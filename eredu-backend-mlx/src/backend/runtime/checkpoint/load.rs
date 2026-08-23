use eredu_checkpoint::AffineQuantization;

use eredu_checkpoint::WeightQuantization;

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::MmapOptions;
#[cfg(test)]
use safemlx::ops::{concatenate_axis, stack_axis};
use safemlx::{
    module::{FlattenedModuleParamMut, ModuleParameters},
    native_quantization::NativeQuantizationFormat,
    ops::{GgufCheckpoint, GgufMetadataValue},
    transforms::async_eval_with_event,
    Array, Stream,
};
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::backend::error::Error;
use crate::backend::runtime::checkpoint::quantization::quantize_tensor;

pub fn gguf_metadata(checkpoint: &GgufCheckpoint) -> HashMap<String, GgufMetadataValue> {
    checkpoint
        .metadata()
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

pub trait GgufTensorNames {
    fn contains_gguf_tensor(&self, name: &str) -> bool;

    fn any_gguf_tensor<F>(&self, predicate: F) -> bool
    where
        F: FnMut(&str) -> bool;

    fn has_affine_gguf_tensor(&self) -> bool;
}

impl GgufTensorNames for GgufCheckpoint {
    fn contains_gguf_tensor(&self, name: &str) -> bool {
        self.catalog()
            .tensors()
            .any(|tensor| tensor.descriptor().name == name)
    }

    fn any_gguf_tensor<F>(&self, mut predicate: F) -> bool
    where
        F: FnMut(&str) -> bool,
    {
        self.catalog()
            .tensors()
            .any(|tensor| predicate(&tensor.descriptor().name))
    }

    fn has_affine_gguf_tensor(&self) -> bool {
        self.catalog()
            .tensors()
            .any(|tensor| tensor.affine().is_some())
    }
}

#[cfg(any(test, feature = "test-support"))]
impl GgufTensorNames for HashMap<String, Array> {
    fn contains_gguf_tensor(&self, name: &str) -> bool {
        self.contains_key(name)
    }

    fn any_gguf_tensor<F>(&self, predicate: F) -> bool
    where
        F: FnMut(&str) -> bool,
    {
        self.keys().map(String::as_str).any(predicate)
    }

    fn has_affine_gguf_tensor(&self) -> bool {
        self.keys().any(|name| {
            name.ends_with(".scales")
                || name.ends_with(".biases")
                || name.ends_with("_scales")
                || name.ends_with("_biases")
        })
    }
}

pub fn gguf_affine_configs<F>(
    checkpoint: &GgufCheckpoint,
    mut translate: F,
) -> Result<HashMap<String, AffineQuantization>, Error>
where
    F: FnMut(&str) -> String,
{
    let mut configs = HashMap::new();
    for tensor in checkpoint.catalog().tensors() {
        let Some((bits, group_size)) = tensor.affine() else {
            continue;
        };
        let weight_name = translate(&tensor.outputs()[0].name);
        let group_size = i32::try_from(group_size).map_err(|_| {
            Error::Quantization(format!(
                "GGUF group size {group_size} does not fit in an i32"
            ))
        })?;
        let config = AffineQuantization::new(group_size, i32::from(bits))?;
        if configs.insert(weight_name.clone(), config).is_some() {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF tensors collide after translating {weight_name:?}"
            )));
        }
    }
    Ok(configs)
}

/// Exact per-weight runtime formats for mixed affine and native-block GGUF files.
pub fn gguf_quantization_configs<F>(
    checkpoint: &GgufCheckpoint,
    mut translate: F,
) -> Result<HashMap<String, WeightQuantization>, Error>
where
    F: FnMut(&str) -> String,
{
    let mut configs = gguf_affine_configs(checkpoint, &mut translate)?
        .into_iter()
        .map(|(name, config)| (name, config.into()))
        .collect::<HashMap<_, _>>();
    for shard in checkpoint.catalog().shards() {
        for tensor in shard.tensors() {
            let descriptor = tensor.descriptor();
            if tensor.is_mxfp4() {
                let weight_name = translate(&descriptor.name);
                if configs
                    .insert(weight_name.clone(), WeightQuantization::MxFp4)
                    .is_some()
                {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "GGUF tensors collide after translating {weight_name:?}"
                    )));
                }
                continue;
            }
            if tensor.affine().is_some()
                || NativeQuantizationFormat::from_ggml_type(descriptor.ggml_type).is_none()
            {
                continue;
            }
            let weight_name = translate(&descriptor.name);
            let config = WeightQuantization::GgufIQuant {
                ggml_type: descriptor.ggml_type,
                endian: shard.endian(),
            };
            if configs.insert(weight_name.clone(), config).is_some() {
                return Err(Error::UnsupportedArchitecture(format!(
                    "GGUF tensors collide after translating {weight_name:?}"
                )));
            }
        }
    }
    Ok(configs)
}

#[cfg(test)]
pub fn load_named_array_strict<M: ModuleParameters>(
    model: &mut M,
    name: String,
    value: Array,
    quantization: Option<(WeightQuantization, &Stream)>,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let mut params = model.parameters_mut().flatten();
    if let Some((quantization, stream)) = quantization {
        load_array_quantized_strict(
            &mut params,
            name,
            value,
            stream,
            quantization,
            config,
            report,
        )
    } else {
        load_array_strict(&mut params, name, value, config, report);
        Ok(())
    }
}

/// Options for strict checkpoint loading.
///
/// This configuration controls how checkpoint tensor names are matched to model
/// parameters and which missing or unused names are accepted.
#[derive(Debug, Clone, Default)]
pub struct StrictLoadConfig {
    allow_all_unused: bool,
    allowed_unused_prefixes: Vec<String>,
    allowed_missing_suffixes: Vec<String>,
    allowed_missing_contains: Vec<String>,
    key_prefixes_to_strip: Vec<String>,
    key_prefix_rewrites: Vec<(String, String)>,
}

impl StrictLoadConfig {
    /// Allows every unused checkpoint tensor while preserving missing-parameter validation.
    pub fn allow_all_unused(mut self) -> Self {
        self.allow_all_unused = true;
        self
    }

    /// Allows unused checkpoint tensors whose names start with `prefix`.
    pub fn allow_unused_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.allowed_unused_prefixes.push(prefix.into());
        self
    }

    /// Allows missing model parameters whose names end with `suffix`.
    pub fn allow_missing_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.allowed_missing_suffixes.push(suffix.into());
        self
    }

    /// Allows missing model parameters whose names contain `needle`.
    pub fn allow_missing_contains(mut self, needle: impl Into<String>) -> Self {
        self.allowed_missing_contains.push(needle.into());
        self
    }

    /// Adds a candidate key with `prefix` stripped from checkpoint tensor names.
    pub fn strip_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefixes_to_strip.push(prefix.into());
        self
    }

    /// Rewrites a checkpoint key prefix before matching it to model parameters.
    pub fn rewrite_prefix(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.key_prefix_rewrites.push((from.into(), to.into()));
        self
    }

    pub fn is_unused_allowed(&self, key: &str) -> bool {
        self.allow_all_unused
            || self
                .allowed_unused_prefixes
                .iter()
                .any(|prefix| key.starts_with(prefix))
    }

    fn is_missing_allowed(&self, key: &str) -> bool {
        self.allowed_missing_suffixes
            .iter()
            .any(|suffix| key.ends_with(suffix))
            || self
                .allowed_missing_contains
                .iter()
                .any(|needle| key.contains(needle))
    }

    pub fn candidates(&self, key: &str) -> Vec<String> {
        let mut candidates = vec![key.to_string()];
        for prefix in &self.key_prefixes_to_strip {
            if let Some(stripped) = key.strip_prefix(prefix) {
                candidates.push(stripped.to_string());
            }
        }
        for (from, to) in &self.key_prefix_rewrites {
            if let Some(stripped) = key.strip_prefix(from) {
                candidates.push(format!("{to}{stripped}"));
            }
        }

        let mut expanded = Vec::with_capacity(candidates.len() * 2);
        for candidate in candidates {
            expanded.push(candidate.clone());
            if candidate == "weight" {
                expanded.push("inner.weight".to_string());
            } else if let Some(inner_key) = candidate.strip_suffix(".weight") {
                expanded.push(format!("{inner_key}.inner.weight"));
            }
            if candidate == "bias" {
                expanded.push("inner.bias".to_string());
            } else if let Some(inner_key) = candidate.strip_suffix(".bias") {
                expanded.push(format!("{inner_key}.inner.bias"));
            }
        }

        let mut seen = HashSet::new();
        expanded
            .into_iter()
            .filter(|candidate| seen.insert(candidate.clone()))
            .collect()
    }
}

/// Accumulates strict checkpoint-loading diagnostics across one or more files.
#[derive(Debug, Clone, Default)]
pub struct StrictLoadReport {
    loaded: HashSet<String>,
    unused: Vec<String>,
    shape_mismatches: Vec<String>,
}

impl StrictLoadReport {
    pub fn record_loaded(&mut self, key: String) {
        self.loaded.insert(key);
    }

    pub fn record_unused(&mut self, key: String) {
        self.unused.push(key);
    }

    pub fn record_shape_mismatch(
        &mut self,
        weight_key: String,
        param_key: String,
        expected_shape: Vec<i32>,
        actual_shape: Vec<i32>,
    ) {
        self.shape_mismatches.push(format!(
            "{weight_key} -> {param_key}: expected {expected_shape:?}, got {actual_shape:?}"
        ));
    }

    /// Validates the report against the model parameters and load configuration.
    pub fn finish<M: ModuleParameters + ?Sized>(
        self,
        model: &M,
        config: &StrictLoadConfig,
    ) -> Result<(), Error> {
        self.finish_excluding(model, config, |_| false)
    }

    /// Validates a partial strict load while leaving an independently managed
    /// parameter class untouched.
    pub fn finish_excluding<M, F>(
        self,
        model: &M,
        config: &StrictLoadConfig,
        excluded: F,
    ) -> Result<(), Error>
    where
        M: ModuleParameters + ?Sized,
        F: Fn(&str) -> bool,
    {
        let mut missing = model
            .parameters()
            .flatten()
            .keys()
            .map(|key| key.to_string())
            .filter(|key| !excluded(key))
            .filter(|key| !self.loaded.contains(key))
            .filter(|key| !config.is_missing_allowed(key))
            .collect::<Vec<_>>();

        let mut unused = self
            .unused
            .into_iter()
            .filter(|key| !config.is_unused_allowed(key))
            .collect::<Vec<_>>();
        unused.extend(self.shape_mismatches);

        missing.sort();
        unused.sort();

        if missing.is_empty() && unused.is_empty() {
            Ok(())
        } else {
            Err(Error::StrictLoadValidation { missing, unused })
        }
    }
}

/// Strict-loads and quantizes eligible tensors from an in-memory named-array
/// source such as an unquantized GGUF. The map is consumed so each dense source
/// array can be released after its packed replacement is materialized.
#[cfg(test)]
pub fn load_arrays_quantized_strict<M: ModuleParameters>(
    model: &mut M,
    loaded: HashMap<String, Array>,
    quantization_stream: &Stream,
    quantization: WeightQuantization,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    quantization.validate()?;
    let mut params = model.parameters_mut().flatten();
    for (key, value) in loaded {
        load_array_quantized_strict(
            &mut params,
            key,
            value,
            quantization_stream,
            quantization,
            config,
            report,
        )?;
    }
    Ok(())
}

pub fn for_each_safetensor_array<F>(
    path: impl AsRef<Path>,
    stream: &Stream,
    mut f: F,
) -> Result<(), Error>
where
    F: FnMut(String, Array) -> Result<(), Error>,
{
    let file = File::open(path)?;
    // The mmap only has to live until each TensorView is copied into an MLX-owned Array.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(|err| Error::Other(Box::new(err)))?;

    for (key, view) in tensors.iter() {
        let value = Array::try_from(view).map_err(|err| Error::Other(Box::new(err)))?;
        let value = value.copy(stream)?;
        f(key.to_string(), value)?;
    }

    Ok(())
}

#[cfg(test)]
fn quantize_safetensors_for_test<M: ModuleParameters>(
    model: &mut M,
    path: impl AsRef<Path>,
    weights_stream: &Stream,
    quantization_stream: &Stream,
    quantization: WeightQuantization,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    quantization.validate()?;
    let mut params = model.parameters_mut().flatten();
    for_each_safetensor_array(path, weights_stream, |key, value| {
        load_array_quantized_strict(
            &mut params,
            key,
            value,
            quantization_stream,
            quantization,
            config,
            report,
        )
    })
}

pub fn load_array_strict(
    params: &mut FlattenedModuleParamMut<'_>,
    key: String,
    value: Array,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) {
    let mut matched = None;
    for candidate in config.candidates(&key) {
        if params.contains_key(candidate.as_str()) {
            matched = Some(candidate);
            break;
        }
    }

    if let Some(candidate) = matched {
        if let Some(param) = params.get_mut(candidate.as_str()) {
            let expected_shape = param.shape().to_vec();
            let actual_shape = value.shape().to_vec();
            if expected_shape == actual_shape {
                let checkpoint_native_blocks = value.dtype() == safemlx::Dtype::Uint8;
                **param = value;
                report.record_loaded(candidate.clone());
                if checkpoint_native_blocks {
                    let prefix = candidate
                        .strip_suffix(".inner.weight")
                        .or_else(|| candidate.strip_suffix(".weight"));
                    if let Some(prefix) = prefix {
                        let scales = format!("{prefix}.scales");
                        if params
                            .get(scales.as_str())
                            .is_some_and(|scales| scales.shape() == [1])
                        {
                            // Native-block modules retain a one-element unloaded
                            // scale placeholder solely for parameter compatibility.
                            report.record_loaded(scales);
                        }
                    }
                }
            } else {
                report.record_shape_mismatch(key, candidate, expected_shape, actual_shape);
            }
        }
    } else {
        report.record_unused(key);
    }
}

/// Strict-loads a dense safetensors file into a model whose selected parameters
/// use the standard MLX affine quantized layout.
///
/// Dense matrices are quantized and materialized one at a time as they are
/// read, bounding the lazy graph and active allocation peak. A target module is
/// recognized either by the standard safemlx `inner.weight` parameter plus
/// sibling `scales`/`biases`, or by a packed `weight` with those siblings.
pub fn load_array_quantized_strict(
    params: &mut FlattenedModuleParamMut<'_>,
    key: String,
    value: Array,
    quantization_stream: &Stream,
    quantization: WeightQuantization,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    {
        let target = config.candidates(&key).into_iter().find_map(|candidate| {
            let (prefix, weight_key, underscore_companions) = if candidate == "inner.weight" {
                (String::new(), candidate, false)
            } else if let Some(prefix) = candidate.strip_suffix(".inner.weight") {
                (prefix.to_string(), candidate, false)
            } else if let Some(prefix) = candidate.strip_suffix(".weight") {
                (prefix.to_string(), candidate, false)
            } else if candidate == "weight" {
                (String::new(), candidate, false)
            } else {
                (candidate.clone(), candidate, true)
            };
            let scales_key = if prefix.is_empty() {
                "scales".to_string()
            } else if underscore_companions {
                format!("{prefix}_scales")
            } else {
                format!("{prefix}.scales")
            };
            let biases_key = if prefix.is_empty() {
                "biases".to_string()
            } else if underscore_companions {
                format!("{prefix}_biases")
            } else {
                format!("{prefix}.biases")
            };
            let has_quantized_parameters = params.contains_key(weight_key.as_str())
                && params.contains_key(scales_key.as_str())
                && (!quantization.has_biases() || params.contains_key(biases_key.as_str()));
            let packed_direct_weight = !weight_key.ends_with(".inner.weight")
                && params
                    .get(weight_key.as_str())
                    .is_some_and(|target| target.shape() != value.shape());
            (has_quantized_parameters
                && (weight_key.ends_with(".inner.weight") || packed_direct_weight))
                .then_some((weight_key, scales_key, biases_key))
        });

        if let Some((weight_key, scales_key, biases_key)) = target {
            let quantized = quantize_tensor(&value, quantization, quantization_stream)?;
            // MLX quantization is lazy. Materialize this tensor before the
            // source value leaves the streaming callback so subsequent
            // weights do not accumulate a checkpoint-sized dense graph.
            let mut arrays = vec![&quantized.weight, &quantized.scales];
            if let Some(biases) = &quantized.biases {
                arrays.push(biases);
            }
            async_eval_with_event(arrays)?.synchronize()?;
            load_array_strict(params, weight_key, quantized.weight, config, report);
            load_array_strict(params, scales_key, quantized.scales, config, report);
            if let Some(biases) = quantized.biases {
                load_array_strict(params, biases_key, biases, config, report);
            }
            return Ok(());
        }
    }
    load_array_strict(params, key, value, config, report);
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
/// Hugging Face safetensors index file.
pub struct WeightMap {
    /// Mapping from tensor name to shard file name.
    pub weight_map: HashMap<String, String>,
}

/// Returns the safetensors files referenced by a Hugging Face model directory.
pub fn safetensors_files(model_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>, Error> {
    let model_dir = model_dir.as_ref();
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let mut files = weight_map
            .weight_map
            .values()
            .map(|file| model_dir.join(file))
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        return Ok(files);
    }

    Ok(vec![model_dir.join("model.safetensors")])
}

/// Strict-loads a model directory while streaming and packing split ReLU2 experts.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub fn load_safetensors_dir_strict_with_split_relu2_experts<M, F>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    weights_stream: &Stream,
    transform_stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
    num_experts: i32,
    rewrite_key: F,
) -> Result<(), Error>
where
    M: ModuleParameters,
    F: Fn(&str) -> Result<String, Error>,
{
    let mut expert_parts: HashMap<(String, i32), Relu2ExpertParts> = HashMap::new();
    let mut params = model.parameters_mut().flatten();

    for file in safetensors_files(model_dir)? {
        for_each_safetensor_array(file, weights_stream, |key, value| {
            let key = rewrite_key(&key)?;
            if let Some((prefix, expert, projection)) =
                parse_split_relu2_expert_projection_key(&key)
            {
                let parts = expert_parts.entry((prefix, expert)).or_default();
                match projection {
                    Relu2ExpertProjection::Up => parts.up = Some(value),
                    Relu2ExpertProjection::Down => parts.down = Some(value),
                }
            } else {
                load_array_strict(&mut params, key, value, config, report);
            }
            Ok(())
        })?;

        let mut complete_prefixes = expert_parts
            .keys()
            .map(|(prefix, _)| prefix.clone())
            .collect::<Vec<_>>();
        complete_prefixes.sort();
        complete_prefixes.dedup();
        for prefix in complete_prefixes {
            if split_relu2_expert_prefix_complete(&expert_parts, &prefix, num_experts) {
                let packed = pack_split_relu2_expert_prefix(
                    &mut expert_parts,
                    &prefix,
                    num_experts,
                    transform_stream,
                )?;
                for (key, value) in packed {
                    load_array_strict(&mut params, key, value, config, report);
                }
            }
        }
    }

    if let Some((prefix, _)) = expert_parts.keys().next().cloned() {
        pack_split_relu2_expert_prefix(&mut expert_parts, &prefix, num_experts, transform_stream)?;
    }

    Ok(())
}

/// Strict-loads a model directory while streaming and packing split gated-product experts.
///
/// Public checkpoints commonly store `w1`, `w2`, and `w3` per expert, while the
/// runtime uses one expert-major gate/up bank plus a down bank. Completed layer
/// banks are loaded immediately so all expert layers are never resident at once.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub fn load_safetensors_dir_strict_with_split_gated_product_experts<M>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    weights_stream: &Stream,
    transform_stream: &Stream,
    quantization: Option<WeightQuantization>,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
    num_experts: i32,
) -> Result<(), Error>
where
    M: ModuleParameters,
{
    load_safetensors_dir_strict_with_split_gated_product_experts_and_transform(
        model,
        model_dir,
        weights_stream,
        transform_stream,
        quantization,
        config,
        report,
        num_experts,
        |key, value| Ok(vec![(key, value)]),
    )
}

/// Strict-loads and packs split gated-product experts after applying a streaming key/value transform.
///
/// The transform can split or rewrite architecture-specific tensors before
/// expert detection and strict parameter matching without buffering a shard.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub fn load_safetensors_dir_strict_with_split_gated_product_experts_and_transform<M, F>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    weights_stream: &Stream,
    transform_stream: &Stream,
    quantization: Option<WeightQuantization>,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
    num_experts: i32,
    transform: F,
) -> Result<(), Error>
where
    M: ModuleParameters,
    F: Fn(String, Array) -> Result<Vec<(String, Array)>, Error>,
{
    if let Some(quantization) = quantization {
        quantization.validate()?;
    }
    let mut expert_parts: HashMap<
        (String, GatedProductExpertComponent, i32),
        GatedProductExpertParts,
    > = HashMap::new();
    let mut params = model.parameters_mut().flatten();

    for file in safetensors_files(model_dir)? {
        for_each_safetensor_array(file, weights_stream, |key, value| {
            for (key, value) in transform(key, value)? {
                if let Some((prefix, expert, projection, component)) =
                    parse_split_gated_product_expert_projection_key(&key)
                {
                    {
                        let parts = expert_parts
                            .entry((prefix.clone(), component, expert))
                            .or_default();
                        match projection {
                            GatedProductExpertProjection::Gate => parts.gate = Some(value),
                            GatedProductExpertProjection::Down => parts.down = Some(value),
                            GatedProductExpertProjection::Up => parts.up = Some(value),
                        }
                    }
                    if split_gated_product_expert_prefix_complete(
                        &expert_parts,
                        &prefix,
                        component,
                        num_experts,
                    ) {
                        for (key, value) in pack_split_gated_product_expert_prefix(
                            &mut expert_parts,
                            &prefix,
                            component,
                            num_experts,
                            transform_stream,
                        )? {
                            if let Some(quantization) = quantization
                                .filter(|_| component == GatedProductExpertComponent::Weight)
                            {
                                load_array_quantized_strict(
                                    &mut params,
                                    key,
                                    value,
                                    transform_stream,
                                    quantization,
                                    config,
                                    report,
                                )?;
                            } else {
                                load_array_strict(&mut params, key, value, config, report);
                            }
                        }
                    }
                } else if let Some(quantization) = quantization {
                    load_array_quantized_strict(
                        &mut params,
                        key,
                        value,
                        transform_stream,
                        quantization,
                        config,
                        report,
                    )?;
                } else {
                    load_array_strict(&mut params, key, value, config, report);
                }
            }
            Ok(())
        })?;
    }

    if let Some((prefix, component, _)) = expert_parts.keys().next().cloned() {
        pack_split_gated_product_expert_prefix(
            &mut expert_parts,
            &prefix,
            component,
            num_experts,
            transform_stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Projection kind in a split gated-product expert checkpoint.
enum GatedProductExpertProjection {
    /// Gate projection (`w1`).
    Gate,
    /// Down projection (`w2`).
    Down,
    /// Up projection (`w3`).
    Up,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
/// Stored component of one split expert projection.
enum GatedProductExpertComponent {
    /// Projection values.
    Weight,
    /// Quantization scales.
    Scales,
    /// Quantization biases.
    Biases,
}

#[cfg(test)]
impl GatedProductExpertComponent {
    fn packed_suffix(self) -> &'static str {
        match self {
            Self::Weight => "",
            Self::Scales => "_scales",
            Self::Biases => "_biases",
        }
    }
}

#[derive(Default)]
#[cfg(test)]
struct GatedProductExpertParts {
    gate: Option<Array>,
    down: Option<Array>,
    up: Option<Array>,
}

/// Parses keys like `prefix.experts.17.w1.weight`.
#[cfg(test)]
fn parse_split_gated_product_expert_projection_key(
    key: &str,
) -> Option<(
    String,
    i32,
    GatedProductExpertProjection,
    GatedProductExpertComponent,
)> {
    let (prefix, rest) = if let Some((prefix, rest)) = key.split_once(".experts.") {
        (format!("{prefix}.experts"), rest)
    } else if let Some((prefix, rest)) = key.split_once(".switch_mlp.") {
        (format!("{prefix}.switch_mlp"), rest)
    } else {
        return None;
    };
    let mut parts = rest.split('.');
    let expert = parts.next()?.parse().ok()?;
    let projection = match parts.next()? {
        "w1" | "gate_proj" => GatedProductExpertProjection::Gate,
        "w2" | "down_proj" => GatedProductExpertProjection::Down,
        "w3" | "up_proj" => GatedProductExpertProjection::Up,
        _ => return None,
    };
    let component = match parts.next()? {
        "weight" => GatedProductExpertComponent::Weight,
        "scale" | "scales" => GatedProductExpertComponent::Scales,
        "bias" | "biases" => GatedProductExpertComponent::Biases,
        _ => return None,
    };
    if parts.next().is_some() {
        return None;
    }
    Some((prefix, expert, projection, component))
}

#[cfg(test)]
fn split_gated_product_expert_prefix_complete(
    expert_parts: &HashMap<(String, GatedProductExpertComponent, i32), GatedProductExpertParts>,
    prefix: &str,
    component: GatedProductExpertComponent,
    num_experts: i32,
) -> bool {
    (0..num_experts).all(|expert| {
        expert_parts
            .get(&(prefix.to_string(), component, expert))
            .is_some_and(|parts| parts.gate.is_some() && parts.down.is_some() && parts.up.is_some())
    })
}

#[cfg(test)]
fn pack_split_gated_product_expert_prefix(
    expert_parts: &mut HashMap<(String, GatedProductExpertComponent, i32), GatedProductExpertParts>,
    prefix: &str,
    component: GatedProductExpertComponent,
    num_experts: i32,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let mut gate_up = Vec::with_capacity(num_experts as usize);
    let mut down = Vec::with_capacity(num_experts as usize);
    for expert in 0..num_experts {
        let parts = expert_parts
            .remove(&(prefix.to_string(), component, expert))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "checkpoint is missing expert {expert} for '{prefix}'"
                ))
            })?;
        let gate = parts.gate.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "checkpoint is missing {prefix}.{expert}.w1 component {component:?}"
            ))
        })?;
        let up = parts.up.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "checkpoint is missing {prefix}.{expert}.w3 component {component:?}"
            ))
        })?;
        gate_up.push(concatenate_axis(&[gate, up], 0, stream)?);
        down.push(parts.down.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "checkpoint is missing {prefix}.{expert}.w2 component {component:?}"
            ))
        })?);
    }
    let gate_up_proj = stack_axis(&gate_up, 0, stream)?;
    let down_proj = stack_axis(&down, 0, stream)?;
    async_eval_with_event([&gate_up_proj, &down_proj])?.synchronize()?;
    let suffix = component.packed_suffix();
    Ok(HashMap::from([
        (format!("{prefix}.gate_up_proj{suffix}"), gate_up_proj),
        (format!("{prefix}.down_proj{suffix}"), down_proj),
    ]))
}

/// Packs a map of split gated-product experts into local expert-major banks.
///
/// Expert ids in `loaded` must already be dense local ids `0..num_experts`.
#[cfg(test)]
pub fn transform_split_gated_product_experts(
    loaded: HashMap<String, Array>,
    num_experts: i32,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let mut transformed = HashMap::with_capacity(loaded.len());
    let mut expert_parts: HashMap<
        (String, GatedProductExpertComponent, i32),
        GatedProductExpertParts,
    > = HashMap::new();
    for (key, value) in loaded {
        if let Some((prefix, expert, projection, component)) =
            parse_split_gated_product_expert_projection_key(&key)
        {
            let parts = expert_parts.entry((prefix, component, expert)).or_default();
            match projection {
                GatedProductExpertProjection::Gate => parts.gate = Some(value),
                GatedProductExpertProjection::Down => parts.down = Some(value),
                GatedProductExpertProjection::Up => parts.up = Some(value),
            }
        } else {
            transformed.insert(key, value);
        }
    }
    let mut prefixes = expert_parts
        .keys()
        .map(|(prefix, component, _)| (prefix.clone(), *component))
        .collect::<Vec<_>>();
    prefixes.sort();
    prefixes.dedup();
    for (prefix, component) in prefixes {
        transformed.extend(pack_split_gated_product_expert_prefix(
            &mut expert_parts,
            &prefix,
            component,
            num_experts,
            stream,
        )?);
    }
    Ok(transformed)
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Projection kind in a split ReLU2 expert checkpoint.
enum Relu2ExpertProjection {
    /// Expert up projection.
    Up,
    /// Expert down projection.
    Down,
}

#[derive(Default)]
#[cfg(test)]
struct Relu2ExpertParts {
    up: Option<Array>,
    down: Option<Array>,
}

/// Parses keys like `prefix.experts.17.up_proj.weight`.
#[cfg(test)]
fn parse_split_relu2_expert_projection_key(
    key: &str,
) -> Option<(String, i32, Relu2ExpertProjection)> {
    let (prefix, rest) = key.split_once(".experts.")?;
    let mut parts = rest.split('.');
    let expert = parts.next()?.parse().ok()?;
    let projection = match parts.next()? {
        "up_proj" => Relu2ExpertProjection::Up,
        "down_proj" => Relu2ExpertProjection::Down,
        _ => return None,
    };
    if parts.next()? != "weight" || parts.next().is_some() {
        return None;
    }
    Some((format!("{prefix}.experts"), expert, projection))
}

/// Packs split ReLU2 expert tensors into `prefix.experts.{up,down}_proj` banks.
#[cfg(test)]
pub fn transform_split_relu2_experts(
    loaded: HashMap<String, Array>,
    num_experts: i32,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let mut transformed = HashMap::with_capacity(loaded.len());
    let mut expert_parts: HashMap<(String, i32), Relu2ExpertParts> = HashMap::new();

    for (key, value) in loaded {
        if let Some((prefix, expert, projection)) = parse_split_relu2_expert_projection_key(&key) {
            let parts = expert_parts.entry((prefix, expert)).or_default();
            match projection {
                Relu2ExpertProjection::Up => parts.up = Some(value),
                Relu2ExpertProjection::Down => parts.down = Some(value),
            }
        } else {
            transformed.insert(key, value);
        }
    }

    let mut layer_prefixes = expert_parts
        .keys()
        .map(|(prefix, _)| prefix.clone())
        .collect::<Vec<_>>();
    layer_prefixes.sort();
    layer_prefixes.dedup();

    for prefix in layer_prefixes {
        transformed.extend(pack_split_relu2_expert_prefix(
            &mut expert_parts,
            &prefix,
            num_experts,
            stream,
        )?);
    }

    Ok(transformed)
}

#[cfg(test)]
fn split_relu2_expert_prefix_complete(
    expert_parts: &HashMap<(String, i32), Relu2ExpertParts>,
    prefix: &str,
    num_experts: i32,
) -> bool {
    (0..num_experts).all(|expert| {
        expert_parts
            .get(&(prefix.to_string(), expert))
            .is_some_and(|parts| parts.up.is_some() && parts.down.is_some())
    })
}

#[cfg(test)]
fn pack_split_relu2_expert_prefix(
    expert_parts: &mut HashMap<(String, i32), Relu2ExpertParts>,
    prefix: &str,
    num_experts: i32,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let mut up = Vec::with_capacity(num_experts as usize);
    let mut down = Vec::with_capacity(num_experts as usize);
    for expert in 0..num_experts {
        let parts = expert_parts
            .remove(&(prefix.to_string(), expert))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "checkpoint is missing expert {expert} for '{prefix}'"
                ))
            })?;
        up.push(parts.up.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "checkpoint is missing {prefix}.{expert}.up_proj.weight"
            ))
        })?);
        down.push(parts.down.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "checkpoint is missing {prefix}.{expert}.down_proj.weight"
            ))
        })?);
    }

    let up_proj = stack_axis(&up, 0, stream)?;
    let down_proj = stack_axis(&down, 0, stream)?;
    async_eval_with_event([&up_proj, &down_proj])?.synchronize()?;
    Ok(HashMap::from([
        (format!("{prefix}.up_proj"), up_proj),
        (format!("{prefix}.down_proj"), down_proj),
    ]))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, HashMap},
        time::{SystemTime, UNIX_EPOCH},
    };

    use eredu_checkpoint::{AffineQuantization, WeightQuantization};
    use eredu_gguf::{Endian, GgmlType, TensorInput, Writer, WriterOptions};
    use safemlx::{
        macros::ModuleParameters, module::Param, quantization::MaybeQuantized, Array, Device,
        DeviceType, Dtype, ExecutionContext,
    };

    use crate::{
        backend::nn::linear::unloaded_maybe_quantized_linear,
        backend::runtime::checkpoint::quantization::quantize_tensor,
    };

    use super::{
        gguf_quantization_configs, load_arrays_quantized_strict,
        parse_split_gated_product_expert_projection_key, quantize_safetensors_for_test,
        StrictLoadConfig, StrictLoadReport,
    };

    #[test]
    fn gguf_runtime_configs_preserve_every_native_affine_format() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("native.gguf");
        let formats = [GgmlType::Q4K, GgmlType::Q5_1, GgmlType::Q8_0];
        let names = formats
            .iter()
            .map(|format| format!("{format:?}.weight"))
            .collect::<Vec<_>>();
        let dimensions = formats
            .iter()
            .map(|format| [format.block_and_bytes().unwrap().0])
            .collect::<Vec<_>>();
        let payloads = formats
            .iter()
            .map(|format| vec![0; format.block_and_bytes().unwrap().1 as usize])
            .collect::<Vec<_>>();
        let tensors = formats
            .iter()
            .enumerate()
            .map(|(index, format)| TensorInput {
                name: &names[index],
                dimensions: &dimensions[index],
                ggml_type: *format,
                data: &payloads[index],
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(&path).unwrap(),
                &BTreeMap::new(),
                &tensors,
            )
            .unwrap();

        let checkpoint = safemlx::ops::GgufCheckpoint::open(path).unwrap();
        let configs = gguf_quantization_configs(&checkpoint, str::to_string).unwrap();
        for (name, expected) in names.iter().zip(formats) {
            assert!(matches!(
                configs[name],
                WeightQuantization::GgufIQuant { ggml_type, .. } if ggml_type == expected
            ));
        }
    }

    #[test]
    fn big_endian_native_affine_format_keeps_portable_runtime_config() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("big-endian.gguf");
        let payload = vec![0; GgmlType::Q4K.block_and_bytes().unwrap().1 as usize];
        Writer::new(WriterOptions {
            endian: Endian::Big,
            ..WriterOptions::default()
        })
        .unwrap()
        .write(
            std::fs::File::create(&path).unwrap(),
            &BTreeMap::new(),
            &[TensorInput {
                name: "projection.weight",
                dimensions: &[256],
                ggml_type: GgmlType::Q4K,
                data: &payload,
            }],
        )
        .unwrap();

        let checkpoint = safemlx::ops::GgufCheckpoint::open(path).unwrap();
        let configs = gguf_quantization_configs(&checkpoint, str::to_string).unwrap();
        assert!(matches!(
            configs["projection.weight"],
            WeightQuantization::Affine(config) if config.group_size == 32 && config.bits == 4
        ));
    }

    #[test]
    fn parses_split_gated_product_expert_names() {
        let (prefix, expert, projection, component) =
            parse_split_gated_product_expert_projection_key(
                "model.layers.3.feed_forward.experts.17.w3.weight",
            )
            .unwrap();
        assert_eq!(prefix, "model.layers.3.feed_forward.experts");
        assert_eq!(expert, 17);
        assert_eq!(projection, super::GatedProductExpertProjection::Up);
        assert_eq!(component, super::GatedProductExpertComponent::Weight);
        let (_, _, projection, component) = parse_split_gated_product_expert_projection_key(
            "model.layers.3.mlp.experts.17.gate_proj.weight",
        )
        .unwrap();
        assert_eq!(projection, super::GatedProductExpertProjection::Gate);
        assert_eq!(component, super::GatedProductExpertComponent::Weight);
        assert!(parse_split_gated_product_expert_projection_key(
            "model.layers.3.feed_forward.experts.17.bias"
        )
        .is_none());
    }

    #[test]
    fn strict_load_candidates_do_not_invent_family_aliases() {
        let config = StrictLoadConfig::default();
        assert_eq!(
            config.candidates("model.language_model.embed_tokens.scales"),
            ["model.language_model.embed_tokens.scales"]
        );
        assert_eq!(
            config.candidates("model.language_model.embed_tokens_per_layer.biases"),
            ["model.language_model.embed_tokens_per_layer.biases"]
        );
    }

    #[derive(Debug, Clone, ModuleParameters)]
    struct RewrittenLinear {
        #[param]
        projection: MaybeQuantized<safemlx::nn::Linear>,
    }

    #[derive(Debug, Clone, ModuleParameters)]
    struct PackedExperts {
        #[param]
        experts: Param<Array>,
        #[param]
        experts_scales: Param<Option<Array>>,
        #[param]
        experts_biases: Param<Option<Array>>,
    }

    #[test]
    fn named_array_quantization_packs_rank_three_experts() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut model = PackedExperts {
            experts: Param::<Array>::unloaded(&[3, 8, 8], Dtype::Uint32, stream).unwrap(),
            experts_scales: Param::<Option<Array>>::unloaded_some(&[3, 8, 2], Dtype::Uint8, stream)
                .unwrap(),
            experts_biases: Param::new(None),
        };
        let dense = Array::from_slice(&vec![0.25f32; 3 * 8 * 64], &[3, 8, 64]);
        let config = StrictLoadConfig::default();
        let mut report = StrictLoadReport::default();
        load_arrays_quantized_strict(
            &mut model,
            HashMap::from([("experts".into(), dense)]),
            stream,
            WeightQuantization::MxFp4,
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&model, &config).unwrap();
        assert_eq!(model.experts.shape(), &[3, 8, 8]);
        assert_eq!(
            model.experts_scales.value.as_ref().unwrap().shape(),
            &[3, 8, 2]
        );
        assert!(model.experts_biases.value.is_none());
    }

    #[test]
    fn quantized_strict_load_applies_key_rewrites_before_target_selection() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights_stream = weights_context.stream();
        let quantization = AffineQuantization::default();
        let mut model = RewrittenLinear {
            projection: unloaded_maybe_quantized_linear(
                64,
                8,
                false,
                Some(quantization.into()),
                stream,
            )
            .unwrap(),
        };
        let values = (0..(8 * 64))
            .map(|index| (index as f32 - 255.5) / 64.0)
            .collect::<Vec<_>>();
        let dense = Array::from_slice(&values, &[8, 64]);
        let expected = quantize_tensor(&dense, quantization, stream).unwrap();
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "eredu-mlx-rewritten-quantized-load-{}-{suffix}.safetensors",
            std::process::id()
        ));
        Array::save_safetensors([("checkpoint.projection.weight", &dense)], None, &path).unwrap();

        let config = StrictLoadConfig::default().rewrite_prefix("checkpoint.", "");
        let mut report = StrictLoadReport::default();
        quantize_safetensors_for_test(
            &mut model,
            &path,
            weights_stream,
            stream,
            quantization.into(),
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&model, &config).unwrap();

        let MaybeQuantized::Quantized(projection) = model.projection else {
            panic!("target projection should use affine storage")
        };
        assert_eq!(
            projection
                .inner
                .weight
                .evaluated()
                .unwrap()
                .as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
        assert_eq!(
            projection.scales.evaluated().unwrap().as_slice::<f32>(),
            expected.scales.evaluated().unwrap().as_slice::<f32>()
        );
        assert_eq!(
            projection
                .biases
                .value
                .as_ref()
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            expected
                .biases
                .as_ref()
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mxfp4_strict_load_streams_weight_and_scales_without_biases() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights_stream = weights_context.stream();
        let mut model = RewrittenLinear {
            projection: unloaded_maybe_quantized_linear(
                64,
                8,
                false,
                Some(WeightQuantization::MxFp4),
                stream,
            )
            .unwrap(),
        };
        let dense = Array::from_slice(&vec![0.5f32; 8 * 64], &[8, 64]);
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "eredu-mlx-mxfp4-strict-load-{}-{suffix}.safetensors",
            std::process::id()
        ));
        Array::save_safetensors([("projection.weight", &dense)], None, &path).unwrap();
        let config = StrictLoadConfig::default();
        let mut report = StrictLoadReport::default();
        quantize_safetensors_for_test(
            &mut model,
            &path,
            weights_stream,
            stream,
            WeightQuantization::MxFp4,
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&model, &config).unwrap();
        let MaybeQuantized::Quantized(projection) = model.projection else {
            panic!("target projection should use MXFP4 storage")
        };
        assert_eq!(projection.mode, safemlx::ops::QuantizationMode::MxFp4);
        assert!(projection.biases.value.is_none());
        std::fs::remove_file(path).unwrap();
    }
}
