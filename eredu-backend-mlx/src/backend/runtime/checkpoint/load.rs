use eredu_checkpoint::AffineQuantization;

use eredu_checkpoint::WeightQuantization;

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::MmapOptions;
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

#[cfg(test)]
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
            return Err(Error::ArchitectureModel(format!(
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
                    return Err(Error::ArchitectureModel(format!(
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
                return Err(Error::ArchitectureModel(format!(
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
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let mut params = model.parameters_mut().flatten();
    if let Some((quantization, stream)) = quantization {
        load_array_quantized_strict(&mut params, name, value, stream, quantization, report)
    } else {
        load_array_strict(&mut params, name, value, report);
        Ok(())
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

    /// Validates the report against the model parameters.
    pub fn finish<M: ModuleParameters + ?Sized>(self, model: &M) -> Result<(), Error> {
        self.finish_excluding(model, |_| false)
    }

    /// Validates a partial strict load while leaving an independently managed
    /// parameter class untouched.
    pub fn finish_excluding<M, F>(self, model: &M, excluded: F) -> Result<(), Error>
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
            .collect::<Vec<_>>();

        let mut unused = self.unused;
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
            report,
        )
    })
}

pub fn load_array_strict(
    params: &mut FlattenedModuleParamMut<'_>,
    key: String,
    value: Array,
    report: &mut StrictLoadReport,
) {
    if params.contains_key(key.as_str()) {
        if let Some(param) = params.get_mut(key.as_str()) {
            let expected_shape = param.shape().to_vec();
            let actual_shape = value.shape().to_vec();
            if expected_shape == actual_shape {
                let checkpoint_native_blocks = value.dtype() == safemlx::Dtype::Uint8;
                **param = value;
                report.record_loaded(key.clone());
                if checkpoint_native_blocks {
                    let prefix = key
                        .strip_suffix(".inner.weight")
                        .or_else(|| key.strip_suffix(".weight"));
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
                report.record_shape_mismatch(key.clone(), key, expected_shape, actual_shape);
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
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    {
        let weight_key = key.clone();
        let (prefix, underscore_companions) = if weight_key == "inner.weight" {
            (String::new(), false)
        } else if let Some(prefix) = weight_key.strip_suffix(".inner.weight") {
            (prefix.to_string(), false)
        } else if let Some(prefix) = weight_key.strip_suffix(".weight") {
            (prefix.to_string(), false)
        } else if weight_key == "weight" {
            (String::new(), false)
        } else {
            (weight_key.clone(), true)
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
        let target = (has_quantized_parameters
            && (weight_key.ends_with(".inner.weight") || packed_direct_weight))
            .then_some((weight_key, scales_key, biases_key));

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
            load_array_strict(params, weight_key, quantized.weight, report);
            load_array_strict(params, scales_key, quantized.scales, report);
            if let Some(biases) = quantized.biases {
                load_array_strict(params, biases_key, biases, report);
            }
            return Ok(());
        }
    }
    load_array_strict(params, key, value, report);
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
        gguf_quantization_configs, load_arrays_quantized_strict, quantize_safetensors_for_test,
        StrictLoadReport,
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

    #[derive(Debug, Clone, ModuleParameters)]
    struct QuantizedLinear {
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
        let mut report = StrictLoadReport::default();
        load_arrays_quantized_strict(
            &mut model,
            HashMap::from([("experts".into(), dense)]),
            stream,
            WeightQuantization::MxFp4,
            &mut report,
        )
        .unwrap();
        report.finish(&model).unwrap();
        assert_eq!(model.experts.shape(), &[3, 8, 8]);
        assert_eq!(
            model.experts_scales.value.as_ref().unwrap().shape(),
            &[3, 8, 2]
        );
        assert!(model.experts_biases.value.is_none());
    }

    #[test]
    fn quantized_strict_load_requires_exact_parameter_names() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights_stream = weights_context.stream();
        let quantization = AffineQuantization::default();
        let mut model = QuantizedLinear {
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
            "eredu-mlx-exact-quantized-load-{}-{suffix}.safetensors",
            std::process::id()
        ));
        Array::save_safetensors([("projection.inner.weight", &dense)], None, &path).unwrap();

        let mut report = StrictLoadReport::default();
        quantize_safetensors_for_test(
            &mut model,
            &path,
            weights_stream,
            stream,
            quantization.into(),
            &mut report,
        )
        .unwrap();
        report.finish(&model).unwrap();

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
        let mut model = QuantizedLinear {
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
        Array::save_safetensors([("projection.inner.weight", &dense)], None, &path).unwrap();
        let mut report = StrictLoadReport::default();
        quantize_safetensors_for_test(
            &mut model,
            &path,
            weights_stream,
            stream,
            WeightQuantization::MxFp4,
            &mut report,
        )
        .unwrap();
        report.finish(&model).unwrap();
        let MaybeQuantized::Quantized(projection) = model.projection else {
            panic!("target projection should use MXFP4 storage")
        };
        assert_eq!(projection.mode, safemlx::ops::QuantizationMode::MxFp4);
        assert!(projection.biases.value.is_none());
        std::fs::remove_file(path).unwrap();
    }
}
