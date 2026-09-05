#[cfg(test)]
use eredu_checkpoint::{AffineQuantization, WeightQuantization};

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use safemlx::{Array, Stream};
#[cfg(test)]
use std::collections::HashMap;

use crate::backend::error::Error;
#[cfg(test)]
use crate::backend::runtime::checkpoint::gguf::GgufCheckpoint;
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
use crate::backend::runtime::checkpoint::quantization::quantize_tensor;
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
use crate::module::FlattenedModuleParamMut;
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
use crate::module::PhysicalParameters;
#[cfg(test)]
use crate::native_quantization::NativeQuantizationFormat;
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
use safemlx::transforms::async_eval_with_event;
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
use std::collections::HashSet;

/// Lowers affine GGUF encodings under an admitted canonical tensor mapping.
#[cfg(test)]
fn gguf_affine_configs(
    checkpoint: &GgufCheckpoint,
    tensor_mapping: &[eredu_gguf::TranslatedTensorLayout],
) -> Result<HashMap<String, AffineQuantization>, Error> {
    let mut configs = HashMap::new();
    for tensor in checkpoint.catalog().tensors() {
        let Some((bits, group_size)) = tensor.affine() else {
            continue;
        };
        let weight_name = canonical_gguf_name(
            tensor_mapping,
            &tensor.descriptor().name,
            &tensor.outputs()[0].name,
        )?;
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

/// Lowers exact mixed affine and native-block GGUF encodings under an admitted
/// canonical tensor mapping.
#[cfg(test)]
pub(crate) fn gguf_quantization_configs(
    checkpoint: &GgufCheckpoint,
    tensor_mapping: &[eredu_gguf::TranslatedTensorLayout],
) -> Result<HashMap<String, WeightQuantization>, Error> {
    let mut configs = gguf_affine_configs(checkpoint, tensor_mapping)?
        .into_iter()
        .map(|(name, config)| (name, config.into()))
        .collect::<HashMap<_, _>>();
    for shard in checkpoint.catalog().shards() {
        for tensor in shard.tensors() {
            let descriptor = tensor.descriptor();
            if tensor.is_mxfp4() {
                let weight_name =
                    canonical_gguf_name(tensor_mapping, &descriptor.name, &descriptor.name)?;
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
            let weight_name =
                canonical_gguf_name(tensor_mapping, &descriptor.name, &descriptor.name)?;
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
fn canonical_gguf_name(
    tensor_mapping: &[eredu_gguf::TranslatedTensorLayout],
    physical_name: &str,
    original_name: &str,
) -> Result<String, Error> {
    tensor_mapping
        .iter()
        .find(|mapped| {
            mapped.physical_name == physical_name && mapped.original_name == original_name
        })
        .map(|mapped| mapped.layout.name.clone())
        .ok_or_else(|| {
            Error::ArchitectureModel(format!(
                "admitted GGUF tensor mapping omits {physical_name:?} output {original_name:?}"
            ))
        })
}

/// Accumulates strict checkpoint-loading diagnostics across one or more files.
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
#[derive(Debug, Clone, Default)]
pub(crate) struct StrictLoadReport {
    loaded: HashSet<String>,
    unused: Vec<String>,
    shape_mismatches: Vec<String>,
}

#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
impl StrictLoadReport {
    /// Records a checkpoint tensor successfully assigned to a parameter.
    fn record_loaded(&mut self, key: String) {
        self.loaded.insert(key);
    }

    /// Records an unused checkpoint tensor.
    fn record_unused(&mut self, key: String) {
        self.unused.push(key);
    }

    /// Records a checkpoint tensor whose shape did not match its parameter.
    fn record_shape_mismatch(
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
    #[cfg(all(
        test,
        any(feature = "cuda", all(feature = "metal", target_os = "macos"))
    ))]
    pub(crate) fn finish<M: PhysicalParameters + ?Sized>(self, model: &M) -> Result<(), Error> {
        self.finish_excluding(model, |_| false)
    }

    /// Validates a partial strict load while leaving an independently managed
    /// parameter class untouched.
    #[cfg(all(
        test,
        any(feature = "cuda", all(feature = "metal", target_os = "macos"))
    ))]
    pub(crate) fn finish_excluding<M, F>(self, model: &M, excluded: F) -> Result<(), Error>
    where
        M: PhysicalParameters + ?Sized,
        F: Fn(&str) -> bool,
    {
        self.finish_parameter_names(
            model
                .parameters()
                .flatten()
                .keys()
                .map(|key| key.to_string()),
            excluded,
        )
    }

    /// Validates the report against names from an explicit parameter topology.
    pub(crate) fn finish_parameter_names<I, F>(
        self,
        parameter_names: I,
        excluded: F,
    ) -> Result<(), Error>
    where
        I: IntoIterator<Item = String>,
        F: Fn(&str) -> bool,
    {
        let mut missing = parameter_names
            .into_iter()
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

/// Strict-loads one exact checkpoint tensor identity into a module parameter.
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
pub(crate) fn load_array_strict(
    params: &mut FlattenedModuleParamMut<'_>,
    key: String,
    value: Array,
    report: &mut StrictLoadReport,
) {
    load_array_for_parameter_strict(params, key.clone(), key, value, report);
}

#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
fn load_array_for_parameter_strict(
    params: &mut FlattenedModuleParamMut<'_>,
    source_key: String,
    parameter_key: String,
    value: Array,
    report: &mut StrictLoadReport,
) {
    if params.contains_key(parameter_key.as_str()) {
        if let Some(param) = params.get_mut(parameter_key.as_str()) {
            let expected_shape = param.shape().to_vec();
            let actual_shape = value.shape().to_vec();
            if expected_shape == actual_shape {
                **param = value;
                report.record_loaded(parameter_key);
            } else {
                report.record_shape_mismatch(
                    source_key,
                    parameter_key,
                    expected_shape,
                    actual_shape,
                );
            }
        }
    } else {
        report.record_unused(source_key);
    }
}

/// Exact parameter destinations for quantizing one dense source tensor.
///
/// The caller obtains these identities from architecture parameter metadata;
/// this loader never derives private slots or companion names.
#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
pub(in crate::backend::runtime::checkpoint) struct QuantizedLoadRecipe {
    weight: String,
    scales: String,
    biases: Option<String>,
}

#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
impl QuantizedLoadRecipe {
    pub(in crate::backend::runtime::checkpoint) fn new(
        weight: impl Into<String>,
        scales: impl Into<String>,
        biases: Option<String>,
    ) -> Self {
        Self {
            weight: weight.into(),
            scales: scales.into(),
            biases,
        }
    }
}

/// Strict-loads or explicitly quantizes one named array.
///
/// Dense matrices are quantized and materialized one at a time as they are
/// read, bounding the lazy graph and active allocation peak. A quantization
/// recipe contains every exact destination; without one, loading is exact.
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
pub(in crate::backend::runtime::checkpoint) fn load_array_quantized_strict(
    params: &mut FlattenedModuleParamMut<'_>,
    key: String,
    value: Array,
    quantization_stream: &Stream,
    quantization: WeightQuantization,
    recipe: Option<&QuantizedLoadRecipe>,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let Some(recipe) = recipe else {
        load_array_strict(params, key, value, report);
        return Ok(());
    };
    let quantized = quantize_tensor(&value, quantization, quantization_stream)?;
    // MLX quantization is lazy. Materialize this tensor before the source
    // value leaves the streaming callback so subsequent weights do not
    // accumulate a checkpoint-sized dense graph.
    let mut arrays = vec![&quantized.weight, &quantized.scales];
    if let Some(biases) = &quantized.biases {
        arrays.push(biases);
    }
    async_eval_with_event(arrays)?.synchronize()?;
    load_array_for_parameter_strict(params, key, recipe.weight.clone(), quantized.weight, report);
    load_array_strict(params, recipe.scales.clone(), quantized.scales, report);
    if let Some(biases) = quantized.biases {
        let Some(biases_key) = &recipe.biases else {
            return Err(Error::ArchitectureModel(format!(
                "quantized load recipe for {:?} has no affine-bias destination",
                recipe.weight
            )));
        };
        load_array_strict(params, biases_key.clone(), biases, report);
    }
    Ok(())
}

#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
fn load_arrays_quantized_strict<M: PhysicalParameters>(
    model: &mut M,
    loaded: HashMap<String, Array>,
    quantization_stream: &Stream,
    quantization: WeightQuantization,
    recipes: &HashMap<String, QuantizedLoadRecipe>,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    quantization.validate()?;
    let mut params = model.parameters_mut().flatten();
    for (key, value) in loaded {
        load_array_quantized_strict(
            &mut params,
            key.clone(),
            value,
            quantization_stream,
            quantization,
            recipes.get(&key),
            report,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
