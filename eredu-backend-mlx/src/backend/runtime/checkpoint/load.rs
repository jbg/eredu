use eredu_checkpoint::AffineQuantization;

use eredu_checkpoint::WeightQuantization;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use eredu_gguf::MetadataValue as GgufMetadataValue;
use safemlx::{Array, Stream};

use crate::backend::error::Error;
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
use crate::{
    backend::runtime::checkpoint::gguf::GgufCheckpoint,
    native_quantization::NativeQuantizationFormat,
};
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
use safemlx::transforms::async_eval_with_event;
use safetensors::SafeTensors;
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
use std::collections::HashSet;

/// Copies decoded GGUF metadata into a name-addressable map.
pub(crate) fn gguf_metadata(checkpoint: &GgufCheckpoint) -> HashMap<String, GgufMetadataValue> {
    checkpoint
        .metadata()
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// Lowers affine GGUF encodings under an admitted canonical tensor mapping.
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

/// Visits every tensor in one safetensors file as an MLX-owned array.
pub(super) fn for_each_safetensor_array<F>(
    path: impl AsRef<Path>,
    stream: &Stream,
    mut f: F,
) -> Result<(), Error>
where
    F: FnMut(String, Array) -> Result<(), Error>,
{
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|err| Error::Other(Box::new(err)))?;

    for (key, view) in tensors.iter() {
        let value = Array::try_from(view).map_err(|err| Error::Other(Box::new(err)))?;
        let value = value.copy(stream)?;
        f(key.to_string(), value)?;
    }

    Ok(())
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

/// Returns the validated safetensors payloads referenced by a model directory.
pub(crate) fn safetensors_files(model_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>, Error> {
    Ok(eredu_checkpoint::safetensors::SafetensorsShards::discover(model_dir)?.into_payload_paths())
}

#[cfg(test)]
mod tests {
    use crate::backend::runtime::checkpoint::gguf::GgufCheckpoint;
    use std::collections::BTreeMap;
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use std::collections::HashMap;

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use eredu_backend_mlx_macros::PhysicalParameters;
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use eredu_checkpoint::AffineQuantization;
    use eredu_checkpoint::WeightQuantization;
    use eredu_gguf::{Endian, GgmlType, TensorInput, Writer, WriterOptions};
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use safemlx::{Array, Device, DeviceType, Dtype};

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use crate::{
        backend::ExecutionContext,
        module::{PhysicalParam, PhysicalParameters as _},
    };

    use super::{gguf_quantization_configs, safetensors_files};
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use super::{
        load_array_strict, load_arrays_quantized_strict, Error, QuantizedLoadRecipe,
        StrictLoadReport,
    };

    #[test]
    fn safetensors_file_discovery_rejects_untrusted_index_paths_and_duplicates() {
        let traversal = tempfile::tempdir().unwrap();
        std::fs::write(
            traversal.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
        )
        .unwrap();
        let traversal_error = safetensors_files(traversal.path()).unwrap_err();
        assert!(matches!(
            traversal_error,
            crate::backend::Error::CheckpointShards(
                eredu_checkpoint::safetensors::SafetensorsShardError::UnsafeShardPath { .. }
            )
        ));

        let duplicate = tempfile::tempdir().unwrap();
        std::fs::write(
            duplicate.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"weight":"one.safetensors","weight":"two.safetensors"}}"#,
        )
        .unwrap();
        let duplicate_error = safetensors_files(duplicate.path()).unwrap_err();
        assert!(matches!(
            duplicate_error,
            crate::backend::Error::CheckpointShards(
                eredu_checkpoint::safetensors::SafetensorsShardError::MalformedIndex { .. }
            )
        ));
    }

    #[cfg(unix)]
    #[test]
    fn safetensors_file_discovery_rejects_external_payload_symlinks() {
        let parent = tempfile::tempdir().unwrap();
        let checkpoint = parent.path().join("checkpoint");
        std::fs::create_dir(&checkpoint).unwrap();
        let outside = parent.path().join("outside.safetensors");
        std::fs::write(&outside, []).unwrap();
        std::os::unix::fs::symlink(&outside, checkpoint.join("linked.safetensors")).unwrap();
        std::fs::write(
            checkpoint.join("model.safetensors.index.json"),
            r#"{"weight_map":{"weight":"linked.safetensors"}}"#,
        )
        .unwrap();

        assert!(matches!(
            safetensors_files(&checkpoint),
            Err(crate::backend::Error::CheckpointShards(
                eredu_checkpoint::safetensors::SafetensorsShardError::UnsafeShardPath { .. }
            ))
        ));
    }

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

        let checkpoint = GgufCheckpoint::open(path).unwrap();
        let tensor_mapping = checkpoint
            .catalog()
            .translated_outputs(str::to_string)
            .unwrap();
        let configs = gguf_quantization_configs(&checkpoint, &tensor_mapping).unwrap();
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

        let checkpoint = GgufCheckpoint::open(path).unwrap();
        let tensor_mapping = checkpoint
            .catalog()
            .translated_outputs(str::to_string)
            .unwrap();
        let configs = gguf_quantization_configs(&checkpoint, &tensor_mapping).unwrap();
        assert!(matches!(
            configs["projection.weight"],
            WeightQuantization::Affine(config) if config.group_size == 32 && config.bits == 4
        ));
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[derive(Debug, Clone, PhysicalParameters)]
    struct PackedExperts {
        #[param]
        experts: PhysicalParam<Array>,
        #[param]
        experts_scales: PhysicalParam<Option<Array>>,
        #[param]
        experts_biases: PhysicalParam<Option<Array>>,
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[derive(Debug, Clone, PhysicalParameters)]
    struct ExactInnerWeight {
        #[param]
        projection: ExactInnerProjection,
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[derive(Debug, Clone, PhysicalParameters)]
    struct ExactInnerProjection {
        #[param]
        inner: ExactPackedLinear,
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[derive(Debug, Clone, PhysicalParameters)]
    struct ExactPackedLinear {
        #[param]
        weight: PhysicalParam<Array>,
        #[param]
        scales: PhysicalParam<Array>,
        #[param]
        biases: PhysicalParam<Option<Array>>,
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn strict_load_accepts_exact_inner_weight_identity() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut model = ExactInnerWeight {
            projection: ExactInnerProjection {
                inner: ExactPackedLinear {
                    weight: PhysicalParam::<Array>::unloaded(&[8, 64], Dtype::Float32, stream)
                        .unwrap(),
                    scales: PhysicalParam::<Array>::unloaded(&[8, 2], Dtype::Float32, stream)
                        .unwrap(),
                    biases: PhysicalParam::new(None),
                },
            },
        };
        let value = Array::from_slice(&vec![0.25f32; 8 * 64], &[8, 64]);
        let mut report = StrictLoadReport::default();
        load_array_strict(
            &mut model.parameters_mut().flatten(),
            "projection.inner.weight".into(),
            value,
            &mut report,
        );
        report
            .finish_excluding(&model, |name| name != "projection.inner.weight")
            .unwrap();
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn strict_load_does_not_translate_canonical_weight_to_private_slot() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut model = ExactInnerWeight {
            projection: ExactInnerProjection {
                inner: ExactPackedLinear {
                    weight: PhysicalParam::<Array>::unloaded(&[8, 64], Dtype::Float32, stream)
                        .unwrap(),
                    scales: PhysicalParam::<Array>::unloaded(&[8, 2], Dtype::Float32, stream)
                        .unwrap(),
                    biases: PhysicalParam::new(None),
                },
            },
        };
        let value = Array::from_slice(&vec![0.25f32; 8 * 64], &[8, 64]);
        let mut report = StrictLoadReport::default();
        load_array_strict(
            &mut model.parameters_mut().flatten(),
            "projection.weight".into(),
            value,
            &mut report,
        );
        assert!(matches!(
            report.finish_excluding(&model, |name| name != "projection.inner.weight"),
            Err(Error::StrictLoadValidation { missing, unused })
                if missing == ["projection.inner.weight"] && unused == ["projection.weight"]
        ));
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn named_array_quantization_packs_rank_three_experts() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut model = PackedExperts {
            experts: PhysicalParam::<Array>::unloaded(&[3, 8, 8], Dtype::Uint32, stream).unwrap(),
            experts_scales: PhysicalParam::<Option<Array>>::unloaded_some(
                &[3, 8, 2],
                Dtype::Uint8,
                stream,
            )
            .unwrap(),
            experts_biases: PhysicalParam::new(None),
        };
        let dense = Array::from_slice(&vec![0.25f32; 3 * 8 * 64], &[3, 8, 64]);
        let mut report = StrictLoadReport::default();
        load_arrays_quantized_strict(
            &mut model,
            HashMap::from([("experts".into(), dense)]),
            stream,
            WeightQuantization::MxFp4,
            &HashMap::from([(
                "experts".into(),
                QuantizedLoadRecipe::new("experts", "experts_scales", None),
            )]),
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

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn quantized_strict_load_preserves_exact_inner_weight_and_companions() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let quantization = AffineQuantization::default();
        let mut model = ExactInnerWeight {
            projection: ExactInnerProjection {
                inner: ExactPackedLinear {
                    weight: PhysicalParam::<Array>::unloaded(&[8, 8], Dtype::Uint32, stream)
                        .unwrap(),
                    scales: PhysicalParam::<Array>::unloaded(&[8, 1], Dtype::Uint8, stream)
                        .unwrap(),
                    biases: PhysicalParam::<Option<Array>>::unloaded_some(
                        &[8, 1],
                        Dtype::Float32,
                        stream,
                    )
                    .unwrap(),
                },
            },
        };
        let dense = Array::from_slice(&vec![0.25f32; 8 * 64], &[8, 64]);
        let mut report = StrictLoadReport::default();
        load_arrays_quantized_strict(
            &mut model,
            HashMap::from([("projection.inner.weight".into(), dense)]),
            stream,
            quantization.into(),
            &HashMap::from([(
                "projection.inner.weight".into(),
                QuantizedLoadRecipe::new(
                    "projection.inner.weight",
                    "projection.inner.scales",
                    Some("projection.inner.biases".into()),
                ),
            )]),
            &mut report,
        )
        .unwrap();
        report.finish(&model).unwrap();

        assert_eq!(model.projection.inner.weight.shape(), &[8, 8]);
        assert_eq!(model.projection.inner.scales.shape(), &[8, 1]);
        assert_eq!(
            model
                .projection
                .inner
                .biases
                .value
                .as_ref()
                .unwrap()
                .shape(),
            &[8, 1]
        );
    }
}
