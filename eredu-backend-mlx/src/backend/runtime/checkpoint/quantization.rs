//! Execution of architecture-owned dense SafeTensors quantization plans.
//!
//! This module owns MLX packing, streaming, and shard persistence. Exact
//! source eligibility, output identities, companion presence, and output
//! configuration come from `eredu-architectures` and are consumed literally.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    num::NonZeroI32,
    path::{Path, PathBuf},
};

use safemlx::{ops, Array, Stream};
use serde_json::{json, Value};

use eredu_architectures::checkpoint_conversion::{
    SafetensorsQuantizationPlan, SafetensorsQuantizationTarget,
};
#[cfg(test)]
use eredu_checkpoint::AffineQuantization;
use eredu_checkpoint::WeightQuantization;

use crate::{backend::error::Error, backend::runtime::checkpoint::load as weights};

/// Resolves an on-load request against checkpoint quantization metadata.
///
/// Returns `true` for a dense checkpoint that must be quantized and `false`
/// when a matching pre-quantized checkpoint should be loaded directly.
pub fn should_quantize_on_load(
    architecture: &str,
    existing: Option<WeightQuantization>,
    requested: WeightQuantization,
) -> Result<bool, Error> {
    requested.validate()?;
    match existing {
        None => Ok(true),
        Some(existing) if existing == requested => Ok(false),
        Some(existing) => Err(Error::Quantization(format!(
            "{architecture} checkpoint is already quantized as {existing:?}, requested {requested:?}; implicit dequantization and requantization is unsupported"
        ))),
    }
}

/// Architecture plan and output-sharding options for checkpoint conversion.
#[derive(Debug, Clone)]
pub struct CheckpointQuantizationOptions {
    /// Exact architecture-owned conversion decision.
    pub plan: SafetensorsQuantizationPlan,
    /// Maximum uncompressed tensor bytes accumulated before writing a shard.
    pub shard_size_bytes: usize,
}

impl CheckpointQuantizationOptions {
    /// Creates conversion options with the default 512 MiB shard bound.
    pub fn new(plan: SafetensorsQuantizationPlan) -> Self {
        Self {
            plan,
            shard_size_bytes: 512 * 1024 * 1024,
        }
    }

    fn validate(&self) -> Result<(), Error> {
        if self.shard_size_bytes == 0 {
            return Err(Error::Quantization(
                "shard_size_bytes must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// Tensors produced from one dense quantized matrix.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed unsigned-integer weights.
    pub weight: Array,
    /// Per-group scales.
    pub scales: Array,
    /// Per-group affine biases.
    pub biases: Option<Array>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuantizationMatrixGeometry {
    leading_size: NonZeroI32,
    input_dims: NonZeroI32,
}

impl QuantizationMatrixGeometry {
    fn from_shape(shape: &[i32]) -> Result<Self, Error> {
        let Some((&input_dims, leading_dims)) = shape.split_last() else {
            return Err(Error::Quantization(
                "quantization requires a tensor with at least two dimensions".into(),
            ));
        };
        if leading_dims.is_empty() {
            return Err(Error::Quantization(
                "quantization requires a tensor with at least two dimensions".into(),
            ));
        }
        let input_dims = positive_dimension(input_dims).ok_or_else(|| {
            Error::Quantization(format!(
                "quantization input dimension must be nonzero, got shape {shape:?}"
            ))
        })?;
        let leading_size = leading_dims.iter().try_fold(1_i32, |size, &dimension| {
            let dimension = positive_dimension(dimension).ok_or_else(|| {
                Error::Quantization(format!(
                    "quantization leading dimensions must be nonzero, got shape {shape:?}"
                ))
            })?;
            size.checked_mul(dimension.get()).ok_or_else(|| {
                Error::Quantization(format!(
                    "quantization leading geometry cannot be flattened into an MLX matrix: {shape:?}"
                ))
            })
        })?;
        let leading_size = NonZeroI32::new(leading_size).ok_or_else(|| {
            Error::Quantization("quantization leading geometry must be nonzero".into())
        })?;
        Ok(Self {
            leading_size,
            input_dims,
        })
    }
}

fn positive_dimension(dimension: i32) -> Option<NonZeroI32> {
    NonZeroI32::new(dimension).filter(|_| dimension > 0)
}

fn packed_dimension(input_dims: NonZeroI32, bits: NonZeroI32) -> Result<NonZeroI32, Error> {
    i64::from(input_dims.get())
        .checked_mul(i64::from(bits.get()))
        .and_then(|bits| bits.checked_div(32))
        .and_then(|values| i32::try_from(values).ok())
        .and_then(NonZeroI32::new)
        .ok_or_else(|| Error::Quantization("quantized packed dimension overflow".into()))
}

impl QuantizedTensor {
    /// Associates packed arrays with the architecture-declared identities.
    pub fn into_named_arrays(
        self,
        target: &SafetensorsQuantizationTarget,
    ) -> Result<Vec<(String, Array)>, Error> {
        let mut arrays = vec![
            (target.weight_name().to_owned(), self.weight),
            (target.scales_name().to_owned(), self.scales),
        ];
        match (self.biases, target.biases_name()) {
            (Some(biases), Some(name)) => arrays.push((name.to_owned(), biases)),
            (None, None) => {}
            (Some(_), None) => {
                return Err(Error::Quantization(format!(
                    "architecture target {:?} omits the affine-bias output produced by its encoding",
                    target.source_name()
                )))
            }
            (None, Some(name)) => {
                return Err(Error::Quantization(format!(
                    "architecture target {:?} requires affine-bias output {name:?}, but its encoding produced none",
                    target.source_name()
                )))
            }
        }
        Ok(arrays)
    }
}

/// Quantizes one floating-point weight using an explicit execution stream.
///
/// The last dimension is grouped and packed. Leading dimensions, including an
/// expert-bank dimension, are retained. Both on-the-fly model loading and
/// checkpoint conversion call this function.
pub fn quantize_tensor(
    weight: &Array,
    config: impl Into<WeightQuantization>,
    stream: &Stream,
) -> Result<QuantizedTensor, Error> {
    let config = config.into();
    config.validate()?;
    let mode = mlx_quantization_mode(config)?;
    if weight.ndim() < 2 || !weight.dtype().is_float() {
        return Err(Error::Quantization(format!(
            "expected a floating-point weight with at least two dimensions, got shape {:?} and dtype {:?}",
            weight.shape(),
            weight.dtype()
        )));
    }
    let geometry = QuantizationMatrixGeometry::from_shape(weight.shape())?;
    let input_dims = geometry.input_dims;
    let group_size = positive_dimension(config.group_size())
        .ok_or_else(|| Error::Quantization("quantization group size must be positive".into()))?;
    let bits = positive_dimension(config.bits())
        .ok_or_else(|| Error::Quantization("quantization bit width must be positive".into()))?;
    if input_dims.get() % group_size.get() != 0 || input_dims.get() % 32 != 0 {
        return Err(Error::Quantization(format!(
            "input dimension {} must be divisible by group_size {} and 32",
            input_dims, group_size
        )));
    }
    let original_shape = weight.shape();
    let matrix = if weight.ndim() == 2 {
        weight.clone()
    } else {
        weight.reshape(&[geometry.leading_size.get(), input_dims.get()], stream)?
    };
    let packed_dims = packed_dimension(input_dims, bits)?;
    let group_dims = NonZeroI32::new(input_dims.get() / group_size.get()).ok_or_else(|| {
        Error::Quantization("quantization group dimension must be nonzero".into())
    })?;
    let arrays = ops::quantize_with_mode(&matrix, group_size.get(), bits.get(), mode, stream)?;
    let restore_shape = |array: Array, last_dim: NonZeroI32| -> Result<Array, Error> {
        if weight.ndim() == 2 {
            Ok(array)
        } else {
            let mut shape = original_shape[..original_shape.len() - 1].to_vec();
            shape.push(last_dim.get());
            Ok(array.reshape(&shape, stream)?)
        }
    };
    Ok(QuantizedTensor {
        weight: restore_shape(arrays.weight, packed_dims)?,
        scales: restore_shape(arrays.scales, group_dims)?,
        biases: arrays
            .biases
            .map(|biases| restore_shape(biases, group_dims))
            .transpose()?,
    })
}

/// Maps a producible checkpoint quantization format to its MLX operator mode.
///
/// Checkpoint-native GGUF blocks are valid stored formats, but dense MLX
/// quantization cannot produce them.
pub fn mlx_quantization_mode(config: WeightQuantization) -> Result<ops::QuantizationMode, Error> {
    match config {
        WeightQuantization::Affine(_) => Ok(ops::QuantizationMode::Affine),
        WeightQuantization::MxFp4 => Ok(ops::QuantizationMode::MxFp4),
        WeightQuantization::GgufIQuant { .. } => Err(Error::Quantization(
            "checkpoint-native GGUF blocks cannot be produced by dense quantization".into(),
        )),
    }
}

/// Summary returned after converting and saving a checkpoint directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointQuantizationReport {
    /// Number of source matrices converted to packed tensors.
    pub quantized_tensors: usize,
    /// Number of source tensors copied without conversion.
    pub copied_tensors: usize,
    /// Number of output safetensors shards.
    pub shards: usize,
    /// Uncompressed bytes represented by all output tensors.
    pub total_size: usize,
}

struct PendingShard {
    arrays: HashMap<String, Array>,
    bytes: usize,
}

impl PendingShard {
    fn new() -> Self {
        Self {
            arrays: HashMap::new(),
            bytes: 0,
        }
    }

    fn insert(&mut self, name: String, array: Array) {
        self.bytes += array.nbytes();
        self.arrays.insert(name, array);
    }
}

/// Executes an architecture-owned conversion plan and saves the checkpoint.
///
/// The source directory may contain a single `model.safetensors` file or a
/// Hugging Face sharded checkpoint index. Non-weight files are copied, while
/// `config.json` is replaced by the exact architecture-authored output value.
pub(crate) fn quantize_checkpoint(
    source_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    options: &CheckpointQuantizationOptions,
    stream: &Stream,
) -> Result<CheckpointQuantizationReport, Error> {
    options.validate()?;
    let source_dir = source_dir.as_ref();
    let output_dir = output_dir.as_ref();
    if !source_dir.is_dir() {
        return Err(Error::Quantization(format!(
            "source is not a directory: {}",
            source_dir.display()
        )));
    }
    fs::create_dir(output_dir).map_err(|error| {
        Error::Quantization(format!(
            "could not create empty output directory {}: {error}",
            output_dir.display()
        ))
    })?;

    let result = quantize_checkpoint_inner(source_dir, output_dir, options, stream);
    if result.is_err() {
        // The directory was created by this call and contains only partial output.
        let _ = fs::remove_dir_all(output_dir);
    }
    result
}

fn quantize_checkpoint_inner(
    source_dir: &Path,
    output_dir: &Path,
    options: &CheckpointQuantizationOptions,
    stream: &Stream,
) -> Result<CheckpointQuantizationReport, Error> {
    let weight_files = weights::safetensors_files(source_dir)?;
    copy_checkpoint_assets(source_dir, output_dir, &weight_files)?;
    write_planned_config(output_dir, options.plan.output_config())?;

    let targets = options
        .plan
        .targets()
        .iter()
        .map(|target| (target.source_name(), target))
        .collect::<BTreeMap<_, _>>();
    let mut converted_sources = std::collections::BTreeSet::new();
    let mut emitted_names = std::collections::BTreeSet::new();

    let mut pending = PendingShard::new();
    let mut temporary_shards = Vec::new();
    let mut locations = BTreeMap::<String, usize>::new();
    let mut quantized_tensors = 0;
    let mut copied_tensors = 0;
    let mut total_size = 0;

    for file in weight_files {
        crate::backend::runtime::checkpoint::load::for_each_safetensor_array(
            file,
            stream,
            |name, tensor| {
                let arrays = if let Some(target) = targets.get(name.as_str()) {
                    quantized_tensors += 1;
                    converted_sources.insert(name.clone());
                    quantize_tensor(&tensor, options.plan.quantization(), stream)?
                        .into_named_arrays(target)?
                } else {
                    copied_tensors += 1;
                    vec![(name, tensor)]
                };

                let incoming_bytes = arrays
                    .iter()
                    .map(|(_, array)| array.nbytes())
                    .sum::<usize>();
                if !pending.arrays.is_empty()
                    && pending.bytes.saturating_add(incoming_bytes) > options.shard_size_bytes
                {
                    flush_temporary_shard(
                        output_dir,
                        &mut pending,
                        &mut temporary_shards,
                        &mut locations,
                    )?;
                }
                for (name, array) in arrays {
                    if !emitted_names.insert(name.clone()) {
                        return Err(Error::Quantization(format!(
                            "architecture quantization output {name:?} collides with another checkpoint tensor"
                        )));
                    }
                    total_size += array.nbytes();
                    pending.insert(name, array);
                }
                Ok(())
            },
        )?;
    }
    let missing = targets
        .keys()
        .filter(|name| !converted_sources.contains(**name))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(Error::Quantization(format!(
            "checkpoint is missing architecture quantization targets: {}",
            missing
                .into_iter()
                .map(|name| format!("{name:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if !pending.arrays.is_empty() {
        flush_temporary_shard(
            output_dir,
            &mut pending,
            &mut temporary_shards,
            &mut locations,
        )?;
    }
    if temporary_shards.is_empty() {
        return Err(Error::Quantization("checkpoint contains no tensors".into()));
    }

    finalize_shards(output_dir, &temporary_shards, &locations, total_size)?;
    Ok(CheckpointQuantizationReport {
        quantized_tensors,
        copied_tensors,
        shards: temporary_shards.len(),
        total_size,
    })
}

fn flush_temporary_shard(
    output_dir: &Path,
    pending: &mut PendingShard,
    temporary_shards: &mut Vec<PathBuf>,
    locations: &mut BTreeMap<String, usize>,
) -> Result<(), Error> {
    let shard_index = temporary_shards.len();
    let path = output_dir.join(format!(".quantized-{shard_index:05}.safetensors"));
    Array::save_safetensors(pending.arrays.iter(), None, &path)?;
    for name in pending.arrays.keys() {
        locations.insert(name.clone(), shard_index);
    }
    pending.arrays.clear();
    pending.bytes = 0;
    temporary_shards.push(path);
    Ok(())
}

fn finalize_shards(
    output_dir: &Path,
    temporary_shards: &[PathBuf],
    locations: &BTreeMap<String, usize>,
    total_size: usize,
) -> Result<(), Error> {
    if temporary_shards.len() == 1 {
        fs::rename(&temporary_shards[0], output_dir.join("model.safetensors"))?;
        return Ok(());
    }

    let count = temporary_shards.len();
    let mut shard_names = Vec::with_capacity(count);
    for (index, temporary) in temporary_shards.iter().enumerate() {
        let name = format!("model-{:05}-of-{count:05}.safetensors", index + 1);
        fs::rename(temporary, output_dir.join(&name))?;
        shard_names.push(name);
    }
    let weight_map = locations
        .iter()
        .map(|(name, index)| (name.clone(), Value::String(shard_names[*index].clone())))
        .collect::<serde_json::Map<_, _>>();
    let index = json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map,
    });
    fs::write(
        output_dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    Ok(())
}

fn copy_checkpoint_assets(
    source_dir: &Path,
    output_dir: &Path,
    weight_files: &[PathBuf],
) -> Result<(), Error> {
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type()?.is_file() {
            continue;
        }
        let file_name = entry.file_name();
        let file_name_lossy = file_name.to_string_lossy();
        let is_weight_file = path
            .canonicalize()
            .is_ok_and(|canonical| weight_files.contains(&canonical));
        if file_name_lossy == "config.json"
            || file_name_lossy == "model.safetensors.index.json"
            || is_weight_file
        {
            continue;
        }
        fs::copy(path, output_dir.join(file_name))?;
    }
    Ok(())
}

fn write_planned_config(output_dir: &Path, config: &Value) -> Result<(), Error> {
    fs::write(
        output_dir.join("config.json"),
        serde_json::to_vec_pretty(config)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use crate::backend::ExecutionContext;
    use eredu_architectures::checkpoint_conversion::{
        SafetensorsQuantizationPlan, SafetensorsQuantizationTarget,
    };
    use eredu_gguf::{Endian, GgmlType};
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use safemlx::{Device, DeviceType, Dtype};

    use super::*;

    #[test]
    fn affine_config_uses_mlx_spelling() {
        let value = serde_json::to_value(AffineQuantization::default()).unwrap();
        assert_eq!(value["group_size"], 64);
        assert_eq!(value["bits"], 4);
        assert_eq!(value["mode"], "affine");
    }

    #[test]
    fn mxfp4_metadata_is_fixed_and_round_trips() {
        let value = serde_json::to_value(WeightQuantization::MxFp4).unwrap();
        assert_eq!(value, json!({"group_size": 32, "bits": 4, "mode": "mxfp4"}));
        assert_eq!(
            serde_json::from_value::<WeightQuantization>(value).unwrap(),
            WeightQuantization::MxFp4
        );
        assert!(serde_json::from_value::<WeightQuantization>(
            json!({"group_size": 64, "bits": 4, "mode": "mxfp4"})
        )
        .is_err());
        assert!(serde_json::from_value::<WeightQuantization>(
            json!({"group_size": 32, "bits": 8, "mode": "mxfp4"})
        )
        .is_err());
    }

    #[test]
    fn omitted_quantization_mode_defaults_to_affine() {
        let quantization =
            serde_json::from_value::<WeightQuantization>(json!({"group_size": 64, "bits": 4}))
                .unwrap();
        assert_eq!(
            quantization,
            WeightQuantization::Affine(AffineQuantization::new(64, 4).unwrap())
        );
    }

    #[test]
    fn valid_gguf_quantization_is_not_a_producible_mlx_format() {
        let quantization = WeightQuantization::GgufIQuant {
            ggml_type: GgmlType::Q4_0,
            endian: Endian::Little,
        };
        quantization.validate().unwrap();

        let error = mlx_quantization_mode(quantization).unwrap_err();
        assert!(matches!(error, Error::Quantization(_)));
        assert!(error
            .to_string()
            .contains("cannot be produced by dense quantization"));
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn quantize_tensor_returns_an_error_for_valid_gguf_quantization() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weight = Array::from_slice(&[0.25_f32; 64], &[2, 32]);
        let quantization = WeightQuantization::GgufIQuant {
            ggml_type: GgmlType::Q4_0,
            endian: Endian::Little,
        };

        let error = quantize_tensor(&weight, quantization, context.stream()).unwrap_err();
        assert!(matches!(error, Error::Quantization(_)));
        assert!(error
            .to_string()
            .contains("cannot be produced by dense quantization"));
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn quantize_tensor_rejects_zero_width_without_panicking() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weight = Array::from_slice(&[] as &[f32], &[2, 0]);

        let error =
            quantize_tensor(&weight, AffineQuantization::default(), context.stream()).unwrap_err();
        assert!(error
            .to_string()
            .contains("input dimension must be nonzero"));
    }

    #[test]
    fn planned_config_is_written_without_backend_metadata_rewrites() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "eredu-mlx-neutral-quantization-config-{}-{suffix}",
            std::process::id()
        ));
        let output = root.join("output");
        fs::create_dir_all(&output).unwrap();
        let planned = json!({
            "model_type": "opaque",
            "artifact_filename": "custom.safetensors",
            "architecture_quantization": {"encoding": "mxfp4"}
        });
        write_planned_config(&output, &planned).unwrap();
        let config: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(config["artifact_filename"], "custom.safetensors");
        assert_eq!(config, planned);
        assert!(config.get("quantization").is_none());
        assert!(config.get("quantization_config").is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn options_have_no_tensor_name_selection_policy() {
        let plan = SafetensorsQuantizationPlan::new(
            WeightQuantization::MxFp4,
            [SafetensorsQuantizationTarget::new(
                "architecture.exact.source",
                "architecture.exact.packed",
                "architecture.exact.scale",
                None::<String>,
            )],
            json!({"model_type": "test"}),
        )
        .unwrap();
        let options = CheckpointQuantizationOptions::new(plan);
        assert_eq!(options.plan.targets().len(), 1);
        assert_eq!(
            options.plan.targets()[0].source_name(),
            "architecture.exact.source"
        );
    }

    #[test]
    fn on_load_resolution_reuses_matching_metadata_and_rejects_mismatch() {
        let q4 = AffineQuantization::default();
        assert!(should_quantize_on_load("test", None, q4.into()).unwrap());
        assert!(!should_quantize_on_load("test", Some(q4.into()), q4.into()).unwrap());

        let q8 = AffineQuantization::new(64, 8).unwrap();
        let error = should_quantize_on_load("test", Some(q4.into()), q8.into()).unwrap_err();
        assert!(error.to_string().contains("already quantized"));
        assert!(error.to_string().contains("implicit dequantization"));
        assert!(!should_quantize_on_load(
            "test",
            Some(WeightQuantization::MxFp4),
            WeightQuantization::MxFp4
        )
        .unwrap());
    }

    #[test]
    fn affine_config_accepts_mlx_non_power_of_two_widths() {
        assert!(AffineQuantization::new(32, 3).is_ok());
        assert!(AffineQuantization::new(32, 5).is_ok());
        assert!(AffineQuantization::new(32, 6).is_ok());
        assert!(AffineQuantization::new(32, 7).is_err());
    }

    #[test]
    fn quantization_geometry_rejects_empty_dimensions() {
        let error = QuantizationMatrixGeometry::from_shape(&[2, 0]).unwrap_err();
        assert!(error
            .to_string()
            .contains("input dimension must be nonzero"));

        let error = QuantizationMatrixGeometry::from_shape(&[0, 2, 32]).unwrap_err();
        assert!(error
            .to_string()
            .contains("leading dimensions must be nonzero"));
    }

    #[test]
    fn quantization_geometry_preserves_large_expert_banks() {
        let shape = [256, 65_536, 65_536];
        let geometry = QuantizationMatrixGeometry::from_shape(&shape).unwrap();

        assert!(
            shape
                .iter()
                .map(|&dimension| i64::from(dimension))
                .product::<i64>()
                > i64::from(i32::MAX)
        );
        assert_eq!(geometry.leading_size.get(), 16_777_216);
        assert_eq!(geometry.input_dims.get(), 65_536);
    }

    #[test]
    fn quantization_geometry_rejects_unrepresentable_flattened_rows() {
        let error = QuantizationMatrixGeometry::from_shape(&[46_341, 46_341, 32]).unwrap_err();
        assert!(error.to_string().contains("cannot be flattened"));
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn mxfp4_quantizes_rank_three_expert_banks() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let experts = Array::from_slice(&vec![0.25f32; 3 * 8 * 64], &[3, 8, 64]);
        let quantized =
            quantize_tensor(&experts, WeightQuantization::MxFp4, context.stream()).unwrap();
        assert_eq!(quantized.weight.shape(), &[3, 8, 8]);
        assert_eq!(quantized.scales.shape(), &[3, 8, 2]);
        assert!(quantized.biases.is_none());
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn saved_mxfp4_checkpoint_has_no_affine_bias_tensors() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "eredu-mlx-mxfp4-save-test-{}-{suffix}",
            std::process::id()
        ));
        let source = root.join("source");
        let output = root.join("output");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("config.json"), br#"{"model_type":"test"}"#).unwrap();
        let weight = Array::from_slice(&vec![0.25f32; 2 * 64], &[2, 64]);
        Array::save_safetensors(
            [("model.proj.weight", &weight)],
            None,
            source.join("model.safetensors"),
        )
        .unwrap();

        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let plan = SafetensorsQuantizationPlan::new(
            WeightQuantization::MxFp4,
            [SafetensorsQuantizationTarget::new(
                "model.proj.weight",
                "model.proj.weight",
                "model.proj.scales",
                None::<String>,
            )],
            json!({
                "model_type": "test",
                "architecture_quantization": {"mode": "mxfp4"}
            }),
        )
        .unwrap();
        let options = CheckpointQuantizationOptions::new(plan);
        quantize_checkpoint(&source, &output, &options, stream).unwrap();
        let arrays =
            Array::load_safetensors(output.join("model.safetensors"), weights_context.stream())
                .unwrap();
        assert!(arrays.contains_key("model.proj.weight"));
        assert!(arrays.contains_key("model.proj.scales"));
        assert!(!arrays.contains_key("model.proj.biases"));
        let config: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(config["architecture_quantization"]["mode"], "mxfp4");
        assert!(config.get("quantization").is_none());
        assert!(config.get("quantization_config").is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn saved_checkpoint_matches_direct_tensor_quantization() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "eredu-mlx-quantization-test-{}-{suffix}",
            std::process::id()
        ));
        let source = root.join("source");
        let output = root.join("output");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("config.json"), br#"{"model_type":"test"}"#).unwrap();

        let values = (0..(8 * 64))
            .map(|index| (index as f32 - 255.5) / 64.0)
            .collect::<Vec<_>>();
        let weight = Array::from_slice(&values, &[8, 64]);
        let embedding = Array::from_slice(&vec![1.0f32; 3 * 64], &[3, 64]);
        Array::save_safetensors(
            [
                ("model.proj.weight", &weight),
                ("model.embed_tokens.weight", &embedding),
            ],
            None,
            source.join("model.safetensors"),
        )
        .unwrap();
        let norm = Array::from_slice(&vec![1.0f32; 64], &[64]);
        Array::save_safetensors(
            [("auxiliary.weight", &norm)],
            None,
            source.join("auxiliary.safetensors"),
        )
        .unwrap();

        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights_stream = weights_context.stream();
        let expected = quantize_tensor(&weight, AffineQuantization::default(), stream).unwrap();
        let plan = SafetensorsQuantizationPlan::new(
            AffineQuantization::default(),
            [SafetensorsQuantizationTarget::new(
                "model.proj.weight",
                "runtime.projection.packed",
                "runtime.projection.scale",
                Some("runtime.projection.zero"),
            )],
            json!({
                "model_type": "test",
                "architecture_quantization": {"mode": "affine"}
            }),
        )
        .unwrap();
        let mut options = CheckpointQuantizationOptions::new(plan);
        options.shard_size_bytes = 1;
        let report = quantize_checkpoint(&source, &output, &options, stream).unwrap();
        assert_eq!(report.quantized_tensors, 1);
        assert_eq!(report.copied_tensors, 1);
        assert_eq!(report.shards, 2);

        let mut saved = HashMap::new();
        for file in weights::safetensors_files(&output).unwrap() {
            saved.extend(Array::load_safetensors(file, weights_stream).unwrap());
        }
        let saved_weight = &saved["runtime.projection.packed"];
        assert_eq!(saved_weight.dtype(), Dtype::Uint32);
        assert_eq!(
            saved_weight.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
        assert_eq!(
            saved["runtime.projection.scale"]
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            expected.scales.evaluated().unwrap().as_slice::<f32>()
        );
        assert_eq!(
            saved["runtime.projection.zero"]
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
        assert_eq!(
            saved["model.embed_tokens.weight"]
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            embedding.evaluated().unwrap().as_slice::<f32>()
        );

        let config: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(config["architecture_quantization"]["mode"], "affine");
        assert!(config.get("quantization").is_none());
        assert!(config.get("quantization_config").is_none());
        assert!(output.join("auxiliary.safetensors").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
