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

use crate::backend::error::Error;

pub(super) mod io;
use io::{for_each_safetensor_array, safetensors_files};

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
/// leading bank dimensions are retained. Both on-the-fly model loading and
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
    let weight_files = safetensors_files(source_dir)?;
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
        for_each_safetensor_array(file, stream, |name, tensor| {
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
        })?;
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
mod tests;
