use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype, SafeTensorError, SafeTensors,
};
use serde_json::json;

use super::{SafetensorsQuantizationPlan, SafetensorsQuantizationTarget};
use crate::{
    safetensors::{checkpoint_member_path, SafetensorsShardError, SafetensorsShards},
    store::StoreError,
    WeightQuantization,
};

struct CopiedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl CopiedTensor {
    fn copy(view: TensorView<'_>) -> Self {
        Self {
            dtype: view.dtype(),
            shape: view.shape().to_vec(),
            bytes: view.data().to_vec(),
        }
    }

    fn view(&self) -> Result<TensorView<'_>, SafeTensorError> {
        TensorView::new(self.dtype, self.shape.clone(), &self.bytes)
    }
}

/// Physical tensors produced by packing one dense weight.
pub struct PackedSafetensorsWeight<T> {
    /// Packed weight payload.
    pub weight: T,
    /// Per-group scales.
    pub scales: T,
    /// Per-group affine biases, when present.
    pub biases: Option<T>,
}

impl<T> PackedSafetensorsWeight<T> {
    fn named(
        self,
        target: &SafetensorsQuantizationTarget,
    ) -> Result<Vec<(String, T)>, CheckpointConversionError> {
        let mut tensors = vec![
            (target.weight_name().into(), self.weight),
            (target.scales_name().into(), self.scales),
        ];
        match (target.biases_name(), self.biases) {
            (Some(name), Some(value)) => tensors.push((name.into(), value)),
            (None, None) => {}
            _ => {
                return Err(CheckpointConversionError::BiasMismatch(
                    target.source_name().into(),
                ))
            }
        }
        Ok(tensors)
    }
}

/// Backend operations for packing tensors selected by an offline conversion plan.
pub trait SafetensorsWeightPacker {
    /// A packed tensor retained until the checkpoint writer needs its bytes.
    type Pending;
    /// Host-readable backing for a packed tensor.
    type Ready;
    /// Failure to pack or read back a tensor.
    type Error: Error + Send + Sync + 'static;

    /// Stages packing for one selected dense weight.
    fn pack(
        &mut self,
        weight: TensorView<'_>,
        quantization: WeightQuantization,
    ) -> Result<PackedSafetensorsWeight<Self::Pending>, Self::Error>;

    /// Returns the exact encoded byte length without waiting for packing.
    fn packed_bytes(&self, tensor: &Self::Pending) -> usize;

    /// Makes one output shard's packed tensors readable in input order.
    fn materialize(&mut self, tensors: Vec<Self::Pending>)
        -> Result<Vec<Self::Ready>, Self::Error>;

    /// Borrows the little-endian tensor payload for serialization.
    fn view<'a>(&self, tensor: &'a Self::Ready) -> Result<TensorView<'a>, Self::Error>;
}

enum ShardTensor {
    Copied(CopiedTensor),
    Packed { index: usize, bytes: usize },
}

/// Exact conversion plan and output shard sizing.
pub struct CheckpointQuantizationOptions {
    /// Validated source and output tensor identities.
    pub plan: SafetensorsQuantizationPlan,
    /// Target maximum bytes per shard. A single tensor group can exceed it.
    pub shard_size_bytes: usize,
}

impl CheckpointQuantizationOptions {
    /// Uses the default 512 MiB shard size.
    pub fn new(plan: SafetensorsQuantizationPlan) -> Self {
        Self {
            plan,
            shard_size_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Summary of the published checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointQuantizationReport {
    /// Number of source matrices converted.
    pub quantized_tensors: usize,
    /// Number of source tensors copied byte for byte.
    pub copied_tensors: usize,
    /// Number of output SafeTensors shards.
    pub shards: usize,
    /// Total output tensor bytes excluding headers.
    pub total_size: usize,
}

/// Failure while producing a SafeTensors checkpoint.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointConversionError {
    /// The source path must be a directory.
    #[error("checkpoint source is not a directory: {0}")]
    SourceDirectory(PathBuf),
    /// The target shard size must be positive.
    #[error("shard size must be greater than zero")]
    ZeroShardSize,
    /// The admitted source does not contain a selected matrix.
    #[error("missing checkpoint quantization targets: {0:?}")]
    MissingTargets(Vec<String>),
    /// Two source entries produce the same output identity.
    #[error("duplicate output tensor {0:?}")]
    DuplicateOutput(String),
    /// The packed payload disagrees with the plan's required bias companion.
    #[error("packed bias presence disagrees with the conversion target for {0:?}")]
    BiasMismatch(String),
    /// A copied asset would overwrite a generated checkpoint file.
    #[error("checkpoint asset collides with generated output: {0}")]
    AssetCollision(PathBuf),
    /// Output size cannot be represented by the host.
    #[error("checkpoint output byte count overflow")]
    SizeOverflow,
    /// The backend could not pack a selected tensor.
    #[error("packing checkpoint tensor {tensor:?} failed")]
    Pack {
        /// Source tensor identity.
        tensor: String,
        /// Packing failure.
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
    /// The backend could not make the packed batch available for writing.
    #[error("reading packed checkpoint tensors failed")]
    ReadPacked(#[source] Box<dyn Error + Send + Sync>),
    /// The backend returned a different number of tensors than it staged.
    #[error("packed checkpoint batch returned {actual} tensors, expected {expected}")]
    PackedCount {
        /// Number of staged packed tensors.
        expected: usize,
        /// Number of readable packed tensors.
        actual: usize,
    },
    /// The backend returned a tensor whose encoded size disagrees with its staging metadata.
    #[error("packed checkpoint tensor {tensor:?} contains {actual} bytes, expected {expected}")]
    PackedSize {
        /// Output tensor identity.
        tensor: String,
        /// Staged byte length.
        expected: usize,
        /// Readable byte length.
        actual: usize,
    },
    /// Source shard discovery failed.
    #[error(transparent)]
    Shards(#[from] SafetensorsShardError),
    /// Payload access failed an admission check.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// Tensor encoding or serialization failed.
    #[error(transparent)]
    Safetensors(#[from] SafeTensorError),
    /// Filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Configuration or index serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Applies an exact SafeTensors conversion plan to a checkpoint directory.
///
/// The packer receives only selected tensor views and their encoding. All other
/// tensor payloads are copied unchanged. A failed conversion removes the
/// output directory created by this call.
pub fn quantize_checkpoint<P: SafetensorsWeightPacker>(
    source_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    options: &CheckpointQuantizationOptions,
    packer: &mut P,
) -> Result<CheckpointQuantizationReport, CheckpointConversionError> {
    let source_dir = source_dir.as_ref();
    let output_dir = output_dir.as_ref();
    if options.shard_size_bytes == 0 {
        return Err(CheckpointConversionError::ZeroShardSize);
    }
    if !source_dir.is_dir() {
        return Err(CheckpointConversionError::SourceDirectory(
            source_dir.into(),
        ));
    }
    let shards = SafetensorsShards::discover(source_dir)?;
    fs::create_dir(output_dir)?;
    let result = write_checkpoint(source_dir, output_dir, &shards, options, packer);
    if result.is_err() {
        let _ = fs::remove_dir_all(output_dir);
    }
    result
}

fn write_checkpoint<P: SafetensorsWeightPacker>(
    source_dir: &Path,
    output_dir: &Path,
    shards: &SafetensorsShards,
    options: &CheckpointQuantizationOptions,
    packer: &mut P,
) -> Result<CheckpointQuantizationReport, CheckpointConversionError> {
    let mut targets: BTreeMap<_, _> = options
        .plan
        .targets()
        .iter()
        .map(|target| (target.source_name(), target))
        .collect();
    let mut pending = PendingShard {
        tensors: Vec::new(),
        packed: Vec::new(),
        bytes: 0,
    };
    let mut locations = BTreeMap::new();
    let mut names = BTreeSet::new();
    let mut temporary = Vec::new();
    let mut total_size = 0usize;
    let mut quantized_tensors = 0;
    let mut copied_tensors = 0;

    for path in shards.payload_paths() {
        let bytes = shards.admission(path).file.read_all(path)?;
        let source = SafeTensors::deserialize(&bytes)?;
        let mut tensors: Vec<_> = source.iter().collect();
        tensors.sort_unstable_by_key(|(name, _)| *name);
        for (name, view) in tensors {
            let packed = targets
                .remove(name)
                .map(|target| {
                    let packed = packer
                        .pack(view.clone(), options.plan.quantization())
                        .map_err(|source| CheckpointConversionError::Pack {
                            tensor: name.into(),
                            source: Box::new(source),
                        })?;
                    packed.named(target).map(|tensors| {
                        tensors
                            .into_iter()
                            .map(|(name, tensor)| {
                                let bytes = packer.packed_bytes(&tensor);
                                (name, tensor, bytes)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .transpose()?;
            let incoming = match &packed {
                Some(tensors) => tensors.iter().try_fold(0usize, |bytes, (_, _, size)| {
                    bytes
                        .checked_add(*size)
                        .ok_or(CheckpointConversionError::SizeOverflow)
                })?,
                None => view.data().len(),
            };
            if !pending.tensors.is_empty()
                && pending.bytes.saturating_add(incoming) > options.shard_size_bytes
            {
                pending.write(output_dir, &mut temporary, &mut locations, packer)?;
            }
            let mut append = |name: String, tensor| {
                if !names.insert(name.clone()) {
                    return Err(CheckpointConversionError::DuplicateOutput(name));
                }
                pending.tensors.push((name, tensor));
                Ok(())
            };
            if let Some(tensors) = packed {
                quantized_tensors += 1;
                for (name, tensor, bytes) in tensors {
                    let index = pending.packed.len();
                    pending.packed.push(tensor);
                    append(name, ShardTensor::Packed { index, bytes })?;
                }
            } else {
                copied_tensors += 1;
                append(name.into(), ShardTensor::Copied(CopiedTensor::copy(view)))?;
            }
            pending.bytes += incoming;
            total_size = total_size
                .checked_add(incoming)
                .ok_or(CheckpointConversionError::SizeOverflow)?;
        }
    }
    if !targets.is_empty() {
        return Err(CheckpointConversionError::MissingTargets(
            targets.into_keys().map(str::to_owned).collect(),
        ));
    }
    if !pending.tensors.is_empty() {
        pending.write(output_dir, &mut temporary, &mut locations, packer)?;
    }
    finalize_shards(output_dir, &temporary, &locations, total_size)?;
    fs::write(
        output_dir.join("config.json"),
        serde_json::to_vec_pretty(options.plan.output_config())?,
    )?;
    copy_checkpoint_assets(source_dir, output_dir, shards.payload_paths())?;
    Ok(CheckpointQuantizationReport {
        quantized_tensors,
        copied_tensors,
        shards: temporary.len(),
        total_size,
    })
}

struct PendingShard<T> {
    tensors: Vec<(String, ShardTensor)>,
    packed: Vec<T>,
    bytes: usize,
}

impl<T> PendingShard<T> {
    fn write(
        &mut self,
        output_dir: &Path,
        temporary: &mut Vec<PathBuf>,
        locations: &mut BTreeMap<String, usize>,
        packer: &mut impl SafetensorsWeightPacker<Pending = T>,
    ) -> Result<(), CheckpointConversionError> {
        let expected = self.packed.len();
        let packed = if expected == 0 {
            Vec::new()
        } else {
            packer
                .materialize(std::mem::take(&mut self.packed))
                .map_err(|error| CheckpointConversionError::ReadPacked(Box::new(error)))?
        };
        if packed.len() != expected {
            return Err(CheckpointConversionError::PackedCount {
                expected,
                actual: packed.len(),
            });
        }
        let mut views = Vec::with_capacity(self.tensors.len());
        for (name, tensor) in &self.tensors {
            let view = match tensor {
                ShardTensor::Copied(tensor) => tensor.view()?,
                ShardTensor::Packed { index, bytes } => {
                    let view = packer
                        .view(&packed[*index])
                        .map_err(|error| CheckpointConversionError::ReadPacked(Box::new(error)))?;
                    let actual = view.data().len();
                    if actual != *bytes {
                        return Err(CheckpointConversionError::PackedSize {
                            tensor: name.clone(),
                            expected: *bytes,
                            actual,
                        });
                    }
                    view
                }
            };
            views.push((name.as_str(), view));
        }
        let index = temporary.len();
        let path = output_dir.join(format!(".quantized-{index:05}.safetensors"));
        serialize_to_file(views, None, &path)?;
        for (name, _) in self.tensors.drain(..) {
            locations.insert(name, index);
        }
        temporary.push(path);
        self.bytes = 0;
        Ok(())
    }
}

fn finalize_shards(
    output_dir: &Path,
    temporary: &[PathBuf],
    locations: &BTreeMap<String, usize>,
    total_size: usize,
) -> Result<(), CheckpointConversionError> {
    if let [path] = temporary {
        fs::rename(path, output_dir.join("model.safetensors"))?;
        return Ok(());
    }
    let count = temporary.len();
    let mut members = Vec::with_capacity(count);
    for (index, path) in temporary.iter().enumerate() {
        let name = format!("model-{:05}-of-{count:05}.safetensors", index + 1);
        fs::rename(path, output_dir.join(&name))?;
        members.push(name);
    }
    let weight_map: BTreeMap<_, _> = locations
        .iter()
        .map(|(name, &index)| (name, &members[index]))
        .collect();
    let index = json!({ "metadata": { "total_size": total_size }, "weight_map": weight_map });
    fs::write(
        output_dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    Ok(())
}

fn copy_checkpoint_assets(
    source_dir: &Path,
    output_dir: &Path,
    payloads: &[PathBuf],
) -> Result<(), CheckpointConversionError> {
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        if matches!(
            file_name.to_str(),
            Some("config.json" | "model.safetensors.index.json")
        ) {
            continue;
        }
        let kind = entry.file_type()?;
        if !kind.is_file() && !kind.is_symlink() {
            continue;
        }
        let path = checkpoint_member_path(source_dir, Path::new(&file_name))?;
        if payloads.contains(&path) || !path.metadata()?.is_file() {
            continue;
        }
        let destination = output_dir.join(file_name);
        if destination.try_exists()? {
            return Err(CheckpointConversionError::AssetCollision(path));
        }
        fs::copy(path, destination)?;
    }
    Ok(())
}
