//! One closed logical catalog compiler. Inputs remain owned on every refusal.
use super::catalog::{CatalogData, CatalogHandle, CatalogRows, UnclaimedRows};
use super::*;
use crate::prepared_index::{PreparedIndex, PreparedIndexNode};
use std::{alloc::Layout, any::Any, collections::TryReserveError, fmt, mem::size_of};

type Mapping<'a> = PreparedIndex<(&'a str, &'a str), &'a str, ()>;
#[derive(Debug, Clone, Copy, Default)]
struct Position {
    shard: usize,
    tensor: usize,
    output: usize,
}
#[derive(Debug)]
enum Issue {
    Empty,
    Limit,
    MappingDuplicate(usize),
    MappingMissing,
    Collision,
    PhysicalDimension,
    LogicalDimension,
    LogicalBytes,
    Projection,
    Block(eredu_gguf::Error),
    PhysicalBlocks { physical: usize, block: usize },
    LogicalBlocks { logical: usize, blocks: usize },
    Reserve(TryReserveError),
    Overflow,
}
#[derive(Debug)]
pub(super) struct WorkerFailure {
    position: Position,
    issue: Issue,
}
impl WorkerFailure {
    fn at(position: Position, issue: Issue) -> Self {
        Self { position, issue }
    }
    fn diagnostic_key<'b>(
        &self,
        checkpoint: &'b Checkpoint,
        mapping: &'b [eredu_gguf::TranslatedTensorLayout],
    ) -> &'b str {
        if let Issue::MappingDuplicate(i) = self.issue {
            return &mapping[i].layout.name;
        }
        let p = self.position;
        let Some(tensor) = checkpoint
            .shards()
            .get(p.shard)
            .and_then(|s| s.tensors().get(p.tensor))
        else {
            return "";
        };
        let physical = tensor.descriptor().name.as_str();
        let logical = tensor
            .outputs()
            .get(p.output)
            .map_or("", |o| o.name.as_str());
        match self.issue {
            Issue::Empty | Issue::Limit | Issue::Reserve(_) | Issue::Overflow => "",
            Issue::PhysicalDimension => physical,
            Issue::Collision => mapping
                .iter()
                .find(|m| m.physical_name == physical && m.original_name == logical)
                .map_or(logical, |m| m.layout.name.as_str()),
            _ => logical,
        }
    }
    fn display_with(
        &self,
        key: &str,
        checkpoint: &Checkpoint,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match &self.issue {
            Issue::Limit => return f.write_str("maximum cached-shard count must be nonzero"),
            Issue::Reserve(e) => return write!(f, "checkpoint store state is unavailable: {e}"),
            Issue::Overflow => {
                return f.write_str("checkpoint size overflow: GGUF catalog storage")
            }
            Issue::PhysicalDimension => {
                return write!(
                    f,
                    "checkpoint size overflow: GGUF physical shape for tensor {key:?}"
                )
            }
            Issue::LogicalDimension => {
                return write!(
                    f,
                    "checkpoint size overflow: GGUF logical shape for tensor {key:?}"
                )
            }
            Issue::LogicalBytes => {
                return write!(
                    f,
                    "checkpoint size overflow: GGUF logical byte length for tensor {key:?}"
                )
            }
            _ => {}
        }
        write!(f, "GGUF checkpoint operation failed for tensor {key:?}: ")?;
        match &self.issue {
            Issue::Empty=>f.write_str("GGUF logical catalog is empty"),
            Issue::MappingDuplicate(_)=>f.write_str("admitted GGUF tensor mapping contains a duplicate source output"),
            Issue::MappingMissing=>f.write_str("admitted GGUF tensor mapping omits a catalog output"),
            Issue::Collision=>f.write_str("translated logical tensor collides with an existing output"),
            Issue::Projection=>{
                struct Dims<'a>(&'a [u64],bool);
                impl fmt::Debug for Dims<'_>{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{let mut list=f.debug_list();if self.1{for &n in self.0.iter().rev(){list.entry(&(n as usize));}}else{for &n in self.0{list.entry(&(n as usize));}}list.finish()}}
                let p=self.position;let tensor=&checkpoint.shards()[p.shard].tensors()[p.tensor];let output=&tensor.outputs()[p.output];
                write!(f,"logical shape {:?} is not an innermost-axis projection of physical shape {:?}",Dims(&output.shape,false),Dims(&tensor.descriptor().dimensions,true))
            },
            Issue::Block(e)=>fmt::Display::fmt(e, f),
            Issue::PhysicalBlocks{physical,block}=>write!(f,"physical innermost dimension {physical} is not divisible by block length {block}"),
            Issue::LogicalBlocks{logical,blocks}=>write!(f,"logical innermost dimension {logical} is not divisible by {blocks} physical blocks"),
            _=>unreachable!("handled diagnostic"),
        }
    }
    pub(super) fn into_store_error(
        self,
        checkpoint: &Checkpoint,
        mapping: &[eredu_gguf::TranslatedTensorLayout],
    ) -> StoreError {
        let p = self.position;
        let tensor = checkpoint
            .shards()
            .get(p.shard)
            .and_then(|s| s.tensors().get(p.tensor));
        let output = tensor.and_then(|t| t.outputs().get(p.output));
        let logical = output.map_or("", |o| o.name.as_str());
        let physical = tensor.map_or("", |t| t.descriptor().name.as_str());
        match self.issue {
            Issue::Empty=>gguf_error("","GGUF logical catalog is empty"),
            Issue::Limit=>StoreError::InvalidShardCacheLimit,
            Issue::MappingDuplicate(i)=>gguf_error(&mapping[i].layout.name,"admitted GGUF tensor mapping contains a duplicate source output"),
            Issue::MappingMissing=>gguf_error(logical,"admitted GGUF tensor mapping omits a catalog output"),
            Issue::Collision=>{
                let name=mapping.iter().find(|m|m.physical_name==physical && m.original_name==logical).map_or(logical,|m|m.layout.name.as_str());
                gguf_error(name,"translated logical tensor collides with an existing output")
            }
            Issue::PhysicalDimension=>StoreError::Overflow{context:format!("GGUF physical shape for tensor {physical:?}")},
            Issue::LogicalDimension=>StoreError::Overflow{context:format!("GGUF logical shape for tensor {logical:?}")},
            Issue::LogicalBytes=>StoreError::Overflow{context:format!("GGUF logical byte length for tensor {logical:?}")},
            Issue::Projection=>{
                let shape=output.expect("output").shape.iter().map(|&v|v as usize).collect::<Vec<_>>();
                let physical=tensor.expect("tensor").descriptor().row_major_shape().into_iter().map(|v|v as usize).collect::<Vec<_>>();
                gguf_error(logical,format!("logical shape {shape:?} is not an innermost-axis projection of physical shape {physical:?}"))
            }
            Issue::Block(error)=>gguf_error(logical,error),
            Issue::PhysicalBlocks{physical,block}=>gguf_error(logical,format!("physical innermost dimension {physical} is not divisible by block length {block}")),
            Issue::LogicalBlocks{logical:dimension,blocks}=>gguf_error(logical,format!("logical innermost dimension {dimension} is not divisible by {blocks} physical blocks")),
            Issue::Reserve(error)=>StoreError::Internal(error.to_string()),
            Issue::Overflow=>StoreError::Overflow{context:"GGUF catalog storage".into()},
        }
    }
}
impl fmt::Display for WorkerFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Fixed typed diagnostic; formatting is a caller observation, never an
        // allocation performed while the admitted catalog is being built.
        write!(
            f,
            "GGUF catalog at shard {}, tensor {}, output {}: ",
            self.position.shard, self.position.tensor, self.position.output
        )?;
        match &self.issue {
            Issue::Reserve(cause) => cause.fmt(f),
            Issue::Block(cause) => cause.fmt(f),
            issue => write!(f, "{issue:?}"),
        }
    }
}
impl std::error::Error for WorkerFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.issue {
            Issue::Reserve(e) => Some(e),
            Issue::Block(e) => Some(e),
            _ => None,
        }
    }
}

/// Original checkpoint retained by an uncalled catalog plan. Its private
/// representation cannot be used as a catalog or admission proof.
#[derive(Debug)]
pub struct GgufCatalogInput(Checkpoint);
impl GgufCatalogInput {
    /// Borrow the unchanged source input, without reopening or cloning it.
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.0
    }
}
/// Actual fixed input plan; no future catalog key or source callback can be
/// appended. This owns the checkpoint and borrows retained resolution/mapping.
#[derive(Debug)]
pub struct GgufCatalogPlan<'a> {
    checkpoint: Checkpoint,
    resolved: &'a ResolvedCheckpointPlan,
    mapping: &'a [eredu_gguf::TranslatedTensorLayout],
    maximum_readers: usize,
}
impl<'a> GgufCatalogPlan<'a> {
    /// Bind this exact owned checkpoint to its already resolved architecture
    /// contract. No header, mapping, descriptor or payload is cloned here.
    pub fn new(
        checkpoint: Checkpoint,
        resolved: &'a ResolvedCheckpointPlan,
        mapping: &'a [eredu_gguf::TranslatedTensorLayout],
        maximum_readers: usize,
    ) -> Self {
        Self {
            checkpoint,
            resolved,
            mapping,
            maximum_readers,
        }
    }
    /// Return the original input on an uncalled-plan refusal. This cannot
    /// produce a catalog or preserve authority for a differently bound retry.
    pub fn into_input(self) -> GgufCatalogInput {
        GgufCatalogInput(self.checkpoint)
    }
    /// Borrow the actual input for diagnostics/baseline custody, without a new
    /// source selection or independently replaceable catalog binding.
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }
    /// Preserve the existing limit refusal before any compiler admission or
    /// mapping/node work. The returned error has no allocated text.
    pub fn validate_input(&self) -> Result<(), StoreError> {
        if self.maximum_readers == 0 {
            Err(StoreError::InvalidShardCacheLimit)
        } else {
            Ok(())
        }
    }
    /// Actual closed requested storage. The source's validated checkpoint has
    /// unique physical/logical outputs; each admitted mapping name can therefore
    /// supply at most one output. Sum their actual lengths, including unused
    /// mappings conservatively, instead of creating a second lookup catalog.
    /// Qualification and pre-debit belong to the concrete runtime caller.
    pub fn requested_storage<C>(&self) -> Option<GgufCatalogStorageRequest> {
        let mut bytes = 0usize;
        let mut selected_rows = 0usize;
        let mut unclaimed_rows = 0usize;
        let mut scratch = 0usize;
        let mut diagnostic_name = 0usize;
        for mapped in self.mapping {
            diagnostic_name = diagnostic_name.max(mapped.layout.name.len());
            bytes = bytes.checked_add(mapped.layout.name.len().checked_mul(2)?)?;
        }
        for shard in self.checkpoint.shards() {
            for tensor in shard.tensors() {
                let descriptor = tensor.descriptor();
                diagnostic_name = diagnostic_name.max(descriptor.name.len());
                let selected = self.resolved.source_keys().contains(&descriptor.name);
                let unclaimed = self.resolved.unclaimed_keys().contains(&descriptor.name);
                if !selected && !unclaimed {
                    continue;
                }
                let rank = descriptor.dimensions.len();
                let descriptor_dimensions = Layout::array::<u64>(rank).ok()?.size();
                let physical_shape = Layout::array::<usize>(rank).ok()?.size();
                scratch = scratch.max(
                    descriptor
                        .name
                        .len()
                        .checked_add(descriptor_dimensions)?
                        .checked_add(physical_shape)?,
                );
                for output in tensor.outputs() {
                    diagnostic_name = diagnostic_name.max(output.name.len());
                    if unclaimed {
                        unclaimed_rows = unclaimed_rows.checked_add(1)?;
                        continue;
                    }
                    selected_rows = selected_rows.checked_add(1)?;
                    bytes = bytes
                        .checked_add(descriptor.name.len().checked_mul(2)?)?
                        .checked_add(output.name.len())?
                        .checked_add(descriptor_dimensions)?
                        .checked_add(physical_shape)?
                        .checked_add(Layout::array::<usize>(output.shape.len()).ok()?.size())?
                        .checked_add(shard.path().as_os_str().as_encoded_bytes().len())?;
                }
            }
        }
        let nodes = CatalogRows::node_layout()
            .size()
            .checked_mul(selected_rows)?
            .checked_add(
                UnclaimedRows::node_layout()
                    .size()
                    .checked_mul(unclaimed_rows)?,
            )?
            .checked_add(
                PreparedIndexNode::<(&str, &str), &str, ()>::storage_layout()
                    .size()
                    .checked_mul(self.mapping.len())?,
            )?;
        let worker = PreparedIndex::<String, CatalogEntry, ()>::worker_control_layout()?
            .size()
            .checked_add(PreparedIndex::<String, (), ()>::worker_control_layout()?.size())?
            .checked_add(Mapping::worker_control_layout()?.size())?
            .checked_add(
                PreparedIndex::<String, CatalogEntry, ()>::iteration_control_layout().size(),
            )?
            .checked_add(PreparedIndex::<String, (), ()>::iteration_control_layout().size())?;
        let controls = [
            size_of::<Self>(),
            size_of::<GgufWeightStoreBuilder>(),
            size_of::<PreparedGgufCatalog>(),
            size_of::<GgufCatalogCompileFailure<C>>(),
            size_of::<Result<PreparedGgufCatalog, GgufCatalogCompileFailure<C>>>(),
            size_of::<CatalogData>(),
            size_of::<CatalogHandle>(),
            size_of::<C>(),
            size_of::<Mapping<'_>>(),
            size_of::<Option<PreparedIndexNode<(&str, &str), &str, ()>>>(),
            size_of::<TensorDescriptor>(),
            size_of::<Vec<usize>>().checked_mul(2)?,
            size_of::<String>(),
            size_of::<TensorMetadata>(),
            size_of::<CatalogEntry>(),
            size_of::<Position>(),
            size_of::<WorkerFailure>(),
            size_of::<Issue>(),
            size_of::<Result<(), WorkerFailure>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<&mut Vec<usize>>(),
            size_of::<&mut Vec<Checkpoint>>(),
            size_of::<usize>(), // reached reserve request
            size_of::<bool>(),  // selected
            size_of::<bool>(),  // unclaimed
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, eredu_gguf::CatalogShard>>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, eredu_gguf::CatalogTensor>>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, eredu_gguf::LogicalTensorLayout>>>(
            ),
            size_of::<std::iter::Rev<std::slice::Iter<'_, u64>>>(),
            size_of::<std::slice::Iter<'_, u64>>(),
            size_of::<std::slice::Iter<'_, usize>>(),
            size_of::<Option<usize>>(),
            size_of::<std::slice::Iter<'_, eredu_gguf::TranslatedTensorLayout>>(),
            size_of::<&Checkpoint>(),
            size_of::<&ResolvedCheckpointPlan>(),
            size_of::<&[eredu_gguf::TranslatedTensorLayout]>(),
            // Actual numeric/result transports of this query and row geometry.
            size_of::<usize>(), // bytes
            size_of::<usize>(), // selected_rows
            size_of::<usize>(), // unclaimed_rows
            size_of::<usize>(), // scratch
            size_of::<usize>(), // diagnostic_name
            size_of::<usize>(), // rank
            size_of::<usize>(), // descriptor_dimensions
            size_of::<usize>(), // physical_shape
            size_of::<usize>(), // nodes
            size_of::<usize>(), // worker
            size_of::<u64>(),   // width / checked byte result
            size_of::<Result<Option<usize>, Issue>>(),
            size_of::<(&[usize], &[usize])>(),
            size_of::<(usize, usize)>(), // enumerated source position
            CatalogHandle::owner_control_bytes::<C>()?,
        ];
        let bytes = controls
            .into_iter()
            .try_fold(
                bytes
                    .checked_add(nodes)?
                    .checked_add(scratch)?
                    .checked_add(diagnostic_name)?
                    .checked_add(worker)?
                    .checked_add(Layout::array::<Checkpoint>(1).ok()?.size())?,
                usize::checked_add,
            )?
            .checked_add(std::mem::size_of_val(&controls))?;
        Some(GgufCatalogStorageRequest {
            other_bytes: bytes,
            shared_body: CatalogHandle::body_layout::<C>(),
        })
    }
    /// Compile only these rows under the supplied caller-owned custody. This
    /// built-in worker accepts no provider callbacks and certifies no budget.
    /// The concrete runtime adapter must qualify/debit its owning requests first.
    pub fn compile<C: Any + fmt::Debug + Send + Sync>(
        self,
        custody: C,
    ) -> Result<PreparedGgufCatalog, GgufCatalogCompileFailure<C>> {
        let mut builder = GgufWeightStoreBuilder::default();
        builder.max_cached_readers = self.maximum_readers;
        let result = if self.maximum_readers == 0 {
            Err(WorkerFailure::at(Position::default(), Issue::Limit))
        } else {
            append_rows(&mut builder, &self.checkpoint, self.resolved, self.mapping)
                .and_then(|()| {
                    if builder.catalog.is_empty() {
                        Err(WorkerFailure::at(Position::default(), Issue::Empty))
                    } else {
                        Ok(())
                    }
                })
                .and_then(|()| {
                    reserve(&mut builder.checkpoints, 1)
                        .map_err(|e| WorkerFailure::at(Position::default(), Issue::Reserve(e)))
                })
        };
        if let Err(cause) = result {
            let key = cause
                .diagnostic_key(&self.checkpoint, self.mapping)
                .to_owned();
            return Err(GgufCatalogCompileFailure {
                cause,
                key,
                builder,
                checkpoint: GgufCatalogInput(self.checkpoint),
                custody,
            });
        }
        let GgufCatalogPlan { checkpoint, .. } = self;
        builder.checkpoints.push(checkpoint);
        let data = CatalogData {
            rows: std::mem::take(&mut builder.catalog),
            unclaimed: std::mem::take(&mut builder.unclaimed_keys),
        };
        builder.sealed_catalog = Some(CatalogHandle::new(data, custody));
        Ok(PreparedGgufCatalog { builder })
    }
}
/// Requested extents from the actual source compiler, not an allocation grant.
#[derive(Clone, Copy, Debug)]
pub struct GgufCatalogStorageRequest {
    other_bytes: usize,
    shared_body: Layout,
}
impl GgufCatalogStorageRequest {
    /// Concrete nested buffers, node Boxes, input Vec and compiler transports.
    pub const fn other_bytes(&self) -> usize {
        self.other_bytes
    }
    /// Concrete shared body payload; qualified Arc headers/padding are separate.
    pub const fn shared_body(&self) -> Layout {
        self.shared_body
    }
}
/// A catalog compiled from the same checkpoint still owned by this carrier.
/// There is no append, catalog replacement, Arc or raw custody extraction.
#[derive(Debug)]
pub struct PreparedGgufCatalog {
    pub(super) builder: GgufWeightStoreBuilder,
}
impl PreparedGgufCatalog {
    /// Continue the existing cold reader/source producer. Catalog admission is
    /// not coverage for the independent readers, headers or StoreInner shell.
    pub fn build_with_prepared_reader_buffers(
        self,
    ) -> Result<GgufWeightStore, GgufSourcePreparationFailure> {
        self.builder.build_with_prepared_reader_buffers()
    }
    /// Build through the ordinary reader policy, preserving the same catalog.
    pub fn build(self) -> Result<GgufWeightStore, StoreError> {
        self.builder.build()
    }
    /// The actual raw catalog origin only; no amount/clone/authority is granted.
    pub fn catalog_control_owner<C: Any>(&self) -> Option<&C> {
        self.builder.sealed_catalog.as_ref()?.origin()
    }
}
/// Actual unconsumed checkpoint, successful row prefix and original custody.
/// Construction temporaries retire synchronously before this value is returned.
#[derive(Debug)]
pub struct GgufCatalogCompileFailure<C> {
    cause: WorkerFailure,
    key: String,
    builder: GgufWeightStoreBuilder,
    checkpoint: GgufCatalogInput,
    custody: C,
}
impl<C> GgufCatalogCompileFailure<C> {
    /// Borrow the same input, never a reconstructed checkpoint.
    pub fn checkpoint(&self) -> &Checkpoint {
        self.checkpoint.checkpoint()
    }
    /// The same owned input through a type that exposes no raw custody.
    pub fn input(&self) -> &GgufCatalogInput {
        &self.checkpoint
    }
    /// Successfully installed rows in the retained partial builder.
    pub fn completed_rows(&self) -> usize {
        self.builder.catalog.len()
    }
}
impl<C> fmt::Display for GgufCatalogCompileFailure<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause
            .display_with(&self.key, self.checkpoint.checkpoint(), f)
    }
}
impl<C: fmt::Debug> std::error::Error for GgufCatalogCompileFailure<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

pub(super) fn append_rows(
    builder: &mut GgufWeightStoreBuilder,
    checkpoint: &Checkpoint,
    resolved: &ResolvedCheckpointPlan,
    mapping: &[eredu_gguf::TranslatedTensorLayout],
) -> Result<(), WorkerFailure> {
    let mut names = Mapping::default();
    for (i, mapped) in mapping.iter().enumerate() {
        let key = (mapped.physical_name.as_str(), mapped.original_name.as_str());
        let mut candidate = Some(PreparedIndexNode::new(key, mapped.layout.name.as_str(), ()));
        if !names.insert(&mut candidate) {
            return Err(WorkerFailure::at(
                Position::default(),
                Issue::MappingDuplicate(i),
            ));
        }
    }
    let checkpoint_index = builder.checkpoints.len();
    for (shard_index, shard) in checkpoint.shards().iter().enumerate() {
        for (tensor_index, tensor) in shard.tensors().iter().enumerate() {
            let mut p = Position {
                shard: shard_index,
                tensor: tensor_index,
                output: 0,
            };
            let physical_name = &tensor.descriptor().name;
            let selected = resolved.source_keys().contains(physical_name);
            let unclaimed = resolved.unclaimed_keys().contains(physical_name);
            if !selected && !unclaimed {
                continue;
            }
            let descriptor = tensor.descriptor().clone();
            let mut physical_shape = Vec::new();
            reserve(&mut physical_shape, descriptor.dimensions.len())
                .map_err(|e| WorkerFailure::at(p, Issue::Reserve(e)))?;
            for &dimension in descriptor.dimensions.iter().rev() {
                physical_shape.push(
                    usize::try_from(dimension)
                        .map_err(|_| WorkerFailure::at(p, Issue::PhysicalDimension))?,
                );
            }
            for (output_index, output) in tensor.outputs().iter().enumerate() {
                p.output = output_index;
                let key = (physical_name.as_str(), output.name.as_str());
                let mapped_name = *names
                    .get_by(|stored| key.cmp(stored))
                    .ok_or_else(|| WorkerFailure::at(p, Issue::MappingMissing))?;
                let name = mapped_name.to_owned();
                if unclaimed {
                    builder.unclaimed_keys.insert(name);
                    continue;
                }
                if builder.catalog.contains_key(&name) {
                    return Err(WorkerFailure::at(p, Issue::Collision));
                }
                let mut shape = Vec::new();
                reserve(&mut shape, output.shape.len())
                    .map_err(|e| WorkerFailure::at(p, Issue::Reserve(e)))?;
                for &dimension in &output.shape {
                    shape.push(
                        usize::try_from(dimension)
                            .map_err(|_| WorkerFailure::at(p, Issue::LogicalDimension))?,
                    );
                }
                let width = match output.dtype {
                    LogicalDtype::U8 | LogicalDtype::I8 => 1u64,
                    LogicalDtype::F16 | LogicalDtype::Bf16 | LogicalDtype::I16 => 2,
                    LogicalDtype::F32 | LogicalDtype::U32 | LogicalDtype::I32 => 4,
                    LogicalDtype::I64 | LogicalDtype::F64 => 8,
                };
                let byte_len = shape
                    .iter()
                    .try_fold(width, |bytes, &n| bytes.checked_mul(u64::try_from(n).ok()?))
                    .ok_or_else(|| WorkerFailure::at(p, Issue::LogicalBytes))?;
                let units = logical_units(&descriptor, &physical_shape, &shape)
                    .map_err(|issue| WorkerFailure::at(p, issue))?;
                let metadata = TensorMetadata {
                    name: name.clone(),
                    logical_shape: shape,
                    physical_shape: physical_shape.clone(),
                    stored_dtype: stored_dtype(output.dtype),
                    encoded_byte_len: byte_len,
                    backing_shard: Some(shard.path().to_path_buf()),
                };
                builder.catalog.insert(
                    name,
                    CatalogEntry {
                        checkpoint: checkpoint_index,
                        physical_name: physical_name.clone(),
                        original_name: output.name.clone(),
                        metadata,
                        physical_descriptor: descriptor.clone(),
                        logical_last_units_per_block: units,
                        source_encoding: crate::SourceTensorEncoding::Gguf {
                            ggml_type: descriptor.ggml_type,
                            endian: shard.endian(),
                        },
                    },
                );
            }
        }
    }
    Ok(())
}
fn logical_units(
    descriptor: &TensorDescriptor,
    physical: &[usize],
    logical: &[usize],
) -> Result<Option<usize>, Issue> {
    if physical.len() != logical.len()
        || physical
            .iter()
            .zip(logical)
            .take(physical.len().saturating_sub(1))
            .any(|(a, b)| a != b)
    {
        return Err(Issue::Projection);
    }
    let (Some(&physical_last), Some(&logical_last)) = (physical.last(), logical.last()) else {
        return Ok(None);
    };
    let (block_values, _) = descriptor
        .ggml_type
        .block_and_bytes()
        .map_err(Issue::Block)?;
    let block = usize::try_from(block_values).map_err(|_| Issue::Overflow)?;
    if !physical_last.is_multiple_of(block) {
        return Err(Issue::PhysicalBlocks {
            physical: physical_last,
            block,
        });
    }
    let blocks = physical_last / block;
    if blocks == 0 || !logical_last.is_multiple_of(blocks) {
        return Err(Issue::LogicalBlocks {
            logical: logical_last,
            blocks,
        });
    }
    Ok(Some(logical_last / blocks))
}

// Same fresh reserve worker on ordinary and admitted paths. The private test
// hook makes the reached reserve return a real capacity refusal, without
// constructing a second catalog worker or exposing a production callback.
fn reserve<T>(values: &mut Vec<T>, requested: usize) -> Result<(), TryReserveError> {
    #[cfg(test)]
    if FAIL_RESERVE.with(|remaining| match remaining.get() {
        Some(0) => {
            remaining.set(None);
            true
        }
        Some(n) => {
            remaining.set(Some(n - 1));
            false
        }
        None => false,
    }) {
        return values.try_reserve_exact(usize::MAX);
    }
    values.try_reserve_exact(requested)
}
#[cfg(test)]
thread_local! {
    static FAIL_RESERVE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}
#[cfg(test)]
mod tests;
