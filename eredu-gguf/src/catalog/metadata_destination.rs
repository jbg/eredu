//! Final result metadata and the original physical planner's storage.
use super::*;
use crate::reader::plan_storage::{
    self, AxisPlan, AxisScratch, AxisStorage, Descriptor, DescriptorSlot, SpanPlan,
};
use crate::{MetadataDestinationError, TensorDescriptorView};
mod supplied;
use std::alloc::Layout;
use supplied::SelectionLoan;
pub use supplied::{
    StoredCheckpointTensor, StoredMetadataFailure, StoredOutputNames, StoredTensorMetadata,
    StoredTensorPair,
};

/// A borrowed physical selection; the source wrapper supplies its retained value.
#[derive(Clone, Copy)]
pub enum MetadataSelection<'a> {
    /// The complete physical tensor.
    Full,
    /// An actual axis selection, including ordered repeated indices.
    Axis(&'a TensorSelection),
    /// An actual contiguous physical span.
    Span(&'a DenseTensorSpan),
}

/// Immutable catalog rows borrowed from the actual retained materializer.
pub struct TensorMetadataSource<'a> {
    tensor: &'a CatalogTensor,
    location: TensorLocation,
    endian: Endian,
}
/// Retains the actual immutable catalog row outside its mutable reader cache.
/// Its scalar coordinate indexes this same Arc, never an equivalent catalog.
/// Clones share existing source allocations and introduce no new allocation.
#[derive(Clone, Debug)]
pub struct SharedTensorMetadataSource {
    checkpoint: super::prepared_materializer::SharedCheckpoint,
    location: TensorLocation,
}
impl SharedTensorMetadataSource {
    #[cfg(test)]
    pub(super) fn test_checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }
    /// Borrow the exact descriptor and ordered output-name row held at capture.
    pub fn source(&self) -> TensorMetadataSource<'_> {
        let shard = &self.checkpoint.shards[self.location.shard_index];
        TensorMetadataSource {
            tensor: &shard.tensors[self.location.tensor_index],
            location: self.location,
            endian: shard.endian,
        }
    }
}
impl TensorMaterializer {
    /// Loans the actual catalog without opening a reader or changing its cache state.
    pub fn metadata_source(&self, name: &str) -> Result<TensorMetadataSource<'_>> {
        let location = self.metadata_location(name)?;
        let shard = &self.checkpoint.shards[location.shard_index];
        Ok(TensorMetadataSource {
            tensor: &shard.tensors[location.tensor_index],
            location,
            endian: shard.endian,
        })
    }

    /// Retain the actual metadata row without retaining a reader-cache lock.
    /// Returns `None` for an ordinary inline catalog, without promoting it or
    /// allocating. Use the explicit cold shared constructor when this is needed.
    pub fn shared_metadata_source(&self, name: &str) -> Result<Option<SharedTensorMetadataSource>> {
        let location = self.metadata_location(name)?;
        Ok(match &self.checkpoint {
            MaterializerCheckpoint::Owned(_) => None,
            MaterializerCheckpoint::Shared(checkpoint) => Some(SharedTensorMetadataSource {
                checkpoint: checkpoint.clone(),
                location,
            }),
        })
    }

    fn metadata_location(&self, name: &str) -> Result<TensorLocation> {
        let location =
            self.locations
                .get(name, &self.checkpoint)
                .ok_or_else(|| Error::InvalidTensor {
                    tensor: name.to_string(),
                    reason: "tensor is not present in the checkpoint".into(),
                })?;
        Ok(location)
    }
}

/// Requested layouts, excluding allocator bookkeeping and previously retained sources.
#[derive(Clone, Copy, Debug)]
pub struct MetadataLayouts {
    owner: Layout,
    buffer_bytes: usize,
    nonempty_buffers: usize,
}
impl MetadataLayouts {
    /// Inline metadata owner, including all Vec/String headers and completed-output slot.
    pub fn owner(&self) -> Layout {
        self.owner
    }
    /// Sum of requested typed buffer layouts; actual capacity can be larger.
    pub fn requested_buffer_bytes(&self) -> usize {
        self.buffer_bytes
    }
    /// Number of nonempty requested buffer allocations.
    pub fn nonempty_buffers(&self) -> usize {
        self.nonempty_buffers
    }
    fn add<T>(&mut self, count: usize) -> Option<()> {
        let layout = Layout::array::<T>(count).ok()?;
        self.buffer_bytes = self.buffer_bytes.checked_add(layout.size())?;
        self.nonempty_buffers = self
            .nonempty_buffers
            .checked_add(usize::from(layout.size() != 0))?;
        Some(())
    }
    fn descriptor(&mut self, name: usize, rank: usize) -> Option<()> {
        self.add::<u8>(name)?;
        self.add::<u64>(rank)
    }
}
impl TensorMetadataSource<'_> {
    /// Physical descriptor and lazy planner scratch requests from this exact row.
    pub fn physical_storage_bound(
        &self,
        selection: MetadataSelection<'_>,
    ) -> Option<crate::StorageRequestBound> {
        crate::reader::plan_storage::physical_storage_bound(&self.tensor.descriptor, selection)
    }
    /// G3 supplied fixed-name/descriptor/selection storage; includes zero calls.
    /// This is distinct from ordinary Vec<String> metadata storage.
    pub fn supplied_storage_bound(
        &self,
        selection: MetadataSelection<'_>,
    ) -> Option<crate::StorageRequestBound> {
        self.layouts(selection)?; // unchanged complete-layout eligibility
        let d = &self.tensor.descriptor;
        let rank = d.dimensions.len();
        let mut out = crate::StorageRequestBound::default();
        out.descriptor(d.name.len(), rank)?;
        for output in &self.tensor.outputs {
            out.add::<u8>(output.name.len())?;
        }
        match selection {
            MetadataSelection::Full => {}
            MetadataSelection::Axis(selection) => {
                if let TensorSelection::Indices { indices, .. } = selection {
                    out.add::<usize>(indices.len())?;
                }
                let (indices, ranges) = supplied::axis_requests(selection);
                out.add::<usize>(indices)?;
                out.add::<(usize, usize)>(ranges)?;
                out.add::<u64>(rank)?;
                out.add::<crate::reader::RelativeEncodedSpan>(ranges)?;
                out.descriptor(d.name.len(), rank)?;
                out.descriptor(d.name.len(), rank)?;
            }
            MetadataSelection::Span(span) => {
                out.add::<u64>(span.shape().len())?;
                out.descriptor(d.name.len(), rank.max(span.shape().len()))?;
                out.descriptor(d.name.len(), span.shape().len())?;
            }
        }
        Some(out)
    }

    /// Computes storage ceilings from actual source lengths, without running a planner.
    /// Index scratch reserves the full index count even when blocks/coalescing use less.
    pub fn layouts(&self, selection: MetadataSelection<'_>) -> Option<MetadataLayouts> {
        let d = &self.tensor.descriptor;
        let r = d.dimensions.len();
        let n = d.name.len();
        let mut result = MetadataLayouts {
            owner: Layout::new::<PreparedTensorMetadata>(),
            buffer_bytes: 0,
            nonempty_buffers: 0,
        };
        result.descriptor(n, r)?;
        result.add::<String>(self.tensor.outputs.len())?;
        for output in &self.tensor.outputs {
            result.add::<u8>(output.name.len())?;
        }
        match selection {
            MetadataSelection::Full => {}
            MetadataSelection::Axis(selection) => {
                result.descriptor(n, r)?;
                result.descriptor(n, r)?;
                let (indices, ranges) = match selection {
                    TensorSelection::Range { .. } => (0, 1),
                    TensorSelection::Indices { indices, .. } => (indices.len(), indices.len()),
                };
                result.add::<usize>(indices)?; // retained exact selection
                result.add::<usize>(indices)?; // encoded index scratch
                result.add::<(usize, usize)>(ranges)?;
                result.add::<u64>(r)?;
                result.add::<crate::reader::RelativeEncodedSpan>(ranges)?;
            }
            MetadataSelection::Span(selection) => {
                let s = selection.shape().len();
                result.descriptor(n, s)?;
                result.descriptor(n, r.max(s))?;
                result.add::<u64>(s)?;
            }
        }
        Some(result)
    }
}

#[derive(Debug)]
enum SelectionStorage {
    Full,
    Axis {
        selection: TensorSelection,
        scratch: AxisScratch,
    },
    Span {
        selection: DenseTensorSpan,
        selected: TensorDescriptor,
    },
}
/// Owns final descriptor/name destinations and compact planner scratch.
/// This is storage, not accepted-account or source-custody authority.
#[derive(Debug)]
pub struct PreparedTensorMetadata {
    base: TensorDescriptor,
    final_descriptor: TensorDescriptor,
    names: Vec<String>,
    selection: SelectionStorage,
    completed: Option<ConvertedTensor>,
    location: TensorLocation,
    endian: Endian,
    layouts: MetadataLayouts,
    used: bool,
}
/// Preparation failure retaining every successful metadata allocation prefix.
#[derive(Debug)]
pub struct MetadataPreparationFailure {
    cause: MetadataDestinationError,
    destination: PreparedTensorMetadata,
}
impl MetadataPreparationFailure {
    /// Original reserve/layout cause and its complete retained destination prefix.
    pub fn into_parts(self) -> (MetadataDestinationError, PreparedTensorMetadata) {
        (self.cause, self.destination)
    }
}
impl std::fmt::Display for MetadataPreparationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for MetadataPreparationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl PreparedTensorMetadata {
    /// Prepares actual catalog names and selection destinations without reader I/O.
    /// All semantic plan checks still run at the ordinary materialization point.
    pub fn prepare(
        source: TensorMetadataSource<'_>,
        selection: MetadataSelection<'_>,
    ) -> std::result::Result<Self, MetadataPreparationFailure> {
        let mut destination = Self {
            base: plan_storage::empty_descriptor(),
            final_descriptor: plan_storage::empty_descriptor(),
            names: Vec::new(),
            selection: SelectionStorage::Full,
            completed: None,
            location: source.location,
            endian: source.endian,
            layouts: MetadataLayouts {
                owner: Layout::new::<Self>(),
                buffer_bytes: 0,
                nonempty_buffers: 0,
            },
            used: false,
        };
        let result = (|| {
            destination.layouts = source
                .layouts(selection)
                .ok_or(MetadataDestinationError::Layout)?;
            let tensor = &source.tensor.descriptor;
            plan_storage::prepare_descriptor(
                &mut destination.base,
                tensor,
                tensor.dimensions.len(),
            )?;
            plan_storage::reserve(&mut destination.names, source.tensor.outputs.len())?;
            for output in &source.tensor.outputs {
                destination.names.push(String::new());
                let name = destination.names.last_mut().expect("new final name");
                name.try_reserve_exact(output.name.len())
                    .map_err(MetadataDestinationError::Reserve)?;
                name.push_str(&output.name);
            }
            match selection {
                MetadataSelection::Full => {}
                MetadataSelection::Axis(selection) => {
                    destination.selection = SelectionStorage::Axis {
                        selection: plan_storage::empty_selection(selection),
                        scratch: AxisScratch::empty(),
                    };
                    let SelectionStorage::Axis {
                        selection: bound,
                        scratch,
                    } = &mut destination.selection
                    else {
                        unreachable!()
                    };
                    plan_storage::prepare_selection(bound, selection)?;
                    scratch.prepare(tensor, selection)?;
                    prepare_final_descriptor(
                        &mut destination.final_descriptor,
                        tensor,
                        tensor.dimensions.len(),
                    )?;
                }
                MetadataSelection::Span(selection) => {
                    destination.selection = SelectionStorage::Span {
                        selection: plan_storage::empty_span(selection),
                        selected: plan_storage::empty_descriptor(),
                    };
                    let SelectionStorage::Span {
                        selection: bound,
                        selected,
                    } = &mut destination.selection
                    else {
                        unreachable!()
                    };
                    plan_storage::prepare_span(bound, selection)?;
                    plan_storage::prepare_descriptor(
                        selected,
                        tensor,
                        tensor.dimensions.len().max(selection.shape().len()),
                    )?;
                    prepare_final_descriptor(
                        &mut destination.final_descriptor,
                        tensor,
                        selection.shape().len(),
                    )?;
                }
            }
            Ok::<_, MetadataDestinationError>(())
        })();
        match result {
            Ok(()) => Ok(destination),
            Err(cause) => Err(MetadataPreparationFailure { cause, destination }),
        }
    }
    /// Requested layouts from the genuine catalog lengths; not allocator charge.
    pub fn layouts(&self) -> MetadataLayouts {
        self.layouts
    }
    /// Completed payload retained if a later metadata operation refused storage.
    pub fn completed(&self) -> Option<&ConvertedTensor> {
        self.completed.as_ref()
    }
    /// Actual metadata buffer capacity bytes, excluding the G2 raw/output buffers.
    pub fn capacity_bytes(&self) -> Option<usize> {
        let mut n = 0usize;
        fn add<T>(n: &mut usize, v: &Vec<T>) -> Option<()> {
            *n = n.checked_add(Layout::array::<T>(v.capacity()).ok()?.size())?;
            Some(())
        }
        fn descriptor(n: &mut usize, d: &TensorDescriptor) -> Option<()> {
            *n = n.checked_add(d.name.capacity())?;
            add(n, &d.dimensions)
        }
        descriptor(&mut n, &self.base)?;
        descriptor(&mut n, &self.final_descriptor)?;
        add(&mut n, &self.names)?;
        for name in &self.names {
            n = n.checked_add(name.capacity())?;
        }
        match &self.selection {
            SelectionStorage::Full => {}
            SelectionStorage::Axis { selection, scratch } => {
                if let TensorSelection::Indices { indices, .. } = selection {
                    add(&mut n, indices)?;
                }
                add(&mut n, &scratch.indices)?;
                add(&mut n, &scratch.ranges)?;
                add(&mut n, &scratch.dimensions)?;
                add(&mut n, &scratch.spans)?;
                descriptor(&mut n, &scratch.selected)?;
            }
            SelectionStorage::Span {
                selection,
                selected,
            } => {
                n = n.checked_add(plan_storage::span_capacity_bytes(selection)?)?;
                descriptor(&mut n, selected)?;
            }
        }
        Some(n)
    }
}
fn prepare_final_descriptor(
    target: &mut TensorDescriptor,
    source: &TensorDescriptor,
    rank: usize,
) -> std::result::Result<(), MetadataDestinationError> {
    target
        .name
        .try_reserve_exact(source.name.len())
        .map_err(MetadataDestinationError::Reserve)?;
    target.name.push_str(&source.name);
    plan_storage::reserve(&mut target.dimensions, rank)
}

pub(super) struct MetadataPolicy<'a> {
    base: Option<DescriptorSlot<'a>>,
    final_descriptor: Option<DescriptorSlot<'a>>,
    names: Option<&'a mut Vec<String>>,
    selection: Option<&'a mut SelectionStorage>,
    completed: Option<&'a mut Option<ConvertedTensor>>,
    binding: Option<(TensorLocation, Endian, &'a mut bool)>,
    supplied_names: Option<[Option<&'a str>; 3]>,
    supplied_count: usize,
    supplied_selection: Option<SelectionLoan<'a>>,
}
impl<'a> MetadataPolicy<'a> {
    pub(super) fn new(destination: Option<&'a mut PreparedTensorMetadata>) -> Self {
        match destination {
            None => Self {
                base: None,
                final_descriptor: None,
                names: None,
                selection: None,
                completed: None,
                binding: None,
                supplied_names: None,
                supplied_count: 0,
                supplied_selection: None,
            },
            Some(d) => Self {
                base: Some(DescriptorSlot::Vec(&mut d.base)),
                final_descriptor: Some(DescriptorSlot::Vec(&mut d.final_descriptor)),
                names: Some(&mut d.names),
                selection: Some(&mut d.selection),
                completed: Some(&mut d.completed),
                binding: Some((d.location, d.endian, &mut d.used)),
                supplied_names: None,
                supplied_count: 0,
                supplied_selection: None,
            },
        }
    }
    pub(super) fn base(
        &mut self,
        tensor: &TensorDescriptor,
        location: TensorLocation,
        endian: Endian,
    ) -> std::result::Result<Descriptor<'a>, MetadataDestinationError> {
        match self.base.take() {
            None => Descriptor::copy(tensor, None),
            Some(base) => {
                let (bound, order, used) = self.binding.take().expect("prepared binding");
                if *used {
                    return Err(MetadataDestinationError::Used);
                }
                *used = true;
                let base = match base {
                    DescriptorSlot::Vec(base) => Descriptor::Borrowed(base),
                    DescriptorSlot::Fixed(base) => Descriptor::Fixed(base),
                    DescriptorSlot::Lazy(_) => unreachable!("metadata prepared before execution"),
                };
                if bound != location || order != endian || base.view() != tensor.view() {
                    return Err(MetadataDestinationError::Binding);
                }
                Ok(base)
            }
        }
    }
    pub(super) fn names(
        &mut self,
        outputs: &[LogicalTensorLayout],
    ) -> std::result::Result<Names<'a>, MetadataDestinationError> {
        if let Some(names) = self.supplied_names.take() {
            if self.supplied_count != outputs.len()
                || outputs
                    .iter()
                    .enumerate()
                    .any(|(i, row)| names[i] != Some(row.name.as_str()))
            {
                return Err(MetadataDestinationError::Binding);
            }
            return Ok(Names::Supplied);
        }
        match self.names.take() {
            None => Ok(Names::Owned(
                outputs.iter().map(|output| output.name.clone()).collect(),
            )),
            Some(names) => {
                if names.len() != outputs.len()
                    || names
                        .iter()
                        .zip(outputs)
                        .any(|(name, row)| name != &row.name)
                {
                    return Err(MetadataDestinationError::Binding);
                };
                Ok(Names::Borrowed(names))
            }
        }
    }
    pub(super) fn full(&mut self) -> std::result::Result<(), MetadataDestinationError> {
        if let Some(selection) = self.supplied_selection.take() {
            return match selection {
                SelectionLoan::Full => Ok(()),
                _ => Err(MetadataDestinationError::Binding),
            };
        }
        match self.selection.take() {
            None | Some(SelectionStorage::Full) => Ok(()),
            _ => Err(MetadataDestinationError::Binding),
        }
    }
    pub(super) fn axis(
        &mut self,
        tensor: &Descriptor<'_>,
        selection: &TensorSelection,
    ) -> std::result::Result<AxisMetadataPlan<'a>, MetadataDestinationError> {
        if let Some(prepared) = self.supplied_selection.take() {
            return prepared.axis(tensor.view(), selection);
        }
        match self.selection.take() {
            None => Ok(AxisMetadataPlan::Owned(TensorSelectionPlan::new(
                tensor,
                selection.clone(),
            )?)),
            Some(SelectionStorage::Axis {
                selection: bound,
                scratch,
            }) if bound == selection => Ok(AxisMetadataPlan::Prepared(plan_storage::axis_plan(
                tensor,
                bound,
                AxisStorage::new(Some(scratch)),
            )?)),
            _ => Err(MetadataDestinationError::Binding),
        }
    }
    pub(super) fn span(
        &mut self,
        tensor: &Descriptor<'_>,
        selection: &DenseTensorSpan,
    ) -> std::result::Result<SpanMetadataPlan<'a>, MetadataDestinationError> {
        if let Some(prepared) = self.supplied_selection.take() {
            return prepared.span(tensor.view(), selection);
        }
        match self.selection.take() {
            None => Ok(SpanMetadataPlan::Owned(DenseTensorSpanPlan::new(
                tensor,
                selection.clone(),
            )?)),
            Some(SelectionStorage::Span {
                selection: bound,
                selected,
            }) if bound == selection => Ok(SpanMetadataPlan::Prepared(plan_storage::span_plan(
                tensor,
                bound,
                Some(selected),
            )?)),
            _ => Err(MetadataDestinationError::Binding),
        }
    }
    pub(super) fn selected(
        &mut self,
        source: TensorDescriptorView<'_>,
    ) -> std::result::Result<Descriptor<'a>, MetadataDestinationError> {
        Descriptor::copy_view(source, self.final_descriptor.take())
    }
    pub(super) fn completed(&mut self, value: ConvertedTensor) -> Completed<'a> {
        match self.completed.take() {
            None => Completed::Owned(value),
            Some(slot) => {
                *slot = Some(value);
                Completed::Borrowed(slot)
            }
        }
    }
}
pub(super) enum Names<'a> {
    Supplied,
    Owned(Vec<String>),
    Borrowed(&'a mut Vec<String>),
}
impl Names<'_> {
    pub(super) fn finish(self) -> Vec<String> {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => std::mem::take(v),
            Self::Supplied => unreachable!("supplied names retain their allocation owners"),
        }
    }
}
pub(super) enum Completed<'a> {
    Owned(ConvertedTensor),
    Borrowed(&'a mut Option<ConvertedTensor>),
}
impl Completed<'_> {
    pub(super) fn finish(self) -> ConvertedTensor {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => v.take().expect("completed payload"),
        }
    }
}
pub(super) enum AxisMetadataPlan<'a> {
    Owned(TensorSelectionPlan),
    Prepared(AxisPlan<'a>),
}
impl AxisMetadataPlan<'_> {
    pub(super) fn view(&self) -> plan_storage::AxisView<'_> {
        match self {
            Self::Owned(p) => p.view(),
            Self::Prepared(p) => p.view(),
        }
    }
    pub(super) fn selected_descriptor(&self) -> TensorDescriptorView<'_> {
        match self {
            Self::Owned(p) => p.selected_descriptor().view(),
            Self::Prepared(p) => p.selected_descriptor.view(),
        }
    }
}
pub(super) enum SpanMetadataPlan<'a> {
    Owned(DenseTensorSpanPlan),
    Prepared(SpanPlan<'a>),
}
impl SpanMetadataPlan<'_> {
    pub(super) fn view(&self) -> plan_storage::SpanView<'_> {
        match self {
            Self::Owned(p) => p.view(),
            Self::Prepared(p) => p.view(),
        }
    }
    pub(super) fn selected_descriptor(&self) -> TensorDescriptorView<'_> {
        match self {
            Self::Owned(p) => p.selected_descriptor().view(),
            Self::Prepared(p) => p.selected_descriptor.view(),
        }
    }
}

#[cfg(test)]
mod tests;

// The same driver parks ordinary converted output before the late metadata copy.
// Supplied conversion keeps its completed buffers in the caller's intact owner.
pub(super) trait CatalogOutput: Sized {
    type Completed<'a>;
    type Result;
    fn park<'a>(self, metadata: &mut MetadataPolicy<'a>) -> Self::Completed<'a>;
    fn finish<'a>(
        location: TensorLocation,
        descriptor: Descriptor<'a>,
        names: Names<'a>,
        completed: Self::Completed<'a>,
    ) -> Self::Result;
}
impl CatalogOutput for ConvertedTensor {
    type Completed<'a> = Completed<'a>;
    type Result = ConvertedCheckpointTensor;
    fn park<'a>(self, metadata: &mut MetadataPolicy<'a>) -> Completed<'a> {
        metadata.completed(self)
    }
    fn finish<'a>(
        location: TensorLocation,
        descriptor: Descriptor<'a>,
        names: Names<'a>,
        completed: Completed<'a>,
    ) -> Self::Result {
        ConvertedCheckpointTensor {
            shard_index: location.shard_index,
            tensor_index: location.tensor_index,
            descriptor: descriptor.finish(),
            output_names: names.finish(),
            converted: completed.finish(),
        }
    }
}
impl CatalogOutput for () {
    type Completed<'a> = ();
    type Result = ();
    fn park<'a>(self, _: &mut MetadataPolicy<'a>) {}
    fn finish<'a>(_: TensorLocation, _: Descriptor<'a>, _: Names<'a>, _: ()) {}
}
