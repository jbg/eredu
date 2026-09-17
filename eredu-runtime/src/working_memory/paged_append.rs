//! Metadata realization of the same ordered native paged append and scan operations.
mod host;
mod scan;
pub use host::{WorkspacePagedHostEntry, WorkspacePagedHostLoad, WorkspacePagedHostTrace};
mod visible;
use crate::cache::{PagedAppendMechanisms, PagedAppendPlan};
use eredu_nn::{
    Error, Index, Tensor,
    workspace::{WorkspaceContext, WorkspaceDtype, WorkspaceMetadataError, WorkspaceTensor},
};

/// Actual selected paging and array geometry. This descriptor grants no native
/// source permission; composition keeps its manager/source pins separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspacePagedGeometry {
    /// Actual immutable block size.
    pub block_size: i32,
    /// Array batch, key heads, and key width.
    pub dimensions: [i32; 3],
    /// Absolute exclusive position after the current tail.
    pub offset: i64,
    /// Absolute first token in the current partial tail.
    pub tail_start: i64,
    /// Causal sliding width, if selected.
    pub window: Option<i32>,
    /// Absolute pinned prefix boundary.
    pub prefix_tokens: i32,
    /// Values use the actual one-channel persistence sentinel.
    pub key_only: bool,
    /// Actual manager persistence policy keeps otherwise expired blocks.
    pub retain_discarded: bool,
}

#[derive(Clone, Debug)]
enum PagedBlockValues {
    Tensor([WorkspaceTensor; 2]),
    Backed {
        tensors: [WorkspaceTensor; 2],
        read: [eredu_nn::workspace::WorkspaceHostReadValue; 2],
    },
    Read([eredu_nn::workspace::WorkspaceHostReadValue; 2]),
}
impl PagedBlockValues {
    fn tensors(&self) -> Option<&[WorkspaceTensor; 2]> {
        match self {
            Self::Tensor(values)
            | Self::Backed {
                tensors: values, ..
            } => Some(values),
            Self::Read(_) => None,
        }
    }
}

/// One ordered symbolic block with its actual source retention condition.
#[derive(Debug)]
pub struct WorkspacePagedBlock {
    start: i64,
    end: i64,
    values: PagedBlockValues,
    retained_source: bool,
    host: Option<[eredu_nn::workspace::WorkspaceFloatingType; 2]>,
    stored: Option<[eredu_nn::workspace::WorkspaceStoredHostValue; 2]>,
    host_entry: Option<usize>,
}
impl WorkspacePagedBlock {
    /// Supplies a source block; full interval/array validation occurs at import.
    pub fn new(start: i64, end: i64, values: [WorkspaceTensor; 2], retained_source: bool) -> Self {
        Self {
            start,
            end,
            values: PagedBlockValues::Tensor(values),
            retained_source,
            host: None,
            stored: None,
            host_entry: None,
        }
    }
    /// Describes a retained host block with each actual scalar representation.
    /// Source custody and transfer permission stay in the native source owner.
    pub fn host(
        start: i64,
        end: i64,
        values: [WorkspaceTensor; 2],
        types: [eredu_nn::workspace::WorkspaceFloatingType; 2],
    ) -> Self {
        Self {
            start,
            end,
            values: PagedBlockValues::Tensor(values),
            retained_source: true,
            host: Some(types),
            stored: None,
            host_entry: None,
        }
    }
    /// Retains actual hot tensors together with their existing immutable read
    /// backing. The transfer itinerary never needs to store this page again.
    pub fn backed(
        start: i64,
        end: i64,
        tensors: [WorkspaceTensor; 2],
        read: [eredu_nn::workspace::WorkspaceHostReadValue; 2],
    ) -> Self {
        Self {
            start,
            end,
            values: PagedBlockValues::Backed { tensors, read },
            retained_source: true,
            host: None,
            stored: None,
            host_entry: None,
        }
    }
    /// Describes an independently prepared read without inventing tensor storage.
    /// The exact native file and its source pins remain provider-owned.
    pub fn read_source(
        start: i64,
        end: i64,
        values: [eredu_nn::workspace::WorkspaceHostReadValue; 2],
    ) -> Self {
        Self {
            start,
            end,
            values: PagedBlockValues::Read(values),
            retained_source: true,
            host: None,
            stored: None,
            host_entry: None,
        }
    }
    /// Whether the exact described values still require host promotion.
    pub fn requires_host_transfer(&self) -> bool {
        self.host.is_some()
            || self.stored.is_some()
            || matches!(self.values, PagedBlockValues::Read(_))
    }

    /// Absolute half-open token interval.
    pub fn range(&self) -> std::ops::Range<i64> {
        self.start..self.end
    }
    /// Last tensor geometry. A stored-Host block retains this descriptive
    /// origin but does not retain it as current Device-state storage; storage
    /// visitors must use `tensor_values`. A declared read has no Tensor origin
    /// and returns `None` until the shared transfer has produced real values.
    pub fn values(&self) -> Option<&[WorkspaceTensor; 2]> {
        self.values.tensors()
    }
    /// Current tensor-backed source values. Newly stored Host pages are opaque
    /// source effects and have no portable Tensor backing to visit.
    pub fn tensor_values(&self) -> Option<&[WorkspaceTensor; 2]> {
        self.stored
            .is_none()
            .then(|| self.values.tensors())
            .flatten()
    }
    fn promote(&mut self, context: &WorkspaceContext) -> Result<(), Error> {
        context.charge_metadata(std::mem::size_of::<(
            &mut Self,
            &WorkspaceContext,
            [eredu_nn::workspace::WorkspaceFloatingType; 2],
            [WorkspaceTensor; 2],
            Result<[WorkspaceTensor; 2], Error>,
            Result<(), Error>,
        )>())?;
        let values = if let PagedBlockValues::Read(values) = &self.values {
            [values[0].load(context)?, values[1].load(context)?]
        } else if let Some(stored) = &self.stored {
            [stored[0].load(context)?, stored[1].load(context)?]
        } else if let Some(types) = self.host {
            [
                self.values
                    .tensors()
                    .ok_or(WorkspaceMetadataError::Unqualified)?[0]
                    .transfer_host_floating(types[0], context)?,
                self.values
                    .tensors()
                    .ok_or(WorkspaceMetadataError::Unqualified)?[1]
                    .transfer_host_floating(types[1], context)?,
            ]
        } else {
            return Ok(());
        };
        context.retain_values(&[&values[0], &values[1]])?;
        self.values = PagedBlockValues::Tensor(values);
        self.host = None;
        self.stored = None;
        Ok(())
    }
}

/// Ordered blocks plus an independent partial tail. This is deliberately not a
/// contiguous attention state. Scan/completion and native admission have their
/// own required consumers before an original managed entry can use it.
#[derive(Debug)]
pub struct WorkspacePagedAppendState {
    geometry: WorkspacePagedGeometry,
    blocks: Vec<WorkspacePagedBlock>,
    tail: Option<[WorkspaceTensor; 2]>,
    host_trace: Option<WorkspacePagedHostTrace>,
    context: WorkspaceContext,
}
impl WorkspacePagedAppendState {
    /// Imports exact source metadata without replaying or combining history.
    pub fn project<I: ExactSizeIterator<Item = WorkspacePagedBlock>>(
        geometry: WorkspacePagedGeometry,
        blocks: I,
        tail: Option<[WorkspaceTensor; 2]>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(std::mem::size_of::<(
            Self,
            WorkspacePagedGeometry,
            Option<[WorkspaceTensor; 2]>,
            &WorkspaceContext,
            I,
            WorkspacePagedBlock,
            Result<Self, Error>,
            (usize, usize, i64),
        )>())?;
        if geometry.block_size <= 0
            || geometry.dimensions.iter().any(|n| *n <= 0)
            || geometry.offset < 0
            || geometry.tail_start < 0
            || geometry.tail_start > geometry.offset
            || geometry.prefix_tokens < 0
            || geometry.window.is_some_and(|n| n <= 0)
        {
            return Err(context.metadata_error(format_args!("invalid paged source geometry")));
        }
        let tail_len = geometry.offset - geometry.tail_start;
        if tail_len >= i64::from(geometry.block_size) || tail.is_some() != (tail_len > 0) {
            return Err(
                context.metadata_error(format_args!("paged source tail differs from frontier"))
            );
        }
        let count = blocks.len();
        let mut prepared = context.metadata_vec(count)?;
        let mut previous_end = 0;
        for block in blocks {
            if prepared.len() == count {
                return Err(
                    context.metadata_error(format_args!("paged source block count changed"))
                );
            }
            if block.start < previous_end
                || block.end <= block.start
                || block.end > geometry.tail_start
                || block.end - block.start > i64::from(geometry.block_size)
            {
                return Err(context
                    .metadata_error(format_args!("paged source block order or extent differs")));
            }
            match &block.values {
                PagedBlockValues::Tensor(values) => {
                    validate_pair(&geometry, values, block.end - block.start, context)?
                }
                PagedBlockValues::Read(values) | PagedBlockValues::Backed { read: values, .. } => {
                    if values.iter().any(|value| !value.shares_context(context))
                        || values[0].floating_type() != values[1].floating_type()
                    {
                        return Err(WorkspaceMetadataError::Unqualified.into());
                    }
                    validate_pair_shapes(
                        &geometry,
                        [values[0].shape(), values[1].shape()],
                        block.end - block.start,
                        context,
                    )?;
                }
            }
            if let PagedBlockValues::Backed { tensors, .. } = &block.values {
                validate_pair(&geometry, tensors, block.end - block.start, context)?;
            }
            previous_end = block.end;
            prepared.push(block);
        }
        if prepared.len() != count {
            return Err(context.metadata_error(format_args!("paged source block count changed")));
        }
        if let Some(tail) = &tail {
            validate_pair(&geometry, tail, tail_len, context)?;
        }
        Ok(Self {
            geometry,
            blocks: prepared,
            tail,
            host_trace: None,
            context: context.clone(),
        })
    }
    pub(in crate::working_memory) fn workspace_context(&self) -> &WorkspaceContext {
        &self.context
    }

    /// Copies only the metadata container and immutable tensor handles. Native
    /// copy/manager authority remains a separate selected mechanism.
    pub(in crate::working_memory) fn copy_metadata(
        &self,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            &WorkspaceContext,
            Self,
            WorkspacePagedBlock,
            [WorkspaceTensor; 2],
            Option<[WorkspaceTensor; 2]>,
            Result<Self, Error>,
        )>())?;
        if !context.shares_trace(&self.context) {
            return Err(
                context.metadata_error(format_args!("paged state copy uses another context"))
            );
        }
        let mut blocks = context.metadata_vec(self.blocks.len())?;
        for block in &self.blocks {
            blocks.push(WorkspacePagedBlock {
                start: block.start,
                end: block.end,
                values: block.values.clone(),
                retained_source: block.retained_source,
                host: block.host,
                stored: block.stored.clone(),
                host_entry: block.host_entry,
            });
        }
        Ok(Self {
            geometry: self.geometry,
            blocks,
            tail: self.tail.clone(),
            host_trace: self.host_trace.clone(),
            context: context.clone(),
        })
    }
    /// Trace independent copies of each real block/tail value. The shared
    /// isolated-copy worker creates destination storage; it is never an alias
    /// relabel. Interval and geometry metadata remain unchanged.
    pub(in crate::working_memory) fn copy_isolated(
        &self,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        self.copy_values(context, false)
    }
    pub(in crate::working_memory) fn copy_for_resume(
        &self,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        self.copy_values(context, true)
    }
    fn copy_values(&self, context: &WorkspaceContext, retain_host: bool) -> Result<Self, Error> {
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            &WorkspaceContext,
            Self,
            Result<Self, Error>,
            eredu_nn::WorkspaceIsolatedCopy<'_>,
            [WorkspaceTensor; 2],
            bool,
        )>())?;
        // An active store trace is not an immutable saved Host source. Resume
        // may retain only actual imported Host values; it never fabricates a
        // Device tensor or reruns an uncompleted Host store.
        if self.blocks.iter().any(|block| {
            block.stored.is_some()
                || block.host_entry.is_some()
                || (!retain_host && block.requires_host_transfer())
        }) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        if let Some(trace) = &self.host_trace {
            // Complete native source projection installs the future itinerary
            // before copying saved state. Only its still-empty, same-context
            // source may follow the retained Host roots into resumed equations.
            if !retain_host {
                return Err(WorkspaceMetadataError::Unqualified.into());
            }
            trace.validate_unstarted(context)?;
        }
        let mut output = self.copy_metadata(context)?;
        let copy = |value: &WorkspaceTensor| {
            eredu_nn::isolated_copy(eredu_nn::WorkspaceIsolatedCopy::new(value.clone(), context))
        };
        for block in &mut output.blocks {
            if retain_host
                && (block.host.is_some() || matches!(block.values, PagedBlockValues::Read(_)))
            {
                // Same immutable root and scalar representation. Native state
                // receives a distinct manager and keeps its actual B owner.
                continue;
            }
            let values = block
                .values
                .tensors()
                .ok_or(WorkspaceMetadataError::Unqualified)?;
            let copied = [copy(&values[0])?, copy(&values[1])?];
            block.values = match &block.values {
                PagedBlockValues::Backed { read, .. } => PagedBlockValues::Backed {
                    tensors: copied,
                    read: read.clone(),
                },
                _ => PagedBlockValues::Tensor(copied),
            };
            block.retained_source = false;
        }
        if let Some(tail) = &mut output.tail {
            *tail = [copy(&tail[0])?, copy(&tail[1])?];
        }
        Ok(output)
    }
    /// Record one selected page's two Host-store effects. The caller owns the
    /// shared eviction policy; this method grants no native source or capacity.
    /// A failure retains prior canonical state and all already accepted trace.
    pub fn store_block_host(
        &mut self,
        index: usize,
        types: [eredu_nn::workspace::WorkspaceFloatingType; 2],
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        if !self.context.shares_trace(context) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        context.charge_metadata(std::mem::size_of::<(
            &mut Self,
            usize,
            [eredu_nn::workspace::WorkspaceFloatingType; 2],
            [eredu_nn::workspace::WorkspaceStoredHostValue; 2],
            Result<(), Error>,
        )>())?;
        let block = self
            .blocks
            .get_mut(index)
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        if block.requires_host_transfer() {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let values = block
            .values
            .tensors()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let stored = [
            context.store_host_value(&values[0], types[0])?,
            context.store_host_value(&values[1], types[1])?,
        ];
        block.stored = Some(stored);
        Ok(())
    }
    pub(in crate::working_memory) fn reset_metadata(&mut self) {
        self.blocks.clear();
        self.tail = None;
        self.geometry.offset = 0;
        self.geometry.tail_start = 0;
    }

    /// Current descriptive geometry after successful append operations.
    pub fn geometry(&self) -> WorkspacePagedGeometry {
        self.geometry
    }
    /// Exact ordered retained blocks; no flattened history is synthesized.
    pub fn blocks(&self) -> &[WorkspacePagedBlock] {
        &self.blocks
    }
    /// Current partial tail, if any.
    pub fn tail(&self) -> Option<&[WorkspaceTensor; 2]> {
        self.tail.as_ref()
    }

    fn discard_after_attention(&mut self, context: &WorkspaceContext) -> Result<(), Error> {
        context.charge_metadata(std::mem::size_of::<(
            &mut Self,
            &WorkspaceContext,
            Option<i32>,
            i64,
            i64,
            &WorkspacePagedBlock,
            Result<(), Error>,
        )>())?;
        if !self.geometry.retain_discarded {
            if let Some(window) = self.geometry.window {
                let prefix = i64::from(self.geometry.prefix_tokens);
                let visible = (self.geometry.offset - i64::from(window)).max(prefix);
                self.blocks.retain(|block| {
                    block.retained_source
                        || !crate::cache::CacheBlockSelection::outside_retained_window(
                            block.start,
                            block.end,
                            visible,
                            prefix,
                        )
                });
            }
        }
        Ok(())
    }

    /// Executes an already-normalized input, matching native append_inner. The
    /// key-only caller supplies its real one-channel persistence sentinel.
    /// Failure restores the prior state while accepted trace costs remain spent.
    pub fn append_normalized(
        &mut self,
        values: [WorkspaceTensor; 2],
        retain_for_attention: bool,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        if !context.shares_trace(&self.context) {
            return Err(context.metadata_error(format_args!("paged append uses another context")));
        }
        context.charge_metadata(
            PagedAppendPlan::control_bytes::<Append<'_>>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let input_len = *values[0].shape().get(2).ok_or_else(|| {
            context.metadata_error(format_args!("paged append requires rank-four input"))
        })?;
        validate_pair(&self.geometry, &values, i64::from(input_len), context)?;
        let tail_len = i32::try_from(self.geometry.offset - self.geometry.tail_start)
            .map_err(|cause| context.metadata_source(cause))?;
        let plan = PagedAppendPlan::new(
            self.geometry.block_size,
            input_len,
            tail_len,
            self.geometry.tail_start,
            self.geometry.offset,
        )
        .map_err(|cause| context.metadata_source(cause))?;
        context.reserve_metadata_vec(&mut self.blocks, plan.sealed_blocks())?;
        context.charge_metadata(std::mem::size_of::<(
            Option<[WorkspaceTensor; 2]>,
            WorkspacePagedGeometry,
            usize,
        )>())?;
        let previous_tail = self.tail.clone();
        let previous_geometry = self.geometry;
        let previous_blocks = self.blocks.len();
        let result = plan.run(
            &mut Append {
                state: self,
                input: values,
                context,
            },
            retain_for_attention,
        );
        if result.is_err() {
            self.tail = previous_tail;
            self.geometry = previous_geometry;
            self.blocks.truncate(previous_blocks);
        }
        result
    }
}
fn validate_pair(
    geometry: &WorkspacePagedGeometry,
    values: &[WorkspaceTensor; 2],
    tokens: i64,
    context: &WorkspaceContext,
) -> Result<(), Error> {
    context.charge_metadata(std::mem::size_of::<(
        &WorkspacePagedGeometry,
        &[WorkspaceTensor; 2],
        i64,
        i32,
        [i32; 3],
        &WorkspaceContext,
        Result<(), Error>,
    )>())?;
    context.validate_values(values)?;
    if values
        .iter()
        .any(|value| value.layout().dtype() != WorkspaceDtype::Float32)
    {
        return Err(
            context.metadata_error(format_args!("paged arrays differ from selected geometry"))
        );
    }
    validate_pair_shapes(
        geometry,
        [values[0].shape(), values[1].shape()],
        tokens,
        context,
    )
}
fn validate_pair_shapes(
    geometry: &WorkspacePagedGeometry,
    shapes: [&[i32]; 2],
    tokens: i64,
    context: &WorkspaceContext,
) -> Result<(), Error> {
    context.charge_metadata(std::mem::size_of::<(
        &WorkspacePagedGeometry,
        [&[i32]; 2],
        i64,
        i32,
        [i32; 3],
        &WorkspaceContext,
        Result<(), Error>,
    )>())?;
    let tokens = i32::try_from(tokens).map_err(|cause| context.metadata_source(cause))?;
    let [batch, heads, width] = geometry.dimensions;
    if tokens <= 0
        || shapes[0] != [batch, heads, tokens, width]
        || shapes[1]
            != [
                batch,
                heads,
                tokens,
                if geometry.key_only { 1 } else { width },
            ]
    {
        return Err(
            context.metadata_error(format_args!("paged arrays differ from selected geometry"))
        );
    }
    Ok(())
}

struct Append<'a> {
    state: &'a mut WorkspacePagedAppendState,
    input: [WorkspaceTensor; 2],
    context: &'a WorkspaceContext,
}
impl PagedAppendMechanisms for Append<'_> {
    type Pair = [WorkspaceTensor; 2];
    type Error = Error;
    fn slice_input(&mut self, range: std::ops::Range<i32>) -> Result<Self::Pair, Error> {
        let axes = [
            Index::Full,
            Index::Full,
            Index::Range(range.start, range.end),
            Index::Full,
        ];
        let pair = [
            self.input[0].index(&axes, self.context)?,
            self.input[1].index(&axes, self.context)?,
        ];
        self.context.retain_values(&[&pair[0], &pair[1]])?;
        Ok(pair)
    }
    fn join_tail(&mut self, [keys, values]: Self::Pair) -> Result<Self::Pair, Error> {
        let pair = match &self.state.tail {
            Some([old_keys, old_values]) => [
                WorkspaceTensor::concatenate(&[old_keys.clone(), keys], -2, self.context)?,
                WorkspaceTensor::concatenate(&[old_values.clone(), values], -2, self.context)?,
            ],
            None => [keys, values],
        };
        self.context.retain_values(&[&pair[0], &pair[1]])?;
        Ok(pair)
    }
    fn publish_tail(&mut self, start: i64, _end: i64, pair: Self::Pair) -> Result<(), Error> {
        self.state.geometry.tail_start = start;
        self.state.tail = Some(pair);
        Ok(())
    }
    fn seal_tail(&mut self) -> Result<(), Error> {
        let values = self
            .state
            .tail
            .as_ref()
            .expect("shared append published complete tail");
        // The native seal lends two independently owned handles to the
        // manager while retaining the original pair for rollback on failure.
        let publication = values.clone();
        self.context
            .complete_values(&[&publication[0], &publication[1]])?;
        let values = self.state.tail.take().expect("validated complete tail");
        let start = self.state.geometry.tail_start;
        let end = start + i64::from(self.state.geometry.block_size);
        self.state
            .blocks
            .push(WorkspacePagedBlock::new(start, end, publication, false));
        drop(values);
        self.state.geometry.tail_start = end;
        Ok(())
    }
    fn finish(&mut self, end: i64, retain_for_attention: bool) -> Result<(), Error> {
        let start = self.state.geometry.offset;
        self.state.geometry.offset = end;
        if !retain_for_attention {
            // A newly sealed page can be a policy victim before final-window
            // discard, so retain its conditional store even when it expires in
            // this same invocation. This is the actual pre-discard population.
            self.context
                .charge_metadata(std::mem::size_of::<(&mut Self, i64, i64, bool)>())?;
            self.state.add_host_sources(start, end, self.context)?;
            self.state.discard_after_attention(self.context)?;
        }
        Ok(())
    }
}
