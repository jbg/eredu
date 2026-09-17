//! Finite conditional Host transfer inventory from actual page equations.
use super::*;
use eredu_nn::workspace::{WorkspaceFloatingType, WorkspaceStoredHostValue};
use std::{cell::RefCell, mem::size_of, rc::Rc};

#[derive(Debug)]
enum Source {
    Retained {
        values: [WorkspaceTensor; 2],
        types: [WorkspaceFloatingType; 2],
    },
    Stored([WorkspaceStoredHostValue; 2]),
    Read([eredu_nn::workspace::WorkspaceHostReadValue; 2]),
}
/// One exact page's possible immutable Host source. This descriptive entry is
/// neither a manager ID nor permission to copy any actual native array.
#[derive(Debug)]
pub struct WorkspacePagedHostEntry {
    start: i64,
    end: i64,
    query_start: i64,
    context_end: i64,
    source: Source,
}
impl WorkspacePagedHostEntry {
    /// Absolute page interval supplied by the actual append/source worker.
    pub fn range(&self) -> std::ops::Range<i64> {
        self.start..self.end
    }
    /// First selected equation that may need this stored backing.
    pub fn first_invocation(&self) -> (i64, i64) {
        (self.query_start, self.context_end)
    }
    /// Whether this entry requires a separate accepted Host store destination.
    pub fn requires_store(&self) -> bool {
        matches!(self.source, Source::Stored(_))
    }
    /// Exact key/value geometry; an invalid component has no description.
    pub fn shape(&self, component: usize) -> Option<&[i32]> {
        match &self.source {
            Source::Retained { values, .. } => values.get(component).map(|v| v.layout().shape()),
            Source::Stored(values) => values.get(component).map(WorkspaceStoredHostValue::shape),
            Source::Read(values) => values
                .get(component)
                .map(eredu_nn::workspace::WorkspaceHostReadValue::shape),
        }
    }
    /// Actual scalar representation, independent of logical extent class.
    pub fn floating_type(&self, component: usize) -> Option<WorkspaceFloatingType> {
        match &self.source {
            Source::Retained { types, .. } => types.get(component).copied(),
            Source::Stored(values) => values
                .get(component)
                .map(WorkspaceStoredHostValue::floating_type),
            Source::Read(values) => values
                .get(component)
                .map(eredu_nn::workspace::WorkspaceHostReadValue::floating_type),
        }
    }
    /// Exact optional store occurrence for matching the native source program.
    pub fn stored(&self) -> Option<&[WorkspaceStoredHostValue; 2]> {
        match &self.source {
            Source::Stored(values) => Some(values),
            _ => None,
        }
    }
}
/// One selected scan-pass occurrence, preserving absolute rather than dense
/// geometry. A native consumer may skip copying only after checking hot storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspacePagedHostLoad {
    /// Index in this same immutable source trace, not a physical module ordinal.
    pub entry: usize,
    /// Absolute first query position for the actual equation.
    pub query_start: i64,
    /// Actual exclusive cache frontier.
    pub context_end: i64,
    /// Actual normalization/value pass selected by the shared scan driver.
    pub pass: usize,
}
#[derive(Debug)]
struct Trace {
    entries: Vec<WorkspacePagedHostEntry>,
    loads: Vec<WorkspacePagedHostLoad>,
}
/// Paid per-source trace of the maximal selected transfer path. Native policy,
/// source authentication, allocation and completion remain separate consumers.
#[derive(Clone, Debug)]
pub struct WorkspacePagedHostTrace {
    trace: Rc<RefCell<Trace>>,
    context: WorkspaceContext,
}
impl WorkspacePagedHostTrace {
    /// Constructs only paid empty metadata storage, without native resources.
    pub fn new(context: &WorkspaceContext) -> Result<Self, Error> {
        context.charge_metadata(size_of::<(
            &WorkspaceContext,
            Self,
            Trace,
            Result<Self, Error>,
        )>())?;
        Ok(Self {
            trace: context.metadata_rc(RefCell::new(Trace {
                entries: Vec::new(),
                loads: Vec::new(),
            }))?,
            context: context.clone(),
        })
    }
    /// Saved-source copying may share this exact future itinerary only before
    /// any source entry, deferred store or selected load has been recorded.
    pub fn validate_unstarted(&self, context: &WorkspaceContext) -> Result<(), Error> {
        context.charge_metadata(size_of::<(
            &Self,
            &WorkspaceContext,
            std::cell::Ref<'_, Trace>,
            Result<std::cell::Ref<'_, Trace>, std::cell::BorrowError>,
            Result<(), Error>,
        )>())?;
        if !context.shares_trace(&self.context) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let trace = self
            .trace
            .try_borrow()
            .map_err(|_| WorkspaceMetadataError::Unqualified)?;
        if !trace.entries.is_empty() || !trace.loads.is_empty() {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        Ok(())
    }
    /// Lends the same exact source inventory while its mutation is excluded.
    pub fn with_entries<R>(
        &self,
        inspect: impl FnOnce(&[WorkspacePagedHostEntry]) -> R,
    ) -> Result<R, Error> {
        self.context.charge_metadata(
            size_of::<(
                &Self,
                &[WorkspacePagedHostEntry],
                std::cell::Ref<'_, Trace>,
                Result<R, Error>,
            )>()
            .checked_add(std::mem::size_of_val(&inspect))
            .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let trace = self
            .trace
            .try_borrow()
            .map_err(|_| WorkspaceMetadataError::Unqualified)?;
        Ok(inspect(&trace.entries))
    }
    /// Lends the actual ordered conditional loads from the shared scan driver.
    pub fn with_loads<R>(
        &self,
        inspect: impl FnOnce(&[WorkspacePagedHostLoad]) -> R,
    ) -> Result<R, Error> {
        self.context.charge_metadata(
            size_of::<(
                &Self,
                &[WorkspacePagedHostLoad],
                std::cell::Ref<'_, Trace>,
                Result<R, Error>,
            )>()
            .checked_add(std::mem::size_of_val(&inspect))
            .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let trace = self
            .trace
            .try_borrow()
            .map_err(|_| WorkspaceMetadataError::Unqualified)?;
        Ok(inspect(&trace.loads))
    }
    // Any retained page may become a later policy victim, including another
    // layer's page within the same Model role. Keep each exact source's possible
    // one-time store in every subsequent invocation, without minting another
    // Host destination or choosing a cache policy during quotation.
    fn declare_deferred_stores(&self, context: &WorkspaceContext) -> Result<(), Error> {
        if !context.shares_trace(&self.context) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        context.charge_metadata(size_of::<(
            &Self,
            &WorkspaceContext,
            std::cell::Ref<'_, Trace>,
            std::slice::Iter<'_, WorkspacePagedHostEntry>,
            std::slice::Iter<'_, WorkspaceStoredHostValue>,
            Result<(), Error>,
        )>())?;
        let trace = self
            .trace
            .try_borrow()
            .map_err(|_| WorkspaceMetadataError::Unqualified)?;
        for entry in &trace.entries {
            if let Source::Stored(values) = &entry.source {
                for value in values {
                    value.declare_deferred_store(context)?;
                }
            }
        }
        Ok(())
    }
    fn add(
        &self,
        block: &WorkspacePagedBlock,
        query_start: i64,
        context_end: i64,
        context: &WorkspaceContext,
    ) -> Result<usize, Error> {
        context.charge_metadata(size_of::<(
            &Self,
            &WorkspacePagedBlock,
            i64,
            i64,
            &WorkspaceContext,
            Source,
            WorkspacePagedHostEntry,
            std::cell::RefMut<'_, Trace>,
            [WorkspaceFloatingType; 2],
            Result<usize, Error>,
        )>())?;
        if !context.shares_trace(&self.context) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let mut trace = self
            .trace
            .try_borrow_mut()
            .map_err(|_| WorkspaceMetadataError::Unqualified)?;
        context.reserve_metadata_vec(&mut trace.entries, 1)?;
        let source = if let PagedBlockValues::Read(values)
        | PagedBlockValues::Backed { read: values, .. } = &block.values
        {
            Source::Read(values.clone())
        } else if let Some(types) = block.host {
            Source::Retained {
                values: block
                    .values
                    .tensors()
                    .ok_or(WorkspaceMetadataError::Unqualified)?
                    .clone(),
                types,
            }
        } else if let Some(values) = &block.stored {
            Source::Stored(values.clone())
        } else {
            let values = block
                .values
                .tensors()
                .ok_or(WorkspaceMetadataError::Unqualified)?;
            let types = [
                values[0]
                    .layout()
                    .representation()
                    .ok_or(WorkspaceMetadataError::Unqualified)?
                    .dtype(),
                values[1]
                    .layout()
                    .representation()
                    .ok_or(WorkspaceMetadataError::Unqualified)?
                    .dtype(),
            ];
            Source::Stored([
                context.store_host_value(&values[0], types[0])?,
                context.store_host_value(&values[1], types[1])?,
            ])
        };
        let index = trace.entries.len();
        trace.entries.push(WorkspacePagedHostEntry {
            start: block.start,
            end: block.end,
            query_start,
            context_end,
            source,
        });
        Ok(index)
    }
    pub(super) fn load(
        &self,
        index: usize,
        query_start: i64,
        context_end: i64,
        pass: usize,
        context: &WorkspaceContext,
    ) -> Result<[WorkspaceTensor; 2], Error> {
        context.charge_metadata(size_of::<(
            &Self,
            usize,
            i64,
            i64,
            usize,
            &WorkspaceContext,
            WorkspacePagedHostLoad,
            std::cell::RefMut<'_, Trace>,
            [WorkspaceTensor; 2],
            Result<[WorkspaceTensor; 2], Error>,
        )>())?;
        if !context.shares_trace(&self.context) || pass > 1 {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let mut trace = self
            .trace
            .try_borrow_mut()
            .map_err(|_| WorkspaceMetadataError::Unqualified)?;
        context.reserve_metadata_vec(&mut trace.loads, 1)?;
        let entry = trace
            .entries
            .get(index)
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        if query_start < entry.query_start || context_end < entry.context_end {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let values = match &entry.source {
            Source::Retained { values, types } => [
                values[0].transfer_host_floating(types[0], context)?,
                values[1].transfer_host_floating(types[1], context)?,
            ],
            Source::Stored(values) => [values[0].load(context)?, values[1].load(context)?],
            Source::Read(values) => [values[0].load(context)?, values[1].load(context)?],
        };
        trace.loads.push(WorkspacePagedHostLoad {
            entry: index,
            query_start,
            context_end,
            pass,
        });
        Ok(values)
    }
}
impl WorkspacePagedAppendState {
    /// Attaches one source-owned conditional transfer program. This describes
    /// its selected Host mechanism but never grants manager or native access.
    pub fn with_host_transfers(mut self, trace: WorkspacePagedHostTrace) -> Result<Self, Error> {
        self.context.charge_metadata(size_of::<(
            Self,
            WorkspacePagedHostTrace,
            Result<Self, Error>,
        )>())?;
        if self.host_trace.is_some() || !trace.context.shares_trace(&self.context) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        self.host_trace = Some(trace);
        Ok(self)
    }
    pub(super) fn prepare_host_sources(
        &mut self,
        query_start: i64,
        context_end: i64,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        context.charge_metadata(size_of::<(
            &mut Self,
            i64,
            i64,
            &WorkspaceContext,
            Option<WorkspacePagedHostTrace>,
            Result<(), Error>,
        )>())?;
        let Some(trace) = &self.host_trace else {
            return Ok(());
        };
        // New entries below already emit their first store in this invocation.
        // Existing entries need the same effect again because the native policy
        // can first select them for demotion here instead of at their birth.
        trace.declare_deferred_stores(context)?;
        self.add_host_sources(query_start, context_end, context)
    }
    /// Adds only newly published pages after a local visible update. Existing
    /// deferred stores were already declared before reading that old window.
    pub(super) fn add_host_sources(
        &mut self,
        query_start: i64,
        context_end: i64,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        context.charge_metadata(size_of::<(
            &mut Self,
            i64,
            i64,
            &WorkspaceContext,
            std::slice::IterMut<'_, WorkspacePagedBlock>,
            Result<(), Error>,
        )>())?;
        let Some(trace) = &self.host_trace else {
            return Ok(());
        };
        for block in &mut self.blocks {
            if block.host_entry.is_none() {
                block.host_entry = Some(trace.add(block, query_start, context_end, context)?);
            }
        }
        Ok(())
    }
    /// One shared actual source operation for blockwise attention and local
    /// visible readout. The canonical invocation coordinates remain separate
    /// from the requested old-page window and the current mutable frontier.
    pub(super) fn load_block_for_invocation(
        &mut self,
        index: usize,
        query_start: i64,
        context_end: i64,
        pass: usize,
        context: &WorkspaceContext,
    ) -> Result<[WorkspaceTensor; 2], Error> {
        context.charge_metadata(size_of::<(
            &mut Self,
            usize,
            i64,
            i64,
            usize,
            &WorkspaceContext,
            Option<WorkspacePagedHostTrace>,
            &mut WorkspacePagedBlock,
            [WorkspaceTensor; 2],
            Result<[WorkspaceTensor; 2], Error>,
        )>())?;
        if !context.shares_trace(&self.context) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let host_trace = self.host_trace.clone();
        let block = self
            .blocks
            .get_mut(index)
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        if let Some(trace) = host_trace {
            let entry = block
                .host_entry
                .ok_or(WorkspaceMetadataError::Unqualified)?;
            let values = trace.load(entry, query_start, context_end, pass, context)?;
            context.retain_values(&[&values[0], &values[1]])?;
            block.values = PagedBlockValues::Tensor(values);
            block.host = None;
            block.stored = None;
        } else {
            block.promote(context)?;
        }
        Ok(block
            .values
            .tensors()
            .ok_or(WorkspaceMetadataError::Unqualified)?
            .clone())
    }
}
