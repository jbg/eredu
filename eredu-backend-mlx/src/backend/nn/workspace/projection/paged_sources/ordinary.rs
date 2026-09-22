//! Ordinary publication destinations from the retained canonical paged source.
mod host;
use super::*;
use crate::backend::{
    error::Error as NativeError,
    nn::workspace::{OrdinaryCallControls, OrdinaryNativeControls},
    runtime::cache::{
        kv::ProjectedPagedSource,
        residency::{
            CacheBlockArrays, CacheBlockMetadata, CacheResidencyError, CacheResidencyManager,
            CacheSourceError, CacheSourceFailure, InstalledManagerCatalog,
            PreparedFloatingBlockMetadata, PreparedManagerCatalog,
        },
    },
};
use eredu_core::{
    HostPreparationAuthority,
    cache::{CacheBlockId, CacheRepresentation},
};
use eredu_runtime::{
    cache::PagedAppendPlan,
    working_memory::{
        InferenceSpanWorkspacePlan, InferenceTextStep, InferenceWorkspaceSpan, WorkingMemoryError,
    },
};
pub(crate) use host::OrdinaryPagedHostScan;
use safemlx::error::Exception;
use std::{
    mem::{size_of, size_of_val},
    ops::Range,
};

/// Every alias retains the independent planning payer after its shared node.
/// No raw or weak node handle escapes this descriptive source owner.
#[derive(Clone)]
pub(crate) struct OrdinaryPagedProgram {
    inner: Rc<Program>,
    _funding: Option<HostMetadataFunding>,
}
struct Program {
    host: RefCell<Option<host::HostProgram>>,
    transfer_requirements: Option<eredu_core::DomainMemoryRequirements>,
    pool: eredu_runtime::working_memory::MemoryLedger,
    sources: ProjectedPagedSources,
    plan: InferenceSpanWorkspacePlan,
    prefill: usize,
    forwards: usize,
    state: RefCell<State>,
    context: WorkspaceContext,
}
struct State {
    catalogs: Vec<Option<PreparedManagerCatalog>>,
    installed: Vec<InstalledManagerCatalog>,
    rows: Vec<Option<Row>>,
    initialized: bool,
    installing: bool,
    failed: bool,
}
struct Row {
    source: usize,
    ordinal: usize,
    append: PagedAppendPlan,
    publications: Vec<Publication>,
    used: bool,
    completed: bool,
    reporting: usize,
    scan: Option<super::ordinary_scan::PreparedOrdinaryPagedScan>,
    scan_used: bool,
    checkpoint: crate::backend::runtime::cache::kv::PreparedPagedCheckpoint,
}
impl super::host_program::HostAppendSource for Row {
    fn source_index(&self) -> usize {
        self.source
    }
    fn ordinal(&self) -> usize {
        self.ordinal
    }
    fn scan_coordinates(&self) -> Option<(i64, i64)> {
        self.scan
            .as_ref()
            .map(|scan| (scan.query_start, scan.context_end))
    }
    fn publication(&self, range: &Range<i64>) -> Option<&CacheBlockId> {
        self.publications
            .iter()
            .find(|row| row.id.start == range.start && row.id.end == range.end)
            .map(|row| &row.id)
    }
}
struct Publication {
    id: CacheBlockId,
    protected: bool,
    metadata: Option<PreparedFloatingBlockMetadata>,
}
#[derive(Clone)]
pub(crate) struct OrdinaryPagedWork {
    program: OrdinaryPagedProgram,
    ordinals: Range<usize>,
}
impl std::fmt::Debug for OrdinaryPagedProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinaryPagedProgram")
            .field("forwards", &self.inner.forwards)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Debug for OrdinaryPagedWork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinaryPagedWork")
            .field("ordinals", &self.ordinals)
            .finish_non_exhaustive()
    }
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum OrdinaryPagedCause {
    #[error(transparent)]
    Source(#[from] CacheSourceError),
    #[error(transparent)]
    Preparation(#[from] CacheSourceFailure),
    #[error(transparent)]
    Residency(#[from] CacheResidencyError),
    #[error(transparent)]
    Stream(#[from] safemlx::StreamCopyCause),
    #[error(transparent)]
    Native(#[from] Exception),
}
#[derive(Debug, thiserror::Error)]
#[error("ordinary paged publication: {cause}")]
struct Failure {
    #[source]
    cause: OrdinaryPagedCause,
    _host: HostPreparationAuthority,
}
fn failure(cause: impl Into<OrdinaryPagedCause>, host: &HostPreparationAuthority) -> Exception {
    Exception::from_retained_source(Failure {
        cause: cause.into(),
        _host: host.clone(),
    })
}
fn same_layer(a: &ProjectedPagedSource, b: &ProjectedPagedSource) -> bool {
    a.manager().same_catalog(b.manager()) && a.geometry().global_layer == b.geometry().global_layer
}
impl ProjectedPagedSources {
    pub(crate) fn prepare_ordinary(
        &self,
        plan: &InferenceSpanWorkspacePlan,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        context: &WorkspaceContext,
    ) -> Result<OrdinaryPagedProgram, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let metadata = |cause| CacheSourceFailure::metadata(cause, context);
        let frames = [
            size_of::<OrdinaryPagedProgram>(),
            size_of::<Program>(),
            size_of::<State>(),
            size_of::<Row>(),
            size_of::<Publication>(),
            size_of::<Option<CacheSourceFailure>>(),
            size_of::<(
                &Self,
                &InferenceSpanWorkspacePlan,
                &eredu_runtime::working_memory::MemoryLedger,
                &WorkspaceContext,
            )>(),
            size_of::<Result<OrdinaryPagedProgram, CacheSourceFailure>>(),
        ];
        context
            .charge_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|e| metadata(e.into()))?;
        let forwards = plan
            .generation_forward_count()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let prefill = plan
            .generation_records()
            .ok_or_else(|| fail(CacheSourceError::Identity))?
            .take_while(|record| matches!(record.span(), InferenceWorkspaceSpan::Prefill(_)))
            .count();
        let sources = &self.inner.sources;
        let managers = sources
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                !sources[..*i]
                    .iter()
                    .any(|prior| prior.manager().same_catalog(s.manager()))
            })
            .count();
        let layers = sources
            .iter()
            .enumerate()
            .filter(|(i, s)| !sources[..*i].iter().any(|prior| same_layer(prior, s)))
            .count();
        let mut catalogs = context.metadata_vec(managers).map_err(metadata)?;
        for (index, source) in sources.iter().enumerate() {
            if sources[..index]
                .iter()
                .any(|p| p.manager().same_catalog(source.manager()))
            {
                continue;
            }
            let work = catalogs::CatalogWork {
                sources,
                first: source,
                plan,
                context,
            };
            let catalog =
                source
                    .manager()
                    .with_source_loan(source.selection(), context, move |loan| {
                        work.prepare(loan)
                    })?;
            catalogs.push(Some(catalog));
        }
        let mut rows = context
            .metadata_vec(
                layers
                    .checked_mul(forwards)
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(metadata)?;
        for (index, source) in sources.iter().enumerate() {
            if let Some(prior) = sources[..index]
                .iter()
                .find(|prior| same_layer(prior, source))
            {
                if prior.append_geometry() != source.append_geometry() {
                    return Err(fail(CacheSourceError::Geometry));
                }
                continue;
            }
            let reporting = catalogs
                .iter()
                .flatten()
                .find(|c| c.manager().same_catalog(source.manager()))
                .ok_or_else(|| fail(CacheSourceError::Identity))?
                .publication_control_bytes();
            let mut failed = None;
            let mut checkpoint_blocks = source.geometry().blocks.len();
            let result = catalogs::visit_append_plans(source, plan, |ordinal, append| {
                match prepare_row(
                    source,
                    index,
                    ordinal,
                    append,
                    reporting,
                    checkpoint_blocks,
                    context,
                ) {
                    Ok(row) => {
                        checkpoint_blocks = checkpoint_blocks
                            .checked_add(row.publications.len())
                            .ok_or(CacheSourceError::Overflow)?;
                        rows.push(Some(row));
                        Ok(())
                    }
                    Err(cause) => {
                        failed = Some(cause);
                        Err(CacheSourceError::Geometry)
                    }
                }
            });
            if let Some(cause) = failed {
                return Err(cause);
            }
            result.map_err(fail)?;
        }
        // The future scan inventory comes only from retained existing IDs and
        // actual earlier/current publication rows of this same physical source.
        for index in 0..rows.len() {
            let row = rows[index].as_ref().expect("prepared row");
            let source = &sources[row.source];
            let ids = source
                .geometry()
                .blocks
                .iter()
                .map(|block| &block.id)
                .chain(
                    rows.iter()
                        .flatten()
                        .filter(|candidate| {
                            candidate.source == row.source && candidate.ordinal <= row.ordinal
                        })
                        .flat_map(|candidate| {
                            candidate
                                .publications
                                .iter()
                                .map(|publication| &publication.id)
                        }),
                );
            let scan = super::ordinary_scan::PreparedOrdinaryPagedScan::prepare(
                source,
                &row.append,
                ids,
                context,
            )?;
            rows[index].as_mut().expect("prepared row").scan = Some(scan);
        }
        let host = host::HostProgram::prepare(sources, &rows, plan, pool, context)?;
        let (host, transfer_requirements) = match host {
            Some(prepared) => (Some(prepared.program), Some(prepared.requirements)),
            None => (None, None),
        };
        let installed = context.metadata_vec(managers).map_err(metadata)?;
        let inner = context
            .metadata_rc(Program {
                host: RefCell::new(host),
                transfer_requirements,
                pool: pool.clone(),
                sources: self.clone(),
                plan: plan.clone(),
                prefill,
                forwards,
                state: RefCell::new(State {
                    catalogs,
                    installed,
                    rows,
                    initialized: false,
                    installing: false,
                    failed: false,
                }),
                context: context.clone(),
            })
            .map_err(|e| metadata(e.into()))?;
        Ok(OrdinaryPagedProgram {
            inner,
            _funding: context.metadata_funding(),
        })
    }
}
fn prepare_row(
    source: &ProjectedPagedSource,
    index: usize,
    ordinal: usize,
    append: PagedAppendPlan,
    reporting: usize,
    checkpoint_blocks: usize,
    context: &WorkspaceContext,
) -> Result<Row, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    context
        .charge_metadata(size_of::<(
            Row,
            Publication,
            PagedAppendPlan,
            [i32; 3],
            [[i32; 4]; 2],
            CacheBlockId,
            Result<Row, CacheSourceFailure>,
        )>())
        .map_err(|e| CacheSourceFailure::metadata(e.into(), context))?;
    let geometry = source.append_geometry();
    let [batch, heads, width] = geometry.dimensions;
    let shapes = [
        [batch, heads, geometry.block_size, width],
        [
            batch,
            heads,
            geometry.block_size,
            if geometry.key_only { 1 } else { width },
        ],
    ];
    let mut publications = context
        .metadata_vec(append.sealed_blocks())
        .map_err(|e| CacheSourceFailure::metadata(e, context))?;
    for step in append.steps().filter(|step| step.seals()) {
        let tail = step.tail();
        let id = CacheBlockId {
            session_id: source.geometry().session_id,
            global_layer: source.geometry().global_layer,
            representation: CacheRepresentation::KeyValue,
            start: tail.start,
            end: tail.end,
            rank: source.geometry().rank,
        };
        let metadata = PreparedFloatingBlockMetadata::prepare(
            CacheRepresentation::KeyValue,
            [&shapes[0], &shapes[1]],
            context,
        )?;
        publications.push(Publication {
            id,
            protected: tail.end <= i64::from(geometry.prefix_tokens),
            metadata: Some(metadata),
        });
    }
    if publications.len() != append.sealed_blocks() {
        return Err(fail(CacheSourceError::Identity));
    }
    Ok(Row {
        source: index,
        ordinal,
        append,
        publications,
        used: false,
        completed: false,
        reporting,
        scan: None,
        scan_used: false,
        checkpoint: crate::backend::runtime::cache::kv::PreparedPagedCheckpoint::prepare(
            source,
            append,
            checkpoint_blocks,
            context,
        )?,
    })
}
impl OrdinaryPagedProgram {
    pub(crate) fn for_step(
        &self,
        step: &InferenceTextStep,
        prefill: bool,
    ) -> Result<OrdinaryPagedWork, NativeError> {
        let invalid = || NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch);
        if step.request().geometry() != self.inner.plan.geometry()
            || prefill != (step.attempt() == 0)
        {
            return Err(invalid());
        }
        let ordinals = if prefill {
            0..self.inner.prefill
        } else {
            let index = usize::try_from(step.attempt().checked_sub(1).ok_or_else(invalid)?)
                .map_err(|_| invalid())?
                .checked_add(self.inner.prefill)
                .filter(|n| *n < self.inner.forwards)
                .ok_or_else(invalid)?;
            index..index.checked_add(1).ok_or_else(invalid)?
        };
        Ok(OrdinaryPagedWork {
            program: self.clone(),
            ordinals,
        })
    }
}
impl OrdinaryPagedWork {
    pub(crate) fn source_error(
        cause: impl Into<OrdinaryPagedCause>,
        host: &HostPreparationAuthority,
    ) -> Exception {
        failure(cause, host)
    }
    pub(crate) fn activate(&self, host: &HostPreparationAuthority) -> Result<(), Exception> {
        let program = &self.program.inner;
        let (mut catalogs, mut installed) = {
            let mut state = program
                .state
                .try_borrow_mut()
                .map_err(|_| failure(CacheSourceError::Busy, host))?;
            if state.failed || state.installing {
                return Err(failure(CacheSourceError::Identity, host));
            }
            if state.initialized {
                return Ok(());
            }
            state.installing = true;
            (
                std::mem::take(&mut state.catalogs),
                std::mem::take(&mut state.installed),
            )
        };
        let mut failed = None;
        for slot in &mut catalogs {
            let Some(catalog) = slot.take() else { continue };
            match catalog.install() {
                Ok(value) => installed.push(value),
                Err(error) => {
                    let (cause, retained) = error.into_parts();
                    *slot = Some(retained);
                    failed = Some(cause);
                    break;
                }
            }
        }
        if failed.is_none() {
            let result = match program.host.try_borrow_mut() {
                Ok(mut slot) => match slot.as_mut() {
                    Some(value) => value.install_disk_workers(&program.context),
                    None => Ok(()),
                },
                Err(_) => Err(CacheSourceFailure::source(
                    CacheSourceError::Busy,
                    &program.context,
                )),
            };
            if let Err(cause) = result {
                failed = Some(cause);
            }
        }
        let mut state = program
            .state
            .try_borrow_mut()
            .map_err(|_| failure(CacheSourceError::Busy, host))?;
        state.catalogs = catalogs;
        state.installed = installed;
        state.installing = false;
        state.failed = failed.is_some();
        state.initialized = failed.is_none();
        drop(state);
        match failed {
            Some(cause) => Err(failure(cause, host)),
            None => Ok(()),
        }
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;

/// One checked-out destination from the actual request and canonical manager.
/// Native execution remains authorized by the enclosing ordinary Work.
pub(crate) struct OrdinaryPagedAppend {
    work: OrdinaryPagedWork,
    row: Option<Row>,
    index: usize,
    source: usize,
    installed: InstalledManagerCatalog,
    host: HostPreparationAuthority,
    published: usize,
    succeeded: bool,
    step: usize,
    tail_ready: bool,
    tail_cleared: bool,
    prepared_tail_bytes: Option<u64>,
    initial_tail_bytes: u64,
}
impl Drop for OrdinaryPagedAppend {
    fn drop(&mut self) {
        let Some(mut row) = self.row.take() else {
            return;
        };
        row.completed = self.succeeded;
        match self.work.program.inner.state.try_borrow_mut() {
            Ok(mut state) => {
                state.failed |= !self.succeeded;
                state.rows[self.index] = Some(row);
            }
            // A live conflicting loan cannot prove safe source retirement.
            Err(_) => {
                std::mem::forget((row, self.work.clone(), self.host.clone()));
            }
        }
    }
}
impl OrdinaryPagedWork {
    pub(crate) fn with_append<R>(
        &self,
        actual: super::PagedAppendInput<'_>,
        host: &HostPreparationAuthority,
        run: impl FnOnce(&mut OrdinaryPagedAppend) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        let owner = crate::backend::nn::shared::current_ordinary_execution_owner()?
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?;
        if owner.paged().is_none_or(|work| {
            !Rc::ptr_eq(&work.program.inner, &self.program.inner) || work.ordinals != self.ordinals
        }) {
            return Err(failure(CacheSourceError::Identity, owner.host()));
        }
        let host = owner.host();
        self.activate(host)?;
        let bank = &self.program.inner;
        let source = bank
            .sources
            .sources()
            .iter()
            .position(|source| {
                source.manager().same_catalog(actual.manager)
                    && source.geometry().global_layer == actual.global_layer
            })
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?;
        let retained = &bank.sources.sources()[source];
        let expected = retained.append_geometry();
        let current = actual.geometry;
        if actual.rank != retained.geometry().rank
            || current.block_size != expected.block_size
            || current.dimensions != expected.dimensions
            || current.window != expected.window
            || current.prefix_tokens != expected.prefix_tokens
            || current.key_only != expected.key_only
            || current.retain_discarded != expected.retain_discarded
            || actual
                .dtypes
                .into_iter()
                .any(|d| CacheBlockMetadata::floating_dtype_bytes(d).is_none())
            || actual.tail_dtypes.is_some_and(|d| d != actual.dtypes)
        {
            return Err(failure(CacheSourceError::Geometry, host));
        }
        let mut state = bank
            .state
            .try_borrow_mut()
            .map_err(|_| failure(CacheSourceError::Busy, host))?;
        if state.failed {
            return Err(failure(CacheSourceError::Identity, host));
        }
        let index = state
            .rows
            .iter()
            .position(|row| {
                row.as_ref().is_some_and(|row| {
                    row.source == source && self.ordinals.contains(&row.ordinal) && !row.used
                })
            })
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?;
        let row = state.rows[index].as_ref().expect("checked row");
        if row.append != actual.plan
            || current.offset != row.append.initial_frontier().2
            || current.tail_start != row.append.initial_frontier().0
        {
            return Err(failure(CacheSourceError::Identity, host));
        }
        let installed = state
            .installed
            .iter()
            .find(|i| i.manager().same_catalog(actual.manager))
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?
            .clone();
        let length = actual.plan.initial_frontier().1;
        let initial_tail_bytes = if length == 0 {
            if actual.tail_dtypes.is_some() {
                return Err(failure(CacheSourceError::Geometry, host));
            }
            0
        } else {
            let dtypes = actual
                .tail_dtypes
                .ok_or_else(|| failure(CacheSourceError::Geometry, host))?;
            let [b, h, w] = current.dimensions;
            CacheBlockMetadata::floating_bytes(
                [
                    &[b, h, length, w],
                    &[b, h, length, if current.key_only { 1 } else { w }],
                ],
                dtypes,
            )
            .map_err(|e| failure(e, host))?
        };
        let mut row = state.rows[index].take().expect("one checked-out row");
        row.used = true;
        drop(state);
        let mut claim = OrdinaryPagedAppend {
            work: self.clone(),
            row: Some(row),
            index,
            source,
            installed,
            host: host.clone(),
            published: 0,
            succeeded: false,
            step: 0,
            tail_ready: false,
            tail_cleared: false,
            prepared_tail_bytes: None,
            initial_tail_bytes,
        };
        let result = run(&mut claim);
        if result.is_ok() {
            if claim.published != claim.row().publications.len()
                || claim.step != claim.plan().steps().count()
                || claim.tail_ready
                || claim.tail_cleared
            {
                return Err(claim.error(CacheSourceError::Identity));
            }
            claim.succeeded = true;
        }
        result.map_err(|cause| failure(cause, host))
    }
}
impl OrdinaryPagedAppend {
    fn row(&self) -> &Row {
        self.row.as_ref().expect("live append row")
    }
    pub(crate) fn error(&self, cause: impl Into<OrdinaryPagedCause>) -> Exception {
        failure(cause, &self.host)
    }
    pub(crate) fn layer(&self) -> usize {
        self.work.program.inner.sources.sources()[self.source]
            .geometry()
            .global_layer
    }
    pub(crate) fn plan(&self) -> PagedAppendPlan {
        self.row().append
    }
    pub(crate) fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        if !self.installed.manager().same_catalog(manager)
            || generation != self.installed.initial_generation()
        {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn publication_id(&self, index: usize) -> Option<&CacheBlockId> {
        self.row().publications.get(index).map(|p| &p.id)
    }
    pub(crate) fn published_count(&self) -> usize {
        self.published
    }
    pub(crate) fn take_publication(
        &mut self,
        id: &CacheBlockId,
        arrays: &CacheBlockArrays,
    ) -> Result<(CacheBlockMetadata, bool), Exception> {
        if !self.tail_ready || !self.tail_cleared {
            return Err(self.error(CacheSourceError::Identity));
        }
        let host = self.host.clone();
        let published = self.published;
        let publication = self
            .row
            .as_mut()
            .unwrap()
            .publications
            .get_mut(published)
            .ok_or_else(|| failure(CacheSourceError::Identity, &host))?;
        if &publication.id != id {
            return Err(failure(CacheSourceError::Identity, &host));
        }
        let metadata = publication
            .metadata
            .as_ref()
            .ok_or_else(|| failure(CacheSourceError::Identity, &host))?;
        metadata
            .validate_arrays(arrays)
            .map_err(|e| failure(e, &host))?;
        Ok((
            publication
                .metadata
                .take()
                .unwrap()
                .bind(arrays)
                .map_err(|e| failure(e, &host))?,
            publication.protected,
        ))
    }
    pub(crate) fn published(&mut self, id: &CacheBlockId) -> Result<(), Exception> {
        if self.publication_id(self.published) != Some(id) {
            return Err(self.error(CacheSourceError::Identity));
        }
        self.published = self
            .published
            .checked_add(1)
            .ok_or_else(|| self.error(CacheSourceError::Overflow))?;
        self.step += 1;
        self.tail_ready = false;
        self.tail_cleared = false;
        self.prepared_tail_bytes = None;
        Ok(())
    }
}
impl OrdinaryPagedProgram {
    pub(crate) fn transfer_requirements(&self) -> Option<&eredu_core::DomainMemoryRequirements> {
        self.inner.transfer_requirements.as_ref()
    }
    pub(crate) fn caller_controls(&self) -> Option<OrdinaryCallControls> {
        let host = self.inner.host.try_borrow().ok()?;
        let state = self.inner.state.try_borrow().ok()?;
        let host = match host.as_ref() {
            Some(host) => host.caller_controls(&state.rows, 0..self.inner.forwards)?,
            None => OrdinaryCallControls::default(),
        };
        state.rows.iter().try_fold(host, |total, row| {
            total.append(row_controls(row.as_ref()?)?)
        })
    }
    pub(crate) fn step_call_controls(
        &self,
        step: &InferenceTextStep,
        prefill: bool,
    ) -> Result<OrdinaryCallControls, NativeError> {
        let work = self.for_step(step, prefill)?;
        let state = self
            .inner
            .state
            .try_borrow()
            .map_err(|_| NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let host = self
            .inner
            .host
            .try_borrow()
            .map_err(|_| NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let host = host
            .as_ref()
            .map(|host| host.caller_controls(&state.rows, work.ordinals.clone()))
            .unwrap_or(Some(OrdinaryCallControls::default()));
        let controls = state
            .rows
            .iter()
            .flatten()
            .filter(|row| work.ordinals.contains(&row.ordinal))
            .fold(host, |total, row| total?.append(row_controls(row)?));
        controls.ok_or(NativeError::PrefillControl(
            WorkingMemoryError::IdentityMismatch,
        ))
    }
}
fn row_controls(row: &Row) -> Option<OrdinaryCallControls> {
    // Eval and numerical wrappers are attached to the actual trace producers.
    // This source adds only the installed catalog, append and failure transports.
    let parts = [
        size_of::<OrdinaryPagedAppend>(),
        crate::backend::runtime::cache::kv::PreparedPagedCheckpoint::control_bytes()?,
        row.reporting,
        size_of::<ScanCheckout>(),
        size_of::<OrdinaryPagedHostScan<'_>>(),
        size_of::<OrdinaryPagedWork>(),
        size_of::<Row>(),
        size_of::<Option<Row>>(),
        size_of::<InstalledManagerCatalog>(),
        size_of::<Failure>().checked_mul(2)?,
        size_of::<super::PagedAppendInput<'_>>(),
        size_of::<Result<(), Exception>>(),
        Exception::retained_source_control_bytes::<Failure>()?.checked_mul(2)?,
        CacheResidencyManager::ordinary_append_control_bytes()?.checked_mul(
            row.append
                .steps()
                .count()
                .checked_add(row.publications.len().checked_mul(4)?)?
                .checked_add(3)?,
        )?,
        safemlx::Array::ordinary_clone_control_bytes()?
            .checked_mul(row.publications.len().checked_mul(2)?.checked_add(2)?)?,
        crate::backend::runtime::cache::kv::PagedKeyValueCache::ordinary_append_control_bytes()?,
        row.reporting.checked_mul(
            row.append
                .steps()
                .count()
                .checked_add(row.publications.len().checked_mul(3)?)?
                .checked_add(3)?,
        )?,
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)?;
    OrdinaryCallControls {
        metadata_bytes: u64::try_from(bytes).ok()?,
        observed: OrdinaryNativeControls::default(),
    }
    .append(row.scan.as_ref()?.caller_controls()?)
}

impl OrdinaryPagedAppend {
    pub(crate) fn validate_tail_update(
        &self,
        bytes: u64,
        end: i64,
        clearing: bool,
        restoring: bool,
    ) -> Result<(), Exception> {
        let step = self
            .plan()
            .steps()
            .nth(self.step)
            .ok_or_else(|| self.error(CacheSourceError::Identity))?;
        let range = step.tail();
        if (clearing && restoring)
            || end != range.end
            || (clearing && (!step.seals() || !self.tail_ready || self.tail_cleared))
            || (restoring && (!self.tail_ready || !self.tail_cleared))
            || (!clearing && !restoring && self.tail_ready)
            || bytes
                != if clearing {
                    0
                } else {
                    self.prepared_tail_bytes
                        .ok_or_else(|| self.error(CacheSourceError::Identity))?
                }
        {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn did_update_tail(&mut self, clearing: bool, restoring: bool) {
        if clearing {
            self.tail_cleared = true;
        } else if restoring {
            self.tail_cleared = false;
        } else if self
            .plan()
            .steps()
            .nth(self.step)
            .expect("validated step")
            .seals()
        {
            self.tail_ready = true;
        } else {
            self.step += 1;
            self.prepared_tail_bytes = None;
        }
    }
    pub(crate) fn validate_rollback_tail(&self, bytes: u64, end: i64) -> Result<(), Exception> {
        let (_, _, offset) = self.plan().initial_frontier();
        if end != offset || bytes != self.initial_tail_bytes {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn prepare_tail(&mut self, arrays: [&safemlx::Array; 2]) -> Result<(), Exception> {
        let step = self
            .plan()
            .steps()
            .nth(self.step)
            .ok_or_else(|| self.error(CacheSourceError::Identity))?;
        let source = &self.work.program.inner.sources.sources()[self.source];
        let g = source.append_geometry();
        let [b, h, w] = g.dimensions;
        let length = i32::try_from(step.tail().end - step.tail().start)
            .map_err(|_| self.error(CacheSourceError::Overflow))?;
        let shapes = [
            [b, h, length, w],
            [b, h, length, if g.key_only { 1 } else { w }],
        ];
        if arrays[0].shape() != shapes[0]
            || arrays[1].shape() != shapes[1]
            || arrays[0].dtype() != arrays[1].dtype()
        {
            return Err(self.error(CacheSourceError::Geometry));
        }
        let bytes = CacheBlockMetadata::floating_bytes(
            [&shapes[0], &shapes[1]],
            [arrays[0].dtype(), arrays[1].dtype()],
        )
        .map_err(|e| self.error(e))?;
        if bytes
            != u64::try_from(arrays[0].nbytes())
                .ok()
                .and_then(|a| a.checked_add(u64::try_from(arrays[1].nbytes()).ok()?))
                .ok_or_else(|| self.error(CacheSourceError::Overflow))?
        {
            return Err(self.error(CacheSourceError::Geometry));
        }
        self.prepared_tail_bytes = Some(bytes);
        Ok(())
    }
}

struct ScanCheckout {
    work: OrdinaryPagedWork,
    row: Option<Row>,
    index: usize,
    failed: bool,
}
impl Drop for ScanCheckout {
    fn drop(&mut self) {
        let Some(row) = self.row.take() else { return };
        match self.work.program.inner.state.try_borrow_mut() {
            Ok(mut state) => {
                state.failed |= self.failed;
                state.rows[self.index] = Some(row);
            }
            Err(_) => std::mem::forget((row, self.work.clone())),
        }
    }
}
impl OrdinaryPagedWork {
    pub(crate) fn with_scan<R>(
        &self,
        actual: super::PagedScanInput<'_>,
        host: &HostPreparationAuthority,
        run: impl FnOnce(
            &ProjectedPagedSource,
            &InstalledManagerCatalog,
            &mut super::ordinary_scan::PreparedOrdinaryPagedScan,
        ) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        self.with_host_scan(actual, host, |source, installed, scan, _host| {
            run(source, installed, scan)
        })
    }
    pub(crate) fn with_host_scan<R>(
        &self,
        actual: super::PagedScanInput<'_>,
        host: &HostPreparationAuthority,
        run: impl FnOnce(
            &ProjectedPagedSource,
            &InstalledManagerCatalog,
            &mut super::ordinary_scan::PreparedOrdinaryPagedScan,
            OrdinaryPagedHostScan<'_>,
        ) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        let owner = crate::backend::nn::shared::current_ordinary_execution_owner()?
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?;
        if owner.paged().is_none_or(|work| {
            !Rc::ptr_eq(&work.program.inner, &self.program.inner) || work.ordinals != self.ordinals
        }) {
            return Err(failure(CacheSourceError::Identity, owner.host()));
        }
        let host = owner.host();
        self.activate(host)?;
        let bank = &self.program.inner;
        let source = bank
            .sources
            .sources()
            .iter()
            .position(|source| {
                source.manager().same_catalog(actual.manager)
                    && source.geometry().global_layer == actual.global_layer
            })
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?;
        let retained = &bank.sources.sources()[source];
        let g = retained.append_geometry();
        if actual.rank != retained.geometry().rank
            || actual.block_size != g.block_size
            || actual.window != g.window
            || actual.prefix != g.prefix_tokens
            || actual.queries[0] != g.dimensions[0]
            || actual.queries[3] != g.dimensions[2]
            || CacheBlockMetadata::floating_dtype_bytes(actual.dtype).is_none()
        {
            return Err(failure(CacheSourceError::Geometry, host));
        }
        let mut state = bank
            .state
            .try_borrow_mut()
            .map_err(|_| failure(CacheSourceError::Busy, host))?;
        if state.failed {
            return Err(failure(CacheSourceError::Identity, host));
        }
        let index = state
            .rows
            .iter()
            .position(|row| {
                row.as_ref().is_some_and(|row| {
                    row.source == source
                        && self.ordinals.contains(&row.ordinal)
                        && row.completed
                        && !row.scan_used
                })
            })
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?;
        let row = state.rows[index].as_ref().unwrap();
        let (tail_start, _, end) = row.append.resulting_tail();
        if actual.offset != end
            || actual.tail_start != tail_start
            || end.checked_sub(row.append.initial_frontier().2)
                != Some(i64::from(actual.queries[2]))
            || actual.queries[1] <= 0
            || g.dimensions[1] <= 0
            || actual.queries[1] % g.dimensions[1] != 0
        {
            return Err(failure(CacheSourceError::Identity, host));
        }
        let installed = state
            .installed
            .iter()
            .find(|entry| entry.manager().same_catalog(actual.manager))
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?
            .clone();
        let mut row = state.rows[index].take().unwrap();
        row.scan_used = true;
        drop(state);
        let mut checkout = ScanCheckout {
            work: self.clone(),
            row: Some(row),
            index,
            failed: true,
        };
        let ordinal = checkout.row.as_ref().unwrap().ordinal;
        let scan = checkout
            .row
            .as_mut()
            .unwrap()
            .scan
            .as_mut()
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?;
        let loan = OrdinaryPagedHostScan {
            work: self,
            source: retained,
            installed: &installed,
            ordinal,
            host,
        };
        let result = run(retained, &installed, scan, loan);
        checkout.failed = result.is_err();
        result.map_err(|cause| failure(cause, host))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::runtime::cache::state::MlxKeyValueState, memory_fixture::LedgerFixture};
    use eredu_core::{
        AttentionPolicy, InferenceGeometry, LayerSchedule, OutputDemand, cache::LayerCachePolicy,
    };
    use eredu_runtime::{
        PagedCacheOptions, StateLayout,
        working_memory::{
            InferenceExecutionIdentity, MemoryLedger, quote_inference_workspace_with_context,
        },
    };
    use std::num::NonZeroU32;

    #[test]
    #[ignore = "requires native CPU source construction"]
    fn ordinary_paged_catalog_refuses_changed_source_and_retains_failure_payer() {
        if !crate::tests::support::native_process::enter("ordinary-paged-catalog-source") {
            return;
        }
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap();
        let layout = StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap();
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let process_ledger = crate::backend::managed_memory::try_ledger().unwrap();
        let execution=crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_device_factory(
            &process_ledger,safemlx::DeviceType::Cpu).unwrap().unwrap();
        manager
            .prepare_transfer_stream(&process_ledger, execution.execution())
            .unwrap();
        let native =
            MlxKeyValueState::paged_with_global_layer_start(layout, manager.clone(), None, 7)
                .unwrap();
        let capacity = 1 << 25;
        let ledger = crate::memory_fixture::ledger(capacity, 0).unwrap();
        let funding = ledger
            .prepare_workspace_metadata(
                &InferenceExecutionIdentity::default(),
                crate::memory_fixture::resolved_limits(capacity),
            )
            .unwrap();
        let mechanism =
            crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context =
            WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone()).unwrap();
        let mut projected = native
            .project_complete_workspace_with_storage(NonZeroU32::new(1).unwrap(), &context)
            .unwrap();
        let sources = projected
            .storage
            .take_paged_sources(&context)
            .unwrap()
            .unwrap();
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 2,
            max_output_tokens: 1,
            prefill_chunk_positions: 2,
            output: OutputDemand::LastPosition,
        };
        // Only the exact traversal schedule is needed for this source test; the
        // empty report is never submitted as an inference allocation grant.
        let report = quote_inference_workspace_with_context(geometry, &context, |_| {
            context.begin_span();
            context.finish_report(&[])
        })
        .unwrap();
        let program = sources
            .prepare_ordinary(report.span_workspace_plan(), &ledger, &context)
            .unwrap();
        let work = OrdinaryPagedWork {
            program: program.clone(),
            ordinals: 0..1,
        };
        let blocks = [
            safemlx::Array::from_slice(&[2.0f32, 3.0], &[1, 1, 2, 1]),
            safemlx::Array::from_slice(&[-2.0f32, -3.0], &[1, 1, 2, 1]),
        ];
        let [keys, values] = blocks;
        let id = manager
            .seal_block(
                7,
                0,
                2,
                None,
                CacheBlockArrays::KeyValue { keys, values },
                false,
            )
            .unwrap();
        let error = work
            .activate(&HostPreparationAuthority::unmanaged())
            .unwrap_err();
        assert!(error.to_string().contains("identity"));
        assert_eq!(
            manager
                .layer_block_ids(7, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
                .unwrap(),
            [id.clone()]
        );
        drop((
            work, program, sources, projected, native, report, context, funding,
        ));
        assert!(
            ledger.fixture_host_charge().unwrap() > 0,
            "escaped source failure must retain its actual planning payer"
        );
        drop(error);
        assert_eq!(ledger.fixture_host_charge().unwrap(), 0);
        manager.remove_block(&id).unwrap();
    }
}

impl OrdinaryPagedWork {
    pub(crate) fn checkpoint(
        &self,
        cache: &crate::backend::runtime::cache::kv::PagedKeyValueCache,
        host: &HostPreparationAuthority,
    ) -> Result<crate::backend::runtime::cache::kv::OrdinaryPagedCheckpoint, Exception> {
        let owner = crate::backend::nn::shared::current_ordinary_execution_owner()?
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?;
        if owner.paged().is_none_or(|work| {
            !Rc::ptr_eq(&work.program.inner, &self.program.inner) || work.ordinals != self.ordinals
        }) {
            return Err(failure(CacheSourceError::Identity, owner.host()));
        }
        let host = owner.host();
        self.activate(host)?;
        let bank = &self.program.inner;
        let source = bank
            .sources
            .sources()
            .iter()
            .position(|source| cache.same_checkpoint_source(source))
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?;
        let state = bank
            .state
            .try_borrow()
            .map_err(|_| failure(CacheSourceError::Busy, host))?;
        if state.failed {
            return Err(failure(CacheSourceError::Identity, host));
        }
        let row = state
            .rows
            .iter()
            .flatten()
            .find(|row| row.source == source && self.ordinals.contains(&row.ordinal) && !row.used)
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?;
        let prepared = row.checkpoint.clone();
        let generation = state
            .installed
            .iter()
            .find(|installed| {
                installed
                    .manager()
                    .same_catalog(bank.sources.sources()[source].manager())
            })
            .ok_or_else(|| failure(CacheSourceError::Identity, host))?
            .initial_generation();
        drop(state);
        prepared.capture(cache, generation, host)
    }
}
