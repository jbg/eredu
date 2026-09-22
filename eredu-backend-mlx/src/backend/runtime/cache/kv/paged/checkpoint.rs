//! A checkpoint loan from the actual admitted pager and its saved tail source.
use super::*;
use crate::backend::runtime::cache::residency::{
    CacheHistoryOwner, CacheSourceError, CacheSourceFailure, PreparedCacheHistory,
};
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataAllocation};
use eredu_runtime::{
    MutableCacheTail,
    cache::{PagedAppendPlan, PagedTruncateMechanisms, PagedTruncatePlan},
};
use std::{
    mem::{size_of, size_of_val},
    sync::{Arc, Mutex, MutexGuard, TryLockError},
};

#[derive(Clone)]
pub(crate) struct PreparedPagedCheckpoint {
    inner: Arc<Mutex<Destination>>,
    _funding: Option<HostMetadataFunding>,
}
struct Destination {
    manager: CacheResidencyManager,
    layer: usize,
    rank: Option<CacheRankIdentity>,
    offset: i64,
    tail_start: i64,
    window: Option<i32>,
    prefix: i32,
    key_only: bool,
    ids: Vec<CacheBlockId>,
    tail: Option<MutableCacheTail>,
    captured: bool,
    history: Option<PreparedCacheHistory>,
}
/// The saved pager carries this loan. Its native tail aliases retire before
/// these actual source, metadata and execution owners.
#[derive(Clone)]
pub(crate) struct OrdinaryPagedCheckpoint {
    prepared: PreparedPagedCheckpoint,
    history: Option<CacheHistoryOwner>,
    host: HostPreparationAuthority,
}
impl std::fmt::Debug for OrdinaryPagedCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinaryPagedCheckpoint")
            .finish_non_exhaustive()
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Source(#[from] CacheSourceError),
    #[error(transparent)]
    Preparation(#[from] CacheSourceFailure),
    #[error(transparent)]
    Native(#[from] Exception),
    #[error(transparent)]
    Plan(#[from] eredu_runtime::cache::PagedTruncateError),
}
#[derive(Debug, thiserror::Error)]
#[error("ordinary paged checkpoint: {cause}")]
struct Failure {
    #[source]
    cause: Cause,
    _host: HostPreparationAuthority,
}
fn fail(cause: impl Into<Cause>, host: &HostPreparationAuthority) -> Exception {
    Exception::from_retained_source(Failure {
        cause: cause.into(),
        _host: host.clone(),
    })
}
impl PreparedPagedCheckpoint {
    pub(crate) fn prepare(
        source: &ProjectedPagedSource,
        append: PagedAppendPlan,
        blocks: usize,
        context: &WorkspaceContext,
    ) -> Result<Self, CacheSourceFailure> {
        Self::prepare_geometry(source.manager(), source.geometry(), append, blocks, context)
    }
    fn prepare_geometry(
        manager: &CacheResidencyManager,
        geometry: &PagedCacheSourceGeometry,
        append: PagedAppendPlan,
        blocks: usize,
        context: &WorkspaceContext,
    ) -> Result<Self, CacheSourceFailure> {
        let (tail_start, _, offset) = append.initial_frontier();
        let source_fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(
                Self::control_bytes().ok_or_else(|| source_fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let ids = context
            .metadata_vec(blocks)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let history = match geometry.sliding_window {
            Some(window) if (offset - i64::from(window)).max(0) < tail_start => {
                Some(PreparedCacheHistory::prepare(
                    manager,
                    geometry.global_layer,
                    (offset - i64::from(window)).max(0),
                    tail_start,
                    context,
                )?)
            }
            _ => None,
        };
        let mutex_bytes =
            eredu_runtime::working_memory::OriginalHostMetadataCustody::initialized_mutex_bytes()
                .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        context
            .charge_metadata(
                usize::try_from(mutex_bytes)
                    .map_err(|_| source_fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let destination = Mutex::new(Destination {
            manager: manager.clone(),
            layer: geometry.global_layer,
            rank: geometry.rank,
            offset,
            tail_start,
            window: geometry.sliding_window,
            prefix: geometry.prefix_tokens,
            key_only: geometry.key_only,
            ids,
            tail: None,
            captured: false,
            history,
        });
        // Select the qualified PAL allocation once while the mutex is private.
        drop(
            destination
                .lock()
                .expect("private checkpoint destination mutex"),
        );
        let inner = context
            .metadata_arc(destination)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        Ok(Self {
            inner,
            _funding: context.metadata_funding(),
        })
    }
    pub(crate) fn capture(
        &self,
        cache: &PagedKeyValueCache,
        generation: u64,
        host: &HostPreparationAuthority,
    ) -> Result<OrdinaryPagedCheckpoint, Exception> {
        let mut destination = self
            .inner
            .try_lock()
            .map_err(|cause| fail(lock_cause(cause), host))?;
        if destination.captured || !destination.matches(cache) {
            return Err(fail(CacheSourceError::Identity, host));
        }
        destination.captured = true;
        cache.inspect_copy_source(
            |cause| fail(cause, host),
            capture_callback(Some(&mut destination), generation, Some(host)),
        )?;
        let history = destination
            .history
            .take()
            .map(|history| history.install(generation))
            .transpose()
            .map_err(|cause| fail(cause, host))?;
        destination.captured = true;
        Ok(OrdinaryPagedCheckpoint {
            prepared: self.clone(),
            history,
            host: host.clone(),
        })
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let callback = capture_callback(None, 0, None);
        let frames = [
            size_of::<Self>(),
            size_of::<Destination>(),
            size_of::<OrdinaryPagedCheckpoint>(),
            size_of::<Failure>(),
            Exception::retained_source_control_bytes::<Failure>()?,
            size_of::<MutexGuard<'_, Destination>>(),
            size_of::<Result<MutexGuard<'_, Destination>, TryLockError<MutexGuard<'_, Destination>>>>(
            ),
            size_of::<Result<OrdinaryPagedCheckpoint, Exception>>(),
            size_of::<Result<(), Exception>>(),
            size_of::<(&Self, &PagedKeyValueCache, u64, &HostPreparationAuthority)>(),
            PreparedCacheHistory::control_bytes()?,
            CacheResidencyManager::ordinary_transaction_rollback_control_bytes()?,
            PagedTruncatePlan::control_bytes::<Restore<'_>>()?,
            // Checkpoint creation and restore each retain two immutable handles.
            safemlx::Array::ordinary_clone_control_bytes()?.checked_mul(4)?,
            PagedKeyValueCache::workspace_source_control_bytes::<(), _>(&callback)?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
fn capture_callback<'a>(
    destination: Option<&'a mut Destination>,
    generation: u64,
    host: Option<&'a HostPreparationAuthority>,
) -> impl for<'loan> FnOnce(PagedKeyValueSource<'loan>) -> Result<(), Exception> + 'a {
    move |source| {
        let host = host.expect("actual source callback carries host custody");
        let destination = destination.expect("actual source callback carries its paid destination");
        let loan = source.manager_source();
        if loan.generation() != generation {
            return Err(fail(CacheSourceError::Identity, host));
        }
        if loan.blocks().count() > destination.ids.capacity() {
            return Err(fail(CacheSourceError::Geometry, host));
        }
        for block in loan.blocks() {
            destination.ids.push(block.id().clone());
        }
        destination.tail = loan.tail();
        Ok(())
    }
}
impl Destination {
    fn matches(&self, cache: &PagedKeyValueCache) -> bool {
        self.manager.same_catalog(&cache.manager)
            && self.layer == cache.global_layer
            && self.rank == cache.rank
            && self.offset == cache.offset
            && self.tail_start == cache.tail_start
            && self.window == cache.sliding_window
            && self.prefix == cache.prefix_tokens
            && self.key_only == cache.key_only
    }
}
impl OrdinaryPagedCheckpoint {
    pub(super) fn restore(
        &self,
        current: &mut PagedKeyValueCache,
        saved: &PagedKeyValueCache,
    ) -> Result<(), Exception> {
        let destination = self
            .prepared
            .inner
            .try_lock()
            .map_err(|cause| fail(lock_cause(cause), &self.host))?;
        if !destination.captured
            || !destination.matches(saved)
            || !current.manager.same_catalog(&saved.manager)
            || current.global_layer != saved.global_layer
            || current.rank != saved.rank
            || current.key_only != saved.key_only
            || current.sliding_window != saved.sliding_window
            || current.prefix_tokens != saved.prefix_tokens
        {
            return Err(fail(CacheSourceError::Identity, &self.host));
        }
        let plan = PagedTruncatePlan::checkpoint(
            current.offset,
            current.tail_start,
            saved.offset,
            saved.tail_start,
        )
        .map_err(|cause| fail(cause, &self.host))?;
        let mut worker = Restore {
            current,
            saved,
            destination: &destination,
            host: &self.host,
        };
        plan.run(&mut worker)
    }
}
struct Restore<'a> {
    current: &'a mut PagedKeyValueCache,
    saved: &'a PagedKeyValueCache,
    destination: &'a Destination,
    host: &'a HostPreparationAuthority,
}
impl PagedTruncateMechanisms for Restore<'_> {
    type Source = ();
    type Pair = ();
    type Error = Exception;
    fn slice_tail(&mut self, _: i32) -> Result<(), Exception> {
        Err(fail(CacheSourceError::Identity, self.host))
    }
    fn publish_tail(&mut self, _: i64, _: i32, _: ()) -> Result<(), Exception> {
        Err(fail(CacheSourceError::Identity, self.host))
    }
    fn acquire_block(&mut self, _: std::ops::Range<i64>) -> Result<(), Exception> {
        Err(fail(CacheSourceError::Identity, self.host))
    }
    fn slice_block(&mut self, _: &(), _: i32) -> Result<(), Exception> {
        Err(fail(CacheSourceError::Identity, self.host))
    }
    fn complete_slices(&mut self, _: &()) -> Result<(), Exception> {
        Err(fail(CacheSourceError::Identity, self.host))
    }
    fn copy_slices(&mut self, _: ()) -> Result<(), Exception> {
        Err(fail(CacheSourceError::Identity, self.host))
    }
    fn publish_catalog(&mut self, _: i64, _: Option<((), ())>) -> Result<(), Exception> {
        Err(fail(CacheSourceError::Identity, self.host))
    }
    fn restore_checkpoint(&mut self, tail_start: i64, offset: i64) -> Result<(), Exception> {
        if tail_start != self.destination.tail_start || offset != self.destination.offset {
            return Err(fail(CacheSourceError::Identity, self.host));
        }
        let visible = self
            .destination
            .window
            .map_or(0, |window| (offset - i64::from(window)).max(0));
        self.current.manager.rollback_ordinary_checkpoint(
            self.current.manager.session_id(),
            self.current.global_layer,
            self.destination.layer,
            self.destination.tail,
            tail_start,
            offset,
            self.current.offset,
            &self.destination.ids,
            visible,
            i64::from(self.destination.prefix),
            self.host,
        )?;
        self.current.clone_from(self.saved);
        self.current.retained_history = None;
        self.current.ordinary_checkpoint = None;
        Ok(())
    }
}

pub(super) fn unqualified_truncate(host: &HostPreparationAuthority) -> Exception {
    fail(CacheSourceError::Identity, host)
}

impl PagedKeyValueCache {
    pub(crate) fn same_checkpoint_source(&self, source: &ProjectedPagedSource) -> bool {
        self.manager.same_catalog(source.manager())
            && self.global_layer == source.geometry().global_layer
    }
}

fn lock_cause<T>(cause: TryLockError<T>) -> CacheSourceError {
    match cause {
        TryLockError::WouldBlock => CacheSourceError::Busy,
        TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_fixture::LedgerFixture;
    use eredu_runtime::working_memory::{InferenceExecutionIdentity, MemoryLedger};

    #[test]
    #[ignore = "requires native CPU cache construction"]
    fn ordinary_checkpoint_restores_saved_nonzero_tail_and_retains_refusal_payer() {
        if !crate::tests::support::native_process::enter("ordinary-paged-checkpoint-source") {
            return;
        }
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let process = crate::backend::managed_memory::try_ledger().unwrap();
        let execution=crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_device_factory(
            &process,safemlx::DeviceType::Cpu).unwrap().unwrap();
        let stream = execution.execution();
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        // A sliding checkpoint also retains exactly the previous visible
        // history while its immutable tail becomes a new sealed block.
        let mut cache = PagedKeyValueCache::new(manager.clone(), 7, Some(3)).unwrap();
        cache
            .append_normalized(
                Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3, 1]),
                Array::from_slice(&[-1.0f32, -2.0, -3.0], &[1, 1, 3, 1]),
                true,
                stream,
            )
            .unwrap();
        safemlx::transforms::eval(cache.retained_arrays()).unwrap();
        let capacity = 1 << 25;
        let ledger = crate::memory_fixture::ledger(capacity, 0).unwrap();
        let funding = ledger
            .prepare_workspace_metadata(
                &InferenceExecutionIdentity::default(),
                crate::memory_fixture::resolved_limits(capacity),
            )
            .unwrap();
        let context = WorkspaceContext::new_with_metadata_funding(
            crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms::current_host().unwrap(),
            funding.clone(),
        )
        .unwrap();
        let projected = cache.project_device_workspace(&context).unwrap();
        let append = PagedAppendPlan::new(2, 3, 1, 2, 3).unwrap();
        let prepared = PreparedPagedCheckpoint::prepare_geometry(
            projected.manager(),
            projected.geometry(),
            append,
            projected.geometry().blocks.len(),
            &context,
        )
        .unwrap();
        let catalog = manager
            .with_source_loan(
                eredu_runtime::CacheBlockSelection::new(
                    7,
                    CacheRepresentation::KeyValue,
                    0,
                    i64::MAX,
                    0,
                ),
                &context,
                |loan| loan.prepare_catalog(3, 1, &context),
            )
            .unwrap()
            .install()
            .unwrap();
        funding
            .reserve_metadata(
                HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().unwrap(),
            )
            .unwrap();
        let host = HostPreparationAuthority::retain(funding.clone());
        let receipt = prepared
            .capture(&cache, catalog.initial_generation(), &host)
            .unwrap();
        let mut saved = cache.clone();
        saved.ordinary_checkpoint = Some(receipt);
        // Source projection pins are gone. Only the actual checkpoint history
        // entry prevents the original visible block from being discarded.
        drop(projected);
        cache
            .append_normalized(
                Array::from_slice(&[4.0f32, 5.0, 6.0], &[1, 1, 3, 1]),
                Array::from_slice(&[-4.0f32, -5.0, -6.0], &[1, 1, 3, 1]),
                true,
                stream,
            )
            .unwrap();
        manager
            .discard_before(7, CacheRepresentation::KeyValue, 4, 0)
            .unwrap();
        assert!(
            manager
                .layer_block_ids(7, CacheRepresentation::KeyValue, 0, 6, 0)
                .unwrap()
                .iter()
                .any(|id| id.start == 0)
        );
        cache.restore_checkpoint(&saved, stream).unwrap();
        assert_eq!((cache.tail_start, cache.offset), (2, 3));
        assert_eq!(
            cache
                .tail_keys
                .as_ref()
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[3.0]
        );
        assert_eq!(
            cache
                .tail_values
                .as_ref()
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[-3.0]
        );
        let ids = manager
            .layer_block_ids(7, CacheRepresentation::KeyValue, 0, 6, 0)
            .unwrap();
        assert_eq!(
            ids.iter().map(|id| (id.start, id.end)).collect::<Vec<_>>(),
            [(0, 2)]
        );
        let foreign_manager = CacheResidencyManager::new(
            PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let mut foreign = PagedKeyValueCache::new(foreign_manager, 7, Some(3)).unwrap();
        let error = foreign.restore_checkpoint(&saved, stream).unwrap_err();
        assert_eq!(foreign.offset, 0);
        assert!(std::error::Error::source(&error).is_some());
        drop((
            foreign, cache, saved, prepared, catalog, manager, context, host, funding,
        ));
        assert!(
            ledger.fixture_host_charge().unwrap() > 0,
            "escaped mismatch retains its actual prepaid payer"
        );
        drop(error);
        assert_eq!(ledger.fixture_host_charge().unwrap(), 0);
    }
}
