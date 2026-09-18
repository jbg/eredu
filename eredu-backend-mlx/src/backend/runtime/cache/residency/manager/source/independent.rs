//! Paid empty canonical destination for actual paged state copying.
use super::*;
use eredu_runtime::cache::CachePoolReservation;
#[path = "independent/tails.rs"]
mod tails;
use tails::CopyTail;
pub(crate) use tails::CopyTailClaim;

/// An empty independent namespace with source-sized canonical destinations.
/// No copied block or runnable state exists until the registered copy consumer
/// validates and publishes its exact source arrays into this owner.
pub(crate) struct PreparedIndependentCacheManager {
    destination: CacheResidencyManager,
    installed: InstalledManagerCatalog,
    source: CacheResidencyManager,
    source_generation: u64,
    copy_reservation: Option<CachePoolReservation>,
    copy_tails: Vec<CopyTail>,
    copy_blocks: usize,
    _funding: Option<HostMetadataFunding>,
}
impl PreparedIndependentCacheManager {
    pub(crate) fn destination(&self) -> &CacheResidencyManager {
        &self.destination
    }
    pub(crate) fn installed(&self) -> &InstalledManagerCatalog {
        &self.installed
    }
    pub(crate) fn matches_source(&self, manager: &CacheResidencyManager, generation: u64) -> bool {
        self.source.same_catalog(manager) && self.source_generation == generation
    }

    /// Reserves actual source occupancy before any numerical array copy. The
    /// complete canonical inventory, including all layer tails, is borrowed
    /// under the same source guard; caller-provided byte totals are not accepted.
    pub(crate) fn reserve_copy_occupancy(
        &mut self,
        source: &CacheBlockSourceLoan<'_>,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(copy_controls(source).ok_or_else(|| fail(CacheSourceError::Overflow))?)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        if self.copy_reservation.is_some()
            || !self.matches_source(source.manager(), source.generation())
        {
            return Err(fail(CacheSourceError::Identity));
        }
        {
            let state = self.destination.inner.state.try_lock().map_err(|cause| {
                fail(match cause {
                    TryLockError::WouldBlock => CacheSourceError::Busy,
                    TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
                })
            })?;
            if state.generation != self.installed.initial_generation()
                || !state.blocks.is_empty()
                || state.lifecycle.catalog_population() != (0, 0)
                || !self.destination.borrowed_storage_complete(&state)
            {
                return Err(fail(CacheSourceError::Identity));
            }
        }
        let mut tails = context
            .metadata_vec(source.catalog_population().1)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let mut usage = CachePoolUsage::default();
        for block in source.all_blocks() {
            match block.phase() {
                CacheStoragePhase::Device => {
                    usage.device_bytes = usage
                        .device_bytes
                        .checked_add(block.logical_bytes())
                        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                }
                CacheStoragePhase::HostUnbacked | CacheStoragePhase::HostBacked => {
                    let buffers = block
                        .host()
                        .ok_or_else(|| fail(CacheSourceError::Identity))?;
                    for buffer in buffers {
                        let descriptor = buffer
                            .try_fixed_descriptor::<4>()
                            .map_err(|cause| fail(CacheSourceError::HostDescriptor(cause)))?;
                        let capacity = u64::try_from(descriptor.allocation().bytes())
                            .map_err(|_| fail(CacheSourceError::Overflow))?;
                        usage.host_bytes = usage
                            .host_bytes
                            .checked_add(capacity)
                            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                    }
                }
                CacheStoragePhase::DiskReady => {
                    block
                        .retained_disk_types()
                        .map_err(fail)?
                        .ok_or_else(|| fail(CacheSourceError::Identity))?;
                    // The immutable file retains its existing physical Disk
                    // reservation. A shared alias allocates no second file.
                }
                _ => return Err(fail(CacheSourceError::PromotionRequired)),
            }
        }
        for (layer, tail) in source.lifecycle.tails() {
            tails.push(CopyTail::new(layer, tail));
            usage.device_bytes = usage
                .device_bytes
                .checked_add(tail.bytes)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        }
        let reservation = source
            .pool()
            .prepare_reservation(context)
            .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?
            .reserve(usage)
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        self.copy_reservation = Some(reservation);
        self.copy_tails = tails;
        self.copy_blocks = source.catalog_population().0;
        Ok(())
    }

    /// Called after a completed registered copy has entered the canonical
    /// destination. Failed publication retains the partial destination and the
    /// remaining reservation; no storage is refunded by an error alone.
    pub(crate) fn publish_copy_occupancy(
        &mut self,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let controls = self
            .installed
            .publication_control_bytes()
            .checked_add(
                CachePoolReservation::publication_control_bytes()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        context
            .charge_metadata(controls)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let reservation = self
            .copy_reservation
            .as_mut()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let mut state = self.destination.inner.state.try_lock().map_err(|cause| {
            fail(match cause {
                TryLockError::WouldBlock => CacheSourceError::Busy,
                TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
            })
        })?;
        if state.generation != self.installed.initial_generation()
            || !self.destination.borrowed_storage_complete(&state)
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let result = reporting::update_report_totals_prepared_copy(
            &mut state,
            reservation,
            &self.destination.inner.pool_membership,
        );
        drop(state);
        result
            .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))
    }
}
impl CacheBlockSourceLoan<'_> {
    /// Constructs an independent empty destination under the same actual pool.
    /// Source workers are retained aliases, not newly spawned queues/threads.
    /// Stable Device/Host sources share this empty constructor. It reads no
    /// payload and grants no promotion or copy; reserve_copy_occupancy checks
    /// the separately supported numerical source before any copied publication.
    pub(crate) fn prepare_independent_manager(
        &self,
        context: &WorkspaceContext,
    ) -> Result<PreparedIndependentCacheManager, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let metadata = |cause| CacheSourceFailure::metadata(cause, context);
        context
            .charge_metadata(controls().ok_or_else(|| fail(CacheSourceError::Overflow))?)
            .map_err(|cause| metadata(cause.into()))?;
        let source = self.manager();
        self.validate_empty_manager_source().map_err(fail)?;
        let pool = source.pool().clone();
        if source.options().device_budget_bytes() > pool.limits().device_bytes()
            || source.options().host_budget_bytes() > pool.limits().host_bytes()
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let (blocks, tails) = self.catalog_population();
        let prepared = self.prepare_catalog(blocks, tails, context)?;
        let registration = pool
            .prepare_manager_registration(context)
            .map_err(|cause| metadata(context.metadata_source(cause)))?;
        // with_pool uses the ordinary validated configuration worker. Its one
        // actual Arc shell is paid before the call; source limits were checked.
        context
            .charge_metadata(
                WorkspaceContext::metadata_arc_bytes::<CacheResidencyPool>()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| metadata(cause.into()))?;
        let options = source
            .options()
            .clone()
            .with_pool(pool.clone())
            .map_err(|cause| metadata(context.metadata_source(cause)))?;
        let session_id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
        let membership = registration
            .register(session_id)
            .map_err(|cause| metadata(context.metadata_source(cause)))?;
        let membership = context
            .metadata_arc(membership)
            .map_err(|cause| metadata(cause.into()))?;
        let state =
            CacheManagerState::empty(&options, pool, session_id, CacheResidencyTelemetry::new(1));
        let state = context
            .metadata_arc(Mutex::new(state))
            .map_err(|cause| metadata(cause.into()))?;
        let inner = context
            .metadata_arc(CacheResidencyManagerInner {
                options,
                state,
                host_demotion_worker: source.inner.host_demotion_worker.clone(),
                disk_worker: source.inner.disk_worker.clone(),
                pool_membership: membership,
                _metadata_funding: context.metadata_funding(),
            })
            .map_err(|cause| metadata(cause.into()))?;
        let destination = CacheResidencyManager { session_id, inner };
        let prepared = prepared
            .for_empty_manager(destination.clone())
            .map_err(|failure| {
                let (cause, retained) = failure.into_parts();
                drop(retained);
                cause
            })?;
        let installed = prepared.install().map_err(|failure| {
            let (cause, retained) = failure.into_parts();
            drop(retained);
            cause
        })?;
        Ok(PreparedIndependentCacheManager {
            destination,
            installed,
            source: source.clone(),
            source_generation: self.generation(),
            copy_reservation: None,
            copy_tails: Vec::new(),
            copy_blocks: 0,
            _funding: context.metadata_funding(),
        })
    }
}
fn controls() -> Option<usize> {
    let frames = [
        CacheBlockSource::retained_disk_control_bytes(),
        size_of::<PreparedIndependentCacheManager>(),
        size_of::<(
            &CacheBlockSourceLoan<'_>,
            CacheStoragePhase,
            Result<(), CacheSourceError>,
        )>(),
        size_of::<Result<PreparedIndependentCacheManager, CacheSourceFailure>>(),
        size_of::<(&CacheBlockSourceLoan<'_>, &WorkspaceContext)>(),
        size_of::<CacheResidencyManager>(),
        size_of::<CacheResidencyManagerInner>(),
        size_of::<CacheManagerState>(),
        size_of::<CacheResidencyTelemetry>(),
        size_of::<PagedCacheOptions>(),
        size_of::<CacheResidencyPool>(),
        size_of::<Result<PagedCacheOptions, CacheResidencyConfigurationError>>(),
        size_of::<eredu_runtime::cache::PreparedCachePoolRegistration>(),
        size_of::<
            Result<
                eredu_runtime::cache::PreparedCachePoolRegistration,
                eredu_runtime::cache::CachePoolRegistrationPreparationFailure,
            >,
        >(),
        size_of::<Result<CachePoolMembership, eredu_runtime::cache::CachePoolRegistrationFailure>>(
        ),
        size_of::<Result<InstalledManagerCatalog, CatalogInstallFailure>>(),
        size_of::<Result<PreparedManagerCatalog, CatalogInstallFailure>>(),
        size_of::<(usize, usize, u64)>(),
        size_of::<&CacheResidencyManager>(),
        size_of::<Result<(), CacheSourceError>>(),
        size_of::<MutexGuard<'_, CacheManagerState>>(),
        size_of::<
            Result<
                MutexGuard<'_, CacheManagerState>,
                TryLockError<MutexGuard<'_, CacheManagerState>>,
            >,
        >(),
        size_of::<eredu_runtime::cache::CacheRecordTableIter<'_, CacheBlockId, CacheBlockRecord>>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

pub(super) fn copy_controls(source: &CacheBlockSourceLoan<'_>) -> Option<usize> {
    let frames = [
        CacheBlockSource::retained_disk_control_bytes(),
        safemlx::HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
        size_of::<safemlx::HostTransferDescriptor<4>>(),
        size_of::<Result<safemlx::HostTransferDescriptor<4>, safemlx::HostTransferMetadataError>>(),
        size_of::<std::array::IntoIter<&Arc<ImmutableHostTransferBuffer>, 2>>(),
        size_of::<(
            &mut PreparedIndependentCacheManager,
            &CacheBlockSourceLoan<'_>,
            &WorkspaceContext,
        )>(),
        size_of::<CachePoolUsage>(),
        size_of::<Vec<CopyTail>>(),
        size_of::<CopyTail>(),
        size_of::<u64>(),
        size_of::<Option<u64>>(),
        size_of::<CachePoolReservation>(),
        size_of::<CacheSourceFailure>(),
        size_of::<Result<(), CacheSourceFailure>>(),
        size_of::<eredu_runtime::cache::PreparedCachePoolReservation>(),
        size_of::<
            Result<
                eredu_runtime::cache::PreparedCachePoolReservation,
                eredu_runtime::cache::CachePoolReservationPreparationFailure,
            >,
        >(),
        size_of::<
            Result<
                CachePoolReservation,
                eredu_runtime::cache::CachePoolReservationAdmissionFailure,
            >,
        >(),
        size_of::<MutexGuard<'_, CacheManagerState>>(),
        size_of::<
            Result<
                MutexGuard<'_, CacheManagerState>,
                TryLockError<MutexGuard<'_, CacheManagerState>>,
            >,
        >(),
        std::mem::size_of_val(&source.all_blocks()),
        std::mem::size_of_val(&source.lifecycle.tails()),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

/// Exact cold description of the borrowed manager. It supplies no native grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IndependentCacheManagerPlan {
    session: u64,
    generation: u64,
    population: (usize, usize),
    bytes: usize,
}
impl IndependentCacheManagerPlan {
    pub(crate) fn control_bytes(self) -> usize {
        self.bytes
    }
}
impl CacheBlockSourceLoan<'_> {
    // A new namespace shares the actual idle worker and immutable file owners;
    // it does not recreate I/O threads or confer a task/materialization grant.
    fn validate_empty_manager_source(&self) -> Result<(), CacheSourceError> {
        if matches!(
            self.manager().options().live_disk_policy(),
            LiveCacheDiskPolicy::Enabled { .. }
        ) && self.manager().inner.disk_worker.is_none()
        {
            return Err(CacheSourceError::Identity);
        }
        for block in self.all_blocks() {
            if !matches!(
                block.phase(),
                CacheStoragePhase::Device
                    | CacheStoragePhase::HostUnbacked
                    | CacheStoragePhase::HostBacked
                    | CacheStoragePhase::DiskReady
            ) {
                return Err(CacheSourceError::PromotionRequired);
            }
            if block.disk().is_some() {
                block.retained_disk_types()?;
            }
        }
        Ok(())
    }
    pub(super) fn independent_plan(&self) -> Result<IndependentCacheManagerPlan, CacheSourceError> {
        self.validate_empty_manager_source()?;
        let pool =
            self.pool()
                .manager_registration_control_bytes()
                .map_err(|cause| match cause {
                    eredu_runtime::cache::CachePoolError::Busy => CacheSourceError::Busy,
                    eredu_runtime::cache::CachePoolError::Poisoned => CacheSourceError::Poisoned,
                    _ => CacheSourceError::Overflow,
                })?;
        let frames = [
            controls(),
            self.independent_catalog_control_bytes(),
            Some(pool),
            WorkspaceContext::metadata_arc_bytes::<CacheResidencyPool>(),
            WorkspaceContext::metadata_arc_bytes::<CachePoolMembership>(),
            WorkspaceContext::metadata_arc_bytes::<Mutex<CacheManagerState>>(),
            WorkspaceContext::metadata_arc_bytes::<CacheResidencyManagerInner>(),
            CacheResidencyManager::source_loan_control_bytes::<PreparedIndependentCacheManager>(
                size_of::<(&IndependentCacheManagerPlan, &WorkspaceContext)>(),
            ),
            WorkspaceContext::metadata_source_bytes::<
                eredu_runtime::cache::CachePoolRegistrationPreparationFailure,
            >(),
            WorkspaceContext::metadata_source_bytes::<
                eredu_runtime::cache::CachePoolRegistrationFailure,
            >(),
            WorkspaceContext::metadata_source_bytes::<CacheResidencyConfigurationError>(),
            WorkspaceContext::metadata_source_bytes::<CacheSourceFailure>(),
            Some(size_of::<Self>() + size_of::<IndependentCacheManagerPlan>()),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), |sum, n| sum.checked_add(n?))
            .ok_or(CacheSourceError::Overflow)?;
        Ok(IndependentCacheManagerPlan {
            session: self.manager().session_id(),
            generation: self.generation(),
            population: self.catalog_population(),
            bytes,
        })
    }
}
impl CacheResidencyManager {
    /// Inspects the same real source without allocation, demand mutation or native work.
    pub(crate) fn inspect_independent_manager(
        &self,
    ) -> Result<IndependentCacheManagerPlan, CacheSourceError> {
        self.with_source_loan_inner(
            CacheBlockSelection::new(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0),
            None,
            |cause| cause,
            |source| source.independent_plan(),
        )
    }
    pub(crate) fn prepare_empty_manager(
        &self,
        plan: &IndependentCacheManagerPlan,
        context: &WorkspaceContext,
    ) -> Result<CacheResidencyManager, CacheSourceFailure> {
        let prepared = self.with_source_loan(
            CacheBlockSelection::new(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0),
            context,
            |source| {
                let current = source
                    .independent_plan()
                    .map_err(|cause| CacheSourceFailure::source(cause, context))?;
                if &current != plan {
                    return Err(CacheSourceFailure::source(
                        CacheSourceError::Identity,
                        context,
                    ));
                }
                source.prepare_independent_manager(context)
            },
        )?;
        // Source and destination catalog loans are closed before aliases retire.
        Ok(prepared.destination().clone())
    }
}

impl CacheBlockSourceLoan<'_> {
    pub(super) fn copy_occupancy_control_bytes(&self) -> Result<usize, CacheSourceError> {
        let reservation = self
            .pool()
            .reservation_preparation_control_bytes()
            .map_err(|cause| match cause {
                eredu_runtime::cache::CachePoolError::Busy => CacheSourceError::Busy,
                eredu_runtime::cache::CachePoolError::Poisoned => CacheSourceError::Poisoned,
                _ => CacheSourceError::Overflow,
            })?;
        copy_controls(self)
            .and_then(|n| {
                n.checked_add(WorkspaceContext::metadata_vec_bytes::<CopyTail>(
                    self.catalog_population().1,
                )?)?
                .checked_add(reservation)
            })
            .ok_or(CacheSourceError::Overflow)
    }
    pub(super) fn copy_publication_control_bytes(&self) -> Option<usize> {
        self.publication_controls
            .checked_add(CachePoolReservation::publication_control_bytes()?)
    }
}
impl PreparedIndependentCacheManager {
    pub(crate) fn copy_tail_control_bytes() -> Option<usize> {
        tails::tail_controls()
    }
}

impl PreparedIndependentCacheManager {
    pub(crate) fn copy_tail_loan_control_bytes() -> Option<usize> {
        crate::backend::runtime::cache::kv::PagedKeyValueCache::workspace_source_control_for::<
            CopyTailClaim,
            (&mut Self, &WorkspaceContext),
        >()
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
