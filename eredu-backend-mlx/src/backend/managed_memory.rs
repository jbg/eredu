//! Process-local accounting for Eredu-managed MLX host and device storage.
//!
//! The aggregate domain includes every local MLX device and managed host tier.
//! Its unlimited configured ceiling supplies accounting, not a hardware-memory
//! estimate; individual admitted requests can impose a shared-domain ceiling.

use std::sync::{Arc, OnceLock};

pub(crate) mod bf16_projection_kernel;
pub(crate) mod input_allocator;
pub(crate) mod kernel_family;
pub(crate) mod metal_device;
pub(crate) mod pointwise_kernel;
pub(crate) mod recurrent_kernel;
pub(crate) mod router;
pub(crate) mod row_kernels;
pub(crate) mod scheduler;

use eredu_runtime::working_memory::{WorkingMemoryPool, WorkingMemoryUnquotedLease};

use super::{error::Error, submission_recovery};

// This static owns its fixed accounting allocation for process lifetime. It
// contains no native context, runtime Rc or request-specific storage authority.
struct StaticDomain {
    pool: WorkingMemoryPool,
    // A missing fixed-owner qualification must not become a complete zero.
    // Existing unquoted exclusion preserves ordinary behavior and typed strict
    // refusal. This global lease is never released while its domain survives.
    _unqualified: Option<WorkingMemoryUnquotedLease>,
}

impl StaticDomain {
    fn new(
        facts: safemlx::RuntimeStaticBaseline,
    ) -> Result<Self, eredu_runtime::working_memory::WorkingMemoryError> {
        use eredu_runtime::working_memory::WorkingMemoryError;
        let known = facts
            .known_static_storage_bytes()
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<OnceLock<Self>>()))
            .and_then(|bytes| bytes.checked_add(input_allocator::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(metal_device::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(scheduler::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(router::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(pointwise_kernel::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(row_kernels::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(recurrent_kernel::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(bf16_projection_kernel::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(crate::backend::nn::fp8::kernel::static_storage_bytes()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        if facts.fixed_storage_bytes().is_some() {
            match WorkingMemoryPool::new_with_fixed_owner_baseline(u64::MAX, known) {
                Ok(pool) => {
                    return Ok(Self {
                        pool,
                        _unqualified: None,
                    });
                }
                Err(WorkingMemoryError::UnknownBound) => {}
                Err(error) => return Err(error),
            }
        }
        let pool = WorkingMemoryPool::new(u64::MAX, known)?;
        let unqualified = pool.acquire_unquoted()?;
        Ok(Self {
            pool,
            _unqualified: Some(unqualified),
        })
    }
}

/// One aggregate managed domain for all local MLX execution and host payloads.
pub(crate) fn domain() -> WorkingMemoryPool {
    static DOMAIN: OnceLock<StaticDomain> = OnceLock::new();
    DOMAIN
        .get_or_init(|| {
            StaticDomain::new(safemlx::runtime_static_baseline())
                .expect("fixed managed static layout fits the configured ceiling")
        })
        .pool
        .clone()
}

#[derive(Debug)]
struct Owner {
    pool: WorkingMemoryPool,
    // A distinct attachment namespace preserves original unquoted authority
    // even if the same immutable source is later inventoried in the pool.
    host_source_domain: eredu_core::SharedStorageDomain,
    // Allocation authority is immutable. A residency transition can drop its
    // own handle, but cannot revoke exclusion from surviving descendants.
    _unquoted: WorkingMemoryUnquotedLease,
}

/// Shared ownership of unquoted managed work and its surviving payloads.
///
/// Acquire before allocating; retain through native recovery and descendants.
/// This owner holds no native resources and certifies no byte coverage. Its
/// presence excludes quoted request admission until the last clone retires.
/// Merely registering an inventory does not promote or clear this owner.
#[derive(Clone, Debug)]
pub(crate) struct NativeMemoryOwner(Arc<Owner>);

impl NativeMemoryOwner {
    pub(crate) fn acquire(pool: &WorkingMemoryPool) -> Result<Self, Error> {
        Self::acquire_typed(pool).map_err(|error| Error::Other(Box::new(error)))
    }

    /// Same real ordinary lease, with allocation-free policy rejection for a
    /// caller that has not yet acquired any construction/error custody.
    pub(crate) fn acquire_typed(
        pool: &WorkingMemoryPool,
    ) -> Result<Self, eredu_runtime::working_memory::WorkingMemoryError> {
        let unquoted = pool.acquire_unquoted()?;
        Ok(Self(Arc::new(Owner {
            pool: pool.clone(),
            host_source_domain: Default::default(),
            _unquoted: unquoted,
        })))
    }

    pub(crate) fn pool(&self) -> &WorkingMemoryPool {
        &self.0.pool
    }

    pub(crate) fn same_authority(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Retains the neutral exclusion authority with portable state. This never
    /// clears or promotes the original lease, even when native owners retire.
    pub(crate) fn unquoted_lease(&self) -> Result<WorkingMemoryUnquotedLease, Error> {
        Ok(self.0._unquoted.clone())
    }

    /// Retains authority on an already materialized backing without evaluating.
    /// Lazy producers must complete inside their existing recovery scope first.
    pub(crate) fn retain_array(&self, array: &safemlx::Array) -> Result<(), Error> {
        if array
            .allocation_info()?
            .is_some_and(|info| info.bytes() == 0)
        {
            return Ok(());
        }
        array
            .retain_allocation_owner(self.clone())
            .map_err(|failure| {
                let (error, _unattached) = failure.into_parts();
                Error::from(error)
            })
    }

    /// Preserves the original unquoted construction authority on every alias
    /// of immutable host metadata. This is exclusion, not a finite byte grant;
    /// later registration never promotes or revokes that original authority.
    pub(crate) fn retain_metadata(
        &self,
        metadata: &eredu_runtime::SharedHostMetadata,
    ) -> Result<(), Error> {
        if let eredu_runtime::SharedHostMetadata::Input(input) = metadata {
            if let Some(result) = input.original_residence(self.pool()) {
                // This exact closed B source already owns its payload/control.
                // Preserve the source's authority; never attach an ordinary
                // owner or manufacture a second finite source charge.
                return result.map_err(|error| Error::Other(Box::new(error)));
            }
        }
        metadata
            .try_attach(&self.0.host_source_domain, || {
                Ok::<Box<dyn Send + Sync>, std::convert::Infallible>(Box::new(self.clone()))
            })
            .map(|_| ())
            .map_err(|error| Error::Other(Box::new(error)))
    }
}

impl submission_recovery::Retention for NativeMemoryOwner {
    fn observe(&self, _: submission_recovery::Status) {}
}

/// Distinct allocation authorities retained by restored or submitted state.
/// Clones share leases; merging the same authority never creates a new lease.
#[derive(Clone, Debug, Default)]
pub(crate) struct NativeMemoryRetention(Vec<NativeMemoryOwner>);

impl NativeMemoryRetention {
    /// Borrowed exact authorities for a separate finite exchange constructor.
    /// This grants neither storage nor another ordinary/native account.
    pub(crate) fn owners(&self) -> &[NativeMemoryOwner] {
        &self.0
    }

    /// Accepts the caller's already constructed finite, deduplicated rows.
    pub(crate) fn from_prepared_owners(owners: Vec<NativeMemoryOwner>) -> Self {
        Self(owners)
    }

    pub(crate) fn from_owner(owner: &NativeMemoryOwner) -> Self {
        Self(vec![owner.clone()])
    }

    pub(crate) fn singleton_metadata_bytes() -> u64 {
        std::mem::size_of::<NativeMemoryOwner>() as u64
    }

    /// Builds a copied authority list with room for one independent operation.
    /// Reserve its final logical size up front instead of growing a cloned Vec.
    pub(crate) fn with_added_owner(&self, owner: &NativeMemoryOwner) -> Self {
        let mut copied = Self(Vec::with_capacity(self.0.len() + 1));
        copied.0.extend(self.0.iter().cloned());
        copied.retain(owner);
        copied
    }

    pub(crate) fn metadata_bytes_with_added_owner(&self) -> Option<u64> {
        u64::try_from(self.0.len())
            .ok()?
            .checked_add(1)?
            .checked_mul(Self::singleton_metadata_bytes())
    }

    pub(crate) fn retain(&mut self, owner: &NativeMemoryOwner) {
        if !self
            .0
            .iter()
            .any(|existing| Arc::ptr_eq(&existing.0, &owner.0))
        {
            self.0.push(owner.clone());
        }
    }

    pub(crate) fn extend_from(&mut self, other: &Self) {
        for owner in &other.0 {
            self.retain(owner);
        }
    }

    pub(crate) fn covers_pool(&self, pool: &WorkingMemoryPool) -> bool {
        self.0.iter().any(|owner| owner.pool().same_domain(pool))
    }

    /// Whether these local handles all share the given immutable authority.
    /// Empty retention satisfies this check; it cannot conceal another owner.
    pub(crate) fn retains_only(&self, owner: &NativeMemoryOwner) -> bool {
        self.0.iter().all(|retained| retained.same_authority(owner))
    }

    pub(crate) fn logical_metadata_bytes(&self) -> Option<u64> {
        u64::try_from(self.0.capacity())
            .ok()?
            .checked_mul(u64::try_from(std::mem::size_of::<NativeMemoryOwner>()).ok()?)
    }

    pub(crate) fn retain_array(&self, array: &safemlx::Array) -> Result<(), Error> {
        if self.0.is_empty()
            || array
                .allocation_info()?
                .is_some_and(|info| info.bytes() == 0)
        {
            return Ok(());
        }
        array
            .retain_allocation_owner(self.clone())
            .map_err(|failure| {
                let (error, _unattached) = failure.into_parts();
                Error::from(error)
            })
    }
}

impl submission_recovery::Retention for NativeMemoryRetention {
    fn observe(&self, _: submission_recovery::Status) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{
        Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
        InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
        cache::LayerCachePolicy,
    };
    use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryError};
    use std::{cell::Cell, rc::Rc};

    fn zero_admission() -> Admission {
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 1,
            max_output_tokens: 0,
            prefill_chunk_positions: 1,
            output: OutputDemand::StateOnly,
        };
        let layout = StateMemoryLayout::new(
            LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
            vec![0],
            1,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap();
        let zero = || WorkspaceBound::bounded(0, "stateless fixture without managed payload");
        let state = eredu_core::estimate_runtime_state(
            &layout,
            InputTokenCount::text(1),
            0,
            1,
            std::num::NonZeroU8::new(4).unwrap(),
        )
        .unwrap()
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
        })
        .unwrap();
        Admission {
            requested_positions: 1,
            state,
            incremental_required_bytes: 0,
            available_memory_bytes: None,
        }
    }

    fn fixed_baseline_required() -> bool {
        std::env::var_os("EREDU_REQUIRE_STATIC_BASELINE_QUALIFICATION").is_some()
    }

    #[test]
    fn fixed_native_baseline_is_charged_once_at_exact_and_one_short_capacity() {
        let facts = safemlx::runtime_static_baseline();
        let domain = StaticDomain::new(facts).unwrap();
        let bytes = domain.pool.used_bytes().unwrap();
        assert!(bytes > 0);
        if fixed_baseline_required() {
            assert!(domain._unqualified.is_none());
        }
        if domain._unqualified.is_some() {
            assert!(matches!(
                domain
                    .pool
                    .reserve(&InferenceExecutionIdentity::default(), &zero_admission()),
                Err(WorkingMemoryError::UnknownBound)
            ));
            return;
        }
        let same = domain.pool.clone();
        assert!(same.same_domain(&domain.pool));
        let mut admission = zero_admission();
        // A nonzero application request, separate from the fixed baseline.
        admission.incremental_required_bytes = 17;
        admission
            .state
            .execution_workspace
            .as_mut()
            .unwrap()
            .retained = WorkspaceBound::bounded(17, "fixture retained producer storage");
        let reservation = same
            .reserve_with_capacity(
                &InferenceExecutionIdentity::default(),
                &admission,
                bytes + 17,
            )
            .unwrap();
        assert_eq!(domain.pool.used_bytes().unwrap(), bytes + 17);
        drop(reservation);
        assert!(matches!(
            same.reserve_with_capacity(
                &InferenceExecutionIdentity::default(),
                &admission,
                bytes + 16
            ),
            Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: 17,
                available_bytes: 16,
            })
        ));
        drop(same);
        assert_eq!(domain.pool.used_bytes().unwrap(), bytes);
    }

    #[test]
    fn fixed_native_baseline_concurrent_requests_share_one_charge() {
        use std::sync::Barrier;
        let slot = OnceLock::<StaticDomain>::new();
        let starts = Barrier::new(4);
        let all_reserved = Barrier::new(4);
        let all_observed = Barrier::new(4);
        std::thread::scope(|threads| {
            let handles = (0..4)
                .map(|_| {
                    threads.spawn(|| {
                        starts.wait();
                        let domain = slot.get_or_init(|| {
                            StaticDomain::new(safemlx::runtime_static_baseline()).unwrap()
                        });
                        // Existing fixed bytes are independent of each live request.
                        // Every contender uses the same actual constructor and domain.
                        let baseline = domain.pool.used_bytes().unwrap();
                        let mut admission = zero_admission();
                        admission.incremental_required_bytes = 17;
                        admission
                            .state
                            .execution_workspace
                            .as_mut()
                            .unwrap()
                            .retained =
                            WorkspaceBound::bounded(17, "concurrent fixture retained storage");
                        let qualified = domain._unqualified.is_none();
                        // Synchronize BEFORE anyone reserves, so each observed subtotal
                        // is the same fixed baseline, not a sibling's live reservation.
                        starts.wait();
                        let reservation = domain.pool.reserve_with_capacity(
                            &InferenceExecutionIdentity::default(),
                            &admission,
                            baseline + 4 * 17,
                        );
                        all_reserved.wait();
                        let concurrent_bytes = domain.pool.used_bytes();
                        // Keep actual Result owners until all siblings observed
                        // the full charge; a refusal must not hang the barriers.
                        let alias = domain.pool.clone();
                        all_observed.wait();
                        if fixed_baseline_required() {
                            assert!(qualified);
                        }
                        assert_eq!(
                            concurrent_bytes.unwrap(),
                            baseline + if qualified { 4 * 17 } else { 0 }
                        );
                        assert!(alias.same_domain(&domain.pool));
                        if qualified {
                            drop(reservation.unwrap());
                        } else {
                            assert!(matches!(reservation, Err(WorkingMemoryError::UnknownBound)));
                        }
                        (alias, baseline)
                    })
                })
                .collect::<Vec<_>>();
            let mut results = handles.into_iter().map(|thread| thread.join().unwrap());
            let (first, bytes) = results.next().unwrap();
            for (next, next_bytes) in results {
                assert!(first.same_domain(&next));
                assert_eq!(next_bytes, bytes);
            }
            assert_eq!(first.used_bytes().unwrap(), bytes);
        });
    }

    #[test]
    fn unknown_fixed_native_baseline_keeps_ordinary_authority_and_strict_refusal() {
        let mut facts = safemlx::runtime_static_baseline();
        facts.constant_registry = false; // explicit unsupported producer fixture
        let domain = StaticDomain::new(facts).unwrap();
        let bytes = domain.pool.used_bytes().unwrap();
        assert!(bytes > 0);
        assert!(domain._unqualified.is_some());
        assert_eq!(domain.pool.unquoted_owner_count().unwrap(), 1);
        let ordinary = NativeMemoryOwner::acquire_typed(&domain.pool).unwrap();
        assert_eq!(domain.pool.unquoted_owner_count().unwrap(), 2);
        drop(ordinary);
        let alias = domain.pool.clone();
        assert!(matches!(
            alias.reserve(&InferenceExecutionIdentity::default(), &zero_admission()),
            Err(WorkingMemoryError::UnknownBound)
        ));
        assert_eq!(alias.unquoted_owner_count().unwrap(), 1);
        assert_eq!(alias.used_bytes().unwrap(), bytes);
    }

    #[test]
    fn clones_share_one_unquoted_owner_across_threads() {
        let pool = WorkingMemoryPool::new(1024, 13).unwrap();
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let descendant = owner.clone();
        assert_eq!(owner.pool().used_bytes().unwrap(), 13);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop(owner);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        std::thread::spawn(move || {
            assert_eq!(descendant.pool().unquoted_owner_count().unwrap(), 1);
            drop(descendant);
        })
        .join()
        .unwrap();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        assert_eq!(pool.used_bytes().unwrap(), 13);
    }

    #[test]
    fn rejected_acquisition_preserves_zero_byte_reservation_and_error_source() {
        let pool = WorkingMemoryPool::new(0, 0).unwrap();
        let reservation = pool
            .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
            .unwrap();
        let Error::Other(error) = NativeMemoryOwner::acquire(&pool).unwrap_err() else {
            panic!("retain the original neutral accounting error");
        };
        assert!(matches!(
            error.downcast_ref::<WorkingMemoryError>(),
            Some(WorkingMemoryError::ReservedWorkActive)
        ));
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        drop(reservation);
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop(owner);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }

    struct Pending(Rc<Cell<bool>>);
    impl submission_recovery::Probe for Pending {
        fn seal(&mut self) {}
        fn progress(&self) -> submission_recovery::Status {
            submission_recovery::Status {
                settled: self.0.get(),
                failed: true,
                blocked: !self.0.get(),
            }
        }
    }

    #[test]
    fn pending_failure_and_unwind_retain_owner_until_independent_settlement() {
        for unwind in [false, true] {
            let pool = WorkingMemoryPool::new(1024, 0).unwrap();
            let settled = Rc::new(Cell::new(false));
            let owner = NativeMemoryOwner::acquire(&pool).unwrap();
            let recovery = submission_recovery::Recovery::with_probe(
                owner.clone(),
                Pending(Rc::clone(&settled)),
            );
            drop(owner);
            if unwind {
                let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _recovery = recovery;
                    panic!("fixture operation unwind");
                }));
                assert!(panic.is_err());
            } else {
                drop(recovery);
            }
            submission_recovery::reap();
            assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
            settled.set(true);
            submission_recovery::wait_for_retirement(|| pool.unquoted_owner_count().unwrap() == 0);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

pub(crate) mod gpu_stream;
