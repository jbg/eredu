//! Exclusive settled-session readiness followed by originally funded KV publication.
use super::*;
use crate::backend::runtime::{
    cache::state::{MlxHybridState, MlxKeyValueState},
    residency::storage::StorageIdentity,
};
use eredu_runtime::working_memory::{
    PreparedResidentKvReset, ResidentKvResetState, ResidentResetDisplaced,
    ResidentResetPublicationCustody, ResidentResetPublicationProfile, ResidentResetSession,
    ResidentResetSource, WorkingMemoryError, WorkingMemoryPool,
};

type Plan<'a, S = MlxKeyValueState> = PreparedResidentKvReset<'a, S, StorageIdentity>;
struct Displaced<S: ResidentKvResetState = MlxKeyValueState> {
    _state: ResidentResetDisplaced<S>,
    _memory: NativeMemoryRetention,
}
struct Retirement<S: ResidentKvResetState = MlxKeyValueState> {
    displaced: Option<Displaced<S>>,
    #[cfg(all(
        test,
        target_vendor = "apple",
        feature = "metal",
        not(feature = "cuda")
    ))]
    _witness: Option<tests::NodeWitness>,
    // Last: same reset custody survives the concrete node and native payload.
    _custody: ResidentResetPublicationCustody,
}
struct Prepared<S: ResidentKvResetState = MlxKeyValueState>(
    ordinary_retirement::OrdinaryRetirement<Retirement<S>>,
);
impl<S: ResidentKvResetState> ResidentResetPublicationProfile for Prepared<S> {
    fn control_bytes() -> Option<u64> {
        let fixed = [
            size_of::<MlxResidentResetReadiness<'_, '_>>(),
            size_of::<eredu_core::SessionCapabilities>(),
            size_of::<eredu_core::SessionResetClaim<'_>>(),
            size_of::<Result<(), BackendFailure>>(),
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Displaced<S>>(),
            size_of::<Option<Displaced<S>>>(),
            size_of::<Retirement<S>>(),
            size_of::<NativeMemoryRetention>(),
            size_of::<
                Result<
                    (),
                    (
                        WorkingMemoryError,
                        eredu_runtime::working_memory::ResidentResetInstallation<S>,
                    ),
                >,
            >(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        ordinary_retirement::OrdinaryRetirement::<Retirement<S>>::control_bytes()?
            .checked_add(u64::try_from(fixed).ok()?)
    }
    fn prepare(custody: ResidentResetPublicationCustody) -> Self {
        Self(ordinary_retirement::OrdinaryRetirement::new(Retirement {
            displaced: None,
            #[cfg(all(
                test,
                target_vendor = "apple",
                feature = "metal",
                not(feature = "cuda")
            ))]
            _witness: tests::node_witness(),
            _custody: custody,
        }))
    }
}

// Typed backend erasure only. State equations/policy and publication checks stay
// in the shared neutral driver; no family or alternative reset engine lives here.
trait StateProvider: ResidentKvResetState {
    fn source(
        session: &MlxModelSession,
    ) -> Result<ResidentResetSource<'_, Self>, WorkingMemoryError>;
    fn install(
        payload: &mut SessionPayload,
        state: eredu_runtime::working_memory::ResidentResetInstallation<Self>,
    ) -> Result<
        ResidentResetDisplaced<Self>,
        (
            WorkingMemoryError,
            eredu_runtime::working_memory::ResidentResetInstallation<Self>,
        ),
    >;
}
impl StateProvider for MlxKeyValueState {
    fn source(
        session: &MlxModelSession,
    ) -> Result<ResidentResetSource<'_, Self>, WorkingMemoryError> {
        session.payload.model.erased().resident_reset_source()
    }
    fn install(
        payload: &mut SessionPayload,
        state: eredu_runtime::working_memory::ResidentResetInstallation<Self>,
    ) -> Result<
        ResidentResetDisplaced<Self>,
        (
            WorkingMemoryError,
            eredu_runtime::working_memory::ResidentResetInstallation<Self>,
        ),
    > {
        payload.model.erased_mut().install_resident_reset(state)
    }
}
impl StateProvider for MlxHybridState {
    fn source(
        session: &MlxModelSession,
    ) -> Result<ResidentResetSource<'_, Self>, WorkingMemoryError> {
        session
            .payload
            .model
            .erased()
            .resident_hybrid_reset_source()
    }
    fn install(
        payload: &mut SessionPayload,
        state: eredu_runtime::working_memory::ResidentResetInstallation<Self>,
    ) -> Result<
        ResidentResetDisplaced<Self>,
        (
            WorkingMemoryError,
            eredu_runtime::working_memory::ResidentResetInstallation<Self>,
        ),
    > {
        payload
            .model
            .erased_mut()
            .install_resident_hybrid_reset(state)
    }
}

impl<S: StateProvider> ResidentResetSession<S> for MlxModelSession {
    fn validate_resident_reset_source(
        &self,
        source: &ResidentResetSource<'_, S>,
    ) -> Result<(), WorkingMemoryError> {
        self.check_reset_idle()?;
        let actual = S::source(self)?;
        if actual.same_source(source) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl MlxModelSession {
    /// Pure host predicate only: no reap, stream/device query, progress or wait.
    /// A separate readiness boundary must establish completed native work.
    fn check_reset_idle(&self) -> Result<(), WorkingMemoryError> {
        if self.poison.get()
            || self
                .failure
                .try_borrow()
                .map_err(|_| WorkingMemoryError::ResetAdmissionBusy)?
                .is_some()
        {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        if self.payload.distributed.is_some() || self.payload.target.has_retained_world() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        self.authority
            .try_borrow()
            .map_err(|_| WorkingMemoryError::ResetAdmissionBusy)?
            .require_idle()
            .map_err(|_| WorkingMemoryError::ResetAdmissionBusy)
    }
    #[cfg(test)]
    fn resident_reset_plan(&self, pool: &WorkingMemoryPool) -> Result<Plan<'_>, BackendFailure> {
        self.resident_reset_plan_for::<MlxKeyValueState>(pool)
    }
    fn resident_reset_plan_for<S: StateProvider>(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<Plan<'_, S>, BackendFailure> {
        self.check_reset_idle()
            .map_err(BackendFailure::from_error)?;
        if !self.payload.memory_pool.same_domain(pool) {
            return Err(BackendFailure::from_error(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        let source = S::source(self).map_err(BackendFailure::from_error)?;
        let table = source.state().resident_reset_layers().metadata();
        let kind = pool
            .classify_host_slot_source(table)
            .map_err(BackendFailure::from_error)?;
        let ordinary = kind.registered().is_some();
        drop(kind);
        if ordinary {
            let table = StorageIdentity::HostMetadata(table.identity().registry_key().clone());
            let layout = StorageIdentity::HostMetadata(
                source
                    .state()
                    .resident_reset_layout()
                    .identity()
                    .registry_key()
                    .clone(),
            );
            PreparedResidentKvReset::prepare_registered(source, table, layout)
        } else {
            PreparedResidentKvReset::prepare_original(source)
        }
        .map_err(BackendFailure::from_error)
    }

    /// Consuming readiness (or the explicitly settled test fixture) supplies
    /// the genuine core claim. No native work or global housekeeping occurs in
    /// construction or publication here.
    fn publish_prepared_resident_reset(
        &mut self,
        pool: &WorkingMemoryPool,
        claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        use crate::composition::mlx::replicated_text::ResidentResetProfile;
        match self.payload.model.erased().resident_reset_profile() {
            Some(ResidentResetProfile::KeyValue) => {
                self.publish_resident_reset_for::<MlxKeyValueState>(pool, claim)
            }
            Some(ResidentResetProfile::Hybrid) => {
                self.publish_resident_reset_for::<MlxHybridState>(pool, claim)
            }
            None => Err(readiness_memory(WorkingMemoryError::UnknownBound)),
        }
    }
    fn publish_resident_reset_for<S: StateProvider>(
        &mut self,
        pool: &WorkingMemoryPool,
        claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        claim
            .validate_session(self)
            .map_err(BackendFailure::from_error)?;
        self.check_reset_idle()
            .map_err(BackendFailure::from_error)?;
        if self.payload.get_mut().is_none() {
            return Err(BackendFailure::from_error(
                WorkingMemoryError::ResetAdmissionBusy,
            ));
        }
        let (installation, mut prepared) = self
            .resident_reset_plan_for::<S>(pool)?
            .construct_for_publication::<Self, Prepared<S>>(self, claim, pool)
            .map_err(BackendFailure::from_error)?;
        // Constructors above clone source metadata/custody, never this payload.
        // Still return the complete destination if a future change breaks that
        // invariant, before touching installed state.
        #[cfg(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        ))]
        let rejected = tests::reject_constructed(self);
        #[cfg(not(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        )))]
        let rejected = false;
        let installed = if rejected {
            Err((WorkingMemoryError::ExecutionFenced, installation))
        } else {
            match self.payload.get_mut() {
                None => Err((WorkingMemoryError::ResetAdmissionBusy, installation)),
                Some(payload) => match S::install(payload, installation) {
                    Err(error) => Err(error),
                    Ok(state) => {
                        let memory = std::mem::take(&mut payload.state_memory);
                        // This private slot was prepared empty and has one caller.
                        // Only moves follow the shared session's first mutation.
                        prepared.0.displaced = Some(Displaced {
                            _state: state,
                            _memory: memory,
                        });
                        Ok(())
                    }
                },
            }
        };
        // All native session and state loans ended. This only queues the node;
        // its old arrays and complete allocation retire at ordinary housekeeping.
        drop(prepared);
        installed.map_err(|(cause, installation)| {
            BackendFailure::from_error(installation.into_error(cause))
        })
    }
}

/// Closed readiness for one actual resident session. No session accessor, clone,
/// callback or Drop work exists; the exclusive borrow lasts through consumption.
#[must_use]
pub struct MlxResidentResetReadiness<'s, 'backend> {
    backend: &'s MlxBackend<'backend>,
    session: &'s mut MlxModelSession,
}
impl eredu_core::SessionResetReadiness for MlxResidentResetReadiness<'_, '_> {
    fn capabilities(&self) -> eredu_core::SessionCapabilities {
        self.session.capabilities
    }
}

fn readiness_memory(cause: WorkingMemoryError) -> BackendFailure {
    let kind = match cause {
        WorkingMemoryError::ResetAdmissionBusy => BackendFailureKind::Busy,
        WorkingMemoryError::UnknownBound => BackendFailureKind::Unsupported,
        WorkingMemoryError::IdentityMismatch => BackendFailureKind::InvalidSession,
        _ => BackendFailureKind::Other,
    };
    BackendFailure::new(kind, cause)
}
fn readiness_stream(
    cause: crate::backend::managed_memory::gpu_stream::MlxStreamOwnershipError,
) -> BackendFailure {
    use crate::backend::managed_memory::gpu_stream::MlxStreamOwnershipError as E;
    let kind = match &cause {
        E::Gpu(safemlx::GpuStreamRegistrationCause::Busy)
        | E::CpuWorker(safemlx::CpuWorkerCause::Busy)
        | E::CpuStream(safemlx::StreamRegistrationCause::Busy) => BackendFailureKind::Busy,
        E::Accounting(WorkingMemoryError::UnknownBound) => BackendFailureKind::Unsupported,
        _ => BackendFailureKind::Other,
    };
    BackendFailure::new(kind, cause)
}

impl<'backend> eredu_core::SessionResetPreparationBackend for MlxBackend<'backend> {
    type Readiness<'s>
        = MlxResidentResetReadiness<'s, 'backend>
    where
        Self: 's;

    fn prepare_session_reset_ordinary<'s>(
        backend: &'s Self,
        session: &'s mut MlxModelSession,
    ) -> Result<Self::Readiness<'s>, BackendFailure>
    where
        Self: 's,
    {
        session.check_reset_idle().map_err(readiness_memory)?;
        if !session
            .payload
            .memory_pool
            .same_domain(backend.memory_pool())
            || !backend.matches_prepared_target(&session.payload.target)
        {
            return Err(readiness_memory(WorkingMemoryError::IdentityMismatch));
        }
        // No exported Weak can recreate a payload owner after this succeeds.
        // Existing completion/recovery aliases keep the owner non-unique.
        if session.payload.get_mut().is_none() {
            return Err(readiness_memory(WorkingMemoryError::ResetAdmissionBusy));
        }
        let model = &session.payload.model;
        let selected = model
            .inference_blueprint()
            .ok_or_else(|| readiness_memory(WorkingMemoryError::UnknownBound))?
            .selected();
        if selected.text_realization().residency()
            != eredu_runtime::LayerWeightResidency::FullyResident
            || !crate::backend::runtime::cache::kv::PagedKeyValueCache::original_reset_residency_supported(
                selected.text_realization().state().policy(),
                model
                    .erased()
                    .capability_estimate()
                    .state_layout()
                    .layer_layout(),
                session.floating_state_dtype_bytes,
            )
            || selected.communication_manifest().is_some()
            || selected.prediction_extension().is_some()
            || model.erased().has_embedded_prediction()
            || session.payload.parameter_state.active.is_some()
            || !model.has_published_idle_storage()
        {
            return Err(readiness_memory(WorkingMemoryError::UnknownBound));
        }
        // Validate the actual whole-state constructor through its typed native
        // representation. Both profiles use the same neutral original worker.
        use crate::composition::mlx::replicated_text::ResidentResetProfile;
        match model.erased().resident_reset_profile() {
            Some(ResidentResetProfile::KeyValue) => {
                session.resident_reset_plan_for::<MlxKeyValueState>(backend.memory_pool())?;
            }
            Some(ResidentResetProfile::Hybrid) => {
                session.resident_reset_plan_for::<MlxHybridState>(backend.memory_pool())?;
            }
            None => return Err(readiness_memory(WorkingMemoryError::UnknownBound)),
        }
        // Target compatibility allows ordinary stream changes on one device.
        // Therefore current stream counters alone cannot establish loading or
        // transfer completion. Reuse the actual complete borrowed inventory:
        // every retained array must expose certified completed backing, and
        // external/in-flight owners make these same collectors incomplete.
        let pool = backend.memory_pool();
        let mut nonstate =
            crate::backend::runtime::residency::storage::RetainedStorage::original_census(pool);
        let mut decoder =
            crate::backend::runtime::residency::storage::RetainedStorage::original_census(pool);
        session
            .payload
            .collect_retained_idle_storage(&mut nonstate, &mut decoder)
            .map_err(BackendFailure::from_error)?;
        if nonstate
            .original_publication_rows(pool)
            .map_err(BackendFailure::from_error)?
            .is_none()
            || decoder
                .original_publication_rows(pool)
                .map_err(BackendFailure::from_error)?
                .is_none()
        {
            return Err(readiness_memory(WorkingMemoryError::UnknownBound));
        }
        backend
            .observe_original_streams_idle()
            .map_err(readiness_stream)?;
        Ok(MlxResidentResetReadiness { backend, session })
    }

    fn reset_ready_session_admitted<'s>(
        backend: &'s Self,
        ready: Self::Readiness<'s>,
        claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure>
    where
        Self: 's,
    {
        if !std::ptr::eq(backend, ready.backend) {
            return Err(readiness_memory(WorkingMemoryError::IdentityMismatch));
        }
        // The retained exclusive borrow excluded all further session producers.
        // Publication revalidates current authority, domain, exact source and
        // genuine claim before admission; there is no wait or global retirement.
        ready
            .session
            .publish_prepared_resident_reset(backend.memory_pool(), claim)
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
pub(super) mod tests;
