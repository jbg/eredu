//! Host-only reset slots for the prepared parameter publication transaction.
use super::*;
use crate::backend::{
    ordinary_retirement::OrdinaryRetirement,
    runtime::{execution::generic::MlxParameterPreparation, residency::storage::StorageIdentity},
};
use eredu_runtime::working_memory::{
    PreparedParameterStateReset, PreparedResidentKvReset, ResidentResetProjection,
    ResidentResetPublicationCustody, ResidentResetPublicationProfile, ResidentResetSession,
    ResidentResetSource, ResidentTableResetState, WorkingMemoryError,
};
use std::{
    any::Any,
    mem::{size_of, size_of_val},
};

struct SlotData<S: ResidentTableResetState> {
    state: Option<PreparedParameterStateReset<S>>,
    memory: crate::backend::managed_memory::NativeMemoryRetention,
    publication: Option<crate::backend::runtime::residency::storage::RetainedStoragePublication>,
    // The retirement node and its erased outer Box die before this original account.
    _custody: ResidentResetPublicationCustody,
}
struct Slot<S: ResidentTableResetState>(OrdinaryRetirement<SlotData<S>>);
impl<S: ResidentTableResetState> ResidentResetPublicationProfile for Slot<S> {
    fn control_bytes() -> Option<u64> {
        let frames = [
            size_of::<Self>(),
            size_of::<Box<Self>>(),
            size_of::<Box<dyn Any>>(),
            size_of::<Option<PreparedParameterStateReset<S>>>(),
            size_of::<Result<Box<dyn Any>, Error>>(),
        ];
        OrdinaryRetirement::<SlotData<S>>::control_bytes()?.checked_add(
            u64::try_from(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)?,
            )
            .ok()?,
        )
    }
    fn prepare(custody: ResidentResetPublicationCustody) -> Self {
        Self(OrdinaryRetirement::new(SlotData {
            state: None,
            memory: Default::default(),
            publication: None,
            _custody: custody,
        }))
    }
}
struct SourceLoan<F>(F);
impl<S, F> ResidentResetSession<S> for SourceLoan<F>
where
    S: ResidentTableResetState,
    F: Fn(&ResidentResetSource<'_, S>) -> Result<(), WorkingMemoryError>,
{
    fn validate_resident_reset_source(
        &self,
        source: &ResidentResetSource<'_, S>,
    ) -> Result<(), WorkingMemoryError> {
        (self.0)(source)
    }
}
fn missing() -> Error {
    Error::PrefillControl(WorkingMemoryError::UnknownBound)
}

pub(super) fn prepare_parameter_reset<A, M, D>(
    session: &ReplicatedTextSession<A, MlxNeuralBackend, M, D>,
    preparation: &MlxParameterPreparation<'_>,
    profile: Option<ResidentResetProfile>,
) -> Result<Box<dyn Any>, Error>
where
    M: ReplicatedTextSessionMechanisms<A, MlxNeuralBackend>,
    M::State: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, M::State>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        M::State,
        M::ResidentPolicy,
        M::BoundedPolicy,
    >,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    match profile {
        Some(ResidentResetProfile::KeyValue) => {
            prepare::<MlxKeyValueState, _, _, _>(session, preparation)
        }
        Some(ResidentResetProfile::Hybrid) => {
            prepare::<MlxHybridState, _, _, _>(session, preparation)
        }
        Some(ResidentResetProfile::Pooling) => {
            prepare::<MlxPoolingAttentionState, _, _, _>(session, preparation)
        }
        None => Err(missing()),
    }
}
fn prepare<S, A, M, D>(
    session: &ReplicatedTextSession<A, MlxNeuralBackend, M, D>,
    preparation: &MlxParameterPreparation<'_>,
) -> Result<Box<dyn Any>, Error>
where
    S: ResidentTableResetState,
    M: ReplicatedTextSessionMechanisms<A, MlxNeuralBackend>,
    M::State: ResidentResetProjection<S>,
    A: LayeredArchitecture<MlxNeuralBackend, M::State>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        M::State,
        M::ResidentPolicy,
        M::BoundedPolicy,
    >,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let pool = preparation.environment.pool();
    let source = session
        .projected_resident_reset_source::<S>()
        .map_err(Error::PrefillControl)?;
    let table = source.state().resident_reset_layers().metadata();
    let ordinary = pool
        .classify_host_slot_source(table)
        .map_err(Error::PrefillControl)?
        .registered()
        .is_some();
    let keys = ordinary.then(|| {
        (
            StorageIdentity::HostMetadata(table.identity().registry_key().clone()),
            StorageIdentity::HostMetadata(
                source
                    .state()
                    .resident_reset_layout()
                    .identity()
                    .registry_key()
                    .clone(),
            ),
        )
    });
    let plan = match &keys {
        Some((table, layout)) => {
            PreparedResidentKvReset::prepare_registered(source, table.clone(), layout.clone(), pool)
        }
        None => PreparedResidentKvReset::prepare_original(source, pool),
    }
    .map_err(|cause| Error::Neural(preparation.funding.metadata_source(cause)))?;
    let loan = SourceLoan(|expected: &ResidentResetSource<'_, S>| {
        if session
            .projected_resident_reset_source::<S>()?
            .same_source(expected)
        {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    });
    let (installation, mut slot) = plan
        .construct_for_parameter_publication::<_, Slot<S>>(
            &loan,
            preparation.execution,
            &pool.configured_limits(),
            pool,
        )
        .map_err(|cause| Error::Neural(preparation.funding.metadata_source(cause)))?;
    let state = session
        .prepare_parameter_state_reset(installation)
        .map_err(|(cause, installation)| {
            Error::Neural(
                preparation
                    .funding
                    .metadata_source(installation.into_error(cause)),
            )
        })?;
    slot.0.state = Some(state);
    Ok(Box::new(slot))
}

pub(super) fn validate_prepared_parameter_reset<A, M, D>(
    session: &ReplicatedTextSession<A, MlxNeuralBackend, M, D>,
    slot: &dyn Any,
) -> Result<(), Error>
where
    M: ReplicatedTextSessionMechanisms<A, MlxNeuralBackend>,
    M::State: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, M::State>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        M::State,
        M::ResidentPolicy,
        M::BoundedPolicy,
    >,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    if let Some(slot) = slot.downcast_ref::<Slot<MlxKeyValueState>>() {
        session
            .validate_parameter_state_reset(slot.0.state.as_ref().ok_or_else(missing)?)
            .map_err(Error::PrefillControl)
    } else if let Some(slot) = slot.downcast_ref::<Slot<MlxHybridState>>() {
        session
            .validate_parameter_state_reset(slot.0.state.as_ref().ok_or_else(missing)?)
            .map_err(Error::PrefillControl)
    } else if let Some(slot) = slot.downcast_ref::<Slot<MlxPoolingAttentionState>>() {
        session
            .validate_parameter_state_reset(slot.0.state.as_ref().ok_or_else(missing)?)
            .map_err(Error::PrefillControl)
    } else {
        Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
    }
}
pub(super) fn exchange_prepared_parameter_reset<A, M, D>(
    session: &mut ReplicatedTextSession<A, MlxNeuralBackend, M, D>,
    slot: &mut dyn Any,
) -> Result<(), Error>
where
    M: ReplicatedTextSessionMechanisms<A, MlxNeuralBackend>,
    M::State: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, M::State>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        M::State,
        M::ResidentPolicy,
        M::BoundedPolicy,
    >,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    if let Some(slot) = slot.downcast_mut::<Slot<MlxKeyValueState>>() {
        session
            .exchange_parameter_state_reset(slot.0.state.as_mut().ok_or_else(missing)?)
            .map_err(Error::PrefillControl)
    } else if let Some(slot) = slot.downcast_mut::<Slot<MlxHybridState>>() {
        session
            .exchange_parameter_state_reset(slot.0.state.as_mut().ok_or_else(missing)?)
            .map_err(Error::PrefillControl)
    } else if let Some(slot) = slot.downcast_mut::<Slot<MlxPoolingAttentionState>>() {
        session
            .exchange_parameter_state_reset(slot.0.state.as_mut().ok_or_else(missing)?)
            .map_err(Error::PrefillControl)
    } else {
        Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
    }
}

/// These existing payload owners follow their exact displaced state into the
/// same prepaid retirement node. All casts are checked before the first move.
pub(in crate::composition::mlx) fn exchange_parameter_reset_memory(
    slot: &mut dyn Any,
    memory: &mut crate::backend::managed_memory::NativeMemoryRetention,
    publication: &mut Option<
        crate::backend::runtime::residency::storage::RetainedStoragePublication,
    >,
    covered: bool,
) -> Result<(), Error> {
    fn exchange<S: ResidentTableResetState>(
        slot: &mut Slot<S>,
        memory: &mut crate::backend::managed_memory::NativeMemoryRetention,
        publication: &mut Option<
            crate::backend::runtime::residency::storage::RetainedStoragePublication,
        >,
        covered: bool,
    ) {
        std::mem::swap(memory, &mut slot.0.memory);
        if covered {
            std::mem::swap(publication, &mut slot.0.publication);
        }
    }
    if let Some(slot) = slot.downcast_mut::<Slot<MlxKeyValueState>>() {
        exchange(slot, memory, publication, covered);
    } else if let Some(slot) = slot.downcast_mut::<Slot<MlxHybridState>>() {
        exchange(slot, memory, publication, covered);
    } else if let Some(slot) = slot.downcast_mut::<Slot<MlxPoolingAttentionState>>() {
        exchange(slot, memory, publication, covered);
    } else {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    }
    Ok(())
}
