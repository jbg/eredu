//! Common source-authenticated host preparation for both actual strategies.
use super::*;
use crate::composition::mlx::session::MlxModelSession;
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer};
use eredu_runtime::working_memory::OriginalSpeculativeSemanticPreparation;

pub(super) fn buffer<T>(
    count: usize,
    funding: &WorkspaceMetadataFunding,
) -> Result<SpeculativeBuffer<T>, Error> {
    let bytes = SpeculativeBuffer::<T>::retained_control_bytes(count)
        .and_then(|n| {
            n.checked_add(HostPreparationAuthority::retention_bytes::<
                WorkspaceMetadataFunding,
            >()?)
        })
        .and_then(|n| {
            n.checked_add(size_of::<(
                usize,
                &WorkspaceMetadataFunding,
                Result<SpeculativeBuffer<T>, Error>,
            )>())
        })
        .ok_or(Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Overflow,
        ))?;
    funding
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)?;
    SpeculativeBuffer::try_new_retained(count, HostPreparationAuthority::retain(funding.clone()))
        .map_err(|cause| super::super::super::model::retain_planning_error(cause, funding.clone()))
}

pub(super) fn inspect<C: SpeculativeTokenFilterController>(
    backend: &MlxBackend<'_>,
    session: &MlxModelSession,
    lane: &SpeculativeGenerationLane<'_, MlxBackend<'_>, C>,
) -> Result<(OriginalSpeculativeSemanticPreparation, NonZeroU64), Error> {
    let semantic = lane
        .semantic()
        .prepared_source()
        .and_then(|source| {
            source.downcast_ref::<eredu_runtime::working_memory::OriginalSpeculativePlainText>()
        })
        .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
    let preparation = semantic.preparation();
    let model = session
        .original_model_source()
        .map_err(Error::PrefillControl)?;
    preparation
        .validate(
            backend.memory_pool(),
            model.erased().inference_execution_identity(),
        )
        .map_err(Error::PrefillControl)?;
    let frames = [
        size_of::<(&MlxBackend<'_>, &MlxModelSession)>(),
        size_of::<&SpeculativeGenerationLane<'_, MlxBackend<'_>, C>>(),
        size_of::<OriginalSpeculativeSemanticPreparation>(),
        size_of::<Result<(OriginalSpeculativeSemanticPreparation, NonZeroU64), Error>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<[usize; 2]>(),
    ];
    preparation
        .metadata_funding()
        .reserve_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(Error::WorkspacePlanning(
                    WorkspaceMetadataFundingError::Overflow,
                ))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    semantic
        .validate_pool(backend.memory_pool())
        .map_err(Error::PrefillControl)?;
    preparation
        .validate_configuration(lane.configuration())
        .map_err(Error::PrefillControl)?;
    preparation
        .validate_callback(lane.event_callback())
        .map_err(Error::PrefillControl)?;
    let tokens = lane
        .prompt()
        .plain_token_array()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let [1, positions] = tokens.shape() else {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    };
    let positions = u64::try_from(*positions)
        .ok()
        .and_then(NonZeroU64::new)
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    Ok((preparation.clone(), positions))
}
