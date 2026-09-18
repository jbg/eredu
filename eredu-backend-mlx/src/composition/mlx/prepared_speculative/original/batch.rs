//! Common source-authenticated host preparation for both actual strategies.
use super::*;
use crate::composition::mlx::session::MlxModelSession;
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer};
use eredu_runtime::working_memory::PreparedSemanticSource;

pub(super) fn buffer<T>(
    count: usize,
    funding: &HostMetadataFunding,
) -> Result<SpeculativeBuffer<T>, Error> {
    let bytes = SpeculativeBuffer::<T>::retained_control_bytes(count)
        .and_then(|n| {
            n.checked_add(HostPreparationAuthority::retention_bytes::<
                HostMetadataFunding,
            >()?)
        })
        .and_then(|n| {
            n.checked_add(size_of::<(
                usize,
                &HostMetadataFunding,
                Result<SpeculativeBuffer<T>, Error>,
            )>())
        })
        .ok_or(Error::WorkspacePlanning(
            HostMetadataFundingError::Overflow,
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
) -> Result<(PreparedSemanticSource, NonZeroU64), Error> {
    let semantic = lane
        .semantic()
        .prepared_source()
        .and_then(|source| {
            source.downcast_ref::<eredu_runtime::working_memory::PreparedSemanticState>()
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
        size_of::<PreparedSemanticSource>(),
        size_of::<Result<(PreparedSemanticSource, NonZeroU64), Error>>(),
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
                    HostMetadataFundingError::Overflow,
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
    let prompt = lane.prompt();
    // Authenticate the complete actual owner before reading either text or
    // media geometry. Publicly replaceable descriptors cannot select a lane.
    prompt.original_prediction_source(backend.memory_pool()).map_err(Error::PrefillControl)?;
    prompt.with_borrowed(|input| -> Result<(), Error> {
    if let Some(crate::backend::runtime::media::input::OriginalMediaPacket::Original(packet)) = input.original_media() {
        preparation.metadata_funding().reserve_metadata(size_of::<(
            eredu_runtime::working_memory::MediaSessionBinding,
            Result<eredu_runtime::working_memory::MediaSessionBinding,
                eredu_runtime::replicated_session::MediaSemanticBindingError<Error>>,
        )>()).map_err(Error::WorkspacePlanning)?;
        let current = model.erased().current_media_semantic_binding()
            .map_err(|cause| crate::composition::mlx::model::retain_planning_error(cause, preparation.metadata_funding().clone()))?;
        if !packet.borrowed_semantics().binding().matches(&current) {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
    }
        Ok(())
    })?;
    let [batch, positions] = prompt.with_borrowed(|input| match input.original_media() {
        Some(packet) => Some(packet.shape()),
        None => prompt.plain_token_array().and_then(|tokens| match tokens.shape() {
            [batch, positions] => Some([u64::try_from(*batch).ok()?, u64::try_from(*positions).ok()?]),
            _ => None,
        }),
    }).ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    if batch != 1 { return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)); }
    let positions = NonZeroU64::new(positions)
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    Ok((preparation.clone(), positions))
}
