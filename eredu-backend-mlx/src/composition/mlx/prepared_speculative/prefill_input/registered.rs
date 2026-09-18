//! One real full B copy, then shared admitted sequence views of that copy.
use super::*;
use crate::backend::array_copy::{
    IsolatedArrayCopy, OriginalPreparedArrayCopySource, RegisteredArrayCopy,
};
use crate::composition::mlx::speculative::{
    OriginalNumericalValue, OriginalSpeculativeNumericalSources, prepared_registered_token_range,
    registered_copy_input,
};
use eredu_architectures::speculative_execution::EmbeddedPredictionTensor;
use eredu_runtime::working_memory::{OriginalSpeculativeSourceIdentity, WorkingMemoryError};

pub(crate) struct OriginalEmbeddedPrefillInput {
    value: OriginalNumericalValue,
    prepared: PreparedModelInputOwner<MlxTensor>,
    identity: OriginalSpeculativeSourceIdentity,
    media: Option<eredu_architectures::media_plan::BoundPreparedMediaSemantics>,
}
impl std::fmt::Debug for OriginalEmbeddedPrefillInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OriginalEmbeddedPrefillInput")
    }
}
fn invalid() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
impl OriginalEmbeddedPrefillInput {
    pub(super) fn prepare<A, S, P>(
        plan: &P, prepared: &PreparedModelInputOwner<MlxTensor>,
        media: Option<eredu_architectures::media_plan::BoundPreparedMediaSemantics>,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<Self, Error>
    where S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
        A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
        P: eredu_architectures::speculative_execution::PredictionPrefillPlan<A, MlxNeuralBackend, S>,
    {
        let (sources, environment) = context.original_numerical().ok_or_else(invalid)?;
        let funding = sources.metadata_funding();
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<OriginalNumericalValue>(),
            size_of::<RegisteredArrayCopy>(),
            size_of::<Result<RegisteredArrayCopy, Error>>(),
            size_of::<OriginalSpeculativeSourceIdentity>(),
            size_of::<PreparedModelInputOwner<MlxTensor>>(),
            size_of::<Result<Option<OriginalPreparedArrayCopySource<'_>>, WorkingMemoryError>>(),
            OriginalPreparedArrayCopySource::control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            size_of::<IsolatedArrayCopy<'_>>(),
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(
                        HostMetadataFundingError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        sources.validate_environment(environment)?;
        if prepared.original_source().is_none() {
            return Err(invalid());
        }
        use eredu_architectures::composite_execution::PredictionTokenPart;
        use crate::composition::mlx::speculative::{repeated_token_input, concatenate_token_inputs};
        let mut part_count = 0usize;
        plan.visit_token_parts(&mut |_| {
            part_count = part_count.checked_add(1).ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            Ok(())
        }).map_err(Error::Neural)?;
        let mut values = funding.metadata_vec(part_count).map_err(Error::Neural)?;
        let mut positions = 0u64;
        plan.visit_token_parts(&mut |part| {
            let result = (|| {
                funding.reserve_metadata(size_of::<(PredictionTokenPart<'_, MlxTensor>, Option<OriginalNumericalValue>,
                    OriginalNumericalValue, Result<OriginalNumericalValue, Error>, u64)>())
                    .map_err(Error::WorkspacePlanning)?;
                let (next, count) = match part {
                    PredictionTokenPart::Tokens(tokens) => {
                        let original = OriginalPreparedArrayCopySource::from_payload(prepared, tokens)
                            .map_err(Error::PrefillControl)?;
                        input::validate_token_ids_with_diagnostic(original.array(), |message| Error::Neural(funding.metadata_error(message)))?;
                        let [1, count] = original.array().shape() else { return Err(invalid()); };
                        let count = u64::try_from(*count).map_err(|_| invalid())?;
                        let (roots, mechanisms) = sources.numerical_prerequisites();
                        let copy = IsolatedArrayCopy::new(original.array()).copy_prepared(&original, environment,
                            roots, mechanisms, funding, sources.request().capacity_bytes())?;
                        (registered_copy_input(copy, sources, environment)?, count)
                    }
                    PredictionTokenPart::Repeated { token, positions } =>
                        (repeated_token_input(token, positions, sources, environment)?, positions),
                };
                positions = positions.checked_add(count).ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
                if values.len() == part_count { return Err(invalid()); }
                values.push(next);
                Ok(())
            })();
            result.map_err(|cause: Error| super::super::embedded_error::neural_cause(cause, context))
        }).map_err(Error::Neural)?;
        if plan.shape().map_err(Error::Neural)? != [1, positions] { return Err(invalid()); }
        if values.len() != part_count || part_count == 0 { return Err(invalid()); }
        let value = if values.len() == 1 { values.pop().expect("one token part") }
            else { concatenate_token_inputs(values, sources, environment)? };
        Ok(Self {
            value,
            prepared: prepared.clone(),
            identity: sources.request().source_identity(),
            media,
        })
    }
    pub(crate) fn project_media(
        &self, context: &WorkspaceContext, sources: &OriginalSpeculativeNumericalSources,
    ) -> Result<Option<eredu_architectures::prepared_execution::OriginalMediaWorkspaceInput>, Error> {
        self.validate_request(sources)?;
        context.charge_metadata(size_of::<(
            Option<eredu_architectures::media_plan::BoundPreparedMediaSemantics>,
            Option<eredu_architectures::prepared_execution::OriginalMediaWorkspaceInput>,
        )>()).map_err(|cause| Error::Neural(cause.into()))?;
        self.media.as_ref().map(|media| {
            eredu_architectures::prepared_execution::OriginalMediaWorkspaceInput::project_with_metadata(
                &self.prepared, media.clone(), context, sources.pool(),
            ).map_err(|cause| sources.retain_startup_error(cause))
        }).transpose()
    }
    pub(crate) fn validate_request(
        &self,
        sources: &OriginalSpeculativeNumericalSources,
    ) -> Result<(), Error> {
        if self.identity.belongs_to_request(sources.request()) {
            Ok(())
        } else {
            Err(invalid())
        }
    }
    pub(super) fn range(
        &self,
        prepared: Option<&PreparedModelInputOwner<MlxTensor>>,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<(MlxTensor, EmbeddedPredictionTensor<MlxTensor>), Error> {
        let (sources, environment) = context.original_numerical().ok_or_else(invalid)?;
        self.validate_request(sources)?;
        let prepared = prepared.ok_or_else(invalid)?;
        if !std::ptr::eq(prepared.parts(), self.prepared.parts())
            || prepared
                .original_source()
                .zip(self.prepared.original_source())
                .is_none_or(|(left, right)| !left.same_source(right))
        {
            return Err(invalid());
        }
        let preparation = context.original_cache_preparation().ok_or_else(invalid)?;
        let funding = sources.metadata_funding();
        let frames = [
            size_of::<(MlxTensor, EmbeddedPredictionTensor<MlxTensor>)>(),
            size_of::<Result<(MlxTensor, EmbeddedPredictionTensor<MlxTensor>), Error>>(),
            size_of::<Option<&PreparedModelInputOwner<MlxTensor>>>(),
            size_of::<Result<usize, std::num::TryFromIntError>>(),
            size_of::<safemlx::PreparedArrayClone>(),
            size_of::<Result<safemlx::PreparedArrayClone, safemlx::PreparedArrayCloneCause>>(),
            size_of::<Result<Array, safemlx::PreparedArrayCloneCause>>(),
            safemlx::PreparedArrayClone::control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            Array::inspection_clone_handle_bytes(),
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(
                        HostMetadataFundingError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let start = usize::try_from(chunk.input.start)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let end = usize::try_from(chunk.input.end)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let packet = prepared_registered_token_range(
            &self.value,
            start,
            end,
            sources,
            environment,
            preparation,
        )?;
        // This paid descriptor loan preserves the very same completed backing.
        // The tuple and chunk field order retire it before the actual packet/Q/H.
        let mut slot = safemlx::PreparedArrayClone::try_prepare_for_inspection()
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let alias = slot
            .fill_for_inspection(packet.as_array())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        Ok((MlxTensor::from_array(alias), packet))
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
