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
    pub(super) fn prepare(
        input: &MlxModelInput,
        prepared: &PreparedModelInputOwner<MlxTensor>,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<Self, Error> {
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
                        WorkspaceMetadataFundingError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        sources.validate_environment(environment)?;
        if prepared.original_source().is_none() {
            return Err(invalid());
        }
        let original = input
            .original_text_copy_source()
            .transpose()
            .map_err(Error::PrefillControl)?
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        input::validate_token_ids_with_diagnostic(original.array(), |message| {
            Error::Neural(funding.metadata_error(message))
        })?;
        // The selected speculative lane is one sequence. Payload shape and
        // dtype above use the very same ordinary token-input validator.
        if original.array().shape()[0] != 1 {
            return Err(invalid());
        }
        let (roots, mechanisms) = sources.numerical_prerequisites();
        // Same isolated prepared-source copy worker; signed token storage is
        // preserved rather than passed through a token-normalization program.
        let copy = IsolatedArrayCopy::new(original.array()).copy_prepared(
            &original,
            environment,
            roots,
            mechanisms,
            funding,
            sources.request().capacity_bytes(),
        )?;
        let value = registered_copy_input(copy, sources, environment)?;
        Ok(Self {
            value,
            prepared: prepared.clone(),
            identity: sources.request().source_identity(),
        })
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
                        WorkspaceMetadataFundingError::Overflow,
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
