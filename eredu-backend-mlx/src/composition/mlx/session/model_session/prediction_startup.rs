//! Source-authenticated host preparation for the loaded embedded prediction lane.
use super::*;
use crate::{
    backend::OriginalCopyEnvironment,
    composition::mlx::replicated_text::{
        OriginalPredictionLane, OriginalPredictionStartupContext, StartupCause,
    },
};
use eredu_runtime::working_memory::{OriginalSpeculativeSemanticPreparation, WorkingMemoryError};

impl MlxModelSession {
    /// Borrows the actual idle loaded extension. It neither fabricates an external
    /// schedule nor acquires an ordinary memory owner. A missing extension returns
    /// None; a present but unqualified source returns its typed preparation refusal.
    pub(crate) fn prepare_original_prediction_lane(
        &self,
        preparation: &OriginalSpeculativeSemanticPreparation,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Option<OriginalPredictionLane>, Error> {
        let model = self
            .original_model_source()
            .map_err(Error::PrefillControl)?;
        preparation
            .validate(
                environment.pool(),
                model.erased().inference_execution_identity(),
            )
            .map_err(Error::PrefillControl)?;
        let bytes = OriginalPredictionStartupContext::control_bytes()
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        preparation
            .metadata_funding()
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let prepared: Result<Option<OriginalPredictionLane>, StartupCause> = (|| {
            if !model.erased().has_embedded_prediction() {
                return Ok(None);
            }
            let initialized = model.erased().prefill_roots_runtime()?;
            let mechanisms = model
                .workspace_mechanisms()
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            let mut context = OriginalPredictionStartupContext::new(
                environment,
                initialized,
                mechanisms,
                preparation,
            );
            model
                .erased()
                .prepare_original_prediction(&mut context)
                .ok_or(StartupCause::Backend(Error::PrefillControl(
                    WorkingMemoryError::UnknownBound,
                )))?
                .map(Some)
        })();
        prepared.map_err(|cause| OriginalPredictionStartupContext::failure(preparation, cause))
    }
}

impl MlxModelSession {
    /// Copy the actual idle target before embedded request admission. Existing
    /// text completion installed its source registrations; the shared copy
    /// worker pins those exact backings and rejects ordinary/unregistered arrays.
    /// The result is a one-use target payload, never fabricated role evidence.
    pub(crate) fn prepare_original_prediction_target(
        &self,
        lane: &OriginalPredictionLane,
        preparation: &OriginalSpeculativeSemanticPreparation,
        environment: &OriginalCopyEnvironment<'_>,
        expected_frontier: u64,
    ) -> Result<crate::composition::mlx::replicated_text::OriginalPredictionTarget, Error> {
        use crate::backend::runtime::cache::state::PreparedResidentDecoderCopy;
        use crate::composition::mlx::replicated_text::OriginalPredictionTarget;
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<(
                &Self,
                &OriginalPredictionLane,
                &OriginalSpeculativeSemanticPreparation,
                &OriginalCopyEnvironment<'_>,
                u64,
            )>(),
            size_of::<PreparedResidentDecoderCopy<'_>>(),
            size_of::<OriginalPredictionTarget>(),
            size_of::<Result<OriginalPredictionTarget, StartupCause>>(),
            size_of::<eredu_runtime::replicated_session::ReplicatedTextControlOrigin>(),
            size_of::<
                Result<
                    eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
                    eredu_runtime::replicated_session::PreparedControlBindingError,
                >,
            >(),
            size_of::<Option<u64>>(),
            OriginalPredictionStartupContext::copy_error_control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        ];
        preparation
            .metadata_funding()
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let result: Result<OriginalPredictionTarget, StartupCause> = (|| {
            let model = self
                .original_model_source()
                .map_err(Error::PrefillControl)?;
            preparation
                .validate(
                    environment.pool(),
                    model.erased().inference_execution_identity(),
                )
                .map_err(Error::PrefillControl)?;
            let frontier = model.erased().original_text_frontier().map_err(|cause| {
                Error::Neural(preparation.metadata_funding().metadata_source(cause))
            })?;
            if frontier != Some(expected_frontier) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch).into());
            }
            let (source, origin) = model.erased().prepare_original_prediction_target_source()?;
            lane.prepare_registered_target(source, preparation, &origin, expected_frontier)
        })();
        result.map_err(|cause| OriginalPredictionStartupContext::failure(preparation, cause))
    }
}
