//! MLX erasure of a fully bound portable retained prefill session.

use std::{any::Any, path::Path};

use eredu_architectures::{
    capability::CapabilityEstimate, media_plan::PreparedInputPartPlan,
    prepared_execution::PreparedRetainedTextSession,
};
use eredu_core::{
    cache::{
        PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
        PromptCacheStore,
    },
    capture::CaptureRequestShape,
    execution_control::{ControlSupport, SnapshotEstimate},
    residency::{
        ParameterConversionRetentionReport, ParameterConversionRetentionTrimReport,
        ParameterConversionTrimError,
    },
    BackendFailure, CapabilityError, Observed, SessionCapabilities,
};
use eredu_runtime::{
    prefill::{drain_retained_prefill, PrefillStep, RetainedPrefillAdvance, RetainedPrefillCursor},
    ActivationObserver, CacheResidencyReport, DenseDiskStreamReport, ResidencyReport,
    WeightMaterializationReport,
};
use safemlx::{error::Exception, Array, Stream};

use super::super::{
    ErasedPrefillCursor, ErasedReplicatedTextExecutable, MlxPredictionTargetState,
    MlxStateMechanisms,
};
use crate::{
    backend::{
        error::Error,
        runtime::{cache::state::MlxHybridState, media::input},
        MlxDistributedSession,
    },
    MlxTensor,
};

type SavedState = eredu_runtime::execution_control::SessionStateSnapshot<MlxHybridState>;

pub(crate) fn erase_retained_session<R>(
    session: R,
    communication: Option<MlxDistributedSession>,
    stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    R: PreparedRetainedTextSession<MlxTensor, MlxHybridState> + 'static,
{
    if communication.is_some() != session.publication().is_some() {
        return Err(Error::Parallel(
            "retained session publication and native communication disagree".into(),
        ));
    }
    let selected = session.session_capabilities();
    eredu_core::SessionAdmission::new(selected)
        .validate(SessionCapabilities::new(
            false,
            selected.output_observation(),
            false,
        ))
        .map_err(native)?;
    Ok(Box::new(RetainedExecutable {
        session,
        communication,
        stream: stream.clone(),
    }))
}

struct RetainedExecutable<R> {
    session: R,
    communication: Option<MlxDistributedSession>,
    stream: Stream,
}

fn native(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Other(Box::new(error))
}

fn unavailable(feature: &str) -> Error {
    Error::ArchitectureModel(format!("retained executable does not support {feature}"))
}

impl<R: PreparedRetainedTextSession<MlxTensor, MlxHybridState>> RetainedExecutable<R> {
    fn start(
        &mut self,
        input: Result<input::ModelInput<'_>, Error>,
        max_tokens: usize,
        geometry: Option<CaptureRequestShape>,
        stream: &Stream,
    ) -> Result<R::Cursor, Error> {
        let identity = input
            .as_ref()
            .ok()
            .and_then(|input| input.cache_identity())
            .cloned();
        let prepared = input.and_then(|input| {
            input::validate_composite_input_selection(self.session.processor(), input)?;
            input::prepared_composite_input(input)
        });
        self.session
            .start_prefill(
                prepared.map_err(BackendFailure::from_error),
                &input::MlxTensorInputInspector,
                identity.as_ref(),
                max_tokens,
                geometry,
                stream,
            )
            .map_err(|error| {
                if self.communication.is_some() {
                    native(error)
                } else {
                    Error::before_model_mutation(error)
                }
            })
    }

    fn cursor<'a>(
        &self,
        cursor: &'a mut dyn ErasedPrefillCursor,
    ) -> Result<&'a mut R::Cursor, Error> {
        cursor
            .as_any_mut()
            .downcast_mut()
            .ok_or_else(|| unavailable("foreign retained cursor"))
    }

    fn saved<'a>(&self, saved: &'a dyn Any) -> Result<&'a MlxHybridState, Error> {
        let saved = saved
            .downcast_ref::<SavedState>()
            .ok_or_else(|| unavailable("foreign native snapshot"))?;
        self.session.state().saved_state(saved).map_err(native)
    }
}

impl<R: PreparedRetainedTextSession<MlxTensor, MlxHybridState>> ErasedReplicatedTextExecutable
    for RetainedExecutable<R>
{
    fn realized_session_capabilities(&self) -> SessionCapabilities {
        self.session.session_capabilities()
    }

    fn effective_model_type(&self) -> &str {
        &self
            .session
            .capabilities()
            .capabilities()
            .effective_model_type
    }

    fn capability_estimate(&self) -> &CapabilityEstimate {
        self.session.capabilities()
    }

    fn forecast_state_offset(&self) -> Result<Option<i32>, Exception> {
        Ok(Some(self.session.state().get().offset()))
    }

    fn supports_chunked_prefill(&self) -> bool {
        true
    }
    fn supports_retained_prefill(&self) -> bool {
        true
    }

    fn start_prefill_cursor(
        &mut self,
        input: Result<input::ModelInput<'_>, Error>,
        max_tokens: usize,
        geometry: Option<CaptureRequestShape>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn ErasedPrefillCursor>>, Error> {
        self.start(input, max_tokens, geometry, stream)
            .map(|cursor| Some(Box::new(cursor) as Box<dyn ErasedPrefillCursor>))
    }

    fn advance_prefill_ingress(
        &mut self,
        cursor: &mut dyn ErasedPrefillCursor,
        stream: &Stream,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<(), Error> {
        if cursor.next_step()? != PrefillStep::PrepareInput {
            return Err(unavailable("unexpected prefill ingress step"));
        }
        let cursor = self.cursor(cursor)?;
        match self
            .session
            .advance_prefill(
                cursor,
                stream,
                &mut crate::composition::NeutralActivationObserver::new(observer),
            )
            .map_err(native)?
        {
            Some(RetainedPrefillAdvance::Ingress) => Ok(()),
            _ => Err(unavailable("invalid prefill ingress result")),
        }
    }

    fn advance_prefill_cursor(
        &mut self,
        cursor: &mut dyn ErasedPrefillCursor,
        stream: &Stream,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        if !matches!(cursor.next_step()?, PrefillStep::Decoder(_)) {
            return Err(unavailable("unexpected prefill decoder step"));
        }
        let cursor = self.cursor(cursor)?;
        match self
            .session
            .advance_prefill(
                cursor,
                stream,
                &mut crate::composition::NeutralActivationObserver::new(observer),
            )
            .map_err(native)?
        {
            Some(RetainedPrefillAdvance::Decoder(scores)) => Ok(scores.into_array()),
            _ => Err(unavailable("invalid prefill decoder result")),
        }
    }

    fn prepared_input_token_ids(
        &self,
        input: input::ModelInput<'_>,
        _: &Stream,
    ) -> Result<Vec<u32>, Error> {
        input::validate_composite_input_selection(self.session.processor(), input)?;
        let prepared = input::prepared_composite_input(input)?;
        self.session
            .input_tokens(
                &prepared,
                &input::MlxTensorInputInspector,
                input.cache_identity(),
            )
            .map_err(native)
    }

    fn prepared_input_plans(
        &self,
        input: input::ModelInput<'_>,
    ) -> Result<Vec<PreparedInputPartPlan>, CapabilityError> {
        let invalid = |error: Error| CapabilityError::UnsupportedInput {
            architecture: self.effective_model_type().into(),
            reason: error.to_string(),
        };
        input::validate_composite_input_selection(self.session.processor(), input)
            .map_err(invalid)?;
        let prepared = input::prepared_composite_input(input).map_err(invalid)?;
        self.session
            .input_plans(&prepared, &input::MlxTensorInputInspector)
    }

    fn prefill_result_with_observer(
        &mut self,
        input: Result<input::ModelInput<'_>, Error>,
        mask: Option<&Array>,
        geometry: Option<CaptureRequestShape>,
        stream: &Stream,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        if mask.is_some() {
            return Err(Error::before_model_mutation(unavailable(
                "explicit prefill mask",
            )));
        }
        let mut cursor = self.start(input, usize::MAX, geometry, stream)?;
        drain_retained_prefill(|| {
            let step = RetainedPrefillCursor::next_step(&cursor).map_err(native)?;
            if step == PrefillStep::Complete {
                return Ok(None);
            }
            match (step, self.session.advance_prefill(&mut cursor, stream, &mut crate::composition::NeutralActivationObserver::new(observer)).map_err(native)?) {
                (PrefillStep::PrepareInput, progress @ Some(RetainedPrefillAdvance::Ingress))
                | (PrefillStep::Decoder(_), progress @ Some(RetainedPrefillAdvance::Decoder(_))) => {
                    Ok(progress)
                }
                _ => Err(unavailable("invalid prefill step result")),
            }
        })?
            .map(MlxTensor::into_array)
            .ok_or_else(|| unavailable("empty prefill"))
    }

    fn decode_result_with_observer(
        &mut self,
        tokens: Result<&Array, Error>,
        stream: &Stream,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        self.session
            .decode(
                tokens
                    .map(|tokens| MlxTensor::from_array(tokens.clone()))
                    .map_err(BackendFailure::from_error),
                stream,
                &mut crate::composition::NeutralActivationObserver::new(observer),
            )
            .map(MlxTensor::into_array)
            .map_err(native)
    }

    fn has_partition_control(&self) -> bool {
        self.communication.is_some()
    }
    fn partition_sampling_context(
        &self,
    ) -> Option<(
        &crate::backend::runtime::distributed::Group,
        &eredu_runtime::PartitionCommunicationAuthority,
        &Stream,
        usize,
    )> {
        let (publication, authority) = self.session.publication()?;
        let communication = self.communication.as_ref()?;
        Some((
            communication.selected_group(publication.group)?,
            communication.authority(),
            &self.stream,
            authority.owner_group_rank(),
        ))
    }
    fn partition_public_output(&self) -> bool {
        self.session
            .publication()
            .is_none_or(|(_, authority)| authority.local_public_output())
    }

    fn native_control_support(&self) -> ControlSupport {
        if self
            .session
            .state()
            .get()
            .isolated_snapshot_estimate()
            .is_some()
        {
            ControlSupport::Supported
        } else {
            ControlSupport::Unsupported {
                reason: "decoder state storage is unavailable for an isolated snapshot".into(),
            }
        }
    }

    fn estimate_native_control_state(
        &self,
        saved: Option<&dyn Any>,
    ) -> Result<Option<SnapshotEstimate>, Error> {
        Ok(match saved {
            Some(saved) => self.saved(saved)?.isolated_snapshot_estimate(),
            None => self.session.state().get().isolated_snapshot_estimate(),
        })
    }

    fn capture_native_control_state(&mut self) -> Result<Box<dyn Any>, Error> {
        self.session
            .state()
            .capture_state(|state| state.isolated_snapshot(&self.stream))
            .map(|saved| Box::new(saved) as Box<dyn Any>)
            .map_err(Into::into)
    }

    fn copy_native_control_state(&mut self, saved: &dyn Any) -> Result<Box<dyn Any>, Error> {
        let saved = saved
            .downcast_ref::<SavedState>()
            .ok_or_else(|| unavailable("foreign native snapshot"))?;
        self.session
            .state()
            .copy_state(
                saved,
                |state| state.isolated_snapshot(&self.stream).map_err(Error::from),
                native,
            )
            .map(|saved| Box::new(saved) as Box<dyn Any>)
    }

    fn validate_native_control_state(&self, saved: &dyn Any) -> Result<(), Error> {
        self.saved(saved).map(|_| ())
    }

    fn exchange_native_control_state(&mut self, slot: &mut dyn Any) -> Result<(), Error> {
        let slot = slot
            .downcast_mut::<SavedState>()
            .ok_or_else(|| unavailable("foreign native snapshot"))?;
        self.session
            .state_mut()
            .exchange_state(slot)
            .map_err(native)
    }

    fn estimate_installed_control_growth(&self, additional: u64) -> Result<Option<u64>, Error> {
        Ok(self
            .session
            .state()
            .get()
            .isolated_snapshot_growth(additional))
    }

    fn estimate_native_control_growth(
        &self,
        saved: &dyn Any,
        additional: u64,
    ) -> Result<Option<u64>, Error> {
        Ok(self.saved(saved)?.isolated_snapshot_growth(additional))
    }

    fn exchange_autoregressive_cache(
        &mut self,
        _: &mut MlxPredictionTargetState,
        _: &Stream,
    ) -> Result<(), Error> {
        Err(unavailable("autoregressive prediction exchange"))
    }

    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        self.session.trim_parameter_conversions()
    }
    fn parameter_conversion_retention(
        &self,
    ) -> Result<Observed<Vec<ParameterConversionRetentionReport>>, BackendFailure> {
        Ok(match self.session.residency()? {
            Some(report) => report.offload().parameter_conversion_retention().clone(),
            None => Observed::Unsupported {
                reason: "selected session does not select conversion retention".into(),
            },
        })
    }
    fn component_residency_reports(&self) -> Result<Vec<ResidencyReport>, Error> {
        self.session.component_residency().map_err(native)
    }
    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        self.session.residency().map_err(native)
    }
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.session.dense_stream().map_err(native)
    }
    fn materialization_report(&self) -> Option<&WeightMaterializationReport> {
        None
    }
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        Ok(None)
    }
    fn prompt_cache_model_identity(&self) -> Option<&PromptCacheModelIdentity> {
        None
    }
    fn reset_cache(&mut self) -> Result<(), Exception> {
        *self.session.state_mut().get_mut() =
            self.session.fresh_state().map_err(Exception::from_source)?;
        Ok(())
    }
    fn reset_cache_distributed(&mut self) -> Result<(), Error> {
        self.session.reset(&self.stream).map_err(native)
    }

    fn load_prompt_cache(
        &mut self,
        _: &dyn PromptCacheStore,
        _: &PromptCacheDescriptor,
        _: &[u32],
    ) -> Result<PromptCacheManifest, Error> {
        Err(unavailable("prompt cache import"))
    }
    fn save_prompt_cache(
        &mut self,
        _: &dyn PromptCacheStore,
        _: PromptCacheDescriptor,
        _: &[u32],
        _: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        Err(unavailable("prompt cache export"))
    }
    fn load_prompt_cache_distributed(
        &mut self,
        _: &Path,
        _: &PromptCacheDescriptor,
        _: &[u32],
    ) -> Result<Option<PromptCacheManifest>, Error> {
        Err(unavailable("distributed prompt cache import"))
    }
    fn load_prompt_cache_for_input_distributed(
        &mut self,
        _: &Path,
        _: &PromptCacheDescriptor,
        _: &[u32],
        _: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        Err(unavailable("distributed prompt cache import"))
    }
    fn save_prompt_cache_distributed(
        &mut self,
        _: &Path,
        _: PromptCacheDescriptor,
        _: &[u32],
        _: &PromptCacheOptions,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        Err(unavailable("distributed prompt cache export"))
    }
    fn save_prompt_cache_for_input_distributed(
        &mut self,
        _: &Path,
        _: PromptCacheDescriptor,
        _: &[u32],
        _: &PromptCacheOptions,
        _: &eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        Err(unavailable("distributed prompt cache export"))
    }

    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.session.state().get().residency_report()
    }

    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.session.decoder_residency()
    }
    #[cfg(test)]
    fn state_snapshot(&self) -> super::super::StatePresenceSnapshot {
        self.session.state().get().state_snapshot()
    }
    #[cfg(test)]
    fn fixed_numeric_state_snapshot(
        &self,
    ) -> Result<super::super::FixedNumericStateSnapshot, Exception> {
        self.session.state().get().fixed_numeric_snapshot()
    }
    #[cfg(test)]
    fn checkpoint_restore_probe(
        &mut self,
        _: &Array,
        _: &Stream,
    ) -> Result<super::super::CheckpointRestoreProbe, Error> {
        Err(unavailable("prediction checkpoint probe"))
    }
    #[cfg(test)]
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        if mask.is_some() {
            return Err(unavailable("explicit decode mask"));
        }
        self.decode_result_with_observer(Ok(tokens), stream, observer)
    }
}
