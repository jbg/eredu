//! MLX session adaptation of a selected portable VL graph.

use std::{any::Any, num::NonZeroUsize, path::Path, rc::Rc};

use eredu_architectures::{
    capability::CapabilityEstimate,
    decoder::LayeredInput,
    lfm3::vision::{
        PreparedVlGraph, PreparedVlRankGraph, VisionPromptAdmission, VisionPromptPreparation,
        VlInput, VlPrefill, VlPrefillProgress, VL_SESSION_CAPABILITIES,
    },
    lfm3::Lfm3Input,
    media_plan::PreparedInputPartPlan,
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
    BackendFailure, CapabilityError, Observed, PreparedModel, SessionAdmission,
    SessionCapabilities,
};
use eredu_runtime::{
    ActivationObserver, CacheResidencyReport, DenseDiskStreamReport, ResidencyReport,
    SelectedProcessorExecution, WeightMaterializationReport,
};
use safemlx::{error::Exception, Array, Stream};

use crate::{
    backend::{
        error::Error,
        runtime::{cache::state::MlxHybridState, media::input},
        MlxBackend, MlxModel, MlxPreparedTarget,
    },
    composition::{
        lfm3_vision::{
            bind_vl_rank_with_session, settle_vl_prefill, MlxVlDecoder, MlxVlGraph, MlxVlVision,
        },
        NeutralActivationObserver,
    },
    MlxTensor,
};

use super::{
    replicated_text::{
        ErasedPrefillCursor, ErasedReplicatedTextExecutable, MlxPredictionTargetState,
        MlxStateMechanisms,
    },
    Executable,
};

enum VlGraph {
    Replicated(MlxVlGraph),
    Partitioned {
        vision: MlxVlVision,
        decoder: MlxVlDecoder,
    },
}

impl VlGraph {
    fn new_state(&self) -> Result<MlxHybridState, BackendFailure> {
        match self {
            Self::Replicated(graph) => graph.new_state(),
            Self::Partitioned { decoder, .. } => decoder.new_state(),
        }
    }
}

struct VlSession {
    graph: VlGraph,
    state: MlxHybridState,
    admission: VisionPromptAdmission,
    processor: SelectedProcessorExecution,
    capabilities: CapabilityEstimate,
    stream: Stream,
    identity: Rc<()>,
    #[cfg(test)]
    residency: eredu_runtime::LayerWeightResidency,
}

struct SavedState {
    identity: Rc<()>,
    state: MlxHybridState,
}

impl MlxBackend<'_> {
    /// Binds an already prepared VL graph to the ordinary MLX generation session.
    ///
    /// The selected session must request output observation, with prompt-cache
    /// persistence and named activation inspection disabled.
    pub fn bind_prepared_vl(
        &self,
        prepared: PreparedVlGraph,
    ) -> Result<PreparedModel<MlxModel>, BackendFailure> {
        let execution = prepared.selected().execution();
        let session_capabilities = execution.session();
        let state = execution.state();
        let dtype = state.floating_dtype().ok_or_else(|| {
            BackendFailure::from_error(Error::ArchitectureModel(
                "selected VL state omitted its dtype".into(),
            ))
        })?;
        let state_policy = state.policy().clone();
        let (executable, raw_processor) =
            bind_vl_session(prepared, self.stream(), self.weights_stream())
                .map_err(BackendFailure::from_error)?;
        let executable = Executable::new(executable);
        let target =
            MlxPreparedTarget::new(self.stream(), None).map_err(BackendFailure::from_error)?;
        let model = MlxModel::new(executable, dtype.bytes(), state_policy, target);
        #[cfg(any(feature = "image", feature = "audio"))]
        let model = model.with_processor(raw_processor.map(super::ModelProcessor::from_prepared));
        #[cfg(not(any(feature = "image", feature = "audio")))]
        let _ = raw_processor;
        Ok(PreparedModel::new(model, session_capabilities))
    }

    /// Binds the selected VL rank to the same native session as replicated VL.
    pub fn bind_prepared_vl_rank(
        &self,
        prepared: PreparedVlRankGraph,
    ) -> Result<PreparedModel<MlxModel>, BackendFailure> {
        let bind = || -> Result<_, Error> {
            let world = self
                .native_world()
                .ok_or_else(|| unavailable("native rank world"))?;
            let distributed = crate::backend::MlxDistributedSession::from_manifest(
                prepared.weights().communication_manifest(),
                world,
                self.stream(),
            )?;
            let target = MlxPreparedTarget::new(self.stream(), Some(&distributed))?;
            let (session, raw_processor, capabilities, dtype) =
                bind_vl_rank_session(prepared, &distributed, self.stream(), self.weights_stream())?;
            let executable = Executable::new(session);
            let model = MlxModel::new(
                executable,
                dtype,
                eredu_runtime::CacheResidencyPolicy::Device,
                target,
            )
            .with_distributed(distributed);
            #[cfg(any(feature = "image", feature = "audio"))]
            let model =
                model.with_processor(raw_processor.map(super::ModelProcessor::from_prepared));
            #[cfg(not(any(feature = "image", feature = "audio")))]
            let _ = raw_processor;
            Ok(PreparedModel::new(model, capabilities))
        };
        bind().map_err(BackendFailure::from_error)
    }
}

pub(super) fn bind_vl_rank_session(
    prepared: PreparedVlRankGraph,
    distributed: &crate::backend::MlxDistributedSession,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<
    (
        Box<dyn ErasedReplicatedTextExecutable>,
        Option<eredu_architectures::processor_execution::PreparedProcessor>,
        SessionCapabilities,
        std::num::NonZeroU8,
    ),
    Error,
> {
    let (weights, admission, capabilities, processor, raw_processor, session, dtype) =
        prepared.into_parts();
    SessionAdmission::new(session)
        .validate(VL_SESSION_CAPABILITIES)
        .map_err(native)?;
    #[cfg(test)]
    let residency = weights.decoder_residency();
    let (vision, decoder) =
        bind_vl_rank_with_session(weights, distributed.clone(), weights_stream, stream)
            .map_err(native)?;
    let state = decoder.new_state().map_err(native)?;
    let executable = Box::new(VlSession {
        graph: VlGraph::Partitioned { vision, decoder },
        state,
        admission,
        processor,
        capabilities,
        stream: stream.clone(),
        identity: Rc::new(()),
        #[cfg(test)]
        residency,
    });
    Ok((executable, raw_processor, session, dtype))
}

pub(super) fn bind_vl_session(
    prepared: PreparedVlGraph,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<
    (
        Box<dyn ErasedReplicatedTextExecutable>,
        Option<eredu_architectures::processor_execution::PreparedProcessor>,
    ),
    Error,
> {
    let admission = prepared.prompt_admission().map_err(native)?;
    let capabilities = prepared.capabilities().map_err(native)?;
    let execution = prepared.selected().execution();
    let session_capabilities = execution.session();
    if execution.prompt_cache() {
        return Err(unavailable("prompt cache persistence"));
    }
    SessionAdmission::new(session_capabilities)
        .validate(VL_SESSION_CAPABILITIES)
        .map_err(native)?;

    #[cfg(test)]
    let residency = execution.residency();
    let (graph, processor, raw_processor) =
        MlxVlGraph::bind_selected(prepared, weights_stream, stream).map_err(native)?;
    let state = graph.new_state().map_err(native)?;
    let executable = Box::new(VlSession {
        graph: VlGraph::Replicated(graph),
        state,
        admission,
        processor,
        capabilities,
        stream: stream.clone(),
        identity: Rc::new(()),
        #[cfg(test)]
        residency,
    });
    Ok((executable, raw_processor))
}

fn unavailable(feature: &str) -> Error {
    Error::ArchitectureModel(format!(
        "{feature} is unavailable for the selected VL session"
    ))
}

fn native(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Other(Box::new(error))
}

impl VlSession {
    fn prepare(
        &self,
        input: input::ModelInput<'_>,
    ) -> Result<VisionPromptPreparation<MlxTensor>, Error> {
        input::validate_composite_input_selection(&self.processor, input)?;
        let prepared = input::prepared_composite_input(input)?;
        self.admission
            .admit_with_identity(
                &prepared,
                &input::MlxTensorInputInspector,
                input.cache_identity(),
            )
            .map_err(native)
    }

    fn advance(
        &mut self,
        cursor: &mut VlPrefill<MlxTensor>,
        stream: &Stream,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<Option<VlPrefillProgress<MlxTensor>>, Error> {
        match &mut self.graph {
            VlGraph::Replicated(graph) => {
                let mut observer = NeutralActivationObserver::new(observer);
                let progress = cursor
                    .advance_input(stream, |input| {
                        graph.forward_observed(input, &mut self.state, &mut observer)
                    })
                    .map_err(native)?;
                settle_vl_prefill(cursor)?;
                Ok(progress)
            }
            VlGraph::Partitioned { vision, decoder } => decoder
                .advance_prefill_step(vision, cursor, &mut self.state, Ok(()))
                .map_err(native),
        }
    }

    fn cursor<'a>(
        &self,
        cursor: &'a mut dyn ErasedPrefillCursor,
    ) -> Result<&'a mut VlPrefill<MlxTensor>, Error> {
        cursor
            .as_any_mut()
            .downcast_mut()
            .ok_or_else(|| unavailable("foreign retained cursor"))
    }

    fn capture_extent(
        preparation: &VisionPromptPreparation<MlxTensor>,
        geometry: Option<CaptureRequestShape>,
    ) -> Result<(), Error> {
        if let Some(geometry) = geometry {
            geometry
                .validate_prefill(1, preparation.token_ids().len() as u64)
                .map_err(Error::before_model_mutation)?;
        }
        Ok(())
    }

    fn saved<'a>(&self, saved: &'a dyn Any) -> Result<&'a SavedState, Error> {
        saved
            .downcast_ref::<SavedState>()
            .filter(|s| Rc::ptr_eq(&self.identity, &s.identity))
            .ok_or_else(|| unavailable("foreign native snapshot"))
    }
}

impl ErasedReplicatedTextExecutable for VlSession {
    fn realized_session_capabilities(&self) -> SessionCapabilities {
        VL_SESSION_CAPABILITIES
    }

    fn effective_model_type(&self) -> &str {
        &self.capabilities.capabilities().effective_model_type
    }

    fn capability_estimate(&self) -> &CapabilityEstimate {
        &self.capabilities
    }

    fn forecast_state_offset(&self) -> Result<Option<i32>, Exception> {
        Ok(Some(self.state.offset()))
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
        _stream: &Stream,
    ) -> Result<Option<Box<dyn ErasedPrefillCursor>>, Error> {
        let prepared = input
            .and_then(|input| self.prepare(input))
            .and_then(|preparation| {
                Self::capture_extent(&preparation, geometry)?;
                let chunk = NonZeroUsize::new(max_tokens)
                    .ok_or_else(|| unavailable("zero prefill chunk"))?;
                Ok((preparation, chunk))
            });
        let cursor = match &mut self.graph {
            VlGraph::Replicated(_) => {
                let (preparation, chunk) = prepared.map_err(Error::before_model_mutation)?;
                VlPrefill::new(preparation, chunk)
            }
            VlGraph::Partitioned { decoder, .. } => decoder
                .admit_prefill(
                    prepared
                        .map(|prompt| (prompt, ()))
                        .map_err(BackendFailure::from_error),
                )
                .map(|(cursor, ())| cursor)
                .map_err(native)?,
        };
        Ok(Some(Box::new(cursor)))
    }

    fn advance_prefill_ingress(
        &mut self,
        cursor: &mut dyn ErasedPrefillCursor,
        stream: &Stream,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<(), Error> {
        let cursor = self.cursor(cursor)?;
        if cursor.next_step().map_err(native)? != eredu_runtime::prefill::PrefillStep::PrepareInput
        {
            return Err(unavailable("unexpected prefill ingress step"));
        }
        match self.advance(cursor, stream, observer)? {
            Some(VlPrefillProgress::Media) => Ok(()),
            _ => Err(unavailable("invalid prefill ingress result")),
        }
    }

    fn advance_prefill_cursor(
        &mut self,
        cursor: &mut dyn ErasedPrefillCursor,
        stream: &Stream,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        let cursor = self.cursor(cursor)?;
        if !matches!(
            cursor.next_step().map_err(native)?,
            eredu_runtime::prefill::PrefillStep::Decoder(_)
        ) {
            return Err(unavailable("unexpected prefill decoder step"));
        }
        match self.advance(cursor, stream, observer)? {
            Some(VlPrefillProgress::Scores { scores, .. }) => Ok(scores.into_array()),
            _ => Err(unavailable("invalid prefill decoder result")),
        }
    }

    fn prepared_input_token_ids(
        &self,
        input: input::ModelInput<'_>,
        _stream: &Stream,
    ) -> Result<Vec<u32>, Error> {
        self.prepare(input)?
            .token_ids()
            .iter()
            .map(|&id| u32::try_from(id).map_err(native))
            .collect()
    }

    fn prepared_input_plans(
        &self,
        input: input::ModelInput<'_>,
    ) -> Result<Vec<PreparedInputPartPlan>, CapabilityError> {
        let invalid = |error: Error| CapabilityError::UnsupportedInput {
            architecture: self.effective_model_type().into(),
            reason: error.to_string(),
        };
        input::validate_composite_input_selection(&self.processor, input).map_err(invalid)?;
        let prepared = input::prepared_composite_input(input).map_err(invalid)?;
        Ok(self
            .admission
            .admit_composite(&prepared, &input::MlxTensorInputInspector)?
            .parts()
            .iter()
            .cloned()
            .map(Into::into)
            .collect())
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
                "explicit VL prefill mask",
            )));
        }
        let mut cursor = self
            .start_prefill_cursor(input, usize::MAX, geometry, stream)?
            .expect("VL always provides a retained prefill cursor");
        use eredu_runtime::prefill::PrefillStep;
        let mut output = None;
        loop {
            match cursor.next_step()? {
                PrefillStep::PrepareInput => {
                    self.advance_prefill_ingress(cursor.as_mut(), stream, observer)?;
                }
                PrefillStep::Decoder(_) => {
                    output =
                        Some(self.advance_prefill_cursor(cursor.as_mut(), stream, observer)?);
                }
                PrefillStep::Complete => return output.ok_or_else(|| unavailable("empty prefill")),
            }
        }
    }

    fn decode_result_with_observer(
        &mut self,
        tokens: Result<&Array, Error>,
        _stream: &Stream,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        let tokens = tokens.map(|tokens| MlxTensor::from_array(tokens.clone()));
        let input = |tokens| LayeredInput { tokens, mask: None };
        match &mut self.graph {
            VlGraph::Replicated(graph) => {
                let mut observer = NeutralActivationObserver::new(observer);
                let tokens = tokens?;
                graph
                    .forward_observed(
                        VlInput::Text(input(&tokens)),
                        &mut self.state,
                        &mut observer,
                    )
                    .map(MlxTensor::into_array)
                    .map_err(native)
            }
            VlGraph::Partitioned { decoder, .. } => {
                let output = match tokens {
                    Ok(tokens) => decoder.forward(Lfm3Input::Text(input(&tokens)), &mut self.state),
                    Err(error) => decoder
                        .forward_checked(Err(BackendFailure::from_error(error)), &mut self.state),
                };
                output.map(MlxTensor::into_array).map_err(native)
            }
        }
    }

    fn has_partition_control(&self) -> bool {
        matches!(&self.graph, VlGraph::Partitioned { .. })
    }

    fn partition_sampling_context(
        &self,
    ) -> Option<(
        &crate::backend::runtime::distributed::Group,
        &eredu_runtime::PartitionCommunicationAuthority,
        &Stream,
        usize,
    )> {
        match &self.graph {
            VlGraph::Replicated(_) => None,
            VlGraph::Partitioned { decoder, .. } => decoder.partition_sampling_context(),
        }
    }

    fn partition_public_output(&self) -> bool {
        match &self.graph {
            VlGraph::Replicated(_) => true,
            VlGraph::Partitioned { decoder, .. } => decoder.partition_public_output(),
        }
    }

    fn native_control_support(&self) -> ControlSupport {
        if self.state.isolated_snapshot_estimate().is_some() {
            ControlSupport::Supported
        } else {
            ControlSupport::Unsupported {
                reason: "VL decoder state storage is unavailable for an isolated snapshot".into(),
            }
        }
    }

    fn estimate_native_control_state(
        &self,
        saved: Option<&dyn Any>,
    ) -> Result<Option<SnapshotEstimate>, Error> {
        Ok(match saved {
            Some(saved) => self.saved(saved)?.state.isolated_snapshot_estimate(),
            None => self.state.isolated_snapshot_estimate(),
        })
    }

    fn capture_native_control_state(&mut self) -> Result<Box<dyn Any>, Error> {
        Ok(Box::new(SavedState {
            identity: self.identity.clone(),
            state: self.state.isolated_snapshot(&self.stream)?,
        }))
    }

    fn copy_native_control_state(&mut self, saved: &dyn Any) -> Result<Box<dyn Any>, Error> {
        let saved = self.saved(saved)?;
        Ok(Box::new(SavedState {
            identity: self.identity.clone(),
            state: saved.state.isolated_snapshot(&self.stream)?,
        }))
    }

    fn validate_native_control_state(&self, saved: &dyn Any) -> Result<(), Error> {
        self.saved(saved).map(|_| ())
    }

    fn exchange_native_control_state(&mut self, slot: &mut dyn Any) -> Result<(), Error> {
        self.saved(slot)?;
        let slot = slot
            .downcast_mut::<SavedState>()
            .expect("validated snapshot");
        std::mem::swap(&mut self.state, &mut slot.state);
        Ok(())
    }

    fn estimate_installed_control_growth(&self, additional: u64) -> Result<Option<u64>, Error> {
        Ok(self.state.isolated_snapshot_growth(additional))
    }

    fn estimate_native_control_growth(
        &self,
        saved: &dyn Any,
        additional: u64,
    ) -> Result<Option<u64>, Error> {
        Ok(self
            .saved(saved)?
            .state
            .isolated_snapshot_growth(additional))
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
        match &self.graph {
            VlGraph::Replicated(graph) => graph.trim_parameter_conversions(),
            VlGraph::Partitioned { .. } => Err(ParameterConversionTrimError::Unsupported),
        }
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<Observed<Vec<ParameterConversionRetentionReport>>, BackendFailure> {
        match &self.graph {
            VlGraph::Replicated(graph) => graph
                .residency()
                .map(|(report, _)| report.offload().parameter_conversion_retention().clone()),
            VlGraph::Partitioned { .. } => Ok(Observed::Unsupported {
                reason: "VL rank does not select conversion retention".into(),
            }),
        }
    }

    fn component_residency_reports(&self) -> Result<Vec<ResidencyReport>, Error> {
        match &self.graph {
            VlGraph::Replicated(_) => Ok(Vec::new()),
            VlGraph::Partitioned { vision, decoder } => {
                let (vision, _) = vision.residency().map_err(native)?;
                let (decoder, _) = decoder.residency().map_err(native)?;
                Ok(vec![vision, decoder])
            }
        }
    }

    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        match &self.graph {
            VlGraph::Replicated(graph) => graph
                .residency()
                .map(|(report, _)| Some(report))
                .map_err(native),
            VlGraph::Partitioned { .. } => Ok(None),
        }
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.graph {
            VlGraph::Replicated(graph) => {
                graph.residency().map(|(_, report)| report).map_err(native)
            }
            VlGraph::Partitioned { .. } => Ok(None),
        }
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
        self.state = self.graph.new_state().map_err(Exception::from_source)?;
        Ok(())
    }

    fn reset_cache_distributed(&mut self) -> Result<(), Error> {
        match &mut self.graph {
            VlGraph::Replicated(_) => self.reset_cache().map_err(Into::into),
            VlGraph::Partitioned { decoder, .. } => {
                decoder.reset_state(&mut self.state).map_err(native)
            }
        }
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
        self.state.residency_report()
    }

    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.residency
    }
    #[cfg(test)]
    fn state_snapshot(&self) -> super::replicated_text::StatePresenceSnapshot {
        self.state.state_snapshot()
    }
    #[cfg(test)]
    fn fixed_numeric_state_snapshot(
        &self,
    ) -> Result<super::replicated_text::FixedNumericStateSnapshot, Exception> {
        self.state.fixed_numeric_snapshot()
    }
    #[cfg(test)]
    fn checkpoint_restore_probe(
        &mut self,
        _: &Array,
        _: &Stream,
    ) -> Result<super::replicated_text::CheckpointRestoreProbe, Error> {
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
            return Err(unavailable("explicit VL decode mask"));
        }
        self.decode_result_with_observer(Ok(tokens), stream, observer)
    }
}
