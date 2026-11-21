//! Portable session policy for a bound replicated VL graph.

use eredu_core::input::PreparedInputInspector;
use eredu_core::PreparedModelInput;
use eredu_core::{
    capture::CaptureRequestShape,
    residency::{ParameterConversionRetentionTrimReport, ParameterConversionTrimError},
    BackendFailure, CapabilityError, SessionCapabilities,
};
use eredu_nn::{AttentionCache, DistributedNeuralBackend, Error, GroupedNeuralBackend, Tensor};
use eredu_runtime::{
    execution_control::SessionState, prefill::RetainedPrefillAdvance, ActivationObserver,
    BoundLocalComponent, DenseDiskStreamReport, LayerRuntimeState, LayerWeightResidency,
    PreparedInputCacheIdentity, ResidencyReport, RuntimeStateComponents,
    SelectedProcessorExecution,
};

use super::{
    VisionPromptAdmission, VlInput, VlLayeredModel, VlPrefill, VlPrefillProgress, VlSessionDriver,
};
use crate::{
    capability::CapabilityEstimate, decoder::LayeredInput, media_plan::PreparedInputPartPlan,
    prepared_execution::PreparedRetainedTextSession,
};

/// Builds the typed replicated VL session for the shared executable adapter.
pub(crate) fn local_vl_session<B, S, G>(
    graph: G,
    admission: VisionPromptAdmission,
    capabilities: CapabilityEstimate,
    processor: SelectedProcessorExecution,
    residency: LayerWeightResidency,
) -> Result<impl PreparedRetainedTextSession<B::Tensor, S>, BackendFailure>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + 'static,
    B::Tensor: 'static,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    G: BoundLocalComponent<VlLayeredModel<B>, B, S>,
{
    let state = graph.fresh_state()?;
    Ok(LocalVlSession::<B, _, _> {
        backend: std::marker::PhantomData,
        graph,
        driver: VlSessionDriver::new(admission, processor, state),
        capabilities,
        residency,
    })
}

struct LocalVlSession<B, G, S> {
    backend: std::marker::PhantomData<fn() -> B>,
    graph: G,
    driver: VlSessionDriver<S>,
    capabilities: CapabilityEstimate,
    residency: LayerWeightResidency,
}

impl<B, S, G> PreparedRetainedTextSession<B::Tensor, S> for LocalVlSession<B, G, S>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + 'static,
    B::Tensor: 'static,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    G: BoundLocalComponent<VlLayeredModel<B>, B, S>,
{
    type Cursor = VlPrefill<B::Tensor>;

    fn capabilities(&self) -> &CapabilityEstimate {
        &self.capabilities
    }
    fn session_capabilities(&self) -> SessionCapabilities {
        super::VL_SESSION_CAPABILITIES
    }
    fn state(&self) -> &SessionState<S> {
        self.driver.state()
    }
    fn state_mut(&mut self) -> &mut SessionState<S> {
        self.driver.state_mut()
    }
    fn decoder_residency(&self) -> LayerWeightResidency {
        self.residency
    }
    fn residency(&self) -> Result<Option<ResidencyReport>, BackendFailure> {
        self.graph.residency().map(|(report, _)| Some(report))
    }
    fn dense_stream(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        self.graph.residency().map(|(_, report)| report)
    }
    fn trim_parameter_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        self.graph.trim_conversions()
    }
    fn input_plans(
        &self,
        input: &PreparedModelInput<B::Tensor>,
        inspector: &impl PreparedInputInspector<B::Tensor>,
    ) -> Result<Vec<PreparedInputPartPlan>, CapabilityError> {
        self.driver.input_plans(input, inspector)
    }
    fn input_tokens(
        &self,
        input: &PreparedModelInput<B::Tensor>,
        inspector: &impl PreparedInputInspector<B::Tensor>,
        identity: Option<&PreparedInputCacheIdentity>,
    ) -> Result<Vec<u32>, BackendFailure> {
        self.driver.input_tokens(input, inspector, identity)
    }
    fn start_prefill(
        &mut self,
        input: Result<PreparedModelInput<B::Tensor>, BackendFailure>,
        inspector: &impl PreparedInputInspector<B::Tensor>,
        identity: Option<&PreparedInputCacheIdentity>,
        max_tokens: usize,
        capture: Option<CaptureRequestShape>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Cursor, BackendFailure> {
        let (prompt, chunk) = self
            .driver
            .prepare_prefill(input, inspector, identity, max_tokens, capture)?;
        Ok(VlPrefill::new(prompt, chunk))
    }
    fn advance_prefill(
        &mut self,
        cursor: &mut Self::Cursor,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut dyn ActivationObserver<B::Tensor, Error>,
    ) -> Result<Option<RetainedPrefillAdvance<B::Tensor>>, BackendFailure> {
        let progress = cursor
            .advance_input(context, |input| {
                self.graph
                    .execute(input, self.driver.state_mut().get_mut(), observer)
            })
            .map_err(BackendFailure::from_error)?;
        if let Err(error) = G::settle(cursor.retained_values()) {
            cursor.invalidate();
            return Err(error);
        }
        Ok(progress.map(|step| match step {
            VlPrefillProgress::Media => RetainedPrefillAdvance::Ingress,
            VlPrefillProgress::Scores { scores, .. } => RetainedPrefillAdvance::Decoder(scores),
        }))
    }
    fn decode(
        &mut self,
        tokens: Result<B::Tensor, BackendFailure>,
        _: &<B::Tensor as Tensor>::Context,
        observer: &mut dyn ActivationObserver<B::Tensor, Error>,
    ) -> Result<B::Tensor, BackendFailure> {
        self.graph
            .execute(
                VlInput::Text(LayeredInput {
                    tokens: &tokens?,
                    mask: None,
                }),
                self.driver.state_mut().get_mut(),
                observer,
            )
            .map_err(BackendFailure::from_error)
    }
    fn fresh_state(&self) -> Result<S, BackendFailure> {
        self.graph.fresh_state()
    }
    fn reset(&mut self, _: &<B::Tensor as Tensor>::Context) -> Result<(), BackendFailure> {
        *self.driver.state_mut().get_mut() = self.fresh_state()?;
        Ok(())
    }
}
