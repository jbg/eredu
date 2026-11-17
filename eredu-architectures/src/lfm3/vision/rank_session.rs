//! Portable execution of a bound vision and decoder rank.

use eredu_core::{
    capture::CaptureRequestShape, BackendFailure, CapabilityError, DistributedCommitEpoch,
};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, Tensor, TensorParallelGroupedNeuralBackend,
};
use eredu_runtime::{
    execution_control::SessionState, prefill::RetainedPrefillAdvance, BoundComponentPipeline,
    DecoderPipelineState, LayerwisePolicy, PreparedInputCacheIdentity, PreparedInputInspector,
    PreparedModelInput, RuntimeStateComponents, UnitPolicyResidency,
};

use super::{
    BoundVlRankSession, PolicyVision, PreparedVlDecoderStage, VisionModule, VlPrefill,
    VlPrefillProgress, VlSessionDriver,
};
use crate::{
    decoder::LayeredInput,
    lfm3::{Lfm3Input, PartitionedLayeredModel},
    prepared_execution::PreparedRetainedTextSession,
};

/// Architecture-owned VL rank assembled from portable component mechanisms.
pub(crate) struct VlRankSession<B: eredu_nn::NeuralBackend, P, D, S> {
    vision: PolicyVision<B, P>,
    decoder: PreparedVlDecoderStage<D>,
    driver: VlSessionDriver<S>,
    processor: eredu_runtime::SelectedProcessorExecution,
    capabilities: crate::capability::CapabilityEstimate,
    session: eredu_core::SessionCapabilities,
    residency: eredu_runtime::LayerWeightResidency,
    epoch: DistributedCommitEpoch,
}

impl<B: eredu_nn::NeuralBackend, P, D, S> VlRankSession<B, P, D, S> {
    pub(crate) fn new(
        bound: BoundVlRankSession<PolicyVision<B, P>, PreparedVlDecoderStage<D>, S>,
    ) -> Self {
        Self {
            vision: bound.vision,
            decoder: bound.decoder,
            driver: VlSessionDriver::new(bound.admission, bound.state),
            processor: bound.processor,
            capabilities: bound.capabilities,
            session: bound.session,
            residency: bound.decoder_residency,
            epoch: DistributedCommitEpoch::FIRST,
        }
    }
}

impl<B, P, D, S> PreparedRetainedTextSession<B::Tensor, S> for VlRankSession<B, P, D, S>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend + 'static,
    B::Tensor: 'static,
    S: DecoderPipelineState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    P: LayerwisePolicy<B, VisionModule<B>, Error: std::error::Error + Send + Sync + 'static>
        + UnitPolicyResidency,
    D: BoundComponentPipeline<PartitionedLayeredModel<B>, B, S>,
{
    type Cursor = VlPrefill<B::Tensor>;

    fn processor(&self) -> &eredu_runtime::SelectedProcessorExecution {
        &self.processor
    }
    fn capabilities(&self) -> &crate::capability::CapabilityEstimate {
        &self.capabilities
    }
    fn session_capabilities(&self) -> eredu_core::SessionCapabilities {
        self.session
    }
    fn state(&self) -> &SessionState<S> {
        self.driver.state()
    }
    fn state_mut(&mut self) -> &mut SessionState<S> {
        self.driver.state_mut()
    }
    fn publication(
        &self,
    ) -> Option<(
        eredu_runtime::PartitionOutputPublication,
        eredu_runtime::PartitionOutputAuthority,
    )> {
        Some((self.decoder.publication, self.decoder.output))
    }
    fn component_residency(&self) -> Result<Vec<eredu_runtime::ResidencyReport>, BackendFailure> {
        Ok(vec![
            self.vision.policy().residency()?,
            self.decoder.pipeline.residency()?,
        ])
    }
    fn decoder_residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.residency
    }
    fn input_plans(
        &self,
        input: &PreparedModelInput<B::Tensor>,
        inspector: &impl PreparedInputInspector<B::Tensor>,
    ) -> Result<Vec<crate::media_plan::PreparedInputPartPlan>, CapabilityError> {
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
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Cursor, BackendFailure> {
        super::admit_partitioned_vl_prefill(
            self.driver
                .prepare_prefill(input, inspector, identity, max_tokens, capture)
                .map(|request| (request, ())),
            Some(self.decoder.prefix),
            |descriptor, ready| self.decoder.pipeline.agree(descriptor, ready, context),
        )
        .map(|(cursor, ())| cursor)
    }
    fn advance_prefill(
        &mut self,
        cursor: &mut Self::Cursor,
        context: &<B::Tensor as Tensor>::Context,
        _: &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>,
    ) -> Result<Option<RetainedPrefillAdvance<B::Tensor>>, BackendFailure> {
        let vision = &mut self.vision;
        let state = self.driver.state_mut().get_mut();
        let epoch = &mut self.epoch;
        let prefix = self.decoder.prefix;
        super::prefill::advance_partitioned_vl_prefill(
            cursor,
            context,
            vision,
            &mut self.decoder.pipeline,
            true,
            |pipeline, descriptor, ready| pipeline.agree(descriptor, ready, context),
            |pipeline, prompt| {
                execute::<B, S, D>(
                    pipeline,
                    Ok(Lfm3Input::Vision(prompt)),
                    state,
                    epoch,
                    prefix,
                    context,
                )
            },
            |cursor| D::settle_values(cursor.retained_values()),
        )
        .map(|progress| {
            progress.map(|step| match step {
                VlPrefillProgress::Media => RetainedPrefillAdvance::Ingress,
                VlPrefillProgress::Scores { scores, .. } => RetainedPrefillAdvance::Decoder(scores),
            })
        })
    }
    fn decode(
        &mut self,
        tokens: Result<B::Tensor, BackendFailure>,
        context: &<B::Tensor as Tensor>::Context,
        _: &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>,
    ) -> Result<B::Tensor, BackendFailure> {
        let input = match tokens {
            Ok(ref tokens) => Ok(Lfm3Input::Text(LayeredInput { tokens, mask: None })),
            Err(error) => Err(error),
        };
        execute::<B, S, D>(
            &mut self.decoder.pipeline,
            input,
            self.driver.state_mut().get_mut(),
            &mut self.epoch,
            self.decoder.prefix,
            context,
        )
    }
    fn fresh_state(&self) -> Result<S, BackendFailure> {
        S::allocate_pipeline_state(
            self.decoder.state_layout.clone(),
            self.decoder.global_layer_start,
        )
    }
    fn reset(&mut self, context: &<B::Tensor as Tensor>::Context) -> Result<(), BackendFailure> {
        let replacement = self.fresh_state();
        super::reset_partitioned_vl_state(
            self.driver.state_mut().get_mut(),
            replacement,
            |descriptor, ready| self.decoder.pipeline.agree(descriptor, ready, context),
        )
    }
}

pub(super) fn execute<B, S, D>(
    pipeline: &mut D,
    input: Result<Lfm3Input<'_, B::Tensor>, BackendFailure>,
    state: &mut S,
    epoch: &mut DistributedCommitEpoch,
    prefix: std::num::NonZeroUsize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, BackendFailure>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend + 'static,
    S: DecoderPipelineState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    D: BoundComponentPipeline<PartitionedLayeredModel<B>, B, S>,
{
    super::run_partitioned_vl_decoder(
        input,
        state.pipeline_offset(),
        state,
        epoch,
        prefix,
        context,
        pipeline,
        |state| state.fork_pipeline_state(context),
        |pipeline, descriptor, ready| pipeline.agree(descriptor, ready, context),
        |pipeline, input, state, descriptor, epoch| {
            let roots = input.vl_pipeline_roots().cloned().collect();
            pipeline.execute(input, roots, state, descriptor, epoch, context)
        },
    )
}
