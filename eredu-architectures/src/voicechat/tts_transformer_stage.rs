//! Portable execution of a bound pipeline stage.

use super::{
    tts_causal_pipeline::CausalTransformerPipelineExecution,
    tts_transformer_layered::{OwnedTransformerRuntimeState, TransformerInput, TransformerKernel},
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::Tensor;
use eredu_runtime::StateLayout;
use eredu_runtime::{BoundComponentPipeline, ComponentAdapter, ComponentDefinition};
use std::marker::PhantomData;

/// Typed architecture retained by a selected stage.
pub type TtsTransformerPipelineArchitecture<B> =
    ComponentAdapter<ComponentDefinition, TransformerKernel<B>>;

/// One rank of an architecture-selected transformer pipeline.
/// The shared driver owns routing and commit agreement. The enclosing session
/// owns candidate state branches and publication of speech or semantic events.
pub struct TtsTransformerPipeline<P, C> {
    execution: P,
    cache: PhantomData<fn() -> C>,
    state_layout: Option<StateLayout>,
}
impl<P, C> TtsTransformerPipeline<P, C> {
    /// Wraps a bound component and its retained stage facts.
    pub fn new(execution: P, state_layout: Option<StateLayout>) -> Self {
        Self {
            execution,
            state_layout,
            cache: PhantomData,
        }
    }

    /// Borrows the bound execution for residency and disk telemetry.
    pub fn pipeline(&self) -> &P {
        &self.execution
    }

    /// Exact local mutable-state ownership retained during preparation.
    pub fn state_layout(&self) -> Option<&StateLayout> {
        self.state_layout.as_ref()
    }

    /// Executes and completes an unpublished component transition.
    /// All ranks supply the same recorded invocation descriptor and commit epoch;
    /// the enclosing scheduler assigns request, turn and frame identities. A
    /// returned tensor is externally publishable only for `Committed`. On error
    /// the caller discards its candidate state; communication failures and
    /// noncommitted outcomes require recovery before reusing this runtime.
    pub fn forward<B>(
        &mut self,
        input: TransformerInput<'_, B::Tensor>,
        state: &mut OwnedTransformerRuntimeState<B, C>,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DistributedCommitOutcome), BackendFailure>
    where
        B: eredu_nn::DistributedNeuralBackend,
        P: BoundComponentPipeline<
            TtsTransformerPipelineArchitecture<B>,
            B,
            OwnedTransformerRuntimeState<B, C>,
        >,
        C: eredu_runtime::RuntimeLayerState<B> + eredu_nn::AttentionCache<B::Tensor> + 'static,
    {
        let roots = std::iter::once(input.embedded.clone())
            .chain(input.key_mask.cloned())
            .collect();
        self.execution
            .execute(input, roots, state, descriptor, epoch, stream)
    }
}

impl<B, C, P> CausalTransformerPipelineExecution<B> for TtsTransformerPipeline<P, C>
where
    B: eredu_nn::DistributedNeuralBackend,
    P: BoundComponentPipeline<
        TtsTransformerPipelineArchitecture<B>,
        B,
        OwnedTransformerRuntimeState<B, C>,
    >,
    C: eredu_runtime::RuntimeLayerState<B> + eredu_nn::AttentionCache<B::Tensor> + 'static,
{
    type State = OwnedTransformerRuntimeState<B, C>;

    fn config(&self) -> &super::tts_transformer::TransformerConfig {
        self.execution.architecture().kernel().config()
    }

    fn causal_position(
        &self,
        state: &Self::State,
    ) -> Result<i32, super::tts_transformer_layered::TransformerError> {
        self.execution
            .architecture()
            .kernel()
            .validate_causal_state(state)
    }

    fn forward(
        &mut self,
        embedded: &B::Tensor,
        state: &mut Self::State,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DistributedCommitOutcome), BackendFailure> {
        TtsTransformerPipeline::forward(
            self,
            TransformerInput {
                embedded,
                key_mask: None,
            },
            state,
            descriptor,
            epoch,
            context,
        )
    }
}
