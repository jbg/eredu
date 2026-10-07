//! One causal generator invocation over the shared transformer pipeline.
//!
//! The enclosing transaction owns unpublished cache branches and agrees any
//! conditioning failures before entering this component. This adapter owns no
//! scheduler, epoch allocator or cache cloning policy.

use super::{
    tts_transformer::TransformerConfig,
    tts_transformer_layered::{CausalTransformerExecution, TransformerError, TransformerExecution},
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::replicated_execution::ReplicatedCoordinator;

/// Native or portable realization of architecture-selected causal stages.
pub trait CausalTransformerPipelineExecution<B: NeuralBackend> {
    /// Caller-owned unpublished stage state, including its logical frontier.
    type State;
    /// Global transformer equations retained during preparation.
    fn config(&self) -> &TransformerConfig;
    /// Validates exact local ownership and returns the common causal frontier.
    fn causal_position(&self, state: &Self::State) -> Result<i32, TransformerError>;
    /// Executes through shared transport and completion without publishing state.
    fn forward(
        &mut self,
        embedded: &B::Tensor,
        state: &mut Self::State,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DistributedCommitOutcome), BackendFailure>;
}

/// Explicit authority for one causal call in the ordinary TTS generator.
///
/// An admitted invocation consumes its epoch even when execution fails. Replay
/// and retry require fresh authority and an unpublished owner-supplied branch.
pub struct PipelineCausalTransformer<'a, B: NeuralBackend, P, C> {
    pipeline: &'a mut P,
    coordinator: &'a mut C,
    identity: [u64; 3],
    epoch: Option<DistributedCommitEpoch>,
    backend: std::marker::PhantomData<fn() -> B>,
}
impl<'a, B: NeuralBackend, P, C> PipelineCausalTransformer<'a, B, P, C> {
    /// Borrows retained mechanisms and recorded request, turn and invocation identity.
    pub fn new(
        pipeline: &'a mut P,
        coordinator: &'a mut C,
        identity: [u64; 3],
        epoch: DistributedCommitEpoch,
    ) -> Self {
        Self {
            pipeline,
            coordinator,
            identity,
            epoch: Some(epoch),
            backend: std::marker::PhantomData,
        }
    }
    /// Whether this authority has admitted its single causal call.
    pub fn is_consumed(&self) -> bool {
        self.epoch.is_none()
    }
}

impl<B, P, C> TransformerExecution<B> for PipelineCausalTransformer<'_, B, P, C>
where
    B: NeuralBackend,
    P: CausalTransformerPipelineExecution<B>,
{
    fn config(&self) -> &TransformerConfig {
        self.pipeline.config()
    }
    fn forward_encoder(
        &mut self,
        _: &B::Tensor,
        _: &B::Tensor,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        Err(TransformerError::Configuration(
            "causal pipeline authority cannot execute a character encoder",
        ))
    }
}

impl<B, P, C> CausalTransformerExecution<B, P::State> for PipelineCausalTransformer<'_, B, P, C>
where
    B: NeuralBackend,
    P: CausalTransformerPipelineExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    fn causal_position(&self, state: &P::State) -> Result<i32, TransformerError> {
        self.pipeline.causal_position(state)
    }
    fn forward_causal(
        &mut self,
        embedded: &B::Tensor,
        state: &mut P::State,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        let local = (|| {
            if self.epoch.is_none() {
                return Err(TransformerError::Configuration(
                    "causal pipeline invocation has consumed its epoch",
                ));
            }
            if !self.pipeline.config().geometry().causal
                || !matches!(embedded.shape(), [batch, sequence, hidden]
                    if *batch > 0 && *sequence > 0 && *hidden == self.pipeline.config().geometry().hidden_size as i32)
            {
                return Err(TransformerError::Configuration(
                    "causal pipeline requires matching embedded geometry",
                ));
            }
            self.pipeline.causal_position(state)
        })();
        let descriptor = [
            0x5454534341550001,
            self.identity[0],
            self.identity[1],
            self.identity[2],
            self.epoch.map_or(0, DistributedCommitEpoch::value),
            local.as_ref().copied().unwrap_or(-1) as u64,
            embedded.shape().first().copied().unwrap_or(0) as u64,
            embedded.shape().get(1).copied().unwrap_or(0) as u64,
        ];
        if !self.coordinator.agree(descriptor, local.is_ok(), context)? {
            return Err(local.err().unwrap_or(TransformerError::Configuration(
                "causal pipeline participants disagree on invocation or state",
            )));
        }
        local?;
        let epoch = self.epoch.take().expect("agreed unspent causal authority");
        let result = self
            .pipeline
            .forward(embedded, state, descriptor, epoch, context);
        let committed = matches!(&result, Ok((_, DistributedCommitOutcome::Committed(actual))) if *actual == epoch);
        if !self.coordinator.agree(descriptor, committed, context)? {
            return Err(result.err().map(TransformerError::Backend).unwrap_or(
                TransformerError::Configuration("causal pipeline transition did not commit"),
            ));
        }
        let (hidden, outcome) = result?;
        if outcome != DistributedCommitOutcome::Committed(epoch) {
            return Err(TransformerError::Configuration(
                "causal pipeline returned a different completion epoch",
            ));
        }
        Ok(hidden)
    }
}
