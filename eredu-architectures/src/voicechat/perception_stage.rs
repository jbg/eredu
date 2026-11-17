//! Portable execution of a bound pipeline stage.

use super::{
    perception::{PerceptionDimensions, PerceptionError},
    perception_layered::{
        unpack_perception_pipeline_output, PerceptionInput, PerceptionKernel,
        PerceptionRuntimeState,
    },
    perception_rnnt_pipeline::PerceptionPipelineExecution,
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::StateLayout;
use eredu_runtime::{BoundComponentPipeline, ComponentAdapter, ComponentDefinition};

/// Typed architecture retained by a selected stage.
pub type PerceptionPipelineArchitecture<B> =
    ComponentAdapter<ComponentDefinition, PerceptionKernel<B>>;

/// One selected perception stage, with the same shared pipeline driver
/// used by transformer stages. Architecture code owns the encoder/projection
/// boundary and the packed publication representation.
pub struct PerceptionPipeline<B: NeuralBackend, P> {
    execution: P,
    dimensions: PerceptionDimensions,
    state_layout: Option<StateLayout>,
    initial: PerceptionRuntimeState<B::Tensor>,
}

impl<B, P> PerceptionPipelineExecution<B> for PerceptionPipeline<B, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    P: BoundComponentPipeline<
        PerceptionPipelineArchitecture<B>,
        B,
        PerceptionRuntimeState<B::Tensor>,
    >,
{
    type State = PerceptionRuntimeState<B::Tensor>;

    fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
    }

    fn validate_parent(&self, parent: &Self::State, batch: i32) -> Result<(), PerceptionError> {
        self.execution
            .architecture()
            .kernel()
            .validate_parent_for_batch(parent, batch)
    }

    fn forward(
        &mut self,
        input: PerceptionInput<'_, B::Tensor>,
        parent: &Self::State,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor, Self::State, DistributedCommitOutcome), PerceptionError>
    {
        let mut candidate = parent.clone();
        let (encoder, projected, outcome) =
            self.forward(input, &mut candidate, descriptor, epoch, stream)?;
        Ok((encoder, projected, candidate, outcome))
    }
}

impl<B, P> PerceptionPipeline<B, P>
where
    B: NeuralBackend,
{
    /// Wraps a bound component and its retained stage facts.
    pub fn new(
        execution: P,
        dimensions: PerceptionDimensions,
        state_layout: Option<StateLayout>,
        initial: PerceptionRuntimeState<B::Tensor>,
    ) -> Self {
        Self {
            execution,
            dimensions,
            state_layout,
            initial,
        }
    }

    /// Borrows the bound execution for residency and disk telemetry.
    pub fn pipeline(&self) -> &P {
        &self.execution
    }

    /// Exact raw-encoder and language-projection geometry of the selected stage.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
    }

    /// Exact local mutable-state ownership, absent on stateless component stages.
    pub fn state_layout(&self) -> Option<&StateLayout> {
        self.state_layout.as_ref()
    }

    /// Creates an empty unpublished branch with the same selected local ownership.
    pub fn initial_state(&self) -> PerceptionRuntimeState<B::Tensor> {
        self.initial.clone()
    }

    /// Executes a caller-owned candidate and returns completed unprojected
    /// encoder features, projected features and the shared commit decision.
    /// Every rank supplies the same recorded descriptor and epoch. The enclosing
    /// scheduler publishes outputs and retains candidate state only on `Committed`;
    /// a rejected candidate must be discarded before retrying.
    pub fn forward(
        &mut self,
        input: PerceptionInput<'_, B::Tensor>,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor, DistributedCommitOutcome), BackendFailure>
    where
        B: eredu_nn::DistributedNeuralBackend,
        P: BoundComponentPipeline<
            PerceptionPipelineArchitecture<B>,
            B,
            PerceptionRuntimeState<B::Tensor>,
        >,
    {
        let dimensions = self.dimensions;
        let roots = vec![input.mel.clone()];
        let ((encoder, projected), outcome) =
            self.execution
                .invoke(input, roots, state, descriptor, epoch, stream, |packed| {
                    let (encoder, projected) =
                        unpack_perception_pipeline_output(dimensions, &packed, stream)
                            .map_err(BackendFailure::from_error)?;
                    let roots = vec![encoder.clone(), projected.clone()];
                    Ok(((encoder, projected), roots))
                })?;
        Ok((encoder, projected, outcome))
    }
}
