//! RNNT embedding and joint affines without recurrent parameter ownership.
use super::{
    rnnt::{RnntDimensions, RnntError, RnntProjectionExecution},
    rnnt_layered::{
        rnnt_local_dimensions_for_description, runtime_error, RnntForward, RnntInput, RnntKernel,
        RnntLocalDimensions, RnntUnit,
    },
};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::*;

/// Exact projection-only ownership from the common RNNT semantic declaration.
pub fn rnnt_projection_parameter_description(
    dimensions: RnntDimensions,
) -> Result<ArchitectureParameterDescription, RnntError> {
    super::rnnt_layered::rnnt_projection_description(dimensions)
}
/// Checks vocabulary slices and joint replicas in the compact projection-only topology.
pub fn rnnt_projection_local_dimensions(
    dimensions: RnntDimensions,
    layout: &LocalModelLayout,
) -> Result<RnntLocalDimensions, RnntError> {
    rnnt_local_dimensions_for_description(
        dimensions,
        layout,
        &rnnt_projection_parameter_description(dimensions)?,
    )
}
/// One nonrecurrent phase; no predictor request can enter this graph.
pub enum RnntProjectionInput<'a, T> {
    /// Global vocabulary token, including blank.
    Embed(usize),
    /// Unprojected perception row.
    Encode(&'a T),
    /// Published final predictor readout.
    Prediction(&'a T),
    /// Rectified joint hidden row.
    Output(&'a T),
}
impl<'a, T> RnntProjectionInput<'a, T> {
    fn into_full(self) -> RnntInput<'a, T> {
        match self {
            Self::Embed(token) => RnntInput::Embed(token),
            Self::Encode(input) => RnntInput::Encode(input),
            Self::Prediction(input) => RnntInput::Prediction(input),
            Self::Output(input) => RnntInput::Output(input),
        }
    }
}
/// Ordinary bounded or resident projection phases over the shared unit lifecycle.
pub struct LayerwiseRnntProjection<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
{
    runtime: LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, RnntKernel<B>>,
        B,
        StatelessComponentState,
        P,
    >,
}
impl<B, P> LayerwiseRnntProjection<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
    P::Error: std::fmt::Display,
{
    /// Adopts the compact projection graph and its exact parameter provider.
    pub fn new(
        architecture: ComponentAdapter<ComponentDefinition, RnntKernel<B>>,
        policy: P,
    ) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
        }
    }
    /// Retained shared runtime and cold parameter ownership.
    pub fn runtime(
        &self,
    ) -> &LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, RnntKernel<B>>,
        B,
        StatelessComponentState,
        P,
    > {
        &self.runtime
    }
    /// Shared provider inspection and fault-recovery surface.
    pub fn runtime_mut(
        &mut self,
    ) -> &mut LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, RnntKernel<B>>,
        B,
        StatelessComponentState,
        P,
    > {
        &mut self.runtime
    }
}
impl<B, P> RnntProjectionExecution<B> for LayerwiseRnntProjection<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> RnntDimensions {
        self.runtime.architecture().kernel().dimensions()
    }
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntProjectionInput::Embed(token).into_full(),
                &mut StatelessComponentState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
    fn encode_frame(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntProjectionInput::Encode(input).into_full(),
                &mut StatelessComponentState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
    fn project_prediction(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntProjectionInput::Prediction(input).into_full(),
                &mut StatelessComponentState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
    fn project_output(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntProjectionInput::Output(input).into_full(),
                &mut StatelessComponentState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
}
impl<B, P> LayerwiseRnntProjection<B, P>
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn invoke_parallel(
        &mut self,
        input: RnntProjectionInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .forward_parallel_with_context_hook(
                input.into_full(),
                &mut StatelessComponentState::new()?,
                parallel,
                context,
                |_, _, _| Ok(()),
            )
            .map(|value| value.0)
            .map_err(runtime_error)
    }
    /// Runs a projection through shared traversal with cohort failure agreement.
    /// The hook participates before parameter acquisition and collective execution.
    pub fn invoke_parallel_with_hook<H>(
        &mut self,
        input: RnntProjectionInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> Result<B::Tensor, RnntError>
    where
        H: LayeredTraversalHook<B, RnntForward<B::Tensor>, RnntError> + ?Sized,
    {
        self.runtime
            .forward_parallel_with_unit_executor_and_traversal_hook(
                input.into_full(),
                &mut StatelessComponentState::new()?,
                parallel,
                context,
                |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                    architecture.forward_unit_parallel(
                        group, index, unit, hidden, state, forward, parallel, context,
                    )
                },
                hook,
            )
            .map(|value| value.0)
            .map_err(runtime_error)
    }
    /// Vocabulary-sharded lookup with complete output on every cohort participant.
    pub fn embed_token_parallel(
        &mut self,
        token: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Embed(token), parallel, context)
    }
    /// Replicated perception projection using the selected cohort's exact authority.
    pub fn encode_frame_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Encode(input), parallel, context)
    }
    /// Replicated predictor affine over the authoritative published readout.
    pub fn project_prediction_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Prediction(input), parallel, context)
    }
    /// Vocabulary-sharded affine and exact row-order gather.
    pub fn project_output_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Output(input), parallel, context)
    }
}
