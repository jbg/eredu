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
/// Projection-only semantic definition, without recurrent parameter or state ownership.
pub fn rnnt_projection_component_definition(
    dimensions: RnntDimensions,
) -> Result<ComponentDefinition, RnntError> {
    super::rnnt_layered::rnnt_definition_for_selection(
        dimensions,
        super::rnnt_layered::RnntSelection::Projections,
    )
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
impl<B, P> RnntProjectionExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, RnntKernel<B, false>>,
        B,
        StatelessComponentState,
        P,
    >
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> RnntDimensions {
        self.architecture().kernel().dimensions()
    }
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke(
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
        self.invoke(
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
        self.invoke(
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
        self.invoke(
            RnntProjectionInput::Output(input).into_full(),
            &mut StatelessComponentState::new()?,
            context,
        )
        .map_err(runtime_error)
    }
}
/// Tensor-parallel projection phases over the shared component runtime lifecycle.
/// Implementors supply typed invocation and explicit cohort traversal participation.
pub trait ParallelRnntProjectionExecution<B: DistributedNeuralBackend>:
    RnntProjectionExecution<B>
{
    /// Executes a projection using the selected tensor cohort.
    fn invoke_parallel(
        &mut self,
        input: RnntProjectionInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError>;

    /// Runs a projection through shared traversal with cohort failure agreement.
    /// The hook participates before parameter acquisition and collective execution.
    fn invoke_parallel_with_hook<H>(
        &mut self,
        input: RnntProjectionInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> Result<B::Tensor, RnntError>
    where
        H: LayeredTraversalHook<B, RnntForward<B::Tensor>, RnntError> + ?Sized;

    /// Vocabulary-sharded lookup with complete output on every cohort participant.
    fn embed_token_parallel(
        &mut self,
        token: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Embed(token), parallel, context)
    }
    /// Replicated perception projection using the selected cohort's exact authority.
    fn encode_frame_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Encode(input), parallel, context)
    }
    /// Replicated predictor affine over the authoritative published readout.
    fn project_prediction_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Prediction(input), parallel, context)
    }
    /// Vocabulary-sharded affine and exact row-order gather.
    fn project_output_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Output(input), parallel, context)
    }
}

impl<B, P> ParallelRnntProjectionExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, RnntKernel<B, false>>,
        B,
        StatelessComponentState,
        P,
    >
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
        self.forward_parallel_with_context_hook(
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
    fn invoke_parallel_with_hook<H>(
        &mut self,
        input: RnntProjectionInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> Result<B::Tensor, RnntError>
    where
        H: LayeredTraversalHook<B, RnntForward<B::Tensor>, RnntError> + ?Sized,
    {
        self.forward_parallel_with_unit_executor_and_traversal_hook(
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
}
