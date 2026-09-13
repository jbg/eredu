//! Parameter work borrows the exact architecture and policy retained by each executor.
use super::*;
use eredu_runtime::parameter_operations::LayeredParameterOwner;

impl<A, B, S, P> LayeredParameterOwner<B, S> for DirectPartitionExecutor<A, B, S, P>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    type Architecture = A;
    type Policy = P;
    fn parameter_parts(&mut self) -> Option<(&mut A, &mut P)> {
        self.runtime.parameter_parts()
    }
}

impl<A, B, S, P, Provider, Movement> LayeredParameterOwner<B, S>
    for RoutedPartitionExecutor<A, B, S, P, Provider, Movement>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    type Architecture = A;
    type Policy = P;
    fn parameter_parts(&mut self) -> Option<(&mut A, &mut P)> {
        self.runtime.parameter_parts()
    }
}

impl<A, B, S, P, F, U> LayeredParameterOwner<B, S> for PipelinePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    type Architecture = A;
    type Policy = P;
    fn parameter_parts(&mut self) -> Option<(&mut A, &mut P)> {
        Some((&mut self.architecture, &mut self.policy))
    }
}

impl<A, B, S, P, F, U> LayeredParameterOwner<B, S> for CompositePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    type Architecture = crate::composite_execution::PreparedCompositeArchitecture<A>;
    type Policy = P;
    fn parameter_parts(&mut self) -> Option<(&mut Self::Architecture, &mut P)> {
        Some((&mut self.architecture, &mut self.policy))
    }
}
