//! Parameter work borrows the exact architecture and policy retained by each executor.
use super::*;
use eredu_runtime::parameter_operations::LayeredParameterOwner;

macro_rules! runtime_parameter_workers {
    () => {
        fn visit_loaded_parameters(
            &mut self,
            visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
        ) -> bool {
            self.runtime.visit_loaded_parameters(visitor)
        }
        fn with_parameter_slots(
            &mut self,
            location: &eredu_runtime::parameter_operations::PreparedParameterLocation,
            operation: &mut eredu_runtime::parameter_operations::ParameterSlotOperation<
                '_,
                B::Tensor,
                P::Error,
            >,
            context: &<B::Tensor as eredu_nn::Tensor>::Context,
            preparation: Option<&B::ParameterPreparation<'_>>,
        ) -> Result<bool, eredu_runtime::LayerwiseAcquireError<A::Error, P::Error>> {
            self.runtime
                .with_parameter_slots(location, operation, context, preparation)
        }
        fn visit_parameter_publication(
            &mut self,
            publication: &mut dyn eredu_runtime::parameter_operations::ParameterPublication<
                B::Tensor,
            >,
        ) -> Result<bool, P::Error> {
            self.runtime.visit_parameter_publication(publication)
        }
    };
}

macro_rules! field_parameter_workers {
    () => {
        fn visit_loaded_parameters(
            &mut self,
            visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
        ) -> bool {
            eredu_runtime::parameter_operations::visit_loaded_parameters_in_parts::<
                Self::Architecture,
                B,
                S,
                P,
            >(&mut self.architecture, &mut self.policy, visitor)
        }
        fn with_parameter_slots(
            &mut self,
            location: &eredu_runtime::parameter_operations::PreparedParameterLocation,
            operation: &mut eredu_runtime::parameter_operations::ParameterSlotOperation<
                '_,
                B::Tensor,
                P::Error,
            >,
            context: &<B::Tensor as eredu_nn::Tensor>::Context,
            preparation: Option<&B::ParameterPreparation<'_>>,
        ) -> Result<bool, eredu_runtime::LayerwiseAcquireError<A::Error, P::Error>> {
            eredu_runtime::parameter_operations::with_parameter_slots_in_parts::<
                Self::Architecture,
                B,
                S,
                P,
            >(
                &mut self.architecture,
                &mut self.policy,
                location,
                operation,
                context,
                preparation,
            )
        }
        fn visit_parameter_publication(
            &mut self,
            publication: &mut dyn eredu_runtime::parameter_operations::ParameterPublication<
                B::Tensor,
            >,
        ) -> Result<bool, P::Error> {
            eredu_runtime::parameter_operations::visit_parameter_publication_in_parts::<
                Self::Architecture,
                B,
                S,
                P,
            >(&mut self.architecture, &mut self.policy, publication)
        }
    };
}

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
    fn parameter_parts_ref(&self) -> Option<(&A, &P)> {
        self.runtime.parameter_parts_ref()
    }
    runtime_parameter_workers!();
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
    fn parameter_parts_ref(&self) -> Option<(&A, &P)> {
        self.runtime.parameter_parts_ref()
    }
    runtime_parameter_workers!();
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
        self.observation_binding.invalidate();
        Some((&mut self.architecture, &mut self.policy))
    }
    fn parameter_parts_ref(&self) -> Option<(&A, &P)> {
        Some((&self.architecture, &self.policy))
    }
    field_parameter_workers!();
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
        self.observation_binding.invalidate();
        Some((&mut self.architecture, &mut self.policy))
    }
    fn parameter_parts_ref(&self) -> Option<(&Self::Architecture, &P)> {
        Some((&self.architecture, &self.policy))
    }
    field_parameter_workers!();
}
