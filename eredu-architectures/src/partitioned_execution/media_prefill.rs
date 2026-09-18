//! Borrowed media control in the existing selected composite pass.
use super::*;
use crate::composite_execution::{
    CompositeArchitecture, CompositeMediaIngressArchitecture, PreparedCompositeArchitecture,
};
use eredu_runtime::media_prefill::MediaInvocation;
use eredu_runtime::LayeredArchitecture;

pub(super) trait CompositeMediaPassControl<A, B, S>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeArchitecture<B, S> + eredu_runtime::PartitionedLayeredArchitecture<B, S>,
{
    fn begin(
        &mut self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        group: usize,
        owns_group_input: bool,
        received: Option<&B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<eredu_runtime::LayeredForwardState<B::Tensor, A::ForwardContext>, eredu_nn::Error>;
    fn enter(
        &mut self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        group: usize,
        initial: &mut B::Tensor,
        forward: &mut A::ForwardContext,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        cut: &mut dyn FnMut(&mut dyn FnMut(&mut dyn FnMut(&B::Tensor))) -> Result<(), eredu_nn::Error>,
    ) -> Result<bool, eredu_nn::Error>;
    fn retain(&mut self, group: usize, value: &B::Tensor);
    fn inactive(&mut self, group: usize);
}

impl<A, B, S> CompositeMediaPassControl<A, B, S>
    for MediaInvocation<'_, PreparedCompositeArchitecture<A>, B, S>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeMediaIngressArchitecture<B, S>
        + eredu_runtime::PartitionedLayeredArchitecture<B, S, Error = eredu_nn::Error>
        + 'static,
    A::InputPartPlan: 'static,
{
    fn begin(
        &mut self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        group: usize,
        owns_group_input: bool,
        received: Option<&B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<eredu_runtime::LayeredForwardState<B::Tensor, A::ForwardContext>, eredu_nn::Error>
    {
        if self.is_initial() && (group == self.primary_group() || !owns_group_input) {
            if let Some(received) = received {
                return self.begin_received(
                    architecture,
                    received,
                    group != self.primary_group(),
                    state,
                    parallel,
                    context,
                );
            }
            // A zero-unit encoder still runs its real begin/complete pair in the
            // existing dependency path; an absent remote boundary is an error.
            if group != self.primary_group()
                || architecture
                    .execution_graph()?
                    .dependencies(group)
                    .into_iter()
                    .flatten()
                    .any(|&dependency| {
                        architecture.inner().should_execute_prepared_group(
                            dependency,
                            A::prepared_ingress_input(self.plan()),
                        ) && architecture
                            .group_unit_count(dependency, None)
                            .is_ok_and(|count| count != 0)
                    })
            {
                return Err(eredu_nn::Error::backend(
                    "media continuation has no actual selected boundary",
                ));
            }
        }
        self.begin_selected(architecture, state, parallel, context)
    }
    fn enter(
        &mut self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        group: usize,
        initial: &mut B::Tensor,
        forward: &mut A::ForwardContext,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        cut: &mut dyn FnMut(&mut dyn FnMut(&mut dyn FnMut(&B::Tensor))) -> Result<(), eredu_nn::Error>,
    ) -> Result<bool, eredu_nn::Error> {
        if !self.owns_decoder_ingress() {
            return Ok(false);
        }
        self.enter_group_with_cut(
            architecture,
            group,
            initial,
            forward,
            state,
            parallel,
            context,
            cut,
        )
    }
    fn retain(&mut self, group: usize, value: &B::Tensor) {
        self.retain_group_result(group, value);
    }
    fn inactive(&mut self, group: usize) {
        self.retain_inactive_group(group);
    }
}

impl<A, B, S, P, F, U, G, R, I>
    eredu_runtime::PartitionedMediaGroupExecutor<PreparedCompositeArchitecture<A>, B, S, G, R, I>
    for CompositePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_runtime::SumReductionBackend
        + eredu_runtime::UnevenGatherBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeMediaIngressArchitecture<B, S>
        + eredu_runtime::PartitionedLayeredArchitecture<B, S, Error = eredu_nn::Error>
        + 'static,
    A::InputPartPlan: 'static,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
    F: PartitionTensorAllocator<B>,
    U: CompositePartitionUnitStrategy<A, B, S>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_runtime::CommunicationTensorMetadata<B>,
    B::ParallelContext: Sized,
{
    fn begin_media<'a, 'source>(
        &mut self,
        invocation: &'a mut MediaInvocation<'source, PreparedCompositeArchitecture<A>, B, S>,
        state: &mut S,
        demand: eredu_core::OutputDemand,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        paths: Option<&'a eredu_runtime::PreparedLayeredObservationPaths>,
    ) -> Result<Self::Pass<'a, 'a>, eredu_nn::Error>
    where
        'source: 'a,
        S: 'a,
    {
        let metadata=B::construction_metadata(context).filter(|source|source.uses_checked_metadata());
        if let Some(metadata)=&metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                Option<&eredu_runtime::PreparedLayeredObservationPaths>,
                Result<(),eredu_runtime::PreparedLayeredObservationError<std::convert::Infallible>>,
            )>())?;
        }
        if let Some(paths)=paths {
            self.observation_binding.validate_binding(paths).map_err(|cause|match &metadata {
                Some(metadata)=>metadata.metadata_source(cause),None=>eredu_nn::Error::backend_retained_source(cause),
            })?;
        }
        match &metadata {
            Some(metadata)=>self.architecture.inner().validate_ingress_plan(invocation.plan(), Some(metadata))?,
            None=>self.architecture.inner().validate_ingress_plan(invocation.plan(), None)?,
        }
        let input = A::prepared_ingress_input(invocation.plan());
        let mut pass = self
            .prepare_composite_pass(
                input,
                state,
                eredu_runtime::ExpertPass::Prefill,
                demand,
                context,
            )?
            .without_input();
        let tensor_partitions = self.parallel.as_ref().map_or(1, B::parallel_size);
        pass.batch_size = i32::try_from(A::ingress_geometry(invocation.plan()).batch_size)
            .map_err(eredu_nn::Error::backend)?;
        pass.sequence_length =
            i32::try_from(invocation.span().input.end - invocation.span().input.start)
                .map_err(eredu_nn::Error::backend)?;
        pass.group_boundary_sequences[self.primary_group] = pass.sequence_length;
        pass.primary_ingress_collectives = match &metadata {
            Some(metadata)=>self.architecture.inner().media_primary_ingress_collectives_with_metadata(
                invocation.plan(),invocation.span(),tensor_partitions,metadata)?,
            None=>self.architecture.inner().media_primary_ingress_collectives(
                invocation.plan(),invocation.span(),tensor_partitions).map_err(eredu_nn::Error::backend)?,
        };
        for group in 0..self.units.len() {
            if let Some(value) = invocation.imported_group(group) {
                pass.group_activity[group] = false;
                pass.group_outputs[group] = value;
                pass.group_collective_waves[group] = None;
            } else {
                pass.group_collective_waves[group] = match &metadata {
                    Some(metadata)=>self.architecture.inner().media_group_collective_waves_with_metadata(
                        invocation.plan(),group,tensor_partitions,self.pipeline_stages,metadata)?,
                    None=>self.architecture.inner().media_group_collective_waves(
                        invocation.plan(),group,tensor_partitions,self.pipeline_stages).map_err(eredu_nn::Error::backend)?,
                };
            }
        }
        pass.paths=paths;
        pass.media = Some(invocation);
        Ok(pass)
    }
}
