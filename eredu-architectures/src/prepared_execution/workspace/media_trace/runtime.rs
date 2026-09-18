//! The selected runtime supplies mechanisms to one retained-media interval loop.
use super::*;

pub(in crate::prepared_execution::workspace) type Source<A> =
    eredu_runtime::media_prefill::workspace::MediaEquationSource<
        PreparedCompositeArchitecture<A>,
        ResidentState,
    >;

pub(in crate::prepared_execution::workspace) trait Driver<A>
where
    A: CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error> + 'static,
    A::InputPartPlan: 'static,
{
    fn observation_host_peak_bytes(&self, context: &WorkspaceContext) -> Result<u64, Error>;
    fn prepare_source(
        &self,
        plan: A::IngressPlan,
        context: &WorkspaceContext,
    ) -> Result<Source<A>, Error>;
    fn prefill(
        &mut self,
        source: &mut Source<A>,
        span: &eredu_runtime::prefill::PrefillChunk,
        state: &mut ResidentState,
        context: &WorkspaceContext,
        hook: &mut CutHook,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
    ) -> Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>;
    fn decode(
        &mut self,
        input: PreparedCompositeInput<'_, WorkspaceTensor, A::InputPartPlan>,
        state: &mut ResidentState,
        context: &WorkspaceContext,
        demand: eredu_core::OutputDemand,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
    ) -> Result<Option<WorkspaceTensor>, Error>;
    fn finish_logits(
        &mut self,
        scores: Option<WorkspaceTensor>,
        demand: eredu_core::OutputDemand,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        match observer {
            Some(observer) => observed::finish_logits(observer, scores, demand, context),
            None => Ok(scores),
        }
    }
}

pub(super) struct Serial<'a, A>
where
    A: CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error> + 'static,
    A::InputPartPlan: 'static,
{
    pub(super) runtime: EquationRuntime<'a, PreparedCompositeArchitecture<A>>,
    pub(super) provider: super::super::EquationRoutedProvider,
}
impl<A> Driver<A> for Serial<'_, A>
where
    A: CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error>
        + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
        + 'static,
    A::InputPartPlan: 'static,
{
    fn observation_host_peak_bytes(&self, context: &WorkspaceContext) -> Result<u64, Error> {
        self.runtime.observation_host_peak_bytes(context)
    }
    fn prepare_source(
        &self,
        plan: A::IngressPlan,
        context: &WorkspaceContext,
    ) -> Result<Source<A>, Error> {
        Source::<A>::new_with_prepared_graph(
            self.runtime.architecture(),
            plan,
            self.runtime.prepared_graph()?,
            context,
        )
    }
    fn prefill(
        &mut self,
        source: &mut Source<A>,
        span: &eredu_runtime::prefill::PrefillChunk,
        state: &mut ResidentState,
        context: &WorkspaceContext,
        hook: &mut CutHook,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
    ) -> Result<(Option<WorkspaceTensor>, A::ForwardContext), Error> {
        self.runtime.forward_media_routed(
            source,
            span,
            state,
            context,
            hook,
            &mut self.provider,
            observer,
        )
    }
    fn decode(
        &mut self,
        input: PreparedCompositeInput<'_, WorkspaceTensor, A::InputPartPlan>,
        state: &mut ResidentState,
        context: &WorkspaceContext,
        demand: eredu_core::OutputDemand,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        self.runtime
            .forward_routed_with_capture(
                input,
                state,
                context,
                demand,
                observer,
                false,
                eredu_runtime::ExpertPass::Decode,
                &mut self.provider,
            )
            .map(|value| value.0)
    }
}
