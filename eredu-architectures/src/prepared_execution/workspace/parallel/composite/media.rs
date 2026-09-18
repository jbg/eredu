//! Media spans use the same retained partition executor and communication source.
use super::super::super::media_trace::{
    CutHook,
    runtime::{Driver, Source},
};
use super::*;
use eredu_runtime::LayeredTraversalHook;

pub(super) struct Partition<A, Q>
where
    A: CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error>
        + eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend, ResidentState>
        + eredu_runtime::ParallelLayeredArchitecture<WorkspaceBackend, ResidentState>
        + 'static,
    A::InputPartPlan: 'static,
    Q: eredu_runtime::LayerwisePolicy<WorkspaceBackend, A::Unit>,
    Q::Error: std::error::Error + Send + Sync + 'static,
{
    pub(super) runtime: Runtime<A, Q>,
    pub(super) paths: Option<eredu_runtime::PreparedLayeredObservationPaths>,
    pub(super) publication: WorkspaceParallelContext,
    pub(super) control: Option<Box<WorkspaceParallelContext>>,
    pub(super) output_owner: usize,
    pub(super) local_rank: usize,
    pub(super) funding: eredu_nn::workspace::HostMetadataFunding,
}
impl<A, Q> Driver<A> for Partition<A, Q>
where
    A: CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error>
        + eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend, ResidentState>
        + eredu_runtime::ParallelLayeredArchitecture<WorkspaceBackend, ResidentState>
        + 'static,
    A::InputPartPlan: 'static,
    Q: eredu_runtime::LayerwisePolicy<WorkspaceBackend, A::Unit>,
    Q::Error: std::error::Error + Send + Sync + 'static,
{
    fn observation_host_peak_bytes(&self, _context: &WorkspaceContext) -> Result<u64, Error> {
        self.paths
            .as_ref()
            .map(|paths| {
                paths
                    .traversal_host_peak_bytes()
                    .ok_or_else(|| eredu_nn::workspace::WorkspaceMetadataError::Overflow.into())
            })
            .transpose()
            .map(|bytes| bytes.unwrap_or(0))
    }
    fn prepare_source(
        &self,
        plan: A::IngressPlan,
        context: &WorkspaceContext,
    ) -> Result<Source<A>, Error> {
        Source::<A>::new_selected::<Q, Q, Strategy<A, Q>>(&self.runtime, plan, context)
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
        context.charge_metadata(std::mem::size_of::<(
            Strategy<A, Q>,
            super::cut_observer::CutObserver<'_>,
            eredu_runtime::inspection::NoopObserver,
            &mut CutHook,
            &WorkspaceContext,
            Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>,
        )>())?;
        let mut noop = eredu_runtime::inspection::NoopObserver;
        let observer: &mut dyn eredu_runtime::ActivationObserver<WorkspaceTensor, Error> =
            match observer {
                Some(observer) => observer,
                None => &mut noop,
            };
        let mut cut = |visit: &mut dyn FnMut(&mut dyn FnMut(&WorkspaceTensor))| {
            <CutHook as LayeredTraversalHook<WorkspaceBackend,A::ForwardContext,Error>>::retained_media_cut(hook,visit,context)
        };
        let mut observer = super::cut_observer::CutObserver {
            observer,
            cut: &mut cut,
        };
        let paths = self.paths.as_ref();
        <Strategy<A, Q> as ReplicatedTextExecutionStrategy<
            Model<A>,
            WorkspaceBackend,
            ResidentState,
            Q,
            Q,
        >>::with_borrowed_parallel_control_context(
            &mut self.runtime,
            &mut self.control,
            &self.funding,
            |runtime| {
                source.forward_selected::<Q, Q, Strategy<A, Q>, _>(
                    &mut Strategy::<A, Q>::new(),
                    runtime,
                    span,
                    state,
                    context,
                    &mut observer,
                    paths,
                )
            },
        )
        .map_err(|cause| context.metadata_source(cause))?
    }
    fn decode(
        &mut self,
        input: PreparedCompositeInput<'_, WorkspaceTensor, A::InputPartPlan>,
        state: &mut ResidentState,
        context: &WorkspaceContext,
        demand: eredu_core::OutputDemand,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        context.charge_metadata(std::mem::size_of::<(
            Strategy<A, Q>,
            eredu_runtime::inspection::NoopObserver,
            PreparedCompositeInput<'_, WorkspaceTensor, A::InputPartPlan>,
            &mut ResidentState,
            &WorkspaceContext,
            eredu_core::OutputDemand,
            Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
            Result<Option<WorkspaceTensor>, Error>,
        )>())?;
        let paths = self.paths.as_ref();
        <Strategy<A, Q> as ReplicatedTextExecutionStrategy<
            Model<A>,
            WorkspaceBackend,
            ResidentState,
            Q,
            Q,
        >>::with_borrowed_parallel_control_context(
            &mut self.runtime,
            &mut self.control,
            &self.funding,
            |runtime| {
                let mut strategy = Strategy::<A, Q>::new();
                let mut noop = eredu_runtime::inspection::NoopObserver;
                let observer: &mut dyn eredu_runtime::ActivationObserver<WorkspaceTensor, Error> =
                    match observer {
                        Some(observer) => observer,
                        None => &mut noop,
                    };
                match paths {
                    Some(paths) => <Strategy<A, Q> as ReplicatedTextExecutionStrategy<
                        Model<A>,
                        WorkspaceBackend,
                        ResidentState,
                        Q,
                        Q,
                    >>::forward_with_prepared_observer(
                        &mut strategy,
                        runtime,
                        input,
                        state,
                        eredu_runtime::ExpertPass::Decode,
                        context,
                        observer,
                        paths,
                        demand,
                    ),
                    None => <Strategy<A, Q> as ReplicatedTextExecutionStrategy<
                        Model<A>,
                        WorkspaceBackend,
                        ResidentState,
                        Q,
                        Q,
                    >>::forward_with_observer(
                        &mut strategy,
                        runtime,
                        input,
                        state,
                        eredu_runtime::ExpertPass::Decode,
                        context,
                        observer,
                        demand,
                    ),
                }
                .map(|value| value.0)
                .map_err(|cause| context.metadata_source(cause))
            },
        )
        .map_err(|cause| context.metadata_source(cause))?
    }
    fn finish_logits(
        &mut self,
        scores: Option<WorkspaceTensor>,
        _demand: eredu_core::OutputDemand,
        mut observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        context.charge_metadata(std::mem::size_of::<(
            Option<WorkspaceTensor>,
            Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
            &WorkspaceContext,
            Result<Option<WorkspaceTensor>, Error>,
        )>())?;
        let scores = match (scores, observer.as_deref_mut()) {
            (Some(scores), Some(observer)) => Some(
                <Strategy<A, Q> as ReplicatedTextExecutionStrategy<
                    Model<A>,
                    WorkspaceBackend,
                    ResidentState,
                    Q,
                    Q,
                >>::observe_output(&mut self.runtime, &scores, observer, context)
                .map_err(|cause| context.metadata_source(cause))?,
            ),
            (scores, _) => scores,
        };
        let scores = scores
            .map(|scores| {
                <Strategy<A, Q> as ReplicatedTextExecutionStrategy<
                    Model<A>,
                    WorkspaceBackend,
                    ResidentState,
                    Q,
                    Q,
                >>::publish_observed_output_with_parallel(
                    &mut self.runtime,
                    scores,
                    context,
                    Some((&self.publication, &self.funding)),
                )
                .map_err(|cause| context.metadata_source(cause))
            })
            .transpose()?;
        if self.local_rank != self.output_owner {
            if let (Some(scores), Some(observer)) = (&scores, observer) {
                observer.observe_remote_output(scores, context)?;
            }
        }
        Ok(scores)
    }
}
