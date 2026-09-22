//! Retained composite constructor lowered through its ordinary partition executor.
use super::pipeline::{Allocator, parallel_context};
use super::*;
use crate::composite_execution::{
    CompositeMediaIngressArchitecture, PreparedCompositeArchitecture, PreparedCompositeInput,
};
use crate::prepared_execution::workspace::layerwise::QuoteError;
mod cut_observer;
mod media;
use crate::composite_partitioned::PreparedCompositeExecutorPlan;
use eredu_nn::workspace::WorkspaceParallelContext;
use eredu_runtime::working_memory::{
    WorkspaceCommunicationGroup, WorkspaceCommunicationMetadata, WorkspaceCommunicationRoute,
    workspace_partition_communication,
};
use eredu_runtime::{
    OpaqueBoundaryTransport, OpaqueFailureAgreement, OpaqueOutputPublisher,
    PartitionedTextExecution, PartitionedTextRuntime, PreparedInputPart, PreparedInputPayload,
    PreparedModelInput, ReplicatedTextExecutionStrategy,
};

#[derive(Debug)]
pub(crate) enum PreparedCompositeModelSource {
    Inkling(
        crate::inkling::RetainedModelSource,
        Option<crate::inkling::RetainedModelSource>,
        Arc<crate::inkling::LocalGeometry>,
    ),
    Gemma4(
        crate::gemma4::model::RetainedModelSource,
        Option<crate::gemma4::model::RetainedModelSource>,
    ),
    QwenVl(
        crate::qwen::vl::RetainedModelSource,
        Option<crate::qwen::vl::RetainedModelSource>,
    ),
}
struct Composite {
    model: PreparedCompositeModelSource,
    selected: crate::SelectedPreparation,
    rank: eredu_core::ParallelRankTopology,
    state: eredu_runtime::SelectedStateRealization,
    state_offset: usize,
    executor: PreparedCompositeExecutorPlan,
    plan: Arc<eredu_runtime::PartitionedExecutionPlan>,
    addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
}
impl PreparedDirectPartitionSource {
    pub(crate) fn composite(
        model: PreparedCompositeModelSource,
        selected: crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
        state: eredu_runtime::SelectedStateRealization,
        state_offset: usize,
        executor: PreparedCompositeExecutorPlan,
        plan: Arc<eredu_runtime::PartitionedExecutionPlan>,
        addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
    ) -> Self {
        Self(Arc::new(Composite {
            model,
            selected,
            rank,
            state,
            state_offset,
            executor,
            plan,
            addresses,
        }))
    }
}
impl Composite {
    fn validate(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<(), String> {
        if !selected.same_complete_selection(&self.selected) || rank != self.rank {
            return Err("composite workspace source differs from its retained constructor".into());
        }
        Ok(())
    }
}
impl DirectPartitionSource for Composite {
    fn matches_gemma_admission(&self, admission: &crate::gemma4::FamilyConfig) -> bool {
        matches!(&self.model, PreparedCompositeModelSource::Gemma4(target, _) if target.matches_admission(admission))
    }
    fn matches_inkling_admission(&self, admission: &crate::inkling::ModelArgs) -> bool {
        matches!(&self.model, PreparedCompositeModelSource::Inkling(target, _, _) if target.matches_admission(admission))
    }
    fn composite_executor(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<Option<PreparedCompositeExecutorPlan>, String> {
        self.validate(selected, rank)?;
        Ok(Some(self.executor.clone()))
    }
    fn tensor_waves(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<Option<Arc<crate::partitioned_execution::TensorPipelineCollectiveWaves>>, String>
    {
        self.validate(selected, rank)?;
        Ok(None)
    }
    fn quote(
        &self,
        actual: &PreparedModelSources,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        visitor: EquationVisitor<'_, '_, '_>,
    ) -> Result<EquationQuote, Error> {
        let context = visitor.context;
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            &PreparedModelSources,
            EquationVisitor<'_, '_, '_>,
            Result<EquationQuote, Error>,
        )>())?;
        if !actual.selected().same_complete_selection(&self.selected)
            || actual.selected().execution().parallel_topology() != Some(self.rank)
        {
            return Err(context.metadata_error(format_args!(
                "composite quote differs from its retained constructor selection"
            )));
        }
        let communication = communication.ok_or_else(|| {
            context.metadata_error(format_args!(
                "composite quote has no retained communication source"
            ))
        })?;
        if actual.selected().communication_manifest() != Some(communication.manifest()) {
            return Err(context.metadata_error(format_args!(
                "composite communication differs from the retained manifest"
            )));
        }
        eredu_runtime::working_memory::validate_workspace_state_realization(
            visitor.state,
            &self.state,
            context,
        )?;
        match &self.model {
            PreparedCompositeModelSource::Inkling(target, source, geometry) => {
                context.charge_metadata(std::mem::size_of::<(
                    crate::inkling::LayeredModel<WorkspaceBackend>,
                    Option<crate::inkling::LayeredModel<WorkspaceBackend>>,
                )>())?;
                let _source = source
                    .as_ref()
                    .map(|source| {
                        crate::inkling::LayeredModel::<WorkspaceBackend>::new_parallel_with_source(
                            source.clone(),
                            geometry.clone(),
                            context,
                        )
                    })
                    .transpose()?;
                let architecture =
                    crate::inkling::LayeredModel::<WorkspaceBackend>::new_parallel_with_source(
                        target.clone(),
                        geometry.clone(),
                        context,
                    )?;
                quote_model(self, architecture, communication, visitor)
            }
            PreparedCompositeModelSource::Gemma4(target, source) => {
                context.charge_metadata(std::mem::size_of::<(
                    crate::gemma4::LayeredModel<WorkspaceBackend>,
                    Option<crate::gemma4::LayeredModel<WorkspaceBackend>>,
                )>())?;
                let _source = source
                    .as_ref()
                    .map(|source| {
                        crate::gemma4::LayeredModel::<WorkspaceBackend>::new_with_source(
                            source.clone(),
                            context,
                        )
                    })
                    .transpose()?;
                let architecture =
                    crate::gemma4::LayeredModel::<WorkspaceBackend>::new_with_source(
                        target.clone(),
                        context,
                    )?;
                quote_model(self, architecture, communication, visitor)
            }
            PreparedCompositeModelSource::QwenVl(target, source) => {
                context.charge_metadata(std::mem::size_of::<(
                    crate::qwen::vl::LayeredModel<WorkspaceBackend>,
                    Option<crate::qwen::vl::LayeredModel<WorkspaceBackend>>,
                )>())?;
                let _source = source
                    .as_ref()
                    .map(|source| {
                        crate::qwen::vl::LayeredModel::<WorkspaceBackend>::new_with_source(
                            source.clone(),
                            context,
                        )
                    })
                    .transpose()?;
                let architecture =
                    crate::qwen::vl::LayeredModel::<WorkspaceBackend>::new_with_source(
                        target.clone(),
                        context,
                    )?;
                quote_model(self, architecture, communication, visitor)
            }
        }
    }
}

type Model<A> = PreparedCompositeArchitecture<A>;
type Executor<A, Q> = crate::partitioned_execution::CompositePartitionExecutor<
    A,
    WorkspaceBackend,
    ResidentState,
    Q,
    Allocator,
>;
type Strategy<A, Q> = PartitionedTextExecution<
    Executor<A, Q>,
    WorkspaceCommunicationGroup,
    WorkspaceCommunicationRoute,
    WorkspaceCommunicationMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;
type Runtime<A, Q> = PartitionedTextRuntime<
    Model<A>,
    WorkspaceBackend,
    ResidentState,
    Q,
    Executor<A, Q>,
    WorkspaceCommunicationGroup,
    WorkspaceCommunicationRoute,
    WorkspaceCommunicationMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

fn quote_model<A>(
    source: &Composite,
    architecture: A,
    communication: &eredu_runtime::RetainedCommunicationSource,
    visitor: EquationVisitor<'_, '_, '_>,
) -> Result<EquationQuote, Error>
where
    A: CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error>
        + eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend, ResidentState>
        + eredu_runtime::ParallelLayeredArchitecture<WorkspaceBackend, ResidentState>
        + 'static,
    A::InputPartPlan: 'static,
{
    let context = visitor.context;
    context.charge_metadata(std::mem::size_of::<(
        A,
        Model<A>,
        eredu_runtime::StateLayout,
        A::AdmissionConfig,
        Vec<eredu_runtime::ExecutionUnitAddress>,
        Result<EquationQuote, Error>,
    )>())?;
    let admission =
        A::retain_admission_config_with_metadata(&architecture.admission_config(), context)?;
    let architecture = PreparedCompositeArchitecture::new(architecture);
    let layout = architecture.state_layout(Some(context))?;
    let end = source
        .state_offset
        .checked_add(visitor.state.layout().len())
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
    let local = layout
        .slice_with_metadata(source.state_offset..end, context)
        .map_err(|cause| context.metadata_source(cause))?;
    if &local != visitor.state.layout() {
        return Err(context.metadata_error(format_args!(
            "composite state differs from its retained local constructor"
        )));
    }
    let mut addresses = context.metadata_vec(source.addresses.len())?;
    addresses.extend_from_slice(&source.addresses);
    match visitor.parameters {
        Some(parameters) => {
            let policy = WorkspaceLayerwisePolicy::for_partition(
                parameters,
                &architecture,
                &addresses,
                context,
            )?;
            let bounded =
                WorkspaceLayerwisePolicy::for_layout(parameters, parameters.layout(), context)?;
            spans(
                source,
                architecture.into_inner(),
                admission,
                policy,
                Some(bounded),
                communication,
                visitor,
            )
        }
        None => {
            let policy = ResidentUnitWindow::<A::Unit>::from_workspace_addresses::<
                Model<A>,
                ResidentState,
            >(&architecture, &addresses, context)?;
            spans(
                source,
                architecture.into_inner(),
                admission,
                policy,
                None,
                communication,
                visitor,
            )
        }
    }
}

fn spans<A, Q>(
    source: &Composite,
    mut architecture: A,
    admission: A::AdmissionConfig,
    policy: Q,
    bounded: Option<Q>,
    communication: &eredu_runtime::RetainedCommunicationSource,
    visitor: EquationVisitor<'_, '_, '_>,
) -> Result<EquationQuote, Error>
where
    A: CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error>
        + eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend, ResidentState>
        + eredu_runtime::ParallelLayeredArchitecture<WorkspaceBackend, ResidentState>
        + 'static,
    A::InputPartPlan: 'static,
    Q: eredu_runtime::LayerwisePolicy<WorkspaceBackend, A::Unit>,
    Q::Error: std::error::Error + Send + Sync + 'static,
{
    let context = visitor.context;
    context.charge_metadata(std::mem::size_of::<(
        A,
        Q,
        Option<Q>,
        Executor<A, Q>,
        Runtime<A, Q>,
        Strategy<A, Q>,
        A::AdmissionConfig,
        Option<WorkspaceParallelContext>,
        Option<Box<WorkspaceParallelContext>>,
        WorkspaceParallelContext,
        Option<eredu_runtime::PreparedLayeredObservationPaths>,
        Result<EquationQuote, Error>,
        PreparedInputPart<WorkspaceTensor>,
        PreparedModelInput<WorkspaceTensor>,
        crate::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
        PreparedCompositeInput<'_, WorkspaceTensor, A::InputPartPlan>,
        usize,
    )>())?;
    let target_capture = visitor.target_capture;
    if target_capture {
        architecture.retain_prediction_target_capture();
    }
    let parallel = parallel_context(&source.selected, source.rank, communication, context)?;
    let executor = source
        .executor
        .bind_borrowed_direct::<A, WorkspaceBackend, ResidentState, Q, Allocator>(
            architecture,
            policy,
            parallel,
            Allocator,
            context,
        )?;
    let bindings = workspace_partition_communication(communication, context)?;
    let mut runtime = Runtime::<A, Q>::with_retained_plan(
        source.plan.clone(),
        executor,
        bindings,
        context.clone(),
        OpaqueBoundaryTransport,
        OpaqueOutputPublisher,
        OpaqueFailureAgreement,
        source
            .selected
            .text_realization()
            .residency()
            .execution_residency(),
        bounded,
    )
    .map_err(|cause| context.metadata_source(cause))?;
    context.charge_metadata(std::mem::size_of::<WorkspaceParallelContext>())?;
    let publication = WorkspaceParallelContext::new(
        communication.manifest().rank(),
        communication.manifest().world_size(),
    )?;
    let mut control = Some(Box::new(WorkspaceParallelContext::new(
        communication.manifest().rank(),
        communication.manifest().world_size(),
    )?));
    let funding = context
        .metadata_funding()
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)?;
    let output_owner = source
        .selected
        .execution()
        .partitioned_output_publication()
        .ok_or_else(|| {
            context.metadata_error(format_args!(
                "composite selection has no output publication owner"
            ))
        })?
        .owner_rank;
    let paths = visitor
        .observation
        .map(|observation| {
            <Strategy<A, Q> as ReplicatedTextExecutionStrategy<
                Model<A>,
                WorkspaceBackend,
                ResidentState,
                Q,
                Q,
            >>::bind_observation_paths(
                &runtime,
                observation.paths,
                Some(eredu_runtime::layered::LayeredMetadata::new(
                    context,
                    |error| error,
                )),
            )
        })
        .transpose()
        .map_err(|cause| cause.into_quote_error(context))?;
    let hook_bytes = paths
        .as_ref()
        .map(|paths| {
            paths
                .traversal_host_peak_bytes()
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)
        })
        .transpose()?
        .unwrap_or(0);
    if visitor.media.is_some() {
        context.charge_metadata(std::mem::size_of::<media::Partition<A, Q>>())?;
        let mut driver = media::Partition {
            runtime,
            paths,
            publication,
            control,
            output_owner,
            local_rank: communication.manifest().rank(),
            funding,
        };
        let (roots, plan) = visitor.prepare_media_interval_source::<A>()?;
        return visitor.quote_media_intervals::<A, _>(admission, roots, plan, &mut driver);
    }
    visitor.quote_spans_with_prepublication_observation(
        hook_bytes,
        |tokens, state, demand, mut observer, span| {
            <Strategy<A, Q> as ReplicatedTextExecutionStrategy<
                Model<A>,
                WorkspaceBackend,
                ResidentState,
                Q,
                Q,
            >>::with_borrowed_parallel_control_context(
                &mut runtime,
                &mut control,
                &funding,
                |runtime| {
                    let part = PreparedInputPart::new(
                        eredu_core::InputModality::Text,
                        PreparedInputPayload::TokenIds(tokens.clone()),
                        [],
                    )
                    .map_err(|cause| context.metadata_source(cause))?;
                    let mut parts = context.metadata_vec(1)?;
                    parts.push(part);
                    let input = PreparedModelInput::new_with_metadata(parts, context, |tensor| {
                        super::super::composite::TextInspector::identity_with_metadata(
                            tensor, context,
                        )
                    })?;
                    let admitted = A::admit_prepared_input_with_metadata(
                        &admission,
                        &input,
                        &super::super::composite::TextInspector,
                        context,
                    )?;
                    let input = PreparedCompositeInput::new_with_diagnostic(
                        &input,
                        &admitted,
                        |message| context.metadata_error(format_args!("{message}")),
                    )?;
                    let mut strategy = Strategy::<A, Q>::new();
                    let pass = match span {
                        InferenceWorkspaceSpan::Sampling(_) => {
                            unreachable!("model equation scheduler emits only prefill/decode spans")
                        }
                        InferenceWorkspaceSpan::Prefill(_) => eredu_runtime::ExpertPass::Prefill,
                        InferenceWorkspaceSpan::Decode { .. } => eredu_runtime::ExpertPass::Decode,
                    };
                    let mut execute = |observer: &mut dyn eredu_runtime::ActivationObserver<
                        WorkspaceTensor,
                        Error,
                    >| {
                        match paths.as_ref() {
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
                                pass,
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
                                pass,
                                context,
                                observer,
                                demand,
                            ),
                        }
                        .map_err(|cause| context.metadata_source(cause))
                    };
                    let (scores, forward) = match observer.as_deref_mut() {
                        Some(observer) => execute(observer)?,
                        None => execute(&mut eredu_runtime::inspection::NoopObserver)?,
                    };
                    let capture = super::pipeline::prediction_capture::<Model<A>,Q,Strategy<A,Q>>(
                        runtime, &forward, target_capture, context,
                    )?;
                    let scores = match (scores, observer.as_deref_mut()) {
                        (Some(scores), Some(observer)) => Some(
                            <Strategy<A, Q> as ReplicatedTextExecutionStrategy<
                                Model<A>,
                                WorkspaceBackend,
                                ResidentState,
                                Q,
                                Q,
                            >>::observe_output(
                                runtime, &scores, observer, context
                            )
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
                                runtime,
                                scores,
                                context,
                                Some((&publication, &funding)),
                            )
                            .map_err(|cause| context.metadata_source(cause))
                        })
                        .transpose()?;
                    if communication.manifest().rank() != output_owner {
                        if let (Some(scores), Some(observer)) = (&scores, observer) {
                            observer.observe_remote_output(scores, context)?;
                        }
                    }
                    Ok((scores, capture))
                },
            )
            .map_err(|cause| context.metadata_source(cause))?
        },
    )
}
