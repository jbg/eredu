//! The actual PP or combined TP/PP plan traversed by the ordinary partition executor.
use super::*;
use crate::partitioned_execution::{PartitionTensorAllocator, PipelinePartitionExecutor};
use crate::prepared_execution::workspace::layerwise::QuoteError;
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceFloatingType, WorkspaceParallelContext};
use eredu_runtime::working_memory::{
    WorkspaceCommunicationGroup, WorkspaceCommunicationMetadata, WorkspaceCommunicationRoute,
    workspace_partition_communication,
};
use eredu_runtime::{
    OpaqueBoundaryTransport, OpaqueFailureAgreement, OpaqueOutputPublisher,
    PartitionedTextExecution, PartitionedTextRuntime, ReplicatedTextExecutionStrategy,
};

type Model<C, P> = PartitionedLayeredModel<WorkspaceBackend, C, P>;
type Policy<C, P> = ResidentUnitWindow<Unit<C, P>>;
type Executor<A, Q, U> =
    PipelinePartitionExecutor<A, WorkspaceBackend, ResidentState, Q, Allocator, U>;
type Strategy<A, Q, U> = PartitionedTextExecution<
    Executor<A, Q, U>,
    WorkspaceCommunicationGroup,
    WorkspaceCommunicationRoute,
    WorkspaceCommunicationMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;
type Runtime<A, Q, U> = PartitionedTextRuntime<
    A,
    WorkspaceBackend,
    ResidentState,
    Q,
    Executor<A, Q, U>,
    WorkspaceCommunicationGroup,
    WorkspaceCommunicationRoute,
    WorkspaceCommunicationMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

pub(super) fn pipeline_quote<C, P>(
    source: &Dense<C, P>,
    pipeline: &PipelineSource,
    actual: &PreparedModelSources,
    communication: Option<&eredu_runtime::RetainedCommunicationSource>,
    visitor: EquationVisitor<'_, '_, '_>,
) -> Result<EquationQuote, Error>
where
    C: PartitionedConfig + Send + Sync,
    P: BlockFactory<WorkspaceBackend, C> + 'static,
    P::FeedForward: TensorParallelProjectionOperator<WorkspaceBackend>,
{
    let context = visitor.context;
    context.charge_metadata(std::mem::size_of::<(
        Model<C, P>,
        Option<Model<C, P>>,
        Result<EquationQuote, Error>,
        Option<Box<WorkspaceParallelContext>>,
        Box<WorkspaceParallelContext>,
        Option<WorkspaceParallelContext>,
        eredu_runtime::CommunicationGroupOperation<'_>,
        Result<
            eredu_runtime::CommunicationGroupOperation<'_>,
            eredu_runtime::CommunicationGroupOperationError,
        >,
        eredu_runtime::StateLayout,
        &crate::decoder::PartitionLocalGeometry<C>,
        &eredu_runtime::SelectedStateRealization,
        Result<(), Error>,
        std::ops::Range<usize>,
        Result<eredu_runtime::StateLayout, eredu_runtime::StateError>,
    )>())?;
    if !actual.selected().same_complete_selection(&source.selected)
        || actual.selected().execution().parallel_topology() != Some(source.rank)
        || source.rank.pipeline_parallel_size() <= 1
    {
        return Err(context.metadata_error(format_args!(
            "pipeline quote differs from its retained partition constructor"
        )));
    }
    let communication = communication.ok_or_else(|| {
        context.metadata_error(format_args!(
            "pipeline quote has no retained native communication declaration source"
        ))
    })?;
    if actual.selected().communication_manifest() != Some(communication.manifest()) {
        return Err(context.metadata_error(format_args!(
            "pipeline source differs from the selected manifest"
        )));
    }
    eredu_runtime::working_memory::validate_workspace_state_realization(
        visitor.state,
        &source.local_state,
        context,
    )?;
    let _transform = source
        .transform
        .as_ref()
        .map(|source| Model::<C, P>::from_retained_partition_source(source.clone(), context))
        .transpose()?;
    let architecture =
        Model::<C, P>::from_retained_partition_source(source.target.clone(), context)?;
    // ArchitectureParameters exposes the complete TP-local state so the
    // ordinary partition driver can select global unit ranges. The session's
    // actual state already contains only this rank's retained owned range.
    let local_geometry = architecture.local_geometry();
    let local_state = local_geometry
        .complete_state_layout()
        .slice_with_metadata(local_geometry.owned_units(), context)
        .map_err(|cause| context.metadata_source(cause))?;
    if visitor.state.layout() != &local_state {
        return Err(context.metadata_error(format_args!(
            "pipeline state differs from its retained local constructor"
        )));
    }
    let mut addresses = context.metadata_vec(pipeline.addresses.len())?;
    addresses.extend_from_slice(&pipeline.addresses);
    // These are the exact same global addresses used by ordinary partition
    // construction. The policy stores local slots in this order; it must not
    // construct the architecture's unowned full graph before that conversion.
    match visitor.parameters {
        Some(parameters) => {
            let policy = WorkspaceLayerwisePolicy::for_partition(
                parameters,
                &architecture,
                &addresses,
                context,
            )?;
            // The runtime's descriptive bounded-policy loan shares this exact
            // source. Its independent empty projection never executes an acquire.
            let bounded =
                WorkspaceLayerwisePolicy::for_layout(parameters, parameters.layout(), context)?;
            pipeline_spans::<Model<C, P>, _, _>(
                &source.selected,
                source.rank,
                pipeline,
                architecture,
                policy,
                Some(bounded),
                addresses,
                communication,
                crate::partitioned_execution::OrdinaryPipelinePartitionUnitStrategy,
                visitor,
            )
        }
        None => {
            let policy = Policy::<C, P>::from_workspace_addresses::<Model<C, P>, ResidentState>(
                &architecture,
                &addresses,
                context,
            )?;
            pipeline_spans::<Model<C, P>, _, _>(
                &source.selected,
                source.rank,
                pipeline,
                architecture,
                policy,
                None,
                addresses,
                communication,
                crate::partitioned_execution::OrdinaryPipelinePartitionUnitStrategy,
                visitor,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn pipeline_spans<A, Q, U>(
    selected: &crate::SelectedPreparation,
    rank: eredu_core::ParallelRankTopology,
    pipeline: &PipelineSource,
    mut architecture: A,
    policy: Q,
    bounded: Option<Q>,
    addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
    communication: &eredu_runtime::RetainedCommunicationSource,
    unit_strategy: U,
    visitor: EquationVisitor<'_, '_, '_>,
) -> Result<EquationQuote, Error>
where
    A: crate::partitioned_execution::TextPartitionArchitecture<
            WorkspaceBackend,
            ResidentState,
            Error = Error,
        > + eredu_runtime::ReplicatedTextArchitecture<WorkspaceBackend, ResidentState>
        + 'static,
    Q: eredu_runtime::LayerwisePolicy<WorkspaceBackend, A::Unit>,
    Q::Error: std::error::Error + Send + Sync + 'static,
    U: crate::partitioned_execution::PipelinePartitionUnitStrategy<
            A,
            WorkspaceBackend,
            ResidentState,
        >,
{
    let context = visitor.context;
    context.charge_metadata(std::mem::size_of::<(
        Runtime<A, Q, U>,
        Executor<A, Q, U>,
        Strategy<A, Q, U>,
        A,
        Q,
        Option<Q>,
        Vec<eredu_runtime::ExecutionUnitAddress>,
        Result<EquationQuote, Error>,
        &crate::SelectedPreparation,
        eredu_core::ParallelRankTopology,
        U,
        &PipelineSource,
        &eredu_runtime::RetainedCommunicationSource,
        EquationVisitor<'_, '_, '_>,
    )>())?;
    let target_capture = visitor.target_capture;
    if target_capture {
        architecture.retain_prediction_target_capture();
    }
    // Neural TP and the independent world-control context remain separate.
    // The retained semantic selection, not native world size, chooses this slot.
    let parallel = parallel_context(selected, rank, communication, context)?;
    let executor = Executor::<A, Q, U>::new_with_unit_strategy(
        architecture,
        policy,
        addresses,
        parallel,
        Allocator,
        pipeline.dtype,
        unit_strategy,
    )?
    .with_retained_tensor_collective_waves(pipeline.tensor_waves.as_ref(), context)?;
    let bindings = workspace_partition_communication(communication, context)?;
    let mut runtime = Runtime::<A, Q, U>::with_retained_plan(
        pipeline.plan.clone(),
        executor,
        bindings,
        context.clone(),
        OpaqueBoundaryTransport,
        OpaqueOutputPublisher,
        OpaqueFailureAgreement,
        selected
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
    let output_owner = selected
        .execution()
        .partitioned_output_publication()
        .ok_or_else(|| {
            context.metadata_error(format_args!("partition has no selected observation owner"))
        })?
        .owner_rank;
    context.charge_metadata(
        std::mem::size_of::<Option<eredu_runtime::PreparedLayeredObservationPaths>>()
            .checked_add(2 * std::mem::size_of::<usize>())
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
    )?;
    let paths = visitor
        .observation
        .map(|observation| {
            <Strategy<A, Q, U> as ReplicatedTextExecutionStrategy<
                A,
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
    visitor.quote_spans_with_prepublication_observation(
        hook_bytes,
        |tokens, state, demand, mut observer, span| {
            <Strategy<A, Q, U> as ReplicatedTextExecutionStrategy<
                A,
                WorkspaceBackend,
                ResidentState,
                Q,
                Q,
            >>::with_borrowed_parallel_control_context(
                &mut runtime,
                &mut control,
                &funding,
                |runtime| {
                    let mut strategy = Strategy::<A, Q, U>::new();
                    let pass = match span {
                        InferenceWorkspaceSpan::Sampling(_) => {
                            unreachable!("model equation scheduler emits only prefill/decode spans")
                        }
                        InferenceWorkspaceSpan::Prefill(_) => eredu_runtime::ExpertPass::Prefill,
                        InferenceWorkspaceSpan::Decode { .. } => eredu_runtime::ExpertPass::Decode,
                    };
                    let mut execute = |observer: &mut dyn eredu_runtime::ActivationObserver<
                        eredu_nn::workspace::WorkspaceTensor,
                        Error,
                    >| {
                        match paths.as_ref() {
                            Some(paths) => <Strategy<A, Q, U> as ReplicatedTextExecutionStrategy<
                                A,
                                WorkspaceBackend,
                                ResidentState,
                                Q,
                                Q,
                            >>::forward_with_prepared_observer(
                                &mut strategy,
                                runtime,
                                A::text_input(tokens, None),
                                state,
                                pass,
                                context,
                                observer,
                                paths,
                                demand,
                            ),
                            None => <Strategy<A, Q, U> as ReplicatedTextExecutionStrategy<
                                A,
                                WorkspaceBackend,
                                ResidentState,
                                Q,
                                Q,
                            >>::forward_with_observer(
                                &mut strategy,
                                runtime,
                                A::text_input(tokens, None),
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
                    let capture = prediction_capture::<A, Q, Strategy<A, Q, U>>(
                        runtime,
                        &forward,
                        target_capture,
                        context,
                    )?;
                    let scores = match (scores, observer.as_deref_mut()) {
                        (Some(scores), Some(observer)) => Some(
                            <Strategy<A, Q, U> as ReplicatedTextExecutionStrategy<
                                A,
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
                            <Strategy<A, Q, U> as ReplicatedTextExecutionStrategy<
                                A,
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

/// Reads and publishes the capture retained by the selected ordinary executor.
/// The same helper prices its actual fixed callback/error transports per span.
pub(super) fn prediction_capture<A, Q, D>(
    runtime: &mut D::Runtime,
    forward: &A::ForwardContext,
    enabled: bool,
    context: &WorkspaceContext,
) -> Result<Option<EquationCapture>, Error>
where
    A: eredu_runtime::LayeredArchitecture<WorkspaceBackend, ResidentState, Error = Error>,
    Q: eredu_runtime::LayerwisePolicy<WorkspaceBackend, A::Unit>,
    Q::Error: std::error::Error + Send + Sync + 'static,
    D: ReplicatedTextExecutionStrategy<A, WorkspaceBackend, ResidentState, Q, Q>,
{
    type CaptureError<P> =
        eredu_runtime::ReplicatedTextSessionError<Error, P, std::convert::Infallible>;
    context.charge_metadata(std::mem::size_of::<(
        &mut D::Runtime,
        &A::ForwardContext,
        bool,
        &WorkspaceContext,
        Option<WorkspaceTensor>,
        WorkspaceTensor,
        Option<EquationCapture>,
        Result<Option<WorkspaceTensor>, CaptureError<Q::Error>>,
        Result<WorkspaceTensor, CaptureError<Q::Error>>,
        Result<Option<EquationCapture>, Error>,
    )>())?;
    if !enabled {
        return Ok(None);
    }
    let capture = D::prediction_target_capture(runtime, forward, context)
        .map_err(|cause| context.metadata_source(cause))?
        .ok_or_else(|| {
            context.metadata_error(format_args!(
                "selected partition target forward did not retain its declared prediction capture"
            ))
        })?;
    D::publish_prediction_target_capture(runtime, capture, context)
        .map(|capture| Some(EquationCapture::Embedded(capture)))
        .map_err(|cause| context.metadata_source(cause))
}

pub(super) struct Allocator;
fn floating(
    dtype: eredu_runtime::PipelineActivationDtype,
    context: &WorkspaceContext,
) -> Result<WorkspaceFloatingType, Error> {
    match dtype {
        eredu_runtime::PipelineActivationDtype::Float16 => Ok(WorkspaceFloatingType::Float16),
        eredu_runtime::PipelineActivationDtype::Bfloat16 => Ok(WorkspaceFloatingType::Bfloat16),
        eredu_runtime::PipelineActivationDtype::Float32 => Ok(WorkspaceFloatingType::Float32),
        _ => Err(context.metadata_error(format_args!(
            "unknown selected pipeline floating representation"
        ))),
    }
}
impl PartitionTensorAllocator<WorkspaceBackend> for Allocator {
    fn tensor_to_wire(
        &mut self,
        tensor: WorkspaceTensor,
        logical: eredu_runtime::BoundaryTensorDtype,
        dtype: eredu_runtime::PipelineActivationDtype,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        match logical {
            eredu_runtime::BoundaryTensorDtype::Activation => {
                tensor.cast_floating(floating(dtype, context)?, context)
            }
            eredu_runtime::BoundaryTensorDtype::Uint32
                if tensor.layout().dtype() == WorkspaceDtype::Uint32 =>
            {
                Ok(tensor)
            }
            eredu_runtime::BoundaryTensorDtype::Int32
                if tensor.layout().dtype() == WorkspaceDtype::Int32 =>
            {
                Ok(tensor)
            }
            _ => Err(context
                .metadata_error(format_args!("pipeline logical wire representation differs"))),
        }
    }
    fn tensor_placeholder(
        &mut self,
        shape: &[i32],
        logical: eredu_runtime::BoundaryTensorDtype,
        dtype: eredu_runtime::PipelineActivationDtype,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        // The native allocator uses its typed zero constructor for every
        // receiver placeholder. Describe that same source rather than an
        // unspecified floating/host initialization.
        use eredu_nn::workspace::{WorkspaceLayoutView, WorkspaceRepresentation};
        context.charge_metadata(
            std::mem::size_of::<WorkspaceLayoutView<'_>>()
                + std::mem::size_of::<Option<WorkspaceRepresentation>>()
                + std::mem::size_of::<
                    Result<WorkspaceLayoutView<'_>, eredu_nn::workspace::WorkspaceLayoutError>,
                >(),
        )?;
        let prototype = match logical {
            eredu_runtime::BoundaryTensorDtype::Activation => {
                WorkspaceLayoutView::new(&[], WorkspaceDtype::Float32)?.with_representation(Some(
                    WorkspaceRepresentation::new(floating(dtype, context)?, true),
                ))
            }
            eredu_runtime::BoundaryTensorDtype::Uint32 => {
                WorkspaceLayoutView::new(&[], WorkspaceDtype::Uint32)?
            }
            eredu_runtime::BoundaryTensorDtype::Int32 => {
                WorkspaceLayoutView::new(&[], WorkspaceDtype::Int32)?
            }
            _ => {
                return Err(context.metadata_error(format_args!(
                    "unknown selected pipeline logical wire representation"
                )));
            }
        };

        WorkspaceTensor::zeros_from_prototype(shape, prototype, context)
    }
}

pub(in crate::prepared_execution::workspace) fn parallel_context(
    selected: &crate::SelectedPreparation,
    rank: eredu_core::ParallelRankTopology,
    communication: &eredu_runtime::RetainedCommunicationSource,
    context: &WorkspaceContext,
) -> Result<Option<WorkspaceParallelContext>, Error> {
    context.charge_metadata(std::mem::size_of::<(
        &crate::SelectedPreparation,
        eredu_core::ParallelRankTopology,
        &eredu_runtime::RetainedCommunicationSource,
        Result<Option<WorkspaceParallelContext>, Error>,
        eredu_runtime::CommunicationGroupOperation<'_>,
    )>())?;
    Ok(match selected.execution().partitioned_tensor_group() {
        Some(id) => {
            context.charge_metadata(
                eredu_runtime::CommunicationManifest::group_operation_control_bytes().ok_or_else(
                    || context.metadata_error(format_args!("tensor group metadata overflow")),
                )?,
            )?;
            let group = communication
                .manifest()
                .select_group_operation(id, eredu_runtime::CommunicationOperation::AllReduceSum)
                .map_err(|cause| context.metadata_source(cause))?;
            let descriptor = group.descriptor();
            if descriptor.members().len() != rank.tensor_parallel_size()
                || descriptor.local_index() != Some(rank.tensor_parallel_rank())
            {
                return Err(context.metadata_error(format_args!(
                    "tensor subgroup differs from the retained partition"
                )));
            }
            Some(WorkspaceParallelContext::new(
                rank.tensor_parallel_rank(),
                rank.tensor_parallel_size(),
            )?)
        }
        None if rank.tensor_parallel_size() == 1 => None,
        None => {
            return Err(
                context.metadata_error(format_args!("partition has no selected tensor subgroup"))
            );
        }
    })
}

#[cfg(test)]
mod token_source_tests {
    use super::*;
    use eredu_nn::workspace::{WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound};
    #[derive(Debug)]
    struct Tokens(Option<WorkspaceDtype>);
    impl WorkspaceMechanisms for Tokens {
        fn prepared_text_input_dtype(&self) -> Option<WorkspaceDtype> {
            self.0
        }
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            Ok(None)
        }
    }
    #[test]
    fn prepared_token_source_preserves_exact_pipeline_wire_dtype() {
        let context = WorkspaceContext::new(Tokens(Some(WorkspaceDtype::Uint32)));
        let tokens = super::super::super::prepared_text_tokens(&[1, 3], None, &context).unwrap();
        let before = context.operation_count();
        let wire = Allocator
            .tensor_to_wire(
                tokens,
                eredu_runtime::BoundaryTensorDtype::Uint32,
                eredu_runtime::PipelineActivationDtype::Float32,
                &context,
            )
            .unwrap();
        assert_eq!(wire.layout().dtype(), WorkspaceDtype::Uint32);
        assert_eq!(wire.shape(), [1, 3]);
        assert_eq!(context.operation_count(), before);
        // An explicit signed source remains signed; no allocator conversion can
        // disguise a different contract, even with identical shape/byte size.
        let signed = super::super::super::prepared_text_tokens(
            &[1, 3],
            Some(WorkspaceDtype::Int32),
            &context,
        )
        .unwrap();
        assert!(
            Allocator
                .tensor_to_wire(
                    signed,
                    eredu_runtime::BoundaryTensorDtype::Uint32,
                    eredu_runtime::PipelineActivationDtype::Float32,
                    &context
                )
                .is_err()
        );
        let portable = WorkspaceContext::new(Tokens(None));
        assert_eq!(
            super::super::super::prepared_text_tokens(&[1, 3], None, &portable)
                .unwrap()
                .layout()
                .dtype(),
            WorkspaceDtype::Int32
        );
    }
}
