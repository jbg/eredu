//! Exact initial partition construction reused by the ordinary parallel quote.
use super::*;
use crate::decoder::{
    BlockFactory, PartitionModelSource, PartitionedConfig, PartitionedLayeredModel,
    TensorParallelProjectionOperator,
};
use crate::prepared_execution::workspace::layerwise::QuoteError;
use eredu_runtime::{
    ArchitectureParameters, LayeredArchitecture, LayerwiseRuntime, ResidentUnitWindow,
};
use std::sync::Arc;
mod family;
mod pipeline;
mod routed;
pub(crate) use family::PreparedFamilyPartitionModelSource;
mod composite;
use super::layerwise::WorkspaceLayerwisePolicy;
pub(crate) use composite::PreparedCompositeModelSource;
pub(super) use pipeline::parallel_context;
use pipeline::pipeline_quote;
// Bind the policy to the concrete shared decoder unit, avoiding a recursive
// normalization obligation through LayeredArchitecture in generic helpers.
type Unit<C, P> = crate::decoder::TransformerBlock<
    WorkspaceBackend,
    <P as BlockFactory<WorkspaceBackend, C>>::FeedForward,
>;

// The source is installed only by a successful architecture-owned partition
// constructor. It contains immutable neutral module semantics, never a native
// group, a request, a Context, source pins, or an admission grant.
#[derive(Clone)]
pub(crate) struct PreparedDirectPartitionSource(Arc<dyn DirectPartitionSource + Send + Sync>);
trait DirectPartitionSource {
    fn matches_gemma_admission(&self, _: &crate::gemma4::FamilyConfig) -> bool {
        false
    }
    fn matches_inkling_admission(&self, _: &crate::inkling::ModelArgs) -> bool {
        false
    }
    fn routed_addressable_source(
        &self,
        _selected: &crate::SelectedPreparation,
        _rank: eredu_core::ParallelRankTopology,
    ) -> Result<
        Option<(
            crate::routed_text::RetainedRoutedBanks,
            crate::partitioned_execution::PreparedRoutedExecutionHandoff,
        )>,
        String,
    > {
        Ok(None)
    }
    fn composite_executor(
        &self,
        _selected: &crate::SelectedPreparation,
        _rank: eredu_core::ParallelRankTopology,
    ) -> Result<Option<crate::composite_partitioned::PreparedCompositeExecutorPlan>, String> {
        Ok(None)
    }
    fn routed_resident_source(
        &self,
        _selected: &crate::SelectedPreparation,
        _rank: eredu_core::ParallelRankTopology,
    ) -> Result<Option<crate::routed_text::RetainedPartitionResidentSource>, String> {
        Ok(None)
    }
    fn tensor_waves(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<Option<Arc<crate::partitioned_execution::TensorPipelineCollectiveWaves>>, String>;
    fn quote(
        &self,
        source: &PreparedModelSources,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        visitor: EquationVisitor<'_, '_, '_>,
    ) -> Result<EquationQuote, Error>;
}
struct Dense<C, P> {
    target: Arc<PartitionModelSource<C>>,
    transform: Option<Arc<PartitionModelSource<C>>>,
    selected: crate::SelectedPreparation,
    rank: eredu_core::ParallelRankTopology,
    // The initial constructor derives this once from its validated local
    // partition. Quotes borrow it; they never clone or rebase a global layout.
    local_state: eredu_runtime::SelectedStateRealization,
    pipeline: Option<PipelineSource>,
    marker: std::marker::PhantomData<fn() -> P>,
}
struct PipelineSource {
    plan: Arc<eredu_runtime::PartitionedExecutionPlan>,
    addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
    dtype: eredu_runtime::PipelineActivationDtype,
    tensor_waves: Option<Arc<crate::partitioned_execution::TensorPipelineCollectiveWaves>>,
}
impl PreparedDirectPartitionSource {
    pub(crate) fn matches_gemma_admission(&self, admission: &crate::gemma4::FamilyConfig) -> bool {
        self.0.matches_gemma_admission(admission)
    }
    pub(crate) fn matches_inkling_admission(&self, admission: &crate::inkling::ModelArgs) -> bool {
        self.0.matches_inkling_admission(admission)
    }
    pub(crate) fn routed_addressable_source(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<
        (
            crate::routed_text::RetainedRoutedBanks,
            crate::partitioned_execution::PreparedRoutedExecutionHandoff,
        ),
        String,
    > {
        self.0
            .routed_addressable_source(selected, rank)?
            .ok_or_else(|| {
                "retained partition source is not the selected addressable constructor".into()
            })
    }
    pub(super) fn quote_media(
        &self,
        source: &PreparedModelSources,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        visitor: EquationVisitor<'_, '_, '_>,
    ) -> Result<EquationQuote, Error> {
        self.0.quote(source, communication, visitor)
    }
    pub(crate) fn composite_executor(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<crate::composite_partitioned::PreparedCompositeExecutorPlan, String> {
        self.0.composite_executor(selected, rank)?.ok_or_else(|| {
            "retained partition source is not the selected composite constructor".into()
        })
    }
    pub(crate) fn routed_resident_source(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<crate::routed_text::RetainedPartitionResidentSource, String> {
        self.0
            .routed_resident_source(selected, rank)?
            .ok_or_else(|| {
                "retained partition source is not the selected resident routed constructor".into()
            })
    }
    pub(crate) fn tensor_waves(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<Option<Arc<crate::partitioned_execution::TensorPipelineCollectiveWaves>>, String>
    {
        self.0.tensor_waves(selected, rank)
    }
    pub(crate) fn dense<C, P>(
        target: Arc<PartitionModelSource<C>>,
        transform: Option<Arc<PartitionModelSource<C>>>,
        selected: crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
        local_state: eredu_runtime::SelectedStateRealization,
        pipeline: Option<(
            Arc<eredu_runtime::PartitionedExecutionPlan>,
            Vec<eredu_runtime::ExecutionUnitAddress>,
            eredu_runtime::PipelineActivationDtype,
            Option<Arc<crate::partitioned_execution::TensorPipelineCollectiveWaves>>,
        )>,
    ) -> Self
    where
        C: PartitionedConfig + Send + Sync,
        P: BlockFactory<WorkspaceBackend, C> + 'static,
        P::FeedForward: TensorParallelProjectionOperator<WorkspaceBackend>,
    {
        Self(Arc::new(Dense::<C, P> {
            target,
            transform,
            selected,
            rank,
            local_state,
            pipeline: pipeline.map(|(plan, addresses, dtype, tensor_waves)| PipelineSource {
                plan,
                addresses,
                dtype,
                tensor_waves,
            }),
            marker: std::marker::PhantomData,
        }))
    }
}
impl<C, P> DirectPartitionSource for Dense<C, P>
where
    C: PartitionedConfig + Send + Sync,
    P: BlockFactory<WorkspaceBackend, C> + 'static,
    P::FeedForward: TensorParallelProjectionOperator<WorkspaceBackend>,
{
    fn tensor_waves(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<Option<Arc<crate::partitioned_execution::TensorPipelineCollectiveWaves>>, String>
    {
        if !selected.same_complete_selection(&self.selected) || rank != self.rank {
            return Err(
                "retained tensor pipeline source differs from the selected constructor".into(),
            );
        }
        Ok(self
            .pipeline
            .as_ref()
            .and_then(|pipeline| pipeline.tensor_waves.clone()))
    }
    fn quote(
        &self,
        source: &PreparedModelSources,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        visitor: EquationVisitor<'_, '_, '_>,
    ) -> Result<EquationQuote, Error> {
        if let Some(pipeline) = &self.pipeline {
            return pipeline_quote::<C, P>(self, pipeline, source, communication, visitor);
        }
        type Backend = WorkspaceBackend;
        let context = visitor.context;
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            &PreparedModelSources,
            EquationVisitor<'_, '_, '_>,
            PartitionedLayeredModel<Backend, C, P>,
            Option<PartitionedLayeredModel<Backend, C, P>>,
            eredu_nn::workspace::WorkspaceParallelContext,
            Result<EquationQuote, Error>,
            &eredu_runtime::SelectedStateRealization,
            Result<(), Error>,
        )>())?;
        if !source.selected().same_complete_selection(&self.selected)
            || source.selected().execution().parallel_topology() != Some(self.rank)
            || self.rank.pipeline_parallel_size() != 1
            || self.rank.tensor_parallel_size() <= 1
        {
            return Err(context.metadata_error(format_args!(
                "parallel quote differs from its completed direct partition source"
            )));
        }
        eredu_runtime::working_memory::validate_workspace_state_realization(
            visitor.state,
            &self.local_state,
            context,
        )?;
        let parallel = eredu_nn::workspace::WorkspaceParallelContext::new(
            self.rank.tensor_parallel_rank(),
            self.rank.tensor_parallel_size(),
        )?;
        // Keep the actual transformed source modules until all target traces
        // retire, exactly as the replicated source-preserving constructor does.
        let _transform = self
            .transform
            .as_ref()
            .map(|source| {
                PartitionedLayeredModel::<Backend, C, P>::from_retained_partition_source(
                    source.clone(),
                    context,
                )
            })
            .transpose()?;
        let architecture =
            PartitionedLayeredModel::<Backend, C, P>::from_retained_partition_source(
                self.target.clone(),
                context,
            )?;
        let layout = architecture.state_layout(Some(context))?;
        if visitor.state.layout() != &layout {
            return Err(context.metadata_error(format_args!(
                "parallel state projection differs from its exact local constructor"
            )));
        }
        match visitor.parameters {
            Some(parameters) => {
                let runtime = LayerwiseRuntime::new_workspace_with_policy(
                    architecture,
                    |layout| WorkspaceLayerwisePolicy::for_layout(parameters, layout, context),
                    context,
                )?;
                direct_spans::<C, P, _>(self, runtime, &parallel, visitor)
            }
            None => {
                let runtime = ResidentRuntime::<_, Backend, ResidentState>::new_workspace(
                    architecture,
                    context,
                )?
                .into_layerwise_workspace(context)?;
                direct_spans::<C, P, _>(self, runtime, &parallel, visitor)
            }
        }
    }
}

fn direct_spans<C, P, Q>(
    source: &Dense<C, P>,
    mut runtime: LayerwiseRuntime<
        PartitionedLayeredModel<WorkspaceBackend, C, P>,
        WorkspaceBackend,
        ResidentState,
        Q,
    >,
    parallel: &eredu_nn::workspace::WorkspaceParallelContext,
    visitor: EquationVisitor<'_, '_, '_>,
) -> Result<EquationQuote, Error>
where
    C: PartitionedConfig + Send + Sync,
    P: BlockFactory<WorkspaceBackend, C> + 'static,
    P::FeedForward: TensorParallelProjectionOperator<WorkspaceBackend>,
    Q: eredu_runtime::LayerwisePolicy<WorkspaceBackend, Unit<C, P>>,
    <Q as eredu_runtime::LayerwisePolicy<WorkspaceBackend, Unit<C, P>>>::Error:
        std::error::Error + Send + Sync + 'static,
{
    let context = visitor.context;
    context.charge_metadata(std::mem::size_of::<(
        LayerwiseRuntime<
            PartitionedLayeredModel<WorkspaceBackend, C, P>,
            WorkspaceBackend,
            ResidentState,
            Q,
        >,
        <PartitionedLayeredModel<WorkspaceBackend, C, P> as LayeredArchitecture<
            WorkspaceBackend,
            ResidentState,
        >>::ForwardContext,
        &Dense<C, P>,
        &eredu_nn::workspace::WorkspaceParallelContext,
        EquationVisitor<'_, '_, '_>,
        Result<EquationQuote, Error>,
    )>())?;
    context.charge_metadata(std::mem::size_of::<(
        Option<eredu_runtime::PreparedLayeredObservationPaths>,
        Option<&eredu_runtime::SharedLayeredObservationPaths>,
        Result<
            eredu_runtime::PreparedLayeredObservationPaths,
            eredu_runtime::PreparedLayeredObservationError<Error>,
        >,
        u64,
    )>())?;
    let paths = visitor
        .observation
        .map(|observation| {
            runtime.bind_observation_paths(
                observation.paths,
                Some(eredu_runtime::layered::LayeredMetadata::new(
                    context,
                    |error| error,
                )),
            )
        })
        .transpose()
        .map_err(|cause| cause.into_quote_error(context))?;
    let capture = visitor.target_capture;
    if capture {
        <PartitionedLayeredModel<WorkspaceBackend, C, P> as LayeredArchitecture<
            WorkspaceBackend,
            ResidentState,
        >>::retain_prediction_target_capture(runtime.architecture_mut());
    }
    context.charge_metadata(std::mem::size_of::<(bool, Option<WorkspaceTensor>)>())?;
    direct_publication_spans(
        &source.selected,
        visitor,
        paths,
        |tokens, state, demand, observer, paths, _span| {
            let (scores, forward) = match (observer, paths) {
                (observer, Some(paths)) => runtime
                    .forward_parallel_with_prepared_observer_and_context_with_readout(
                        crate::decoder::LayeredInput { tokens, mask: None },
                        state,
                        parallel,
                        context,
                        observer,
                        paths,
                        demand,
                    )
                    .map_err(|cause| context.metadata_source(cause))?,
                (None, None) => runtime
                    .forward_parallel_fixed_with_readout(
                        crate::decoder::LayeredInput { tokens, mask: None },
                        state,
                        parallel,
                        context,
                        demand,
                    )
                    .map_err(|cause| context.metadata_source(cause))?,
                _ => {
                    return Err(context.metadata_source(
                        eredu_runtime::working_memory::InferenceObservationError::RemoteOutput,
                    ))
                }
            };
            let hidden = if capture {
                Some(
                    <PartitionedLayeredModel<WorkspaceBackend, C, P> as LayeredArchitecture<
                        WorkspaceBackend,
                        ResidentState,
                    >>::prediction_target_capture(&forward)
                    .ok_or_else(|| {
                        context.metadata_error(format_args!(
                            "partition target equation did not retain its prediction capture"
                        ))
                    })?
                    .clone(),
                )
            } else {
                None
            };
            Ok((scores, hidden))
        },
    )
}

// All direct partition sources use the same selected publication occurrence,
// output observation timing and receipt ownership. Only the fixed architecture
// forward worker differs between dense and routed resident constructors.
fn direct_publication_spans<F>(
    selected_source: &crate::SelectedPreparation,
    visitor: EquationVisitor<'_, '_, '_>,
    paths: Option<eredu_runtime::PreparedLayeredObservationPaths>,
    mut forward: F,
) -> Result<EquationQuote, Error>
where
    F: FnMut(
        &WorkspaceTensor,
        &mut ResidentState,
        eredu_core::OutputDemand,
        Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
        Option<&eredu_runtime::PreparedLayeredObservationPaths>,
        &InferenceWorkspaceSpan,
    ) -> Result<(Option<WorkspaceTensor>, Option<WorkspaceTensor>), Error>,
{
    let context = visitor.context;
    context.charge_metadata(std::mem::size_of::<(
        &crate::SelectedPreparation,
        EquationVisitor<'_, '_, '_>,
        Option<eredu_runtime::PreparedLayeredObservationPaths>,
        F,
        Option<WorkspaceTensor>,
        Option<EquationCapture>,
        Result<EquationQuote, Error>,
    )>())?;
    context.charge_metadata(
        std::mem::size_of::<(
            eredu_runtime::PartitionOutputPublication,
            eredu_runtime::CommunicationGroupOperation<'_>,
            usize,
            usize,
            &[usize],
        )>()
        .checked_add(
            eredu_runtime::CommunicationManifest::group_operation_control_bytes().ok_or_else(
                || context.metadata_error(format_args!("publication control size overflow")),
            )?,
        )
        .ok_or_else(|| context.metadata_error(format_args!("publication control size overflow")))?,
    )?;
    let publication = selected_source
        .execution()
        .partitioned_output_publication()
        .ok_or_else(|| {
            context.metadata_error(format_args!("partition has no selected publication"))
        })?;
    let selected = selected_source
        .communication_manifest()
        .ok_or_else(|| context.metadata_error(format_args!("missing publication manifest")))?
        .select_group_operation(
            publication.group,
            eredu_runtime::CommunicationOperation::Broadcast,
        )
        .map_err(|cause| context.metadata_source(cause))?;
    let ranks = selected.descriptor().members();
    let root = ranks
        .iter()
        .position(|&rank| rank == publication.owner_rank)
        .ok_or_else(|| {
            context.metadata_error(format_args!("publication owner is not a selected member"))
        })?;
    let rank = selected.descriptor().local_index().ok_or_else(|| {
        context.metadata_error(format_args!("publication rank is not a selected member"))
    })?;
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
            let (scores, capture) = match observer.as_deref_mut() {
                Some(observer) => {
                    forward(tokens, state, demand, Some(observer), paths.as_ref(), span)?
                }
                None => forward(tokens, state, demand, None, paths.as_ref(), span)?,
            };
            let scores = match (scores, observer.as_deref_mut()) {
                (Some(scores), Some(observer)) if rank == root => {
                    Some(eredu_runtime::observe_model_logits(observer, &scores)?)
                }
                (Some(scores), Some(observer)) => {
                    observer.observe_replica(eredu_core::MODEL_LOGITS_OBSERVATION_PATH, &scores)?;
                    Some(scores)
                }
                (scores, _) => scores,
            };
            let scores = scores
                .map(|scores| {
                    scores.broadcast_publication(
                        publication.group,
                        root,
                        rank,
                        ranks.len(),
                        context,
                    )
                })
                .transpose()?;
            if rank != root {
                if let (Some(scores), Some(observer)) = (&scores, observer) {
                    observer.observe_remote_output(scores, context)?;
                }
            }
            Ok((scores, capture.map(EquationCapture::Embedded)))
        },
    )
}

impl PreparedInferenceBlueprint {
    /// Quotes the actual direct TP model equations, selected output publication
    /// contribution/Sum, and ordinary configured sampling from a completed
    /// same-source partition constructor. This does not quote PP transport,
    /// world agreement, native storage or completion.
    /// Those mechanism producers remain required for whole-request admission.
    pub fn quote_direct_parallel_text_with_sampling_and_trace<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        observer: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_partitioned_text_with_sampling_and_trace(
            geometry, state, context, config, filter, observer, None,
        )
    }

    /// Reuses the same selected partition constructor and pipeline driver with
    /// its actual retained communication source. This is a cold description;
    /// the backend still supplies exact native route bounds and submission.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_partitioned_text_with_sampling_and_trace<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        observer: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_partitioned_sampling_input(
            geometry,
            state,
            context,
            TextSamplingInput::Configured(config, filter.into()),
            observer,
            communication,
            None,
            None,
        )
    }

    /// Quotes the independently copied local state and the actual populated
    /// saved sampler through the same retained partition constructor and spans.
    /// This borrows source state; it creates no request or copy authority.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_partitioned_text_with_existing_sampling_and_trace(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        observer: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_partitioned_sampling_input(
            geometry,
            state,
            context,
            TextSamplingInput::Borrowed(sampling),
            observer,
            communication,
            None,
            None,
        )
    }

    /// Quotes the same selected partition traversal with its exact retained
    /// layerwise parameter source. This source describes unit binding only;
    /// native transfer, residency roles and completion remain separately required.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_partitioned_layerwise_text_with_sampling_and_trace<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        observer: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        parameters: &dyn WorkspaceLayerwiseParameters,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_partitioned_sampling_input(
            geometry,
            state,
            context,
            TextSamplingInput::Configured(config, filter.into()),
            observer,
            communication,
            Some(parameters),
            None,
        )
    }

    /// Quotes the same independently copied state and populated sampler with
    /// the matching retained layerwise source used for resume materialization.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_partitioned_layerwise_text_with_existing_sampling_and_trace(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        observer: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        parameters: &dyn WorkspaceLayerwiseParameters,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_partitioned_sampling_input(
            geometry,
            state,
            context,
            TextSamplingInput::Borrowed(sampling),
            observer,
            communication,
            Some(parameters),
            None,
        )
    }

    /// Quote authoritative final-logits capture before the selected partition
    /// publication. The same retained TP/PP constructor, checked parameter loan,
    /// span driver and sampling worker are used. Native capture/receipt transport
    /// still needs its independently retained producers and admission.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_partitioned_text_with_sampling_observed_and_trace<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        paths: &eredu_runtime::SharedLayeredObservationPaths,
        capture: &mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver,
        trace: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_partitioned_sampling_observed(
            geometry,
            state,
            context,
            TextSamplingInput::Configured(config, filter.into()),
            paths,
            capture,
            trace,
            communication,
            parameters,
        )
    }

    /// Observes an independently copied local state using its populated saved
    /// sampler and exact retained paths. The same partition span driver runs;
    /// the observer retains the saved absolute capture frontier and usage.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_partitioned_text_with_existing_sampling_observed_and_trace(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        paths: &eredu_runtime::SharedLayeredObservationPaths,
        capture: &mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver,
        trace: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_partitioned_sampling_observed(
            geometry,
            state,
            context,
            TextSamplingInput::Borrowed(sampling),
            paths,
            capture,
            trace,
            communication,
            parameters,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn quote_partitioned_sampling_observed(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: TextSamplingInput<'_>,
        paths: &eredu_runtime::SharedLayeredObservationPaths,
        capture: &mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver,
        trace: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        context
            .charge_metadata(std::mem::size_of::<(
                ObservationRef<'_, '_>,
                std::cell::RefCell<
                    &mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver,
                >,
                [TextSamplingInput<'_>; 2],
                [Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>>; 2],
                [(
                    &Self,
                    InferenceGeometry,
                    &ResidentState,
                    &WorkspaceContext,
                    &eredu_runtime::SharedLayeredObservationPaths,
                    &mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver,
                    &mut dyn InferenceEquationTraceObserver,
                    Option<&eredu_runtime::RetainedCommunicationSource>,
                    Option<&dyn WorkspaceLayerwiseParameters>,
                ); 2],
            )>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let capture = std::cell::RefCell::new(capture);
        let observation = ObservationRef {
            paths,
            observer: &capture,
            invocation_prediction: None,
        };
        self.quote_partitioned_sampling_input(
            geometry,
            state,
            context,
            sampling,
            trace,
            communication,
            parameters,
            Some(observation),
        )
    }

    fn quote_partitioned_sampling_input(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: TextSamplingInput<'_>,
        observer: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observation: Option<ObservationRef<'_, '_>>,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        context
            .charge_metadata(std::mem::size_of::<(
                EquationVisitor<'_, '_, '_>,
                std::cell::RefCell<&mut dyn InferenceEquationTraceObserver>,
                TextSamplingInput<'_>,
                Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>>,
                // Public configured/borrowed entry and the shared input worker.
                TextSamplingInput<'_>,
                &Self,
                InferenceGeometry,
                &ResidentState,
                &WorkspaceContext,
                Option<&eredu_runtime::RetainedCommunicationSource>,
                Option<&dyn WorkspaceLayerwiseParameters>,
                Option<ObservationRef<'_, '_>>,
                bool,
                Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>>,
            )>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        geometry
            .validate()
            .map_err(|cause| preparation_message(context, format_args!("{cause}")))?;
        if geometry.batch_size != 1
            || geometry.output == eredu_core::OutputDemand::StateOnly
            || (observation.is_none() && geometry.output != eredu_core::OutputDemand::LastPosition)
        {
            return Err(preparation_message(
                context,
                format_args!(
                    "parallel sampling quote requires one sequence with an admitted output demand"
                ),
            ));
        }
        if let Some(observation) = observation {
            observation
                .validate_selection(geometry, context)
                .map_err(PreparedExecutionError::Backend)?;
            if observation.requires_sequence_readout()
                && geometry.output != eredu_core::OutputDemand::Sequence
            {
                return Err(preparation_message(
                    context,
                    format_args!("parallel observation requires sequence output"),
                ));
            }
        }
        if matches!(
            self.selected().text_realization().residency(),
            eredu_runtime::LayerWeightResidency::FullyResident
        ) != parameters.is_none()
        {
            return Err(preparation_message(
                context,
                format_args!(
                    "parallel parameter source differs from the selected weight residency"
                ),
            ));
        }
        let source = self
            .sources
            .construction_semantics()
            .direct_partition
            .get()
            .ok_or_else(|| {
                preparation_message(
                    context,
                    format_args!("exact completed direct partition quote source is unavailable"),
                )
            })?;
        let observer = std::cell::RefCell::new(observer);
        let visitor = EquationVisitor {
            geometry,
            state,
            context,
            sampling: Some(sampling),
            parameters,
            unpriced_execution: None,
            observation,
            trace: Some(EquationTraceRef(&observer)),
            media: None,
            input_dtype: None,
            target_capture: false,
            routed_pass: None,
            external_target: None,
        };
        let (equations, sampling) = source
            .0
            .quote(&self.sources, communication, visitor)
            .map_err(PreparedExecutionError::Metadata)?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("configured parallel quote includes sampling"),
        })
    }
}
