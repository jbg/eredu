//! Exact routed partition source; equations use the selected ordinary provider driver.
use super::*;
use crate::prepared_execution::workspace::layerwise::QuoteError;
mod movement;
pub(crate) mod provider;
mod region;
use crate::decoder::TensorParallelRoutedProjectionOperator;
use crate::routed_text::RetainedPartitionResidentSource;
use eredu_runtime::working_memory::{
    workspace_partition_communication, WorkspaceCommunicationGroup, WorkspaceCommunicationMetadata,
    WorkspaceCommunicationRoute,
};

type Model<C, P> = PartitionedLayeredModel<WorkspaceBackend, C, P>;
type Communication = eredu_runtime::PartitionCommunication<
    WorkspaceBackend,
    WorkspaceCommunicationGroup,
    WorkspaceCommunicationRoute,
    WorkspaceCommunicationMetadata,
>;

struct Routed<C, P> {
    target: Arc<PartitionModelSource<C>>,
    selected: crate::SelectedPreparation,
    rank: eredu_core::ParallelRankTopology,
    local_state: eredu_runtime::SelectedStateRealization,
    execution: crate::partitioned_execution::PreparedRoutedExecutionHandoff,
    providers: provider::Source,
    pipeline: Option<PipelineSource>,
    marker: std::marker::PhantomData<fn() -> P>,
}

impl PreparedDirectPartitionSource {
    pub(crate) fn routed<C, P>(
        target: Arc<PartitionModelSource<C>>,
        selected: crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
        local_state: eredu_runtime::SelectedStateRealization,
        execution: crate::partitioned_execution::PreparedRoutedExecutionHandoff,
        providers: Option<RetainedPartitionResidentSource>,
        banks: crate::routed_text::RetainedRoutedBanks,
        residency: eredu_runtime::ParameterBankResidency,
        pipeline_addresses: Option<Vec<eredu_runtime::ExecutionUnitAddress>>,
    ) -> Self
    where
        C: PartitionedConfig + Send + Sync,
        P: BlockFactory<WorkspaceBackend, C> + 'static,
        P::FeedForward: TensorParallelRoutedProjectionOperator<WorkspaceBackend>,
    {
        let providers = match providers {
            Some(source) => provider::Source::Resident(source),
            None => provider::Source::Addressable { banks, residency },
        };
        let pipeline = pipeline_addresses.map(|addresses| PipelineSource {
            plan: execution.retained_execution_plan(),
            addresses,
            dtype: execution.activation_dtype(),
            tensor_waves: execution.tensor_pipeline_collective_waves(),
        });
        Self(Arc::new(Routed::<C, P> {
            target,
            selected,
            rank,
            local_state,
            execution,
            providers,
            pipeline,
            marker: std::marker::PhantomData,
        }))
    }
}

impl<C, P> DirectPartitionSource for Routed<C, P>
where
    C: PartitionedConfig + Send + Sync,
    P: BlockFactory<WorkspaceBackend, C> + 'static,
    P::FeedForward: TensorParallelRoutedProjectionOperator<WorkspaceBackend>,
{
    fn routed_addressable_source(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<
        Option<(
            crate::routed_text::RetainedRoutedBanks,
            crate::partitioned_execution::PreparedRoutedExecutionHandoff,
        )>,
        String,
    > {
        self.tensor_waves(selected, rank)?;
        Ok(match &self.providers {
            provider::Source::Addressable { banks, .. } => {
                Some((banks.clone(), self.execution.clone()))
            }
            provider::Source::Resident(_) => None,
        })
    }
    fn routed_resident_source(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<Option<RetainedPartitionResidentSource>, String> {
        self.tensor_waves(selected, rank)?;
        Ok(self.providers.resident())
    }
    fn tensor_waves(
        &self,
        selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<Option<Arc<crate::partitioned_execution::TensorPipelineCollectiveWaves>>, String>
    {
        if !selected.same_complete_selection(&self.selected) || rank != self.rank {
            return Err(
                "retained routed partition source differs from its selected constructor".into(),
            );
        }
        Ok(self
            .pipeline
            .as_ref()
            .and_then(|pipeline| pipeline.tensor_waves.clone()))
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
            Model<C, P>,
            Communication,
            eredu_nn::workspace::WorkspaceParallelContext,
            eredu_runtime::StateLayout,
            Result<EquationQuote, Error>,
        )>())?;
        if !actual.selected().same_complete_selection(&self.selected)
            || actual.selected().execution().parallel_topology() != Some(self.rank)
            || self.pipeline.is_some() != (self.rank.pipeline_parallel_size() > 1)
            || (self.rank.pipeline_parallel_size() == 1
                && self.rank.tensor_parallel_size() <= 1
                && self.rank.expert_parallel_size() <= 1)
        {
            return Err(context.metadata_error(format_args!(
                "routed quote differs from its retained local tensor partition"
            )));
        }
        let communication = communication.ok_or_else(|| {
            context.metadata_error(format_args!(
                "routed provider agreement has no retained communication source"
            ))
        })?;
        if actual.selected().communication_manifest() != Some(communication.manifest()) {
            return Err(context.metadata_error(format_args!(
                "routed provider agreement differs from the retained manifest"
            )));
        }
        eredu_runtime::working_memory::validate_workspace_state_realization(
            visitor.state,
            &self.local_state,
            context,
        )?;
        let architecture =
            Model::<C, P>::from_retained_partition_source(self.target.clone(), context)?;
        if let Some(pipeline) = &self.pipeline {
            return pipeline_quote(self, pipeline, architecture, communication, visitor);
        }
        let layout = architecture.state_layout(Some(context))?;
        if visitor.state.layout() != &layout {
            return Err(context.metadata_error(format_args!(
                "routed state projection differs from its exact local constructor"
            )));
        }
        let parallel = eredu_nn::workspace::WorkspaceParallelContext::new(
            self.rank.tensor_parallel_rank(),
            self.rank.tensor_parallel_size(),
        )?;
        let communication = workspace_partition_communication(communication, context)?;
        match visitor.parameters {
            Some(parameters) => {
                let runtime = LayerwiseRuntime::new_workspace_with_policy(
                    architecture,
                    |layout| WorkspaceLayerwisePolicy::for_layout(parameters, layout, context),
                    context,
                )?;
                spans::<Model<C, P>, _, _>(
                    &self.selected,
                    self.rank,
                    &self.execution,
                    self.providers.provider(context)?,
                    runtime,
                    &parallel,
                    &communication,
                    visitor,
                )
            }
            None => {
                let runtime = ResidentRuntime::<_, WorkspaceBackend, ResidentState>::new_workspace(
                    architecture,
                    context,
                )?
                .into_layerwise_workspace(context)?;
                spans::<Model<C, P>, _, _>(
                    &self.selected,
                    self.rank,
                    &self.execution,
                    self.providers.provider(context)?,
                    runtime,
                    &parallel,
                    &communication,
                    visitor,
                )
            }
        }
    }
}

pub(super) fn spans<A, Q, Provider>(
    selected: &crate::SelectedPreparation,
    rank: eredu_core::ParallelRankTopology,
    execution: &crate::partitioned_execution::PreparedRoutedExecutionHandoff,
    mut provider: Provider,
    mut runtime: LayerwiseRuntime<A, WorkspaceBackend, ResidentState, Q>,
    parallel: &eredu_nn::workspace::WorkspaceParallelContext,
    communication: &Communication,
    visitor: EquationVisitor<'_, '_, '_>,
) -> Result<EquationQuote, Error>
where
    A: eredu_runtime::ParallelRoutedLayeredArchitecture<
            WorkspaceBackend,
            ResidentState,
            Error = Error,
        > + eredu_runtime::ReplicatedTextArchitecture<WorkspaceBackend, ResidentState>
        + 'static,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<WorkspaceBackend>,
    Q: eredu_runtime::LayerwisePolicy<WorkspaceBackend, A::Unit>,
    Q::Error: std::error::Error + Send + Sync + 'static,
{
    let context = visitor.context;
    context.charge_metadata(std::mem::size_of::<(
        &crate::SelectedPreparation,
        eredu_core::ParallelRankTopology,
        &crate::partitioned_execution::PreparedRoutedExecutionHandoff,
        Provider,
        LayerwiseRuntime<A, WorkspaceBackend, ResidentState, Q>,
        <A as LayeredArchitecture<WorkspaceBackend, ResidentState>>::ForwardContext,
        &eredu_nn::workspace::WorkspaceParallelContext,
        &Communication,
        EquationVisitor<'_, '_, '_>,
        Result<EquationQuote, Error>,
        Option<eredu_runtime::PreparedLayeredObservationPaths>,
        eredu_runtime::ExpertPass,
        Option<eredu_core::CollectiveGroupId>,
    )>())?;
    let group = if rank.tensor_parallel_size() > 1 {
        Some(execution.communication_tensor_group().ok_or_else(|| {
            context.metadata_error(format_args!(
                "routed constructor retained no tensor provider agreement group"
            ))
        })?)
    } else {
        None
    };
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
    let mut provider = region::RegionProvider::new(execution, &mut provider);
    context.charge_metadata(
        std::mem::size_of_val(&provider)
            .checked_mul(2)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
    )?;
    let capture = visitor.target_capture;
    if capture {
        runtime
            .architecture_mut()
            .retain_prediction_target_capture();
    }
    context.charge_metadata(std::mem::size_of::<(bool, Option<WorkspaceTensor>)>())?;
    direct_publication_spans(
        selected,
        visitor,
        paths,
        |tokens, state, demand, observer, paths, span| {
            let pass = match span {
                InferenceWorkspaceSpan::Sampling(_) => {
                    unreachable!("model equation scheduler emits only prefill/decode spans")
                }
                InferenceWorkspaceSpan::Prefill(_) => eredu_runtime::ExpertPass::Prefill,
                InferenceWorkspaceSpan::Decode { .. } => eredu_runtime::ExpertPass::Decode,
            };
            let mut agreed = eredu_runtime::expert::AgreeingRoutedExpertProvider::new(
                &mut provider,
                |success| {
                    crate::partitioned_execution::agree_provider_tensor_work(
                        communication,
                        context,
                        Some(parallel),
                        group,
                        success,
                    )
                },
            );
            context.charge_metadata(
                std::mem::size_of_val(&agreed)
                    .checked_mul(2)
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
            )?;
            let mut execute =
                |observer: &mut dyn eredu_runtime::ActivationObserver<WorkspaceTensor, Error>| {
                    runtime
                        .forward_routed_with_prepared_paths(
                            A::text_input(tokens, None),
                            state,
                            (rank.tensor_parallel_size() > 1).then_some(parallel),
                            context,
                            &mut agreed,
                            pass,
                            observer,
                            demand,
                            paths,
                        )
                        .map_err(|cause| context.metadata_source(cause))
                };
            let (scores, forward) = match observer {
                Some(observer) => execute(observer),
                None => execute(&mut eredu_runtime::inspection::NoopObserver),
            }?;
            let hidden = if capture {
                Some(
                    A::prediction_target_capture(&forward)
                        .ok_or_else(|| {
                            context.metadata_error(format_args!(
                        "routed partition target equation did not retain its prediction capture"
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

/// Uses the same ordinary pipeline strategy with its immutable provider table.
/// No row movement is selected when the retained expert group is absent.
fn pipeline_quote<C, P>(
    source: &Routed<C, P>,
    pipeline: &PipelineSource,
    architecture: Model<C, P>,
    communication: &eredu_runtime::RetainedCommunicationSource,
    visitor: EquationVisitor<'_, '_, '_>,
) -> Result<EquationQuote, Error>
where
    C: PartitionedConfig + Send + Sync,
    P: BlockFactory<WorkspaceBackend, C> + 'static,
    P::FeedForward: TensorParallelRoutedProjectionOperator<WorkspaceBackend>,
{
    let context = visitor.context;
    context.charge_metadata(std::mem::size_of::<(
        &Routed<C, P>,
        &PipelineSource,
        Model<C, P>,
        &eredu_runtime::RetainedCommunicationSource,
        EquationVisitor<'_, '_, '_>,
        eredu_runtime::StateLayout,
        Vec<eredu_runtime::ExecutionUnitAddress>,
        Result<EquationQuote, Error>,
    )>())?;
    let geometry = architecture.local_geometry();
    let layout = geometry
        .complete_state_layout()
        .slice_with_metadata(geometry.owned_units(), context)
        .map_err(|cause| context.metadata_source(cause))?;
    if visitor.state.layout() != &layout {
        return Err(context.metadata_error(format_args!(
            "routed pipeline state differs from its retained local constructor"
        )));
    }
    let mut addresses = context.metadata_vec(pipeline.addresses.len())?;
    addresses.extend_from_slice(&pipeline.addresses);
    let unit_strategy = pipeline_strategy(
        &source.execution,
        source.providers.provider(context)?,
        context,
    )?;
    context.charge_metadata(
        std::mem::size_of_val(&unit_strategy)
            .checked_mul(2)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
    )?;
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
            pipeline::pipeline_spans::<Model<C, P>, _, _>(
                &source.selected,
                source.rank,
                pipeline,
                architecture,
                policy,
                Some(bounded),
                addresses,
                communication,
                unit_strategy,
                visitor,
            )
        }
        None => {
            let policy = ResidentUnitWindow::<Unit<C, P>>::from_workspace_addresses::<
                Model<C, P>,
                ResidentState,
            >(&architecture, &addresses, context)?;
            pipeline::pipeline_spans::<Model<C, P>, _, _>(
                &source.selected,
                source.rank,
                pipeline,
                architecture,
                policy,
                None,
                addresses,
                communication,
                unit_strategy,
                visitor,
            )
        }
    }
}

/// Reuses the ordinary selected strategy, including its inactive PP/EP waves.
/// WorkspaceBackend records the native regions instead of entering their route
/// movement closure; the retained provider remains real for replicated units.
pub(super) fn pipeline_strategy<Provider>(
    execution: &crate::partitioned_execution::PreparedRoutedExecutionHandoff,
    provider: Provider,
    context: &WorkspaceContext,
) -> Result<
    crate::partitioned_execution::RoutedPipelinePartitionUnitStrategy<
        Provider,
        movement::SymbolicRouteMovement,
    >,
    Error,
> {
    context.charge_metadata(std::mem::size_of::<(
        &crate::partitioned_execution::PreparedRoutedExecutionHandoff,
        Provider,
        &WorkspaceContext,
        Result<
            crate::partitioned_execution::RoutedPipelinePartitionUnitStrategy<
                Provider,
                movement::SymbolicRouteMovement,
            >,
            Error,
        >,
        Option<usize>,
    )>())?;
    context.charge_metadata(
        execution
            .pipeline_unit_strategy_control_bytes::<Provider, movement::SymbolicRouteMovement>()
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
    )?;
    execution
        .pipeline_unit_strategy(provider, movement::SymbolicRouteMovement)
        .map_err(|cause| context.metadata_error(format_args!("{cause}")))
}
