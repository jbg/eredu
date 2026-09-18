//! Completed family model sources consumed by the shared routed TP/PP workers.
use super::*;
use crate::routed_text::RetainedPartitionResidentSource;
use eredu_runtime::{ParallelRoutedLayeredArchitecture, ReplicatedTextArchitecture};

pub(crate) enum PreparedFamilyPartitionModelSource {
    DeepSeekV3(crate::deepseek::v3::RetainedParallelModelSource,
        Option<crate::deepseek::v3::RetainedParallelModelSource>),
}
struct Family {
    model: PreparedFamilyPartitionModelSource,
    selected: crate::SelectedPreparation,
    rank: eredu_core::ParallelRankTopology,
    state: eredu_runtime::SelectedStateRealization,
    state_offset: usize,
    execution: crate::partitioned_execution::PreparedRoutedExecutionHandoff,
    providers: RetainedPartitionResidentSource,
    pipeline: Option<PipelineSource>,
}
impl PreparedDirectPartitionSource {
    pub(crate) fn family_routed(model: PreparedFamilyPartitionModelSource,
        selected: crate::SelectedPreparation, rank: eredu_core::ParallelRankTopology,
        state: eredu_runtime::SelectedStateRealization, state_offset: usize,
        execution: crate::partitioned_execution::PreparedRoutedExecutionHandoff,
        providers: RetainedPartitionResidentSource,
        addresses: Option<Vec<eredu_runtime::ExecutionUnitAddress>>) -> Self {
        let pipeline = addresses.map(|addresses| PipelineSource {
            plan: execution.retained_execution_plan(), addresses,
            dtype: execution.activation_dtype(),
            tensor_waves: execution.tensor_pipeline_collective_waves(),
        });
        Self(Arc::new(Family {model, selected, rank, state, state_offset, execution, providers, pipeline}))
    }
}
impl Family {
    fn validate(&self, selected: &crate::SelectedPreparation,
        rank: eredu_core::ParallelRankTopology) -> Result<(), String> {
        if !selected.same_complete_selection(&self.selected) || rank != self.rank {
            return Err("family routed quote differs from its completed constructor".into());
        }
        Ok(())
    }
}
impl DirectPartitionSource for Family {
    fn tensor_waves(&self, selected: &crate::SelectedPreparation, rank: eredu_core::ParallelRankTopology)
        -> Result<Option<Arc<crate::partitioned_execution::TensorPipelineCollectiveWaves>>, String> {
        self.validate(selected, rank)?;
        Ok(self.execution.tensor_pipeline_collective_waves())
    }
    fn routed_resident_source(&self, selected: &crate::SelectedPreparation, rank: eredu_core::ParallelRankTopology)
        -> Result<Option<RetainedPartitionResidentSource>, String> {
        self.validate(selected, rank)?;
        Ok(Some(self.providers.clone()))
    }
    fn quote(&self, actual: &PreparedModelSources, communication: Option<&eredu_runtime::RetainedCommunicationSource>,
        visitor: EquationVisitor<'_, '_, '_>) -> Result<EquationQuote, Error> {
        let context = visitor.context;
        context.charge_metadata(std::mem::size_of::<(&Self, &PreparedModelSources,
            EquationVisitor<'_, '_, '_>, Result<EquationQuote, Error>,
            crate::deepseek::v3::Model<WorkspaceBackend>, Option<crate::deepseek::v3::Model<WorkspaceBackend>>) >())?;
        self.validate(actual.selected(), actual.selected().execution().parallel_topology()
            .ok_or_else(|| context.metadata_error(format_args!("family quote has no selected parallel rank")))?)
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
        let communication = communication.ok_or_else(|| context.metadata_error(format_args!(
            "family routed quote has no retained communication source")))?;
        if actual.selected().communication_manifest() != Some(communication.manifest()) {
            return Err(context.metadata_error(format_args!("family routed communication source differs")));
        }
        eredu_runtime::working_memory::validate_workspace_state_realization(visitor.state, &self.state, context)?;
        match &self.model {
            PreparedFamilyPartitionModelSource::DeepSeekV3(target, transform) => {
                let _transform = transform.as_ref().map(|source|
                    crate::deepseek::v3::Model::<WorkspaceBackend>::from_retained_parallel_source(source.clone(), context)).transpose()?;
                let model = crate::deepseek::v3::Model::<WorkspaceBackend>::from_retained_parallel_source(target.clone(), context)?;
                quote_model(self, model, communication, visitor)
            }
        }
    }
}
fn quote_model<A>(source: &Family, architecture: A,
    communication: &eredu_runtime::RetainedCommunicationSource, visitor: EquationVisitor<'_, '_, '_>) -> Result<EquationQuote, Error>
where A: crate::partitioned_execution::TextPartitionArchitecture<WorkspaceBackend, ResidentState, Error=Error>
        +ParallelRoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
        +ReplicatedTextArchitecture<WorkspaceBackend, ResidentState>+'static,
{
    let context = visitor.context;
    context.charge_metadata(std::mem::size_of::<(A, eredu_runtime::StateLayout,
        Vec<eredu_runtime::ExecutionUnitAddress>, eredu_nn::workspace::WorkspaceParallelContext,
        Result<EquationQuote, Error>)>())?;
    let layout = architecture.state_layout(Some(context))?;
    let end = source.state_offset.checked_add(visitor.state.layout().len())
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
    let local = layout.slice_with_metadata(source.state_offset..end, context)
        .map_err(|cause| context.metadata_source(cause))?;
    if &local != visitor.state.layout() {
        return Err(context.metadata_error(format_args!("family state differs from completed local geometry")));
    }
    if let Some(pipeline) = &source.pipeline {
        let mut addresses = context.metadata_vec(pipeline.addresses.len())?;
        addresses.extend_from_slice(&pipeline.addresses);
        let strategy = routed::pipeline_strategy(&source.execution, source.providers.borrowed(), context)?;
        context.charge_metadata(std::mem::size_of_val(&strategy).checked_mul(2)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
        match visitor.parameters {
            Some(parameters) => {
                let policy = WorkspaceLayerwisePolicy::for_partition(parameters, &architecture, &addresses, context)?;
                let bounded = WorkspaceLayerwisePolicy::for_layout(parameters, parameters.layout(), context)?;
                pipeline::pipeline_spans(&source.selected, source.rank, pipeline, architecture,
                    policy, Some(bounded), addresses, communication, strategy, visitor)
            }
            None => {
                let policy = ResidentUnitWindow::<A::Unit>::from_workspace_addresses::<A,ResidentState>(
                    &architecture, &addresses, context)?;
                pipeline::pipeline_spans(&source.selected, source.rank, pipeline, architecture,
                    policy, None, addresses, communication, strategy, visitor)
            }
        }
    } else {
        let parallel = eredu_nn::workspace::WorkspaceParallelContext::new(
            source.rank.tensor_parallel_rank(), source.rank.tensor_parallel_size())?;
        let bindings = eredu_runtime::working_memory::workspace_partition_communication(communication, context)?;
        match visitor.parameters {
            Some(parameters) => {
                let runtime = LayerwiseRuntime::new_workspace_with_policy(architecture,
                    |layout| WorkspaceLayerwisePolicy::for_layout(parameters, layout, context), context)?;
                routed::spans(&source.selected, source.rank, &source.execution, source.providers.borrowed(),
                    runtime, &parallel, &bindings, visitor)
            }
            None => {
                let runtime = ResidentRuntime::<_,WorkspaceBackend,ResidentState>::new_workspace(architecture,context)?
                    .into_layerwise_workspace(context)?;
                routed::spans(&source.selected, source.rank, &source.execution, source.providers.borrowed(),
                    runtime, &parallel, &bindings, visitor)
            }
        }
    }
}
