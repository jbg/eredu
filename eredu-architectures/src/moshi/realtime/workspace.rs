//! Real temporal/depth traversal with exact projected native source inputs.
use eredu_nn::{Error,workspace::{WorkspaceBackend,WorkspaceContext,WorkspaceTensor}};
use eredu_runtime::{DeviceState,LayeredArchitecture,LayerwiseRuntime,
    working_memory::{WorkspaceResidentLayerState,WorkspaceSamplingBackend}};
use crate::prepared_execution::{WorkspaceLayerwiseParameters,WorkspaceLayerwisePolicy};
use super::{MoshiRealtimeExecutionError,execute_layerwise_moshi_realtime_with_observation};
use std::mem::{size_of,size_of_val};

type State=DeviceState<WorkspaceBackend,WorkspaceResidentLayerState>;
type Model=crate::moshi::LayeredModel<WorkspaceBackend>;

/// Traces the retained model's actual temporal/depth equations using the shared
/// frame traversal. Static bindings, unit layouts, state, temporal tensors and
/// sampling driver are projections of their respective existing owners. Missing
/// physical sources still refuse in those owners and the native recipe reducer.
/// This function neither invents token inputs nor replaces current state with a
/// frontier-derived empty cache. The caller observes the complete common trace,
/// including sampling and output/state retention, before granting admission.
#[allow(clippy::too_many_arguments)]
pub fn execute_moshi_workspace_frame<S,O>(source:&crate::moshi::MoshiRealtimeModelSource,
    static_bindings:&mut [eredu_runtime::PreparedParameterBinding<'_,WorkspaceTensor>],
    parameters:&dyn WorkspaceLayerwiseParameters,state:&mut State,
    temporal:&[WorkspaceTensor],
    driver:&mut eredu_runtime::SequentialDecisionDriver<WorkspaceSamplingBackend,S>,
    context:&WorkspaceContext,observer:&mut O,demand:eredu_core::OutputDemand,
    observation:eredu_runtime::SequentialDecisionObservation)
    ->Result<(Option<WorkspaceTensor>,crate::moshi::ForwardContext<WorkspaceTensor>),MoshiRealtimeExecutionError<Error>>
where S:eredu_runtime::Sampler<WorkspaceSamplingBackend>,
    O:eredu_runtime::ActivationObserver<WorkspaceTensor,Error>+?Sized {
    // The replicated frame and partitioned frame have distinct ordinary
    // traversals. A selected partition must use its actual communication owner
    // before its cold source can qualify; serial equations cannot stand in.
    if source.is_partitioned() {
        return Err(MoshiRealtimeExecutionError::Execution(eredu_runtime::LayerwiseRuntimeError::Architecture(
            eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())));
    }
    let mut prepare=||->Result<LayerwiseRuntime<Model,WorkspaceBackend,State,WorkspaceLayerwisePolicy<'_>>,Error> {
        let parts=[size_of::<Model>(),size_of::<Vec<usize>>(),
            size_of::<eredu_runtime::ExecutionUnitLayout>(),size_of::<eredu_runtime::ArchitectureExecutionGraph<'_>>(),
            size_of::<LayerwiseRuntime<Model,WorkspaceBackend,State,WorkspaceLayerwisePolicy<'_>>>(),
            size_of::<(&crate::moshi::MoshiRealtimeModelSource,&dyn WorkspaceLayerwiseParameters,&WorkspaceContext)>(),
            size_of::<Result<Model,Error>>()];
        context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
        context.validate_values(temporal)?;
        let mut model=source.workspace_model(context)?;
        eredu_runtime::working_memory::bind_prepared_workspace_parameters(
            model.static_modules_mut(),static_bindings,context)?;
        let graph=source.execution_graph();
        let mut counts=context.metadata_vec(graph.groups().len())?;
        for group in 0..graph.groups().len() {
            counts.push(<Model as LayeredArchitecture<WorkspaceBackend,State>>::group_unit_count(&model,group)?);
        }
        let layout=eredu_runtime::ExecutionUnitLayout::new_with_metadata(graph,&counts,context)?;
        let policy=WorkspaceLayerwisePolicy::for_layout(parameters,&layout,context)?;
        Ok(LayerwiseRuntime::new(model,policy))
    };
    let mut runtime=prepare().map_err(|cause|MoshiRealtimeExecutionError::Execution(
        eredu_runtime::LayerwiseRuntimeError::Architecture(cause)))?;
    execute_layerwise_moshi_realtime_with_observation(&mut runtime,state,temporal,driver,context,
        observer,demand,observation)
}
