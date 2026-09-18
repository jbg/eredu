//! Prepared metadata execution through the ordinary temporal/depth traversal.
use eredu_nn::{Error,workspace::{WorkspaceBackend,WorkspaceContext,WorkspaceTensor,WorkspaceMetadataError}};
use eredu_runtime::{DeviceState,LayeredArchitecture,LayerwiseRuntime,ResidentRuntime,ResidentUnitWindow,
    working_memory::{WorkspaceResidentLayerState,WorkspaceSamplingBackend}};
use crate::prepared_execution::{WorkspaceLayerwiseParameters,WorkspaceLayerwisePolicy};
use super::execute_layerwise_moshi_realtime_with_parallel_observation;
use std::mem::{size_of,size_of_val};

type State=DeviceState<WorkspaceBackend,WorkspaceResidentLayerState>;
type Model=crate::moshi::LayeredModel<WorkspaceBackend>;
type Unit=crate::moshi::Unit<WorkspaceBackend>;
type Resident=LayerwiseRuntime<Model,WorkspaceBackend,State,ResidentUnitWindow<Unit>>;
type Layerwise<'a>=LayerwiseRuntime<Model,WorkspaceBackend,State,WorkspaceLayerwisePolicy<'a>>;

/// An exact metadata model prepared before the enclosing frame trace begins.
/// Native owners supply static bindings and either actual resident units or a
/// retained layerwise parameter source. The shared traversal owns all equations.
pub struct PreparedMoshiWorkspaceFrame<'a> {
    runtime:Runtime<'a>,
    parallel:Option<eredu_nn::workspace::WorkspaceParallelContext>,
}
enum Runtime<'a> { Resident(Resident), Layerwise(Layerwise<'a>) }

#[derive(Debug,thiserror::Error)]
#[error("Moshi prepared workspace {stage}: {cause}")]
struct PreparationFailure {stage:&'static str,#[source]cause:Error}
fn source_failure(context:&WorkspaceContext,stage:&'static str,cause:Error)->Error {
    context.metadata_source(PreparationFailure{stage,cause})
}

impl<'a> PreparedMoshiWorkspaceFrame<'a> {
    fn model(source:&crate::moshi::MoshiRealtimeModelSource,
        static_bindings:&mut [eredu_runtime::PreparedParameterBinding<'_,WorkspaceTensor>],
        context:&WorkspaceContext,parallel:bool)->Result<Model,Error> {
        if source.is_partitioned()!=parallel { return Err(context.metadata_error(format_args!(
            "Moshi workspace source partition={} differs from supplied parallel source={parallel}",source.is_partitioned()))); }
        context.charge_metadata(size_of::<(Model,Result<Model,Error>,Self)>())?;
        let mut model=source.workspace_model(context)
            .map_err(|cause|source_failure(context,"static model construction",cause))?;
        eredu_runtime::working_memory::bind_prepared_workspace_parameters(
            model.static_modules_mut(),static_bindings,context, |_| false)
            .map_err(|cause|source_failure(context,"static parameter binding",cause))?;
        Ok(model)
    }

    /// Builds the ordinary resident units and binds each exact existing native
    /// source before any frame input, interpolation or sampling trace begins.
    pub fn resident<F>(source:&crate::moshi::MoshiRealtimeModelSource,
        static_bindings:&mut [eredu_runtime::PreparedParameterBinding<'_,WorkspaceTensor>],
        context:&WorkspaceContext,bind:F)->Result<Self,Error>
    where F:FnMut(usize,eredu_runtime::ExecutionUnitAddress,&mut Unit,&WorkspaceContext)->Result<(),Error> {Self::resident_in(source,static_bindings,context,bind,None)}

    /// Projects the exact local TP model and parameter owners before tracing.
    /// The caller validates the actual fully-local partition source first.
    pub fn resident_parallel<F>(source:&crate::moshi::MoshiRealtimeModelSource,
        static_bindings:&mut [eredu_runtime::PreparedParameterBinding<'_,WorkspaceTensor>],
        context:&WorkspaceContext,bind:F,parallel:eredu_nn::workspace::WorkspaceParallelContext)->Result<Self,Error>
    where F:FnMut(usize,eredu_runtime::ExecutionUnitAddress,&mut Unit,&WorkspaceContext)->Result<(),Error> {
        Self::resident_in(source,static_bindings,context,bind,Some(parallel))
    }

    fn resident_in<F>(source:&crate::moshi::MoshiRealtimeModelSource,
        static_bindings:&mut [eredu_runtime::PreparedParameterBinding<'_,WorkspaceTensor>],
        context:&WorkspaceContext,mut bind:F,parallel:Option<eredu_nn::workspace::WorkspaceParallelContext>)->Result<Self,Error>
    where F:FnMut(usize,eredu_runtime::ExecutionUnitAddress,&mut Unit,&WorkspaceContext)->Result<(),Error> {
        let frames=[size_of::<Self>(),size_of::<F>(),size_of::<ResidentRuntime<Model,WorkspaceBackend,State>>(),
            size_of::<Result<Self,Error>>(),size_of::<Result<Resident,Error>>(),
            size_of::<(usize,usize,usize,eredu_runtime::ExecutionUnitAddress)>(),
            size_of::<Vec<usize>>(),size_of::<eredu_runtime::ExecutionUnitLayout>(),
            size_of::<Result<eredu_runtime::ExecutionUnitLayout,Error>>(),
            size_of::<std::slice::IterMut<'_,Vec<Unit>>>(),size_of::<std::slice::IterMut<'_,Unit>>()];
        context.charge_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?)?;
        let model=Self::model(source,static_bindings,context,parallel.is_some())?;
        let mut runtime=ResidentRuntime::<Model,WorkspaceBackend,State>::new_workspace(model,context)
            .map_err(|cause|source_failure(context,"resident unit construction",cause))?;
        let mut counts=context.metadata_vec(runtime.units().len())?;
        counts.extend(runtime.units().iter().map(Vec::len));
        let layout=eredu_runtime::ExecutionUnitLayout::new_with_metadata(source.execution_graph(),&counts,context)?;
        let mut ordinal=0usize;
        for units in runtime.units_mut() {
            for unit in units {
                let address=layout.address(ordinal).ok_or_else(||context.metadata_error(format_args!(
                    "Moshi workspace unit {ordinal} is absent from its retained layout")))?;
                bind(ordinal,address,unit,context)
                    .map_err(|cause|source_failure(context,"resident unit binding",cause))?;
                ordinal=ordinal.checked_add(1).ok_or(WorkspaceMetadataError::Overflow)?;
            }
        }
        Ok(Self{runtime:Runtime::Resident(runtime.into_layerwise_workspace(context)?),parallel})
    }

    /// Prepares the same model against the actual source-bearing host/disk
    /// policy. Unit acquisition remains in the shared layerwise driver.
    pub fn layerwise(source:&crate::moshi::MoshiRealtimeModelSource,
        static_bindings:&mut [eredu_runtime::PreparedParameterBinding<'_,WorkspaceTensor>],
        parameters:&'a dyn WorkspaceLayerwiseParameters,context:&WorkspaceContext)->Result<Self,Error> {
        Self::layerwise_in(source,static_bindings,parameters,context,None)
    }
    /// Uses the original local TP layerwise parameter source and ordinary acquire driver.
    pub fn layerwise_parallel(source:&crate::moshi::MoshiRealtimeModelSource,
        static_bindings:&mut [eredu_runtime::PreparedParameterBinding<'_,WorkspaceTensor>],
        parameters:&'a dyn WorkspaceLayerwiseParameters,context:&WorkspaceContext,
        parallel:eredu_nn::workspace::WorkspaceParallelContext)->Result<Self,Error> {
        Self::layerwise_in(source,static_bindings,parameters,context,Some(parallel))
    }
    fn layerwise_in(source:&crate::moshi::MoshiRealtimeModelSource,
        static_bindings:&mut [eredu_runtime::PreparedParameterBinding<'_,WorkspaceTensor>],
        parameters:&'a dyn WorkspaceLayerwiseParameters,context:&WorkspaceContext,
        parallel:Option<eredu_nn::workspace::WorkspaceParallelContext>)->Result<Self,Error> {
        let model=Self::model(source,static_bindings,context,parallel.is_some())?;
        let runtime=LayerwiseRuntime::new_workspace_with_policy(model,|layout|
            WorkspaceLayerwisePolicy::for_layout(parameters,layout,context),context)?;
        Ok(Self {runtime:Runtime::Layerwise(runtime),parallel})
    }

    /// Executes the already prepared model without clearing the caller's trace.
    /// Ingress, history interpretation, temporal/depth work and ordered sampling
    /// therefore remain in one complete enclosing workspace report.
    #[allow(clippy::too_many_arguments)]
    pub fn execute<S,O>(&mut self,state:&mut State,temporal:&[WorkspaceTensor],
        driver:&mut eredu_runtime::SequentialDecisionDriver<WorkspaceSamplingBackend,S>,
        context:&WorkspaceContext,observer:&mut O,demand:eredu_core::OutputDemand,
        observation:eredu_runtime::SequentialDecisionObservation)
        ->Result<(Option<WorkspaceTensor>,crate::moshi::ForwardContext<WorkspaceTensor>),Error>
    where S:eredu_runtime::Sampler<WorkspaceSamplingBackend>,
        O:eredu_runtime::ActivationObserver<WorkspaceTensor,Error>+?Sized {
        context.validate_values(temporal)?;
        match &mut self.runtime {
            Runtime::Resident(runtime)=>execute_layerwise_moshi_realtime_with_parallel_observation(
                runtime,state,temporal,driver,context,observer,demand,observation,self.parallel.as_ref())
                .map_err(|cause|context.metadata_source(cause)),
            Runtime::Layerwise(runtime)=>execute_layerwise_moshi_realtime_with_parallel_observation(
                runtime,state,temporal,driver,context,observer,demand,observation,self.parallel.as_ref())
                .map_err(|cause|context.metadata_source(cause)),
        }
    }
}

/// Adapter used directly by the common full-frame coordinator.
pub struct MoshiWorkspaceFrameExecutor<'a,'source,O: ?Sized> {
    frame:&'a mut PreparedMoshiWorkspaceFrame<'source>,
    observer:&'a mut O,
    demand:eredu_core::OutputDemand,
    observation:eredu_runtime::SequentialDecisionObservation,
}
impl<'a,'source,O: ?Sized> MoshiWorkspaceFrameExecutor<'a,'source,O> {
    /// Borrows the prepared model and the exact selected observation policy.
    pub fn new(frame:&'a mut PreparedMoshiWorkspaceFrame<'source>,observer:&'a mut O,
        demand:eredu_core::OutputDemand,observation:eredu_runtime::SequentialDecisionObservation)->Self {
        Self {frame,observer,demand,observation}
    }
}
impl<S,O> eredu_runtime::PreparedRealtimeFrameExecutor<WorkspaceSamplingBackend,S,State>
    for MoshiWorkspaceFrameExecutor<'_,'_,O>
where S:eredu_runtime::Sampler<WorkspaceSamplingBackend>,
    O:eredu_runtime::ActivationObserver<WorkspaceTensor,Error>+?Sized {
    type Error=Error;
    type Retained=(Option<WorkspaceTensor>,crate::moshi::ForwardContext<WorkspaceTensor>);
    fn execute(&mut self,state:&mut State,temporal:&[WorkspaceTensor],
        driver:&mut eredu_runtime::SequentialDecisionDriver<WorkspaceSamplingBackend,S>,
        context:&WorkspaceContext)->Result<Self::Retained,Error> {
        self.frame.execute(state,temporal,driver,context,self.observer,self.demand,self.observation)
    }
}
