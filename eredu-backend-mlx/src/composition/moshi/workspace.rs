//! Mechanical projections of the actual selected realtime native owners.
use super::*;
use crate::backend::nn::workspace::{ExistingArrayProjection,MetalAllocationFacts};
use crate::backend::runtime::execution::generic::LayerwiseWorkspace;
use eredu_nn::{ParameterMetadataView,ParameterSourceVisitor,workspace::{WorkspaceContext,WorkspaceTensor,WorkspaceMetadataError,WorkspaceParallelContext}};
use std::mem::{size_of,size_of_val};

#[derive(Debug,thiserror::Error)]
#[error("realtime model workspace {stage}: {cause}")]
struct WorkspaceSourceFailure {stage:&'static str,#[source]cause:eredu_nn::Error}
fn source_failure(context:&WorkspaceContext,stage:&'static str,cause:eredu_nn::Error)->Error {
    Error::Neural(context.metadata_source(WorkspaceSourceFailure{stage,cause}))
}

/// One lexical cold visit. Neither the source, projected state nor prepared
/// frame may be used as native submission authority. Physical witnesses remain
/// borrowed from the exact selected model until this callback returns.
pub(crate) trait RealtimeWorkspaceVisitor {
    fn visit(&mut self,frame:&mut moshi::PreparedMoshiWorkspaceFrame<'_>,
        projection:&mut ExistingArrayProjection<'_>,layerwise:Option<&LayerwiseWorkspace>)
        ->Result<(),Error>;
}

/// Mechanical binding selected by the actual policy type. No checkpoint,
/// family, or residency decision is repeated by this visitor.
pub(super) trait RealtimeWorkspacePolicy<A>:eredu_runtime::LayerwisePolicy<MlxNeuralBackend,A::Unit>+Sized
where A:moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend,MlxKeyValueState>+'static,
    A::Error:std::fmt::Display {
    fn with_workspace_frame(runtime:&SelectedLayerwiseRuntime<A,Self>,allocation:MetalAllocationFacts,
        context:&WorkspaceContext,parallel:Option<WorkspaceParallelContext>,visitor:&mut dyn RealtimeWorkspaceVisitor)
        ->Result<(),Error>;
}
impl<A> RealtimeWorkspacePolicy<A> for MlxResidentPolicy<A::Unit>
where A:moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend,MlxKeyValueState>+'static,
    A::Error:std::fmt::Display {
    fn with_workspace_frame(runtime:&SelectedLayerwiseRuntime<A,Self>,_:MetalAllocationFacts,
        context:&WorkspaceContext,parallel:Option<WorkspaceParallelContext>,visitor:&mut dyn RealtimeWorkspaceVisitor)
        ->Result<(),Error> {resident_in(runtime,context,parallel,visitor)}
}
impl<A> RealtimeWorkspacePolicy<A> for MlxLayerwisePolicy<A::Unit,()>
where A:moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend,MlxKeyValueState>+'static,
    A::Error:std::fmt::Display {
    fn with_workspace_frame(runtime:&SelectedLayerwiseRuntime<A,Self>,allocation:MetalAllocationFacts,
        context:&WorkspaceContext,parallel:Option<WorkspaceParallelContext>,visitor:&mut dyn RealtimeWorkspaceVisitor)
        ->Result<(),Error> {layerwise_in(runtime,allocation,context,parallel,visitor)}
}

#[derive(Default)]
struct Count { named:usize,values:usize,overflow:bool }
impl<'a> ParameterSourceVisitor<'a,crate::MlxTensor> for Count {
    fn parameter(&mut self,_:ParameterMetadataView<'a>,_:&'a crate::MlxTensor) {
        match self.named.checked_add(1) {Some(n)=>self.named=n,None=>self.overflow=true}
        self.value();
    }
    fn retained(&mut self,_:&'a crate::MlxTensor) { self.value(); }
}
impl Count {
    fn value(&mut self) {
        match self.values.checked_add(1) {Some(n)=>self.values=n,None=>self.overflow=true}
    }
    fn observe<U:Parameterized<crate::MlxTensor>>(&mut self,source:&U,context:&WorkspaceContext)
        ->Result<(),eredu_nn::Error> {
        source.visit_parameter_sources(self).map_err(|cause|context.metadata_source(cause))?;
        if self.overflow {return Err(WorkspaceMetadataError::Overflow.into());}
        Ok(())
    }
}

struct Fill<'a,'p> {
    projection:&'p mut ExistingArrayProjection<'a>,
    rows:Vec<eredu_runtime::PreparedParameterBinding<'a,WorkspaceTensor>>,
    error:Option<eredu_nn::Error>,
    maximum:usize,
}
impl<'a> ParameterSourceVisitor<'a,crate::MlxTensor> for Fill<'a,'_> {
    fn parameter(&mut self,metadata:ParameterMetadataView<'a>,value:&'a crate::MlxTensor) {
        if self.error.is_some() {return;}
        if self.rows.len()==self.maximum {
            self.error=Some(self.projection.context().metadata_error(format_args!("realtime parameter source exceeds its counted rows: {} of {}",self.rows.len(),self.maximum)));return;
        }
        match self.projection.project(value.as_array()) {
            Ok(value)=>self.rows.push(eredu_runtime::PreparedParameterBinding::new(metadata.id().as_str(),value)),
            Err(cause)=>self.error=Some(cause),
        }
    }
    fn retained(&mut self,value:&'a crate::MlxTensor) {
        if self.error.is_none() {
            if let Err(cause)=self.projection.project(value.as_array()) {self.error=Some(cause);}
        }
    }
}
fn bindings<'a,U:Parameterized<crate::MlxTensor>>(source:&'a U,
    projection:&mut ExistingArrayProjection<'a>)->Result<Vec<eredu_runtime::PreparedParameterBinding<'a,WorkspaceTensor>>,eredu_nn::Error> {
    let context=projection.context();
    let parts=[size_of::<Count>(),size_of::<Fill<'_,'_>>(),
        size_of::<Result<Vec<eredu_runtime::PreparedParameterBinding<'_,WorkspaceTensor>>,eredu_nn::Error>>(),
        size_of::<(&U,&mut ExistingArrayProjection<'_>)>()];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(WorkspaceMetadataError::Overflow)?)?;
    let mut count=Count::default();count.observe(source,context)?;
    let rows=context.metadata_vec(count.named)?;
    let mut fill=Fill {projection,rows,error:None,maximum:count.named};
    source.visit_parameter_sources(&mut fill).map_err(|cause|context.metadata_source(cause))?;
    if let Some(cause)=fill.error {return Err(cause);}
    if fill.rows.len()!=count.named {return Err(context.metadata_error(format_args!("realtime parameter source count changed: {} of {}",fill.rows.len(),count.named)));}
    Ok(fill.rows)
}

fn source_controls<A>(context:&WorkspaceContext)->Result<(),Error> {
    let parts=[size_of::<Count>(),size_of::<moshi::MoshiRealtimeModelSource>(),
        size_of::<moshi::PreparedMoshiWorkspaceFrame<'_>>(),
        size_of::<Result<moshi::PreparedMoshiWorkspaceFrame<'_>,eredu_nn::Error>>(),
        size_of::<ExistingArrayProjection<'_>>(),size_of::<Result<ExistingArrayProjection<'_>,crate::backend::nn::workspace::ProjectionInventoryError>>(),
        size_of::<LayerwiseWorkspace>(),size_of::<Result<LayerwiseWorkspace,Error>>(),
        size_of::<(&A,&WorkspaceContext,&mut dyn RealtimeWorkspaceVisitor)>(),
        size_of::<Option<WorkspaceParallelContext>>(),size_of::<Result<WorkspaceParallelContext,eredu_nn::Error>>(),
        size_of::<(usize,usize,eredu_runtime::ExecutionUnitAddress)>(),size_of::<Result<(),Error>>()];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or_else(||Error::Neural(WorkspaceMetadataError::Overflow.into()))?)
        .map_err(|cause|Error::Neural(cause.into()))
}

pub(super) fn resident<'a,A>(runtime:&'a SelectedLayerwiseRuntime<A,MlxResidentPolicy<A::Unit>>,
    context:&'a WorkspaceContext,visitor:&mut dyn RealtimeWorkspaceVisitor)->Result<(),Error>
where A:moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend,MlxKeyValueState>+'static,
    A::Error:std::fmt::Display {resident_in(runtime,context,None,visitor)}
fn resident_in<'a,A>(runtime:&'a SelectedLayerwiseRuntime<A,MlxResidentPolicy<A::Unit>>,
    context:&'a WorkspaceContext,parallel:Option<WorkspaceParallelContext>,visitor:&mut dyn RealtimeWorkspaceVisitor)->Result<(),Error>
where A:moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend,MlxKeyValueState>+'static,
    A::Error:std::fmt::Display {
    source_controls::<A>(context)?;
    let source=runtime.architecture().realtime_workspace_source();
    let parameters=runtime.policy().parameter_sources().map_err(|cause|Error::Neural(context.metadata_source(cause)))?;
    let mut count=Count::default();
    count.observe(runtime.architecture().static_modules(),context).map_err(Error::Neural)?;
    for ordinal in 0..parameters.layout().len() {
        let address=parameters.layout().address(ordinal).ok_or_else(||Error::Neural(WorkspaceMetadataError::Unqualified.into()))?;
        let unit=parameters.unit(ordinal,address).map_err(|cause|Error::Neural(context.metadata_source(cause)))?;
        count.observe(unit,context).map_err(Error::Neural)?;
    }
    let mut projection=ExistingArrayProjection::with_source_count(context,count.values)
        .map_err(|cause|Error::Neural(context.metadata_source(cause)))?;
    let mut static_bindings=bindings(runtime.architecture().static_modules(),&mut projection)
        .map_err(|cause|source_failure(context,"static parameter projection",cause))?;
    let mut visited=0usize;
    let bind=|ordinal,address,unit:&mut eredu_architectures::moshi::Unit<eredu_nn::workspace::WorkspaceBackend>,context:&WorkspaceContext| {
            let native=parameters.unit(ordinal,address).map_err(|cause|context.metadata_source(cause))?;
            let mut values=bindings(native,&mut projection)?;
            eredu_runtime::working_memory::bind_prepared_workspace_parameters(unit,&mut values,context, |_| false)?;
            visited=visited.checked_add(1).ok_or(WorkspaceMetadataError::Overflow)?;
            Ok(())
        };
    let mut frame=match parallel {
        Some(parallel)=>moshi::PreparedMoshiWorkspaceFrame::resident_parallel(&source,&mut static_bindings,context,bind,parallel),
        None=>moshi::PreparedMoshiWorkspaceFrame::resident(&source,&mut static_bindings,context,bind),
    }.map_err(|cause|source_failure(context,"resident prepared model",cause))?;
    if visited!=parameters.layout().len() {
        return Err(Error::Neural(context.metadata_error(format_args!("realtime resident source unit count changed: {visited} of {}",parameters.layout().len()))));
    }
    if !projection.is_complete() {
        return Err(Error::Neural(context.metadata_error(format_args!("realtime resident parameter source has incomplete native backing"))));
    }
    visitor.visit(&mut frame,&mut projection,None)
}

pub(super) fn layerwise<'a,A>(runtime:&'a SelectedLayerwiseRuntime<A,MlxLayerwisePolicy<A::Unit,()>>,
    allocation:MetalAllocationFacts,context:&'a WorkspaceContext,visitor:&mut dyn RealtimeWorkspaceVisitor)
    ->Result<(),Error>
where A:moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend,MlxKeyValueState>+'static,
    A::Error:std::fmt::Display {layerwise_in(runtime,allocation,context,None,visitor)}
fn layerwise_in<'a,A>(runtime:&'a SelectedLayerwiseRuntime<A,MlxLayerwisePolicy<A::Unit,()>>,
    allocation:MetalAllocationFacts,context:&'a WorkspaceContext,parallel:Option<WorkspaceParallelContext>,visitor:&mut dyn RealtimeWorkspaceVisitor)
    ->Result<(),Error>
where A:moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend,MlxKeyValueState>+'static,
    A::Error:std::fmt::Display {
    source_controls::<A>(context)?;
    let source=runtime.architecture().realtime_workspace_source();
    let parameters=runtime.policy().layerwise_workspace_with_metadata(allocation,context)?;
    parameters.begin_execution_trace(context).map_err(Error::Neural)?;
    let mut count=Count::default();count.observe(runtime.architecture().static_modules(),context).map_err(Error::Neural)?;
    let mut projection=ExistingArrayProjection::with_source_count(context,count.values)
        .map_err(|cause|Error::Neural(context.metadata_source(cause)))?;
    let mut static_bindings=bindings(runtime.architecture().static_modules(),&mut projection)
        .map_err(|cause|source_failure(context,"static parameter projection",cause))?;
    let mut frame=match parallel {
        Some(parallel)=>moshi::PreparedMoshiWorkspaceFrame::layerwise_parallel(&source,&mut static_bindings,&parameters,context,parallel),
        None=>moshi::PreparedMoshiWorkspaceFrame::layerwise(&source,&mut static_bindings,&parameters,context),
    }.map_err(|cause|source_failure(context,"layerwise prepared model",cause))?;
    if !projection.is_complete() {return Err(Error::Neural(context.metadata_error(format_args!("realtime layerwise static source has incomplete native backing"))));}
    visitor.visit(&mut frame,&mut projection,Some(&parameters))
}
