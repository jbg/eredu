//! One accepted frame entering the existing coordinator and selected equation.
use super::*;
use super::{input_source::{RealtimeInputSource,RealtimeInputSourceError,RealtimeInputMaterializer},
    retained_frame_source::{CompiledRealtimeFrameSource,RetainedFrameSources,FrameShape},
    original_completion::OriginalRealtimeCompletion,
    original_observation::{RealtimeHostReadPlan,OriginalRealtimeHostObserver}};
use crate::{backend::{nn::workspace::SpeculativeNumericalRecipe,
    submission_recovery::native_role::{self,NativeRoleCapacity,realtime::RealtimeRoleContext}},
    composition::moshi::{OriginalRealtimeModelLease,RealtimeOperationRecipe}};
use eredu_nn::workspace::{WorkspaceContext,HostMetadataFunding};
use eredu_runtime::{RealtimePayloadContract,RealtimeIngressContract,RealtimeFrameCoordinatorError,
    working_memory::{OriginalRealtimeFrame,OriginalHostSourceBank,InferenceExecutionIdentity,WorkingMemoryError}};
use safemlx::{PreparedInputRuntime,PreparedPipelineCachePlan,OriginalBufferBudget};
use std::{cell::RefCell,convert::Infallible,mem::{size_of,size_of_val},time::Duration};

type CoordinatorError=RealtimeFrameCoordinatorError<RealtimeInputSourceError,Error,Error,
    safemlx::error::Exception,Error>;
/// Actual descriptive native capacities. The accepted frame issuer is separate;
/// this object cannot construct a native quota or recover a consumed phase.
pub(super) struct FrameNativeLayout {
    pub(super) capacity:NativeRoleCapacity,
    pipeline:PreparedPipelineCachePlan,
    controls:usize,
}
#[derive(Debug,thiserror::Error)]
#[error("original realtime frame: {cause}")]
struct FrameFailure {
    #[source] cause:Error,
    _funding:Option<HostMetadataFunding>,
    _custody:eredu_runtime::working_memory::OriginalRealtimeBudgetCustody,
}
// The portable coordinator deliberately has a compact model-error display.
// Retain its actual backend cause at this concrete adapter before that label
// would hide a reached native/source failure from the public report.
#[derive(Debug)]
struct ModelExecutionFailure { cause:Error }
impl std::fmt::Display for ModelExecutionFailure {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        write!(f,"realtime prepared model execution failed: {}",self.cause)?;
        let mut source=std::error::Error::source(&self.cause);
        while let Some(cause)=source {
            write!(f,": {cause}")?;
            source=cause.source();
        }
        Ok(())
    }
}
impl std::error::Error for ModelExecutionFailure {
    fn source(&self)->Option<&(dyn std::error::Error+'static)> {Some(&self.cause)}
}
fn coordinator_failure(cause:CoordinatorError,funding:&HostMetadataFunding)->Error {
    match cause {
        CoordinatorError::Model(cause)=>Error::Neural(funding.metadata_source(ModelExecutionFailure{cause})),
        cause=>Error::Neural(funding.metadata_source(cause)),
    }
}
fn retain(cause:Error,custody:eredu_runtime::working_memory::OriginalRealtimeBudgetCustody,
    funding:Option<HostMetadataFunding>)->Error {
    Error::retained_original(eredu_core::SharedBackendFailure::new(eredu_core::BackendFailureKind::Other,
        FrameFailure{cause,_funding:funding,_custody:custody}),false)
}
struct Invocation {
    completion:OriginalRealtimeCompletion,
    model:OriginalRealtimeModelLease,
    operations:RefCell<Option<RealtimeOperationRecipe>>,
    host:RefCell<Option<RealtimeHostReadPlan>>,
    inputs:RefCell<Option<OriginalHostSourceBank>>,
    operation_source:RefCell<Option<OriginalHostSourceBank>>,
    _source_program:Option<eredu_runtime::working_memory::OriginalHostSourceProgramBanks>,
    ingress:RealtimeIngressContract,
    payload:RealtimePayloadContract,
    shape:FrameShape,
    recipe:SpeculativeNumericalRecipe,
    resources:Arc<neutral_moshi::SelectedRealtimeResources>,
    // Cold-account custody retires after all ingress/policy/source descriptors.
    sources:RetainedFrameSources,
    funding:HostMetadataFunding,
}
/// Move-only scheduler preparation. The source issuer and complete native
/// invocation remain together until the scheduler lends this actual branch.
pub struct PreparedOriginalRealtimeFrame {
    account:OriginalRealtimeFrame,
    invocation:Invocation,
    layout:FrameNativeLayout,
}
struct Executor<'a,'r> {
    model:&'a mut MlxRealtimeExecution,
    source:&'a OriginalRealtimeModelLease,
    parallel:Option<&'a crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation>,
    role:&'a RealtimeRoleContext<'r>,
}
impl PreparedRealtimeFrameExecutor<MlxSamplingBackend,GenerationSampler,MlxKeyValueTransactionBranch>
    for Executor<'_,'_> {
    type Error=Error;
    type Retained=(Option<MlxTensor>,eredu_architectures::moshi::ForwardContext<MlxTensor>);
    fn execute(&mut self,state:&mut MlxKeyValueTransactionBranch,temporal:&[MlxTensor],
        driver:&mut eredu_runtime::SequentialDecisionDriver<MlxSamplingBackend,GenerationSampler>,
        context:&Stream)->Result<Self::Retained,Error> {
        self.model.execute_selected_realtime_original(state,temporal,driver,context,self.source,self.role,self.parallel)
    }
}
fn wrapper_controls()->Option<usize> {
    let equation=equation_callback(None);
    let frames=[size_of_val(&equation),size_of::<FrameEquation<'_,'_,'_>>(),size_of::<FrameNativeLayout>(),size_of::<Invocation>(),size_of::<PreparedOriginalRealtimeFrame>(),
        size_of::<Result<PreparedOriginalRealtimeFrame,Error>>(),size_of::<Result<MlxPrepublicationFrame,Error>>(),
        size_of::<Result<Result<MlxPrepublicationFrame,Infallible>,eredu_core::BackendFailure>>(),
        size_of::<Executor<'_,'_>>(),size_of::<CoordinatorError>(),size_of::<RealtimeInputSourceError>(),
        size_of::<RealtimeInputMaterializer<'_>>(),size_of::<Option<OriginalHostSourceBank>>(),
        size_of::<std::cell::RefMut<'_,Option<RealtimeOperationRecipe>>>(),
        size_of::<std::cell::RefMut<'_,Option<RealtimeHostReadPlan>>>(),
        size_of::<std::cell::RefMut<'_,Option<OriginalHostSourceBank>>>(),
        size_of::<OriginalRealtimeFrame>(),size_of::<NativeRoleCapacity>(),
        size_of::<PreparedPipelineCachePlan>(),size_of::<PreparedInputRuntime>(),size_of::<FrameFailure>(),
        eredu_core::SharedBackendFailure::control_bytes::<FrameFailure>()?,
        size_of::<Option<Duration>>(),size_of::<MlxRealtimeFrameTensorMechanisms<'_>>()];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)?
        .checked_add(size_of::<ModelExecutionFailure>())?
        .checked_add(size_of::<Option<&(dyn std::error::Error+'static)>>())?
        .checked_add(size_of::<&mut std::fmt::Formatter<'_>>())?
        .checked_add(size_of::<std::fmt::Result>())?
        .checked_add(size_of::<(CoordinatorError,&HostMetadataFunding)>())?
        .checked_add(WorkspaceContext::metadata_source_bytes::<CoordinatorError>()?.max(
            WorkspaceContext::metadata_source_bytes::<ModelExecutionFailure>()?))?
        .checked_add(WorkspaceContext::metadata_source_bytes::<RealtimeInputSourceError>()?)
}
impl FrameNativeLayout {
    /// This includes native execution, input-adapter and output-completion
    /// controls. Source compilation and portable branch/history controls are
    /// independently retained domains in the enclosing complete frame quote.
    pub(super) fn inspect(source:&CompiledRealtimeFrameSource,runtime:&PreparedInputRuntime,
        input:eredu_runtime::RealtimeIngressSource<'_>)->Result<Self,Error> {
        let recipe=*source.operations.recipe();
        let population=OriginalBufferBudget::population_layout(runtime,
            usize::try_from(recipe.storage.mutable_bytes()).map_err(|_|overflow())?,
            recipe.storage.maximum_births()).map_err(|cause|Error::Neural(
                source.funding().metadata_source(cause)))?;
        let capacity=NativeRoleCapacity{graph:recipe.graph_capacity,records:recipe.record_capacity,
            backing:population.capacity()};
        let pipeline=PreparedPipelineCachePlan::new(recipe.kernels);
        let input=RealtimeInputSource::inspect(runtime,input)
            .map_err(|cause|Error::Neural(source.funding().metadata_source(cause)))?;
        if input.facts()!=source.initialized_inputs {return Err(identity());}
        let callback=worker(None,None,None,None,None);
        let parts=[wrapper_controls().ok_or_else(overflow)?,
            usize::try_from(recipe.controls).map_err(|_|overflow())?,population.control_bytes(),
            usize::try_from(source.operations.control_bytes().ok_or_else(overflow)?).map_err(|_|overflow())?,
            OriginalRealtimeCompletion::control_bytes(recipe.completion,source.completion_roots).ok_or_else(overflow)?,
            OriginalRealtimeModelLease::control_bytes::<MlxPrepublicationFrame>().ok_or_else(overflow)?,
            source.host.control_bytes().ok_or_else(overflow)?,
            if source.parallel.is_some(){MlxRealtimeExecution::original_parallel_control_bytes().ok_or_else(overflow)?}else{0},
            super::source_program::FrameSourceProgram::control_bytes().ok_or_else(overflow)?,
            source.coordinator.control_bytes::<MlxSamplingBackend,GenerationSampler,
                MlxKeyValueTransactionBranch,MlxTensor,MlxRealtimeCompletion,RealtimeInputMaterializer<'_>,
                MlxRealtimeFrameTensorMechanisms<'_>,Executor<'_,'_>,
                super::original_completion::OriginalFrameCompletionMechanism>().ok_or_else(overflow)?,
            super::original_admission::submission_control_bytes().ok_or_else(overflow)?,
            input.control_bytes().map_err(|cause|Error::Neural(source.funding().metadata_source(cause)))?,
            native_role::realtime::control_bytes::<Invocation,MlxPrepublicationFrame,Infallible>(
                runtime,capacity,Some(pipeline),size_of_val(&callback))
                .map_err(|cause|Error::Neural(source.funding().metadata_source(cause)))?];
        let controls=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add).ok_or_else(overflow)?;
        Ok(Self{capacity,pipeline,controls})
    }
    pub(super) fn control_bytes(&self)->usize {self.controls}
}
impl PreparedOriginalRealtimeFrame {
    /// Called only after the complete neutral requirements have been admitted.
    /// The returned funding is also used by the source-aware scheduler branch.
    pub(super) fn accept(mut account:OriginalRealtimeFrame,source:CompiledRealtimeFrameSource,
        layout:FrameNativeLayout,payload:RealtimePayloadContract,model:&MlxRealtimeExecution)
        ->Result<Self,Error> {
        let custody=account.budget_custody();
        let funding:HostMetadataFunding=account.metadata_funding()
            .map_err(|cause|retain(Error::WorkspacePlanning(cause),custody.clone(),None))?.into();
        let failure_funding=funding.clone();
        let result=(|| {
        funding.reserve_metadata(wrapper_controls().ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)?;
        let recipe=*source.operations.recipe();
        if layout.capacity.graph!=recipe.graph_capacity||layout.capacity.records!=recipe.record_capacity{return Err(identity());}
        let completion=OriginalRealtimeCompletion::prepare(recipe.completion,source.completion_roots,
            account.budget_custody(),&funding)?;
        let model_source=OriginalRealtimeModelLease::prepare::<MlxPrepublicationFrame>(model,account.budget_custody(),&funding)?;
        let inputs=account.take_source_constructions().map_err(Error::PrefillControl)?;
        let super::source_program::FrameSourceBanks{input:inputs,operation:operation_source,program:source_program}=
            source.source_program.accept(inputs,&funding)?;
        let (operations,host,sources,ingress,shape)=source.into_parts();
        Ok(Self{account,layout,invocation:Invocation{completion,model:model_source,
            operations:RefCell::new(Some(operations)),host:RefCell::new(Some(host)),inputs:RefCell::new(Some(inputs)),
            operation_source:RefCell::new(operation_source),_source_program:source_program,
            sources,ingress,payload,shape,recipe,resources:model.completion_resources(),funding}})
        })();
        result.map_err(|cause|retain(cause,custody,Some(failure_funding)))
    }
    pub(super) fn funding(&self)->&HostMetadataFunding {&self.invocation.funding}
    /// Preparation copy uses the same frame's independent one-use native phase.
    pub(super) fn claim_preparation(&mut self)
        ->Result<eredu_runtime::working_memory::OriginalRealtimeNative,Error> {
        self.account.claim_preparation_native().map_err(Error::PrefillControl)
    }
    pub(super) fn run(mut self,model:&mut MoshiRealtimeExecution<MlxRealtimeExecution>,
        branch:&mut MlxFrameSessionBranch<Self>,frame:&RealtimeInputFrame,identity_source:&InferenceExecutionIdentity,
        runtime:&PreparedInputRuntime,stream:&Stream,timeout:Option<Duration>)->Result<MlxPrepublicationFrame,Error> {
        let custody=self.account.budget_custody();let failure_funding=self.invocation.funding.clone();
        let result=(|| {
        let occurrence=branch.frame_occurrence().map_err(|cause|Error::Neural(self.funding().metadata_source(cause)))?;
        self.account.validate(identity_source,occurrence).map_err(Error::PrefillControl)?;
        if !self.invocation.shape.matches(frame){return Err(identity());}
        let claim=self.account.claim_native().map_err(Error::PrefillControl)?;
        if usize::try_from(claim.physical_bytes()).ok()!=Some(self.layout.capacity.backing)
            ||usize::try_from(claim.graph_bytes()).ok()!=Some(self.layout.capacity.graph)
            ||usize::try_from(claim.record_bytes()).ok()!=Some(self.layout.capacity.records){return Err(identity());}
        let funding=self.invocation.funding.clone();
        let callback=worker(Some(model.executor_mut()),Some(branch),Some(frame),Some(runtime),Some(stream));
        let result=native_role::realtime::run(self.invocation,claim,runtime,Some(self.layout.pipeline),
            &funding,timeout,callback).map_err(Error::StorageSource)?;
        match result {Ok(value)=>Ok(value),Err(never)=>match never{}}
        })();
        result.map_err(|cause|retain(cause,custody,Some(failure_funding)))
    }
}
fn worker<'a>(mut model:Option<&'a mut MlxRealtimeExecution>,
    mut branch:Option<&'a mut MlxFrameSessionBranch<PreparedOriginalRealtimeFrame>>,
    frame:Option<&'a RealtimeInputFrame>,runtime:Option<&'a PreparedInputRuntime>,stream:Option<&'a Stream>)
    ->impl FnMut(&Invocation,&RealtimeRoleContext<'_>)->Result<Result<MlxPrepublicationFrame,Infallible>,Error>+'a {
    move |invocation,role| {
        let model=model.as_deref_mut().ok_or_else(identity)?;
        let branch=branch.as_deref_mut().ok_or_else(identity)?;
        let frame=frame.ok_or_else(identity)?;let stream=stream.ok_or_else(identity)?;
        let runtime=runtime.ok_or_else(identity)?;
        let input=invocation.ingress.inspect(frame)
            .map_err(|cause|Error::Neural(invocation.funding.metadata_source(cause)))?;
        let input=RealtimeInputSource::inspect(runtime,input)
            .map_err(|cause|Error::Neural(invocation.funding.metadata_source(cause)))?;
        let bank=invocation.inputs.borrow_mut().take().ok_or_else(identity)?;
        let mut inputs=input.accept(bank,role.budget_custody().into(),role.metadata_funding())
            .map_err(|cause|Error::Neural(invocation.funding.metadata_source(cause)))?;
        let host=invocation.host.borrow_mut().take().ok_or_else(identity)?;
        let observer=OriginalRealtimeHostObserver::prepare(host,role)?;
        let validations=invocation.completion.begin(invocation.recipe,role)?;
        let operations=invocation.operations.borrow_mut().take().ok_or_else(identity)?;
        let operation_source=invocation.operation_source.borrow_mut().take();
        let operations=operations.activate(role.claim_operations()?,operation_source)?;
        let mut equation=FrameEquation{invocation,role,branch,frame,stream,inputs:&mut inputs,observer:&observer};
        let result=invocation.model.during(model,role,&mut equation_callback(Some(&mut equation)));
        drop(equation);
        drop(operations);
        // Close graph construction and preserve validation owners before native
        // Recovery seals. A secondary source check never replaces the cause.
        drop(validations);
        match result {
            Err(cause)=>Err(cause),
            Ok(value)=>{inputs.finish().map_err(|cause|Error::Neural(invocation.funding.metadata_source(cause)))?;Ok(Ok(value))}
        }
    }
}
struct FrameEquation<'a,'input,'role> {
    invocation:&'a Invocation,role:&'a RealtimeRoleContext<'role>,
    branch:&'a mut MlxFrameSessionBranch<PreparedOriginalRealtimeFrame>,
    frame:&'a RealtimeInputFrame,stream:&'a Stream,
    inputs:&'a mut RealtimeInputMaterializer<'input>,observer:&'a OriginalRealtimeHostObserver,
}
impl FrameEquation<'_,'_,'_> {
    fn run(&mut self,model:&mut MlxRealtimeExecution)->Result<MlxPrepublicationFrame,Error> {
        let invocation=self.invocation;let stream=self.stream;
        let mut tensors=MlxRealtimeFrameTensorMechanisms::new(stream);
        let mut completion=invocation.completion.mechanism();
        let mut executor=Executor{model,source:&invocation.model,parallel:invocation.sources.parallel.as_ref(),role:self.role};
        let submitted=execute_realtime_frame::<MlxSamplingBackend,_,_,_,_,_,_,_,_>(
            &invocation.ingress,&invocation.payload,self.frame,self.branch.generation_mut(),
            &eredu_architectures::moshi::realtime_decision_execution(),self.inputs,&mut tensors,
            &mut executor,&mut completion,stream)
            .map_err(|cause|coordinator_failure(cause,&invocation.funding))?;
        invocation.completion.finish(self.branch.generation_mut().random_state(),stream)?;
        Ok(PrepublicationRealtimeFrame::new(submitted,MlxRealtimeHostObserver::from_original(self.observer.clone())))
    }
}
fn equation_callback<'a,'source:'a,'input:'source,'role:'source>(mut equation:Option<&'a mut FrameEquation<'source,'input,'role>>)
    ->impl FnMut(&mut MlxRealtimeExecution)->Result<MlxPrepublicationFrame,Error>+'a + use<'a,'source,'input,'role> {
    move |model|equation.as_deref_mut().ok_or_else(identity)?.run(model)
}
fn identity()->Error{Error::PrefillControl(WorkingMemoryError::IdentityMismatch)}
fn overflow()->Error{Error::PrefillControl(WorkingMemoryError::Overflow)}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
