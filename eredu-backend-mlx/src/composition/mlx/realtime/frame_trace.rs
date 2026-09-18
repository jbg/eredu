//! Full shared frame coordinator traced from exact projected state and inputs.
use crate::backend::error::Error;
use eredu_core::{Completion,RealtimeFrameScheduleState,RealtimeSampling};
use eredu_nn::{Tensor,workspace::{WorkspaceContext,WorkspaceTensor,WorkspaceBackend,
    WorkspaceDtype,WorkspaceExistingStorage,WorkspaceMetadataError,WorkspaceTraceReport}};
use eredu_runtime::{DeviceState,RuntimeState,GenerationSampler,RealtimeIngressSource,
    RealtimePayloadContract,RealtimePayloadHistory,RealtimeInputMatrix,RealtimePayloadKind,
    RealtimeHostTokenMaterializer,RealtimeFrameCompletionMechanism,RealtimeCompletionCreationError,
    MaterializedRealtimeInput,CompletedRealtimeFrame,RealtimeFrameExecutionView,
    RealtimeFrameExecutionUpdates,working_memory::{WorkspaceResidentLayerState,
        WorkspaceSamplingBackend,WorkspaceSamplingRandomState,HostSourceConstructionFacts}};
use safemlx::PreparedInputRuntime;
use std::mem::{size_of,size_of_val};

type State=DeviceState<WorkspaceBackend,WorkspaceResidentLayerState>;
type Forward=(Option<WorkspaceTensor>,eredu_architectures::moshi::ForwardContext<WorkspaceTensor>);
const KINDS:[RealtimePayloadKind;3]=[RealtimePayloadKind::InputAudio,
    RealtimePayloadKind::ForcedAudio,RealtimePayloadKind::ForcedText];
fn invalid(context:&WorkspaceContext)->eredu_nn::Error {
    context.metadata_error(format_args!("realtime workspace source differs from the retained frame"))
}
fn reserve(context:&WorkspaceContext,parts:&[usize])->Result<(),eredu_nn::Error> {
    Ok(context.charge_metadata(parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
        .ok_or(WorkspaceMetadataError::Overflow)?)?)
}

/// All portable fields are the actual source of this occurrence. The caller
/// projects tensor/model/random owners without replacing schedule policy.
pub(crate) struct RealtimeFrameTraceSource<'a> {
    pub(crate) ingress:RealtimeIngressSource<'a>,
    pub(crate) payload:&'a RealtimePayloadContract,
    pub(crate) state:&'a mut State,
    pub(crate) history:&'a RealtimePayloadHistory<WorkspaceTensor>,
    pub(crate) schedule:&'a RealtimeFrameScheduleState,
    pub(crate) sampling:RealtimeSampling,
    pub(crate) samplers:&'a [GenerationSampler],
    pub(crate) random:Option<&'a WorkspaceSamplingRandomState>,
    pub(crate) native_alias_bytes:usize,
    pub(crate) random_copy_bytes:Option<usize>,
}
pub(crate) struct RealtimeFrameTrace {
    pub(crate) report:WorkspaceTraceReport,
    pub(crate) host:super::original_observation::RealtimeHostReadPlan,
    pub(crate) completion_roots:usize,
    pub(crate) initialized_inputs:HostSourceConstructionFacts,
    pub(crate) coordinator:eredu_runtime::RealtimeCoordinatorHostSource,
}

/// Exact prospective initialized-leaf descriptors. These source constructors
/// are paid outside the numerical span by the accepted initialized-input bank.
/// No Initialize operation is invented or suppressed inside the equation trace.
struct InputSource<'a> {
    source:RealtimeIngressSource<'a>,
    values:[Option<WorkspaceTensor>;3],
    cursor:usize,
    started:bool,
    context:&'a WorkspaceContext,
}
impl<'a> InputSource<'a> {
    fn prepare(source:RealtimeIngressSource<'a>,runtime:&PreparedInputRuntime,
        context:&'a WorkspaceContext)->Result<Self,eredu_nn::Error> {
        reserve(context,&[size_of::<Self>(),size_of::<Result<Self,eredu_nn::Error>>(),
            size_of::<safemlx::PreparedInputPlan<'_>>(),size_of::<[usize;2]>(),
            size_of::<[i32;2]>(),size_of::<WorkspaceExistingStorage>()])?;
        let mut values=std::array::from_fn(|_|None);
        for (index,kind) in KINDS.into_iter().enumerate() {
            if let Some(matrix)=source.matrix(kind) {
                let shape=matrix.shape();
                let plan=runtime.i32(matrix.values(),&shape).map_err(|cause|context.metadata_source(cause))?;
                let shape=[i32::try_from(shape[0]).map_err(|_|invalid(context))?,
                    i32::try_from(shape[1]).map_err(|_|invalid(context))?];
                let storage=WorkspaceExistingStorage::try_new(Some(
                    u64::try_from(plan.backing_bytes()).map_err(|_|invalid(context))?),context)?;
                values[index]=Some(WorkspaceTensor::existing_with_storage(
                    context.layout(&shape,WorkspaceDtype::Int32)?,&storage,context)?);
            }
        }
        Ok(Self{source,values,cursor:0,started:false,context})
    }
    fn complete(&self)->bool {self.started && self.values.iter().all(Option::is_none)}
}
impl RealtimeHostTokenMaterializer for InputSource<'_> {
    type Tensor=WorkspaceTensor;
    type Error=eredu_nn::Error;
    fn prepare_source(&mut self,source:&RealtimeIngressSource<'_>)->Result<(),Self::Error> {
        if self.started || !self.source.same_source(source) {return Err(invalid(self.context));}
        self.started=true;Ok(())
    }
    fn materialize_matrix(&mut self,matrix:RealtimeInputMatrix<'_>)->Result<WorkspaceTensor,Self::Error> {
        if !self.started {return Err(invalid(self.context));}
        while self.cursor<KINDS.len() && self.source.matrix(KINDS[self.cursor]).is_none() {self.cursor+=1;}
        let index=self.cursor;self.cursor=self.cursor.checked_add(1).ok_or_else(||invalid(self.context))?;
        let same=KINDS.get(index).and_then(|kind|self.source.matrix(*kind)).is_some_and(|expected|
            expected.kind()==matrix.kind() && expected.shape()==matrix.shape()
                && std::ptr::eq(expected.values(),matrix.values()));
        if !same {return Err(invalid(self.context));}
        self.values[index].take().ok_or_else(||invalid(self.context))
    }
    fn materialize_i32(&mut self,_:&[i32],_:[usize;2])->Result<WorkspaceTensor,Self::Error> {
        Err(invalid(self.context))
    }
}

// This private marker describes synchronous metadata only. It never escapes
// into a native session or grants completion/publication of native work.
#[derive(Clone,Copy)]
struct MetadataDone;
impl Completion for MetadataDone {
    type Error=std::convert::Infallible;
    fn is_complete(&self)->Result<bool,Self::Error> {Ok(true)}
    fn wait(&self)->Result<(),Self::Error> {Ok(())}
}
struct CompletionSource<'a> {context:&'a WorkspaceContext,roots:Vec<WorkspaceTensor>,complete:bool}
impl CompletionSource<'_> {
    fn push(&mut self,value:&WorkspaceTensor)->Result<(),eredu_nn::Error> {
        self.context.reserve_metadata_vec(&mut self.roots,1)?;
        self.roots.push(value.clone());Ok(())
    }
    fn state(&mut self,state:&State)->Result<(),eredu_nn::Error> {
        let mut failure=None;
        state.visit_all_retained_values(&mut |value| {
            if failure.is_none() {failure=self.push(value).err();}
        }).map_err(|cause|self.context.metadata_source(cause))?;
        match failure {Some(cause)=>Err(cause),None=>Ok(())}
    }
}
impl RealtimeFrameCompletionMechanism<WorkspaceTensor,State,Forward> for CompletionSource<'_> {
    type Completion=MetadataDone;
    type Error=eredu_nn::Error;
    fn complete(&mut self,input:MaterializedRealtimeInput<WorkspaceTensor>,
        output:&CompletedRealtimeFrame<WorkspaceTensor,WorkspaceTensor>,state:&State,
        history:&RealtimePayloadHistory<WorkspaceTensor>,execution:Option<Forward>)
        ->Result<MetadataDone,RealtimeCompletionCreationError<MetadataDone,eredu_nn::Error>> {
        let result=(|| {
            if self.complete {return Err(invalid(self.context));} self.complete=true;
            self.push(input.input_audio())?;
            for value in input.forced_audio().into_iter().chain(input.forced_text()) {self.push(value)?;}
            for value in [output.text(),output.decision_audio(),output.sampled_audio()] {self.push(value)?;}
            for value in output.aligned_audio().into_iter().chain(output.diagnostics()).chain(history.retained_values()) {
                self.push(value)?;
            }
            self.state(state)?;
            if let Some((text,forward))=execution {
                for value in text.as_ref().into_iter().chain(forward.temporal_mask())
                    .chain(forward.temporal_output()).chain(forward.text_logits()).chain(forward.previous_depth_token()) {
                    self.push(value)?;
                }
            }
            Ok(MetadataDone)
        })();
        result.map_err(RealtimeCompletionCreationError::before_submission)
    }
    fn retained_resources(&self,_:&MetadataDone)->usize {self.roots.len()}
}

pub(crate) fn trace_frame(model:&mut eredu_architectures::moshi::PreparedMoshiWorkspaceFrame<'_>,
    source:RealtimeFrameTraceSource<'_>,runtime:&PreparedInputRuntime,context:&WorkspaceContext)
    ->Result<RealtimeFrameTrace,Error> {
    reserve(context,&[size_of::<RealtimeFrameTraceSource<'_>>(),size_of::<RealtimeFrameTrace>(),
        size_of::<CompletionSource<'_>>(),size_of::<Forward>(),size_of::<MetadataDone>(),
        size_of::<RealtimeFrameExecutionUpdates<WorkspaceTensor,GenerationSampler,WorkspaceSamplingRandomState,MetadataDone>>(),
        size_of::<Result<RealtimeFrameTrace,Error>>()]).map_err(Error::Neural)?;
    let initialized_inputs=super::input_source::RealtimeInputSource::inspect(runtime,source.ingress)
        .map_err(|cause|Error::Neural(context.metadata_source(cause)))?.facts();
    let mut inputs=InputSource::prepare(source.ingress,runtime,context).map_err(Error::Neural)?;
    let mut completion=CompletionSource{context,roots:Vec::new(),complete:false};
    completion.state(source.state).map_err(Error::Neural)?;
    for value in source.history.retained_values().chain(source.random.map(WorkspaceSamplingRandomState::key)) {
        completion.push(value).map_err(Error::Neural)?;
    }
    context.begin_state_span(completion.roots.iter()).map_err(Error::Neural)?;
    completion.roots.clear();
    let coordinator=std::cell::RefCell::new(None);
    let observe_transition=|transition:&eredu_core::RealtimeFrameTransition| {
        if coordinator.borrow().is_some(){return Err(invalid(context));}
        let value=eredu_runtime::RealtimeCoordinatorHostSource::prepare(source.ingress,source.payload,
            source.history,source.schedule,transition,&eredu_architectures::moshi::realtime_decision_execution(),
            source.samplers,source.native_alias_bytes,source.random_copy_bytes)
            .map_err(|cause|context.metadata_source(cause))?;
        *coordinator.borrow_mut()=Some(value);Ok(())
    };
    let view=RealtimeFrameExecutionView::new(source.state,source.history,source.schedule,
        source.sampling,source.samplers,source.random,false);
    let mut updates=RealtimeFrameExecutionUpdates::new();
    let mut observer=eredu_runtime::NoopObserver;
    let executor=eredu_architectures::moshi::MoshiWorkspaceFrameExecutor::new(model,&mut observer,
        eredu_core::OutputDemand::StateOnly,eredu_runtime::SequentialDecisionObservation::Opportunistic);
    let mut executor=transition_source(executor,observe_transition,context).map_err(Error::Neural)?;
    let mut tensors=TensorSource{inner:eredu_runtime::NeuralRealtimeFrameTensorMechanisms::<WorkspaceTensor>::new(context),context};
    let submitted=eredu_runtime::execute_realtime_frame_view::<WorkspaceSamplingBackend,_,_,_,_,_,_,_,_>(
        source.ingress.contract(),source.payload,source.ingress.frame(),view,&mut updates,
        &eredu_architectures::moshi::realtime_decision_execution(),&mut inputs,&mut tensors,
        &mut executor,&mut completion,context).map_err(|cause|Error::Neural(context.metadata_source(cause)))?;
    if !inputs.complete() || !completion.complete {return Err(Error::Neural(invalid(context)));}
    if let Some((_,Some(random)))=updates.sampling_state() {completion.push(random.key()).map_err(Error::Neural)?;}
    let completion_roots=completion.roots.len();
    let host=super::original_observation::RealtimeHostReadPlan::inspect(submitted.frame(),context)?;
    let report=context.finish_report(&completion.roots).map_err(Error::Neural)?;
    drop(executor);
    let coordinator=coordinator.into_inner().ok_or_else(||Error::Neural(invalid(context)))?;
    Ok(RealtimeFrameTrace{report,host,completion_roots,initialized_inputs,coordinator})
}

struct TransitionSource<E,F>{inner:E,observe:F}
impl<B,S,M,E,F> eredu_runtime::PreparedRealtimeFrameExecutor<B,S,M> for TransitionSource<E,F>
where B:eredu_runtime::SamplingBackend,S:eredu_runtime::Sampler<B>,
    E:eredu_runtime::PreparedRealtimeFrameExecutor<B,S,M>,
    F:Fn(&eredu_core::RealtimeFrameTransition)->Result<(),E::Error> {
    type Error=E::Error;type Retained=E::Retained;
    fn validate_transition(&self,transition:&eredu_core::RealtimeFrameTransition)->Result<(),Self::Error> {
        self.inner.validate_transition(transition)?;(self.observe)(transition)
    }
    fn execute(&mut self,state:&mut M,temporal:&[B::Token],
        driver:&mut eredu_runtime::SequentialDecisionDriver<B,S>,context:&B::Context)->Result<Self::Retained,Self::Error> {
        self.inner.execute(state,temporal,driver,context)
    }
}
struct TensorSource<'a>{inner:eredu_runtime::NeuralRealtimeFrameTensorMechanisms<'a,WorkspaceTensor>,context:&'a WorkspaceContext}
impl eredu_runtime::RealtimeFrameTensorMechanisms for TensorSource<'_> {
    type Tensor=WorkspaceTensor;type Error=eredu_nn::Error;
    fn clone_with_host_source(&mut self,value:&WorkspaceTensor,funding:&eredu_core::HostMetadataFunding)
        ->Result<WorkspaceTensor,eredu_core::BackendFailure> {
        <WorkspaceSamplingBackend as eredu_runtime::SamplingBackend>::clone_token_with_host_source(value,funding,self.context)
    }
    fn column(&mut self,value:&WorkspaceTensor,column:usize)->Result<WorkspaceTensor,Self::Error> {self.inner.column(value,column)}
    fn filled_column(&mut self,token:i32,batch:usize)->Result<WorkspaceTensor,Self::Error> {self.inner.filled_column(token,batch)}
    fn stack_columns(&mut self,values:&[WorkspaceTensor],batch:usize)->Result<WorkspaceTensor,Self::Error> {self.inner.stack_columns(values,batch)}
}

fn transition_source<E,F>(inner:E,observe:F,context:&WorkspaceContext)->Result<TransitionSource<E,F>,eredu_nn::Error> {
    reserve(context,&[size_of::<TransitionSource<E,F>>(),size_of::<TensorSource<'_>>(),
        size_of::<std::cell::RefCell<Option<eredu_runtime::RealtimeCoordinatorHostSource>>>(),
        size_of::<Result<TransitionSource<E,F>,eredu_nn::Error>>()])?;
    Ok(TransitionSource{inner,observe})
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
