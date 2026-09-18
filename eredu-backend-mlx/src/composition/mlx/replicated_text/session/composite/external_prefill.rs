//! Typed external captures use the existing paid source and guarded span driver.
use super::*;
use crate::backend::runtime::cache::state::CompletedResidentSource;
use crate::composition::mlx::{
    prepared_speculative::prefill_input,
    speculative::SpeculativeExecutionStreams,
    replicated_text::prediction::phase::external_target::{self, Capture, Output},
};
use eredu_architectures::speculative_execution::{PredictionPrefillPlan,PredictionPrefillSource};
use eredu_runtime::replicated_session::{
    PrefillSpanOperation, PrefillSourceProgress, PrefillScoreLayout, ReplicatedTextSessionError,
};
use std::marker::PhantomData;

type Architecture<A> = PreparedCompositeArchitecture<A>;
type Session<A,D> = ReplicatedTextSession<Architecture<A>,MlxNeuralBackend,
    MlxReplicatedTextMechanisms<Architecture<A>,MlxHybridState>,D>;
type Receiver<'a> = dyn eredu_architectures::external_assistant::ExternalPrefillReceiver<MlxTensor,Error>+'a;
type SpanError = ReplicatedTextSessionError<eredu_nn::Error,Error,Error>;
fn neural(cause:Error,context:SpeculativeExecutionStreams<'_>)->eredu_nn::Error {
    match context.original_numerical() {
        Some((sources,_))=>sources.metadata_funding().metadata_source(cause),
        None=>eredu_nn::Error::backend_retained_source(cause),
    }
}
fn charge(context:SpeculativeExecutionStreams<'_>,bytes:usize)->Result<(),Error>{
    if let Some((sources,_))=context.original_numerical(){
        sources.metadata_funding().reserve_metadata(bytes).map_err(Error::WorkspacePlanning)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<A,D>(session:&mut Session<A,D>,admission:&A::AdmissionConfig,
    processor:&eredu_runtime::SelectedProcessorExecution,input:crate::composition::mlx::MlxModelInput,
    request:&ExternalPredictionCaptureRequest,cache:&mut MlxPredictionTargetState,
    receiver:&mut Receiver<'_>,cancellation:&eredu_core::GenerationCancellationToken,
    context:SpeculativeExecutionStreams<'_>)->Result<PrefillSourceProgress<Option<MlxTensor>>,Error>
where A:CompositeArchitecture<MlxNeuralBackend,MlxHybridState,Error=eredu_nn::Error>+'static,
    A::InputPartPlan:'static,
    D:eredu_runtime::ReplicatedTextExecutionStrategy<Architecture<A>,MlxNeuralBackend,MlxHybridState,
        MlxArchitectureLayerwisePolicy<Architecture<A>,MlxHybridState>,
        MlxArchitectureLayerwisePolicy<Architecture<A>,MlxHybridState>>,
{
    let mut lowerer=MlxCompositePredictionInput::<A>{admission,processor};
    prefill_input::with_source::<Architecture<A>,MlxHybridState,_,_>(&mut lowerer,input,context,|prepared,context|{
        let mut prepared=Some(prepared.and_then(|p|{
            let shape=<_ as PredictionPrefillPlan<Architecture<A>,MlxNeuralBackend,MlxHybridState>>::shape(&p)?;
            let chunk=<_ as PredictionPrefillPlan<Architecture<A>,MlxNeuralBackend,MlxHybridState>>::chunk_positions(&p);
            let whole=<_ as PredictionPrefillPlan<Architecture<A>,MlxNeuralBackend,MlxHybridState>>::requires_whole_input(&p);
            Ok((p,shape,chunk,whole))
        }));
        let facts=prepared.as_ref().and_then(|p|p.as_ref().ok()).map(|(_,shape,chunk,whole)|(*shape,*chunk,*whole));
        let shape=facts.map(|f|f.0);
        let chunk=receiver.prefill_chunk_positions(facts.and_then(|f|f.1),shape.map(|s|s[1]),facts.is_some_and(|f|!f.2));
        let progress=external_state::with_state::<A,D,_>(session,cache,context.target(),Some(context),|session,prior|{
            charge(context,std::mem::size_of::<(
                Span<'_,'_,'_,A,D>,Result<Span<'_,'_,'_,A,D>,Error>,
                PrefillSourceProgress<Option<MlxTensor>>,Result<PrefillSourceProgress<Option<MlxTensor>>,SpanError>,
            )>())?;
            let demand=receiver.output_demand();
            let span=Span::<A,D>{request,receiver,context,prior,
                prompt:shape.map_or(0,|s|s[1]),_types:PhantomData};
            session.try_prefill_unbudgeted_source_with_operation(shape,chunk,demand,|geometry|{
                let (source,_,_,_)=prepared.take().expect("one source factory")
                    .map_err(|e|neural(e,context))?;
                <_ as PredictionPrefillPlan<Architecture<A>,MlxNeuralBackend,MlxHybridState>>::into_source(source,geometry)
            },cancellation,context.target(),&mut eredu_runtime::NoopObserver,span)
            .map_err(|e|external_state::failure(e,Some(context)))
        })?;
        if let Some(Err(cause))=prepared {return Err(cause);}
        Ok(progress)
    })
}

struct Span<'a,'r,'c,A,D> {
    request:&'a ExternalPredictionCaptureRequest,
    receiver:&'a mut Receiver<'r>,
    context:SpeculativeExecutionStreams<'c>,
    prior:&'a mut Option<CompletedResidentSource>,
    prompt:u64,
    _types:PhantomData<fn()->(A,D)>,
}
impl<A,D,P,O> PrefillSpanOperation<Architecture<A>,MlxNeuralBackend,
    MlxReplicatedTextMechanisms<Architecture<A>,MlxHybridState>,D,P,O>
    for Span<'_,'_,'_,A,D>
where A:CompositeArchitecture<MlxNeuralBackend,MlxHybridState,Error=eredu_nn::Error>+'static,
    A::InputPartPlan:'static,
    D:eredu_runtime::ReplicatedTextExecutionStrategy<Architecture<A>,MlxNeuralBackend,MlxHybridState,
        MlxArchitectureLayerwisePolicy<Architecture<A>,MlxHybridState>,
        MlxArchitectureLayerwisePolicy<Architecture<A>,MlxHybridState>>,
    P:PredictionPrefillSource<Architecture<A>,MlxNeuralBackend,MlxHybridState>,
    O:eredu_runtime::ActivationObserver<MlxTensor,eredu_nn::Error>+?Sized,
{
    fn score_layout(&self)->PrefillScoreLayout{
        if self.context.original_external().is_some(){PrefillScoreLayout::SelectedPositions}
        else{PrefillScoreLayout::FinalScores}
    }
    fn prepare(&mut self,source:&P,chunk:&eredu_runtime::prefill::PrefillChunk,_:&Stream)->Result<P::Chunk,eredu_nn::Error>{
        prefill_input::prepare_chunk::<Architecture<A>,MlxHybridState,P>(source,chunk,self.context)
    }
    fn execute<'s>(&mut self,session:&mut Session<A,D>,source:&'s P,prepared:Option<&'s P::Chunk>,
        input:Result<<Architecture<A> as eredu_runtime::LayeredArchitecture<MlxNeuralBackend,MlxHybridState>>::Input<'s>,SpanError>,
        _identity:Option<eredu_runtime::SharedPreparedInputCacheIdentity>,chunk:&eredu_runtime::prefill::PrefillChunk,
        stream:&Stream,_observer:&mut O)->Result<(Option<MlxTensor>,Option<eredu_core::DistributedCommitOutcome>),SpanError>
    {
        let context=self.context;
        let preparation=self.receiver.prepare(chunk).map_err(|e|neural(e,context));
        session.agree_prediction_prefill_preparation(preparation,stream)?;
        let request=self.request;
        let execute=|session:&mut Session<A,D>,capture:Option<&Capture>,metadata:Option<&eredu_nn::workspace::WorkspaceContext>|{
            let capture=capture.ok_or(Error::PrefillScopeUnavailable)?;
            let mut observer=capture.observer();
            let (scores,captured,frontier,commit)=session.prefill_capture_span(input,chunk.output,stream,&mut observer,|forward|{
                let values=capture.values()?;
                let output=match metadata {
                    Some(metadata)=>A::external_prediction_capture_with_metadata(request,forward,values,metadata),
                    None=>A::external_prediction_capture(request,forward,values),
                }?;
                output.ok_or_else(||match metadata {
                    Some(metadata)=>metadata.metadata_error(format_args!("target omitted selected external capture")),
                    None=>eredu_nn::Error::backend("target omitted selected external capture"),
                })
            }).map_err(|e|external_state::failure(e,Some(context)))?;
            Ok(Output{scores,capture:Some(captured),frontier,commit,evidence:None})
        };
        let output=if context.original_external().is_some(){
            charge(context,std::mem::size_of::<(
                [&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence;1],
                eredu_core::speculative::SpeculativePrefillSpan,Output,Result<Output,Error>,
                SpeculativeExecutionStreams<'_>,
            )>()).map_err(|e|SpanError::Architecture(neural(e,context)))?;
            let prepared=prepared.ok_or_else(||SpanError::Architecture(neural(
                Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch),context)))?;
            let evidence=source.completed_token_packet(prepared).and_then(|p|p.evidence())
                .ok_or_else(||SpanError::Architecture(neural(
                    Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch),context)))?;
            let inputs=[evidence];
            let execution=context.with_external_tensor_sources(&inputs)
                .map_err(|e|SpanError::Architecture(neural(e,context)))?;
            let span=eredu_core::speculative::SpeculativePrefillSpan{prompt_tokens:self.prompt,
                input_start:chunk.input.start,input_end:chunk.input.end,position:chunk.position,
                hidden_start:chunk.input.start,token_start:chunk.input.start,
                sequence:chunk.input.end-chunk.input.start,seed_start:chunk.position};
            external_target::run::<A,D,_>(session,source.tokens(prepared),Some(request),
                eredu_runtime::speculative::external_occurrence::ExternalInvocationKind::TargetPrefill,
                Some(span),chunk.output,execution,self.prior,execute)
        }else{
            Capture::prepare::<A>(request,None).and_then(|capture|execute(session,Some(&capture),None))
        }.map_err(|e|SpanError::Architecture(neural(e,context)))?;
        let Output{mut scores,capture,frontier,commit,evidence}=output;
        let consumed=self.receiver.consume_with_evidence(chunk,frontier,&mut scores,capture.ok_or_else(||SpanError::Architecture(neural(Error::PrefillScopeUnavailable,context)))?,evidence)
            .map_err(|e|neural(e,context));
        session.agree_prediction_prefill_preparation(consumed,stream)?;
        Ok((scores,commit))
    }
}
