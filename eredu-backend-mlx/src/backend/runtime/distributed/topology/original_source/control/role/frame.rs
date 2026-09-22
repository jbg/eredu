//! Source-bound canonical frame encoding through the common numerical worker.
use super::*;
use crate::backend::{nn::{boundary_frame,workspace::{ExistingArrayProjection,
    MlxMetalWorkspaceMechanisms,ResidentExecutionMechanisms,selected_parallel_numerical,SpeculativeNumericalRecipe}},
    runtime::distributed::completion::{OriginalCommunicationCompletion,
        prepared::{PreparedCompletionResources,CompletionResourceLayout}}};
use super::super::super::OriginalBoundaryHeaders;
use super::super::super::parallel::RetainedPipelineBoundary;
use eredu_nn::workspace::{WorkspaceContext,WorkspaceTensor,WorkspaceTraceReport};
use eredu_runtime::{CommunicationRouteId,DistributedExecutionPhase,PartitionExecutionError};
use safemlx::{Array,PreparedStreamCopy,StreamCopyPlan};

#[derive(Debug,thiserror::Error)]
enum FrameCause {
    #[error("boundary frame source or execution context differs from its exact packet")]
    Identity,
    #[error("boundary frame execution worker is not yet qualified for this device")]
    Worker,
    #[error(transparent)] Neural(#[from] eredu_nn::Error),
    #[error(transparent)] Native(#[from] safemlx::error::Exception),
    #[error(transparent)] Buffer(#[from] safemlx::OriginalBufferCause),
    #[error(transparent)] Stream(#[from] safemlx::StreamCopyCause),
    #[error(transparent)] Communication(#[from] PartitionExecutionError),
    #[error("boundary frame destination allocation failed")]
    Capacity(#[source] std::collections::TryReserveError),
}
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct FrameFailure { #[source] cause:FrameCause,custody:Custody }
fn fail(cause:FrameCause,custody:&Custody)->Error {
    Error::with_original_control_source(eredu_core::BackendFailure::from_error(
        FrameFailure{cause,custody:custody.clone()}),false)
}
fn failure_bytes()->Option<usize>{
    let parts=[size_of::<FrameFailure>(),size_of::<FrameCause>(),size_of::<Custody>(),
        size_of::<Error>(),eredu_core::BackendFailure::source_retention_peak_bytes::<FrameFailure>()?];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
struct Encoding {
    quotes:Vec<RetainedPipelineBoundary>,
    headers:OriginalBoundaryHeaders,
    stream:PreparedStreamCopy<Custody>,
    report:WorkspaceTraceReport,
    context:WorkspaceContext,
    recipe:SpeculativeNumericalRecipe,
    capacity:AgreementCapacity,
    claim:ParallelControlClaim,
    owner:OriginalParallelControlOwner,
    custody:Custody,
}
impl OriginalParallelControlProjection {
    /// Consume the canonical packet. The existing logical pair worker sends
    /// an encoded frame on both endpoints, including the receiver placeholder.
    pub(crate) fn encode_boundary_frames(&self,prepared:&Group,headers:OriginalBoundaryHeaders,quotes:&[RetainedPipelineBoundary],
        route:&crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
        stream:&Stream)->Result<(OriginalBoundaryHeaders,Vec<Array>),Error>{
        self.with_context(prepared,|bound|{
            reserve(&self.custody.funding,&[size_of::<Encoding>(),size_of::<Rc<Encoding>>(),
                size_of::<OriginalParallelControlOwner>(),size_of::<ParallelControlClaim>(),
                size_of::<Result<(OriginalBoundaryHeaders,Vec<Array>),Error>>(),
                size_of::<Result<Vec<Array>,Error>>(),size_of::<Result<Result<Vec<Array>,Error>,eredu_core::BackendFailure>>(),
                Layout::new::<[usize;2]>().extend(Layout::new::<Encoding>()).map_err(|_|overflow())?.0.pad_to_align().size(),
                failure_bytes().ok_or_else(overflow)?])?;
            if bound.is_none() || headers.route()!=route.descriptor().id()
                || !headers.source().same_source(&self.custody.source)
                || !headers.funding().same_account(&self.custody.funding)
                || !self.has_model_boundary()
                || route.endpoint().is_none() {
                return Err(fail(FrameCause::Identity,&self.custody));
            }
            let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
            let claim=owner.owner().request.cursor.try_borrow_mut().map_err(|_|fail(FrameCause::Identity,&self.custody))?
                .claim(ParallelControlEvent::Phase(DistributedExecutionPhase::BoundarySourceCompletion(headers.route())))
                .map_err(|_|fail(FrameCause::Identity,&self.custody))?;
            if owner.owner().failed.get() || owner.owner().running.replace(true){
                owner.owner().failed.set(true);return Err(fail(FrameCause::Identity,&self.custody));
            }
            let _running=Running{running:&owner.owner().running,failed:&owner.owner().failed};
            let result=(||{
                let source=owner.owner().request.source.communication_source()?;
                let order=source.source().manifest().routes().iter().position(|item|item==route.descriptor())
                    .ok_or_else(||fail(FrameCause::Identity,&self.custody))?;
                if !source.matches_route(order,route){return Err(fail(FrameCause::Identity,&self.custody));}
                drop(source);
                if matches!(route.endpoint(),Some(crate::backend::runtime::distributed::topology::CommunicationRouteEndpoint::Destination)) {
                    self.complete_boundary_placeholder_inputs(&headers,route,stream)?;
                }
                self.prepare_boundary_encoding_inputs(&headers)?;
                let encoding=Encoding::prepare(headers,quotes,stream,claim,
                    OriginalParallelControlOwner(owner.0.clone()),self.custody.clone())?;
                let plan=Rc::new(encoding);
                let output=run_native_role_with_pipeline(plan.clone(),plan.capacity,
                    Some(safemlx::PreparedPipelineCachePlan::new(plan.recipe.kernels)),
                    &owner.owner().native,&self.custody,
                    |plan,observer|Ok(plan.run(observer)))
                    .map_err(|cause|Error::with_original_control_source(cause,false))??;
                // Completed Recovery no longer owns its plan. A surviving
                // alias would mean native custody has not safely retired.
                let plan=Rc::try_unwrap(plan).map_err(|_|fail(FrameCause::Identity,&self.custody))?;
                Ok((plan.headers,output))
            })();
            if result.is_err(){owner.owner().failed.set(true);}
            result
        })?
    }
}
impl Encoding {
    fn prepare(headers:OriginalBoundaryHeaders,quotes:&[RetainedPipelineBoundary],stream:&Stream,claim:ParallelControlClaim,
        owner:OriginalParallelControlOwner,custody:Custody)->Result<Self,Error>{
        let funding=&custody.funding;
        reserve(funding,&[size_of::<Self>(),size_of::<WorkspaceContext>(),size_of::<WorkspaceTraceReport>(),
            size_of::<SpeculativeNumericalRecipe>(),size_of::<AgreementCapacity>(),
            size_of::<ExistingArrayProjection<'_>>(),size_of::<Vec<WorkspaceTensor>>(),
            size_of::<(WorkspaceTensor,WorkspaceTensor)>(),size_of::<ResidentExecutionMechanisms>(),
            size_of::<Result<Self,Error>>(),size_of::<StreamCopyPlan<Custody>>(),
            size_of::<PreparedStreamCopy<Custody>>(),Stream::device_type_control_bytes().ok_or_else(overflow)?,
            failure_bytes().ok_or_else(overflow)?])?;
        let invalid=||fail(FrameCause::Identity,&custody);
        if headers.values().is_empty() || headers.values().len()!=quotes.len() {
            return Err(fail(FrameCause::Worker,&custody));
        }
        #[cfg(all(target_vendor="apple",feature="metal",not(feature="cuda")))]
        let mechanism=MlxMetalWorkspaceMechanisms::current_host().map_err(|cause|fail(cause.into(),&custody))?;
        #[cfg(not(all(target_vendor="apple",feature="metal",not(feature="cuda"))))]
        return Err(fail(FrameCause::Worker,&custody));
        #[cfg(all(target_vendor="apple",feature="metal",not(feature="cuda")))]
        {
            let mechanism=ResidentExecutionMechanisms::from_stream(mechanism,stream,funding)
                .map_err(|cause|fail(cause.into(),&custody))?;
            let context=WorkspaceContext::new_with_metadata_funding(mechanism,funding.clone())
                .map_err(|cause|fail(FrameCause::Neural(cause.into()),&custody))?;
            let source_count=headers.values().len().checked_mul(2).ok_or_else(overflow)?;
            let mut projection=ExistingArrayProjection::with_source_count(&context,source_count)
                .map_err(|cause|fail(FrameCause::Neural(context.metadata_source(cause)),&custody))?;
            let mut sources=context.metadata_vec(headers.values().len()).map_err(|cause|fail(FrameCause::Neural(cause.into()),&custody))?;
            for (header,quote) in headers.values().iter().zip(quotes){
                let quote=quote.value();let actual=header.value().tensor().as_array();
                if quote.route!=headers.route() || quote.header_bytes!=header.value().header().len()
                    || quote.input.shape()!=actual.shape() || crate::backend::nn::workspace::byte_view::Dtype::from_layout(quote.input.as_view())
                        .is_none_or(|dtype|dtype.native()!=actual.dtype()) {return Err(invalid());}
                header.header().observe().map_err(|cause|fail(cause.into(),&custody))?;
                let input=projection.project(header.value().tensor().as_array()).map_err(|cause|fail(cause.into(),&custody))?;
                let bytes=projection.project(header.header().array()).map_err(|cause|fail(cause.into(),&custody))?;
                sources.push((input,bytes));
            }
            if !projection.is_complete(){return Err(invalid());}
            drop(projection);
            context.begin_span();
            let mut outputs=context.metadata_vec(sources.len()).map_err(|cause|fail(FrameCause::Neural(cause.into()),&custody))?;
            for (input,header) in &sources {
                outputs.push(boundary_frame::encode(&boundary_frame::Workspace(&context),input,header)
                    .map_err(|cause|fail(cause.into(),&custody))?);
            }
            let report=context.finish_report(&outputs).map_err(|cause|fail(cause.into(),&custody))?;
            let recipe=selected_parallel_numerical(&report,outputs.len(),mechanism,&context)
                .map_err(|cause|fail(cause.into(),&custody))?;
            drop((outputs,sources));
            let runtime=owner.owner().request.source.agreement_inputs().ok_or_else(invalid)?.runtime();
            let backing=OriginalBufferBudget::population_layout(runtime,
                usize::try_from(recipe.storage.mutable_bytes()).map_err(|_|overflow())?,recipe.storage.maximum_births())
                .map_err(|cause|fail(cause.into(),&custody))?;
            reserve(funding,&[usize::try_from(recipe.controls).map_err(|_|overflow())?,
                boundary_frame::control_bytes::<boundary_frame::Native<'_>>().and_then(|n|n.checked_mul(headers.values().len())).ok_or_else(overflow)?,
                Layout::array::<Array>(headers.values().len()).map_err(|_|overflow())?.size(),
                size_of::<Vec<Array>>(),size_of::<Result<(),std::collections::TryReserveError>>(),
                safemlx::OperationEvent::traversal_leaf_control_bytes().and_then(|n|n.checked_mul(source_count)).ok_or_else(overflow)?])?;
            let actual=AgreementCapacity{graph:recipe.graph_capacity,records:recipe.record_capacity,backing:backing.capacity()};
            let mut capacity=AgreementCapacity{graph:0,records:0,backing:0};
            for quote in quotes {
                let selected=quote.value().encode_capacity;
                capacity.graph=capacity.graph.checked_add(selected.graph).ok_or_else(overflow)?;
                capacity.records=capacity.records.checked_add(selected.records).ok_or_else(overflow)?;
                capacity.backing=capacity.backing.checked_add(selected.backing).ok_or_else(overflow)?;
            }
            if !capacity.covers(actual){return Err(invalid());}
            reserve(funding,&[size_of::<Vec<RetainedPipelineBoundary>>(),
                Layout::array::<RetainedPipelineBoundary>(quotes.len()).map_err(|_|overflow())?.size(),
                size_of::<AgreementCapacity>()*2,size_of::<Result<(),std::collections::TryReserveError>>()])?;
            let mut retained=Vec::new();retained.try_reserve_exact(quotes.len()).map_err(|cause|fail(FrameCause::Capacity(cause),&custody))?;
            retained.extend(quotes.iter().cloned());
            let stream=StreamCopyPlan::<Custody>::capture(stream).map_err(|cause|fail(cause.into(),&custody))?;
            reserve(funding,&[stream.control_bytes().ok_or_else(overflow)?,stream.native_wrapper_bytes(),stream.owner_node_layout().size(),
                Layout::new::<[usize;2]>().extend(stream.shared_body_layout()).map_err(|_|overflow())?.0.pad_to_align().size()])?;
            let stream=stream.realize(custody.clone()).map_err(|error|{
                let(cause,owner)=error.into_parts();let error=fail(cause.into(),&custody);drop(owner);error
            })?;
            Ok(Self{quotes:retained,headers,stream,report,context,recipe,capacity,
                claim,owner,custody})
        }
    }
    fn run(&self,observer:&OriginalScopeObserver)->Result<Vec<Array>,Error>{
        let c=&self.custody;let stream=self.stream.as_stream();
        reserve(&c.funding,&[size_of::<Vec<Array>>(),size_of::<OriginalCommunicationCompletion>(),
            size_of::<Result<Vec<Array>,Error>>(),size_of::<Result<OriginalCommunicationCompletion,Error>>(),
            size_of::<PartitionExecutionError>(),failure_bytes().and_then(|n|n.checked_mul(2)).ok_or_else(overflow)?])?;
        if self.claim.identity()!=self.owner.owner().request.cursor.borrow().identity()
            || self.claim.event()!=ParallelControlEvent::Phase(DistributedExecutionPhase::BoundarySourceCompletion(self.headers.route())) {
            return Err(fail(FrameCause::Identity,c));
        }
        let source=self.owner.owner().request.source.communication_source()?;
        let mut ready=PreparedCompletionResources::prepare_original(&source,
            CompletionResourceLayout{arrays:0,counts:&[],groups:0,routes:1,streams:0},self.recipe.completion.traversal)?;
        let order=source.source().manifest().routes().iter().position(|route|route.id()==self.headers.route())
            .ok_or_else(||fail(FrameCause::Identity,c))?;
        ready.retain_route(order)?;
        let ready=ready.finish()?;
        let mut outputs=Vec::new();outputs.try_reserve_exact(self.headers.values().len())
            .map_err(|cause|fail(FrameCause::Capacity(cause),c))?;
        for header in self.headers.values(){
            safemlx::OperationEvent::validate_traversal_leaf(header.value().tensor().as_array(),observer)
                .map_err(|cause|fail(cause.into(),c))?;
            safemlx::OperationEvent::validate_traversal_leaf(header.header().array(),observer)
                .map_err(|cause|fail(cause.into(),c))?;
        }
        let bank=safemlx::OperationEvent::prepare_resident_graph(self.recipe.completion.graph,observer)
            .map_err(|cause|fail(cause.into(),c))?;
        for header in self.headers.values(){
            outputs.push(boundary_frame::encode(&boundary_frame::Native(stream),
                header.value().tensor().as_array(),header.header().array()).map_err(|cause|fail(cause.into(),c))?);
        }
        drop(bank);
        let phase=DistributedExecutionPhase::BoundarySourceCompletion(self.headers.route());
        let completion=ready.submit_original(&source,observer,stream,&outputs)?;
        let outputs=source.authority.wait_with_error(eredu_core::Submission{output:outputs,completion},
            CommunicationOperation::SendReceive,phase,Some(self.headers.route()),|cause|PartitionExecutionError::PreparedCommunication{
                operation:CommunicationOperation::SendReceive,phase,completion:true,
                source:fail(FrameCause::Native(cause),c).into_backend_failure(),
            }).map_err(|cause|fail(cause.into(),c))?;
        for output in &outputs{
            safemlx::OperationEvent::validate_traversal_leaf(output,observer).map_err(|cause|fail(cause.into(),c))?;
        }
        Ok(outputs)
    }
}

mod transport;

mod decode;

impl OriginalParallelControlProjection {
    pub(crate) fn transfer_boundary_frames(&self,prepared:&Group,
        frames:eredu_runtime::PreparedBoundaryFrames<crate::MlxTensor>,
        route:&crate::backend::runtime::distributed::topology::CommunicationRouteRealization,stream:&Stream)
        ->Result<eredu_core::Submission<Vec<crate::MlxTensor>,crate::backend::runtime::distributed::completion::MlxNeuralCommunicationCompletion>,Error>{
        let c=&self.custody;
        reserve(&c.funding,&[size_of::<eredu_runtime::PreparedBoundaryFrames<crate::MlxTensor>>(),
            size_of::<OriginalBoundaryHeaders>(),size_of::<Vec<Array>>(),size_of::<Vec<crate::MlxTensor>>(),
            size_of::<Option<OriginalCommunicationCompletion>>(),size_of::<Result<(OriginalBoundaryHeaders,Vec<Array>),Error>>(),
            size_of::<std::vec::IntoIter<Array>>(),size_of::<std::vec::IntoIter<super::super::super::OriginalBoundaryHeader>>(),
            OriginalScopeObserver::control_bytes().ok_or_else(overflow)?,failure_bytes().ok_or_else(overflow)?])?;
        self.with_context(prepared,|bound|{
            if bound.is_none() || frames.values().is_empty() || !frames.source().same_source(&c.source)
                || !frames.funding().same_account(&c.funding) || frames.route()!=route.descriptor().id() || !self.has_model_boundary(){
                return Err(fail(FrameCause::Identity,c));
            }
            let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
            let source=owner.owner().request.source.communication_source()?;
            let order=source.source().manifest().routes().iter().position(|item|item==route.descriptor())
                .ok_or_else(||fail(FrameCause::Identity,c))?;
            if !source.matches_route(order,route) || source.framed_route_exchange(order)?.is_none(){return Err(fail(FrameCause::Identity,c));}
            Ok(())
        })??;
        let boundary_call=self.claim_boundary_sources(&frames)?;
        let observer=OriginalScopeObserver::require_current().map_err(|cause|fail(cause.into(),c))?;
        let headers=self.prepare_boundary_headers(prepared,frames,route,&observer)?;
        let (headers,frames)=self.encode_boundary_frames(prepared,headers,boundary_call.quotes(),route,stream)?;
        if headers.values().len()!=frames.len(){return Err(fail(FrameCause::Identity,c));}
        reserve(&c.funding,&[Layout::array::<crate::MlxTensor>(frames.len()).map_err(|_|overflow())?.size(),
            size_of::<Result<(),std::collections::TryReserveError>>()])?;
        let mut output=Vec::new();output.try_reserve_exact(frames.len()).map_err(|cause|fail(FrameCause::Capacity(cause),c))?;
        let receiving=matches!(route.endpoint(),Some(crate::backend::runtime::distributed::topology::CommunicationRouteEndpoint::Destination));
        let id=headers.route();let (headers,source,funding)=headers.into_parts();
        let mut completion=None;
        for ((header,frame),quote) in headers.into_iter().zip(frames).zip(boundary_call.quotes()){
            let (received,settled)=self.exchange_boundary_frame(frame,id,quote.clone())?;
            if receiving {
                let (payload,decoded)=self.decode_boundary_frame(header,received,id,stream,quote.clone())?;
                drop(settled);output.push(crate::MlxTensor::from_array(payload));completion=Some(decoded);
            } else {
                drop(received);let (native,value)=header.into_parts();let (_,logical)=value.into_parts();
                drop(native);output.push(logical);completion=Some(settled);
            }
        }
        let completion=completion.ok_or_else(||fail(FrameCause::Identity,c))?;
        drop((source,funding));
        Ok(eredu_core::Submission{output,completion:completion.into()})
    }
}
