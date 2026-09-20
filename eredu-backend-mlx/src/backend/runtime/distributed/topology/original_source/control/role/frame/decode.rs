//! Actual received bytes, canonical expected header and ordinary typed payload
//! reconstruction share one source-funded numerical role and resolver.
use super::*;
use super::super::super::super::OriginalBoundaryHeader;
use crate::backend::runtime::distributed::completion::PreparedCommunicationHeader;
struct Decoding {
    quote:RetainedPipelineBoundary,
    header:OriginalBoundaryHeader,
    received:Array,
    stream:PreparedStreamCopy<Custody>,
    report:WorkspaceTraceReport,
    context:WorkspaceContext,
    recipe:SpeculativeNumericalRecipe,
    capacity:AgreementCapacity,
    order:usize,
    claim:ParallelControlClaim,
    owner:OriginalParallelControlOwner,
    custody:Custody,
}
impl OriginalParallelControlProjection {
    pub(super) fn decode_boundary_frame(&self,header:OriginalBoundaryHeader,received:Array,
        route:CommunicationRouteId,stream:&Stream,quote:RetainedPipelineBoundary)->Result<(Array,OriginalCommunicationCompletion),Error>{
        let c=&self.custody;
        reserve(&c.funding,&[size_of::<Decoding>(),size_of::<Rc<Decoding>>(),size_of::<ParallelControlClaim>(),
            size_of::<OriginalParallelControlOwner>(),size_of::<Result<(Array,OriginalCommunicationCompletion),Error>>(),
            Layout::new::<[usize;2]>().extend(Layout::new::<Decoding>()).map_err(|_|overflow())?.0.pad_to_align().size(),
            failure_bytes().ok_or_else(overflow)?])?;
        let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
        let claim=owner.owner().request.cursor.try_borrow_mut().map_err(|_|fail(FrameCause::Identity,c))?
            .claim(ParallelControlEvent::Phase(DistributedExecutionPhase::Execution))
            .map_err(|_|fail(FrameCause::Identity,c))?;
        if !self.has_model_boundary() || owner.owner().failed.get() || owner.owner().running.replace(true){
            owner.owner().failed.set(true);return Err(fail(FrameCause::Identity,c));
        }
        let _running=Running{running:&owner.owner().running,failed:&owner.owner().failed};
        let result=(||{
            let source=owner.owner().request.source.communication_source()?;
            let order=source.source().manifest().routes().iter().position(|item|item.id()==route)
                .ok_or_else(||fail(FrameCause::Identity,c))?;
            drop(source);
            let plan=Rc::new(Decoding::prepare(header,received,stream,order,claim,quote,
                OriginalParallelControlOwner(owner.0.clone()),c.clone())?);
            run_native_role_with_pipeline(plan.clone(),plan.capacity,
                Some(safemlx::PreparedPipelineCachePlan::new(plan.recipe.kernels)),
                &owner.owner().bank,&owner.owner().controls,c,
                |value,observer|Ok(value.run(observer))).map_err(|cause|Error::with_original_control_source(cause,false))?
        })();
        if result.is_err(){owner.owner().failed.set(true);}
        result
    }
}
impl Decoding {
    fn prepare(header:OriginalBoundaryHeader,received:Array,stream:&Stream,order:usize,claim:ParallelControlClaim,quote:RetainedPipelineBoundary,
        owner:OriginalParallelControlOwner,custody:Custody)->Result<Self,Error>{
        let funding=&custody.funding;
        reserve(funding,&[size_of::<Self>(),size_of::<WorkspaceContext>(),size_of::<WorkspaceTraceReport>(),
            size_of::<SpeculativeNumericalRecipe>(),size_of::<AgreementCapacity>(),size_of::<ExistingArrayProjection<'_>>(),
            size_of::<[WorkspaceTensor;2]>(),size_of::<(WorkspaceTensor,WorkspaceTensor)>(),size_of::<ResidentExecutionMechanisms>(),
            size_of::<Result<Self,Error>>(),size_of::<StreamCopyPlan<Custody>>(),size_of::<PreparedStreamCopy<Custody>>(),
            Stream::device_type_control_bytes().ok_or_else(overflow)?,failure_bytes().ok_or_else(overflow)?])?;
        let invalid=||fail(FrameCause::Identity,&custody);
        if !quote.value().receiving || quote.value().decoding.is_none()
            || quote.value().header_bytes!=header.value().header().len()
            || quote.value().input.shape()!=header.value().tensor().as_array().shape()
            || quote.value().wire_shape()!=received.shape() || received.dtype()!=safemlx::Dtype::Uint8 || received.ndim()!=1 {
            return Err(fail(FrameCause::Worker,&custody));
        }
        let header_len=i32::try_from(header.value().header().len()).map_err(|_|invalid())?;
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
            let mut projection=ExistingArrayProjection::with_source_count(&context,2)
                .map_err(|cause|fail(FrameCause::Neural(context.metadata_source(cause)),&custody))?;
            let prototype=projection.project(header.value().tensor().as_array()).map_err(|cause|fail(cause.into(),&custody))?;
            let wire=projection.project(&received).map_err(|cause|fail(cause.into(),&custody))?;
            if !projection.is_complete(){return Err(invalid());}
            drop(projection);context.begin_span();
            let (head,payload)=boundary_frame::split(&boundary_frame::Workspace(&context),&wire,header_len,&prototype)
                .map_err(|cause|fail(cause.into(),&custody))?;
            let report=context.finish_report(&[head,payload]).map_err(|cause|fail(cause.into(),&custody))?;
            let recipe=selected_parallel_numerical(&report,2,mechanism,&context).map_err(|cause|fail(cause.into(),&custody))?;
            drop((prototype,wire));
            let runtime=owner.owner().request.source.agreement_inputs().ok_or_else(invalid)?.runtime();
            let backing=OriginalBufferBudget::population_layout(runtime,
                usize::try_from(recipe.storage.mutable_bytes()).map_err(|_|overflow())?,recipe.storage.maximum_births())
                .map_err(|cause|fail(cause.into(),&custody))?;
            reserve(funding,&[usize::try_from(recipe.controls).map_err(|_|overflow())?,
                boundary_frame::control_bytes::<boundary_frame::Native<'_>>().ok_or_else(overflow)?,size_of::<(Array,Array)>(),
                safemlx::OperationEvent::traversal_leaf_control_bytes().and_then(|n|n.checked_mul(3)).ok_or_else(overflow)?])?;
            let selected=quote.value().decode_capacity.ok_or_else(invalid)?;
            let capacity=AgreementCapacity{graph:selected.graph,records:selected.records,backing:selected.backing};
            if !capacity.covers(AgreementCapacity{graph:recipe.graph_capacity,records:recipe.record_capacity,backing:backing.capacity()}) {
                return Err(invalid());
            }
            let stream=StreamCopyPlan::<Custody>::capture(stream).map_err(|cause|fail(cause.into(),&custody))?;
            reserve(funding,&[stream.control_bytes().ok_or_else(overflow)?,stream.native_wrapper_bytes(),stream.owner_node_layout().size(),
                Layout::new::<[usize;2]>().extend(stream.shared_body_layout()).map_err(|_|overflow())?.0.pad_to_align().size()])?;
            let stream=stream.realize(custody.clone()).map_err(|error|{
                let(cause,owner)=error.into_parts();let error=fail(cause.into(),&custody);drop(owner);error
            })?;
            Ok(Self{quote,header,received,stream,report,context,recipe,order,claim,owner,custody,capacity})
        }
    }
    fn run(&self,observer:&OriginalScopeObserver)->Result<(Array,OriginalCommunicationCompletion),Error>{
        let c=&self.custody;let stream=self.stream.as_stream();
        reserve(&c.funding,&[size_of::<Self>(),size_of::<OriginalCommunicationCompletion>(),size_of::<PreparedCommunicationHeader>(),
            size_of::<(Array,Array)>(),size_of::<Result<(Array,OriginalCommunicationCompletion),Error>>(),
            failure_bytes().and_then(|n|n.checked_mul(2)).ok_or_else(overflow)?])?;
        let source=self.owner.owner().request.source.communication_source()?;
        if self.claim.identity()!=self.owner.owner().request.cursor.borrow().identity()
            || self.claim.event()!=ParallelControlEvent::Phase(DistributedExecutionPhase::Execution) {
            return Err(fail(FrameCause::Identity,c));
        }
        let id=source.route(self.order).ok_or_else(||fail(FrameCause::Identity,c))?.0.descriptor().id();
        let header_len=i32::try_from(self.header.value().header().len()).map_err(|_|fail(FrameCause::Identity,c))?;
        safemlx::OperationEvent::validate_traversal_leaf(&self.received,observer).map_err(|cause|fail(cause.into(),c))?;
        let bank=safemlx::OperationEvent::prepare_resident_graph(self.recipe.completion.graph,observer)
            .map_err(|cause|fail(cause.into(),c))?;
        let (header,payload)=boundary_frame::split(&boundary_frame::Native(stream),&self.received,header_len,
            self.header.value().tensor().as_array()).map_err(|cause|fail(cause.into(),c))?;
        drop(bank);
        let mut ready=PreparedCompletionResources::prepare_original(&source,
            CompletionResourceLayout{arrays:0,counts:&[],groups:0,routes:1,streams:0},self.recipe.completion.traversal)?;
        ready.retain_route(self.order)?;
        let header=PreparedCommunicationHeader::prepare(&source,header,self.header.value().header())?;
        let completion=header.submit_with_payload(ready.finish()?,&source,observer,stream,&payload)?;
        let completion=super::transport::wait_retaining(&source,completion,DistributedExecutionPhase::Execution,id,c)?;
        safemlx::OperationEvent::validate_traversal_leaf(&payload,observer).map_err(|cause|fail(cause.into(),c))?;
        Ok((payload,completion))
    }
}
