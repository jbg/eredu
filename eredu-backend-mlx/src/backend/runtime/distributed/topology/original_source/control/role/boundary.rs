//! Explicit model-root loan for the shared PP dependency completion boundary.
use super::*;
use crate::{MlxTensor,backend::{submission_recovery::prefill::TransientRootsProjection,
    runtime::distributed::completion::{OriginalCommunicationCompletion,
        prepared::{PreparedCompletionResources,CompletionResourceLayout}}}};
use eredu_runtime::{ArchitectureBoundaryValue,PreparedBoundarySource};
use safemlx::PreparedNestedRoots;

/// Every clone shares lexical close state. Only weak model roots are retained;
/// an escaped context cannot prolong Q or submit after the shared forward exits.
pub(super) struct ModelBoundaryContext {
    roots:TransientRootsProjection, active:Cell<bool>,
    binding:Option<super::super::super::parallel::OriginalParallelBinding>,
}
impl ModelBoundaryContext {
    pub(super) fn active_binding(&self) -> Option<&super::super::super::parallel::OriginalParallelBinding> {
        self.active.get().then_some(self.binding.as_ref()).flatten()
    }
}
pub(super) struct ModelBoundaryLoan(Option<Rc<ModelBoundaryContext>>);
impl ModelBoundaryLoan {
    pub(super) fn new(roots:Option<TransientRootsProjection>,
        binding:Option<super::super::super::parallel::OriginalParallelBinding>,custody:&Custody)->Result<Self,Error>{
        reserve(&custody.funding,&[size_of::<Self>(),size_of::<Option<TransientRootsProjection>>(),
            size_of::<Option<Rc<ModelBoundaryContext>>>(),size_of::<ModelBoundaryContext>(),
            size_of::<Result<Self,Error>>(),TransientRootsProjection::control_bytes().ok_or_else(overflow)?,
            Layout::new::<[usize;2]>().extend(Layout::new::<ModelBoundaryContext>())
                .map_err(|_|overflow())?.0.pad_to_align().size()])?;
        Ok(Self(roots.map(|roots|Rc::new(ModelBoundaryContext{roots,active:Cell::new(true),binding}))))
    }
    pub(super) fn context(&self)->Option<Rc<ModelBoundaryContext>>{self.0.clone()}
}
impl Drop for ModelBoundaryLoan {
    fn drop(&mut self){if let Some(context)=self.0.take(){context.active.set(false);drop(Rc::into_inner(context));}}
}
struct Values<'a>(std::slice::Iter<'a,ArchitectureBoundaryValue<MlxTensor>>);
impl<'a> Iterator for Values<'a> {
    type Item=&'a safemlx::Array;
    fn next(&mut self)->Option<Self::Item>{self.0.next().map(|value|value.tensor().as_array())}
}
impl OriginalParallelControlProjection {
    pub(super) fn prepare_boundary_encoding_inputs(&self,headers:&super::super::super::OriginalBoundaryHeaders)->Result<(),Error>{
        let model=self.model_roots.as_ref().filter(|model|model.active.get())
            .ok_or_else(||control_error(ControlCause::Identity,&self.custody.source,&self.custody.funding))?;
        let controls=TransientRootsProjection::control_bytes()
            .and_then(|n|n.checked_add(safemlx::OperationEvent::traversal_leaf_control_bytes()?))
            .and_then(|n|n.checked_mul(headers.values().len())).ok_or_else(overflow)?;
        reserve(&self.custody.funding,&[controls])?;
        for header in headers.values(){model.roots.validate_boundary_leaf(header.value().tensor().as_array())?;}
        Ok(())
    }
    pub(super) fn claim_boundary_sources(&self,frames:&eredu_runtime::PreparedBoundaryFrames<MlxTensor>)
        ->Result<super::super::super::parallel::OriginalBoundaryCall,Error>{
        let model=self.model_roots.as_ref().filter(|model|model.active.get())
            .ok_or_else(||control_error(ControlCause::Identity,&self.custody.source,&self.custody.funding))?;
        let binding=model.binding.as_ref().ok_or_else(||control_error(ControlCause::Identity,&self.custody.source,&self.custody.funding))?;
        let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
        let source=owner.owner().request.source.model().ok_or_else(||control_error(ControlCause::Identity,&self.custody.source,&self.custody.funding))?;
        binding.claim_boundaries(source,frames)
    }
    pub(super) fn has_model_boundary(&self)->bool {
        self.model_roots.as_ref().is_some_and(|model|model.active.get())
    }

    pub(crate) fn submit_boundary_dependencies(&self,prepared:&Group,
        values:&[ArchitectureBoundaryValue<MlxTensor>],boundary:&PreparedBoundarySource,
        route:&crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
        stream:&Stream)->Result<OriginalCommunicationCompletion,Error>{
        self.with_context(prepared,|bound|{
            reserve(&self.custody.funding,&[
                size_of::<(&Self,&Group,&[ArchitectureBoundaryValue<MlxTensor>],&PreparedBoundarySource,&Stream)>(),
                size_of::<OriginalParallelControlOwner>(),size_of::<Option<usize>>(),size_of::<Values<'_>>(),
                size_of::<safemlx::OriginalScopeObserver>(),size_of::<safemlx::OperationEvalTraversalLayout>(),
                size_of::<Result<OriginalCommunicationCompletion,Error>>(),
                TransientRootsProjection::boundary_traversal_control_bytes().ok_or_else(overflow)?,
                PreparedNestedRoots::<Custody>::control_bytes(values.len()).ok_or_else(overflow)?,
                PreparedNestedRoots::<Custody>::submission_control_bytes::<Values<'_>>().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?])?;
            let fail=||control_error(ControlCause::Identity,&self.custody.source,&self.custody.funding);
            let model=self.model_roots.as_ref().ok_or_else(fail)?;
            if bound.is_none() || !model.active.get() || values.is_empty()
                || !boundary.source().same_source(&self.custody.source)
                || !boundary.funding().same_account(&self.custody.funding)
                || boundary.descriptor()!=route.descriptor()
                || !matches!(route.endpoint(),Some(crate::backend::runtime::distributed::topology::CommunicationRouteEndpoint::Source)) {
                return Err(fail());
            }
            let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
            let source=owner.owner().request.source.communication_source()?;
            let order=source.source().manifest().routes().iter().position(|entry|entry==boundary.descriptor())
                .ok_or_else(fail)?;
            if !source.matches_route(order,route){return Err(fail());}
            let (observer,traversal)=model.roots.boundary_traversal(values.len(),&self.custody.funding)?;
            let mut roots=PreparedNestedRoots::try_new(values.len(),self.custody.clone()).map_err(|failed|{
                let(cause,custody)=failed.into_parts();
                let error=failure(Cause::BoundaryRoots(cause),&custody.source,&custody.funding);drop(custody);error
            })?;
            let mut ready=PreparedCompletionResources::prepare_original(&source,
                CompletionResourceLayout{arrays:0,counts:&[],groups:0,routes:1,streams:0},traversal)?;
            ready.retain_route(order)?;
            let ready=ready.finish()?;
            ready.submit_model_nested(&source,&observer,stream,&mut roots,Values(values.iter()),values.len())
        })?
    }
}

struct HeaderValues<'a>(std::slice::Iter<'a,super::super::super::OriginalBoundaryHeader>);
impl<'a> Iterator for HeaderValues<'a>{
    type Item=&'a safemlx::Array;
    fn next(&mut self)->Option<Self::Item>{self.0.next().map(|value|value.value().tensor().as_array())}
}
impl OriginalParallelControlProjection {
    /// The ordinary logical-pair receive worker also sends its placeholder.
    /// Settle that actual selected graph under the same model recipe before
    /// its bytes cross into a separate frame role; no new nested grant is made.
    pub(super) fn complete_boundary_placeholder_inputs(&self,headers:&super::super::super::OriginalBoundaryHeaders,
        route:&crate::backend::runtime::distributed::topology::CommunicationRouteRealization,stream:&Stream)->Result<(),Error>{
        let c=&self.custody;
        reserve(&c.funding,&[size_of::<HeaderValues<'_>>(),size_of::<OriginalCommunicationCompletion>(),
            size_of::<Result<OriginalCommunicationCompletion,Error>>(),size_of::<Result<(),Error>>(),
            TransientRootsProjection::boundary_traversal_control_bytes().ok_or_else(overflow)?,
            PreparedNestedRoots::<Custody>::control_bytes(headers.values().len()).ok_or_else(overflow)?,
            PreparedNestedRoots::<Custody>::submission_control_bytes::<HeaderValues<'_>>().ok_or_else(overflow)?,
            failure_control_bytes().and_then(|n|n.checked_mul(2)).ok_or_else(overflow)?])?;
        let fail=||control_error(ControlCause::Identity,&c.source,&c.funding);
        let model=self.model_roots.as_ref().ok_or_else(fail)?;
        if !model.active.get() || headers.values().is_empty() || headers.route()!=route.descriptor().id()
            || !headers.source().same_source(&c.source) || !headers.funding().same_account(&c.funding)
            || !matches!(route.endpoint(),Some(crate::backend::runtime::distributed::topology::CommunicationRouteEndpoint::Destination)) {
            return Err(fail());
        }
        let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
        let source=owner.owner().request.source.communication_source()?;
        let order=source.source().manifest().routes().iter().position(|item|item==route.descriptor()).ok_or_else(fail)?;
        if !source.matches_route(order,route) || source.framed_route_exchange(order)?.is_none(){return Err(fail());}
        let (observer,traversal)=model.roots.boundary_traversal(headers.values().len(),&c.funding)?;
        let mut roots=PreparedNestedRoots::try_new(headers.values().len(),c.clone()).map_err(|failed|{
            let(cause,custody)=failed.into_parts();let error=failure(Cause::BoundaryRoots(cause),&custody.source,&custody.funding);
            drop(custody);error
        })?;
        let mut ready=PreparedCompletionResources::prepare_original(&source,
            CompletionResourceLayout{arrays:0,counts:&[],groups:0,routes:1,streams:0},traversal)?;
        ready.retain_route(order)?;
        let completion=ready.finish()?.submit_model_nested(&source,&observer,stream,&mut roots,HeaderValues(headers.values().iter()),headers.values().len())?;
        let phase=eredu_runtime::DistributedExecutionPhase::BoundarySourceCompletion(headers.route());
        source.authority.wait_with_error(eredu_core::Submission{output:(),completion},CommunicationOperation::SendReceive,phase,Some(headers.route()),
            |cause|eredu_runtime::PartitionExecutionError::PreparedCommunication{operation:CommunicationOperation::SendReceive,
                phase,completion:true,source:failure(Cause::Native(cause),&c.source,&c.funding).into_backend_failure()})
            .map_err(|cause|failure(Cause::BoundaryCommunication(cause),&c.source,&c.funding))
    }
}
