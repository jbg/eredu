//! One actual shared-session agreement, bound to its retained request source.
//! The enclosing request supplies the admitted native role; this worker issues
//! no Graph/Record/buffer grant and never substitutes the model-forward scope.
use super::*;
use super::{agreement::AgreementCapacity,parallel::OriginalParallelSource};
use crate::backend::runtime::distributed::completion::{OriginalCommunicationBool,MlxNeuralCommunicationCompletion};
use eredu_runtime::replicated_session::{ParallelControlClaim,ParallelControlCursor,ParallelControlEvent,ParallelControlIdentity};
use safemlx::{OriginalScopeObserver,StreamCopyPlan};
use std::{alloc::Layout,cell::{Cell,OnceCell,RefCell},rc::{Rc,Weak}};
mod source;
use source::ControlSource;
pub(crate) use source::OriginalParallelControlSource;

#[derive(Debug,thiserror::Error)]
pub(crate) enum ControlCause {
    #[error("parallel control request or occurrence differs from its exact source")]
    Identity,
    #[error("parallel control occurrence has no admitted scope")]
    Unbound,
    #[error("parallel control construction is closed or its owner has retired")]
    Closed,
    #[error("parallel control occurrence has already been attempted")]
    Exhausted,
    #[error("parallel control callback did not consume its agreement")]
    Incomplete,
    #[error(transparent)]
    WorkingMemory(eredu_runtime::working_memory::WorkingMemoryError),
}
fn overflow()->Error{Error::WorkspacePlanning(HostMetadataFundingError::Overflow)}
fn reserve(funding:&HostMetadataFunding,parts:&[usize])->Result<(),Error>{
    funding.reserve_metadata(parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)
}
fn control_error(cause:ControlCause,source:&RetainedCommunicationSource,funding:&HostMetadataFunding)->Error{
    failure(Cause::Control(cause),source,funding)
}

/// Exactly one cursor lives in request control ownership, outside state copies.
/// A prepared source can create this owner once; dropping it cannot reissue it.
impl OriginalParallelSource {
    /// Read both actual immutable status programs before selected native storage
    /// is admitted. This issues no occurrence and creates no native owner.
    pub(crate) fn control_capacity(&self)->Result<AgreementCapacity,Error>{
        reserve(self.funding(),&[size_of::<&Self>(),size_of::<AgreementCapacity>(),
            size_of::<Result<AgreementCapacity,Error>>(),
            size_of::<Option<&agreement::OriginalAgreementInputs>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        let source=self.communication_source()?;
        let inputs=self.agreement_inputs().ok_or_else(||failure(Cause::Resource,source.source(),self.funding()))?;
        source::model_agreement_capacity(&source,inputs)
    }
}

pub(crate) struct OriginalParallelControlRequest {
    cursor:RefCell<ParallelControlCursor>,
    source:ControlSource,
    retained:RetainedCommunicationSource,
    fallback:eredu_core::SharedBackendFailure,
    funding:HostMetadataFunding,
}
impl OriginalParallelControlRequest {
    pub(crate) fn new(source:OriginalParallelSource)->Result<Self,Error>{
        Self::from_source(ControlSource::Model(source))
    }
    fn from_source(source:ControlSource)->Result<Self,Error>{
        reserve(source.funding(),&[size_of::<Self>(),size_of::<Result<Self,Error>>(),
            size_of::<RefCell<ParallelControlCursor>>(),ParallelControlCursor::control_bytes().ok_or_else(overflow)?,
            eredu_core::SharedBackendFailure::control_bytes::<Failure>().ok_or_else(overflow)?,
            size_of::<eredu_core::SharedBackendFailure>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        source.take_control_owner()?;
        let actual=source.communication_source()?;
        let retained=actual.source().clone();
        let funding=source.funding().clone();
        let cursor=ParallelControlCursor::new()
            .map_err(|cause|control_error(ControlCause::WorkingMemory(cause),&retained,&funding))?;
        let fallback=eredu_core::SharedBackendFailure::new(eredu_core::BackendFailureKind::Other,
            Failure{cause:Cause::Control(ControlCause::Closed),_source:retained.clone(),funding:funding.clone()});
        drop(actual);
        Ok(Self{cursor:RefCell::new(cursor),source,retained,fallback,funding})
    }
    pub(crate) fn prepare(&self,event:ParallelControlEvent,target:Option<&Group>)->Result<OriginalParallelControlInvocation,Error>{
        // The fixed claim is consumed before any fallible source, funding or
        // native query. A refused attempt cannot rewind this cursor.
        let mut cursor=self.cursor.try_borrow_mut()
            .map_err(|_|Error::with_original_control_source(self.fallback.retained().into_failure(),false))?;
        let identity=cursor.identity();
        let claim=cursor.claim(event)
            .map_err(|_|Error::with_original_control_source(self.fallback.retained().into_failure(),false))?;
        drop(cursor);
        reserve(&self.funding,&[size_of::<Self>(),size_of::<ParallelControlClaim>(),
            size_of::<ParallelControlIdentity>(),size_of::<ParallelControlEvent>(),
            size_of::<OriginalParallelControlInvocation>(),size_of::<State>(),size_of::<OriginalControlBinding>(),
            size_of::<Result<OriginalParallelControlInvocation,Error>>(),
            size_of::<Option<Rc<State>>>(),size_of::<OnceCell<Bound>>(),size_of::<Bound>(),
            size_of::<std::cell::RefMut<'_,ParallelControlCursor>>(),
            size_of::<Result<std::cell::RefMut<'_,ParallelControlCursor>,std::cell::BorrowMutError>>(),
            size_of::<Result<ParallelControlClaim,eredu_runtime::working_memory::WorkingMemoryError>>(),
            size_of::<Option<&agreement::OriginalAgreementInputs>>(),
            size_of::<Option<(&Group,&CommunicationGroupDescriptor,bool)>>(),
            size_of::<[AgreementCapacity;2]>(),size_of::<ControlSource>(),
            size_of::<RetainedCommunicationSource>(),size_of::<eredu_core::SharedBackendFailure>(),
            Layout::new::<[usize;2]>().extend(Layout::new::<State>()).map_err(|_|overflow())?.0.pad_to_align().size(),
            CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        let source=self.source.communication_source()?;
        let inputs=self.source.agreement_inputs()
            .ok_or_else(||control_error(ControlCause::Identity,&self.retained,&self.funding))?;
        reserve(&self.funding,&[size_of::<Option<&Group>>(),size_of::<CollectiveGroupId>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_,CommunicationGroupDescriptor>>>(),
            size_of::<Option<CollectiveGroupId>>()])?;
        let target_id=match target {
            Some(group)=>source.source().manifest().groups().iter().enumerate()
                .find_map(|(order,descriptor)|source.matches_group(order,group).then_some(descriptor.id()))
                .ok_or_else(||control_error(ControlCause::Identity,&self.retained,&self.funding))?,
            None=>inputs.group(),
        };
        // Source-only control owners retain their one declared protocol group.
        // A model source may lend another actual initialized agreement group.
        if self.source.model().is_none() && target_id!=inputs.group(){
            return Err(control_error(ControlCause::Identity,&self.retained,&self.funding));
        }
        let selected=source.source().manifest().select_group_operation(target_id,CommunicationOperation::FailureAgreement)
            .map_err(|cause|failure(Cause::Rank(cause),&self.retained,&self.funding))?;
        let group=source.group(selected.order())
            .ok_or_else(||control_error(ControlCause::Identity,&self.retained,&self.funding))?.0;
        if group.has_original_parallel() || group.has_original_control() {
            return Err(control_error(ControlCause::Identity,&self.retained,&self.funding));
        }
        // Both branches are real, immutable source leaves. Their one-operation
        // union prices either status; no phase count or unknown DAG is inferred.
        let capacity=source::agreement_capacity_for_group(&source,inputs,target_id)?;
        reserve(&self.funding,&[group.retention_copy_bytes().and_then(|n|n.checked_mul(2)).ok_or_else(overflow)?])?;
        let group=group.try_copy_for_retention().map_err(|_|failure(Cause::Resource,&self.retained,&self.funding))?;
        let context=group.try_copy_for_retention().map_err(|_|failure(Cause::Resource,&self.retained,&self.funding))?;
        let state=Rc::new(State{group,group_order:selected.order(),source:self.source.clone(),claim,identity,capacity,
            bound:OnceCell::new(),consumed:Cell::new(false),closed:Cell::new(false),
            retained:self.retained.clone(),funding:self.funding.clone()});
        let context=context.with_original_control(OriginalControlBinding{state:Rc::downgrade(&state),
            retained:self.retained.clone(),fallback:self.fallback.retained(),funding:self.funding.clone()});
        Ok(OriginalParallelControlInvocation{context,state:Some(state),funding:self.funding.clone()})
    }
}
struct Bound {observer:OriginalScopeObserver,stream:StreamCopyPlan<()>}
struct State {
    group:Group,
    group_order:usize,
    source:ControlSource,
    claim:ParallelControlClaim,
    identity:ParallelControlIdentity,
    capacity:AgreementCapacity,
    bound:OnceCell<Bound>,
    consumed:Cell<bool>,
    closed:Cell<bool>,
    retained:RetainedCommunicationSource,
    funding:HostMetadataFunding,
}
/// Group clones carry only a weak occurrence loan and account-only custody.
/// They cannot extend native ownership, rebind a cursor, or find a latest scope.
pub(crate) struct OriginalControlBinding {
    state:Weak<State>,
    retained:RetainedCommunicationSource,
    fallback:eredu_core::SharedBackendFailure,
    funding:HostMetadataFunding,
}
impl Clone for OriginalControlBinding {
    fn clone(&self)->Self {
        Self{state:self.state.clone(),retained:self.retained.clone(),
            fallback:self.fallback.retained(),funding:self.funding.clone()}
    }
}
/// The admitted request's Recovery owns this through completion or quarantine.
pub(crate) struct OriginalParallelControlInvocation {
    context:Group,
    state:Option<Rc<State>>,
    funding:HostMetadataFunding,
}
impl Drop for OriginalParallelControlInvocation {
    fn drop(&mut self){if let Some(state)=self.state.take(){drop(Rc::into_inner(state));}}
}
impl OriginalParallelControlInvocation {
    fn state(&self)->&State{self.state.as_deref().expect("live control occurrence")}
    pub(crate) fn capacity(&self)->AgreementCapacity{self.state().capacity}
    pub(crate) fn claim(&self)->&ParallelControlClaim{&self.state().claim}
    /// The observer is already admitted with capacity() by the request's
    /// control-role producer. Native constructors independently validate it.
    pub(crate) fn with_context<T,E,F>(&self,observer:&OriginalScopeObserver,stream:&Stream,run:F)
        ->Result<Result<T,E>,Error>
    where F:FnOnce(&Group,&HostMetadataFunding)->Result<T,E>, {
        let state=self.state();
        reserve(&self.funding,&[size_of::<F>(),size_of::<T>(),size_of::<E>(),
            size_of::<Result<T,E>>(),size_of::<Result<Result<T,E>,Error>>(),
            size_of::<(&Self,&OriginalScopeObserver,&Stream)>(),size_of::<Close<'_>>(),
            size_of::<Result<(),Bound>>(),size_of::<StreamCopyPlan<()>>(),
            size_of::<Result<StreamCopyPlan<()>,safemlx::StreamCopyCause>>(),
            OriginalScopeObserver::control_bytes().and_then(|n|n.checked_mul(2)).ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?])?;
        if state.closed.get() || state.bound.get().is_some() || state.claim.identity()!=state.identity {
            return Err(control_error(ControlCause::Closed,&state.retained,&self.funding));
        }
        let current=OriginalScopeObserver::require_current()
            .map_err(|cause|failure(Cause::Native(cause),&state.retained,&self.funding))?;
        if !current.same_scope(observer){return Err(control_error(ControlCause::Identity,&state.retained,&self.funding));}
        let stream=StreamCopyPlan::capture(stream).map_err(|_|control_error(ControlCause::Identity,&state.retained,&self.funding))?;
        state.bound.set(Bound{observer:observer.clone(),stream})
            .map_err(|_|control_error(ControlCause::Closed,&state.retained,&self.funding))?;
        let _close=Close(&state.closed);
        let result=run(&self.context,&self.funding);
        if result.is_ok() && !state.consumed.get(){return Err(control_error(ControlCause::Incomplete,&state.retained,&self.funding));}
        Ok(result)
    }
}
// This closes construction on return/unwind only. It never closes the observer
// or discards accepted roots; Q still retains State until independent evidence.
struct Close<'a>(&'a Cell<bool>);
impl Drop for Close<'_>{fn drop(&mut self){self.0.set(true);}}
struct Loan(Option<Rc<State>>);
impl Loan{fn state(&self)->&State{self.0.as_deref().expect("live control loan")}}
impl Drop for Loan{fn drop(&mut self){if let Some(state)=self.0.take(){drop(Rc::into_inner(state));}}}
impl OriginalControlBinding {
    fn loan(&self)->Result<Loan,Error>{
        reserve(&self.funding,&[size_of::<Loan>(),size_of::<Option<Rc<State>>>(),
            size_of::<Result<Loan,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let state=self.state.upgrade().ok_or_else(||Error::with_original_control_source(self.fallback.retained().into_failure(),false))?;
        Ok(Loan(Some(state)))
    }
    pub(crate) fn with_context<T,E,F>(&self,prepared:&Group,run:F)->Result<Result<T,E>,Error>
    where F:FnOnce(Option<(&Group,&HostMetadataFunding)>)->Result<T,E>, {
        reserve(&self.funding,&[size_of::<F>(),size_of::<T>(),size_of::<E>(),size_of::<Result<T,E>>(),
            size_of::<Result<Result<T,E>,Error>>(),size_of::<Option<(&Group,&HostMetadataFunding)>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        let loan=self.loan()?;let state=loan.state();
        if state.closed.get() || state.bound.get().is_none() {
            return Err(control_error(ControlCause::Closed,&self.retained,&self.funding));
        }
        Ok(run(Some((prepared,&self.funding))))
    }
    pub(crate) fn with_group<T,E,F>(&self,event:ParallelControlEvent,group:&Group,prepared:&Group,
        funding:&HostMetadataFunding,executor:&Stream,run:F)->Result<Result<T,E>,Error>
    where F:FnOnce(Option<&Group>)->Result<T,E>, {
        reserve(&self.funding,&[size_of::<F>(),size_of::<T>(),size_of::<E>(),size_of::<Result<T,E>>(),
            size_of::<Result<Result<T,E>,Error>>(),size_of::<Option<&Group>>(),
            size_of::<(ParallelControlEvent,&Group,&Group,&HostMetadataFunding,&Stream)>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        let loan=self.loan()?;let state=loan.state();
        state.validate(prepared,executor)?;
        let source=state.source.communication_source()?;
        if event!=state.claim.event() || !funding.same_account(&state.funding)
            || !source.matches_group(state.group_order,group) {
            return Err(control_error(ControlCause::Identity,&state.retained,&self.funding));
        }
        Ok(run(Some(prepared)))
    }
    pub(crate) fn agree(&self,group:&Group,success:bool,executor:&Stream)
        ->Result<(OriginalCommunicationBool,MlxNeuralCommunicationCompletion),Error> {
        reserve(&self.funding,&[size_of::<(&Self,&Group,bool,&Stream)>(),
            size_of::<Result<(OriginalCommunicationBool,MlxNeuralCommunicationCompletion),Error>>(),
            size_of::<agreement::PreparedOriginalAgreement<'_>>(),size_of::<AgreementCapacity>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        let loan=self.loan()?;let state=loan.state();
        state.validate(group,executor)?;
        if state.consumed.replace(true){return Err(control_error(ControlCause::Exhausted,&state.retained,&self.funding));}
        let source=state.source.communication_source()?;
        let inputs=state.source.agreement_inputs().ok_or_else(||control_error(ControlCause::Identity,&state.retained,&self.funding))?;
        let actual_group=source.group(state.group_order)
            .ok_or_else(||control_error(ControlCause::Identity,&state.retained,&self.funding))?.1.id();
        let operation=inputs.prepare_for_group(&source,actual_group,success)?;
        let actual=AgreementCapacity{graph:operation.graph_capacity(),records:operation.record_capacity(),backing:operation.backing_capacity()};
        if !state.capacity.covers(actual){return Err(control_error(ControlCause::Identity,&state.retained,&self.funding));}
        let transport=state.group.retained_transport_stream()
            .ok_or_else(||control_error(ControlCause::Identity,&state.retained,&self.funding))?;
        operation.submit(&source,&state.bound.get().expect("validated control binding").observer,transport)
    }
}
impl State {
    fn validate(&self,group:&Group,stream:&Stream)->Result<(),Error>{
        reserve(&self.funding,&[size_of::<(&Self,&Group,&Stream)>(),size_of::<Result<(),Error>>(),
            size_of::<Option<&Bound>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let bound=self.bound.get().ok_or_else(||control_error(ControlCause::Unbound,&self.retained,&self.funding))?;
        reserve(&self.funding,&[bound.stream.source_comparison_control_bytes().ok_or_else(overflow)?])?;
        if self.closed.get(){return Err(control_error(ControlCause::Closed,&self.retained,&self.funding));}
        let source=self.source.communication_source()?;
        if self.claim.identity()!=self.identity || !bound.stream.matches_source(stream)
            || !source.matches_group(self.group_order,group) {
            return Err(control_error(ControlCause::Identity,&self.retained,&self.funding));
        }
        Ok(())
    }
}

mod role;
pub(crate) use role::{OriginalCaptureTransport,OriginalParallelControlOwner,OriginalParallelControlProjection,OriginalParallelControlInstallation};

pub(crate) use role::{OriginalSamplingSource, BoundSamplingSource};

pub(crate) use role::OriginalExpertMovementSource;
