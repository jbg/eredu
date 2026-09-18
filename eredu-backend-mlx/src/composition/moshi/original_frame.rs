//! Source-funded retention around the unchanged selected realtime equation.
use super::*;
use crate::backend::submission_recovery::native_role::realtime::RealtimeRoleContext;
use eredu_nn::workspace::{WorkspaceContext,HostMetadataFunding};
use eredu_runtime::working_memory::{OriginalRealtimeBudgetCustody,WorkingMemoryError};
use safemlx::OriginalScopeObserver;
use crate::backend::runtime::distributed::{Group,topology::original_source::parallel::OriginalParallelInvocation};
use std::mem::{size_of,size_of_val};

struct State {
    payload:RefCell<Option<Rc<OrdinaryRetirement<RealtimeExecutionPayload>>>>,
    poisoned:Rc<Cell<bool>>,
    entered:Cell<bool>,
    observer:RefCell<Option<OriginalScopeObserver>>,
    custody:OriginalRealtimeBudgetCustody,
    funding:HostMetadataFunding,
}
impl Drop for State {
    fn drop(&mut self) {
        if self.observer.get_mut().as_ref().is_some_and(|observer| {
            let status=observer.status();status.failed()||status.blocked()
        }) {self.poisoned.set(true);}
    }
}
/// Held by the accepted native role before any mutable model borrow. The
/// payload pin is published only after that borrow ends, including unwinding.
#[derive(Clone)]
pub(crate) struct OriginalRealtimeModelLease(Option<Rc<State>>);
impl Drop for OriginalRealtimeModelLease {
    fn drop(&mut self){if let Some(value)=self.0.take(){drop(Rc::into_inner(value));}}
}
impl OriginalRealtimeModelLease {
    fn state(&self)->&State {self.0.as_deref().expect("live original model lease")}
    pub(crate) fn control_bytes<T>()->Option<usize> {
        let parts=[WorkspaceContext::metadata_rc_bytes::<State>()?,size_of::<State>(),size_of::<Self>(),
            size_of::<Result<Self,Error>>(),size_of::<Guard<'_>>(),size_of::<Result<T,Error>>(),
            size_of::<&mut dyn FnMut(&mut MlxRealtimeExecution)->Result<T,Error>>(),
            size_of::<(&MlxRealtimeExecution,&RealtimeRoleContext<'_>)>(),
            size_of::<Option<OriginalScopeObserver>>(),OriginalScopeObserver::control_bytes()?.checked_mul(2)?,
            size_of::<std::cell::RefMut<'static,Option<Rc<OrdinaryRetirement<RealtimeExecutionPayload>>>>>(),
            size_of::<OriginalRealtimeBudgetCustody>(),size_of::<HostMetadataFunding>(),
            size_of::<Option<&HostMetadataFunding>>(),
            size_of::<moshi::MoshiRealtimeExecutionError<Error>>(),
            WorkspaceContext::metadata_source_bytes::<execution_failure::Source>()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(crate) fn prepare<T>(model:&MlxRealtimeExecution,custody:OriginalRealtimeBudgetCustody,
        funding:&HostMetadataFunding)->Result<Self,Error> {
        funding.reserve_metadata(Self::control_bytes::<T>().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        model.ensure_healthy()?;
        if Rc::strong_count(&model.payload)!=1 {return Err(Error::PrefillScopeUnavailable);}
        Ok(Self(Some(Rc::new(State{payload:RefCell::new(None),poisoned:model.poisoned.clone(),
            entered:Cell::new(false),observer:RefCell::new(None),custody,funding:funding.clone()}))))
    }
    fn validate(&self,model:&MlxRealtimeExecution,role:&RealtimeRoleContext<'_>)->Result<(),Error> {
        let source=self.state();
        if !Rc::ptr_eq(&source.poisoned,&model.poisoned)
            || !source.custody.same_account(&role.budget_custody())
            || !OriginalScopeObserver::require_current()?.same_scope(role.native().observer()) {
            return Err(identity());
        }
        Ok(())
    }
    /// The accepted outer Recovery already owns this lease. No ordinary scope
    /// or additional completion engine is opened around the shared equation.
    pub(crate) fn during<T>(&self,model:&mut MlxRealtimeExecution,role:&RealtimeRoleContext<'_>,
        run:&mut dyn FnMut(&mut MlxRealtimeExecution)->Result<T,Error>)->Result<T,Error> {
        self.validate(model,role)?;
        if self.state().entered.replace(true){return Err(identity());}
        model.ensure_healthy()?;
        if Rc::strong_count(&model.payload)!=1{return Err(Error::PrefillScopeUnavailable);}
        *self.state().observer.borrow_mut()=Some(role.native().observer().clone());
        let mut guard=Guard{model,source:self,succeeded:false};
        let result=run(guard.model);guard.succeeded=result.is_ok();drop(guard);result
    }
}
struct Guard<'a>{model:&'a mut MlxRealtimeExecution,source:&'a OriginalRealtimeModelLease,succeeded:bool}
impl Drop for Guard<'_> {
    fn drop(&mut self){
        if !self.succeeded {self.model.poisoned.set(true);}
        *self.source.state().payload.borrow_mut()=Some(self.model.payload.clone());
    }
}
impl MlxRealtimeExecution {
    pub(crate) fn execute_selected_realtime_original(&mut self,state:&mut MlxKeyValueState,
        temporal:&[crate::MlxTensor],
        driver:&mut SequentialDecisionDriver<MlxSamplingBackend,eredu_runtime::GenerationSampler>,
        stream:&Stream,source:&OriginalRealtimeModelLease,role:&RealtimeRoleContext<'_>,
        parallel:Option<&OriginalParallelInvocation>)
        ->Result<(Option<crate::MlxTensor>,moshi::ForwardContext<crate::MlxTensor>),Error> {
        source.validate(self,role)?;
        if !source.state().entered.get()||source.state().payload.borrow().is_some(){return Err(identity());}
        if self.parallel_communication().is_some()!=parallel.is_some(){return Err(identity());}
        match parallel {
            None=>self.execute_realtime_body(state,temporal,driver,stream,Some(role.metadata_funding())),
            Some(parallel)=>{
                role.metadata_funding().reserve_metadata(parallel_frame_controls().ok_or_else(overflow)?)
                    .map_err(Error::WorkspacePlanning)?;
                let mut frame=ParallelFrame{model:self,state,temporal,driver,stream,funding:parallel.source().funding()};
                let result=parallel.with_borrowed_context(role.native().observer(),stream,&mut parallel_callback(Some(&mut frame)));
                result
            }
        }
    }
    pub(crate) fn original_parallel_control_bytes()->Option<usize> {
        parallel_frame_controls()?.checked_add(OriginalParallelInvocation::borrowed_context_control_bytes::<Forward>()?)
    }
}
type Forward=(Option<crate::MlxTensor>,moshi::ForwardContext<crate::MlxTensor>);
struct ParallelFrame<'a> {
    model:&'a mut MlxRealtimeExecution,state:&'a mut MlxKeyValueState,temporal:&'a [crate::MlxTensor],
    driver:&'a mut SequentialDecisionDriver<MlxSamplingBackend,eredu_runtime::GenerationSampler>,
    stream:&'a Stream,funding:&'a HostMetadataFunding,
}
impl ParallelFrame<'_> {
    fn run(&mut self,parallel:&Group)->Result<Forward,Error> {
        Rc::get_mut(&mut self.model.payload).ok_or(Error::PrefillScopeUnavailable)?
            .execution.execute_original_decisions(self.state,self.temporal,self.driver,self.stream,parallel,self.funding)
    }
}
fn parallel_callback<'a,'source:'a>(mut frame:Option<&'a mut ParallelFrame<'source>>)
    ->impl FnMut(&Group)->Result<Forward,Error>+'a + use<'a,'source> {
    move |parallel|frame.as_deref_mut().ok_or_else(identity)?.run(parallel)
}
fn parallel_frame_controls()->Option<usize> {
    let callback=parallel_callback(None);
    let parts=[size_of_val(&callback),size_of::<ParallelFrame<'_>>(),size_of::<Forward>(),
        size_of::<Result<Forward,Error>>(),size_of::<Option<&OriginalParallelInvocation>>(),
        size_of::<(&Group,&Group)>(),
        WorkspaceContext::metadata_source_bytes::<execution_failure::Source>()?];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
fn identity()->Error{Error::PrefillControl(WorkingMemoryError::IdentityMismatch)}
fn overflow()->Error{Error::PrefillControl(WorkingMemoryError::Overflow)}
