//! Finite exact-role stream waits using the existing observed operation bank.
use super::*;
use super::prepared::{Cause,ResourceCustody,error};
use crate::backend::{runtime::distributed::topology::OriginalCommunicationSource,
    submission_recovery::observed::{ObservedRecovery,PreparedObservedRecovery,
        bank::{PreparedOperationBank,BankPreparationCause}}};
use crate::backend::submission_recovery::observed::RetirementAttempt;
use safemlx::{OriginalScopeObserver,OperationWaitRecordLayout,error::Exception};
use std::{alloc::Layout,convert::Infallible,mem::{size_of,size_of_val}};

type Ready=PreparedObservedRecovery<NativeChildResources,ResourceCustody>;
type Active=ObservedRecovery<NativeChildResources,ResourceCustody>;
struct Slots { ready:PreparedOperationBank<Ready>, active:Vec<Active> }
/// No backedge from NativeResources to this bank. Issued children retain the
/// actual resource owner until the same role retires, even after primary drop.
pub(super) struct PreparedConsumers {
    slots:RefCell<Slots>,
    _custody:ResourceCustody,
}
impl std::fmt::Debug for PreparedConsumers {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result { f.debug_struct("PreparedCommunicationConsumers").finish_non_exhaustive() }
}
impl PreparedConsumers {
    pub(super) fn prepare(source:&OriginalCommunicationSource<'_>,waits:OperationWaitRecordLayout)->Result<Self,Error> {
        source.validate()?;
        let custody=ResourceCustody { source:source.source().clone(),funding:source.funding().clone() };
        let factory_custody=custody.clone();
        let factory=move |_:usize|Ok::<_,Infallible>(Ready::new(factory_custody.clone()));
        let bytes=controls(&factory,waits).ok_or(Error::WorkspacePlanning(eredu_nn::workspace::HostMetadataFundingError::Overflow))?;
        custody.funding.reserve_metadata(bytes).map_err(Error::WorkspacePlanning)?;
        let ready=PreparedOperationBank::try_new(waits.wait_count(),factory).map_err(|failed| {
            let (cause,prefix,factory)=failed.into_parts();
            let cause=match cause {
                BankPreparationCause::Overflow=>Cause::Identity,
                BankPreparationCause::Reserve(cause)=>Cause::Capacity(cause),
                BankPreparationCause::Slot { cause,.. }=>match cause {},
            };
            let retained=error(cause,&custody);drop(prefix);drop(factory);retained
        })?;
        let mut active=Vec::new();active.try_reserve_exact(waits.wait_count()).map_err(|cause|error(Cause::Capacity(cause),&custody))?;
        Ok(Self { slots:RefCell::new(Slots { ready,active }),_custody:custody })
    }
    pub(super) fn wait(&self,event:&NativeEvent,owner:&NativeOwner,stream:&Stream,
        observer:&OriginalScopeObserver)->Result<(),Exception> {
        let producer=event.original_observer().ok_or_else(||observer.domain_error())?;
        let current=OriginalScopeObserver::require_current()?;
        if !current.same_scope(observer) || !producer.same_scope(observer)
            || !owner._streams.iter().any(|retained|retained==stream) {
            return Err(producer.domain_error());
        }
        // Do not progress a frontier opened by another predecessor of this
        // same join. Native wait validates actual controls and selected device.
        let status=producer.status();
        if status.failed() || status.blocked() || owner.host_failed.get() || owner.child_failed.get() {
            return Err(producer.retained_failure().unwrap_or_else(||producer.invalid_input_error()));
        }
        let children=owner.children.get().checked_add(1).ok_or_else(||producer.capacity_error())?;
        let ready={
            let mut slots=self.slots.try_borrow_mut().map_err(|_|busy(producer))?;
            slots.ready.checkout().map_err(|_|producer.capacity_error())?
        };
        owner.children.set(children);
        let mut child=ready.activate(NativeChildResources { _arrays:Vec::new(),_stream:None,owner:owner.clone() },observer.clone());
        let _unwind=ChildUnwind(&owner.host_failed);
        let result=event.wait_on(stream);
        child.seal();
        // Fixed refusal spends its slot but does not invent native failure.
        // Actual failed/blocked observations flow through Retention::observe.
        self.slots.borrow_mut().active.push(child);
        result
    }
    pub(super) fn retire_completed(&self,observer:&OriginalScopeObserver)->Result<RetirementAttempt,Exception> {
        let mut slots=self.slots.try_borrow_mut().map_err(|_|busy(observer))?;
        let mut index=0;
        while index<slots.active.len() {
            // Keep the actual owner through a consuming cleanup refusal.
            let owner = slots.active[index].retention().owner.clone();
            match slots.active[index].try_finish_successfully()? {
                RetirementAttempt::Retired => drop(slots.active.swap_remove(index)),
                RetirementAttempt::Pending => index+=1,
                RetirementAttempt::Stopped(cause) => {
                    drop(slots.active.swap_remove(index));
                    owner.child_failed.set(true);
                    // These observed child nodes currently have no registered
                    // prediction role. Preserve a future stop unchanged until
                    // the actual caller chooses synchronous or deferred handling.
                    return Ok(RetirementAttempt::Stopped(cause));
                }
            }
        }
        Ok(if slots.active.is_empty() { RetirementAttempt::Retired } else { RetirementAttempt::Pending })
    }
}
fn busy(observer:&OriginalScopeObserver)->Exception {
    observer.observation_error(safemlx::ScopedSubmissionProgress::Busy).unwrap_or_else(||observer.invalid_input_error())
}
fn controls<F>(_:&F,waits:OperationWaitRecordLayout)->Option<usize>
where F:FnMut(usize)->Result<Ready,Infallible> {
    let bank=PreparedOperationBank::<Ready>::layout::<F,Infallible>(waits.wait_count(),Ready::control_bytes::<Exception>()?)?;
    let fixed=[size_of::<PreparedConsumers>(),size_of::<Slots>(),size_of::<RefCell<Slots>>(),
        size_of::<std::cell::RefMut<'_,Slots>>(),size_of::<std::cell::BorrowMutError>(),
        size_of::<Active>(),size_of::<Option<Active>>(),size_of::<Vec<Active>>(),
        size_of::<Result<PreparedConsumers,Error>>(),size_of::<Result<(),Exception>>(),
        size_of::<Result<bool,Exception>>(),size_of::<Result<RetirementAttempt,Exception>>(),size_of::<NativeOwner>(),size_of::<NativeChildResources>(),size_of::<ChildUnwind<'_>>(),
        size_of::<(&NativeEvent,&NativeOwner,&Stream,&OriginalScopeObserver)>(),
        size_of::<std::slice::Iter<'_,Stream>>(),size_of::<(usize,usize)>(),
        size_of::<Result<(),std::collections::TryReserveError>>(),
        size_of::<safemlx::SubmissionStatus>(),size_of::<OperationWaitRecordLayout>(),
        Layout::array::<Active>(waits.wait_count()).ok()?.size(),
        usize::try_from(bank.total_control_bytes).ok()?,
        usize::try_from(Active::try_finish_control_bytes()?).ok()?,
        safemlx::OriginalScopeObserver::control_bytes()?,waits.named_control_bytes(),
        prepared::error_control_bytes()?,
    ];
    fixed.into_iter().try_fold(size_of_val(&fixed),usize::checked_add)
}

#[cfg(test)]
mod tests;
