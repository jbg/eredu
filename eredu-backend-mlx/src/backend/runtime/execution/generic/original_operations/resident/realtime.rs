//! One frame's exact selected graph boundaries in the existing resident slot.
use super::*;
use crate::backend::nn::workspace::SpeculativeNumericalRecipe;
use crate::backend::submission_recovery::native_role::realtime::RealtimeOperationClaim;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::working_memory::{OriginalOperationMetadataCustody,OriginalRealtimeBudgetCustody};
use safemlx::OriginalScopeObserver;

#[derive(Clone)]
pub(super) struct Projection {
    value:Weak<Bank>,
    custody:OriginalRealtimeBudgetCustody,
}
impl Projection {
    pub(super) fn is_live(&self)->bool {self.value.strong_count()!=0}
    fn access(&self,stream:&Stream)->Result<Rc<Bank>,Error> {
        let bank=self.value.upgrade().ok_or_else(identity)?;
        if !bank.active.get() || !bank.custody.same_account(&self.custody)
            || !bank.stream.matches_source(stream)
            || !bank.observer.same_scope(&OriginalScopeObserver::require_current()?) {
            return Err(identity());
        }
        Ok(bank)
    }
    pub(super) fn uses_shared_executor(&self,stream:&Stream)->Result<bool,Error> {
        self.access(stream).map(|_|true).map_err(|cause|neural::boundary_error(cause,&self.custody))
    }
    pub(super) fn submit(&self,stream:&Stream,value:&MlxTensor)->Result<OrderedNeuralCompletion,Error> {
        let result=(|| {
            let bank=self.access(stream)?;
            let prepared=bank.prepared.try_borrow_mut().map_err(|_|Error::PrefillScopeReentrant)?
                .checkout().map_err(|_|Error::PrefillScopeUnavailable)?;
            prepared.submit_nested(value,bank.observer.clone(),stream)
                .map(|completion|OrderedNeuralCompletion::Realtime{completion,controls:self.custody.clone()})
                .map_err(|failure|Error::from(failure.into_cause()))
        })();
        result.map_err(|cause|neural::boundary_error(cause,&self.custody))
    }
}
struct Bank {
    prepared:RefCell<PreparedOperationBank<PreparedNeuralSubmission>>,
    observer:OriginalScopeObserver,
    stream:safemlx::StreamCopyPlan<()>,
    active:Cell<bool>,
    custody:OriginalRealtimeBudgetCustody,
}
struct Owner<U:'static> {bank:Rc<Bank>,slot:ResidentNeuralSlot<U>}
impl<U:'static> ErasedOwner for Owner<U> {}
impl<U:'static> Drop for Owner<U> {
    fn drop(&mut self) {
        let previous=self.slot.0.resident.take();
        if previous.as_ref().is_some_and(|value|matches!(value,super::Projection::Realtime(source)
            if Weak::ptr_eq(&source.value,&Rc::downgrade(&self.bank)))) {
            drop(previous);
        } else {self.slot.0.resident.set(previous);}
        self.bank.active.set(false);
    }
}
/// Installed only during the accepted frame callback. Issued completions retain
/// their own account and native objects after this lexical slot owner retires.
pub(crate) struct RealtimeNeuralOwner {
    value:Box<dyn ErasedOwner>,
    custody:OriginalRealtimeBudgetCustody,
}
impl RealtimeNeuralOwner {
    pub(in crate::backend::runtime::execution::generic::original_operations) fn from_bounded(
        value:Box<dyn ErasedOwner>,custody:OriginalRealtimeBudgetCustody)->Self {Self{value,custody}}

    pub(crate) fn during<T>(self,run:impl FnOnce()->T)->T {
        let result=run();drop(self);result
    }
}

/// Descriptive source from the actual resident slot, layout and selected stream.
/// Qualifying it consumes the complete numerical recipe; no text geometry is
/// created, and the resulting plan remains move-only until native activation.
pub(crate) struct RealtimeNeuralPlan<U:'static> {
    slot:ResidentNeuralSlot<U>,
    neural:NeuralPopulation,
    stream:safemlx::StreamCopyPlan<()>,
}
pub(crate) struct QualifiedRealtimeNeuralPlan<U:'static> {
    source:RealtimeNeuralPlan<U>,
    recipe:SpeculativeNumericalRecipe,
    fit:NeuralProducerFit,
}
impl<U:'static> ResidentNeuralSlot<U> {
    pub(crate) fn realtime_plan(&self,layout:&ExecutionUnitLayout,stream:&Stream,
        context:&WorkspaceContext)->Result<RealtimeNeuralPlan<U>,Error> {
        // Resident policy finish returns the retained module without an
        // additional submission; use the same shared population producer.
        let neural=NeuralPopulation::single_forward(layout,false)?;
        let controls=safemlx::StreamCopyPlan::<()>::capture_control_bytes().map_err(|_|unknown())?
            .checked_add(size_of::<(RealtimeNeuralPlan<U>,NeuralPopulation,ResidentNeuralSlot<U>)>())
            .ok_or_else(overflow)?;
        context.charge_metadata(controls).map_err(|cause|Error::Neural(cause.into()))?;
        let stream=safemlx::StreamCopyPlan::capture(stream).map_err(|_|identity())?;
        Ok(RealtimeNeuralPlan{slot:self.clone(),neural,stream})
    }
}
impl<U:'static> RealtimeNeuralPlan<U> {
    pub(crate) fn qualify(self,recipe:SpeculativeNumericalRecipe,context:&WorkspaceContext)
        ->Result<QualifiedRealtimeNeuralPlan<U>,Error> {
        let recipe=recipe.with_realtime_boundaries(self.neural.submissions,self.neural.shape.consumers(),context)
            .map_err(Error::Neural)?;
        let fit=NeuralProducerFit::pending(self.neural).ok_or_else(unknown)?;
        Ok(QualifiedRealtimeNeuralPlan{source:self,recipe,fit:NeuralProducerFit::Recipe(fit.requirement())})
    }
}
impl<U:'static> QualifiedRealtimeNeuralPlan<U> {
    pub(crate) fn recipe(&self)->&SpeculativeNumericalRecipe {&self.recipe}
    pub(crate) fn control_bytes(&self)->Option<u64> {
        let frames=[size_of::<Self>(),size_of::<RealtimeNeuralOwner>(),size_of::<Owner<U>>(),
            size_of::<Bank>(),size_of::<Rc<Bank>>(),size_of::<Projection>(),
            size_of::<super::Projection>(),size_of::<Option<super::Projection>>(),
            size_of::<Box<dyn ErasedOwner>>(),size_of::<OriginalScopeObserver>(),
            size_of::<OriginalRealtimeBudgetCustody>(),size_of::<OriginalOperationMetadataCustody>(),
            size_of::<Result<RealtimeNeuralOwner,Error>>(),size_of::<RealtimeOperationClaim<'_>>(),
            size_of::<std::cell::RefMut<'_,PreparedOperationBank<PreparedNeuralSubmission>>>(),
            size_of::<(OriginalScopeObserver,OriginalRealtimeBudgetCustody)>(),
            size_of::<Result<(OriginalScopeObserver,OriginalRealtimeBudgetCustody),Error>>()];
        let fixed=frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)?
            .checked_add(rc_layout::<Bank>()?)?
            .checked_add(size_of::<Owner<U>>())?
            .checked_add(eredu_core::BackendFailure::source_retention_peak_bytes::<
                neural::NeuralBoundaryFailure<OriginalRealtimeBudgetCustody>>()?)?
            .checked_add(self.source.stream.source_comparison_control_bytes()?)?;
        self.source.neural.rust_control_bytes()?.checked_add(self.fit.additional_control_bytes()?)?
            .checked_add(u64::try_from(fixed).ok()?)
    }
    pub(crate) fn prepare(self,claim:RealtimeOperationClaim<'_>)->Result<RealtimeNeuralOwner,Error> {
        let controls=usize::try_from(self.control_bytes().ok_or_else(unknown)?).map_err(|_|overflow())?;
        let (observer,custody)=claim.prepare(controls)?;
        let result=(|| {
            let previous=self.source.slot.0.resident.take();
            let busy=previous.as_ref().is_some_and(super::Projection::is_live);
            self.source.slot.0.resident.set(previous);
            if busy {return Err(Error::PrefillScopeReentrant);}
            let prepared=neural::prepare_with_custody(self.source.neural,custody.clone().into())?;
            let bank=Rc::new(Bank{prepared:RefCell::new(prepared),observer,
                stream:self.source.stream,active:Cell::new(true),custody:custody.clone()});
            let previous=self.source.slot.0.resident.replace(Some(super::Projection::Realtime(Projection {
                value:Rc::downgrade(&bank),custody:custody.clone()})));
            drop(previous);
            Ok(RealtimeNeuralOwner{value:Box::new(Owner{bank,slot:self.source.slot}),custody:custody.clone()})
        })();
        result.map_err(|cause|neural::boundary_error(cause,&custody))
    }
}
