//! A lexical movement producer names one retained dynamic-region occurrence.
//! It cannot select a later equal-shaped region or retain a native Group clone.
use super::*;
use super::super::super::parallel::{OriginalParallelBinding, RetainedExpertRegion};
use eredu_nn::workspace::WorkspaceExpertRegionView;
use eredu_runtime::PreparedExpertMovementLoan;
use safemlx::{Array, OriginalScopeObserver, PreparedInputPlan, PreparedInputRuntime, Stream};

struct Source {
    quote: RetainedExpertRegion,
    binding: OriginalParallelBinding,
    parent: OriginalScopeObserver,
    occurrence: usize,
    attempted: Cell<[usize; 4]>,
    completed: Cell<[usize; 4]>,
    population_checked: Cell<bool>,
    finished: Cell<bool>,
    owner: OriginalParallelControlOwner,
    custody: Custody,
}
#[derive(Clone)]
pub(crate) struct OriginalExpertMovementSource {
    source: Option<Rc<Source>>,
    funding: HostMetadataFunding,
}
impl std::fmt::Debug for OriginalExpertMovementSource {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("OriginalExpertMovementSource").field("retained",&self.source.is_some()).finish_non_exhaustive()
    }
}
impl Drop for OriginalExpertMovementSource {
    fn drop(&mut self) { if let Some(source)=self.source.take(){drop(Rc::into_inner(source));} }
}
impl OriginalExpertMovementSource {
    fn source(&self)->&Source { self.source.as_deref().expect("retained expert movement source") }
    pub(super) fn prepare(binding:&OriginalParallelBinding,model:&OriginalParallelSource,
        declaration:WorkspaceExpertRegionView<'_>,stream:&Stream,owner:&OriginalParallelControlOwner,c:&Custody)
        ->Result<Self,Error> {
        reserve(&c.funding,&[size_of::<Self>(),size_of::<Source>(),size_of::<Result<Self,Error>>(),
            size_of::<PreparedExpertMovementLoan<'_>>(),size_of::<Option<PreparedExpertMovementLoan<'_>>>(),
            Layout::new::<[usize;2]>().extend(Layout::new::<Source>()).map_err(|_|overflow())?.0.pad_to_align().size(),
            OriginalScopeObserver::control_bytes().ok_or_else(overflow)?,failure_control_bytes().ok_or_else(overflow)?])?;
        let (quote,parent,occurrence)=binding.expert_movement_source(model,declaration,stream)?;
        let value=Self{source:Some(Rc::new(Source{quote,binding:binding.clone(),parent,occurrence,
            attempted:Cell::new([0;4]),completed:Cell::new([0;4]),
            population_checked:Cell::new(false),finished:Cell::new(false),
            owner:OriginalParallelControlOwner(owner.0.clone()),custody:c.clone()})),funding:c.funding.clone()};
        value.validate(stream)?;
        Ok(value)
    }
    pub(super) fn loan(&self)->PreparedExpertMovementLoan<'_> { PreparedExpertMovementLoan::new(self,&self.funding) }
    pub(crate) fn from_loan(loan:PreparedExpertMovementLoan<'_>,stream:&Stream)->Result<Self,Error> {
        let invalid=||Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch);
        // Downcast does not create authority: validate the private retained
        // occurrence and parent observer before lending it to the actual worker.
        let source=loan.source::<Self>().ok_or_else(invalid)?;
        let c=&source.source().custody;
        reserve(&c.funding,&[size_of::<Self>(),size_of::<Option<Self>>(),size_of::<Result<Self,Error>>()])?;
        if !loan.funding().same_account(&source.funding){return Err(control_error(ControlCause::Identity,&c.source,&c.funding));}
        source.validate(stream)?;
        Ok(source.clone())
    }
    pub(crate) fn validate(&self,stream:&Stream)->Result<(),Error> {
        let source=self.source();let c=&source.custody;
        reserve(&c.funding,&[size_of::<Result<(),Error>>(),size_of::<OriginalScopeObserver>(),
            OriginalScopeObserver::control_bytes().ok_or_else(overflow)?,failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||control_error(ControlCause::Identity,&c.source,&c.funding);
        let owner=source.owner.owner();
        if owner.failed.get()||owner.running.get()||source.finished.get(){return Err(fail());}
        let model=owner.request.source.model().ok_or_else(fail)?;
        source.binding.validate_expert_movement(model,&source.quote,source.occurrence,stream)?;
        let actual=OriginalScopeObserver::require_current().map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
        if !source.parent.same_scope(&actual){return Err(fail());}
        Ok(())
    }
    pub(crate) fn validate_population(&self,population:eredu_nn::workspace::WorkspaceExpertMovementPopulation,
        transfers:eredu_nn::workspace::WorkspaceExpertTransfers,stream:&Stream)->Result<(),Error> {
        self.validate(stream)?;
        let source=self.source();let c=&source.custody;
        reserve(&c.funding,&[size_of::<eredu_nn::workspace::WorkspaceExpertMovementPopulation>(),
            size_of::<eredu_nn::workspace::WorkspaceExpertTransfers>(),size_of::<[usize;4]>()])?;
        if source.population_checked.replace(true)||source.quote.value().declaration.as_view().movement!=population
            ||source.quote.value().declaration.as_view().transfers!=transfers {
            source.owner.owner().failed.set(true);
            return Err(control_error(ControlCause::Identity,&c.source,&c.funding));
        }
        Ok(())
    }
    pub(super) fn finish(&self,stream:&Stream)->Result<(),Error> {
        self.validate(stream)?;
        let source=self.source();let c=&source.custody;
        reserve(&c.funding,&[size_of::<[[usize;4];3]>()])?;
        if !source.population_checked.get()||source.attempted.get()!=source.completed.get()
            ||source.completed.get()!=source.quote.value().declaration.as_view().movement.counts() {
            source.owner.owner().failed.set(true);
            return Err(control_error(ControlCause::Identity,&c.source,&c.funding));
        }
        source.finished.set(true);
        Ok(())
    }
    pub(crate) fn reserve_call<T>(&self,stream:&Stream)->Result<(),Error> {
        self.validate(stream)?;
        reserve(&self.funding,&[size_of::<T>(),size_of::<Result<T,Error>>()])
    }
    pub(crate) fn destination<T>(&self,count:usize,stream:&Stream)->Result<Vec<T>,Error> {
        self.validate(stream)?;
        let c=&self.source().custody;
        let declaration=self.source().quote.value().declaration.as_view();
        let limit=declaration.selected_rows().ok_or_else(overflow)?.max(declaration.source_rows).max(2);
        if count>limit{return Err(control_error(ControlCause::Identity,&c.source,&c.funding));}
        reserve(&c.funding,&[size_of::<Vec<T>>(),size_of::<Result<Vec<T>,Error>>(),
            size_of::<std::collections::TryReserveError>(),Layout::array::<T>(count).map_err(|_|overflow())?.size()])?;
        let mut output=Vec::new();output.try_reserve_exact(count)
            .map_err(|_|control_error(ControlCause::Identity,&c.source,&c.funding))?;
        Ok(output)
    }
    pub(crate) fn indices(&self,values:&[usize],trailing_axis:bool,stream:&Stream)->Result<Array,Error> {
        self.validate(stream)?;
        let source=self.source();let c=&source.custody;
        let fail=||control_error(ControlCause::Identity,&c.source,&c.funding);
        if values.len()>source.quote.value().declaration.as_view().selected_rows().ok_or_else(overflow)?{return Err(fail());}
        reserve(&c.funding,&[size_of::<[usize;2]>(),size_of::<[[i32;2];3]>(),
            size_of::<PreparedInputPlan<'_>>(),size_of::<[Array;1]>(),size_of::<Result<Array,Error>>(),
            PreparedInputRuntime::zeros_plan_control_bytes(),Array::static_slice_control_bytes().ok_or_else(overflow)?])?;
        let mut original=self.destination::<i32>(values.len(),stream)?;
        for &value in values{original.push(i32::try_from(value).map_err(|_|fail())?);}
        let owner=source.owner.owner();
        let inputs=owner.request.source.agreement_inputs().ok_or_else(fail)?;
        let shape=[values.len().max(1),1];let rank=1+usize::from(trailing_axis);
        let plan=if values.is_empty(){inputs.runtime().zeros(safemlx::Dtype::Int32,&shape[..rank])}
            else{inputs.runtime().i32(&original,&shape[..rank])}
            .map_err(|cause|failure(Cause::Input(cause),&c.source,&c.funding))?;
        let array=super::sampling::token::construct(owner,plan)?;
        if values.is_empty(){array.try_slice(&[0,0][..rank],&[0,1][..rank],&[1,1][..rank],stream)
            .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))}else{Ok(array)}
    }
}

#[path = "expert_movement/child.rs"]
mod child;
