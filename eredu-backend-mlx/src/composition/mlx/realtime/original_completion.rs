//! Completion source for the existing frame coordinator inside its accepted role.
use super::*;
use crate::backend::{nn::{tensor::TokenValidationIngress,workspace::{ResidentCompletionRecipe,SpeculativeNumericalRecipe}},
    submission_recovery::{native_role::realtime::RealtimeRoleContext,prefill::nested::NestedRootCompletion}};
use eredu_nn::workspace::{WorkspaceContext,WorkspaceMetadataFunding};
use eredu_runtime::{RuntimeState,working_memory::{OriginalRealtimeBudgetCustody,WorkingMemoryError}};
use safemlx::{OriginalScopeObserver,PreparedResidentGraph,OperationEvent};
use std::{cell::{Cell,RefCell},rc::Rc,mem::{size_of,size_of_val}};
struct State {
    values:RefCell<Vec<MlxTensor>>,
    nested:RefCell<NestedRootCompletion>,
    ingress:RefCell<TokenValidationIngress>,
    validations:RefCell<Option<TokenValidationBatch>>,
    observer:RefCell<Option<OriginalScopeObserver>>,
    expected:usize,constructed:Cell<bool>,completed:Cell<bool>,
    custody:OriginalRealtimeBudgetCustody,funding:WorkspaceMetadataFunding,
}
// Keep the counted allocation outside the RefCell while native completion
// or ready-event detachment can retire native references. Every exit restores
// the original roots; an error never discards an unpublished frame frontier.
struct CompletingRoots<'a> {
    values:Vec<MlxTensor>,slot:&'a RefCell<Vec<MlxTensor>>,
}
impl<'a> CompletingRoots<'a> {
    fn take(slot:&'a RefCell<Vec<MlxTensor>>)->Result<Self,Error> {
        let values=std::mem::take(&mut *slot.try_borrow_mut().map_err(|_|mismatch())?);
        Ok(Self{values,slot})
    }
}
impl Drop for CompletingRoots<'_> {
    fn drop(&mut self) {
        let mut slot=self.slot.borrow_mut();
        assert!(slot.is_empty()&&slot.capacity()==0,"frame completion root slot changed while borrowed");
        *slot=std::mem::take(&mut self.values);
    }
}

/// Closed Rc handle: final shared allocation retires before account custody.
#[derive(Clone)]
pub(super) struct OriginalRealtimeCompletion(Option<Rc<State>>);
impl Drop for OriginalRealtimeCompletion {
    fn drop(&mut self){if let Some(value)=self.0.take(){drop(Rc::into_inner(value));}}
}
impl OriginalRealtimeCompletion {
    fn state(&self)->&State {self.0.as_deref().expect("live original completion")}
    pub(super) fn retained_resources(&self)->usize {self.state().values.borrow().len()}
    pub(super) fn control_bytes(recipe:ResidentCompletionRecipe,roots:usize)->Option<usize> {
        let parts=[WorkspaceContext::metadata_rc_bytes::<State>()?,
            WorkspaceContext::metadata_vec_bytes::<MlxTensor>(roots)?,
            NestedRootCompletion::control_bytes(roots,recipe.validation_roots)?,
            usize::try_from(TokenValidationIngress::realtime_control_bytes(recipe).ok()?).ok()?,
            OriginalScopeObserver::control_bytes()?.checked_mul(2)?,
            // One existing scope-authenticated completed-array validation per
            // exact retained frontier entry, including history and updated RNG.
            OriginalScopeObserver::control_bytes()?.checked_mul(roots)?,
            size_of::<CompletingRoots<'static>>(),size_of::<Result<CompletingRoots<'static>,Error>>(),
            size_of::<std::slice::Iter<'static,MlxTensor>>(),
            size_of::<Self>(),size_of::<State>(),size_of::<OriginalFrameCompletionMechanism>(),
            size_of::<ValidationScope<'_>>(),size_of::<Result<ValidationScope<'_>,Error>>(),
            size_of::<Option<PreparedResidentGraph>>(),size_of::<Result<Self,Error>>(),
            size_of::<(usize,&RandomState,&Stream)>(),size_of::<Result<(),Error>>(),
            size_of::<std::cell::RefMut<'static,Vec<MlxTensor>>>(),
            size_of::<std::cell::RefMut<'static,NestedRootCompletion>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(super) fn prepare(recipe:ResidentCompletionRecipe,roots:usize,
        custody:OriginalRealtimeBudgetCustody,funding:&WorkspaceMetadataFunding)->Result<Self,Error> {
        let all=Self::control_bytes(recipe,roots).ok_or_else(overflow)?;
        let separate=WorkspaceContext::metadata_vec_bytes::<MlxTensor>(roots).ok_or_else(overflow)?
            .checked_add(NestedRootCompletion::control_bytes(roots,recipe.validation_roots).ok_or_else(overflow)?)
            .and_then(|n|n.checked_add(usize::try_from(TokenValidationIngress::realtime_control_bytes(recipe).ok()?).ok()?))
            .ok_or_else(overflow)?;
        funding.reserve_metadata(all.checked_sub(separate).ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let values=funding.metadata_vec(roots).map_err(Error::Neural)?;
        let nested=NestedRootCompletion::prepare_metadata(roots,recipe.validation_roots,custody.clone().into(),funding)?;
        let ingress=TokenValidationIngress::prepare_realtime(recipe,custody.clone(),funding)?;
        Ok(Self(Some(Rc::new(State{values:RefCell::new(values),nested:RefCell::new(nested),
            ingress:RefCell::new(ingress),validations:RefCell::new(None),observer:RefCell::new(None),
            expected:roots,constructed:Cell::new(false),completed:Cell::new(false),custody,funding:funding.clone()}))))
    }
    pub(super) fn begin<'a>(&'a self,recipe:SpeculativeNumericalRecipe,role:&RealtimeRoleContext<'_>)
        ->Result<ValidationScope<'a>,Error> {
        let state=self.state();
        if !state.custody.same_account(&role.budget_custody())||state.observer.borrow().is_some(){return Err(mismatch());}
        let observer=role.native().observer();
        let mut graph=OperationEvent::prepare_resident_graph(recipe.completion.graph,observer)?;
        if recipe.completion.nested_completions!=0 {
            let nested=recipe.completion.nested_traversal().ok_or_else(mismatch)?;
            graph.configure_nested_completions(&nested,recipe.completion.nested_completions)?;
        }
        let scope=state.ingress.borrow_mut().begin()?;
        *state.observer.borrow_mut()=Some(observer.clone());
        Ok(ValidationScope{scope:Some(scope),graph:Some(graph),owner:self})
    }
    fn push(&self,value:&MlxTensor)->Result<(),Error> {
        let state=self.state();let mut values=state.values.borrow_mut();
        if values.len()==state.expected{return Err(mismatch());}
        values.push(MlxTensor::from_array(value.as_array().try_clone_handle()?));Ok(())
    }
    /// Called after the shared coordinator publishes its updated random state.
    /// The same counted final frontier includes that key and all assertion roots.
    pub(super) fn finish(&self,random:Option<&RandomState>,stream:&Stream)->Result<(),Error> {
        let state=self.state();
        if !state.constructed.get()||state.completed.get(){return Err(mismatch());}
        if let Some(random)=random {
            let mut values=state.values.borrow_mut();if values.len()==state.expected{return Err(mismatch());}
            values.push(MlxTensor::from_array(random.as_array().try_clone_handle()?));
        }
        let observer=state.observer.borrow();let observer=observer.as_ref().ok_or_else(mismatch)?;
        let values=CompletingRoots::take(&state.values)?;
        state.nested.borrow_mut().complete(|visit|{for value in values.values.iter(){visit(value);}Ok(())},observer,stream)?;
        state.nested.borrow().validate_complete()?;
        // Waiting settles the scope, while each native descriptor can still
        // retain its completed event. Publish the same root descriptors through
        // the existing authenticated validator before the next cold visit.
        // This performs no Eval, progress or wait; pending/foreign roots refuse.
        for value in &values.values {observer.validate_completed_array(value.as_array())?;}
        drop(values);
        state.completed.set(true);Ok(())
    }
    pub(super) fn mechanism(&self)->OriginalFrameCompletionMechanism {OriginalFrameCompletionMechanism{owner:self.clone()}}
}
impl Completion for OriginalRealtimeCompletion {
    type Error=Error;
    fn is_complete(&self)->Result<bool,Error> {
        let state=self.state();let observer=state.observer.borrow();
        let Some(observer)=observer.as_ref() else {return Ok(false);};
        if let Some(cause)=observer.retained_failure(){return Err(cause.into());}
        let status=observer.status();
        if status.failed()||status.blocked(){return Err(Error::OriginalOperationCompletion{
            settled:status.is_settled(),failed:status.failed(),blocked:status.blocked()});}
        Ok(state.completed.get()&&status.is_settled())
    }
    fn resources_releasable(&self)->bool {
        self.state().observer.borrow().as_ref().is_some_and(|observer|observer.status().is_settled())
    }
    fn wait(&self)->Result<(),Error>{while !self.is_complete()?{std::thread::yield_now();}Ok(())}
}
/// Finishes the actual collector on every exit, preserving assertions in the
/// outer recovery owner. Construction closes before that recovery seals.
pub(super) struct ValidationScope<'a> {
    scope:Option<TokenValidationScope>,graph:Option<PreparedResidentGraph>,owner:&'a OriginalRealtimeCompletion,
}
impl Drop for ValidationScope<'_> {
    fn drop(&mut self){
        if let Some(scope)=self.scope.take(){*self.owner.state().validations.borrow_mut()=Some(scope.finish());}
        drop(self.graph.take());
    }
}
pub(super) struct OriginalFrameCompletionMechanism {owner:OriginalRealtimeCompletion}
type Forward=(Option<MlxTensor>,eredu_architectures::moshi::ForwardContext<MlxTensor>);
impl RealtimeFrameCompletionMechanism<MlxTensor,MlxKeyValueTransactionBranch,Forward> for OriginalFrameCompletionMechanism {
    type Completion=MlxRealtimeCompletion;type Error=Error;
    fn complete(&mut self,input:MaterializedRealtimeInput<MlxTensor>,output:&CompletedRealtimeFrame<MlxTensor,MlxTensor>,
        state:&MlxKeyValueTransactionBranch,history:&RealtimePayloadHistory<MlxTensor>,execution:Option<Forward>)
        ->Result<MlxRealtimeCompletion,RealtimeCompletionCreationError<MlxRealtimeCompletion,Error>> {
        let owner=&self.owner;
        let result=(||{
            if owner.state().constructed.replace(true){return Err(mismatch());}
            owner.push(input.input_audio())?;
            for value in input.forced_audio().into_iter().chain(input.forced_text()) {owner.push(value)?;}
            for value in [output.text(),output.decision_audio(),output.sampled_audio()] {owner.push(value)?;}
            for value in output.aligned_audio().into_iter().chain(output.diagnostics()).chain(history.retained_values()) {owner.push(value)?;}
            let mut error=None;
            <MlxKeyValueState as RuntimeState<crate::backend::nn::shared::MlxNeuralBackend>>::visit_all_retained_values(
                state.deref(),&mut |value|if error.is_none(){error=owner.push(value).err();})
                .map_err(|cause|Error::StorageSource(eredu_core::BackendFailure::from_error(cause)))?;
            if let Some(cause)=error{return Err(cause);}
            if let Some((text,forward))=execution {
                for value in text.as_ref().into_iter().chain(forward.temporal_mask()).chain(forward.temporal_output())
                    .chain(forward.text_logits()).chain(forward.previous_depth_token()) {owner.push(value)?;}
            }
            Ok(MlxRealtimeCompletion::from_original(owner.clone()))
        })();
        result.map_err(|error|RealtimeCompletionCreationError::AfterSubmission{error,
            completion:MlxRealtimeCompletion::from_original(owner.clone())})
    }
    fn retained_resources(&self,completion:&MlxRealtimeCompletion)->usize {completion.retained_resources()}
}
fn mismatch()->Error {Error::PrefillControl(WorkingMemoryError::IdentityMismatch)}
fn overflow()->Error {Error::PrefillControl(WorkingMemoryError::Overflow)}
