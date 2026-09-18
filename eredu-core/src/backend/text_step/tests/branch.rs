use super::*;
use crate::execution_control::{ControlSupport, NativeTextStateBackend, SnapshotEstimate};

// This mechanism has no model cache; all mutable execution state is the actual
// pending input and sampler already owned by the shared ordinary machine.
impl NativeTextStateBackend for Backend {
    type NativeTextState = ();
    fn native_text_state_support(_: &ModelRuntime<Self>) -> ControlSupport<&'static str> { ControlSupport::Supported }
    fn estimate_native_text_state(_: &ModelRuntime<Self>, _: Option<&()>) -> Result<Option<SnapshotEstimate>, io::Error> {
        Ok(Some(SnapshotEstimate { retained_bytes: 0, copy_bytes: 0 }))
    }
    fn capture_native_text_state(_: &mut ModelRuntime<Self>) -> Result<(), io::Error> { Ok(()) }
    fn copy_native_text_state(_: &mut ModelRuntime<Self>, _: &()) -> Result<(), io::Error> { Ok(()) }
    fn validate_native_text_state(_: &ModelRuntime<Self>, _: &()) -> Result<(), io::Error> { Ok(()) }
    fn exchange_native_text_state(_: &mut ModelRuntime<Self>, _: &mut ()) -> Result<(), io::Error> { Ok(()) }
    fn exchange_text_branch(runtime: &mut ModelRuntime<Self>, installed: TextBranchSource<'_, Self>, incoming: TextBranchSource<'_, Self>, _: &mut ()) -> Result<(), io::Error> {
        assert!(installed.pending().is_some());
        assert!(incoming.pending().is_some());
        assert!(!runtime.backend().0.borrow().active);
        let mut facts = runtime.backend().0.borrow_mut();
        if facts.fail_branch_exchange { return Err(io::Error::other("injected branch placement refusal")); }
        facts.branch_exchanges.push((installed.context().run_identity().clone(), incoming.context().run_identity().clone()));
        Ok(())
    }
}

fn start<'a>(runtime: &'a mut ModelRuntime<Backend>, facts: &Rc<RefCell<Facts>>, first: u32)
    -> ControlledTextGeneration<'a, Backend, Controller> {
    ControlledTextGeneration::new(runtime, vec![first], config(), controller(facts, ControllerFailure::None)).unwrap()
}
type BranchError = TextContinuationError<io::Error, io::Error>;

#[test]
fn borrowed_branch_moves_actual_pending_input_and_keeps_each_attempt_sequence() {
    let (mut runtime, facts) = fixture();
    let mut parent = start(&mut runtime, &facts, 13);
    assert_eq!(parent.next().unwrap().unwrap().token_id, 7);
    let parent_run = parent.inner.step_context.run_identity().clone();
    let (mut branch, marker) = parent.fork_completed(|runtime| {
        Ok::<_, BranchError>(Some((start(runtime, &facts, 29), (), 73)))
    }, |error| error).unwrap().unwrap();
    assert_eq!(marker, 73);
    assert_eq!(parent.inner.step_context.attempt(), 1);
    assert_eq!(parent.inner.step_context.run_identity(), &parent_run);
    assert!(matches!(parent.inner.step, Some(PendingTextInput::Decode(_))));
    parent.exchange_branch(&mut branch).unwrap();
    assert_ne!(parent.inner.step_context.run_identity(), &parent_run);
    assert!(matches!(parent.inner.step.as_ref(), Some(PendingTextInput::Prefill(p)) if p.ids == [29]));
    assert_eq!(parent.next().unwrap().unwrap().token_id, 7);
    parent.exchange_branch(&mut branch).unwrap();
    assert_eq!(parent.inner.step_context.run_identity(), &parent_run);
    assert_eq!(parent.next().unwrap().unwrap().token_id, 7);
    assert_eq!(parent.inner.step_context.attempt(), 2);
    assert_eq!(facts.borrow().branch_exchanges.len(), 3);
    drop(parent);
    let mut replacement = start(&mut runtime, &facts, 41);
    assert!(matches!(replacement.exchange_branch(&mut branch), Err(TextContinuationError::IncompatibleDriver)));
    assert_eq!(facts.borrow().branch_exchanges.len(), 3);
}

#[test]
fn failed_parent_placement_fences_prediction_and_copy_after_child_installation() {
    let (mut runtime, facts) = fixture();
    let mut parent = start(&mut runtime, &facts, 13);
    facts.borrow_mut().fail_branch_exchange = true;
    assert!(parent.fork_completed(|runtime| {
        Ok::<_, BranchError>(Some((start(runtime, &facts, 29), (), ())))
    }, |error| error).is_err());
    facts.borrow_mut().fail_branch_exchange = false;
    assert!(matches!(parent.snapshot_source(), Err(TextContinuationError::Failed)));
    let before = facts.borrow().contexts.len();
    assert!(matches!(parent.next().unwrap(), Err(ControlledTextGenerationError::Preparation(_))));
    assert_eq!(facts.borrow().contexts.len(), before);
}
