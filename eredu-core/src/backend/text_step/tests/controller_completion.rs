use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct QueryController {
    inner: Controller,
    complete: Rc<Cell<bool>>,
    queries: Rc<Cell<usize>>,
    failure: Rc<RefCell<Option<io::Error>>>,
}
impl TokenFilterController for QueryController {
    type Error = io::Error;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.inner.current_filter()
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        self.inner.commit_token(token)
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.queries.set(self.queries.get() + 1);
        if let Some(error) = self.failure.borrow_mut().take() {
            return Err(error);
        }
        Ok(self.complete.get())
    }
}
fn query_controller(facts: &Rc<RefCell<Facts>>) -> QueryController {
    QueryController {
        inner: controller(facts, ControllerFailure::None),
        complete: Rc::new(Cell::new(false)),
        queries: Rc::new(Cell::new(0)),
        failure: Rc::new(RefCell::new(None)),
    }
}
fn unchanged_queries(
    facts: &Rc<RefCell<Facts>>,
    queries: &Rc<Cell<usize>>,
    mut query: impl FnMut() -> Result<bool, io::Error>,
    expected: bool,
) {
    let events = facts.borrow().events.clone();
    let attempts = facts.borrow().contexts.clone();
    let bound = facts.borrow().bound_contexts.clone();
    let before = queries.get();
    for _ in 0..3 {
        assert_eq!(query().unwrap(), expected);
    }
    assert_eq!(
        queries.get(),
        before + 3,
        "each actual mutable query executes"
    );
    let facts = facts.borrow();
    assert_eq!(facts.events, events);
    assert_eq!(facts.contexts, attempts);
    assert_eq!(facts.bound_contexts, bound);
}

#[test]
fn controlled_queries_preserve_bound_policy_and_current_termination_result() {
    let (mut runtime, facts) = fixture();
    let controller = query_controller(&facts);
    let complete = controller.complete.clone();
    let queries = controller.queries.clone();
    let mut run =
        ControlledTextGeneration::new(&mut runtime, vec![1], config(), controller).unwrap();
    let bound = facts.borrow().bound_contexts[0].clone();
    for _ in 0..3 {
        unchanged_queries(&facts, &queries, || run.controller_is_complete(), false);
        assert_eq!(run.next().unwrap().unwrap().token_id(), 7);
        unchanged_queries(&facts, &queries, || run.controller_is_complete(), false);
        assert_eq!(run.inner.step_context.run_identity(), bound.run_identity());
        assert_eq!(
            run.inner.step_context.policy_identity(),
            bound.policy_identity()
        );
    }
    let remaining = run.inner.remaining_tokens;
    let context = run.inner.step_context.clone();
    let completions = run.inner.completions.len();
    let pending = match run.inner.step.as_ref() {
        Some(PendingTextInput::Decode(token)) => Rc::as_ptr(&token.0),
        _ => panic!("three committed predictions leave the next decode input"),
    };
    assert_eq!(remaining, Some(5));
    complete.set(true);
    unchanged_queries(&facts, &queries, || run.controller_is_complete(), true);
    // The facade owns termination policy. A true query only reports the current
    // condition; it does not stop this core machine or consume its next input.
    assert_eq!(run.inner.remaining_tokens, remaining);
    assert_eq!(run.inner.step_context, context);
    assert_eq!(run.inner.completions.len(), completions);
    match run.inner.step.as_ref() {
        Some(PendingTextInput::Decode(token)) => assert_eq!(Rc::as_ptr(&token.0), pending),
        _ => panic!("query must preserve the pending decode input"),
    }
    let facts = facts.borrow();
    assert_eq!(facts.contexts.len(), 3);
    for (attempt, context) in facts.contexts.iter().enumerate() {
        assert_eq!(context.attempt(), attempt as u64);
        assert_eq!(context.run_identity(), bound.run_identity());
        assert_eq!(context.policy_identity(), bound.policy_identity());
    }
}

#[test]
fn detached_queries_do_not_change_pending_input_allowance_or_completed_boundary() {
    let (mut runtime, facts) = fixture();
    let controller = query_controller(&facts);
    let queries = controller.queries.clone();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_input(TextGenerationInput::TokenIds(vec![1]), config(), controller)
        .unwrap();
    let bound = facts.borrow().bound_contexts[0].clone();
    for attempt in 0..3 {
        let remaining = state.remaining_tokens();
        let prefill = state.is_prefill_pending();
        unchanged_queries(&facts, &queries, || state.controller_is_complete(), false);
        assert_eq!(state.remaining_tokens(), remaining);
        assert_eq!(state.is_prefill_pending(), prefill);
        assert_eq!(driver.advance(&mut state).unwrap().unwrap().token_id(), 7);
        // Asking does not implicitly drain the exact pending completion.
        unchanged_queries(&facts, &queries, || state.controller_is_complete(), false);
        assert!(state.require_quiescent().is_err());
        assert!(driver.take_completed_step(&mut state).unwrap().is_none());
        state.require_quiescent().unwrap();
        assert_eq!(
            facts.borrow().contexts[attempt].policy_identity(),
            bound.policy_identity()
        );
    }
}

#[derive(Debug)]
struct OriginalQueryError {
    identity: Arc<()>,
    formats: Arc<AtomicUsize>,
}
impl std::fmt::Display for OriginalQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.formats.fetch_add(1, Ordering::SeqCst);
        f.write_str("original completion query")
    }
}
impl std::error::Error for OriginalQueryError {}

#[test]
fn completion_query_returns_original_typed_error_without_formatting_or_policy_change() {
    for detached in [false, true] {
        let (mut runtime, facts) = fixture();
        let controller = query_controller(&facts);
        let failure = controller.failure.clone();
        let identity = Arc::new(());
        let formats = Arc::new(AtomicUsize::new(0));
        let error = io::Error::new(
            io::ErrorKind::Other,
            OriginalQueryError {
                identity: identity.clone(),
                formats: formats.clone(),
            },
        );
        if detached {
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let mut state = driver
                .start_input(TextGenerationInput::TokenIds(vec![1]), config(), controller)
                .unwrap();
            *failure.borrow_mut() = Some(error);
            let actual = state.controller_is_complete().unwrap_err();
            let original = actual
                .get_ref()
                .unwrap()
                .downcast_ref::<OriginalQueryError>()
                .unwrap();
            assert!(Arc::ptr_eq(&identity, &original.identity));
            assert_eq!(formats.load(Ordering::SeqCst), 0);
            assert_eq!(driver.advance(&mut state).unwrap().unwrap().token_id(), 7);
            driver.take_completed_step(&mut state).unwrap();
        } else {
            let mut run =
                ControlledTextGeneration::new(&mut runtime, vec![1], config(), controller).unwrap();
            let bound = facts.borrow().bound_contexts[0].clone();
            *failure.borrow_mut() = Some(error);
            let actual = run.controller_is_complete().unwrap_err();
            let original = actual
                .get_ref()
                .unwrap()
                .downcast_ref::<OriginalQueryError>()
                .unwrap();
            assert!(Arc::ptr_eq(&identity, &original.identity));
            assert_eq!(formats.load(Ordering::SeqCst), 0);
            assert_eq!(run.inner.step_context, bound);
            assert_eq!(run.next().unwrap().unwrap().token_id(), 7);
        }
        let facts = facts.borrow();
        assert_eq!(facts.contexts.len(), 1);
        assert_eq!(
            facts.contexts[0].policy_identity(),
            facts.bound_contexts[0].policy_identity()
        );
    }
}
