//! Genuine core contexts with the existing neutral finite request fixture.
//! No native workspace proof or extra controller allocation allowance is invented.
use super::*;
use eredu_runtime::execution_control::ManagedTextContinuation;
use std::cell::Cell;

struct QueryController {
    inner: Controller,
    queries: Rc<Cell<usize>>,
    fail: Rc<Cell<bool>>,
}
impl TokenFilterController for QueryController {
    type Error = WorkingMemoryError;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.inner.current_filter()
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        self.inner.commit_token(token)
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.queries.set(self.queries.get() + 1);
        if self.fail.get() {
            Err(WorkingMemoryError::Overflow)
        } else {
            self.inner.is_complete()
        }
    }
}
fn query_controller(facts: &Rc<RefCell<Facts>>) -> QueryController {
    QueryController {
        inner: Controller(facts.clone()),
        queries: Rc::new(Cell::new(0)),
        fail: Rc::new(Cell::new(false)),
    }
}
fn check_query(
    facts: &Rc<RefCell<Facts>>,
    queries: &Rc<Cell<usize>>,
    mut query: impl FnMut() -> Result<bool, WorkingMemoryError>,
) {
    let before = {
        let f = facts.borrow();
        (f.decisions, f.commits, f.submissions, f.attempts.len())
    };
    let count = queries.get();
    for _ in 0..3 {
        assert!(!query().unwrap());
    }
    assert_eq!(queries.get(), count + 3);
    let f = facts.borrow();
    assert_eq!(
        (f.decisions, f.commits, f.submissions, f.attempts.len()),
        before
    );
}

#[test]
fn termination_queries_preserve_original_finite_context_across_cached_decodes() {
    for route in 0..3 {
        let (preparation, pool, _) = new_preparation(3, true);
        let (mut runtime, facts) = runtime(Some(preparation));
        let controller = query_controller(&facts);
        let queries = controller.queries.clone();
        let mut output = Vec::new();
        if route == 0 {
            let mut run =
                ControlledTextGeneration::new(&mut runtime, vec![1; 7], config(3), controller)
                    .unwrap();
            for _ in 0..3 {
                check_query(&facts, &queries, || run.controller_is_complete());
                output.push(run.next().unwrap().unwrap().token_id());
                check_query(&facts, &queries, || run.controller_is_complete());
            }
            assert!(run.next().is_none());
        } else {
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let mut state = driver
                .start_input(
                    eredu_core::TextGenerationInput::TokenIds(vec![1; 7]),
                    config(3),
                    controller,
                )
                .unwrap();
            if route == 1 {
                for _ in 0..3 {
                    check_query(&facts, &queries, || state.controller_is_complete());
                    output.push(driver.advance(&mut state).unwrap().unwrap().token_id());
                    driver.take_completed_delivery(&mut state).unwrap();
                    check_query(&facts, &queries, || state.controller_is_complete());
                }
                assert!(driver.advance(&mut state).unwrap().is_none());
            } else {
                let mut managed = ManagedTextContinuation::root(state);
                for _ in 0..3 {
                    check_query(&facts, &queries, || managed.controller_is_complete());
                    output.push(managed.advance(&mut driver).unwrap().unwrap().token_id());
                    managed.take_completed_delivery(&mut driver).unwrap();
                    check_query(&facts, &queries, || managed.controller_is_complete());
                }
                assert!(managed.advance(&mut driver).unwrap().is_none());
            }
        }
        assert_eq!(output, [7, 8, 9]);
        {
            let f = facts.borrow();
            assert_eq!((f.decisions, f.commits, f.submissions), (3, 3, 3));
            assert_eq!(f.bound.len(), 1);
            for (ordinal, context) in f.attempts.iter().enumerate() {
                assert_eq!(context.attempt(), ordinal as u64);
                assert_eq!(context.run_identity(), f.bound[0].run_identity());
                assert_eq!(context.policy_identity(), f.bound[0].policy_identity());
            }
        }
        drop(runtime);
        assert_eq!(pool.funded_used_bytes().unwrap(), 0);
    }
}

#[test]
fn query_errors_do_not_rebind_or_fence_finite_run_but_mutable_policy_still_invalidates_it() {
    for managed in [false, true] {
        let (preparation, pool, _) = new_preparation(3, true);
        let (mut runtime, facts) = runtime(Some(preparation));
        let controller = query_controller(&facts);
        let fail = controller.fail.clone();
        if managed {
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let state = driver
                .start_input(
                    eredu_core::TextGenerationInput::TokenIds(vec![1; 7]),
                    config(3),
                    controller,
                )
                .unwrap();
            let mut state = ManagedTextContinuation::root(state);
            fail.set(true);
            assert!(matches!(
                state.controller_is_complete(),
                Err(WorkingMemoryError::Overflow)
            ));
            fail.set(false);
            assert_eq!(state.advance(&mut driver).unwrap().unwrap().token_id(), 7);
            state.take_completed_delivery(&mut driver).unwrap();
            assert!(!state.controller_is_complete().unwrap());
            let _ = state.controller_mut();
            assert!(matches!(
                state.advance(&mut driver),
                Err(eredu_core::TextContinuationError::Generation(
                    eredu_core::ControlledTextGenerationError::Backend(
                        WorkingMemoryError::IdentityMismatch
                    )
                ))
            ));
        } else {
            let mut run =
                ControlledTextGeneration::new(&mut runtime, vec![1; 7], config(3), controller)
                    .unwrap();
            fail.set(true);
            assert!(matches!(
                run.controller_is_complete(),
                Err(WorkingMemoryError::Overflow)
            ));
            fail.set(false);
            assert_eq!(run.next().unwrap().unwrap().token_id(), 7);
            assert!(!run.controller_is_complete().unwrap());
            let _ = run.controller_mut();
            assert!(matches!(
                run.next().unwrap(),
                Err(eredu_core::ControlledTextGenerationError::Backend(
                    WorkingMemoryError::IdentityMismatch
                ))
            ));
        }
        {
            let f = facts.borrow();
            assert_eq!((f.decisions, f.commits, f.submissions), (1, 1, 1));
            assert_eq!(f.attempts.len(), 2);
            assert_eq!(
                f.attempts[0].policy_identity(),
                f.bound[0].policy_identity()
            );
            assert_ne!(
                f.attempts[1].policy_identity(),
                f.bound[0].policy_identity()
            );
            assert_eq!(f.attempts[1].run_identity(), f.bound[0].run_identity());
        }
        drop(runtime);
        assert_eq!(pool.funded_used_bytes().unwrap(), 0);
    }
}
