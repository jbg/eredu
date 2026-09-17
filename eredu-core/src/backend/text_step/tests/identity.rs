use super::*;

fn cause(error: &BackendFailure) -> TextContextError {
    *std::error::Error::source(error)
        .unwrap()
        .downcast_ref::<TextContextError>()
        .unwrap()
}

#[test]
fn concurrent_context_issuer_issues_last_ids_once_and_exhaustion_never_rewinds() {
    let issuer = AtomicU64::new(u64::MAX - 16);
    let mut issued = std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for _ in 0..8 {
            let issuer = &issuer;
            workers.push(
                scope.spawn(move || (0..8).filter_map(|_| issue_run(issuer)).collect::<Vec<_>>()),
            );
        }
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    issued.sort_unstable();
    assert_eq!(issued.len(), 16);
    assert_eq!(issued.first().unwrap().get(), u64::MAX - 15);
    assert_eq!(issued.last().unwrap().get(), u64::MAX);
    issued.dedup();
    assert_eq!(issued.len(), 16);
    for _ in 0..4 {
        assert!(issue_run(&issuer).is_none());
    }
    let mut failed = TextStepContext::from_run(None);
    failed.revise_policy();
    assert_eq!(
        cause(&failed.validate().unwrap_err()),
        TextContextError::RunExhausted
    );
    assert!(failed.following_attempt().is_none());
    assert_ne!(failed.run_identity(), TextStepContext::new().run_identity());
}

#[test]
fn exhausted_context_rejects_all_startup_routes_before_preparation_or_callbacks() {
    for route in 0..3 {
        let (mut runtime, facts) = fixture();
        let error = with_exhausted_run(|| match route {
            0 => TextGeneration::new(&mut runtime, vec![1, 2], config())
                .err()
                .unwrap(),
            1 => match ControlledTextGeneration::new(
                &mut runtime,
                vec![1, 2],
                config(),
                controller(&facts, ControllerFailure::None),
            )
            .err()
            .unwrap()
            {
                ControlledTextGenerationError::Preparation(error) => error,
                _ => panic!("fixed context failure"),
            },
            _ => {
                let mut driver = TextGenerationDriver::new(&mut runtime);
                match driver
                    .start_input(
                        TextGenerationInput::TokenIds(vec![1, 2]),
                        config(),
                        controller(&facts, ControllerFailure::None),
                    )
                    .err()
                    .unwrap()
                {
                    ControlledTextGenerationError::Preparation(error) => error,
                    _ => panic!("fixed context failure"),
                }
            }
        });
        assert_eq!(cause(&error), TextContextError::RunExhausted);
        let f = facts.borrow();
        assert!(f.preparation_events.is_empty());
        assert!(f.admission_votes.is_empty());
        assert!(f.preparation_votes.is_empty());
        assert!(f.events.is_empty());
        assert!(f.contexts.is_empty());
    }
}

#[test]
fn policy_exhaustion_is_terminal_before_prediction_and_survives_host_restore() {
    let (mut runtime, facts) = fixture();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2]),
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
    driver.advance(&mut state).unwrap().unwrap();
    driver.take_completed_step(&mut state).unwrap();
    let attempt = state.context_for_test().attempt();
    let old_run = state.context_for_test().run_identity().clone();
    state.context_mut_for_test().policy.revision = u64::MAX;
    let before_policy = state.context_for_test().policy_identity().clone();
    let mut child = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        // Existing infallible restore invalidates policy before installing data.
        boundary.install_host_state(
            controller(&facts, ControllerFailure::None),
            Some(PendingTextInput::Prefill(Prompt {
                ids: vec![4, 5],
                facts: facts.clone(),
            })),
            Some(8),
        );
        // A second restore cannot clear the terminal identity marker.
        boundary.install_host_state(
            controller(&facts, ControllerFailure::None),
            Some(PendingTextInput::Prefill(Prompt {
                ids: vec![6, 7],
                facts: facts.clone(),
            })),
            Some(8),
        );
        with_exhausted_run(|| {
            boundary.fork_host_state(
                State {
                    changes: 0,
                    facts: facts.clone(),
                },
                controller(&facts, ControllerFailure::None),
                Some(PendingTextInput::Prefill(Prompt {
                    ids: vec![8, 9],
                    facts: facts.clone(),
                })),
                Some(8),
            )
        })
    };
    assert_eq!(state.context_for_test().attempt(), attempt);
    assert_eq!(state.context_for_test().run_identity(), &old_run);
    assert_ne!(state.context_for_test().policy_identity(), &before_policy);
    let before = (
        facts.borrow().events.clone(),
        facts.borrow().votes.clone(),
        facts.borrow().contexts.len(),
    );
    for (state, expected) in [
        (&mut state, TextContextError::PolicyExhausted),
        (&mut child, TextContextError::RunExhausted),
    ] {
        for _ in 0..2 {
            let error = match driver.advance(state).err().unwrap() {
                TextContinuationError::Generation(ControlledTextGenerationError::Preparation(
                    error,
                )) => error,
                _ => panic!("typed terminal identity failure"),
            };
            assert_eq!(cause(&error), expected);
        }
    }
    assert_eq!(
        (
            facts.borrow().events.clone(),
            facts.borrow().votes.clone(),
            facts.borrow().contexts.len()
        ),
        before
    );
}

#[test]
fn terminal_controlled_agreement_helpers_preserve_local_errors_and_retire_values_without_hooks() {
    let (mut runtime, facts) = fixture();
    let mut run = ControlledTextGeneration::new(
        &mut runtime,
        vec![2, 3],
        config(),
        controller(&facts, ControllerFailure::None),
    )
    .unwrap();
    run.inner.step_context.policy.revision = u64::MAX;
    let _ = run.controller_mut();
    let before = (
        facts.borrow().events.clone(),
        facts.borrow().votes.clone(),
        facts.borrow().preparation_votes.clone(),
    );
    struct Value(Rc<Cell<usize>>);
    impl Drop for Value {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let dropped = Rc::new(Cell::new(0));
    let error = run
        .finish_text_preparation(Stage::Decision, Ok(Value(dropped.clone())), |error| {
            assert_eq!(dropped.get(), 1);
            error
        })
        .err()
        .unwrap();
    assert_eq!(cause(&error), TextContextError::PolicyExhausted);
    let error = run
        .finish_text_preparation_cancellable(
            Stage::Decision,
            Ok(Some(Value(dropped.clone()))),
            |error| {
                assert_eq!(dropped.get(), 2);
                error
            },
        )
        .err()
        .unwrap();
    assert_eq!(cause(&error), TextContextError::PolicyExhausted);
    let original = TextContextError::RunExhausted.into_backend_failure();
    let error = run
        .finish_text_preparation::<(), _>(Stage::Decision, Err(original), |_| {
            panic!("must preserve local failure")
        })
        .unwrap_err();
    assert_eq!(cause(&error), TextContextError::RunExhausted);
    let original = TextContextError::RunExhausted.into_backend_failure();
    let error = run
        .finish_text_preparation_cancellable::<(), _>(Stage::Decision, Err(original), |_| {
            panic!("must preserve local failure")
        })
        .unwrap_err();
    assert_eq!(cause(&error), TextContextError::RunExhausted);
    assert_eq!(
        (
            facts.borrow().events.clone(),
            facts.borrow().votes.clone(),
            facts.borrow().preparation_votes.clone()
        ),
        before
    );
}

#[test]
fn continuation_identity_tracks_the_existing_run_across_restore_fork_and_retirement() {
    let (mut runtime, facts) = fixture();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2]),
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
    let identity = state.identity_for_test();
    let run = state.context_for_test().run_identity().clone();
    driver.advance(&mut state).unwrap().unwrap();
    driver.take_completed_step(&mut state).unwrap();
    let mut child = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        assert!(identity == boundary.identity());
        boundary.install_host_state(
            controller(&facts, ControllerFailure::None),
            Some(PendingTextInput::Prefill(Prompt {
                ids: vec![3, 4],
                facts: facts.clone(),
            })),
            Some(8),
        );
        assert!(identity == boundary.identity());
        boundary.fork_host_state(
            State {
                changes: 0,
                facts: facts.clone(),
            },
            controller(&facts, ControllerFailure::None),
            Some(PendingTextInput::Prefill(Prompt {
                ids: vec![5, 6],
                facts: facts.clone(),
            })),
            Some(8),
        )
    };
    assert_eq!(state.context_for_test().run_identity(), &run);
    assert!(identity == state.identity_for_test());
    assert!(identity != child.identity_for_test());
    assert_ne!(
        state.context_for_test().run_identity(),
        child.context_for_test().run_identity()
    );
    let retained_child = child.identity_for_test();
    drop(state);
    assert!(identity != child.identity_for_test());
    driver.advance(&mut child).unwrap().unwrap();
    driver.take_completed_step(&mut child).unwrap();
    assert!(retained_child == driver.quiescent(&mut child).unwrap().identity());
    drop(child);
    let fresh = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![7, 8]),
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
    assert!(identity != fresh.identity_for_test());
    assert!(retained_child != fresh.identity_for_test());
}
