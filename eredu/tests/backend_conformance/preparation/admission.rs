use super::*;
use eredu_core::{
    ControlledTextGeneration, TextGeneration, TextGenerationDriver, TextGenerationInput,
};

#[derive(Clone, Default)]
struct Controller(std::rc::Rc<std::cell::Cell<usize>>);
impl eredu_core::TokenFilterController for Controller {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.0.set(self.0.get() + 1);
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        self.0.set(self.0.get() + 1);
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.0.set(self.0.get() + 1);
        Ok(false)
    }
}

fn config() -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            GenerationConfigOverrides {
                max_new_tokens: Some(2),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

#[test]
fn shared_preparation_accounts_spare_host_capacity_without_advancing_controller() {
    let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
    let _guard = probe(Fault::None);
    let mut ids = Vec::with_capacity(257);
    ids.extend([11, 7, 3]);
    let expected_capacity = ids.capacity() as u64 * 4;
    let controller = Controller::default();
    let observed = controller.clone();
    let mut generation =
        ControlledTextGeneration::new(&mut runtime, ids, config(), controller).unwrap();
    let prepared = snapshot();
    assert_eq!(observed.0.get(), 0);
    assert_eq!(prepared.admissions, [Some((3, expected_capacity))]);
    assert_eq!(prepared.lifecycle, ["admit", "prompt", "sampling"]);
    assert_eq!(
        prepared.votes,
        [
            (Stage::Admission, Status::Ready),
            (Stage::Prompt, Status::Ready),
            (Stage::Sampling, Status::Ready),
        ]
    );
    assert_eq!(prepared.forwards, 0);
    assert!(generation.next().unwrap().is_ok());
    assert!(observed.0.get() > 0);
    assert_eq!(snapshot().completion_charge, [true]);
    assert!(snapshot().charge.strong_count() > 0);
    drop(generation);
    let completed = snapshot();
    assert_eq!(completed.completion_charge, [true]);
    assert_eq!(completed.lifecycle.last(), Some(&"release"));
    assert_eq!(completed.charge.strong_count(), 0);
}

#[test]
fn ordinary_prepared_input_uses_admission_and_retains_it_until_drop_completion() {
    for prepared in [false, true] {
        let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
        let _guard = probe(Fault::None);
        let mut generation = if prepared {
            TextGeneration::from_prompt(&mut runtime, vec![11, 7, 3], config())
        } else {
            TextGeneration::new(&mut runtime, vec![11, 7, 3], config())
        }
        .unwrap();
        assert_eq!(
            snapshot().admissions,
            if prepared {
                vec![None]
            } else {
                vec![Some((3, 12))]
            }
        );
        assert_eq!(
            snapshot().native,
            if prepared {
                vec![Stage::Sampling]
            } else {
                vec![Stage::Prompt, Stage::Sampling]
            }
        );
        assert!(generation.next().unwrap().is_ok());
        assert!(snapshot().completion_charge.is_empty());
        assert!(snapshot().charge.strong_count() > 0);
        drop(generation);
        assert_eq!(snapshot().completion_charge, [true]);
        assert_eq!(snapshot().charge.strong_count(), 0);
    }
}

#[test]
fn detached_fork_retains_one_admission_after_driver_and_parent_drop() {
    let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
    let _guard = probe(Fault::None);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut parent = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![11, 7, 3]),
            config(),
            Controller::default(),
        )
        .unwrap();
    let child = {
        let boundary = driver.quiescent(&mut parent).unwrap();
        // Fixture components are independently staged, as required by the
        // snapshot contract. This only tests retention of the original charge;
        // copied native payloads require their own snapshot reservation.
        boundary.fork_host_state(
            observed_mock::State::default(),
            boundary.controller().clone(),
            Some(eredu_core::PendingTextInput::Prefill(vec![11, 7, 3])),
            Some(2),
        )
    };
    assert_eq!(snapshot().admissions.len(), 1);
    assert_eq!(snapshot().charge.strong_count(), 2);
    drop(driver);
    drop(parent);
    assert_eq!(snapshot().charge.strong_count(), 1);
    assert!(!snapshot().lifecycle.contains(&"release"));
    drop(child);
    assert_eq!(snapshot().charge.strong_count(), 0);
    assert_eq!(
        snapshot()
            .lifecycle
            .iter()
            .filter(|&&event| event == "release")
            .count(),
        1
    );
}

#[test]
fn ordinary_and_controlled_facades_share_admission_order_and_output() {
    let mut outputs = vec![];
    for controlled in [false, true] {
        let (mut model, chat, settings) = setup();
        let _guard = probe(Fault::None);
        outputs.push(run(&mut model, &chat, settings, controlled).unwrap());
        let seen = snapshot();
        assert_eq!(seen.admissions.len(), 1);
        assert_eq!(seen.inference_settings, [(Default::default(), Some(2))]);
        assert!(seen.admissions[0].is_some());
        assert_eq!(&seen.lifecycle[..3], ["admit", "prompt", "sampling"]);
        let admitted = seen
            .votes
            .iter()
            .position(|&(stage, _)| stage == Stage::Admission)
            .unwrap();
        assert!(admitted > 0);
        assert!(seen.votes[..admitted]
            .iter()
            .all(|&vote| vote == (Stage::Request, Status::Ready)));
        assert_eq!(
            &seen.votes[admitted..admitted + 5],
            [
                (Stage::Admission, Status::Ready),
                (Stage::Prompt, Status::Ready),
                (Stage::Sampling, Status::Ready),
                (Stage::Instrumentation, Status::Ready),
                (Stage::Delivery, Status::Ready),
            ]
        );
        assert_eq!(seen.charge.strong_count(), 0);
    }
    assert!(!outputs[0].is_empty());
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
fn facade_inference_policy_reaches_shared_admission_with_exact_output_allowance() {
    use eredu::api::TextInferencePolicy;
    for controlled in [false, true] {
        for explicit_allowance in [false, true] {
            for budget in [None, Some(0), Some(16 << 20)] {
                let (mut model, chat, mut settings) = setup();
                settings.overrides.max_new_tokens = explicit_allowance.then_some(2);
                settings.inference = TextInferencePolicy {
                    prefill_chunk_positions: std::num::NonZeroU64::new(3),
                    managed_memory_capacity_bytes: budget,
                    submission_tracking_capacity_bytes: None,
                    graph_metadata_capacity_bytes: None,
                };
                // Fail at admission so the test exercises preparation without
                // asking this neutral fixture to execute an unbounded output.
                let _guard = probe(Fault::Local(Stage::Admission));
                let error = run(&mut model, &chat, settings, controlled).unwrap_err();
                assert!(has_source::<MockError>(error.as_ref()));
                let seen = snapshot();
                assert_eq!(
                    seen.inference_settings,
                    [(
                        settings.inference,
                        if explicit_allowance {
                            Some(2)
                        } else if budget.is_some() {
                            Some(256)
                        } else {
                            None
                        }
                    )]
                );
                assert!(seen.native.is_empty());
                assert_eq!(seen.forwards, 0);
            }
        }
    }
}

#[test]
fn enforced_unbounded_core_requests_reject_before_backend_or_controller() {
    let config = TextGenerationConfig::new(
        eredu_core::resolve_generation_config(None, Default::default()).unwrap(),
    )
    .with_inference_policy(eredu_core::TextInferencePolicy {
        managed_memory_capacity_bytes: Some(16 << 20),
        ..Default::default()
    });
    assert_eq!(config.sampling().max_new_tokens, None);
    for controlled in [false, true] {
        for prepared in [false, true] {
            let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
            let _guard = probe(Fault::None);
            let controller = Controller::default();
            let observed = controller.clone();
            let error: Box<dyn std::error::Error> = if controlled {
                let input = if prepared {
                    TextGenerationInput::Prepared(vec![11, 7, 3])
                } else {
                    TextGenerationInput::TokenIds(vec![11, 7, 3])
                };
                ControlledTextGeneration::from_input(&mut runtime, input, config, controller)
                    .err()
                    .expect("missing output bound must reject")
                    .into()
            } else {
                let result = if prepared {
                    TextGeneration::from_prompt(&mut runtime, vec![11, 7, 3], config)
                } else {
                    TextGeneration::new(&mut runtime, vec![11, 7, 3], config)
                };
                result
                    .err()
                    .expect("missing output bound must reject")
                    .into()
            };
            assert!(
                has_source::<eredu_core::CapabilityError>(error.as_ref()),
                "{error}"
            );
            assert!(error.to_string().contains("finite output-token allowance"));
            assert_eq!(observed.0.get(), 0);
            let seen = snapshot();
            assert_eq!(seen.votes, [(Stage::Admission, Status::Failed)]);
            assert!(seen.admissions.is_empty());
            assert!(seen.native.is_empty());
            assert_eq!(seen.forwards, 0);
        }
    }
}
