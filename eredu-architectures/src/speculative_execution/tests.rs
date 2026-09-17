use std::{
    fmt,
    sync::{Arc, Mutex},
};

use eredu_core::{Completion, SpeculativeExecutor};

use super::*;
use eredu_runtime::inspection::with_speculative_activation;

#[path = "phase_tests.rs"]
mod phase_tests;

#[path = "snapshot_tests.rs"]
mod snapshot_tests;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Tensor(Vec<i32>);

#[derive(Clone, Debug, Eq, PartialEq)]
struct Cache {
    fail_restore: bool,
    target: Vec<i32>,
    prediction: Vec<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Failure {
    None,
    Advance,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TestError {
    Message(String),
    Rollback(Arc<eredu_core::speculative::SpeculativeRollbackFailure<TestError, TestError>>),
}

impl fmt::Display for TestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => formatter.write_str(message),
            Self::Rollback(failure) => fmt::Display::fmt(failure, formatter),
        }
    }
}
impl std::error::Error for TestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self { Self::Message(_) => None, Self::Rollback(failure) => Some(failure.as_ref()) }
    }
}

#[derive(Debug)]
struct Done {
    retained: Arc<[Tensor]>,
}

impl Completion for Done {
    type Error = TestError;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl eredu_core::BoundedCompletion for Done {
    fn wait_bounded(
        self,
        _policy: eredu_core::BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
        Ok(eredu_core::BoundedCompletionOutcome::Completed)
    }
}

struct Mechanisms;

impl SpeculativeTensorMechanisms for Mechanisms {
    type Tensor = Tensor;
    type Logits = i32;
    type Context<'a> = ();
    type Completion = Done;
    type Error = TestError;

    fn observation_error(message: &'static str) -> Self::Error {
        TestError::Message(message.into())
    }

    fn control_tensor_estimate(
        value: &Tensor,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        Some(snapshot_tests::estimate(value.0.len()))
    }

    fn control_tensor_snapshot<'a>(
        value: &Tensor,
        _: (),
    ) -> Result<Option<Tensor>, SpeculativeControlError> {
        Ok(Some(value.clone()))
    }

    fn empty_prediction_input() -> Self::Error {
        TestError::Message("empty prediction input".into())
    }

    fn fused_prediction_exhausted() -> Self::Error {
        TestError::Message("prediction block exhausted".into())
    }

    fn invalid_prediction_commit(verified: usize, available: usize) -> Self::Error {
        TestError::Message(format!("invalid commit {verified}/{available}"))
    }

    fn invalid_prediction_output(
        logits: usize,
        capture: usize,
        tokens: usize,
        expected: Option<usize>,
    ) -> Self::Error {
        TestError::Message(format!(
            "invalid output {logits}/{capture}/{tokens}/{expected:?}"
        ))
    }

    fn invalid_fused_capacity(requested: usize, available: usize) -> Self::Error {
        TestError::Message(format!("invalid fused capacity {requested}/{available}"))
    }

    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
        Ok(value.0.len())
    }

    fn selected_prefill_logits(value: Self::Tensor) -> Result<Self::Logits, Self::Error> {
        value
            .0
            .last()
            .copied()
            .ok_or_else(Self::empty_prediction_input)
    }

    fn logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        value
            .0
            .get(row)
            .copied()
            .ok_or_else(|| TestError::Message("logits row is missing".into()))
    }

    fn tensor_row<'a>(
        value: &Self::Tensor,
        row: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        Self::logits_row(value, row, ()).map(|value| Tensor(vec![value]))
    }

    fn tensor_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(Tensor(value.0[..end].to_vec()))
    }

    fn token_range<'a>(
        value: &Self::Tensor,
        start: usize,
        end: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(Tensor(value.0[start..end].to_vec()))
    }

    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        Self::tensor_prefix(value, end, ())
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(Tensor(tokens.iter().map(|&token| token as i32).collect()))
    }

    fn fused_logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        Self::logits_row(value, row, ())
    }

    fn submit_verification_completion<'a>(
        output: &EmbeddedPredictionOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        _: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error> {
        Ok(Done {
            retained: Arc::from([
                output.logits.clone(),
                output.capture.clone(),
                output.tokens().clone(),
                inputs.clone(),
            ]),
        })
    }
}

#[derive(Clone, Debug)]
struct Strategy {
    fused_rows: Option<usize>,
    corrupt_capture: bool,
    failure: Failure,
}

impl Strategy {
    fn output(tokens: Tensor, corrupt_capture: bool) -> EmbeddedPredictionOutput<Tensor> {
        let logits = Tensor(tokens.0.iter().map(|token| token + 100).collect());
        let mut capture = Tensor(tokens.0.iter().map(|token| token + 200).collect());
        if corrupt_capture {
            capture.0.pop();
        }
        EmbeddedPredictionOutput::new(logits, capture, tokens)
    }
}

impl EmbeddedPredictionStrategy<Mechanisms> for Strategy {
    type Input = Vec<u32>;
    type TargetCache = Cache;
    type PredictionCache = Vec<i32>;
    type Telemetry = ();

    fn prepare_rollback_failure<'context: 'context>(_: ())
        -> Result<impl FnOnce(eredu_core::speculative::SpeculativeRollbackFailure<TestError, TestError>) -> TestError, TestError> {
        Ok(|failure| TestError::Rollback(Arc::new(failure)))
    }

    fn supports_internal_observations(&self) -> bool {
        self.fused_rows.is_none()
    }

    fn proposal_capacity(&self) -> usize {
        3
    }

    fn checkpoint_target(cache: &Self::TargetCache) -> Result<Self::TargetCache, TestError> {
        Ok(cache.clone())
    }

    fn control_target_estimate(
        &self,
        cache: &Cache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        Some(snapshot_tests::estimate(
            cache.target.len() + cache.prediction.len(),
        ))
    }

    fn control_prediction_estimate(
        &self,
        cache: &Vec<i32>,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        (!self.corrupt_capture).then(|| snapshot_tests::estimate(cache.len()))
    }

    fn control_target_snapshot<'a>(
        &self,
        cache: &Cache,
        _: (),
    ) -> Result<Option<Cache>, SpeculativeControlError> {
        assert!(!self.corrupt_capture, "unbounded copy must not start");
        Ok(Some(cache.clone()))
    }

    fn control_prediction_snapshot<'a>(
        &self,
        cache: &Vec<i32>,
        _: (),
    ) -> Result<Option<Vec<i32>>, SpeculativeControlError> {
        if self.failure == Failure::Advance {
            return Err(SpeculativeControlError::backend(TestError::Message(
                "seed copy failed".into(),
            )));
        }
        Ok(Some(cache.clone()))
    }

    fn take_telemetry(&mut self) -> Result<Self::Telemetry, TestError> {
        Ok(())
    }

    fn take_verification_telemetry(
        &mut self,
        _: &mut EmbeddedPredictionOutput<Tensor>,
    ) -> Result<Self::Telemetry, TestError> {
        Ok(())
    }

    fn prefill_target<'a>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::TargetCache,
        _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<Tensor, TestError>>,
    ) -> Result<EmbeddedPredictionOutput<Tensor>, TestError> {
        let tokens = Tensor(input.into_iter().map(|token| token as i32).collect());
        with_speculative_activation(
            observer,
            SpeculativeActivationPhase::TargetPrefill,
            tokens.0.len(),
            |observer| {
                cache.target.extend_from_slice(&tokens.0);
                let mut output = Self::output(tokens, self.corrupt_capture);
                if let Some(observer) = observer {
                    output.logits = eredu_runtime::observe_and_intervene(
                        observer,
                        "fixture.target.logits",
                        &output.logits,
                    )?;
                }
                Ok(output)
            },
        )
    }

    fn verify_target<'a>(
        &mut self,
        tokens: &EmbeddedPredictionTensor<Tensor>,
        cache: &mut Self::TargetCache,
        _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<Tensor, TestError>>,
        phase: SpeculativeActivationPhase,
    ) -> Result<EmbeddedPredictionOutput<Tensor>, TestError> {
        with_speculative_activation(observer, phase, tokens.0.len(), |observer| {
            cache.target.extend_from_slice(&tokens.0);
            let mut output = Self::output((**tokens).clone(), self.corrupt_capture);
            if let Some(observer) = observer {
                output.logits = eredu_runtime::observe_and_intervene(
                    observer,
                    "fixture.target.logits",
                    &output.logits,
                )?;
            }
            Ok(output)
        })
    }

    fn seed_prediction_cache<'a>(
        &mut self,
        _: &EmbeddedPredictionOutput<Tensor>,
        tokens: &Tensor,
        cache: &mut Self::TargetCache,
        _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<Tensor, TestError>>,
    ) -> Result<(), TestError> {
        if tokens.0.len() <= 1 {
            cache.prediction.clone_from(&tokens.0);
            return Ok(());
        }
        with_speculative_activation(
            observer,
            SpeculativeActivationPhase::PredictionPrefill,
            tokens.0.len() - 1,
            |observer| {
                cache.prediction.clone_from(&tokens.0);
                if let Some(observer) = observer {
                    observer.observe("fixture.prediction.seed", &Tensor(tokens.0[1..].to_vec()))?;
                }
                Ok(())
            },
        )
    }

    fn prediction_cache(&self, cache: &Self::TargetCache) -> Result<Self::PredictionCache, TestError> {
        Ok(cache.prediction.clone())
    }

    fn copy_prediction_cache(&self, cache: &Self::PredictionCache) -> Result<Self::PredictionCache, TestError> {
        Ok(cache.clone())
    }

    fn commit_prediction_cache(
        &self,
        cache: &mut Self::TargetCache,
        prediction: &Self::PredictionCache,
    ) -> Result<(), TestError> {
        cache.prediction.clone_from(prediction);
        Ok(())
    }

    fn restore_target_checkpoint<'a>(
        cache: &mut Self::TargetCache,
        checkpoint: &Self::TargetCache,
        _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
    ) -> Result<(), TestError> {
        if cache.fail_restore {
            return Err(TestError::Message("injected target restoration failure".into()));
        }
        cache.clone_from(checkpoint);
        Ok(())
    }

    fn sequential_logits<'a>(
        &mut self,
        capture: &Tensor,
        last_token: u32,
        depth: usize,
        cache: &mut Self::PredictionCache,
        _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<Tensor, TestError>>,
    ) -> Result<(i32, Tensor), TestError> {
        with_speculative_activation(
            observer,
            SpeculativeActivationPhase::Proposal { depth },
            1,
            |observer| {
                cache.push(last_token as i32);
                let mut logits = Tensor(vec![last_token as i32 + depth as i32]);
                if let Some(observer) = observer {
                    logits = eredu_runtime::observe_and_intervene(
                        observer,
                        "fixture.prediction.logits",
                        &logits,
                    )?;
                }
                Ok((logits.0[0], capture.clone()))
            },
        )
    }

    fn fused_logits<'a>(
        &mut self,
        _: &Tensor,
        _: u32,
        _: usize,
        _: &mut Self::PredictionCache,
        _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<Tensor, TestError>>,
    ) -> Result<Option<EmbeddedPredictionLogitBlock<Tensor>>, TestError> {
        if observer.is_some() && self.fused_rows.is_some() {
            return Err(Mechanisms::observation_error("fixture fused hooks absent"));
        }
        Ok(self
            .fused_rows
            .map(|rows| EmbeddedPredictionLogitBlock::ordinary(Tensor((0..rows as i32).collect()))))
    }

    fn advance_prediction_cache<'a>(
        &mut self,
        _: &Tensor,
        tokens: &Tensor,
        cache: &mut Self::PredictionCache,
        _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<Tensor, TestError>>,
    ) -> Result<(), TestError> {
        with_speculative_activation(
            observer,
            SpeculativeActivationPhase::PredictionReplay,
            tokens.0.len(),
            |observer| {
                if self.failure == Failure::Advance {
                    return Err(TestError::Message("injected prediction advance failure".into()));
                }
                cache.extend_from_slice(&tokens.0);
                if let Some(observer) = observer {
                    observer.observe("fixture.prediction.replay", tokens)?;
                }
                Ok(())
            },
        )
    }
}

fn cache() -> Cache {
    Cache {
        fail_restore: false,
        target: vec![9],
        prediction: vec![9],
    }
}

#[test]
fn embedded_cache_envelope_owns_prediction_fork_commit_and_target_membership() {
    let mut cache = EmbeddedPredictionCache::new(7_i32, vec![1_i32]);
    let checkpoint = cache
        .checkpoint(|target| Ok::<_, TestError>(*target))
        .unwrap();
    let mut draft = cache.prediction_fork().unwrap();
    draft.prediction_mut().push(2);
    cache.commit_prediction(&draft).unwrap();
    assert_eq!(cache.prediction(), &[1, 2]);

    let active = cache.take_target().unwrap();
    let error = cache
        .restore(&checkpoint, |current, previous| {
            *current = *previous;
            Ok::<_, TestError>(())
        })
        .unwrap_err();
    assert!(matches!(
        error,
        EmbeddedPredictionCacheAccessError::Cache(
            EmbeddedPredictionCacheError::TargetPresenceChanged
        )
    ));
    assert_eq!(cache.prediction(), &[1, 2]);
    cache.restore_target(active);
    cache
        .restore(&checkpoint, |current, previous| {
            *current = *previous;
            Ok::<_, TestError>(())
        })
        .unwrap();
    assert_eq!(cache.target(), Some(&7));
    assert_eq!(cache.prediction(), &[1]);
}

#[test]
fn sequential_partial_commit_replays_target_and_commits_prediction_state() {
    let mut strategy = Strategy {
        fused_rows: None,
        corrupt_capture: false,
        failure: Failure::None,
    };
    let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
    let mut cache = cache();
    let prefill = executor.prefill(vec![1, 2], &mut cache, ()).unwrap();
    let (_, state, _) = prefill.into_parts();
    let mut draft = executor.begin_proposal(&state, 2, 2, ()).unwrap();
    assert_eq!(executor.proposal_logits(&mut draft, 2, ()).unwrap(), 2);
    assert_eq!(executor.proposal_logits(&mut draft, 3, ()).unwrap(), 4);
    assert_eq!(
        executor.proposal_logits(&mut draft, 4, ()).unwrap_err().to_string(),
        "prediction block exhausted"
    );
    let checkpoint = executor.checkpoint(&cache).unwrap();
    let submission = executor
        .submit_verification(&[2, 3, 4], &mut cache, ())
        .unwrap();
    assert_eq!(submission.completion.retained.len(), 4);
    let commit = executor
        .commit_verification(submission.output, draft, &mut cache, &checkpoint, 2, ())
        .unwrap();
    let (_, replayed_tokens) = commit.into_parts();
    assert_eq!(replayed_tokens, 2);
    assert_eq!(cache.target, vec![9, 1, 2, 2, 3]);
    assert_eq!(cache.prediction, vec![1, 2, 2, 3, 3]);
}

#[test]
fn prefill_geometry_failure_restores_target_and_prediction_cache() {
    let mut strategy = Strategy {
        fused_rows: None,
        corrupt_capture: true,
        failure: Failure::None,
    };
    let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
    let mut cache = cache();
    let checkpoint = executor.checkpoint(&cache).unwrap();
    let error = executor.prefill(vec![1, 2], &mut cache, ()).err().unwrap();
    assert!(error.to_string().starts_with("invalid output"));
    assert_eq!(cache, checkpoint.cache);
}

#[test]
fn fused_capacity_is_rejected_before_any_proposal_row() {
    let mut strategy = Strategy {
        fused_rows: Some(1),
        corrupt_capture: false,
        failure: Failure::None,
    };
    let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
    let mut cache = cache();
    let prefill = executor.prefill(vec![1], &mut cache, ()).unwrap();
    let (_, state, _) = prefill.into_parts();
    let error = executor.begin_proposal(&state, 1, 2, ()).err().unwrap();
    assert_eq!(error.to_string(), "invalid fused capacity 2/1");
}

#[test]
fn commit_failure_restores_exact_preverification_checkpoint() {
    let mut strategy = Strategy {
        fused_rows: None,
        corrupt_capture: false,
        failure: Failure::Advance,
    };
    let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
    let mut cache = cache();
    let prefill = executor.prefill(vec![1, 2], &mut cache, ()).unwrap();
    let (_, state, _) = prefill.into_parts();
    let draft = executor.begin_proposal(&state, 2, 2, ()).unwrap();
    let checkpoint = executor.checkpoint(&cache).unwrap();
    let submission = executor
        .submit_verification(&[2, 3, 4], &mut cache, ())
        .unwrap();
    let error = executor
        .commit_verification(submission.output, draft, &mut cache, &checkpoint, 2, ())
        .err()
        .unwrap();
    assert_eq!(error.to_string(), "injected prediction advance failure");
    assert_eq!(cache, checkpoint.cache);
}

#[test]
fn production_observers_reach_causal_embedded_boundaries_and_can_intervene() {
    struct TensorTrace(Arc<Mutex<Vec<String>>>);

    impl eredu_runtime::ActivationObserver<Tensor, TestError> for TensorTrace {
        fn observe(&mut self, path: &str, _: &Tensor) -> Result<(), TestError> {
            self.0.lock().unwrap().push(path.into());
            Ok(())
        }

        fn intervene(&mut self, path: &str, value: &Tensor) -> Result<Option<Tensor>, TestError> {
            Ok((path == EMBEDDED_PREDICTION_OUTPUT_PATH).then(|| {
                let mut replacement = value.clone();
                replacement.0.fill(77);
                replacement
            }))
        }
    }

    struct LogitsTrace(Arc<Mutex<Vec<String>>>);

    impl eredu_runtime::ActivationObserver<i32, TestError> for LogitsTrace {
        fn observe(&mut self, path: &str, _: &i32) -> Result<(), TestError> {
            self.0.lock().unwrap().push(path.into());
            Ok(())
        }

        fn intervene(&mut self, _: &str, value: &i32) -> Result<Option<i32>, TestError> {
            Ok(Some(value + 100))
        }
    }

    let paths = Arc::new(Mutex::new(Vec::new()));
    let observers = EmbeddedPredictionObservers::new(
        TensorTrace(Arc::clone(&paths)),
        LogitsTrace(Arc::clone(&paths)),
    );
    let mut strategy = Strategy {
        fused_rows: None,
        corrupt_capture: false,
        failure: Failure::None,
    };
    let mut executor =
        EmbeddedPredictionExecutor::<_, Mechanisms>::with_observers(&mut strategy, observers);
    let mut cache = cache();
    let prefill = executor.prefill(vec![1, 2], &mut cache, ()).unwrap();
    let (_, state, _) = prefill.into_parts();
    let mut draft = executor.begin_proposal(&state, 2, 1, ()).unwrap();
    assert_eq!(executor.proposal_logits(&mut draft, 2, ()).unwrap(), 102);
    let _ = executor
        .submit_verification(&[2, 3], &mut cache, ())
        .unwrap();
    assert_eq!(
        *paths.lock().unwrap(),
        [
            EMBEDDED_TARGET_CAPTURE_PATH,
            EMBEDDED_PREDICTION_OUTPUT_PATH,
            EMBEDDED_PROPOSAL_LOGITS_PATH,
            EMBEDDED_VERIFICATION_LOGITS_PATH,
            EMBEDDED_TARGET_CAPTURE_PATH,
        ]
    );
}

#[test]
fn embedded_commit_keeps_operation_and_restoration_failures() {
    let mut strategy = Strategy { fused_rows: None, corrupt_capture: false, failure: Failure::Advance };
    let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
    let mut cache = cache();
    let prefill = executor.prefill(vec![1, 2], &mut cache, ()).unwrap();
    let (_, state, _) = prefill.into_parts();
    let draft = executor.begin_proposal(&state, 2, 2, ()).unwrap();
    let checkpoint = executor.checkpoint(&cache).unwrap();
    let submission = executor.submit_verification(&[2, 3, 4], &mut cache, ()).unwrap();
    cache.fail_restore = true;
    let error = executor.commit_verification(submission.output, draft, &mut cache, &checkpoint, 2, ())
        .err().unwrap();
    let pair = std::error::Error::source(&error).unwrap()
        .downcast_ref::<eredu_core::speculative::SpeculativeRollbackFailure<TestError, TestError>>().unwrap();
    assert_eq!(pair.operation.to_string(), "injected prediction advance failure");
    assert_eq!(pair.rollback.to_string(), "injected target restoration failure");
    assert_eq!(std::error::Error::source(pair).unwrap().to_string(), pair.operation.to_string());
    assert_ne!(cache, checkpoint.cache, "failed restoration must not certify reusable cache state");
}
