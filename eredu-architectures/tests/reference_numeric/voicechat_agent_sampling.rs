//! Original NVIDIA filtering probabilities with a shared supplied uniform draw.
use eredu_architectures::voicechat::agent_sampling::{
    AgentSampler, AgentSamplingError, AgentSamplingPolicy,
};
use eredu_core::TokenFilter;
use eredu_runtime::{PenaltyConfig, Sampler, SamplingBackend, TokenDomain};
use serde_json::Value;
use std::cell::RefCell;

pub(super) struct Sampling;
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Draw {
    pub(super) uniform: f32,
    pub(super) count: usize,
}

fn probabilities(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<_> = logits.iter().map(|x| (*x - max).exp()).collect();
    let total: f32 = weights.iter().sum();
    weights.iter().map(|x| *x / total).collect()
}

impl SamplingBackend for Sampling {
    type Logits = Vec<f32>;
    type Token = u32;
    type RandomState = Draw;
    type Context = RefCell<Option<Vec<f32>>>;
    type Error = String;

    fn error(message: String) -> String {
        message
    }
    fn validate_token(token: &u32, domain: TokenDomain, _: &Self::Context) -> Result<u32, String> {
        assert!((*token as usize) < domain.cardinality());
        Ok(*token)
    }
    fn scale_temperature(
        logits: &Vec<f32>,
        temperature: f32,
        _: &Self::Context,
    ) -> Result<Vec<f32>, String> {
        Ok(logits.iter().map(|x| *x / temperature).collect())
    }
    fn apply_penalties(
        logits: &Vec<f32>,
        history: &[u32],
        config: PenaltyConfig,
        _: &Self::Context,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(config.frequency_penalty, 0.);
        assert_eq!(config.presence_penalty, 0.);
        let mut unique = history.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            history.len(),
            "architecture supplies unique penalty IDs"
        );
        assert!(history.iter().all(|id| ![1, 2, 12].contains(id)));
        let mut output = logits.clone();
        for token in unique {
            let value = &mut output[token as usize];
            *value = if *value > 0. {
                *value / config.repeat_penalty
            } else {
                *value * config.repeat_penalty
            };
        }
        Ok(output)
    }
    fn apply_top_p(
        mut logits: Vec<f32>,
        top_p: f32,
        _: &Self::Context,
    ) -> Result<Vec<f32>, String> {
        let weights = probabilities(&logits);
        let mut order: Vec<_> = (0..logits.len()).collect();
        order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
        let mut cumulative = 0.;
        for index in order {
            if cumulative > top_p {
                logits[index] = f32::NEG_INFINITY;
            }
            cumulative += weights[index];
        }
        Ok(logits)
    }
    fn sample_raw(
        logits: &Vec<f32>,
        temperature: f32,
        random: Option<&mut Draw>,
        _: &Self::Context,
    ) -> Result<u32, String> {
        assert_eq!(temperature, 0.);
        assert!(random.is_none(), "greedy probe cannot own RNG");
        let mut best = 0;
        for index in 1..logits.len() {
            if logits[index] > logits[best] {
                best = index;
            }
        }
        Ok(best as u32)
    }
    fn sample_processed(
        logits: &Vec<f32>,
        temperature: f32,
        random: Option<&mut Draw>,
        context: &Self::Context,
    ) -> Result<u32, String> {
        assert_eq!(temperature, 1.);
        let random = random.expect("supplied uniform");
        random.count += 1;
        let weights = probabilities(logits);
        let mut cumulative = 0.;
        let mut selected = weights.len() - 1;
        for (index, value) in weights.iter().enumerate() {
            cumulative += value;
            if random.uniform < cumulative {
                selected = index;
                break;
            }
        }
        *context.borrow_mut() = Some(weights);
        Ok(selected as u32)
    }
    fn token_id(token: &u32, _: &Self::Context) -> Result<u32, String> {
        Ok(*token)
    }
    fn token_probability(logits: &Vec<f32>, token: u32, _: &Self::Context) -> Result<f32, String> {
        Ok(probabilities(logits)[token as usize])
    }
    fn apply_top_k(_: Vec<f32>, _: i32, _: &Self::Context) -> Result<Vec<f32>, String> {
        panic!("VoiceChat has no top-k policy")
    }
    fn apply_min_p(_: Vec<f32>, _: f32, _: &Self::Context) -> Result<Vec<f32>, String> {
        panic!("VoiceChat has no min-p policy")
    }
    fn apply_token_filter(
        _: &Vec<f32>,
        _: &TokenFilter,
        _: &Self::Context,
    ) -> Result<Vec<f32>, String> {
        panic!("VoiceChat has no vocabulary filter")
    }
    fn apply_mirostat(
        _: &Vec<f32>,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        _: &Self::Context,
    ) -> Result<Vec<f32>, String> {
        panic!("VoiceChat has no Mirostat policy")
    }
}

#[test]
fn voicechat_agent_sampling_matches_nvidia_probabilities_and_draw_ownership() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::AGENT_SAMPLING_REFERENCE_JSON).unwrap();
    let absolute = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let relative = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for case in fixture["cases"].as_array().unwrap() {
        let policy = AgentSamplingPolicy::new(
            case["temperature"].as_f64().unwrap() as f32,
            case["top_p"].as_f64().unwrap() as f32,
            case["repetition_penalty"].as_f64().unwrap() as f32,
            32,
        )
        .unwrap();
        let history: Vec<_> = case["history"]
            .as_array()
            .unwrap()
            .iter()
            .take(case["current_step"].as_u64().unwrap() as usize)
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let logits = case["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let mut sampler = AgentSampler::new(policy);
        sampler.set_history(&history).unwrap();
        let mut random = Draw {
            uniform: case["uniform"].as_f64().unwrap() as f32,
            count: 0,
        };
        let context = RefCell::new(None);
        let token = <AgentSampler as Sampler<Sampling>>::sample(
            &mut sampler,
            &logits,
            case["temperature"].as_f64().unwrap() as f32,
            Some(&mut random),
            &context,
        )
        .unwrap();
        assert_eq!(
            token as u64,
            case["expected"]["token"].as_u64().unwrap(),
            "{}",
            case["name"]
        );
        assert_eq!(
            random.count as u64,
            case["expected"]["draws"].as_u64().unwrap(),
            "{}",
            case["name"]
        );
        assert_eq!(
            sampler.history(),
            history,
            "raw decisions do not become effective feedback implicitly"
        );
        match context.into_inner() {
            Some(actual) => {
                let expected = case["expected"]["probabilities"].as_array().unwrap();
                for (actual, expected) in actual.iter().zip(expected) {
                    let expected = expected.as_f64().unwrap() as f32;
                    assert!(
                        (*actual - expected).abs() <= absolute + relative * expected.abs(),
                        "{}: {actual} versus NVIDIA {expected}",
                        case["name"]
                    );
                }
            }
            None => assert!(case["expected"]["probabilities"].is_null()),
        }
    }
}

// This dense mechanism records stream position rather than evaluating a normal
// distribution. Distribution equations have their own numerical fixtures; here
// the risk is adoption overwriting a later dense draw or publishing a failed step.
impl eredu_runtime::DenseSamplingBackend for Sampling {
    fn uniform_f32(_: &[i32], _: &mut Draw, _: &Self::Context) -> Result<Vec<f32>, String> {
        unreachable!()
    }
    fn standard_normal_f32(
        shape: &[i32],
        random: &mut Draw,
        _: &Self::Context,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(shape, &[1]);
        random.count += 1;
        Ok(vec![-(random.count as f32)])
    }
    fn dense_shape(_: &Vec<f32>) -> &[i32] {
        &[1]
    }
    fn gumbel_max(
        _: &Vec<f32>,
        _: &Vec<f32>,
        _: eredu_runtime::GumbelTruncation,
        _: f32,
        _: &Self::Context,
    ) -> Result<u32, String> {
        unreachable!()
    }
}

#[test]
fn voicechat_agent_feedback_and_dense_rng_publish_together_after_completion() {
    use eredu_core::{scheduler::SemanticStateTransaction, Completion};
    use eredu_runtime::{
        PredictionDirective, RealtimeGenerationBranch, RealtimeGenerationContract,
        RealtimeGenerationState, RealtimeScheduleState, SequentialDecisionPlan,
    };
    use std::convert::Infallible;

    #[derive(Clone, Default)]
    struct Model;
    impl SemanticStateTransaction for Model {
        type Branch = Self;
        type Error = Infallible;
        fn branch(&self) -> Result<Self, Infallible> {
            Ok(self.clone())
        }
        fn commit_branch(&mut self, branch: Self) -> Result<(), Infallible> {
            *self = branch;
            Ok(())
        }
    }
    #[derive(Clone)]
    struct Schedule(u64);
    impl RealtimeScheduleState for Schedule {
        type Identity = ();
        type Error = std::io::Error;
        fn identity(&self) -> &() {
            &()
        }
        fn validate_successor(&self, next: &Self) -> Result<(), Self::Error> {
            if next.0 == self.0 + 1 {
                Ok(())
            } else {
                Err(std::io::Error::other("invalid successor"))
            }
        }
    }
    struct Ready;
    impl Completion for Ready {
        type Error = Infallible;
        fn is_complete(&self) -> Result<bool, Infallible> {
            Ok(true)
        }
        fn wait(&self) -> Result<(), Infallible> {
            Ok(())
        }
    }
    type Generation = RealtimeGenerationState<Model, Schedule, (), AgentSampler, Draw, Ready>;
    type Branch = RealtimeGenerationBranch<Model, Schedule, (), AgentSampler, Draw, Ready>;
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::AGENT_SAMPLING_REFERENCE_JSON).unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "nucleus_crossing")
        .unwrap();
    let temperature = case["temperature"].as_f64().unwrap() as f32;
    let policy = AgentSamplingPolicy::new(
        temperature,
        case["top_p"].as_f64().unwrap() as f32,
        case["repetition_penalty"].as_f64().unwrap() as f32,
        32,
    )
    .unwrap();
    let mut sampler = AgentSampler::new(policy);
    let history: Vec<_> = case["history"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    sampler.set_history(&history).unwrap();
    let initial_random = Draw {
        uniform: case["uniform"].as_f64().unwrap() as f32,
        count: 0,
    };
    let mut state = Generation::new(
        Model,
        RealtimeGenerationContract::new((), 1),
        Schedule(0),
        (),
        vec![sampler.clone()],
        Some(initial_random.clone()),
    )
    .unwrap();
    let logits = case["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let execute = |branch: &mut Branch| {
        let context = RefCell::new(None);
        // A function/turn override is applied AFTER the Sample directive, so
        // its raw categorical RNG draw remains part of the committed stream.
        let plan =
            SequentialDecisionPlan::new([PredictionDirective::Sample], false, false).unwrap();
        let mut driver = branch
            .decision_driver::<Sampling>(plan, vec![temperature])
            .unwrap();
        let raw = driver
            .resolve(0, &logits, TokenDomain::new(131_072), &context)
            .unwrap();
        branch.adopt_decision_driver(driver).unwrap();
        branch.samplers_mut()[0].accept_effective(12).unwrap();
        let dense = branch
            .draw_standard_normal::<Sampling>(&[1], &context)
            .unwrap();
        branch.schedule_state_mut().0 += 1;
        (raw, dense)
    };

    let mut discarded = state.branch().unwrap();
    let abandoned_trace = execute(&mut discarded);
    assert!(
        state.commit_branch(discarded).is_err(),
        "missing completion rejects publication"
    );
    assert_eq!(state.samplers(), &[sampler]);
    assert_eq!(state.random_state(), Some(&initial_random));
    assert_eq!(state.schedule_state().0, 0);

    // Two unpublished branches are independent forks of exactly the same state.
    let mut replay = state.branch().unwrap();
    let mut fork = state.branch().unwrap();
    let replay_trace = execute(&mut replay);
    assert_eq!(replay_trace, abandoned_trace);
    assert_eq!(execute(&mut fork), replay_trace);
    assert_eq!(
        replay_trace.0 as u64,
        case["expected"]["token"].as_u64().unwrap()
    );
    assert_ne!(
        replay_trace.0, 12,
        "raw choice differs from the effective override"
    );
    assert_eq!(
        replay_trace.1,
        vec![-2.],
        "dense draw follows the categorical draw"
    );
    assert_eq!(state.samplers()[0].history(), history);
    assert_eq!(state.random_state().unwrap().count, 0);
    replay.attach_submission_completion(Ready).unwrap();
    state.commit_branch(replay).unwrap();
    let mut effective = history;
    effective.push(12);
    assert_eq!(state.samplers()[0].history(), effective);
    assert_eq!(state.random_state().unwrap().count, 2);
    assert_eq!(state.schedule_state().0, 1);
    fork.attach_submission_completion(Ready).unwrap();
    assert!(
        state.commit_branch(fork).is_err(),
        "stale fork cannot publish again"
    );
    assert_eq!(state.samplers()[0].history(), effective);
    assert_eq!(state.random_state().unwrap().count, 2);
}

#[test]
fn voicechat_agent_history_failure_preserves_feedback_and_randomness() {
    let mut sampler = AgentSampler::new(AgentSamplingPolicy::new(0.5, 0.9, 1.2, 3).unwrap());
    sampler.set_history(&[3, 4]).unwrap();
    assert_eq!(
        sampler.set_history(&[5, 131_072]),
        Err(AgentSamplingError::Token { token: 131_072 })
    );
    assert_eq!(sampler.history(), &[3, 4]);
    assert_eq!(
        sampler.set_history(&[5, 6, 7, 8]),
        Err(AgentSamplingError::HistoryCapacity { capacity: 3 })
    );
    assert_eq!(sampler.history(), &[3, 4]);
    assert_eq!(
        sampler.accept_effective(131_072),
        Err(AgentSamplingError::Token { token: 131_072 })
    );
    assert_eq!(sampler.history(), &[3, 4]);

    // Control feedback occupies a history position even though repetition
    // penalties exclude it. Full history rejects before consuming randomness.
    sampler.accept_effective(12).unwrap();
    assert_eq!(
        sampler.preflight(),
        Err(AgentSamplingError::HistoryCapacity { capacity: 3 })
    );
    let mut random = Draw {
        uniform: 0.75,
        count: 7,
    };
    let context = RefCell::new(None);
    let mut logits = vec![-2.; 13];
    logits[0] = 3.;
    assert!(<AgentSampler as Sampler<Sampling>>::sample(
        &mut sampler,
        &logits,
        0.5,
        Some(&mut random),
        &context,
    )
    .is_err());
    assert_eq!(random.count, 7);
    assert!(context.into_inner().is_none());
    assert_eq!(sampler.history(), &[3, 4, 12]);
}
