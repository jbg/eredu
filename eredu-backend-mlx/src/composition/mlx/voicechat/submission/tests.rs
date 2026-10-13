use super::*;
use crate::{
    backend::runtime::cache::state::{
        JointCopyLimits, MlxHybridState, MlxJointState, MlxKeyValueState,
    },
    composition::mlx::voicechat::MlxVoiceChatGeneration,
};
use eredu_architectures::voicechat::{
    channel_decision::FeedbackTokens,
    config::VoiceChatConfig,
    frontend::FrontendOptions,
    input::InputState,
    schedule::{ScheduleLimits, SchedulePolicy},
    state::State,
};
use eredu_core::{
    cache::LayerCachePolicy, scheduler::SemanticStateTransaction, AttentionPolicy, LayerSchedule,
};
use eredu_nn::AttentionCache;
use eredu_runtime::{RealtimeGenerationState, RealtimeGenerationTransactionError, StateLayout};
use safemlx::{transforms::async_eval_with_event, Device, DeviceType};
use std::{panic::AssertUnwindSafe, rc::Rc};

mod backbone;
mod coupled;
mod preparation;

fn generation(stream: &Stream) -> MlxVoiceChatGeneration {
    let config =
        VoiceChatConfig::from_json(eredu_evaluation::voicechat::OFFICIAL_CONFIG_JSON).unwrap();
    let input =
        InputState::new(&config.perception.preprocessor, FrontendOptions::default()).unwrap();
    let hybrid = MlxHybridState::device(
        StateLayout::new(LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap()).unwrap(),
        &[],
    )
    .unwrap();
    let key_value = MlxKeyValueState::device(
        StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap()],
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let neural = MlxJointState::new(
        hybrid,
        key_value,
        stream,
        JointCopyLimits {
            max_branch_bytes: 1 << 20,
            max_total_copy_bytes: 1 << 20,
        },
    )
    .unwrap();
    let schedule = SchedulePolicy::from_config(
        &config,
        ScheduleLimits {
            max_audio_frames: 4,
            max_backbone_positions: 64,
            max_tts_positions: 128,
            max_codec_frames: 4,
        },
    )
    .unwrap();
    let sampling = AgentSamplingPolicy::streaming_defaults(16).unwrap();
    RealtimeGenerationState::new(
        State::new(input, neural, FeedbackTokens::new(2, 2).unwrap()),
        schedule.generation_contract(),
        ScheduleState::new(schedule),
        sampling,
        vec![AgentSampler::new(sampling)],
        Some(RandomState::with_seed(7).unwrap()),
    )
    .unwrap()
}

struct Owner<N> {
    _neural: N,
    poisoned: Rc<Cell<bool>>,
}
impl<N: 'static> MlxRealtimeExecutionResources for Owner<N> {
    fn poison(&self) {
        self.poisoned.set(true);
    }
}

#[test]
fn failed_and_unwinding_voicechat_invocations_attach_completion_without_publication() {
    for unwind in [false, true] {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let mut canonical = generation(&stream);
        let mut branch = canonical.branch().unwrap();
        let poisoned = Rc::new(Cell::new(false));
        let owner = Arc::new(Owner {
            _neural: branch.model_state().neural.retention(),
            poisoned: poisoned.clone(),
        });
        let weak = Arc::downgrade(&owner);
        let calls = Cell::new(0);
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            submit_invocation(&mut branch, owner, vec![], &stream, |candidate| {
                calls.set(calls.get() + 1);
                assert!(candidate.model_state_mut().begin_execution());
                let values = Array::from_slice(&[2_f32, 3.], &[2, 1, 1, 1])
                    .multiply(Array::from_f32(7.), &stream)
                    .unwrap();
                let (_, caches) = candidate.model_state_mut().neural.parts_mut();
                AttentionCache::<MlxTensor>::update_for_attention(
                    &mut caches[0],
                    MlxTensor::from_array(values.clone()),
                    MlxTensor::from_array(values.clone()),
                    &stream,
                )
                .unwrap();
                // Work has escaped lazy construction before either failure.
                let _event = async_eval_with_event([&values]).unwrap();
                if unwind {
                    panic!("injected coupled execution unwind");
                }
                Err(ExecutionError::Preflight("injected after eager work"))
            })
        }));
        if unwind {
            assert!(outcome.is_err());
        } else {
            assert!(matches!(
                outcome,
                Ok(Err(VoiceChatSubmissionError::Execution(
                    ExecutionError::Preflight("injected after eager work")
                )))
            ));
        }
        assert!(branch.has_submission_completion());
        assert!(poisoned.get());
        assert_eq!(calls.get(), 1);
        assert_eq!(
            AttentionCache::<MlxTensor>::offset(&branch.model_state_mut().neural.parts_mut().1[0]),
            1
        );
        // Failed submission never publishes. A nonblocking poll may report
        // pending while native work or runtime-lock contention prevents an
        // observation; both outcomes discard the candidate without retrying it.
        let committed = canonical.commit_branch(branch);
        assert!(
            matches!(
                committed,
                Err(RealtimeGenerationTransactionError::Completion(_)
                    | RealtimeGenerationTransactionError::CompletionPending)
            ),
            "failed submission unexpectedly committed: {committed:?}"
        );
        assert_eq!(canonical.revision(), 0);
        assert_eq!(
            AttentionCache::<MlxTensor>::offset(
                &canonical.model_state().neural().parts().1.as_ref()[0]
            ),
            0
        );
        assert_eq!(calls.get(), 1, "failure recovery cannot retry execution");
        submission_recovery::wait_for_retirement(|| weak.upgrade().is_none());
    }
}

#[test]
fn successful_voicechat_submission_releases_next_branch_after_host_delivery() {
    use eredu_architectures::voicechat::{
        output::{self, SpeechPayload},
        schedule::Step,
    };
    use eredu_core::scheduler::TransitionOutput;

    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut canonical = generation(&stream);
    let mut branch = canonical.branch().unwrap();
    let poisoned = Rc::new(Cell::new(false));
    let owner = Arc::new(Owner {
        _neural: branch.model_state().neural.retention(),
        poisoned: poisoned.clone(),
    });
    let weak = Arc::downgrade(&owner);
    // This synthetic prefill receipt isolates submission and host publication;
    // learned execution and its cache geometry have separate numerical coverage.
    let frame = submit_invocation(&mut branch, owner, vec![], &stream, |candidate| {
        assert!(candidate.model_state_mut().begin_execution());
        let plan = candidate
            .schedule_state()
            .plan(Step::Prefill { backbone_tokens: 0 })?;
        let observed = *plan.expected();
        let output = output::assemble(plan, observed, None, SpeechPayload::None, &stream)?;
        *candidate.schedule_state_mut() = output.schedule.clone();
        let hidden = Array::from_slice(&[2_f32, 3.], &[1, 1, 2])
            .multiply(Array::from_f32(7.), &stream)
            .unwrap();
        candidate.model_state_mut().retained_audio = Some(MlxTensor::from_array(hidden.clone()));
        Ok(ExecutionOutput {
            output,
            reminder_token: None,
            input: None,
            prompt_hidden: Some(MlxTensor::from_array(hidden)),
            backbone: None,
            decisions: None,
            speech: None,
            tail: None,
        })
    })
    .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !frame.is_complete().unwrap() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    canonical.commit_branch(branch).unwrap();
    assert_eq!(canonical.revision(), 1);
    assert_eq!(
        canonical
            .model_state()
            .retained_audio()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
        vec![14., 21.]
    );
    assert!(
        canonical.branch().is_err(),
        "host frame still retains cache admission"
    );
    let host = frame.into_host_output().unwrap();
    assert_eq!(host.invocation_index, 0);
    assert!(matches!(host.step, Step::Prefill { backbone_tokens: 0 }));
    assert!(host.speech.is_none());
    assert!(!poisoned.get());
    submission_recovery::wait_for_retirement(|| weak.upgrade().is_none());
    let next = canonical.branch().unwrap();
    MlxVoiceChatGeneration::discard_branch(next).unwrap();
}
