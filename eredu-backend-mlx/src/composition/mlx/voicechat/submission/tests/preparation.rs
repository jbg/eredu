//! Startup output completion and failed preparation lifetime ownership.

use super::*;
use eredu_architectures::voicechat::tts_prompt::{AriaPromptConfig, AriaPromptPlan};
use eredu_nn::Tensor;

#[test]
fn prompt_preparation_retains_outputs_and_execution_until_completion_release() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let poisoned = Rc::new(Cell::new(false));
    let owner = Arc::new(Owner {
        _neural: (),
        poisoned: poisoned.clone(),
    });
    let weak = Arc::downgrade(&owner);
    let plan = AriaPromptPlan::new(
        AriaPromptConfig {
            sample_rate: 8,
            samples_per_frame: 2,
            audio_prompt_duration: 1.0,
            codebooks: 2,
            speech_pad: 16,
        },
        10,
    )
    .unwrap();
    let (prompt, completion) = submit_preparation(
        owner,
        || {
            let codes =
                MlxTensor::from_i32_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 5, 2], &stream)
                    .unwrap();
            Ok(plan.prepare_encoded(&codes, 1, 4, &stream)?)
        },
        |prompt, visitor| prompt.visit_retained_tensors(visitor),
    )
    .unwrap();
    assert!(
        weak.upgrade().is_some(),
        "completed output still retains its native owner"
    );
    assert_eq!(
        completion.retained_resources(),
        1,
        "prompt root belongs to completion"
    );
    completion.wait().unwrap();
    assert_eq!(
        prompt.codes().to_i32_vec(&stream).unwrap(),
        [16, 16, 2, 3, 4, 5, 16, 16]
    );
    assert!(!poisoned.get());
    drop(completion);
    submission_recovery::wait_for_retirement(|| weak.upgrade().is_none());
}

#[test]
fn failed_and_unwinding_preparation_poison_execution_without_publishing_output() {
    for unwind in [false, true] {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let poisoned = Rc::new(Cell::new(false));
        let owner = Arc::new(Owner {
            _neural: (),
            poisoned: poisoned.clone(),
        });
        let weak = Arc::downgrade(&owner);
        let visited = Cell::new(false);
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            submit_preparation::<MlxTensor>(
                owner,
                || {
                    let values = Array::from_slice(&[2_f32, 3.], &[2])
                        .multiply(Array::from_f32(7.), &stream)
                        .unwrap();
                    let _event = async_eval_with_event([&values]).unwrap();
                    if unwind {
                        panic!("injected startup preparation unwind");
                    }
                    Err(VoiceChatSubmissionError::Speech(TtsError::Configuration(
                        "injected after eager preparation",
                    )))
                },
                |output, visitor| {
                    visited.set(true);
                    visitor(output);
                },
            )
        }));
        if unwind {
            assert!(outcome.is_err());
        } else {
            assert!(matches!(
                outcome,
                Ok(Err(VoiceChatSubmissionError::Speech(
                    TtsError::Configuration("injected after eager preparation")
                )))
            ));
        }
        assert!(
            !visited.get(),
            "failed preparation has no publishable output"
        );
        assert!(poisoned.get());
        submission_recovery::wait_for_retirement(|| weak.upgrade().is_none());
    }
}
