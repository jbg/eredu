use eredu_architectures::voicechat::{
    config::VoiceChatConfig,
    frontend::{
        validate_input_pcm, FrontendError, FrontendOptions, PcmPrecision, VoiceChatFrontend,
        INPUT_FRAME_SAMPLES,
    },
};
use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
use eredu_media::audio::streaming::AudioStreamError;
use eredu_media::audio::streaming::{FinalPcmFrame, PcmFinishPolicy, PcmFrameSource};
use eredu_runtime::execution_control::{SnapshotBudget, SnapshotValue};
use serde_json::Value;

#[test]
fn pcm_admission_rejects_selected_precision_overflow_before_queueing() {
    assert!(validate_input_pcm(&[65_504.0], 16_000, PcmPrecision::F16).is_ok());
    assert!(validate_input_pcm(&[70_000.0], 16_000, PcmPrecision::F32).is_ok());
    assert!(validate_input_pcm(&[70_000.0], 16_000, PcmPrecision::Bf16).is_ok());
    assert!(validate_input_pcm(&[70_000.0], 16_000, PcmPrecision::F16).is_err());
    assert!(validate_input_pcm(&[f32::MAX], 16_000, PcmPrecision::Bf16).is_err());
    assert!(validate_input_pcm(&[f32::NAN], 16_000, PcmPrecision::F32).is_err());
    assert!(validate_input_pcm(&[0.0], 8_000, PcmPrecision::F32).is_err());

    let mut frontend = frontend_with_precision(INPUT_FRAME_SAMPLES, false, PcmPrecision::F16);
    frontend.submit(&[0.5; 17], 16_000).unwrap();
    assert!(frontend
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::Partial,
            trailing_silence_frames: 1,
        })
        .is_err());
    assert!(!frontend.is_finished());
    assert!(matches!(
        frontend.submit(&[70_000.0], 16_000),
        Err(FrontendError::Audio(AudioStreamError::Invalid(_)))
    ));
    assert_eq!(frontend.queued_samples(), 17);
    frontend
        .submit(&[0.5; INPUT_FRAME_SAMPLES - 17], 16_000)
        .unwrap();
    assert!(frontend.has_ready_frame());
}

fn configured_frontend(capacity: usize, padded: bool) -> VoiceChatFrontend {
    frontend_with_precision(capacity, padded, PcmPrecision::F32)
}

#[test]
fn finite_silence_finish_matches_explicit_zero_pcm_without_source_sample_attribution() {
    let mut automatic = configured_frontend(INPUT_FRAME_SAMPLES, true);
    let mut manual = configured_frontend(INPUT_FRAME_SAMPLES, true);
    // Warm the retained window, then finish a nonzero partial frame so this
    // comparison exercises feature history across the input/continuation boundary.
    for chunk in pcm(3 * INPUT_FRAME_SAMPLES).chunks(INPUT_FRAME_SAMPLES) {
        automatic.submit(chunk, 16_000).unwrap();
        manual.submit(chunk, 16_000).unwrap();
        automatic = automatic.advance().unwrap().unwrap().state;
        manual = manual.advance().unwrap().unwrap().state;
    }
    let partial = pcm(17);
    automatic.submit(&partial, 16_000).unwrap();
    automatic
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::ZeroPad,
            trailing_silence_frames: 3,
        })
        .unwrap();
    assert!(automatic.is_finished());
    assert!(!automatic.is_drained());
    let mut full = vec![0.0; INPUT_FRAME_SAMPLES];
    full[..partial.len()].copy_from_slice(&partial);
    manual.submit(&full, 16_000).unwrap();
    let input = automatic.advance().unwrap().unwrap();
    let explicit = manual.advance().unwrap().unwrap();
    assert_eq!(input.mel, explicit.mel);
    assert_eq!(input.source, PcmFrameSource::Input);
    assert_eq!(input.valid_samples, partial.len());
    automatic = input.state;
    manual = explicit.state;

    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 0,
        retained_bytes: 1 << 24,
        cumulative_copy_bytes: 1 << 24,
    });
    let retention = budget
        .reserve(
            SnapshotResourceKind::Snapshot,
            Some(automatic.estimate_snapshot_value().unwrap()),
        )
        .unwrap();
    let mut replay = automatic.copy_snapshot_value(&retention).unwrap();
    for frame in 4..7 {
        manual.submit(&[0.0; INPUT_FRAME_SAMPLES], 16_000).unwrap();
        let explicit = manual.advance().unwrap().unwrap();
        let generated = automatic.advance().unwrap().unwrap();
        let restored = replay.advance().unwrap().unwrap();
        assert_eq!(generated.mel, explicit.mel);
        assert_eq!(restored.mel, explicit.mel);
        assert_eq!(generated.source, PcmFrameSource::Continuation);
        assert_eq!(generated.valid_samples, 0);
        assert_eq!(generated.frame_index, frame);
        assert_eq!(generated.start_sample, frame * INPUT_FRAME_SAMPLES as u64);
        assert_eq!(
            automatic.frames(),
            frame,
            "uncommitted extraction preserves the suffix"
        );
        automatic = generated.state;
        replay = restored.state;
        manual = explicit.state;
    }
    assert!(automatic.is_drained());
    assert!(replay.is_drained());
    assert!(automatic.advance().unwrap().is_none());
    assert_eq!(automatic.frames(), 7);
}
fn frontend_with_precision(
    capacity: usize,
    padded: bool,
    pcm_precision: PcmPrecision,
) -> VoiceChatFrontend {
    let config = VoiceChatConfig::from_json(include_str!(
        "../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    VoiceChatFrontend::new(
        &config.perception.preprocessor,
        FrontendOptions {
            pcm_precision,
            buffer_frames: 3,
            queue_capacity_samples: capacity,
            pad_and_drop_preencoded: padded,
        },
    )
    .unwrap()
}
fn pcm(count: usize) -> Vec<f32> {
    (0..count as i64)
        .map(|i| (((i * 73 + i * i * 19) % 2048) - 1024) as f32 / 8192.0)
        .collect()
}
fn compare(actual: &[f32], expected: &Value, tolerance: f32) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    let max = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b.as_f64().unwrap() as f32).abs())
        .fold(0f32, f32::max);
    assert!(
        max <= tolerance,
        "logmel max error {max}, tolerance {tolerance}"
    );
}

#[test]
fn voicechat_window_schedule_matches_nvidia_through_pcm_rollover() {
    let reference: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MEL_REFERENCE_JSON).unwrap();
    let audio = pcm(4 * INPUT_FRAME_SAMPLES);
    let tolerance = reference["absolute_tolerance"].as_f64().unwrap() as f32;
    for padded in [false, true] {
        for chunk_size in [157, INPUT_FRAME_SAMPLES, audio.len()] {
            let mut frontend = configured_frontend(audio.len(), padded);
            let mut outputs = Vec::new();
            for chunk in audio.chunks(chunk_size) {
                frontend.submit(chunk, 16_000).unwrap();
                while let Some(step) = frontend.advance().unwrap() {
                    assert_eq!(frontend.frames(), step.frame_index);
                    assert_eq!(
                        step.start_sample,
                        step.frame_index * INPUT_FRAME_SAMPLES as u64
                    );
                    assert_eq!(step.valid_samples, INPUT_FRAME_SAMPLES);
                    assert_eq!(frontend.advance().unwrap().unwrap().mel, step.mel);
                    let schedule = if padded {
                        "padded_schedule"
                    } else {
                        "schedule"
                    };
                    let expected = &reference[schedule][step.frame_index as usize];
                    assert_eq!(
                        step.mel.frames,
                        expected["frames"].as_u64().unwrap() as usize
                    );
                    assert_eq!(
                        step.drop_preencoded,
                        expected["drop_extra_pre_encoded"].as_u64().unwrap() as usize
                    );
                    compare(&step.mel.values, &expected["values"], tolerance);
                    outputs.push(step.mel);
                    frontend = step.state;
                }
            }
            assert_eq!(outputs.len(), 4);
            assert_eq!(frontend.retained_samples(), 3 * INPUT_FRAME_SAMPLES);
            frontend
                .finish(PcmFinishPolicy {
                    final_frame: FinalPcmFrame::RequireComplete,
                    trailing_silence_frames: 0,
                })
                .unwrap();
            assert!(frontend.is_drained());
            assert!(frontend.advance().unwrap().is_none());
        }
    }
}

#[test]
fn frontend_backpressure_snapshot_and_explicit_partial_finish_preserve_audio() {
    let mut frontend = configured_frontend(INPUT_FRAME_SAMPLES, false);
    // Populate cached mel rows before preserving a partial ingress frame. The
    // next frame rolls the finite history origin and invalidates boundary rows.
    for frame in pcm(3 * INPUT_FRAME_SAMPLES).chunks(INPUT_FRAME_SAMPLES) {
        frontend.submit(frame, 16_000).unwrap();
        frontend = frontend.advance().unwrap().unwrap().state;
    }
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 2,
        max_branches: 0,
        retained_bytes: 1 << 24,
        cumulative_copy_bytes: 1 << 24,
    });
    let audio = pcm(INPUT_FRAME_SAMPLES + 3);
    frontend.submit(&audio[..17], 16_000).unwrap();
    assert!(frontend.advance().unwrap().is_none());
    let _snapshot_retention = budget
        .reserve(
            SnapshotResourceKind::Snapshot,
            Some(frontend.estimate_snapshot_value().unwrap()),
        )
        .unwrap();
    let snapshot = frontend.copy_snapshot_value(&_snapshot_retention).unwrap();
    assert!(matches!(
        frontend.submit(&audio[17..], 16_000),
        Err(FrontendError::Audio(AudioStreamError::Backpressure { .. }))
    ));
    assert_eq!(frontend.queued_samples(), 17);
    assert!(matches!(
        frontend.finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::RequireComplete,
            trailing_silence_frames: 0
        }),
        Err(FrontendError::Audio(
            AudioStreamError::NeedMoreSamples { .. }
        ))
    ));
    frontend
        .submit(&audio[17..INPUT_FRAME_SAMPLES], 16_000)
        .unwrap();
    let mut fork = snapshot;
    fork.submit(&audio[17..INPUT_FRAME_SAMPLES], 16_000)
        .unwrap();
    let step = frontend.advance().unwrap().unwrap();
    assert_eq!(step.mel, fork.advance().unwrap().unwrap().mel);
    let mut frontend = step.state;
    frontend
        .submit(&audio[INPUT_FRAME_SAMPLES..], 16_000)
        .unwrap();
    frontend
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::ZeroPad,
            trailing_silence_frames: 0,
        })
        .unwrap();
    let _finished_retention = budget
        .reserve(
            SnapshotResourceKind::Snapshot,
            Some(frontend.estimate_snapshot_value().unwrap()),
        )
        .unwrap();
    let finished = frontend.copy_snapshot_value(&_finished_retention).unwrap();
    let last = frontend.advance().unwrap().unwrap();
    let replayed = finished.advance().unwrap().unwrap();
    assert_eq!(last.mel, replayed.mel);
    assert_eq!(last.valid_samples, replayed.valid_samples);
    assert!(replayed.state.is_drained());
    assert_eq!(
        frontend.frames(),
        4,
        "tentative final extraction leaves the source unchanged"
    );
    assert_eq!(frontend.queued_samples(), 3);
    assert_eq!(last.start_sample, (4 * INPUT_FRAME_SAMPLES) as u64);
    assert_eq!(last.valid_samples, 3);
    assert_eq!(last.frame_index, 4);
    assert!(last.state.is_drained());
    assert_eq!(fork.frames(), 3);
    assert_eq!(last.state.retained_samples(), 3 * INPUT_FRAME_SAMPLES);
}

#[test]
fn bf16_pcm_ingress_matches_the_pinned_wrapper_before_f32_features() {
    let reference: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MEL_REFERENCE_JSON).unwrap();
    let audio = pcm(4 * INPUT_FRAME_SAMPLES);
    let mut frontend = frontend_with_precision(
        INPUT_FRAME_SAMPLES,
        false,
        FrontendOptions::default().pcm_precision,
    );
    let tolerance = reference["absolute_tolerance"].as_f64().unwrap() as f32;
    for (index, tick) in audio.chunks(INPUT_FRAME_SAMPLES).enumerate() {
        frontend.submit(&tick[..153], 16_000).unwrap();
        assert!(matches!(
            frontend.submit(&[f32::NAN], 16_000),
            Err(FrontendError::Audio(AudioStreamError::Invalid(_)))
        ));
        assert_eq!(frontend.queued_samples(), 153);
        frontend.submit(&tick[153..], 16_000).unwrap();
        let step = frontend.advance().unwrap().unwrap();
        compare(
            &step.mel.values,
            &reference["bf16_schedule"][index]["values"],
            tolerance,
        );
        frontend = step.state;
    }
    frontend
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::RequireComplete,
            trailing_silence_frames: 0,
        })
        .unwrap();
    assert!(matches!(
        frontend.submit(&[], 16_000),
        Err(FrontendError::Audio(AudioStreamError::Finished))
    ));
}
