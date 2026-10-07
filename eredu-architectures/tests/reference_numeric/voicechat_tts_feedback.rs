use super::*;
use eredu_architectures::voicechat::tts_feedback::{
    FeedbackControls, OutputPolicy, WaveformSilence,
};

#[test]
fn voicechat_tts_feedback_preserves_waveform_before_eos_and_pad_substitution() {
    let context = NumericContext::default();
    let controls = FeedbackControls::new(8, 2, 12, true, &[10, 9, 8], &[3, 1, 6]).unwrap();
    // EOS, ordinary text, and two PAD rows with distinct facade permissions.
    let text = [2, 42, 12, 12];
    let generated =
        NumericTensor::from_i32_slice(&[0, 5, 7, 2, 4, 0, 4, 5, 7, 7, 2, 0], &[4, 1, 3], &context)
            .unwrap();
    let before = generated.to_i32_vec(&context).unwrap();
    let input = controls.prepare_input(&generated, &text, &context).unwrap();
    assert_eq!(
        input.to_i32_vec(&context).unwrap(),
        [3, 1, 6, 2, 4, 0, 4, 5, 7, 7, 2, 0],
        "only current EOS replaces incoming previous audio"
    );
    let policy = OutputPolicy {
        force_silence_on_eos: true,
        silence_on_pad: &[false, true, true, false],
        waveform: WaveformSilence::PreserveGenerated,
    };
    let main = controls
        .finish_output(&generated, &text, policy, &context)
        .unwrap();
    assert_eq!(main.waveform.to_i32_vec(&context).unwrap(), before);
    assert_eq!(
        main.next_feedback.to_i32_vec(&context).unwrap(),
        [3, 1, 6, 2, 4, 0, 3, 1, 6, 7, 2, 0],
        "ordinary content ignores PAD permission; unpermitted PAD keeps its tail"
    );
    let tool_wait = controls
        .finish_output(
            &generated,
            &text,
            OutputPolicy {
                waveform: WaveformSilence::ApplySilence,
                ..policy
            },
            &context,
        )
        .unwrap();
    assert_eq!(
        tool_wait.waveform.to_i32_vec(&context).unwrap(),
        main.next_feedback.to_i32_vec(&context).unwrap()
    );
    let no_post_eos = controls
        .finish_output(
            &generated,
            &text,
            OutputPolicy {
                force_silence_on_eos: false,
                silence_on_pad: &[false; 4],
                ..policy
            },
            &context,
        )
        .unwrap();
    assert_eq!(
        no_post_eos.next_feedback.to_i32_vec(&context).unwrap(),
        before,
        "input EOS setting does not enable the separate streaming output setting"
    );
    assert_eq!(generated.to_i32_vec(&context).unwrap(), before);
}

#[test]
fn voicechat_tts_feedback_matches_original_control_replacement_and_eos_input() {
    let context = NumericContext::default();
    let fixture: serde_json::Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let values = |value: &serde_json::Value| {
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap() as i32)
            .collect::<Vec<_>>()
    };
    let tensor = |value: &serde_json::Value| {
        let shape = value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        NumericTensor::from_i32_slice(&values(value), &shape, &context).unwrap()
    };
    let replacement = &fixture["control_replacement"];
    let controls = FeedbackControls::new(
        fixture["dimensions"]["cardinality"].as_i64().unwrap() as i32,
        2,
        12,
        true,
        &values(&replacement["controls"]),
        &values(&fixture["silence"]),
    )
    .unwrap();
    assert_eq!(
        controls
            .replace_controls(&tensor(&replacement["input"]), &context)
            .unwrap()
            .to_i32_vec(&context)
            .unwrap(),
        values(&replacement["output"]),
        "original replace_control_speech_codes preserves ordinary entries within each frame"
    );
    for call in fixture["calls"].as_array().unwrap().iter().skip(1) {
        assert_eq!(
            controls
                .prepare_input(
                    &tensor(&call["previous_codes"]),
                    &values(&call["subword_ids"]),
                    &context
                )
                .unwrap()
                .to_i32_vec(&context)
                .unwrap(),
            values(&call["input_codes"]),
            "original Duplex infer_codes_one_step applies current-EOS input substitution"
        );
    }
}
