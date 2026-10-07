use super::*;
use serde_json::Value;

fn controls() -> TextControls {
    TextControls {
        bos: 1,
        eos: 2,
        pad: 12,
        listening: 11,
        user_bos: 3,
        user_eos: 4,
        tool_call_start: 20,
        tool_call_end: 21,
        tool_response_end: 22,
    }
}

fn fixture_state(value: &Value) -> State {
    // The optional post-EOS fallback is disabled by the selected upstream
    // configuration; it is outside this RNNT policy's state contract.
    assert_eq!(value["post_eos_fired"].as_bool(), Some(false));
    State {
        blank_count: value["blank_count"].as_u64().unwrap(),
        nonblank_consec: value["nonblank_consec"].as_u64().unwrap(),
        nonblank_total: value["nonblank_total"].as_u64().unwrap(),
        speech_confirmed: value["speech_confirmed"].as_bool().unwrap(),
        agent_speaking: value["agent_speaking"].as_bool().unwrap(),
        first_turn: value["first_turn"].as_bool().unwrap(),
        user_first_turn: value["user_first_turn"].as_bool().unwrap(),
        rolling_density: value["rolling_density"].as_f64().unwrap() as f32,
        ..State::default()
    }
}

#[test]
fn rnnt_turn_policy_matches_original_nvidia_host_method() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TURN_REFERENCE_JSON).unwrap();
    let policy = VoiceChatTurnPolicy::default();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut state = fixture_state(&case["state"]);
        let history = case["history"].as_array().unwrap();
        state.model_steps = history.len() as u64;
        state.history = history[history.len().saturating_sub(policy.lookback)..]
            .iter()
            .map(|token| token.as_u64().unwrap() as u32)
            .collect();
        let mut expected = fixture_state(&case["expected"]["state"]);
        expected.model_steps = state.model_steps;
        expected.history = state.history.clone();
        let (token, event) = state.turn(
            case["raw_agent"].as_u64().unwrap() as u32,
            policy,
            controls(),
        );
        assert_eq!(
            token,
            case["expected"]["agent_token"].as_u64().unwrap() as u32,
            "{name}: effective agent token"
        );
        assert_eq!(state, expected, "{name}: RNNT policy successor");
        assert_eq!(
            event == Some(VoiceChatTurnEvent::Interrupted),
            case["expected"]["agent_eos_just_fired"].as_bool().unwrap(),
            "{name}: interruption flag"
        );
    }
}

#[test]
fn first_prediction_updates_frame_counters_and_density_freezes_during_agent_speech() {
    let limits = VoiceChatTurnLimits {
        max_audio_frames: 4,
        max_model_steps: 16,
        max_history: 40,
    };
    let mut state = State::default();
    // A speech frame may finish its RNNT label loop with blank. Evidence names
    // the first decision, so it still contributes exactly one speech frame.
    state
        .accept_audio(
            DecisionEvidence {
                first_blank: Some(false),
                audio_frame_index: Some(0),
            },
            limits,
        )
        .unwrap();
    assert_eq!(
        (
            state.blank_count,
            state.nonblank_consec,
            state.nonblank_total
        ),
        (0, 1, 1)
    );
    assert_eq!(state.rolling_density, 0.1);
    state
        .accept_audio(
            DecisionEvidence {
                first_blank: Some(false),
                audio_frame_index: Some(1),
            },
            limits,
        )
        .unwrap();
    assert_eq!(
        (
            state.blank_count,
            state.nonblank_consec,
            state.nonblank_total
        ),
        (0, 2, 2)
    );
    assert!((state.rolling_density - 0.19).abs() <= f32::EPSILON);
    let speech_density = state.rolling_density;
    state.agent_speaking = true;
    state
        .accept_audio(
            DecisionEvidence {
                first_blank: Some(true),
                audio_frame_index: Some(2),
            },
            limits,
        )
        .unwrap();
    assert_eq!(
        (
            state.blank_count,
            state.nonblank_consec,
            state.nonblank_total
        ),
        (1, 0, 2)
    );
    assert_eq!(state.rolling_density, speech_density);
    state.agent_speaking = false;
    state
        .accept_audio(
            DecisionEvidence {
                first_blank: Some(true),
                audio_frame_index: Some(3),
            },
            limits,
        )
        .unwrap();
    assert_eq!(
        (
            state.blank_count,
            state.nonblank_consec,
            state.nonblank_total
        ),
        (2, 0, 2)
    );
    assert!((state.rolling_density - 0.171).abs() <= f32::EPSILON);
    assert_eq!(state.audio_frames, 4);
    assert_eq!(state.model_steps, 0);
}
