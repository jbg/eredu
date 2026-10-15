use super::*;
use crate::runtime::voicechat::VoiceChatTextLimits;
use eredu_architectures::voicechat::host::TextControls;
use eredu_text::tokenizer::Tokenizer;
use serde_json::json;
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit};

pub(crate) fn channel(request: u64, limits: VoiceChatToolLimits) -> VoiceChatToolChannel {
    let words = [
        ("[UNK]", 0),
        ("<s>", 1),
        ("</s>", 2),
        ("user-start", 3),
        ("user-end", 4),
        ("listen", 11),
        ("pad", 12),
        ("call-start", 20),
        ("call-end", 21),
        ("response-end", 22),
        (r#"{"name":"weather","arguments":{"city":"Paris"}}"#, 30),
        (
            r#"{"name":"weather","parameters":"{\"city\":\"Rome\"}"}"#,
            31,
        ),
        (r#"{"name":"weather","arguments":{"city":123}}"#, 32),
        (r#"{"name":"unknown","arguments":{}}"#, 33),
        ("[", 40),
        ("]", 41),
        (",", 42),
        ("<TOOLCALL>", 45),
        ("</TOOLCALL>", 46),
        ("<TOOL_RESPONSE>", 50),
        ("ok", 51),
        ("</TOOL_RESPONSE>", 52),
    ]
    .into_iter()
    .map(|(word, id)| (word.to_owned(), id))
    .collect();
    let model = WordLevel::builder()
        .vocab(words)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    let text = VoiceChatTextPreparation::from_tokenizer(
        &Tokenizer::from_tokenizer(tokenizer),
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
        },
        VoiceChatTextLimits {
            max_bytes: 8192,
            max_tokens: 2048,
        },
    )
    .unwrap();
    VoiceChatToolChannel::new(
        text,
        RequestId::new(request),
        vec![json!({
            "type":"function", "function": {"name":"weather", "parameters":{
                "type":"object", "properties":{"city":{"type":"string"}},
                "required":["city"], "additionalProperties":false
            }}
        })],
        limits,
    )
    .unwrap()
}
pub(crate) fn limits() -> VoiceChatToolLimits {
    VoiceChatToolLimits {
        max_payload_tokens: 32,
        max_payload_bytes: 2048,
        max_calls: 2,
    }
}
pub(crate) fn call(channel: &mut VoiceChatToolChannel) -> VoiceChatToolBatch {
    assert!(channel
        .observe_committed_function(channel.turn_identity(), 20)
        .unwrap()
        .is_none());
    assert!(channel
        .observe_committed_function(channel.turn_identity(), 30)
        .unwrap()
        .is_none());
    channel
        .observe_committed_function(channel.turn_identity(), 21)
        .unwrap()
        .unwrap()
}
fn complete_events(batch: &VoiceChatToolBatch) -> Vec<(usize, String, String, String)> {
    let mut calls = Vec::new();
    let mut open = None;
    for event in &batch.events {
        match event {
            SemanticEvent::ToolCallStart { index, id, name } => {
                assert!(open.is_none());
                open = Some((*index, id.clone(), name.clone(), String::new()));
            }
            SemanticEvent::ToolArgumentsDelta {
                index,
                json_fragment,
            } => {
                let current = open.as_mut().unwrap();
                assert_eq!(*index, current.0);
                current.3.push_str(json_fragment);
            }
            SemanticEvent::ToolCallCompleted {
                index,
                id,
                name,
                arguments,
            } => {
                let provisional = open.take().unwrap();
                assert_eq!(
                    (*index, id, name),
                    (provisional.0, &provisional.1, &provisional.2)
                );
                assert_eq!(
                    serde_json::Value::Object(arguments.clone()),
                    serde_json::from_str::<serde_json::Value>(&provisional.3).unwrap()
                );
                calls.push((
                    *index,
                    id.clone(),
                    name.clone(),
                    serde_json::to_string(arguments).unwrap(),
                ));
            }
            other => panic!("unexpected tool event {other:?}"),
        }
    }
    assert!(open.is_none());
    calls
}

#[test]
fn committed_function_batch_reuses_semantics_and_injects_only_committed_responses() {
    let mut channel = channel(7, limits());
    for token in [12, 20, 45, 40, 30, 42, 12, 31, 41, 46] {
        assert!(channel
            .observe_committed_function(channel.turn_identity(), token)
            .unwrap()
            .is_none());
    }
    let batch = channel
        .observe_committed_function(channel.turn_identity(), 21)
        .unwrap()
        .unwrap();
    let calls = complete_events(&batch);
    assert_eq!(calls.len(), 2);
    assert_eq!((calls[0].0, calls[1].0), (0, 1));
    assert_eq!(calls[0].2, "weather");
    assert_eq!(
        serde_json::from_str::<Value>(&calls[0].3).unwrap(),
        json!({"city":"Paris"})
    );
    assert_eq!(
        serde_json::from_str::<Value>(&calls[1].3).unwrap(),
        json!({"city":"Rome"})
    );
    assert_ne!(calls[0].1, calls[1].1);
    assert_eq!(channel.pending(), Some(batch.ticket));
    assert!(matches!(
        channel.observe_committed_function(channel.turn_identity(), 20),
        Err(VoiceChatToolError::Waiting)
    ));
    assert!(matches!(
        channel.submit_response(batch.ticket, "caf\u{e9}"),
        Err(VoiceChatToolError::Text(
            VoiceChatTextError::NonAscii { .. }
        ))
    ));
    assert_eq!(channel.pending(), Some(batch.ticket));
    channel
        .submit_response(batch.ticket, "<TOOL_RESPONSE> ok </TOOL_RESPONSE>")
        .unwrap();
    for token in [50, 51, 52] {
        let policy = Some(ChannelDecisionPolicy::Tool {
            function_feedback: Some(token),
            spoken_text: None,
        });
        assert_eq!(channel.response_policy(), policy);
        // Preparing or abandoning an unpublished invocation consumes nothing.
        assert_eq!(channel.response_policy(), policy);
        assert!(matches!(
            channel.observe_committed_function(channel.turn_identity(), 21),
            Err(VoiceChatToolError::Protocol(_))
        ));
        assert_eq!(channel.response_policy(), policy);
        assert!(channel
            .observe_committed_function(channel.turn_identity(), token)
            .unwrap()
            .is_none());
    }
    let terminal_policy = Some(ChannelDecisionPolicy::Tool {
        function_feedback: None,
        spoken_text: None,
    });
    assert_eq!(channel.response_policy(), terminal_policy);
    assert_eq!(channel.pending(), Some(batch.ticket));
    assert!(matches!(
        channel.observe_committed_function(channel.turn_identity(), 20),
        Err(VoiceChatToolError::Protocol(_))
    ));
    // Supplied text exhaustion does not finish model-driven function decoding.
    for token in [12, 51] {
        assert!(channel
            .observe_committed_function(channel.turn_identity(), token)
            .unwrap()
            .is_none());
        assert_eq!(channel.response_policy(), terminal_policy);
        assert_eq!(channel.pending(), Some(batch.ticket));
    }
    assert!(channel
        .observe_committed_function(channel.turn_identity(), 22)
        .unwrap()
        .is_none());
    assert!(channel.response_policy().is_none());
    assert!(channel.pending().is_none());
    let second = call(&mut channel);
    assert_eq!(complete_events(&second)[0].0, 2);
    assert_ne!(second.ticket, batch.ticket);
    assert!(matches!(
        channel.submit_response(batch.ticket, "ok"),
        Err(VoiceChatToolError::StaleResult)
    ));
}

#[test]
fn incomplete_and_invalid_calls_stay_bounded_without_partial_events_or_reset() {
    let mut short = channel(
        1,
        VoiceChatToolLimits {
            max_payload_tokens: 1,
            ..limits()
        },
    );
    short
        .observe_committed_function(short.turn_identity(), 20)
        .unwrap();
    short
        .observe_committed_function(short.turn_identity(), 30)
        .unwrap();
    assert!(matches!(
        short.observe_committed_function(short.turn_identity(), 31),
        Err(VoiceChatToolError::Capacity {
            resource: "payload tokens",
            ..
        })
    ));
    assert!(matches!(
        short.observe_committed_function(short.turn_identity(), 20),
        Err(VoiceChatToolError::Protocol(_))
    ));
    assert!(matches!(
        short.check_complete_payload(),
        Err(VoiceChatToolError::Incomplete)
    ));
    // Rejected appends do not destroy the already complete original payload.
    assert_eq!(
        complete_events(
            &short
                .observe_committed_function(short.turn_identity(), 21)
                .unwrap()
                .unwrap()
        )
        .len(),
        1
    );
    for token in [32, 33] {
        let mut invalid = channel(1, limits());
        invalid
            .observe_committed_function(invalid.turn_identity(), 20)
            .unwrap();
        invalid
            .observe_committed_function(invalid.turn_identity(), token)
            .unwrap();
        assert!(matches!(
            invalid.observe_committed_function(invalid.turn_identity(), 21),
            Err(VoiceChatToolError::Protocol(_))
        ));
        assert!(invalid.pending().is_none());
        assert!(matches!(
            invalid.check_complete_payload(),
            Err(VoiceChatToolError::Incomplete)
        ));
    }
    let mut bytes = channel(
        1,
        VoiceChatToolLimits {
            max_payload_bytes: 8,
            ..limits()
        },
    );
    bytes
        .observe_committed_function(bytes.turn_identity(), 20)
        .unwrap();
    bytes
        .observe_committed_function(bytes.turn_identity(), 30)
        .unwrap();
    assert!(matches!(
        bytes.observe_committed_function(bytes.turn_identity(), 21),
        Err(VoiceChatToolError::Capacity {
            resource: "decoded bytes",
            capacity: 8
        })
    ));
    let mut calls = channel(
        1,
        VoiceChatToolLimits {
            max_calls: 1,
            ..limits()
        },
    );
    for token in [20, 40, 30, 42, 31, 41] {
        calls
            .observe_committed_function(calls.turn_identity(), token)
            .unwrap();
    }
    assert!(matches!(
        calls.observe_committed_function(calls.turn_identity(), 21),
        Err(VoiceChatToolError::Capacity {
            resource: "calls",
            capacity: 1
        })
    ));
}

#[test]
fn interruptions_invalidate_waiting_and_partially_injected_results() {
    let mut original = channel(4, limits());
    let admitted_turn = original.turn_identity();
    let first = call(&mut original);
    let mut foreign = channel(5, limits());
    assert!(matches!(
        original.observe_committed_function(foreign.turn_identity(), 20),
        Err(VoiceChatToolError::StaleObservation)
    ));
    let foreign_ticket = call(&mut foreign).ticket;
    assert!(matches!(
        original.submit_response(foreign_ticket, "ok"),
        Err(VoiceChatToolError::StaleResult)
    ));
    assert_eq!(original.interrupt().unwrap(), 1);
    assert!(matches!(
        original.observe_committed_function(admitted_turn, 20),
        Err(VoiceChatToolError::StaleObservation)
    ));
    assert!(original.check_complete_payload().is_ok());
    assert!(matches!(
        original.submit_response(first.ticket, "ok"),
        Err(VoiceChatToolError::StaleResult)
    ));
    let next = call(&mut original);
    assert_eq!(next.ticket.turn(), 1);
    assert_eq!(complete_events(&next)[0].0, 0);
    assert_ne!(complete_events(&next)[0].1, complete_events(&first)[0].1);
    original
        .submit_response(next.ticket, "<TOOL_RESPONSE> ok </TOOL_RESPONSE>")
        .unwrap();
    original
        .observe_committed_function(original.turn_identity(), 50)
        .unwrap();
    assert_eq!(original.interrupt().unwrap(), 2);
    assert!(original.response_policy().is_none());
    assert!(matches!(
        original.submit_response(next.ticket, "ok"),
        Err(VoiceChatToolError::StaleResult)
    ));
    original
        .observe_committed_function(original.turn_identity(), 20)
        .unwrap();
    original
        .observe_committed_function(original.turn_identity(), 30)
        .unwrap();
    original.interrupt().unwrap();
    assert!(original.check_complete_payload().is_ok());
    assert!(original
        .observe_committed_function(original.turn_identity(), 21)
        .unwrap()
        .is_none());
}
