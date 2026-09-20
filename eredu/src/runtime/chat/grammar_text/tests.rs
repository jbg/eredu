use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[test]
fn literal_spelling_matches_json_for_controls_unicode_and_delimiters() {
    let controls = (0..=0x1f).map(char::from).collect::<String>();
    for text in ["", "\"\\/\n\r\t", "mañana 水 🦀", &controls] {
        assert_eq!(
            format!("{}", Literal(text)),
            serde_json::to_string(text).unwrap()
        );
    }
    let source = StructuralTokens::new(&["<x>", "<x>y", "<z>"], &[7, 11, 99]).unwrap();
    assert_eq!(
        source.literal("α<x>y<z>\n").to_string(),
        "\"α\" <[7]> \"y\" <[99]> \"\\n\""
    );
    assert_eq!(source.literal("").to_string(), "\"\"");
    assert!(StructuralTokens::new(&[""], &[1]).is_err());
    assert!(StructuralTokens::new(&["x"], &[]).is_err());
}


fn render(funding: &PreparationFunding, schema: &serde_json::Value) -> Result<String, Error> {
    let source = StructuralTokens::new(&["<call>"], &[19])?;
    let mut output = Text::new(funding)?;
    output.push_str("start: ")?;
    output.push_fmt(format_args!("{}\n", source.literal("<call>")))?;
    output.push_str("args: %json ")?;
    output.push_json(schema)?;
    output.push_str("\n")?;
    let grammar = lark(output.finish(), funding)?;
    Ok(grammar
        .grammars
        .into_iter()
        .next()
        .unwrap()
        .lark_grammar
        .unwrap())
}

#[test]
fn every_reached_text_producer_refuses_with_original_cause() {
    let schema = serde_json::json!({"type":"object","properties":{"count":{"minimum":3}}});
    let calls = Arc::new(AtomicUsize::new(0));
    let tally = calls.clone();
    let funding = crate::runtime::chat::preparation_memory::test_funding(move |_| {
        tally.fetch_add(1, Ordering::Relaxed);
        Ok::<_, eredu_core::HostMetadataFundingError>(())
    })
    .unwrap();
    let expected = render(&funding, &schema).unwrap();
    assert!(expected.contains("start: <[19]>\nargs: %json"));
    let reached = calls.load(Ordering::Relaxed);
    assert!(reached > 4);
    for fail in 1..reached {
        let calls = Arc::new(AtomicUsize::new(0));
        let tally = calls.clone();
        let funding = crate::runtime::chat::preparation_memory::test_funding(move |_| {
            if tally.fetch_add(1, Ordering::Relaxed) == fail {
                Err(eredu_core::HostMetadataFundingError::Capacity { required: 1, available: 0 })
            } else {
                Ok(())
            }
        })
        .unwrap();
        let error = render(&funding, &schema).unwrap_err();
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        while let Some(next) = cause.source() {
            cause = next;
        }
        assert!(cause.is::<eredu_core::HostMetadataFundingError>(), "failure {fail}: {error:?}");
        assert!(calls.load(Ordering::Relaxed) <= fail + 1);
    }
}

#[test]
fn changing_display_cannot_grow_a_destination_after_measurement() {
    struct Changes(std::cell::Cell<bool>);
    impl fmt::Display for Changes {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(if self.0.replace(true) { "longer" } else { "x" })
        }
    }
    let funding = PreparationFunding::unmanaged();
    let mut output = Text::new(&funding).unwrap();
    assert!(matches!(
        output.push_fmt(format_args!("{}", Changes(std::cell::Cell::new(false)))),
        Err(Error::Destination)
    ));
}

#[test]
fn funded_protocol_producers_preserve_source_and_refuse_without_retry() {
    use crate::runtime::chat::{
        ParallelToolCallPolicy, ToolChoice,
        atem::{ATEM_DIALECT, parameters as atem_parameters},
        dialect::{DialectParameters, FormatDialect},
        harmony::{GPT_OSS_HARMONY_PARAMETERS, HARMONY_DIALECT},
        inkling::{INKLING_TOOL_DIALECT, parameters},
        tool_schema::ToolDefinition,
    };
    let schema = serde_json::json!({
        "$defs": {"count": {"type":"integer", "minimum":3}},
        "type":"object", "properties":{"count":{"$ref":"#/$defs/count"}},
        "required":["count"], "additionalProperties":false
    });
    let tools = [ToolDefinition {
        name: "measure",
        parameters: &schema,
    }];
    let dialects: [(&dyn FormatDialect, DialectParameters); 3] = [
        (
            &HARMONY_DIALECT,
            DialectParameters::Custom(&GPT_OSS_HARMONY_PARAMETERS),
        ),
        (&INKLING_TOOL_DIALECT, parameters()),
        (&ATEM_DIALECT, atem_parameters()),
    ];
    for (dialect, parameters) in dialects {
        let ids = (0..dialect
            .required_structural_tokens(parameters)
            .unwrap()
            .len())
            .map(|index| 100 + u32::try_from(index).unwrap())
            .collect::<Vec<_>>();
        for choice in [ToolChoice::None, ToolChoice::Auto, ToolChoice::Required] {
            let build = |funding: &PreparationFunding| {
                dialect.constraint_configuration(
                    parameters,
                    &tools,
                    choice,
                    ParallelToolCallPolicy::Disabled,
                    &ids,
                    funding,
                )
            };
            let ordinary = build(&PreparationFunding::unmanaged()).unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let tally = calls.clone();
            let funding = crate::runtime::chat::preparation_memory::test_funding(move |_| {
                tally.fetch_add(1, Ordering::Relaxed);
                Ok::<_, eredu_core::HostMetadataFundingError>(())
            })
            .unwrap();
            let funded = build(&funding).unwrap();
            assert_eq!(
                serde_json::to_value(&funded.grammar).unwrap(),
                serde_json::to_value(&ordinary.grammar).unwrap()
            );
            let count = calls.load(Ordering::Relaxed);
            for fail in 1..count {
                let calls = Arc::new(AtomicUsize::new(0));
                let tally = calls.clone();
                let funding = crate::runtime::chat::preparation_memory::test_funding(move |_| {
                    if tally.fetch_add(1, Ordering::Relaxed) == fail {
                        Err(eredu_core::HostMetadataFundingError::Capacity { required: 1, available: 0 })
                    } else {
                        Ok(())
                    }
                })
                .unwrap();
                let error = build(&funding).unwrap_err();
                let mut cause: &(dyn std::error::Error + 'static) = &error;
                while let Some(next) = cause.source() {
                    cause = next;
                }
                assert!(
                    cause.is::<eredu_core::HostMetadataFundingError>(),
                    "{choice:?} failed producer {fail}: {error:?}"
                );
                assert_eq!(calls.load(Ordering::Relaxed), fail + 1);
            }
        }
    }
}
