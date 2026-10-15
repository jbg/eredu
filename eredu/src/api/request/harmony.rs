//! Harmony channel admission independently of function-call rendering.

use super::*;
use crate::runtime::chat::{
    dialect::DialectParameters,
    harmony::{GPT_OSS_HARMONY_PARAMETERS, HARMONY_DIALECT},
    PreparedFormatProfile,
};

pub(super) fn recognize(
    tokenizer: &mut ChatTokenizer,
    template: &ModelChatTemplate,
    model_id: &str,
) -> Option<PreparedFormatProfile> {
    let mut tokens = [
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "<|return|>",
    ]
    .map(str::to_owned)
    .to_vec();
    resolve_structural_tokens(tokenizer, &tokens).ok()?;
    let history = render_reasoning_protocol_probe(tokenizer, template, model_id)?;
    if !history.contains(concat!(
        "<|start|>assistant<|channel|>analysis<|message|>__eredu_reasoning_probe__<|end|>",
        "<|start|>assistant<|channel|>final<|message|>__eredu_visible_probe__<|return|>"
    )) {
        return None;
    }
    let prompt = tokenizer
        .apply_chat_template_json(
            template.clone(),
            [vec![
                serde_json::json!({"role":"user", "content":"__eredu_user_probe__"}),
            ]],
            Some(&[]),
            model_id,
            true,
            None,
        )
        .ok()?
        .into_iter()
        .next()?;
    if !prompt.ends_with("<|start|>assistant") {
        return None;
    }

    // These tokens may occur in headers, but neither their presence nor tool
    // rendering is necessary to establish analysis/final channel semantics.
    for token in ["<|constrain|>", "<|call|>"] {
        if resolve_structural_tokens(tokenizer, &[token.into()]).is_ok() {
            tokens.push(token.into());
        }
    }
    let mut profile = prepare_format_profile("");
    profile.identity = Some("harmony.channels.v1".into());
    profile.dialect = Some(&HARMONY_DIALECT);
    profile.dialect_parameters = Some(DialectParameters::Custom(&GPT_OSS_HARMONY_PARAMETERS));

    profile.required_structural_tokens = tokens;
    profile.stop_sequences = vec!["<|return|>".into()];
    if ["low", "medium", "high"].iter().all(|effort| {
        let kwargs =
            serde_json::Map::from_iter([("reasoning_effort".into(), serde_json::json!(effort))]);
        tokenizer
            .apply_chat_template_json(
                template.clone(),
                [vec![
                    serde_json::json!({"role":"user", "content":"__eredu_effort_probe__"}),
                ]],
                Some(&[]),
                model_id,
                true,
                Some(&kwargs),
            )
            .ok()
            .and_then(|rendered| rendered.into_iter().next())
            .is_some_and(|text| text.contains(&format!("Reasoning: {effort}\n")))
    }) {
        profile.reasoning_effort_control = Some(crate::runtime::chat::ReasoningEffortControl {
            kwarg: "reasoning_effort",
            supported: &["low", "medium", "high"],
        });
    }
    profile.native_tool_unavailable_reason = Some("Harmony channels were recognized, but function-call rendering or tokens were not established".into());

    let supports = |rendered: Option<String>| {
        rendered.is_some_and(|rendered| {
            [
                "assistant to=functions.eredu_probe_7c91",
                "<|message|>",
                "<|call|>",
                "functions.eredu_probe_7c91 to=assistant",
                "__eredu_tool_result_probe__",
            ]
            .iter()
            .all(|marker| rendered.contains(marker))
        })
    };
    let mapping = supports(render_protocol_probe(
        tokenizer,
        template,
        model_id,
        serde_json::json!({"value":"__eredu_mapping_argument_probe__"}),
    ));
    let string = supports(render_protocol_probe(
        tokenizer,
        template,
        model_id,
        serde_json::json!(r#"{"value":"__eredu_string_argument_probe__"}"#),
    ));
    if mapping || string {
        if let Some(tools) = recognized_tool_profile(
            tokenizer,
            "harmony.channels.v1",
            &HARMONY_DIALECT,
            DialectParameters::Custom(&GPT_OSS_HARMONY_PARAMETERS),
            mapping,
            string,
        ) {
            profile.tool_dialect = tools.tool_dialect;
            profile.tool_dialect_parameters = tools.tool_dialect_parameters;
            profile.tool_required_structural_tokens = tools.tool_required_structural_tokens;
            profile.supports_mapping_tool_arguments = mapping;
            profile.supports_string_tool_arguments = string;
            profile.supports_tool_input_rendering = true;
            profile.native_tool_unavailable_reason = None;
            profile.stop_sequences = tools.stop_sequences;
        }
    }
    Some(profile)
}
