//! Behavioral admission for reasoning independently of tool-history support.

use super::*;
use crate::runtime::chat::{
    dialect::DialectParameters,
    reasoning::{REASONING_DIALECT, THINK_FORMATS},
    PreparedFormatProfile,
};
use serde_json::json;

pub(super) fn recognize(
    tokenizer: &mut ChatTokenizer,
    template: &ModelChatTemplate,
    model_id: &str,
    request: &ChatTemplateRequest,
) -> Option<PreparedFormatProfile> {
    let mut kwargs = request.extra_template_kwargs.clone();
    if let Some(enabled) = request.enable_thinking {
        kwargs.insert("enable_thinking".into(), json!(enabled));
    }
    let mut render = |messages, generation| {
        tokenizer
            .apply_chat_template_json(
                template.clone(),
                [messages],
                Some(&[]),
                model_id,
                generation,
                Some(&kwargs),
            )
            .ok()?
            .into_iter()
            .next()
    };
    // Only the suffix contributed by the generation toggle can establish a
    // prefilled channel. Markers in user messages or older turns are irrelevant.
    let messages = vec![json!({"role":"user", "content":"__eredu_user_probe__"})];
    let without = render(messages.clone(), false)?;
    let with = render(messages, true)?;
    let suffix = with.strip_prefix(&without)?;
    let prefilled = suffix.trim_end().ends_with("<think>");
    let history = render(
        vec![
            json!({"role":"user", "content":"__eredu_user_probe__"}),
            json!({"role":"assistant", "reasoning_content":"__eredu_reasoning_probe__", "content":"__eredu_visible_probe__"}),
        ],
        false,
    );
    let framed_history = history.as_deref().is_some_and(|rendered| {
        rendered
            .split_once("<think>")
            .and_then(|(_, rest)| rest.split_once("</think>"))
            .is_some_and(|(reasoning, rest)| {
                reasoning.trim() == "__eredu_reasoning_probe__"
                    && rest.trim_start().starts_with("__eredu_visible_probe__")
            })
    });
    if !prefilled && !framed_history {
        return None;
    }
    let format = &THINK_FORMATS[0];
    let mut profile = prepare_format_profile("");
    profile.identity = Some("reasoning.think.v1".into());
    profile.dialect = Some(&REASONING_DIALECT);
    profile.dialect_parameters = Some(DialectParameters::Custom(format));
    profile.supports_reasoning_parsing = true;
    // Ordinary multi-token delimiters work too. Atomic added tokens must survive
    // special-token decoding, but their presence never establishes support alone.
    profile.required_structural_tokens = [format.opening, format.closing]
        .into_iter()
        .filter(|marker| resolve_structural_tokens(tokenizer, &[(*marker).into()]).is_ok())
        .map(str::to_owned)
        .collect();
    profile.native_tool_unavailable_reason =
        Some("reasoning framing does not establish native tool support".into());
    Some(profile)
}

/// Bind the parser to this conversation's actual suffix, never the probe history.
pub(super) fn bind_generation_prompt(
    profile: &mut PreparedFormatProfile,
    suffix: &str,
    added: bool,
    thinking: Option<bool>,
) -> Result<(), TextModelError> {
    if !profile.dialect_parameters.is_some_and(|parameters| {
        parameters
            .custom::<crate::runtime::chat::reasoning::ReasoningFormat>()
            .is_ok()
    }) {
        return Ok(());
    }
    let prefilled = added && suffix.trim_end().ends_with("<think>");
    let closed = added && suffix.trim_end().ends_with("</think>");
    if thinking == Some(false) && prefilled {
        return Err(TextModelError::UnsupportedReasoning {
            reason: "the selected template always opens reasoning and does not honor enable_thinking=false".into(),
        });
    }
    if thinking == Some(true) && closed {
        return Err(TextModelError::UnsupportedReasoning {
            reason: "the selected template closes reasoning despite enable_thinking=true".into(),
        });
    }
    let index = if prefilled {
        1
    } else if closed {
        2
    } else {
        0
    };
    profile.dialect_parameters = Some(DialectParameters::Custom(&THINK_FORMATS[index]));
    if can_bind_tools(profile) {
        use crate::runtime::chat::{QWEN_XML_TOOL_SPEC, XML_THINK_GENERATED, XML_THINK_PREFILLED};
        let spec = match index {
            1 => &XML_THINK_PREFILLED,
            2 => &QWEN_XML_TOOL_SPEC,
            _ => &XML_THINK_GENERATED,
        };
        profile.tool_dialect_parameters = Some(DialectParameters::Declarative(spec));
        profile.identity = Some("xml-tools.think.v1".into());
    }
    Ok(())
}

/// Existing XML envelopes can compose the admitted reasoning channel directly.
pub(super) fn can_bind_tools(profile: &PreparedFormatProfile) -> bool {
    matches!(profile.tool_dialect_parameters, Some(DialectParameters::Declarative(spec))
        if spec == &crate::runtime::chat::QWEN_XML_TOOL_SPEC)
}
