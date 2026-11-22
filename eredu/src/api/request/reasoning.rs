//! Behavioral admission for reasoning independently of tool-history support.

use super::*;
use crate::runtime::chat::{
    dialect::{DelimitedChannel, DialectParameters, ReasoningFraming},
    reasoning::{DELIMITED_FORMATS, REASONING_DIALECT},
    PreparedFormatProfile,
};
use serde_json::json;

pub(super) fn recognize(
    tokenizer: &mut ChatTokenizer,
    template: &ModelChatTemplate,
    model_id: &str,
    request: &ChatTemplateRequest,
    tools: &[serde_json::Value],
) -> Option<(PreparedFormatProfile, DelimitedChannel)> {
    use crate::runtime::chat::{ReasoningEffortControl, ReasoningTemplateControl};
    let user = vec![json!({"role":"user", "content":"__eredu_user_probe__"})];
    let history = vec![
        user[0].clone(),
        json!({"role":"assistant",
        "reasoning_content":"__eredu_reasoning_probe__", "thinking":"__eredu_reasoning_probe__",
        "think":"__eredu_reasoning_probe__", "content":"__eredu_visible_probe__"}),
    ];
    let render = |tokenizer: &mut ChatTokenizer,
                  messages,
                  generation,
                  kwargs: &serde_json::Map<String, serde_json::Value>| {
        tokenizer
            .apply_chat_template_json(
                template.clone(),
                [messages],
                Some(tools),
                model_id,
                generation,
                Some(kwargs),
            )
            .ok()?
            .into_iter()
            .next()
    };
    for (identity, formats) in &DELIMITED_FORMATS {
        let format = &formats[0];
        for control in [Some("enable_thinking"), Some("thinking"), None] {
            let mut kwargs = request.extra_template_kwargs.clone();
            if let Some(control) = control {
                kwargs.insert(control.into(), json!(true));
            }
            let Some(without) = render(tokenizer, user.clone(), false, &kwargs) else {
                continue;
            };
            let Some(with) = render(tokenizer, user.clone(), true, &kwargs) else {
                continue;
            };
            let Some(suffix) = with.strip_prefix(&without) else {
                continue;
            };
            let history = render(tokenizer, history.clone(), false, &kwargs);
            let framed = history
                .as_deref()
                .and_then(|text| text.split_once(format.opening))
                .and_then(|(_, rest)| rest.split_once(format.closing))
                .filter(|(reasoning, rest)| {
                    reasoning.trim() == "__eredu_reasoning_probe__"
                        && rest.trim_start().starts_with("__eredu_visible_probe__")
                })
                .map(|(reasoning, _)| reasoning);
            if !suffix.trim_end().ends_with(format.opening) && framed.is_none() {
                continue;
            }
            let channel = DelimitedChannel {
                prefix: if framed.is_some_and(|body| body.starts_with('\n'))
                    || suffix.ends_with(format.opening_with_newline)
                {
                    format.opening_with_newline
                } else {
                    format.opening
                },
                suffix: if framed.is_some_and(|body| body.ends_with('\n')) {
                    format.closing_with_newline
                } else {
                    format.closing
                },
                required: false,
                prefix_in_prompt: false,
            };
            // A Boolean control must change rendered response framing. Accepting
            // an ignored kwarg is not evidence that the template can disable reasoning.
            if let Some(control) = control {
                kwargs.insert(control.into(), json!(false));
                let Some(off_without) = render(tokenizer, user.clone(), false, &kwargs) else {
                    continue;
                };
                let Some(off_with) = render(tokenizer, user.clone(), true, &kwargs) else {
                    continue;
                };
                let Some(off_suffix) = off_with.strip_prefix(&off_without) else {
                    continue;
                };
                if suffix == off_suffix || off_suffix.trim_end().ends_with(format.opening) {
                    continue;
                }
            }
            let mut profile = prepare_format_profile("");
            profile.identity = Some((*identity).into());
            profile.dialect = Some(&REASONING_DIALECT);
            profile.dialect_parameters = Some(DialectParameters::Custom(format));
            profile.reasoning_template_control =
                ReasoningTemplateControl::Boolean(control.unwrap_or("enable_thinking"));

            profile.required_structural_tokens = [format.opening, format.closing]
                .into_iter()
                .filter(|marker| resolve_structural_tokens(tokenizer, &[(*marker).into()]).is_ok())
                .map(str::to_owned)
                .collect();
            profile.native_tool_unavailable_reason =
                Some("reasoning framing does not establish native tool support".into());
            // Named effort is admitted from rendered control behavior, independently
            // of the template's optional tool-call envelope.
            let effort_prompts = ["low", "medium", "xhigh"].map(|effort| {
                let mut kwargs = request.extra_template_kwargs.clone();
                kwargs.insert("reasoning_effort".into(), json!(effort));
                render(tokenizer, user.clone(), true, &kwargs)
            });
            if let [Some(low), Some(medium), Some(xhigh)] = &effort_prompts {
                if low.contains("Reasoning effort is set to low.")
                    && xhigh.contains("Reasoning effort is set to xhigh.")
                    && !medium.contains("Reasoning effort is set to low.")
                    && !medium.contains("Reasoning effort is set to xhigh.")
                {
                    profile.reasoning_effort_control = Some(ReasoningEffortControl {
                        kwarg: "reasoning_effort",
                        supported: &["low", "medium", "xhigh"],
                    });
                }
            }
            return Some((profile, channel));
        }
    }
    None
}

/// Bind the parser to this conversation's actual suffix, never the probe history.
pub(super) fn bind_generation_prompt(
    profile: &mut PreparedFormatProfile,
    tokenizer: &ChatTokenizer,
    channel: Option<DelimitedChannel>,
    suffix: &str,
    added: bool,
    thinking: Option<bool>,
) -> Result<(), TextModelError> {
    let Some(format) = profile.dialect_parameters.and_then(|parameters| {
        parameters
            .custom::<crate::runtime::chat::reasoning::ReasoningFormat>()
            .ok()
    }) else {
        return Ok(());
    };
    let formats = &DELIMITED_FORMATS
        .iter()
        .find(|(_, formats)| formats[0].opening == format.opening)
        .expect("admitted delimited format")
        .1;
    let prefilled = added && suffix.trim_end().ends_with(format.opening);
    let closed = added && suffix.trim_end().ends_with(format.closing);
    if thinking == Some(false) && prefilled {
        return Err(super::ChatCapabilityRejection::ReasoningControl {
            control: "enable_thinking".into(),
            reason: "the selected template always opens reasoning and does not honor enable_thinking=false".into(),
        }.into());
    }
    if thinking == Some(true) && closed {
        return Err(super::ChatCapabilityRejection::ReasoningControl {
            control: "enable_thinking".into(),
            reason: "the selected template closes reasoning despite enable_thinking=true".into(),
        }
        .into());
    }
    let index = if prefilled {
        1
    } else if closed || thinking == Some(false) {
        2
    } else {
        0
    };
    profile.dialect_parameters = Some(DialectParameters::Custom(&formats[index]));
    if let (Some(channel), Some(DialectParameters::Declarative(spec))) =
        (channel, profile.tool_dialect_parameters)
    {
        if !can_bind_tools(profile) {
            return Ok(());
        }
        let channel = if index == 2 {
            None
        } else {
            Some(DelimitedChannel {
                prefix_in_prompt: prefilled,
                ..spec
                    .reasoning_channel
                    .filter(|declared| {
                        declared.prefix.trim_matches('\n') == format.opening
                            && declared.suffix.trim_matches('\n') == format.closing
                    })
                    .unwrap_or(channel)
            })
        };
        let mut special_tokens = [None; 2];
        if channel.is_some() {
            let added_tokens = tokenizer.get_added_tokens_decoder();
            for (slot, marker) in special_tokens
                .iter_mut()
                .zip([format.opening, format.closing])
            {
                let spelling = marker.to_owned();
                if let Ok(ids) =
                    resolve_structural_tokens(tokenizer, std::slice::from_ref(&spelling))
                {
                    if !profile.tool_required_structural_tokens.contains(&spelling) {
                        profile.tool_required_structural_tokens.push(spelling);
                    }
                    if added_tokens.get(&ids[0]).is_some_and(|token| token.special) {
                        *slot = Some((marker, ids[0]));
                    }
                }
            }
        }
        profile.tool_dialect_parameters = Some(DialectParameters::BoundDeclarative(
            spec,
            ReasoningFraming {
                channel,
                special_tokens,
            },
        ));
        if spec == &crate::runtime::chat::QWEN_XML_TOOL_SPEC && format.opening == "<think>" {
            profile.identity = Some("xml-tools.think.v1".into());
        }
    }
    Ok(())
}

/// Declarative tool envelopes can compose independently admitted tag framing.
pub(super) fn can_bind_tools(profile: &PreparedFormatProfile) -> bool {
    let Some(DialectParameters::Declarative(spec)) = profile.tool_dialect_parameters else {
        return false;
    };
    spec.supports_tool_reasoning
        && profile
            .dialect_parameters
            .and_then(|parameters| {
                parameters
                    .custom::<crate::runtime::chat::reasoning::ReasoningFormat>()
                    .ok()
            })
            .is_some()
        && spec.reasoning_channel.is_none_or(|channel| {
            DELIMITED_FORMATS.iter().any(|(_, formats)| {
                let format = &formats[0];
                channel.prefix.trim_matches('\n') == format.opening
                    && channel.suffix.trim_matches('\n') == format.closing
            })
        })
}
