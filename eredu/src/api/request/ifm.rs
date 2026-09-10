//! Recognition from rendered IFM protocol behavior and tokenizer properties.

use eredu_text::tokenizer::{ModelChatTemplate, Tokenizer};
use serde_json::{json, Map, Value};

use super::{ChatTemplateRequest, TextModelError};
use crate::runtime::chat::{
    dialect::{DialectParameters, DECLARATIVE_DIALECT},
    ifm, PreparedFormatProfile, ReasoningEffortControl,
};

pub(super) fn recognize(
    tokenizer: &mut Tokenizer,
    template: &ModelChatTemplate,
    model_id: &str,
    request: &ChatTemplateRequest,
) -> Result<Option<PreparedFormatProfile>, TextModelError> {
    if super::resolve_structural_tokens(
        tokenizer,
        &["<|ifm|im_start|>".into(), "<|ifm|im_end|>".into()],
    )
    .is_err()
    {
        return Ok(None);
    }
    let user = json!({"role": "user", "content": "__eredu_ifm_probe__"});
    // Establish the protocol independently of model names or source text.
    for &(effort, prefix, _) in &ifm::REASONING {
        let kwargs = Map::from_iter([("reasoning_effort".into(), json!(effort))]);
        let rendered = tokenizer.apply_chat_template_json(
            template.clone(),
            [vec![user.clone()]],
            Some(&[]),
            model_id,
            true,
            Some(&kwargs),
        );
        if !rendered
            .ok()
            .and_then(|v| v.into_iter().next())
            .is_some_and(|text| {
                text.contains("<|ifm|im_start|>user\n__eredu_ifm_probe__<|ifm|im_end|>")
                    && text.ends_with(&format!("<|ifm|im_start|>assistant\n{prefix}"))
            })
        {
            return Ok(None);
        }
    }
    let mut kwargs = request.extra_template_kwargs.clone();
    if let Some(effort) = &request.reasoning_effort {
        kwargs.insert("reasoning_effort".into(), json!(effort));
    }
    if let Some(enabled) = request.enable_thinking {
        kwargs.insert("enable_thinking".into(), json!(enabled));
    }
    let effort = kwargs
        .get("reasoning_effort")
        .map_or(Some("high"), Value::as_str)
        .filter(|effort| ["high", "medium", "low"].contains(effort))
        .ok_or_else(|| {
            TextModelError::ToolConstraint(
                "IFM reasoning_effort must be high, medium, or low".into(),
            )
        })?;
    let format = kwargs
        .get("tool_call_format")
        .map_or(Some("xml"), Value::as_str)
        .filter(|format| ["json", "xml", "xml_typed"].contains(format))
        .ok_or_else(|| {
            TextModelError::ToolConstraint(
                "IFM tool_call_format must be json, xml, or xml_typed".into(),
            )
        })?;
    let disabled = kwargs.get("enable_thinking") == Some(&Value::Bool(false));
    if disabled {
        let rendered = tokenizer.apply_chat_template_json(
            template.clone(),
            [vec![user.clone()]],
            Some(&[]),
            model_id,
            true,
            Some(&kwargs),
        )?;
        if !rendered[0].ends_with("<ifm|think>\n</ifm|think>\n") {
            return Err(TextModelError::ToolConstraint(
                "The selected IFM template does not expose a reasoning-disable control".into(),
            ));
        }
    }
    let tools = vec![json!({"type": "function", "function": {
        "name": "eredu_ifm_probe", "description": "Probe", "parameters": {
            "type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]
        }
    }})];
    let messages = vec![
        user,
        json!({"role": "assistant", "think": "__eredu_think_probe__", "content": "", "tool_calls": [{
            "type": "function", "id": "probe", "function": {"name": "eredu_ifm_probe", "arguments": {"value": "__eredu_value_probe__"}}
        }]}),
        json!({"role": "tool", "tool_call_id": "probe", "content": "__eredu_result_probe__"}),
    ];
    let rendered = tokenizer.apply_chat_template_json(
        template.clone(),
        [messages],
        Some(&tools),
        model_id,
        true,
        Some(&kwargs),
    )?;
    let text = &rendered[0];
    let expected_call = match format {
        "json" => "<ifm|tool_call>{\"name\": \"eredu_ifm_probe\", \"arguments\": {\"value\": \"__eredu_value_probe__\"}}</ifm|tool_call>",
        "xml" => "<ifm|tool_call>eredu_ifm_probe\n<ifm|arg_key>value</ifm|arg_key>\n<ifm|arg_value>__eredu_value_probe__</ifm|arg_value>\n</ifm|tool_call>",
        "xml_typed" => "<ifm|tool_call>eredu_ifm_probe\n<ifm|arg_key>value</ifm|arg_key>\n<ifm|arg_type>string</ifm|arg_type>\n<ifm|arg_value>__eredu_value_probe__</ifm|arg_value>\n</ifm|tool_call>",
        _ => unreachable!(),
    };
    if !text.contains(expected_call)
        || !text.contains("<|ifm|im_start|>tool\n__eredu_result_probe__<|ifm|im_end|>")
        || !text.contains("<ifm|think>\n__eredu_think_probe__")
        || !text.contains("</ifm|think>")
        || !text.contains("<ifm|tool_calls>\n")
        || !text.contains("\n</ifm|tool_calls>")
    {
        return Ok(None);
    }
    let prefilled = request.add_generation_prompt;
    let spec = ifm::spec(format, effort, prefilled, disabled).expect("validated IFM controls");
    let mut profile = match super::recognized_dialect_profile(
        tokenizer,
        "ifm.tools.v1",
        &DECLARATIVE_DIALECT,
        DialectParameters::Declarative(spec),
        true,
        false,
    ) {
        Some(profile) => profile,
        None => return Ok(None),
    };
    profile.reasoning_effort_control = Some(ReasoningEffortControl {
        kwarg: "reasoning_effort",
        supported: &["high", "medium", "low"],
    });
    if format != "json" {
        super::validate_tagged_history(&request.messages, "</ifm|arg_value>", "IFM")?;
    }
    Ok(Some(profile))
}
