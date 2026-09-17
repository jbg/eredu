//! IFM behavior and request precedence over actual shared source operations.
use super::{Operations, Probe};
use crate::api::request::ChatTemplateRequest;
use eredu_text::chat_storage::{ChatScalarBinding, ChatScalarBindingValue};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Effort {
    High,
    Medium,
    Low,
}
impl Effort {
    pub(crate) fn spelling(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
        }
    }
    fn expected(self) -> &'static str {
        match self {
            Self::High => "<|ifm|im_start|>assistant\n<ifm|think>\n",
            Self::Medium => "<|ifm|im_start|>assistant\n<ifm|think_fast>\n",
            Self::Low => "<|ifm|im_start|>assistant\n<ifm|think_faster>\n",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Format {
    Json,
    Xml,
    XmlTyped,
}
impl Format {
    pub(crate) fn spelling(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Xml => "xml",
            Self::XmlTyped => "xml_typed",
        }
    }
    fn expected(self) -> &'static str {
        match self {
            Self::Json => {
                "<ifm|tool_call>{\"name\": \"eredu_ifm_probe\", \"arguments\": {\"value\": \"__eredu_value_probe__\"}}</ifm|tool_call>"
            }
            Self::Xml => {
                "<ifm|tool_call>eredu_ifm_probe\n<ifm|arg_key>value</ifm|arg_key>\n<ifm|arg_value>__eredu_value_probe__</ifm|arg_value>\n</ifm|tool_call>"
            }
            Self::XmlTyped => {
                "<ifm|tool_call>eredu_ifm_probe\n<ifm|arg_key>value</ifm|arg_key>\n<ifm|arg_type>string</ifm|arg_type>\n<ifm|arg_value>__eredu_value_probe__</ifm|arg_value>\n</ifm|tool_call>"
            }
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum ControlError {
    #[error("IFM reasoning_effort must be high, medium, or low")]
    Effort,
    #[error("IFM tool_call_format must be json, xml, or xml_typed")]
    Format,
    #[error("The selected IFM template does not expose a reasoning-disable control")]
    Disable,
}
/// Fixed facts from behavior, not a complete dialect or token-source witness.
/// Dialect token validation and tagged history validation still follow this step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Recognition {
    pub(crate) effort: Effort,
    pub(crate) format: Format,
    pub(crate) disabled: bool,
}
impl Recognition {
    fn controls(request: &ChatTemplateRequest) -> Result<Self, ControlError> {
        let effort = match request.reasoning_effort.as_deref() {
            Some(effort) => Some(effort),
            None => request
                .extra_template_kwargs
                .get("reasoning_effort")
                .map_or(Some("high"), Value::as_str),
        };
        let effort = match effort {
            Some("high") => Effort::High,
            Some("medium") => Effort::Medium,
            Some("low") => Effort::Low,
            _ => return Err(ControlError::Effort),
        };
        let format = match request
            .extra_template_kwargs
            .get("tool_call_format")
            .map_or(Some("xml"), Value::as_str)
        {
            Some("json") => Format::Json,
            Some("xml") => Format::Xml,
            Some("xml_typed") => Format::XmlTyped,
            _ => return Err(ControlError::Format),
        };
        let disabled = request.enable_thinking.map_or_else(
            || request.extra_template_kwargs.get("enable_thinking") == Some(&Value::Bool(false)),
            |enabled| !enabled,
        );
        Ok(Self {
            effort,
            format,
            disabled,
        })
    }
}
/// Exact explicit replacements, in the ordinary request's order. Other kwargs
/// remain borrowed from the request map; defaults remain a renderer source.
pub(crate) struct Overrides<'a> {
    entries: [ChatScalarBinding<'a>; 2],
    len: usize,
}
impl<'a> Overrides<'a> {
    pub(crate) fn new(request: &'a ChatTemplateRequest) -> Self {
        let mut out = Self {
            entries: [ChatScalarBinding {
                name: "",
                value: ChatScalarBindingValue::Bool(false),
            }; 2],
            len: 0,
        };
        if let Some(effort) = request.reasoning_effort.as_deref() {
            out.entries[out.len] = ChatScalarBinding {
                name: "reasoning_effort",
                value: ChatScalarBindingValue::Text(effort),
            };
            out.len += 1;
        }
        if let Some(enabled) = request.enable_thinking {
            out.entries[out.len] = ChatScalarBinding {
                name: "enable_thinking",
                value: ChatScalarBindingValue::Bool(enabled),
            };
            out.len += 1;
        }
        out
    }
    pub(crate) fn as_slice(&self) -> &[ChatScalarBinding<'a>] {
        &self.entries[..self.len]
    }
}
/// Request probes propagate actual render errors. Initial protocol probes use
/// the unchanged negative-recognition behavior of the ordinary adapter.
pub(crate) trait RequestOperations: Operations {
    fn render_request_matches(
        &mut self,
        probe: Probe,
        request: &ChatTemplateRequest,
        contains: &[&str],
        suffix: Option<&str>,
    ) -> Result<bool, Self::Error>;
    fn control_failure(&self, cause: ControlError) -> Self::Error;
}
pub(crate) fn recognize<O: RequestOperations>(
    ops: &mut O,
    request: &ChatTemplateRequest,
) -> Result<Option<Recognition>, O::Error> {
    if !ops.structural(&["<|ifm|im_start|>", "<|ifm|im_end|>"])? {
        return Ok(None);
    }
    for effort in [Effort::High, Effort::Medium, Effort::Low] {
        if !ops.render_matches(
            Probe::IfmEffort(effort),
            &["<|ifm|im_start|>user\n__eredu_ifm_probe__<|ifm|im_end|>"],
            Some(effort.expected()),
        )? {
            return Ok(None);
        }
    }
    let controls = Recognition::controls(request).map_err(|cause| ops.control_failure(cause))?;
    if controls.disabled
        && !ops.render_request_matches(
            Probe::IfmUser,
            request,
            &[],
            Some("<ifm|think>\n</ifm|think>\n"),
        )?
    {
        return Err(ops.control_failure(ControlError::Disable));
    }
    if !ops.render_request_matches(
        Probe::IfmTools,
        request,
        &[
            controls.format.expected(),
            "<|ifm|im_start|>tool\n__eredu_result_probe__<|ifm|im_end|>",
            "<ifm|think>\n__eredu_think_probe__",
            "</ifm|think>",
            "<ifm|tool_calls>\n",
            "\n</ifm|tool_calls>",
        ],
        None,
    )? {
        return Ok(None);
    }
    Ok(Some(controls))
}
pub(crate) fn control_bytes<E>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<Recognition>(),
        size_of::<Result<Recognition, ControlError>>(),
        size_of::<Option<Recognition>>(),
        size_of::<Result<Option<Recognition>, E>>(),
        size_of::<Overrides<'_>>(),
        size_of::<ControlError>(),
        size_of::<E>(),
        size_of::<[Effort; 3]>(),
        size_of::<std::array::IntoIter<Effort, 3>>(),
        size_of::<[&str; 6]>(),
        size_of::<[&str; 2]>(),
        size_of::<(&ChatTemplateRequest, Option<&str>, Option<&Value>, bool)>(),
        size_of::<(&str, Option<&Value>, Option<bool>)>(),
        size_of::<Result<bool, E>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
