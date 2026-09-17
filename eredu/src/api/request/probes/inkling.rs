//! Existing Inkling behavior recognition, independent of render/encoding storage.
use super::{Operations, Probe};
use crate::runtime::chat::inkling::{
    CONTENT_INVOKE_TOOL_JSON, CONTENT_TEXT, CONTENT_THINKING, END_MESSAGE, END_SAMPLING,
    MESSAGE_MODEL,
};

pub(crate) const STRUCTURAL: [&str; 5] = [
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    END_MESSAGE,
    END_SAMPLING,
];
pub(crate) const TOOL_STRUCTURAL: [&str; 6] = [
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_INVOKE_TOOL_JSON,
    END_MESSAGE,
    END_SAMPLING,
];
#[derive(Clone, Copy, Debug)]
pub(crate) struct Recognition {
    pub(crate) mapping_tool_arguments: bool,
    pub(crate) string_tool_arguments: bool,
}

pub(crate) fn recognize<O: Operations>(ops: &mut O) -> Result<Option<Recognition>, O::Error> {
    if !ops.structural(&STRUCTURAL)? {
        return Ok(None);
    }
    if !ops.render_matches(Probe::InklingHistory, &[concat!(
        "<|message_model|><|content_thinking|>__eredu_inkling_reasoning_probe_7c91__<|end_message|>",
        "<|message_model|><|content_text|>__eredu_inkling_visible_probe_28ad__<|end_message|><|content_model_end_sampling|>"
    )], None)? { return Ok(None); }
    for (enabled, expected) in [
        (
            false,
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>",
        ),
        (
            true,
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
        ),
    ] {
        if !ops.render_matches(
            Probe::InklingEffort { enabled },
            &[expected],
            Some(MESSAGE_MODEL),
        )? {
            return Ok(None);
        }
    }
    let tool_tokens_valid = ops.structural(&TOOL_STRUCTURAL)?;
    let mapping_tool_arguments = tool_tokens_valid
        && ops.render_matches(
            Probe::Tool { mapping: true },
            &[
                concat!(
                    "<|message_system|>tool_declare<|content_xml|>",
                    "[{\"description\":\"protocol recognition probe\",\"name\":",
                    "\"eredu_probe_7c91\",\"parameters\":"
                ),
                concat!(
                    "<|message_model|>eredu_probe_7c91<|content_invoke_tool_json|>",
                    "{\"name\":\"eredu_probe_7c91\",\"args\":{\"value\":\"probe-value\"}}",
                    "<|end_message|>"
                ),
                concat!(
                    "<|message_tool|>eredu_probe_7c91<|content_text|>",
                    "\"__eredu_tool_result_probe__\"<|end_message|>"
                ),
            ],
            None,
        )?;
    let string_tool_arguments = tool_tokens_valid
        && ops.render_matches(
            Probe::Tool { mapping: false },
            &[CONTENT_INVOKE_TOOL_JSON, "__eredu_tool_result_probe__"],
            None,
        )?;
    Ok(Some(Recognition {
        mapping_tool_arguments,
        string_tool_arguments,
    }))
}
