//! Shared Gemma channel and tool recognition from actual rendered behavior.
use super::{Operations, Probe};
use crate::runtime::chat::gemma::{
    CHANNEL_CLOSE, CHANNEL_OPEN, STRING_DELIMITER, TOOL_CALL_CLOSE, TOOL_CALL_OPEN,
    TOOL_RESPONSE_OPEN, TURN_CLOSE,
};
pub(crate) const CHANNELS: [&str; 2] = [CHANNEL_OPEN, CHANNEL_CLOSE];
pub(crate) const TOOLS: [&str; 7] = [
    CHANNEL_OPEN,
    CHANNEL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    STRING_DELIMITER,
    TOOL_RESPONSE_OPEN,
    TURN_CLOSE,
];
#[derive(Clone, Copy, Debug)]
pub(crate) struct Recognition {
    pub(crate) response_token: bool,
    pub(crate) turn_token: bool,
    pub(crate) mapping_tool_arguments: bool,
    pub(crate) string_tool_arguments: bool,
}
pub(crate) fn recognize<O: Operations>(ops: &mut O) -> Result<Option<Recognition>, O::Error> {
    if !ops.structural(&CHANNELS)? {
        return Ok(None);
    }
    if !ops.render_matches(
        Probe::GemmaHistory,
        &[
            "<|channel>thought\n__eredu_reasoning_probe_7c91__\n<channel|>",
            "__eredu_visible_probe_28ad__",
        ],
        None,
    )? {
        return Ok(None);
    }
    for thinking in [true, false] {
        if !ops.render_matches(Probe::GemmaPrompt(thinking), &["<|turn>model\n"], None)? {
            return Ok(None);
        }
    }
    let response_token = ops.structural(&[TOOL_RESPONSE_OPEN])?;
    let turn_token = ops.structural(&[TURN_CLOSE])?;
    let tools = ops.structural(&TOOLS)?;
    let mapping_tool_arguments = tools
        && ops.render_matches(
            Probe::GemmaTool { mapping: true },
            &[
                "<|tool_call>call:eredu_probe{",
                TOOL_CALL_CLOSE,
                TOOL_RESPONSE_OPEN,
                "probe-response",
            ],
            None,
        )?;
    let string_tool_arguments = tools
        && ops.render_matches(
            Probe::GemmaTool { mapping: false },
            &[
                "<|tool_call>call:eredu_probe{",
                TOOL_CALL_CLOSE,
                TOOL_RESPONSE_OPEN,
                "probe-response",
            ],
            None,
        )?;
    Ok(Some(Recognition {
        response_token,
        turn_token,
        mapping_tool_arguments,
        string_tool_arguments,
    }))
}
pub(crate) fn control_bytes<E>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<Recognition>(),
        size_of::<Option<Recognition>>(),
        size_of::<Result<Option<Recognition>, E>>(),
        size_of::<[bool; 2]>(),
        size_of::<std::array::IntoIter<bool, 2>>(),
        size_of::<[&str; 7]>(),
        size_of::<[&str; 4]>(),
        size_of::<[&str; 2]>(),
        size_of::<[&str; 1]>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
