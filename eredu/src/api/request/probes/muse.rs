//! Actual ATEM history/strength/tool decisions over the same probe operations.
use super::{Operations, Probe};
use crate::runtime::chat::atem::{EOM, EOT, MESSAGE, START};
pub(crate) const STRUCTURAL: [&str; 4] = [START, MESSAGE, EOM, EOT];
#[derive(Clone, Copy, Debug)]
pub(crate) enum Strength {
    Low,
    Medium,
    High,
    ExtraHigh,
}
impl Strength {
    pub(crate) fn spelling(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::ExtraHigh => "xhigh",
        }
    }
    fn expected(self) -> &'static str {
        match self {
            Self::Low => "Reasoning strength: low.",
            Self::Medium => "Reasoning strength: medium.",
            Self::High => "Reasoning strength: high.",
            Self::ExtraHigh => "Reasoning strength: xhigh.",
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Recognition {
    pub(crate) mapping_tool_arguments: bool,
}
pub(crate) fn recognize<O: Operations>(ops: &mut O) -> Result<Option<Recognition>, O::Error> {
    if !ops.structural(&STRUCTURAL)? {
        return Ok(None);
    }
    if !ops.render_matches(
        Probe::ReasoningHistory,
        &[concat!(
            "<|start|>assistant to=self<|message|>__eredu_reasoning_probe__<|eom|>",
            "<|start|>assistant to=user<|message|>__eredu_visible_probe__<|eot|>"
        )],
        None,
    )? {
        return Ok(None);
    }
    for strength in [
        Strength::Low,
        Strength::Medium,
        Strength::High,
        Strength::ExtraHigh,
    ] {
        if !ops.render_matches(
            Probe::MuseStrength(strength),
            &[strength.expected()],
            Some("<|start|>assistant"),
        )? {
            return Ok(None);
        }
    }
    let mapping_tool_arguments = ops.render_matches(
        Probe::Tool { mapping: true },
        &[
            "<|start|>assistant to=self<|message|>__eredu_reasoning_probe__<|eom|>",
            "<|start|>assistant to=eredu_probe_7c91<|message|>",
            "<atem:function_calls>",
            "<atem:invoke name=\"eredu_probe_7c91\">",
            "<atem:parameter name=\"value\">probe-value</atem:parameter>",
            "<|start|>tool eredu_probe_7c91<|message|><tool_output name=\"eredu_probe_7c91\">",
            "__eredu_tool_result_probe__",
        ],
        None,
    )?;
    Ok(Some(Recognition {
        mapping_tool_arguments,
    }))
}
pub(crate) fn control_bytes<E>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<Recognition>(),
        size_of::<Option<Recognition>>(),
        size_of::<Result<Option<Recognition>, E>>(),
        size_of::<[Strength; 4]>(),
        size_of::<std::array::IntoIter<Strength, 4>>(),
        size_of::<[&str; 7]>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
