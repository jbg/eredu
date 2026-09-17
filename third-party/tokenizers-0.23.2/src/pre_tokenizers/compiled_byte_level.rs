//! Closed fresh source for the ordinary implicit ByteLevel regex.
use super::{
    byte_level::ByteLevel,
    compiled_split::{self, CompiledRegexSplit},
};
use crate::{PreTokenizedString, PreTokenizer};
use fancy_regex::workspace::{Plan, PlanError};
use serde::{Serialize, Serializer};
use std::mem::size_of;

/// Actual ByteLevel settings and one closed compiled source. Ordinary JSON
/// restoration remains ordinary ByteLevel; no source constructor is exposed.
/// ```compile_fail
/// use tokenizers::pre_tokenizers::compiled_byte_level::CompiledByteLevel;
/// fn escape(value: CompiledByteLevel) { let _ = value.into_source(); }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledByteLevel {
    settings: ByteLevel,
    split: CompiledRegexSplit,
}
impl CompiledByteLevel {
    pub(crate) fn from_parts(settings: ByteLevel, split: CompiledRegexSplit) -> Self {
        debug_assert!(!settings.add_prefix_space && settings.use_regex);
        debug_assert_eq!(split.pattern(), super::byte_level::DEFAULT_PATTERN);
        Self { settings, split }
    }
    pub(crate) fn settings(&self) -> ByteLevel {
        self.settings
    }
    pub(crate) fn workspace_plan(&self) -> Result<Plan<'_>, PlanError> {
        self.split.workspace_plan()
    }
    pub(crate) fn wrapper_control_bytes() -> Option<usize> {
        size_of::<Self>().checked_add(size_of::<
            Result<Self, fancy_regex::workspace::construction::Failure>,
        >())
    }
}
impl Serialize for CompiledByteLevel {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.settings.serialize(serializer)
    }
}
impl PreTokenizer for CompiledByteLevel {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> crate::Result<()> {
        let mut workspace = self.workspace_plan()?.prepare().map_err(|e| e.retire())?;
        self.settings.pre_tokenize_with(pretokenized, |normalized| {
            let mut pieces = Vec::new();
            compiled_split::visit_spans::<crate::Error>(
                &mut workspace,
                normalized.get(),
                |offsets| {
                    pieces.push(normalized.isolated_piece(offsets));
                    Ok(())
                },
            )?;
            Ok(pieces)
        })
    }
}
#[cfg(test)]
mod tests;
