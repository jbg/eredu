//! Borrowed profile metadata and allocation-free ordinary validation failures.
use super::{
    DeclarativeDialectSpec, DelimitedChannel, GenerationPromptBehavior, ToolNameConstraint,
};
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProfileDeclaration {
    pub(crate) generation: GenerationPromptBehavior,
    pub(crate) reasoning_kwarg: &'static str,
    pub(crate) tool_reasoning: bool,
    pub(crate) reasoning_parsing: bool,
    pub(crate) structural: &'static [&'static str],
    pub(crate) stops: &'static [&'static str],
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum ToolNameError<'a> {
    #[error("declarative tool-name limit must be positive")]
    ZeroLimit,
    #[error(
        "tool function name {name:?} must contain at most {max_length} ASCII letters, digits, underscores, or dashes"
    )]
    Invalid { name: &'a str, max_length: usize },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum DeclarationError {
    #[error("{0}")]
    Message(&'static str),
    #[error("a bare JSON object must use {0:?} as its exact activation trigger")]
    BareActivation(&'static str),
    #[error("declarative {0} channel requires non-empty delimiters")]
    Channel(&'static str),
    #[error(transparent)]
    Name(#[from] ToolNameError<'static>),
}
impl DeclarationError {
    pub(super) fn ordinary(self) -> String {
        self.to_string()
    }
}
impl ProfileDeclaration {
    pub(crate) fn control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Self>(),
            size_of::<(
                super::DeclarativePayloadShape,
                super::JsonFunctionEnvelope,
                Option<super::DeclarativeCallId>,
                super::NamedJsonArgumentsEncoding,
                super::TaggedParametersEncoding,
                super::StructuralObjectEncoding,
            )>(),
            size_of::<(
                Option<super::ExactEnvelope>,
                Option<super::NamedCallIdEncoding>,
                Option<DelimitedChannel>,
                &DeclarativeDialectSpec,
            )>(),
            size_of::<DeclarationError>(),
            size_of::<Result<Self, DeclarationError>>(),
            size_of::<Result<(), DeclarationError>>(),
            size_of::<Result<&DeclarativeDialectSpec, DeclarationError>>(),
            size_of::<ToolNameError<'_>>(),
            size_of::<Result<(), ToolNameError<'_>>>(),
            size_of::<(ToolNameConstraint, &str, std::str::Bytes<'_>)>(),
            size_of::<[&str; 5]>(),
            size_of::<std::slice::Iter<'_, &str>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, &str>>>(),
            size_of::<std::iter::Take<std::slice::Iter<'_, &str>>>(),
            size_of::<[(&str, Option<DelimitedChannel>); 2]>(),
            size_of::<std::array::IntoIter<(&str, Option<DelimitedChannel>), 2>>(),
            size_of::<(usize, &str, &str, bool, bool, bool)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
