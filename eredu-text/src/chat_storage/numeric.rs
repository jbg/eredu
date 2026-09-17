//! Numeric validation retains the same source borrow until admitted compilation.
use super::ChatSourceError;
use serde_json::bounded_number::{Plan, PlanError};
use std::{fmt, mem::size_of};
use tokenizers::utils::json_view::{Document, NumberValue, Numbers};

/// Fixed numeric storage qualification failure, before any parser allocation.
#[derive(Clone, Copy, Debug)]
pub struct ChatNumericPlanError {
    /// Original numeric token start.
    pub offset: usize,
    /// Dependency-owned representation or feature-profile refusal.
    pub cause: PlanError,
}
impl fmt::Display for ChatNumericPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for ChatNumericPlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// Actual serde numeric error, with original document position and source custody
/// supplied by the enclosing source-compilation owner.
#[derive(Debug)]
pub struct ChatNumericError {
    offset: usize,
    error: serde_json::Error,
    bytes: usize,
}
impl ChatNumericError {
    /// Start of the numeric token in the original document.
    pub fn offset(&self) -> usize {
        self.offset
    }
    /// Exact original parser error, including finite-range behavior and position.
    pub fn error(&self) -> &serde_json::Error {
        &self.error
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.bytes
    }
}
impl fmt::Display for ChatNumericError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for ChatNumericError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
#[derive(Debug)]
pub(super) struct NumericPlan<'a> {
    document: Document<'a>,
    buffers: usize,
    controls: usize,
    has_numbers: bool,
}
impl<'a> NumericPlan<'a> {
    pub(super) fn prepare(document: Document<'a>) -> Result<Self, ChatSourceError> {
        let mut buffers = 0;
        let mut controls = 0;
        let mut has_numbers = false;
        for number in document.numbers() {
            let plan = Plan::prepare(document.source(), number.range()).map_err(|cause| {
                ChatSourceError::Numeric(ChatNumericPlanError {
                    offset: number.offset(),
                    cause,
                })
            })?;
            let required = plan.requirements();
            // Numeric calls are sequential. Scratch retires synchronously and
            // the first error ends the pass; only one error Box can escape.
            let parser_buffers = required
                .failure_bytes()
                .checked_add(required.temporary_bytes())
                .ok_or(ChatSourceError::Overflow)?;
            buffers = buffers.max(parser_buffers);
            controls = controls.max(required.control_bytes());
            has_numbers = true;
        }
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, ChatSourceError>>(),
            size_of::<Numbers<'a>>(),
            size_of::<Option<NumberValue<'a>>>(),
            size_of::<NumberValue<'a>>(),
            size_of::<Plan<'a>>(),
            size_of::<Result<serde_json::Number, serde_json::Error>>(),
            size_of::<ChatNumericPlanError>(),
            size_of::<ChatNumericError>(),
            size_of::<Result<(), ChatNumericError>>(),
            controls,
        ];
        controls = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(ChatSourceError::Overflow)?;
        Ok(Self {
            document,
            buffers,
            controls,
            has_numbers,
        })
    }
    pub(super) fn buffers(&self) -> usize {
        self.buffers
    }
    pub(super) fn controls(&self) -> usize {
        self.controls
    }
    pub(super) fn has_numbers(&self) -> bool {
        self.has_numbers
    }
    pub(super) fn validate(self) -> Result<(), ChatNumericError> {
        for number in self.document.numbers() {
            let plan = Plan::prepare(self.document.source(), number.range())
                .expect("same immutable source and dependency profile as planning");
            let bytes = plan.requirements().failure_bytes();
            plan.parse().map_err(|error| ChatNumericError {
                offset: number.offset(),
                error,
                bytes,
            })?;
        }
        Ok(())
    }
}
