//! Source-bound construction and rendering through checked compiler storage.
//!
//! This API does not initialize an Environment or provide an allocation-account
//! grant. Callers admit the returned concrete requirements before constructing.
//! Ordinary constructors, Value semantics and rendering remain available.

#![forbid(unsafe_code)]

/// Shared default-syntax lexical frontend and closed literal storage.
pub mod frontend;

/// Retained local clock facts and the shared allocation-free date formatter.
#[cfg(feature = "chat-clock")]
pub mod clock;
mod compile;
mod input;
mod recipe;
mod render;
mod source;
mod worker;

pub use compile::SourceCompilePlan;
pub use input::{
    InputError, Messages, RenderContext, ScalarBinding, ScalarBindingValue, TextMessage,
};
#[cfg(feature = "json")]
pub use input::{InputArray, RecordField, RecordFields, RecordValue};

pub use render::{
    JsonCapacity, RenderBuffer, RenderCause, RenderError, RenderFailure, RenderPlan,
    RenderPlanError, RenderRequirements, Rendered, TextCapacity, ValueCapacity,
};

pub use source::{
    CompileCause, CompileError, CompilePlan, CompileRequirements, PreparedTemplate, SourceBuffer,
    SourceError, TemplateName, TemplateSettings, supported_source,
};

#[cfg(test)]
mod tests;

/// Source-bound expression parsing through the ordinary grammar.
pub mod expression;

/// Source-bound full-template parsing through the ordinary shared grammar.
pub mod template_syntax;

/// Shared Python-compatible JSON formatting used by the registered chat filter.
#[cfg(feature = "json")]
pub mod json_format;
