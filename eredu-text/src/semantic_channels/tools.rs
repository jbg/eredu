//! Borrowed exact JSON tool declarations; selection and authority stay with callers.
use crate::json_fragments::JsonFieldNames;
/// Exact opening and closing bytes around a selected wire unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonEnvelope<'a> {
    /// Opening bytes, possibly empty.
    pub prefix: &'a str,
    /// Closing bytes, possibly empty.
    pub suffix: &'a str,
}
/// Selected JSON payload shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonToolShape {
    /// One JSON object per call.
    Object,
    /// A JSON list of call objects.
    List,
}
/// Existing parallel framing selected by the facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonToolLayout {
    /// One call envelope for each JSON object.
    RepeatedEnvelopes,
    /// A single envelope for the list of call objects.
    SingleEnvelope,
}
/// Literal JSON tool declaration projected from one selected protocol. It owns
/// no storage, schema validator, immutable source identity or execution permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonToolProgram<'a> {
    /// Outer collection wrapper.
    pub output: JsonEnvelope<'a>,
    /// Per-call or whole-list wrapper according to layout.
    pub call: JsonEnvelope<'a>,
    /// Wrapper around the JSON function object itself.
    pub function: JsonEnvelope<'a>,
    /// Exact field names and ID constraint.
    pub fields: JsonFieldNames<'a>,
    /// Object or list payload.
    pub shape: JsonToolShape,
    /// Exact bytes separating successive calls.
    pub separator: &'a str,
    /// Selected parallel call layout.
    pub layout: JsonToolLayout,
}
impl<'a> JsonToolProgram<'a> {
    /// Exact copied literal population in a stable constructor order. Absent
    /// IDs contribute an empty field and retain their separate absence bit.
    pub fn literals(self) -> [&'a str; 10] {
        [
            self.output.prefix,
            self.output.suffix,
            self.call.prefix,
            self.call.suffix,
            self.function.prefix,
            self.function.suffix,
            self.fields.name,
            self.fields.arguments,
            self.fields.call_id.map_or("", |id| id.field),
            self.separator,
        ]
    }
}
