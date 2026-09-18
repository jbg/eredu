//! Lark quoted strings use the canonical serde event parser with paid output.
use derivre::{ParserAllocationFunding, ParserStorageError};
use serde_json::bounded_events::{Event, Plan, Sink};
struct StringValue<'a> {
    funding: &'a ParserAllocationFunding,
    value: Option<String>,
    failure: Option<ParserStorageError>,
    invalid: bool,
}
impl Sink for StringValue<'_> {
    fn event(&mut self, event: Event<'_>) {
        if self.failure.is_some() { return; }
        match event {
            Event::String(value) if self.value.is_none() => match self.funding.try_copy_str(value) {
                Ok(value) => self.value = Some(value),
                Err(error) => self.failure = Some(error),
            },
            _ => self.invalid = true,
        }
    }
}
pub(super) fn parse(source: &str, funding: &ParserAllocationFunding) -> derivre::ParserResult<String> {
    let plan = Plan::prepare(source.as_bytes()).map_err(|error| derivre::ParserError::cause(error, funding))?;
    funding.reserve(plan.requirements::<StringValue<'_>>().map_err(|error| derivre::ParserError::cause(error, funding))?.required_bytes())?;
    let mut sink = StringValue { funding, value: None, failure: None, invalid: false };
    let parsed = plan.parse(&mut sink,&crate::allocation::CompilerAllocation(funding));
    if let Some(error) = sink.failure { return Err(error.into()); }
    parsed.map_err(|error| derivre::ParserError::cause(error, funding))?;
    derivre::parser_ensure!(funding, !sink.invalid, "expected a Lark string literal");
    sink.value.ok_or_else(|| derivre::parser_error!(funding, "expected a Lark string literal"))
}
