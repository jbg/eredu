//! Grammar directive destinations over the ordinary funded JSON event worker.
use derivre::{ParserResult as Result, ParserError, parser_error as anyhow, parser_bail as bail, parser_ensure as ensure};
use derivre::ParserAllocationFunding;
use serde_json::{bounded_events::{Event, Plan, Sink}, Value};
use crate::api::RegexExt;

use crate::allocation::CompilerAllocation;

pub(super) fn parse(data: &[u8], funding: &ParserAllocationFunding) -> Result<(Value, usize)> {
    serde_json::bounded_events::from_prefix_with_allocations(data, &CompilerAllocation(funding))
        .map_err(|error| match funding.failure() {
            Some(error) => error.into(),
            None => ParserError::cause(error, funding),
        })
}

// Value objects intentionally retain the last duplicate key. A typed %regex
// declaration instead rejects duplicate known fields, as its Deserialize does.
// This fixed event check uses the same parser and no duplicate owned JSON tree.
#[derive(Default)]
struct RegexFields { depth: usize, seen: u8, duplicate: bool }
impl Sink for RegexFields {
    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Object | Event::Array => self.depth += 1,
            Event::EndObject | Event::EndArray => self.depth -= 1,
            Event::Key(key) if self.depth == 1 => {
                let bit = match key { "substring_chunks" => 1, "substring_words" => 2, "substring_chars" => 4, _ => 0 };
                self.duplicate |= self.seen & bit != 0;
                self.seen |= bit;
            }
            _ => (),
        }
    }
}

pub(super) fn parse_regex(data: &[u8], funding: &ParserAllocationFunding) -> Result<(RegexExt, usize)> {
    let (value, consumed) = parse(data, funding)?;
    let plan = Plan::prepare(&data[..consumed]).map_err(|error| derivre::ParserError::cause(error, funding))?;
    funding.reserve(plan.requirements::<RegexFields>().map_err(|error| derivre::ParserError::cause(error, funding))?.required_bytes())?;
    let mut fields = RegexFields::default();
    plan.parse(&mut fields,&CompilerAllocation(funding)).map_err(|error| derivre::ParserError::cause(error, funding))?;
    if fields.duplicate { bail!(funding, "duplicate field in %regex declaration"); }
    let Value::Object(mut object) = value else { bail!(funding, "%regex declaration must be an object"); };
    let words = optional_string(object.remove("substring_words"), funding)?;
    let chars = optional_string(object.remove("substring_chars"), funding)?;
    let chunks = match object.remove("substring_chunks") {
        None | Some(Value::Null) => None,
        Some(Value::Array(values)) => {
            let mut chunks = Vec::new();
            for value in values {
                let Value::String(value) = value else { bail!(funding, "substring_chunks must contain strings"); };
                funding.try_push(&mut chunks, value)?;
            }
            Some(chunks)
        }
        Some(_) => bail!(funding, "substring_chunks must be an array or null"),
    };
    if !object.is_empty() { bail!(funding, "unknown field in %regex declaration"); }
    Ok((RegexExt { substring_words: words, substring_chars: chars, substring_chunks: chunks }, consumed))
}
fn optional_string(value: Option<Value>, funding: &ParserAllocationFunding) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => bail!(funding, "%regex string field must be a string or null"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn regex_directives_preserve_typed_fields_and_duplicate_rejection() {
        for source in [
            r#"{}"#, r#"{"substring_chars":"a😀b"}"#,
            r#"{"substring_chunks":["ab","cd"],"substring_words":null}"#,
            r#"{"substring_words":"ab","substring_words":"cd"}"#,
            r#"{"unknown":"x"}"#, r#"{"substring_chars":1}"#,
            r#"{"substring_chunks":[1]}"#, r#"[]"#,
        ] {
            let expected = serde_json::from_str::<RegexExt>(source);
            let actual = parse_regex(source.as_bytes(), &ParserAllocationFunding::unenforced());
            match (expected, actual) {
                (Ok(expected), Ok((actual, consumed))) => {
                    assert_eq!(consumed, source.len());
                    assert_eq!(expected.substring_words, actual.substring_words);
                    assert_eq!(expected.substring_chars, actual.substring_chars);
                    assert_eq!(expected.substring_chunks, actual.substring_chunks);
                }
                (Err(_), Err(_)) => (),
                (expected, actual) => panic!("directive mismatch {source}: {expected:?}, {actual:?}"),
            }
        }
    }
}
