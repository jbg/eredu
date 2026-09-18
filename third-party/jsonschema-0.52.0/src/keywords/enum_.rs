use crate::{
    compiler,
    error::ValidationError,
    keywords::CompilationResult,
    paths::{LazyLocation, Location, RefTracker},
    types::{JsonType, JsonTypeSet},
    validator::{Validate, ValidationContext},
    Json, Node,
};

use jsonschema_value::literal::Literal;
use serde_json::{Map, Value};
use std::borrow::Cow;

const STRING_ENUM_THRESHOLD: usize = 10;

#[derive(Debug)]
pub(crate) struct EnumValidator {
    options: Literal,
    // Types that occur in items
    types: JsonTypeSet,
    items: Vec<Literal>,
    location: Location,
}

impl EnumValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        schema: &'a Value,
        items: &'a [Value],
        location: Location,
    ) -> CompilationResult<'a, F> {
        let mut types = JsonTypeSet::empty();
        for item in items {
            types = types.insert(JsonType::from(item));
        }
        Ok(ctx.funding().boxed(EnumValidator {
            options: ctx.funding().literal(schema)?,
            items: ctx.funding().literals(items)?,
            types,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for EnumValidator {
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        if cfg!(feature="arbitrary-precision") {return Err(crate::validator::workspace::Error::Unqualified(crate::validator::workspace::Component::Validator("arbitrary-precision integer classification")));}

        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<&Literal>(),
            std::mem::size_of::<std::slice::Iter<'_, Literal>>(),
            std::mem::size_of::<crate::types::JsonTypeSet>(),
        ])
    }

    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.literal(&self.options)?;
        source.vector(&self.items)?;
        for value in &self.items {
            source.literal(value)?;
        }
        source.location(&self.location)
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if Validate::<F>::is_valid(self, instance, ctx) {
            Ok(())
        } else {
            ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
                Ok(crate::error::ValidationErrorKind::Enum {
                    options: funding.literal_value(&self.options)?,
                })
            })
        }
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        // If the input value type is not in the types present among the enum options, then there
        // is no reason to compare it against all items - we know that
        // there are no items with such type at all
        if self.types.contains_value_type::<F>(instance) {
            self.items
                .iter()
                .any(|item| ctx.equals_literal::<F>(instance, item))
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub(crate) struct SingleValueEnumValidator {
    value: Literal,
    options: Literal,
    location: Location,
}

impl SingleValueEnumValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        schema: &'a Value,
        value: &'a Value,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(SingleValueEnumValidator {
            options: ctx.funding().literal(schema)?,
            value: ctx.funding().literal(value)?,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for SingleValueEnumValidator {
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<&Literal>(),
            std::mem::size_of::<std::slice::Iter<'_, Literal>>(),
            std::mem::size_of::<crate::types::JsonTypeSet>(),
        ])
    }

    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.literal(&self.options)?;
        source.literal(&self.value)?;
        source.location(&self.location)
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if Validate::<F>::is_valid(self, instance, ctx) {
            Ok(())
        } else {
            ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
                Ok(crate::error::ValidationErrorKind::Enum {
                    options: funding.literal_value(&self.options)?,
                })
            })
        }
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        ctx.equals_literal::<F>(instance, &self.value)
    }
}

#[derive(Debug)]
pub(crate) struct SmallStringEnumValidator {
    options: Value,
    items: Vec<Box<str>>,
    location: Location,
}

impl SmallStringEnumValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        schema: &'a Value,
        items: &'a [Value],
        location: Location,
    ) -> CompilationResult<'a, F> {
        let mut strings = Vec::new();
        ctx.funding().grow(&mut strings, items.len())?;
        for item in items {
            strings.push(
                ctx.funding()
                    .boxed_str(item.as_str().expect("all items are strings"))?,
            );
        }
        Ok(ctx.funding().boxed(SmallStringEnumValidator {
            options: ctx.funding().value(schema)?,
            items: strings,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for SmallStringEnumValidator {
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<Option<bool>>(),
            std::mem::size_of::<bool>(),
            std::mem::size_of::<Option<std::borrow::Cow<'_, str>>>(),
            std::mem::size_of::<(&str, &str)>(),
            std::mem::size_of::<std::slice::Iter<'_, Box<str>>>(),
            std::mem::size_of::<(&Box<str>, &std::borrow::Cow<'_, str>)>(),
        ])
    }

    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.value(&self.options)?;
        source.vector(&self.items)?;
        for value in &self.items {
            source.add(value.len())?;
        }
        source.location(&self.location)
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if Validate::<F>::is_valid(self, instance, ctx) {
            Ok(())
        } else {
            ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
                Ok(crate::error::ValidationErrorKind::Enum {
                    options: funding.value(&self.options)?,
                })
            })
        }
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(s) = instance.as_string() {
            self.items.iter().any(|item| item.as_ref() == s.as_ref())
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub(crate) struct BigStringEnumValidator {
    options: Value,
    items: hashbrown::HashSet<Box<str>, ahash::RandomState>,
    location: Location,
}

impl BigStringEnumValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        schema: &'a Value,
        items: &'a [Value],
        location: Location,
    ) -> CompilationResult<'a, F> {
        let mut strings = hashbrown::HashSet::with_hasher(ctx.funding().random_state()?);
        for item in items {
            let value = ctx
                .funding()
                .boxed_str(item.as_str().expect("all items are strings"))?;
            ctx.funding().insert_set(&mut strings, value)?;
        }
        Ok(ctx.funding().boxed(BigStringEnumValidator {
            options: ctx.funding().value(schema)?,
            items: strings,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for BigStringEnumValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.value(&self.options)?;
        source.add(self.items.allocation_size())?;
        for value in &self.items {
            source.add(value.len())?;
        }
        source.location(&self.location)
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            self.items
                .lookup_control_bytes("")
                .ok_or(crate::validator::workspace::Error::Overflow)?,
            std::mem::size_of::<Option<std::borrow::Cow<'_, str>>>(),
            std::mem::size_of::<(&str, &str)>(),
            std::mem::size_of::<ahash::AHasher>(),
            std::mem::size_of::<(u64, usize, usize)>(),
            std::mem::size_of::<Option<&Box<str>>>(),
        ])
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if Validate::<F>::is_valid(self, instance, ctx) {
            Ok(())
        } else {
            ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
                Ok(crate::error::ValidationErrorKind::Enum {
                    options: funding.value(&self.options)?,
                })
            })
        }
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(s) = instance.as_string() {
            self.items.contains(s.as_ref())
        } else {
            false
        }
    }
}

#[inline]
pub(crate) fn compile<'a, F: Json>(
    ctx: &compiler::Context<F>,
    _: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    if let Value::Array(items) = schema {
        let location =
            crate::keywords::try_compile!(ctx.location().join_with_funding("enum", ctx.funding()));
        if items.len() == 1 {
            let value = items.iter().next().expect("Vec is not empty");
            Some(SingleValueEnumValidator::compile(
                ctx, schema, value, location,
            ))
        } else if items.iter().all(|v| matches!(v, Value::String(_))) {
            if items.len() <= STRING_ENUM_THRESHOLD {
                Some(SmallStringEnumValidator::compile(
                    ctx, schema, items, location,
                ))
            } else {
                Some(BigStringEnumValidator::compile(
                    ctx, schema, items, location,
                ))
            }
        } else {
            Some(EnumValidator::compile(ctx, schema, items, location))
        }
    } else {
        let location =
            crate::keywords::try_compile!(ctx.location().join_with_funding("enum", ctx.funding()));
        Some(Err(crate::keywords::try_compile!(
            ValidationError::single_type_error_with_funding(
                location.clone(),
                location,
                crate::keywords::try_compile!(Location::new_with_funding(ctx.funding())),
                Cow::Borrowed(schema),
                JsonType::Array,
                ctx.funding()
            )
        )
        .into()))
    }
}

#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::{json, Value};
    use test_case::test_case;

    #[test_case(&json!({"enum": [1]}), &json!(2), "/enum")]
    #[test_case(&json!({"enum": [1, 3]}), &json!(2), "/enum")]
    fn location(schema: &Value, instance: &Value, expected: &str) {
        tests_util::assert_schema_location(schema, instance, expected);
    }

    // 10 entries — exercises BigStringEnumValidator
    const BIG_STRING_ENUM: &str = r#"{
        "enum": ["a","b","c","d","e","f","g","h","i","j","k"]
    }"#;

    #[test]
    fn big_string_enum_valid() {
        let schema: Value = serde_json::from_str(BIG_STRING_ENUM).unwrap();
        for s in &["a", "e", "j"] {
            tests_util::is_valid(&schema, &json!(s));
        }
    }

    #[test]
    fn big_string_enum_invalid_string() {
        let schema: Value = serde_json::from_str(BIG_STRING_ENUM).unwrap();
        tests_util::is_not_valid(&schema, &json!("z"));
    }

    #[test]
    fn big_string_enum_invalid_type() {
        let schema: Value = serde_json::from_str(BIG_STRING_ENUM).unwrap();
        tests_util::is_not_valid(&schema, &json!(1));
        tests_util::is_not_valid(&schema, &json!(null));
    }

    #[test]
    fn big_string_enum_location() {
        let schema: Value = serde_json::from_str(BIG_STRING_ENUM).unwrap();
        tests_util::assert_schema_location(&schema, &json!("z"), "/enum");
    }

    #[test]
    fn big_string_enum_error_message() {
        let schema: Value = serde_json::from_str(BIG_STRING_ENUM).unwrap();
        tests_util::expect_errors(
            &schema,
            &json!("z"),
            &[r#""z" is not one of "a", "b" or 9 other candidates"#],
        );
    }
}
