use crate::{
    compiler,
    error::ValidationError,
    keywords::CompilationResult,
    paths::Location,
    validator::{Validate, ValidationContext},
    Array, Json, Node,
};
use jsonschema_value::literal::Literal;
use serde_json::{Map, Number, Value};

use crate::paths::{LazyLocation, RefTracker};

struct ConstArrayValidator {
    value: Vec<Literal>,
    location: Location,
}
impl ConstArrayValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        value: &'a [Value],
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(ConstArrayValidator {
            value: ctx.funding().literals(value)?,
            location,
        })?)
    }
}
impl<F: Json> Validate<F> for ConstArrayValidator {
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<&Literal>(),
            std::mem::size_of::<std::slice::Iter<'_, Literal>>(),
        ])
    }

    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(&self.value)?;
        for value in &self.value {
            source.literal(value)?;
        }
        source.location(&self.location)
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
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
                Ok(crate::error::ValidationErrorKind::Constant {
                    expected_value: Value::Array(
                        funding.copy_vec(&self.value, |value| funding.literal_value(value))?,
                    ),
                })
            })
        }
    }

    #[inline]
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(items) = instance.as_array() {
            items.len() == self.value.len()
                && items
                    .elements()
                    .zip(&self.value)
                    .all(|(item, expected)| ctx.equals_literal::<F>(&item, expected))
        } else {
            false
        }
    }
}

struct ConstBooleanValidator {
    value: bool,
    location: Location,
}
impl ConstBooleanValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        value: bool,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx
            .funding()
            .boxed(ConstBooleanValidator { value, location })?)
    }
}
impl<F: Json> Validate<F> for ConstBooleanValidator {
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<Option<bool>>(),
            std::mem::size_of::<bool>(),
            std::mem::size_of::<Option<std::borrow::Cow<'_, str>>>(),
            std::mem::size_of::<(&str, &str)>(),
        ])
    }

    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.location(&self.location)
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
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
                Ok(crate::error::ValidationErrorKind::Constant {
                    expected_value: Value::Bool(self.value),
                })
            })
        }
    }

    #[inline]
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        instance.as_boolean() == Some(self.value)
    }
}

struct ConstNullValidator {
    location: Location,
}
impl ConstNullValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(ConstNullValidator { location })?)
    }
}
impl<F: Json> Validate<F> for ConstNullValidator {
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<Option<bool>>(),
            std::mem::size_of::<bool>(),
            std::mem::size_of::<Option<std::borrow::Cow<'_, str>>>(),
            std::mem::size_of::<(&str, &str)>(),
        ])
    }

    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.location(&self.location)
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
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
                Ok(crate::error::ValidationErrorKind::Constant {
                    expected_value: Value::Null,
                })
            })
        }
    }
    #[inline]
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        instance.is_null()
    }
}

struct ConstNumberValidator {
    // This is saved in order to ensure that the error message is not altered by precision loss
    original_value: Number,
    location: Location,
}

impl ConstNumberValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        original_value: &Number,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(ConstNumberValidator {
            original_value: ctx.funding().number(original_value)?,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for ConstNumberValidator {
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        let numeric = crate::cmp::original_number_equality_control_bytes::<
            <F::Node<'_> as crate::Node<'_, F>>::Number,
        >()
        .ok_or(crate::validator::workspace::Error::Unqualified(
            crate::validator::workspace::Component::Validator(
                "arbitrary-precision number equality",
            ),
        ))?;
        crate::validator::workspace::body_controls::<F, Self>(&[
            numeric,
            std::mem::size_of::<&serde_json::Number>(),
        ])
    }

    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.number(&self.original_value)?;
        source.location(&self.location)
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
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
                Ok(crate::error::ValidationErrorKind::Constant {
                    expected_value: Value::Number(funding.number(&self.original_value)?),
                })
            })
        }
    }

    #[inline]
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(item) = instance.as_number() {
            crate::cmp::equal_numbers(&item, &self.original_value)
        } else {
            false
        }
    }
}

pub(crate) struct ConstObjectValidator {
    value: Literal,
    location: Location,
}

impl ConstObjectValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        value: &'a Map<String, Value>,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(ConstObjectValidator {
            value: ctx.funding().literal_object(value)?,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for ConstObjectValidator {
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<&Literal>(),
            std::mem::size_of::<std::slice::Iter<'_, Literal>>(),
        ])
    }
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.literal(&self.value)?;
        source.location(&self.location)
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
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
                Ok(crate::error::ValidationErrorKind::Constant {
                    expected_value: funding.literal_value(&self.value)?,
                })
            })
        }
    }

    #[inline]
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        ctx.equals_literal::<F>(instance, &self.value)
    }
}

pub(crate) struct ConstStringValidator {
    value: String,
    location: Location,
}

impl ConstStringValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        value: &'a str,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(ConstStringValidator {
            value: ctx.funding().copy_str(value)?,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for ConstStringValidator {
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<Option<bool>>(),
            std::mem::size_of::<bool>(),
            std::mem::size_of::<Option<std::borrow::Cow<'_, str>>>(),
            std::mem::size_of::<(&str, &str)>(),
        ])
    }

    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.string(&self.value)?;
        source.location(&self.location)
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
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
                Ok(crate::error::ValidationErrorKind::Constant {
                    expected_value: Value::String(funding.copy_str(&self.value)?),
                })
            })
        }
    }

    #[inline]
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(item) = instance.as_string() {
            self.value == item.as_ref()
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
    let location =
        crate::keywords::try_compile!(ctx.location().join_with_funding("const", ctx.funding()));
    match schema {
        Value::Array(items) => Some(ConstArrayValidator::compile(ctx, items, location)),
        Value::Bool(item) => Some(ConstBooleanValidator::compile(ctx, *item, location)),
        Value::Null => Some(ConstNullValidator::compile(ctx, location)),
        Value::Number(item) => Some(ConstNumberValidator::compile(ctx, item, location)),
        Value::Object(map) => Some(ConstObjectValidator::compile(ctx, map, location)),
        Value::String(string) => Some(ConstStringValidator::compile(ctx, string, location)),
    }
}

#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::{json, Value};
    use test_case::test_case;

    #[test_case(&json!({"const": 1}), &json!(2), "/const")]
    #[test_case(&json!({"const": null}), &json!(3), "/const")]
    #[test_case(&json!({"const": false}), &json!(4), "/const")]
    #[test_case(&json!({"const": []}), &json!(5), "/const")]
    #[test_case(&json!({"const": {}}), &json!(6), "/const")]
    #[test_case(&json!({"const": ""}), &json!(7), "/const")]
    fn location(schema: &Value, instance: &Value, expected: &str) {
        tests_util::assert_schema_location(schema, instance, expected);
    }

    // Tests for arbitrary-precision const validation
    #[cfg(feature = "arbitrary-precision")]
    mod arbitrary_precision {
        use crate::tests_util;
        use serde_json::Value;
        use test_case::test_case;

        fn parse_json(json: &str) -> Value {
            serde_json::from_str(json).unwrap()
        }

        #[test_case(r#"{"const": 18446744073709551617}"#, "18446744073709551617", true; "large int exact match")]
        #[test_case(r#"{"const": 18446744073709551617}"#, "18446744073709551616", false; "large int different by one")]
        #[test_case(r#"{"const": 18446744073709551617}"#, "18446744073709551618", false; "large int different by one above")]
        #[test_case(r#"{"const": -9223372036854775809}"#, "-9223372036854775809", true; "large negative int match")]
        #[test_case(r#"{"const": -9223372036854775809}"#, "-9223372036854775808", false; "large negative int different")]
        #[test_case(r#"{"const": 0.1}"#, "0.1", true; "decimal exact match")]
        #[test_case(r#"{"const": 0.1}"#, "0.10000000000000001", false; "decimal precision difference")]
        fn const_arbitrary_precision(schema_json: &str, instance_json: &str, expected_valid: bool) {
            let schema = parse_json(schema_json);
            let instance = parse_json(instance_json);
            if expected_valid {
                tests_util::is_valid(&schema, &instance);
            } else {
                tests_util::is_not_valid(&schema, &instance);
            }
        }
    }
}
