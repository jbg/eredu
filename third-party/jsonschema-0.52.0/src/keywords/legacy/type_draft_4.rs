use crate::{
    compiler,
    error::ValidationError,
    keywords::{type_, CompilationResult},
    paths::{LazyLocation, Location, RefTracker},
    types::{JsonType, JsonTypeSet},
    validator::{Validate, ValidationContext},
    Json, Node,
};
use serde_json::{json, Map, Value};
use std::{borrow::Cow, str::FromStr};

pub(crate) struct MultipleTypesValidator {
    types: JsonTypeSet,
    location: Location,
}

impl MultipleTypesValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        items: &'a [Value],
        location: Location,
    ) -> CompilationResult<'a, F> {
        let mut types = JsonTypeSet::empty();
        for item in items {
            match item {
                Value::String(string) => {
                    if let Ok(ty) = JsonType::from_str(string.as_str()) {
                        types = types.insert(ty);
                    } else {
                        let options = ctx.funding().copy_vec(
                            &[
                                "array", "boolean", "integer", "null", "number", "object", "string",
                            ],
                            |name| Ok(Value::String(ctx.funding().copy_str(name)?)),
                        )?;
                        return Err(ValidationError::new_with_funding(
                            Cow::Borrowed(item),
                            crate::error::ValidationErrorKind::Enum {
                                options: Value::Array(options),
                            },
                            Location::new_with_funding(ctx.funding())?,
                            location.clone(),
                            location,
                            ctx.funding(),
                        )?
                        .into());
                    }
                }
                _ => {
                    return Err(ValidationError::single_type_error_with_funding(
                        location.clone(),
                        location,
                        Location::new_with_funding(ctx.funding())?,
                        Cow::Borrowed(item),
                        JsonType::String,
                        ctx.funding(),
                    )?
                    .into())
                }
            }
        }
        Ok(ctx
            .funding()
            .boxed(MultipleTypesValidator { types, location })?)
    }
}

impl<F: Json> Validate<F> for MultipleTypesValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.location(&self.location)
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        if cfg!(feature = "arbitrary-precision") {
            return Err(crate::validator::workspace::Error::Unqualified(
                crate::validator::workspace::Component::Validator(
                    "draft4 arbitrary-precision integer",
                ),
            ));
        }
        crate::validator::workspace::body_controls::<F, Self>(&[std::mem::size_of::<(
            crate::types::JsonType,
            crate::types::JsonTypeSet,
            Option<f64>,
            Option<u64>,
            Option<i64>,
        )>()])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        match instance.json_type() {
            JsonType::Number => {
                let Some(n) = instance.as_number() else {
                    return false;
                };
                if is_integer(&n) {
                    self.types.contains(JsonType::Integer) || self.types.contains(JsonType::Number)
                } else {
                    self.types.contains(JsonType::Number)
                }
            }
            other => self.types.contains(other),
        }
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
            ctx.diagnostic::<F>(instance, location, tracker, &self.location, |_| {
                Ok(crate::error::ValidationErrorKind::Type {
                    kind: crate::error::TypeKind::Multiple(self.types),
                })
            })
        }
    }
}

pub(crate) struct IntegerTypeValidator {
    location: Location,
}

impl IntegerTypeValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(IntegerTypeValidator { location })?)
    }
}

impl<F: Json> Validate<F> for IntegerTypeValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.location(&self.location)
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        if cfg!(feature = "arbitrary-precision") {
            return Err(crate::validator::workspace::Error::Unqualified(
                crate::validator::workspace::Component::Validator(
                    "draft4 arbitrary-precision integer",
                ),
            ));
        }
        crate::validator::workspace::body_controls::<F, Self>(&[std::mem::size_of::<(
            crate::types::JsonType,
            crate::types::JsonTypeSet,
            Option<f64>,
            Option<u64>,
            Option<i64>,
        )>()])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(num) = instance.as_number() {
            is_integer(&num)
        } else {
            false
        }
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
            ctx.diagnostic::<F>(instance, location, tracker, &self.location, |_| {
                Ok(crate::error::ValidationErrorKind::Type {
                    kind: crate::error::TypeKind::Single(JsonType::Integer),
                })
            })
        }
    }
}

pub(crate) fn is_integer<N: jsonschema_value::JsonNumber>(num: &N) -> bool {
    if num.as_u64().is_some() || num.as_i64().is_some() {
        return true;
    }
    // Draft 4 is strict: numbers written with decimal points are NOT integers,
    // regardless of whether the fractional part is zero.
    // See: tests/suite/tests/draft4/optional/zeroTerminatedFloats.json
    #[cfg(feature = "arbitrary-precision")]
    {
        use crate::numeric::bignum;

        if num.as_str().contains('.') {
            return false;
        }
        // Plain integers or scientific notation without decimal point
        bignum::try_parse_bigint(&num.to_number()).is_some()
    }
    #[cfg(not(feature = "arbitrary-precision"))]
    {
        // Without the raw text, a number past `i64`/`u64` is an `f64` - and every `f64` of that
        // magnitude is an exact integer, so there is no decimal point left to be strict about.
        num.as_f64()
            .is_some_and(|value| value.abs() >= crate::canonical::json::I64_UPPER_EXCLUSIVE_F64)
    }
}

#[inline]
pub(crate) fn compile<'a, F: Json>(
    ctx: &compiler::Context<F>,
    parent: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    // Absorbed by the fused array-shape validator emitted from `items`.
    if crate::keywords::items::array_shape_fusion(ctx, parent) {
        return None;
    }
    let location =
        crate::keywords::try_compile!(ctx.location().join_with_funding("type", ctx.funding()));
    match schema {
        Value::String(item) => Some(compile_single_type(ctx, item.as_str(), location, schema)),
        Value::Array(items) => {
            if items.len() == 1 {
                let item = &items[0];
                if let Value::String(ty) = item {
                    Some(compile_single_type(ctx, ty.as_str(), location, item))
                } else {
                    Some(Err(crate::keywords::try_compile!(
                        ValidationError::single_type_error_with_funding(
                            location.clone(),
                            location,
                            crate::keywords::try_compile!(Location::new_with_funding(
                                ctx.funding()
                            )),
                            Cow::Borrowed(item),
                            JsonType::String,
                            ctx.funding()
                        )
                    )
                    .into()))
                }
            } else {
                Some(MultipleTypesValidator::compile(ctx, items, location))
            }
        }
        _ => {
            let location = crate::keywords::try_compile!(ctx
                .location()
                .join_with_funding("type", ctx.funding()));
            Some(Err(crate::keywords::try_compile!(
                ValidationError::multiple_type_error_with_funding(
                    location.clone(),
                    location,
                    crate::keywords::try_compile!(Location::new_with_funding(ctx.funding())),
                    Cow::Borrowed(schema),
                    JsonTypeSet::from(JsonType::String).insert(JsonType::Array),
                    ctx.funding()
                )
            )
            .into()))
        }
    }
}

fn compile_single_type<'a, F: Json>(
    ctx: &compiler::Context<F>,
    item: &str,
    location: Location,
    instance: &'a Value,
) -> CompilationResult<'a, F> {
    match JsonType::from_str(item) {
        Ok(JsonType::Array) => type_::ArrayTypeValidator::compile(ctx, location),
        Ok(JsonType::Boolean) => type_::BooleanTypeValidator::compile(ctx, location),
        Ok(JsonType::Integer) => IntegerTypeValidator::compile(ctx, location),
        Ok(JsonType::Null) => type_::NullTypeValidator::compile(ctx, location),
        Ok(JsonType::Number) => type_::NumberTypeValidator::compile(ctx, location),
        Ok(JsonType::Object) => type_::ObjectTypeValidator::compile(ctx, location),
        Ok(JsonType::String) => type_::StringTypeValidator::compile(ctx, location),
        Err(()) => Err(ValidationError::compile_error(
            location.clone(),
            location,
            Location::new_with_funding(ctx.funding())?,
            Cow::Borrowed(instance),
            "Unexpected type",
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::Value;
    use test_case::test_case;

    fn parse_json(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    // Draft 4 is strict: floats like 1.0 are NOT integers
    #[test_case(r#"{"type": "integer"}"#, "42", true; "plain integer")]
    #[test_case(r#"{"type": "integer"}"#, "-42", true; "negative integer")]
    #[test_case(r#"{"type": "integer"}"#, "0", true; "zero")]
    #[test_case(r#"{"type": "integer"}"#, "1.0", false; "float with .0 is not integer in draft4")]
    #[test_case(r#"{"type": "integer"}"#, "42.0", false; "integer as float is not integer in draft4")]
    #[test_case(r#"{"type": "integer"}"#, "-42.0", false; "negative float with .0 is not integer in draft4")]
    #[test_case(r#"{"type": "integer"}"#, "1.5", false; "decimal")]
    #[test_case(r#"{"type": "integer"}"#, "0.1", false; "small decimal")]
    #[test_case(r#"{"type": "integer"}"#, "42.7", false; "float with decimal")]
    #[test_case(r#"{"type": "integer"}"#, "9223372036854775807", true; "i64::MAX")]
    #[test_case(r#"{"type": "integer"}"#, "-9223372036854775808", true; "i64::MIN")]
    #[test_case(r#"{"type": "integer"}"#, "18446744073709551615", true; "u64::MAX")]
    #[test_case(r#"{"type": "integer"}"#, "true", false; "boolean")]
    #[test_case(r#"{"type": "integer"}"#, r#""42""#, false; "string")]
    #[test_case(r#"{"type": "integer"}"#, "[]", false; "array")]
    #[test_case(r#"{"type": "integer"}"#, "{}", false; "object")]
    #[test_case(r#"{"type": "integer"}"#, "null", false; "null")]
    fn integer_type_validation_draft4(schema_json: &str, instance_json: &str, expected: bool) {
        let schema = parse_json(schema_json);
        let instance = parse_json(instance_json);
        if expected {
            tests_util::is_valid_with_draft4(&schema, &instance);
        } else {
            tests_util::is_not_valid_with_draft4(&schema, &instance);
        }
    }

    #[cfg(feature = "arbitrary-precision")]
    mod arbitrary_precision {
        use crate::tests_util;
        use serde_json::Value;
        use test_case::test_case;

        fn parse_json(s: &str) -> Value {
            serde_json::from_str(s).unwrap()
        }

        // Tests for huge integers beyond i64/u64 range - these must be parsed from JSON string
        // to avoid Python/Rust int conversion issues
        #[test_case(r#"{"type": "integer"}"#, "18446744073709551616", true; "u64_max_plus_1_plain")]
        #[test_case(r#"{"type": "integer"}"#, "-9223372036854775809", true; "i64_min_minus_1")]
        #[test_case(r#"{"type": "integer"}"#, "99999999999999999999", true; "huge_plain_integer")]
        #[test_case(
            r#"{"type": "integer"}"#,
            "999999999999999999999999999999999999999999999999999999999999999999999999999999",
            true;
            "very_huge_plain_integer"
        )]
        #[test_case(r#"{"type": "integer"}"#, "-18446744073709551616", true; "negative_huge_integer")]
        #[test_case(r#"{"type": "integer"}"#, "-99999999999999999999", true; "negative_huge_plain")]
        // Numbers with decimal points are NOT integers in Draft 4, even if fractional part is zero
        #[test_case(r#"{"type": "integer"}"#, "18446744073709551616.0", false; "huge_with_dot_0_not_integer")]
        #[test_case(r#"{"type": "integer"}"#, "99999999999999999999.0", false; "huge_integer_with_dot_0_not_integer")]
        #[test_case(r#"{"type": "integer"}"#, "-18446744073709551616.0", false; "negative_huge_with_dot_0_not_integer")]
        #[test_case(r#"{"type": "integer"}"#, "-99999999999999999999.0", false; "negative_very_huge_with_dot_0_not_integer")]
        #[test_case(r#"{"type": "integer"}"#, "18446744073709551616.5", false; "huge decimal")]
        #[test_case(r#"{"type": "integer"}"#, "99999999999999999999.5", false; "huge float")]
        #[test_case(
            r#"{"type": "integer"}"#,
            "999999999999999999999999999999999999999999999999999999999999999999999999999999.5",
            false;
            "very huge float"
        )]
        #[test_case(r#"{"type": "integer"}"#, "1e1000", true; "huge scientific notation integer")]
        #[test_case(r#"{"type": "integer"}"#, "1e1000001", false; "infinity positive")]
        #[test_case(r#"{"type": "integer"}"#, "-1e1000001", false; "infinity negative")]
        #[test_case(r#"{"type": ["integer", "string"]}"#, "18446744073709551616", true; "huge int in union")]
        #[test_case(r#"{"type": ["integer", "string"]}"#, "-9223372036854775809", true; "huge negative int in union")]
        #[test_case(r#"{"type": ["integer", "string"]}"#, "18446744073709551616.0", false; "huge .0 not integer in union")]
        #[test_case(r#"{"type": ["integer", "string"]}"#, "18446744073709551616.5", false; "huge float not in union")]
        fn huge_number_integer_validation_draft4(
            schema_json: &str,
            instance_json: &str,
            expected: bool,
        ) {
            let schema = parse_json(schema_json);
            let instance = parse_json(instance_json);
            if expected {
                tests_util::is_valid_with_draft4(&schema, &instance);
            } else {
                tests_util::is_not_valid_with_draft4(&schema, &instance);
            }
        }
    }
}
