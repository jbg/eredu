use std::borrow::Cow;

use crate::{
    compiler,
    error::ValidationError,
    keywords::CompilationResult,
    paths::{LazyLocation, Location, RefTracker},
    properties::HASHMAP_THRESHOLD,
    types::JsonType,
    validator::{Validate, ValidationContext},
    Json, Node, Object, SerdeJson,
};
use serde_json::{Map, Value};

pub(super) fn required_errors<'i, 'r, F: Json>(
    required: impl IntoIterator<Item = (&'r str, &'r F::PreparedKey)>,
    schema_path: &Location,
    instance: &F::Node<'i>,
    location: &LazyLocation,
    tracker: Option<&RefTracker>,
    ctx: &mut ValidationContext,
    mut output: Option<&mut Vec<ValidationError<'i>>>,
) -> Result<(), ValidationError<'i>> {
    if ctx.workspace.failed() {
        return Ok(());
    }
    let Some(object) = instance.as_object() else {
        return Ok(());
    };
    let required = required.into_iter();
    if !ctx.workspace.reserve(
        std::mem::size_of_val(&required).checked_add(std::mem::size_of::<(
            &str,
            &F::PreparedKey,
            Option<&mut Vec<ValidationError<'i>>>,
            Result<(), ValidationError<'i>>,
        )>()),
    ) {
        return Ok(());
    }
    for (name, key) in required {
        if object.get(key).is_none() {
            if let Err(error) =
                ctx.diagnostic::<F>(instance, location, tracker, schema_path, |funding| {
                    Ok(crate::error::ValidationErrorKind::Required {
                        property: Value::String(funding.copy_str(name)?),
                    })
                })
            {
                if let Some(errors) = output.as_deref_mut() {
                    if !ctx.workspace.push(errors, error) {
                        return Ok(());
                    }
                } else {
                    return Err(error);
                }
            }
            if ctx.workspace.failed() {
                return Ok(());
            }
        }
    }
    Ok(())
}

pub(crate) struct RequiredValidator<F: Json = SerdeJson> {
    required: Vec<(String, F::PreparedKey)>,
    location: Location,
}

impl RequiredValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        items: &'a [Value],
        location: Location,
    ) -> CompilationResult<'a, F> {
        let mut required = Vec::new();
        ctx.funding().grow(&mut required, items.len())?;
        for item in items {
            match item {
                Value::String(string) => {
                    required.push((
                        ctx.funding().copy_str(string)?,
                        ctx.funding().key::<F>(string)?,
                    ));
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
            .boxed(RequiredValidator { required, location })?)
    }
}

impl<F: Json> Validate<F> for RequiredValidator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(&self.required)?;
        for (name, key) in &self.required {
            source.string(name)?;
            source.key(key)?;
        }
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::slice::Iter<'_, F::PreparedKey>>(),
            std::mem::size_of::<std::slice::Iter<'_, String>>(),
            std::mem::size_of::<(&str, &F::PreparedKey)>(),
            std::mem::size_of::<ahash::AHasher>(),
            std::mem::size_of::<(bool, bool, usize)>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            if object.len() < self.required.len() {
                return false;
            }
            self.required
                .iter()
                .all(|(_, key)| object.get(key).is_some())
        } else {
            true
        }
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        required_errors::<F>(
            self.required.iter().map(|(name, key)| (name.as_str(), key)),
            &self.location,
            instance,
            location,
            tracker,
            ctx,
            None,
        )
    }
    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let _ = required_errors::<F>(
            self.required.iter().map(|(name, key)| (name.as_str(), key)),
            &self.location,
            instance,
            location,
            tracker,
            ctx,
            Some(errors),
        );
    }
}

pub(crate) struct SingleItemRequiredValidator<F: Json = SerdeJson> {
    value: String,
    key: F::PreparedKey,
    location: Location,
}

impl SingleItemRequiredValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        value: &'a str,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(SingleItemRequiredValidator {
            value: ctx.funding().copy_str(value)?,
            key: ctx.funding().key::<F>(value)?,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for SingleItemRequiredValidator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.string(&self.value)?;
        source.key(&self.key)?;
        source.location(&self.location)
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        required_errors::<F>(
            [(self.value.as_str(), &self.key)],
            &self.location,
            instance,
            location,
            tracker,
            ctx,
            None,
        )
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::slice::Iter<'_, F::PreparedKey>>(),
            std::mem::size_of::<std::slice::Iter<'_, String>>(),
            std::mem::size_of::<(&str, &F::PreparedKey)>(),
            std::mem::size_of::<ahash::AHasher>(),
            std::mem::size_of::<(bool, bool, usize)>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            if object.is_empty() {
                return false;
            }
            object.get(&self.key).is_some()
        } else {
            true
        }
    }
}

/// Specialized validator for exactly 2 required properties.
/// Uses fixed-size array and unrolled checks to avoid Vec/iterator overhead.
pub(crate) struct Required2Validator<F: Json = SerdeJson> {
    first: String,
    first_key: F::PreparedKey,
    second: String,
    second_key: F::PreparedKey,
    location: Location,
}

impl Required2Validator {
    #[inline]
    pub(crate) fn compile<F: Json>(
        ctx: &crate::compiler::Context<F>,
        first: String,
        second: String,
        location: Location,
    ) -> CompilationResult<'static, F> {
        Ok(ctx.funding().boxed(Required2Validator {
            first_key: ctx.funding().key::<F>(&first)?,
            second_key: ctx.funding().key::<F>(&second)?,
            first,
            second,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for Required2Validator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.string(&self.first)?;
        source.key(&self.first_key)?;
        source.string(&self.second)?;
        source.key(&self.second_key)?;
        source.location(&self.location)
    }

    #[inline]
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::slice::Iter<'_, F::PreparedKey>>(),
            std::mem::size_of::<std::slice::Iter<'_, String>>(),
            std::mem::size_of::<(&str, &F::PreparedKey)>(),
            std::mem::size_of::<ahash::AHasher>(),
            std::mem::size_of::<(bool, bool, usize)>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            object.len() >= 2
                && object.get(&self.first_key).is_some()
                && object.get(&self.second_key).is_some()
        } else {
            true
        }
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        required_errors::<F>(
            [
                (self.first.as_str(), &self.first_key),
                (self.second.as_str(), &self.second_key),
            ],
            &self.location,
            instance,
            location,
            tracker,
            ctx,
            None,
        )
    }

    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let _ = required_errors::<F>(
            [
                (self.first.as_str(), &self.first_key),
                (self.second.as_str(), &self.second_key),
            ],
            &self.location,
            instance,
            location,
            tracker,
            ctx,
            Some(errors),
        );
    }
}

/// Specialized validator for exactly 3 required properties.
/// Uses fixed-size fields and unrolled checks to avoid Vec/iterator overhead.
pub(crate) struct Required3Validator<F: Json = SerdeJson> {
    first: String,
    first_key: F::PreparedKey,
    second: String,
    second_key: F::PreparedKey,
    third: String,
    third_key: F::PreparedKey,
    location: Location,
}

impl Required3Validator {
    #[inline]
    pub(crate) fn compile<F: Json>(
        ctx: &crate::compiler::Context<F>,
        first: String,
        second: String,
        third: String,
        location: Location,
    ) -> CompilationResult<'static, F> {
        Ok(ctx.funding().boxed(Required3Validator {
            first_key: ctx.funding().key::<F>(&first)?,
            second_key: ctx.funding().key::<F>(&second)?,
            third_key: ctx.funding().key::<F>(&third)?,
            first,
            second,
            third,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for Required3Validator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.string(&self.first)?;
        source.key(&self.first_key)?;
        source.string(&self.second)?;
        source.key(&self.second_key)?;
        source.string(&self.third)?;
        source.key(&self.third_key)?;
        source.location(&self.location)
    }

    #[inline]
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::slice::Iter<'_, F::PreparedKey>>(),
            std::mem::size_of::<std::slice::Iter<'_, String>>(),
            std::mem::size_of::<(&str, &F::PreparedKey)>(),
            std::mem::size_of::<ahash::AHasher>(),
            std::mem::size_of::<(bool, bool, usize)>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            object.len() >= 3
                && object.get(&self.first_key).is_some()
                && object.get(&self.second_key).is_some()
                && object.get(&self.third_key).is_some()
        } else {
            true
        }
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        required_errors::<F>(
            [
                (self.first.as_str(), &self.first_key),
                (self.second.as_str(), &self.second_key),
                (self.third.as_str(), &self.third_key),
            ],
            &self.location,
            instance,
            location,
            tracker,
            ctx,
            None,
        )
    }

    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let _ = required_errors::<F>(
            [
                (self.first.as_str(), &self.first_key),
                (self.second.as_str(), &self.second_key),
                (self.third.as_str(), &self.third_key),
            ],
            &self.location,
            instance,
            location,
            tracker,
            ctx,
            Some(errors),
        );
    }
}

#[inline]
pub(crate) fn compile<'a, F: Json>(
    ctx: &compiler::Context<F>,
    parent: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    // Check if fused validators handle this case
    if let Value::Array(items) = schema {
        let has_properties = parent.contains_key("properties");
        let has_pattern_properties = parent.contains_key("patternProperties");
        let additional_props_false =
            matches!(parent.get("additionalProperties"), Some(Value::Bool(false)));

        // Case 1: properties + additionalProperties: false + required: [1 item], no patternProperties
        // Handled by AdditionalPropertiesNotEmptyFalseWithRequired1Validator
        if items.len() == 1 && additional_props_false && has_properties && !has_pattern_properties {
            return None;
        }

        // Case 2: properties + required: [2 items], no additionalProperties, no patternProperties
        // Handled by SmallPropertiesWithRequired2Validator — only when the map is below the
        // threshold; above it BigPropertiesValidator is used and does not include required checks.
        // Must NOT skip when additionalProperties is a schema object: properties::compile returns
        // None in that case (AdditionalPropertiesNotEmptyValidator takes over), so
        // SmallPropertiesWithRequired2Validator is never created — required would be silently dropped.
        let additional_props_is_schema =
            matches!(parent.get("additionalProperties"), Some(Value::Object(_)));
        let properties_below_threshold = parent
            .get("properties")
            .and_then(Value::as_object)
            .is_some_and(|m| m.len() < HASHMAP_THRESHOLD);
        if items.len() == 2
            && has_properties
            && properties_below_threshold
            && !additional_props_false
            && !additional_props_is_schema
            && !has_pattern_properties
        {
            return None;
        }
    }
    let location =
        crate::keywords::try_compile!(ctx.location().join_with_funding("required", ctx.funding()));
    compile_with_path(ctx, schema, location)
}

#[inline]
pub(crate) fn compile_with_path<'a, F: Json>(
    ctx: &compiler::Context<F>,
    schema: &'a Value,
    location: Location,
) -> Option<CompilationResult<'a, F>> {
    // IMPORTANT: If this function will ever return `None`, adjust `dependencies.rs` accordingly
    match schema {
        Value::Array(items) => match items.len() {
            1 => {
                let item = &items[0];
                if let Value::String(item) = item {
                    Some(SingleItemRequiredValidator::compile(ctx, item, location))
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
            }
            2 => {
                let (first, second) = (&items[0], &items[1]);
                match (first, second) {
                    (Value::String(first), Value::String(second)) => {
                        Some(Required2Validator::compile(
                            ctx,
                            crate::keywords::try_compile!(ctx.funding().copy_str(first)),
                            crate::keywords::try_compile!(ctx.funding().copy_str(second)),
                            location,
                        ))
                    }
                    (Value::String(_), other) | (other, _) => {
                        Some(Err(crate::keywords::try_compile!(
                            ValidationError::single_type_error_with_funding(
                                location.clone(),
                                location,
                                crate::keywords::try_compile!(Location::new_with_funding(
                                    ctx.funding()
                                )),
                                Cow::Borrowed(other),
                                JsonType::String,
                                ctx.funding()
                            )
                        )
                        .into()))
                    }
                }
            }
            3 => {
                let (first, second, third) = (&items[0], &items[1], &items[2]);
                match (first, second, third) {
                    (Value::String(first), Value::String(second), Value::String(third)) => {
                        Some(Required3Validator::compile(
                            ctx,
                            crate::keywords::try_compile!(ctx.funding().copy_str(first)),
                            crate::keywords::try_compile!(ctx.funding().copy_str(second)),
                            crate::keywords::try_compile!(ctx.funding().copy_str(third)),
                            location,
                        ))
                    }
                    (Value::String(_), Value::String(_), other)
                    | (Value::String(_), other, _)
                    | (other, _, _) => Some(Err(crate::keywords::try_compile!(
                        ValidationError::single_type_error_with_funding(
                            location.clone(),
                            location,
                            crate::keywords::try_compile!(Location::new_with_funding(
                                ctx.funding()
                            )),
                            Cow::Borrowed(other),
                            JsonType::String,
                            ctx.funding()
                        )
                    )
                    .into())),
                }
            }
            _ => Some(RequiredValidator::compile(ctx, items, location)),
        },
        _ => Some(Err(crate::keywords::try_compile!(
            ValidationError::single_type_error_with_funding(
                location.clone(),
                location,
                crate::keywords::try_compile!(Location::new_with_funding(ctx.funding())),
                Cow::Borrowed(schema),
                JsonType::Array,
                ctx.funding()
            )
        )
        .into())),
    }
}

#[cfg(test)]
mod tests {
    use super::HASHMAP_THRESHOLD;
    use crate::tests_util;
    use serde_json::{json, Value};
    use test_case::test_case;

    #[test_case(&json!({"required": ["a"]}), &json!({}), "/required")]
    #[test_case(&json!({"required": ["a", "b"]}), &json!({}), "/required")]
    #[test_case(&json!({"required": ["a", "b", "c"]}), &json!({}), "/required")]
    fn location(schema: &Value, instance: &Value, expected: &str) {
        tests_util::assert_schema_location(schema, instance, expected);
    }

    // Required2Validator tests
    #[test_case(&json!({"a": 1, "b": 2}), true)]
    #[test_case(&json!({"a": 1, "b": 2, "c": 3}), true)]
    #[test_case(&json!({"a": 1}), false)]
    #[test_case(&json!({"b": 2}), false)]
    #[test_case(&json!({}), false)]
    #[test_case(&json!([1, 2]), true)] // Non-object passes
    fn required_2(instance: &Value, expected: bool) {
        let schema = json!({"required": ["a", "b"]});
        let validator = crate::validator_for(&schema).unwrap();
        assert_eq!(validator.is_valid(instance), expected);
    }

    // Required3Validator tests
    #[test_case(&json!({"a": 1, "b": 2, "c": 3}), true)]
    #[test_case(&json!({"a": 1, "b": 2, "c": 3, "d": 4}), true)]
    #[test_case(&json!({"a": 1, "b": 2}), false)]
    #[test_case(&json!({"a": 1, "c": 3}), false)]
    #[test_case(&json!({"b": 2, "c": 3}), false)]
    #[test_case(&json!({}), false)]
    #[test_case(&json!("string"), true)] // Non-object passes
    fn required_3(instance: &Value, expected: bool) {
        let schema = json!({"required": ["a", "b", "c"]});
        let validator = crate::validator_for(&schema).unwrap();
        assert_eq!(validator.is_valid(instance), expected);
    }

    #[test]
    fn required_2_iter_errors() {
        let schema = json!({"required": ["a", "b"]});
        let validator = crate::validator_for(&schema).unwrap();

        // Missing both
        let instance = json!({});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert_eq!(errors.len(), 2);

        // Missing one
        let instance = json!({"a": 1});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert_eq!(errors.len(), 1);

        // All present
        let instance = json!({"a": 1, "b": 2});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn required_3_iter_errors() {
        let schema = json!({"required": ["a", "b", "c"]});
        let validator = crate::validator_for(&schema).unwrap();

        // Missing all
        let instance = json!({});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert_eq!(errors.len(), 3);

        // Missing two
        let instance = json!({"a": 1});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert_eq!(errors.len(), 2);

        // Missing one
        let instance = json!({"a": 1, "b": 2});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert_eq!(errors.len(), 1);

        // All present
        let instance = json!({"a": 1, "b": 2, "c": 3});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert!(errors.is_empty());
    }

    // When `additionalProperties` is a schema object, properties::compile returns None so
    // SmallPropertiesWithRequired2Validator is never created; required must still be enforced.
    #[test]
    fn required_2_enforced_with_additional_properties_schema() {
        let schema = json!({
            "properties": {
                "type": {"type": "string"},
                "linkedServiceName": {"type": "object"},
            },
            "additionalProperties": {"type": "object"},
            "required": ["type", "linkedServiceName"],
        });
        let validator = crate::validator_for(&schema).unwrap();

        assert!(!validator.is_valid(&json!({"type": "x"})));
        assert!(!validator.is_valid(&json!({"linkedServiceName": {}})));
        assert!(!validator.is_valid(&json!({})));
        assert!(validator.is_valid(&json!({"type": "x", "linkedServiceName": {}})));
    }

    // When `properties` has >= HASHMAP_THRESHOLD entries the fused SmallPropertiesWithRequired2
    // is not used; the standalone required validator must still fire.
    #[test]
    fn required_2_enforced_with_large_properties_map() {
        let mut props = serde_json::Map::new();
        for i in 0..HASHMAP_THRESHOLD {
            props.insert(format!("prop{i}"), json!({"type": "string"}));
        }
        props.insert("vmSize".to_string(), json!({"type": "string"}));
        props.insert("count".to_string(), json!({"type": "integer"}));

        let schema = json!({
            "properties": Value::Object(props),
            "required": ["vmSize", "count"]
        });
        let validator = crate::validator_for(&schema).unwrap();

        assert!(!validator.is_valid(&json!({"count": 1})));
        assert!(!validator.is_valid(&json!({"vmSize": "x"})));
        assert!(validator.is_valid(&json!({"vmSize": "x", "count": 1})));

        let instance = json!({"count": 1});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert_eq!(errors.len(), 1);
    }
}
