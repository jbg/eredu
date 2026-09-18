//! # Description
//! This module contains various validators for the `additionalProperties` keyword.
//!
//! The goal here is to compute intersections with another keywords affecting properties validation:
//!   - `properties`
//!   - `patternProperties`
//!
//! Each valid combination of these keywords has a validator here.
use crate::{
    compiler,
    error::ValidationError,
    evaluation::{Annotations, ErrorDescription, EvaluationNode},
    keywords::CompilationResult,
    node::SchemaNode,
    options::PatternEngineOptions,
    paths::{LazyLocation, Location, RefTracker},
    properties::{
        are_properties_valid, compile_big_map, compile_dynamic_prop_map_validator,
        compile_fancy_regex_patterns, compile_regex_patterns, compile_small_map, BigValidatorsMap,
        CompiledPattern, PropertiesValidatorsMap, SmallValidatorsMap, HASHMAP_THRESHOLD,
    },
    regex::RegexEngine,
    types::JsonType,
    validator::{EvaluationResult, Validate, ValidationContext},
    Json, Node, Object, SerdeJson,
};
type PatternIndices = hashbrown::HashMap<String, Vec<usize>, ahash::RandomState>;
use referencing::Uri;
use serde_json::{Map, Value};
use std::{borrow::Cow, sync::Arc};

/// # Schema example
///
/// ```json
/// {
///     "additionalProperties": {"type": "integer"},
/// }
/// ```
///
/// # Valid value
///
/// ```json
/// {
///     "bar": 6
/// }
/// ```
pub(crate) struct AdditionalPropertiesValidator<F: Json = SerdeJson> {
    node: SchemaNode<F>,
}
impl AdditionalPropertiesValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        schema: &'a Value,
        ctx: &compiler::Context<F>,
    ) -> CompilationResult<'a, F> {
        let ctx = ctx.new_at_location("additionalProperties")?;
        Ok(ctx.funding().boxed(AdditionalPropertiesValidator {
            node: compiler::compile(&ctx, ctx.as_resource_ref(schema))?,
        })?)
    }
}
impl<F: Json> Validate<F> for AdditionalPropertiesValidator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.node(&self.node)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<(&str, &crate::node::SchemaNode<F>)>(),
            std::mem::size_of::<(usize, bool, bool)>(),
            std::mem::size_of::<ahash::AHasher>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            object
                .members()
                .all(|(_, value)| self.node.is_valid(&value, ctx))
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
        if let Some(object) = instance.as_object() {
            for (name, value) in object.members() {
                self.node
                    .validate(&value, &location.push(name.as_ref()), tracker, ctx)?;
            }
        }
        Ok(())
    }

    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let Some(object) = instance.as_object() else {
            return;
        };
        for (name, value) in object.members() {
            self.node
                .collect_errors(&value, &location.push(name.as_ref()), tracker, ctx, errors);
        }
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if let Some(object) = instance.as_object() {
            let mut children = Vec::with_capacity(object.len());
            for (name, value) in object.members() {
                children.push(self.node.evaluate_instance(
                    &value,
                    &location.push(name.as_ref()),
                    tracker,
                    ctx,
                ));
            }
            let mut result = EvaluationResult::from_children(children);
            let annotated_props = object
                .members()
                .map(|(name, _)| serde_json::Value::String(name.as_ref().to_owned()))
                .collect();
            result.annotate(Annotations::new(serde_json::Value::Array(annotated_props)));
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

/// # Schema example
///
/// ```json
/// {
///     "additionalProperties": false
/// }
/// ```
///
/// # Valid value
///
/// ```json
/// {}
/// ```
pub(crate) struct AdditionalPropertiesFalseValidator {
    location: Location,
}
impl AdditionalPropertiesFalseValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx
            .funding()
            .boxed(AdditionalPropertiesFalseValidator { location })?)
    }
}
impl<F: Json> Validate<F> for AdditionalPropertiesFalseValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<(&str, &crate::node::SchemaNode<F>)>(),
            std::mem::size_of::<(usize, bool, bool)>(),
            std::mem::size_of::<ahash::AHasher>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)?
            .checked_add(std::mem::size_of::<Vec<String>>())
            .ok_or(crate::validator::workspace::Error::Overflow)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            object.members().next().is_none()
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
        if let Some(object) = instance.as_object() {
            if let Some((_, value)) = object.members().next() {
                return ctx.diagnostic::<F>(&value, location, tracker, &self.location, |_| {
                    Ok(crate::error::ValidationErrorKind::FalseSchema)
                });
            }
        }
        Ok(())
    }
}

/// # Schema example
///
/// ```json
/// {
///     "additionalProperties": false,
///     "properties": {
///         "foo": {"type": "string"}
///     },
/// }
/// ```
///
/// # Valid value
///
/// ```json
/// {
///     "foo": "bar",
/// }
/// ```
pub(crate) struct AdditionalPropertiesNotEmptyFalseValidator<M> {
    properties: M,
    location: Location,
}
impl<F: Json> AdditionalPropertiesNotEmptyFalseValidator<SmallValidatorsMap<F>> {
    #[inline]
    pub(crate) fn compile<'a>(
        map: &'a Map<String, Value>,
        ctx: &compiler::Context<F>,
    ) -> CompilationResult<'a, F> {
        Ok(ctx
            .funding()
            .boxed(AdditionalPropertiesNotEmptyFalseValidator {
                properties: compile_small_map(ctx, map)?,
                location: ctx
                    .location()
                    .join_with_funding("additionalProperties", ctx.funding())?,
            })?)
    }
}
impl<F: Json> AdditionalPropertiesNotEmptyFalseValidator<BigValidatorsMap<F>> {
    #[inline]
    pub(crate) fn compile<'a>(
        map: &'a Map<String, Value>,
        ctx: &compiler::Context<F>,
    ) -> CompilationResult<'a, F> {
        Ok(ctx
            .funding()
            .boxed(AdditionalPropertiesNotEmptyFalseValidator {
                properties: compile_big_map(ctx, map)?,
                location: ctx
                    .location()
                    .join_with_funding("additionalProperties", ctx.funding())?,
            })?)
    }
}
impl<F: Json, M: PropertiesValidatorsMap<F>> Validate<F>
    for AdditionalPropertiesNotEmptyFalseValidator<M>
{
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        self.properties.original_source(source)?;
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            self.properties
                .original_lookup_controls()
                .ok_or(crate::validator::workspace::Error::Overflow)?,
            std::mem::size_of::<(&str, &crate::node::SchemaNode<F>)>(),
            std::mem::size_of::<(usize, bool, bool)>(),
            std::mem::size_of::<ahash::AHasher>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)?
            .checked_add(std::mem::size_of::<Vec<String>>())
            .ok_or(crate::validator::workspace::Error::Overflow)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            are_properties_valid(&self.properties, &object, ctx, |_, _| false)
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
        if let Some(object) = instance.as_object() {
            for (property, value) in object.members() {
                if let Some((name, node)) = self.properties.get_key_validator(property.as_ref()) {
                    node.validate(&value, &location.push(name), tracker, ctx)?;
                } else {
                    return ctx.diagnostic::<F>(
                        instance,
                        location,
                        tracker,
                        &self.location,
                        |funding| {
                            Ok(crate::error::ValidationErrorKind::AdditionalProperties {
                                unexpected: funding.copy_vec(&[property.as_ref()], |name| {
                                    funding.copy_str(name)
                                })?,
                            })
                        },
                    );
                }
            }
        }
        Ok(())
    }

    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let Some(object) = instance.as_object() else {
            return;
        };
        let mut unexpected = vec![];
        for (property, value) in object.members() {
            if let Some((name, node)) = self.properties.get_key_validator(property.as_ref()) {
                node.collect_errors(&value, &location.push(name), tracker, ctx, errors);
            } else {
                let Some(name) = ctx.produce(|funding| funding.copy_str(property.as_ref())) else {
                    return;
                };
                if !ctx.workspace.push(&mut unexpected, name) {
                    return;
                }
            }
        }
        if !unexpected.is_empty() {
            if let Err(error) =
                ctx.diagnostic::<F>(instance, location, tracker, &self.location, |_| {
                    Ok(crate::error::ValidationErrorKind::AdditionalProperties { unexpected })
                })
            {
                if !ctx.workspace.push(errors, error) {
                    return;
                }
            }
        }
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if let Some(object) = instance.as_object() {
            let mut unexpected = Vec::with_capacity(object.len());
            let mut children = Vec::with_capacity(object.len());
            for (property, value) in object.members() {
                if let Some((_name, node)) = self.properties.get_key_validator(property.as_ref()) {
                    children.push(node.evaluate_instance(
                        &value,
                        &location.push(property.as_ref()),
                        tracker,
                        ctx,
                    ));
                } else {
                    unexpected.push(property.as_ref().to_owned());
                }
            }
            let mut result = EvaluationResult::from_children(children);
            if !unexpected.is_empty() {
                let eval_path = crate::paths::capture_evaluation_path(tracker, &self.location);
                result.mark_errored(ErrorDescription::from_validation_error(
                    &ValidationError::additional_properties(
                        self.location.clone(),
                        eval_path,
                        location.into(),
                        instance.to_value(),
                        unexpected,
                    ),
                ));
            }
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

/// Fused validator for properties + additionalProperties: false + required
/// Eliminates separate required validation pass by tracking the required property during iteration.
pub(crate) struct AdditionalPropertiesNotEmptyFalseWithRequired1Validator<M> {
    properties: M,
    required: String,
    location: Location,
    required_location: Location,
}
impl<F: Json> AdditionalPropertiesNotEmptyFalseWithRequired1Validator<SmallValidatorsMap<F>> {
    #[inline]
    pub(crate) fn compile<'a>(
        map: &'a Map<String, Value>,
        ctx: &compiler::Context<F>,
        required: String,
    ) -> CompilationResult<'a, F> {
        Ok(ctx
            .funding()
            .boxed(AdditionalPropertiesNotEmptyFalseWithRequired1Validator {
                properties: compile_small_map(ctx, map)?,
                required,
                location: ctx
                    .location()
                    .join_with_funding("additionalProperties", ctx.funding())?,
                required_location: ctx
                    .location()
                    .join_with_funding("required", ctx.funding())?,
            })?)
    }
}
impl<F: Json> AdditionalPropertiesNotEmptyFalseWithRequired1Validator<BigValidatorsMap<F>> {
    #[inline]
    pub(crate) fn compile<'a>(
        map: &'a Map<String, Value>,
        ctx: &compiler::Context<F>,
        required: String,
    ) -> CompilationResult<'a, F> {
        Ok(ctx
            .funding()
            .boxed(AdditionalPropertiesNotEmptyFalseWithRequired1Validator {
                properties: compile_big_map(ctx, map)?,
                required,
                location: ctx
                    .location()
                    .join_with_funding("additionalProperties", ctx.funding())?,
                required_location: ctx
                    .location()
                    .join_with_funding("required", ctx.funding())?,
            })?)
    }
}
impl<F: Json, M: PropertiesValidatorsMap<F>> Validate<F>
    for AdditionalPropertiesNotEmptyFalseWithRequired1Validator<M>
{
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        self.properties.original_source(source)?;
        source.string(&self.required)?;
        source.location(&self.required_location)?;
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            self.properties
                .original_lookup_controls()
                .ok_or(crate::validator::workspace::Error::Overflow)?,
            std::mem::size_of::<(&str, &crate::node::SchemaNode<F>)>(),
            std::mem::size_of::<(usize, bool, bool)>(),
            std::mem::size_of::<ahash::AHasher>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)?
            .checked_add(std::mem::size_of::<Vec<String>>())
            .ok_or(crate::validator::workspace::Error::Overflow)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            if object.is_empty() {
                return false;
            }
            let mut found_required = false;
            for (property, value) in object.members() {
                if let Some(node) = self.properties.get_validator(property.as_ref()) {
                    if !node.is_valid(&value, ctx) {
                        return false;
                    }
                    if property.as_ref() == self.required.as_str() {
                        found_required = true;
                    }
                } else {
                    return false;
                }
            }
            found_required
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
        if let Some(object) = instance.as_object() {
            let mut found_required = false;
            for (property, value) in object.members() {
                if let Some((name, node)) = self.properties.get_key_validator(property.as_ref()) {
                    node.validate(&value, &location.push(name), tracker, ctx)?;
                    if property.as_ref() == self.required.as_str() {
                        found_required = true;
                    }
                } else {
                    return ctx.diagnostic::<F>(
                        instance,
                        location,
                        tracker,
                        &self.location,
                        |funding| {
                            Ok(crate::error::ValidationErrorKind::AdditionalProperties {
                                unexpected: funding.copy_vec(&[property.as_ref()], |name| {
                                    funding.copy_str(name)
                                })?,
                            })
                        },
                    );
                }
            }
            if !found_required {
                return ctx.diagnostic::<F>(
                    instance,
                    location,
                    tracker,
                    &self.required_location,
                    |funding| {
                        Ok(crate::error::ValidationErrorKind::Required {
                            property: Value::String(funding.copy_str(&self.required)?),
                        })
                    },
                );
            }
        }
        Ok(())
    }

    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let Some(object) = instance.as_object() else {
            return;
        };
        let mut unexpected = vec![];
        let mut found_required = false;
        for (property, value) in object.members() {
            if let Some((name, node)) = self.properties.get_key_validator(property.as_ref()) {
                node.collect_errors(&value, &location.push(name), tracker, ctx, errors);
                if property.as_ref() == self.required.as_str() {
                    found_required = true;
                }
            } else {
                let Some(name) = ctx.produce(|funding| funding.copy_str(property.as_ref())) else {
                    return;
                };
                if !ctx.workspace.push(&mut unexpected, name) {
                    return;
                }
            }
        }
        if !unexpected.is_empty() {
            if let Err(error) =
                ctx.diagnostic::<F>(instance, location, tracker, &self.location, |_| {
                    Ok(crate::error::ValidationErrorKind::AdditionalProperties { unexpected })
                })
            {
                if !ctx.workspace.push(errors, error) {
                    return;
                }
            }
        }
        if !found_required {
            if let Err(error) = ctx.diagnostic::<F>(
                instance,
                location,
                tracker,
                &self.required_location,
                |funding| {
                    Ok(crate::error::ValidationErrorKind::Required {
                        property: Value::String(funding.copy_str(&self.required)?),
                    })
                },
            ) {
                ctx.workspace.push(errors, error);
            }
        }
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if let Some(object) = instance.as_object() {
            let mut unexpected = Vec::with_capacity(object.len());
            let mut children = Vec::with_capacity(object.len());
            let mut found_required = false;
            for (property, value) in object.members() {
                if let Some((_name, node)) = self.properties.get_key_validator(property.as_ref()) {
                    children.push(node.evaluate_instance(
                        &value,
                        &location.push(property.as_ref()),
                        tracker,
                        ctx,
                    ));
                    if property.as_ref() == self.required.as_str() {
                        found_required = true;
                    }
                } else {
                    unexpected.push(property.as_ref().to_owned());
                }
            }
            let mut result = EvaluationResult::from_children(children);
            if !unexpected.is_empty() {
                let eval_path = crate::paths::capture_evaluation_path(tracker, &self.location);
                result.mark_errored(ErrorDescription::from_validation_error(
                    &ValidationError::additional_properties(
                        self.location.clone(),
                        eval_path,
                        location.into(),
                        instance.to_value(),
                        unexpected,
                    ),
                ));
            }
            if !found_required {
                let eval_path =
                    crate::paths::capture_evaluation_path(tracker, &self.required_location);
                result.mark_errored(ErrorDescription::from_validation_error(
                    &ValidationError::required(
                        self.required_location.clone(),
                        eval_path,
                        location.into(),
                        instance.to_value(),
                        Value::String(self.required.clone()),
                    ),
                ));
            }
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

/// # Schema example
///
/// ```json
/// {
///     "additionalProperties": {"type": "integer"},
///     "properties": {
///         "foo": {"type": "string"}
///     }
/// }
/// ```
///
/// # Valid value
///
/// ```json
/// {
///     "foo": "bar",
///     "bar": 6
/// }
/// ```
pub(crate) struct AdditionalPropertiesNotEmptyValidator<M, F: Json = SerdeJson> {
    node: SchemaNode<F>,
    properties: M,
}
impl<F: Json> AdditionalPropertiesNotEmptyValidator<SmallValidatorsMap<F>> {
    #[inline]
    pub(crate) fn compile<'a>(
        map: &'a Map<String, Value>,
        ctx: &compiler::Context<F>,
        schema: &'a Value,
    ) -> CompilationResult<'a, F> {
        let kctx = ctx.new_at_location("additionalProperties")?;
        Ok(ctx.funding().boxed(AdditionalPropertiesNotEmptyValidator {
            properties: compile_small_map(ctx, map)?,
            node: compiler::compile(&kctx, kctx.as_resource_ref(schema))?,
        })?)
    }
}
impl<F: Json> AdditionalPropertiesNotEmptyValidator<BigValidatorsMap<F>> {
    #[inline]
    pub(crate) fn compile<'a>(
        map: &'a Map<String, Value>,
        ctx: &compiler::Context<F>,
        schema: &'a Value,
    ) -> CompilationResult<'a, F> {
        let kctx = ctx.new_at_location("additionalProperties")?;
        Ok(ctx.funding().boxed(AdditionalPropertiesNotEmptyValidator {
            properties: compile_big_map(ctx, map)?,
            node: compiler::compile(&kctx, kctx.as_resource_ref(schema))?,
        })?)
    }
}
impl<F: Json, M: PropertiesValidatorsMap<F>> Validate<F>
    for AdditionalPropertiesNotEmptyValidator<M, F>
{
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        self.properties.original_source(source)?;
        source.node(&self.node)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            self.properties
                .original_lookup_controls()
                .ok_or(crate::validator::workspace::Error::Overflow)?,
            std::mem::size_of::<(&str, &crate::node::SchemaNode<F>)>(),
            std::mem::size_of::<(usize, bool, bool)>(),
            std::mem::size_of::<ahash::AHasher>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            are_properties_valid(&self.properties, &object, ctx, |instance, ctx| {
                self.node.is_valid(instance, ctx)
            })
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
        if let Some(object) = instance.as_object() {
            for (property, value) in object.members() {
                let property_location = location.push(property.as_ref());
                if let Some(validator) = self.properties.get_validator(property.as_ref()) {
                    validator.validate(&value, &property_location, tracker, ctx)?;
                } else {
                    self.node
                        .validate(&value, &property_location, tracker, ctx)?;
                }
            }
        }
        Ok(())
    }

    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let Some(object) = instance.as_object() else {
            return;
        };
        for (property, value) in object.members() {
            if let Some((name, property_validators)) =
                self.properties.get_key_validator(property.as_ref())
            {
                property_validators.collect_errors(
                    &value,
                    &location.push(name),
                    tracker,
                    ctx,
                    errors,
                );
            } else {
                self.node.collect_errors(
                    &value,
                    &location.push(property.as_ref()),
                    tracker,
                    ctx,
                    errors,
                );
            }
        }
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if let Some(object) = instance.as_object() {
            let mut matched_propnames = Vec::with_capacity(object.len());
            let mut children = Vec::with_capacity(object.len());
            for (property, value) in object.members() {
                let path = location.push(property.as_ref());
                if let Some((_name, property_validators)) =
                    self.properties.get_key_validator(property.as_ref())
                {
                    children
                        .push(property_validators.evaluate_instance(&value, &path, tracker, ctx));
                } else {
                    children.push(self.node.evaluate_instance(&value, &path, tracker, ctx));
                    matched_propnames.push(property.as_ref().to_owned());
                }
            }
            let mut result = EvaluationResult::from_children(children);
            if !matched_propnames.is_empty() {
                result.annotate(Annotations::new(Value::from(matched_propnames)));
            }
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

/// # Schema example
///
/// ```json
/// {
///     "additionalProperties": {"type": "integer"},
///     "patternProperties": {
///         "^x-": {"type": "integer", "minimum": 5},
///         "-x$": {"type": "integer", "maximum": 10}
///     }
/// }
/// ```
///
/// # Valid value
///
/// ```json
/// {
///     "x-foo": 6,
///     "foo-x": 7,
///     "bar": 8
/// }
/// ```
pub(crate) struct AdditionalPropertiesWithPatternsValidator<R, F: Json = SerdeJson> {
    node: SchemaNode<F>,
    patterns: Vec<(R, SchemaNode<F>)>,
    /// We need this because `compiler::compile` uses the additionalProperties keyword to compile
    /// this validator. That means that the schema node which contains this validator has
    /// "additionalProperties" as it's path. However, we need to produce annotations which have the
    /// patternProperties keyword as their path so we store the paths here.
    pattern_keyword_path: Location,
    pattern_keyword_absolute_location: Option<Arc<Uri<String>>>,
}

impl<F: Json, R: RegexEngine> Validate<F> for AdditionalPropertiesWithPatternsValidator<R, F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(&self.patterns)?;
        for (pattern, node) in &self.patterns {
            pattern.original_source(source)?;
            source.node(node)?;
        }
        source.node(&self.node)?;
        source.location(&self.pattern_keyword_path)?;
        source.uri(&self.pattern_keyword_absolute_location)?;
        Ok(())
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        let regex = self
            .patterns
            .iter()
            .try_fold(0usize, |peak, (pattern, _)| {
                pattern.original_controls().map(|bytes| peak.max(bytes))
            })?;
        crate::validator::workspace::body_controls::<F, Self>(&[
            regex,
            std::mem::size_of::<(
                &str,
                bool,
                usize,
                &SchemaNode<F>,
                std::slice::Iter<'_, (R, SchemaNode<F>)>,
                std::slice::Iter<'_, usize>,
                ahash::AHasher,
            )>(),
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)?
            .checked_add(std::mem::size_of::<Vec<String>>())
            .ok_or(crate::validator::workspace::Error::Overflow)
    }

    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            for (property, value) in object.members() {
                let mut has_match = false;
                for (re, node) in &self.patterns {
                    if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                        has_match = true;
                        if !node.is_valid(&value, ctx) {
                            return false;
                        }
                    }
                }
                if !has_match && !self.node.is_valid(&value, ctx) {
                    return false;
                }
            }
        }
        true
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if let Some(object) = instance.as_object() {
            for (property, value) in object.members() {
                let property_location = location.push(property.as_ref());
                let mut has_match = false;
                for (re, node) in &self.patterns {
                    if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                        has_match = true;
                        node.validate(&value, &property_location, tracker, ctx)?;
                    }
                }
                if !has_match {
                    self.node
                        .validate(&value, &property_location, tracker, ctx)?;
                }
            }
        }
        Ok(())
    }

    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let Some(object) = instance.as_object() else {
            return;
        };
        for (property, value) in object.members() {
            let mut has_match = false;
            for (re, node) in &self.patterns {
                if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                    has_match = true;
                    node.collect_errors(
                        &value,
                        &location.push(property.as_ref()),
                        tracker,
                        ctx,
                        errors,
                    );
                }
            }
            if !has_match {
                self.node.collect_errors(
                    &value,
                    &location.push(property.as_ref()),
                    tracker,
                    ctx,
                    errors,
                );
            }
        }
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if let Some(object) = instance.as_object() {
            let mut pattern_matched_propnames = Vec::with_capacity(object.len());
            let mut additional_matched_propnames = Vec::with_capacity(object.len());
            let mut children = Vec::with_capacity(object.len());
            for (property, value) in object.members() {
                let path = location.push(property.as_ref());
                let mut has_match = false;
                for (pattern, node) in &self.patterns {
                    if pattern.is_match(property.as_ref(), ctx).unwrap_or(false) {
                        has_match = true;
                        pattern_matched_propnames.push(property.as_ref().to_owned());
                        children.push(node.evaluate_instance(&value, &path, tracker, ctx));
                    }
                }
                if !has_match {
                    additional_matched_propnames.push(property.as_ref().to_owned());
                    children.push(self.node.evaluate_instance(&value, &path, tracker, ctx));
                }
            }
            if !pattern_matched_propnames.is_empty() {
                let annotation = Annotations::new(Value::from(pattern_matched_propnames));
                let schema_location = crate::evaluation::format_schema_location(
                    &self.pattern_keyword_path,
                    self.pattern_keyword_absolute_location.as_ref(),
                );
                children.push(EvaluationNode::valid(
                    self.pattern_keyword_path.clone(),
                    self.pattern_keyword_absolute_location.clone(),
                    schema_location,
                    location.into(),
                    Some(annotation),
                    Vec::new(),
                ));
            }
            let mut result = EvaluationResult::from_children(children);
            if !additional_matched_propnames.is_empty() {
                result.annotate(Annotations::new(Value::from(additional_matched_propnames)));
            }
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

/// # Schema example
///
/// ```json
/// {
///     "additionalProperties": false,
///     "patternProperties": {
///         "^x-": {"type": "integer", "minimum": 5},
///         "-x$": {"type": "integer", "maximum": 10}
///     }
/// }
/// ```
///
/// # Valid value
///
/// ```json
/// {
///     "x-bar": 6,
///     "spam-x": 7,
///     "x-baz-x": 8,
/// }
/// ```
pub(crate) struct AdditionalPropertiesWithPatternsFalseValidator<R, F: Json = SerdeJson> {
    patterns: Vec<(R, SchemaNode<F>)>,
    location: Location,
    pattern_keyword_path: Location,
    pattern_keyword_absolute_location: Option<Arc<Uri<String>>>,
}

impl<F: Json, R: RegexEngine> Validate<F> for AdditionalPropertiesWithPatternsFalseValidator<R, F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(&self.patterns)?;
        for (pattern, node) in &self.patterns {
            pattern.original_source(source)?;
            source.node(node)?;
        }
        source.location(&self.location)?;
        source.location(&self.pattern_keyword_path)?;
        source.uri(&self.pattern_keyword_absolute_location)?;
        Ok(())
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        let regex = self
            .patterns
            .iter()
            .try_fold(0usize, |peak, (pattern, _)| {
                pattern.original_controls().map(|bytes| peak.max(bytes))
            })?;
        crate::validator::workspace::body_controls::<F, Self>(&[
            regex,
            std::mem::size_of::<(
                &str,
                bool,
                usize,
                &SchemaNode<F>,
                std::slice::Iter<'_, (R, SchemaNode<F>)>,
                std::slice::Iter<'_, usize>,
                ahash::AHasher,
            )>(),
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)?
            .checked_add(std::mem::size_of::<Vec<String>>())
            .ok_or(crate::validator::workspace::Error::Overflow)
    }

    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            for (property, value) in object.members() {
                let mut has_match = false;
                for (re, node) in &self.patterns {
                    if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                        has_match = true;
                        if !node.is_valid(&value, ctx) {
                            return false;
                        }
                    }
                }
                if !has_match {
                    return false;
                }
            }
        }
        true
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if let Some(object) = instance.as_object() {
            for (property, value) in object.members() {
                let property_location = location.push(property.as_ref());
                let mut has_match = false;
                for (re, node) in &self.patterns {
                    if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                        has_match = true;
                        node.validate(&value, &property_location, tracker, ctx)?;
                    }
                }
                if !has_match {
                    return ctx.diagnostic::<F>(
                        instance,
                        location,
                        tracker,
                        &self.location,
                        |funding| {
                            Ok(crate::error::ValidationErrorKind::AdditionalProperties {
                                unexpected: funding.copy_vec(&[property.as_ref()], |name| {
                                    funding.copy_str(name)
                                })?,
                            })
                        },
                    );
                }
            }
        }
        Ok(())
    }

    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let Some(object) = instance.as_object() else {
            return;
        };
        let mut unexpected = vec![];
        for (property, value) in object.members() {
            let mut has_match = false;
            for (re, node) in &self.patterns {
                if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                    has_match = true;
                    node.collect_errors(
                        &value,
                        &location.push(property.as_ref()),
                        tracker,
                        ctx,
                        errors,
                    );
                }
            }
            if !has_match {
                let Some(name) = ctx.produce(|funding| funding.copy_str(property.as_ref())) else {
                    return;
                };
                if !ctx.workspace.push(&mut unexpected, name) {
                    return;
                }
            }
        }
        if !unexpected.is_empty() {
            if let Err(error) =
                ctx.diagnostic::<F>(instance, location, tracker, &self.location, |_| {
                    Ok(crate::error::ValidationErrorKind::AdditionalProperties { unexpected })
                })
            {
                if !ctx.workspace.push(errors, error) {
                    return;
                }
            }
        }
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if let Some(object) = instance.as_object() {
            let mut unexpected = Vec::with_capacity(object.len());
            let mut pattern_matched_props = Vec::with_capacity(object.len());
            let mut children = Vec::with_capacity(object.len());
            for (property, value) in object.members() {
                let path = location.push(property.as_ref());
                let mut has_match = false;
                for (pattern, node) in &self.patterns {
                    if pattern.is_match(property.as_ref(), ctx).unwrap_or(false) {
                        has_match = true;
                        pattern_matched_props.push(property.as_ref().to_owned());
                        children.push(node.evaluate_instance(&value, &path, tracker, ctx));
                    }
                }
                if !has_match {
                    unexpected.push(property.as_ref().to_owned());
                }
            }
            if !pattern_matched_props.is_empty() {
                let annotation = Annotations::new(Value::from(pattern_matched_props));
                let schema_location = crate::evaluation::format_schema_location(
                    &self.pattern_keyword_path,
                    self.pattern_keyword_absolute_location.as_ref(),
                );
                children.push(EvaluationNode::valid(
                    self.pattern_keyword_path.clone(),
                    self.pattern_keyword_absolute_location.clone(),
                    schema_location,
                    location.into(),
                    Some(annotation),
                    Vec::new(),
                ));
            }
            let mut result = EvaluationResult::from_children(children);
            if !unexpected.is_empty() {
                let eval_path = crate::paths::capture_evaluation_path(tracker, &self.location);
                result.mark_errored(ErrorDescription::from_validation_error(
                    &ValidationError::additional_properties(
                        self.location.clone(),
                        eval_path,
                        location.into(),
                        instance.to_value(),
                        unexpected,
                    ),
                ));
            }
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

/// # Schema example
///
/// ```json
/// {
///     "additionalProperties": {"type": "integer"},
///     "properties": {
///         "foo": {"type": "string"}
///     },
///     "patternProperties": {
///         "^x-": {"type": "integer", "minimum": 5},
///         "-x$": {"type": "integer", "maximum": 10}
///     }
/// }
/// ```
///
/// # Valid value
///
/// ```json
/// {
///     "foo": "a",
///     "x-spam": 6,
///     "spam-x": 7,
///     "x-spam-x": 8,
///     "bar": 42
/// }
/// ```
pub(crate) struct AdditionalPropertiesWithPatternsNotEmptyValidator<M, R, F: Json = SerdeJson> {
    node: SchemaNode<F>,
    properties: M,
    patterns: Vec<(R, SchemaNode<F>)>,
    /// Pre-computed pattern indices for properties defined in `properties`.
    /// Maps property name -> indices into `patterns` Vec for patterns that match.
    /// Eliminates regex matching at validation time for known properties.
    property_pattern_indices: PatternIndices,
}

impl<F: Json, M: PropertiesValidatorsMap<F>, R: RegexEngine> Validate<F>
    for AdditionalPropertiesWithPatternsNotEmptyValidator<M, R, F>
{
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(&self.patterns)?;
        for (pattern, node) in &self.patterns {
            pattern.original_source(source)?;
            source.node(node)?;
        }
        self.properties.original_source(source)?;
        source.add(self.property_pattern_indices.allocation_size())?;
        for (name, indices) in &self.property_pattern_indices {
            source.string(name)?;
            source.vector(indices)?;
        }
        source.node(&self.node)?;
        Ok(())
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        let regex = self
            .patterns
            .iter()
            .try_fold(0usize, |peak, (pattern, _)| {
                pattern.original_controls().map(|bytes| peak.max(bytes))
            })?;
        crate::validator::workspace::body_controls::<F, Self>(&[
            regex,
            std::mem::size_of::<(
                &str,
                bool,
                usize,
                &SchemaNode<F>,
                std::slice::Iter<'_, (R, SchemaNode<F>)>,
                std::slice::Iter<'_, usize>,
                ahash::AHasher,
            )>(),
            self.properties
                .original_lookup_controls()
                .ok_or(crate::validator::workspace::Error::Overflow)?,
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)?
            .checked_add(std::mem::size_of::<Vec<String>>())
            .ok_or(crate::validator::workspace::Error::Overflow)
    }

    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            for (property, value) in object.members() {
                if let Some(node) = self.properties.get_validator(property.as_ref()) {
                    if !node.is_valid(&value, ctx) {
                        return false;
                    }
                    // Use pre-computed pattern indices - no regex at runtime
                    if let Some(pattern_indices) =
                        self.property_pattern_indices.get(property.as_ref())
                    {
                        for &idx in pattern_indices {
                            if !self.patterns[idx].1.is_valid(&value, ctx) {
                                return false;
                            }
                        }
                    }
                } else {
                    // Unknown property - need runtime regex matching
                    let mut has_match = false;
                    for (re, node) in &self.patterns {
                        if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                            has_match = true;
                            if !node.is_valid(&value, ctx) {
                                return false;
                            }
                        }
                    }
                    if !has_match && !self.node.is_valid(&value, ctx) {
                        return false;
                    }
                }
            }
            true
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
        if let Some(object) = instance.as_object() {
            for (property, value) in object.members() {
                if let Some((name, node)) = self.properties.get_key_validator(property.as_ref()) {
                    let name_location = location.push(name);
                    node.validate(&value, &name_location, tracker, ctx)?;
                    // Use pre-computed pattern indices - no regex at runtime
                    if let Some(pattern_indices) =
                        self.property_pattern_indices.get(property.as_ref())
                    {
                        for &idx in pattern_indices {
                            self.patterns[idx]
                                .1
                                .validate(&value, &name_location, tracker, ctx)?;
                        }
                    }
                } else {
                    // Unknown property - need runtime regex matching
                    let property_location = location.push(property.as_ref());
                    let mut has_match = false;
                    for (re, node) in &self.patterns {
                        if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                            has_match = true;
                            node.validate(&value, &property_location, tracker, ctx)?;
                        }
                    }
                    if !has_match {
                        self.node
                            .validate(&value, &property_location, tracker, ctx)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let Some(object) = instance.as_object() else {
            return;
        };
        for (property, value) in object.members() {
            if let Some((name, node)) = self.properties.get_key_validator(property.as_ref()) {
                let name_location = location.push(name);
                node.collect_errors(&value, &name_location, tracker, ctx, errors);
                // Use pre-computed pattern indices - no regex at runtime
                if let Some(pattern_indices) = self.property_pattern_indices.get(property.as_ref())
                {
                    for &idx in pattern_indices {
                        self.patterns[idx].1.collect_errors(
                            &value,
                            &name_location,
                            tracker,
                            ctx,
                            errors,
                        );
                    }
                }
            } else {
                // Unknown property - need runtime regex matching
                let mut has_match = false;
                for (re, node) in &self.patterns {
                    if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                        has_match = true;
                        node.collect_errors(
                            &value,
                            &location.push(property.as_ref()),
                            tracker,
                            ctx,
                            errors,
                        );
                    }
                }
                if !has_match {
                    self.node.collect_errors(
                        &value,
                        &location.push(property.as_ref()),
                        tracker,
                        ctx,
                        errors,
                    );
                }
            }
        }
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if let Some(object) = instance.as_object() {
            let mut additional_matches = Vec::with_capacity(object.len());
            let mut children = Vec::with_capacity(object.len());
            for (property, value) in object.members() {
                let path = location.push(property.as_ref());
                if let Some((_name, node)) = self.properties.get_key_validator(property.as_ref()) {
                    children.push(node.evaluate_instance(&value, &path, tracker, ctx));
                    // Use pre-computed pattern indices - no regex at runtime
                    if let Some(pattern_indices) =
                        self.property_pattern_indices.get(property.as_ref())
                    {
                        for &idx in pattern_indices {
                            children.push(
                                self.patterns[idx]
                                    .1
                                    .evaluate_instance(&value, &path, tracker, ctx),
                            );
                        }
                    }
                } else {
                    // Unknown property - need runtime regex matching
                    let mut has_match = false;
                    for (pattern, node) in &self.patterns {
                        if pattern.is_match(property.as_ref(), ctx).unwrap_or(false) {
                            has_match = true;
                            children.push(node.evaluate_instance(&value, &path, tracker, ctx));
                        }
                    }
                    if !has_match {
                        additional_matches.push(property.as_ref().to_owned());
                        children.push(self.node.evaluate_instance(&value, &path, tracker, ctx));
                    }
                }
            }
            let mut result = EvaluationResult::from_children(children);
            result.annotate(Annotations::new(Value::from(additional_matches)));
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

/// # Schema example
///
/// ```json
/// {
///     "additionalProperties": false,
///     "properties": {
///         "foo": {"type": "string"}
///     },
///     "patternProperties": {
///         "^x-": {"type": "integer", "minimum": 5},
///         "-x$": {"type": "integer", "maximum": 10}
///     }
/// }
/// ```
///
/// # Valid value
///
/// ```json
/// {
///     "foo": "bar",
///     "x-bar": 6,
///     "spam-x": 7,
///     "x-baz-x": 8,
/// }
/// ```
pub(crate) struct AdditionalPropertiesWithPatternsNotEmptyFalseValidator<M, R, F: Json = SerdeJson>
{
    properties: M,
    patterns: Vec<(R, SchemaNode<F>)>,
    /// Pre-computed pattern indices for properties defined in `properties`.
    /// Maps property name -> indices into `patterns` Vec for patterns that match.
    /// Eliminates regex matching at validation time for known properties.
    property_pattern_indices: PatternIndices,
    location: Location,
}

impl<F: Json, M: PropertiesValidatorsMap<F>, R: RegexEngine> Validate<F>
    for AdditionalPropertiesWithPatternsNotEmptyFalseValidator<M, R, F>
{
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(&self.patterns)?;
        for (pattern, node) in &self.patterns {
            pattern.original_source(source)?;
            source.node(node)?;
        }
        self.properties.original_source(source)?;
        source.add(self.property_pattern_indices.allocation_size())?;
        for (name, indices) in &self.property_pattern_indices {
            source.string(name)?;
            source.vector(indices)?;
        }
        source.location(&self.location)?;
        Ok(())
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        let regex = self
            .patterns
            .iter()
            .try_fold(0usize, |peak, (pattern, _)| {
                pattern.original_controls().map(|bytes| peak.max(bytes))
            })?;
        crate::validator::workspace::body_controls::<F, Self>(&[
            regex,
            std::mem::size_of::<(
                &str,
                bool,
                usize,
                &SchemaNode<F>,
                std::slice::Iter<'_, (R, SchemaNode<F>)>,
                std::slice::Iter<'_, usize>,
                ahash::AHasher,
            )>(),
            self.properties
                .original_lookup_controls()
                .ok_or(crate::validator::workspace::Error::Overflow)?,
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)?
            .checked_add(std::mem::size_of::<Vec<String>>())
            .ok_or(crate::validator::workspace::Error::Overflow)
    }

    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            for (property, value) in object.members() {
                if let Some(node) = self.properties.get_validator(property.as_ref()) {
                    if !node.is_valid(&value, ctx) {
                        return false;
                    }
                    // Use pre-computed pattern indices - no regex at runtime for known properties
                    if let Some(pattern_indices) =
                        self.property_pattern_indices.get(property.as_ref())
                    {
                        for &idx in pattern_indices {
                            if !self.patterns[idx].1.is_valid(&value, ctx) {
                                return false;
                            }
                        }
                    }
                } else {
                    // Unknown property - need runtime regex matching
                    let mut has_match = false;
                    for (re, node) in &self.patterns {
                        if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                            has_match = true;
                            if !node.is_valid(&value, ctx) {
                                return false;
                            }
                        }
                    }
                    if !has_match {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if let Some(object) = instance.as_object() {
            for (property, value) in object.members() {
                if let Some((name, node)) = self.properties.get_key_validator(property.as_ref()) {
                    let name_location = location.push(name);
                    node.validate(&value, &name_location, tracker, ctx)?;
                    // Use pre-computed pattern indices - no regex at runtime
                    if let Some(pattern_indices) =
                        self.property_pattern_indices.get(property.as_ref())
                    {
                        for &idx in pattern_indices {
                            self.patterns[idx]
                                .1
                                .validate(&value, &name_location, tracker, ctx)?;
                        }
                    }
                } else {
                    // Unknown property - need runtime regex matching
                    let property_location = location.push(property.as_ref());
                    let mut has_match = false;
                    for (re, node) in &self.patterns {
                        if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                            has_match = true;
                            node.validate(&value, &property_location, tracker, ctx)?;
                        }
                    }
                    if !has_match {
                        return ctx.diagnostic::<F>(
                            instance,
                            location,
                            tracker,
                            &self.location,
                            |funding| {
                                Ok(crate::error::ValidationErrorKind::AdditionalProperties {
                                    unexpected: funding.copy_vec(&[property.as_ref()], |name| {
                                        funding.copy_str(name)
                                    })?,
                                })
                            },
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn collect_errors_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
        errors: &mut Vec<ValidationError<'i>>,
    ) {
        let Some(object) = instance.as_object() else {
            return;
        };
        let mut unexpected = vec![];
        for (property, value) in object.members() {
            if let Some((name, node)) = self.properties.get_key_validator(property.as_ref()) {
                let name_location = location.push(name);
                node.collect_errors(&value, &name_location, tracker, ctx, errors);
                // Use pre-computed pattern indices - no regex at runtime
                if let Some(pattern_indices) = self.property_pattern_indices.get(property.as_ref())
                {
                    for &idx in pattern_indices {
                        self.patterns[idx].1.collect_errors(
                            &value,
                            &name_location,
                            tracker,
                            ctx,
                            errors,
                        );
                    }
                }
            } else {
                // Unknown property - need runtime regex matching
                let mut has_match = false;
                for (re, node) in &self.patterns {
                    if re.is_match(property.as_ref(), ctx).unwrap_or(false) {
                        has_match = true;
                        node.collect_errors(
                            &value,
                            &location.push(property.as_ref()),
                            tracker,
                            ctx,
                            errors,
                        );
                    }
                }
                if !has_match {
                    let Some(name) = ctx.produce(|funding| funding.copy_str(property.as_ref()))
                    else {
                        return;
                    };
                    if !ctx.workspace.push(&mut unexpected, name) {
                        return;
                    }
                }
            }
        }
        if !unexpected.is_empty() {
            if let Err(error) =
                ctx.diagnostic::<F>(instance, location, tracker, &self.location, |_| {
                    Ok(crate::error::ValidationErrorKind::AdditionalProperties { unexpected })
                })
            {
                if !ctx.workspace.push(errors, error) {
                    return;
                }
            }
        }
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if let Some(object) = instance.as_object() {
            let mut unexpected = vec![];
            let mut children = Vec::with_capacity(object.len());
            for (property, value) in object.members() {
                let path = location.push(property.as_ref());
                if let Some((_name, node)) = self.properties.get_key_validator(property.as_ref()) {
                    children.push(node.evaluate_instance(&value, &path, tracker, ctx));
                    // Use pre-computed pattern indices - no regex at runtime
                    if let Some(pattern_indices) =
                        self.property_pattern_indices.get(property.as_ref())
                    {
                        for &idx in pattern_indices {
                            children.push(
                                self.patterns[idx]
                                    .1
                                    .evaluate_instance(&value, &path, tracker, ctx),
                            );
                        }
                    }
                } else {
                    // Unknown property - need runtime regex matching
                    let mut has_match = false;
                    for (pattern, node) in &self.patterns {
                        if pattern.is_match(property.as_ref(), ctx).unwrap_or(false) {
                            has_match = true;
                            children.push(node.evaluate_instance(&value, &path, tracker, ctx));
                        }
                    }
                    if !has_match {
                        unexpected.push(property.as_ref().to_owned());
                    }
                }
            }
            let mut result = EvaluationResult::from_children(children);
            if !unexpected.is_empty() {
                let eval_path = crate::paths::capture_evaluation_path(tracker, &self.location);
                result.mark_errored(ErrorDescription::from_validation_error(
                    &ValidationError::additional_properties(
                        self.location.clone(),
                        eval_path,
                        location.into(),
                        instance.to_value(),
                        unexpected,
                    ),
                ));
            }
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

macro_rules! try_compile {
    ($expr:expr) => {
        match $expr {
            Ok(result) => result,
            Err(error) => return Some(Err(error)),
        }
    };
}

/// Pre-compute which `patternProperties` patterns match each property name from `properties`.
/// This eliminates regex matching at validation time for known properties.
fn precompute_property_pattern_indices<R: RegexEngine, F: Json>(
    property_names: impl Iterator<Item = impl AsRef<str>>,
    patterns: &[(R, SchemaNode<F>)],
    funding: &crate::compilation::Funding,
) -> Result<PatternIndices, crate::CompilationError> {
    let mut result = PatternIndices::with_hasher(funding.random_state()?);
    for prop_name in property_names {
        let mut matching_indices = Vec::new();
        for (index, (regex, _)) in patterns.iter().enumerate() {
            if funding.with_validation_context::<crate::SerdeJson, _>(|context| {
                regex.is_match(prop_name.as_ref(), context).unwrap_or(false)
            })? {
                funding.push(&mut matching_indices, index)?;
            }
        }
        if !matching_indices.is_empty() {
            let name = funding.copy_str(prop_name.as_ref())?;
            funding.insert(&mut result, name, matching_indices)?;
        }
    }
    Ok(result)
}

fn compile_pattern_non_empty<'a, R, F: Json>(
    ctx: &compiler::Context<F>,
    map: &'a Map<String, Value>,
    patterns: Vec<(R, SchemaNode<F>)>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>>
where
    R: RegexEngine + 'static,
{
    let kctx = (match ctx.new_at_location("additionalProperties") {
        Ok(context) => context,
        Err(error) => return Some(Err(error.into())),
    });
    let property_pattern_indices = crate::keywords::try_compile!(
        precompute_property_pattern_indices(map.keys(), &patterns, ctx.funding())
    );

    if map.len() < HASHMAP_THRESHOLD {
        Some(Ok(
            match ctx
                .funding()
                .boxed(AdditionalPropertiesWithPatternsNotEmptyValidator::<
                    SmallValidatorsMap<F>,
                    R,
                    F,
                > {
                    node: try_compile!(compiler::compile(&kctx, kctx.as_resource_ref(schema))),
                    properties: try_compile!(compile_small_map(ctx, map)),
                    patterns,
                    property_pattern_indices,
                }) {
                Ok(value) => value,
                Err(error) => return Some(Err(error.into())),
            },
        ))
    } else {
        Some(Ok(
            match ctx
                .funding()
                .boxed(AdditionalPropertiesWithPatternsNotEmptyValidator::<
                    BigValidatorsMap<F>,
                    R,
                    F,
                > {
                    node: try_compile!(compiler::compile(&kctx, kctx.as_resource_ref(schema))),
                    properties: try_compile!(compile_big_map(ctx, map)),
                    patterns,
                    property_pattern_indices,
                }) {
                Ok(value) => value,
                Err(error) => return Some(Err(error.into())),
            },
        ))
    }
}

fn compile_pattern_non_empty_false<'a, R, F: Json>(
    ctx: &compiler::Context<F>,
    map: &'a Map<String, Value>,
    patterns: Vec<(R, SchemaNode<F>)>,
) -> Option<CompilationResult<'a, F>>
where
    R: RegexEngine + 'static,
{
    let kctx = (match ctx.new_at_location("additionalProperties") {
        Ok(context) => context,
        Err(error) => return Some(Err(error.into())),
    });
    let property_pattern_indices = crate::keywords::try_compile!(
        precompute_property_pattern_indices(map.keys(), &patterns, ctx.funding())
    );

    if map.len() < HASHMAP_THRESHOLD {
        Some(Ok(
            match ctx
                .funding()
                .boxed(AdditionalPropertiesWithPatternsNotEmptyFalseValidator::<
                    SmallValidatorsMap<F>,
                    R,
                    F,
                > {
                    properties: try_compile!(compile_small_map(ctx, map)),
                    patterns,
                    property_pattern_indices,
                    location: kctx.location().clone(),
                }) {
                Ok(value) => value,
                Err(error) => return Some(Err(error.into())),
            },
        ))
    } else {
        Some(Ok(
            match ctx
                .funding()
                .boxed(AdditionalPropertiesWithPatternsNotEmptyFalseValidator::<
                    BigValidatorsMap<F>,
                    R,
                    F,
                > {
                    properties: try_compile!(compile_big_map(ctx, map)),
                    patterns,
                    property_pattern_indices,
                    location: kctx.location().clone(),
                }) {
                Ok(value) => value,
                Err(error) => return Some(Err(error.into())),
            },
        ))
    }
}

#[inline]
pub(crate) fn compile<'a, F: Json>(
    ctx: &compiler::Context<F>,
    parent: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    let properties = parent.get("properties");
    if let Some(patterns) = parent.get("patternProperties") {
        if let Value::Object(obj) = patterns {
            // Compile all patterns & their validators to avoid doing work in the `patternProperties` validator
            match ctx.config().pattern_options() {
                PatternEngineOptions::FancyRegex { .. } => {
                    let patterns = match compile_fancy_regex_patterns(ctx, obj) {
                        Ok(patterns) => patterns,
                        Err(error) => return Some(Err(error.into())),
                    };
                    match schema {
                        Value::Bool(true) => None, // "additionalProperties" are "true" by default
                        Value::Bool(false) => {
                            if let Some(properties) = properties {
                                if let Value::Object(map) = properties {
                                    compile_pattern_non_empty_false::<
                                        CompiledPattern<Arc<fancy_regex::Regex>>,
                                        F,
                                    >(ctx, map, patterns)
                                } else {
                                    let location = crate::keywords::try_compile!(ctx
                                        .location()
                                        .join_with_funding("properties", ctx.funding()));
                                    Some(Err(ValidationError::compile_error(
                                        location.clone(),
                                        location,
                                        crate::keywords::try_compile!(Location::new_with_funding(
                                            ctx.funding()
                                        )),
                                        Cow::Borrowed(properties),
                                        "Unexpected type",
                                    )
                                    .into()))
                                }
                            } else {
                                Some(Ok(
                                    match ctx.funding().boxed(
                                        AdditionalPropertiesWithPatternsFalseValidator {
                                            patterns,
                                            location: crate::keywords::try_compile!(ctx
                                                .location()
                                                .join_with_funding(
                                                    "additionalProperties",
                                                    ctx.funding()
                                                )),
                                            pattern_keyword_path: crate::keywords::try_compile!(
                                                ctx.location().join_with_funding(
                                                    "patternProperties",
                                                    ctx.funding()
                                                )
                                            ),
                                            pattern_keyword_absolute_location: ctx.base_uri(),
                                        },
                                    ) {
                                        Ok(value) => value,
                                        Err(error) => return Some(Err(error.into())),
                                    },
                                ))
                            }
                        }
                        _ => {
                            if let Some(properties) = properties {
                                if let Value::Object(map) = properties {
                                    compile_pattern_non_empty::<
                                        CompiledPattern<Arc<fancy_regex::Regex>>,
                                        F,
                                    >(ctx, map, patterns, schema)
                                } else {
                                    let location = crate::keywords::try_compile!(ctx
                                        .location()
                                        .join_with_funding("properties", ctx.funding()));
                                    Some(Err(ValidationError::compile_error(
                                        location.clone(),
                                        location,
                                        crate::keywords::try_compile!(Location::new_with_funding(
                                            ctx.funding()
                                        )),
                                        Cow::Borrowed(properties),
                                        "Unexpected type",
                                    )
                                    .into()))
                                }
                            } else {
                                let kctx = (match ctx.new_at_location("additionalProperties") {
                                    Ok(context) => context,
                                    Err(error) => return Some(Err(error.into())),
                                });
                                Some(Ok(
                                    match ctx.funding().boxed(
                                        AdditionalPropertiesWithPatternsValidator {
                                            node: try_compile!(compiler::compile(
                                                &kctx,
                                                kctx.as_resource_ref(schema),
                                            )),
                                            patterns,
                                            pattern_keyword_path: crate::keywords::try_compile!(
                                                ctx.location().join_with_funding(
                                                    "patternProperties",
                                                    ctx.funding()
                                                )
                                            ),
                                            pattern_keyword_absolute_location: ctx.base_uri(),
                                        },
                                    ) {
                                        Ok(value) => value,
                                        Err(error) => return Some(Err(error.into())),
                                    },
                                ))
                            }
                        }
                    }
                }
                PatternEngineOptions::Regex { .. } => {
                    let patterns = match compile_regex_patterns(ctx, obj) {
                        Ok(patterns) => patterns,
                        Err(error) => return Some(Err(error.into())),
                    };
                    match schema {
                        Value::Bool(true) => None, // "additionalProperties" are "true" by default
                        Value::Bool(false) => {
                            if let Some(properties) = properties {
                                if let Value::Object(map) = properties {
                                    compile_pattern_non_empty_false::<
                                        CompiledPattern<Arc<regex::Regex>>,
                                        F,
                                    >(ctx, map, patterns)
                                } else {
                                    let location = crate::keywords::try_compile!(ctx
                                        .location()
                                        .join_with_funding("properties", ctx.funding()));
                                    Some(Err(ValidationError::compile_error(
                                        location.clone(),
                                        location,
                                        crate::keywords::try_compile!(Location::new_with_funding(
                                            ctx.funding()
                                        )),
                                        Cow::Borrowed(properties),
                                        "Unexpected type",
                                    )
                                    .into()))
                                }
                            } else {
                                Some(Ok(
                                    match ctx.funding().boxed(
                                        AdditionalPropertiesWithPatternsFalseValidator {
                                            patterns,
                                            location: crate::keywords::try_compile!(ctx
                                                .location()
                                                .join_with_funding(
                                                    "additionalProperties",
                                                    ctx.funding()
                                                )),
                                            pattern_keyword_path: crate::keywords::try_compile!(
                                                ctx.location().join_with_funding(
                                                    "patternProperties",
                                                    ctx.funding()
                                                )
                                            ),
                                            pattern_keyword_absolute_location: ctx.base_uri(),
                                        },
                                    ) {
                                        Ok(value) => value,
                                        Err(error) => return Some(Err(error.into())),
                                    },
                                ))
                            }
                        }
                        _ => {
                            if let Some(properties) = properties {
                                if let Value::Object(map) = properties {
                                    compile_pattern_non_empty::<CompiledPattern<Arc<regex::Regex>>, F>(
                                        ctx, map, patterns, schema,
                                    )
                                } else {
                                    let location = crate::keywords::try_compile!(ctx
                                        .location()
                                        .join_with_funding("properties", ctx.funding()));
                                    Some(Err(ValidationError::compile_error(
                                        location.clone(),
                                        location,
                                        crate::keywords::try_compile!(Location::new_with_funding(
                                            ctx.funding()
                                        )),
                                        Cow::Borrowed(properties),
                                        "Unexpected type",
                                    )
                                    .into()))
                                }
                            } else {
                                let kctx = (match ctx.new_at_location("additionalProperties") {
                                    Ok(context) => context,
                                    Err(error) => return Some(Err(error.into())),
                                });
                                Some(Ok(
                                    match ctx.funding().boxed(
                                        AdditionalPropertiesWithPatternsValidator {
                                            node: try_compile!(compiler::compile(
                                                &kctx,
                                                kctx.as_resource_ref(schema),
                                            )),
                                            patterns,
                                            pattern_keyword_path: crate::keywords::try_compile!(
                                                ctx.location().join_with_funding(
                                                    "patternProperties",
                                                    ctx.funding()
                                                )
                                            ),
                                            pattern_keyword_absolute_location: ctx.base_uri(),
                                        },
                                    ) {
                                        Ok(value) => value,
                                        Err(error) => return Some(Err(error.into())),
                                    },
                                ))
                            }
                        }
                    }
                }
            }
        } else {
            let location = crate::keywords::try_compile!(ctx
                .location()
                .join_with_funding("patternProperties", ctx.funding()));
            Some(Err(crate::keywords::try_compile!(
                ValidationError::single_type_error_with_funding(
                    location.clone(),
                    location,
                    crate::keywords::try_compile!(Location::new_with_funding(ctx.funding())),
                    Cow::Borrowed(patterns),
                    JsonType::Object,
                    ctx.funding()
                )
            )
            .into()))
        }
    } else {
        match schema {
            Value::Bool(true) => None, // "additionalProperties" are "true" by default
            Value::Bool(false) => {
                if let Some(properties) = properties {
                    // Check if we can use fused validator with required
                    if let Some(Value::Array(required)) = parent.get("required") {
                        if required.len() == 1 {
                            if let Some(Value::String(req)) = required.first() {
                                if let Value::Object(map) = properties {
                                    return if map.len() < HASHMAP_THRESHOLD {
                                        Some(
                                            AdditionalPropertiesNotEmptyFalseWithRequired1Validator::<
                                                SmallValidatorsMap<F>,
                                            >::compile(
                                                map, ctx, req.clone()
                                            ),
                                        )
                                    } else {
                                        Some(
                                            AdditionalPropertiesNotEmptyFalseWithRequired1Validator::<
                                                BigValidatorsMap<F>,
                                            >::compile(
                                                map, ctx, req.clone()
                                            ),
                                        )
                                    };
                                }
                            }
                        }
                    }
                    compile_dynamic_prop_map_validator!(
                        AdditionalPropertiesNotEmptyFalseValidator,
                        properties,
                        ctx,
                    )
                } else {
                    let location = crate::keywords::try_compile!(ctx
                        .location()
                        .join_with_funding("additionalProperties", ctx.funding()));
                    Some(AdditionalPropertiesFalseValidator::compile(ctx, location))
                }
            }
            _ => {
                if let Some(properties) = properties {
                    compile_dynamic_prop_map_validator!(
                        AdditionalPropertiesNotEmptyValidator,
                        properties,
                        ctx,
                        schema,
                    )
                } else {
                    Some(AdditionalPropertiesValidator::compile(schema, ctx))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::{json, Value};
    use test_case::test_case;

    fn schema_1() -> Value {
        // For `AdditionalPropertiesWithPatternsNotEmptyFalseValidator`
        json!({
            "additionalProperties": false,
            "properties": {
                "foo": {"type": "string"},
                "barbaz": {"type": "integer", "multipleOf": 3},
            },
            "patternProperties": {
                "^bar": {"type": "integer", "minimum": 5},
                "spam$": {"type": "integer", "maximum": 10},
            }
        })
    }

    // Another type
    #[test_case(&json!([1]))]
    // The right type
    #[test_case(&json!({}))]
    // Match `properties.foo`
    #[test_case(&json!({"foo": "a"}))]
    // Match `properties.barbaz` & `patternProperties.^bar`
    #[test_case(&json!({"barbaz": 6}))]
    // Match `patternProperties.^bar`
    #[test_case(&json!({"bar": 6}))]
    // Match `patternProperties.spam$`
    #[test_case(&json!({"spam": 7}))]
    // All `patternProperties` rules match on different values
    #[test_case(&json!({"bar": 6, "spam": 7}))]
    // All `patternProperties` rules match on the same value
    #[test_case(&json!({"barspam": 7}))]
    // All combined
    #[test_case(&json!({"barspam": 7, "bar": 6, "spam": 7, "foo": "a", "barbaz": 6}))]
    fn schema_1_valid(instance: &Value) {
        let schema = schema_1();
        tests_util::is_valid(&schema, instance);
    }

    // `properties.foo` - should be a string
    #[test_case(&json!({"foo": 3}), &["3 is not of type \"string\""], &["/properties/foo/type"])]
    // `additionalProperties` - extra keyword & not in `properties` / `patternProperties`
    #[test_case(&json!({"faz": 1}), &["Additional properties are not allowed (\'faz\' was unexpected)"], &["/additionalProperties"])]
    #[test_case(&json!({"faz": 1, "haz": 1}), &["Additional properties are not allowed (\'faz\', \'haz\' were unexpected)"], &["/additionalProperties"])]
    // `properties.foo` - should be a string & `patternProperties.^bar` - invalid
    #[test_case(&json!({"foo": 3, "bar": 4}), &["3 is not of type \"string\"","4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum","/properties/foo/type"])]
    // `properties.barbaz` - valid; `patternProperties.^bar` - invalid
    #[test_case(&json!({"barbaz": 3}), &["3 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.^bar` (should be >=5)
    #[test_case(&json!({"bar": 4}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.spam$` (should be <=10)
    #[test_case(&json!({"spam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // `patternProperties` - both values are invalid
    #[test_case(&json!({"bar": 4, "spam": 11}), &["11 is greater than the maximum of 10","4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum", "/patternProperties/spam$/maximum"])]
    // `patternProperties` - `bar` is valid, `spam` is invalid
    #[test_case(&json!({"bar": 6, "spam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // `patternProperties` - `bar` is invalid, `spam` is valid
    #[test_case(&json!({"bar": 4, "spam": 8}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.^bar` - (should be >=5), but valid for `patternProperties.spam$`
    #[test_case(&json!({"barspam": 4}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.spam$` - (should be <=10), but valid for `patternProperties.^bar`
    #[test_case(&json!({"barspam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // All combined
    #[test_case(
      &json!({"bar": 4, "spam": 11, "foo": 3, "faz": 1}),
      &[
          "11 is greater than the maximum of 10",
          "3 is not of type \"string\"",
          "4 is less than the minimum of 5",
          "Additional properties are not allowed (\'faz\' was unexpected)",
      ],
      &[
          "/additionalProperties",
          "/patternProperties/^bar/minimum",
          "/patternProperties/spam$/maximum",
          "/properties/foo/type",
      ]
    )]
    fn schema_1_invalid(instance: &Value, expected: &[&str], locations: &[&str]) {
        let schema = schema_1();
        tests_util::is_not_valid(&schema, instance);
        tests_util::expect_errors(&schema, instance, expected);
        tests_util::assert_locations(&schema, instance, locations);
    }

    fn schema_2() -> Value {
        // For `AdditionalPropertiesWithPatternsFalseValidator`
        json!({
            "additionalProperties": false,
            "patternProperties": {
                "^bar": {"type": "integer", "minimum": 5},
                "spam$": {"type": "integer", "maximum": 10},
            }
        })
    }

    // Another type
    #[test_case(&json!([1]))]
    // The right type
    #[test_case(&json!({}))]
    // Match `patternProperties.^bar`
    #[test_case(&json!({"bar": 6}))]
    // Match `patternProperties.spam$`
    #[test_case(&json!({"spam": 7}))]
    // All `patternProperties` rules match on different values
    #[test_case(&json!({"bar": 6, "spam": 7}))]
    // All `patternProperties` rules match on the same value
    #[test_case(&json!({"barspam": 7}))]
    // All combined
    #[test_case(&json!({"barspam": 7, "bar": 6, "spam": 7}))]
    fn schema_2_valid(instance: &Value) {
        let schema = schema_2();
        tests_util::is_valid(&schema, instance);
    }

    // `additionalProperties` - extra keyword & not in `patternProperties`
    #[test_case(&json!({"faz": "a"}), &["Additional properties are not allowed (\'faz\' was unexpected)"], &["/additionalProperties"])]
    // `patternProperties.^bar` (should be >=5)
    #[test_case(&json!({"bar": 4}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.spam$` (should be <=10)
    #[test_case(&json!({"spam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // `patternProperties` - both values are invalid
    #[test_case(&json!({"bar": 4, "spam": 11}), &["11 is greater than the maximum of 10","4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum", "/patternProperties/spam$/maximum"])]
    // `patternProperties` - `bar` is valid, `spam` is invalid
    #[test_case(&json!({"bar": 6, "spam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // `patternProperties` - `bar` is invalid, `spam` is valid
    #[test_case(&json!({"bar": 4, "spam": 8}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.^bar` - (should be >=5), but valid for `patternProperties.spam$`
    #[test_case(&json!({"barspam": 4}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.spam$` - (should be <=10), but valid for `patternProperties.^bar`
    #[test_case(&json!({"barspam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // All combined
    #[test_case(
      &json!({"bar": 4, "spam": 11, "faz": 1}),
      &[
          "11 is greater than the maximum of 10",
          "4 is less than the minimum of 5",
          "Additional properties are not allowed (\'faz\' was unexpected)",
      ],
      &[
          "/additionalProperties",
          "/patternProperties/^bar/minimum",
          "/patternProperties/spam$/maximum",
      ]
    )]
    fn schema_2_invalid(instance: &Value, expected: &[&str], locations: &[&str]) {
        let schema = schema_2();
        tests_util::is_not_valid(&schema, instance);
        tests_util::expect_errors(&schema, instance, expected);
        tests_util::assert_locations(&schema, instance, locations);
    }

    fn schema_3() -> Value {
        // For `AdditionalPropertiesNotEmptyFalseValidator`
        json!({
            "additionalProperties": false,
            "properties": {
                "foo": {"type": "string"}
            }
        })
    }

    // Another type
    #[test_case(&json!([1]))]
    // The right type
    #[test_case(&json!({}))]
    // Match `properties`
    #[test_case(&json!({"foo": "a"}))]
    fn schema_3_valid(instance: &Value) {
        let schema = schema_3();
        tests_util::is_valid(&schema, instance);
    }

    // `properties` - should be a string
    #[test_case(&json!({"foo": 3}), &["3 is not of type \"string\""], &["/properties/foo/type"])]
    // `additionalProperties` - extra keyword & not in `properties`
    #[test_case(&json!({"faz": "a"}), &["Additional properties are not allowed (\'faz\' was unexpected)"], &["/additionalProperties"])]
    // All combined
    #[test_case(
      &json!(
        {"foo": 3, "faz": "a"}),
        &[
            "3 is not of type \"string\"",
            "Additional properties are not allowed (\'faz\' was unexpected)",
        ],
        &[
            "/additionalProperties",
            "/properties/foo/type",
        ]
    )]
    fn schema_3_invalid(instance: &Value, expected: &[&str], locations: &[&str]) {
        let schema = schema_3();
        tests_util::is_not_valid(&schema, instance);
        tests_util::expect_errors(&schema, instance, expected);
        tests_util::assert_locations(&schema, instance, locations);
    }

    fn schema_4() -> Value {
        // For `AdditionalPropertiesNotEmptyValidator`
        json!({
            "additionalProperties": {"type": "integer"},
            "properties": {
                "foo": {"type": "string"}
            }
        })
    }

    // Another type
    #[test_case(&json!([1]))]
    // The right type
    #[test_case(&json!({}))]
    // Match `properties`
    #[test_case(&json!({"foo": "a"}))]
    // Match `additionalProperties`
    #[test_case(&json!({"bar": 4}))]
    // All combined
    #[test_case(&json!({"foo": "a", "bar": 4}))]
    fn schema_4_valid(instance: &Value) {
        let schema = schema_4();
        tests_util::is_valid(&schema, instance);
    }

    // `properties` - should be a string
    #[test_case(&json!({"foo": 3}), &["3 is not of type \"string\""], &["/properties/foo/type"])]
    // `additionalProperties` - should be an integer
    #[test_case(&json!({"bar": "a"}), &["\"a\" is not of type \"integer\""], &["/additionalProperties/type"])]
    // All combined
    #[test_case(
      &json!(
        {"foo": 3, "bar": "a"}),
        &[
            "\"a\" is not of type \"integer\"",
            "3 is not of type \"string\"",
        ],
        &[
            "/additionalProperties/type",
            "/properties/foo/type",
        ]
    )]
    fn schema_4_invalid(instance: &Value, expected: &[&str], locations: &[&str]) {
        let schema = schema_4();
        tests_util::is_not_valid(&schema, instance);
        tests_util::expect_errors(&schema, instance, expected);
        tests_util::assert_locations(&schema, instance, locations);
    }

    fn schema_5() -> Value {
        // For `AdditionalPropertiesWithPatternsNotEmptyValidator`
        json!({
            "additionalProperties": {"type": "integer"},
            "properties": {
                "foo": {"type": "string"},
                "barbaz": {"type": "integer", "multipleOf": 3},
            },
            "patternProperties": {
                "^bar": {"type": "integer", "minimum": 5},
                "spam$": {"type": "integer", "maximum": 10},
            }
        })
    }

    // Another type
    #[test_case(&json!([1]))]
    // The right type
    #[test_case(&json!({}))]
    // Match `properties.foo`
    #[test_case(&json!({"foo": "a"}))]
    // Match `additionalProperties`
    #[test_case(&json!({"faz": 42}))]
    // Match `properties.barbaz` & `patternProperties.^bar`
    #[test_case(&json!({"barbaz": 6}))]
    // Match `patternProperties.^bar`
    #[test_case(&json!({"bar": 6}))]
    // Match `patternProperties.spam$`
    #[test_case(&json!({"spam": 7}))]
    // All `patternProperties` rules match on different values
    #[test_case(&json!({"bar": 6, "spam": 7}))]
    // All `patternProperties` rules match on the same value
    #[test_case(&json!({"barspam": 7}))]
    // All combined
    #[test_case(&json!({"barspam": 7, "bar": 6, "spam": 7, "foo": "a", "barbaz": 6, "faz": 42}))]
    fn schema_5_valid(instance: &Value) {
        let schema = schema_5();
        tests_util::is_valid(&schema, instance);
    }

    // `properties.bar` - should be a string
    #[test_case(&json!({"foo": 3}), &["3 is not of type \"string\""], &["/properties/foo/type"])]
    // `additionalProperties` - extra keyword that doesn't match `additionalProperties`
    #[test_case(&json!({"faz": "a"}), &["\"a\" is not of type \"integer\""], &["/additionalProperties/type"])]
    #[test_case(&json!({"faz": "a", "haz": "a"}), &["\"a\" is not of type \"integer\"", "\"a\" is not of type \"integer\""], &["/additionalProperties/type", "/additionalProperties/type"])]
    // `properties.foo` - should be a string & `patternProperties.^bar` - invalid
    #[test_case(&json!({"foo": 3, "bar": 4}), &["3 is not of type \"string\"","4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum","/properties/foo/type"])]
    // `properties.barbaz` - valid; `patternProperties.^bar` - invalid
    #[test_case(&json!({"barbaz": 3}), &["3 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.^bar` (should be >=5)
    #[test_case(&json!({"bar": 4}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.spam$` (should be <=10)
    #[test_case(&json!({"spam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // `patternProperties` - both values are invalid
    #[test_case(&json!({"bar": 4, "spam": 11}), &["11 is greater than the maximum of 10","4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum", "/patternProperties/spam$/maximum"])]
    // `patternProperties` - `bar` is valid, `spam` is invalid
    #[test_case(&json!({"bar": 6, "spam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // `patternProperties` - `bar` is invalid, `spam` is valid
    #[test_case(&json!({"bar": 4, "spam": 8}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.^bar` - (should be >=5), but valid for `patternProperties.spam$`
    #[test_case(&json!({"barspam": 4}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.spam$` - (should be <=10), but valid for `patternProperties.^bar`
    #[test_case(&json!({"barspam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // All combined + valid via `additionalProperties`
    #[test_case(
      &json!({"bar": 4, "spam": 11, "foo": 3, "faz": "a", "fam": 42}),
      &[
          "\"a\" is not of type \"integer\"",
          "11 is greater than the maximum of 10",
          "3 is not of type \"string\"",
          "4 is less than the minimum of 5",
      ],
      &[
          "/additionalProperties/type",
          "/patternProperties/^bar/minimum",
          "/patternProperties/spam$/maximum",
          "/properties/foo/type",
      ]
    )]
    fn schema_5_invalid(instance: &Value, expected: &[&str], locations: &[&str]) {
        let schema = schema_5();
        tests_util::is_not_valid(&schema, instance);
        tests_util::expect_errors(&schema, instance, expected);
        tests_util::assert_locations(&schema, instance, locations);
    }

    fn schema_6() -> Value {
        // For `AdditionalPropertiesWithPatternsValidator`
        json!({
            "additionalProperties": {"type": "integer"},
            "patternProperties": {
                "^bar": {"type": "integer", "minimum": 5},
                "spam$": {"type": "integer", "maximum": 10},
            }
        })
    }

    // Another type
    #[test_case(&json!([1]))]
    // The right type
    #[test_case(&json!({}))]
    // Match `additionalProperties`
    #[test_case(&json!({"faz": 42}))]
    // Match `patternProperties.^bar`
    #[test_case(&json!({"bar": 6}))]
    // Match `patternProperties.spam$`
    #[test_case(&json!({"spam": 7}))]
    // All `patternProperties` rules match on different values
    #[test_case(&json!({"bar": 6, "spam": 7}))]
    // All `patternProperties` rules match on the same value
    #[test_case(&json!({"barspam": 7}))]
    // All combined
    #[test_case(&json!({"barspam": 7, "bar": 6, "spam": 7, "faz": 42}))]
    fn schema_6_valid(instance: &Value) {
        let schema = schema_6();
        tests_util::is_valid(&schema, instance);
    }

    // `additionalProperties` - extra keyword that doesn't match `additionalProperties`
    #[test_case(&json!({"faz": "a"}), &["\"a\" is not of type \"integer\""], &["/additionalProperties/type"])]
    #[test_case(&json!({"faz": "a", "haz": "a"}), &["\"a\" is not of type \"integer\"", "\"a\" is not of type \"integer\""], &["/additionalProperties/type", "/additionalProperties/type"])]
    // `additionalProperties` - should be an integer & `patternProperties.^bar` - invalid
    #[test_case(&json!({"foo": "a", "bar": 4}), &["\"a\" is not of type \"integer\"","4 is less than the minimum of 5"], &["/additionalProperties/type","/patternProperties/^bar/minimum"])]
    // `patternProperties.^bar` (should be >=5)
    #[test_case(&json!({"bar": 4}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.spam$` (should be <=10)
    #[test_case(&json!({"spam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // `patternProperties` - both values are invalid
    #[test_case(&json!({"bar": 4, "spam": 11}), &["11 is greater than the maximum of 10","4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum", "/patternProperties/spam$/maximum"])]
    // `patternProperties` - `bar` is valid, `spam` is invalid
    #[test_case(&json!({"bar": 6, "spam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // `patternProperties` - `bar` is invalid, `spam` is valid
    #[test_case(&json!({"bar": 4, "spam": 8}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.^bar` - (should be >=5), but valid for `patternProperties.spam$`
    #[test_case(&json!({"barspam": 4}), &["4 is less than the minimum of 5"], &["/patternProperties/^bar/minimum"])]
    // `patternProperties.spam$` - (should be <=10), but valid for `patternProperties.^bar`
    #[test_case(&json!({"barspam": 11}), &["11 is greater than the maximum of 10"], &["/patternProperties/spam$/maximum"])]
    // All combined + valid via `additionalProperties`
    #[test_case(
      &json!({"bar": 4, "spam": 11, "faz": "a", "fam": 42}),
      &[
          "\"a\" is not of type \"integer\"",
          "11 is greater than the maximum of 10",
          "4 is less than the minimum of 5",
      ],
      &[
          "/additionalProperties/type",
          "/patternProperties/^bar/minimum",
          "/patternProperties/spam$/maximum",
      ]
    )]
    fn schema_6_invalid(instance: &Value, expected: &[&str], locations: &[&str]) {
        let schema = schema_6();
        tests_util::is_not_valid(&schema, instance);
        tests_util::expect_errors(&schema, instance, expected);
        tests_util::assert_locations(&schema, instance, locations);
    }

    // Invalid regex pattern in `patternProperties` with `additionalProperties: false`
    #[test_case(&json!({"additionalProperties": false, "patternProperties": {"[invalid": {"type": "string"}}}))]
    // Invalid regex pattern in `patternProperties` with `additionalProperties` as an object
    #[test_case(&json!({"additionalProperties": {"type": "integer"}, "patternProperties": {"[invalid": {"type": "string"}}}))]
    // Invalid regex pattern in `patternProperties` with `properties` and `additionalProperties: false`
    #[test_case(&json!({"properties": {"foo": {"type": "string"}}, "additionalProperties": false, "patternProperties": {"[invalid": {"type": "string"}}}))]
    fn invalid_pattern_properties_fancy_regex(schema: &Value) {
        // Default engine is fancy_regex
        let error = crate::validator_for(schema).expect_err("Should fail to compile");
        assert!(error.to_string().contains("regex"));
    }

    #[test_case(&json!({"additionalProperties": false, "patternProperties": {"[invalid": {"type": "string"}}}))]
    #[test_case(&json!({"additionalProperties": {"type": "integer"}, "patternProperties": {"[invalid": {"type": "string"}}}))]
    #[test_case(&json!({"properties": {"foo": {"type": "string"}}, "additionalProperties": false, "patternProperties": {"[invalid": {"type": "string"}}}))]
    fn invalid_pattern_properties_standard_regex(schema: &Value) {
        use crate::PatternOptions;

        let error = crate::options()
            .with_pattern_options(PatternOptions::regex())
            .build(schema)
            .expect_err("Should fail to compile");
        assert!(error.to_string().contains("regex"));
    }

    // Test prefix optimization with additionalProperties: false
    #[test_case("^x-", "x-custom", true)]
    #[test_case("^x-", "y-other", false)]
    #[test_case("^eo_", "eo_bands", true)]
    #[test_case("^eo_", "other", false)]
    fn prefix_pattern_with_additional_false(pattern: &str, key: &str, matches_pattern: bool) {
        let schema = json!({
            "additionalProperties": false,
            "patternProperties": {
                pattern: {"type": "string"}
            }
        });
        let validator = crate::validator_for(&schema).unwrap();

        // Valid string value
        let string_instance = json!({ key: "value" });
        // If matches_pattern: key matches, string is valid type -> valid
        // If !matches_pattern: key is additional property, not allowed -> invalid
        assert_eq!(validator.is_valid(&string_instance), matches_pattern);

        // Invalid type (number instead of string)
        let number_instance = json!({ key: 42 });
        // If matches_pattern: key matches but wrong type -> invalid
        // If !matches_pattern: additional property not allowed -> invalid
        assert!(!validator.is_valid(&number_instance));
    }

    // Test prefix optimization with additionalProperties as schema
    #[test]
    fn prefix_pattern_with_additional_schema() {
        let schema = json!({
            "additionalProperties": {"type": "integer"},
            "patternProperties": {
                "^x-": {"type": "string"}
            }
        });
        let validator = crate::validator_for(&schema).unwrap();

        // x-foo matches pattern, must be string
        assert!(validator.is_valid(&json!({"x-foo": "bar"})));
        assert!(!validator.is_valid(&json!({"x-foo": 42})));

        // other doesn't match pattern, goes to additionalProperties (must be integer)
        assert!(validator.is_valid(&json!({"other": 42})));
        assert!(!validator.is_valid(&json!({"other": "str"})));

        // Both patterns
        assert!(validator.is_valid(&json!({"x-foo": "bar", "other": 42})));
        assert!(!validator.is_valid(&json!({"x-foo": 42, "other": "str"})));
    }

    // Test prefix optimization with properties + additionalProperties: false
    #[test]
    fn prefix_pattern_with_properties_and_additional_false() {
        let schema = json!({
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": false,
            "patternProperties": {
                "^x-": {"type": "string"}
            }
        });
        let validator = crate::validator_for(&schema).unwrap();

        // name is in properties
        assert!(validator.is_valid(&json!({"name": "test"})));
        assert!(!validator.is_valid(&json!({"name": 42})));

        // x-foo matches pattern
        assert!(validator.is_valid(&json!({"x-foo": "bar"})));
        assert!(!validator.is_valid(&json!({"x-foo": 42})));

        // other is neither in properties nor matches pattern -> additional, not allowed
        assert!(!validator.is_valid(&json!({"other": "value"})));

        // Combined valid
        assert!(validator.is_valid(&json!({"name": "test", "x-foo": "bar"})));
    }

    // Test fused validator with properties + additionalProperties: false + single required
    #[test]
    fn fused_additional_properties_false_with_required_1() {
        let schema = json!({
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"}
            },
            "additionalProperties": false,
            "required": ["id"]
        });
        let validator = crate::validator_for(&schema).unwrap();

        // Valid: has required and valid additional
        assert!(validator.is_valid(&json!({"id": "123"})));
        assert!(validator.is_valid(&json!({"id": "123", "name": "test"})));

        // Invalid: missing required
        assert!(!validator.is_valid(&json!({})));
        assert!(!validator.is_valid(&json!({"name": "test"})));

        // Invalid: has additional property
        assert!(!validator.is_valid(&json!({"id": "123", "extra": "val"})));

        // Invalid: wrong type for property
        assert!(!validator.is_valid(&json!({"id": 123})));

        // Non-object passes
        assert!(validator.is_valid(&json!("string")));
        assert!(validator.is_valid(&json!([1, 2, 3])));
    }

    #[test]
    fn fused_additional_properties_false_with_required_1_errors() {
        let schema = json!({
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"}
            },
            "additionalProperties": false,
            "required": ["id"]
        });
        let validator = crate::validator_for(&schema).unwrap();

        // Missing required should produce required error
        let instance = json!({});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].to_string().contains("required"));

        // Missing required with valid other property
        let instance = json!({"name": "test"});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].to_string().contains("required"));

        // Additional property error
        let instance = json!({"id": "123", "extra": "val"});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].to_string().contains("Additional properties"));

        // Both missing required and extra property
        let instance = json!({"extra": "val"});
        let errors: Vec<_> = validator.iter_errors(&instance).collect();
        assert_eq!(errors.len(), 2);
    }
}
