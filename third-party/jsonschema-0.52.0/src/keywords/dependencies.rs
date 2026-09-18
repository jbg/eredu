use std::borrow::Cow;

use crate::{
    compiler,
    error::ValidationError,
    keywords::{required, CompilationResult},
    node::SchemaNode,
    paths::{LazyLocation, Location, RefTracker},
    types::JsonType,
    validator::{EvaluationResult, Validate, ValidationContext},
    Json, Node, Object, SerdeJson,
};
use serde_json::{Map, Value};

pub(crate) struct DependenciesValidator<F: Json = SerdeJson> {
    dependencies: Vec<(F::PreparedKey, SchemaNode<F>)>,
}

impl DependenciesValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &compiler::Context<F>,
        schema: &'a Value,
    ) -> CompilationResult<'a, F> {
        if let Value::Object(map) = schema {
            let kctx = ctx.new_at_location("dependencies")?;
            let mut dependencies = Vec::new();
            ctx.funding().grow(&mut dependencies, map.len())?;
            for (key, subschema) in map {
                let ctx = kctx.new_at_location(key.as_str())?;
                let s = match subschema {
                    Value::Array(_) => {
                        let mut validators = Vec::new();
                        ctx.funding().grow(&mut validators, 1)?;
                        validators.push(
                            required::compile_with_path(&ctx, subschema, kctx.location().clone())
                                .expect("The required validator compilation does not return None")?,
                        );
                        SchemaNode::from_array(&kctx, validators)?
                    }
                    _ => compiler::compile(&ctx, ctx.as_resource_ref(subschema))?,
                };
                dependencies.push((ctx.funding().key::<F>(key)?, s));
            }
            Ok(ctx
                .funding()
                .boxed(DependenciesValidator { dependencies })?)
        } else {
            let location = ctx
                .location()
                .join_with_funding("dependencies", ctx.funding())?;
            Err(ValidationError::single_type_error_with_funding(
                location.clone(),
                location,
                Location::new_with_funding(ctx.funding())?,
                Cow::Borrowed(schema),
                JsonType::Object,
                ctx.funding(),
            )?
            .into())
        }
    }
}

impl<F: Json> Validate<F> for DependenciesValidator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(&self.dependencies)?;
        for (key, node) in &self.dependencies {
            source.key(key)?;
            source.node(node)?;
        }
        Ok(())
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::slice::Iter<'_, (F::PreparedKey, SchemaNode<F>)>>(),
            std::mem::size_of::<(&F::PreparedKey, &SchemaNode<F>, bool)>(),
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            for (property, node) in &self.dependencies {
                if object.get(property).is_some() && !node.is_valid(instance, ctx) {
                    return false;
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
            for (property, dependency) in &self.dependencies {
                if object.get(property).is_some() {
                    dependency.validate(instance, location, tracker, ctx)?;
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
        for (property, node) in &self.dependencies {
            if object.get(property).is_some() {
                node.collect_errors(instance, location, tracker, ctx, errors);
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
            let mut children = Vec::new();
            for (property, dependency) in &self.dependencies {
                if object.get(property).is_some() {
                    children.push(dependency.evaluate_instance(instance, location, tracker, ctx));
                }
            }
            EvaluationResult::from_children(children)
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

pub(crate) struct DependentRequiredValidator<F: Json = SerdeJson> {
    dependencies: Vec<(F::PreparedKey, SchemaNode<F>)>,
}

impl DependentRequiredValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &compiler::Context<F>,
        schema: &'a Value,
    ) -> CompilationResult<'a, F> {
        if let Value::Object(map) = schema {
            let kctx = ctx.new_at_location("dependentRequired")?;
            let mut dependencies = Vec::new();
            ctx.funding().grow(&mut dependencies, map.len())?;
            for (key, subschema) in map {
                let ictx = kctx.new_at_location(key.as_str())?;
                if let Value::Array(dependency_array) = subschema {
                    if !ctx.funding().unique(dependency_array)? {
                        let location = ictx.location().clone();
                        return Err(ValidationError::unique_items_with_funding(
                            location.clone(),
                            location,
                            Location::new_with_funding(ctx.funding())?,
                            Cow::Borrowed(subschema),
                            ctx.funding(),
                        )?
                        .into());
                    }
                    let mut validators = Vec::new();
                    ctx.funding().grow(&mut validators, 1)?;
                    validators.push(
                        required::compile_with_path(&ctx, subschema, kctx.location().clone())
                            .expect("The required validator compilation does not return None")?,
                    );
                    dependencies.push((
                        ctx.funding().key::<F>(key)?,
                        SchemaNode::from_array(&kctx, validators)?,
                    ));
                } else {
                    let location = ictx.location().clone();
                    return Err(ValidationError::single_type_error_with_funding(
                        location.clone(),
                        location,
                        Location::new_with_funding(ctx.funding())?,
                        Cow::Borrowed(subschema),
                        JsonType::Array,
                        ctx.funding(),
                    )?
                    .into());
                }
            }
            Ok(ctx
                .funding()
                .boxed(DependentRequiredValidator { dependencies })?)
        } else {
            let location = ctx
                .location()
                .join_with_funding("dependentRequired", ctx.funding())?;
            Err(ValidationError::single_type_error_with_funding(
                location.clone(),
                location,
                Location::new_with_funding(ctx.funding())?,
                Cow::Borrowed(schema),
                JsonType::Object,
                ctx.funding(),
            )?
            .into())
        }
    }
}
impl<F: Json> Validate<F> for DependentRequiredValidator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(&self.dependencies)?;
        for (key, node) in &self.dependencies {
            source.key(key)?;
            source.node(node)?;
        }
        Ok(())
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::slice::Iter<'_, (F::PreparedKey, SchemaNode<F>)>>(),
            std::mem::size_of::<(&F::PreparedKey, &SchemaNode<F>, bool)>(),
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            for (property, node) in &self.dependencies {
                if object.get(property).is_some() && !node.is_valid(instance, ctx) {
                    return false;
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
            for (property, dependency) in &self.dependencies {
                if object.get(property).is_some() {
                    dependency.validate(instance, location, tracker, ctx)?;
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
        for (property, node) in &self.dependencies {
            if object.get(property).is_some() {
                node.collect_errors(instance, location, tracker, ctx, errors);
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
            let mut children = Vec::new();
            for (property, dependency) in &self.dependencies {
                if object.get(property).is_some() {
                    children.push(dependency.evaluate_instance(instance, location, tracker, ctx));
                }
            }
            EvaluationResult::from_children(children)
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

pub(crate) struct DependentSchemasValidator<F: Json = SerdeJson> {
    dependencies: Vec<(F::PreparedKey, SchemaNode<F>)>,
}
impl DependentSchemasValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &compiler::Context<F>,
        schema: &'a Value,
    ) -> CompilationResult<'a, F> {
        if let Value::Object(map) = schema {
            let ctx = ctx.new_at_location("dependentSchemas")?;
            let mut dependencies = Vec::new();
            ctx.funding().grow(&mut dependencies, map.len())?;
            for (key, subschema) in map {
                let ctx = ctx.new_at_location(key.as_str())?;
                let schema_nodes = compiler::compile(&ctx, ctx.as_resource_ref(subschema))?;
                dependencies.push((ctx.funding().key::<F>(key)?, schema_nodes));
            }
            Ok(ctx
                .funding()
                .boxed(DependentSchemasValidator { dependencies })?)
        } else {
            let location = ctx
                .location()
                .join_with_funding("dependentSchemas", ctx.funding())?;
            Err(ValidationError::single_type_error_with_funding(
                location.clone(),
                location,
                Location::new_with_funding(ctx.funding())?,
                Cow::Borrowed(schema),
                JsonType::Object,
                ctx.funding(),
            )?
            .into())
        }
    }
}
impl<F: Json> Validate<F> for DependentSchemasValidator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(&self.dependencies)?;
        for (key, node) in &self.dependencies {
            source.key(key)?;
            source.node(node)?;
        }
        Ok(())
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::slice::Iter<'_, (F::PreparedKey, SchemaNode<F>)>>(),
            std::mem::size_of::<(&F::PreparedKey, &SchemaNode<F>, bool)>(),
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            for (property, node) in &self.dependencies {
                if object.get(property).is_some() && !node.is_valid(instance, ctx) {
                    return false;
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
            for (property, dependency) in &self.dependencies {
                if object.get(property).is_some() {
                    dependency.validate(instance, location, tracker, ctx)?;
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
        for (property, node) in &self.dependencies {
            if object.get(property).is_some() {
                node.collect_errors(instance, location, tracker, ctx, errors);
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
            let mut children = Vec::new();
            for (property, dependency) in &self.dependencies {
                if object.get(property).is_some() {
                    children.push(dependency.evaluate_instance(instance, location, tracker, ctx));
                }
            }
            EvaluationResult::from_children(children)
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

#[inline]
pub(crate) fn compile<'a, F: Json>(
    ctx: &compiler::Context<F>,
    _: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    Some(DependenciesValidator::compile(ctx, schema))
}
#[inline]
pub(crate) fn compile_dependent_required<'a, F: Json>(
    ctx: &compiler::Context<F>,
    _: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    Some(DependentRequiredValidator::compile(ctx, schema))
}
#[inline]
pub(crate) fn compile_dependent_schemas<'a, F: Json>(
    ctx: &compiler::Context<F>,
    _: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    Some(DependentSchemasValidator::compile(ctx, schema))
}
#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::{json, Value};
    use test_case::test_case;

    #[test_case(&json!({"dependencies": {"bar": ["foo"]}}), &json!({"bar": 1}), "/dependencies")]
    #[test_case(&json!({"dependencies": {"bar": {"type": "string"}}}), &json!({"bar": 1}), "/dependencies/bar/type")]
    fn location(schema: &Value, instance: &Value, expected: &str) {
        tests_util::assert_schema_location(schema, instance, expected);
    }
}
