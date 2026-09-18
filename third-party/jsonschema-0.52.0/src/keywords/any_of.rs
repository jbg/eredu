use std::borrow::Cow;

use crate::{
    compiler,
    error::ValidationError,
    node::SchemaNode,
    paths::{LazyLocation, Location, RefTracker},
    types::JsonType,
    validator::{EvaluationResult, Validate, ValidationContext},
    Json, Node, SerdeJson,
};
use serde_json::{Map, Value};

use super::CompilationResult;

fn any_of_failure<'i, F: Json>(
    schemas: &[SchemaNode<F>],
    schema_path: &Location,
    instance: &F::Node<'i>,
    location: &LazyLocation,
    tracker: Option<&RefTracker>,
    ctx: &mut ValidationContext,
) -> Result<(), ValidationError<'i>> {
    let Some(branches) = ctx.branch_errors(schemas, instance, location, tracker) else {
        return Ok(());
    };
    ctx.diagnostic::<F>(instance, location, tracker, schema_path, |funding| {
        Ok(crate::error::ValidationErrorKind::AnyOf {
            context: funding.error_context(branches)?,
        })
    })
}

pub(crate) struct AnyOfValidator<F: Json> {
    schemas: Vec<SchemaNode<F>>,
    location: Location,
}

impl AnyOfValidator<SerdeJson> {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &compiler::Context<F>,
        schema: &'a Value,
    ) -> CompilationResult<'a, F> {
        if let Value::Array(items) = schema {
            let ctx = ctx.new_at_location("anyOf")?;
            let mut schemas = Vec::new();
            ctx.funding().grow(&mut schemas, items.len())?;
            for (idx, item) in items.iter().enumerate() {
                let ctx = ctx.new_at_location(idx)?;
                let node = compiler::compile(&ctx, ctx.as_resource_ref(item))?;
                schemas.push(node);
            }
            Ok(ctx.funding().boxed(AnyOfValidator {
                schemas,
                location: ctx.location().clone(),
            })?)
        } else {
            let location = ctx.location().join_with_funding("anyOf", ctx.funding())?;
            Err(ValidationError::single_type_error_with_funding(
                location.clone(),
                location,
                Location::new_with_funding(ctx.funding())?,
                Cow::Borrowed(schema),
                JsonType::Array,
                ctx.funding(),
            )?
            .into())
        }
    }
}

impl<F: Json> Validate<F> for AnyOfValidator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.nodes(&self.schemas)?;
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::slice::Iter<'_, crate::node::SchemaNode<F>>>(),
            std::mem::size_of::<&crate::node::SchemaNode<F>>(),
            std::mem::size_of::<(bool, usize)>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        let base = <Self as Validate<F>>::original_controls(self)?;
        base.checked_add(std::mem::size_of::<(
            Vec<Vec<ValidationError<'_>>>,
            Vec<ValidationError<'_>>,
            &SchemaNode<F>,
        )>())
        .ok_or(crate::validator::workspace::Error::Overflow)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        self.schemas.iter().any(|s| s.is_valid(instance, ctx))
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if self.is_valid(instance, ctx) {
            return Ok(());
        }
        any_of_failure(
            &self.schemas,
            &self.location,
            instance,
            location,
            tracker,
            ctx,
        )
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        // Per spec §10.2.1.2, annotations must be collected from ALL valid branches.
        // First detect all valid branches cheaply, then evaluate only those branches to avoid
        // constructing dropped error trees for invalid branches in the common case.
        let valid_indices: Vec<_> = self
            .schemas
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| node.is_valid(instance, ctx).then_some(idx))
            .collect();

        if valid_indices.is_empty() {
            // No valid schemas - evaluate all for error output.
            let failures: Vec<_> = self
                .schemas
                .iter()
                .map(|node| node.evaluate_instance(instance, location, tracker, ctx))
                .collect();
            EvaluationResult::from_children(failures)
        } else {
            let valid_results: Vec<_> = valid_indices
                .into_iter()
                .map(|idx| self.schemas[idx].evaluate_instance(instance, location, tracker, ctx))
                .collect();
            EvaluationResult::from_children(valid_results)
        }
    }
}

/// Optimized validator for `anyOf` with a single subschema.
pub(crate) struct SingleAnyOfValidator<F: Json> {
    node: SchemaNode<F>,
    location: Location,
}

impl SingleAnyOfValidator<SerdeJson> {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &compiler::Context<F>,
        schema: &'a Value,
    ) -> CompilationResult<'a, F> {
        let any_of_ctx = ctx.new_at_location("anyOf")?;
        let item_ctx = any_of_ctx.new_at_location(0)?;
        let node = compiler::compile(&item_ctx, item_ctx.as_resource_ref(schema))?;
        Ok(ctx.funding().boxed(SingleAnyOfValidator {
            node,
            location: any_of_ctx.location().clone(),
        })?)
    }
}

impl<F: Json> Validate<F> for SingleAnyOfValidator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.node(&self.node)?;
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::slice::Iter<'_, crate::node::SchemaNode<F>>>(),
            std::mem::size_of::<&crate::node::SchemaNode<F>>(),
            std::mem::size_of::<(bool, usize)>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        let base = <Self as Validate<F>>::original_controls(self)?;
        base.checked_add(std::mem::size_of::<(
            Vec<Vec<ValidationError<'_>>>,
            Vec<ValidationError<'_>>,
            &SchemaNode<F>,
        )>())
        .ok_or(crate::validator::workspace::Error::Overflow)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        self.node.is_valid(instance, ctx)
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if self.is_valid(instance, ctx) {
            return Ok(());
        }
        any_of_failure(
            std::slice::from_ref(&self.node),
            &self.location,
            instance,
            location,
            tracker,
            ctx,
        )
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        EvaluationResult::from(
            self.node
                .evaluate_instance(instance, location, tracker, ctx),
        )
    }
}

#[inline]
pub(crate) fn compile<'a, F: Json>(
    ctx: &compiler::Context<F>,
    _: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    if let Value::Array(items) = schema {
        match items.as_slice() {
            [item] => Some(SingleAnyOfValidator::compile(ctx, item)),
            _ => Some(AnyOfValidator::compile(ctx, schema)),
        }
    } else {
        let location =
            crate::keywords::try_compile!(ctx.location().join_with_funding("anyOf", ctx.funding()));
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

    #[test_case(&json!({"anyOf": [{"type": "string"}]}), &json!(1), "/anyOf")]
    #[test_case(&json!({"anyOf": [{"type": "integer"}, {"type": "string"}]}), &json!({}), "/anyOf")]
    fn location(schema: &Value, instance: &Value, expected: &str) {
        tests_util::assert_schema_location(schema, instance, expected);
    }
}
