use crate::{
    compiler,
    error::ValidationError,
    keywords::CompilationResult,
    node::SchemaNode,
    paths::{LazyLocation, Location, RefTracker},
    validator::{EvaluationResult, Validate, ValidationContext},
    Json, Node, Object,
};
use serde_json::{Map, Value};

pub(crate) struct PropertyNamesObjectValidator<F: Json> {
    node: SchemaNode<F>,
}

impl<F: Json> PropertyNamesObjectValidator<F> {
    #[inline]
    pub(crate) fn compile<'a>(
        ctx: &compiler::Context<F>,
        schema: &'a Value,
    ) -> CompilationResult<'a, F> {
        let ctx = ctx.new_at_location("propertyNames")?;
        Ok(ctx.funding().boxed(PropertyNamesObjectValidator {
            node: compiler::compile(&ctx, ctx.as_resource_ref(schema))?,
        })?)
    }
}

impl<F: Json> Validate<F> for PropertyNamesObjectValidator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.node(&self.node)
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<F::StringBuffer>(),
            std::mem::size_of::<(&Self, &mut ValidationContext<'_>)>(),
            std::mem::size_of::<(&mut F::StringBuffer, &str, F::Node<'_>, bool)>(),
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            if !ctx.workspace.string_constructor(true) {
                return false;
            }
            let mut buffer = F::StringBuffer::default();
            for (name, _) in object.members() {
                let Some(valid) =
                    ctx.with_string_node::<F, _>(&mut buffer, name.as_ref(), |node, ctx| {
                        self.node.is_valid(&node, ctx)
                    })
                else {
                    return false;
                };
                if !valid {
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
            if !ctx.workspace.string_constructor(true) {
                return Ok(());
            }
            let mut buffer = F::StringBuffer::default();
            for (name, _) in object.members() {
                let result = ctx.with_string_node::<F, _>(
                    &mut buffer,
                    name.as_ref(),
                    |node, ctx| match self.node.validate(&node, location, tracker, ctx) {
                        Ok(()) => None,
                        Err(error) => ctx.produce(|funding| error.to_owned_with_funding(funding)),
                    },
                );
                if ctx.workspace.failed() {
                    return Ok(());
                }
                if let Some(Some(error)) = result {
                    let schema_path = error.schema_path().clone();
                    return ctx.diagnostic::<F>(
                        instance,
                        location,
                        tracker,
                        &schema_path,
                        |funding| {
                            Ok(crate::error::ValidationErrorKind::PropertyNames {
                                error: funding.boxed(error)?,
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
        if !ctx.workspace.string_constructor(true) {
            return;
        }
        let mut buffer = F::StringBuffer::default();
        for (name, _) in object.members() {
            let collected =
                ctx.with_string_node::<F, _>(&mut buffer, name.as_ref(), |node, ctx| {
                    let mut collected = Vec::new();
                    self.node
                        .collect_errors(&node, location, tracker, ctx, &mut collected);
                    ctx.produce(|funding| funding.owned_errors(collected))
                });
            let Some(Some(collected)) = collected else {
                return;
            };
            for error in collected {
                let schema_path = error.schema_path().clone();
                if let Err(error) =
                    ctx.diagnostic::<F>(instance, location, tracker, &schema_path, |funding| {
                        Ok(crate::error::ValidationErrorKind::PropertyNames {
                            error: funding.boxed(error)?,
                        })
                    })
                {
                    if !ctx.workspace.push(errors, error) {
                        return;
                    }
                }
                if ctx.workspace.failed() {
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
            let mut children = Vec::with_capacity(object.len());
            let mut buffer = F::StringBuffer::default();
            for (name, _) in object.members() {
                children.push(F::with_string_node(&mut buffer, name.as_ref(), |node| {
                    self.node.evaluate_instance(&node, location, tracker, ctx)
                }));
            }
            EvaluationResult::from_children(children)
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

pub(crate) struct PropertyNamesBooleanValidator {
    location: Location,
}

impl PropertyNamesBooleanValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(ctx: &compiler::Context<F>) -> CompilationResult<'a, F> {
        let location = ctx
            .location()
            .join_with_funding("propertyNames", ctx.funding())?;
        Ok(ctx
            .funding()
            .boxed(PropertyNamesBooleanValidator { location })?)
    }
}

impl<F: Json> Validate<F> for PropertyNamesBooleanValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.location(&self.location)
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[std::mem::size_of::<bool>()])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(object) = instance.as_object() {
            if !object.is_empty() {
                return false;
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
        if <Self as Validate<F>>::is_valid(self, instance, ctx) {
            Ok(())
        } else {
            ctx.diagnostic::<F>(instance, location, tracker, &self.location, |_| {
                Ok(crate::error::ValidationErrorKind::FalseSchema)
            })
        }
    }
}

#[inline]
pub(crate) fn compile<'a, F: Json>(
    ctx: &compiler::Context<F>,
    _: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    match schema {
        Value::Object(_) => Some(PropertyNamesObjectValidator::compile(ctx, schema)),
        Value::Bool(false) => Some(PropertyNamesBooleanValidator::compile(ctx)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::{json, Value};
    use test_case::test_case;

    // Each key is validated on its own, not concatenated with the ones before it
    #[test_case(&json!({"propertyNames": {"maxLength": 3}}), &json!({"abc": 1, "de": 1}))]
    #[test_case(&json!({"propertyNames": {"maxLength": 3}}), &json!({"a": 1, "bc": 1, "def": 1}))]
    fn keys_validate_independently(schema: &Value, instance: &Value) {
        tests_util::is_valid(schema, instance);
    }

    #[test_case(&json!({"propertyNames": {"maxLength": 2}}), &json!({"ab": 1, "cde": 1}))]
    #[test_case(&json!({"propertyNames": {"minLength": 2}}), &json!({"ab": 1, "c": 1}))]
    fn invalid_key_is_reported(schema: &Value, instance: &Value) {
        tests_util::is_not_valid(schema, instance);
    }

    #[test_case(&json!({"propertyNames": false}), &json!({"foo": 1}), "/propertyNames")]
    #[test_case(&json!({"propertyNames": {"minLength": 2}}), &json!({"f": 1}), "/propertyNames/minLength")]
    fn location(schema: &Value, instance: &Value, expected: &str) {
        tests_util::assert_schema_location(schema, instance, expected);
    }
}
