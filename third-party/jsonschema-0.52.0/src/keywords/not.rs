use crate::{
    compiler,
    error::ValidationError,
    keywords::CompilationResult,
    node::SchemaNode,
    paths::{LazyLocation, RefTracker},
    validator::{Validate, ValidationContext},
    Json, Node, SerdeJson,
};
use serde_json::{Map, Value};

pub(crate) struct NotValidator<F: Json> {
    // needed only for error representation
    original: Value,
    node: SchemaNode<F>,
}

impl NotValidator<SerdeJson> {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &compiler::Context<F>,
        schema: &'a Value,
    ) -> CompilationResult<'a, F> {
        let ctx = ctx.new_at_location("not")?;
        Ok(ctx.funding().boxed(NotValidator {
            original: ctx.funding().value(schema)?,
            node: compiler::compile(&ctx, ctx.as_resource_ref(schema))?,
        })?)
    }
}

impl<F: Json> Validate<F> for NotValidator<F> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.value(&self.original)?;
        source.node(&self.node)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[std::mem::size_of::<
            &crate::node::SchemaNode<F>,
        >()])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        !self.node.is_valid(instance, ctx)
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if self.is_valid(instance, ctx) {
            Ok(())
        } else {
            ctx.diagnostic::<F>(
                instance,
                location,
                tracker,
                self.node.location(),
                |funding| {
                    Ok(crate::error::ValidationErrorKind::Not {
                        schema: funding.value(&self.original)?,
                    })
                },
            )
        }
    }
}

#[inline]
pub(crate) fn compile<'a, F: Json>(
    ctx: &compiler::Context<F>,
    _: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    Some(NotValidator::compile(ctx, schema))
}

#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::json;

    #[test]
    fn location() {
        tests_util::assert_schema_location(
            &json!({"not": {"type": "string"}}),
            &json!("foo"),
            "/not",
        );
    }
}
