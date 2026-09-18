use crate::{
    compiler,
    error::ValidationError,
    keywords::{
        helpers::{fail_on_non_positive_integer, size_limit},
        CompilationResult,
    },
    paths::{LazyLocation, Location, RefTracker},
    validator::{Validate, ValidationContext},
    Array, Json, Node,
};
use serde_json::{Map, Value};

pub(crate) struct MinItemsValidator {
    limit: u64,
    location: Location,
}

impl MinItemsValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &compiler::Context<F>,
        schema: &'a Value,
        location: Location,
    ) -> CompilationResult<'a, F> {
        let Some(limit) = size_limit(ctx, schema) else {
            return Err(fail_on_non_positive_integer(schema, location).into());
        };
        Ok(ctx.funding().boxed(MinItemsValidator { limit, location })?)
    }
}

impl<F: Json> Validate<F> for MinItemsValidator {
    fn original_source(&self, source: &mut crate::validator::source::Inspector<F>) -> Result<(), crate::validator::workspace::Error> {
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<(usize, u64)>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(array) = instance.as_array() {
            if (array.len() as u64) < self.limit {
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
        if let Some(array) = instance.as_array() {
            if (array.len() as u64) < self.limit {
                return ctx.diagnostic::<F>(instance, location, tracker, &self.location, |_| Ok(crate::error::ValidationErrorKind::MinItems { limit: self.limit }));
            }
        }
        Ok(())
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
    let location = crate::keywords::try_compile!(ctx.location().join_with_funding("minItems", ctx.funding()));
    Some(MinItemsValidator::compile(ctx, schema, location))
}

#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::json;

    #[test]
    fn location() {
        tests_util::assert_schema_location(&json!({"minItems": 1}), &json!([]), "/minItems");
    }
}
