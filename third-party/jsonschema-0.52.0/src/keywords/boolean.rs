use crate::paths::{LazyLocation, Location, RefTracker};

use crate::{
    error::ValidationError,
    keywords::CompilationResult,
    validator::{Validate, ValidationContext},
    Json, Node,
};

pub(crate) struct FalseValidator {
    location: Location,
}
impl FalseValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(ctx: &crate::compiler::Context<F>, location: Location) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(FalseValidator { location })?)
    }
}
impl<F: Json> Validate<F> for FalseValidator {
    fn original_source(&self, source: &mut crate::validator::source::Inspector<F>) -> Result<(), crate::validator::workspace::Error> {
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, _: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        false
    }

    fn validate_body<'i>(
        &self, instance: &F::Node<'i>, location: &LazyLocation, tracker: Option<&RefTracker>, ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        ctx.diagnostic::<F>(instance, location, tracker, &self.location, |_| Ok(crate::error::ValidationErrorKind::FalseSchema))
    }

}

#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::json;

    #[test]
    fn location() {
        tests_util::assert_schema_location(&json!(false), &json!(1), "");
    }
}
