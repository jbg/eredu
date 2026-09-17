use crate::{
    compiler,
    error::ValidationError,
    keywords::CompilationResult,
    paths::{LazyLocation, Location, RefTracker},
    validator::{Validate, ValidationContext},
    Array, Json, Node,
};
use serde_json::{Map, Value};

fn has_unique_items<F: Json>(instance: &F::Node<'_>) -> bool {
    let Some(array) = instance.as_array() else {
        return true;
    };
    array.is_unique()
}

pub(crate) struct UniqueItemsValidator {
    location: Location,
}

impl UniqueItemsValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(location: Location) -> CompilationResult<'a, F> {
        Ok(Box::new(UniqueItemsValidator { location }))
    }
}

impl<F: Json> Validate<F> for UniqueItemsValidator {
    fn original_source(&self, source: &mut crate::validator::source::Inspector<F>) -> Result<(), crate::validator::workspace::Error> {
        source.location(&self.location)
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F,Self>(&[std::mem::size_of::<bool>()])
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if !ctx.workspace.original() { return has_unique_items::<F>(instance); }
        let Some(array) = instance.as_array() else { return true; };
        ctx.unique_items::<F>(&array)
    }

    fn validate<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        _ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if has_unique_items::<F>(instance) {
            Ok(())
        } else {
            Err(ValidationError::unique_items(
                self.location.clone(),
                crate::paths::capture_evaluation_path(tracker, &self.location),
                location.into(),
                instance.to_value(),
            ))
        }
    }
}

#[inline]
pub(crate) fn compile<'a, F: Json>(
    ctx: &compiler::Context<F>,
    _: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    if let Value::Bool(value) = schema {
        if *value {
            let location = ctx.location().join("uniqueItems");
            Some(UniqueItemsValidator::compile(location))
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::json;

    #[test]
    fn location() {
        tests_util::assert_schema_location(
            &json!({"uniqueItems": true}),
            &json!([1, 1]),
            "/uniqueItems",
        );
    }
}
