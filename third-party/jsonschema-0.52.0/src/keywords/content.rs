//! Validators for `contentMediaType`, `contentEncoding`, and `contentSchema` keywords.
use crate::{
    compiler,
    content_encoding::{ContentEncodingSource, ConversionError},
    content_media_type::ContentMediaTypeSource,
    error::{ValidationError, ValidationErrorKind},
    evaluation::Annotations,
    keywords::CompilationResult,
    paths::{LazyLocation, Location, RefTracker},
    types::JsonType,
    validator::{EvaluationResult, Validate, ValidationContext},
    Json, Node,
};
use serde_json::{Map, Value};
use std::{borrow::Cow, sync::Arc};

/// Validator for `contentMediaType` keyword.
pub(crate) struct ContentMediaTypeValidator {
    media_type: String,
    func: ContentMediaTypeSource,
    location: Location,
}

impl ContentMediaTypeValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        media_type: &'a str,
        func: ContentMediaTypeSource,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(ContentMediaTypeValidator {
            media_type: ctx.funding().copy_str(media_type)?,
            func,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for ContentMediaTypeValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        self.func.qualify()?;
        source.string(&self.media_type)?;
        source.location(&self.location)
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        self.func.qualify()?;
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<Option<Cow<'_, str>>>(),
            std::mem::size_of::<(ContentMediaTypeSource, bool)>(),
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        instance
            .as_string()
            .map_or(true, |item| self.func.check(&item, ctx))
    }
    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if !<Self as Validate<F>>::is_valid(self, instance, ctx) {
            return ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
                Ok(ValidationErrorKind::ContentMediaType {
                    content_media_type: funding.copy_str(&self.media_type)?,
                })
            });
        }
        Ok(())
    }
}

/// Validator for `contentEncoding` keyword.
pub(crate) struct ContentEncodingValidator {
    encoding: String,
    func: ContentEncodingSource,
    location: Location,
}

impl ContentEncodingValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        encoding: &'a str,
        func: ContentEncodingSource,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(ContentEncodingValidator {
            encoding: ctx.funding().copy_str(encoding)?,
            func,
            location,
        })?)
    }
}

impl<F: Json> Validate<F> for ContentEncodingValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        self.func.qualify()?;
        source.string(&self.encoding)?;
        source.location(&self.location)
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        self.func.qualify()?;
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<Option<Cow<'_, str>>>(),
            std::mem::size_of::<(ContentEncodingSource, bool)>(),
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        instance
            .as_string()
            .map_or(true, |item| self.func.check(&item, ctx))
    }
    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if !<Self as Validate<F>>::is_valid(self, instance, ctx) {
            return ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
                Ok(ValidationErrorKind::ContentEncoding {
                    content_encoding: funding.copy_str(&self.encoding)?,
                })
            });
        }
        Ok(())
    }
}

/// Combined validator for both `contentEncoding` and `contentMediaType` keywords.
pub(crate) struct ContentMediaTypeAndEncodingValidator {
    media_type: String,
    encoding: String,
    func: ContentMediaTypeSource,
    converter: ContentEncodingSource,
    location: Location,
}

impl ContentMediaTypeAndEncodingValidator {
    #[inline]
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        media_type: &'a str,
        encoding: &'a str,
        func: ContentMediaTypeSource,
        converter: ContentEncodingSource,
        location: Location,
    ) -> CompilationResult<'a, F> {
        Ok(ctx.funding().boxed(ContentMediaTypeAndEncodingValidator {
            media_type: ctx.funding().copy_str(media_type)?,
            encoding: ctx.funding().copy_str(encoding)?,
            func,
            converter,
            location,
        })?)
    }
}

/// Decode the input value and check the selected media type with the same workers.
impl<F: Json> Validate<F> for ContentMediaTypeAndEncodingValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        self.func.qualify()?;
        self.converter.qualify()?;
        source.string(&self.media_type)?;
        source.string(&self.encoding)?;
        source.location(&self.location)
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        self.func.qualify()?;
        self.converter.qualify()?;
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<Option<Cow<'_, str>>>(),
            std::mem::size_of::<Option<Result<Option<String>, ConversionError>>>(),
            std::mem::size_of::<(ContentMediaTypeSource, ContentEncodingSource, bool)>(),
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)?
            .checked_add(std::mem::size_of::<Location>())
            .ok_or(crate::validator::workspace::Error::Overflow)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(item) = instance.as_string() {
            match self.converter.convert(&item, ctx) {
                Some(Ok(Some(converted))) => self.func.check(&converted, ctx),
                _ => false,
            }
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
        if let Some(item) = instance.as_string() {
            let Some(converted) = self.converter.convert(&item, ctx) else {
                return Ok(());
            };
            match converted {
                Ok(None) => {
                    let Some(path) = ctx.produce(|funding| {
                        self.location.join_with_funding("contentEncoding", funding)
                    }) else {
                        return Ok(());
                    };
                    return ctx.diagnostic::<F>(instance, location, tracker, &path, |funding| {
                        Ok(ValidationErrorKind::ContentEncoding {
                            content_encoding: funding.copy_str(&self.encoding)?,
                        })
                    });
                }
                Ok(Some(converted)) => {
                    if !self.func.check(&converted, ctx) {
                        let Some(path) = ctx.produce(|funding| {
                            self.location.join_with_funding("contentMediaType", funding)
                        }) else {
                            return Ok(());
                        };
                        return ctx.diagnostic::<F>(
                            instance,
                            location,
                            tracker,
                            &path,
                            |funding| {
                                Ok(ValidationErrorKind::ContentMediaType {
                                    content_media_type: funding.copy_str(&self.media_type)?,
                                })
                            },
                        );
                    }
                }
                Err(error) => {
                    let Some(path) = ctx.produce(|funding| {
                        self.location.join_with_funding("contentEncoding", funding)
                    }) else {
                        return Ok(());
                    };
                    return ctx.diagnostic::<F>(instance, location, tracker, &path, |_| {
                        Ok(match error {
                            ConversionError::Utf8(error) => ValidationErrorKind::FromUtf8 { error },
                            ConversionError::Schema(error) => error.into_parts().kind,
                        })
                    });
                }
            }
        }
        Ok(())
    }
}

#[inline]
pub(crate) fn compile_media_type<'a, F: Json>(
    ctx: &compiler::Context<F>,
    schema: &'a Map<String, Value>,
    subschema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    if let Value::String(media_type) = subschema {
        let func = ctx.get_content_media_type_check(media_type.as_str())?;
        if let Some(content_encoding) = schema.get("contentEncoding") {
            if let Value::String(content_encoding) = content_encoding {
                let converter = ctx.get_content_encoding(content_encoding)?;
                Some(ContentMediaTypeAndEncodingValidator::compile(
                    ctx,
                    media_type,
                    content_encoding,
                    func,
                    converter,
                    ctx.location().clone(),
                ))
            } else {
                let location = crate::keywords::try_compile!(ctx
                    .location()
                    .join_with_funding("contentEncoding", ctx.funding()));
                Some(Err(crate::keywords::try_compile!(
                    ValidationError::single_type_error_with_funding(
                        location.clone(),
                        location,
                        crate::keywords::try_compile!(Location::new_with_funding(ctx.funding())),
                        Cow::Borrowed(content_encoding),
                        JsonType::String,
                        ctx.funding()
                    )
                )
                .into()))
            }
        } else {
            Some(ContentMediaTypeValidator::compile(
                ctx,
                media_type,
                func,
                crate::keywords::try_compile!(ctx
                    .location()
                    .join_with_funding("contentMediaType", ctx.funding())),
            ))
        }
    } else {
        let location = crate::keywords::try_compile!(ctx
            .location()
            .join_with_funding("contentMediaType", ctx.funding()));
        Some(Err(crate::keywords::try_compile!(
            ValidationError::single_type_error_with_funding(
                location.clone(),
                location,
                crate::keywords::try_compile!(Location::new_with_funding(ctx.funding())),
                Cow::Borrowed(subschema),
                JsonType::String,
                ctx.funding()
            )
        )
        .into()))
    }
}

#[inline]
pub(crate) fn compile_content_encoding<'a, F: Json>(
    ctx: &compiler::Context<F>,
    schema: &'a Map<String, Value>,
    subschema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    // Performed during media type validation
    if schema.get("contentMediaType").is_some() {
        // TODO. what if media type is not supported?
        return None;
    }
    if let Value::String(content_encoding) = subschema {
        let func = ctx.get_content_encoding(content_encoding)?;
        Some(ContentEncodingValidator::compile(
            ctx,
            content_encoding,
            func,
            crate::keywords::try_compile!(ctx
                .location()
                .join_with_funding("contentEncoding", ctx.funding())),
        ))
    } else {
        let location = crate::keywords::try_compile!(ctx
            .location()
            .join_with_funding("contentEncoding", ctx.funding()));
        Some(Err(crate::keywords::try_compile!(
            ValidationError::single_type_error_with_funding(
                location.clone(),
                location,
                crate::keywords::try_compile!(Location::new_with_funding(ctx.funding())),
                Cow::Borrowed(subschema),
                JsonType::String,
                ctx.funding()
            )
        )
        .into()))
    }
}

/// Annotation-only validator for `contentMediaType` (Draft 2019-09 / 2020-12).
///
/// Per spec, annotations are only produced for string instances.
pub(crate) struct ContentMediaTypeAnnotationValidator {
    annotation: Arc<Value>,
}

impl ContentMediaTypeAnnotationValidator {
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        _ctx: &compiler::Context<F>,
        _schema: &'a Map<String, Value>,
        subschema: &'a Value,
    ) -> Option<CompilationResult<'a, F>> {
        if let Value::String(_) = subschema {
            Some(Ok(
                match ctx.funding().boxed(ContentMediaTypeAnnotationValidator {
                    annotation: crate::keywords::try_compile!(ctx.funding().annotation(subschema)),
                }) {
                    Ok(value) => value,
                    Err(error) => return Some(Err(error.into())),
                },
            ))
        } else {
            None
        }
    }
}

impl<F: Json> Validate<F> for ContentMediaTypeAnnotationValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        if source.arc(&self.annotation)? {
            source.value(&self.annotation)?;
        }
        Ok(())
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        // The selected ordinary assertion body is the existing constant true.
        // Annotation emission belongs to the separate ordinary evaluate API.
        crate::validator::workspace::body_controls::<F, Self>(&[std::mem::size_of::<bool>()])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, _instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        true
    }

    fn validate_body<'i>(
        &self,
        _instance: &F::Node<'i>,
        _location: &LazyLocation,
        _tracker: Option<&RefTracker>,
        _ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        Ok(())
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        _location: &LazyLocation,
        _tracker: Option<&RefTracker>,
        _ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if instance.is_string() {
            let mut result = EvaluationResult::valid_empty();
            result.annotate(Annotations::from_arc(Arc::clone(&self.annotation)));
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

pub(crate) fn compile_media_type_annotation<'a, F: Json>(
    ctx: &compiler::Context<F>,
    schema: &'a Map<String, Value>,
    subschema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    ContentMediaTypeAnnotationValidator::compile(ctx, ctx, schema, subschema)
}

/// Annotation-only validator for `contentEncoding` (Draft 2019-09 / 2020-12).
///
/// Per spec, annotations are only produced for string instances.
pub(crate) struct ContentEncodingAnnotationValidator {
    annotation: Arc<Value>,
}

impl ContentEncodingAnnotationValidator {
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        _ctx: &compiler::Context<F>,
        _schema: &'a Map<String, Value>,
        subschema: &'a Value,
    ) -> Option<CompilationResult<'a, F>> {
        if let Value::String(_) = subschema {
            Some(Ok(
                match ctx.funding().boxed(ContentEncodingAnnotationValidator {
                    annotation: crate::keywords::try_compile!(ctx.funding().annotation(subschema)),
                }) {
                    Ok(value) => value,
                    Err(error) => return Some(Err(error.into())),
                },
            ))
        } else {
            None
        }
    }
}

impl<F: Json> Validate<F> for ContentEncodingAnnotationValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        if source.arc(&self.annotation)? {
            source.value(&self.annotation)?;
        }
        Ok(())
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        // The selected ordinary assertion body is the existing constant true.
        // Annotation emission belongs to the separate ordinary evaluate API.
        crate::validator::workspace::body_controls::<F, Self>(&[std::mem::size_of::<bool>()])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, _instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        true
    }

    fn validate_body<'i>(
        &self,
        _instance: &F::Node<'i>,
        _location: &LazyLocation,
        _tracker: Option<&RefTracker>,
        _ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        Ok(())
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        _location: &LazyLocation,
        _tracker: Option<&RefTracker>,
        _ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if instance.is_string() {
            let mut result = EvaluationResult::valid_empty();
            result.annotate(Annotations::from_arc(Arc::clone(&self.annotation)));
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

pub(crate) fn compile_content_encoding_annotation<'a, F: Json>(
    ctx: &compiler::Context<F>,
    schema: &'a Map<String, Value>,
    subschema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    ContentEncodingAnnotationValidator::compile(ctx, ctx, schema, subschema)
}

/// Annotation-only validator for `contentSchema` (Draft 2019-09 / 2020-12).
///
/// Per spec, the annotation is only produced when the instance is a string AND
/// `contentMediaType` is also present in the same schema object.
pub(crate) struct ContentSchemaAnnotationValidator {
    annotation: Arc<Value>,
}

impl ContentSchemaAnnotationValidator {
    pub(crate) fn compile<'a, F: Json>(
        ctx: &crate::compiler::Context<F>,
        _ctx: &compiler::Context<F>,
        schema: &'a Map<String, Value>,
        subschema: &'a Value,
    ) -> Option<CompilationResult<'a, F>> {
        // contentSchema only annotates when contentMediaType is also present
        if schema.contains_key("contentMediaType") {
            Some(Ok(
                match ctx.funding().boxed(ContentSchemaAnnotationValidator {
                    annotation: crate::keywords::try_compile!(ctx.funding().annotation(subschema)),
                }) {
                    Ok(value) => value,
                    Err(error) => return Some(Err(error.into())),
                },
            ))
        } else {
            None
        }
    }
}

impl<F: Json> Validate<F> for ContentSchemaAnnotationValidator {
    fn is_valid_body(&self, _instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        true
    }

    fn validate_body<'i>(
        &self,
        _instance: &F::Node<'i>,
        _location: &LazyLocation,
        _tracker: Option<&RefTracker>,
        _ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        Ok(())
    }

    fn evaluate(
        &self,
        instance: &F::Node<'_>,
        _location: &LazyLocation,
        _tracker: Option<&RefTracker>,
        _ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        if instance.is_string() {
            let mut result = EvaluationResult::valid_empty();
            result.annotate(Annotations::from_arc(Arc::clone(&self.annotation)));
            result
        } else {
            EvaluationResult::valid_empty()
        }
    }
}

pub(crate) fn compile_content_schema_annotation<'a, F: Json>(
    ctx: &compiler::Context<F>,
    schema: &'a Map<String, Value>,
    subschema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    ContentSchemaAnnotationValidator::compile(ctx, ctx, schema, subschema)
}

#[cfg(test)]
mod tests {
    use referencing::Draft;
    use serde_json::{json, Value};
    use test_case::test_case;

    #[test_case(&json!({"contentEncoding": "base64"}), &json!("asd"), "/contentEncoding")]
    #[test_case(&json!({"contentMediaType": "application/json"}), &json!("asd"), "/contentMediaType")]
    #[test_case(&json!({"contentMediaType": "application/json", "contentEncoding": "base64"}), &json!("ezp9Cg=="), "/contentMediaType")]
    #[test_case(&json!({"contentMediaType": "application/json", "contentEncoding": "base64"}), &json!("{}"), "/contentEncoding")]
    fn location(schema: &Value, instance: &Value, expected: &str) {
        let validator = crate::options()
            .with_draft(Draft::Draft7)
            .build(schema)
            .expect("Invalid schema");
        let error = validator.validate(instance).expect_err("Should fail");
        assert_eq!(error.schema_path().as_str(), expected);
    }

    #[test]
    fn invalid_utf8_after_base64_decode_has_content_encoding_location() {
        let schema = json!({
            "properties": {
                "data": {
                    "contentMediaType": "application/json",
                    "contentEncoding": "base64"
                }
            }
        });
        // "//4=" decodes to 0xFF 0xFE, which is not valid UTF-8
        let instance = json!({"data": "//4="});
        let validator = crate::options()
            .with_draft(Draft::Draft7)
            .build(&schema)
            .expect("Invalid schema");
        let error = validator.validate(&instance).expect_err("Should fail");
        assert_eq!(error.instance_path().as_str(), "/data");
        assert_eq!(
            error.schema_path().as_str(),
            "/properties/data/contentEncoding"
        );
        assert_eq!(error.instance().as_ref(), &json!("//4="));
        assert!(
            matches!(
                error.kind(),
                crate::error::ValidationErrorKind::FromUtf8 { .. }
            ),
            "expected FromUtf8, got {:?}",
            error.kind()
        );
    }
}
