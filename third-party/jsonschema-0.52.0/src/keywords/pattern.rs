use std::{borrow::Cow, sync::Arc};

use crate::{
    compiler,
    error::ValidationError,
    keywords::CompilationResult,
    options::PatternEngineOptions,
    paths::{LazyEvaluationPath, LazyLocation, Location, RefTracker},
    regex::{
        analyze_pattern, is_ecma_whitespace, PatternOptimization, RegexEngine, RegexError,
        RegexFailureReason,
    },
    types::JsonType,
    validator::{Validate, ValidationContext},
    Json, Node,
};
use serde_json::{Map, Value};

/// Validator for patterns that are simple prefixes (optimized path).
pub(crate) struct PrefixPatternValidator {
    prefix: String,
    pattern: String,
    location: Location,
}

impl<F: Json> Validate<F> for PrefixPatternValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.string(&self.prefix)?;
        source.string(&self.pattern)?;
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[std::mem::size_of::<(
            Cow<'_, str>,
            &str,
            &str,
            usize,
            bool,
        )>()])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(item) = instance.as_string() {
            item.starts_with(&self.prefix)
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
        if <Self as Validate<F>>::is_valid(self, instance, ctx) {
            return Ok(());
        }
        ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
            Ok(crate::error::ValidationErrorKind::Pattern {
                pattern: funding.copy_str(&self.pattern)?,
            })
        })
    }
}

/// Validator for patterns that are exact-match anchored patterns.
pub(crate) struct ExactPatternValidator {
    exact: String,
    pattern: String,
    location: Location,
}

impl<F: Json> Validate<F> for ExactPatternValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.string(&self.exact)?;
        source.string(&self.pattern)?;
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[std::mem::size_of::<(
            Cow<'_, str>,
            &str,
            &str,
            usize,
            bool,
        )>()])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(item) = instance.as_string() {
            item.as_ref() == self.exact
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
        if <Self as Validate<F>>::is_valid(self, instance, ctx) {
            return Ok(());
        }
        ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
            Ok(crate::error::ValidationErrorKind::Pattern {
                pattern: funding.copy_str(&self.pattern)?,
            })
        })
    }
}

/// Validator for `^(a|b|c)$` alternation patterns (linear scan).
pub(crate) struct AlternationPatternValidator {
    alternatives: Vec<String>,
    pattern: String,
    location: Location,
}

impl<F: Json> Validate<F> for AlternationPatternValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(&self.alternatives)?;
        for alternative in &self.alternatives {
            source.string(alternative)?;
        }
        source.string(&self.pattern)?;
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::slice::Iter<'_, String>>(),
            std::mem::size_of::<(Cow<'_, str>, &Cow<'_, str>, &String, &str, bool)>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(item) = instance.as_string() {
            self.alternatives
                .iter()
                .any(|a| a.as_str() == item.as_ref())
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
        if <Self as Validate<F>>::is_valid(self, instance, ctx) {
            return Ok(());
        }
        ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
            Ok(crate::error::ValidationErrorKind::Pattern {
                pattern: funding.copy_str(&self.pattern)?,
            })
        })
    }
}

/// Validator for `^\S*$` — rejects any string containing ECMA-262 whitespace.
pub(crate) struct NoWhitespacePatternValidator {
    pattern: String,
    location: Location,
}

impl<F: Json> Validate<F> for NoWhitespacePatternValidator {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.string(&self.pattern)?;
        source.location(&self.location)
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            std::mem::size_of::<std::str::Chars<'_>>(),
            std::mem::size_of::<(Cow<'_, str>, char, Option<char>, bool, fn(char) -> bool)>(),
        ])
    }

    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }
    fn is_valid_body(&self, instance: &F::Node<'_>, _ctx: &mut ValidationContext) -> bool {
        if let Some(item) = instance.as_string() {
            !item.chars().any(is_ecma_whitespace)
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
        if <Self as Validate<F>>::is_valid(self, instance, ctx) {
            return Ok(());
        }
        ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
            Ok(crate::error::ValidationErrorKind::Pattern {
                pattern: funding.copy_str(&self.pattern)?,
            })
        })
    }
}

pub(crate) struct PatternValidator<R> {
    regex: R,
    /// Original schema pattern, kept for error messages. The compiled `regex` stores the
    /// ECMA->Rust translated form (e.g. `\S` expanded into a verbose class), which is unreadable.
    pattern: String,
    location: Location,
}

impl<R: RegexEngine, F: Json> Validate<F> for PatternValidator<R> {
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        self.regex.original_source(source)?;
        source.string(&self.pattern)?;
        source.location(&self.location)
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        crate::validator::workspace::body_controls::<F, Self>(&[
            self.regex.original_controls()?,
            std::mem::size_of::<(Cow<'_, str>, Result<bool, R::Error>)>(),
        ])
    }
    fn original_diagnostic_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        <Self as Validate<F>>::original_controls(self)
    }

    fn validate_body<'i>(
        &self,
        instance: &F::Node<'i>,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        let Some(item) = instance.as_string() else {
            return Ok(());
        };
        let reason = match self.regex.is_match(&item, ctx) {
            Ok(true) => return Ok(()),
            Ok(false) => None,
            Err(error) => Some(error.into_failure_reason()),
        };
        ctx.diagnostic::<F>(instance, location, tracker, &self.location, |funding| {
            Ok(match reason {
                None => crate::error::ValidationErrorKind::Pattern {
                    pattern: funding.copy_str(&self.pattern)?,
                },
                Some(RegexFailureReason::FancyRegex(error)) => {
                    crate::error::ValidationErrorKind::BacktrackLimitExceeded { error }
                }
                Some(reason @ (RegexFailureReason::Panicked | RegexFailureReason::Delegate(_))) => {
                    let error = match reason {
                        RegexFailureReason::Delegate(error) => Some(error),
                        _ => None,
                    };
                    crate::error::ValidationErrorKind::RegexEngineFailure {
                        error,
                        message: funding.format(format_args!(
                            "Regex engine failed to evaluate pattern '{}'",
                            self.pattern
                        ))?,
                    }
                }
            })
        })
    }

    fn is_valid_body(&self, instance: &F::Node<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(item) = instance.as_string() {
            return self.regex.is_match(&item, ctx).unwrap_or(false);
        }
        true
    }
}

#[inline]
pub(crate) fn compile<'a, F: Json>(
    ctx: &compiler::Context<F>,
    _: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a, F>> {
    if let Value::String(item) = schema {
        // Try literal optimizations before compiling a full regex.
        match crate::keywords::try_compile!(crate::regex::analyze_pattern_with_funding(
            item,
            ctx.funding()
        )) {
            Some(PatternOptimization::Exact(exact)) => {
                return Some(Ok(
                    match ctx.funding().boxed(ExactPatternValidator {
                        exact,
                        pattern: crate::keywords::try_compile!(ctx.funding().copy_str(item)),
                        location: crate::keywords::try_compile!(ctx
                            .location()
                            .join_with_funding("pattern", ctx.funding())),
                    }) {
                        Ok(value) => value,
                        Err(error) => return Some(Err(error.into())),
                    },
                ));
            }
            Some(PatternOptimization::Prefix(prefix)) => {
                return Some(Ok(
                    match ctx.funding().boxed(PrefixPatternValidator {
                        prefix,
                        pattern: crate::keywords::try_compile!(ctx.funding().copy_str(item)),
                        location: crate::keywords::try_compile!(ctx
                            .location()
                            .join_with_funding("pattern", ctx.funding())),
                    }) {
                        Ok(value) => value,
                        Err(error) => return Some(Err(error.into())),
                    },
                ));
            }
            Some(PatternOptimization::Alternation(alternatives)) => {
                return Some(Ok(
                    match ctx.funding().boxed(AlternationPatternValidator {
                        alternatives,
                        pattern: crate::keywords::try_compile!(ctx.funding().copy_str(item)),
                        location: crate::keywords::try_compile!(ctx
                            .location()
                            .join_with_funding("pattern", ctx.funding())),
                    }) {
                        Ok(value) => value,
                        Err(error) => return Some(Err(error.into())),
                    },
                ));
            }
            Some(PatternOptimization::NoWhitespace) => {
                return Some(Ok(
                    match ctx.funding().boxed(NoWhitespacePatternValidator {
                        pattern: crate::keywords::try_compile!(ctx.funding().copy_str(item)),
                        location: crate::keywords::try_compile!(ctx
                            .location()
                            .join_with_funding("pattern", ctx.funding())),
                    }) {
                        Ok(value) => value,
                        Err(error) => return Some(Err(error.into())),
                    },
                ));
            }
            None => {}
        }
        // Fall back to regex compilation
        match ctx.config().pattern_options() {
            PatternEngineOptions::FancyRegex { .. } => {
                let regex = match ctx.get_or_compile_regex(item) {
                    Ok(regex) => regex,
                    Err(crate::compilation::PatternError::Storage(error)) => {
                        return Some(Err(error.into()))
                    }
                    Err(error) => {
                        let location = crate::keywords::try_compile!(ctx
                            .location()
                            .join_with_funding("pattern", ctx.funding()));
                        return Some(Err(error.diagnostic(ctx.funding(), &location, schema)));
                    }
                };
                Some(Ok(
                    match ctx.funding().boxed(PatternValidator {
                        regex,
                        pattern: crate::keywords::try_compile!(ctx.funding().copy_str(item)),
                        location: crate::keywords::try_compile!(ctx
                            .location()
                            .join_with_funding("pattern", ctx.funding())),
                    }) {
                        Ok(value) => value,
                        Err(error) => return Some(Err(error.into())),
                    },
                ))
            }
            PatternEngineOptions::Regex { .. } => {
                let regex = match ctx.get_or_compile_standard_regex(item) {
                    Ok(regex) => regex,
                    Err(crate::compilation::PatternError::Storage(error)) => {
                        return Some(Err(error.into()))
                    }
                    Err(error) => {
                        let location = crate::keywords::try_compile!(ctx
                            .location()
                            .join_with_funding("pattern", ctx.funding()));
                        return Some(Err(error.diagnostic(ctx.funding(), &location, schema)));
                    }
                };
                Some(Ok(
                    match ctx.funding().boxed(PatternValidator {
                        regex,
                        pattern: crate::keywords::try_compile!(ctx.funding().copy_str(item)),
                        location: crate::keywords::try_compile!(ctx
                            .location()
                            .join_with_funding("pattern", ctx.funding())),
                    }) {
                        Ok(value) => value,
                        Err(error) => return Some(Err(error.into())),
                    },
                ))
            }
        }
    } else {
        let location = crate::keywords::try_compile!(ctx
            .location()
            .join_with_funding("pattern", ctx.funding()));
        Some(Err(crate::keywords::try_compile!(
            ValidationError::single_type_error_with_funding(
                location.clone(),
                location,
                crate::keywords::try_compile!(Location::new_with_funding(ctx.funding())),
                Cow::Borrowed(schema),
                JsonType::String,
                ctx.funding()
            )
        )
        .into()))
    }
}

#[cfg(test)]
mod tests {
    use crate::{tests_util, PatternOptions};
    use serde_json::json;
    use test_case::test_case;

    #[test_case("^(?!eo:)", "eo:bands", false)]
    #[test_case("^(?!eo:)", "proj:epsg", true)]
    fn negative_lookbehind_match(pattern: &str, text: &str, is_matching: bool) {
        let text = json!(text);
        let schema = json!({"pattern": pattern});
        let validator = crate::validator_for(&schema).unwrap();
        assert_eq!(validator.is_valid(&text), is_matching);
    }

    #[test]
    fn location() {
        tests_util::assert_schema_location(&json!({"pattern": "^f"}), &json!("b"), "/pattern");
    }

    #[test_case("^/", "/api/users", true)]
    #[test_case("^/", "api/users", false)]
    #[test_case("^x-", "x-custom-header", true)]
    #[test_case("^x-", "custom-header", false)]
    #[test_case("^foo", "foobar", true)]
    #[test_case("^foo", "barfoo", false)]
    #[test_case("^\\/", "/api/users", true; "escaped slash match")]
    #[test_case("^\\/", "api/users", false; "escaped slash no match")]
    fn prefix_pattern_optimization(pattern: &str, text: &str, is_matching: bool) {
        let text = json!(text);
        let schema = json!({"pattern": pattern});
        let validator = crate::validator_for(&schema).unwrap();
        assert_eq!(validator.is_valid(&text), is_matching);
    }

    #[test_case("^\\$ref$", "$ref", true; "dollar ref exact match")]
    #[test_case("^\\$ref$", "$refs", false; "dollar ref suffix no match")]
    #[test_case("^\\$ref$", "ref", false; "dollar ref no dollar no match")]
    #[test_case("^\\$ref$", "$ref_", false; "dollar ref trailing no match")]
    fn exact_pattern_optimization(pattern: &str, text: &str, is_matching: bool) {
        let text = json!(text);
        let schema = json!({"pattern": pattern});
        let validator = crate::validator_for(&schema).unwrap();
        assert_eq!(validator.is_valid(&text), is_matching);
        assert_eq!(validator.validate(&text).is_ok(), is_matching);
    }

    #[test_case(r"^(get|put|post)$", "get", true ; "alternation match get")]
    #[test_case(r"^(get|put|post)$", "put", true ; "alternation match put")]
    #[test_case(r"^(get|put|post)$", "post", true ; "alternation match post")]
    #[test_case(r"^(get|put|post)$", "patch", false ; "alternation no match")]
    #[test_case(r"^(get|put|post)$", "GET", false ; "alternation case sensitive")]
    fn alternation_pattern_optimization(pattern: &str, text: &str, is_matching: bool) {
        let text = json!(text);
        let schema = json!({"pattern": pattern});
        let validator = crate::validator_for(&schema).unwrap();
        assert_eq!(validator.is_valid(&text), is_matching);
        assert_eq!(validator.validate(&text).is_ok(), is_matching);
    }

    #[test_case(r"^\S*$", "hello", true ; "no whitespace match")]
    #[test_case(r"^\S*$", "hello world", false ; "no whitespace space fail")]
    #[test_case(r"^\S*$", "hello\tworld", false ; "no whitespace tab fail")]
    #[test_case(r"^\S*$", "", true ; "no whitespace empty string")]
    fn no_whitespace_pattern_optimization(pattern: &str, text: &str, is_matching: bool) {
        let text = json!(text);
        let schema = json!({"pattern": pattern});
        let validator = crate::validator_for(&schema).unwrap();
        assert_eq!(validator.is_valid(&text), is_matching);
        assert_eq!(validator.validate(&text).is_ok(), is_matching);
    }

    // Error messages must show the original schema pattern, not the ECMA->Rust translated form
    // (e.g. `\S` expanded into a verbose Unicode class).
    fn assert_original_pattern_in_error(validator: &crate::Validator) {
        let instance = json!("");
        let error = validator
            .validate(&instance)
            .expect_err("expected a validation error");
        assert_eq!(error.to_string(), r#""" does not match "^[\S]{1,5}$""#);
    }

    #[test]
    fn original_pattern_in_error() {
        let schema = json!({"pattern": r"^[\S]{1,5}$"});
        assert_original_pattern_in_error(
            &crate::options()
                .with_pattern_options(PatternOptions::fancy_regex())
                .build(&schema)
                .expect("Schema should be valid"),
        );
        assert_original_pattern_in_error(
            &crate::options()
                .with_pattern_options(PatternOptions::regex())
                .build(&schema)
                .expect("Schema should be valid"),
        );
    }

    #[test]
    fn test_regex_engine_validation() {
        let schema = json!({"pattern": "^[a-z]+$"});
        let validator = crate::options()
            .with_pattern_options(PatternOptions::regex())
            .build(&schema)
            .expect("Schema should be valid");

        let valid = json!("hello");
        assert!(validator.is_valid(&valid));
        let invalid = json!("Hello123");
        assert!(!validator.is_valid(&invalid));
    }

    // `catch_unwind` is a no-op under `panic = "abort"` (e.g. the wasm32-wasip1 default), so the
    // recovery path can't be exercised there.
    #[cfg(panic = "unwind")]
    #[test]
    fn empty_string_with_large_bounded_quantifier_fancy_regex() {
        // Recovery for https://github.com/rust-lang/regex/issues/1344.
        let schema = json!({"type": "string", "pattern": r"^.{0,404600}$"});
        let validator = crate::options()
            .with_pattern_options(PatternOptions::fancy_regex().size_limit(1_000_000_000))
            .build(&schema)
            .expect("Schema should be valid");

        assert!(validator.is_valid(&json!("x")));
        assert!(!validator.is_valid(&json!("")));
        let empty = json!("");
        let error = validator
            .validate(&empty)
            .expect_err("expected a validation error");
        assert_eq!(
            error.to_string(),
            "Regex engine failed to evaluate pattern '^.{0,404600}$'"
        );
    }

    #[test]
    fn fancy_regex_backtrack_limit_exceeded() {
        let schema = json!({"type": "string", "pattern": r"(?<=ab)c"});
        let validator = crate::options()
            .with_pattern_options(PatternOptions::fancy_regex().backtrack_limit(1))
            .build(&schema)
            .expect("Schema should be valid");

        let instance = json!("abc");
        let error = validator
            .validate(&instance)
            .expect_err("expected a validation error");
        assert!(
            matches!(
                error.kind(),
                crate::error::ValidationErrorKind::BacktrackLimitExceeded { .. }
            ),
            "expected BacktrackLimitExceeded, got {:?}",
            error.kind()
        );
        assert_eq!(
            error.to_string(),
            "Error executing regex: Max limit for backtracking count exceeded"
        );
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn empty_string_with_large_bounded_quantifier_regex() {
        // Recovery for https://github.com/rust-lang/regex/issues/1344.
        let schema = json!({"type": "string", "pattern": r"^.{0,404600}$"});
        let validator = crate::options()
            .with_pattern_options(PatternOptions::regex().size_limit(1_000_000_000))
            .build(&schema)
            .expect("Schema should be valid");

        assert!(validator.is_valid(&json!("x")));
        assert!(!validator.is_valid(&json!("")));
        let empty = json!("");
        let error = validator
            .validate(&empty)
            .expect_err("expected a validation error");
        assert_eq!(
            error.to_string(),
            "Regex engine failed to evaluate pattern '^.{0,404600}$'"
        );
    }
}
