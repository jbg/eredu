use std::sync::Arc;
pub(crate) trait RegexEngine: Sized + Send + Sync {
    type Error: RegexError;
    fn original_source<F: crate::Json>(
        &self,
        _: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        Err(crate::validator::workspace::Error::Unqualified(
            crate::validator::workspace::Component::Source("selected regex retained source"),
        ))
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        Err(crate::validator::workspace::Error::Unqualified(
            crate::validator::workspace::Component::Validator("selected regex invocation"),
        ))
    }
    fn is_match(
        &self,
        text: &str,
        context: &mut crate::ValidationContext,
    ) -> Result<bool, Self::Error>;
}

/// Reason a regex match failed, distinguishing real engine errors from recovered panics.
#[derive(Debug)]
pub(crate) enum RegexFailureReason {
    /// Real `fancy-regex` runtime error (e.g., the configured backtrack limit was hit).
    FancyRegex(fancy_regex::Error),
    /// Engine panicked during matching and was recovered via `catch_unwind`.
    Panicked,
    /// A concrete delegated engine failure from scoped search.
    Delegate(crate::error::RegexEngineError),
}

pub(crate) trait RegexError {
    fn into_failure_reason(self) -> RegexFailureReason;
}

/// Failure mode for the `fancy-regex` backend: either a real engine error or a recovered panic.
#[derive(Debug)]
pub(crate) enum FancyRegexError {
    Engine(fancy_regex::Error),
    Panicked,
}

impl RegexError for FancyRegexError {
    fn into_failure_reason(self) -> RegexFailureReason {
        match self {
            Self::Engine(
                e @ fancy_regex::Error::RuntimeError(fancy_regex::RuntimeError::DelegateError(_)),
            ) => RegexFailureReason::Delegate(crate::error::RegexEngineError::Fancy(e)),
            Self::Engine(e) => RegexFailureReason::FancyRegex(e),
            Self::Panicked => RegexFailureReason::Panicked,
        }
    }
}

impl RegexEngine for Arc<fancy_regex::Regex> {
    type Error = FancyRegexError;
    fn original_source<F: crate::Json>(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        use crate::validator::workspace::{Component, Error};
        if !source.arc(self)? {
            return Ok(());
        }
        let mut failure = None;
        let result = self.visit_source_storage(&mut |identity, bytes| {
            if failure.is_some() {
                return false;
            }
            match source.allocation(identity, bytes) {
                Ok(descend) => descend,
                Err(error) => {
                    failure = Some(error);
                    false
                }
            }
        });
        if let Some(error) = failure {
            return Err(error);
        }
        result.map_err(|error| match error {
            fancy_regex::source_storage::Error::SizeOverflow => Error::Overflow,
            fancy_regex::source_storage::Error::WarmedPool => {
                Error::Unqualified(Component::Source("used persistent regex search pool"))
            }
        })
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        Ok(std::mem::size_of::<(&Self, &str, Result<bool, Self::Error>)>())
    }

    fn is_match(
        &self,
        text: &str,
        context: &mut crate::ValidationContext,
    ) -> Result<bool, Self::Error> {
        // `regex-automata` 0.4 panics on some patterns (https://github.com/rust-lang/regex/issues/1344); catch to surface a regular error instead of aborting the host process.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            context.workspace.fancy_match(self, text)
        })) {
            Ok(Ok(matched)) => Ok(matched),
            Ok(Err(e)) => Err(FancyRegexError::Engine(e)),
            Err(_) => Err(FancyRegexError::Panicked),
        }
    }
}

/// Failure from the selected standard engine, preserving the original cause.
#[derive(Debug)]
pub(crate) enum StandardRegexError {
    Engine(regex::SearchError),
    Panicked,
}
impl RegexError for StandardRegexError {
    fn into_failure_reason(self) -> RegexFailureReason {
        match self {
            Self::Engine(error) => {
                RegexFailureReason::Delegate(crate::error::RegexEngineError::Standard(error))
            }
            Self::Panicked => RegexFailureReason::Panicked,
        }
    }
}
impl RegexEngine for Arc<regex::Regex> {
    type Error = StandardRegexError;
    fn original_source<F: crate::Json>(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        use crate::validator::workspace::{Component, Error};
        if !source.arc(self)? {
            return Ok(());
        }
        let mut failure = None;
        let result = self.visit_source_storage(&mut |identity, bytes| {
            if failure.is_some() {
                return false;
            }
            match source.allocation(identity, bytes) {
                Ok(descend) => descend,
                Err(error) => {
                    failure = Some(error);
                    false
                }
            }
        });
        if let Some(error) = failure {
            return Err(error);
        }
        result.map_err(|error| match error {
            regex::source_storage::Error::SizeOverflow => Error::Overflow,
            regex::source_storage::Error::WarmedPool => {
                Error::Unqualified(Component::Source("used persistent regex search pool"))
            }
        })
    }

    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        Ok(std::mem::size_of::<(&Self, &str, Result<bool, Self::Error>)>())
    }
    fn is_match(
        &self,
        text: &str,
        context: &mut crate::ValidationContext,
    ) -> Result<bool, Self::Error> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            context.workspace.standard_match(self, text)
        })) {
            Ok(result) => result.map_err(StandardRegexError::Engine),
            Err(_) => Err(StandardRegexError::Panicked),
        }
    }
}

/// Uninhabited error type for literal matchers — matching never fails.
#[derive(Debug)]
pub(crate) enum LiteralMatchError {}

impl RegexError for LiteralMatchError {
    fn into_failure_reason(self) -> RegexFailureReason {
        match self {}
    }
}

/// [`RegexEngine`] for literal patterns — either `starts_with` (prefix) or `==` (exact).
pub(crate) enum LiteralMatcher {
    Prefix {
        literal: String,
    },
    Exact {
        exact: String,
    },
    /// `^(a|b|c)$` — linear scan over a small sorted array.
    Alternation {
        alternatives: Vec<String>,
    },
    /// `^\S*$` — no ECMA-262 whitespace characters.
    NoWhitespace,
}

impl RegexEngine for Arc<LiteralMatcher> {
    type Error = LiteralMatchError;
    fn original_source<F: crate::Json>(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        if !source.arc(self)? {
            return Ok(());
        }
        match self.as_ref() {
            LiteralMatcher::Prefix { literal } => source.string(literal),
            LiteralMatcher::Exact { exact } => source.string(exact),
            LiteralMatcher::Alternation { alternatives } => {
                source.vector(alternatives)?;
                for text in alternatives {
                    source.string(text)?;
                }
                Ok(())
            }
            LiteralMatcher::NoWhitespace => Ok(()),
        }
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        Ok(std::mem::size_of::<(
            &Self,
            &str,
            Result<bool, Self::Error>,
            std::str::Chars<'_>,
            std::slice::Iter<'_, String>,
        )>())
    }

    #[inline]
    fn is_match(
        &self,
        text: &str,
        _context: &mut crate::ValidationContext,
    ) -> Result<bool, Self::Error> {
        match self.as_ref() {
            LiteralMatcher::Prefix { literal } => Ok(text.starts_with(literal.as_str())),
            LiteralMatcher::Exact { exact } => Ok(text == exact.as_str()),
            LiteralMatcher::Alternation { alternatives } => {
                Ok(alternatives.iter().any(|a| a.as_str() == text))
            }
            LiteralMatcher::NoWhitespace => Ok(!text.chars().any(is_ecma_whitespace)),
        }
    }
}

/// Result of analyzing a regex pattern for literal-match optimizations.
#[derive(Debug, PartialEq)]
pub(crate) enum PatternOptimization {
    /// `^prefix` — use `starts_with(prefix)`.
    Prefix(String),
    /// `^exact$` — use `== exact`.
    Exact(String),
    /// `^(a|b|c)$` — linear scan over a small sorted array.
    Alternation(Vec<String>),
    /// `^\S*$` — no ECMA-262 whitespace characters.
    NoWhitespace,
}

pub(crate) use jsonschema_regex::is_ecma_whitespace;

/// Build a fancy-regex matcher, applying engine limits; `Err(())` on a rejected pattern.
pub(crate) fn build_fancy_regex(
    translated: &str,
    backtrack_limit: Option<usize>,
    size_limit: Option<usize>,
    dfa_size_limit: Option<usize>,
) -> Result<fancy_regex::Regex, ()> {
    build_fancy_regex_with_funding(
        translated,
        backtrack_limit,
        size_limit,
        dfa_size_limit,
        &crate::compilation::Funding::default(),
    )
    .map_err(|_| ())
}
pub(crate) fn build_fancy_regex_with_funding(
    translated: &str,
    backtrack_limit: Option<usize>,
    size_limit: Option<usize>,
    dfa_size_limit: Option<usize>,
    funding: &crate::compilation::Funding,
) -> Result<fancy_regex::Regex, crate::compilation::PatternError> {
    funding.reserve(
        std::mem::size_of::<fancy_regex::RegexOptionsBuilder>()
            + std::mem::size_of::<Result<fancy_regex::Regex, fancy_regex::Error>>(),
    )?;
    let mut builder = fancy_regex::RegexOptionsBuilder::new();
    if let Some(limit) = backtrack_limit {
        builder.backtrack_limit(limit);
    }
    if let Some(limit) = size_limit {
        builder.delegate_size_limit(limit);
    }
    if let Some(limit) = dfa_size_limit {
        builder.delegate_dfa_size_limit(limit);
    }
    builder
        .build_with_allocations(translated, funding)
        .map_err(|error| match error {
            fancy_regex::Error::Allocation(error) => {
                crate::compilation::PatternError::Storage(funding.syntax_allocation(error))
            }
            _ => crate::compilation::PatternError::Syntax,
        })
}

/// Build a standard-regex matcher, applying engine limits; `Err(())` on a rejected pattern.
pub(crate) fn build_standard_regex(
    translated: &str,
    size_limit: Option<usize>,
    dfa_size_limit: Option<usize>,
) -> Result<regex::Regex, ()> {
    build_standard_regex_with_funding(
        translated,
        size_limit,
        dfa_size_limit,
        &crate::compilation::Funding::default(),
    )
    .map_err(|_| ())
}
pub(crate) fn build_standard_regex_with_funding(
    translated: &str,
    size_limit: Option<usize>,
    dfa_size_limit: Option<usize>,
    funding: &crate::compilation::Funding,
) -> Result<regex::Regex, crate::compilation::PatternError> {
    funding.reserve(std::mem::size_of::<(
        regex::RegexBuilder,
        Result<regex::Regex, regex::Error>,
    )>())?;
    let failure = |error| match error {
        regex::Error::Allocation(error) => {
            crate::compilation::PatternError::Storage(funding.syntax_allocation(error.into()))
        }
        _ => crate::compilation::PatternError::Syntax,
    };
    let mut builder =
        regex::RegexBuilder::new_with_allocations(translated, funding).map_err(failure)?;
    if let Some(limit) = size_limit {
        builder.size_limit(limit);
    }
    if let Some(limit) = dfa_size_limit {
        builder.dfa_size_limit(limit);
    }
    builder.build_with_allocations(funding).map_err(failure)
}

/// Analyze a pattern and return a [`PatternOptimization`] if one applies, or `None` if a full
/// regex engine is required. Shares [`jsonschema_regex::analyze_pattern`] with the codegen backend.
pub(crate) fn analyze_pattern(pattern: &str) -> Option<PatternOptimization> {
    analyze_pattern_with_funding(pattern, &crate::compilation::Funding::default())
        .expect("ordinary pattern analysis")
}
pub(crate) fn analyze_pattern_with_funding(
    pattern: &str,
    funding: &crate::compilation::Funding,
) -> Result<Option<PatternOptimization>, crate::CompilationError> {
    let Some(analysis) = jsonschema_regex::analyze_pattern_with_allocations(pattern, funding)
        .map_err(|error| funding.syntax_allocation(error))?
    else {
        return Ok(None);
    };
    fn owned(
        text: std::borrow::Cow<'_, str>,
        funding: &crate::compilation::Funding,
    ) -> Result<String, crate::CompilationError> {
        match text {
            std::borrow::Cow::Owned(text) => Ok(text),
            std::borrow::Cow::Borrowed(text) => funding.copy_str(text),
        }
    }
    Ok(Some(match analysis {
        jsonschema_regex::PatternAnalysis::Prefix(prefix) => {
            PatternOptimization::Prefix(owned(prefix, funding)?)
        }
        jsonschema_regex::PatternAnalysis::Exact(exact) => {
            PatternOptimization::Exact(owned(exact, funding)?)
        }
        jsonschema_regex::PatternAnalysis::Alternation(alternatives) => {
            PatternOptimization::Alternation(alternatives)
        }
        jsonschema_regex::PatternAnalysis::NoWhitespace => PatternOptimization::NoWhitespace,
    }))
}

#[cfg(test)]
mod tests {
    use super::{analyze_pattern, PatternOptimization};
    use test_case::test_case;

    #[test_case(r"^\S*$", Some(PatternOptimization::NoWhitespace) ; "no whitespace sentinel")]
    #[test_case(
        r"^(get|put|post)$",
        Some(PatternOptimization::Alternation(vec!["get".into(), "post".into(), "put".into()])) ;
        "sorted alternation"
    )]
    #[test_case(r"^(a|b|c^)$", None ; "invalid char in alternative")]
    #[test_case(r"^(x-foo|x-bar)$", Some(PatternOptimization::Alternation(vec!["x-bar".into(), "x-foo".into()])) ; "alternation with dash")]
    #[test_case(r"^(single)$", Some(PatternOptimization::Alternation(vec!["single".into()])) ; "single alternative")]
    #[allow(clippy::needless_pass_by_value)]
    fn test_analyze_pattern_new(pattern: &str, expected: Option<PatternOptimization>) {
        assert_eq!(analyze_pattern(pattern), expected);
    }
}
