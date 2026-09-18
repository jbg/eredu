//! Canonical selected regex program, including closed checked workspace sources.
#[cfg(all(feature = "fancy-regex", not(feature = "onig")))]
use super::fancy::GeneralRegex;
#[cfg(feature = "onig")]
use super::onig::GeneralRegex;
use crate::tokenizer::pattern::coverage;
use crate::{Offsets, Result};
#[cfg(feature = "fancy-regex")]
use fancy_regex::workspace::{construction, Plan, PlanError, Source, Workspace};
use std::sync::Arc;

#[cfg(feature = "fancy-regex")]
#[derive(Debug)]
struct SourceOwner(Option<Arc<Source>>, Option<&'static str>);
#[cfg(feature = "fancy-regex")]
impl SourceOwner {
    fn source(&self) -> &Source {
        self.0.as_deref().expect("live regex source")
    }
}
#[cfg(feature = "fancy-regex")]
impl Clone for SourceOwner {
    fn clone(&self) -> Self {
        Self(
            Some(self.0.as_ref().expect("live regex source").clone()),
            self.1,
        )
    }
}
#[cfg(feature = "fancy-regex")]
impl Drop for SourceOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
#[derive(Debug, Clone)]
enum Engine {
    General(Arc<GeneralRegex>),
    #[cfg(feature = "fancy-regex")]
    Workspace(SourceOwner),
}
/// Immutable regex selected by syntax and engine features. Accepted exact fancy
/// patterns use the same closed source for ordinary and checked construction.
#[derive(Debug, Clone)]
pub struct SysRegex {
    engine: Engine,
}

/// Exact feature-selected word-class program. Oniguruma's pinned Unicode 16
/// table excludes join controls, while its Latin-1 fast path includes these
/// six numeric scalars in the positive shorthand only; its negated class
/// uses the Unicode property table. Preserve that language without invoking Onig in
/// the funded worker; the declared source spelling remains unchanged.
#[cfg(feature = "fancy-regex")]
const ONIG_WORD_PROGRAM: &str = r"[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\u{00B2}\u{00B3}\u{00B9}\u{00BC}\u{00BD}\u{00BE}]+|[^\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\s]+";
#[cfg(feature = "fancy-regex")]
const WORD_PATTERN: &str = r"\w+|[^\w\s]+";

impl SysRegex {
    pub fn new(pattern: &str) -> Result<Self> {
        #[cfg(feature = "fancy-regex")]
        if let Some(declared) = Self::declared_patterns().find(|source| *source == pattern) {
            let plan = Self::construction_plan(declared)?;
            return Self::compile_selected(plan, declared)
                .map_err(|error| Box::new(error) as crate::Error);
        }
        Ok(Self {
            engine: Engine::General(Arc::new(GeneralRegex::new(pattern)?)),
        })
    }
    #[cfg(feature = "fancy-regex")]
    pub(crate) fn compile(
        plan: construction::Plan<'_>,
    ) -> std::result::Result<Self, construction::Failure> {
        Ok(Self {
            engine: Engine::Workspace(SourceOwner(Some(Arc::new(plan.prepare()?)), None)),
        })
    }
    /// Programs used only to lower a feature-selected class are not new caller
    /// declarations. Keep ordinary engine acceptance and source identity exact.
    #[cfg(feature = "fancy-regex")]
    pub(crate) fn declared_patterns() -> impl Iterator<Item = &'static str> {
        construction::patterns().filter(|pattern| *pattern != ONIG_WORD_PROGRAM)
    }
    #[cfg(feature = "fancy-regex")]
    pub(crate) fn construction_plan(
        pattern: &'static str,
    ) -> std::result::Result<construction::Plan<'static>, construction::PlanError> {
        construction::Plan::new(if cfg!(feature = "onig") && pattern == WORD_PATTERN {
            ONIG_WORD_PROGRAM
        } else {
            pattern
        })
    }
    #[cfg(feature = "fancy-regex")]
    pub(crate) fn compile_selected(
        plan: construction::Plan<'_>,
        declared: &'static str,
    ) -> std::result::Result<Self, construction::Failure> {
        let mut regex = Self::compile(plan)?;
        if let Engine::Workspace(owner) = &mut regex.engine {
            owner.1 = Some(declared);
        }
        Ok(regex)
    }
    pub(crate) fn pattern(&self) -> &str {
        match &self.engine {
            Engine::General(regex) => regex.pattern(),
            #[cfg(feature = "fancy-regex")]
            Engine::Workspace(owner) => match owner.1 {
                Some(declared) => declared,
                None => owner.source().pattern(),
            },
        }
    }
    #[cfg(feature = "fancy-regex")]
    pub(crate) fn workspace_plan(&self) -> Option<std::result::Result<Plan<'_>, PlanError>> {
        match &self.engine {
            Engine::Workspace(owner) => Some(owner.source().plan()),
            Engine::General(_) => None,
        }
    }
    pub(crate) fn matcher(&self) -> Result<RegexMatcher<'_>> {
        let engine = match &self.engine {
            Engine::General(regex) => MatcherEngine::General(regex),
            #[cfg(feature = "fancy-regex")]
            Engine::Workspace(owner) => MatcherEngine::Workspace(std::cell::RefCell::new(
                owner
                    .source()
                    .plan()?
                    .prepare()
                    .map_err(|error| error.retire())?,
            )),
        };
        Ok(RegexMatcher { engine })
    }
    pub(crate) fn find_matches(&self, text: &str) -> Result<Vec<(Offsets, bool)>> {
        self.matcher()?.find_matches(text)
    }
    #[cfg(feature = "fancy-regex")]
    pub(crate) fn control_bytes() -> Option<usize> {
        use std::{alloc::Layout, mem::size_of, sync::atomic::AtomicUsize};
        let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Source>())
            .ok()?;
        [
            layout.pad_to_align().size(),
            size_of::<SourceOwner>(),
            size_of::<Engine>(),
            size_of::<Self>(),
            size_of::<&'static str>(),
            size_of::<Option<&'static str>>(),
            size_of::<Result<Self>>(),
            size_of::<std::result::Result<Self, construction::Failure>>(),
        ]
        .iter()
        .copied()
        .try_fold(0usize, usize::checked_add)
    }
}
enum MatcherEngine<'s> {
    General(&'s GeneralRegex),
    #[cfg(feature = "fancy-regex")]
    Workspace(std::cell::RefCell<Workspace<'s>>),
}
/// One operation reuses its workspace across every normalized input segment.
pub(crate) struct RegexMatcher<'s> {
    engine: MatcherEngine<'s>,
}
impl RegexMatcher<'_> {
    fn find_matches(&self, text: &str) -> Result<Vec<(Offsets, bool)>> {
        match &self.engine {
            MatcherEngine::General(regex) => collect(text, regex.find_iter(text)),
            #[cfg(feature = "fancy-regex")]
            MatcherEngine::Workspace(workspace) => collect(
                text,
                workspace
                    .borrow_mut()
                    .find_iter(text)
                    .map(|m| m.map(|m| (m.start(), m.end()))),
            ),
        }
    }
}
impl crate::tokenizer::pattern::Pattern for &RegexMatcher<'_> {
    fn find_matches(&self, text: &str) -> Result<Vec<(Offsets, bool)>> {
        RegexMatcher::find_matches(self, text)
    }
}
fn collect<I, E>(text: &str, matches: I) -> Result<Vec<(Offsets, bool)>>
where
    I: Iterator<Item = std::result::Result<Offsets, E>>,
    E: Into<crate::Error>,
{
    let mut splits = Vec::with_capacity(text.len());
    for item in coverage(text.len(), matches) {
        splits.push(item.map_err(Into::into)?);
    }
    Ok(splits)
}
#[cfg(feature = "fancy-regex")]
struct Matches<'w, 's, 't>(fancy_regex::workspace::Matches<'w, 's, 't>);
#[cfg(feature = "fancy-regex")]
impl Iterator for Matches<'_, '_, '_> {
    type Item = std::result::Result<Offsets, fancy_regex::Error>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|m| m.map(|m| (m.start(), m.end())))
    }
}
/// The same fallible coverage worker with an already admitted encoding workspace.
#[cfg(feature = "fancy-regex")]
pub(crate) fn visit_spans<E: From<fancy_regex::Error>>(
    workspace: &mut Workspace<'_>,
    text: &str,
    matches_only: bool,
    mut visit: impl FnMut(Offsets) -> std::result::Result<(), E>,
) -> std::result::Result<(), E> {
    for item in coverage(text.len(), Matches(workspace.find_iter(text))) {
        let (offsets, matched) = item.map_err(E::from)?;
        if !matches_only || matched {
            visit(offsets)?;
        }
    }
    Ok(())
}
#[cfg(feature = "fancy-regex")]
pub(crate) fn visit_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<crate::tokenizer::pattern::Coverage<Matches<'static, 'static, 'static>>>(),
        size_of::<Option<std::result::Result<(Offsets, bool), fancy_regex::Error>>>(),
        size_of::<Offsets>(),
    ]
    .iter()
    .copied()
    .try_fold(0usize, usize::checked_add)
}

#[cfg(all(test, feature = "fancy-regex"))]
mod tests {
    use super::*;
    use crate::pre_tokenizers::split::{Split, SplitPattern};
    use crate::{
        OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, SplitDelimiterBehavior,
    };
    fn reference(pattern: &str, text: &str) -> Vec<(Offsets, bool)> {
        if text.is_empty() {
            return vec![((0, 0), false)];
        }
        let regex = GeneralRegex::new(pattern).unwrap();
        collect(text, regex.find_iter(text)).unwrap()
    }
    #[test]
    fn all_checked_patterns_match_independent_general_engine_and_ordinary_serde() {
        for pattern in SysRegex::declared_patterns() {
            let compiled =
                SysRegex::compile_selected(SysRegex::construction_plan(pattern).unwrap(), pattern)
                    .unwrap();
            let split = Split::from_regex(compiled);
            let ordinary = Split::new(
                SplitPattern::Regex(pattern.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap();
            let serialized = serde_json::to_string(&split).unwrap();
            assert_eq!(serialized, serde_json::to_string(&ordinary).unwrap());
            let restored: Split = serde_json::from_str(&serialized).unwrap();
            for text in [
                "Hello WORLD's 1234567!\r\n αβé e\u{301} 中文🙂 ",
                "can't I'M\t ",
                "a\u{200c}b\u{200d}c",
                "a²b '²s a³b '³s a¹b '¹s a¼b '¼s a½b '½s a¾b '¾s",
                "",
                "  /\r\n",
            ] {
                let expected = reference(pattern, text);
                for value in [&split, &ordinary, &restored] {
                    let mut input = PreTokenizedString::from(text);
                    value.pre_tokenize(&mut input).unwrap();
                    let actual: Vec<_> = input
                        .get_splits(OffsetReferential::Original, OffsetType::Byte)
                        .into_iter()
                        .map(|(_, offsets, _)| offsets)
                        .collect();
                    assert_eq!(
                        actual,
                        expected
                            .iter()
                            .map(|&(offsets, _)| offsets)
                            .filter(|(a, b)| a != b)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    #[cfg(feature = "onig")]
    #[test]
    fn internal_word_program_does_not_expand_ordinary_onig_declarations() {
        assert!(GeneralRegex::new(ONIG_WORD_PROGRAM).is_err());
        assert!(SysRegex::new(ONIG_WORD_PROGRAM).is_err());
    }
    #[test]
    fn closed_program_aliases_and_repeated_workspaces_preserve_actual_source() {
        let pattern = construction::patterns().nth(2).unwrap();
        let source = SysRegex::compile(construction::Plan::new(pattern).unwrap()).unwrap();
        let alias = source.clone();
        let (Engine::Workspace(a), Engine::Workspace(b)) = (&source.engine, &alias.engine) else {
            panic!("checked workspace source");
        };
        assert!(std::ptr::eq(a.source(), b.source()));
        assert_eq!(Arc::strong_count(a.0.as_ref().unwrap()), 2);
        drop(source);
        let mut workspace = alias.workspace_plan().unwrap().unwrap().prepare().unwrap();
        let capacities = workspace.capacities();
        let text = "ABC's 123456 élève\r\n🙂";
        let expected: Vec<_> = reference(pattern, text)
            .into_iter()
            .map(|(offsets, _)| offsets)
            .collect();
        for _ in 0..100 {
            let mut actual = Vec::new();
            visit_spans::<fancy_regex::Error>(&mut workspace, text, false, |offsets| {
                actual.push(offsets);
                Ok(())
            })
            .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(workspace.capacities(), capacities);
        }
    }
    #[cfg(not(feature = "onig"))]
    #[test]
    fn ordinary_regex_execution_failure_is_not_successful_partial_tokenization() {
        let split = Split::new(
            SplitPattern::Regex("(?i)(a|b|ab)*(?>c)".to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .unwrap();
        let mut input =
            PreTokenizedString::from("abababababababababababababababababababababababababababab");
        let error = split.pre_tokenize(&mut input).unwrap_err();
        assert!(matches!(
            error.downcast_ref::<fancy_regex::Error>(),
            Some(fancy_regex::Error::RuntimeError(
                fancy_regex::RuntimeError::BacktrackLimitExceeded
            ))
        ));
    }
}

#[cfg(all(test, feature = "fancy-regex", not(feature = "onig")))]
#[test]
#[ignore = "release measurement against independent general regex engine"]
fn canonical_regex_performance() {
    use crate::pre_tokenizers::split::Split;
    use crate::{PreTokenizedString, PreTokenizer};
    use std::{
        hint::black_box,
        time::{Duration, Instant},
    };
    fn measure(split: &Split, text: &str) -> f64 {
        let start = Instant::now();
        let mut count = 0;
        while start.elapsed() < Duration::from_millis(300) {
            let mut input = PreTokenizedString::from(black_box(text));
            split.pre_tokenize(&mut input).unwrap();
            black_box(input);
            count += 1;
        }
        start.elapsed().as_secs_f64() * 1e6 / count as f64
    }
    for ordinal in [0, 2].iter().copied() {
        let pattern = construction::patterns().nth(ordinal).unwrap();
        let plan = construction::Plan::new(pattern).unwrap();
        let source_heap = plan.heap_bytes();
        let source = SysRegex::compile(plan).unwrap();
        let workspace = source
            .workspace_plan()
            .unwrap()
            .unwrap()
            .requirements()
            .required_bytes();
        let compiled = Split::from_regex(source);
        let reference = Split::from_regex(SysRegex {
            engine: Engine::General(Arc::new(GeneralRegex::new(pattern).unwrap())),
        });
        for (name, text) in [
            (
                "chat",
                "The assistant's answer has 12345 tokens.\n".repeat(512),
            ),
            ("unicode", "élève 東京🙂 १२३ e\u{301}\r\n".repeat(512)),
            ("whitespace", format!("{}x", " ".repeat(32768))),
        ]
        .iter()
        {
            let mut a = PreTokenizedString::from(text.as_str());
            let mut b = PreTokenizedString::from(text.as_str());
            compiled.pre_tokenize(&mut a).unwrap();
            reference.pre_tokenize(&mut b).unwrap();
            assert_eq!(a, b);
            let old = measure(&reference, text);
            let new = measure(&compiled, text);
            eprintln!("regex,{ordinal},{name},bytes={},general_us={old:.2},canonical_us={new:.2},source_heap={source_heap},workspace={workspace}", text.len());
        }
    }
}
