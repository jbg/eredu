//! Closed fresh-regex implementation of the existing Isolated Split semantics.
//! Ordinary JSON deserialization still constructs `Split` and its chosen engine.
use crate::tokenizer::pattern::{coverage, Coverage};
use crate::{Offsets, PreTokenizedString, PreTokenizer, SplitDelimiterBehavior};
use fancy_regex::workspace::{construction, Plan, PlanError, Source, Workspace};
use serde::{Serialize, Serializer};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::{atomic::AtomicUsize, Arc},
};

// All strong aliases use this owner. No raw Arc/Weak/source extraction exists.
struct SourceOwner(Option<Arc<Source>>);
impl SourceOwner {
    fn source(&self) -> &Source {
        self.0.as_deref().expect("live compiled split")
    }
}
impl Clone for SourceOwner {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live compiled split"),
        )))
    }
}
impl Drop for SourceOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

/// Freshly compiled exact default regex with Isolated/noninverted Split policy.
/// Construction is crate-private and consumes the aggregate's checked plan.
/// No mutable configuration, raw source, Arc or Weak is exposed.
///
/// ```compile_fail
/// use tokenizers::pre_tokenizers::compiled_split::CompiledRegexSplit;
/// fn escape(split: &CompiledRegexSplit) { let _ = split.source(); }
/// ```
#[derive(Clone)]
pub struct CompiledRegexSplit {
    owner: SourceOwner,
}
impl CompiledRegexSplit {
    pub(crate) fn compile(plan: construction::Plan<'_>) -> Result<Self, construction::Failure> {
        let source = plan.prepare()?;
        Ok(Self {
            owner: SourceOwner(Some(Arc::new(source))),
        })
    }
    /// Borrow the original full regex spelling; no new source is constructed.
    pub fn pattern(&self) -> &str {
        self.owner.source().pattern()
    }
    pub(crate) fn workspace_plan(&self) -> Result<Plan<'_>, PlanError> {
        self.owner.source().plan()
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        // Pinned Rust ArcInner<T>: repr(C), two AtomicUsize counters then T.
        // Include the full allocation, not only its pointer or live payload.
        let (layout, _) = Layout::new::<(AtomicUsize, AtomicUsize)>()
            .extend(Layout::new::<Source>())
            .ok()?;
        [
            layout.pad_to_align().size(),
            size_of::<SourceOwner>(),
            size_of::<Self>(),
            size_of::<Result<Self, construction::Failure>>(),
        ]
        .iter()
        .try_fold(0usize, |sum, n| sum.checked_add(*n))
    }
}
impl PartialEq for CompiledRegexSplit {
    fn eq(&self, other: &Self) -> bool {
        self.pattern() == other.pattern()
    }
}
impl fmt::Debug for CompiledRegexSplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledRegexSplit")
            .field("pattern", &self.pattern())
            .finish()
    }
}
impl Serialize for CompiledRegexSplit {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        enum Pattern<'a> {
            Regex(&'a str),
        }
        #[derive(Serialize)]
        struct Split<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            pattern: Pattern<'a>,
            behavior: SplitDelimiterBehavior,
            invert: bool,
        }
        Split {
            kind: "Split",
            pattern: Pattern::Regex(self.pattern()),
            behavior: SplitDelimiterBehavior::Isolated,
            invert: false,
        }
        .serialize(serializer)
    }
}

struct Matches<'w, 's, 't>(fancy_regex::workspace::Matches<'w, 's, 't>);
impl Iterator for Matches<'_, '_, '_> {
    type Item = Result<Offsets, fancy_regex::Error>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .next()
            .map(|result| result.map(|m| (m.start(), m.end())))
    }
}
/// Shared fallible full coverage. Original E supplies its one bound workspace;
/// ordinary pretokenization supplies its own unaccounted compatibility workspace.
pub(crate) fn visit_spans<E: From<fancy_regex::Error>>(
    workspace: &mut Workspace<'_>,
    text: &str,
    mut visit: impl FnMut(Offsets) -> Result<(), E>,
) -> Result<(), E> {
    for item in coverage(text.len(), Matches(workspace.find_iter(text))) {
        let (offsets, _) = item.map_err(E::from)?;
        visit(offsets)?;
    }
    Ok(())
}
pub(crate) fn visit_control_bytes() -> Option<usize> {
    [
        size_of::<Coverage<Matches<'static, 'static, 'static>>>(),
        size_of::<Option<Result<(Offsets, bool), fancy_regex::Error>>>(),
        size_of::<Offsets>(),
    ]
    .iter()
    .try_fold(0usize, |sum, n| sum.checked_add(*n))
}
impl PreTokenizer for CompiledRegexSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> crate::Result<()> {
        let mut workspace = self
            .workspace_plan()?
            .prepare()
            .map_err(|error| error.retire())?;
        pretokenized.split(|_, normalized| {
            let mut pieces = Vec::new();
            visit_spans::<crate::Error>(&mut workspace, normalized.get(), |offsets| {
                pieces.push(normalized.isolated_piece(offsets));
                Ok(())
            })?;
            Ok(pieces)
        })
    }
}

#[cfg(test)]
mod tests;
