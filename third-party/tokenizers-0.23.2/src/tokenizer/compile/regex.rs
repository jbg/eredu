//! One exact decoded-pattern selection in the borrowed root, never a family ID.
use super::*;
#[cfg(feature = "fancy-regex")]
use crate::pre_tokenizers::{
    compiled_byte_level::CompiledByteLevel, compiled_split::CompiledRegexSplit,
};
#[cfg(feature = "fancy-regex")]
use fancy_regex::workspace::construction;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Origin {
    ExplicitSplit,
    ImplicitByteLevel(ByteLevel),
}
#[derive(Debug, Clone, Copy)]
pub(super) struct RegexSelection {
    pattern: &'static str,
    span: Span,
    origin: Origin,
}
pub(super) fn selection(input: &str, span: Span, role: Role) -> Result<RegexSelection, Error> {
    #[cfg(not(feature = "fancy-regex"))]
    {
        let _ = (input, role);
        Err(err(K::RegexProfile, span.start))
    }
    #[cfg(feature = "fancy-regex")]
    {
        if !matches!(role, Role::Pre) {
            return Err(err(K::RegexProfile, span.start));
        }
        let fields = fields(input, span, ["type", "pattern", "behavior", "invert"])?;
        if !text_is(input, required(fields[2], span.start)?, "Isolated")?
            || boolean(input, Some(required(fields[3], span.start)?), false)?
        {
            return Err(err(K::RegexProfile, span.start));
        }
        let pattern = required(fields[1], span.start)?;
        let pattern = super::fields(input, pattern, ["Regex"])?;
        let pattern = required(pattern[0], span.start)?;
        let mut reader = Reader::new(input, pattern);
        let text = reader.string()?;
        reader.finish()?;
        let source = construction::patterns()
            .find(|source| text.is(source))
            .ok_or_else(|| err(K::RegexProfile, pattern.start))?;
        Ok(RegexSelection {
            pattern: source,
            span,
            origin: Origin::ExplicitSplit,
        })
    }
}
impl RegexSelection {
    pub(super) fn is_implicit(&self) -> bool {
        matches!(self.origin, Origin::ImplicitByteLevel(_))
    }
}
pub(super) fn implicit_byte_level(
    span: Span,
    settings: ByteLevel,
) -> Result<RegexSelection, Error> {
    #[cfg(not(feature = "fancy-regex"))]
    {
        let _ = settings;
        Err(err(K::RegexProfile, span.start))
    }
    #[cfg(feature = "fancy-regex")]
    {
        if settings.add_prefix_space || !settings.use_regex {
            return Err(err(K::RegexProfile, span.start));
        }
        Ok(RegexSelection {
            pattern: crate::pre_tokenizers::byte_level::DEFAULT_PATTERN,
            span,
            origin: Origin::ImplicitByteLevel(settings),
        })
    }
}
#[derive(Debug)]
pub(super) struct RegexState {
    selected: Option<RegexSelection>,
    #[cfg(feature = "fancy-regex")]
    plan: Option<construction::Plan<'static>>,
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    failure: Option<construction::ConstructionFailure>,
}
impl RegexState {
    pub(super) fn plan(selected: Option<RegexSelection>) -> Result<Self, Error> {
        #[cfg(feature = "fancy-regex")]
        let plan = selected
            .map(|selection| {
                construction::Plan::new(selection.pattern).map_err(|error| {
                    err(
                        if matches!(error, construction::PlanError::Overflow) {
                            K::Overflow
                        } else {
                            K::RegexProfile
                        },
                        selection.span.start,
                    )
                })
            })
            .transpose()?;
        Ok(Self {
            selected,
            #[cfg(feature = "fancy-regex")]
            plan,
            #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
            failure: None,
        })
    }
    pub(super) fn buffer_bytes(&self) -> usize {
        #[cfg(feature = "fancy-regex")]
        if let Some(plan) = &self.plan {
            return plan.heap_bytes();
        }
        0
    }
    pub(super) fn required_bytes(&self) -> Result<usize, Error> {
        #[cfg(feature = "fancy-regex")]
        if let Some(plan) = &self.plan {
            return [
                plan.required_bytes(),
                CompiledRegexSplit::control_bytes().ok_or_else(|| err(K::Overflow, 0))?,
                size_of::<Self>(),
                size_of::<Result<Self, Error>>(),
                size_of::<RegexSelection>(),
                size_of::<Origin>(),
                if self.selected.is_some_and(|selected| selected.is_implicit()) {
                    CompiledByteLevel::wrapper_control_bytes().ok_or_else(|| err(K::Overflow, 0))?
                } else {
                    0
                },
                size_of::<Option<RegexSelection>>(),
                size_of::<Result<RegexSelection, Error>>(),
                size_of::<Result<PreTokenizerWrapper, Cause>>(),
                size_of::<json::Text<'_>>(),
                size_of::<json::Bytes<'_>>(),
                size_of::<std::str::Bytes<'_>>(),
                size_of::<Result<construction::Plan<'static>, construction::PlanError>>(),
            ]
            .iter()
            .try_fold(0usize, |sum, &bytes| sum.checked_add(bytes))
            .ok_or_else(|| err(K::Overflow, 0));
        }
        Ok(0)
    }
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    pub(super) fn fail(&mut self, target: construction::ConstructionFailure) {
        self.failure = Some(target);
    }
    pub(super) fn compile(
        &mut self,
        selected: RegexSelection,
    ) -> Result<PreTokenizerWrapper, Cause> {
        let actual = self
            .selected
            .ok_or_else(|| err(K::RegexProfile, selected.span.start))?;
        if actual.origin != selected.origin
            || actual.pattern != selected.pattern
            || actual.span.start != selected.span.start
            || actual.span.end != selected.span.end
        {
            return Err(err(K::RegexProfile, selected.span.start).into());
        }
        #[cfg(not(feature = "fancy-regex"))]
        {
            Err(err(K::RegexProfile, selected.span.start).into())
        }
        #[cfg(feature = "fancy-regex")]
        {
            let plan = self
                .plan
                .take()
                .ok_or_else(|| err(K::RegexProfile, selected.span.start))?;
            #[cfg(feature = "tokenizer-compiler-test-support")]
            if let Some(target) = self.failure.take() {
                // The genuine selected plan is attempted once. No replacement source is accepted.
                return plan
                    .prepare_failing(target)
                    .map(|_| unreachable!("fixed failure target succeeded"))
                    .map_err(Cause::Regex);
            }
            let split = CompiledRegexSplit::compile(plan).map_err(Cause::Regex)?;
            Ok(match selected.origin {
                Origin::ExplicitSplit => PreTokenizerWrapper::CompiledRegexSplit(split),
                Origin::ImplicitByteLevel(settings) => PreTokenizerWrapper::CompiledByteLevel(
                    CompiledByteLevel::from_parts(settings, split),
                ),
            })
        }
    }
}
