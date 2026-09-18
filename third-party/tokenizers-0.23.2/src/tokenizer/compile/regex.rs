//! One exact decoded-pattern selection in the borrowed root, never a family ID.
use super::*;
#[cfg(feature = "fancy-regex")]
use crate::{pre_tokenizers::split::Split, utils::SysRegex};
#[cfg(feature = "fancy-regex")]
use fancy_regex::workspace::construction;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Origin {
    ExplicitSplit,
    Whitespace,
    ImplicitByteLevel(ByteLevelSettings),
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
        let source = SysRegex::declared_patterns()
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
    pub(super) fn is_whitespace(&self) -> bool {
        matches!(self.origin, Origin::Whitespace)
    }
    pub(super) fn is_implicit(&self) -> bool {
        matches!(self.origin, Origin::ImplicitByteLevel(_))
    }
}
pub(super) fn whitespace(span: Span) -> Result<RegexSelection, Error> {
    #[cfg(feature = "fancy-regex")]
    {
        Ok(RegexSelection {
            pattern: crate::pre_tokenizers::whitespace::PATTERN,
            span,
            origin: Origin::Whitespace,
        })
    }
    #[cfg(not(feature = "fancy-regex"))]
    {
        Err(err(K::RegexProfile, span.start))
    }
}
pub(super) fn implicit_byte_level(
    span: Span,
    settings: ByteLevelSettings,
) -> Result<RegexSelection, Error> {
    #[cfg(not(feature = "fancy-regex"))]
    {
        let _ = settings;
        Err(err(K::RegexProfile, span.start))
    }
    #[cfg(feature = "fancy-regex")]
    {
        if !settings.use_regex {
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
struct RegexOne {
    selected: Option<RegexSelection>,
    #[cfg(feature = "fancy-regex")]
    plan: Option<construction::Plan<'static>>,
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    failure: Option<construction::ConstructionFailure>,
}
impl RegexOne {
    pub(super) fn plan(selected: Option<RegexSelection>) -> Result<Self, Error> {
        #[cfg(feature = "fancy-regex")]
        let plan = selected
            .map(|selection| {
                SysRegex::construction_plan(selection.pattern).map_err(|error| {
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
                Split::compiled_control_bytes().ok_or_else(|| err(K::Overflow, 0))?,
                size_of::<Self>(),
                size_of::<Result<Self, Error>>(),
                size_of::<RegexSelection>(),
                size_of::<Origin>(),
                if self.selected.is_some_and(|selected| selected.is_implicit()) {
                    size_of::<ByteLevel>()
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
            let regex = SysRegex::compile_selected(plan, actual.pattern).map_err(Cause::Regex)?;
            Ok(match selected.origin {
                Origin::ExplicitSplit => PreTokenizerWrapper::Split(Split::from_regex(regex)),
                Origin::Whitespace => PreTokenizerWrapper::Whitespace(
                    crate::pre_tokenizers::whitespace::Whitespace::from_regex(regex),
                ),
                Origin::ImplicitByteLevel(settings) => {
                    PreTokenizerWrapper::ByteLevel(settings.build().with_compiled_regex(regex))
                }
            })
        }
    }
}

/// Borrowed ordered source inventory. Every constructor is the same RegexOne
/// worker, selected in source order; its retained heap is summed before C.
#[derive(Debug)]
pub(super) struct RegexState<'a> {
    input: &'a str,
    component: Component,
    next: usize,
    buffers: usize,
    controls: usize,
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    failure: Option<(usize, construction::ConstructionFailure)>,
}
impl<'a> RegexState<'a> {
    fn visit(
        input: &str,
        component: Component,
        mut visit: impl FnMut(RegexSelection) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let Some(span) = component.value else {
            return Ok(());
        };
        if let Some(array) = component.array {
            let mut items = pre::Items::new(input, array, Role::Pre)?;
            while let Some(span) = items.next()? {
                if let Inline::Regex(selected) = inline(input, span, Role::Pre)? {
                    visit(selected)?;
                }
            }
        } else if let Inline::Regex(selected) = inline(input, span, Role::Pre)? {
            visit(selected)?;
        }
        Ok(())
    }
    pub(super) fn plan(input: &'a str, component: Component) -> Result<Self, Error> {
        let mut buffers = 0;
        let mut controls = add(size_of::<Self>(), size_of::<Result<Self, Error>>())?;
        controls = add(controls, pre::controls()?)?;
        Self::visit(input, component, |selected| {
            let plan = RegexOne::plan(Some(selected))?;
            buffers = add(buffers, plan.buffer_bytes())?;
            controls = add(
                controls,
                plan.required_bytes()?
                    .checked_sub(plan.buffer_bytes())
                    .ok_or_else(|| err(K::Overflow, 0))?,
            )?;
            Ok(())
        })?;
        Ok(Self {
            input,
            component,
            next: 0,
            buffers,
            controls,
            #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
            failure: None,
        })
    }
    pub(super) fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    pub(super) fn required_bytes(&self) -> Result<usize, Error> {
        add(self.buffers, self.controls)
    }
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    pub(super) fn fail(&mut self, target: construction::ConstructionFailure) {
        self.fail_at(0, target);
    }
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    pub(super) fn fail_at(&mut self, ordinal: usize, target: construction::ConstructionFailure) {
        self.failure = Some((ordinal, target));
    }
    pub(super) fn compile(
        &mut self,
        selected: RegexSelection,
    ) -> Result<PreTokenizerWrapper, Cause> {
        let mut at = 0;
        let mut actual = None;
        Self::visit(self.input, self.component, |item| {
            if at == self.next {
                actual = Some(item);
            }
            at += 1;
            Ok(())
        })?;
        let mut plan = RegexOne::plan(actual)?;
        self.next += 1;
        #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
        if let Some((ordinal, target)) = self.failure {
            if ordinal + 1 == self.next {
                plan.fail(target);
            }
        }
        plan.compile(selected)
    }
}
