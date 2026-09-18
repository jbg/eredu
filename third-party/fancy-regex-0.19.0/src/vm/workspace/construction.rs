//! One-attempt fresh construction of exact emitted default sources.
//!
//! This constructor skips dynamic parsing, not the existing search engine.
//! Its private recipes are emitted by the ordinary pinned compiler. A caller
//! cannot provide a recipe, precompiled delegate, byte allowance or allocator.
//! Static recipe residency and future original account integration are separate.

use super::{Body, PlanError as WorkspacePlanError, Source};
use crate::{
    vm::{Insn, Prog},
    Assertion, RegexOptions,
};
use alloc::{collections::TryReserveError, string::String, vec::Vec};
use core::{alloc::Layout, fmt, mem};
use regex_automata::dfa::dense::source as dfa;

#[derive(Debug)]
struct SyntaxRecipe {
    unicode: bool,
    case_insensitive: bool,
    multi_line: bool,
    dot_matches_new_line: bool,
    crlf: bool,
    line_terminator: u8,
    swap_greed: bool,
    ignore_whitespace: bool,
    utf8: bool,
    nest_limit: u32,
    octal: bool,
}
impl SyntaxRecipe {
    fn is_default(&self) -> bool {
        self.unicode
            && !self.case_insensitive
            && !self.multi_line
            && !self.dot_matches_new_line
            && !self.crlf
            && self.line_terminator == 10
            && !self.swap_greed
            && !self.ignore_whitespace
            && self.utf8
            && self.nest_limit == 250
            && !self.octal
    }
}
#[derive(Debug)]
struct SourceRecipe {
    pattern: &'static str,
    syntax: SyntaxRecipe,
    backtrack_limit: usize,
    saves: usize,
    instructions: &'static [InsnRecipe],
}
#[derive(Debug)]
enum InsnRecipe {
    End,
    Any,
    AnyNoNL,
    AnyNoCRLF,
    CharClassCodepoint(&'static [(char, char)]),
    CharClassByte(&'static [(u8, u8)]),
    Assertion(Assertion),
    Lit(&'static str),
    Split(usize, usize),
    SplitUnanchored(usize, usize),
    SaveCaptureGroupStart(usize),
    Jmp(usize),
    Save(usize),
    Save0(usize),
    Restore(usize),
    RepeatGr {
        lo: usize,
        hi: usize,
        next: usize,
        repeat: usize,
    },
    RepeatNg {
        lo: usize,
        hi: usize,
        next: usize,
        repeat: usize,
    },
    RepeatEpsilonGr {
        lo: usize,
        next: usize,
        repeat: usize,
        check: usize,
    },
    RepeatEpsilonNg {
        lo: usize,
        next: usize,
        repeat: usize,
        check: usize,
    },
    FailNegativeLookAround,
    Regular {
        pattern: &'static str,
        ordinal: usize,
    },
}
include!("construction/recipes.rs");

/// Borrow the exact full default source strings supported by fresh construction.
/// This immutable view supplies no recipe, program, byte count or account grant.
/// Call [`Plan::new`] with a fully validated actual source before construction.
pub fn patterns() -> impl ExactSizeIterator<Item = &'static str> + Clone {
    SOURCES.iter().map(|recipe| recipe.pattern)
}

/// Fixed rejection before any source construction reserve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanError {
    /// The exact full source/default profile is not implemented here.
    Profile,
    /// An emitted instruction/index/configuration is inconsistent.
    Recipe,
    /// A concrete allocation layout or sum cannot be represented.
    Overflow,
    /// Immutable DFA table profile or layout rejection.
    Dfa(dfa::PlanError),
}
impl From<dfa::PlanError> for PlanError {
    fn from(error: dfa::PlanError) -> Self {
        Self::Dfa(error)
    }
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile => f.write_str("fresh source construction profile is not implemented"),
            Self::Recipe => f.write_str("fresh source recipe is inconsistent"),
            Self::Overflow => f.write_str("fresh source construction layout overflow"),
            Self::Dfa(error) => error.fmt(f),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for PlanError {}

/// A borrowed exact source and its checked fresh construction requirements.
///
/// This plan is not Clone; preparation consumes it once. Its existence grants
/// no account capacity. The successful source still uses the existing closed
/// workspace and unchanged search/iterator workers.
///
/// ```compile_fail
/// use fancy_regex::workspace::construction::Plan;
/// let plan = Plan::new("a").unwrap();
/// let _ = plan.prepare();
/// let _ = plan.prepare(); // a construction attempt consumes its plan
/// ```
///
/// ```compile_fail
/// use fancy_regex::workspace::construction::Plan;
/// let plan;
/// {
///     let source = String::from("a");
///     plan = Plan::new(&source).unwrap();
/// }
/// let _ = plan.prepare(); // the source borrow must survive construction
/// ```
#[derive(Debug)]
pub struct Plan<'s> {
    source: &'s str,
    recipe: &'static SourceRecipe,
    bytes: usize,
    heap_bytes: usize,
    allocations: usize,
}
impl<'s> Plan<'s> {
    /// Select exact full pattern bytes with ordinary default settings.
    pub fn new(source: &'s str) -> Result<Self, PlanError> {
        let recipe = SOURCES
            .iter()
            .find(|r| r.pattern == source)
            .ok_or(PlanError::Profile)?;
        if !recipe.syntax.is_default() || recipe.backtrack_limit != 1_000_000 {
            return Err(PlanError::Recipe);
        }
        validate(recipe)?;
        let mut bytes = array::<u8>(source.len())?;
        let mut heap_bytes = bytes;
        let mut allocations = usize::from(bytes != 0);
        let instructions = recipe.instructions;
        {
            let instructions_bytes = array::<Insn>(instructions.len())?;
            bytes = add(bytes, instructions_bytes)?;
            heap_bytes = add(heap_bytes, instructions_bytes)?;
            allocations = add(allocations, usize::from(instructions_bytes != 0))?;
            for instruction in instructions {
                match *instruction {
                    InsnRecipe::Lit(literal) => {
                        bytes = add(bytes, array::<u8>(literal.len())?)?;
                        heap_bytes = add(heap_bytes, array::<u8>(literal.len())?)?;
                        allocations = add(allocations, usize::from(!literal.is_empty()))?;
                    }
                    InsnRecipe::CharClassCodepoint(ranges) => {
                        let backing = array::<(char, char)>(ranges.len())?;
                        bytes = add(bytes, backing)?;
                        heap_bytes = add(heap_bytes, backing)?;
                        allocations = add(allocations, usize::from(backing != 0))?;
                    }
                    InsnRecipe::CharClassByte(ranges) => {
                        let backing = array::<(u8, u8)>(ranges.len())?;
                        bytes = add(bytes, backing)?;
                        heap_bytes = add(heap_bytes, backing)?;
                        allocations = add(allocations, usize::from(backing != 0))?;
                    }
                    InsnRecipe::Regular { pattern, .. } => {
                        let plan = dfa::Plan::new(pattern)?;
                        bytes = add(bytes, plan.required_bytes())?;
                        heap_bytes = add(heap_bytes, plan.heap_bytes())?;
                        allocations = add(allocations, plan.allocation_count())?;
                    }
                    _ => {}
                }
            }
        }
        for value in [
            mem::size_of::<Self>(),
            mem::size_of::<Result<Self, PlanError>>(),
            mem::size_of::<Storage>(),
            mem::size_of::<Failure>(),
            mem::size_of::<Result<Source, Failure>>(),
            mem::size_of::<Source>(),
            mem::size_of::<Body>(),
            mem::size_of::<RegexOptions>(),
            mem::size_of::<Prog>(),
            mem::size_of::<Insn>(),
            mem::size_of::<Result<(), TryReserveError>>(),
            mem::size_of::<Result<super::Plan<'_>, WorkspacePlanError>>(),
            mem::size_of::<Result<(), WorkspacePlanError>>(),
            mem::size_of::<Option<Fault>>(),
        ] {
            bytes = add(bytes, value)?;
        }
        Ok(Self {
            source,
            recipe,
            bytes,
            heap_bytes,
            allocations,
        })
    }
    /// Exact pattern borrowed through construction.
    pub fn source(&self) -> &'s str {
        self.source
    }
    /// Checked heap requests and named simultaneous construction controls.
    pub fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Sum of the actual nonzero heap requests, excluding named controls.
    pub fn heap_bytes(&self) -> usize {
        self.heap_bytes
    }
    /// Number of allocations in the successful fresh source.
    pub fn allocation_count(&self) -> usize {
        self.allocations
    }
    /// Perform one attempt, returning either the fresh closed source or all
    /// current source storage in a by-value failure.
    pub fn prepare(self) -> Result<Source, Failure> {
        self.prepare_inner(None)
    }
    /// Development-only actual failed reserve (or explicit completed-source
    /// rejection). Missing literal/instruction targets reject before allocating;
    /// no arbitrary capacity, replacement source or successful override exists.
    #[cfg(feature = "workspace-test-support")]
    pub fn prepare_failing(self, target: ConstructionFailure) -> Result<Source, Failure> {
        let fault = match target {
            ConstructionFailure::Pattern => Fault::Pattern,
            ConstructionFailure::Instructions => Fault::Instructions,
            ConstructionFailure::FirstLiteral => match self
                .recipe
                .instructions
                .iter()
                .position(|i| matches!(i, InsnRecipe::Lit(_)))
            {
                Some(index) => Fault::Literal(index),
                None => {
                    return Err(Failure {
                        cause: Cause::Profile(WorkspacePlanError::Geometry),
                        storage: Storage::new(),
                    })
                }
            },
            ConstructionFailure::LastDelegate => {
                let count = self
                    .recipe
                    .instructions
                    .iter()
                    .filter(|i| matches!(i, InsnRecipe::Regular { .. }))
                    .count();
                Fault::Delegate(count.checked_sub(1).expect("validated delegate population"))
            }
            ConstructionFailure::Completed => Fault::Completed,
        };
        self.prepare_inner(Some(fault))
    }
    fn prepare_inner(self, fault: Option<Fault>) -> Result<Source, Failure> {
        let mut storage = Storage::new();
        macro_rules! reserve {
            ($field:expr,$count:expr,$target:expr) => {{
                let expected = $count;
                let requested = if fault == Some($target) {
                    usize::MAX
                } else {
                    expected
                };
                if let Err(error) = $field.try_reserve_exact(requested) {
                    return Err(Failure {
                        cause: Cause::Reserve(error),
                        storage,
                    });
                }
                if $field.capacity() != expected {
                    return Err(Failure {
                        cause: Cause::Capacity,
                        storage,
                    });
                }
            }};
        }
        reserve!(storage.pattern, self.source.len(), Fault::Pattern);
        storage.pattern.push_str(self.source);
        let instructions = self.recipe.instructions;
        reserve!(
            storage.instructions,
            instructions.len(),
            Fault::Instructions
        );
        storage.body = Some({
            for (index, recipe) in instructions.iter().enumerate() {
                let instruction = match *recipe {
                    InsnRecipe::End => Insn::End,
                    InsnRecipe::Any => Insn::Any,
                    InsnRecipe::AnyNoNL => Insn::AnyNoNL,
                    InsnRecipe::AnyNoCRLF => Insn::AnyNoCRLF,
                    InsnRecipe::CharClassCodepoint(ranges) => {
                        reserve!(storage.codepoint_ranges, ranges.len(), Fault::Class(index));
                        storage.codepoint_ranges.extend_from_slice(ranges);
                        Insn::CharClass(crate::vm::CharClassMatcher::Codepoint(
                            mem::take(&mut storage.codepoint_ranges).into_boxed_slice(),
                        ))
                    }
                    InsnRecipe::CharClassByte(ranges) => {
                        reserve!(storage.byte_ranges, ranges.len(), Fault::Class(index));
                        storage.byte_ranges.extend_from_slice(ranges);
                        Insn::CharClass(crate::vm::CharClassMatcher::Byte(
                            mem::take(&mut storage.byte_ranges).into_boxed_slice(),
                        ))
                    }
                    InsnRecipe::Assertion(value) => Insn::Assertion(value),
                    InsnRecipe::Lit(literal) => {
                        reserve!(storage.literal, literal.len(), Fault::Literal(index));
                        storage.literal.push_str(literal);
                        Insn::Lit(mem::take(&mut storage.literal))
                    }
                    InsnRecipe::Split(a, b) => Insn::Split(a, b),
                    InsnRecipe::SplitUnanchored(a, b) => Insn::SplitUnanchored(a, b),
                    InsnRecipe::SaveCaptureGroupStart(group) => Insn::SaveCaptureGroupStart(group),
                    InsnRecipe::Jmp(next) => Insn::Jmp(next),
                    InsnRecipe::Save(slot) => Insn::Save(slot),
                    InsnRecipe::Save0(slot) => Insn::Save0(slot),
                    InsnRecipe::Restore(slot) => Insn::Restore(slot),
                    InsnRecipe::RepeatGr {
                        lo,
                        hi,
                        next,
                        repeat,
                    } => Insn::RepeatGr {
                        lo,
                        hi,
                        next,
                        repeat,
                    },
                    InsnRecipe::RepeatNg {
                        lo,
                        hi,
                        next,
                        repeat,
                    } => Insn::RepeatNg {
                        lo,
                        hi,
                        next,
                        repeat,
                    },
                    InsnRecipe::RepeatEpsilonGr {
                        lo,
                        next,
                        repeat,
                        check,
                    } => Insn::RepeatEpsilonGr {
                        lo,
                        next,
                        repeat,
                        check,
                    },
                    InsnRecipe::RepeatEpsilonNg {
                        lo,
                        next,
                        repeat,
                        check,
                    } => Insn::RepeatEpsilonNg {
                        lo,
                        next,
                        repeat,
                        check,
                    },
                    InsnRecipe::FailNegativeLookAround => Insn::FailNegativeLookAround,
                    InsnRecipe::Regular {
                        pattern,
                        ordinal: _ordinal,
                    } => {
                        let plan = match dfa::Plan::new(pattern) {
                            Ok(plan) => plan,
                            Err(error) => {
                                return Err(Failure {
                                    cause: Cause::DfaPlan(error),
                                    storage,
                                })
                            }
                        };
                        #[cfg(any(test, feature = "workspace-test-support"))]
                        let plan = if fault == Some(Fault::Delegate(_ordinal)) {
                            plan.fail_reserve_for_testing(dfa::Buffer::Accelerators)
                        } else {
                            plan
                        };
                        let inner = match plan.prepare() {
                            Ok(inner) => inner,
                            Err(error) => {
                                return Err(Failure {
                                    cause: Cause::Dfa(error),
                                    storage,
                                })
                            }
                        };
                        Insn::DfaDelegate(inner)
                    }
                };
                storage.instructions.push(instruction);
            }
            Body::Fancy(Prog::new_explicit(
                mem::take(&mut storage.instructions),
                self.recipe.saves,
                crate::BytesMode::Unicode,
                String::new(),
            ))
        });
        let options = RegexOptions::default();
        let pattern = mem::take(&mut storage.pattern);
        storage.source = Some(Source {
            body: storage.body.take().unwrap(),
            options,
            pattern,
        });
        // Validation of the same actual final source used by search. Any
        // failure retains the complete source, not merely its diagnostic.
        let validation = match storage.source.as_ref().unwrap().plan() {
            Ok(_) => Ok(()),
            Err(error) => Err(error),
        };
        if let Err(error) = validation {
            return Err(Failure {
                cause: Cause::Profile(error),
                storage,
            });
        }
        #[cfg(any(test, feature = "workspace-test-support"))]
        if fault == Some(Fault::Completed) {
            return Err(Failure {
                cause: Cause::Profile(WorkspacePlanError::Geometry),
                storage,
            });
        }
        Ok(storage.source.take().unwrap())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fault {
    Pattern,
    Instructions,
    Literal(usize),
    Class(usize),
    #[cfg(any(test, feature = "workspace-test-support"))]
    Delegate(usize),
    #[cfg(any(test, feature = "workspace-test-support"))]
    Completed,
}
/// Fixed construction failure selection for dependency-development coverage.
#[cfg(feature = "workspace-test-support")]
#[derive(Clone, Copy, Debug)]
pub enum ConstructionFailure {
    /// Actual source-string reserve.
    Pattern,
    /// Actual instruction-vector reserve, when the source has one.
    Instructions,
    /// First actual literal-string reserve, when one exists.
    FirstLiteral,
    /// Last delegate's actual acceleration-table reserve.
    LastDelegate,
    /// Explicitly injected rejection after complete construction, not a natural error.
    Completed,
}
#[derive(Debug)]
struct Storage {
    pattern: String,
    instructions: Vec<Insn>,
    literal: String,
    codepoint_ranges: Vec<(char, char)>,
    byte_ranges: Vec<(u8, u8)>,
    body: Option<Body>,
    source: Option<Source>,
}
impl Storage {
    fn new() -> Self {
        Self {
            pattern: String::new(),
            instructions: Vec::new(),
            literal: String::new(),
            codepoint_ranges: Vec::new(),
            byte_ranges: Vec::new(),
            body: None,
            source: None,
        }
    }
}
#[derive(Debug)]
enum Cause {
    Reserve(TryReserveError),
    Capacity,
    Dfa(dfa::Failure),
    DfaPlan(dfa::PlanError),
    Profile(WorkspacePlanError),
}
/// One owning failure containing the full partial/completed construction.
#[derive(Debug)]
pub struct Failure {
    cause: Cause,
    storage: Storage,
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Reserve(e) => e.fmt(f),
            Cause::Capacity => f.write_str("fresh source capacity mismatch"),
            Cause::Dfa(e) => e.fmt(f),
            Cause::DfaPlan(e) => e.fmt(f),
            Cause::Profile(e) => e.fmt(f),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Reserve(e) => Some(e),
            Cause::Capacity => None,
            Cause::Dfa(e) => Some(e),
            Cause::DfaPlan(e) => Some(e),
            Cause::Profile(e) => Some(e),
        }
    }
}
fn array<T>(count: usize) -> Result<usize, PlanError> {
    Layout::array::<T>(count)
        .map(|layout| layout.size())
        .map_err(|_| PlanError::Overflow)
}
fn add(a: usize, b: usize) -> Result<usize, PlanError> {
    a.checked_add(b).ok_or(PlanError::Overflow)
}
fn validate_ranges<T: Copy + Ord>(ranges: &[(T, T)]) -> Result<(), PlanError> {
    if ranges.iter().any(|&(start, end)| start > end)
        || ranges.windows(2).any(|pair| pair[0].1 >= pair[1].0)
    {
        return Err(PlanError::Recipe);
    }
    Ok(())
}
fn validate(recipe: &SourceRecipe) -> Result<(), PlanError> {
    let saves = recipe.saves;
    let instructions = recipe.instructions;
    {
        if saves < 2
            || instructions.is_empty()
            || !matches!(instructions.last(), Some(InsnRecipe::End))
        {
            return Err(PlanError::Recipe);
        }
        let target = |n| {
            if n < instructions.len() {
                Ok(())
            } else {
                Err(PlanError::Recipe)
            }
        };
        let slot = |n| {
            if n < saves {
                Ok(())
            } else {
                Err(PlanError::Recipe)
            }
        };
        let mut ordinal = 0;
        for instruction in instructions {
            match *instruction {
                InsnRecipe::Split(a, b) | InsnRecipe::SplitUnanchored(a, b) => {
                    target(a)?;
                    target(b)?;
                }
                InsnRecipe::Jmp(a) => target(a)?,
                InsnRecipe::SaveCaptureGroupStart(group) => {
                    let start = group.checked_mul(2).ok_or(PlanError::Overflow)?;
                    slot(start)?;
                    slot(add(start, 1)?)?;
                }
                InsnRecipe::CharClassCodepoint(ranges) => validate_ranges(ranges)?,
                InsnRecipe::CharClassByte(ranges) => validate_ranges(ranges)?,
                InsnRecipe::Save(n) | InsnRecipe::Save0(n) | InsnRecipe::Restore(n) => slot(n)?,
                InsnRecipe::RepeatGr {
                    lo,
                    hi,
                    next,
                    repeat,
                }
                | InsnRecipe::RepeatNg {
                    lo,
                    hi,
                    next,
                    repeat,
                } => {
                    if lo > hi {
                        return Err(PlanError::Recipe);
                    }
                    target(next)?;
                    slot(repeat)?;
                }
                InsnRecipe::RepeatEpsilonGr {
                    next,
                    repeat,
                    check,
                    ..
                }
                | InsnRecipe::RepeatEpsilonNg {
                    next,
                    repeat,
                    check,
                    ..
                } => {
                    target(next)?;
                    slot(repeat)?;
                    slot(check)?;
                }
                InsnRecipe::Regular {
                    ordinal: actual, ..
                } => {
                    if actual != ordinal {
                        return Err(PlanError::Recipe);
                    }
                    ordinal = add(ordinal, 1)?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
