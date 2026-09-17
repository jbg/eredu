//! One-attempt fresh construction of exact emitted default sources.
//!
//! This constructor skips dynamic parsing, not the existing search engine.
//! Its private recipes are emitted by the ordinary pinned compiler. A caller
//! cannot provide a recipe, precompiled delegate, byte allowance or allocator.
//! Static recipe residency and future original account integration are separate.

use super::{Body, PlanError as WorkspacePlanError, Source};
use crate::{
    vm::{CaptureGroupRange, Insn, PikeDelegate, Prog},
    Assertion, RegexOptions,
};
use alloc::{collections::TryReserveError, string::String, vec::Vec};
use core::{alloc::Layout, fmt, mem};
use regex_automata::nfa::thompson::source as pike;

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
    body: BodyRecipe,
}
#[derive(Debug)]
enum BodyRecipe {
    Wrap {
        pattern: &'static str,
        explicit_group_zero: bool,
    },
    Fancy {
        saves: usize,
        instructions: &'static [InsnRecipe],
    },
}
#[derive(Debug)]
enum InsnRecipe {
    End,
    Any,
    AnyNoNL,
    Assertion(Assertion),
    Lit(&'static str),
    Split(usize, usize),
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
    Pike {
        pattern: &'static str,
        ordinal: usize,
        capture_groups: Option<(usize, usize)>,
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
    /// A genuine delegate plan rejects its source or geometry.
    Delegate(pike::PlanError),
}
impl From<pike::PlanError> for PlanError {
    fn from(error: pike::PlanError) -> Self {
        Self::Delegate(error)
    }
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile => f.write_str("fresh source construction profile is not implemented"),
            Self::Recipe => f.write_str("fresh source recipe is inconsistent"),
            Self::Overflow => f.write_str("fresh source construction layout overflow"),
            Self::Delegate(error) => error.fmt(f),
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
    scratch: pike::ScratchLayout,
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
        let mut scratch = None;
        let mut bytes = array::<u8>(source.len())?;
        let mut heap_bytes = bytes;
        let mut allocations = usize::from(bytes != 0);
        match recipe.body {
            BodyRecipe::Wrap { pattern, .. } => {
                let plan = pike::Plan::new(pattern)?;
                bytes = add(bytes, plan.required_bytes())?;
                heap_bytes = add(heap_bytes, plan.heap_bytes())?;
                allocations = add(allocations, plan.allocation_count())?;
                scratch = Some(plan.scratch_layout());
            }
            BodyRecipe::Fancy { instructions, .. } => {
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
                        InsnRecipe::Pike { pattern, .. } => {
                            let plan = pike::Plan::new(pattern)?;
                            bytes = add(bytes, plan.required_bytes())?;
                            heap_bytes = add(heap_bytes, plan.heap_bytes())?;
                            allocations = add(allocations, plan.allocation_count())?;
                            scratch = Some(scratch.map_or(
                                plan.scratch_layout(),
                                |previous: pike::ScratchLayout| {
                                    previous.union(plan.scratch_layout())
                                },
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        let scratch = scratch.ok_or(PlanError::Recipe)?;
        bytes = add(bytes, scratch.required_bytes()?)?;
        heap_bytes = add(heap_bytes, scratch.heap_bytes()?)?;
        allocations = add(allocations, 3)?;
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
            mem::size_of::<PikeDelegate>(),
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
            scratch,
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
    /// Number of allocations in the successful fresh source plus shared scratch.
    pub fn allocation_count(&self) -> usize {
        self.allocations
    }
    /// Perform one attempt, returning either the fresh closed source or all
    /// current source/scratch storage in a by-value failure.
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
            ConstructionFailure::Instructions => match self.recipe.body {
                BodyRecipe::Fancy { .. } => Fault::Instructions,
                _ => {
                    return Err(Failure {
                        cause: Cause::Profile(WorkspacePlanError::Geometry),
                        storage: Storage::new(),
                    })
                }
            },
            ConstructionFailure::FirstLiteral => match self.recipe.body {
                BodyRecipe::Fancy { instructions, .. } => match instructions
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
                _ => {
                    return Err(Failure {
                        cause: Cause::Profile(WorkspacePlanError::Geometry),
                        storage: Storage::new(),
                    })
                }
            },
            ConstructionFailure::LastDelegate => {
                let count = match self.recipe.body {
                    BodyRecipe::Wrap { .. } => 1,
                    BodyRecipe::Fancy { instructions, .. } => instructions
                        .iter()
                        .filter(|i| matches!(i, InsnRecipe::Pike { .. }))
                        .count(),
                };
                Fault::Delegate(count.checked_sub(1).expect("validated delegate population"))
            }
            ConstructionFailure::Scratch(buffer) => Fault::Scratch(buffer),
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
        if let BodyRecipe::Fancy { instructions, .. } = self.recipe.body {
            reserve!(
                storage.instructions,
                instructions.len(),
                Fault::Instructions
            );
        }
        let scratch_result = {
            #[cfg(feature = "workspace-test-support")]
            if let Some(Fault::Scratch(buffer)) = fault {
                self.scratch.prepare_failing(buffer)
            } else {
                self.scratch.prepare()
            }
            #[cfg(not(feature = "workspace-test-support"))]
            {
                self.scratch.prepare()
            }
        };
        storage.scratch = Some(match scratch_result {
            Ok(value) => value,
            Err(error) => {
                return Err(Failure {
                    cause: Cause::Scratch(error),
                    storage,
                })
            }
        });
        macro_rules! delegate {
            ($pattern:expr,$ordinal:expr) => {{
                let plan = match pike::Plan::new($pattern) {
                    Ok(plan) => plan,
                    Err(error) => {
                        return Err(Failure {
                            cause: Cause::Plan(error),
                            storage,
                        })
                    }
                };
                #[cfg(any(test, feature = "workspace-test-support"))]
                let plan = if fault == Some(Fault::Delegate($ordinal)) {
                    match plan.fail_reserve_for_testing(pike::ConstructionBuffer::LastTransitions) {
                        Ok(plan) => plan,
                        Err(error) => {
                            return Err(Failure {
                                cause: Cause::Plan(error),
                                storage,
                            })
                        }
                    }
                } else {
                    plan
                };
                match plan.prepare(storage.scratch.as_mut().unwrap()) {
                    Ok(value) => value,
                    Err(error) => {
                        return Err(Failure {
                            cause: Cause::Delegate(error),
                            storage,
                        })
                    }
                }
            }};
        }
        storage.body = Some(match self.recipe.body {
            BodyRecipe::Wrap {
                pattern,
                explicit_group_zero,
            } => Body::Wrap {
                inner: delegate!(pattern, 0),
                explicit_group_zero,
            },
            BodyRecipe::Fancy {
                saves,
                instructions,
            } => {
                for (index, recipe) in instructions.iter().enumerate() {
                    let instruction = match *recipe {
                        InsnRecipe::End => Insn::End,
                        InsnRecipe::Any => Insn::Any,
                        InsnRecipe::AnyNoNL => Insn::AnyNoNL,
                        InsnRecipe::Assertion(value) => Insn::Assertion(value),
                        InsnRecipe::Lit(literal) => {
                            reserve!(storage.literal, literal.len(), Fault::Literal(index));
                            storage.literal.push_str(literal);
                            Insn::Lit(mem::take(&mut storage.literal))
                        }
                        InsnRecipe::Split(a, b) => Insn::Split(a, b),
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
                        InsnRecipe::Pike {
                            pattern,
                            ordinal,
                            capture_groups,
                        } => Insn::PikeDelegate(PikeDelegate {
                            inner: delegate!(pattern, ordinal),
                            ordinal,
                            capture_groups: capture_groups.map(|(a, b)| CaptureGroupRange(a, b)),
                        }),
                    };
                    storage.instructions.push(instruction);
                }
                Body::Fancy(Prog::new(mem::take(&mut storage.instructions), saves))
            }
        });
        let mut options = RegexOptions::default();
        options.pattern = mem::take(&mut storage.pattern);
        storage.source = Some(Source {
            body: storage.body.take().unwrap(),
            options,
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
    #[cfg(any(test, feature = "workspace-test-support"))]
    Delegate(usize),
    #[cfg(any(test, feature = "workspace-test-support"))]
    Completed,
    #[cfg(feature = "workspace-test-support")]
    Scratch(pike::ScratchFailureBuffer),
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
    /// Last delegate's last actual sparse-transition reserve.
    LastDelegate,
    /// One actual shared finalization scratch reserve.
    Scratch(ScratchFailureBuffer),
    /// Explicitly injected rejection after complete construction, not a natural error.
    Completed,
}
/// Fixed scratch destination selector, enabled only for development tests.
#[cfg(feature = "workspace-test-support")]
pub use pike::ScratchFailureBuffer;
#[derive(Debug)]
struct Storage {
    pattern: String,
    instructions: Vec<Insn>,
    literal: String,
    scratch: Option<pike::Scratch>,
    body: Option<Body>,
    source: Option<Source>,
}
impl Storage {
    fn new() -> Self {
        Self {
            pattern: String::new(),
            instructions: Vec::new(),
            literal: String::new(),
            scratch: None,
            body: None,
            source: None,
        }
    }
}
#[derive(Debug)]
enum Cause {
    Reserve(TryReserveError),
    Capacity,
    Scratch(pike::ScratchFailure),
    Delegate(pike::Failure),
    Plan(pike::PlanError),
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
            Cause::Scratch(e) => e.fmt(f),
            Cause::Delegate(e) => e.fmt(f),
            Cause::Plan(e) => e.fmt(f),
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
            Cause::Scratch(e) => Some(e),
            Cause::Delegate(e) => Some(e),
            Cause::Plan(e) => Some(e),
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
fn validate(recipe: &SourceRecipe) -> Result<(), PlanError> {
    if let BodyRecipe::Fancy {
        saves,
        instructions,
    } = recipe.body
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
                InsnRecipe::Split(a, b) => {
                    target(a)?;
                    target(b)?;
                }
                InsnRecipe::Jmp(a) => target(a)?,
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
                InsnRecipe::Pike {
                    ordinal: actual,
                    capture_groups,
                    ..
                } => {
                    if actual != ordinal {
                        return Err(PlanError::Recipe);
                    }
                    ordinal = add(ordinal, 1)?;
                    if let Some((a, b)) = capture_groups {
                        if a > b || b.checked_mul(2).ok_or(PlanError::Overflow)? > saves {
                            return Err(PlanError::Recipe);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
