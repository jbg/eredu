//! Fresh construction from a private, compiler-emitted source inventory.
//!
//! This is an allocation mechanism, not an account or a recipe-import API.
//! Planning matches exact source bytes and the fixed default syntax profile.
//! Ordinary builders remain unchanged. Static recipe image residency is not
//! included in the per-construction heap request.

use super::{Inner, SparseTransitions, State, Transition, NFA};
use crate::{
    nfa::thompson::pikevm::PikeVM,
    util::{
        alphabet::{ByteClassSet, ByteClasses},
        captures::construction as groups,
        look::{Look, LookMatcher, LookSet},
        primitives::{PatternID, SmallIndex, StateID},
        sparse_set::SparseSet,
    },
};
use alloc::{collections::TryReserveError, sync::Arc, vec::Vec};
use core::{fmt, mem};

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
struct NfaRecipe {
    pattern: &'static str,
    syntax: SyntaxRecipe,
    anchored: usize,
    unanchored: usize,
    utf8: bool,
    reverse: bool,
    line_terminator: u8,
    starts: &'static [usize],
    groups: &'static [&'static [(usize, usize)]],
    states: &'static [StateRecipe],
    expected: PropertiesRecipe,
}
#[derive(Debug)]
struct PropertiesRecipe {
    has_empty: bool,
    has_capture: bool,
    look_any: &'static [Look],
    look_prefix: &'static [Look],
    byte_classes: &'static [u8],
}
#[derive(Debug)]
enum StateRecipe {
    ByteRange(u8, u8, usize),
    Sparse(&'static [(u8, u8, usize)]),
    Look(Look, usize),
    Union(&'static [usize]),
    BinaryUnion(usize, usize),
    Capture {
        next: usize,
        pattern: usize,
        group: usize,
        slot: usize,
    },
    Fail,
    Match(usize),
}
include!("source/recipes.rs");

/// Fixed preconstruction rejection. No dynamic diagnostic is constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanError {
    /// Exact source or fixed syntax is not in this constructor inventory.
    Profile,
    /// A recipe index, property or representation is inconsistent.
    Recipe,
    /// A concrete layout or checked sum cannot be represented.
    Overflow,
    /// A required look assertion is unavailable in this feature build.
    LookUnavailable,
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Profile => "source construction profile is not implemented",
            Self::Recipe => "source construction recipe is inconsistent",
            Self::Overflow => "source construction layout overflow",
            Self::LookUnavailable => {
                "source construction look assertion unavailable"
            }
        })
    }
}
#[cfg(feature = "std")]
impl std::error::Error for PlanError {}
impl From<groups::Overflow> for PlanError {
    fn from(_: groups::Overflow) -> Self {
        Self::Overflow
    }
}

/// One exact borrowed source and its allocation-free construction facts.
///
/// ```compile_fail
/// use regex_automata::nfa::thompson::source::Plan;
/// let plan = Plan::new("a").unwrap();
/// let mut scratch = plan.scratch_layout().prepare().unwrap();
/// let _ = plan.prepare(&mut scratch);
/// let _ = plan.prepare(&mut scratch);
/// ```
#[derive(Debug)]
pub struct Plan<'s> {
    source: &'s str,
    recipe: &'static NfaRecipe,
    groups: groups::Plan,
    bytes: usize,
    heap_bytes: usize,
    allocations: usize,
    scratch: ScratchLayout,
    #[cfg(feature = "workspace-test-support")]
    fault: Option<Fault>,
}
impl<'s> Plan<'s> {
    /// Select the exact default-syntax source before any allocation.
    pub fn new(source: &'s str) -> Result<Self, PlanError> {
        let recipe = NFAS
            .iter()
            .find(|r| r.pattern == source)
            .ok_or(PlanError::Profile)?;
        Self::from_recipe(source, recipe)
    }
    fn from_recipe(
        source: &'s str,
        recipe: &'static NfaRecipe,
    ) -> Result<Self, PlanError> {
        let scratch = validate(recipe)?;
        let group_plan = groups::Plan::new(recipe.groups[0].len())?;
        let mut heap_bytes = group_plan.heap_bytes();
        for value in [
            groups::array::<State>(recipe.states.len())?,
            groups::array::<StateID>(recipe.starts.len())?,
            groups::arc_layout::<Inner>()?.size(),
        ] {
            heap_bytes = add(heap_bytes, value)?;
        }
        let mut allocations = 8usize; // four capture buffers, two Arcs, states and starts
        let mut bytes = group_plan.required_bytes();
        for value in [
            groups::array::<State>(recipe.states.len())?,
            groups::array::<StateID>(recipe.starts.len())?,
            groups::arc_layout::<Inner>()?.size(),
            mem::size_of::<Self>(),
            mem::size_of::<Result<Self, PlanError>>(),
            mem::size_of::<Storage>(),
            mem::size_of::<Failure>(),
            mem::size_of::<Result<PikeVM, Failure>>(),
            mem::size_of::<Inner>(),
            mem::size_of::<PikeVM>(),
            mem::size_of::<NFA>(),
            mem::size_of::<State>(),
            mem::size_of::<Option<Fault>>(),
            mem::size_of::<Result<(), TryReserveError>>(),
        ] {
            bytes = add(bytes, value)?;
        }
        for state in recipe.states {
            let payload = match state {
                StateRecipe::Sparse(ts) => {
                    groups::array::<Transition>(ts.len())?
                }
                StateRecipe::Union(ids) => {
                    groups::array::<StateID>(ids.len())?
                }
                _ => 0,
            };
            heap_bytes = add(heap_bytes, payload)?;
            allocations = add(allocations, usize::from(payload != 0))?;
            bytes = add(
                bytes,
                match state {
                    StateRecipe::Sparse(ts) => {
                        groups::array::<Transition>(ts.len())?
                    }
                    StateRecipe::Union(ids) => {
                        groups::array::<StateID>(ids.len())?
                    }
                    _ => 0,
                },
            )?;
        }
        Ok(Self {
            source,
            recipe,
            groups: group_plan,
            bytes,
            heap_bytes,
            allocations,
            scratch,
            #[cfg(feature = "workspace-test-support")]
            fault: None,
        })
    }
    /// The exact borrowed pattern selected by this plan.
    pub fn source(&self) -> &'s str {
        self.source
    }
    /// Heap requests and named construction controls, excluding shared scratch.
    pub fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Sum of actual heap allocation requests, excluding named controls and scratch.
    pub fn heap_bytes(&self) -> usize {
        self.heap_bytes
    }
    /// Number of nonzero allocation requests in a successful construction.
    pub fn allocation_count(&self) -> usize {
        self.allocations
    }
    /// Opaque scratch geometry derived only from this genuine plan.
    pub fn scratch_layout(&self) -> ScratchLayout {
        self.scratch
    }
    /// Consume one plan and fill fresh storage using already prepared scratch.
    pub fn prepare(self, scratch: &mut Scratch) -> Result<PikeVM, Failure> {
        {
            #[cfg(feature = "workspace-test-support")]
            let fault = self.fault;
            #[cfg(not(feature = "workspace-test-support"))]
            let fault = None;
            self.prepare_inner(scratch, fault)
        }
    }
    /// Fixed real destination overflow for dependency-development tests only.
    #[cfg(feature = "workspace-test-support")]
    pub fn fail_reserve_for_testing(
        mut self,
        buffer: ConstructionBuffer,
    ) -> Result<Self, PlanError> {
        self.fault = Some(match buffer {
            ConstructionBuffer::States => Fault::States,
            ConstructionBuffer::Starts => Fault::Starts,
            ConstructionBuffer::CaptureNames => {
                Fault::Groups(groups::Buffer::Names)
            }
            ConstructionBuffer::LastTransitions => Fault::Transitions(
                self.recipe
                    .states
                    .iter()
                    .rposition(|s| matches!(s, StateRecipe::Sparse(_)))
                    .ok_or(PlanError::Profile)?,
            ),
        });
        Ok(self)
    }
    fn prepare_inner(
        self,
        scratch: &mut Scratch,
        fault: Option<Fault>,
    ) -> Result<PikeVM, Failure> {
        let mut storage = Storage::new();
        if !scratch.layout.fits(self.scratch) {
            return Err(Failure {
                cause: Cause::Scratch,
                storage,
            });
        }
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
        reserve!(storage.states, self.recipe.states.len(), Fault::States);
        reserve!(storage.starts, self.recipe.starts.len(), Fault::Starts);
        let group_info = match self.groups.prepare(match fault {
            Some(Fault::Groups(buffer)) => Some(buffer),
            _ => None,
        }) {
            Ok(value) => value,
            Err(error) => {
                return Err(Failure {
                    cause: Cause::Groups(error),
                    storage,
                })
            }
        };
        let mut look_matcher = LookMatcher::new();
        look_matcher.set_line_terminator(self.recipe.line_terminator);
        storage.inner = Some(Inner {
            states: mem::take(&mut storage.states),
            start_anchored: sid(self.recipe.anchored),
            start_unanchored: sid(self.recipe.unanchored),
            start_pattern: mem::take(&mut storage.starts),
            group_info,
            byte_class_set: ByteClassSet::empty(),
            byte_classes: ByteClasses::empty(),
            has_capture: false,
            has_empty: false,
            utf8: self.recipe.utf8,
            reverse: self.recipe.reverse,
            look_matcher,
            look_set_any: LookSet::empty(),
            look_set_prefix_any: LookSet::empty(),
            memory_extra: 0,
        });
        for (index, recipe) in self.recipe.states.iter().enumerate() {
            let state = match *recipe {
                StateRecipe::ByteRange(start, end, next) => State::ByteRange {
                    trans: Transition {
                        start,
                        end,
                        next: sid(next),
                    },
                },
                StateRecipe::Sparse(ts) => {
                    reserve!(
                        storage.transitions,
                        ts.len(),
                        Fault::Transitions(index)
                    );
                    for &(start, end, next) in ts {
                        storage.transitions.push(Transition {
                            start,
                            end,
                            next: sid(next),
                        });
                    }
                    State::Sparse(SparseTransitions {
                        transitions: mem::take(&mut storage.transitions)
                            .into_boxed_slice(),
                    })
                }
                StateRecipe::Look(look, next) => State::Look {
                    look,
                    next: sid(next),
                },
                StateRecipe::Union(ids) => {
                    reserve!(
                        storage.alternates,
                        ids.len(),
                        Fault::Alternates(index)
                    );
                    for &id in ids {
                        storage.alternates.push(sid(id));
                    }
                    State::Union {
                        alternates: mem::take(&mut storage.alternates)
                            .into_boxed_slice(),
                    }
                }
                StateRecipe::BinaryUnion(a, b) => State::BinaryUnion {
                    alt1: sid(a),
                    alt2: sid(b),
                },
                StateRecipe::Capture {
                    next,
                    pattern,
                    group,
                    slot,
                } => State::Capture {
                    next: sid(next),
                    pattern_id: PatternID::new(pattern).unwrap(),
                    group_index: SmallIndex::new(group).unwrap(),
                    slot: SmallIndex::new(slot).unwrap(),
                },
                StateRecipe::Fail => State::Fail,
                StateRecipe::Match(pattern) => State::Match {
                    pattern_id: PatternID::new(pattern).unwrap(),
                },
            };
            storage.inner.as_mut().unwrap().add(state);
        }
        let inner = storage.inner.as_mut().unwrap();
        for &start in self.recipe.starts {
            inner.start_pattern.push(sid(start));
        }
        scratch.seen.resize(self.recipe.states.len());
        inner.finalize_properties(&mut scratch.stack, &mut scratch.seen);
        if !properties_match(inner, &self.recipe.expected) {
            return Err(Failure {
                cause: Cause::Properties,
                storage,
            });
        }
        // Equal-capacity buffers move into final owners without shrinking.
        // Arc::new has a real priced request but process-OOM semantics.
        let nfa = NFA(Arc::new(storage.inner.take().unwrap()));
        Ok(PikeVM::from_source_nfa(nfa))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fault {
    States,
    Starts,
    Groups(groups::Buffer),
    Transitions(usize),
    Alternates(usize),
}
#[derive(Debug)]
struct Storage {
    states: Vec<State>,
    starts: Vec<StateID>,
    inner: Option<Inner>,
    transitions: Vec<Transition>,
    alternates: Vec<StateID>,
}
impl Storage {
    fn new() -> Self {
        Self {
            states: Vec::new(),
            starts: Vec::new(),
            inner: None,
            transitions: Vec::new(),
            alternates: Vec::new(),
        }
    }
}
#[derive(Debug)]
enum Cause {
    Reserve(TryReserveError),
    Groups(groups::Failure),
    Capacity,
    Scratch,
    Properties,
}
/// A failed single construction retaining every allocated prefix by value.
#[derive(Debug)]
pub struct Failure {
    cause: Cause,
    storage: Storage,
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Reserve(error) => error.fmt(f),
            Cause::Groups(error) => error.fmt(f),
            Cause::Capacity => {
                f.write_str("source construction capacity mismatch")
            }
            Cause::Scratch => {
                f.write_str("source construction scratch is too small")
            }
            Cause::Properties => {
                f.write_str("source construction property mismatch")
            }
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Reserve(error) => Some(error),
            Cause::Groups(error) => Some(error),
            _ => None,
        }
    }
}

/// Three-buffer finalization geometry; can only originate from actual plans.
#[derive(Clone, Copy, Debug)]
pub struct ScratchLayout {
    stack: usize,
    states: usize,
}
impl ScratchLayout {
    /// Combine genuine layouts for sequential construction with one scratch.
    pub fn union(self, other: Self) -> Self {
        Self {
            stack: self.stack.max(other.stack),
            states: self.states.max(other.states),
        }
    }
    fn fits(self, other: Self) -> bool {
        self.stack >= other.stack && self.states >= other.states
    }
    /// Sum of the three actual nonzero heap requests, excluding controls.
    pub fn heap_bytes(self) -> Result<usize, PlanError> {
        add(
            groups::array::<StateID>(self.stack)?,
            add(
                groups::array::<StateID>(self.states)?,
                groups::array::<StateID>(self.states)?,
            )?,
        )
    }
    /// The actual buffers and their closed preparation/return controls.
    pub fn required_bytes(self) -> Result<usize, PlanError> {
        let mut bytes = groups::array::<StateID>(self.stack)?;
        bytes = add(bytes, groups::array::<StateID>(self.states)?)?;
        bytes = add(bytes, groups::array::<StateID>(self.states)?)?;
        for value in [
            mem::size_of::<Self>(),
            mem::size_of::<Scratch>(),
            mem::size_of::<ScratchFailure>(),
            mem::size_of::<Result<Scratch, ScratchFailure>>(),
            mem::size_of::<Result<(), TryReserveError>>(),
        ] {
            bytes = add(bytes, value)?;
        }
        Ok(bytes)
    }
    /// Prepare each real scratch buffer once, retaining partial failures.
    pub fn prepare(self) -> Result<Scratch, ScratchFailure> {
        self.prepare_inner(None)
    }
    /// Fail one actual scratch reserve by capacity overflow for dependency tests.
    /// The same one-attempt partial owner and real reserve cause are returned.
    #[cfg(feature = "workspace-test-support")]
    pub fn prepare_failing(
        self,
        buffer: ScratchFailureBuffer,
    ) -> Result<Scratch, ScratchFailure> {
        self.prepare_inner(Some(match buffer {
            ScratchFailureBuffer::Stack => ScratchBuffer::Stack,
            ScratchFailureBuffer::Dense => ScratchBuffer::Dense,
            ScratchFailureBuffer::Sparse => ScratchBuffer::Sparse,
        }))
    }
    fn prepare_inner(
        self,
        fault: Option<ScratchBuffer>,
    ) -> Result<Scratch, ScratchFailure> {
        let mut scratch = Scratch {
            layout: self,
            stack: Vec::new(),
            seen: SparseSet::new(0),
        };
        for buffer in [
            ScratchBuffer::Stack,
            ScratchBuffer::Dense,
            ScratchBuffer::Sparse,
        ] {
            let expected = if buffer == ScratchBuffer::Stack {
                self.stack
            } else {
                self.states
            };
            let requested = if fault == Some(buffer) {
                usize::MAX
            } else {
                expected
            };
            let result = match buffer {
                ScratchBuffer::Stack => {
                    scratch.stack.try_reserve_exact(requested)
                }
                ScratchBuffer::Dense => {
                    scratch.seen.try_reserve_workspace_dense(requested)
                }
                ScratchBuffer::Sparse => {
                    scratch.seen.try_reserve_workspace_sparse(requested)
                }
            };
            if let Err(error) = result {
                return Err(ScratchFailure {
                    cause: Some(error),
                    scratch,
                });
            }
            let (dense, sparse) = scratch.seen.workspace_capacities();
            let actual = match buffer {
                ScratchBuffer::Stack => scratch.stack.capacity(),
                ScratchBuffer::Dense => dense,
                ScratchBuffer::Sparse => sparse,
            };
            if actual != expected {
                return Err(ScratchFailure {
                    cause: None,
                    scratch,
                });
            }
        }
        scratch.seen.resize(self.states);
        Ok(scratch)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScratchBuffer {
    Stack,
    Dense,
    Sparse,
}
/// Closed reusable finalization storage, with no raw buffers or source owners.
#[derive(Debug)]
pub struct Scratch {
    layout: ScratchLayout,
    stack: Vec<StateID>,
    seen: SparseSet,
}
/// A failed scratch reserve retaining the real earlier buffers and cause.
#[derive(Debug)]
pub struct ScratchFailure {
    cause: Option<TryReserveError>,
    scratch: Scratch,
}
impl fmt::Display for ScratchFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Some(error) => error.fmt(f),
            None => f.write_str("source scratch capacity mismatch"),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for ScratchFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause.as_ref().map(|e| e as _)
    }
}

fn sid(value: usize) -> StateID {
    StateID::new(value).expect("validated source state")
}
fn add(a: usize, b: usize) -> Result<usize, PlanError> {
    a.checked_add(b).ok_or(PlanError::Overflow)
}
fn looks(values: &[Look]) -> LookSet {
    values
        .iter()
        .fold(LookSet::empty(), |set, &look| set.insert(look))
}
fn properties_match(inner: &Inner, expected: &PropertiesRecipe) -> bool {
    inner.has_empty == expected.has_empty
        && inner.has_capture == expected.has_capture
        && inner.look_set_any == looks(expected.look_any)
        && inner.look_set_prefix_any == looks(expected.look_prefix)
        && (0u16..256).all(|b| {
            inner.byte_classes.get(b as u8)
                == expected.byte_classes[b as usize]
        })
}
fn validate(recipe: &NfaRecipe) -> Result<ScratchLayout, PlanError> {
    if !recipe.syntax.is_default()
        || !recipe.utf8
        || recipe.reverse
        || recipe.line_terminator != 10
    {
        return Err(PlanError::Profile);
    }
    let n = recipe.states.len();
    if n == 0
        || n > StateID::LIMIT
        || recipe.starts.len() != 1
        || recipe.groups.len() != 1
        || recipe.groups[0].is_empty()
        || recipe.expected.byte_classes.len() != 256
    {
        return Err(PlanError::Recipe);
    }
    let edge = |id: usize| {
        if id < n {
            Ok(())
        } else {
            Err(PlanError::Recipe)
        }
    };
    edge(recipe.anchored)?;
    edge(recipe.unanchored)?;
    edge(recipe.starts[0])?;
    if recipe.starts[0] != recipe.anchored {
        return Err(PlanError::Recipe);
    }
    for (group, &slots) in recipe.groups[0].iter().enumerate() {
        let start = group.checked_mul(2).ok_or(PlanError::Overflow)?;
        if slots != (start, add(start, 1)?) {
            return Err(PlanError::Recipe);
        }
    }
    let mut pushes = 1;
    let mut matches = 0;
    for state in recipe.states {
        match *state {
            StateRecipe::ByteRange(a, b, next) => {
                if a > b {
                    return Err(PlanError::Recipe);
                }
                edge(next)?;
            }
            StateRecipe::Sparse(ts) => {
                let mut previous = None;
                for &(a, b, next) in ts {
                    if a > b || previous.map_or(false, |end| end >= a) {
                        return Err(PlanError::Recipe);
                    }
                    previous = Some(b);
                    edge(next)?;
                }
            }
            StateRecipe::Look(look, next) => {
                LookSet::empty()
                    .insert(look)
                    .available()
                    .map_err(|_| PlanError::LookUnavailable)?;
                edge(next)?;
                pushes = add(pushes, 1)?;
            }
            StateRecipe::Union(ids) => {
                for &id in ids {
                    edge(id)?;
                }
                pushes = add(pushes, ids.len())?;
            }
            StateRecipe::BinaryUnion(a, b) => {
                edge(a)?;
                edge(b)?;
                pushes = add(pushes, 2)?;
            }
            StateRecipe::Capture {
                next,
                pattern,
                group,
                slot,
            } => {
                edge(next)?;
                if pattern != 0
                    || group >= recipe.groups[0].len()
                    || (slot != recipe.groups[0][group].0
                        && slot != recipe.groups[0][group].1)
                {
                    return Err(PlanError::Recipe);
                }
                SmallIndex::new(slot).map_err(|_| PlanError::Overflow)?;
                pushes = add(pushes, 1)?;
            }
            StateRecipe::Match(pattern) => {
                if pattern != 0 {
                    return Err(PlanError::Recipe);
                }
                matches = add(matches, 1)?;
            }
            StateRecipe::Fail => {}
        }
    }
    if matches > 1 {
        return Err(PlanError::Recipe);
    }
    groups::array::<StateID>(pushes)?;
    groups::array::<StateID>(n)?;
    Ok(ScratchLayout {
        stack: pushes,
        states: n,
    })
}

/// Fixed finalization scratch reserve target for dependency-development tests.
#[cfg(feature = "workspace-test-support")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScratchFailureBuffer {
    /// Epsilon traversal stack.
    Stack,
    /// Dense visited-state array.
    Dense,
    /// Sparse visited-state array.
    Sparse,
}

/// Fixed destination selector enabled only for dependency-development tests.
#[cfg(feature = "workspace-test-support")]
#[derive(Clone, Copy, Debug)]
pub enum ConstructionBuffer {
    /// State vector.
    States,
    /// Pattern start vector.
    Starts,
    /// Final anonymous capture name row.
    CaptureNames,
    /// Last actual sparse transition buffer in this source.
    LastTransitions,
}

#[cfg(all(test, feature = "syntax"))]
mod tests;
