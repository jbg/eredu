//! Explicit, source-bound reusable search storage for a checked VM profile.
//!
//! Source compilation remains ordinary, unfunded work. Planning inspects its
//! actual instructions and direct PikeVMs without allocating. Preparation makes
//! one reserve attempt per actual destination and retains partial owners on
//! failure. Search and iteration use the same workers as the ordinary APIs.
//! No source, instruction program, delegate or raw cache can be extracted.

mod retirement;
pub use retirement::RetiredPrepareError;

use alloc::{collections::TryReserveError, string::String, vec::Vec};
use core::{alloc::Layout, fmt, mem::size_of};
use regex_automata::nfa::thompson::pikevm::{self, workspace as pike};
use regex_automata::{util::primitives::NonMaxUsize, Input};

use super::{Branch, Insn, Prog, Save, State, MAX_STACK};
use crate::{analyze::analyze, can_compile_as_anchored, compile, optimize};
use crate::{Error, Expr, Match, RegexBuilder, RegexOptions, RuntimeError};

/// Fixed pre-reserve profile or layout rejection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanError {
    /// An opcode requires storage or semantics not covered by this profile.
    Instruction,
    /// The selected Unicode assertion tables are unavailable.
    UnicodeUnavailable,
    /// An instruction target, capture range or save slot is invalid.
    Geometry,
    /// Checked element/layout/control arithmetic overflowed.
    CapacityOverflow,
    /// An actual direct PikeVM rejects its workspace profile.
    Delegate(pike::PlanError),
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnicodeUnavailable => f.write_str("Unicode assertion tables unavailable"),
            Self::Instruction => f.write_str("unsupported workspace instruction"),
            Self::Geometry => f.write_str("invalid workspace instruction geometry"),
            Self::CapacityOverflow => f.write_str("workspace capacity overflow"),
            Self::Delegate(error) => write!(f, "delegate workspace: {}", error),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for PlanError {}

/// An ordinary source compilation or actual instruction-profile failure.
#[derive(Debug)]
pub enum SourceError {
    /// Existing parser/compiler error, including the real PikeVM build error.
    Compile(Error),
    /// Checked instruction/workspace profile rejection.
    Profile(PlanError),
}
impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(e) => e.fmt(f),
            Self::Profile(e) => e.fmt(f),
        }
    }
}
impl From<Error> for SourceError {
    fn from(error: Error) -> Self {
        Self::Compile(error)
    }
}
impl From<PlanError> for SourceError {
    fn from(error: PlanError) -> Self {
        Self::Profile(error)
    }
}
#[cfg(feature = "std")]
impl std::error::Error for SourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compile(error) => {
                if let Error::CompileError(cause) = error {
                    if let crate::CompileError::PikeVmBuildError(cause) = cause.as_ref() {
                        return Some(cause);
                    }
                }
                Some(error)
            }
            Self::Profile(error) => Some(error),
        }
    }
}

#[derive(Debug)]
enum Body {
    Wrap {
        inner: pikevm::PikeVM,
        explicit_group_zero: bool,
    },
    Fancy(Prog),
}

/// One immutable direct-PikeVM source; compilation is not workspace-funded.
///
/// There is no raw program/source export, clone or source replacement API.
#[derive(Debug)]
pub struct Source {
    body: Body,
    options: RegexOptions,
}
impl Source {
    /// Borrow the exact source text used to construct this immutable program.
    /// This exposes no program, delegate, mutable configuration or owner.
    pub fn pattern(&self) -> &str {
        &self.options.pattern
    }

    /// Compile using ordinary default syntax/options and the explicit profile.
    pub fn new(pattern: &str) -> Result<Self, SourceError> {
        RegexBuilder::new(pattern).build_workspace_source()
    }
    pub(crate) fn new_options(options: RegexOptions) -> Result<Self, SourceError> {
        let mut tree = Expr::parse_tree_with_flags(&options.pattern, options.compute_flags())?;
        let fixup = optimize(&mut tree);
        let info = analyze(&tree, fixup)?;
        let body = if !info.hard {
            let mut cooked = String::new();
            tree.expr.to_str(&mut cooked, 0);
            Body::Wrap {
                inner: compile::compile_pikevm(&cooked, &options)?,
                explicit_group_zero: fixup,
            }
        } else {
            Body::Fancy(compile::compile_workspace(
                &info,
                can_compile_as_anchored(&tree.expr),
                &options,
            )?)
        };
        let source = Self { body, options };
        source.plan()?;
        Ok(source)
    }
    /// Inspect this exact source without allocating a destination.
    pub fn plan(&self) -> Result<Plan<'_>, PlanError> {
        Ok(Plan {
            source: self,
            requirements: Requirements::for_source(self)?,
            #[cfg(any(test, feature = "workspace-test-support"))]
            fail: None,
            #[cfg(any(test, feature = "workspace-test-support"))]
            delegate_fail: None,
        })
    }
    fn delegates(&self) -> Delegates<'_> {
        match &self.body {
            Body::Wrap { inner, .. } => Delegates::Wrap(Some(inner)),
            Body::Fancy(prog) => Delegates::Fancy(prog.body.iter()),
        }
    }
}
enum Delegates<'s> {
    Wrap(Option<&'s pikevm::PikeVM>),
    Fancy(core::slice::Iter<'s, Insn>),
}
impl<'s> Iterator for Delegates<'s> {
    type Item = &'s pikevm::PikeVM;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Wrap(inner) => inner.take(),
            Self::Fancy(iter) => iter.find_map(|insn| match insn {
                Insn::PikeDelegate(delegate) => Some(&delegate.inner),
                _ => None,
            }),
        }
    }
}

/// One of the five actual outer reserve targets, in preparation order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Buffer {
    /// Owners of the direct delegate workspaces.
    Delegates,
    /// Current VM save slots.
    Saves,
    /// Backtrack branches, bounded by the existing VM stack limit.
    Branches,
    /// Undo records for at most S distinct slots in each branch segment.
    Undo,
    /// Shared delegate result slots, with an exact per-delegate slice.
    Slots,
}
impl Buffer {
    const ALL: [Self; 5] = [
        Self::Delegates,
        Self::Saves,
        Self::Branches,
        Self::Undo,
        Self::Slots,
    ];
}

/// Checked requested layouts and concrete control overlaps for search only.
/// Allocator overhead and possible excess capacity are not a total heap bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Requirements {
    delegates: usize,
    saves: usize,
    branches: usize,
    undo: usize,
    slots: usize,
    buffers: usize,
    controls: usize,
    nested: usize,
    total: usize,
}
impl Requirements {
    fn for_source(source: &Source) -> Result<Self, PlanError> {
        let saves = match &source.body {
            Body::Wrap { .. } => 0,
            Body::Fancy(prog) => {
                validate(prog)?;
                prog.n_saves
            }
        };
        let mut count = 0usize;
        let mut slots = 0;
        let mut nested = 0usize;
        for inner in source.delegates() {
            let requirements = inner
                .workspace_plan()
                .map_err(PlanError::Delegate)?
                .requirements();
            count = add(count, 1)?;
            slots = slots.max(requirements.maximum_slots());
            nested = add(nested, requirements.required_bytes())?;
        }
        if let Body::Wrap {
            explicit_group_zero: true,
            ..
        } = &source.body
        {
            if slots < 4 {
                return Err(PlanError::Geometry);
            }
        }
        Self::checked(count, saves, slots, nested)
    }
    fn checked(
        delegates: usize,
        saves: usize,
        slots: usize,
        nested: usize,
    ) -> Result<Self, PlanError> {
        let branches = if saves == 0 { 0 } else { MAX_STACK };
        let undo = saves
            .checked_mul(add(branches, 1)?)
            .ok_or(PlanError::CapacityOverflow)?;
        let buffers = [
            array::<pike::Workspace<'static>>(delegates)?,
            array::<usize>(saves)?,
            array::<Branch>(branches)?,
            array::<Save>(undo)?,
            array::<Option<NonMaxUsize>>(slots)?,
        ]
        .iter()
        .try_fold(0, |sum, &n| add(sum, n))?;
        let controls = [
            size_of::<Plan<'static>>(),
            size_of::<Requirements>(),
            size_of::<Storage<'static>>(),
            size_of::<Workspace<'static>>(),
            size_of::<PrepareError<'static>>(),
            size_of::<RetiredPrepareError>(),
            size_of::<[usize; 5]>(),
            size_of::<Result<Plan<'static>, PlanError>>(),
            size_of::<Result<Workspace<'static>, PrepareError<'static>>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Buffer>(),
            size_of::<Matches<'static, 'static, 'static>>(),
            size_of::<Result<Option<Match<'static>>, Error>>(),
            size_of::<Option<Result<Match<'static>, Error>>>(),
        ]
        .iter()
        .try_fold(0, |sum, &n| add(sum, n))?;
        let total = add(add(buffers, controls)?, nested)?;
        Ok(Self {
            delegates,
            saves,
            branches,
            undo,
            slots,
            buffers,
            controls,
            nested,
            total,
        })
    }
    /// Requested elements in the actual outer target.
    pub fn capacity(&self, buffer: Buffer) -> usize {
        match buffer {
            Buffer::Delegates => self.delegates,
            Buffer::Saves => self.saves,
            Buffer::Branches => self.branches,
            Buffer::Undo => self.undo,
            Buffer::Slots => self.slots,
        }
    }
    /// Requested layouts for the five outer buffers.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Named concrete outer owners, errors, iteration and return overlaps.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Sum of each actual nested PikeVM workspace's layouts and controls.
    pub fn delegate_bytes(&self) -> usize {
        self.nested
    }
    /// Checked sum of outer buffers, controls and all actual nested workspaces.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
fn add(a: usize, b: usize) -> Result<usize, PlanError> {
    a.checked_add(b).ok_or(PlanError::CapacityOverflow)
}
fn array<T>(count: usize) -> Result<usize, PlanError> {
    Layout::array::<T>(count)
        .map(|layout| layout.size())
        .map_err(|_| PlanError::CapacityOverflow)
}

// Sources are private compiler output, never arbitrary user instruction arrays.
// Validate all operands used by the allocation-free subset. In particular,
// atomic cuts (BTreeSet/explicit stack), backref case folding (AST/meta), reverse
// pools and any legacy meta delegate cannot enter this search path.
fn validate(prog: &Prog) -> Result<(), PlanError> {
    if prog.n_saves < 2 || !matches!(prog.body.last(), Some(Insn::End)) {
        return Err(PlanError::Geometry);
    }
    let slot = |n: usize| {
        if n < prog.n_saves {
            Ok(())
        } else {
            Err(PlanError::Geometry)
        }
    };
    let target = |n: usize| {
        if n < prog.body.len() {
            Ok(())
        } else {
            Err(PlanError::Geometry)
        }
    };
    let mut ordinal = 0;
    for (pc, insn) in prog.body.iter().enumerate() {
        match insn {
            Insn::End | Insn::Any | Insn::AnyNoNL | Insn::Lit(_) => {}
            Insn::Assertion(assertion) => {
                use crate::Assertion::*;
                if matches!(
                    assertion,
                    LeftWordBoundary
                        | RightWordBoundary
                        | LeftWordHalfBoundary
                        | RightWordHalfBoundary
                        | WordBoundary
                        | NotWordBoundary
                ) && regex_automata::util::look::UnicodeWordBoundaryError::check().is_err()
                {
                    return Err(PlanError::UnicodeUnavailable);
                }
            }
            Insn::Split(x, y) => {
                target(*x)?;
                target(*y)?;
            }
            Insn::Jmp(n) => target(*n)?,
            Insn::Save(n) | Insn::Save0(n) | Insn::Restore(n) => slot(*n)?,
            Insn::RepeatGr { next, repeat, .. } | Insn::RepeatNg { next, repeat, .. } => {
                target(*next)?;
                slot(*repeat)?;
            }
            Insn::RepeatEpsilonGr {
                next,
                repeat,
                check,
                ..
            }
            | Insn::RepeatEpsilonNg {
                next,
                repeat,
                check,
                ..
            } => {
                target(*next)?;
                slot(*repeat)?;
                slot(*check)?;
            }
            Insn::FailNegativeLookAround => {
                target(add(pc, 1)?)?;
            }
            Insn::PikeDelegate(delegate) => {
                if delegate.ordinal != ordinal {
                    return Err(PlanError::Geometry);
                }
                ordinal = add(ordinal, 1)?;
                let width = delegate
                    .inner
                    .workspace_plan()
                    .map_err(PlanError::Delegate)?
                    .requirements()
                    .maximum_slots();
                if let Some(range) = delegate.capture_groups {
                    let end = range
                        .end()
                        .checked_mul(2)
                        .ok_or(PlanError::CapacityOverflow)?;
                    if range.start() >= range.end() || end > prog.n_saves {
                        return Err(PlanError::Geometry);
                    }
                    let needed = add(range.end() - range.start(), 1)?
                        .checked_mul(2)
                        .ok_or(PlanError::CapacityOverflow)?;
                    if width < needed {
                        return Err(PlanError::Geometry);
                    }
                }
            }
            _ => return Err(PlanError::Instruction),
        }
    }
    Ok(())
}

/// A one-attempt plan tied to its immutable source.
///
/// ```compile_fail
/// use fancy_regex::workspace::Source;
/// let source = Source::new("a").unwrap();
/// let plan = source.plan().unwrap();
/// let first = plan.prepare();
/// let second = plan.prepare();
/// ```
#[derive(Debug)]
pub struct Plan<'s> {
    source: &'s Source,
    requirements: Requirements,
    #[cfg(any(test, feature = "workspace-test-support"))]
    fail: Option<Buffer>,
    #[cfg(any(test, feature = "workspace-test-support"))]
    delegate_fail: Option<(usize, pike::Buffer)>,
}
/// Fixed actual reserve target for dependency-development tests only.
#[cfg(feature = "workspace-test-support")]
#[derive(Clone, Copy, Debug)]
pub enum PrepareFailure {
    /// One of the five existing outer destinations.
    Outer(Buffer),
    /// One actual delegate's destination; ordinal is checked against this source.
    Delegate {
        /// Zero-based ordinal of the actual delegate in this source.
        ordinal: usize,
        /// Actual reserve site within the selected delegate workspace.
        buffer: DelegateBuffer,
    },
}
/// Actual nested PikeVM destination selector; this supplies no capacity.
#[cfg(feature = "workspace-test-support")]
pub use pike::Buffer as DelegateBuffer;

impl<'s> Plan<'s> {
    /// Select a real target for capacity overflow, preserving this exact source.
    /// No successful override, arbitrary capacity or callback is accepted.
    #[cfg(feature = "workspace-test-support")]
    pub fn fail_reservation(mut self, failure: PrepareFailure) -> Result<Self, PlanError> {
        match failure {
            PrepareFailure::Outer(buffer) => {
                self.fail = Some(buffer);
                self.delegate_fail = None;
            }
            PrepareFailure::Delegate { ordinal, buffer } => {
                if ordinal >= self.requirements.delegates {
                    return Err(PlanError::Geometry);
                }
                self.fail = None;
                self.delegate_fail = Some((ordinal, buffer));
            }
        }
        Ok(self)
    }

    /// Checked requirements from the bound source.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Reserve each real target once; return all partial ownership on failure.
    pub fn prepare(self) -> Result<Workspace<'s>, PrepareError<'s>> {
        let mut partial = Storage::empty();
        for buffer in Buffer::ALL {
            let capacity = self.requested(buffer);
            let result = match buffer {
                Buffer::Delegates => partial.delegates.try_reserve_exact(capacity),
                Buffer::Saves => partial.state.saves.try_reserve_exact(capacity),
                Buffer::Branches => partial.state.stack.try_reserve_exact(capacity),
                Buffer::Undo => partial.state.oldsave.try_reserve_exact(capacity),
                Buffer::Slots => partial.slots.try_reserve_exact(capacity),
            };
            if let Err(cause) = result {
                return Err(PrepareError {
                    source: self.source,
                    requirements: self.requirements,
                    partial,
                    cause: PrepareCause::Reserve { buffer, cause },
                });
            }
        }
        for (ordinal, source) in self.source.delegates().enumerate() {
            let plan = match source.workspace_plan() {
                Ok(plan) => plan,
                Err(cause) => {
                    return Err(PrepareError {
                        source: self.source,
                        requirements: self.requirements,
                        partial,
                        cause: PrepareCause::Profile(cause),
                    })
                }
            };
            #[cfg(any(test, feature = "workspace-test-support"))]
            let plan = match self.delegate_fail {
                Some((selected, buffer)) if selected == ordinal => plan.fail_reservation(buffer),
                _ => plan,
            };
            match plan.prepare() {
                Ok(workspace) => partial.delegates.push(workspace),
                Err(cause) => {
                    return Err(PrepareError {
                        source: self.source,
                        requirements: self.requirements,
                        partial,
                        cause: PrepareCause::Delegate { ordinal, cause },
                    })
                }
            }
        }
        partial
            .state
            .saves
            .resize(self.requirements.saves, usize::MAX);
        partial.state.explicit_sp = self.requirements.saves;
        partial.slots.resize(self.requirements.slots, None);
        Ok(Workspace {
            source: self.source,
            requirements: self.requirements,
            storage: partial,
        })
    }
    fn requested(&self, buffer: Buffer) -> usize {
        #[cfg(any(test, feature = "workspace-test-support"))]
        if self.fail == Some(buffer) {
            return usize::MAX;
        }
        self.requirements.capacity(buffer)
    }
}
struct Storage<'s> {
    state: State,
    slots: Vec<Option<NonMaxUsize>>,
    delegates: Vec<pike::Workspace<'s>>,
}
impl Storage<'_> {
    fn empty() -> Self {
        Self {
            state: State {
                saves: Vec::new(),
                stack: Vec::new(),
                oldsave: Vec::new(),
                nsave: 0,
                explicit_sp: 0,
                max_stack: MAX_STACK,
                options: 0,
            },
            slots: Vec::new(),
            delegates: Vec::new(),
        }
    }
    fn capacities(&self) -> [usize; 5] {
        [
            self.delegates.capacity(),
            self.state.saves.capacity(),
            self.state.stack.capacity(),
            self.state.oldsave.capacity(),
            self.slots.capacity(),
        ]
    }
    fn reset(&mut self, flags: u32) {
        self.state.saves.fill(usize::MAX);
        self.state.stack.clear();
        self.state.oldsave.clear();
        self.state.nsave = 0;
        self.state.explicit_sp = self.state.saves.len();
        self.state.options = flags;
    }
}
#[derive(Debug)]
enum PrepareCause<'s> {
    Reserve {
        buffer: Buffer,
        cause: TryReserveError,
    },
    Delegate {
        ordinal: usize,
        cause: pike::PrepareError<'s>,
    },
    Profile(pike::PlanError),
}

/// Closed failure retaining the real reserve error, all completed buffers and
/// any partially constructed delegate. Its source remains borrowed throughout.
///
/// ```compile_fail
/// use fancy_regex::workspace::{PrepareError, Source};
/// fn escape<'a>(error: PrepareError<'a>) -> PrepareError<'static> { error }
/// ```
pub struct PrepareError<'s> {
    source: &'s Source,
    requirements: Requirements,
    partial: Storage<'s>,
    cause: PrepareCause<'s>,
}
impl<'s> PrepareError<'s> {
    /// The planned requested layouts.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Actual retained outer capacities, in Buffer order.
    pub fn capacities(&self) -> [usize; 5] {
        self.partial.capacities()
    }
    /// Failed outer target, if failure occurred before delegate construction.
    pub fn buffer(&self) -> Option<Buffer> {
        match self.cause {
            PrepareCause::Reserve { buffer, .. } => Some(buffer),
            _ => None,
        }
    }
    /// Borrow the real reserve cause without extracting any partial storage.
    pub fn reserve_error(&self) -> Option<&TryReserveError> {
        match &self.cause {
            PrepareCause::Reserve { cause, .. } => Some(cause),
            PrepareCause::Delegate { cause, .. } => Some(cause.cause()),
            _ => None,
        }
    }
    /// Borrow the nested real partial failure, if present.
    pub fn delegate_error(&self) -> Option<(usize, &pike::PrepareError<'s>)> {
        match &self.cause {
            PrepareCause::Delegate { ordinal, cause } => Some((*ordinal, cause)),
            _ => None,
        }
    }
    /// Number of completed delegates still retained alongside the failure.
    pub fn completed_delegates(&self) -> usize {
        self.partial.delegates.len()
    }
}
impl fmt::Debug for PrepareError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrepareError")
            .field("requirements", &self.requirements)
            .field("capacities", &self.capacities())
            .field("cause", &self.cause)
            .field("source", &(self.source as *const Source))
            .finish()
    }
}
impl fmt::Display for PrepareError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            PrepareCause::Reserve { cause, .. } => cause.fmt(f),
            PrepareCause::Delegate { cause, .. } => cause.fmt(f),
            PrepareCause::Profile(cause) => cause.fmt(f),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for PrepareError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Some(error) = self.reserve_error() {
            return Some(error);
        }
        match &self.cause {
            PrepareCause::Profile(error) => Some(error),
            _ => None,
        }
    }
}

/// Prepared search storage permanently borrowing the compiled source.
///
/// ```compile_fail
/// use fancy_regex::workspace::Source;
/// let mut source = Source::new("a").unwrap();
/// let workspace = source.plan().unwrap().prepare();
/// source = Source::new("b").unwrap();
/// drop(workspace);
/// ```
/// ```compile_fail
/// use fancy_regex::workspace::Source;
/// let workspace;
/// { let source = Source::new("a").unwrap(); workspace = source.plan().unwrap().prepare(); }
/// drop(workspace);
/// ```
/// ```compile_fail
/// use fancy_regex::workspace::Source;
/// let source = Source::new("a").unwrap();
/// let workspace = source.plan().unwrap().prepare().unwrap();
/// let copy = workspace.clone();
/// ```
pub struct Workspace<'s> {
    source: &'s Source,
    requirements: Requirements,
    storage: Storage<'s>,
}
impl<'s> Workspace<'s> {
    /// Requested checked layouts and concrete control overlaps.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Actual retained outer capacities in Buffer order.
    pub fn capacities(&self) -> [usize; 5] {
        self.storage.capacities()
    }
    /// Search once from the beginning, reusing the same bound storage.
    pub fn find<'t>(&mut self, text: &'t str) -> crate::Result<Option<Match<'t>>> {
        self.find_from_pos(text, 0)
    }
    /// Search from an exact UTF-8 boundary without slicing away context.
    pub fn find_from_pos<'t>(
        &mut self,
        text: &'t str,
        pos: usize,
    ) -> crate::Result<Option<Match<'t>>> {
        self.search(text, pos, 0)
    }
    /// Iterate using the ordinary empty-match/UTF-8/terminal-error progression.
    pub fn find_iter<'w, 't>(&'w mut self, text: &'t str) -> Matches<'w, 's, 't> {
        Matches {
            workspace: self,
            text,
            last_end: 0,
            last_match: None,
        }
    }
    fn search<'t>(
        &mut self,
        text: &'t str,
        pos: usize,
        flags: u32,
    ) -> crate::Result<Option<Match<'t>>> {
        if pos > text.len() || !text.is_char_boundary(pos) {
            return Err(Error::RuntimeError(RuntimeError::WorkspaceInputBoundary));
        }
        self.storage.reset(flags);
        match &self.source.body {
            Body::Wrap {
                explicit_group_zero,
                ..
            } => {
                let found = self.storage.delegates[0]
                    .search_slots(
                        &Input::new(text).span(pos..text.len()),
                        &mut self.storage.slots,
                    )
                    .map_err(|_| Error::RuntimeError(RuntimeError::WorkspaceInvariant))?;
                if found.is_none() {
                    return Ok(None);
                }
                let start = if *explicit_group_zero { 2 } else { 0 };
                Ok(Some(Match::new(
                    text,
                    self.storage.slots[start].unwrap().get(),
                    self.storage.slots[start + 1].unwrap().get(),
                )))
            }
            Body::Fancy(prog) => {
                if super::run_inner(
                    prog,
                    text,
                    pos,
                    flags,
                    &self.source.options,
                    &mut self.storage.state,
                    &mut self.storage.slots,
                    &mut self.storage.delegates,
                )? {
                    Ok(Some(Match::new(
                        text,
                        self.storage.state.saves[0],
                        self.storage.state.saves[1],
                    )))
                } else {
                    Ok(None)
                }
            }
        }
    }
}
impl fmt::Debug for Workspace<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Workspace")
            .field("requirements", &self.requirements)
            .field("capacities", &self.capacities())
            .finish()
    }
}
/// Borrowed iterator; it cannot outlive or concurrently reuse the workspace.
#[derive(Debug)]
pub struct Matches<'w, 's, 't> {
    workspace: &'w mut Workspace<'s>,
    text: &'t str,
    last_end: usize,
    last_match: Option<usize>,
}
impl<'t> Iterator for Matches<'_, '_, 't> {
    type Item = crate::Result<Match<'t>>;
    fn next(&mut self) -> Option<Self::Item> {
        let workspace = &mut self.workspace;
        crate::next_match(
            self.text,
            &mut self.last_end,
            &mut self.last_match,
            |text, pos, flags| workspace.search(text, pos, flags),
        )
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "std"))]
pub(crate) mod recipe_emitter;

/// Checked fresh construction of exact compiler-emitted sources.
pub mod construction;
