/*!
A closed, fallibly prepared PikeVM search workspace.

This API borrows one already constructed [`PikeVM`]. Planning checks the actual
NFA without allocating. Preparation reserves each of seven actual buffers at
most once; failure retains the partial buffers and the real reserve error.
Search uses the ordinary PikeVM worker with a checked capture-slot slice and
never accepts a replacement regex or exposes its cache.

This is search storage only. It does not bound or fund regex parsing, NFA
construction, source residence, caller result storage or diagnostic formatting.
Ordinary PikeVM and Cache APIs retain their existing behavior.
*/

mod retirement;
pub use retirement::RetiredPrepareError;

use alloc::collections::TryReserveError;
use alloc::vec::Vec;
use core::{alloc::Layout, fmt, mem::size_of};

use super::{ActiveStates, Cache, FollowEpsilon, PikeVM, SlotTable};
use crate::{
    nfa::thompson::State,
    util::{
        primitives::{NonMaxUsize, StateID},
        sparse_set::SparseSet,
    },
    Input, PatternID,
};

/// One of the seven actual workspace reserve destinations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Buffer {
    /// Epsilon-closure exploration and capture-restoration frames.
    Epsilon,
    /// Current active-state dense array.
    CurrentDense,
    /// Current active-state sparse array.
    CurrentSparse,
    /// Next active-state dense array.
    NextDense,
    /// Next active-state sparse array.
    NextSparse,
    /// Current state capture-slot table and absent-slot scratch.
    CurrentSlots,
    /// Next state capture-slot table and absent-slot scratch.
    NextSlots,
}

impl Buffer {
    const ALL: [Buffer; 7] = [
        Buffer::Epsilon,
        Buffer::CurrentDense,
        Buffer::CurrentSparse,
        Buffer::NextDense,
        Buffer::NextSparse,
        Buffer::CurrentSlots,
        Buffer::NextSlots,
    ];
}

/// A fixed rejection before any workspace reserve is attempted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanError {
    /// The optional PikeVM instrumentation can allocate during search.
    Instrumentation,
    /// An installed prefilter is outside this workspace's search contract.
    Prefilter,
    /// This first workspace profile requires exactly one actual pattern.
    MultiplePatterns,
    /// The source's capture slots cannot support the checked search path.
    CaptureLayout,
    /// An element count, layout or byte sum cannot be represented.
    CapacityOverflow,
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            PlanError::Instrumentation => "PikeVM instrumentation is enabled",
            PlanError::Prefilter => "PikeVM workspace excludes prefilters",
            PlanError::MultiplePatterns => {
                "PikeVM workspace requires one pattern"
            }
            PlanError::CaptureLayout => "invalid PikeVM workspace slot layout",
            PlanError::CapacityOverflow => {
                "PikeVM workspace capacity overflow"
            }
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PlanError {}

/// Checked requested storage derived from one immutable source NFA.
///
/// These are requested element and byte counts, not measurements of allocator
/// overhead. A successful allocation may report a larger actual capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Requirements {
    states: usize,
    slots: usize,
    minimum_slots: usize,
    epsilon: usize,
    table: usize,
    buffer_bytes: usize,
    control_bytes: usize,
    required_bytes: usize,
}

impl Requirements {
    fn for_source(source: &PikeVM) -> Result<Requirements, PlanError> {
        if cfg!(feature = "internal-instrument-pikevm") {
            return Err(PlanError::Instrumentation);
        }
        if source.get_config().get_prefilter().is_some() {
            return Err(PlanError::Prefilter);
        }
        let nfa = source.get_nfa();
        if nfa.pattern_len() != 1 {
            return Err(PlanError::MultiplePatterns);
        }
        let slots = nfa.group_info().slot_len();
        let minimum_slots = nfa.group_info().implicit_slot_len();
        if minimum_slots != 2 || slots < minimum_slots {
            return Err(PlanError::CaptureLayout);
        }
        // Each state is inserted in next.set before its epsilon edges are
        // followed. A revisit adds nothing. One closure completely drains the
        // stack: one initial frame plus all possible per-state extra frames
        // therefore bounds its maximum length, including capture restores.
        let mut epsilon = 1usize;
        for state in nfa.states() {
            let extra = match *state {
                State::Union { ref alternates } => {
                    alternates.len().saturating_sub(1)
                }
                State::BinaryUnion { .. } => 1,
                State::Capture { slot, .. } => {
                    if slot.as_usize() >= slots {
                        return Err(PlanError::CaptureLayout);
                    }
                    1
                }
                _ => 0,
            };
            epsilon = epsilon
                .checked_add(extra)
                .ok_or(PlanError::CapacityOverflow)?;
        }
        Requirements::checked(
            nfa.states().len(),
            slots,
            minimum_slots,
            epsilon,
        )
    }

    fn checked(
        states: usize,
        slots: usize,
        minimum_slots: usize,
        epsilon: usize,
    ) -> Result<Requirements, PlanError> {
        if states > StateID::LIMIT || slots < minimum_slots {
            return Err(PlanError::CapacityOverflow);
        }
        let table = states
            .checked_mul(slots)
            .and_then(|n| n.checked_add(slots.max(minimum_slots)))
            .ok_or(PlanError::CapacityOverflow)?;
        let frame_bytes = array_bytes::<FollowEpsilon>(epsilon)?;
        let state_bytes = array_bytes::<StateID>(states)?;
        let slot_bytes = array_bytes::<Option<NonMaxUsize>>(table)?;
        let buffer_bytes = frame_bytes
            .checked_add(
                state_bytes
                    .checked_mul(4)
                    .ok_or(PlanError::CapacityOverflow)?,
            )
            .and_then(|n| n.checked_add(slot_bytes.checked_mul(2)?))
            .ok_or(PlanError::CapacityOverflow)?;
        // Named plan, owner, retained failure and return/control overlaps.
        // This excludes the preexisting NFA and caller-provided slots.
        let control_bytes = [
            size_of::<Plan<'static>>(),
            size_of::<Requirements>(),
            size_of::<Workspace<'static>>(),
            size_of::<PrepareError<'static>>(),
            size_of::<RetiredPrepareError>(),
            size_of::<[usize; 7]>(),
            size_of::<Result<Plan<'static>, PlanError>>(),
            size_of::<Result<Workspace<'static>, PrepareError<'static>>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Option<PatternID>, SearchError>>(),
            size_of::<Buffer>(),
        ]
        .iter()
        .try_fold(0usize, |sum, &n| sum.checked_add(n))
        .ok_or(PlanError::CapacityOverflow)?;
        let required_bytes = buffer_bytes
            .checked_add(control_bytes)
            .ok_or(PlanError::CapacityOverflow)?;
        Ok(Requirements {
            states,
            slots,
            minimum_slots,
            epsilon,
            table,
            buffer_bytes,
            control_bytes,
            required_bytes,
        })
    }

    /// Requested capacity, in elements, for the named actual destination.
    pub fn capacity(&self, buffer: Buffer) -> usize {
        match buffer {
            Buffer::Epsilon => self.epsilon,
            Buffer::CurrentDense
            | Buffer::CurrentSparse
            | Buffer::NextDense
            | Buffer::NextSparse => self.states,
            Buffer::CurrentSlots | Buffer::NextSlots => self.table,
        }
    }

    /// Minimum caller slot count, including the whole-match offsets.
    pub fn minimum_slots(&self) -> usize {
        self.minimum_slots
    }

    /// Maximum caller slot count, equal to the actual NFA's slot row width.
    pub fn maximum_slots(&self) -> usize {
        self.slots
    }

    /// Sum of the seven checked requested buffer layouts.
    pub fn buffer_bytes(&self) -> usize {
        self.buffer_bytes
    }

    /// Sum of the named concrete plan, owner, error and return controls.
    pub fn control_bytes(&self) -> usize {
        self.control_bytes
    }

    /// Checked sum of requested buffers and named controls.
    pub fn required_bytes(&self) -> usize {
        self.required_bytes
    }
}

fn array_bytes<T>(count: usize) -> Result<usize, PlanError> {
    Layout::array::<T>(count)
        .map(|layout| layout.size())
        .map_err(|_| PlanError::CapacityOverflow)
}

/// A move-only preparation plan borrowing one actual PikeVM.
///
/// Preparing consumes this plan, including on reserve failure:
///
/// ```compile_fail
/// use regex_automata::nfa::thompson::pikevm::PikeVM;
/// fn twice(source: &PikeVM) {
///     let plan = source.workspace_plan().unwrap();
///     let first = plan.prepare();
///     let second = plan.prepare();
/// }
/// ```
pub struct Plan<'source> {
    source: &'source PikeVM,
    requirements: Requirements,
    #[cfg(any(test, feature = "workspace-test-support"))]
    fail: Option<Buffer>,
}

impl PikeVM {
    /// Inspect this exact source and plan a closed fallible search workspace.
    ///
    /// This performs no workspace allocation. It currently rejects actual
    /// multipattern sources, prefilters, unavailable capture geometry, and
    /// the optional allocating PikeVM instrumentation. Source construction
    /// itself remains outside this API's storage contract.
    pub fn workspace_plan(&self) -> Result<Plan<'_>, PlanError> {
        Ok(Plan {
            source: self,
            requirements: Requirements::for_source(self)?,
            #[cfg(any(test, feature = "workspace-test-support"))]
            fail: None,
        })
    }
}

impl<'source> Plan<'source> {
    /// The checked requirements for this actual source.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }

    /// Select one actual destination reserve to fail by capacity overflow.
    ///
    /// This disabled-by-default development feature cannot supply capacities,
    /// callbacks, a successful override, or source/admission authority. It keeps
    /// the real reserve error and all preceding destinations in `PrepareError`.
    #[cfg(feature = "workspace-test-support")]
    pub fn fail_reservation(mut self, buffer: Buffer) -> Self {
        self.fail = Some(buffer);
        self
    }

    /// Attempt each actual reserve once and retain all partial storage on
    /// failure. No source construction, clone, cache pool or source swap occurs.
    pub fn prepare(self) -> Result<Workspace<'source>, PrepareError<'source>> {
        let mut cache = empty_cache();
        for buffer in Buffer::ALL {
            let requested = self.requested_capacity(buffer);
            let result = match buffer {
                Buffer::Epsilon => cache.stack.try_reserve_exact(requested),
                Buffer::CurrentDense => {
                    cache.curr.set.try_reserve_workspace_dense(requested)
                }
                Buffer::CurrentSparse => {
                    cache.curr.set.try_reserve_workspace_sparse(requested)
                }
                Buffer::NextDense => {
                    cache.next.set.try_reserve_workspace_dense(requested)
                }
                Buffer::NextSparse => {
                    cache.next.set.try_reserve_workspace_sparse(requested)
                }
                Buffer::CurrentSlots => {
                    cache.curr.slot_table.table.try_reserve_exact(requested)
                }
                Buffer::NextSlots => {
                    cache.next.slot_table.table.try_reserve_exact(requested)
                }
            };
            if let Err(cause) = result {
                return Err(PrepareError {
                    source: self.source,
                    requirements: self.requirements,
                    buffer,
                    cause,
                    partial: cache,
                });
            }
        }
        // Every destination is now reserved. The existing sparse-set resize
        // and these table fills stay within those capacities. Do not call
        // legacy SlotTable::reset: it has an external logging callback.
        initialize(&mut cache.curr, self.requirements);
        initialize(&mut cache.next, self.requirements);
        Ok(Workspace {
            source: self.source,
            requirements: self.requirements,
            cache,
        })
    }

    fn requested_capacity(&self, buffer: Buffer) -> usize {
        #[cfg(any(test, feature = "workspace-test-support"))]
        if self.fail == Some(buffer) {
            // Capacity overflow on this actual target. This is not an OOM
            // simulator and does not allocate a substitute failing buffer.
            return usize::MAX;
        }
        self.requirements.capacity(buffer)
    }
}

impl fmt::Debug for Plan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plan")
            .field("requirements", &self.requirements)
            .finish()
    }
}

fn empty_cache() -> Cache {
    fn empty_active() -> ActiveStates {
        ActiveStates {
            set: SparseSet::new(0),
            slot_table: SlotTable::new(),
        }
    }
    Cache {
        stack: Vec::new(),
        curr: empty_active(),
        next: empty_active(),
    }
}

fn initialize(active: &mut ActiveStates, requirements: Requirements) {
    active.set.resize(requirements.states);
    active.slot_table.slots_per_state = requirements.slots;
    active.slot_table.slots_for_captures = requirements.slots;
    active.slot_table.table.resize(requirements.table, None);
}

fn capacities(cache: &Cache) -> [usize; 7] {
    let current = cache.curr.set.workspace_capacities();
    let next = cache.next.set.workspace_capacities();
    [
        cache.stack.capacity(),
        current.0,
        current.1,
        next.0,
        next.1,
        cache.curr.slot_table.table.capacity(),
        cache.next.slot_table.table.capacity(),
    ]
}

/// A closed workspace retaining the immutable borrow of its exact source.
///
/// The workspace cannot outlive that source:
///
/// ```compile_fail
/// use regex_automata::nfa::thompson::pikevm::{PikeVM, workspace::Workspace};
/// fn escape(source: &PikeVM) -> Workspace<'static> {
///     source.workspace_plan().unwrap().prepare().unwrap()
/// }
/// ```
///
/// It cannot be used as an ordinary cache with a foreign source:
///
/// ```compile_fail
/// use regex_automata::{Input, nfa::thompson::pikevm::PikeVM};
/// fn substitute(source: &PikeVM, other: &PikeVM) {
///     let mut workspace = source.workspace_plan().unwrap().prepare().unwrap();
///     other.search_slots(&mut workspace, &Input::new("text"), &mut []);
/// }
/// ```
///
/// It offers no allocating Clone:
///
/// ```compile_fail
/// use regex_automata::nfa::thompson::pikevm::workspace::Workspace;
/// fn duplicate<'s>(workspace: &Workspace<'s>) -> Workspace<'s> {
///     workspace.clone()
/// }
/// ```
pub struct Workspace<'source> {
    source: &'source PikeVM,
    requirements: Requirements,
    cache: Cache,
}

impl Workspace<'_> {
    /// The requirements validated before this workspace was allocated.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }

    /// Actual allocation capacities in [`Buffer`] declaration order.
    pub fn capacities(&self) -> [usize; 7] {
        capacities(&self.cache)
    }

    /// Search with the unchanged PikeVM worker and this workspace's source.
    ///
    /// The caller's slot slice must fit between the source's implicit slots
    /// and full row width. Invalid widths leave the cache and slots untouched.
    /// Valid widths exclude the worker's allocating empty-UTF8 fallback.
    /// If sharing one outer slot buffer among sources, pass each source only
    /// its appropriate prefix. Matching semantics are the ordinary worker's.
    pub fn search_slots(
        &mut self,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, SearchError> {
        if slots.len() < self.requirements.minimum_slots
            || slots.len() > self.requirements.slots
        {
            return Err(SearchError::SlotCount {
                minimum: self.requirements.minimum_slots,
                maximum: self.requirements.slots,
                supplied: slots.len(),
            });
        }
        #[cfg(debug_assertions)]
        let before = self.capacities();
        let result = self.source.search_slots(&mut self.cache, input, slots);
        #[cfg(debug_assertions)]
        debug_assert_eq!(before, self.capacities());
        Ok(result)
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

/// A reserve failure retaining the source borrow and every partial buffer.
///
/// The failure cannot outlive the source either:
///
/// ```compile_fail
/// use regex_automata::nfa::thompson::pikevm::{PikeVM, workspace::PrepareError};
/// fn escape(source: &PikeVM) -> PrepareError<'static> {
///     source.workspace_plan().unwrap().prepare().unwrap_err()
/// }
/// ```
pub struct PrepareError<'source> {
    source: &'source PikeVM,
    requirements: Requirements,
    buffer: Buffer,
    cause: TryReserveError,
    partial: Cache,
}

impl PrepareError<'_> {
    /// The actual destination whose reserve failed.
    pub fn buffer(&self) -> Buffer {
        self.buffer
    }

    /// The unchanged reserve error returned by that destination.
    pub fn cause(&self) -> &TryReserveError {
        &self.cause
    }

    /// Requirements computed before the first reserve.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }

    /// Actual partial capacities in [`Buffer`] declaration order.
    pub fn capacities(&self) -> [usize; 7] {
        capacities(&self.partial)
    }

    /// Number of states in the exact source still borrowed by this failure.
    pub fn source_state_count(&self) -> usize {
        self.source.get_nfa().states().len()
    }
}

impl fmt::Debug for PrepareError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrepareError")
            .field("buffer", &self.buffer)
            .field("cause", &self.cause)
            .field("capacities", &self.capacities())
            .finish()
    }
}

impl fmt::Display for PrepareError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PikeVM workspace {:?} reserve failed: {}",
            self.buffer, self.cause
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PrepareError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// A fixed rejection before a prepared search mutates any destination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchError {
    /// The supplied slice is outside the actual source's valid slot widths.
    SlotCount {
        /// Minimum source-bound width, including implicit capture slots.
        minimum: usize,
        /// Maximum source-bound width, equal to the full row width.
        maximum: usize,
        /// Actual caller-provided slice length.
        supplied: usize,
    },
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            SearchError::SlotCount { minimum, maximum, supplied } => write!(
                f,
                "PikeVM workspace needs {minimum}..={maximum} slots, got {supplied}",
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SearchError {}

#[cfg(all(test, feature = "syntax"))]
mod tests;
