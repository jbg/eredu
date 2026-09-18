//! Fresh immutable DFA construction from private pinned compiler output.
//!
//! Planning validates exact default-syntax source bytes and the complete static
//! tables without allocating. Preparation copies each of the five actual tables
//! once, after its complete layout has been declared. No dynamic DFA compiler or
//! lazy search cache runs on this path. Static recipe image residency is separate
//! from the fresh source's heap storage.

use super::{MatchStates, StartTable, TransitionTable, DFA};
use crate::dfa::{accel::Accels, Automaton, StartKind};
use alloc::{boxed::Box, collections::TryReserveError, vec::Vec};
use core::{alloc::Layout, fmt, mem};

#[repr(C, align(4))]
struct Aligned<const N: usize>([u8; N]);
struct Recipe {
    pattern: &'static str,
    bytes: &'static [u8],
}
include!("source/recipes.rs");

/// Fixed rejection before any source storage is allocated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanError {
    /// The exact source/default profile is not in the compiler inventory.
    Profile,
    /// The emitted DFA fails complete validation or its required profile.
    Recipe,
    /// A concrete table layout or checked sum cannot be represented.
    Overflow,
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Profile => "DFA source profile is not implemented",
            Self::Recipe => "DFA source recipe is inconsistent",
            Self::Overflow => "DFA source layout overflow",
        })
    }
}
#[cfg(feature = "std")]
impl std::error::Error for PlanError {}

/// An actual immutable DFA table, in construction order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Buffer {
    /// Dense transition rows.
    Transitions,
    /// Context-sensitive start states.
    Starts,
    /// Match-state pattern-list ranges.
    Matches,
    /// Pattern identifiers for match states.
    Patterns,
    /// State acceleration records.
    Accelerators,
}
impl Buffer {
    const ALL: [Self; 5] = [
        Self::Transitions,
        Self::Starts,
        Self::Matches,
        Self::Patterns,
        Self::Accelerators,
    ];
}

/// Validated exact source and its prospective copy/control layouts.
#[derive(Debug)]
pub struct Plan<'s> {
    source: &'s str,
    borrowed: DFA<&'static [u32]>,
    heap: usize,
    required: usize,
    #[cfg(any(test, feature = "workspace-test-support"))]
    fail: Option<Buffer>,
}
impl<'s> Plan<'s> {
    /// Validate a private emitted table for this exact default-syntax source.
    pub fn new(source: &'s str) -> Result<Self, PlanError> {
        let recipe = RECIPES
            .iter()
            .find(|r| r.pattern == source)
            .ok_or(PlanError::Profile)?;
        let (borrowed, consumed) =
            DFA::from_bytes(recipe.bytes).map_err(|_| PlanError::Recipe)?;
        if consumed != recipe.bytes.len()
            || borrowed.start_kind() != StartKind::Anchored
            || borrowed.pattern_len() != 1
            || borrowed.has_empty()
        {
            return Err(PlanError::Recipe);
        }
        let mut heap = mem::size_of::<DFA<Vec<u32>>>();
        for table in tables(&borrowed) {
            heap = heap
                .checked_add(
                    Layout::array::<u32>(table.len())
                        .map_err(|_| PlanError::Overflow)?
                        .size(),
                )
                .ok_or(PlanError::Overflow)?;
        }
        let mut required = heap;
        for control in [
            mem::size_of::<Self>(),
            mem::size_of::<Result<Self, PlanError>>(),
            mem::size_of::<[Vec<u32>; 5]>(),
            mem::size_of::<DFA<Vec<u32>>>(),
            mem::size_of::<Failure>(),
            mem::size_of::<Result<Box<DFA<Vec<u32>>>, Failure>>(),
            mem::size_of::<Result<(), TryReserveError>>(),
            mem::size_of::<[&[u32]; 5]>(),
        ] {
            required =
                required.checked_add(control).ok_or(PlanError::Overflow)?;
        }
        Ok(Self {
            source,
            borrowed,
            heap,
            required,
            #[cfg(any(test, feature = "workspace-test-support"))]
            fail: None,
        })
    }
    /// Exact source retained through the construction attempt.
    pub fn source(&self) -> &'s str {
        self.source
    }
    /// All five requested heap layouts.
    pub fn heap_bytes(&self) -> usize {
        self.heap
    }
    /// Heap requests and named simultaneous control carriers.
    pub fn required_bytes(&self) -> usize {
        self.required
    }
    /// Number of nonempty actual table destinations.
    pub fn allocation_count(&self) -> usize {
        1 + tables(&self.borrowed)
            .iter()
            .filter(|t| !t.is_empty())
            .count()
    }
    /// Inject one real reserve overflow in dependency development tests.
    #[cfg(any(test, feature = "workspace-test-support"))]
    pub fn fail_reserve_for_testing(mut self, buffer: Buffer) -> Self {
        self.fail = Some(buffer);
        self
    }
    /// Copy each declared destination once. Failures retain all partial tables.
    pub fn prepare(self) -> Result<Box<DFA<Vec<u32>>>, Failure> {
        let mut storage: [Vec<u32>; 5] = core::array::from_fn(|_| Vec::new());
        for (index, source) in tables(&self.borrowed).into_iter().enumerate() {
            let requested = source.len();
            #[cfg(any(test, feature = "workspace-test-support"))]
            let requested = if self.fail == Some(Buffer::ALL[index]) {
                usize::MAX
            } else {
                requested
            };
            if let Err(cause) = storage[index].try_reserve_exact(requested) {
                return Err(Failure {
                    buffer: Buffer::ALL[index],
                    cause: Cause::Reserve(cause),
                    storage,
                });
            }
            if storage[index].capacity() != source.len() {
                return Err(Failure {
                    buffer: Buffer::ALL[index],
                    cause: Cause::Capacity,
                    storage,
                });
            }
            storage[index].extend_from_slice(source);
        }
        let [transitions, starts, matches, patterns, accelerators] = storage;
        let borrowed = self.borrowed;
        Ok(Box::new(DFA {
            tt: TransitionTable {
                table: transitions,
                classes: borrowed.tt.classes,
                stride2: borrowed.tt.stride2,
            },
            st: StartTable {
                table: starts,
                kind: borrowed.st.kind,
                start_map: borrowed.st.start_map,
                stride: borrowed.st.stride,
                pattern_len: borrowed.st.pattern_len,
                universal_start_unanchored: borrowed
                    .st
                    .universal_start_unanchored,
                universal_start_anchored: borrowed.st.universal_start_anchored,
            },
            ms: MatchStates {
                slices: matches,
                pattern_ids: patterns,
                pattern_len: borrowed.ms.pattern_len,
            },
            special: borrowed.special,
            accels: Accels::from_source_table(accelerators),
            pre: None,
            quitset: borrowed.quitset,
            flags: borrowed.flags,
        }))
    }
}
fn tables<'a>(dfa: &'a DFA<&'a [u32]>) -> [&'a [u32]; 5] {
    [
        dfa.tt.table,
        dfa.st.table,
        dfa.ms.slices,
        dfa.ms.pattern_ids,
        dfa.accels.source_table(),
    ]
}
#[derive(Debug)]
enum Cause {
    Reserve(TryReserveError),
    Capacity,
}
/// Failed destination and all partial table owners, retained until final drop.
#[derive(Debug)]
pub struct Failure {
    buffer: Buffer,
    cause: Cause,
    storage: [Vec<u32>; 5],
}
impl Failure {
    /// The actual failed destination.
    pub fn buffer(&self) -> Buffer {
        self.buffer
    }
    /// Retained partial capacities in buffer declaration order.
    pub fn capacities(&self) -> [usize; 5] {
        core::array::from_fn(|index| self.storage[index].capacity())
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Reserve(error) => error.fmt(f),
            Cause::Capacity => f.write_str("DFA source capacity mismatch"),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Reserve(error) => Some(error),
            Cause::Capacity => None,
        }
    }
}

#[cfg(test)]
mod tests;
