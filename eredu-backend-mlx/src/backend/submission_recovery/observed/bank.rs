//! Once-only storage for complete, caller-prepared operation slots.
use super::*;
use std::{alloc::Layout, collections::TryReserveError};

/// This bank owns storage, not permission or a proof of the caller's count.
pub(crate) struct PreparedOperationBank<S> {
    slots: Vec<Option<S>>,
    next: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BankExhausted {
    pub(crate) prepared: usize,
}

#[derive(Debug)]
pub(crate) enum BankPreparationCause<E> {
    Overflow,
    Reserve(TryReserveError),
    Slot { ordinal: usize, cause: E },
}

/// The factory retains all not-yet-issued source custody. A slot error owns
/// its actual partial construction; the successful prefix stays in this bank.
pub(crate) struct BankPreparationError<S, F, E> {
    pub(crate) cause: BankPreparationCause<E>,
    prepared: PreparedOperationBank<S>,
    prepare: F,
}
impl<S, F, E> BankPreparationError<S, F, E> {
    pub(crate) fn prepared_len(&self) -> usize {
        self.prepared.len()
    }
    pub(crate) fn into_parts(self) -> (BankPreparationCause<E>, PreparedOperationBank<S>, F) {
        (self.cause, self.prepared, self.prepare)
    }
}
impl<S, F, E: std::fmt::Debug> std::fmt::Debug for BankPreparationError<S, F, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BankPreparationError")
            .field("cause", &self.cause)
            .field("prepared", &self.prepared.len())
            .finish_non_exhaustive()
    }
}
impl<S> std::fmt::Debug for PreparedOperationBank<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedOperationBank")
            .field("prepared", &self.len())
            .field("remaining", &self.remaining())
            .finish_non_exhaustive()
    }
}

/// Requested layouts and named Rust controls, not an allocator/producer bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BankLayout {
    pub(crate) count: usize,
    pub(crate) slot_array_bytes: usize,
    pub(crate) prepared_slot_control_bytes: u64,
    pub(crate) total_control_bytes: u64,
}

impl<S> PreparedOperationBank<S> {
    /// The closed caller derives/authenticates count and exact per-slot costs
    /// before Q. F/E are the actual factory and owning partial-failure types.
    /// Dynamic children of F/E and allocator overhead remain caller obligations.
    pub(crate) fn layout<F, E>(count: usize, slot_control_bytes: u64) -> Option<BankLayout> {
        let slot_array_bytes = Layout::array::<Option<S>>(count).ok()?.size();
        let prepared_slot_control_bytes =
            u64::try_from(count).ok()?.checked_mul(slot_control_bytes)?;
        let controls = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Vec<Option<S>>>(),
            size_of::<Result<Self, BankPreparationError<S, F, E>>>(),
            size_of::<BankPreparationError<S, F, E>>(),
            size_of::<BankPreparationCause<E>>(),
            size_of::<F>(),
            size_of::<E>(),
            size_of::<Result<S, E>>(),
            size_of::<S>(),
            size_of::<Option<S>>(),
            size_of::<Option<S>>(),
            size_of::<BankExhausted>(),
            size_of::<Result<S, BankExhausted>>(),
            size_of::<TryReserveError>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Layout>(),
            size_of::<BankLayout>(),
            size_of::<usize>(),
            size_of::<usize>(),
            size_of::<&mut Self>(),
        ]
        .into_iter()
        .try_fold(slot_array_bytes, usize::checked_add)?;
        let total_control_bytes =
            prepared_slot_control_bytes.checked_add(u64::try_from(controls).ok()?)?;
        Some(BankLayout {
            count,
            slot_array_bytes,
            prepared_slot_control_bytes,
            total_control_bytes,
        })
    }

    /// Reserve the complete slot array before calling the supplied factory.
    /// Each ordinal is prepared at most once. No source/native operation may
    /// start here: the caller supplies already-authorized construction custody.
    pub(crate) fn try_new<F, E>(
        count: usize,
        mut prepare: F,
    ) -> Result<Self, BankPreparationError<S, F, E>>
    where
        F: FnMut(usize) -> Result<S, E>,
    {
        let mut prepared = Self {
            slots: Vec::new(),
            next: 0,
        };
        if Layout::array::<Option<S>>(count).is_err() {
            return Err(BankPreparationError {
                cause: BankPreparationCause::Overflow,
                prepared,
                prepare,
            });
        }
        let reserve_count = count;
        #[cfg(test)]
        let reserve_count = if tests::FAIL_RESERVE.with(|flag| flag.replace(false)) {
            usize::MAX
        } else {
            reserve_count
        };
        if let Err(cause) = prepared.slots.try_reserve_exact(reserve_count) {
            return Err(BankPreparationError {
                cause: BankPreparationCause::Reserve(cause),
                prepared,
                prepare,
            });
        }
        for ordinal in 0..count {
            let slot = match prepare(ordinal) {
                Ok(slot) => slot,
                Err(cause) => {
                    return Err(BankPreparationError {
                        cause: BankPreparationCause::Slot { ordinal, cause },
                        prepared,
                        prepare,
                    })
                }
            };
            // The sole append site cannot grow after the successful reserve.
            prepared.slots.push(Some(slot));
        }
        Ok(prepared)
    }

    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }
    pub(crate) fn remaining(&self) -> usize {
        self.slots.len() - self.next
    }

    /// Move out exactly the next prepared slot. No caller replacement, refill,
    /// reuse, callback, allocation or native submission is possible here.
    pub(crate) fn checkout(&mut self) -> Result<S, BankExhausted> {
        let prepared = self.slots.len();
        let Some(slot) = self.slots.get_mut(self.next) else {
            return Err(BankExhausted { prepared });
        };
        let value = slot.take().expect("unconsumed prepared operation slot");
        self.next += 1;
        Ok(value)
    }

    #[cfg(test)]
    fn slot_capacity(&self) -> usize {
        self.slots.capacity()
    }
}

#[cfg(test)]
mod tests;
