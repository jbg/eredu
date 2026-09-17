//! Paid reservations and exact transfer into the same canonical pool ledger.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError, WorkspaceMetadataFunding},
};
use std::{
    mem::size_of,
    sync::{MutexGuard, TryLockError},
};

/// One-use empty slot storage for an actual finite cache occupancy reservation.
#[derive(Debug)]
pub struct PreparedCachePoolReservation {
    storage: Option<PreparedCacheTable<u64, CachePoolUsage>>,
    pool: CacheResidencyPool,
    funding: Option<WorkspaceMetadataFunding>,
}
/// A failed preparation retains the account that paid for its construction.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct CachePoolReservationPreparationFailure {
    #[source]
    cause: PreparationCause,
    _funding: Option<WorkspaceMetadataFunding>,
}
#[derive(Debug, thiserror::Error)]
enum PreparationCause {
    #[error(transparent)]
    Pool(#[from] CachePoolError),
    #[error(transparent)]
    Metadata(#[from] Error),
}
/// A refused admission retains its actual unused paid destination.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct CachePoolReservationAdmissionFailure {
    #[source]
    cause: CachePoolError,
    _prepared: PreparedCachePoolReservation,
}
impl CachePoolReservationAdmissionFailure {
    /// Original typed refusal; the unused destination stays inside this error.
    pub fn cause(&self) -> &CachePoolError {
        &self.cause
    }
}
impl CacheResidencyPool {
    /// Exact current slot population used by the existing reservation producer.
    /// This read-only result supplies no occupancy or installation authority.
    pub fn reservation_preparation_control_bytes(&self) -> Result<usize, CachePoolError> {
        let maximum = self
            .state
            .try_lock()
            .map_err(lock_error)?
            .reservations
            .len()
            .checked_add(1)
            .ok_or(CacheTableCapacityError::Exhausted)?;
        PreparedCachePoolReservation::fixed_controls()
            .and_then(|n| {
                n.checked_add(PreparedCacheTable::<u64, CachePoolUsage>::control_bytes(
                    maximum,
                )?)
            })
            .ok_or_else(|| CacheTableCapacityError::Exhausted.into())
    }
    /// Prepares one canonical reservation entry before a copy/transfer begins.
    /// The later reserve call still compares actual current occupancy and limits.
    pub fn prepare_reservation(
        &self,
        context: &WorkspaceContext,
    ) -> Result<PreparedCachePoolReservation, CachePoolReservationPreparationFailure> {
        self.prepare_reservation_population(1, context)
    }
    /// Prepares one reservation destination for an exact finite producer that
    /// can add `additional_slots` before this destination is used. This pays
    /// actual table storage only; reserve still compares current occupancy and
    /// revalidates its capacity before any ledger mutation.
    pub fn prepare_reservation_population(
        &self,
        additional_slots: usize,
        context: &WorkspaceContext,
    ) -> Result<PreparedCachePoolReservation, CachePoolReservationPreparationFailure> {
        let result = (|| -> Result<_, PreparationCause> {
            if additional_slots == 0 {
                return Err(CachePoolError::from(CacheTableCapacityError::Exhausted).into());
            }
            context
                .charge_metadata(
                    PreparedCachePoolReservation::fixed_controls()
                        .ok_or_else(|| Error::from(WorkspaceMetadataError::Overflow))?,
                )
                .map_err(Error::from)?;
            let maximum = self
                .state
                .try_lock()
                .map_err(lock_error)?
                .reservations
                .len()
                .checked_add(additional_slots)
                .ok_or(CacheTableCapacityError::Exhausted)
                .map_err(CachePoolError::from)?;
            Ok(PreparedCachePoolReservation {
                storage: Some(PreparedCacheTable::prepare(maximum, context)?),
                pool: self.clone(),
                funding: context.metadata_funding(),
            })
        })();
        result.map_err(|cause| CachePoolReservationPreparationFailure {
            cause,
            _funding: context.metadata_funding(),
        })
    }
}
impl PreparedCachePoolReservation {
    /// Commits an actual finite occupancy reservation into this same pool.
    /// Concurrent growth is revalidated before changing the canonical ledger.
    pub fn reserve(
        mut self,
        usage: CachePoolUsage,
    ) -> Result<CachePoolReservation, CachePoolReservationAdmissionFailure> {
        match self.install(usage) {
            Ok((reservation, retired)) => {
                drop(retired);
                Ok(reservation)
            }
            Err(cause) => Err(CachePoolReservationAdmissionFailure {
                cause,
                _prepared: self,
            }),
        }
    }
    fn install(
        &mut self,
        usage: CachePoolUsage,
    ) -> Result<(CachePoolReservation, CacheRecordTable<u64, CachePoolUsage>), CachePoolError> {
        let mut state = self.pool.state.try_lock().map_err(lock_error)?;
        let current = reservation_current(state.current, usage, self.pool.limits)?;
        let maximum = self
            .storage
            .as_ref()
            .expect("one reservation destination")
            .maximum();
        if state
            .reservations
            .len()
            .checked_add(1)
            .is_none_or(|n| n > maximum)
        {
            return Err(CacheTableCapacityError::Exhausted.into());
        }
        let retired = state
            .reservations
            .install(self.storage.take().expect("one reservation destination"))
            .expect("actual reservation population fits");
        let reservation = NEXT_CACHE_POOL_RESERVATION_ID.fetch_add(1, Ordering::Relaxed);
        let previous = state
            .reservations
            .insert_prepared(reservation, usage)
            .expect("one exact free reservation slot");
        debug_assert!(previous.is_none());
        state.current = current;
        update_peaks(&mut state, self.pool.limits);
        Ok((
            CachePoolReservation {
                reservation,
                pool: self.pool.clone(),
                _funding: self.funding.clone(),
            },
            retired,
        ))
    }
    fn fixed_controls() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            pool_table_retirement_control_bytes()?,
            size_of::<CachePoolReservationPreparationFailure>(),
            size_of::<CachePoolReservationAdmissionFailure>(),
            size_of::<PreparationCause>(),
            size_of::<CachePoolReservation>(),
            size_of::<CachePoolUsage>(),
            size_of::<CachePoolError>(),
            size_of::<(&CacheResidencyPool, usize, &WorkspaceContext)>(),
            size_of::<(&mut Self, CachePoolUsage)>(),
            size_of::<MutexGuard<'_, CachePoolState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CachePoolState>,
                    TryLockError<MutexGuard<'_, CachePoolState>>,
                >,
            >(),
            size_of::<Result<Self, PreparationCause>>(),
            size_of::<Result<Self, CachePoolReservationPreparationFailure>>(),
            size_of::<Result<CachePoolReservation, CachePoolReservationAdmissionFailure>>(),
            size_of::<
                Result<
                    (CachePoolReservation, CacheRecordTable<u64, CachePoolUsage>),
                    CachePoolError,
                >,
            >(),
            size_of::<Result<Option<CachePoolUsage>, (CacheTableCapacityError, u64, CachePoolUsage)>>(
            ),
            size_of::<Option<usize>>(),
            size_of::<u64>(),
            CacheRecordTable::<u64, CachePoolUsage>::mutation_control_bytes()?,
            CachePoolReservation::publication_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
impl CachePoolReservation {
    /// Borrows the exact unconsumed reservation, without publishing or releasing
    /// any occupancy. Busy/foreign state never substitutes a caller byte count.
    pub fn remaining_usage(&self) -> Result<CachePoolUsage, CachePoolError> {
        let state = self.pool.state.try_lock().map_err(lock_error)?;
        state.reservations.get(&self.reservation).copied().ok_or(
            CachePoolError::UnknownReservation {
                reservation: self.reservation,
            },
        )
    }
    /// Fixed controls for the same allocation-free reservation observation.
    pub fn remaining_usage_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<&Self>(),
            size_of::<CachePoolUsage>(),
            size_of::<Result<CachePoolUsage, CachePoolError>>(),
            size_of::<MutexGuard<'_, CachePoolState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CachePoolState>,
                    TryLockError<MutexGuard<'_, CachePoolState>>,
                >,
            >(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Publishes actual monotonic destination occupancy by consuming that exact
    /// portion of this reservation. Aggregate usage is unchanged, so publication
    /// cannot double-charge or refund the same admitted bytes. Both identities
    /// are real RAII owners; equal byte counts or limits cannot substitute.
    pub fn publish_to_manager(
        &mut self,
        member: &CachePoolMembership,
        usage: CachePoolUsage,
    ) -> Result<(), CachePoolError> {
        self.publish_manager_storage(member, usage, false)
    }
    /// Publishes a replacement while retaining the removed manager storage in
    /// this same reservation. The backend must keep those removed resources
    /// alive beside this token until their actual retirement. Aggregate usage
    /// and peaks remain unchanged; neither equal bytes nor a foreign membership
    /// can substitute for these exact pool owners.
    pub fn publish_retaining_replaced_storage(
        &mut self,
        member: &CachePoolMembership,
        usage: CachePoolUsage,
    ) -> Result<(), CachePoolError> {
        self.publish_manager_storage(member, usage, true)
    }
    fn publish_manager_storage(
        &mut self,
        member: &CachePoolMembership,
        usage: CachePoolUsage,
        retain_replaced: bool,
    ) -> Result<(), CachePoolError> {
        if self.pool != member.pool {
            return Err(CachePoolError::ForeignMembership);
        }
        let mut state = self.pool.state.try_lock().map_err(lock_error)?;
        let previous =
            *state
                .managers
                .get(&member.manager)
                .ok_or(CachePoolError::UnknownManager {
                    manager: member.manager,
                })?;
        let remaining = *state.reservations.get(&self.reservation).ok_or(
            CachePoolError::UnknownReservation {
                reservation: self.reservation,
            },
        )?;
        let remaining = if retain_replaced {
            remaining
                .checked_add(previous)
                .and_then(|combined| combined.checked_sub(usage))
                .ok_or(CachePoolError::AccountingOverflow {
                    operation: "replacement publication exceeds admitted occupancy",
                })?
        } else {
            let increase =
                usage
                    .checked_sub(previous)
                    .ok_or(CachePoolError::AccountingOverflow {
                        operation: "reservation publication must be monotonic",
                    })?;
            remaining
                .checked_sub(increase)
                .ok_or(CachePoolError::AccountingOverflow {
                    operation: "reservation publication exceeds admitted occupancy",
                })?
        };
        *state
            .managers
            .get_mut(&member.manager)
            .expect("validated manager") = usage;
        *state
            .reservations
            .get_mut(&self.reservation)
            .expect("validated reservation") = remaining;
        Ok(())
    }
    /// Fixed controls for one transfer from reservation to live membership.
    pub fn publication_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<(&mut Self, &CachePoolMembership, CachePoolUsage)>(),
            size_of::<(&mut Self, &CachePoolMembership, CachePoolUsage, bool)>(),
            size_of::<CachePoolError>(),
            size_of::<MutexGuard<'_, CachePoolState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CachePoolState>,
                    TryLockError<MutexGuard<'_, CachePoolState>>,
                >,
            >(),
            size_of::<CachePoolUsage>(),
            size_of::<CachePoolUsage>(),
            size_of::<CachePoolUsage>(),
            size_of::<CachePoolUsage>(),
            size_of::<Result<(), CachePoolError>>(),
            size_of::<Option<&mut CachePoolUsage>>(),
            size_of::<Option<&CachePoolUsage>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
fn lock_error(cause: TryLockError<MutexGuard<'_, CachePoolState>>) -> CachePoolError {
    match cause {
        TryLockError::WouldBlock => CachePoolError::Busy,
        TryLockError::Poisoned(_) => CachePoolError::Poisoned,
    }
}

#[cfg(test)]
#[path = "prepared_reservation/tests.rs"]
mod tests;
