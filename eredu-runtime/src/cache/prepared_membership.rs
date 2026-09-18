//! One paid destination for the existing canonical pool membership table.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError, HostMetadataFunding},
};
use std::{
    mem::size_of,
    sync::{MutexGuard, TryLockError},
};

/// One-use registration storage bound to the actual retained process pool.
/// It grants no cache source, block storage, occupancy or execution authority.
#[derive(Debug)]
pub struct PreparedCachePoolRegistration {
    storage: Option<PreparedCacheTable<u64, CachePoolUsage>>,
    pool: CacheResidencyPool,
    funding: Option<HostMetadataFunding>,
}
/// A refused registration retains its unchanged prepared destination and H.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct CachePoolRegistrationFailure {
    #[source]
    cause: CachePoolError,
    _prepared: PreparedCachePoolRegistration,
}
impl CachePoolRegistrationFailure {
    /// Original typed pool refusal, without formatted reconstruction.
    pub fn cause(&self) -> &CachePoolError {
        &self.cause
    }
}
/// A failed cold preparation keeps its actual paying account beside the cause.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct CachePoolRegistrationPreparationFailure {
    #[source]
    cause: PreparationCause,
    _funding: Option<HostMetadataFunding>,
}
#[derive(Debug, thiserror::Error)]
enum PreparationCause {
    #[error(transparent)]
    Pool(#[from] CachePoolError),
    #[error(transparent)]
    Metadata(#[from] Error),
}
impl CacheResidencyPool {
    /// Exact cold request for the same one-additional-member destination.
    /// Concurrent growth can refuse later preparation; this grants no membership.
    pub fn manager_registration_control_bytes(&self) -> Result<usize, CachePoolError> {
        let maximum = self
            .state
            .try_lock()
            .map_err(lock_error)?
            .managers
            .len()
            .checked_add(1)
            .ok_or(CacheTableCapacityError::Exhausted)?;
        PreparedCachePoolRegistration::fixed_controls()
            .and_then(|n| {
                n.checked_add(PreparedCacheTable::<u64, CachePoolUsage>::control_bytes(
                    maximum,
                )?)
            })
            .ok_or_else(|| CacheTableCapacityError::Exhausted.into())
    }

    /// Reserves space for exactly one additional membership. Concurrent pool
    /// growth can refuse commit; capacity is never inferred from allocator slack.
    pub fn prepare_manager_registration(
        &self,
        context: &WorkspaceContext,
    ) -> Result<PreparedCachePoolRegistration, CachePoolRegistrationPreparationFailure> {
        let prepared = (|| -> Result<PreparedCachePoolRegistration, PreparationCause> {
            context
                .charge_metadata(
                    PreparedCachePoolRegistration::fixed_controls()
                        .ok_or_else(|| Error::from(WorkspaceMetadataError::Overflow))?,
                )
                .map_err(Error::from)?;
            let maximum = {
                let state = self.state.try_lock().map_err(lock_error)?;
                state
                    .managers
                    .len()
                    .checked_add(1)
                    .ok_or(CachePoolError::Capacity(CacheTableCapacityError::Exhausted))?
            };
            Ok(PreparedCachePoolRegistration {
                storage: Some(PreparedCacheTable::prepare(maximum, context)?),
                pool: self.clone(),
                funding: context.metadata_funding(),
            })
        })();
        prepared.map_err(|cause| CachePoolRegistrationPreparationFailure {
            cause,
            _funding: context.metadata_funding(),
        })
    }
}
impl PreparedCachePoolRegistration {
    /// Commits one actual manager identity into this same canonical pool.
    /// Ordinary and prepared memberships share update/removal/accounting workers.
    pub fn register(
        mut self,
        manager: u64,
    ) -> Result<CachePoolMembership, CachePoolRegistrationFailure> {
        match self.install(manager) {
            Ok((membership, retired)) => {
                // Retired backing/funding can run destruction only after unlock.
                drop(retired);
                Ok(membership)
            }
            Err(cause) => Err(CachePoolRegistrationFailure {
                cause,
                _prepared: self,
            }),
        }
    }
    fn install(
        &mut self,
        manager: u64,
    ) -> Result<(CachePoolMembership, CacheRecordTable<u64, CachePoolUsage>), CachePoolError> {
        let mut state = self.pool.state.try_lock().map_err(lock_error)?;
        validate_new_manager(&state, manager)?;
        let maximum = self
            .storage
            .as_ref()
            .expect("one registration destination")
            .maximum();
        if state
            .managers
            .len()
            .checked_add(1)
            .is_none_or(|count| count > maximum)
        {
            return Err(CacheTableCapacityError::Exhausted.into());
        }
        let retired = state
            .managers
            .install(self.storage.take().expect("one registration destination"))
            .expect("actual population validated before installation");
        let previous = state
            .managers
            .insert_prepared(manager, CachePoolUsage::default())
            .expect("one exact free prepared membership");
        debug_assert!(previous.is_none());
        Ok((
            CachePoolMembership {
                manager,
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
            size_of::<CachePoolRegistrationFailure>(),
            size_of::<CachePoolRegistrationPreparationFailure>(),
            size_of::<PreparationCause>(),
            size_of::<Result<Self, PreparationCause>>(),
            size_of::<Result<Self, CachePoolRegistrationPreparationFailure>>(),
            size_of::<CachePoolMembership>(),
            size_of::<CachePoolUsage>(),
            size_of::<CachePoolError>(),
            size_of::<(&CacheResidencyPool, &WorkspaceContext)>(),
            size_of::<(&mut Self, u64)>(),
            size_of::<MutexGuard<'_, CachePoolState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CachePoolState>,
                    TryLockError<MutexGuard<'_, CachePoolState>>,
                >,
            >(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<CachePoolMembership, CachePoolRegistrationFailure>>(),
            size_of::<
                Result<
                    (CachePoolMembership, CacheRecordTable<u64, CachePoolUsage>),
                    CachePoolError,
                >,
            >(),
            size_of::<Result<Option<CachePoolUsage>, (CacheTableCapacityError, u64, CachePoolUsage)>>(
            ),
            size_of::<Option<usize>>(),
            size_of::<u64>(),
            CacheRecordTable::<u64, CachePoolUsage>::mutation_control_bytes()?,
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
