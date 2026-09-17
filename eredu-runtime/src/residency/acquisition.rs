//! Independent input/snapshot loans for one shared acquisition algorithm.
use super::*;
use eredu_core::residency::{
    PreparedResidencyAdmissionFailure, ResidencyAdmissionStorage, ResidencyBatchFailure,
    ResidencyProtection, ResidencyReservationRow,
};

/// Initial hit/miss state borrowing independent IDs and a caller-owned buffer.
/// No controller loan survives construction, reservation, or native publication.
#[derive(Clone, Copy, Debug)]
pub struct ResidencyAcquisitionRef<'ids, 'flags> {
    ids: &'ids [OffloadUnitId],
    missing: &'flags [bool],
}
impl<'ids, 'flags> ResidencyAcquisitionRef<'ids, 'flags> {
    pub(super) fn from_owned(ids: &'ids [OffloadUnitId], missing: &'flags [bool]) -> Self {
        debug_assert_eq!(ids.len(), missing.len());
        Self { ids, missing }
    }
    /// Exact IDs in acquisition order.
    pub fn ids(self) -> &'ids [OffloadUnitId] {
        self.ids
    }
    /// Immutable initial flags, including copies subsequently published.
    pub fn missing(self) -> &'flags [bool] {
        self.missing
    }
    /// Whether every copy was already resident at snapshot time.
    pub fn is_hit(self) -> bool {
        self.missing.iter().all(|missing| !missing)
    }
    /// Borrows only initially missing IDs in acquisition order.
    pub fn missing_ids(self) -> impl Iterator<Item = &'ids OffloadUnitId> + 'flags
    where
        'ids: 'flags,
    {
        self.ids
            .iter()
            .zip(self.missing)
            .filter_map(|(id, missing)| missing.then_some(id))
    }
}
/// Allocation-free snapshot refusal; malformed IDs stay borrowed from the caller.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyAcquisitionFailure<'a> {
    /// Ordinary initialization ordering is preserved.
    #[error("residency manager has not been initialized")]
    NotInitialized,
    /// Exact ordered validation cause, or insufficient index destination.
    #[error("{0}")]
    Batch(ResidencyBatchFailure<'a>),
    /// No partial flags are written when the final destination is too small.
    #[error("prepared residency snapshot needs {required} flags, has {available}")]
    Destination {
        /// Required flags.
        required: usize,
        /// Available flags.
        available: usize,
    },
}
impl ResidencyAcquisitionFailure<'_> {
    pub(super) fn into_owned(self) -> ResidencyLedgerError {
        match self {
            Self::NotInitialized => ResidencyLedgerError::NotInitialized,
            Self::Batch(ResidencyBatchFailure::Ledger(cause)) => cause.into_owned(),
            Self::Batch(ResidencyBatchFailure::Destination { .. }) | Self::Destination { .. } => {
                unreachable!("ordinary exact destinations")
            }
        }
    }
    /// Moves the bounded source/error prefix into a neutral owning cause.
    pub fn retain(self, storage: ResidencyAdmissionStorage) -> PreparedResidencyAdmissionFailure {
        match self {
            Self::NotInitialized => storage.not_initialized_failure(),
            Self::Batch(cause) => storage.batch_failure(cause),
            Self::Destination {
                required,
                available,
            } => storage.destination_failure("snapshot", required, available),
        }
    }
}
impl ResidencyController {
    /// Fills a final initial-state destination after all validation succeeds.
    /// Input IDs and flags are independent of this mutable controller.
    pub fn plan_acquisition_in<'ids, 'flags>(
        &mut self,
        ids: &'ids [OffloadUnitId],
        tier: MemoryTier,
        initializing: bool,
        order: &mut [usize],
        missing: &'flags mut [bool],
    ) -> Result<ResidencyAcquisitionRef<'ids, 'flags>, ResidencyAcquisitionFailure<'ids>> {
        if !initializing && self.ledger.require_initialized().is_err() {
            return Err(ResidencyAcquisitionFailure::NotInitialized);
        }
        self.ledger
            .validate_batch_in(ids, tier, order)
            .map_err(ResidencyAcquisitionFailure::Batch)?;
        if missing.len() < ids.len() {
            return Err(ResidencyAcquisitionFailure::Destination {
                required: ids.len(),
                available: missing.len(),
            });
        }
        let output = &mut missing[..ids.len()];
        for (id, value) in ids.iter().zip(output.iter_mut()) {
            *value = !self
                .ledger
                .is_resident(id, tier)
                .expect("same exclusive validated ledger");
        }
        Ok(ResidencyAcquisitionRef {
            ids,
            missing: output,
        })
    }
    /// Reserves a sorted closure while protecting its initial hits and misses.
    pub fn reserve_acquisition_in(
        &mut self,
        acquisition: ResidencyAcquisitionRef<'_, '_>,
        reservations: &[ResidencyReservationRow],
        tier: MemoryTier,
        storage: ResidencyAdmissionStorage,
    ) -> Result<ResidencyAdmissionStorage, PreparedResidencyAdmissionFailure> {
        let Some(protected) = ResidencyProtection::sorted(acquisition.ids) else {
            return Err(storage.destination_failure(
                "strict ordered protection",
                acquisition.ids.len(),
                0,
            ));
        };
        self.ledger
            .reserve_copies_in(acquisition.ids, reservations, tier, protected, storage)
    }
    /// Updates recency only for the immutable snapshot's original hits.
    pub fn touch_acquisition_hits_ref(
        &mut self,
        acquisition: ResidencyAcquisitionRef<'_, '_>,
        tier: MemoryTier,
    ) -> Result<(), ResidencyLedgerError> {
        for (id, missing) in acquisition.ids.iter().zip(acquisition.missing) {
            if !missing {
                self.ledger.touch(id, tier)?;
            }
        }
        Ok(())
    }
    /// Removes only unpublished reservations from the immutable initial state.
    /// Published copies and committed victims retain their existing semantics.
    pub fn rollback_acquisition_ref(
        &mut self,
        acquisition: ResidencyAcquisitionRef<'_, '_>,
        tier: MemoryTier,
    ) -> Result<(), ResidencyLedgerError> {
        for (id, missing) in acquisition.ids.iter().zip(acquisition.missing) {
            if *missing {
                self.ledger.rollback_reserved(id, tier)?;
            }
        }
        Ok(())
    }
}
