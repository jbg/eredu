//! The original return collection is allocated from the actual window before
//! activation. Its same deferred node owns partial pins and their final IDs.
use super::*;
use std::{alloc::Layout, collections::TryReserveError};

/// The same cold buffer first lends ordered request IDs, then moves each ID
/// into its final lease. Neither transition collects or allocates a new buffer.
enum LeaseIds {
    Prepared(Vec<OffloadUnitId>),
    Consuming(std::vec::IntoIter<OffloadUnitId>),
}

struct LeaseReturnPayload {
    leases: Vec<ResidentUnitLease>,
    ids: LeaseIds,
    manager: ManagerWeak,
    tier: MemoryTier,
    next: usize,
    // OrdinaryRetirement unboxes before destroying this payload. Custody is
    // last, after the final Vec/String and native storage owners are gone.
    _custody: ResidencyControlCustody,
}

pub(in crate::backend::runtime::residency::manager) struct PreparedLeaseCollection {
    payload: OrdinaryRetirement<LeaseReturnPayload>,
}
pub(in crate::backend::runtime::residency::manager) enum LeasePreparationCause {
    Source(NamedArrayError),
    Reserve(TryReserveError),
}
pub(in crate::backend::runtime::residency::manager) struct LeasePreparationError {
    pub(in crate::backend::runtime::residency::manager) cause: LeasePreparationCause,
    pub(in crate::backend::runtime::residency::manager) prefix: PreparedLeaseCollection,
}

impl PreparedLeaseCollection {
    #[cfg(test)]
    pub(in crate::backend::runtime::residency::manager) fn remaining_ids_for_test(
        &self,
    ) -> &[OffloadUnitId] {
        match &self.payload.ids {
            LeaseIds::Prepared(ids) => ids,
            LeaseIds::Consuming(ids) => ids.as_slice(),
        }
    }
    #[cfg(test)]
    pub(in crate::backend::runtime::residency::manager) fn id_addresses(&self) -> Vec<usize> {
        self.requested_ids()
            .expect("unissued prepared ID loan")
            .iter()
            .map(|id| id.as_str().as_ptr() as usize)
            .collect()
    }
    pub(in crate::backend::runtime::residency::manager) fn try_new(
        manager: &ResidencyManager,
        roots: &[OffloadUnitId],
        shape: TransferPayloadShape,
        custody: OriginalOperationMetadataCustody,
    ) -> Result<Self, LeasePreparationError> {
        Self::try_new_in_tier(manager, roots, shape, MemoryTier::Device, custody)
    }

    /// The caller selects this actual acquisition tier before preparing any
    /// lease IDs. The returned collection cannot be reused for another tier.
    pub(in crate::backend::runtime::residency::manager) fn try_new_in_tier(
        manager: &ResidencyManager,
        roots: &[OffloadUnitId],
        shape: TransferPayloadShape,
        tier: MemoryTier,
        custody: OriginalOperationMetadataCustody,
    ) -> Result<Self, LeasePreparationError> {
        Self::try_new_with_custody(manager, roots, shape, tier, custody.into())
    }

    pub(in crate::backend::runtime::residency::manager) fn try_new_with_custody(
        manager: &ResidencyManager,
        roots: &[OffloadUnitId],
        shape: TransferPayloadShape,
        tier: MemoryTier,
        custody: ResidencyControlCustody,
    ) -> Result<Self, LeasePreparationError> {
        let mut ready = Self {
            payload: OrdinaryRetirement::new(LeaseReturnPayload {
                leases: Vec::new(),
                ids: LeaseIds::Prepared(Vec::new()),
                manager: manager.inner.downgrade(),
                tier,
                next: 0,
                _custody: custody,
            }),
        };
        let prepared = (|| {
            let bytes = roots
                .iter()
                .try_fold(0usize, |sum, id| sum.checked_add(id.as_str().len()));
            if !matches!(tier, MemoryTier::Host | MemoryTier::Device)
                || roots.len() != shape.leases
                || bytes != Some(shape.lease_id_bytes)
            {
                return Err(LeasePreparationCause::Source(
                    NamedArrayError::InvalidSource,
                ));
            }
            ready
                .payload
                .leases
                .try_reserve_exact(shape.leases)
                .map_err(LeasePreparationCause::Reserve)?;
            let LeaseIds::Prepared(ids) = &mut ready.payload.ids else {
                unreachable!("cold ID destination is unissued")
            };
            ids.try_reserve_exact(shape.leases)
                .map_err(LeasePreparationCause::Reserve)?;
            // roots is the same retained request slice used for this actual
            // manager's validated named window; cloning occurs only here.
            for id in roots {
                ids.push(id.clone());
            }
            Ok(())
        })();
        match prepared {
            Ok(()) => Ok(ready),
            Err(cause) => Err(LeasePreparationError {
                cause,
                prefix: ready,
            }),
        }
    }

    pub(in crate::backend::runtime::residency::manager) fn control_bytes(
        shape: TransferPayloadShape,
    ) -> Option<u64> {
        let bytes = Layout::array::<ResidentUnitLease>(shape.leases)
            .ok()?
            .size()
            .checked_add(Layout::array::<OffloadUnitId>(shape.leases).ok()?.size())?
            .checked_add(shape.lease_id_bytes)?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<LeasePreparationError>())?
            .checked_add(size_of::<Result<Self, LeasePreparationError>>())?
            .checked_add(size_of::<ResidentLeaseCollection>())?
            .checked_add(size_of::<LeaseBuilder>())?;
        // The actual deferred payload accounts for the enum field. These are
        // additional phase-change, iterator and borrowed-ID return transports.
        let bytes = [
            size_of::<LeaseIds>(),
            size_of::<ResidencyControlCustody>(),
            size_of::<MemoryTier>(),
            size_of::<(
                &ResidencyManager,
                &[OffloadUnitId],
                TransferPayloadShape,
                MemoryTier,
                OriginalOperationMetadataCustody,
            )>(),
            size_of::<Vec<OffloadUnitId>>(),
            size_of::<std::vec::IntoIter<OffloadUnitId>>(),
            size_of::<&[OffloadUnitId]>(),
            size_of::<Result<&[OffloadUnitId], ResidencyError>>(),
            size_of::<Option<Vec<OffloadUnitId>>>(),
            size_of::<Option<OffloadUnitId>>(),
            size_of::<Result<OffloadUnitId, ResidencyError>>(),
        ]
        .into_iter()
        .try_fold(bytes, usize::checked_add)?;
        OrdinaryRetirement::<LeaseReturnPayload>::control_bytes()?
            .checked_add(u64::try_from(bytes).ok()?)
    }

    pub(in crate::backend::runtime::residency::manager) fn validate(
        &self,
        manager: &ManagerOwner,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<(), ResidencyError> {
        let ids = self.requested_ids()?;
        if tier != self.payload.tier
            || !self.payload.manager.ptr_eq(&manager.downgrade())
            || self.payload.next != 0
            || !self.payload.leases.is_empty()
            || ids.len() != requests.len()
            || self.payload.leases.capacity() < requests.len()
            || !ids
                .iter()
                .zip(requests)
                .all(|(expected, (actual, _))| expected == actual)
        {
            return Err(NamedArrayError::InvalidSource.into());
        }
        Ok(())
    }

    /// The manager borrows this exact window population only before lease
    /// extraction. Manager/request/observer validation remains with the caller.
    pub(in crate::backend::runtime::residency::manager) fn requested_ids(
        &self,
    ) -> Result<&[OffloadUnitId], ResidencyError> {
        match &self.payload.ids {
            LeaseIds::Prepared(ids) => Ok(ids),
            LeaseIds::Consuming(_) => Err(NamedArrayError::InvalidSource.into()),
        }
    }

    fn begin_consuming(&mut self) {
        // All callers already validated this unissued collection. Empty Vec
        // is an allocation-free move sentinel, not a replacement destination.
        let LeaseIds::Prepared(ids) =
            std::mem::replace(&mut self.payload.ids, LeaseIds::Prepared(Vec::new()))
        else {
            unreachable!("validated lease collection begins consumption once")
        };
        self.payload.ids = LeaseIds::Consuming(ids.into_iter());
    }

    fn take_id(&mut self, actual: &OffloadUnitId) -> Result<OffloadUnitId, ResidencyError> {
        let payload = &mut *self.payload;
        if payload.leases.len() == payload.leases.capacity() {
            return Err(NamedArrayError::InvalidSource.into());
        }
        let LeaseIds::Consuming(ids) = &mut payload.ids else {
            return Err(NamedArrayError::InvalidSource.into());
        };
        if ids.as_slice().first() != Some(actual) {
            return Err(NamedArrayError::InvalidSource.into());
        }
        let id = ids.next().expect("validated owned lease ID");
        payload.next += 1;
        Ok(id)
    }
}

pub(in crate::backend::runtime::residency::manager) enum ResidentLeaseCollection {
    Ordinary(Vec<ResidentUnitLease>),
    Original(PreparedLeaseCollection),
}
impl ResidentLeaseCollection {
    /// The caller has established sole application ownership, completed work
    /// and the unlocked teardown boundary. Consume the same final pin node.
    pub(super) fn retire_completed(self) {
        match self {
            Self::Ordinary(values) => drop(values),
            Self::Original(values) => drop(values.payload.into_inner()),
        }
    }

    pub(in crate::backend::runtime::residency::manager) fn as_slice(&self) -> &[ResidentUnitLease] {
        match self {
            Self::Ordinary(values) => values,
            Self::Original(values) => &values.payload.leases,
        }
    }
    pub(in crate::backend::runtime::residency::manager) fn into_ordinary(
        self,
    ) -> Vec<ResidentUnitLease> {
        match self {
            Self::Ordinary(values) => values,
            Self::Original(_) => unreachable!("ordinary acquisition owns ordinary collection"),
        }
    }
}

pub(in crate::backend::runtime::residency::manager) enum LeaseBuilder {
    Ordinary(OrdinaryRetirement<Vec<ResidentUnitLease>>),
    Original(PreparedLeaseCollection),
}
impl LeaseBuilder {
    pub(in crate::backend::runtime::residency::manager) fn new(
        original: Option<PreparedLeaseCollection>,
    ) -> Self {
        match original {
            Some(mut prepared) => {
                prepared.begin_consuming();
                Self::Original(prepared)
            }
            None => Self::Ordinary(OrdinaryRetirement::new(Vec::new())),
        }
    }
    /// Preserve ordinary pin/clone order. Original destination checks and its
    /// once-only ID move precede pinning; the following push cannot allocate.
    pub(in crate::backend::runtime::residency::manager) fn acquire(
        &mut self,
        ledger: &mut eredu_core::residency::ResidencyLedger,
        id: &OffloadUnitId,
        tier: MemoryTier,
        demand: u64,
        storage: ResidentLeaseStorage,
        owner: ManagerWeak,
    ) -> Result<(), ResidencyError> {
        match self {
            Self::Ordinary(values) => {
                ledger.pin(id, tier, demand)?;
                values.push(ResidentUnitLease::with_owner(
                    id.clone(),
                    tier,
                    storage,
                    owner,
                ));
            }
            Self::Original(prepared) => {
                let owned_id = prepared.take_id(id)?;
                ledger.pin(&owned_id, tier, demand)?;
                prepared.payload.leases.push(ResidentUnitLease::with_owner(
                    owned_id, tier, storage, owner,
                ));
            }
        }
        Ok(())
    }
    pub(in crate::backend::runtime::residency::manager) fn finish(self) -> ResidentLeaseCollection {
        match self {
            Self::Ordinary(values) => ResidentLeaseCollection::Ordinary(values.into_inner()),
            Self::Original(values) => ResidentLeaseCollection::Original(values),
        }
    }
}
