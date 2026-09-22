//! Complete typed slots for explicitly supplied original-operation banks.
//! The closed selected caller supplies counts and original custody before work.
//! Final empty payload controls may be allocated here; no source lookup, native
//! observer, stream, numerical value, or lease is constructed.
use super::*;
use crate::backend::submission_recovery::observed::{
    bank::{BankLayout, PreparedOperationBank},
    operation::OperationRecovery,
    PreparedObservedRecovery,
};
use eredu_runtime::working_memory::OriginalOperationMetadataCustody;
use safemlx::{error::Exception, OriginalScopeObserver};
use std::alloc::Layout;

/// Scalar destination capacities derived by the selected retained-source plan.
/// Layout/preparation methods live with the concrete native owner, not policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TransferPayloadShape {
    pub(crate) leases: usize,
    pub(crate) lease_id_bytes: usize,
    pub(crate) unit_ids: usize,
    pub(crate) unit_id_bytes: usize,
    pub(crate) pending_sources: usize,
    pub(crate) retained_arrays: usize,
    pub(crate) retained_host: usize,
    pub(crate) retained_events: usize,
    pub(crate) output_arrays: usize,
    pub(crate) prepared_units: usize,
    pub(crate) binding_rows: usize,
    pub(crate) binding_name_bytes: usize,
    pub(crate) alias_assignments: usize,
    pub(crate) alias_owner_name_bytes: usize,
    pub(crate) declaration_clone_bytes: usize,
    pub(crate) declaration_clone_allocations: usize,
}
impl TransferPayloadShape {
    /// Final resource Vec requests owned by each transfer slot. Other shape
    /// fields describe separate declaration, publication and lease owners.
    pub(crate) fn resource_requested_bytes(self) -> Option<usize> {
        [
            Layout::array::<PendingWeightMaterialization>(self.pending_sources)
                .ok()?
                .size(),
            Layout::array::<Array>(self.retained_arrays).ok()?.size(),
            Layout::array::<ResidentHostOwner>(self.retained_host)
                .ok()?
                .size(),
            Layout::array::<Event>(self.retained_events).ok()?.size(),
            Layout::array::<OffloadUnitId>(self.unit_ids).ok()?.size(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}

// Supported Rust 1.98 alloc::rc::RcInner: repr(C, align(2)), two Cell<usize>
// counts followed by T. This implementation fact must be pinned by validation.
fn rc_bytes<T>() -> Option<u64> {
    let layout = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align()
        .extend(Layout::new::<T>())
        .ok()?
        .0
        .pad_to_align();
    u64::try_from(layout.size()).ok()
}

/// Final empty recovery and shared payload controls, prepared before activation.
pub(crate) struct PreparedResidentTransfer {
    // Payload first: unused-slot destruction retains custody until its actual
    // OrdinaryRetirement application is destroyed outside native locks.
    value: Rc<ResidentTransferResources>,
    leases: Option<PreparedLeaseCollection>,
    publication: super::publication::TransferPublication,
    ready:
        PreparedObservedRecovery<Rc<ResidentTransferResources>, OriginalOperationMetadataCustody>,
}
impl PreparedResidentTransfer {
    #[cfg(test)]
    pub(in crate::backend::runtime::residency::manager) fn prepared_lease_id_addresses(
        &self,
    ) -> Vec<usize> {
        self.leases
            .as_ref()
            .expect("prepared lease collection")
            .id_addresses()
    }
    /// Called only during the closed caller's prepaid bank construction.
    /// Uses the existing Box abort-on-OOM contract; custody remains in the node.
    pub(crate) fn new(custody: OriginalOperationMetadataCustody) -> Self {
        let application = Rc::new(OrdinaryRetirement::new(TransferApplication::prepared(
            custody.clone(),
        )));
        Self {
            value: Rc::new(ResidentTransferResources {
                sources: Vec::new(),
                retained_arrays: Vec::new(),
                retained_host: Vec::new(),
                retained_events: Vec::new(),
                event: None,
                application,
                named: None,
            }),
            leases: None,
            publication: super::publication::TransferPublication::original(&custody),
            ready: PreparedObservedRecovery::new(custody),
        }
    }

    /// Prepare the final resource vectors, preserving the slot/prefix/custody
    /// if any reserve fails. ID string storage belongs to the declaration loan.
    pub(crate) fn try_new(
        custody: OriginalOperationMetadataCustody,
        shape: TransferPayloadShape,
    ) -> Result<Self, (Self, std::collections::TryReserveError)> {
        let mut ready = Self::new(custody);
        let value = Rc::get_mut(&mut ready.value).expect("unissued prepared payload");
        let reserved = (|| {
            value.sources.try_reserve_exact(shape.pending_sources)?;
            value
                .retained_arrays
                .try_reserve_exact(shape.retained_arrays)?;
            value.retained_host.try_reserve_exact(shape.retained_host)?;
            value
                .retained_events
                .try_reserve_exact(shape.retained_events)?;
            Rc::get_mut(&mut value.application)
                .expect("unissued prepared application")
                .ids
                .try_reserve_exact(shape.unit_ids)
        })();
        match reserved {
            Ok(()) => Ok(ready),
            Err(error) => Err((ready, error)),
        }
    }

    pub(crate) fn bank_layout_with_payload<F, E>(
        count: usize,
        shape: TransferPayloadShape,
    ) -> Option<BankLayout> {
        let mut layout = Self::bank_layout::<F, E>(count)?;
        let bytes = u64::try_from(shape.resource_requested_bytes()?)
            .ok()?
            .checked_mul(u64::try_from(count).ok()?)?;
        layout.prepared_slot_control_bytes =
            layout.prepared_slot_control_bytes.checked_add(bytes)?;
        layout.total_control_bytes = layout.total_control_bytes.checked_add(bytes)?;
        Some(layout)
    }

    /// Cold final destinations for the actual retained window. The caller's
    /// request-wide catalog is shared; its bytes are charged once separately.
    pub(crate) fn try_new_for_window(
        custody: OriginalOperationMetadataCustody,
        shape: TransferPayloadShape,
        manager: &ResidencyManager,
        catalog: &super::super::named_arrays::NameCatalogOwner,
        roots: &[OffloadUnitId],
        scratch: &mut [eredu_runtime::residency::ResidencyClosureSlot],
    ) -> Result<Self, (Self, PreparedTransferDestinationCause)> {
        Self::try_new_for_window_in_tier(
            custody,
            shape,
            manager,
            catalog,
            roots,
            scratch,
            MemoryTier::Device,
        )
    }

    /// Same final destinations for an explicitly selected Host or Device
    /// acquisition. The exact tier remains with the prepared lease collection.
    pub(crate) fn try_new_for_window_in_tier(
        custody: OriginalOperationMetadataCustody,
        shape: TransferPayloadShape,
        manager: &ResidencyManager,
        catalog: &super::super::named_arrays::NameCatalogOwner,
        roots: &[OffloadUnitId],
        scratch: &mut [eredu_runtime::residency::ResidencyClosureSlot],
        tier: MemoryTier,
    ) -> Result<Self, (Self, PreparedTransferDestinationCause)> {
        let mut ready = Self::try_new(custody.clone(), shape)
            .map_err(|(ready, error)| (ready, PreparedTransferDestinationCause::Reserve(error)))?;
        let names = match catalog.prepare_window(manager, roots, scratch) {
            Ok(names) => names,
            Err(error) => return Err((ready, PreparedTransferDestinationCause::Names(error))),
        };
        Rc::get_mut(&mut ready.value)
            .expect("unissued prepared payload")
            .named = Some(names);
        if let Err(cause) = ready.publication.prepare(manager, roots, scratch, shape) {
            return Err((ready, PreparedTransferDestinationCause::Publication(cause)));
        }
        ready.leases = Some(
            match PreparedLeaseCollection::try_new_in_tier(manager, roots, shape, tier, custody) {
                Ok(leases) => leases,
                Err(error) => return Err((ready, PreparedTransferDestinationCause::Leases(error))),
            },
        );
        Ok(ready)
    }

    pub(crate) fn bank_layout_with_named_payload<F, E>(
        count: usize,
        shape: TransferPayloadShape,
        names: super::super::named_arrays::NamedStorageLayout,
    ) -> Option<BankLayout> {
        let mut layout = Self::bank_layout_with_payload::<F, E>(count, shape)?;
        let bytes = u64::try_from(names.destination_requested_bytes)
            .ok()?
            .checked_add(
                u64::try_from(size_of::<(
                    OriginalOperationMetadataCustody,
                    TransferPayloadShape,
                    &ResidencyManager,
                    &super::super::named_arrays::NameCatalogOwner,
                    &[OffloadUnitId],
                    &mut [eredu_runtime::residency::ResidencyClosureSlot],
                    MemoryTier,
                )>())
                .ok()?,
            )?
            .checked_add(PreparedLeaseCollection::control_bytes(shape)?)?
            .checked_add(super::publication::TransferPublication::control_bytes(
                shape,
            )?)?
            .checked_mul(u64::try_from(count).ok()?)?;
        layout.prepared_slot_control_bytes =
            layout.prepared_slot_control_bytes.checked_add(bytes)?;
        layout.total_control_bytes = layout.total_control_bytes.checked_add(bytes)?;
        Some(layout)
    }

    /// F/E must be the actual factory and its owning failure type. This prices
    /// named controls, not dynamic payloads or a proof of the supplied count.
    pub(crate) fn bank_layout<F, E>(count: usize) -> Option<BankLayout> {
        let node = PreparedObservedRecovery::<
            Rc<ResidentTransferResources>,
            OriginalOperationMetadataCustody,
        >::control_bytes::<Exception>()?;
        let dispatch = OperationRecovery::<
            Rc<ResidentTransferResources>,
            OriginalOperationMetadataCustody,
        >::control_bytes()?;
        let node = node.checked_add(
            u64::try_from(size_of::<
                eredu_runtime::working_memory::OriginalTextMetadataCustody,
            >())
            .ok()?,
        )?;
        let native = u64::try_from(OriginalScopeObserver::control_bytes()?).ok()?;
        let slot = node
            .checked_add(dispatch)?
            .checked_add(native)?
            .checked_add(rc_bytes::<ResidentTransferResources>()?)?
            .checked_add(rc_bytes::<OrdinaryRetirement<TransferApplication>>()?)?
            .checked_add(OrdinaryRetirement::<TransferApplication>::control_bytes()?)?;
        PreparedOperationBank::<Self>::layout::<F, E>(count, slot)
    }

    /// The caller authenticated the current innermost role before any producer.
    /// Moving the final prepared node never allocates or creates a child Scope.
    pub(super) fn activate(
        mut self,
        ids: &[OffloadUnitId],
        missing: &[bool],
        tier: MemoryTier,
        failed: FailureFlag,
        observer: OriginalScopeObserver,
    ) -> Result<
        (
            OperationRecovery<Rc<ResidentTransferResources>, OriginalOperationMetadataCustody>,
            super::publication::TransferPublication,
        ),
        ResidencyError,
    > {
        let value = Rc::get_mut(&mut self.value).expect("unissued prepared payload");
        let application =
            Rc::get_mut(&mut value.application).expect("unissued prepared application");
        self.publication
            .select_application(ids, missing, &mut application.ids)?;
        application.tier = tier;
        application.failed_transfer = Some(failed);
        Ok((
            OperationRecovery::original(self.ready, self.value, observer),
            self.publication,
        ))
    }

    pub(in crate::backend::runtime::residency::manager) fn take_lease_collection(
        &mut self,
        manager: &ManagerOwner,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<PreparedLeaseCollection, ResidencyError> {
        self.leases
            .as_ref()
            .ok_or(NamedArrayError::InvalidSource)?
            .validate(manager, requests, tier)?;
        Ok(self
            .leases
            .take()
            .expect("validated prepared lease destination"))
    }

    pub(super) fn into_immediate_application(
        mut self,
        tier: MemoryTier,
    ) -> Rc<OrdinaryRetirement<TransferApplication>> {
        let value = Rc::get_mut(&mut self.value).expect("unissued prepared payload");
        Rc::get_mut(&mut value.application)
            .expect("unissued prepared application")
            .tier = tier;
        // Rc::try_unwrap frees the resource allocation before its application
        // is returned. The existing application Rc is retained without a new
        // allocation while this unactivated resource payload and its prepared
        // vectors retire under their original custody.
        match Rc::try_unwrap(self.value) {
            Ok(value) => Rc::clone(&value.application),
            Err(_) => unreachable!("prepared payload has no aliases"),
        }
    }
}

pub(crate) enum PreparedTransferDestinationCause {
    Reserve(std::collections::TryReserveError),
    Names(super::super::named_arrays::NamedWindowPreparationError),
    Leases(super::lease_collection::LeasePreparationError),
    Publication(super::super::named_arrays::NamedPreparationSource),
}

/// One final node; its payload remains absent until the selected caller activates it.
pub(crate) struct PreparedTransferObservation {
    ready: PreparedObservedRecovery<TransferObservation, OriginalOperationMetadataCustody>,
}
impl PreparedTransferObservation {
    /// Called only during the closed caller's prepaid bank construction.
    /// Uses the existing Box abort-on-OOM contract; custody remains in the node.
    pub(crate) fn new(custody: OriginalOperationMetadataCustody) -> Self {
        Self {
            ready: PreparedObservedRecovery::new(custody),
        }
    }

    /// F/E must be the actual factory and its owning failure type. This prices
    /// named controls, not dynamic payloads or a proof of the supplied count.
    pub(crate) fn bank_layout<F, E>(count: usize) -> Option<BankLayout> {
        let node = PreparedObservedRecovery::<TransferObservation, OriginalOperationMetadataCustody>
            ::control_bytes::<Exception>()?;
        let node = node.checked_add(
            u64::try_from(size_of::<
                eredu_runtime::working_memory::OriginalTextMetadataCustody,
            >())
            .ok()?,
        )?;
        let dispatch =
            OperationRecovery::<TransferObservation, OriginalOperationMetadataCustody>::control_bytes()?;
        let native = u64::try_from(OriginalScopeObserver::control_bytes()?).ok()?;
        let completion = crate::backend::submission_recovery::observed::ObservedRecovery::<
            TransferObservation,
            OriginalOperationMetadataCustody,
        >::try_finish_control_bytes()?;
        let slot = node
            .checked_add(dispatch)?
            .checked_add(native)?
            .checked_add(completion)?;
        PreparedOperationBank::<Self>::layout::<F, E>(count, slot)
    }

    /// The caller authenticated the current innermost role before any producer.
    /// Moving the final prepared node never allocates or creates a child Scope.
    pub(super) fn activate(
        self,
        value: TransferObservation,
        observer: OriginalScopeObserver,
    ) -> OperationRecovery<TransferObservation, OriginalOperationMetadataCustody> {
        OperationRecovery::original(self.ready, value, observer)
    }
}

impl PreparedTransferDestinationCause {
    pub(crate) fn into_source(self) -> super::super::named_arrays::NamedPreparationSource {
        match self {
            Self::Reserve(cause) => cause.into(),
            Self::Names(cause) => cause.into_source(),
            Self::Publication(cause) => cause,
            Self::Leases(error) => match error.cause {
                super::lease_collection::LeasePreparationCause::Source(cause) => cause.into(),
                super::lease_collection::LeasePreparationCause::Reserve(cause) => cause.into(),
            },
        }
    }
}
