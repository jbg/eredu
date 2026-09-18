//! Physical custody of a completed Device replacement. No array, manager, or
//! native graph is retained by the accounting payload attached to the native
//! allocation or exact Host-backed Device view.
use super::*;
use safemlx::{
    HostTransferArrayViewWitness, OrdinaryBufferInspection, OrdinaryBufferWitness, OriginalBufferAliasWitness,
    OriginalBufferCause, OriginalBufferError, PreparedAllocationOwner,
    PreparedAllocationOwnerCause, PreparedAllocationOwnerError, PreparedAllocationRetirement,
};
use std::sync::OnceLock;

struct Custody {
    reservation: OnceLock<CachePoolReservation>,
    _funding: HostMetadataFunding,
}
type Owner = Arc<Custody>;
type Attachment = PreparedAllocationOwner<Owner>;

pub(super) struct DeviceRetirement {
    attachments: [Option<Attachment>; 2],
    custody: Option<Owner>,
    retirements: [Option<PreparedAllocationRetirement>; 2],
}
impl DeviceRetirement {
    /// Reserve every concrete node and transport before the first allocation.
    pub(super) fn prepare(context: &WorkspaceContext) -> Result<Self, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(Self::control_bytes().ok_or_else(|| fail(CacheSourceError::Overflow))?)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let funding = context
            .metadata_funding()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let custody = Arc::new(Custody {
            reservation: OnceLock::new(),
            _funding: funding,
        });
        let mut result = Self {
            attachments: [None, None],
            custody: Some(custody),
            retirements: [None, None],
        };
        for index in 0..2 {
            let (attachment, retirement) =
                Attachment::try_new_with_retirement(Arc::clone(result.custody.as_ref().unwrap()))
                    .map_err(|error| {
                    let (cause, owner) = error.into_parts();
                    drop(owner);
                    fail(CacheSourceError::DeviceRetirementPreparation(cause))
                })?;
            result.attachments[index] = Some(attachment);
            result.retirements[index] = Some(retirement);
        }
        Ok(result)
    }

    /// Attach to the actual completed allocation or Host-backed Device view
    /// before canonical replacement. Immutable Host source custody is separate.
    /// On any partial failure the enclosing mover keeps the reservation and
    /// attempted payload. Attached empty custody cannot refund that reservation.
    pub(super) fn attach(&mut self, arrays: [&Array; 2]) -> Result<(), CacheSourceError> {
        if self
            .custody
            .as_ref()
            .is_none_or(|owner| owner.reservation.get().is_some())
        {
            return Err(CacheSourceError::Identity);
        }
        for (slot, array) in self.attachments.iter_mut().zip(arrays) {
            let owner = slot.take().ok_or(CacheSourceError::Identity)?;
            let result = match array.inspect_original_buffer_alias() {
                Ok(Some(witness)) => witness.try_attach(owner),
                Ok(None) => match array.inspect_ordinary_buffer() {
                    Ok(OrdinaryBufferInspection::Allocation(witness)) => witness.try_attach(owner),
                    Ok(_) => match array.inspect_host_transfer_view() {
                        Ok(Some(witness)) => witness.try_attach(owner),
                        Ok(None) => {
                            *slot = Some(owner);
                            return Err(CacheSourceError::Identity);
                        }
                        Err(cause) => {
                            *slot = Some(owner);
                            return Err(CacheSourceError::DeviceRetirementAttachment(cause));
                        }
                    },
                    Err(cause) => {
                        *slot = Some(owner);
                        return Err(CacheSourceError::DeviceRetirementAttachment(cause));
                    }
                },
                Err(cause) => {
                    *slot = Some(owner);
                    return Err(CacheSourceError::DeviceRetirementAttachment(cause));
                }
            };
            if let Err(error) = result {
                let (cause, owner) = error.into_parts();
                *slot = Some(owner);
                return Err(CacheSourceError::DeviceRetirementAttachment(cause));
            }
        }
        Ok(())
    }

    /// A successful canonical Host commit moved only the replaced Device and
    /// completed transfer occupancy into this reservation. Both native backing
    /// or Device-view owners keep that exact charge until their final destruction.
    pub(super) fn publish(&mut self, reservation: &mut Option<CachePoolReservation>) {
        assert!(self.attachments.iter().all(Option::is_none));
        let owner = self
            .custody
            .take()
            .expect("one completed replacement publication");
        let reservation = reservation
            .take()
            .expect("original replacement reservation");
        assert!(
            owner.reservation.set(reservation).is_ok(),
            "single replacement publication"
        );
        // This final non-native alias retires before any source arrays are dropped.
        drop(owner);
    }

    /// Drop only payloads whose own actual backing destructor already fired.
    /// No unrelated retirement queue or native progress is entered.
    pub(super) fn reclaim(&mut self) {
        for retirement in self.retirements.iter_mut().flatten() {
            retirement.try_reclaim();
        }
    }

    fn control_bytes() -> Option<usize> {
        let layout = Attachment::layout();
        let frames = [
            WorkspaceContext::metadata_arc_bytes::<Custody>()?,
            layout.allocation_bytes()?.checked_mul(2)?,
            Attachment::retirement_control_bytes()?.checked_mul(2)?,
            layout.preparation_control_bytes().checked_mul(2)?,
            layout.preparation_failure_bytes(),
            layout.attachment_failure_bytes(),
            layout.original_attachment_control_bytes().checked_mul(2)?,
            OriginalBufferAliasWitness::inspection_control_bytes()?.checked_mul(2)?,
            OrdinaryBufferWitness::inspection_control_bytes()?.checked_mul(2)?,
            HostTransferArrayViewWitness::control_bytes()?.checked_mul(2)?,
            size_of::<Self>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<(&WorkspaceContext, Option<HostMetadataFunding>, Owner)>(),
            size_of::<(&mut Self, [&Array; 2])>(),
            size_of::<
                std::iter::Zip<
                    std::slice::IterMut<'_, Option<Attachment>>,
                    std::array::IntoIter<&Array, 2>,
                >,
            >(),
            size_of::<Result<(), OriginalBufferError<Attachment>>>(),
            size_of::<
                Result<
                    (Attachment, PreparedAllocationRetirement),
                    PreparedAllocationOwnerError<Owner>,
                >,
            >(),
            size_of::<(OriginalBufferCause, Attachment)>(),
            size_of::<(PreparedAllocationOwnerCause, Owner)>(),
            size_of::<(
                &mut Self,
                &mut Option<CachePoolReservation>,
                Owner,
                CachePoolReservation,
            )>(),
            size_of::<Result<(), CachePoolReservation>>(),
            size_of::<Result<(), CacheSourceError>>(),
            size_of::<(
                usize,
                std::ops::Range<usize>,
                Attachment,
                PreparedAllocationRetirement,
            )>(),
            size_of::<(&WorkspaceContext, &WorkspaceContext)>(),
            size_of::<(Option<&Owner>, &mut Option<Attachment>, &Array, Attachment)>(),
            size_of::<std::slice::Iter<'_, Option<Attachment>>>(),
            size_of::<
                std::iter::Flatten<std::slice::IterMut<'_, Option<PreparedAllocationRetirement>>>,
            >(),
            size_of::<(&mut Self, &mut PreparedAllocationRetirement, bool)>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}

#[cfg(test)]
#[path = "device_retirement/tests.rs"]
mod tests;
