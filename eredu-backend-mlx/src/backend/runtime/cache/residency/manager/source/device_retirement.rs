//! Logical reservation custody on an exact completed backing or Device view.
//! The attached accounting payload retains no array, manager or native graph;
//! independent Host staging and Device returns use independent prepared owners.
use super::*;
use safemlx::{
    HostTransferArrayViewWitness, ImmutableHostTransferWitness, OrdinaryBufferInspection,
    OrdinaryBufferWitness, OriginalBufferAliasWitness, OriginalBufferCause, OriginalBufferError,
    PreparedAllocationOwner, PreparedAllocationOwnerCause, PreparedAllocationOwnerError,
    PreparedAllocationRetirement,
};
use std::sync::OnceLock;

struct Custody {
    reservation: OnceLock<CachePoolReservation>,
    _funding: HostMetadataFunding,
}
type Owner = Arc<Custody>;
type Attachment = PreparedAllocationOwner<Owner>;

#[derive(Clone, Copy, Eq, PartialEq)]
enum AttachmentKind {
    Device,
    Host,
}

pub(super) struct DeviceRetirement {
    kind: AttachmentKind,
    attachments: [Option<Attachment>; 2],
    custody: Option<Owner>,
    retirements: [Option<PreparedAllocationRetirement>; 2],
}
impl DeviceRetirement {
    /// Reserve every concrete node and transport before the first allocation.
    pub(super) fn prepare(context: &WorkspaceContext) -> Result<Self, CacheSourceFailure> {
        Self::prepare_kind(context, AttachmentKind::Device)
    }

    /// Prepares independent custody for the actual Host staging allocations.
    /// Their final backing owner may be a completed Host-backed Device view.
    pub(super) fn prepare_host(context: &WorkspaceContext) -> Result<Self, CacheSourceFailure> {
        Self::prepare_kind(context, AttachmentKind::Host)
    }

    fn prepare_kind(
        context: &WorkspaceContext,
        kind: AttachmentKind,
    ) -> Result<Self, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(Self::controls(kind).ok_or_else(|| fail(CacheSourceError::Overflow))?)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let funding = context
            .metadata_funding()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let custody = Arc::new(Custody {
            reservation: OnceLock::new(),
            _funding: funding,
        });
        let mut result = Self {
            kind,
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
        if self.kind != AttachmentKind::Device
            || self
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

    /// Attach the two prepared nodes to the exact immutable Host backings,
    /// before canonical replacement. A partial refusal preserves every unused
    /// node and leaves the reservation with its current owner.
    pub(super) fn attach_host(
        &mut self,
        buffers: [&ImmutableHostTransferBuffer; 2],
    ) -> Result<(), CacheSourceError> {
        if self.kind != AttachmentKind::Host
            || self
                .custody
                .as_ref()
                .is_none_or(|owner| owner.reservation.get().is_some())
        {
            return Err(CacheSourceError::Identity);
        }
        for (slot, buffer) in self.attachments.iter_mut().zip(buffers) {
            let witness = buffer
                .inspect_original_source()
                .map_err(CacheSourceError::DeviceRetirementAttachment)?;
            let owner = slot.take().ok_or(CacheSourceError::Identity)?;
            if let Err(error) = witness.try_attach(owner) {
                let (cause, owner) = error.into_parts();
                *slot = Some(owner);
                return Err(CacheSourceError::DeviceRetirementAttachment(cause));
            }
        }
        Ok(())
    }

    /// A successful canonical replacement moved the removed storage and
    /// completed transfer occupancy into this reservation. The exact attached
    /// backings or Device views retain that charge until final destruction.
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

    fn controls(kind: AttachmentKind) -> Option<usize> {
        let attachment_frames = match kind {
            AttachmentKind::Device => [
                OriginalBufferAliasWitness::inspection_control_bytes()?.checked_mul(2)?,
                OrdinaryBufferWitness::inspection_control_bytes()?.checked_mul(2)?,
                HostTransferArrayViewWitness::control_bytes()?.checked_mul(2)?,
                size_of::<(&mut Self, [&Array; 2])>(),
                size_of::<
                    std::iter::Zip<
                        std::slice::IterMut<'_, Option<Attachment>>,
                        std::array::IntoIter<&Array, 2>,
                    >,
                >(),
                size_of::<(Option<&Owner>, &mut Option<Attachment>, &Array, Attachment)>(),
            ],
            AttachmentKind::Host => [
                ImmutableHostTransferWitness::inspection_control_bytes()?.checked_mul(2)?,
                size_of::<Result<ImmutableHostTransferWitness<'_>, OriginalBufferCause>>(),
                size_of::<ImmutableHostTransferWitness<'_>>(),
                size_of::<(&mut Self, [&ImmutableHostTransferBuffer; 2])>(),
                size_of::<
                    std::iter::Zip<
                        std::slice::IterMut<'_, Option<Attachment>>,
                        std::array::IntoIter<&ImmutableHostTransferBuffer, 2>,
                    >,
                >(),
                size_of::<(
                    Option<&Owner>,
                    &mut Option<Attachment>,
                    &ImmutableHostTransferBuffer,
                    Attachment,
                )>(),
            ],
        };
        let attachment_controls = attachment_frames.into_iter().try_fold(
            std::mem::size_of_val(&attachment_frames),
            usize::checked_add,
        )?;
        let layout = Attachment::layout();
        let frames = [
            WorkspaceContext::metadata_arc_bytes::<Custody>()?,
            layout.allocation_bytes()?.checked_mul(2)?,
            Attachment::retirement_control_bytes()?.checked_mul(2)?,
            layout.preparation_control_bytes().checked_mul(2)?,
            layout.preparation_failure_bytes(),
            layout.attachment_failure_bytes(),
            layout.original_attachment_control_bytes().checked_mul(2)?,
            attachment_controls,
            size_of::<AttachmentKind>(),
            size_of::<(&WorkspaceContext, AttachmentKind)>(),
            size_of::<Self>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<(&WorkspaceContext, Option<HostMetadataFunding>, Owner)>(),
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
