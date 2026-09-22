//! Exact completed constructor roots published into the canonical source directory.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::working_memory::{
    NumericalStoragePublicationPlan, NumericalStorageRegistration,
    PreparedNumericalStoragePublication,
};
use safemlx::{OriginalBufferInspection, OriginalBufferWitness, PreparedAllocationOwner};

type Registration = NumericalStorageRegistration<StorageIdentity>;
struct Owner {
    registrations: [Option<Registration>; 2],
    funding: HostMetadataFunding,
}

pub(super) fn publish(
    completed: physical::CompletedNumerical<Array>,
    source: &NativeSource,
) -> Result<Array, Error> {
    let context = &source.context;
    let overflow = || Error::Neural(WorkspaceMetadataError::Overflow.into());
    let layout = PreparedAllocationOwner::<Owner>::layout();
    let frames = [
        layout.allocation_bytes().ok_or_else(overflow)?,
        layout.preparation_control_bytes(),
        layout.prepared_bytes(),
        layout.preparation_failure_bytes(),
        layout.attachment_failure_bytes(),
        layout.original_attachment_control_bytes(),
        OriginalBufferInspection::inspection_control_bytes().ok_or_else(overflow)?,
        safemlx::OrdinaryBufferWitness::inspection_control_bytes().ok_or_else(overflow)?,
        size_of::<safemlx::OrdinaryBufferInspection<'_>>(),
        size_of::<Result<safemlx::OrdinaryBufferInspection<'_>, safemlx::OriginalBufferCause>>(),
        size_of::<physical::CompletedNumerical<Array>>(),
        size_of::<(&NativeSource, &WorkspaceContext)>(),
        size_of::<Owner>(),
        size_of::<std::slice::Iter<'_, Option<Registration>>>(),
        size_of::<std::iter::Flatten<std::slice::Iter<'_, Option<Registration>>>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<safemlx::AllocationInfo>(),
        size_of::<Option<OriginalBufferWitness<'_>>>(),
        size_of::<Result<Option<OriginalBufferWitness<'_>>, safemlx::OriginalBufferCause>>(),
        size_of::<NumericalStoragePublicationPlan<StorageIdentity>>(),
        size_of::<PreparedNumericalStoragePublication<StorageIdentity>>(),
        size_of::<Result<Array, Error>>(),
        size_of::<(
            StorageIdentity,
            u64,
            std::sync::Arc<eredu_core::MemoryPlacement>,
        )>(),
        size_of::<Result<std::sync::Arc<eredu_core::MemoryPlacement>, eredu_core::MemoryDomainError>>(
        ),
    ];
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or_else(overflow)?,
        )
        .map_err(|cause| Error::Neural(cause.into()))?;
    let witness = completed
        .source
        .budget()
        .inspect_array(&completed.value)
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    let Some(witness) = witness else {
        // A completed allocation-free value has no physical identity to publish.
        // Its Data/descriptor still retain their actual Graph allocator custody.
        // Zero-length CPU pages have real backing and never take this branch.
        return match completed
            .value
            .inspect_ordinary_buffer()
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
        {
            safemlx::OrdinaryBufferInspection::Empty => Ok(completed.value),
            _ => Err(Error::Neural(
                context.metadata_source(WorkingMemoryError::IdentityMismatch),
            )),
        };
    };
    let allocation = witness.allocation();
    let placement =
        crate::backend::managed_memory::allocation_placement_handle(&allocation, &source.ledger)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    let controls = u64::try_from(allocation.host_control_bytes()).map_err(|_| overflow())?;
    let count = 1 + usize::from(controls != 0);
    let mut publication = NumericalStoragePublicationPlan::<StorageIdentity>::new(count, 0)
        .and_then(|plan| plan.prepare(completed.source.account(), &source.ledger, &source.funding))
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    publication
        .push(
            StorageIdentity::Native(allocation.identity()),
            u64::try_from(allocation.bytes()).map_err(|_| overflow())?,
            placement,
        )
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    if controls != 0 {
        publication
            .push_host_controls(
                StorageIdentity::NativeControl(allocation.identity()),
                controls,
            )
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    }
    publication
        .publish()
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    let owner = Owner {
        registrations: [
            publication.take_input(0),
            if controls != 0 {
                publication.take_input(1)
            } else {
                None
            },
        ],
        funding: source.funding.clone(),
    };
    if owner.registrations[..count].iter().any(Option::is_none) {
        return Err(Error::Neural(
            context.metadata_source(WorkingMemoryError::IdentityMismatch),
        ));
    }
    let prepared = PreparedAllocationOwner::try_new(owner).map_err(|failure| {
        let (cause, owner) = failure.into_parts();
        // No native publication escaped this constructor; the exact completed
        // array still retains its numerical account while rows roll back.
        drop(owner);
        Error::Neural(context.metadata_source(cause))
    })?;
    for registration in prepared.owner().registrations.iter().flatten() {
        registration.arm_attachment().map_err(|cause| {
            for registration in prepared.owner().registrations.iter().flatten() {
                registration.disarm_attachment();
            }
            Error::Neural(context.metadata_source(cause))
        })?;
    }
    witness.try_attach(prepared).map_err(|failure| {
        let (cause, prepared) = failure.into_parts();
        for registration in prepared.owner().registrations.iter().flatten() {
            registration.disarm_attachment();
        }
        drop(prepared);
        Error::Neural(context.metadata_source(cause))
    })?;
    // The actual backing now retains canonical receipts and the native budget.
    // Neither receipt owns an Array, graph or physical payload.
    Ok(completed.value)
}
