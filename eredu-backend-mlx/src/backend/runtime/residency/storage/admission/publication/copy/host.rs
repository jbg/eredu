//! Actual Copy roots, including aliases of closed completed Host destinations.
use super::*;
use safemlx::{AllocationInfo, Array, HostTransferArrayAliasWitness, OriginalBufferWitness};

pub(super) enum Observation<'a> {
    Native(OriginalBufferWitness<'a>),
    Host(HostTransferArrayAliasWitness<'a>),
}
impl Observation<'_> {
    pub(super) fn allocation(&self) -> AllocationInfo {
        match self {
            Self::Native(w) => w.allocation(),
            Self::Host(w) => w.allocation(),
        }
    }
    pub(super) fn try_attach(
        self,
        owner: PreparedAllocationOwner<Registration>,
    ) -> Result<(), safemlx::OriginalBufferError<PreparedAllocationOwner<Registration>>> {
        match self {
            Self::Native(w) => w.try_attach(owner),
            Self::Host(w) => w.try_attach(owner),
        }
    }
}
pub(super) fn observe<'a>(
    array: &'a Array,
    budget: &'a OriginalBufferBudget,
    sources: &[Arc<ImmutableHostTransferBuffer>],
) -> Result<Observation<'a>, Error> {
    if let Some(witness) = budget.inspect_array(array).map_err(buffer_error)? {
        return Ok(Observation::Native(witness));
    }
    let witness = array
        .inspect_host_transfer_alias()
        .map_err(buffer_error)?
        .ok_or_else(|| {
            if std::env::var_os("EREDU_HOST_SAVED_SOURCE_DIAGNOSTICS").is_some() {
                eprintln!("HOST_SAVED_PUBLICATION_ARRAY no-budget-or-host-witness source_count={} allocation={:?}",
                    sources.len(), array.try_allocation_info());
            }
            Error::PrefillControl(WorkingMemoryError::UnknownBound)
        })?;
    // These owners entered solely through a completed same-budget Copy token.
    // Match native identity and full capacity, never logical byte equality.
    for source in sources {
        let descriptor = source
            .try_fixed_descriptor::<4>()
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        if descriptor.allocation() == witness.allocation() {
            return Ok(Observation::Host(witness));
        }
    }
    Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        OriginalBufferBudget::inspection_control_bytes()?,
        HostTransferArrayAliasWitness::inspection_control_bytes()?,
        safemlx::HostTransferDescriptor::<4>::control_bytes()?,
        size_of::<Observation<'static>>(),
        size_of::<Result<Observation<'static>, Error>>(),
        size_of::<AllocationInfo>(),
        size_of::<(
            &Array,
            &OriginalBufferBudget,
            &[Arc<ImmutableHostTransferBuffer>],
        )>(),
        size_of::<std::slice::Iter<'_, Arc<ImmutableHostTransferBuffer>>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
