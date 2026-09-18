//! Binds the executable's actual independently addressable source sidecars.
use super::*;
use crate::backend::nn::workspace::AddressableSources;
impl Executable {
    pub(in crate::composition::mlx) fn prepare_addressable_workspace_sources(
        &self,
        mechanism: ResidentExecutionMechanisms,
        pool: &WorkingMemoryPool,
        funding: &HostMetadataFunding,
    ) -> Result<Option<AddressableSources>, Error> {
        let Some(banks) = self
            .erased()
            .indexed_bank_sources()
            .filter(|banks| !banks.is_empty())
        else {
            return Ok(None);
        };
        let controls = std::mem::size_of::<(
            AddressableSources,
            Option<AddressableSources>,
            Result<Option<AddressableSources>, Error>,
        )>()
        .checked_add(
            safemlx::InitializedInputAllocator::borrow_control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        )
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        funding
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        // Borrow the already admitted allocator for this exact pool. No device,
        // stream, ordinary initializer or materialized tensor is created here.
        let allocator = crate::backend::managed_memory::input_allocator::admitted_initializer(pool)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let runtime = allocator
            .try_borrow_runtime()
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        AddressableSources::new(
            banks.iter().map(|(bank, source)| (bank.value(), source)),
            mechanism,
            &runtime,
            Some(pool),
            funding,
        )
        .map(Some)
        .map_err(|cause| retain_planning_error(cause, funding.clone()))
    }
}
