//! Independent Host payload custody, retired after its exact buffer.
use super::*;
use eredu_core::DomainMemoryRequirements;
use eredu_runtime::working_memory::{
    HostSourceConstructionFacts, NumericalSourceRequirements, OriginalNumericalLifetime,
};

pub(super) struct Staging<T> {
    values: Vec<T>,
    lifetime: OriginalNumericalLifetime,
}
impl<T> AsRef<[T]> for Staging<T> {
    fn as_ref(&self) -> &[T] {
        &self.values
    }
}
impl<T> AsMut<[T]> for Staging<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}
impl<T> Staging<T> {
    pub(super) fn new(
        source: &NativeSource,
        count: usize,
        mut value: impl FnMut(usize) -> T,
    ) -> Result<Self, eredu_nn::Error> {
        let context = &source.context;
        let overflow = || WorkspaceMetadataError::Overflow;
        let bytes = std::alloc::Layout::array::<T>(count)
            .map_err(|_| overflow())?
            .size();
        let frames = [
            usize::try_from(
                DomainMemoryRequirements::construction_backing_bytes(source.ledger.topology(), 0)
                    .map_err(|cause| context.metadata_source(cause))?,
            )
            .map_err(|_| overflow())?,
            usize::try_from(
                source
                    .ledger
                    .configured_limits()
                    .backing_bytes()
                    .map_err(|cause| context.metadata_source(cause))?,
            )
            .map_err(|_| overflow())?,
            size_of::<Self>(),
            size_of::<NumericalSourceRequirements>(),
            size_of::<DomainMemoryRequirements>(),
            size_of::<eredu_runtime::working_memory::OriginalNumericalSource>(),
            size_of::<eredu_runtime::working_memory::OriginalNumericalNative>(),
            size_of::<OriginalNumericalLifetime>(),
            size_of::<Result<Self, eredu_nn::Error>>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<(&NativeSource, usize, usize)>(),
            size_of_val(&value),
        ];
        context.charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or_else(overflow)?,
        )?;
        let mut domains =
            DomainMemoryRequirements::zero_with_allowance_capacity(source.ledger.topology(), 0);
        let placement = source.ledger.host_placement_handle();
        let bytes = u64::try_from(bytes).map_err(|_| overflow())?;
        domains
            .add_allocation(bytes, &placement)
            .map_err(|cause| context.metadata_source(cause))?;
        let requirements = NumericalSourceRequirements::new(
            domains,
            Some(0),
            Some(0),
            HostSourceConstructionFacts::new(0, 0, 0)
                .map_err(|cause| context.metadata_source(cause))?,
        )
        .map_err(|cause| context.metadata_source(cause))?;
        let mut account = source
            .ledger
            .reserve_numerical_source(
                &source.execution,
                requirements,
                source.ledger.configured_limits().clone(),
            )
            .map_err(|cause| context.metadata_source(cause))?;
        let lifetime = account
            .claim_native()
            .and_then(|claim| claim.bind_lifetime(bytes, placement))
            .map_err(|cause| context.metadata_source(cause))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|cause| context.metadata_source(cause))?;
        for index in 0..count {
            values.push(value(index));
        }
        Ok(Self { values, lifetime })
    }
}
