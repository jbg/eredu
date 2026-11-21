//! Portable continuation reporting from installed owners and allocator facts.

use eredu_core::{
    capability::NativeAllocatorObservations, speculative::SpeculativeModelMemoryObservation,
    BackendFailure,
};

pub(super) fn model<M: NativeAllocatorObservations>(
    (current_positions, current_state_bytes, peak_state_bytes): (u64, Option<u64>, Option<u64>),
    residency: Option<crate::ResidencyReport>,
    banks: Option<crate::expert::ParameterBanksResidencyReport>,
) -> Result<SpeculativeModelMemoryObservation, BackendFailure> {
    use crate::text_session_driver::capability::{
        bank_memory_facts, static_memory_from_residency_and_components,
    };

    let parameter_conversions = residency
        .as_ref()
        .and_then(|report| report.device_parameter_conversions().map(<[_]>::to_vec));
    let parameters = static_memory_from_residency_and_components(
        residency,
        std::iter::empty(),
        bank_memory_facts(banks),
        M::native_memory_facts(),
    )
    .map_err(BackendFailure::from_error)?;
    Ok(SpeculativeModelMemoryObservation {
        current_positions,
        current_state_bytes,
        peak_state_bytes,
        parameters,
        parameter_conversions,
        available: M::available_memory()?,
        allocator_cache_limit: M::allocator_cache_limit(),
    })
}
