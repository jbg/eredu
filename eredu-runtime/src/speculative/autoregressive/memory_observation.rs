//! Portable continuation reporting from installed owners and allocator facts.

use eredu_core::{
    capability::ParameterMemoryFacts, speculative::SpeculativeModelMemoryObservation,
    BackendFailure,
};

pub(crate) fn model(
    (current_positions, current_state_bytes, peak_state_bytes): (u64, Option<u64>, Option<u64>),
    residency: Option<crate::ResidencyReport>,
    banks: Option<crate::expert::ParameterBanksResidencyReport>,
    native: ParameterMemoryFacts,
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
        native.native,
    )
    .map_err(BackendFailure::from_error)?;
    Ok(SpeculativeModelMemoryObservation {
        current_positions,
        current_state_bytes,
        peak_state_bytes,
        parameters,
        parameter_conversions,
        available: native.available,
        allocator_cache_limit: native.allocator_cache_limit,
    })
}
