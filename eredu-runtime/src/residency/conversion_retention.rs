//! Portable selection of the conversion budget for one residency installation.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    OnceLock,
};

use eredu_core::{
    residency::{
        conversion_claim::{
            ConversionRetentionBudget, ConversionRetentionError, ConversionRetentionRegistry,
        },
        ParameterConversionRetentionEligibility as Eligibility, ParameterConversionRetentionGroup,
        ParameterConversionRetentionPolicyReport,
    },
    resources::ResourceIdentity,
};

pub(super) fn execution_budget(
    policy: ParameterConversionRetentionPolicyReport,
) -> Result<ConversionRetentionBudget, ConversionRetentionError> {
    static REGISTRY: OnceLock<ConversionRetentionRegistry> = OnceLock::new();
    static NEXT_GROUP: AtomicU64 = AtomicU64::new(1);
    let group = ParameterConversionRetentionGroup(ResourceIdentity {
        scope: "residency.parameter_conversion_group".into(),
        key: NEXT_GROUP.fetch_add(1, Ordering::Relaxed).to_string(),
    });
    REGISTRY
        .get_or_init(ConversionRetentionRegistry::default)
        .budget(group, policy)
}

pub(super) fn apply_mechanism_eligibility(
    current: Eligibility,
    mechanism: Eligibility,
) -> Eligibility {
    match current {
        Eligibility::DeviceResidencyLimit => Eligibility::DeviceResidencyLimit,
        Eligibility::HostLayerwise => Eligibility::HostLayerwise,
        Eligibility::DiskStreamed => Eligibility::DiskStreamed,
        _ => mechanism,
    }
}

#[cfg(test)]
mod tests;
