use eredu_core::residency::ParameterConversionRetentionEligibility;
use eredu_core::topology::ParallelTopology;

/// A process-local controller cannot grant one execution allowance across ranks.
/// Reject retention through typed eligibility while preserving temporary casts
/// and ordinary partitioned execution. Never give every rank the full allowance.
pub(super) fn partition_retention_exclusion(
    topology: ParallelTopology,
) -> Option<ParameterConversionRetentionEligibility> {
    (!topology.is_replicated()).then(|| ParameterConversionRetentionEligibility::Unsupported {
        reason: "MLX partitioned execution has no cross-rank conversion retention budget authority"
            .into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::residency::{
        ParameterConversionRetentionPolicy, ParameterConversionRetentionPolicyReport,
    };

    #[test]
    fn partition_retention_rejection_covers_every_axis_and_preserves_requested_policy() {
        assert!(
            partition_retention_exclusion(ParallelTopology::new(1, 1, 1, 1).unwrap()).is_none()
        );
        for axes in [
            [2, 1, 1, 1],
            [1, 2, 1, 1],
            [1, 1, 2, 1],
            [1, 1, 1, 2],
            [2, 2, 1, 1],
        ] {
            let [tensor, pipeline, expert, data] = axes;
            let topology = ParallelTopology::new(tensor, pipeline, expert, data).unwrap();
            let eligibility = partition_retention_exclusion(topology).unwrap();
            assert!(matches!(
                eligibility,
                ParameterConversionRetentionEligibility::Unsupported { .. }
            ));
            for requested in [None, Some(ParameterConversionRetentionPolicy::Unlimited)] {
                let report = ParameterConversionRetentionPolicyReport::resolve(
                    requested,
                    eligibility.clone(),
                );
                assert_eq!(
                    report.effective,
                    ParameterConversionRetentionPolicy::Disabled
                );
                assert_eq!(
                    report.requested,
                    requested.unwrap_or(ParameterConversionRetentionPolicy::MANAGED_DEFAULT)
                );
            }
        }
    }
}
