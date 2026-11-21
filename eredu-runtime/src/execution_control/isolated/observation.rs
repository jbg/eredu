#[cfg(test)]
mod tests {
    use crate::execution_control::SnapshotBudget;
    use eredu_core::execution_control::SnapshotValue;
    use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
    use eredu_core::{execution_control::ExecutionControlError, RealtimeDecisionDiagnostics};

    #[test]
    fn diagnostic_spare_capacity_cannot_bypass_retention_admission() {
        let mut values = Vec::with_capacity(4096);
        values.push(0.25);
        let observation = RealtimeDecisionDiagnostics::new(0, vec![1], values).unwrap();
        let budget = SnapshotBudget::new(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 0,
            retained_bytes: 1024,
            cumulative_copy_bytes: u64::MAX,
        });
        assert!(matches!(
            budget.reserve(
                SnapshotResourceKind::Snapshot,
                Some(observation.estimate_snapshot_value().unwrap())
            ),
            Err(ExecutionControlError::Limit("retained bytes"))
        ));
        assert_eq!(budget.usage().retained_bytes, 0);
        assert_eq!(budget.usage().cumulative_copy_bytes, 0);
    }
}
