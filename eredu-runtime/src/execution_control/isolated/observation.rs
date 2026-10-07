//! Exact host diagnostic copies shared by realtime session snapshots.

use eredu_core::{
    execution_control::{ExecutionControlError, SnapshotEstimate},
    BackendFailure, RealtimeDecisionDiagnostics,
};

use super::{snapshot_value_estimate, SnapshotReservation, SnapshotValue};

impl SnapshotValue for RealtimeDecisionDiagnostics {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let bytes = self
            .tensor()
            .retained_storage_bytes()
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        snapshot_value_estimate::<Self>([Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })])
    }

    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(self.shape().len())
            .map_err(BackendFailure::from_error)?;
        shape.extend_from_slice(self.shape());
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(self.logits().len())
            .map_err(BackendFailure::from_error)?;
        logits.extend_from_slice(self.logits());
        Self::new(self.prediction(), shape, logits).map_err(BackendFailure::from_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_control::SnapshotBudget;
    use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};

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
