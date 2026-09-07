//! Host-only selector inputs, separate from native tensors and family dispatch.
use crate::{Error, GroupSelection, TopKGroupSelectionSpec};

mod execution;
pub use execution::{
    execute_routing_intervention, RoutingExecutionError, RoutingMechanism, RoutingRows,
};

/// Precise stage at which router bias is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupScoreStage {
    /// Projection output, including its learned bias, before the score transform.
    RawLogits,
    /// Output of the score transform, before selection-only correction.
    TransformedScores,
    /// Ranking scores including learned corrections; coefficients stay unbiased.
    RankingScores,
}

/// Distinct pre-dispatch operations, all preserving shared-expert ownership.
#[derive(Debug, Clone)]
pub enum GroupSelectionAction {
    /// Remove IDs from eligibility without changing gathered score magnitudes.
    Exclude(Vec<u32>),
    /// Preserve IDs and zero specified coefficients without renormalization.
    ZeroContribution(Vec<u32>),
    /// Add one finite bias per ID at a precise stage.
    Bias {
        /// Score stage.
        stage: GroupScoreStage,
        /// Unique global IDs.
        ids: Vec<u32>,
        /// One finite bias per ID.
        values: Vec<f32>,
    },
    /// Complete per-selected-token IDs, in row-major order and native top-k width.
    Force(Vec<u32>),
}

/// Validated host controls applied to explicit token rows before expert dispatch.
#[derive(Debug, Clone)]
pub struct GroupSelectionControl {
    /// Exact architecture policy retained by admission; must match this selector.
    pub expected: TopKGroupSelectionSpec,
    /// Whether admission declared learned per-expert coefficient multipliers.
    pub learned_coefficient_scale: bool,
    /// Inclusive flattened token-row offset.
    pub first_row: u64,
    /// Exclusive flattened token-row offset.
    pub end_row: u64,
    /// Positive token-row stride.
    pub row_stride: u64,
    /// One operation; conflicting operations are rejected at portable admission.
    pub action: GroupSelectionAction,
    /// Whether to retain the original decision from the same router score tensor.
    pub capture_original: bool,
}

impl GroupSelectionControl {
    /// Rechecks exact runtime policy, row geometry, namespace, counts and grouping
    /// before native score work or expert dispatch. No tensors are inspected here.
    pub fn validate(
        &self,
        actual: TopKGroupSelectionSpec,
        learned_scale: bool,
        token_rows: u64,
    ) -> Result<(), Error> {
        let ensure = |condition, message| {
            if condition {
                Ok(())
            } else {
                Err(Error::backend(message))
            }
        };
        ensure(
            self.expected == actual && self.learned_coefficient_scale == learned_scale,
            "runtime router policy differs from intervention admission",
        )?;
        ensure(
            self.row_stride > 0 && self.first_row < self.end_row && self.end_row <= token_rows,
            "invalid intervention token-row selection",
        )?;
        let rows = (self.end_row - self.first_row).div_ceil(self.row_stride);
        let check_ids = |ids: &[u32]| {
            let mut unique = std::collections::BTreeSet::new();
            ensure(
                !ids.is_empty()
                    && ids
                        .iter()
                        .all(|id| *id < actual.group_count() as u32 && unique.insert(*id)),
                "duplicate or out-of-range intervention expert IDs",
            )
        };
        match &self.action {
            GroupSelectionAction::Exclude(ids) => {
                check_ids(ids)?;
                let width = actual.group_count() / actual.selection_partitions();
                let mut available = vec![width; actual.selection_partitions() as usize];
                for id in ids {
                    available[*id as usize / width as usize] -= 1;
                }
                available.sort_unstable();
                ensure(
                    available
                        .iter()
                        .take(actual.selected_groups() as usize)
                        .sum::<i32>()
                        >= actual.top_k(),
                    "exclusion makes grouped top-k infeasible",
                )?;
            }
            GroupSelectionAction::ZeroContribution(ids) => check_ids(ids)?,
            GroupSelectionAction::Bias { ids, values, .. } => {
                check_ids(ids)?;
                ensure(
                    ids.len() == values.len() && values.iter().all(|v| v.is_finite()),
                    "routing bias requires one finite value per ID",
                )?;
            }
            GroupSelectionAction::Force(ids) => {
                ensure(
                    rows.checked_mul(actual.top_k() as u64) == Some(ids.len() as u64),
                    "forced IDs differ from selected-token/top-k shape",
                )?;
                for row in ids.chunks(actual.top_k() as usize) {
                    check_ids(row)?;
                    let width = actual.group_count() / actual.selection_partitions();
                    let groups: std::collections::BTreeSet<_> =
                        row.iter().map(|id| *id / width as u32).collect();
                    ensure(
                        groups.len() <= actual.selected_groups() as usize,
                        "forced IDs exceed selected-group count",
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// Original (when requested) and effective decisions from one router evaluation.
pub struct IntervenedGroupSelection<T> {
    /// Original selection from the same input/scores, never a second model pass.
    pub original: Option<GroupSelection<T>>,
    /// Exact IDs and coefficients that must reach the provider before execution.
    pub effective: GroupSelection<T>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_control_checks_realized_policy_rows_and_group_feasibility() {
        let policy = TopKGroupSelectionSpec::new(8, 2, crate::GroupScoring::Sigmoid, true)
            .unwrap()
            .with_groups(2, 1)
            .unwrap()
            .with_weight_policy(0.1, 1.5)
            .unwrap();
        let mut control = GroupSelectionControl {
            expected: policy,
            learned_coefficient_scale: false,
            first_row: 1,
            end_row: 4,
            row_stride: 2,
            action: GroupSelectionAction::Force(vec![0, 1, 6, 7]),
            capture_original: true,
        };
        control.validate(policy, false, 4).unwrap();
        assert!(control.validate(policy, true, 4).is_err());
        assert!(control.validate(policy, false, 3).is_err());
        assert!(control
            .validate(policy.with_weight_policy(0.1, 2.0).unwrap(), false, 4)
            .is_err());
        for action in [
            GroupSelectionAction::Force(vec![0, 4, 6, 7]),
            GroupSelectionAction::Force(vec![0, 0, 6, 7]),
            GroupSelectionAction::Force(vec![0, 1]),
            GroupSelectionAction::Exclude(vec![0, 1, 2]),
            GroupSelectionAction::Exclude(vec![8]),
            GroupSelectionAction::Bias {
                stage: GroupScoreStage::RawLogits,
                ids: vec![0],
                values: vec![f32::NAN],
            },
        ] {
            control.action = action;
            assert!(control.validate(policy, false, 4).is_err());
        }
        control.action = GroupSelectionAction::Exclude(vec![0, 1]);
        control.validate(policy, false, 4).unwrap();
        control.action = GroupSelectionAction::ZeroContribution(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        control.validate(policy, false, 4).unwrap();
        control.row_stride = 0;
        assert!(control.validate(policy, false, 4).is_err());
    }
}
