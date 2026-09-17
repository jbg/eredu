//! Host-only selector inputs, separate from native tensors and family dispatch.
use crate::{Error, GroupSelection, TopKGroupSelectionSpec};

mod validation;
pub use validation::GroupSelectionValidationError;

mod execution;
pub use execution::{
    execute_routing_intervention, execute_routing_intervention_fixed, FixedRoutingExecutionError,
    RoutingExecutionError, RoutingInvalidCause, RoutingMechanism, RoutingRows,
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
        self.validate_with_workspace(
            actual,
            learned_scale,
            token_rows,
            validation::ExclusionWorkspace::Ordinary,
        )
        .map_err(Error::backend)
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
