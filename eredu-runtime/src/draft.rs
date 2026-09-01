//! Transactional ownership for embedded draft mutable state.

/// An exact speculative fork retaining both the pre-verification checkpoint
/// and an independently advanceable draft state.
///
/// Model families supply ordinary cloneable runtime state; proposal,
/// verification, commit, cancellation, and rejection all use this one neutral
/// ownership boundary.
#[derive(Debug, Clone)]
pub struct DraftStateTransaction<S: Clone> {
    checkpoint: S,
    draft: S,
}

impl<S: Clone> DraftStateTransaction<S> {
    /// Forks draft state and preserves an exact rollback checkpoint.
    pub fn fork(state: &S) -> Self {
        Self {
            checkpoint: state.clone(),
            draft: state.clone(),
        }
    }

    /// Borrows the independently advanceable proposal state.
    pub const fn draft(&self) -> &S {
        &self.draft
    }

    /// Mutably borrows the independently advanceable proposal state.
    pub fn draft_mut(&mut self) -> &mut S {
        &mut self.draft
    }

    /// Borrows the exact state from before proposal and verification.
    pub const fn checkpoint(&self) -> &S {
        &self.checkpoint
    }

    /// Commits the advanced draft fork into canonical state.
    pub fn commit_draft(self, canonical: &mut S) {
        canonical.clone_from(&self.draft);
    }

    /// Restores canonical state after rejection, cancellation, or failed
    /// verification.
    pub fn rollback(self, canonical: &mut S) {
        canonical.clone_from(&self.checkpoint);
    }

    /// Keeps target state already advanced by successful verification while
    /// consuming the unused fork and checkpoint.
    pub fn commit_verified(self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_transaction_owns_fork_commit_and_rollback() {
        let mut canonical = vec![1, 2];
        let mut commit = DraftStateTransaction::fork(&canonical);
        commit.draft_mut().push(3);
        commit.commit_draft(&mut canonical);
        assert_eq!(canonical, [1, 2, 3]);

        let mut rollback = DraftStateTransaction::fork(&canonical);
        rollback.draft_mut().push(4);
        canonical.push(9); // target verification advanced canonical state
        rollback.rollback(&mut canonical);
        assert_eq!(canonical, [1, 2, 3]);

        let verified = DraftStateTransaction::fork(&canonical);
        canonical.push(5);
        verified.commit_verified();
        assert_eq!(canonical, [1, 2, 3, 5]);
    }

    #[test]
    fn prediction_groups_and_external_drafting_share_transactional_state() {
        let graph = crate::ExecutionGraph::new(
            vec![
                crate::ExecutionGroupSpec::root("target"),
                crate::ExecutionGroupSpec::root("external-drafter"),
                crate::ExecutionGroupSpec::with_dependencies(
                    "prediction.0",
                    ["target", "external-drafter"],
                ),
            ],
            "prediction.0",
        )
        .unwrap();
        assert_eq!(graph.group_index("external-drafter"), Some(1));
        assert_eq!(graph.group_index("prediction.0"), Some(2));

        let mut target = vec![10];
        let mut embedded = DraftStateTransaction::fork(&target);
        embedded.draft_mut().push(11);
        embedded.commit_draft(&mut target);

        let drafter_policy = ("external-drafter", 2usize);
        let mut external = DraftStateTransaction::fork(&target);
        external.draft_mut().extend([12, 13]);
        external.rollback(&mut target);

        assert_eq!(drafter_policy, ("external-drafter", 2));
        assert_eq!(target, [10, 11]);
    }
}
