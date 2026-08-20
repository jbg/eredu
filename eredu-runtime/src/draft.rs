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
}
