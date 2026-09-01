//! Transactional ownership for embedded draft mutable state.

/// Failure while executing one architecture-identified target or draft group.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DraftGroupExecutionError<E> {
    /// The selected architecture graph has no such group.
    #[error("execution graph has no target or draft group {0:?}")]
    UnknownGroup(String),
    /// The architecture-owned group executor failed.
    #[error("target or draft group execution failed")]
    Execution(#[source] E),
}

/// Executes one exact graph identity against the state owned by a target or
/// draft transaction.
pub fn execute_draft_group<S, I, O, E>(
    graph: &crate::ExecutionGraph,
    group: &str,
    input: I,
    state: &mut S,
    execute: impl FnOnce(usize, &str, I, &mut S) -> Result<O, E>,
) -> Result<O, DraftGroupExecutionError<E>> {
    let index = graph
        .group_index(group)
        .ok_or_else(|| DraftGroupExecutionError::UnknownGroup(group.to_owned()))?;
    execute(index, graph.groups()[index].id(), input, state)
        .map_err(DraftGroupExecutionError::Execution)
}

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
    fn prediction_and_external_draft_groups_drive_transactional_acceptance() {
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
        let mut trace = Vec::new();
        let execute = |group: &str, input: u32, state: &mut Vec<u32>, trace: &mut Vec<String>| {
            execute_draft_group(
                &graph,
                group,
                input,
                state,
                |group_index, id, input, state| {
                    let output = match group_index {
                        0 => input + 1,
                        1 => input + 3,
                        2 => input + 2,
                        _ => unreachable!("fixture graph has exactly three groups"),
                    };
                    state.push(output);
                    trace.push(id.to_owned());
                    Ok::<_, std::convert::Infallible>(output)
                },
            )
            .unwrap()
        };

        let mut canonical = vec![10];
        let target_output = execute("target", 0, &mut canonical, &mut trace);
        assert_eq!(target_output, 1);

        let mut embedded = DraftStateTransaction::fork(&canonical);
        let embedded_output = execute("prediction.0", 9, embedded.draft_mut(), &mut trace);
        assert_eq!(embedded_output, 11);
        embedded.commit_draft(&mut canonical);

        let mut accepted_external = DraftStateTransaction::fork(&canonical);
        let accepted = execute(
            "external-drafter",
            9,
            accepted_external.draft_mut(),
            &mut trace,
        );
        assert_eq!(accepted, 12);
        accepted_external.commit_draft(&mut canonical);

        let mut rejected_external = DraftStateTransaction::fork(&canonical);
        let rejected = execute(
            "external-drafter",
            10,
            rejected_external.draft_mut(),
            &mut trace,
        );
        assert_eq!(rejected, 13);
        rejected_external.rollback(&mut canonical);

        assert_eq!(canonical, [10, 1, 11, 12]);
        assert!(!canonical.contains(&rejected));
        assert_eq!(
            trace,
            [
                "target",
                "prediction.0",
                "external-drafter",
                "external-drafter"
            ]
        );
    }
}
