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
/// Model families supply runtime state and its copy mechanism. Proposal,
/// verification, commit, cancellation, and rejection all use this one neutral
/// ownership boundary; retained state may refuse an independent copy.
#[derive(Debug, Clone)]
pub struct DraftStateTransaction<S> {
    checkpoint: S,
    draft: S,
}

impl<S> DraftStateTransaction<S> {
    /// Uses one fallible storage-copy worker for both independent members.
    /// A second-copy refusal retires the completed first copy; source is unchanged.
    pub fn try_fork<E>(
        state: &S,
        mut copy: impl FnMut(&S) -> Result<S, E>,
    ) -> Result<Self, E> {
        let checkpoint = copy(state)?;
        let draft = copy(state)?;
        Ok(Self { checkpoint, draft })
    }

    /// Independently copies both retained members with the same copy mechanism.
    pub fn try_copy<E>(
        &self,
        mut copy: impl FnMut(&S) -> Result<S, E>,
    ) -> Result<Self, E> {
        let checkpoint = copy(&self.checkpoint)?;
        let draft = copy(&self.draft)?;
        Ok(Self { checkpoint, draft })
    }

    /// Borrows the independently advanceable proposal state.
    pub const fn draft(&self) -> &S { &self.draft }

    /// Mutably borrows the independently advanceable proposal state.
    pub fn draft_mut(&mut self) -> &mut S { &mut self.draft }

    /// Borrows the exact state from before proposal and verification.
    pub const fn checkpoint(&self) -> &S { &self.checkpoint }

    /// Keeps target state advanced by verification while retiring both copies.
    pub fn commit_verified(self) {}
}

impl<S: Clone> DraftStateTransaction<S> {
    /// Forks draft state and preserves an exact rollback checkpoint.
    pub fn fork(state: &S) -> Self {
        match Self::try_fork(state, |state| Ok::<_, std::convert::Infallible>(state.clone())) {
            Ok(value) => value,
            Err(never) => match never {},
        }
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

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallible_transaction_retires_completed_prefix_and_retries_without_clone() {
        use std::{cell::Cell, rc::Rc};
        #[derive(Debug)]
        struct State {
            value: u32,
            live: Rc<Cell<usize>>,
        }
        impl State {
            fn new(value: u32, live: &Rc<Cell<usize>>) -> Self {
                live.set(live.get() + 1);
                Self { value, live: live.clone() }
            }
        }
        impl Drop for State {
            fn drop(&mut self) { self.live.set(self.live.get() - 1); }
        }
        let live = Rc::new(Cell::new(0));
        let source = State::new(37, &live);
        let mut attempts = 0;
        let refused = DraftStateTransaction::try_fork(&source, |source| {
            attempts += 1;
            if attempts == 2 { Err("fresh copy capacity exhausted") }
            else { Ok(State::new(source.value, &source.live)) }
        });
        assert_eq!(refused.unwrap_err(), "fresh copy capacity exhausted");
        assert_eq!(attempts, 2);
        assert_eq!(source.value, 37);
        assert_eq!(live.get(), 1);
        let mut fork = DraftStateTransaction::try_fork(&source, |source| {
            Ok::<_, std::convert::Infallible>(State::new(source.value, &source.live))
        }).unwrap();
        fork.draft_mut().value = 91;
        let copied = fork.try_copy(|source| {
            Ok::<_, std::convert::Infallible>(State::new(source.value, &source.live))
        }).unwrap();
        assert_eq!(copied.checkpoint().value, 37);
        assert_eq!(copied.draft().value, 91);
        assert_eq!(live.get(), 5);
        drop(fork);
        assert_eq!(live.get(), 3);
        drop(copied);
        assert_eq!(live.get(), 1);
        drop(source);
        assert_eq!(live.get(), 0);
    }

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
