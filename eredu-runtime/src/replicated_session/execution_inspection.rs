//! Paired, read-only inspection through the existing session boundary.
use super::*;

/// Fixed failure to borrow one coherent session at a quiescent boundary.
/// This value carries neither execution authority nor native completion evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RuntimeInspectionBoundary {
    /// A failed distributed control operation fenced the session.
    #[error("distributed session is fenced after failed operation at {phase:?}")]
    Fenced {
        /// Original phase that installed the fence.
        phase: crate::DistributedExecutionPhase,
    },
    /// The last distributed commit has no determinate outcome.
    #[error("distributed commit {epoch:?} is indeterminate at {phase:?}")]
    Indeterminate {
        /// The actual unresolved commit epoch.
        epoch: DistributedCommitEpoch,
        /// The actual unresolved commit phase.
        phase: DistributedCommitPhase,
    },
    /// A transaction still owns the mutable execution boundary.
    #[error("cannot inspect state during an active transaction")]
    Active {
        /// The actual active commit epoch.
        epoch: DistributedCommitEpoch,
    },
}
impl RuntimeInspectionBoundary {
    // The same precedence as ensure_commit_resolved followed by the active check.
    fn check(
        fence: Option<crate::DistributedExecutionPhase>,
        outcome: Option<DistributedCommitOutcome>,
        active: Option<DistributedCommitEpoch>,
    ) -> Result<(), Self> {
        Self::resolved(fence, outcome)?;
        if let Some(epoch) = active {
            return Err(Self::Active { epoch });
        }
        Ok(())
    }
    pub(super) fn unfenced(fence: Option<crate::DistributedExecutionPhase>) -> Result<(), Self> {
        match fence {
            Some(phase) => Err(Self::Fenced { phase }),
            None => Ok(()),
        }
    }
    pub(super) fn resolved(
        fence: Option<crate::DistributedExecutionPhase>,
        outcome: Option<DistributedCommitOutcome>,
    ) -> Result<(), Self> {
        Self::unfenced(fence)?;
        if let Some(DistributedCommitOutcome::Indeterminate { epoch, phase }) = outcome {
            return Err(Self::Indeterminate { epoch, phase });
        }
        Ok(())
    }
    pub(super) fn into_legacy<A: std::fmt::Display, P: std::fmt::Display, M: std::fmt::Display>(
        self,
    ) -> ReplicatedTextSessionError<A, P, M> {
        match self {
            Self::Fenced { phase } => ReplicatedTextSessionError::Contract(format!(
                "distributed session is fenced after failed operation at {phase:?}"
            )),
            Self::Indeterminate { epoch, phase } => {
                ReplicatedTextSessionError::CommitIndeterminate { epoch, phase }
            }
            Self::Active { .. } => ReplicatedTextSessionError::Contract(
                "cannot inspect state during an active transaction".into(),
            ),
        }
    }
}

impl<A, B, S, R, P> ReplicatedTextRuntime<A, B, S, R, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    /// Checked ceiling for the currently retained static and policy topology.
    /// This scalar does not retain owners, authorize replacement or certify
    /// future acquisitions. Unknown from either actual owner stays unknown.
    pub fn retained_value_slot_bound(&self) -> Option<usize> {
        match &self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime.retained_value_slot_bound(),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime.retained_value_slot_bound(),
        }
    }

    /// Validates an existing prepared token against this exact runtime without
    /// reconstructing declarations, rebinding paths or acquiring a unit.
    pub fn validate_observation_binding(
        &self,
        paths: &crate::PreparedLayeredObservationPaths,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>> {
        self.validate_observation_paths(paths)
    }
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Borrows the exact mechanisms, state and execution retained by one session.
    /// The shared borrow prevents their replacement until all returned borrows
    /// expire. This grants no mutable access, submission, completion, acquisition
    /// or funding authority; inspectors preserve unknown native facts.
    ///
    /// Fenced, indeterminate and active transactions reject before the callback,
    /// using the same errors and precedence as the existing state inspector.
    /// A strategy's runtime remains opaque: ordinary residency branches stay
    /// private, and a partition runtime is not converted to a full runtime.
    /// Strategy-owned auxiliary resources outside Runtime remain separate owners.
    pub fn inspect_runtime_execution<'source, T>(
        &'source self,
        inspect: impl FnOnce(&'source M, &'source M::State, &'source D::Runtime) -> Result<T, M::Error>,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.inspect_runtime_execution_fixed(inspect)
            .map_err(RuntimeInspectionBoundary::into_legacy)?
            .map_err(ReplicatedTextSessionError::Mechanism)
    }

    /// Borrows the same exact selected owners with a fixed outer boundary error.
    /// The inner result is the callback's original value/error, without wrapping,
    /// formatting or cloning it. Arbitrary callback work is not certified bounded.
    /// Boundary rejection occurs before the callback; this performs no native
    /// initialization, evaluation, cleanup, acquisition or state replacement.

    /// The result can borrow the actual source for exactly the session loan:
    ///
    /// ```
    /// use eredu_nn::{NeuralBackend, Tensor};
    /// use eredu_runtime::{SubmissionBackend, LayeredArchitecture, ReplicatedTextSession,
    ///     ReplicatedTextSessionMechanisms, ReplicatedTextExecutionStrategy};
    /// fn borrow_actual<A, B, M, D>(session: &ReplicatedTextSession<A, B, M, D>) -> &M
    /// where
    ///     B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    ///     M: ReplicatedTextSessionMechanisms<A, B>,
    ///     A: LayeredArchitecture<B, M::State>,
    ///     D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    ///     A::Error: std::fmt::Display,
    ///     M::PolicyError: std::fmt::Display,
    ///     M::Error: std::fmt::Display,
    /// {
    ///     session.inspect_runtime_execution_fixed(|m, _, _|
    ///         Ok::<_, std::convert::Infallible>(m)).unwrap().unwrap()
    /// }
    /// ```
    /// The same checked call cannot outlive its source:
    ///
    /// ```compile_fail,E0505
    /// use eredu_nn::{NeuralBackend, Tensor};
    /// use eredu_runtime::{SubmissionBackend, LayeredArchitecture, ReplicatedTextSession,
    ///     ReplicatedTextSessionMechanisms, ReplicatedTextExecutionStrategy};
    /// fn cannot_retire_borrowed<A, B, M, D>(session: ReplicatedTextSession<A, B, M, D>)
    /// where
    ///     B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    ///     M: ReplicatedTextSessionMechanisms<A, B>,
    ///     A: LayeredArchitecture<B, M::State>,
    ///     D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    ///     A::Error: std::fmt::Display,
    ///     M::PolicyError: std::fmt::Display,
    ///     M::Error: std::fmt::Display,
    /// {
    ///     let retained = session.inspect_runtime_execution_fixed(|m, _, _|
    ///         Ok::<_, std::convert::Infallible>(m)).unwrap().unwrap();
    ///     drop(session);
    ///     std::hint::black_box(retained);
    /// }
    /// ```
    pub fn inspect_runtime_execution_fixed<'source, T, E>(
        &'source self,
        inspect: impl FnOnce(&'source M, &'source M::State, &'source D::Runtime) -> Result<T, E>,
    ) -> Result<Result<T, E>, RuntimeInspectionBoundary> {
        RuntimeInspectionBoundary::check(
            self.control_fence,
            self.last_commit_outcome,
            self.active_commit_epoch,
        )?;
        Ok(inspect(&self.mechanisms, &self.state, &self.execution))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_inspection_boundary_preserves_fence_indeterminate_active_precedence() {
        let epoch = DistributedCommitEpoch::FIRST;
        let phase = DistributedCommitPhase::DecisionCompletion;
        let fence = crate::DistributedExecutionPhase::ControlCapturePreparation;
        let unresolved = Some(DistributedCommitOutcome::Indeterminate { epoch, phase });
        assert_eq!(
            RuntimeInspectionBoundary::check(Some(fence), unresolved, Some(epoch)),
            Err(RuntimeInspectionBoundary::Fenced { phase: fence })
        );
        assert_eq!(
            RuntimeInspectionBoundary::check(None, unresolved, Some(epoch)),
            Err(RuntimeInspectionBoundary::Indeterminate { epoch, phase })
        );
        for outcome in [
            None,
            Some(DistributedCommitOutcome::Committed(epoch)),
            Some(DistributedCommitOutcome::Aborted(epoch)),
        ] {
            assert_eq!(
                RuntimeInspectionBoundary::check(None, outcome, Some(epoch)),
                Err(RuntimeInspectionBoundary::Active { epoch })
            );
            assert_eq!(
                RuntimeInspectionBoundary::check(None, outcome, None),
                Ok(())
            );
        }
        let fixed = RuntimeInspectionBoundary::Active { epoch };
        assert_eq!(
            fixed.into_legacy::<&str, &str, &str>().to_string(),
            "replicated text contract mismatch: cannot inspect state during an active transaction"
        );
    }
}
