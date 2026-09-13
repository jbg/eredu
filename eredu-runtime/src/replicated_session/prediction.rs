//! Observed auxiliary phases share ordinary preparation, recovery and commit.
use super::*;

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
    /// Runs a complete observed prediction phase, which may invoke several typed
    /// prediction operations. Preparation precedes mutation; native completion
    /// precedes receipt delivery and the common commit. No ordinary logits or
    /// generated token are published by this transaction.
    ///
    /// Architecture composition owns `work` and supplies its exact execution and
    /// completion callbacks. `complete` must settle every retained prediction
    /// state/output dependency, preserving the original failure and completion
    /// owner. Prediction-state rollback stays with that owner; this session also
    /// restores its own target state after a determinate failure. Restore never
    /// rewinds the shared forward epoch or observer allowances.
    #[allow(clippy::too_many_arguments)]
    pub fn with_prediction_observation<O, W, R, E>(
        &mut self,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        work: &mut W,
        execute: impl FnOnce(&mut Self, &mut W, &mut O) -> Result<R, E>,
        complete: impl FnOnce(&mut Self, &mut W, &R) -> Result<(), E>,
        map_error: impl Fn(ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>) -> E,
    ) -> Result<R, E>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let guard =
            crate::inspection::ObservationTransactionGuard::new(observer, self.next_commit_epoch);
        let result = (|| {
            let epoch = self.begin_commit_epoch().map_err(&map_error)?;
            self.prepare_observation_transaction(guard.observer, epoch, pass, context)
                .map_err(&map_error)?;
            let checkpoint = self
                .checkpoint_observed_state(context)
                .map_err(&map_error)?;
            let completed = (|| {
                let output = execute(self, work, guard.observer);
                let output = self.agree_prediction_observation(
                    crate::DistributedExecutionPhase::PredictionExtensionExecution,
                    output,
                    context,
                    &map_error,
                )?;
                let completion = complete(self, work, &output);
                self.agree_prediction_observation(
                    crate::DistributedExecutionPhase::MechanismCompletion,
                    completion,
                    context,
                    &map_error,
                )?;
                Ok(output)
            })();
            let output = match completed {
                Ok(output) => output,
                Err(error) => {
                    self.restore_failed_work(checkpoint, context)
                        .map_err(&map_error)?;
                    return Err(error);
                }
            };
            self.commit_observation_transaction(checkpoint, context, guard.observer)
                .map_err(&map_error)?;
            Ok(output)
        })();
        guard.finish(result.is_ok());
        result
    }

    fn agree_prediction_observation<R, E>(
        &mut self,
        phase: crate::DistributedExecutionPhase,
        local: Result<R, E>,
        context: &<B::Tensor as Tensor>::Context,
        map_error: &impl Fn(ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>) -> E,
    ) -> Result<R, E> {
        let agreement =
            D::agree_distributed_phase(&mut self.execution, phase, local.is_ok(), context);
        match (local, agreement) {
            (Err(error), _) => Err(error),
            (_, Err(error)) => Err(map_error(widen_infallible(error))),
            (Ok(output), Ok(true)) => Ok(output),
            (Ok(_), Ok(false)) => Err(map_error(ReplicatedTextSessionError::Partition(
                crate::PartitionExecutionError::RemotePhaseFailure(phase),
            ))),
        }
    }
}
