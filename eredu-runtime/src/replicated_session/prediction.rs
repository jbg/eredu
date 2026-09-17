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
    // Capture publication is shared by full verification and selected prefill.
    pub(super) fn prepare_prediction_target_capture(
        &mut self,
        forward: &A::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        let capture = D::prediction_target_capture(&mut self.execution, forward, context)
            .map_err(widen_infallible)
            .and_then(|capture| {
                capture.ok_or_else(|| {
                    ReplicatedTextSessionError::Contract(
                        "prediction target pass did not retain its declared hidden capture".into(),
                    )
                })
            });
        let capture = self.agree_prediction_observation(
            crate::DistributedExecutionPhase::PredictionTargetCapture,
            capture,
            context,
            &|error| error,
        )?;
        let publication =
            D::publish_prediction_target_capture(&mut self.execution, capture, context)
                .map_err(widen_infallible);
        self.agree_prediction_observation(
            crate::DistributedExecutionPhase::PredictionTargetCapturePublication,
            publication,
            context,
            &|error| error,
        )
    }

    /// Selects prefill scores independently from the exact declared target
    /// capture. The returned frontier is inspected from the installed state
    /// after target publication; no saved capture generation supplies it.
    /// Caller retains the enclosing chunk guard through dependent seed work.
    pub fn prefill_prediction_span<O>(
        &mut self,
        input: Result<A::Input<'_>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        PublishedPredictionPrefill<B::Tensor>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let (scores, capture) =
            self.with_observation_transaction(observer, |session, observer| {
                session.require_input_result_agreement(true)?;
                let (output, checkpoint, forward) = session
                    .execute_input_result_before_publication_with_readout(
                        input,
                        ExpertPass::Prefill,
                        context,
                        observer,
                        demand,
                        None,
                    )?;
                let capture = match session.prepare_prediction_target_capture(&forward, context) {
                    Ok(capture) => capture,
                    Err(error) => return session.rollback_failure(checkpoint, error, context),
                };
                let (output, checkpoint, forward) = session
                    .publish_observed_output_transaction_with_readout(
                        output, checkpoint, forward, context,
                    )?;
                let output = session
                    .publish_with_readout(output, checkpoint, forward, context, observer, true)?;
                Ok((output, capture))
            })?;
        let generation = self
            .mechanisms
            .prefill_state_frontier(&self.state)
            .map_err(ReplicatedTextSessionError::Mechanism)
            .and_then(|generation| {
                generation.ok_or_else(|| {
                    ReplicatedTextSessionError::Contract(
                        "captured prefill requires the actual installed target frontier".into(),
                    )
                })
            });
        let generation = self.agree_prediction_observation(
            crate::DistributedExecutionPhase::PredictionTargetCapturePublication,
            generation,
            context,
            &|error| error,
        )?;
        Ok(PublishedPredictionPrefill {
            scores,
            capture,
            generation,
            target_commit: self.last_commit_outcome,
        })
    }

    /// Publishes a selected local span and an architecture-owned typed capture.
    /// Multi-tensor capture publication across selected partitions remains a
    /// separate mechanism. The caller retains the same chunk guard through its
    /// capture consumer and all derived-root completion.
    pub fn prefill_capture_span<C, O>(
        &mut self,
        input: Result<A::Input<'_>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        capture: impl FnOnce(&A::ForwardContext) -> Result<C, A::Error>,
    ) -> Result<
        (
            Option<B::Tensor>,
            C,
            u64,
            Option<eredu_core::DistributedCommitOutcome>,
        ),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        if D::PARTITIONED_SESSION {
            return Err(ReplicatedTextSessionError::Contract(
                "partitioned multi-tensor prediction capture requires selected bundle publication"
                    .into(),
            ));
        }
        let (scores, capture) =
            self.with_observation_transaction(observer, |session, observer| {
                session.require_input_result_agreement(true)?;
                let (output, checkpoint, forward) = session
                    .execute_input_result_before_publication_with_readout(
                        input,
                        ExpertPass::Prefill,
                        context,
                        observer,
                        demand,
                        None,
                    )?;
                let capture = match capture(&forward) {
                    Ok(value) => value,
                    Err(error) => {
                        return session.rollback_failure(
                            checkpoint,
                            ReplicatedTextSessionError::Architecture(error),
                            context,
                        )
                    }
                };
                let (output, checkpoint, forward) = session
                    .publish_observed_output_transaction_with_readout(
                        output, checkpoint, forward, context,
                    )?;
                let output = session
                    .publish_with_readout(output, checkpoint, forward, context, observer, true)?;
                Ok((output, capture))
            })?;
        let generation = self
            .mechanisms
            .prefill_state_frontier(&self.state)
            .map_err(ReplicatedTextSessionError::Mechanism)
            .and_then(|generation| {
                generation.ok_or_else(|| {
                    ReplicatedTextSessionError::Contract(
                        "captured prefill requires the actual installed target frontier".into(),
                    )
                })
            });
        let generation = self.agree_prediction_observation(
            crate::DistributedExecutionPhase::PredictionTargetCapturePublication,
            generation,
            context,
            &|error| error,
        )?;
        Ok((scores, capture, generation, self.last_commit_outcome))
    }

    /// Agrees fallible span-observer/consumer preparation before any participant
    /// enters the next target or auxiliary transaction. Preserves a local cause.
    pub fn agree_prediction_prefill_preparation<T>(
        &mut self,
        local: Result<T, A::Error>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.agree_prediction_observation(
            crate::DistributedExecutionPhase::PredictionExtensionExecution,
            local.map_err(ReplicatedTextSessionError::Architecture),
            context,
            &|error| error,
        )
    }

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
        let agreement = self.agree_execution_phase(phase, local.is_ok(), context);
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

/// A capture published by the installed target, with its own commit identity.
/// Construction is private to the actual session transaction.
pub struct PublishedPredictionPrefill<T> {
    scores: Option<T>,
    capture: T,
    generation: u64,
    target_commit: Option<eredu_core::DistributedCommitOutcome>,
}
impl<T> PublishedPredictionPrefill<T> {
    /// Actual installed target frontier after this capture was published.
    pub fn generation(&self) -> u64 {
        self.generation
    }
    /// Borrows actual published roots for the enclosing native completion.
    pub fn visit_roots(&self, visit: &mut dyn FnMut(&T)) {
        if let Some(scores) = &self.scores { visit(scores); }
        visit(&self.capture);
    }
    /// Moves the published values and the target's pre-auxiliary receipt.
    pub fn into_parts(self) -> (Option<T>, T, Option<eredu_core::DistributedCommitOutcome>) {
        (self.scores, self.capture, self.target_commit)
    }
}
