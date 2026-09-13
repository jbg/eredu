//! Host facts required for a common speculative scheduler action.

use crate::generation::{SpeculativeRequestId, SpeculativeRequestStatus};

/// One request's scheduling facts in stable registration order. Completion is
/// still owned by its executor; this record never releases native resources.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SpeculativeScheduleState {
    /// Exact request within the current registered table.
    pub request: SpeculativeRequestId,
    /// Current committed portable lifecycle state.
    pub status: SpeculativeRequestStatus,
    /// Local cancellation handle or previously deferred cancellation.
    pub cancellation_requested: bool,
    /// Exact local verification completion has been observed.
    pub verification_complete: bool,
    /// Local bounded verification deadline has expired.
    pub verification_deadline_expired: bool,
    /// Local executor and sampler can promote this optimistic branch exactly.
    pub optimistic_eligible: bool,
}

/// Agrees a local phase result before any subsequent native operation can enter
/// a collective. Local policy and native errors retain their original type.
pub(super) fn ready_result<'a, E, T>(
    executor: &mut E,
    local: Result<T, super::SpeculativeDriverError<E::Error>>,
    context: E::Context<'a>,
) -> Result<T, super::SpeculativeDriverError<E::Error>>
where
    E: super::SpeculativeExecutor + 'a,
{
    let stage = crate::run_preparation::TextPreparationStage::Delivery;
    crate::run_preparation::finish_preparation(
        stage,
        local,
        |status| executor.agree_text_preparation(stage, status, context),
        super::SpeculativeDriverError::Preparation,
    )
}

/// Publishes one already computed candidate, then resolves cancellation before
/// any participant installs its terminal lifecycle. Local publication failures
/// remain typed and cause every participant to roll back the same transaction.
pub(super) fn publish_candidate<'a, E, S, C, P>(
    executor: &mut E,
    runtime: &mut super::SpeculativeOutputRuntime<S, C, P>,
    constraint: &mut C,
    sequence: &mut crate::GenerationSequence,
    tokens: &[u32],
    context: E::Context<'a>,
) -> Result<bool, super::SpeculativeDriverError<E::Error>>
where
    E: super::SpeculativeExecutor + 'a,
    S: super::SpeculativeSampling,
    C: super::SpeculativeConstraint,
    P: super::SpeculativePublisher<C>,
{
    use super::SpeculativeDriverError as Error;
    let stage = crate::run_preparation::TextPreparationStage::Delivery;
    let local = runtime
        .publish_candidate(constraint, sequence, tokens)
        .map_err(Error::Output);
    let cancelled_locally = matches!(local, Ok(true));
    let ready = crate::run_preparation::finish_preparation_cancellable(
        stage,
        local.map(|cancelled| (!cancelled).then_some(())),
        |status| executor.agree_text_preparation(stage, status, context),
        Error::Preparation,
    )?;
    if ready.is_some() {
        return Ok(false);
    }
    runtime.cancellation().cancel();
    let local = if cancelled_locally {
        Ok(())
    } else {
        runtime
            .cancel_candidate(constraint, sequence)
            .map_err(Error::Output)
    };
    crate::run_preparation::finish_preparation(
        stage,
        local,
        |status| executor.agree_text_preparation(stage, status, context),
        Error::Preparation,
    )?;
    Ok(true)
}

/// Preserves successful completion ownership through publication and establishes
/// the selected safe disposition before propagating a failed host observation.
/// An error from polling never becomes a completion claim.
pub(super) fn settle_completion<E: super::SpeculativeExecutor>(
    completion: &mut Option<E::Completion>,
    policy: Result<crate::BoundedCompletionWait, crate::GenerationError>,
    observed: Result<(), super::SpeculativeOutputError>,
) -> Result<(), super::SpeculativeDriverError<E::Error>> {
    use super::SpeculativeDriverError as Error;
    use crate::{BoundedCompletion, BoundedCompletionOutcome, Completion};
    let policy = match policy {
        Ok(policy) => policy,
        Err(error) => {
            let emergency = crate::BoundedCompletionWait::new(
                std::time::Duration::from_nanos(1),
                crate::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .expect("emergency completion disposition is positive");
            completion
                .take()
                .expect("pending completion")
                .wait_bounded(emergency)?;
            return Err(Error::Generation(error));
        }
    };
    if let Err(error) = observed {
        completion
            .take()
            .expect("pending completion")
            .wait_bounded(policy)?;
        return Err(Error::Output(error));
    }
    match completion
        .as_ref()
        .expect("pending completion")
        .is_complete()
    {
        Ok(true) => {
            if let Err(error) = completion.as_ref().expect("pending completion").wait() {
                drop(completion.take());
                return Err(error.into());
            }
        }
        Ok(false) => match completion
            .take()
            .expect("pending completion")
            .wait_bounded(policy)?
        {
            BoundedCompletionOutcome::Completed => {}
            BoundedCompletionOutcome::DeadlineExceeded { cancellation } => {
                return Err(Error::CompletionDeadline { cancellation });
            }
        },
        Err(error) => {
            drop(completion.take());
            return Err(error.into());
        }
    }
    Ok(())
}
