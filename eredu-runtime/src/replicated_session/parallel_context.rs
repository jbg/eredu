//! Scoped backend context lending around the ordinary shared session driver.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// Fixed rejection before a scoped parallel context is installed.
#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum PreparedParallelContextCause {
    /// The same session boundary used by immutable source inspection refused.
    #[error(transparent)]
    Boundary(#[from] RuntimeInspectionBoundary),
    /// This actual execution strategy has no context replacement mechanism.
    #[error("selected execution does not accept a prepared parallel context")]
    Unsupported,
    /// The exact scoped replacement controls did not fit their host source.
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
}
/// Rejection with the exact uninstalled context. No user callback has run.
pub struct PreparedParallelContextFailure<C> {
    cause: PreparedParallelContextCause,
    context: C,
}
impl<C> PreparedParallelContextFailure<C> {
    /// Fixed cause, without erasure or formatting allocation.
    pub fn cause(&self) -> PreparedParallelContextCause {
        self.cause
    }
    /// Recover the exact context and its custody.
    pub fn into_parts(self) -> (PreparedParallelContextCause, C) {
        (self.cause, self.context)
    }
}
impl<C> std::fmt::Debug for PreparedParallelContextFailure<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedParallelContextFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
struct Restore<'a, S, C> {
    session: &'a mut S,
    previous: Option<C>,
    replace: fn(&mut S, C) -> Result<C, C>,
}
impl<S, C> Drop for Restore<'_, S, C> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            // Successful installation promised an allocation-free inverse.
            // This is independent of operation success and communication fate.
            assert!(
                (self.replace)(self.session, previous).is_ok(),
                "installed parallel context lost its restoration mechanism"
            );
        }
    }
}
pub(super) fn with_runtime<R, C, T, F>(
    runtime: &mut R,
    context: C,
    funding: &HostMetadataFunding,
    replace: fn(&mut R, C) -> Result<C, C>,
    run: F,
) -> Result<T, PreparedParallelContextFailure<C>>
where
    F: FnOnce(&mut R) -> T,
{
    let controls = [
        size_of::<Restore<'_, R, C>>(),
        size_of::<C>(),
        size_of::<Option<C>>(),
        size_of::<Result<C, C>>(),
        size_of::<PreparedParallelContextFailure<C>>(),
        size_of::<Result<T, PreparedParallelContextFailure<C>>>(),
        size_of::<F>(),
        size_of::<T>(),
        size_of::<(&mut R, &HostMetadataFunding, fn(&mut R, C) -> Result<C, C>)>(),
    ];
    if let Err(cause) = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or(HostMetadataFundingError::Overflow)
        .and_then(|n| funding.reserve_metadata(n))
    {
        return Err(PreparedParallelContextFailure {
            cause: cause.into(),
            context,
        });
    }
    let previous = match replace(runtime, context) {
        Ok(previous) => previous,
        Err(context) => {
            return Err(PreparedParallelContextFailure {
                cause: PreparedParallelContextCause::Unsupported,
                context,
            });
        }
    };
    let guard = Restore {
        session: runtime,
        previous: Some(previous),
        replace,
    };
    Ok(run(&mut *guard.session))
}

struct RestoreBorrowed<'a, R, C: ?Sized> {
    runtime: &'a mut R,
    context: &'a mut C,
    exchange: fn(&mut R, &mut C) -> bool,
}
impl<R, C: ?Sized> Drop for RestoreBorrowed<'_, R, C> {
    fn drop(&mut self) {
        assert!(
            (self.exchange)(self.runtime, self.context),
            "installed parallel context lost its restoration mechanism"
        );
    }
}

pub(super) fn with_borrowed_runtime<R, C: ?Sized, T, F>(
    runtime: &mut R,
    context: &mut C,
    funding: &HostMetadataFunding,
    exchange: fn(&mut R, &mut C) -> bool,
    run: F,
) -> Result<T, PreparedParallelContextCause>
where
    F: FnOnce(&mut R) -> T,
{
    let controls = [
        size_of::<RestoreBorrowed<'_, R, C>>(),
        size_of::<Result<T, PreparedParallelContextCause>>(),
        size_of::<F>(),
        size_of::<T>(),
        size_of::<PreparedParallelContextCause>(),
        size_of::<bool>(),
        size_of::<(
            &mut R,
            &mut C,
            &HostMetadataFunding,
            fn(&mut R, &mut C) -> bool,
        )>(),
    ];
    let bytes = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or(HostMetadataFundingError::Overflow)?;
    funding.reserve_metadata(bytes)?;
    if !exchange(runtime, context) {
        return Err(PreparedParallelContextCause::Unsupported);
    }
    let guard = RestoreBorrowed {
        runtime,
        context,
        exchange,
    };
    Ok(run(&mut *guard.runtime))
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B::ParallelContext: Sized,
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Executes the existing driver under one explicit backend context and
    /// restores the prior context on return, error, or unwind. The caller keeps
    /// any native source/completion owner alive through its enclosing recovery;
    /// restoring the context is not a completion or refund claim.
    pub fn with_prepared_parallel_context<T, F>(
        &mut self,
        context: B::ParallelContext,
        funding: &HostMetadataFunding,
        run: F,
    ) -> Result<T, PreparedParallelContextFailure<B::ParallelContext>>
    where
        F: FnOnce(&mut Self) -> T,
    {
        let controls = [
            size_of::<B::ParallelContext>(),
            size_of::<PreparedParallelContextFailure<B::ParallelContext>>(),
            size_of::<Result<T, PreparedParallelContextFailure<B::ParallelContext>>>(),
            size_of::<F>(),
            size_of::<T>(),
            size_of::<PreparedParallelContextCause>(),
            size_of::<(&mut Self, &HostMetadataFunding)>(),
            size_of::<Result<Result<(), std::convert::Infallible>, RuntimeInspectionBoundary>>(),
        ];
        let reserve = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(HostMetadataFundingError::Overflow)
            .and_then(|bytes| funding.reserve_metadata(bytes));
        if let Err(cause) = reserve {
            return Err(PreparedParallelContextFailure {
                cause: cause.into(),
                context,
            });
        }
        if let Err(cause) =
            self.inspect_runtime_execution_fixed(|_, _, _| Ok::<(), std::convert::Infallible>(()))
        {
            return Err(PreparedParallelContextFailure {
                cause: cause.into(),
                context,
            });
        }
        with_runtime(
            self,
            context,
            funding,
            |session, context| D::replace_parallel_context(&mut session.execution, context),
            run,
        )
    }
}
