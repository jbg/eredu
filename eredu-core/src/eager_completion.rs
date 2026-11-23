//! Exact completion for work that finishes before publication.

use std::marker::PhantomData;

use crate::{Completion, SubmissionLease};

/// Completion of already finished work. A retained lease continues to guard
/// publication after native resources are safe to release.
#[must_use = "retain completion through result processing"]
pub struct EagerCompletion<E, R = ()> {
    _retained: R,
    error: PhantomData<fn() -> E>,
}

impl<E> EagerCompletion<E> {
    /// Creates a completion after all work and native borrows have settled.
    pub const fn completed() -> Self {
        Self::completed_retaining(())
    }
}

impl<E, R> EagerCompletion<E, R> {
    /// Creates a settled completion retaining an opaque publication owner.
    pub const fn completed_retaining(retained: R) -> Self {
        Self {
            _retained: retained,
            error: PhantomData,
        }
    }
}

impl<E, R: Clone> Clone for EagerCompletion<E, R> {
    fn clone(&self) -> Self {
        Self::completed_retaining(self._retained.clone())
    }
}

impl<E> EagerCompletion<E, SubmissionLease> {
    /// Creates a completion after all native work has settled, retaining
    /// submission authority through result processing until drop.
    pub fn completed_with_lease(lease: SubmissionLease) -> Self {
        Self::completed_retaining(lease)
    }
}

impl<E: std::error::Error + Send + Sync + 'static, R> Completion for EagerCompletion<E, R> {
    type Error = E;

    fn is_complete(&self) -> Result<bool, E> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), E> {
        Ok(())
    }
}

impl<E: std::error::Error + Send + Sync + 'static, R> crate::BoundedCompletion
    for EagerCompletion<E, R>
{
    fn wait_bounded(
        self,
        _: crate::BoundedCompletionWait,
    ) -> Result<crate::BoundedCompletionOutcome, E> {
        Ok(crate::BoundedCompletionOutcome::Completed)
    }
}

/// Owns an opaque payload whose native work settles before the operation returns.
/// Unwinding disables further mutation because the resulting payload may be partial.
pub struct EagerRetainedPayload<P, Q> {
    payload: P,
    queue: Q,
    abandoned: bool,
}

impl<P, Q> EagerRetainedPayload<P, Q> {
    /// Transforms settled startup data while preserving its queue and mutation authority.
    pub fn map<U>(self, map: impl FnOnce(P) -> U) -> EagerRetainedPayload<U, Q> {
        EagerRetainedPayload {
            payload: map(self.payload),
            queue: self.queue,
            abandoned: self.abandoned,
        }
    }

    /// Binds the payload to the context in which all its native work runs eagerly.
    pub const fn new(payload: P, queue: Q) -> Self {
        Self {
            payload,
            queue,
            abandoned: false,
        }
    }
}

/// Exclusive eager payload mutation, settled by the end of the call.
pub struct EagerPayloadOperation<'a, P> {
    payload: &'a mut P,
    abandoned: &'a mut bool,
}

impl<P: 'static, Q: Clone> crate::session_authority::RetainedExecutionPayload
    for EagerRetainedPayload<P, Q>
{
    type Payload = P;
    type Queue = Q;
    type Error = crate::BackendFailure;
    type Operation<'a>
        = EagerPayloadOperation<'a, P>
    where
        Self: 'a;

    fn payload(&self) -> &P {
        &self.payload
    }

    fn retain_queue(&self) -> Q {
        self.queue.clone()
    }

    fn begin_operation(&mut self) -> Result<Self::Operation<'_>, Self::Error> {
        if self.abandoned {
            return Err(crate::BackendFailure::new(
                crate::BackendFailureKind::InvalidSession,
                std::io::Error::other("eager payload operation ended without settling"),
            ));
        }
        self.abandoned = true;
        Ok(EagerPayloadOperation {
            payload: &mut self.payload,
            abandoned: &mut self.abandoned,
        })
    }
}

impl<P> crate::session_authority::RetainedPayloadOperation<crate::BackendFailure>
    for EagerPayloadOperation<'_, P>
{
    type Payload = P;

    fn payload_mut(&mut self) -> &mut P {
        self.payload
    }

    fn settle<T>(
        self,
        result: Result<T, crate::BackendFailure>,
    ) -> Result<T, crate::BackendFailure> {
        *self.abandoned = false;
        result
    }
}
