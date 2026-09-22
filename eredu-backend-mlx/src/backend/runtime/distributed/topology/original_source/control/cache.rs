//! One source-owned prompt-cache protocol over the shared native role worker.
use super::*;
use crate::backend::submission_recovery::native_role::{self, NativeRoleCapacity, physical};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataAllocation};
use eredu_runtime::replicated_session::{
    SessionCacheControlCursor, SessionCacheControlPlan, SessionTransactionControlError,
};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, MemoryLedger};

struct Body {
    request: OriginalParallelControlRequest,
    cursor: RefCell<SessionCacheControlCursor>,
    ledger: MemoryLedger,
    execution: InferenceExecutionIdentity,
    context: WorkspaceContext,
    installed: Cell<bool>,
    active: Cell<bool>,
    running: Cell<bool>,
    finished: Cell<bool>,
    failed: Cell<bool>,
    funding: HostMetadataFunding,
}

/// Retains the actual initialized status inputs and exactly one save/load plan.
/// Every occurrence independently admits its native physical role; this owner
/// supplies no model invocation, text bank or replacement inference authority.
pub(crate) struct CacheControlOwner(Option<Rc<Body>>);
impl CacheControlOwner {
    fn body(&self) -> &Body {
        self.0.as_deref().expect("live cache control owner")
    }

    pub(crate) fn new(
        source: OriginalParallelControlSource,
        plan: SessionCacheControlPlan,
        ledger: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let funding = context
            .metadata_funding()
            .ok_or(Error::PrefillScopeUnavailable)?;
        reserve(
            &funding,
            &[
                size_of::<Self>(),
                size_of::<Body>(),
                size_of::<Result<Self, Error>>(),
                size_of::<(
                    OriginalParallelControlSource,
                    SessionCacheControlPlan,
                    &MemoryLedger,
                    &InferenceExecutionIdentity,
                    &WorkspaceContext,
                )>(),
                SessionCacheControlPlan::control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let source = ControlSource::Control(source);
        if !source.funding().same_account(&funding)
            || plan
                .occurrences()
                .any(|row| row.operation() != CommunicationOperation::FailureAgreement)
        {
            return Err(Error::PrefillScopeUnavailable);
        }
        let request = OriginalParallelControlRequest::from_source(source)?;
        let body = context
            .metadata_rc(Body {
                request,
                cursor: RefCell::new(plan.cursor()),
                ledger: ledger.clone(),
                execution: execution.clone(),
                context: context.clone(),
                installed: Cell::new(false),
                active: Cell::new(false),
                running: Cell::new(false),
                finished: Cell::new(false),
                failed: Cell::new(false),
                funding,
            })
            .map_err(|cause| Error::Neural(cause.into()))?;
        Ok(Self(Some(body)))
    }

    pub(crate) fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), Error> {
        reserve(
            &self.body().funding,
            &[
                size_of::<(&Self, &InferenceExecutionIdentity)>(),
                size_of::<Result<(), Error>>(),
            ],
        )?;
        if !self.body().execution.same_execution(execution) {
            return Err(self.body().reject(ControlCause::Identity));
        }
        Ok(())
    }

    /// A projection is valid only during this one lexical installation. The
    /// installation must remain alive until `finish` records the driver result.
    pub(crate) fn install(
        &self,
    ) -> Result<(CacheControlInstallation, CacheControlProjection), Error> {
        let body = self.body();
        reserve(
            &body.funding,
            &[
                size_of::<&Self>(),
                size_of::<CacheControlInstallation>(),
                size_of::<CacheControlProjection>(),
                size_of::<Result<(CacheControlInstallation, CacheControlProjection), Error>>(),
            ],
        )?;
        if body.installed.replace(true)
            || body.failed.get()
            || body.finished.get()
            || body.running.get()
        {
            return Err(body.reject(ControlCause::Exhausted));
        }
        body.active.set(true);
        Ok((
            CacheControlInstallation {
                owner: Self(self.0.clone()),
            },
            CacheControlProjection {
                body: Rc::downgrade(self.0.as_ref().expect("live cache control owner")),
                funding: body.funding.clone(),
            },
        ))
    }

    pub(crate) fn finish(&self, success: bool) -> Result<(), Error> {
        let body = self.body();
        reserve(
            &body.funding,
            &[
                size_of::<(&Self, bool)>(),
                size_of::<Result<(), Error>>(),
                size_of::<Result<(), SessionTransactionControlError>>(),
                size_of::<std::cell::RefMut<'_, SessionCacheControlCursor>>(),
            ],
        )?;
        if !body.active.get()
            || body.running.get()
            || body.finished.replace(true)
            || (success && body.failed.get())
        {
            return Err(body.reject(ControlCause::Closed));
        }
        let result = body
            .cursor
            .try_borrow_mut()
            .map_err(|_| body.reject(ControlCause::Unbound))?
            .finish(success);
        if !success {
            body.failed.set(true);
        }
        result.map_err(|cause| body.cursor_error(cause))
    }
}
impl Drop for CacheControlOwner {
    fn drop(&mut self) {
        if let Some(body) = self.0.take() {
            // Destroy the payload at the last strong owner, independently of
            // weak projections that still retain the paid Rc header/account.
            drop(Rc::into_inner(body));
        }
    }
}

pub(crate) struct CacheControlInstallation {
    owner: CacheControlOwner,
}
impl Drop for CacheControlInstallation {
    fn drop(&mut self) {
        let body = self.owner.body();
        body.active.set(false);
        if std::thread::panicking() || !body.finished.get() || body.running.get() {
            body.failed.set(true);
        }
    }
}

/// Weak execution view; its account retains the shared header, not the source
/// payload. Retired, inactive, repeated and reordered uses fail closed.
#[derive(Clone)]
pub(crate) struct CacheControlProjection {
    body: Weak<Body>,
    funding: HostMetadataFunding,
}
impl CacheControlProjection {
    fn owner(&self) -> Result<CacheControlOwner, Error> {
        reserve(
            &self.funding,
            &[
                size_of::<&Self>(),
                size_of::<Option<Rc<Body>>>(),
                size_of::<Result<CacheControlOwner, Error>>(),
            ],
        )?;
        self.body
            .upgrade()
            .map(|body| CacheControlOwner(Some(body)))
            .ok_or_else(|| Error::Neural(self.funding.metadata_source(ControlCause::Closed)))
    }

    pub(crate) fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), Error> {
        self.owner()?.validate_execution(execution)
    }

    pub(crate) fn run<T, E, F>(
        &self,
        event: ParallelControlEvent,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        let result = self.run_inner(event, stream, run);
        // A refusal before the native/cursor guard is installed still ends
        // this actual protocol attempt; a later call cannot retry its prefix.
        if !matches!(&result, Ok(Ok(_))) {
            if let Some(body) = self.body.upgrade() {
                body.failed.set(true);
            }
        }
        result.map_err(Error::into_backend_failure)
    }

    fn run_inner<T, E, F>(
        &self,
        event: ParallelControlEvent,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        let owner = self.owner()?;
        let body = owner.body();
        reserve(
            &body.funding,
            &[
                size_of::<(&Self, ParallelControlEvent, &Stream, F)>(),
                size_of::<CacheControlOwner>(),
                size_of::<Running<'_>>(),
                size_of::<OriginalParallelControlInvocation>(),
                size_of::<AgreementCapacity>(),
                size_of::<NativeRoleCapacity>(),
                size_of::<Option<OriginalScopeObserver>>(),
                size_of::<std::time::Duration>(),
                size_of::<std::cell::RefMut<'_, SessionCacheControlCursor>>(),
                size_of::<Result<usize, SessionTransactionControlError>>(),
                size_of::<Result<physical::CompletedNumerical<Result<T, E>>, physical::Failure>>(),
                size_of::<Result<Result<T, E>, Error>>(),
            ],
        )?;
        if !body.active.get() || body.failed.get() || body.finished.get() {
            return Err(body.reject(ControlCause::Closed));
        }
        if body.running.replace(true) {
            return Err(body.reject(ControlCause::Unbound));
        }
        let mut running = Running {
            body,
            complete: false,
        };
        // End the cursor loan before invoking any callback or native worker.
        body.cursor
            .try_borrow_mut()
            .map_err(|_| body.reject(ControlCause::Unbound))?
            .claim(event)
            .map_err(|cause| body.cursor_error(cause))?;
        let invocation = body
            .request
            .prepare_with_funding(event, None, &body.funding)?;
        let capacity = invocation.capacity();
        let inputs = body
            .request
            .source
            .agreement_inputs()
            .ok_or_else(|| body.reject(ControlCause::Identity))?;
        let timeout = body
            .request
            .retained
            .manifest()
            .completion_policy()
            .ok_or_else(|| body.reject(ControlCause::Identity))?
            .timeout();
        let parent = OriginalScopeObserver::try_current()
            .map_err(|cause| body.source_error(Cause::Native(cause)))?;
        let operation = |invocation: &OriginalParallelControlInvocation,
                         native: &native_role::NativeRoleContext<'_>| {
            invocation.with_context(native.observer(), stream, |group, funding| {
                run(Some((group, funding)))
            })
        };
        let plan = physical::Plan::new(
            inputs.runtime(),
            NativeRoleCapacity {
                graph: capacity.graph,
                records: capacity.records,
                backing: capacity.backing,
            },
            None,
            invocation,
            operation,
        )
        .with_parent(parent)
        .with_timeout(timeout);
        let result = plan
            .run(&body.ledger, &body.execution, &body.context)
            .map(|completed| completed.value)
            .map_err(|cause| match cause {
                physical::Failure::Admission(cause) => {
                    Error::Neural(body.context.metadata_source(cause))
                }
                physical::Failure::Native(cause) => {
                    Error::with_original_control_source(cause, false)
                }
            });
        running.complete = matches!(&result, Ok(Ok(_)));
        result
    }
}

impl Body {
    fn reject(&self, cause: ControlCause) -> Error {
        self.source_error(Cause::Control(cause))
    }
    fn source_error(&self, cause: Cause) -> Error {
        self.failed.set(true);
        if let Err(error) = reserve_bytes(&self.funding, failure_control_bytes()) {
            return error;
        }
        failure(cause, &self.request.retained, &self.funding)
    }
    fn cursor_error(&self, cause: SessionTransactionControlError) -> Error {
        self.failed.set(true);
        Error::Neural(self.context.metadata_source(cause))
    }
}

struct Running<'a> {
    body: &'a Body,
    complete: bool,
}
impl Drop for Running<'_> {
    fn drop(&mut self) {
        self.body.running.set(false);
        if !self.complete {
            self.body.failed.set(true);
        }
    }
}
