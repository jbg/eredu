//! Preallocated recovery storage observing an existing original native scope.
//!
//! This helper supplies no admission or node-population authority. Its caller
//! must price every concrete node and supply its original custody before `new`.
//! Activation only fills that node; it creates no Scope, carrier, or allocation.
use super::*;
use safemlx::ScopedSubmissionProgress;

pub(crate) mod bank;
pub(crate) mod operation;

/// Fixed observation outcome and independent native lifetime evidence.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Observation {
    pub(crate) outcome: ScopedSubmissionProgress,
    pub(crate) status: Status,
}
impl Observation {
    fn busy() -> Self {
        Self {
            outcome: ScopedSubmissionProgress::Busy,
            status: Status {
                settled: false,
                failed: false,
                blocked: false,
            },
        }
    }
    fn retention_status(self) -> Status {
        Status {
            settled: self.can_retire(),
            ..self.status
        }
    }
    pub(crate) fn can_retire(self) -> bool {
        self.outcome == ScopedSubmissionProgress::Observed && self.status.settled
    }
    fn may_wait(self) -> bool {
        self.outcome == ScopedSubmissionProgress::Observed
            && !self.status.settled
            && !self.status.failed
            && !self.status.blocked
    }
}

/// Private observation seam. Implementations must never enter or seal a Scope.
pub(crate) trait Observer: 'static {
    type Error: 'static;
    fn observe(&self) -> Result<Observation, Self::Error>;
    /// Preserve a concrete cause before a consuming refusal drops this alias.
    /// Fixed observations remain available when a probe has no error adapter.
    #[track_caller]
    fn observation_error(&self, _observed: Observation) -> Option<Self::Error> {
        None
    }
    /// Exact-owner terminal cleanup, separately called after lifetime proof.
    fn retire_terminal(&self) -> Result<safemlx::SubmissionRetirement, Self::Error> {
        Ok(safemlx::SubmissionRetirement::CompleteSnapshot)
    }
}
impl Observer for safemlx::OriginalScopeObserver {
    type Error = Exception;
    fn observe(&self) -> Result<Observation, Exception> {
        self.progress().map(|(outcome, status)| Observation {
            outcome,
            status: Status {
                settled: status.is_settled(),
                failed: status.failed(),
                blocked: status.blocked(),
            },
        })
    }
    fn retire_terminal(&self) -> Result<safemlx::SubmissionRetirement, Exception> {
        self.retire_completed_records()
    }
    #[track_caller]
    fn observation_error(&self, observed: Observation) -> Option<Exception> {
        if observed.outcome != ScopedSubmissionProgress::Observed {
            return safemlx::OriginalScopeObserver::observation_error(self, observed.outcome);
        }
        if observed.status.failed {
            return self.retained_failure().or_else(|| {
                safemlx::OriginalScopeObserver::observation_error(
                    self,
                    ScopedSubmissionProgress::Unobservable,
                )
            });
        }
        if observed.status.blocked {
            return safemlx::OriginalScopeObserver::observation_error(
                self,
                ScopedSubmissionProgress::Unobservable,
            );
        }
        None
    }
}

struct ObservedProbe<P: Observer>(P);
impl<P: Observer> Probe for ObservedProbe<P> {
    fn seal(&mut self) {}
    // No thread-local capture chain was entered by this observer.
    fn seal_after_callback_failure(&mut self) {}
    fn progress(&self) -> Status {
        match self.0.observe() {
            Ok(observed) => observed.retention_status(),
            // No successful observation: retain the node without inventing a
            // native failure or poisoning a healthy transfer on transient Busy.
            Err(_) => Observation::busy().status,
        }
    }
    fn retire_terminal(&self) -> bool {
        matches!(
            self.0.retire_terminal(),
            Ok(safemlx::SubmissionRetirement::CompleteSnapshot)
        )
    }
}

struct ObservedRetention<T: Retention, C: 'static> {
    value: Option<T>,
    // Actual payload, observer, and final Box storage retire before custody.
    _custody: C,
}
impl<T: Retention, C: 'static> Retention for ObservedRetention<T, C> {
    fn observe(&self, status: Status) {
        // Handoff keeps this same node armed through unlocked T destruction
        // and final exact-owner record/wrapper retirement.
        if let Some(value) = &self.value {
            value.observe(status);
        }
    }
    fn retire_node<P: Probe>(mut node: Box<Node<Self, P>>) {
        let Some(value) = node.retention.value.take() else {
            // Never-started nodes and already-handed-off empty cleanup nodes
            // retire directly; neither invokes a user payload hook or cycles.
            retire_typed_node(node);
            return;
        };
        let cleanup = OriginalRetirementCleanup {
            _node: Some(PendingOwner(Some(node))),
        };
        // The same Box/observer/C are owned before the typed hook can unwind.
        value.retire_original(cleanup);
    }
}

/// Opaque same-node cleanup following actual separately deferred resources.
/// Non-Clone. Drop only moves the existing PendingOwner into quarantine; it
/// performs no native callback, allocation or authority construction.
pub(crate) struct OriginalRetirementCleanup {
    _node: Option<PendingOwner>,
}
impl OriginalRetirementCleanup {
    pub(crate) fn control_bytes() -> Option<u64> {
        let bytes = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Option<PendingOwner>>(),
            size_of::<PendingOwner>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        u64::try_from(bytes).ok()
    }
}

/// The final typed recovery Box, prepared before the selected operation.
pub(crate) struct PreparedObservedRecovery<
    T: Retention,
    C: 'static,
    P: Observer = safemlx::OriginalScopeObserver,
> {
    node: Option<NodeOwner<ObservedRetention<T, C>, ObservedProbe<P>>>,
}

/// Failed observer acquisition preserves the same prepared node and payload.
pub(crate) struct ObservedActivationError<T: Retention, C: 'static, P: Observer, E> {
    pub(crate) cause: E,
    pub(crate) retention: T,
    pub(crate) pending: PreparedObservedRecovery<T, C, P>,
}

/// Successful lifetime observation transfers T, retaining the SAME native
/// observer, Box and C until payload destruction and the final exact drain.
/// T precedes cleanup, including abandonment/unwind. Its Drop must retain its
/// existing safe deferral contract; explicit release checks the host boundary.
pub(crate) struct CompletedObservedRetention<
    T: Retention,
    C: 'static,
    P: Observer = safemlx::OriginalScopeObserver,
> {
    retention: Option<T>,
    cleanup: Option<ObservedRecovery<T, C, P>>,
}
impl<T: Retention, C: 'static, P: Observer> Drop for CompletedObservedRetention<T, C, P> {
    fn drop(&mut self) {
        let Some(mut cleanup) = self.cleanup.take() else {
            return;
        };
        if let Some(value) = self.retention.take() {
            // Abandonment returns T to the SAME empty node so nested deferred
            // payloads use retire_original rather than outliving cleanup.
            let target = &mut cleanup.inner.retention_mut().value;
            debug_assert!(target.is_none());
            *target = Some(value);
        }
        // Handoff already sealed local bookkeeping. Queue the same node with
        // no native callback or allocation, including unwind/error Drop.
        drop(
            cleanup
                .inner
                .node
                .take()
                .expect("completed cleanup node")
                .into_pending(),
        );
    }
}
impl<T: Retention, C: 'static, P: Observer> CompletedObservedRetention<T, C, P> {
    pub(crate) fn retention(&self) -> &T {
        self.retention.as_ref().expect("completed payload")
    }
    fn boundary_error(&self) -> FinishRetainingError<P::Error> {
        let observed = Observation::busy();
        match self
            .cleanup
            .as_ref()
            .expect("completed cleanup node")
            .consuming_error(observed)
        {
            Some(cause) => FinishRetainingError::Native(cause),
            None => FinishRetainingError::Observation(observed),
        }
    }
    pub(crate) fn retention_mut(&mut self) -> &mut T {
        self.retention.as_mut().expect("completed payload")
    }
    /// Run the caller's closed targeted teardown only at an unlocked boundary.
    /// Refusal retains both T's complete owner and the uncalled callback. The
    /// same cleanup node remains armed if the callback unwinds. The caller must
    /// price F's actual captures/dynamic children and its teardown work.
    pub(crate) fn release_with<F>(
        mut self,
        release: F,
    ) -> Result<Observation, CompletedReleaseWithError<T, C, P, F>>
    where
        F: FnOnce(&mut T),
    {
        if !safemlx::can_reclaim_submission_resources() {
            return Err(CompletedReleaseWithError {
                cause: self.boundary_error(),
                pending: Some(self),
                callback: Some(release),
            });
        }
        release(self.retention_mut());
        self.release().map_err(|error| CompletedReleaseWithError {
            cause: error.cause,
            pending: error.pending,
            callback: None,
        })
    }
    pub(crate) fn release_with_control_bytes<F>() -> Option<u64> {
        let bytes = [
            size_of::<F>(),
            size_of::<Option<F>>(),
            size_of::<&mut T>(),
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<CompletedReleaseWithError<T, C, P, F>>(),
            size_of::<Result<Observation, CompletedReleaseWithError<T, C, P, F>>>(),
            size_of::<CompletedReleaseError<T, C, P>>(),
            size_of::<Result<Observation, CompletedReleaseError<T, C, P>>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        u64::try_from(bytes).ok()
    }
    /// Destroy T outside the runtime guard, then drain exactly its native owner.
    /// Outer-guard/unwind refusal returns the untouched complete owner. Later
    /// drain refusal/error leaves the SAME empty node and C in quarantine.
    pub(crate) fn release(mut self) -> Result<Observation, CompletedReleaseError<T, C, P>> {
        if !safemlx::can_reclaim_submission_resources() {
            return Err(CompletedReleaseError {
                cause: self.boundary_error(),
                pending: Some(self),
            });
        }
        drop(self.retention.take());
        match self
            .cleanup
            .take()
            .expect("completed cleanup node")
            .finish()
        {
            Ok(observed)
                if observed.can_retire() && !observed.status.failed && !observed.status.blocked =>
            {
                Ok(observed)
            }
            Ok(observed) => Err(CompletedReleaseError {
                cause: FinishRetainingError::Observation(observed),
                pending: None,
            }),
            Err(cause) => Err(CompletedReleaseError {
                cause: FinishRetainingError::Native(cause),
                pending: None,
            }),
        }
    }
}
impl<T: Retention, C: 'static, P: Observer> std::fmt::Debug
    for CompletedObservedRetention<T, C, P>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletedObservedRetention")
            .finish_non_exhaustive()
    }
}
pub(crate) struct CompletedReleaseError<T: Retention, C: 'static, P: Observer> {
    pub(crate) cause: FinishRetainingError<P::Error>,
    pub(crate) pending: Option<CompletedObservedRetention<T, C, P>>,
}
pub(crate) struct CompletedReleaseWithError<T: Retention, C: 'static, P: Observer, F> {
    pub(crate) cause: FinishRetainingError<P::Error>,
    pub(crate) callback: Option<F>,
    // The still-armed node/C outlive an uncalled owning callback's destruction.
    pub(crate) pending: Option<CompletedObservedRetention<T, C, P>>,
}
impl<T: Retention, C: 'static, P: Observer, F> std::fmt::Debug
    for CompletedReleaseWithError<T, C, P, F>
where
    P::Error: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletedReleaseWithError")
            .field("cause", &self.cause)
            .field("pending", &self.pending.is_some())
            .field("callback", &self.callback.is_some())
            .finish()
    }
}
impl<T: Retention, C: 'static, P: Observer> std::fmt::Debug for CompletedReleaseError<T, C, P>
where
    P::Error: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletedReleaseError")
            .field("cause", &self.cause)
            .field("pending", &self.pending.is_some())
            .finish()
    }
}
#[derive(Debug)]
pub(crate) enum FinishRetainingError<E> {
    Observation(Observation),
    Native(E),
}

impl<T: Retention, C: 'static, P: Observer> PreparedObservedRecovery<T, C, P> {
    /// Caller-prepaid allocation, with the existing Box process-abort OOM
    /// contract. There is no native constructor, callback, hook or registration.
    pub(crate) fn new(custody: C) -> Self {
        Self {
            node: Some(NodeOwner(Some(Box::new(Node {
                probe: None,
                next: None,
                seal_attempted: false,
                seal_finished: false,
                callback_failed: Cell::new(false),
                retention: ObservedRetention {
                    value: None,
                    _custody: custody,
                },
            })))),
        }
    }

    /// Authenticate/acquire the native observer before filling or arming this
    /// node. Acquisition failure returns every supplied owner without a retry.
    pub(crate) fn try_activate<E>(
        self,
        retention: T,
        acquire: impl FnOnce() -> Result<P, E>,
    ) -> Result<ObservedRecovery<T, C, P>, ObservedActivationError<T, C, P, E>> {
        match acquire() {
            Ok(observer) => Ok(self.activate(retention, observer)),
            Err(cause) => Err(ObservedActivationError {
                cause,
                retention,
                pending: self,
            }),
        }
    }

    /// The supplied observer must already authenticate the genuine current
    /// original domain. Only moves occur here; no user/native callback runs.
    pub(crate) fn activate(mut self, retention: T, observer: P) -> ObservedRecovery<T, C, P> {
        let node = self
            .node
            .as_mut()
            .expect("prepared observed node")
            .node_mut();
        node.retention.value = Some(retention);
        node.probe = Some(ObservedProbe(observer));
        ObservedRecovery {
            inner: Recovery {
                node: self.node.take(),
            },
        }
    }

    /// One concrete node and named overlap only, excluding dynamic T/C/P
    /// payloads. The caller must additionally prove its finite node population.
    pub(crate) fn control_bytes<E>() -> Option<u64> {
        let extra = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<ObservedRecovery<T, C, P>>(),
            size_of::<ObservedActivationError<T, C, P, E>>(),
            size_of::<Result<ObservedRecovery<T, C, P>, ObservedActivationError<T, C, P, E>>>(),
            size_of::<T>(),
            size_of::<C>(),
            size_of::<P>(),
            size_of::<E>(),
            size_of::<Result<P, E>>(),
            size_of::<Observation>(),
            size_of::<Result<Observation, P::Error>>(),
            size_of::<Option<Result<Observation, P::Error>>>(),
            size_of::<P::Error>(),
            size_of::<Option<P::Error>>(),
            size_of::<Result<safemlx::SubmissionRetirement, P::Error>>(),
            size_of::<safemlx::SubmissionRetirement>(),
            size_of::<Result<bool, P::Error>>(),
            size_of::<CompletedObservedRetention<T, C, P>>(),
            size_of::<CompletedReleaseError<T, C, P>>(),
            size_of::<Result<Observation, CompletedReleaseError<T, C, P>>>(),
            size_of::<FinishRetainingError<P::Error>>(),
            size_of::<Result<CompletedObservedRetention<T, C, P>, FinishRetainingError<P::Error>>>(
            ),
            size_of::<Result<Option<T>, FinishRetainingError<P::Error>>>(),
            size_of::<Option<Result<Option<T>, FinishRetainingError<P::Error>>>>(),
            size_of::<&Self>(),
            size_of::<&mut ObservedRecovery<T, C, P>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        Recovery::<ObservedRetention<T, C>, ObservedProbe<P>>::node_control_bytes()?
            .checked_add(OriginalRetirementCleanup::control_bytes()?)?
            .checked_add(u64::try_from(extra).ok()?)
    }

    #[cfg(test)]
    fn allocation_identity(&self) -> usize {
        std::ptr::from_ref(self.node.as_ref().expect("prepared node").node()) as usize
    }
}

/// Existing quarantine engine with direct fixed-outcome observation for callers.
pub(crate) struct ObservedRecovery<
    T: Retention,
    C: 'static,
    P: Observer = safemlx::OriginalScopeObserver,
> {
    inner: Recovery<ObservedRetention<T, C>, ObservedProbe<P>>,
}
impl<T: Retention, C: 'static, P: Observer> ObservedRecovery<T, C, P> {
    /// Borrow the exact retained observer; this supplies no new authority.
    pub(crate) fn observer(&self) -> &P {
        &self
            .inner
            .node
            .as_ref()
            .expect("active observed node")
            .node()
            .probe
            .as_ref()
            .expect("active observer")
            .0
    }
    pub(crate) fn retention(&self) -> &T {
        self.inner
            .retention()
            .value
            .as_ref()
            .expect("activated retention")
    }
    pub(crate) fn retention_mut(&mut self) -> &mut T {
        self.inner
            .retention_mut()
            .value
            .as_mut()
            .expect("activated retention")
    }
    /// Local node bookkeeping only. The original outer Scope stays active.
    pub(crate) fn seal(&mut self) {
        self.inner.seal();
    }

    fn progress_guarded(&self) -> Result<Observation, P::Error> {
        let node = self
            .inner
            .node
            .as_ref()
            .expect("active observed node")
            .node();
        if node.callback_failed.get() {
            return Ok(Observation {
                outcome: ScopedSubmissionProgress::Unobservable,
                status: callback_blocked(),
            });
        }
        let _health = CallbackHealth::new(&node.callback_failed);
        let observed = node.probe.as_ref().expect("active observer").0.observe()?;
        node.retention.observe(observed.retention_status());
        Ok(observed)
    }
    fn retire_terminal_guarded(&self) -> Result<bool, P::Error> {
        let node = self
            .inner
            .node
            .as_ref()
            .expect("active observed node")
            .node();
        if node.callback_failed.get() {
            return Ok(false);
        }
        let _health = CallbackHealth::new(&node.callback_failed);
        node.probe
            .as_ref()
            .expect("active observer")
            .0
            .retire_terminal()
            .map(|result| result == safemlx::SubmissionRetirement::CompleteSnapshot)
    }
    fn consuming_error(&self, observed: Observation) -> Option<P::Error> {
        let node = self
            .inner
            .node
            .as_ref()
            .expect("active observed node")
            .node();
        if node.callback_failed.get() {
            return None;
        }
        let _health = CallbackHealth::new(&node.callback_failed);
        node.probe
            .as_ref()
            .expect("active observer")
            .0
            .observation_error(observed)
    }
    pub(crate) fn progress(&self) -> Result<Observation, P::Error> {
        safemlx::try_with_submission_retirement(|| self.progress_guarded())
            .unwrap_or_else(|| Ok(Observation::busy()))
    }
    /// Yield only for genuinely observed pending work. Fixed refusal and native
    /// error return promptly while this live handle still owns all resources.
    pub(crate) fn wait(&self) -> Result<Observation, P::Error> {
        loop {
            let observed = self.progress()?;
            if !observed.may_wait() {
                return Ok(observed);
            }
            std::thread::yield_now();
        }
    }
    /// One non-consuming completion attempt. Pending work, runtime/registry
    /// contention and errors leave this exact node owned by the caller. A true
    /// result means the successful terminal payload was retired; do not inspect
    /// the emptied recovery again. This never waits or enters a global reaper.
    pub(crate) fn try_finish_successfully(&mut self) -> Result<bool, P::Error> {
        self.seal();
        safemlx::try_with_submission_retirement(|| {
            let observed = self.progress_guarded()?;
            if observed.outcome == ScopedSubmissionProgress::Busy {
                return Ok(false);
            }
            if let Some(cause) = self.consuming_error(observed) {
                return Err(cause);
            }
            if !observed.can_retire() || observed.status.failed || observed.status.blocked {
                return Ok(false);
            }
            if !self.retire_terminal_guarded()? {
                return Ok(false);
            }
            self.inner
                .node
                .take()
                .expect("active observed node")
                .into_pending()
                .retire();
            Ok(true)
        })
        .unwrap_or(Ok(false))
    }

    pub(crate) fn try_finish_control_bytes() -> Option<u64> {
        [
            size_of::<Observation>(),
            size_of::<Option<P::Error>>(),
            size_of::<Result<bool, P::Error>>(),
            size_of::<Option<Result<bool, P::Error>>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|bytes| u64::try_from(bytes).ok())
    }
    /// Retire only under the same runtime guard that observed terminal native
    /// lifetime. Refusal/error drops into unchanged owning quarantine; it never
    /// abandons the original custody or substitutes ordinary progress.
    pub(crate) fn finish(mut self) -> Result<Observation, P::Error> {
        self.seal();
        loop {
            let observed = safemlx::try_with_submission_retirement(|| {
                let mut observed = self.progress_guarded()?;
                // Capture a failed/fixed observation's owning native cause
                // before a successful terminal pass can retire the observer.
                let mut cause = self.consuming_error(observed);
                if observed.can_retire() {
                    let retired = match self.retire_terminal_guarded() {
                        Ok(retired) => retired,
                        Err(error) => return Err(cause.unwrap_or(error)),
                    };
                    if retired {
                        self.inner
                            .node
                            .take()
                            .expect("active observed node")
                            .into_pending()
                            .retire();
                    } else {
                        observed.outcome = ScopedSubmissionProgress::Busy;
                        if cause.is_none() {
                            cause = self.consuming_error(observed);
                        }
                    }
                }
                match cause {
                    Some(cause) => Err(cause),
                    None => Ok::<Observation, P::Error>(observed),
                }
            })
            .unwrap_or_else(|| {
                let observed = Observation::busy();
                match self.consuming_error(observed) {
                    Some(cause) => Err(cause),
                    None => Ok(observed),
                }
            })?;
            if !observed.may_wait() {
                return Ok(observed);
            }
            std::thread::yield_now();
        }
    }

    /// Move out T after successful terminal observation, keeping this exact
    /// recovery node/observer/C armed through later unlocked T destruction and
    /// its resulting native deferred-wrapper retirement. No node is replaced.
    pub(crate) fn finish_retaining(
        mut self,
    ) -> Result<CompletedObservedRetention<T, C, P>, FinishRetainingError<P::Error>> {
        self.seal();
        loop {
            let result = safemlx::try_with_submission_retirement(|| {
                let mut observed = self
                    .progress_guarded()
                    .map_err(FinishRetainingError::Native)?;
                if observed.can_retire()
                    && !self
                        .retire_terminal_guarded()
                        .map_err(FinishRetainingError::Native)?
                {
                    observed.outcome = ScopedSubmissionProgress::Busy;
                }
                if observed.can_retire() && !observed.status.failed && !observed.status.blocked {
                    return Ok(Some(
                        self.inner
                            .retention_mut()
                            .value
                            .take()
                            .expect("active payload before handoff"),
                    ));
                }
                if observed.may_wait() {
                    Ok(None)
                } else {
                    Err(match self.consuming_error(observed) {
                        Some(cause) => FinishRetainingError::Native(cause),
                        None => FinishRetainingError::Observation(observed),
                    })
                }
            })
            .unwrap_or_else(|| {
                let observed = Observation::busy();
                Err(match self.consuming_error(observed) {
                    Some(cause) => FinishRetainingError::Native(cause),
                    None => FinishRetainingError::Observation(observed),
                })
            })?;
            if let Some(retention) = result {
                return Ok(CompletedObservedRetention {
                    retention: Some(retention),
                    cleanup: Some(self),
                });
            }
            std::thread::yield_now();
        }
    }

    #[cfg(test)]
    fn allocation_identity(&self) -> usize {
        std::ptr::from_ref(self.inner.node.as_ref().expect("active node").node()) as usize
    }
}

#[cfg(test)]
mod tests;
