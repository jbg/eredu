use super::*;

/// A submitted distributed result and its exact backend completion.
///
/// Construction explicitly evaluates the supplied MLX outputs. Merely
/// constructing a send, receive, collective, or pipeline graph does not create
/// a completion. The retained arrays keep every submitted endpoint alive until
/// this value is dropped; MLX additionally retains backend resources while
/// producer work or consumer waits remain outstanding.
///
/// A completion is single-shot but may be queried, host-waited, or waited on by
/// multiple compatible streams. [`Self::wait_on`] is a backend stream
/// dependency and does not block the host. Dropping the value is safe while
/// work remains outstanding, but asynchronous errors are observable only
/// through [`Self::is_complete`], [`Self::synchronize`], or
/// [`Self::into_value`].
///
/// This type is intentionally neither `Send` nor `Sync`, matching `safemlx`'s
/// thread-affine [`Event`] contract.
#[derive(Debug)]
#[must_use = "distributed work has been submitted; retain, wait on, or synchronize its completion"]
pub struct DistributedCompletion<T> {
    value: T,
    event: Rc<Event>,
    recovery: Rc<Recovery<Rc<NativeResources>>>,
    authority: Option<AuthorizedCompletion>,
    quarantined: Cell<bool>,
    #[cfg(test)]
    force_pending: Rc<Cell<bool>>,
}

#[derive(Debug)]
struct AuthorizedCompletion {
    authority: eredu_runtime::PartitionCommunicationAuthority,
    operation: eredu_runtime::CommunicationOperation,
    phase: eredu_runtime::DistributedExecutionPhase,
}

#[derive(Debug)]
pub(super) struct DistributedCompletionOrphan {
    pub(super) _event: Rc<Event>,
    recovery: Rc<Recovery<Rc<NativeResources>>>,
    #[cfg(test)]
    pub(super) force_pending: Rc<Cell<bool>>,
}

#[derive(Debug, Default)]
pub(super) struct DistributedCompletionOrphanQuarantine {
    pub(super) work: Vec<DistributedCompletionOrphan>,
}

impl DistributedCompletionOrphanQuarantine {
    pub(super) fn reap(&mut self) {
        self.work.retain(|work| {
            #[cfg(test)]
            if work.force_pending.get() {
                return true;
            }
            !work.recovery.progress().settled
        });
    }
}

impl Drop for DistributedCompletionOrphanQuarantine {
    fn drop(&mut self) {
        self.reap();
        // Dropping the shared recovery handle transfers unresolved resources to
        // the preallocated nonblocking recovery quarantine, including TLS exit.
        self.work.clear();
    }
}

fn reap_distributed_completion_orphans() {
    let empty = DISTRIBUTED_COMPLETION_ORPHANS.try_with(|orphans| {
        if let Ok(mut orphans) = orphans.try_borrow_mut() {
            orphans.reap();
            return orphans.work.is_empty();
        }
        false
    });
    if matches!(empty, Ok(true)) {
        safemlx::unregister_thread_runtime_housekeeping(reap_distributed_completion_orphans);
    }
}

impl<T> DistributedCompletion<T> {
    fn ensure_authority_active(&self) -> Result<(), Error> {
        match &self.authority {
            Some(context) => context
                .authority
                .ensure_active()
                .map_err(|error| Error::Parallel(error.to_string())),
            None => Ok(()),
        }
    }

    /// Submits the supplied output arrays and couples their event to `value`.
    pub fn submit<'a>(
        value: T,
        outputs: impl IntoIterator<Item = &'a Array>,
    ) -> Result<Self, Error> {
        let retained = outputs.into_iter().cloned().collect::<Vec<_>>();
        let mut recovery = Recovery::begin(NativeResources::new(
            retained,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))?;
        let event = async_eval_with_event(recovery.retention().arrays.iter());
        recovery.seal();
        recovery.retention().host_failed.set(event.is_err());
        recovery.progress();
        let event = Rc::new(event?);
        *recovery.retention().event.borrow_mut() = Some(Rc::clone(&event));
        check_native_status(&recovery)?;
        Ok(Self {
            value,
            event,
            recovery: Rc::new(recovery),
            authority: None,
            quarantined: Cell::new(false),
            #[cfg(test)]
            force_pending: Rc::new(Cell::new(false)),
        })
    }

    /// Submits one selected session operation and retains every native dependency.
    pub(crate) fn submit_authorized<'a>(
        value: T,
        outputs: impl IntoIterator<Item = &'a Array>,
        retained: Vec<Array>,
        count_buffers: Vec<Vec<usize>>,
        groups: Vec<Group>,
        routes: Vec<CommunicationRouteRealization>,
        streams: Vec<Stream>,
        authority: eredu_runtime::PartitionCommunicationAuthority,
        operation: eredu_runtime::CommunicationOperation,
    ) -> Result<Self, Error> {
        let outputs = outputs.into_iter().cloned().collect::<Vec<_>>();
        let mut recovery = Recovery::begin(NativeResources::new(
            retained,
            count_buffers,
            groups,
            routes,
            streams,
        ))?;
        let event = async_eval_with_event(outputs.iter());
        recovery.seal();
        recovery.retention().host_failed.set(event.is_err());
        recovery.progress();
        let event = Rc::new(event.map_err(|error| {
            Error::Parallel(
                authority
                    .submission_error(
                        error,
                        operation,
                        eredu_runtime::DistributedExecutionPhase::Execution,
                        None,
                    )
                    .to_string(),
            )
        })?);
        *recovery.retention().event.borrow_mut() = Some(Rc::clone(&event));
        check_native_status(&recovery).map_err(|error| {
            Error::Parallel(
                authority
                    .submission_error(
                        error,
                        operation,
                        eredu_runtime::DistributedExecutionPhase::Execution,
                        None,
                    )
                    .to_string(),
            )
        })?;
        #[cfg(test)]
        let force_pending = Rc::new(Cell::new(
            FORCE_NEXT_COMMUNICATION_PENDING.with(|force| force.replace(false)),
        ));
        Ok(Self {
            value,
            event,
            recovery: Rc::new(recovery),
            authority: Some(AuthorizedCompletion {
                authority,
                operation,
                phase: eredu_runtime::DistributedExecutionPhase::Execution,
            }),
            quarantined: Cell::new(false),
            #[cfg(test)]
            force_pending,
        })
    }

    fn completion_error(&self, error: safemlx::error::Exception) -> Error {
        match &self.authority {
            Some(context) => Error::Parallel(
                context
                    .authority
                    .completion_error(error, context.operation, context.phase, None)
                    .to_string(),
            ),
            None => Error::from(error),
        }
    }

    fn quarantine(&self) {
        if self.quarantined.replace(true) {
            return;
        }
        let work = DistributedCompletionOrphan {
            _event: self.event.clone(),
            recovery: Rc::clone(&self.recovery),
            #[cfg(test)]
            force_pending: self.force_pending.clone(),
        };
        safemlx::register_thread_runtime_housekeeping(reap_distributed_completion_orphans);
        DISTRIBUTED_COMPLETION_ORPHANS.with(|orphans| orphans.borrow_mut().work.push(work));
    }

    /// Returns the submitted value without waiting for its backend completion.
    ///
    /// Host access still requires synchronization. Work evaluated later on a
    /// different compatible stream must first call [`Self::wait_on`].
    pub const fn value(&self) -> &T {
        &self.value
    }

    /// Orders later evaluation on `stream` after this completion.
    ///
    /// Because MLX graphs are lazy, the consumer graph must be evaluated after
    /// this call. Constructing it before or after the call does not submit it.
    pub fn wait_on(&self, stream: &Stream) -> Result<(), Error> {
        check_native_status(&self.recovery).map_err(|error| self.completion_error(error))?;
        if self.quarantined.get() {
            return Err(Error::Parallel(
                "distributed completion is quarantined after a bounded timeout".into(),
            ));
        }
        self.ensure_authority_active()?;
        observe_native_child(
            self.recovery.retention(),
            Vec::new(),
            Some(stream.clone()),
            || self.event.wait_on(stream),
        )
        .map_err(|error| self.completion_error(error))?;
        self.ensure_authority_active()?;
        Ok(())
    }

    /// Returns whether the exact distributed operation has completed.
    pub fn is_complete(&self) -> Result<bool, Error> {
        safemlx::try_with_submission_retirement(|| {
            if !check_native_status(&self.recovery).map_err(|error| self.completion_error(error))? {
                return Ok(false);
            }
            self.ensure_authority_active()?;
            let complete = self
                .event
                .is_complete()
                .map_err(|error| self.completion_error(error))?;
            if complete {
                self.ensure_authority_active()?;
            }
            Ok(complete)
        })
        .unwrap_or(Ok(false))
    }

    /// Returns the backend which owns this exact completion.
    pub fn backend(&self) -> Result<EventBackend, Error> {
        Ok(self.event.backend()?)
    }

    /// Returns the arrays explicitly retained through exact completion.
    pub fn retained_resources(&self) -> usize {
        self.recovery.retention().arrays.len()
    }

    /// Returns retained count buffers, groups, routes, and streams in that order.
    #[cfg(test)]
    pub(crate) fn retained_native_resources(&self) -> (usize, usize, usize, usize) {
        (
            self.recovery.retention()._count_buffers.len(),
            self.recovery.retention().groups.len(),
            self.recovery.retention()._routes.len(),
            self.recovery.retention()._streams.len(),
        )
    }

    /// Blocks the host for this exact completion, not the remainder of a stream.
    pub fn synchronize(&self) -> Result<(), Error> {
        check_native_status(&self.recovery).map_err(|error| self.completion_error(error))?;
        self.ensure_authority_active()?;
        let Some(context) = &self.authority else {
            while !self.is_complete()? {
                std::thread::yield_now();
            }
            return Ok(());
        };
        let policy = context.authority.completion_policy().ok_or_else(|| {
            Error::Parallel("authorized distributed completion has no bounded policy".into())
        })?;
        let Some(deadline) = std::time::Instant::now().checked_add(policy.timeout()) else {
            self.quarantine();
            return Err(self.completion_error(safemlx::error::Exception::custom(
                "distributed completion deadline overflowed; live work was quarantined",
            )));
        };
        loop {
            #[cfg(test)]
            let complete = if self.force_pending.get() {
                safemlx::try_with_submission_retirement(|| {
                    check_native_status(&self.recovery)
                        .map(|_| false)
                        .map_err(|error| self.completion_error(error))
                })
                .unwrap_or(Ok(false))
            } else {
                self.is_complete()
            };
            #[cfg(not(test))]
            let complete = self.is_complete();
            match complete {
                Ok(true) => {
                    self.ensure_authority_active()?;
                    return Ok(());
                }
                Ok(false) if std::time::Instant::now() < deadline => std::thread::yield_now(),
                Ok(false) => {
                    self.quarantine();
                    return Err(self.completion_error(safemlx::error::Exception::custom(
                        "bounded distributed completion deadline exceeded; live work was quarantined",
                    )));
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Waits for exact completion and returns the owned result.
    pub fn into_value(self) -> Result<T, Error> {
        self.synchronize()?;
        Ok(self.value)
    }
}

impl<T> eredu_core::Completion for DistributedCompletion<T> {
    type Error = Error;

    fn resources_releasable(&self) -> bool {
        #[cfg(test)]
        if self.force_pending.get() {
            return false;
        }
        native_resources_releasable(&self.recovery)
    }

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.is_complete()
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.synchronize()
    }
}
