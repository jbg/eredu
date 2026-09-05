use super::*;

/// Exact completion for one fine-grained communication submission.
///
/// Unlike [`DistributedCompletion`], this completion explicitly owns every
/// array, count buffer, group, route, and stream borrowed while the lazy MLX
/// communication graph was constructed. It is intentionally thread-affine:
/// native MLX groups, streams, and events are not `Send` or `Sync`.
#[derive(Debug)]
#[must_use = "communication work has been submitted; retain or wait on its completion"]
pub struct MlxCommunicationCompletion {
    pub(super) event: Event,
    _arrays: Vec<Array>,
    _count_buffers: Vec<Vec<usize>>,
    _groups: Vec<Group>,
    _routes: Vec<CommunicationRouteRealization>,
    _streams: Vec<Stream>,
    agreement: Option<FailureAgreementResolution>,
    flag: Option<FlagResolution>,
    words: Option<WordResolution>,
    boundary_headers: Vec<BoundaryHeaderResolution>,
    authority: Option<AuthorizedCommunicationCompletion>,
    #[cfg(test)]
    submitted_outputs: usize,
    #[cfg(test)]
    pub(super) force_pending: bool,
    #[cfg(test)]
    pub(super) teardown_observed: Option<Arc<AtomicBool>>,
    #[cfg(test)]
    pub(super) owner_exit_completion_observed: Option<Arc<AtomicBool>>,
}

#[derive(Debug)]
struct AuthorizedCommunicationCompletion {
    authority: eredu_runtime::PartitionCommunicationAuthority,
    operation: eredu_runtime::CommunicationOperation,
    phase: eredu_runtime::DistributedExecutionPhase,
}

impl Drop for MlxCommunicationCompletion {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(observed) = &self.teardown_observed {
            observed.store(true, Ordering::Release);
        }
    }
}

#[derive(Debug)]
struct FailureAgreementResolution {
    output: Array,
    member_count: i32,
    resolved: Rc<Cell<Option<bool>>>,
}

#[derive(Debug)]
struct FlagResolution {
    output: Array,
    resolved: Rc<Cell<Option<bool>>>,
}

#[derive(Debug)]
struct WordResolution {
    output: Array,
    resolved: Rc<RefCell<Option<Vec<i32>>>>,
}

#[derive(Debug)]
struct BoundaryHeaderResolution {
    received: Array,
    expected: Vec<u8>,
}

/// Deferred host words for a manifest-consensus collective.
#[derive(Debug)]
pub(crate) struct MlxCommunicationWords {
    resolved: Rc<RefCell<Option<Vec<i32>>>>,
}

impl MlxCommunicationWords {
    pub(crate) fn resolve(self) -> Result<Vec<i32>, safemlx::error::Exception> {
        self.resolved.borrow_mut().take().ok_or_else(|| {
            safemlx::error::Exception::custom(
                "communication words were requested before exact completion",
            )
        })
    }
}

/// Deferred host result for a failure-agreement collective.
#[derive(Debug)]
pub struct MlxFailureAgreement {
    resolved: Rc<Cell<Option<bool>>>,
}

/// Deferred host boolean resolved while the exact communication event is complete.
#[derive(Debug)]
pub(crate) struct MlxCommunicationFlag {
    resolved: Rc<Cell<Option<bool>>>,
}

impl MlxCommunicationFlag {
    pub(crate) fn resolve(self) -> Result<bool, safemlx::error::Exception> {
        self.resolved.get().ok_or_else(|| {
            safemlx::error::Exception::custom(
                "communication flag was requested before exact completion",
            )
        })
    }
}

impl MlxFailureAgreement {
    pub(crate) fn resolve(self) -> Result<bool, safemlx::error::Exception> {
        self.resolved.get().ok_or_else(|| {
            safemlx::error::Exception::custom(
                "failure-agreement result was requested before exact completion",
            )
        })
    }
}

#[derive(Debug, Default)]
pub(super) struct CommunicationOrphanQuarantine {
    pub(super) work: Vec<MlxCommunicationCompletion>,
}

impl CommunicationOrphanQuarantine {
    pub(super) fn reap(&mut self) {
        let mut index = 0;
        while index < self.work.len() {
            let finished = match self.work[index].event.try_is_complete() {
                Ok(Some(complete)) => complete,
                Ok(None) => false,
                // An event query reports an asynchronous failure only after the
                // backend has resolved that event, so its retained work is releasable.
                Err(_) => true,
            };
            #[cfg(test)]
            let finished = finished && !self.work[index].force_pending;
            if finished {
                // Exact completion (or a terminal asynchronous error) was observed,
                // so explicitly releasing the retained owner is now safe.
                drop(self.work.swap_remove(index));
            } else {
                index += 1;
            }
        }
    }
}

impl Drop for CommunicationOrphanQuarantine {
    fn drop(&mut self) {
        self.reap();
        // MLX does not expose native event abort and Event is thread-affine. The
        // originating thread therefore remains the deterministic owner: at thread
        // exit it waits for each outstanding event to reach completion (or a
        // terminal asynchronous error) before releasing every retained native
        // resource. Nothing is transferred to a foreign reaper or leaked.
        for work in self.work.drain(..) {
            let _ = work.event.synchronize();
            #[cfg(test)]
            if let Some(observed) = &work.owner_exit_completion_observed {
                observed.store(true, Ordering::Release);
            }
            drop(work);
        }
    }
}

#[cfg(test)]
pub(crate) fn force_next_communication_pending() {
    FORCE_NEXT_COMMUNICATION_PENDING.with(|force| force.set(true));
}

#[cfg(test)]
pub(crate) fn distributed_completion_orphan_count() -> usize {
    DISTRIBUTED_COMPLETION_ORPHANS.with(|orphans| orphans.borrow().work.len())
}

#[cfg(test)]
pub(crate) fn release_forced_pending_orphans() {
    DISTRIBUTED_COMPLETION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        for work in &mut orphans.work {
            work.force_pending.set(false);
            work.event.synchronize().unwrap();
        }
        orphans.reap();
    });
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        for work in &mut orphans.work {
            work.force_pending = false;
            work.event.synchronize().unwrap();
        }
        orphans.reap();
    });
}

fn quarantine(work: MlxCommunicationCompletion) {
    safemlx::register_thread_runtime_housekeeping(reap_communication_orphans);
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        orphans.reap();
        orphans.work.push(work);
    });
}

fn reap_communication_orphans() {
    let empty = COMMUNICATION_ORPHANS.try_with(|orphans| {
        if let Ok(mut orphans) = orphans.try_borrow_mut() {
            orphans.reap();
            return orphans.work.is_empty();
        }
        false
    });
    if matches!(empty, Ok(true)) {
        safemlx::unregister_thread_runtime_housekeeping(reap_communication_orphans);
    }
}

pub(crate) fn ensure_group_available(group: &Group) -> Result<(), Error> {
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        orphans.reap();
        if orphans.work.iter().any(|work| {
            work._groups
                .iter()
                .any(|retained| retained.shares_native_world(group))
        }) {
            Err(Error::Parallel(
                "native communicator is quarantined after a bounded communication timeout".into(),
            ))
        } else {
            Ok(())
        }
    })
}

impl MlxCommunicationCompletion {
    /// Submits exactly `outputs` and retains every supplied native resource.
    pub(crate) fn submit<'a>(
        outputs: impl IntoIterator<Item = &'a Array>,
        arrays: Vec<Array>,
        count_buffers: Vec<Vec<usize>>,
        groups: Vec<Group>,
        routes: Vec<CommunicationRouteRealization>,
        streams: Vec<Stream>,
    ) -> Result<Self, safemlx::error::Exception> {
        let outputs = outputs.into_iter().cloned().collect::<Vec<_>>();
        #[cfg(test)]
        let submitted_outputs = outputs.len();
        let event = async_eval_with_event(outputs.iter())?;
        #[cfg(test)]
        let force_pending = FORCE_NEXT_COMMUNICATION_PENDING.with(|force| force.replace(false));
        Ok(Self {
            event,
            _arrays: arrays,
            _count_buffers: count_buffers,
            _groups: groups,
            _routes: routes,
            _streams: streams,
            agreement: None,
            flag: None,
            words: None,
            boundary_headers: Vec::new(),
            authority: None,
            #[cfg(test)]
            submitted_outputs,
            #[cfg(test)]
            force_pending,
            #[cfg(test)]
            teardown_observed: None,
            #[cfg(test)]
            owner_exit_completion_observed: None,
        })
    }

    /// Joins this exact completion to the selected session poison authority.
    pub(crate) fn with_authority(
        mut self,
        authority: eredu_runtime::PartitionCommunicationAuthority,
        operation: eredu_runtime::CommunicationOperation,
        phase: eredu_runtime::DistributedExecutionPhase,
    ) -> Self {
        self.authority = Some(AuthorizedCommunicationCompletion {
            authority,
            operation,
            phase,
        });
        self
    }

    fn mark_authority_failure(&self, error: impl std::fmt::Display) {
        if let Some(context) = &self.authority {
            let _ =
                context
                    .authority
                    .completion_error(error, context.operation, context.phase, None);
        }
    }

    pub(crate) fn with_failure_agreement(
        mut self,
        output: Array,
        member_count: i32,
    ) -> (MlxFailureAgreement, Self) {
        let resolved = Rc::new(Cell::new(None));
        self.agreement = Some(FailureAgreementResolution {
            output,
            member_count,
            resolved: Rc::clone(&resolved),
        });
        (MlxFailureAgreement { resolved }, self)
    }

    pub(crate) fn with_i32_words(mut self, output: Array) -> (MlxCommunicationWords, Self) {
        let resolved = Rc::new(RefCell::new(None));
        self.words = Some(WordResolution {
            output,
            resolved: Rc::clone(&resolved),
        });
        (MlxCommunicationWords { resolved }, self)
    }

    pub(crate) fn with_f32_flag(mut self, output: Array) -> (MlxCommunicationFlag, Self) {
        let resolved = Rc::new(Cell::new(None));
        self.flag = Some(FlagResolution {
            output,
            resolved: Rc::clone(&resolved),
        });
        (MlxCommunicationFlag { resolved }, self)
    }

    /// Requires exact bytes sliced from received in-band boundary frames.
    pub(crate) fn with_boundary_headers(
        mut self,
        headers: impl IntoIterator<Item = (Array, Vec<u8>)>,
    ) -> Self {
        self.boundary_headers.extend(
            headers
                .into_iter()
                .map(|(received, expected)| BoundaryHeaderResolution { received, expected }),
        );
        self
    }

    fn resolve_host_results(&self) -> Result<(), safemlx::error::Exception> {
        if let Some(agreement) = &self.agreement {
            let evaluated = agreement.output.evaluated()?;
            let counts = evaluated.try_as_slice::<i32>().map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "failure-agreement result is not an i32 status count: {error}"
                ))
            })?;
            let agreed = match counts {
                [successes] => *successes == agreement.member_count,
                _ => {
                    return Err(safemlx::error::Exception::custom(
                        "failure-agreement result is not one scalar status count",
                    ))
                }
            };
            agreement.resolved.set(Some(agreed));
        }
        if let Some(flag) = &self.flag {
            let evaluated = flag.output.evaluated()?;
            let values = evaluated.try_as_slice::<f32>().map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "communication flag result is not f32: {error}"
                ))
            })?;
            let value = match values {
                [value] => *value != 0.0,
                _ => {
                    return Err(safemlx::error::Exception::custom(
                        "communication flag result is not one scalar",
                    ))
                }
            };
            flag.resolved.set(Some(value));
        }
        if let Some(words) = &self.words {
            let evaluated = words.output.evaluated()?;
            let values = evaluated.try_as_slice::<i32>().map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "communication word result is not i32: {error}"
                ))
            })?;
            *words.resolved.borrow_mut() = Some(values.to_vec());
        }
        for header in &self.boundary_headers {
            let evaluated = header.received.evaluated()?;
            let actual = evaluated.try_as_slice::<u8>().map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "received boundary frame header is not U8: {error}"
                ))
            })?;
            if actual != header.expected {
                return Err(safemlx::error::Exception::custom(
                    "received boundary frame header differs from the selected route/schema/role contract",
                ));
            }
        }
        Ok(())
    }

    /// Number of explicitly retained array handles.
    #[cfg(test)]
    pub(crate) fn retained_arrays(&self) -> usize {
        self._arrays.len()
    }

    /// Number of explicitly retained count buffers.
    #[cfg(test)]
    pub(crate) fn retained_count_buffers(&self) -> usize {
        self._count_buffers.len()
    }

    /// Number of explicitly retained group handles.
    #[cfg(test)]
    pub(crate) fn retained_groups(&self) -> usize {
        self._groups.len()
    }

    /// Number of explicitly retained route handles.
    #[cfg(test)]
    pub(crate) fn retained_routes(&self) -> usize {
        self._routes.len()
    }

    /// Number of explicitly retained stream handles.
    #[cfg(test)]
    pub(crate) fn retained_streams(&self) -> usize {
        self._streams.len()
    }

    /// Number of native graph outputs certified by the exact event.
    #[cfg(test)]
    pub(crate) fn submitted_outputs(&self) -> usize {
        self.submitted_outputs
    }
}

impl eredu_core::BoundedCompletion for MlxCommunicationCompletion {
    fn wait_bounded(
        self,
        policy: eredu_core::BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
        COMMUNICATION_ORPHANS.with(|orphans| orphans.borrow_mut().reap());
        let Some(deadline) = std::time::Instant::now().checked_add(policy.timeout()) else {
            quarantine(self);
            return Err(safemlx::error::Exception::custom(
                "bounded communication deadline exceeds the host monotonic clock range; live work was quarantined safely",
            ));
        };
        loop {
            #[cfg(test)]
            let complete = !self.force_pending
                && match self.event.try_with_complete(|| self.resolve_host_results()) {
                    Ok(result) => result.is_some(),
                    Err(error) => {
                        self.mark_authority_failure(&error);
                        return Err(error);
                    }
                };
            #[cfg(not(test))]
            let complete = match self.event.try_with_complete(|| self.resolve_host_results()) {
                Ok(result) => result.is_some(),
                Err(error) => {
                    self.mark_authority_failure(&error);
                    return Err(error);
                }
            };
            if complete {
                // A completed query is authoritative and reports any retained
                // asynchronous error without a second lock-taking host wait.
                return Ok(eredu_core::BoundedCompletionOutcome::Completed);
            }
            if std::time::Instant::now() >= deadline {
                let selected = policy.cancellation();
                self.mark_authority_failure("bounded communication deadline exceeded");
                quarantine(self);
                if selected != eredu_core::CompletionCancellationMode::QuarantineUntilComplete {
                    return Err(safemlx::error::Exception::custom(
                        "MLX communication has no native cancellation; timed-out work was quarantined safely",
                    ));
                }
                return Ok(eredu_core::BoundedCompletionOutcome::DeadlineExceeded {
                    cancellation: eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                });
            }
            std::thread::yield_now();
        }
    }
}

impl eredu_core::Completion for MlxCommunicationCompletion {
    type Error = safemlx::error::Exception;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        match self.event.try_with_complete(|| self.resolve_host_results()) {
            Ok(result) => Ok(result.is_some()),
            Err(error) => {
                self.mark_authority_failure(&error);
                Err(error)
            }
        }
    }

    fn wait(&self) -> Result<(), Self::Error> {
        let result = self
            .event
            .synchronize()
            .and_then(|()| self.resolve_host_results());
        if let Err(error) = &result {
            self.mark_authority_failure(error);
        }
        result
    }
}

/// Submits and host-synchronizes exactly the supplied output arrays.
pub fn synchronize_outputs<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
) -> safemlx::error::Result<()> {
    async_eval_with_event(outputs)?.synchronize()
}
