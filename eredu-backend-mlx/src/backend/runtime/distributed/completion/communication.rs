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
    pub(super) event: Rc<Event>,
    pub(super) recovery: Recovery<Rc<NativeResources>>,
    agreement: Option<FailureAgreementResolution>,
    words: Option<WordResolution>,
    boundary_headers: Vec<BoundaryHeaderResolution>,
    authority: Option<AuthorizedCommunicationCompletion>,
}

#[derive(Debug)]
struct AuthorizedCommunicationCompletion {
    authority: eredu_core::communication::CommunicationAuthority,
    operation: eredu_core::communication::CommunicationOperation,
}

impl Drop for MlxCommunicationCompletion {
    fn drop(&mut self) {}
}

#[derive(Debug)]
struct FailureAgreementResolution {
    output: Array,
    member_count: i32,
    descriptor: bool,
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

impl MlxFailureAgreement {
    /// One status word and two exact i32 bit-pattern halves per u64 field.
    pub(crate) const DESCRIPTOR_WORDS: usize = 17;

    pub(crate) fn descriptor_words(success: bool, descriptor: [u64; 8]) -> [i32; 17] {
        let mut words = [0; Self::DESCRIPTOR_WORDS];
        words[0] = i32::from(success);
        for (index, value) in descriptor.into_iter().enumerate() {
            words[1 + index * 2] = value as u32 as i32;
            words[2 + index * 2] = (value >> 32) as u32 as i32;
        }
        words
    }

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
            let finished = self.work[index].recovery.progress().settled;
            if finished {
                // Native terminal evidence, not an error return, permits release.
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
        // Hand unresolved owners to recovery. Its final owner-thread boundary
        // waits for terminal proof before releasing the retained resources.
        for work in self.work.drain(..) {
            drop(work);
        }
    }
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
    crate::backend::submission_recovery::reap();
    let unresolved = NATIVE_RESOURCE_OWNERS.with(|owners| {
        owners
            .borrow()
            .iter()
            .filter_map(std::rc::Weak::upgrade)
            .any(|resources| {
                resources.unavailable()
                    && resources
                        .groups
                        .iter()
                        .any(|retained| retained.shares_native_world(group))
            })
    });
    if unresolved {
        return Err(Error::Parallel(
            "native communicator has unresolved failed work".into(),
        ));
    }
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        orphans.reap();
        if orphans.work.iter().any(|work| {
            work.recovery
                .retention()
                .groups
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
        let mut recovery = Recovery::begin(NativeResources::new(
            arrays,
            count_buffers,
            groups,
            routes,
            streams,
        ))?;
        let event = async_eval_with_event(outputs.iter());
        recovery.seal();
        recovery.retention().host_failed.set(event.is_err());
        recovery.progress();
        let event = Rc::new(event?);
        *recovery.retention().event.borrow_mut() = Some(Rc::clone(&event));
        check_native_status(&recovery)?;
        Ok(Self {
            event,
            recovery,
            agreement: None,
            words: None,
            boundary_headers: Vec::new(),
            authority: None,
        })
    }

    /// Joins this exact completion to the selected session poison authority.
    pub(crate) fn with_authority(
        mut self,
        authority: eredu_core::communication::CommunicationAuthority,
        operation: eredu_core::communication::CommunicationOperation,
    ) -> Self {
        self.authority = Some(AuthorizedCommunicationCompletion {
            authority,
            operation,
        });
        self
    }

    fn fence_authority(&self) {
        if let Some(context) = &self.authority {
            context.authority.fence(context.operation, None);
        }
    }

    pub(crate) fn with_failure_agreement(
        mut self,
        output: Array,
        member_count: i32,
        descriptor: bool,
    ) -> (MlxFailureAgreement, Self) {
        let resolved = Rc::new(Cell::new(None));
        self.agreement = Some(FailureAgreementResolution {
            output,
            member_count,
            descriptor,
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
            let agreed = if agreement.descriptor {
                let count = agreement.member_count as usize;
                if counts.len() != count * MlxFailureAgreement::DESCRIPTOR_WORDS || count == 0 {
                    return Err(safemlx::error::Exception::custom(
                        "failure-agreement descriptor result has invalid member geometry",
                    ));
                }
                counts
                    .chunks_exact(MlxFailureAgreement::DESCRIPTOR_WORDS)
                    .all(|frame| {
                        frame[0] == 1
                            && frame[1..] == counts[1..MlxFailureAgreement::DESCRIPTOR_WORDS]
                    })
            } else {
                match counts {
                    [successes] => *successes == agreement.member_count,
                    _ => {
                        return Err(safemlx::error::Exception::custom(
                            "failure-agreement result is not one scalar status count",
                        ));
                    }
                }
            };
            agreement.resolved.set(Some(agreed));
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
                let mismatch = actual
                    .iter()
                    .zip(&header.expected)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(actual.len().min(header.expected.len()));
                return Err(safemlx::error::Exception::custom(format!(
                    "received boundary frame header differs from the selected route/schema/role contract at byte {mismatch}: received {:?}, expected {:?} (lengths {} / {})",
                    &actual[..actual.len().min(48)], &header.expected[..header.expected.len().min(48)],
                    actual.len(), header.expected.len(),
                )));
            }
        }
        Ok(())
    }

    fn observe_completion(&self) -> Result<bool, safemlx::error::Exception> {
        let arrays = self
            .agreement
            .iter()
            .map(|value| value.output.clone())
            .chain(self.words.iter().map(|value| value.output.clone()))
            .chain(
                self.boundary_headers
                    .iter()
                    .map(|value| value.received.clone()),
            )
            .collect();
        let (result, settled) =
            observe_native_child(self.recovery.retention(), arrays, None, || {
                self.event.try_with_complete(|| self.resolve_host_results())
            })?;
        Ok(result.is_some() && settled)
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
            let complete = eredu_core::Completion::is_complete(&self)?;
            if complete {
                // A completed query is authoritative and reports any retained
                // asynchronous error without a second lock-taking host wait.
                return Ok(eredu_core::BoundedCompletionOutcome::Completed);
            }
            if std::time::Instant::now() >= deadline {
                let selected = policy.cancellation();
                self.fence_authority();
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

    fn resources_releasable(&self) -> bool {
        native_resources_releasable(&self.recovery)
    }

    fn is_complete(&self) -> Result<bool, Self::Error> {
        safemlx::try_with_submission_retirement(|| {
            if !check_native_status(&self.recovery).inspect_err(|_| self.fence_authority())? {
                return Ok(false);
            }
            match self.observe_completion() {
                Ok(result) => Ok(result),
                Err(error) => {
                    self.fence_authority();
                    Err(error)
                }
            }
        })
        .unwrap_or(Ok(false))
    }

    fn wait(&self) -> Result<(), Self::Error> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}

/// Submits and host-synchronizes exactly the supplied output arrays.
pub fn synchronize_outputs<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
) -> safemlx::error::Result<()> {
    let outputs = outputs.into_iter().cloned().collect::<Vec<_>>();
    let completion = MlxCommunicationCompletion::submit(
        outputs.iter(),
        outputs.clone(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )?;
    eredu_core::Completion::wait(&completion)
}
