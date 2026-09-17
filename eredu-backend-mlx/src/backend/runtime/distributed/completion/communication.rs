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
    pub(super) event: NativeEvent,
    pub(super) recovery: CompletionRecovery,
    agreement: Option<FailureAgreementResolution>,
    flag: Option<FlagResolution>,
    words: Option<WordResolution>,
    boundary_headers: BoundaryHeaders,
    authority: Option<AuthorizedCommunicationCompletion>,
    #[cfg(test)]
    submitted_outputs: usize,
    #[cfg(test)]
    pub(super) force_pending: bool,
    #[cfg(test)]
    pub(super) teardown_observed: Option<Arc<AtomicBool>>,
    #[cfg(test)]
    pub(super) owner_exit_completion_observed: Option<Arc<AtomicBool>>,
    pub(super) consumers: Option<consumers::PreparedConsumers>,
    pub(super) orphan_destination: Option<destinations::Destination<Self>>,
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
    resolved: BoolResult,
}

#[derive(Debug)]
struct FlagResolution {
    output: Array,
    resolved: BoolResult,
}

#[derive(Debug)]
enum WordResolution {
    Signed { output:Array,resolved:WordsResult<i32> },
    Unsigned { output:Array,resolved:WordsResult<u32> },
}
impl WordResolution {
    fn output(&self)->&Array { match self { Self::Signed{output,..}|Self::Unsigned{output,..}=>output } }
}

#[derive(Debug)]
pub(super) struct BoundaryHeaderResolution {
    pub(super) received: Array,
    pub(super) expected: Vec<u8>,
}

/// Deferred host words for a manifest-consensus collective.
#[derive(Debug)]
pub(crate) struct MlxCommunicationWords {
    resolved: WordsResult,
}

impl MlxCommunicationWords {
    pub(crate) fn resolve(self) -> Result<Vec<i32>, safemlx::error::Exception> {
        self.resolved.take().ok_or_else(|| {
            safemlx::error::Exception::custom(
                "communication words were requested before exact completion",
            )
        })
    }
}

/// Deferred host result for a failure-agreement collective.
#[derive(Debug)]
pub struct MlxFailureAgreement {
    resolved: BoolResult,
}

/// Deferred host boolean resolved while the exact communication event is complete.
#[derive(Debug)]
pub(crate) struct MlxCommunicationFlag {
    resolved: BoolResult,
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
    pub(super) fn original(resolved:BoolResult)->Self {Self{resolved}}
    /// The shared resolver preserves the paid original source on premature
    /// access, while ordinary callers retain the native Exception path.
    pub(crate) fn resolve_neural(self)->Result<bool,Error> {
        if let Some(custody)=self.resolved.original_custody() {
            return self.resolved.get().ok_or_else(||prepared::error(prepared::Cause::Identity,custody));
        }
        self.resolve().map_err(Into::into)
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
    pub(super) work: destinations::Destinations<MlxCommunicationCompletion>,
}

impl CommunicationOrphanQuarantine {
    pub(super) fn reap(&mut self) {
        self.work.retain(|work| {
            let finished = work.recovery.settled();
            #[cfg(test)]
            let finished = finished && !work.force_pending;
            // Native terminal evidence, not an error return, permits release.
            !finished
        });
    }
}

impl Drop for CommunicationOrphanQuarantine {
    fn drop(&mut self) {
        self.reap();
        // Never wait at thread exit. Recovery retains unresolved native resources
        // independently; an inaccessible submitting thread is not terminal proof.
        #[cfg(test)]
        for work in self.work.iter() {
            if work.recovery.settled() {
                if let Some(observed) = &work.owner_exit_completion_observed {
                    observed.store(true, Ordering::Release);
                }
            }
        }
        self.work.clear();
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
            work._event.synchronize().unwrap();
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

fn quarantine(mut work: MlxCommunicationCompletion) {
    let destination = work.orphan_destination.take();
    if !work.recovery.is_original() {
        safemlx::register_thread_runtime_housekeeping(reap_communication_orphans);
    }
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        orphans.reap();
        match destination {
            Some(destination) => orphans.work.push_prepared(work, destination),
            None => orphans.work.push(work),
        }
    });
}

pub(super) fn reap_communication_orphans() {
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

/// Cold conservative source query: no reaping, event polling, native call,
/// registration or mutation. An unresolved orphan remains unavailable until the
/// existing ordinary/qualified completion worker actually retires it.
pub(crate) fn group_source_available(group: &Group) -> bool {
    let live = NATIVE_RESOURCE_OWNERS.try_with(|owners| {
        owners.try_borrow().map(|owners| !owners.iter().filter_map(std::rc::Weak::upgrade)
            .any(|resources| resources.unavailable() && resources.groups.iter()
                .any(|retained| retained.shares_native_world(group)))).unwrap_or(false)
    }).unwrap_or(false);
    live && COMMUNICATION_ORPHANS.try_with(|orphans| {
        orphans.try_borrow().map(|orphans| !orphans.work.iter().any(|work|
            work.recovery.retention().groups.iter().any(|retained|
                retained.shares_native_world(group)))).unwrap_or(false)
    }).unwrap_or(false)
}
pub(crate) fn group_source_controls() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [size_of::<&Group>(), size_of::<(bool, bool)>(),
        size_of::<std::cell::Ref<'_, destinations::Destinations<std::rc::Weak<NativeResources>>>>(),
        size_of::<Result<std::cell::Ref<'_, destinations::Destinations<std::rc::Weak<NativeResources>>>, std::cell::BorrowError>>(),
        size_of::<Option<Rc<NativeResources>>>(), size_of::<destinations::Iter<'_, std::rc::Weak<NativeResources>>>(),
        size_of::<std::cell::Ref<'_, CommunicationOrphanQuarantine>>(),
        size_of::<Result<std::cell::Ref<'_, CommunicationOrphanQuarantine>, std::cell::BorrowError>>(),
        size_of::<std::slice::Iter<'_, Group>>(),
        size_of::<Result<bool, std::thread::AccessError>>(),
    ];
    controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
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
        #[cfg(test)]
        let submitted_outputs = outputs.len();
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
        #[cfg(test)]
        let force_pending = FORCE_NEXT_COMMUNICATION_PENDING.with(|force| force.replace(false));
        Ok(Self {
            event: NativeEvent::Ordinary(event),
            recovery: CompletionRecovery::ordinary(recovery),
            agreement: None,
            flag: None,
            words: None,
            boundary_headers: BoundaryHeaders::default(),
            authority: None,
            consumers: None,
            orphan_destination: None,
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

    pub(super) fn from_original(event: NativeEvent, recovery: CompletionRecovery,
        destination: destinations::Destination<Self>, outputs: usize, consumers:Option<consumers::PreparedConsumers>) -> Self {
        #[cfg(not(test))] let _ = outputs;
        Self { event, recovery, agreement:None, flag:None, words:None,
            boundary_headers:BoundaryHeaders::default(), authority:None, consumers, orphan_destination:Some(destination),
            #[cfg(test)] submitted_outputs:outputs,
            #[cfg(test)] force_pending:false,
            #[cfg(test)] teardown_observed:None,
            #[cfg(test)] owner_exit_completion_observed:None }
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
        let resolved = BoolResult::ordinary();
        self.agreement = Some(FailureAgreementResolution {
            output,
            member_count,
            resolved: resolved.clone(),
        });
        (MlxFailureAgreement { resolved }, self)
    }

    pub(crate) fn with_i32_words(mut self, output: Array) -> (MlxCommunicationWords, Self) {
        let resolved = WordsResult::ordinary();
        self.words = Some(WordResolution::Signed {
            output,
            resolved: resolved.clone(),
        });
        (MlxCommunicationWords { resolved }, self)
    }

    pub(crate) fn with_f32_flag(mut self, output: Array) -> (MlxCommunicationFlag, Self) {
        let resolved = BoolResult::ordinary();
        self.flag = Some(FlagResolution {
            output,
            resolved: resolved.clone(),
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

    fn completed_readout<'a>(&self,output:&'a Array)->Result<safemlx::EvaluatedArray<'a>,safemlx::error::Exception> {
        match self.event.original_observer() {
            Some(observer)=>output.completed_in_original_scope(observer),
            None=>output.evaluated(),
        }
    }
    fn readout_error(&self,arguments:std::fmt::Arguments<'_>)->safemlx::error::Exception {
        match self.event.original_observer() {
            Some(observer)=>observer.invalid_input_error(),
            None=>safemlx::error::Exception::custom(arguments.to_string()),
        }
    }
    pub(super) fn install_original_scalar(&mut self,output:Array,result:BoolResult,kind:scalar::ScalarKind) {
        match kind {
            scalar::ScalarKind::Flag=>self.flag=Some(FlagResolution { output,resolved:result }),
            scalar::ScalarKind::Agreement(member_count)=>self.agreement=Some(FailureAgreementResolution { output,resolved:result,member_count }),
        }
    }

    pub(super) fn install_original_words(&mut self,output:Array,result:WordsResult) {
        self.words=Some(WordResolution::Signed { output,resolved:result });
    }
    pub(super) fn install_original_u32_words(&mut self,output:Array,result:WordsResult<u32>) {
        self.words=Some(WordResolution::Unsigned { output,resolved:result });
    }
    fn resolve_words<T:readouts::CommunicationWord>(&self,output:&Array,result:&WordsResult<T>)
        ->Result<(),safemlx::error::Exception> {
        let evaluated=self.completed_readout(output)?;
        let stored=readouts::store_words(&evaluated,result).map_err(|error| {
            self.readout_error(format_args!("communication word result differs from its prepared dtype: {error}"))
        })?;
        if !stored {
            return Err(self.readout_error(format_args!("communication word result differs from its prepared population")));
        }
        Ok(())
    }
    pub(super) fn install_original_header(&mut self,headers:BoundaryHeaders) { self.boundary_headers=headers; }

    fn resolve_host_results(&self) -> Result<(), safemlx::error::Exception> {
        if let Some(agreement) = &self.agreement {
            let evaluated = self.completed_readout(&agreement.output)?;
            let counts = evaluated.try_as_slice::<i32>().map_err(|error| {
                self.readout_error(format_args!("failure-agreement result is not an i32 status count: {error}"))
            })?;
            let agreed = match counts {
                [successes] => *successes == agreement.member_count,
                _ => {
                    return Err(self.readout_error(format_args!("failure-agreement result is not one scalar status count")));
                }
            };
            agreement.resolved.set(Some(agreed));
        }
        if let Some(flag) = &self.flag {
            let evaluated = self.completed_readout(&flag.output)?;
            let values = evaluated.try_as_slice::<f32>().map_err(|error| {
                self.readout_error(format_args!("communication flag result is not f32: {error}"))
            })?;
            let value = match values {
                [value] => *value != 0.0,
                _ => {
                    return Err(self.readout_error(format_args!("communication flag result is not one scalar")));
                }
            };
            flag.resolved.set(Some(value));
        }
        if let Some(words) = &self.words {
            match words {
                WordResolution::Signed{output,resolved}=>self.resolve_words(output,resolved)?,
                WordResolution::Unsigned{output,resolved}=>self.resolve_words(output,resolved)?,
            }
        }
        for header in &self.boundary_headers {
            let evaluated = self.completed_readout(&header.received)?;
            let actual = evaluated.try_as_slice::<u8>().map_err(|error| {
                self.readout_error(format_args!("received boundary frame header is not U8: {error}"))
            })?;
            if actual != header.expected {
                return Err(self.readout_error(format_args!("received boundary frame header differs from the selected route/schema/role contract")));
            }
        }
        Ok(())
    }

    fn observe_completion(&self) -> Result<bool, safemlx::error::Exception> {
        if self.event.original_observer().is_some() {
            // Only closed prepaid adapters attach original host readouts. Each
            // one borrows completed storage; none starts another evaluation.
            return self.event.try_with_complete(||self.resolve_host_results()).map(|ready|ready.is_some());
        }
        let arrays = self
            .agreement
            .iter()
            .map(|value| value.output.clone())
            .chain(self.flag.iter().map(|value| value.output.clone()))
            .chain(self.words.iter().map(|value| value.output().clone()))
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

    /// Number of explicitly retained array handles.
    #[cfg(test)]
    pub(crate) fn retained_arrays(&self) -> usize {
        self.recovery.retention().arrays.len()
    }

    /// Number of explicitly retained count buffers.
    #[cfg(test)]
    pub(crate) fn retained_count_buffers(&self) -> usize {
        self.recovery.retention()._count_buffers.len()
    }

    /// Number of explicitly retained group handles.
    #[cfg(test)]
    pub(crate) fn retained_groups(&self) -> usize {
        self.recovery.retention().groups.len()
    }

    /// Number of explicitly retained route handles.
    #[cfg(test)]
    pub(crate) fn retained_routes(&self) -> usize {
        self.recovery.retention()._routes.len()
    }

    /// Number of explicitly retained stream handles.
    #[cfg(test)]
    pub(crate) fn retained_streams(&self) -> usize {
        self.recovery.retention()._streams.len()
    }

    /// Number of native graph outputs certified by the exact event.
    #[cfg(test)]
    pub(crate) fn submitted_outputs(&self) -> usize {
        self.submitted_outputs
    }
}

impl MlxCommunicationCompletion {
    pub(super) fn wait_bounded_retaining(
        self,
        policy: eredu_core::BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedSubmissionOutcome<Self>, safemlx::error::Exception> {
        COMMUNICATION_ORPHANS.with(|orphans| orphans.borrow_mut().reap());
        let Some(deadline) = std::time::Instant::now().checked_add(policy.timeout()) else {
            let cause = self.event.refusal("bounded communication deadline exceeds the host monotonic clock range; live work was quarantined safely");
            quarantine(self);
            return Err(cause);
        };
        loop {
            #[cfg(test)]
            let complete = if self.force_pending {
                safemlx::try_with_submission_retirement(|| {
                    check_native_status(&self.recovery)
                        .inspect_err(|error| self.mark_authority_failure(error))
                        .map(|_| false)
                })
                .unwrap_or(Ok(false))?
            } else {
                eredu_core::Completion::is_complete(&self)?
            };
            #[cfg(not(test))]
            let complete = eredu_core::Completion::is_complete(&self)?;
            if complete {
                // A completed query is authoritative and reports any retained
                // asynchronous error without a second lock-taking host wait.
                return Ok(eredu_core::BoundedSubmissionOutcome::Completed(self));
            }
            if std::time::Instant::now() >= deadline {
                let selected = policy.cancellation();
                self.mark_authority_failure("bounded communication deadline exceeded");
                let cause = (selected != eredu_core::CompletionCancellationMode::QuarantineUntilComplete).then(||
                    self.event.refusal("MLX communication has no native cancellation; timed-out work was quarantined safely"));
                quarantine(self);
                if let Some(cause) = cause { return Err(cause); }
                return Ok(eredu_core::BoundedSubmissionOutcome::DeadlineExceeded {
                    cancellation: eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                });
            }
            std::thread::yield_now();
        }
    }
}

impl eredu_core::BoundedCompletion for MlxCommunicationCompletion {
    fn wait_bounded(self, policy: eredu_core::BoundedCompletionWait)
        -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
        match self.wait_bounded_retaining(policy)? {
            eredu_core::BoundedSubmissionOutcome::Completed(completion) => {
                drop(completion);
                Ok(eredu_core::BoundedCompletionOutcome::Completed)
            }
            eredu_core::BoundedSubmissionOutcome::DeadlineExceeded { cancellation } =>
                Ok(eredu_core::BoundedCompletionOutcome::DeadlineExceeded { cancellation }),
        }
    }
}

impl eredu_core::Completion for MlxCommunicationCompletion {
    type Error = safemlx::error::Exception;

    fn resources_releasable(&self) -> bool {
        #[cfg(test)]
        if self.force_pending {
            return false;
        }
        if let Some(consumers)=&self.consumers {
            let Some(observer)=self.event.original_observer() else { return false; };
            if !consumers.retire_completed(observer).unwrap_or(false) { return false; }
        }
        native_resources_releasable(&self.recovery)
    }

    fn is_complete(&self) -> Result<bool, Self::Error> {
        safemlx::try_with_submission_retirement(|| {
            if !check_native_status(&self.recovery)
                .inspect_err(|error| self.mark_authority_failure(error))?
            {
                return Ok(false);
            }
            match self.observe_completion() {
                Ok(result) => Ok(result),
                Err(error) => {
                    self.mark_authority_failure(&error);
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
