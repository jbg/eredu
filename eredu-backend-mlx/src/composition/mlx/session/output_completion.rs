use super::*;
use super::{
    model_session::{
        CompletionRootsOwner, ObservationRoots, ScopeRetention, SubmissionResourcesOwner,
    },
    recovery::{Probe, Recovery, Retention, Status},
};
use eredu_runtime::working_memory::{
    InferenceStateRevision, InferenceTextStepReceipt, WorkingMemoryError,
};
use std::cell::RefCell;

mod observation;
pub(in crate::composition::mlx::session) mod scalar;
pub(in crate::composition::mlx::session) use observation::token_control_bytes as token_observation_control_bytes;
pub(super) use observation::Observation;
use observation::TokenObservationOwner;
pub(in crate::composition::mlx::session) use scalar::control_bytes as token_scalar_control_bytes;

pub(in crate::composition::mlx::session) fn finish_control_bytes() -> Option<u64> {
    use std::mem::size_of;
    use crate::backend::runtime::execution::generic::RegisteredScopeRetirementCause;
    let parts = [
        size_of::<&MlxTextCompletion>(), size_of::<&SubmissionResourcesOwner>(),
        size_of::<Option<Result<Status, RegisteredScopeRetirementCause>>>(),
        size_of::<Result<Option<Status>, RegisteredScopeRetirementCause>>(),
        size_of::<Result<Option<Status>, Error>>(),
    ];
    u64::try_from(parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)?).ok()
}

// Return the extracted owner only after the slot loan ends. Finalization and
// Drop may reenter the completion; they must never run under its RefMut.
fn take_recovery<T: Retention, P: Probe>(
    slot: &RefCell<Option<Recovery<T, P>>>,
) -> Option<Recovery<T, P>> {
    slot.borrow_mut().take()
}

// The owning recovery is detached for progress as well as final retirement.
// Its callbacks may reenter the completion; the caller's observation gate
// rejects that reentry while this slot temporarily has no owner.
fn progress_recovery<T: Retention, P: Probe>(
    slot: &RefCell<Option<Recovery<T, P>>>,
) -> Option<super::recovery::Status> {
    let recovery = take_recovery(slot);
    let status = recovery.as_ref().map(Recovery::progress);
    let previous = slot.replace(recovery);
    debug_assert!(previous.is_none());
    drop(previous);
    status
}

// Cached output and owner health are Rust-side facts. Replaying them does not
// require the native retirement lock, even when another thread holds it.
fn cached_observation<T: Copy>(
    observation: &observation::Loan<'_, T>,
    owner: &SubmissionResourcesOwner,
) -> Option<Result<T, Error>> {
    observation.cached().map(|result| match result {
        Ok(value) => match owner.ensure_healthy() {
            Ok(()) => Ok(value),
            Err(error) => observation.complete(Err(error)),
        },
        Err(error) => Err(error),
    })
}

/// Backend-owned output of one MLX model-session submission.
///
/// Pipeline ranks that do not own the output projection complete with no local
/// logits. The final rank and every non-pipeline session complete with logits.
#[derive(Debug, Clone)]
pub struct MlxModelOutput {
    logits: Option<MlxTensor>,
}

impl MlxModelOutput {
    pub(super) const fn new(logits: Option<MlxTensor>) -> Self {
        Self { logits }
    }

    /// Borrows local logits when this rank owns them.
    pub const fn logits(&self) -> Option<&MlxTensor> {
        self.logits.as_ref()
    }

    /// Consumes the output and returns local logits when present.
    pub fn into_logits(self) -> Option<MlxTensor> {
        self.logits
    }
}

/// Opaque exact completion for any MLX model-session submission.
pub struct MlxSessionCompletion {
    pub(super) inner: MlxSessionCompletionKind,
}

/// MLX token handle yielded by backend-generic text generation.
#[derive(Clone)]
pub struct MlxTextToken {
    pub(super) value: Array,
    pub(super) stream: Stream,
    // Exact output evidence only. Clones share the original runtime authority;
    // neither token values nor retained request charges can create a receipt.
    step_receipt: Option<InferenceTextStepReceipt>,
    // Snapshot the producing operation's resulting branch, independently of
    // any later state change or request retention on the shared owner.
    state_revision: Option<InferenceStateRevision>,
    observation: TokenObservationOwner,
    pub(super) owner: SubmissionResourcesOwner,
}

impl MlxTextToken {
    pub(super) fn new_with_sampling_source(
        value: Array,
        stream: Stream,
        owner: SubmissionResourcesOwner,
        role: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
        host: Option<eredu_core::HostPreparationAuthority>,
        source: Option<crate::backend::SamplingEventSource>,
    ) -> Self {
        Self::new_with_sources(value, stream, owner, role, host, source)
    }

    pub(super) fn new(value: Array, stream: Stream, owner: SubmissionResourcesOwner) -> Self {
        // Ordinary/copied tokens keep fresh observation state and acquire no
        // original role from their retained producing owner.
        Self::new_with_scalar_scope(value, stream, owner, None)
    }

    pub(super) fn new_with_scalar_scope(
        value: Array,
        stream: Stream,
        owner: SubmissionResourcesOwner,
        role: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
    ) -> Self {
        Self::new_with_scalar_scope_and_capture(value, stream, owner, role, None)
    }
    pub(super) fn new_with_scalar_scope_and_capture(
        value: Array,
        stream: Stream,
        owner: SubmissionResourcesOwner,
        role: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
        host: Option<eredu_core::HostPreparationAuthority>,
    ) -> Self {
        Self::new_with_sources(value, stream, owner, role, host, None)
    }
    fn new_with_sources(
        value: Array,
        stream: Stream,
        owner: SubmissionResourcesOwner,
        role: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
        host: Option<eredu_core::HostPreparationAuthority>,
        source: Option<crate::backend::SamplingEventSource>,
    ) -> Self {
        let state_revision = owner.state_revision();
        Self {
            value,
            stream,
            step_receipt: None,
            state_revision,
            observation: TokenObservationOwner::new_with_sources(owner.clone(), role, host, source),
            owner,
        }
    }

    pub(super) fn ordinary_error_custody(&self) -> Option<&eredu_core::HostPreparationAuthority> {
        self.observation.value.ordinary_error_custody()
    }
    pub(super) fn step_receipt(&self) -> Option<&InferenceTextStepReceipt> {
        self.step_receipt.as_ref()
    }

    pub(super) fn state_revision(&self) -> Option<&InferenceStateRevision> {
        self.state_revision.as_ref()
    }

    pub(super) fn apply_branch_placement(&mut self, placement: &eredu_runtime::replicated_session::ControlBranchPlacement) {
        assert!(self.state_revision.as_ref().is_some_and(|revision| placement.matches_source(revision)),
            "closed branch source was validated before its atomic exchange");
        self.state_revision = Some(placement.revision().clone());
    }

    pub(super) fn attach_step_receipt(
        &mut self,
        receipt: InferenceTextStepReceipt,
    ) -> Result<(), WorkingMemoryError> {
        if self.step_receipt.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.step_receipt = Some(receipt);
        Ok(())
    }
}

impl TokenOutput for MlxTextToken {
    type Error = Error;

    fn token_id(&self) -> Result<u32, Self::Error> {
        if self.observation.original_controls.is_some() {
            return scalar::token_id(&self.value, &self.stream, &self.owner, &self.observation);
        }
        let observation = self.observation.value.enter()?;
        if let Some(result) = cached_observation(&observation, &self.owner) {
            return result;
        }
        // Only this first attempt may convert/evaluate the exact scalar. Token
        // clones share its state; a copied token constructor gets fresh state.
        let result = (|| {
            self.owner.ensure_healthy()?;
            let _unwind = self.owner.poison_on_unwind();
            let role = self.observation.take_scope()?;
            read_ordinary_token_scalar(&self.value, &self.stream, &self.owner, role)
        })();
        observation.complete(result)
    }
}

// Shared ordinary conversion/evaluation worker. Both callers own the observation
// loan, health check and unwind guard before transferring the same optional role.
fn read_ordinary_token_scalar(
    value: &Array,
    stream: &Stream,
    owner: &SubmissionResourcesOwner,
    role: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
) -> Result<u32, Error> {
    #[cfg(test)]
    crate::backend::submission_recovery::prediction::test_counts::record(4, role.is_some());
    let mut recovery = owner.observation_recovery_with_prediction(
        ObservationRoots::Scalar {
            _array: value.clone(),
        },
        role,
    )?;
    let result = value.clone().try_item::<u32>(stream).map_err(Error::from);
    recovery.seal();
    // Keep the first successful read's exact finish boundary. Runtime
    // contention is pending ownership; native failures retain recovery.
    let status = if result.is_ok() {
        recovery.finish().map_err(|cause| { owner.reject_unresolved(); cause.into_error() })?
    } else {
        recovery.progress()
    };
    if !status.settled || status.failed || status.blocked {
        owner.reject_unresolved();
        return match result {
            Err(error) => Err(error),
            Ok(_) => Err(Error::ArchitectureModel(
                "native token read is unresolved or failed".into(),
            )),
        };
    }
    if result.is_err() {
        owner.reject_unresolved();
    }
    result
}

/// Exact completion retaining both model execution and sampled token output.
pub struct MlxTextCompletion {
    // Sampling scope tickets retain authority independently of field drop order.
    pub(super) token: MlxCompletion,
    pub(super) model: MlxSessionCompletion,
    pub(super) recovery: RefCell<Option<Recovery<ScopeRetention>>>,
    pub(super) observation: Observation<bool>,
}

impl Completion for MlxTextCompletion {
    type Error = Error;

    fn resources_releasable(&self) -> bool {
        let Ok(_loan) = self.observation.enter() else {
            return false;
        };
        safemlx::try_with_submission_retirement(|| {
            let settled = progress_recovery(&self.recovery).is_none_or(|scope| scope.settled);
            if settled {
                let retired = take_recovery(&self.recovery);
                drop(retired);
            }
            let token = self.token.resources_releasable();
            let model = self.model.resources_releasable();
            settled && token && model
        })
        .unwrap_or(false)
    }

    fn is_complete(&self) -> Result<bool, Self::Error> {
        // This method accepts no arbitrary callback. It queries only the exact
        // submitted token/model roots and the already-existing sampling scope.
        let observation = self.observation.enter()?;
        if let Some(result) = cached_observation(&observation, self.model.owner()) {
            return result;
        }
        safemlx::try_with_submission_retirement(|| self.observe(false, &observation))
            .unwrap_or(Ok(false))
    }

    fn wait(&self) -> Result<(), Self::Error> {
        let observation = self.observation.enter()?;
        if let Some(result) = cached_observation(&observation, self.model.owner()) {
            return result.map(|_| ());
        }
        self.observe(true, &observation).and_then(|complete| {
            if complete {
                Ok(())
            } else {
                observation
                    .complete(Err(Error::ArchitectureModel(
                        "sampled output still has unresolved native work".into(),
                    )))
                    .map(|_| ())
            }
        })
    }
}

impl MlxTextCompletion {
    fn observe(
        &self,
        wait: bool,
        observation: &observation::Loan<'_, bool>,
    ) -> Result<bool, Error> {
        let result = (|| {
            self.model.owner().ensure_healthy()?;
            let _unwind = self.model.owner().poison_on_unwind();
            // These fixed operations only inspect submitted event roots. Model
            // finalization owns its separate one-shot validation observation.
            // There is no outer Scope per poll and no callback extension point.
            let result = if wait {
                token_then_model_wait(&self.token, &self.model).map(|()| true)
            } else {
                token_then_model_is_complete(&self.token, &self.model)
            };
            let sampling = if wait && matches!(result, Ok(true)) {
                let sampling = take_recovery(&self.recovery);
                sampling.map(Recovery::finish).transpose().map_err(|cause| {
                    self.model.owner().reject_unresolved();
                    cause.into_error()
                })?
            } else {
                progress_recovery(&self.recovery)
            };
            let sampling_settled = sampling.is_none_or(|status| status.settled);
            if sampling_settled && !matches!(result, Ok(false)) {
                let retired = take_recovery(&self.recovery);
                drop(retired);
            }
            if sampling.is_some_and(|status| status.failed || status.blocked) {
                self.model.owner().reject_unresolved();
                return match result {
                    Err(error) => Err(error),
                    Ok(_) => Err(Error::ArchitectureModel(
                        "native sampled output failed or is unobservable".into(),
                    )),
                };
            }
            if result.is_err() {
                self.model.owner().reject_unresolved();
            }
            result.and_then(|complete| {
                // Retiring the last sampling scope can publish model storage.
                // Surface that callback's failure before caching completion or
                // allowing the machine to submit its next decode.
                self.model.owner().ensure_healthy()?;
                Ok(complete && sampling_settled)
            })
        })();
        if matches!(result, Ok(false)) {
            result // Pending queries leave their existing owners in place.
        } else {
            observation.complete(result)
        }
    }
}

// Keep native token-event observation ahead of model authority resolution.
// The generic completion parameter permits deterministic pending/error tests
// without timing-dependent accelerator events; production remains monomorphized.
pub(super) fn token_then_model_is_complete(
    token: &impl Completion<Error = Error>,
    model: &MlxSessionCompletion,
) -> Result<bool, Error> {
    match token.is_complete() {
        Ok(false) => Ok(false),
        Ok(true) => model.is_complete(),
        Err(error) => Err(error),
    }
}

pub(super) fn token_then_model_wait(
    token: &impl Completion<Error = Error>,
    model: &MlxSessionCompletion,
) -> Result<(), Error> {
    match token.wait() {
        Ok(()) => model.wait(),
        Err(error) => {
            // A failure permits model resolution only with independent native
            // resource evidence. Never start an unbounded cleanup wait here.
            if token.resources_releasable() {
                let _ = model.is_complete();
            }
            Err(error)
        }
    }
}

pub(super) enum MlxSessionCompletionKind {
    Model {
        roots: CompletionRootsOwner,
        owner: SubmissionResourcesOwner,
        recovery: RefCell<Option<Recovery<ScopeRetention>>>,
        observation: Observation<()>,
        _funding_retirement: Option<super::model_session::text_funding::RetireFundedCompletion>,
    },
}

impl MlxSessionCompletion {
    pub(super) fn retain_ordinary_capture(
        &mut self,
        host: Option<eredu_core::HostPreparationAuthority>,
    ) {
        let MlxSessionCompletionKind::Model { observation, .. } = &mut self.inner;
        observation.retain_ordinary_capture(host);
    }
    pub(super) fn owner(&self) -> &SubmissionResourcesOwner {
        match &self.inner {
            MlxSessionCompletionKind::Model { owner, .. } => owner,
        }
    }

    // Called only while the local observation gate is held, after its cached
    // result and original execution status have been checked.
    fn resolve(&self) -> Result<(), Error> {
        self.owner().ensure_healthy()?;
        let MlxSessionCompletionKind::Model {
            roots,
            owner,
            recovery,
            ..
        } = &self.inner;
        // Preserve independent ownership before taking execution recovery.
        // This native validation scope is constructed at most once per result.
        let _unwind = owner.poison_on_unwind();
        let retirement = owner.model_retirement_observer().map_err(|cause| {
            owner.reject_unresolved();
            cause
        })?;
        let role = owner.take_model_validation_scope()?;
        #[cfg(test)]
        crate::backend::submission_recovery::prediction::test_counts::record(3, role.is_some());
        let mut observation = owner.observation_recovery_with_prediction(
            ObservationRoots::Model {
                roots: roots.clone(),
            },
            role,
        )?;
        let retired = take_recovery(recovery);
        drop(retired);
        let result = roots.validate_completed().map_err(Error::from);
        observation.seal();
        let status = observation.progress();
        owner.request_release();
        if !status.settled || status.failed || status.blocked {
            owner.reject_unresolved();
            return match result {
                Err(error) => Err(error),
                Ok(()) => Err(Error::ArchitectureModel(
                    "model output observation failed or is unresolved".into(),
                )),
            };
        }
        drop(observation);
        if result.is_err() {
            owner.reject_unresolved();
        }
        // Dropping observation above may be the final scope retirement and may
        // discover an invalid retained backing during publication. Only after
        // both validation and publication succeed may original Record payloads
        // retire. Busy/error fences the owner before a success can be cached.
        let result = result.and_then(|()| owner.ensure_healthy()).and_then(|()| {
            if let Some(observer) = &retirement {
                crate::backend::submission_recovery::retirement::complete(observer)?;
            }
            Ok(())
        });
        if result.is_err() {
            owner.reject_unresolved();
        }
        result
    }
}

impl Completion for MlxSessionCompletion {
    type Error = Error;

    fn resources_releasable(&self) -> bool {
        let MlxSessionCompletionKind::Model { observation, .. } = &self.inner;
        let Ok(_loan) = observation.enter() else {
            return false;
        };
        safemlx::try_with_submission_retirement(|| {
            crate::backend::submission_recovery::reap();
            let MlxSessionCompletionKind::Model {
                recovery, owner, ..
            } = &self.inner;
            let settled = progress_recovery(recovery).is_none_or(|scope| scope.settled);
            if settled {
                let retired = take_recovery(recovery);
                drop(retired);
            }
            settled && owner.resources_releasable()
        })
        .unwrap_or(false)
    }

    fn is_complete(&self) -> Result<bool, Self::Error> {
        let MlxSessionCompletionKind::Model { observation, .. } = &self.inner;
        let observation = observation.enter()?;
        if let Some(result) = cached_observation(&observation, self.owner()) {
            return result.map(|()| true);
        }
        safemlx::try_with_submission_retirement(|| {
            let MlxSessionCompletionKind::Model {
                recovery, owner, ..
            } = &self.inner;
            if let Some(status) = progress_recovery(recovery) {
                if status.failed || status.blocked {
                    return observation
                        .complete(Err(Error::ArchitectureModel(
                            "model submission failed or became unobservable".into(),
                        )))
                        .map(|()| true);
                }
                if !status.settled {
                    return Ok(false);
                }
            }
            let _unwind = owner.poison_on_unwind();
            observation.complete(self.resolve()).map(|()| true)
        })
        .unwrap_or(Ok(false))
    }

    fn wait(&self) -> Result<(), Self::Error> {
        loop {
            if self.is_complete()? {
                return Ok(());
            }
            std::thread::yield_now();
        }
    }
}

impl Drop for MlxSessionCompletion {
    fn drop(&mut self) {
        match &self.inner {
            MlxSessionCompletionKind::Model {
                owner, recovery, ..
            } => {
                owner.request_release();
                let retired = take_recovery(recovery);
                drop(retired);
            }
        }
    }
}

#[cfg(test)]
#[path = "output_completion/recovery_slot_tests.rs"]
mod recovery_slot_tests;

#[cfg(test)]
#[path = "output_completion/once_tests.rs"]
mod once_tests;
