//! Exact host-only saved components for the neutral synchronous model.
//!
//! The backend has no persistent tensor state. Its model identity is one String;
//! its sampler is three scalars and its prompt is already immutable shared IDs.
//! Copying never preserves a historical prediction receipt as future authority.
use super::*;
use eredu_core::{HostMetadataFunding, HostPreparationAuthority};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryError, WorkingMemoryPool, WorkspaceCopyLimits};
use std::{alloc::Layout, mem::size_of, rc::Rc};

#[derive(Clone)]
pub(super) struct Source(Option<Rc<SourceData>>);
impl std::ops::Deref for Source { type Target = SourceData; fn deref(&self) -> &SourceData { self.0.as_ref().unwrap() } }
impl Drop for Source { fn drop(&mut self) { if let Some(owner) = self.0.take() { drop(Rc::into_inner(owner)); } } }
pub(super) struct SourceData {
    pub(super) native: Native,
    pub(super) sampling: SavedSampling,
    execution: InferenceExecutionIdentity,
    pool: WorkingMemoryPool,
    pub(super) frontier: u64,
    pub(super) capture: Option<eredu_runtime::capture::FundedCaptureCheckpoint>,
    // Keep both constructor and independently reserved copy accounts after data.
    _host: HostPreparationAuthority,
    _funding: HostMetadataFunding,
}
impl Source {
    pub(super) fn validate(&self, runtime: &ModelRuntime<MockBackend>) -> Result<(), MockError> {
        let env = MockBackend::source_environment(runtime);
        if !self.execution.same_execution(&env.execution) || !self.pool.same_domain(&env.pool) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        runtime.session().authority.require_idle()?;
        Ok(())
    }
}

pub(super) fn preparation_bytes() -> Result<u64, WorkingMemoryError> {
    [size_of::<Source>(), size_of::<SavedComponents>(), size_of::<SavedSampling>(),
        size_of::<Sampling>(), size_of::<Native>(), size_of::<HostMetadataFunding>(),
        size_of::<WorkspaceCopyLimits>(), size_of::<Option<PendingTextInput<Prompt, MockToken>>>(),
        size_of::<Result<SavedComponents, MockError>>()]
        .into_iter().try_fold(0usize, usize::checked_add)
        .and_then(|n| u64::try_from(n).ok()).ok_or(WorkingMemoryError::Overflow)
}

pub(super) fn estimates(
    runtime: &ModelRuntime<MockBackend>, input: Option<PendingTextInput<&Prompt, &MockToken>>,
) -> Option<[SnapshotEstimate; 3]> {
    runtime.session().authority.require_idle().ok()?;
    // Immutable prompt owners alias their already paid source. The only new
    // variable backing is the copied model identity, measured at its exact len.
    let native = Layout::new::<[usize;2]>().extend(Layout::new::<SourceData>()).ok()?.0.pad_to_align().size()
        .checked_add(runtime.session().intervention_identity.len())?
        .checked_add(HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()?)?;
    let sampling = size_of::<Sampling>();
    let pending = input.map_or(0, |_| size_of::<PendingTextInput<Prompt, MockToken>>());
    Some([native, sampling, pending].map(|n| SnapshotEstimate { retained_bytes: n as u64, copy_bytes: n as u64 }))
}

pub(super) fn capture(
    runtime: &mut ModelRuntime<MockBackend>, state: &observed_mock::State,
    input: Option<PendingTextInput<&Prompt, &MockToken>>, limits: WorkspaceCopyLimits,
    host: &HostPreparationAuthority,
) -> Result<SavedComponents, MockError> {
    runtime.session().authority.require_idle()?;
    if state.original.is_none() || host.is_unmanaged() {
        return Err(WorkingMemoryError::IdentityMismatch.into());
    }
    super::super::provider_errors::check("capture")?;
    let env = MockBackend::source_environment(runtime);
    let bytes = runtime.session().intervention_identity.len();
    let shell = Layout::new::<[usize; 2]>().extend(Layout::new::<SourceData>()).unwrap().0.pad_to_align().size();
    let capture_plan = state.funded.as_ref().map(|c| c.prepare_checkpoint()).transpose()
        .map_err(|e| MockError::ProviderRetained(admitted_text::funded_error(e,state.original.as_ref().unwrap().funding())))?;
    let capture_bytes = capture_plan.as_ref().map(|p|p.required_bytes()).transpose()
        .map_err(|_| WorkingMemoryError::UnknownBound)?.unwrap_or(0);
    let copy_bytes = bytes.checked_add(shell).and_then(|n| n.checked_add(usize::try_from(capture_bytes).ok()?))
        .ok_or(WorkingMemoryError::Overflow)?;
    let required = (copy_bytes as u64).checked_add(limits.safety_reserve_bytes).ok_or(WorkingMemoryError::Overflow)?;
    if limits.application_memory_budget_bytes.is_some_and(|limit| required > limit) {
        return Err(WorkingMemoryError::BudgetExceeded {
            required_bytes: required, available_bytes: limits.application_memory_budget_bytes.unwrap(),
        }.into());
    }
    let funding = env.pool.prepare_workspace_metadata(&env.execution, limits.capacity_bytes)?;
    funding.reserve_metadata(copy_bytes.checked_add(usize::try_from(limits.safety_reserve_bytes)
        .map_err(|_| WorkingMemoryError::Overflow)?).ok_or(WorkingMemoryError::Overflow)?)?;
    let mut identity = String::new();
    identity.try_reserve_exact(bytes).map_err(|_| WorkingMemoryError::UnknownBound)?;
    identity.push_str(&runtime.session().intervention_identity);
    let pending = input.map(|input| match input {
        PendingTextInput::Prefill(prompt) => PendingTextInput::Prefill(prompt.clone()),
        PendingTextInput::Decode(token) => PendingTextInput::Decode(MockToken::new(token.0)),
    });
    funding.reserve_metadata(HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().ok_or(WorkingMemoryError::Overflow)?)?;
    let capture_host = HostPreparationAuthority::retain(funding.clone());
    let capture = capture_plan.map(|p|p.construct(&capture_host)).transpose()
        .map_err(|e|MockError::ProviderRetained(admitted_text::funded_error(e,&funding)))?;
    let source = Source(Some(Rc::new(SourceData {
        native: Native(identity, Some(funding.clone())),
        sampling: SavedSampling { sampling: state.sampling.clone(), pending },
        execution: env.execution.clone(), pool: env.pool.clone(),
        frontier: state.original.as_ref().unwrap().completed_positions(state.sampling.prediction)?,
        capture, _host: host.clone(), _funding: funding,
    })));
    Ok(SavedComponents { ordinary: None, original: Some(source) })
}
