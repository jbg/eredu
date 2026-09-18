//! Replacement sampling shares the original tracer, native roles and completion
//! engine. Only its independently admitted future sampling account is new.
use super::*;
use crate::backend::nn::workspace::{ExistingArrayProjection, ResidentSamplingProgram};
use crate::backend::runtime::residency::storage::native_storage::{BankOwner, MlxNativeStorage};
use eredu_nn::workspace::{WorkspaceDtype, HostMetadataFunding};
use eredu_runtime::working_memory::{
    InferenceTextStep, OriginalNativeStorageMechanism,
    OriginalTextSamplingExtension, SamplingWorkspaceReport, TextHostControlFacts,
    WorkspaceSamplingRandomState, quote_sampling_workspace_with_observer,
};
use std::mem::{size_of, size_of_val};

/// Optional replacements do not enlarge every initial quote transport. The
/// closed owner releases its paid allocation before the payload's accounts.
#[derive(Debug)]
pub(super) struct SamplingRevisionOwner(Option<Box<SamplingRevision>>);

impl SamplingRevisionOwner {
    fn new(revision: SamplingRevision) -> Result<Self, Error> {
        let bytes = [size_of::<SamplingRevision>(), size_of::<Self>(),
            size_of::<Result<Self, Error>>(), size_of::<Box<SamplingRevision>>(),
            size_of::<Option<Box<SamplingRevision>>>(), size_of::<SamplingRevision>()]
            .into_iter().try_fold(0usize, usize::checked_add)
            .ok_or(Error::WorkspacePlanning(eredu_core::HostMetadataFundingError::Overflow))?;
        revision._planning.reserve_metadata(bytes).map_err(Error::WorkspacePlanning)?;
        Ok(Self(Some(Box::new(revision))))
    }

    fn commit(&mut self) -> Result<(), WorkingMemoryError> {
        self.0.as_mut().expect("live sampling revision").extension.commit()
    }
}
impl std::ops::Deref for SamplingRevisionOwner {
    type Target = SamplingRevision;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live sampling revision")
    }
}
impl Drop for SamplingRevisionOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            let revision = *owner;
            drop(revision);
        }
    }
}

#[derive(Debug)]
pub(super) struct SamplingRevision {
    temperature: f32,
    first: u64,
    program: ResidentSamplingProgram,
    scopes: prediction::PredictionScopes,
    bank: BankOwner,
    extension: OriginalTextSamplingExtension,
    // New native/plan payload and their shared shells retire before planning.
    _planning: HostMetadataFunding,
}
impl SamplingRevision {
    pub(super) fn claim(&self, step: &InferenceTextStep,
        source: Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalSamplingSource>)
        -> Result<crate::backend::submission_recovery::prediction::PredictionSet, Error> {
        let index = step.attempt().checked_sub(self.first).and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
        let completion = self.program.completion(index).ok_or_else(unknown)?;
        Ok(self.scopes.claim(step)?.with_sampling(Some(completion)).with_sampling_source(source))
    }
    fn work(&self) -> Result<super::super::text_funding::FundedWorkOwner, Error> {
        super::super::text_funding::FundedWork::new_model_with_native(
            self.extension.funding().scope().map_err(memory)?, Some(self.extension.control_guard()),
            None, None, None, None, Some(self.bank.clone()))
    }
}
impl TextExecutionQuote {
    pub(in crate::composition::mlx::session) fn sampling_temperature(&self) -> Result<f32, Error> {
        let revision = self.sampling_revision.try_borrow().map_err(|_| Error::PredictionScopeReentrant)?;
        Ok(revision.as_ref().map_or(self.config.sampling().temperature, |revision| revision.temperature))
    }
    pub(in crate::composition::mlx::session::model_session) fn sampling_work(&self, step: &InferenceTextStep)
        -> Result<Option<super::super::text_funding::FundedWorkOwner>, Error> {
        self.request.validate_same_request(step.request()).map_err(memory)?;
        self.sampling_revision.try_borrow().map_err(|_| Error::PredictionScopeReentrant)?
            .as_ref().map(|revision| revision.work()).transpose()
    }
    pub(in crate::composition::mlx::session) fn replace_sampling(
        &self, runtime: &mut ModelRuntime<MlxBackend<'_>>,
        state: &mut generation::MlxTextSamplingState,
        context: &eredu_core::TextStepContext,
        change: eredu_runtime::execution_control::ValidatedSamplingOverride,
    ) -> Result<(), Error> {
        runtime.validate_session_admission()?;
        self.validate_frontier(runtime, state.next_prediction)?;
        runtime.session().ensure_no_submission_in_flight()?;
        if state.temperature != self.sampling_temperature()? {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        self.replace_sampling_at_boundary(runtime, state, context, change)
    }

    pub(super) fn replace_saved_sampling(
        &self, runtime: &ModelRuntime<MlxBackend<'_>>, state: &mut generation::MlxTextSamplingState,
        context: &eredu_core::TextStepContext, change: eredu_runtime::execution_control::ValidatedSamplingOverride,
    ) -> Result<(), Error> {
        self.opening.require_pending().map_err(memory)?;
        runtime.session().ensure_no_submission_in_flight()?;
        if state.temperature != self.sampling_temperature()? {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        // The private resume owner authenticated this copied pair and the exact
        // new core context; native model placement has not happened yet.
        self.replace_sampling_at_boundary(runtime, state, context, change)
    }

    fn replace_sampling_at_boundary(
        &self, runtime: &ModelRuntime<MlxBackend<'_>>, state: &mut generation::MlxTextSamplingState,
        context: &eredu_core::TextStepContext, change: eredu_runtime::execution_control::ValidatedSamplingOverride,
    ) -> Result<(), Error> {
        let remaining = self.request.sampling_extension_remaining(context).map_err(memory)?;
        if change.reseed().is_none() && change.temperature() == state.temperature {
            return Ok(());
        }
        let capacity = self.config.inference_policy().managed_memory_capacity_bytes
            .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?;
        let funding = self.model_pool.prepare_workspace_metadata(
            runtime.session().payload.model.erased().inference_execution_identity(), capacity)
            .map_err(Error::WorkspacePlanning)?;
        let result = self.prepare_sampling_revision(runtime, state, context, change, remaining, capacity, &funding);
        match result {
            Ok((mut revision, random)) => {
                // Acquire the destination slot before committing the new epoch.
                // After that infallible moves publish all revised facts together.
                let mut slot = self.sampling_revision.try_borrow_mut().map_err(|_| Error::PredictionScopeReentrant)?;
                revision.commit().map_err(memory)?;
                let previous = slot.replace(revision);
                let old_random = random.map(|random| state.prng.replace(random));
                state.temperature = change.temperature();
                drop(slot);
                drop((old_random, previous));
                Ok(())
            }
            Err(cause) => Err(crate::composition::mlx::model::retain_planning_error(cause, funding)),
        }
    }
    fn prepare_sampling_revision(
        &self, runtime: &ModelRuntime<MlxBackend<'_>>,
        state: &generation::MlxTextSamplingState,
        context: &eredu_core::TextStepContext,
        change: eredu_runtime::execution_control::ValidatedSamplingOverride,
        remaining: u64, capacity: u64, funding: &HostMetadataFunding,
    ) -> Result<(SamplingRevisionOwner, Option<RandomState>), Error> {
        let controls = [size_of::<SamplingRevision>(), size_of::<Option<SamplingRevisionOwner>>(),
            size_of::<Result<(SamplingRevisionOwner, Option<RandomState>), Error>>(),
            size_of::<SamplingWorkspaceReport>(), size_of::<Option<RandomState>>(),
            size_of::<std::cell::RefMut<'_, Option<SamplingRevisionOwner>>>(),
            size_of::<eredu_runtime::working_memory::SamplingExtensionQuote>(),
            size_of::<eredu_runtime::working_memory::OriginalTextSamplingExtension>()];
        let controls = controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(eredu_core::HostMetadataFundingError::Overflow))?;
        funding.reserve_metadata(controls).map_err(Error::WorkspacePlanning)?;
        let pending = self.request.begin_sampling_extension(context, funding).map_err(memory)?;
        if pending.remaining_steps() != remaining { return Err(memory(WorkingMemoryError::IdentityMismatch)); }
        let model = runtime.session().payload.model.erased();
        let mechanism = runtime.session().payload.model.native_storage_mechanism()?.ok_or_else(unknown)?;
        let workspace_mechanisms = runtime.session().payload.model.resident_workspace_mechanisms().ok_or_else(unknown)?;
        let workspace = workspace_mechanisms.context(funding.clone()).map_err(eredu_nn::Error::from)?;
        let input = self.native_recipe.as_ref().and_then(|recipe| recipe.sampling_program().input())
            .ok_or_else(unknown)?;
        let layout = workspace.layout(input.shape(), WorkspaceDtype::Float32)?;
        let mut projection = ExistingArrayProjection::with_source_count(&workspace, usize::from(state.prng.is_some()))
            .map_err(|cause| Error::Neural(workspace.metadata_source(cause)))?;
        let random = if change.reseed().is_some() {
            Some(WorkspaceSamplingRandomState::from_seed(&workspace)?)
        } else {
            state.prng.as_ref().map(|random| projection.project(random.as_array())
                .and_then(WorkspaceSamplingRandomState::from_key)).transpose()?
        };
        let mut observer = workspace_mechanisms.recorder(pending.geometry(), &workspace)?
            .sampling_collector(remaining, &workspace)?;
        let filter = self.controller.sampling_workspace_bound()
            .map_err(|_| memory(WorkingMemoryError::UnknownBound))?;
        let report = quote_sampling_workspace_with_observer(state.sampler.as_sampler(), change.temperature(),
            random.as_ref(), input.source(&layout)?, filter, remaining, &workspace, Some(&mut observer))?;
        let (program, plan) = observer.finish(&report)?;
        let quote = pending.compose(&report, plan).map_err(memory)?;
        let physical = mechanism.sampling_program(quote.workspace(), &program, change.reseed().is_some(), funding)?;
        let collector = super::super::text_funding::native_collector_control_bytes(
            physical.attempts(), physical.rows(), physical.works())?.ok_or_else(unknown)?;
        let native = mechanism.sampling_plan(quote.workspace(), &physical, collector)?;
        let policy = self.config.inference_policy();
        let graph_fit = program.graph_storage_requirement()?;
        let graph_capacity = graph_fit.select_for_policy(policy.graph_metadata_capacity_bytes).map_err(memory)?;
        let graph_facts = graph::selected_facts(policy.graph_metadata_capacity_bytes, Some(graph_capacity),
            graph_fit.control_bytes().ok_or_else(unknown)?)?.ok_or_else(unknown)?;
        let record_capacity = program.record_storage_requirement()?
            .select_for_policy(policy.submission_tracking_capacity_bytes).map_err(memory)?;
        let tracking_facts = tracking::selected_facts(policy.submission_tracking_capacity_bytes, Some(record_capacity))?
            .ok_or_else(unknown)?;
        let work_controls = super::super::text_funding::work_control_bytes()?
            .checked_mul(u64::try_from(physical.works()).map_err(|_| memory(WorkingMemoryError::Overflow))?)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let pipeline_controls = pipeline_cache::control_bytes_for_attempts(program.kernel_attempts())?.ok_or_else(unknown)?;
        let admission = u64::try_from(controls).ok().and_then(|n| n.checked_add(pipeline_controls))
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let (sampling, event) = prediction::sampling_facts()?;
        let mut prepared = quote.prepare_controls(TextHostControlFacts::new(
            Some(admission), Some(0), Some(work_controls)))
            .map_err(memory)?
            .with_sampling_prediction_scopes(sampling, event).map_err(memory)?
            .with_submission_tracking(tracking_facts).map_err(memory)?
            .with_graph_metadata(graph_facts).map_err(memory)?
            .with_native_storage(native).map_err(memory)?;
        if change.reseed().is_some() {
            prepared = prepared.with_sampling_reseed_scope(Some(preparation::reseed_control_bytes()?)).map_err(memory)?;
        }
        let mut extension = quote.with_controls(prepared).map_err(memory)?.admit(capacity).map_err(memory)?;
        let mut bank = extension.take_native_storage_bank::<MlxNativeStorage>(mechanism.selection())
            .map_err(memory)?.ok_or_else(unknown)?;
        bank.install(mechanism).map_err(|cause|
            crate::backend::runtime::residency::storage::native_storage::retained_failure(
                cause, extension.control_guard(), true))?;
        let bank = BankOwner::new(bank);
        let graph = graph::allocate(extension.take_graph_metadata().map_err(memory)?.ok_or_else(unknown)?)?;
        let tracking = tracking::allocate(extension.take_submission_tracking().map_err(memory)?.ok_or_else(unknown)?)?;
        pipeline_cache::install_for_attempts(program.kernel_attempts().ok_or_else(unknown)?, &extension.control_guard(), &graph)?;
        let roles = extension.take_prediction_scopes().map_err(memory)?.ok_or_else(unknown)?;
        let scopes = prediction::PredictionScopes::new(roles, Some(tracking.clone()), Some(graph.clone()), extension.control_guard())
            .with_native_storage(Some(bank.clone()));
        let replacement = change.reseed().map(|seed| preparation::reseed(
            &self.request, seed, &program, &mut extension, &bank, &tracking, &graph)).transpose()?;
        Ok((SamplingRevisionOwner::new(SamplingRevision { temperature: change.temperature(), first: self.request.geometry().max_output_tokens
            .checked_sub(remaining).ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            program, scopes, bank, extension, _planning: funding.clone() })?, replacement))
    }
}
