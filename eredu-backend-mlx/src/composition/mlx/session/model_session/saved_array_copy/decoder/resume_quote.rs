//! Full cold resume diagnostics bound to one independently saved source pair.
//!
//! This is not a request, accepted quote, health certificate or copy authority.
//! Source accounts remain charged. A separately sealed isolated-copy program
//! binds registered-root credit before its private trace. Admission must revalidate source health, controller custody,
//! the destination session and parameter epoch under its actual account commit.

use super::*;
mod media;
mod child;
pub(in crate::composition::mlx::session) use child::ResumeCapture;
mod parallel;
mod preparation;
mod phases;
use crate::composition::mlx::session::model_session::pending_prompt::PreparedPendingPrompt;
use eredu_architectures::prepared_execution::{
    BorrowedTextSamplingWorkspace, PreparedTextGenerationWorkspace,
};
use eredu_core::{
    ExecutionWorkspaceEstimate, InferenceGeometry, InputTokenCount, OutputDemand,
    RuntimeStateEstimate, TextControllerContract, TextControllerWorkspace, WorkspaceBound,
};
use eredu_nn::workspace::{
    WorkspaceBackend, HostMetadataFunding, HostMetadataFundingError, WorkspaceTensor,
    WorkspaceTraceReport,
};
use eredu_runtime::{
    CacheResidencyPolicy, DeviceState, LayerWeightResidency,
    working_memory::{
        WorkspaceReportMetadata, WorkspaceResidentLayerState, WorkspaceSamplingRandomState,
    },
};
pub(super) use preparation::ResumeSourceCause;
use std::num::NonZeroU32;

/// Read-only diagnostics for the exact saved values. The source borrow retains
/// its host holds, native backing, shared input/layout and completed copy account;
/// it cannot be replaced by an ordinal, byte total or unrelated sampler/decoder.
/// No current installed decoder is projected, cloned or used as the future state.
///
/// Pending prompt host construction uses its closed retained-part worker.
/// Numerical/host bookkeeping exclusions are those of the
/// existing workspace contracts; this is never a total-process memory estimate.
pub(in crate::composition::mlx::session) struct PreparedSavedTextResumeQuote<'a> {
    source: &'a CopiedTextComponents,
    child_capture: Option<ResumeCapture>,
    geometry: InferenceGeometry,
    state_input: InputTokenCount,
    config: TextGenerationConfig,
    controller: TextControllerContract,
    full: RuntimeStateEstimate,
    incremental: eredu_runtime::working_memory::CopyPreparationInferenceQuote<StorageIdentity>,
    generation: PreparedTextGenerationWorkspace,
    native_recipe: Option<crate::backend::nn::workspace::ResidentNativeRecipe>,
    // Exact source/parameter proof. The legacy quote consumes only its existing
    // identity after pinning, without cloning or extending snapshot ownership.
    layerwise: Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    decoder_copy: WorkspaceTraceReport,
    sampling_copy: WorkspaceTraceReport,
    decoder_host_peak: u64,
    sampler_host_peak: u64,
    pending_host: WorkspaceBound,
    // Actual prepared source aliases and page pins, separate from copied roots.
    source_native: Option<crate::backend::nn::workspace::ProjectedNativeStorage>,
    // These are portable metadata roots, not installable native values.
    copied_state: DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
    copied_key: Option<WorkspaceTensor>,
    copied_input: Option<WorkspaceTensor>,
    // Exact completed B mapping, separate from the copied decoder/key proof.
    media_storage:
        Option<eredu_runtime::working_memory::RegisteredPreparedWorkspaceStorage<StorageIdentity>>,
    // Existing-only pins add no charge or authority; source values remain above.
    _registered_source: WorkingMemoryStorage<StorageIdentity>,
    // Existing funded Context for this exact adapter's continuation constructors.
    // It supplies no completed-B source or old request authority.
    continuation_metadata: Option<WorkspaceContext>,
    // The copied paged manager has separate construction custody. Its presence
    // never opts a plain text adapter into composite input construction.
    paged_source_metadata: Option<WorkspaceContext>,
    paged_host_facts: Option<eredu_runtime::working_memory::HostSourceConstructionFacts>,
    // Exact attempted row/source owners from this fresh cold continuation.
    text_interventions: Option<crate::composition::mlx::session::intervention::PreparedTextInterventions>,
    // Independent cumulative metadata account, after every funded result field.
    planning_metadata: Option<HostMetadataFunding>,
}

impl<'a> PreparedSavedTextResumeQuote<'a> {
    pub(in crate::composition::mlx::session) fn validate_resume_options(
        source: &CopiedTextComponents, config: TextGenerationConfig,
        options: &eredu_core::OriginalTextResumeOptions<'_>,
    ) -> Result<(), WorkingMemoryError> {
        if options.terminal != (config.sampling().max_new_tokens == Some(0))
            || (options.capture_limits.is_some() && source.capture_checkpoint().is_none())
            || options.session_id.is_some_and(str::is_empty)
            || (options.kind != eredu_core::OriginalTextResumeKind::Branch
                && (options.capture_limits.is_some() || options.intervention.is_some() || options.sampling.is_some())) {
            return Err(WorkingMemoryError::PreparationConfigurationMismatch);
        }
        if let Some(request) = options.sampling {
            eredu_runtime::execution_control::validate_sampling_override::<Error>(
                source.sampling.sampling_state_facts(), request)
                .map_err(|_| WorkingMemoryError::PreparationConfigurationMismatch)?;
        }
        Ok(())
    }
    pub(in crate::composition::mlx::session) fn take_child_capture(&mut self) -> Option<ResumeCapture> {
        self.child_capture.take()
    }
    pub(in crate::composition::mlx::session) fn take_text_interventions(&mut self)
        -> Option<crate::composition::mlx::session::intervention::PreparedTextInterventions> {
        self.text_interventions.take()
    }

    /// Finite source, table, pending-host, admission and credited-copy metadata.
    /// This is a component of host preparation, not the complete resume quote.
    pub(in crate::composition::mlx::session) fn known_source_preparation_component_bytes(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        source: &CopiedTextComponents,
    ) -> Result<usize, preparation::ResumeSourceCause> {
        source
            .validate_resume_origin_fixed(runtime)
            .map_err(|cause| cause.into_memory())?;
        let owner = DecoderCopyOwner::Saved(source.decoder.clone());
        let prepared = source.decoder.native.prepare_copy_fixed()?;
        let pending_plan = if source.sampling.arrays.pending_metadata.is_none() { None }
            else { source.sampling.prepare_pending_tokens()? };
        let pending = pending_plan
            .as_ref()
            .map(|pending| pending.numerical().source());
        let binding = TextArrayBinding::Saved(&source.sampling);
        let mechanisms = runtime
            .session()
            .payload
            .model
            .resident_workspace_mechanisms()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let plan = preparation::ResumeSourcePreparation::inspect(
            &owner,
            &prepared,
            binding.key(),
            pending,
            runtime.backend().memory_pool(),
            mechanisms,
        )?;
        let dense_result = source
            .decoder
            .native
            .prepare_copy_fixed()?
            .into_dense_fixed();
        let dense_result_bytes = std::mem::size_of_val(&dense_result);
        let dense = dense_result?;
        let exchange = crate::composition::mlx::session::model_session::control_slot::OriginalControlExchangePlan::inspect(runtime)
            .map_err(|cause| cause.into_memory())?;
        let slot_bytes = runtime
            .session()
            .payload
            .model
            .erased()
            .prepared_resident_control_state_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let parts = [
            CopiedTextComponents::resume_origin_control_bytes()
                .ok_or(WorkingMemoryError::Overflow)?,
            slot_bytes,
            exchange.control_bytes()?,
            plan.known_component_bytes(),
            dense.host_preparation_control_bytes()?,
            super::resume_prompt::preparation_control_bytes(
                &prepared,
                pending_plan.as_ref(),
                binding.key().is_some(),
            )?,
            if pending_plan.is_some() {
                PreparedPendingPrompt::source_control_bytes().ok_or(WorkingMemoryError::Overflow)?
            } else {
                std::mem::size_of::<crate::composition::mlx::CompletedOriginalModelInput>()
                    .checked_add(std::mem::size_of::<
                        Option<eredu_runtime::working_memory::InferenceRequest>,
                    >())
                    .and_then(|bytes| {
                        bytes.checked_add(std::mem::size_of::<Option<std::num::NonZeroU64>>())
                    })
                    .ok_or(WorkingMemoryError::Overflow)?
            },
            super::super::super::text_quote::original_resume_admission_control_bytes()
                .ok_or(WorkingMemoryError::Overflow)?,
            std::mem::size_of_val(&dense),
            dense_result_bytes,
            std::mem::size_of::<Result<usize, preparation::ResumeSourceCause>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| WorkingMemoryError::Overflow.into())
    }

    #[cfg(test)]
    pub(in crate::composition::mlx::session) fn layerwise_workspace(
        &self,
    ) -> Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace> {
        self.layerwise.as_ref()
    }

    pub(in crate::composition::mlx::session) fn take_layerwise_workspace(
        &mut self,
    ) -> Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace> {
        self.layerwise.take()
    }

    pub(in crate::composition::mlx::session) fn prepare(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        source: &'a CopiedTextComponents,
        config: TextGenerationConfig,
        controller: TextControllerWorkspace<'_>,
    ) -> Result<Self, Error> {
        Self::prepare_impl(runtime, source, config, controller, None, None, None)
    }

    pub(in crate::composition::mlx::session) fn prepare_original(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        source: &'a CopiedTextComponents,
        config: TextGenerationConfig,
        controller: TextControllerWorkspace<'_>,
        host: &eredu_core::HostPreparationAuthority,
        options: &eredu_core::OriginalTextResumeOptions<'_>,
    ) -> Result<Self, Error> {
        let capacity = config
            .inference_policy()
            .managed_memory_capacity_bytes
            .ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Unavailable,
            ))?;
        let funding = runtime
            .backend()
            .memory_pool()
            .prepare_workspace_metadata(
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .inference_execution_identity(),
                capacity,
            )
            .map_err(Error::WorkspacePlanning)?;
        let controls = Self::planning_control_bytes().ok_or(Error::WorkspacePlanning(
            HostMetadataFundingError::Overflow,
        ))?;
        funding
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        Self::prepare_impl(
            runtime,
            source,
            config,
            controller,
            Some(host),
            Some(&funding),
            Some(options),
        )
        .map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(
                super::resume_prompt::ResumeFailure::new(cause, host),
                funding,
            )
        })
    }

    fn prepare_impl(
        runtime: &ModelRuntime<MlxBackend<'_>>, source: &'a CopiedTextComponents,
        config: TextGenerationConfig, controller: TextControllerWorkspace<'_>,
        host: Option<&eredu_core::HostPreparationAuthority>,
        planning_metadata: Option<&HostMetadataFunding>,
        options: Option<&eredu_core::OriginalTextResumeOptions<'_>>,
    ) -> Result<Self, Error> {
        // Retire completed preparation frames before reconstructing a composite
        // model; final report composition starts only after its trace returns.
        let child_capture = options.map(|options| child::ResumeCapture::prepare(runtime, source, options,
            planning_metadata.ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?)).transpose()?.flatten();
        let view = child::Source { saved: source, capture: child_capture.as_ref() };
        let inputs=phases::prepare(runtime,&view,config,controller,host,planning_metadata)?;
        let traced=phases::trace(runtime,&view,controller,inputs,planning_metadata)?;
        let mut quote = phases::finish(runtime,&view,config,controller,traced,planning_metadata)?;
        quote.child_capture = child_capture;
        Ok(quote)
    }

    pub(in crate::composition::mlx::session) fn state_input(&self) -> InputTokenCount {
        self.state_input
    }

    pub(in crate::composition::mlx::session) fn prepared_media_source(
        &self,
    ) -> Result<
        Option<super::super::super::text_quote::original_prepared::PreparedMediaQuoteSource>,
        Error,
    > {
        use super::super::super::text_quote::original_prepared::PreparedMediaQuoteSource;
        let Some(storage) = &self.media_storage else {
            return Ok(None);
        };
        let context = self
            .continuation_metadata
            .as_ref()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        context
            .charge_metadata(std::mem::size_of::<(
                PreparedMediaQuoteSource,
                Option<PreparedMediaQuoteSource>,
                Result<Option<PreparedMediaQuoteSource>, Error>,
            )>())
            .map_err(|cause| planned_error(cause, self.planning_metadata.as_ref()))?;
        Ok(Some(PreparedMediaQuoteSource::new(
            storage.prepared_source().clone(),
            context.clone(),
        )))
    }

    /// Moves only the actual continuation construction destination. The
    /// original source, copy custody and new run admission remain separate.
    pub(in crate::composition::mlx::session) fn take_continuation_metadata(
        &mut self,
    ) -> Option<WorkspaceContext> {
        self.continuation_metadata.take()
    }

    /// Moves the constructor account for the actual independently copied
    /// paged manager. It is consumed by source installation, not text decoding.
    pub(in crate::composition::mlx::session) fn take_paged_source_metadata(
        &mut self,
    ) -> Option<WorkspaceContext> {
        self.paged_source_metadata.take()
    }

    pub(in crate::composition::mlx::session) fn paged_host_facts(
        &self,
    ) -> Option<eredu_runtime::working_memory::HostSourceConstructionFacts> {
        self.paged_host_facts
    }

    /// A clone of the same metadata account, never native work authority. The
    /// admission keeps it through consuming the diagnostic and final quote.
    pub(in crate::composition::mlx::session) fn planning_metadata(
        &self,
    ) -> Option<HostMetadataFunding> {
        self.planning_metadata.clone()
    }

    fn planning_control_bytes() -> Option<usize> {
        let controls = [
            phases::control_bytes()?,
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<Option<HostMetadataFunding>>(),
            std::mem::size_of::<HostMetadataFunding>(),
            std::mem::size_of::<HostMetadataFundingError>(),
            std::mem::size_of::<crate::backend::nn::workspace::ProjectedResidentState>(),
            std::mem::size_of::<WorkspaceReportMetadata<'_>>(),
            std::mem::size_of::<
                Option<(
                    &eredu_runtime::capture::FundedCaptureCheckpoint,
                    &eredu_runtime::layered::PreparedCaptureSelection,
                    &eredu_runtime::working_memory::RegisteredInferenceSourceWitness,
                )>,
            >(),
            std::mem::size_of::<RetainedStorage>(),
            std::mem::size_of::<RetainedStorage>(),
            std::mem::size_of::<Result<Option<usize>, Error>>(),
            std::mem::size_of::<(
                &ModelRuntime<MlxBackend<'_>>,
                &CopiedTextComponents,
                TextGenerationConfig,
                TextControllerWorkspace<'_>,
                &eredu_core::HostPreparationAuthority,
                Option<&HostMetadataFunding>,
            )>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }

    pub(in crate::composition::mlx::session) fn take_source_native(
        &mut self,
    ) -> Option<crate::backend::nn::workspace::ProjectedNativeStorage> {
        self.source_native.take()
    }

    pub(in crate::composition::mlx::session) fn take_native_recipe(
        &mut self,
    ) -> Option<crate::backend::nn::workspace::ResidentNativeRecipe> {
        self.native_recipe.take()
    }

    /// Binds accounting health to the exact source retained by this diagnostic.
    /// Full-source pins add no credit and are revalidated inside reservation.
    pub(in crate::composition::mlx::session) fn registered_source(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::RegisteredSavedSamplingSource<'a, StorageIdentity>,
        Error,
    > {
        self.source
            .sampling
            .arrays
            .custody
            .bind_saved_sampling_source(
                &self.source.sampling.sampler,
                self._registered_source.clone(),
            )
            .map_err(memory)
    }

    pub(in crate::composition::mlx::session) fn retain_source_registration(
        &self,
    ) -> WorkingMemoryStorage<StorageIdentity> {
        self._registered_source.clone()
    }

    pub(in crate::composition::mlx::session) fn controller_contract(
        &self,
    ) -> &TextControllerContract {
        &self.controller
    }

    pub(in crate::composition::mlx::session) fn full(&self) -> &RuntimeStateEstimate {
        &self.full
    }
    pub(in crate::composition::mlx::session) fn incremental(
        &self,
    ) -> &eredu_runtime::working_memory::CopyPreparationInferenceQuote<StorageIdentity> {
        &self.incremental
    }

    pub(in crate::composition::mlx::session) fn into_incremental(
        self,
    ) -> eredu_runtime::working_memory::CopyPreparationInferenceQuote<StorageIdentity> {
        self.incremental
    }

    pub(in crate::composition::mlx::session) fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    pub(in crate::composition::mlx::session) fn output_width(&self) -> usize {
        self.generation.sampling.output_width
    }
}

impl CopiedTextComponents {
    /// Internal recheck for an inventory retained by the closed admission owner.
    /// The caller cannot substitute an inventory at any public resume boundary.
    pub(in crate::composition::mlx::session) fn validate_resume_account(
        &self,
        registered: &WorkingMemoryStorage<StorageIdentity>,
    ) -> Result<(), Error> {
        self.sampling
            .arrays
            .custody
            .bind_saved_sampling_source(&self.sampling.sampler, registered.clone())
            .map(drop)
            .map_err(memory)
    }
}

#[cfg(test)]
fn resume_geometry(
    source: &CopiedTextComponents,
    config: TextGenerationConfig,
) -> Result<InferenceGeometry, Error> {
    resume_geometry_impl(&child::Source { saved: source, capture: None }, config, None)
}

/// The retained capture selection fixes physical readout for both quotation
/// and the later source-bound prompt constructor. Ordinary saved text retains
/// its existing final-position demand. This grants no capture/source authority.
fn resume_output_demand(source: &child::Source<'_, '_>) -> OutputDemand {
    let demand = source.capture_selection().map_or(OutputDemand::LastPosition,
        |selection| selection.physical_output(OutputDemand::LastPosition));
    let first=source.sampling.next_prediction;
    let phase=if first==0{eredu_core::capture::CapturePhase::Prefill}else{eredu_core::capture::CapturePhase::Decode};
    if source.capture_checkpoint().and_then(|checkpoint|checkpoint.intervention_source())
        .is_some_and(|source|source.plan().admission().requires_sequence_scores(phase,first)) {
        OutputDemand::Sequence
    } else { demand }
}

fn resume_geometry_impl(
    source: &child::Source<'_, '_>,
    config: TextGenerationConfig,
    planning_metadata: Option<&HostMetadataFunding>,
) -> Result<InferenceGeometry, Error> {
    let memory = |cause: WorkingMemoryError| planned_error(cause, planning_metadata);
    let unknown = || memory(WorkingMemoryError::UnknownBound);
    let outputs = u64::try_from(config.sampling().max_new_tokens.ok_or_else(unknown)?)
        .map_err(|_| memory(WorkingMemoryError::Overflow))?;
    if outputs == 0 {
        return Ok(InferenceGeometry { batch_size: 1, cached_positions: source.sampling.frontier(),
            input_positions: 0, max_output_tokens: 0, prefill_chunk_positions: 0,
            output: OutputDemand::StateOnly });
    }
    source
        .sampling
        .next_prediction
        .checked_add(outputs)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let (input_positions, saved_chunk) = source.sampling.pending_geometry().map_err(memory)?;
    let prefill_chunk_positions = config
        .inference_policy()
        .prefill_chunk_positions
        .map_or(saved_chunk, |requested| saved_chunk.min(requested.get()));
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: source.sampling.frontier(),
        input_positions,
        max_output_tokens: outputs,
        prefill_chunk_positions,
        output: resume_output_demand(source),
    };
    geometry
        .validate()
        .map_err(|cause| planned_error(cause, planning_metadata))?;
    if let Some(checkpoint) = source.capture_checkpoint() {
        checkpoint
            .validate_continuation_geometry(geometry)
            .map_err(|cause| planned_error(cause, planning_metadata))?;
    }
    Ok(geometry)
}

/// `total_bytes` alone omits existing opening roots. Full copy composition uses
/// complete closing backing plus complete transient overlap, including host.
fn full_span(
    report: &WorkspaceTraceReport,
    label: &'static str,
    metadata: WorkspaceReportMetadata<'_>,
) -> Result<WorkspaceBound, Error> {
    match report
        .state
        .as_ref()
        .and_then(|state| state.retained_bytes.zip(state.transient_bytes))
    {
        Some((retained, transient)) => metadata
            .bounded(
                retained
                    .checked_add(transient)
                    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
                format_args!("{label}"),
            )
            .map_err(|cause| {
                if metadata.is_checked() {
                    Error::Neural(metadata.error(cause))
                } else {
                    other(cause.into_capability())
                }
            }),
        None => metadata
            .unknown(format_args!(
                "{label} has missing tensor, host or retained-source facts"
            ))
            .map_err(|cause| {
                if metadata.is_checked() {
                    Error::Neural(metadata.error(cause))
                } else {
                    other(cause.into_capability())
                }
            }),
    }
}
fn planned_error(
    cause: impl std::error::Error + Send + Sync + 'static,
    funding: Option<&HostMetadataFunding>,
) -> Error {
    match funding {
        Some(funding) => {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        }
        None => other(cause),
    }
}

fn other(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Other(Box::new(error))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
