//! Fixed completed-source authentication and the actual shared-planner candidate.
//! Native completeness comes from the observed quote and common control seal.
use super::*;
use eredu_core::PreparedRequestRejection as R;

pub(super) fn preflight(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    prompt: &MlxModelInput,
    config: TextGenerationConfig,
    _capture: Option<&eredu_core::capture::SharedCapturePlan>,
    claim: Option<&eredu_core::GenerationSequencePreparation<'_, '_>>,
) -> R {
    match validate(runtime, prompt, config.clone(), claim) {
        Err(error) => error,
        // Ordinary preinstalled collectors are not original capture authority.
        // The supplied immutable plan enters the common authenticated bank below.
        // Fixed source inspection precedes the funded equation/recipe adapter.
        // No controller callback, diagnostic trace or native operation has run.
        Ok(source) => match source.inspect(config) {
            Err(error) => error,
            // Borrowed logical facts are exercised, but never promoted to full
            // backing/primitive/workspace or request admission authority.
            Ok(facts) => facts.missing_contribution(),
        },
    }
}

/// Actual source geometry for the common admission comparison. The caller has
/// already completed fixed source/capture authentication above.
pub(super) fn admission_input(prompt: &MlxModelInput) -> Result<InputTokenCount, R> {
    let Some(input::OriginalMediaPacket::Original(packet)) = prompt.original_media.as_ref() else {
        return Err(R::SourceUnavailable);
    };
    packet
        .borrowed_semantics()
        .request_position_facts()
        .and_then(|positions| {
            positions.legacy_input_accounting(std::num::NonZeroU8::new(4).unwrap())
        })
        .map_err(semantic)
}

/// One source-bound candidate in the same shared chunk planner. The caller
/// lends its already validated controller and retained source selection.
pub(super) fn quote_candidate(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    prompt: &MlxModelInput,
    geometry: InferenceGeometry,
    config: TextGenerationConfig,
    workspace: TextControllerWorkspace<'_>,
    storage: &ControllerStorageContract,
    claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
    capacity: eredu_core::MemoryLimits,
    original_table: bool,
    retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    capture: Option<&CaptureAdmission<'_>>,
) -> Result<TextWorkspaceCandidate, Error> {
    let mut recipe = prompt.quote_original_media_recipe_funded(
        runtime,
        geometry,
        config.clone(),
        workspace.filter,
        capacity,
        capture
            .map(|capture| capture.bind_geometry(geometry))
            .transpose()?,
        capture.and_then(CaptureAdmission::intervention_quote),
    )?;
    if recipe.has_missing_operation() {
        return Err(recipe.into_pending_admission());
    }
    recipe.quote_complete(
        runtime,
        config.clone(),
        workspace,
        storage,
        claim,
        original_table,
        retained_sources,
        capture,
    )
}

fn validate<'s>(
    runtime: &'s ModelRuntime<MlxBackend<'_>>,
    prompt: &'s MlxModelInput,
    config: TextGenerationConfig,
    claim: Option<&eredu_core::GenerationSequencePreparation<'_, '_>>,
) -> Result<AuthenticatedPreparedSource<'s>, R> {
    let claim = claim.ok_or(R::MissingSequence)?;
    if claim.request().token_input().is_some()
        || config.sampling().max_new_tokens != Some(claim.request().max_new_tokens())
        || claim.context().attempt() != 0
    {
        return Err(R::RequestMismatch);
    }
    let session = runtime.session();
    let pool = runtime.backend().memory_ledger();
    let input::OriginalMediaPacket::Original(source) =
        prompt.original_media.as_ref().ok_or(R::SourceUnavailable)?
    else {
        return Err(R::SourceUnavailable);
    };
    if !matches!(
        prompt.parts,
        super::super::pending_prompt::ModelInputParts::Original(_)
    ) || prompt.memory_owner.is_some()
        || prompt.quote.is_some()
        || prompt.inference_request.is_some()
    {
        return Err(R::SourceUnavailable);
    }
    let cache = prompt.cache_identity.as_ref().ok_or(R::SourceUnavailable)?;
    let accepted = source.validate_request_source(pool, &prompt.parts, cache)?;
    if !session.payload.memory_ledger.same_ledger(pool)
        || !runtime
            .backend()
            .matches_prepared_target(&session.payload.target)
    {
        return Err(R::IdentityMismatch);
    }
    if session.poison.get() {
        return Err(R::SourceUnavailable);
    }
    let authority = session.authority.try_borrow().map_err(|_| R::Busy)?;
    authority.require_idle().map_err(|_| R::Busy)?;
    drop(authority);
    let current = session
        .payload
        .model
        .erased()
        .original_request_media_binding()
        .map_err(|error| match error {
            WorkingMemoryError::Overflow => R::Overflow,
            WorkingMemoryError::ResetAdmissionBusy => R::Busy,
            _ => R::SourceUnavailable,
        })?;
    if !accepted.matches(&current) || current.frontier() != 0 {
        return Err(R::IdentityMismatch);
    }
    let shape = source.shape();
    if shape[0] != 1 || shape[1] == 0 {
        return Err(R::IdentityMismatch);
    }
    let output = u64::try_from(claim.request().max_new_tokens()).map_err(|_| R::Overflow)?;
    shape[1].checked_add(output).ok_or(R::Overflow)?;
    let model = &session.payload.model;
    Ok(AuthenticatedPreparedSource {
        prompt,
        semantics: source.borrowed_semantics(),
        capability: model.erased().capability_estimate(),
        blueprint: model.inference_blueprint().ok_or(R::SourceUnavailable)?,
        executable: model,
        dtype: session.floating_state_dtype_bytes(),
        binding: accepted,
    })
}

// Private closed source authentication, before capture and fact derivation. A
// borrow of the exact prompt keeps all B/native/source custody alive; none of
// these views can bind a second B or escape as an accepted request certificate.
struct AuthenticatedPreparedSource<'s> {
    prompt: &'s MlxModelInput,
    semantics: &'s eredu_architectures::media_plan::BoundPreparedMediaSemantics,
    capability: &'s eredu_architectures::capability::CapabilityEstimate,
    blueprint: &'s eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
    executable: &'s crate::composition::mlx::model::Executable,
    dtype: std::num::NonZeroU8,
    binding: &'s eredu_runtime::working_memory::MediaSessionBinding,
}
struct OriginalPreparedColdFacts<'s> {
    source: AuthenticatedPreparedSource<'s>,
    positions: eredu_architectures::media_plan::PreparedMediaPositionFacts<'s>,
    state: eredu_core::RuntimeStateFacts<'s>,
    state_slots: Option<crate::backend::runtime::cache::state::NativeStateSlotCounts>,
    // Borrowed protocol output geometry comes from the retained architecture.
    // Neither this width, logical bytes nor slots prove native backing.
    output_width: Option<usize>,
    native_state_backing_bytes: Option<u64>,
    native_media_workspace_bytes: Option<u64>,
}
fn semantic(error: eredu_architectures::media_plan::MediaSemanticError) -> R {
    if error.is_overflow() {
        R::Overflow
    } else {
        R::IdentityMismatch
    }
}
impl<'s> AuthenticatedPreparedSource<'s> {
    fn inspect(self, config: TextGenerationConfig) -> Result<OriginalPreparedColdFacts<'s>, R> {
        let positions = self.semantics.request_position_facts().map_err(semantic)?;
        if positions.decoder_positions() == 0 || self.binding.frontier() != 0 {
            return Err(R::IdentityMismatch);
        }
        // This is precisely the existing logical F32 media equation. Its source
        // bytes and scalar counts are separate from B3's complete native interval.
        let input = positions
            .legacy_input_accounting(std::num::NonZeroU8::new(4).unwrap())
            .map_err(semantic)?;
        let output = config
            .sampling()
            .max_new_tokens
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(R::Overflow)?;
        let state = eredu_core::estimate_runtime_state_facts(
            self.capability.state_layout(),
            input,
            output,
            1,
            self.dtype,
        )
        .map_err(|error| match error {
            eredu_core::AdmissionPolicyError::ArithmeticOverflow { .. } => R::Overflow,
            _ => R::IdentityMismatch,
        })?;
        let state_slots = self
            .executable
            .erased()
            .original_state_slot_facts()
            .map_err(|_| R::SourceUnavailable)?;
        let output_width = self.blueprint.architecture().text_output_width();
        Ok(OriginalPreparedColdFacts {
            source: self,
            positions,
            state,
            state_slots,
            output_width,
            native_state_backing_bytes: None,
            native_media_workspace_bytes: None,
        })
    }
}
impl OriginalPreparedColdFacts<'_> {
    fn missing_contribution(self) -> R {
        #[cfg(test)]
        LAST_COLD_FACTS.set(Some(ColdFactsObservation {
            canonical: self.positions.canonical_tokens(),
            projected_text: self.positions.non_tokenized_text_positions(),
            media: self.positions.media_positions(),
            decoder: self.positions.decoder_positions(),
            requested: self.state.requested_positions,
            fixed_bytes: self.state.fixed_state_bytes,
            context_bytes: self.state.context_state_bytes,
            window_count: self.state.sliding_windows.len(),
            capability: self.source.capability as *const _ as usize,
            blueprint: self.source.blueprint as *const _ as usize,
            source: self.positions.source() as *const _ as usize,
            state_slots: self.state_slots,
            complete: self.state.completeness == eredu_core::EstimationCompleteness::Complete,
            output_width: self.output_width,
            native_backing: self.native_state_backing_bytes,
            native_workspace: self.native_media_workspace_bytes,
        }));

        // Keep the actual source/selection/capability borrow through all facts.
        // No cloned blueprint, ordinary diagnostic owner, native projection or
        // workspace factory is constructed by this path.
        let _source = (
            &self.source.prompt,
            self.source.blueprint,
            self.source.capability,
            self.positions.source(),
            self.state,
            self.state_slots,
        );
        debug_assert!(self.native_state_backing_bytes.is_none());
        debug_assert!(self.native_media_workspace_bytes.is_none());
        R::MissingInspectionStorage
    }
}

// Fixed test observation only. It neither changes production branching nor
// introduces a successful native admission path. No observer Vec is allocated.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(in crate::composition::mlx::session::model_session) struct ColdFactsObservation {
    pub canonical: u64,
    pub projected_text: u64,
    pub media: u64,
    pub decoder: u64,
    pub requested: u64,
    pub fixed_bytes: u64,
    pub context_bytes: u64,
    pub window_count: usize,
    pub capability: usize,
    pub blueprint: usize,
    pub source: usize,
    pub state_slots: Option<crate::backend::runtime::cache::state::NativeStateSlotCounts>,
    pub complete: bool,
    pub output_width: Option<usize>,
    pub native_backing: Option<u64>,
    pub native_workspace: Option<u64>,
}
#[cfg(test)]
thread_local! { static LAST_COLD_FACTS: std::cell::Cell<Option<ColdFactsObservation>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub(in crate::composition::mlx::session::model_session) fn take_original_cold_facts()
-> Option<ColdFactsObservation> {
    LAST_COLD_FACTS.take()
}

impl TextExecutionQuoteOwner {
    pub(in crate::composition::mlx::session::model_session) fn has_completed_input(&self) -> bool {
        self.prepared_source.is_some()
    }

    /// Snapshot entry borrows an already bound exact B. It cannot recreate the
    /// one-shot bind or substitute equal descriptors from another input owner.
    pub(in crate::composition::mlx::session::model_session) fn validate_completed_pending_input(
        &self,
        prompt: &MlxModelInput,
    ) -> Result<(), WorkingMemoryError> {
        let source = self
            .prepared_source
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let Some(input::OriginalMediaPacket::Original(packet)) = prompt.original_media.as_ref()
        else {
            return Err(WorkingMemoryError::IdentityMismatch);
        };
        let geometry = self.request().geometry();
        if !packet.matches_workspace_source(&source.source)
            || packet.shape() != [geometry.batch_size, geometry.input_positions]
            || geometry.cached_positions != 0
            || prompt.memory_owner.is_some()
            || !matches!(
                prompt.parts,
                super::super::pending_prompt::ModelInputParts::Original(_)
            )
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        packet
            .validate_request_source(
                source.source.pool(),
                &prompt.parts,
                prompt
                    .cache_identity
                    .as_ref()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?,
            )
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        Ok(())
    }

    /// Moves the exact completed B into the existing one-use prompt binding.
    /// No native constructor, input copy, source rebind or new account is needed.
    pub(in crate::composition::mlx::session::model_session) fn bind_completed_input_prompt(
        &self,
        backend: &MlxBackend<'_>,
        preparation: &InferenceTextPreparation,
        mut prompt: MlxModelInput,
    ) -> Result<MlxModelInput, Error> {
        let mismatch = || Error::PrefillControl(WorkingMemoryError::IdentityMismatch);
        self.validate_original_prompt_preflight(backend, preparation)
            .map_err(|_| mismatch())?;
        let source = self.prepared_source.as_ref().ok_or_else(mismatch)?;
        let Some(input::OriginalMediaPacket::Original(packet)) = prompt.original_media.as_ref()
        else {
            return Err(mismatch());
        };
        let geometry = self.request().geometry();
        if !source.source.pool().same_ledger(backend.memory_ledger())
            || !packet.matches_workspace_source(&source.source)
            || packet.shape() != [geometry.batch_size, geometry.input_positions]
            || geometry.cached_positions != 0
            || prompt.quote.is_some()
            || prompt.inference_request.is_some()
            || prompt.memory_owner.is_some()
            || !matches!(
                prompt.parts,
                super::super::pending_prompt::ModelInputParts::Original(_)
            )
        {
            return Err(mismatch());
        }
        let cache = prompt.cache_identity.as_ref().ok_or_else(mismatch)?;
        packet
            .validate_request_source(backend.memory_ledger(), &prompt.parts, cache)
            .map_err(|_| mismatch())?;
        // The existing preparation explicitly permits already constructed inputs.
        // Publish READY exactly once, before attaching any accepted owner alias.
        preparation.bind_prompt().map_err(Error::PrefillControl)?;
        prompt.prefill_chunk_positions =
            std::num::NonZeroU64::new(geometry.prefill_chunk_positions);
        prompt.inference_request = Some(preparation.request().clone());
        prompt.quote = Some(self.clone());
        Ok(prompt)
    }
}

pub(super) fn prompt_binding_control_bytes() -> Option<u64> {
    [
        std::mem::size_of::<MlxModelInput>(),
        std::mem::size_of::<Result<MlxModelInput, Error>>(),
        std::mem::size_of::<InferenceGeometry>(),
        std::mem::size_of::<Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>>(),
        std::mem::size_of::<Option<&eredu_runtime::SharedPreparedInputCacheIdentity>>(),
        std::mem::size_of::<Result<&eredu_runtime::working_memory::MediaSessionBinding, R>>(),
        std::mem::size_of::<(
            &TextExecutionQuoteOwner,
            &MlxBackend<'_>,
            &InferenceTextPreparation,
        )>(),
        std::mem::size_of::<Result<(), WorkingMemoryError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|bytes| u64::try_from(bytes).ok())
}

/// Actual source observation retained from the candidate that won admission.
/// The context shares its original cumulative ledger; no new account or census.
#[derive(Debug)]
pub(in crate::composition::mlx::session::model_session) struct PreparedMediaQuoteSource {
    source: eredu_runtime::input::OriginalPreparedWorkspaceSource,
    copied_semantics:
        std::cell::OnceCell<eredu_architectures::media_plan::BoundPreparedMediaSemantics>,
    context: eredu_nn::workspace::WorkspaceContext,
}
impl PreparedMediaQuoteSource {
    pub(in crate::composition::mlx::session::model_session) fn new(
        source: eredu_runtime::input::OriginalPreparedWorkspaceSource,
        context: eredu_nn::workspace::WorkspaceContext,
    ) -> Self {
        Self {
            source,
            copied_semantics: std::cell::OnceCell::new(),
            context,
        }
    }
    pub(super) fn publication_source(
        &self,
    ) -> eredu_runtime::input::OriginalPreparedWorkspaceSource {
        self.source.clone()
    }
}
impl TextExecutionQuoteOwner {
    pub(in crate::composition::mlx::session::model_session) fn copied_media_semantics(
        &self,
    ) -> Option<&eredu_architectures::media_plan::BoundPreparedMediaSemantics> {
        self.prepared_source.as_ref()?.copied_semantics.get()
    }

    pub(in crate::composition::mlx::session::model_session) fn seal_copied_media(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        semantics: &eredu_architectures::media_plan::BoundPreparedMediaSemantics,
        transition: &eredu_runtime::working_memory::CopiedMediaStateBinding,
    ) -> Result<(), Error> {
        let source = self
            .prepared_source
            .as_ref()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        source
            .context
            .charge_metadata(std::mem::size_of::<(
                eredu_architectures::media_plan::BoundPreparedMediaSemantics,
                Result<
                    eredu_architectures::media_plan::BoundPreparedMediaSemantics,
                    WorkingMemoryError,
                >,
                eredu_runtime::working_memory::MediaSessionBinding,
                Result<(), eredu_architectures::media_plan::BoundPreparedMediaSemantics>,
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        let current = runtime
            .session()
            .payload
            .model
            .erased()
            .original_request_media_binding()
            .map_err(Error::PrefillControl)?;
        if !transition.matches_installed(&current) || source.copied_semantics.get().is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let semantics = semantics
            .for_copied_state(transition)
            .map_err(Error::PrefillControl)?;
        source
            .copied_semantics
            .set(semantics)
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
    }

    pub(in crate::composition::mlx::session::model_session) fn execution_metadata(
        &self,
    ) -> Option<&eredu_nn::workspace::WorkspaceContext> {
        self.prepared_source
            .as_ref()
            .map(|source| &source.context)
            .or(self.continuation_metadata.as_ref())
    }
}
