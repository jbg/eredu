//! Semantic and cursor copies join the existing native snapshot transaction.
use super::*;
use crate::api::GenerationSnapshotError;
use eredu_runtime::{
    execution_control::{
        SnapshotBudget, TextContinuationSnapshot, TextSnapshotBackend, TextSnapshotError, PreparedTextHostJournal,
    },
    working_memory::WorkspaceCopyLimits,
};

type SnapshotEnclosure<B, J> = (
                PreparedChatSnapshot<B>,
                GenerationSnapshotError,
                Result<(PreparedChatSnapshot<B>, <J as PreparedTextHostJournal>::Copied), GenerationSnapshotError>,
                Result<(PreparedChatSnapshot<B>, <J as PreparedTextHostJournal>::Copied), TextSnapshotError<BackendFailure>>,
                J,
                <J as PreparedTextHostJournal>::Copied,
                Duration,
                Option<Duration>,
                WorkspaceCopyLimits,

);
/// Prospective execution limits for an exact saved continuation. Sampling,
/// randomness, penalties and adaptive history come from the saved state.
#[derive(Debug, Clone, Copy, Default)]
pub struct PreparedChatResumeSettings {
    /// May shorten the saved output allowance; omission preserves it.
    pub max_new_tokens: Option<usize>,
    /// Omission preserves the saved execution policy. A replacement is admitted
    /// before restoring and must retain the same managed memory domain capacity.
    pub inference: Option<eredu_core::TextInferencePolicy>,
}

/// Immutable committed chat state. Native state, semantic parser and cursor
/// retain their original copy accounts. Restoration and branching use fresh
/// admission and share the snapshot's cumulative copy budget.
pub struct PreparedChatSnapshot<B: TextSnapshotBackend> {
    pub(super) driver: eredu_core::TextDriverIdentity,
    state: TextContinuationSnapshot<B, ChatController>,
    semantic: SemanticStateOwner,
    effective_config: TextGenerationConfig,
    cursor: Cursor,
    lifecycle: eredu_runtime::execution_control::GenerationBoundary,
    attribution: Option<eredu_core::SharedPromptAttribution>,
    active: Duration,
    time_to_first_token: Option<Duration>,
    input_funding: Option<eredu_runtime::input::OriginalModelInputCustody>,
    preparation: PreparedSemanticSource,
}
impl<B: TextSnapshotBackend> PreparedChatSnapshot<B> {
    pub(crate) fn storage_reservation(&self) -> &eredu_runtime::execution_control::SnapshotReservation {
        self.state.storage_reservation()
    }
    pub(crate) fn host_preparation(&self) -> &eredu_core::HostPreparationAuthority { self.state.host_preparation() }
    pub(crate) fn effective_config(&self) -> TextGenerationConfig { self.effective_config }
    pub(crate) fn pending_forced_token(&self) -> Option<u32> { self.state.controller().pending_forced() }
    pub(crate) fn status(&self) -> eredu_core::execution_control::GenerationStatus {
        eredu_runtime::execution_control::GenerationLifecycle::fork(&self.lifecycle).status()
    }
    pub(super) fn lifecycle(&self) -> &eredu_runtime::execution_control::GenerationBoundary { &self.lifecycle }
    /// Exact committed prefix at the saved boundary.
    pub fn token_ids(&self) -> &[u32] {
        self.cursor.token_ids()
    }
    /// Terminal status of that saved prefix.
    pub fn finish_reason(&self) -> Option<eredu_core::FinishReason> {
        self.cursor.finish_reason()
    }
    /// Logical storage held by the shared snapshot reservation.
    pub fn retained_bytes(&self) -> u64 {
        self.state.retained_bytes()
    }
    /// Absolute prediction index; copying never resets the observation schedule.
    pub fn next_prediction(&self) -> u64 {
        self.state.next_prediction()
    }
    /// Remaining output allowance; restoration cannot increase it.
    pub fn remaining_tokens(&self) -> Option<usize> {
        self.state.remaining_tokens()
    }
}
impl<B: TextSnapshotBackend> PreparedChatSession<'_, B> {
    /// Cold estimate from the exact original native source and cursor/parser
    /// copy provider. The eventual snapshot rechecks it with its record journal.
    pub(crate) fn snapshot_estimate(&mut self, capacity: u64)
        -> Result<eredu_core::execution_control::SnapshotEstimate, GenerationSnapshotError> {
        use eredu_runtime::execution_control::{PreparedTextHostCopy, SnapshotTokenController};
        use eredu_core::execution_control::ExecutionControlError;
        let result = (|| {
            self.lifecycle.checkpoint().map_err(TextSnapshotError::Control)?;
            let mut boundary = self.source.generator.snapshot_source().map_err(|_| TextSnapshotError::Unsupported("chat source is not a completed drained boundary"))?;
            let controller = boundary.controller().original_snapshot_storage_bytes()
                .ok_or(ExecutionControlError::UnknownEstimate)?;
            let (runtime, state, pending) = boundary.parts();
            let preparation = B::original_saved_generation_preparation_bytes(runtime, state, pending.as_ref().map(|input| match input {
                eredu_core::PendingTextInput::Prefill(prompt) => eredu_core::PendingTextInput::Prefill(*prompt),
                eredu_core::PendingTextInput::Decode(token) => eredu_core::PendingTextInput::Decode(*token),
            }))
                .map_err(TextSnapshotError::HostAdmission)?.ok_or(ExecutionControlError::UnknownEstimate)?;
            let native = B::original_generation_snapshot_estimates(runtime, state, pending)
                .ok_or(ExecutionControlError::UnknownEstimate)?;
            let pool = B::original_snapshot_host_pool(runtime).ok_or(ExecutionControlError::UnknownEstimate)?;
            let host = self.cursor.prepare_semantic_snapshot_host_copy::<B, ChatController, SnapshotEnclosure<B, ()>, _>(
                &self.semantic, pool, capacity.min(self.preparation.capacity_bytes()), preparation, ())
                .map_err(TextSnapshotError::HostAdmission)?;
            let bytes = host.storage_bytes().ok_or(ExecutionControlError::UnknownEstimate)?
                .checked_add(controller).and_then(|n| n.checked_add(std::mem::size_of::<TextContinuationSnapshot<B, ChatController>>() as u64))
                .ok_or(ExecutionControlError::Overflow)?;
            let mut total = eredu_core::execution_control::SnapshotEstimate { retained_bytes: bytes, copy_bytes: bytes };
            for estimate in native {
                total.retained_bytes = total.retained_bytes.checked_add(estimate.retained_bytes).ok_or(ExecutionControlError::Overflow)?;
                total.copy_bytes = total.copy_bytes.checked_add(estimate.copy_bytes).ok_or(ExecutionControlError::Overflow)?;
            }
            Ok(total)
        })();
        result.map_err(GenerationSnapshotError)
    }
    /// Copies a completed boundary using the existing nonrefundable snapshot
    /// transaction. Missing native copy bounds refuse before destination work.
    /// An unsuccessful copy leaves this live session unchanged.
    pub fn snapshot(
        &mut self,
        budget: &SnapshotBudget,
        host_capacity_bytes: u64,
        native: WorkspaceCopyLimits,
    ) -> Result<PreparedChatSnapshot<B>, GenerationSnapshotError> {
        self.snapshot_with_host(budget, host_capacity_bytes, native, ())
            .map(|(snapshot, ())| snapshot)
    }
    /// Composes a concrete borrowed record journal into the same original host
    /// destination and nonrefundable snapshot transaction as the cursor/parser.
    pub(crate) fn snapshot_with_host<J: PreparedTextHostJournal>(
        &mut self,
        budget: &SnapshotBudget,
        capacity: u64,
        native: WorkspaceCopyLimits,
        journal: J,
    ) -> Result<(PreparedChatSnapshot<B>, J::Copied), GenerationSnapshotError> {
        self.snapshot_inner(budget, capacity, native, journal).map_err(GenerationSnapshotError)
    }
    fn snapshot_inner<J: PreparedTextHostJournal>(
        &mut self,
        budget: &SnapshotBudget,
        capacity: u64,
        native: WorkspaceCopyLimits,
        journal: J,
    ) -> Result<(PreparedChatSnapshot<B>, J::Copied), TextSnapshotError<BackendFailure>> {
        let lifecycle = self.lifecycle.checkpoint().map_err(TextSnapshotError::Control)?;
        let driver = self.source.generator.driver_identity();
        let mut boundary = self.source.generator.snapshot_source().map_err(|error| {
            use eredu_core::TextContinuationError as E;
            match error {
                E::Generation(ControlledTextGenerationError::Preparation(cause)) => {
                    TextSnapshotError::HostPreparation(cause)
                }
                E::Generation(ControlledTextGenerationError::Backend(cause)) => {
                    TextSnapshotError::Backend(B::into_backend_failure(cause))
                }
                E::Generation(ControlledTextGenerationError::Controller(_)) => {
                    TextSnapshotError::Unsupported("chat controller boundary failed")
                }
                E::IncompatibleDriver | E::Failed | E::NotQuiescent => {
                    TextSnapshotError::Unsupported(
                        "chat source is not a completed drained boundary",
                    )
                }
            }
        })?;
        let (runtime, state, pending) = boundary.parts();
        let native_bytes = B::original_saved_generation_preparation_bytes(runtime, state, pending)
            .map_err(TextSnapshotError::HostAdmission)?
            .ok_or(TextSnapshotError::Unsupported(
                "original native copy preparation",
            ))?;
        let pool = B::original_snapshot_host_pool(runtime).ok_or(
            TextSnapshotError::Unsupported("original snapshot host domain"),
        )?;
        let host = self
            .cursor
            .prepare_semantic_snapshot_host_copy::<B, ChatController, SnapshotEnclosure<B, J>, _>(
                &self.semantic,
                pool,
                capacity.min(self.preparation.capacity_bytes()),
                native_bytes,
                journal,
            )
            .map_err(TextSnapshotError::HostAdmission)?;
        let (state, (cursor, semantic, journal)) =
            TextContinuationSnapshot::capture_original_host(&mut boundary, budget, host, native)
                .map_err(|error| {
                    crate::api::control::map_snapshot_failure(error, B::into_backend_failure)
                })?;
        Ok((PreparedChatSnapshot {
            driver,
            state,
            semantic,
            effective_config: self.effective_config,
            cursor,
            lifecycle,
            attribution: self.attribution.clone(),
            active: self.active,
            time_to_first_token: self.source.time_to_first_token,
            input_funding: self.input_funding.clone(),
            preparation: self.preparation.clone(),
        }, journal))
    }
}
impl<B> LoadedModel<B>
where
    B: OriginalChatBackend
        + TextSnapshotBackend
        + eredu_core::TextResumeBackend<
            ResumeSource = <B as TextSnapshotBackend>::SavedTextComponents,
        >,
{
    /// Restores a saved committed chat prefix through the ordinary driver.
    /// Omitted output count uses the saved allowance; overrides may shorten it.
    /// Attach a new borrowed observer to the returned session when required.
    pub fn restore_prepared_chat<'a>(
        &'a mut self,
        snapshot: &PreparedChatSnapshot<B>,
        settings: PreparedChatResumeSettings,
        host_capacity_bytes: u64,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<PreparedChatSession<'a, B>>, PreparedChatSessionError> {
        self.resume_prepared_chat(snapshot, settings, host_capacity_bytes, false, cancellation)
    }
    /// Starts an independent branch through the same restore worker, consuming
    /// its branch slot and cumulative copy allowance without changing the source.
    pub fn fork_prepared_chat<'a>(
        &'a mut self,
        snapshot: &PreparedChatSnapshot<B>,
        settings: PreparedChatResumeSettings,
        host_capacity_bytes: u64,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<PreparedChatSession<'a, B>>, PreparedChatSessionError> {
        self.resume_prepared_chat(snapshot, settings, host_capacity_bytes, true, cancellation)
    }
    fn resume_prepared_chat<'a>(
        &'a mut self,
        snapshot: &PreparedChatSnapshot<B>,
        settings: PreparedChatResumeSettings,
        host_capacity: u64,
        branch: bool,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<PreparedChatSession<'a, B>>, PreparedChatSessionError> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        let funding = snapshot.preparation.metadata_funding().clone();
        let retain = |cause| PreparedChatSessionError {
            cause,
            funding: Some(funding.clone()),
            input_funding: snapshot.input_funding.clone(),
        };
        let prepared = (|| -> Result<_, Cause> {
            if !snapshot.preparation.tokenizer().matches_configuration(&self.tokenizer) {
                return Err(TokenInputRejection::IdentityMismatch.into());
            }
            snapshot.resume_configuration(&self.runtime, settings, cancellation)
                .map(|config| config.map(|config| (config, host_capacity.min(snapshot.preparation.capacity_bytes()))))
        })().map_err(&retain)?;
        let Some((config, capacity)) = prepared else {
            return Ok(None);
        };
        snapshot
            .resume_with_host(&mut self.runtime, config, capacity, &eredu_core::OriginalTextResumeOptions {
                terminal: snapshot.finish_reason().is_some(),
                ..eredu_core::OriginalTextResumeOptions::new(if branch { eredu_core::OriginalTextResumeKind::Branch }
                    else { eredu_core::OriginalTextResumeKind::Restore })
            }, cancellation, ())
            .map(|result| result.map(|(session, ())| session))
            .map_err(|cause| retain(Cause::Snapshot(cause)))
    }
}
impl<B> PreparedChatSnapshot<B>
where
    B: TextSnapshotBackend
        + eredu_core::TextResumeBackend<
            ResumeSource = <B as TextSnapshotBackend>::SavedTextComponents,
        >,
{
    /// Saved capture consumption and sampler facts from the actual immutable
    /// source. Reading these scalar facts grants no restore or execution authority.
    pub fn resume_source_facts(&self) -> Option<eredu_core::TextResumeSourceFacts> {
        self.state.resume_source_facts()
    }
    pub(super) fn resume_configuration(
        &self, runtime: &ModelRuntime<B>, settings: PreparedChatResumeSettings,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<TextGenerationConfig>, Cause>
    where B: OriginalChatBackend,
    {
        if cancellation.is_cancelled() { return Ok(None) }
        let inference = settings.inference.unwrap_or(self.effective_config.inference_policy());
        if inference.managed_memory_capacity_bytes != Some(self.preparation.capacity_bytes()) {
            return Err(TokenInputRejection::IdentityMismatch.into());
        }
        B::validate_semantic_source(runtime, &self.preparation)?;
        let mut sampling = self.effective_config.sampling();
        sampling.max_new_tokens = if self.finish_reason().is_some() { Some(0) } else { settings.max_new_tokens.or(self.remaining_tokens()) };
        if sampling.max_new_tokens == Some(0) && self.finish_reason().is_none() { return Ok(None) }
        Ok(Some(saved_configuration(self.effective_config, sampling, self.effective_config.seed(), inference)))
    }
    pub(crate) fn resume_with_host<'a, J: PreparedTextHostJournal>(
        &self,
        runtime: &'a mut ModelRuntime<B>,
        config: TextGenerationConfig,
        capacity: u64,
        options: &eredu_core::OriginalTextResumeOptions<'_>,
        cancellation: &GenerationCancellationToken,
        journal: J,
    ) -> Result<Option<(PreparedChatSession<'a, B>, J::Copied)>, TextSnapshotError<BackendFailure>> {
        self.resume_parts_with_host(runtime, config, capacity, options, cancellation, journal)
            .map(|result| result.map(|(generation, displaced, host, copied)| {
                drop(displaced);
                (PreparedChatSession::from_host(generation, host), copied)
            }))
    }
    pub(super) fn resume_parts_with_host<'a, J: PreparedTextHostJournal>(
        &self, runtime: &'a mut ModelRuntime<B>, config: TextGenerationConfig,
        capacity: u64, options: &eredu_core::OriginalTextResumeOptions<'_>, cancellation: &GenerationCancellationToken, journal: J,
    ) -> Result<Option<(ControlledTextGeneration<'a, B, ChatController>, B::DisplacedState,
        super::branch::PreparedChatHost, J::Copied)>, TextSnapshotError<BackendFailure>> {
        let started = Instant::now();
        let bytes = self
            .state
            .original_resume_preparation_bytes(runtime, config, options)
            .map_err(|error| {
                crate::api::control::map_snapshot_failure(error, B::into_backend_failure)
            })?;
        let pool = B::original_snapshot_host_pool(runtime).ok_or(
            TextSnapshotError::Unsupported("original resume host domain"),
        )?;
        let host = self
            .cursor
            .prepare_semantic_resume_host_copy::<B, ChatController, (
                PreparedChatSession<'a, B>,
                ScopedBackendGenerationTokenSource<'_, '_, '_, B, ChatController>,
                eredu_core::TextTokenChoiceBoundary<'_, ChatController>,
                eredu_core::TextSamplingBoundary<'_, B>,
                eredu_core::execution_control::SamplingOverride,
                eredu_core::execution_control::SamplingStateFacts,
                Result<eredu_core::execution_control::SamplingStateFacts,
                    eredu_core::execution_control::SamplingOverrideError<B::Error>>,
                PreparedChatSessionError,
                Result<Option<PreparedChatSession<'a, B>>, PreparedChatSessionError>,
                Result<Option<(PreparedChatSession<'a, B>, J::Copied)>, TextSnapshotError<BackendFailure>>,
                J,
                J::Copied,
                Duration,
                Option<Duration>,
                (TextGenerationConfig, eredu_core::ResolvedGenerationConfig, u64, eredu_core::TextInferencePolicy),
                PreparedChatResumeSettings,
                crate::api::request::TokenDeliveryFacts,
                super::branch::PreparedChatHost,
                eredu_core::TextGenerationBranch<B, ChatController>,
                super::branch::PreparedChatBranch<B>,
                B::DisplacedState,
                eredu_core::OriginalTextResumeOptions<'_>,
                bool,
            ), _>(&self.semantic, pool, capacity.min(self.preparation.capacity_bytes()), bytes, journal)
            .map_err(TextSnapshotError::HostAdmission)?;
        let result = self.state.resume_original_host_with_displaced(runtime, config, host, cancellation,
            options)
        .map_err(|error| {
            crate::api::control::map_snapshot_failure(error, B::into_backend_failure)
        })?;
        Ok(result.map(|(generator, displaced, (mut cursor, semantic, journal))| {
            let effective_config = {
                let facts = generator.resume_facts();
                let mut sampling = config.sampling();
                sampling.temperature = facts.sampling_after.temperature;
                sampling.do_sample = facts.sampling_after.temperature > 0.0;
                saved_configuration(config, sampling,
                    options.sampling.and_then(|request| request.reseed).unwrap_or(config.seed()),
                    config.inference_policy())
            };
            if let Some(remaining) = config.sampling().max_new_tokens {
                cursor.restrict_remaining(remaining);
            }
            (generator, displaced, super::branch::PreparedChatHost {
                semantic,
                effective_config,
                cursor,
                lifecycle: eredu_runtime::execution_control::GenerationLifecycle::fork(&self.lifecycle),
                attribution: self.attribution.clone(),
                active: self.active + started.elapsed(),
                time_to_first_token: self.time_to_first_token,
                input_funding: self.input_funding.clone(),
                preparation: self.preparation.clone(),
            }, journal)
        }))
    }
}
