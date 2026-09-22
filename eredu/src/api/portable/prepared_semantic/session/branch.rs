//! Serial branches reuse the same ordinary machine and semantic host state.
use super::*;
use eredu_runtime::execution_control::{
    PreparedTextHostJournal, TextSnapshotBackend, TextSnapshotError,
};

pub(super) struct PreparedChatHost {
    pub(super) semantic: SemanticStateOwner,
    pub(super) effective_config: TextGenerationConfig,
    pub(super) active: Duration,
    pub(super) cursor: Cursor,
    pub(super) lifecycle: eredu_runtime::execution_control::GenerationLifecycle,
    pub(super) attribution: Option<eredu_core::SharedPromptAttribution>,
    pub(super) time_to_first_token: Option<Duration>,
    pub(super) input_funding: Option<eredu_runtime::input::OriginalModelInputCustody>,
    pub(super) preparation: PreparedSemanticSource,
}

/// An inactive serial branch retains its own native, sampler, semantic, output
/// and cumulative budget state. Only its originating exclusive session can
/// exchange it into the loaded executable; dropping it retires that branch.
pub struct PreparedChatBranch<B: TextSnapshotBackend> {
    machine: eredu_core::TextGenerationBranch<B, ChatController>,
    host: PreparedChatHost,
}
impl<B: TextSnapshotBackend> PreparedChatBranch<B> {
    pub(crate) fn resume_facts(&self) -> eredu_core::TextResumeFacts<'_>
    where
        B: eredu_core::TextResumeBackend,
    {
        self.machine.resume_facts()
    }
    /// Exact committed prefix, excluding any pending token restriction.
    pub fn token_ids(&self) -> &[u32] {
        self.host.cursor.token_ids()
    }
    /// Saved completed-token lifecycle.
    pub fn status(&self) -> eredu_core::execution_control::GenerationStatus {
        self.host.lifecycle.status()
    }
    /// Absolute next prediction, without resetting inherited budgets.
    pub fn next_prediction(&self) -> u64 {
        self.host.lifecycle.next_prediction()
    }
    /// Prospective canonical token restriction inherited by the branch.
    pub fn pending_forced_token(&self) -> Option<u32> {
        self.machine.controller().pending_forced()
    }
    pub(crate) fn effective_config(&self) -> TextGenerationConfig {
        self.host.effective_config.clone()
    }
}
impl<'a, B: TextGenerationBackend> PreparedChatSession<'a, B> {
    pub(super) fn from_host(
        generator: ControlledTextGeneration<'a, B, ChatController>,
        host: PreparedChatHost,
    ) -> Self {
        let PreparedChatHost {
            semantic,
            effective_config,
            active,
            cursor,
            lifecycle,
            attribution,
            time_to_first_token,
            input_funding,
            preparation,
        } = host;
        Self {
            source: BackendGenerationTokenSource {
                generator,
                on_token: None,
                delivery_failure: None,
                generation_started: Instant::now(),
                time_to_first_token,
            },
            semantic,
            effective_config,
            active,
            cursor,
            lifecycle,
            attribution,
            input_funding,
            preparation,
        }
    }
    fn exchange_host(&mut self, host: &mut PreparedChatHost) {
        let Self {
            source,
            semantic,
            effective_config,
            active,
            cursor,
            lifecycle,
            attribution,
            input_funding,
            preparation,
        } = self;
        std::mem::swap(semantic, &mut host.semantic);
        std::mem::swap(effective_config, &mut host.effective_config);
        std::mem::swap(active, &mut host.active);
        std::mem::swap(cursor, &mut host.cursor);
        std::mem::swap(lifecycle, &mut host.lifecycle);
        std::mem::swap(attribution, &mut host.attribution);
        std::mem::swap(
            &mut source.time_to_first_token,
            &mut host.time_to_first_token,
        );
        std::mem::swap(input_funding, &mut host.input_funding);
        std::mem::swap(preparation, &mut host.preparation);
    }
}
impl<B> PreparedChatSession<'_, B>
where B: OriginalChatBackend + TextSnapshotBackend + eredu_core::TextResumeBackend<
    ResumeSource = <B as TextSnapshotBackend>::SavedTextComponents,
    DisplacedState = <B as eredu_core::execution_control::NativeTextStateBackend>::NativeTextState>,
{
    /// Restores a snapshot from this exclusive branch tree through the same
    /// original copy transaction. Observer attachments remain on this session.
    pub fn restore_snapshot(
        &mut self, snapshot: &PreparedChatSnapshot<B>, settings: PreparedChatResumeSettings,
        capacity: eredu_core::MemoryLimitDeclarations, cancellation: &GenerationCancellationToken,
    ) -> Result<bool, PreparedChatSessionError> {
        self.restore_with_host(snapshot, settings, capacity, cancellation, ()).map(|result| result.is_some())
    }
    /// Creates an inactive serial child without advancing or replacing the parent.
    pub fn fork_snapshot(
        &mut self, snapshot: &PreparedChatSnapshot<B>, settings: PreparedChatResumeSettings,
        capacity: eredu_core::MemoryLimitDeclarations, cancellation: &GenerationCancellationToken,
    ) -> Result<Option<PreparedChatBranch<B>>, PreparedChatSessionError> {
        self.fork_with_host(snapshot, settings, eredu_core::OriginalTextResumeOptions::new(eredu_core::OriginalTextResumeKind::Branch), capacity, cancellation, ()).map(|result| result.map(|(branch, ())| branch))
    }
    pub(crate) fn restore_with_host<J: PreparedTextHostJournal>(
        &mut self, snapshot: &PreparedChatSnapshot<B>, settings: PreparedChatResumeSettings,
        capacity: eredu_core::MemoryLimitDeclarations, cancellation: &GenerationCancellationToken, journal: J,
    ) -> Result<Option<J::Copied>, PreparedChatSessionError> {
        if self.source.generator.driver_identity() != snapshot.driver {
            return Err(self.control_failure(Cause::Snapshot(TextSnapshotError::IncompatibleRun)));
        }
        self.lifecycle.validate_restore().map_err(|cause| self.control_failure(cause.into()))?;
        let Some(config) = snapshot.resume_configuration(self.source.generator.runtime(), settings, cancellation)
            .map_err(|cause| self.control_failure(cause))? else { return Ok(None) };
        let funding = self.preparation.metadata_funding().clone();
        let input_funding = self.input_funding.clone();
        let retain = |cause| PreparedChatSessionError { cause, funding: Some(funding.clone()), input_funding: input_funding.clone() };
        let result = self.source.generator.replace_completed(|runtime| {
            snapshot.resume_parts_with_host(runtime, config, capacity, &eredu_core::OriginalTextResumeOptions {
                terminal: snapshot.finish_reason().is_some(),
                ..eredu_core::OriginalTextResumeOptions::new(eredu_core::OriginalTextResumeKind::Restore)
            }, cancellation, journal)
                .map(|result| result.map(|(generation, displaced, host, copied)| {
                    drop(displaced); (generation, (host, copied))
                })).map_err(|cause| retain(Cause::Snapshot(cause)))
        }, |error| retain(Cause::Boundary(crate::api::control::map_continuation_failure(error, B::into_backend_failure))))?;
        Ok(result.map(|(mut host, copied)| {
            self.lifecycle.restore(snapshot.lifecycle()).expect("validated same boundary restore");
            std::mem::swap(&mut self.lifecycle, &mut host.lifecycle);
            self.exchange_host(&mut host);
            copied
        }))
    }
    pub(crate) fn fork_with_host<J: PreparedTextHostJournal>(
        &mut self, snapshot: &PreparedChatSnapshot<B>, settings: PreparedChatResumeSettings,
        mut options: eredu_core::OriginalTextResumeOptions<'_>, capacity: eredu_core::MemoryLimitDeclarations, cancellation: &GenerationCancellationToken, journal: J,
    ) -> Result<Option<(PreparedChatBranch<B>, J::Copied)>, PreparedChatSessionError> {
        if self.source.generator.driver_identity() != snapshot.driver {
            return Err(self.control_failure(Cause::Snapshot(TextSnapshotError::IncompatibleRun)));
        }
        self.lifecycle.checkpoint().map_err(|cause| self.control_failure(cause.into()))?;
        let Some(config) = snapshot.resume_configuration(self.source.generator.runtime(), settings, cancellation)
            .map_err(|cause| self.control_failure(cause))? else { return Ok(None) };
        let funding = self.preparation.metadata_funding().clone();
        let input_funding = self.input_funding.clone();
        let retain = |cause| PreparedChatSessionError { cause, funding: Some(funding.clone()), input_funding: input_funding.clone() };
        options.kind = eredu_core::OriginalTextResumeKind::Branch;
        options.terminal = snapshot.finish_reason().is_some();
        let result = self.source.generator.fork_completed(|runtime| {
            snapshot.resume_parts_with_host(runtime, config, capacity, &options, cancellation, journal)
                .map(|result| result.map(|(generation, displaced, host, copied)| (generation, displaced, (host, copied))))
                .map_err(|cause| retain(Cause::Snapshot(cause)))
        }, |error| retain(Cause::Boundary(crate::api::control::map_continuation_failure(error, B::into_backend_failure))));
        if result.is_err() && self.source.generator.branch_is_fenced() { self.lifecycle.fail(); }
        Ok(result?.map(|(machine, (host, copied))| (PreparedChatBranch { machine, host }, copied)))
    }
}
impl<B: TextSnapshotBackend> PreparedChatSession<'_, B> {
    /// Exchanges native and host state together without advancing either branch.
    /// Borrowed capture observers remain attached to this active delivery surface.
    pub fn exchange(
        &mut self,
        branch: &mut PreparedChatBranch<B>,
    ) -> Result<(), PreparedChatSessionError> {
        self.lifecycle
            .validate_placement()
            .map_err(|cause| self.control_failure(cause.into()))?;
        branch
            .host
            .lifecycle
            .validate_placement()
            .map_err(|cause| self.control_failure(cause.into()))?;
        self.source
            .generator
            .exchange_branch(&mut branch.machine)
            .map_err(|cause| {
                self.control_failure(Cause::Boundary(
                    crate::api::control::map_continuation_failure(cause, B::into_backend_failure),
                ))
            })?;
        self.exchange_host(&mut branch.host);
        Ok(())
    }
}
