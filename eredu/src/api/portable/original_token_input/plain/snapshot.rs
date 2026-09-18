//! The original plain cursor and native pair use one shared snapshot transaction.
use super::*;
use eredu_runtime::{
    execution_control::{
        SnapshotBudget, SnapshotTokenController, TextContinuationSnapshot, TextSnapshotBackend,
        TextSnapshotError,
    },
    working_memory::WorkspaceCopyLimits,
};

impl SnapshotTokenController for OriginalDomainController {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        self.original_snapshot_storage_bytes()
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(Self {
            source: self.source.clone(),
        })
    }
    fn original_snapshot_storage_bytes(&self) -> Option<u64> {
        u64::try_from(std::mem::size_of::<Self>()).ok()
    }
    fn fork_original_snapshot(&self) -> Option<Self> {
        Some(Self {
            source: self.source.clone(),
        })
    }
}

/// Immutable saved data, not a runnable continuation or native installation right.
/// All controller/native payloads retire before the independent cursor provider.
pub(crate) struct OriginalPlainSnapshot<B: TextSnapshotBackend> {
    state: TextContinuationSnapshot<B, OriginalDomainController>,
    // Preserve active time and the first-token observation without advancing a clock.
    active: Duration,
    time_to_first_token: Option<Duration>,
    cursor: PlainCursor,
    input_custody: Option<eredu_runtime::input::OriginalModelInputCustody>,
}
impl<B: TextSnapshotBackend> OriginalPlainSnapshot<B> {
    pub(crate) fn token_ids(&self) -> &[u32] {
        self.cursor.token_ids()
    }
    pub(crate) fn finish_reason(&self) -> Option<eredu_core::FinishReason> {
        self.cursor.finish_reason()
    }
    pub(crate) fn retained_bytes(&self) -> u64 {
        self.state.retained_bytes()
    }
    pub(crate) fn next_prediction(&self) -> u64 {
        self.state.next_prediction()
    }
}
impl<B: TextSnapshotBackend> OriginalPlainSession<'_, B> {
    pub(crate) fn snapshot(
        &mut self,
        budget: &SnapshotBudget,
        host_capacity: u64,
        native: WorkspaceCopyLimits,
    ) -> Result<OriginalPlainSnapshot<B>, TextSnapshotError<BackendFailure>> {
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
                    TextSnapshotError::Unsupported("plain controller boundary failed")
                }
                E::IncompatibleDriver | E::Failed | E::NotQuiescent => {
                    TextSnapshotError::Unsupported(
                        "plain source is not a completed drained boundary",
                    )
                }
            }
        })?;
        let (runtime, state, pending) = boundary.parts();
        let native_preparation_bytes =
            B::original_saved_generation_preparation_bytes(runtime, state, pending)
                .map_err(TextSnapshotError::HostAdmission)?
                .ok_or(TextSnapshotError::Unsupported(
                    "original native copy preparation",
                ))?;
        let pool = B::original_snapshot_host_pool(runtime).ok_or(
            TextSnapshotError::Unsupported("original snapshot host domain"),
        )?;
        type PublicError = crate::api::portable::managed_plain::GenerationSnapshotError;
        let host = self
            .cursor
            .prepare_snapshot_host_copy::<B, OriginalDomainController, (
                OriginalPlainSnapshot<B>,
                PublicError,
                Result<OriginalPlainSnapshot<B>, TextSnapshotError<BackendFailure>>,
                Result<
                    crate::api::portable::managed_plain::ManagedPlainTextSnapshot<B>,
                    PublicError,
                >,
                Duration,
                Option<Duration>,
                WorkspaceCopyLimits,
            )>(pool, host_capacity, native_preparation_bytes)
            .map_err(TextSnapshotError::HostAdmission)?;
        let (state, cursor) =
            TextContinuationSnapshot::capture_original_host(&mut boundary, budget, host, native)
                .map_err(|error| {
                    crate::api::control::map_snapshot_failure(error, B::into_backend_failure)
                })?;
        Ok(OriginalPlainSnapshot {
            input_custody: self.input_custody.clone(),
            state,
            cursor,
            active: self.active,
            time_to_first_token: self.source.time_to_first_token,
        })
    }
}

impl<B> OriginalPlainSnapshot<B>
where
    B: TextSnapshotBackend
        + eredu_core::TextResumeBackend<
            ResumeSource = <B as TextSnapshotBackend>::SavedTextComponents,
        >,
{
    pub(crate) fn source(&self) -> &OriginalTokenizer {
        &self.state.controller().source
    }
    pub(crate) fn remaining_tokens(&self) -> Option<usize> {
        self.state.remaining_tokens()
    }

    pub(crate) fn resume<'a>(
        &self,
        runtime: &'a mut ModelRuntime<B>,
        config: TextGenerationConfig,
        host_capacity: u64,
        branch: bool,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<OriginalPlainSession<'a, B>>, TextSnapshotError<BackendFailure>> {
        if cancellation.is_cancelled()
            || self.finish_reason().is_some()
            || config.sampling().max_new_tokens == Some(0)
        {
            return Ok(None);
        }
        let started = Instant::now();
        let preparation = self
            .state
            .original_resume_preparation_bytes(runtime, config, &eredu_core::OriginalTextResumeOptions::new(
                if branch { eredu_core::OriginalTextResumeKind::Branch } else { eredu_core::OriginalTextResumeKind::Restore }))
            .map_err(|e| crate::api::control::map_snapshot_failure(e, B::into_backend_failure))?;
        let pool = B::original_snapshot_host_pool(runtime).ok_or(
            TextSnapshotError::Unsupported("original resume host domain"),
        )?;
        type PublicError = crate::api::portable::managed_plain::ManagedPlainTextError;
        let host = self
            .cursor
            .prepare_resume_host_copy::<B, OriginalDomainController, (
                OriginalPlainSession<'a, B>,
                PublicError,
                Result<Option<OriginalPlainSession<'a, B>>, TextSnapshotError<BackendFailure>>,
                Result<
                    Option<crate::api::portable::managed_plain::ManagedPlainTextSession<'a, B>>,
                    PublicError,
                >,
                Duration,
                Option<Duration>,
                TextGenerationConfig,
                bool,
            )>(pool, host_capacity, preparation)
            .map_err(TextSnapshotError::HostAdmission)?;
        let result = if branch {
            self.state
                .fork_original_host(runtime, config, host, cancellation)
        } else {
            self.state
                .restore_original_host(runtime, config, host, cancellation)
        }
        .map_err(|e| crate::api::control::map_snapshot_failure(e, B::into_backend_failure))?;
        Ok(result.map(|(generator, mut cursor)| {
            if let Some(remaining) = config.sampling().max_new_tokens {
                cursor.restrict_remaining(remaining);
            }
            OriginalPlainSession {
                input_custody: self.input_custody.clone(),
                source: BackendGenerationTokenSource {
                    generator,
                    on_token: None,
                    delivery_failure: None,
                    generation_started: started,
                    time_to_first_token: self.time_to_first_token,
                },
                active: self.active + started.elapsed(),
                cursor,
            }
        }))
    }
}
