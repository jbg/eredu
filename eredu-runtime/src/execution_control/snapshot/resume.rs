//! Fresh original execution from the same immutable snapshot and host provider.
use super::*;
use eredu_core::{
    ControlledTextGeneration, ControlledTextGenerationError, GenerationCancellationToken,
    TextGenerationConfig, TextResumeBackend,
};
use std::mem::size_of;

/// Move-only logical reservation awaiting an independently admitted host copy.
/// It carries no storage or execution authority. The concrete provider installs
/// it in the same closed owner retained by every copied payload/output alias.
pub struct PendingSnapshotResumeRetention(super::super::PendingSnapshotReservation);
impl PendingSnapshotResumeRetention {
    pub(crate) fn reserve(
        budget: &SnapshotBudget,
        kind: SnapshotResourceKind,
        estimate: SnapshotEstimate,
    ) -> Result<Self, ExecutionControlError> {
        budget.reserve_pending(kind, Some(estimate)).map(Self)
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        super::super::PendingSnapshotReservation::control_bytes()?.checked_add(size_of::<Self>())
    }
    pub(crate) fn publish(self) -> SnapshotReservation {
        self.0.publish()
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct ResumeFailure<C: std::error::Error + 'static> {
    #[source]
    cause: ControlledTextGenerationError<BackendFailure, C>,
    _host: HostPreparationAuthority,
}

impl<B, C> TextContinuationSnapshot<B, C>
where
    B: TextSnapshotBackend
        + TextResumeBackend<ResumeSource = <B as TextSnapshotBackend>::SavedTextComponents>,
    C: SnapshotTokenController,
{
    /// Fixed source facts from the actual opaque saved native components.
    pub fn resume_source_facts(&self) -> Option<eredu_core::TextResumeSourceFacts> {
        B::saved_text_resume_facts(&self.saved)
    }
    /// Borrowed original resume constructor facts, with no host allocation or
    /// source replacement. The eventual native admission revalidates its source.
    pub fn original_resume_preparation_bytes(
        &self,
        runtime: &ModelRuntime<B>,
        config: TextGenerationConfig,
        options: &eredu_core::OriginalTextResumeOptions<'_>,
    ) -> Result<u64, TextSnapshotError<B::Error>> {
        self.validate_original_resume(config.clone())?;
        B::original_saved_components_resume_preparation_bytes(
            runtime,
            &self.saved,
            config,
            &self.controller,
            options,
        )
        .map_err(TextSnapshotError::HostAdmission)?
        .ok_or(TextSnapshotError::Unsupported(
            "original saved resume preparation",
        ))
    }

    /// Fixed shared constructor and error controls required by the host provider.
    /// Backend planning/native payloads are supplied by its separate exact query.
    pub fn original_resume_control_bytes<H>() -> Option<usize> {
        [
            PendingSnapshotResumeRetention::control_bytes()?,
            eredu_core::text_resume_control_bytes::<B, C>()?,
            size_of::<H>(),
            size_of::<C>(),
            size_of::<Option<C>>(),
            size_of::<SnapshotEstimate>(),
            size_of::<TextGenerationConfig>(),
            size_of::<TextSnapshotError<B::Error>>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<B::DisplacedState>(),
            size_of::<Option<(ControlledTextGeneration<'_, B, C>, B::DisplacedState, H)>>(),
            size_of::<Result<(H, HostPreparationAuthority), TextHostCopyError>>(),
            size_of::<
                Result<
                    Option<(ControlledTextGeneration<'_, B, C>, H)>,
                    TextSnapshotError<B::Error>,
                >,
            >(),
            BackendFailure::source_retention_peak_bytes::<ResumeFailure<C::Error>>()?,
            BackendFailure::source_retention_peak_bytes::<B::Error>()?,
            size_of::<ControlledTextGenerationError<BackendFailure, C::Error>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// Restores the saved frontier through a fresh original admission and the
    /// ordinary shared generation loop. Copy use is charged to the snapshot's
    /// existing budget; no caller can replace it with a fresh cumulative budget.
    /// The destination host provider independently owns decoder/stop/delivery
    /// state. Its provider aliases and the generation owner retain the same logical
    /// reservation independently until the final copied owner retires.
    pub fn restore_original_host<'a, H: PreparedTextHostCopy>(
        &self,
        runtime: &'a mut ModelRuntime<B>,
        config: TextGenerationConfig,
        host: H,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<(ControlledTextGeneration<'a, B, C>, H::Copied)>, TextSnapshotError<B::Error>>
    {
        self.resume_original_host_with_displaced(
            runtime,
            config,
            host,
            cancellation,
            &eredu_core::OriginalTextResumeOptions::new(
                eredu_core::OriginalTextResumeKind::Restore,
            ),
        )
        .map(|result| {
            result.map(|(generation, displaced, host)| {
                drop(displaced);
                (generation, host)
            })
        })
    }

    /// Makes another independently admitted runnable branch from the immutable
    /// saved frontier. Branch count, retained cost and cumulative copies remain
    /// in the same snapshot budget and never rewind with the copied host state.
    pub fn fork_original_host<'a, H: PreparedTextHostCopy>(
        &self,
        runtime: &'a mut ModelRuntime<B>,
        config: TextGenerationConfig,
        host: H,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<(ControlledTextGeneration<'a, B, C>, H::Copied)>, TextSnapshotError<B::Error>>
    {
        self.resume_original_host_with_displaced(
            runtime,
            config,
            host,
            cancellation,
            &eredu_core::OriginalTextResumeOptions::new(eredu_core::OriginalTextResumeKind::Branch),
        )
        .map(|result| {
            result.map(|(generation, displaced, host)| {
                drop(displaced);
                (generation, host)
            })
        })
    }

    fn validate_original_resume(
        &self,
        config: TextGenerationConfig,
    ) -> Result<(), TextSnapshotError<B::Error>> {
        match (self.remaining_tokens, config.sampling().max_new_tokens) {
            (Some(saved), Some(requested)) if requested <= saved => Ok(()),
            _ => Err(TextSnapshotError::InconsistentState),
        }
    }
    /// Same original copy transaction while retaining the native slot displaced
    /// by successful installation for serial branch composition. The slot grants
    /// no new run or copy allowance; only the source's cumulative budget is used.
    pub fn resume_original_host_with_displaced<'a, H: PreparedTextHostCopy>(
        &self,
        runtime: &'a mut ModelRuntime<B>,
        config: TextGenerationConfig,
        host: H,
        cancellation: &GenerationCancellationToken,
        options: &eredu_core::OriginalTextResumeOptions<'_>,
    ) -> Result<
        Option<(
            ControlledTextGeneration<'a, B, C>,
            B::DisplacedState,
            H::Copied,
        )>,
        TextSnapshotError<B::Error>,
    > {
        if cancellation.is_cancelled()
            || (config.sampling().max_new_tokens == Some(0) && !options.terminal)
        {
            return Ok(None);
        }
        let kind = match options.kind {
            eredu_core::OriginalTextResumeKind::Restore => SnapshotResourceKind::Restore,
            eredu_core::OriginalTextResumeKind::Branch => SnapshotResourceKind::Branch,
        };
        let preparation =
            self.original_resume_preparation_bytes(runtime, config.clone(), options)?;
        let controls = Self::original_resume_control_bytes::<H::Copied>()
            .ok_or(ExecutionControlError::Overflow)?;
        if host
            .original_preparation_bytes()
            .is_none_or(|n| n < preparation)
            || host.original_control_bytes().is_none_or(|n| n < controls)
        {
            return Err(TextSnapshotError::Unsupported(
                "original resume host constructor",
            ));
        }
        let host_bytes = host
            .storage_bytes()
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let controller_bytes = self
            .controller
            .original_snapshot_storage_bytes()
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let estimate = combine_estimates(
            [B::original_saved_components_resume_estimate(
                runtime,
                &self.saved,
                config.clone(),
                options,
            )],
            host_bytes
                .checked_add(controller_bytes)
                .ok_or(ExecutionControlError::Overflow)?,
            0,
            size_of::<PendingSnapshotResumeRetention>(),
        )?;
        let reservation = PendingSnapshotResumeRetention::reserve(
            &self._reservation.lease().budget,
            kind,
            estimate,
        )?;
        // Publish the logical lease inside the independent host account before
        // any copied aliases escape. Failed attempts never refund copied work.
        let (copied, host) = host.copy_original_resume(estimate.retained_bytes, reservation)?;
        let controller =
            self.controller
                .fork_original_snapshot()
                .ok_or(TextSnapshotError::Unsupported(
                    "original resume controller copy",
                ))?;
        let generation = ControlledTextGeneration::resume_saved_original_with_displaced(
            runtime,
            &self.saved,
            config,
            controller,
            cancellation,
            &host,
            options,
        )
        .map_err(|cause| {
            let cause = match cause {
                ControlledTextGenerationError::Preparation(error) => {
                    ControlledTextGenerationError::Preparation(error)
                }
                ControlledTextGenerationError::Backend(error) => {
                    ControlledTextGenerationError::Backend(B::into_backend_failure(error))
                }
                ControlledTextGenerationError::Controller(error) => {
                    ControlledTextGenerationError::Controller(error)
                }
            };
            let (kind, operation) = match &cause {
                ControlledTextGenerationError::Preparation(error)
                | ControlledTextGenerationError::Backend(error) => {
                    (error.kind(), error.operation())
                }
                _ => (eredu_core::BackendFailureKind::Other, "snapshot resume"),
            };
            TextSnapshotError::Resume(
                BackendFailure::new(
                    kind,
                    ResumeFailure {
                        cause,
                        _host: host.clone(),
                    },
                )
                .with_operation(operation),
            )
        })?;
        Ok(generation.map(|(generation, displaced)| (generation, displaced, copied)))
    }
}
