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
struct ResumeFailure<B: std::error::Error + 'static, C: std::error::Error + 'static> {
    #[source]
    cause: ControlledTextGenerationError<B, C>,
    _host: HostPreparationAuthority,
}

impl<B, C> TextContinuationSnapshot<B, C>
where
    B: TextSnapshotBackend
        + TextResumeBackend<ResumeSource = <B as TextSnapshotBackend>::SavedTextComponents>,
    C: SnapshotTokenController,
{
    /// Remaining admitted output positions represented by the saved frontier.
    /// A fresh run may shorten this allowance, but cannot refund or extend it.
    pub fn remaining_tokens(&self) -> Option<usize> {
        self.remaining_tokens
    }

    /// Borrowed original resume constructor facts, with no host allocation or
    /// source replacement. The eventual native admission revalidates its source.
    pub fn original_resume_preparation_bytes(
        &self,
        runtime: &ModelRuntime<B>,
        config: TextGenerationConfig,
    ) -> Result<u64, TextSnapshotError<B::Error>> {
        self.validate_original_resume(config)?;
        B::original_saved_components_resume_preparation_bytes(
            runtime,
            &self.saved,
            config,
            &self.controller,
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
            size_of::<Result<(H, HostPreparationAuthority), TextHostCopyError>>(),
            size_of::<
                Result<
                    Option<(ControlledTextGeneration<'_, B, C>, H)>,
                    TextSnapshotError<B::Error>,
                >,
            >(),
            BackendFailure::source_retention_peak_bytes::<ResumeFailure<B::Error, C::Error>>()?,
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
        self.resume_original_host(
            runtime,
            config,
            host,
            cancellation,
            SnapshotResourceKind::Restore,
        )
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
        self.resume_original_host(
            runtime,
            config,
            host,
            cancellation,
            SnapshotResourceKind::Branch,
        )
    }

    fn validate_original_resume(
        &self,
        config: TextGenerationConfig,
    ) -> Result<(), TextSnapshotError<B::Error>> {
        if self.driver.is_some() || self.capture.is_some() {
            return Err(TextSnapshotError::Unsupported(
                "original resume source contract",
            ));
        }
        match (self.remaining_tokens, config.sampling().max_new_tokens) {
            (Some(saved), Some(requested)) if requested <= saved => Ok(()),
            _ => Err(TextSnapshotError::InconsistentState),
        }
    }
    fn resume_original_host<'a, H: PreparedTextHostCopy>(
        &self,
        runtime: &'a mut ModelRuntime<B>,
        config: TextGenerationConfig,
        host: H,
        cancellation: &GenerationCancellationToken,
        kind: SnapshotResourceKind,
    ) -> Result<Option<(ControlledTextGeneration<'a, B, C>, H::Copied)>, TextSnapshotError<B::Error>>
    {
        if cancellation.is_cancelled() || config.sampling().max_new_tokens == Some(0) {
            return Ok(None);
        }
        let preparation = self.original_resume_preparation_bytes(runtime, config)?;
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
                config,
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
        let generation = ControlledTextGeneration::resume_saved_original_with_kind(
            runtime,
            &self.saved,
            config,
            controller,
            cancellation,
            &host,
            match kind {
                SnapshotResourceKind::Restore => eredu_core::OriginalTextResumeKind::Restore,
                SnapshotResourceKind::Branch => eredu_core::OriginalTextResumeKind::Branch,
                _ => return Err(TextSnapshotError::InconsistentState),
            },
        )
        .map_err(|cause| {
            let kind = match &cause {
                ControlledTextGenerationError::Preparation(cause) => cause.kind(),
                _ => eredu_core::BackendFailureKind::Other,
            };
            TextSnapshotError::Resume(BackendFailure::new(
                kind,
                ResumeFailure {
                    cause,
                    _host: host.clone(),
                },
            ))
        })?;
        Ok(generation.map(|generation| (generation, copied)))
    }
}
