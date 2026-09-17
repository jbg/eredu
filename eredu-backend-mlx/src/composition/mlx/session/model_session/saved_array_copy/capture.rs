//! Exact live generation capture source and its independently funded checkpoint.
use super::*;
use decoder::cold_source::SourcePreparationCause;
use eredu_core::HostPreparationAuthority;
use eredu_runtime::capture::{
    FundedCaptureCheckpoint, FundedCaptureCheckpointError, PreparedFundedCaptureCheckpoint,
};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

/// This loan can only be derived from the complete live generation state.
/// Neither an H token nor a same-shaped source can construct this proof.
pub(in crate::composition::mlx::session) struct PreparedCaptureCopy<'a> {
    checkpoint: PreparedFundedCaptureCheckpoint<'a>,
    source: &'a eredu_core::capture::SharedCapturePlan,
    selection: &'a eredu_runtime::layered::PreparedCaptureSelection,
    witness: &'a eredu_runtime::working_memory::RegisteredInferenceSourceWitness,
}
impl<'a> PreparedCaptureCopy<'a> {
    pub(in crate::composition::mlx::session) fn inspect(
        state: &'a super::super::super::generation::MlxTextGenerationState,
    ) -> Result<Option<Self>, SourcePreparationCause> {
        if state.capture.is_some() {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        let quote = state
            .sampling
            .quote
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        quote.validate_paired_capture_copy_with_source(
            state
                .funded_capture
                .as_ref()
                .map(|capture| capture.source()),
        )?;
        let Some(capture) = &state.funded_capture else {
            return Ok(None);
        };
        let checkpoint = capture.collector().prepare_checkpoint()?;
        if checkpoint.next_prediction() != state.sampling.next_prediction {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(Some(Self {
            checkpoint,
            source: capture.source(),
            selection: capture.saved_selection(),
            witness: capture.source_witness(),
        }))
    }

    pub(in crate::composition::mlx::session) fn source(
        &self,
    ) -> &eredu_core::capture::SharedCapturePlan {
        self.source
    }

    pub(in crate::composition::mlx::session) fn logical_bytes(&self) -> Option<u64> {
        u64::try_from(
            SavedCaptureCheckpoint::block_bytes()?
                .checked_add(size_of::<SavedCaptureCheckpoint>())?,
        )
        .ok()
    }

    pub(in crate::composition::mlx::session) fn required_bytes(
        &self,
    ) -> Result<usize, SourcePreparationCause> {
        let checkpoint = usize::try_from(
            self.checkpoint
                .required_bytes()
                .map_err(FundedCaptureCheckpointError::from)?,
        )
        .map_err(|_| WorkingMemoryError::Overflow)?;
        let parts = [
            checkpoint,
            SavedCaptureCheckpoint::block_bytes().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<Self>(),
            size_of::<SavedCapturePayload>(),
            size_of::<eredu_runtime::layered::PreparedCaptureSelection>(),
            size_of::<eredu_runtime::working_memory::RegisteredInferenceSourceWitness>(),
            size_of::<Option<Self>>(),
            size_of::<SavedCaptureCheckpoint>(),
            size_of::<Option<SavedCaptureCheckpoint>>(),
            size_of::<Result<Option<Self>, SourcePreparationCause>>(),
            size_of::<Result<usize, SourcePreparationCause>>(),
            size_of::<Result<SavedCaptureCheckpoint, SourcePreparationCause>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| WorkingMemoryError::Overflow.into())
    }

    pub(in crate::composition::mlx::session) fn construct(
        self,
        host: &HostPreparationAuthority,
    ) -> Result<SavedCaptureCheckpoint, SourcePreparationCause> {
        let checkpoint = self.checkpoint.construct(host)?;
        Ok(SavedCaptureCheckpoint(Some(Rc::new(SavedCapturePayload {
            selection: self.selection.clone(),
            witness: self.witness.clone(),
            checkpoint,
        }))))
    }
}

/// No raw Rc/Weak escapes. The allocation shell retires before the checkpoint's
/// source aliases and host authority. No live request, native root, or model is
/// retained by this owner; later copies share the immutable snapshot frontier.
struct SavedCapturePayload {
    // Retained source/path aliases retire before checkpoint construction H.
    selection: eredu_runtime::layered::PreparedCaptureSelection,
    witness: eredu_runtime::working_memory::RegisteredInferenceSourceWitness,
    checkpoint: FundedCaptureCheckpoint,
}
#[derive(Clone)]
pub(in crate::composition::mlx::session) struct SavedCaptureCheckpoint(
    Option<Rc<SavedCapturePayload>>,
);
impl SavedCaptureCheckpoint {
    pub(in crate::composition::mlx::session) fn witness(
        &self,
    ) -> &eredu_runtime::working_memory::RegisteredInferenceSourceWitness {
        &self.0.as_deref().expect("live saved capture").witness
    }
    fn block_bytes() -> Option<usize> {
        Layout::new::<[std::cell::Cell<usize>; 2]>()
            .extend(Layout::new::<SavedCapturePayload>())
            .ok()
            .map(|(layout, _)| layout.pad_to_align().size())
    }
    pub(in crate::composition::mlx::session) fn checkpoint(&self) -> &FundedCaptureCheckpoint {
        &self.0.as_deref().expect("live saved capture").checkpoint
    }
    pub(in crate::composition::mlx::session) fn selection(
        &self,
    ) -> &eredu_runtime::layered::PreparedCaptureSelection {
        &self.0.as_deref().expect("live saved capture").selection
    }
}
impl Drop for SavedCaptureCheckpoint {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}

pub(in crate::composition::mlx::session) fn into_memory(
    cause: SourcePreparationCause,
) -> WorkingMemoryError {
    match cause {
        SourcePreparationCause::Memory(cause) => cause,
        SourcePreparationCause::Capture(FundedCaptureCheckpointError::Host(
            eredu_runtime::working_memory::CaptureRunHostError::Memory(cause),
        )) => cause,
        _ => WorkingMemoryError::UnknownBound,
    }
}

// Fixed complete-state hook/query/return transports exist even without capture.
// The existing native preparation worker includes these before granting H.
pub(super) fn handoff_control_bytes() -> Option<usize> {
    [
        size_of::<Option<PreparedCaptureCopy<'static>>>(),
        size_of::<Result<Option<PreparedCaptureCopy<'static>>, SourcePreparationCause>>(),
        size_of::<Option<SavedCaptureCheckpoint>>(),
        size_of::<Result<Option<SavedCaptureCheckpoint>, SourcePreparationCause>>(),
        size_of::<Result<super::super::super::text_snapshot::MlxSavedTextComponents, Error>>(),
        size_of::<Option<&eredu_core::capture::SharedCapturePlan>>(),
        size_of::<Option<&SavedCaptureCheckpoint>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
