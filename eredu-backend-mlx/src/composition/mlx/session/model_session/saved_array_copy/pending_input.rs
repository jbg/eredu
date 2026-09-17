//! Borrow pending text copies or retain the exact completed immutable media source.
use super::*;
use eredu_core::HostPreparationAuthority;
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(in crate::composition::mlx::session) enum PromptCopyCause {
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Descriptor(#[from] safemlx::ArrayDescriptorError),
    #[error("saved prompt differs from its complete prepared source")]
    Geometry,
}

#[derive(Clone, Copy)]
pub(in crate::composition::mlx::session) struct PromptCopySource<'a> {
    prompt: &'a MlxModelInput,
    kind: PromptCopyKind<'a>,
    positions: u64,
}
#[derive(Clone, Copy)]
enum PromptCopyKind<'a> {
    Tokens(&'a Array),
    Media(
        &'a crate::composition::mlx::CompletedOriginalModelInput,
        &'a eredu_architectures::media_plan::BoundPreparedMediaSemantics,
    ),
}
impl<'a> PromptCopySource<'a> {
    pub(in crate::composition::mlx::session) fn prepare(
        sampling: &super::super::super::generation::MlxTextSamplingState,
        prompt: &'a MlxModelInput,
    ) -> Result<Self, PromptCopyCause> {
        let quote = sampling
            .quote
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        super::super::text_step::validate_prompt_binding_fixed(prompt, quote)?;
        if quote.local_prediction_fixed(sampling.next_prediction)? != 0
            || prompt.controlled_attribution.is_some()
            || prompt.prepared_capture.is_some()
        {
            return Err(PromptCopyCause::Geometry);
        }
        if let Some(packet) = prompt.original_media.as_ref() {
            let input::OriginalMediaPacket::Original(packet) = packet else {
                return Err(PromptCopyCause::Geometry);
            };
            quote.validate_completed_pending_input(prompt)?;
            return Ok(Self {
                prompt,
                kind: PromptCopyKind::Media(
                    packet,
                    prompt
                        .quote
                        .as_ref()
                        .and_then(|quote| quote.copied_media_semantics())
                        .unwrap_or_else(|| packet.borrowed_semantics()),
                ),
                positions: packet.shape()[1],
            });
        }
        let [part] = &*prompt.parts else {
            return Err(PromptCopyCause::Geometry);
        };
        if part.modality() != InputModality::Text
            || !part.metadata().is_empty()
            || !part.extents().is_empty()
        {
            return Err(PromptCopyCause::Geometry);
        }
        let input::InputPayload::TokenIds(tokens) = part.payload() else {
            return Err(PromptCopyCause::Geometry);
        };
        let descriptor = tokens.try_descriptor()?;
        let [1, positions] = descriptor.shape() else {
            return Err(PromptCopyCause::Geometry);
        };
        let positions = u64::try_from(*positions).map_err(|_| PromptCopyCause::Geometry)?;
        if positions == 0
            || quote.request().geometry().batch_size != 1
            || positions != quote.request().geometry().input_positions
            || !matches!(descriptor.facts().dtype(), Dtype::Int32 | Dtype::Uint32)
        {
            return Err(PromptCopyCause::Geometry);
        }
        Ok(Self {
            prompt,
            kind: PromptCopyKind::Tokens(tokens),
            positions,
        })
    }
    pub(in crate::composition::mlx::session) fn tokens(self) -> Option<&'a Array> {
        match self.kind {
            PromptCopyKind::Tokens(tokens) => Some(tokens),
            PromptCopyKind::Media(_, _) => None,
        }
    }
    pub(in crate::composition::mlx::session) fn positions(self) -> u64 {
        self.positions
    }
    pub(super) fn validate(
        self,
        sampling: &super::super::super::generation::MlxTextSamplingState,
    ) -> Result<(), PromptCopyCause> {
        let actual = Self::prepare(sampling, self.prompt)?;
        let same = match (actual.kind, self.kind) {
            (PromptCopyKind::Tokens(a), PromptCopyKind::Tokens(b)) => std::ptr::eq(a, b),
            (PromptCopyKind::Media(a, sa), PromptCopyKind::Media(b, sb)) => {
                std::ptr::eq(a, b) && std::ptr::eq(sa, sb)
            }
            _ => false,
        };
        if !same || actual.positions != self.positions {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(())
    }
    pub(super) fn metadata(self) -> SavedPendingInput {
        match self.kind {
            PromptCopyKind::Tokens(_) => SavedPendingInput::Prefill {
                positions: self.positions,
                _chunk_positions: self.prompt.prefill_chunk_positions,
                _cache_identity: self.prompt.cache_identity.clone(),
            },
            PromptCopyKind::Media(packet, semantics) => {
                SavedPendingInput::Media(SavedPendingMedia {
                    chunk_positions: self.prompt.prefill_chunk_positions,
                    packet: packet.clone(),
                    semantics: semantics.clone(),
                })
            }
        }
    }
}

/// Immutable source semantics only. This carries no inference request, quote,
/// input wrapper or completed decode receipt, and cannot be installed as input.
#[derive(Clone)]
pub(super) enum SavedPendingInput {
    Decode,
    Media(SavedPendingMedia),
    Prefill {
        positions: u64,
        _chunk_positions: Option<std::num::NonZeroU64>,
        _cache_identity: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
    },
}
impl SavedPendingInput {
    pub(super) fn positions(&self) -> u64 {
        match self {
            Self::Decode => 1,
            Self::Media(media) => media.packet.shape()[1],
            Self::Prefill { positions, .. } => *positions,
        }
    }
}

/// Immutable completed B and source semantics only. No generation quote,
/// request, ready receipt, sampler or mutable decoder state is retained here.
#[derive(Clone)]
pub(in crate::composition::mlx::session::model_session) struct SavedPendingMedia {
    chunk_positions: Option<std::num::NonZeroU64>,
    packet: crate::composition::mlx::CompletedOriginalModelInput,
    semantics: eredu_architectures::media_plan::BoundPreparedMediaSemantics,
}
impl SavedPendingMedia {
    pub(in crate::composition::mlx::session::model_session) fn semantics(
        &self,
    ) -> &eredu_architectures::media_plan::BoundPreparedMediaSemantics {
        &self.semantics
    }

    pub(super) fn packet(&self) -> &crate::composition::mlx::CompletedOriginalModelInput {
        &self.packet
    }

    pub(super) fn geometry(&self) -> Result<(u64, u64), WorkingMemoryError> {
        let [batch, positions] = self.packet.shape();
        let chunk = self
            .chunk_positions
            .map_or(positions, std::num::NonZeroU64::get);
        if batch != 1 || positions == 0 || chunk == 0 || chunk > positions {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok((positions, chunk))
    }
}

impl CopiedTextSampling {
    /// The immutable variant, not array absence, determines media preparation.
    pub(in crate::composition::mlx::session::model_session) fn pending_media(
        &self,
    ) -> Option<&SavedPendingMedia> {
        match self.arrays.pending_metadata.as_ref()? {
            SavedPendingInput::Media(media) => Some(media),
            _ => None,
        }
    }

    pub(super) fn prepare_pending_tokens(
        &self,
    ) -> Result<
        Option<super::super::pending_prompt::PreparedPendingPrompt<'_>>,
        super::super::pending_prompt::PendingPromptPreparationCause,
    > {
        if let Some(media) = self.pending_media() {
            media.geometry()?;
            if self.arrays.pending.is_some() {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            Ok(None)
        } else {
            self.prepare_pending_prompt().map(Some)
        }
    }

    /// Reconstructs only a source-bound numerical/host plan. The immutable
    /// pending kind prevents a one-token prompt from becoming a decode token;
    /// the fresh request must still admit and execute this complete program.
    pub(in crate::composition::mlx::session) fn prepare_pending_prompt(
        &self,
    ) -> Result<
        super::super::pending_prompt::PreparedPendingPrompt<'_>,
        super::super::pending_prompt::PendingPromptPreparationCause,
    > {
        use super::super::pending_prompt::PreparedPendingPrompt;
        let source = self
            .arrays
            .pending
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        match self
            .arrays
            .pending_metadata
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?
        {
            SavedPendingInput::Decode => PreparedPendingPrompt::new_fixed(source),
            // The media resume consumer uses the genuine completed source,
            // never the scalar/packed-token numerical preparation program.
            SavedPendingInput::Media(_) => Err(WorkingMemoryError::UnknownBound.into()),
            SavedPendingInput::Prefill {
                positions,
                _chunk_positions,
                _cache_identity,
            } => {
                let positions = std::num::NonZeroU64::new(*positions)
                    .ok_or(WorkingMemoryError::IdentityMismatch)?;
                PreparedPendingPrompt::new_prefill_fixed(
                    source,
                    positions,
                    *_chunk_positions,
                    _cache_identity.as_ref(),
                )
            }
        }
    }

    pub(in crate::composition::mlx::session) fn pending_geometry(
        &self,
    ) -> Result<(u64, u64), WorkingMemoryError> {
        match self
            .arrays
            .pending_metadata
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?
        {
            SavedPendingInput::Decode => Ok((1, 1)),
            SavedPendingInput::Media(media) => media.geometry(),
            SavedPendingInput::Prefill {
                positions,
                _chunk_positions,
                ..
            } => {
                let chunk = _chunk_positions.map_or(*positions, std::num::NonZeroU64::get);
                if *positions == 0 || chunk == 0 || chunk > *positions {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                Ok((*positions, chunk))
            }
        }
    }
}

pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<PromptCopySource<'_>>(),
        size_of::<PromptCopyKind<'_>>(),
        size_of::<SavedPendingMedia>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<
            Result<
                &eredu_runtime::working_memory::MediaSessionBinding,
                eredu_core::PreparedRequestRejection,
            >,
        >(),
        size_of::<Option<PromptCopySource<'_>>>(),
        size_of::<Option<eredu_core::PendingTextInput<&MlxModelInput, &MlxTextToken>>>(),
        size_of::<Result<Option<u64>, WorkingMemoryError>>(),
        size_of::<Result<PromptCopySource<'_>, PromptCopyCause>>(),
        size_of::<PromptCopyCause>(),
        std::alloc::Layout::new::<PromptCopyCause>().size(),
        size_of::<Box<PromptCopyCause>>(),
        size_of::<Result<(), Error>>(),
        size_of::<safemlx::OwnedArrayDescriptorLoan<'_>>(),
        size_of::<Result<safemlx::OwnedArrayDescriptorLoan<'_>, safemlx::ArrayDescriptorError>>(),
        size_of::<SavedPendingInput>(),
        size_of::<Option<SavedPendingInput>>(),
        size_of::<Result<(), PromptCopyCause>>(),
        size_of::<Option<&HostPreparationAuthority>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
