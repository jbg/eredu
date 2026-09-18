//! Exact host/numerical input preparation for an already committed pending token.

use super::{input, Array, Error, MlxModelInput, RefCell};
use crate::backend::{
    array_copy::{
        PendingTokenNativePopulation, PendingTokenSourceCause, PreparedPendingTokenInput,
    },
    nn::workspace::ExistingArrayProjection,
};
use eredu_nn::workspace::WorkspaceTensor;
use eredu_runtime::working_memory::{
    FundedPendingTokenInput, InferencePendingPromptCompletion, InferencePromptCompletion,
    PendingTokenInputHostPlan, PreparedPendingTokenInputHost, WorkingMemoryFundingRun,
};
use safemlx::Stream;
use std::ops::Deref;

/// Ordinary compatibility inputs retain their existing allocating Clone.
/// Closed pending inputs share the actual part and its protected host custody.
#[derive(Debug, Clone)]
pub(super) enum ModelInputParts {
    Owned(Vec<input::InputPart>),
    Original(eredu_runtime::input::SharedPreparedInputParts<Array>),
    OriginalText(super::original_host_input::CompletedOriginalTextInput),
    Pending(FundedPendingTokenInput<Array>),
}

impl Deref for ModelInputParts {
    type Target = [input::InputPart];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(parts) => parts,
            Self::Original(parts) => parts.as_ref(),
            Self::OriginalText(source) => source.parts(),
            Self::Pending(parts) => parts.parts(),
        }
    }
}

/// Fixed pregrant constructor causes; ordinary callers wrap these only after
/// their own admission boundary. Native/source layout facts remain borrowed.
#[derive(Debug, thiserror::Error)]
pub(super) enum PendingPromptPreparationCause {
    #[error(transparent)]
    Source(#[from] PendingTokenSourceCause),
    #[error(transparent)]
    Host(#[from] eredu_runtime::working_memory::WorkingMemoryError),
}

/// The numerical program is bound to the actual settled saved scalar. Host
/// facts describe its one fixed output part, without downloading token values
/// or inventing a fingerprint. Source ownership and numerical recovery remain
/// the caller's existing saved-copy/preparation responsibilities.
pub(in crate::composition::mlx::session) struct PreparedPendingPrompt<'a> {
    numerical: PreparedPendingTokenInput<'a>,
    host: PendingTokenInputHostPlan<Array>,
    kind: PendingPromptKind<'a>,
}

enum PendingPromptKind<'a> {
    Decode,
    Prefill {
        chunk_positions: Option<std::num::NonZeroU64>,
        cache_identity: Option<&'a eredu_runtime::SharedPreparedInputCacheIdentity>,
    },
}

impl<'a> PreparedPendingPrompt<'a> {
    pub(super) fn new(source: &'a Array) -> Result<Self, Error> {
        Self::new_fixed(source).map_err(other)
    }

    pub(super) fn new_fixed(source: &'a Array) -> Result<Self, PendingPromptPreparationCause> {
        Ok(Self {
            numerical: PreparedPendingTokenInput::new_fixed(source)?,
            host: PendingTokenInputHostPlan::prepare()?,
            kind: PendingPromptKind::Decode,
        })
    }

    pub(super) fn new_prefill_fixed(
        source: &'a Array,
        positions: std::num::NonZeroU64,
        chunk_positions: Option<std::num::NonZeroU64>,
        cache_identity: Option<&'a eredu_runtime::SharedPreparedInputCacheIdentity>,
    ) -> Result<Self, PendingPromptPreparationCause> {
        if chunk_positions.is_some_and(|chunk| chunk > positions) {
            return Err(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(Self {
            numerical: PreparedPendingTokenInput::new_prefill_fixed(source, positions)?,
            host: PendingTokenInputHostPlan::prepare()?,
            kind: PendingPromptKind::Prefill {
                chunk_positions,
                cache_identity,
            },
        })
    }

    /// Same cache metadata copied by the one-part pending-input constructor.
    pub(in crate::composition::mlx::session) fn logical_metadata_bytes(&self) -> Option<u64> {
        match self.kind {
            PendingPromptKind::Decode => Some(0),
            PendingPromptKind::Prefill { cache_identity, .. } => cache_identity
                .map_or(Some(0), |identity| {
                    identity.as_ref().logical_metadata_bytes()
                }),
        }
    }

    pub(in crate::composition::mlx::session) fn numerical(&self) -> &PreparedPendingTokenInput<'a> {
        &self.numerical
    }

    /// The fresh request may select a smaller chunk without re-encoding the
    /// saved matrix. Numerical preparation and cache identity remain unchanged.
    pub(super) fn select_chunk_positions(
        &mut self,
        chunk: std::num::NonZeroU64,
    ) -> Result<(), PendingPromptPreparationCause> {
        if chunk.get() > self.numerical.positions() {
            return Err(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch.into());
        }
        if let PendingPromptKind::Prefill {
            chunk_positions, ..
        } = &mut self.kind
        {
            *chunk_positions = Some(chunk);
        }
        Ok(())
    }

    /// Shared numerical native population for the fresh prompt request. This
    /// includes two true nested frontiers and gives no old snapshot grant.
    pub(super) fn original_population(&self) -> Option<PendingTokenNativePopulation> {
        self.numerical.original_population()
    }

    /// Source query/constructor controls only. The pending-part payload and
    /// resumed request/native arenas remain independently admitted components.
    pub(super) fn source_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        PreparedPendingTokenInput::source_control_bytes()?
            .checked_add(PendingTokenInputHostPlan::<Array>::original_control_bytes()?)?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<PendingTokenInputHostPlan<Array>>())?
            .checked_add(size_of::<PendingPromptPreparationCause>())?
            .checked_add(size_of::<Result<Self, PendingPromptPreparationCause>>())
    }

    /// Retained host payload and protected construction P, kept separate from
    /// the exact numerical trace and existing enclosing bookkeeping exclusions.
    pub(super) fn host_plan(&self) -> &PendingTokenInputHostPlan<Array> {
        &self.host
    }

    pub(super) fn trace(
        &self,
        projection: &mut ExistingArrayProjection<'a>,
    ) -> Result<WorkspaceTensor, Error> {
        self.numerical.trace(projection).map_err(other)
    }

    pub(super) fn retained_descriptor_count(&self) -> usize {
        self.numerical.retained_descriptor_count()
    }

    /// Call before numerical work: this checks the exact fresh prompt/account
    /// and protects P before any input part can be constructed. It does not
    /// authorize the independently quoted numerical operations.
    pub(super) fn prepare_host(
        &self,
        completion: InferencePromptCompletion,
        funding: &WorkingMemoryFundingRun,
    ) -> Result<PreparedPendingTokenInputHost<Array>, Error> {
        self.prepare_host_with_authority(completion, funding, None)
    }

    pub(super) fn prepare_host_with_authority(
        &self,
        completion: InferencePromptCompletion,
        funding: &WorkingMemoryFundingRun,
        authority: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<PreparedPendingTokenInputHost<Array>, Error> {
        let positions =
            std::num::NonZeroU64::new(self.numerical.positions()).expect("prepared nonempty input");
        completion
            .prepare_pending_text_input(funding, positions, authority)
            .map_err(other)
    }

    /// Consumes both closed workers. Intermediates remain in the caller's
    /// collector until native settlement/publication; the returned input shares
    /// only its fixed host container and original fresh request. No quote marker
    /// is installed and no runnable resume is enabled here.
    pub(super) fn copy_into_prompt(
        self,
        host: PreparedPendingTokenInputHost<Array>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<(MlxModelInput, InferencePendingPromptCompletion), Error> {
        host.validate().map_err(other)?;
        let tokens = self.numerical.copy_retained(stream, roots).map_err(other)?;
        let (parts, completion) = host.construct(tokens.into_array()).map_err(other)?;
        let request = parts.request().clone();
        let (cache_identity, prefill_chunk_positions) = match self.kind {
            // A decode token preserves the prefix already bound to its state.
            PendingPromptKind::Decode => (None, std::num::NonZeroU64::new(1)),
            PendingPromptKind::Prefill {
                chunk_positions,
                cache_identity,
            } => (cache_identity.cloned(), chunk_positions),
        };
        Ok((
            MlxModelInput {
                controlled_attribution: None,
                prepared_capture: None,
                original_media: None,
                placement_semantics: None,
                parts: ModelInputParts::Pending(parts),
                cache_identity,
                prefill_chunk_positions,
                inference_request: Some(request),
                memory_owner: None,
                quote: None,
            },
            completion,
        ))
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
