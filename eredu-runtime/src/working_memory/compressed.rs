//! Cold metadata for a selected resident compressed-cache storage mechanism.

#[cfg(test)]
mod tests;

use crate::{ResettableRuntimeLayerState, RuntimeLayerState, StateError};
use eredu_nn::{
    CompressedAttentionBlock, CompressedAttentionCache, CompressedAttentionScan,
    CompressedAttentionState, CompressedAttentionView, Error, Index, Tensor,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceDtype, WorkspaceTensor},
};
use std::num::NonZeroU32;

/// Resident compressed-cache metadata with capacity rounded to an explicitly
/// selected positive step. Initialization aliases exact-capacity input or uses
/// zero padding; growth concatenates padding, then replaces the new token range.
/// Native composition must supply the retained mechanism's step and facts.
///
/// Logical views retain their full backing allocation. Metadata checkpoints
/// share identities; restore traces independent backing and logical copies,
/// matching a mechanism which restores all four arrays. Isolated snapshots
/// compact the logical arrays before copying. No native completion is implied.
#[derive(Debug, Clone)]
pub struct WorkspaceCompressedCache {
    context: WorkspaceContext,
    batch: i32,
    latent_width: i32,
    rotary_width: i32,
    step: i32,
    capacity: i32,
    position: i32,
    storage: Option<CompressedAttentionState<WorkspaceTensor>>,
    logical: Option<CompressedAttentionState<WorkspaceTensor>>,
}

const CONSTRUCTOR_EXTENT: &str = "workspace compressed extent exceeds i32";
const PROJECTED_GEOMETRY: &str =
    "existing compressed state differs from selected floating geometry";
const PROJECTED_CAPACITY: &str = "existing compressed state exceeds its backing capacity";

impl WorkspaceCompressedCache {
    /// Validates the actual trace owner, including an empty cache with no arrays.
    pub fn validate_projection_context(&self, context: &WorkspaceContext) -> Result<(), Error> {
        self.validate_context(context)
    }

    /// Validates the exact declared policy and trace of a projected current cache.
    /// This grants no native source, copy or completion authority.
    pub fn validate_projected_policy(
        &self,
        policy: &eredu_core::cache::LayerCachePolicy,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        self.validate_context(context)?;
        match policy {
            eredu_core::cache::LayerCachePolicy::CompressedLatentRotary {
                attention: eredu_core::AttentionPolicy::Full,
                latent_dim,
                rotary_dim,
            } if i32::try_from(latent_dim.get()).ok() == Some(self.latent_width)
                && i32::try_from(rotary_dim.get()).ok() == Some(self.rotary_width) => Ok(()),
            _ => Err(context.metadata_error(format_args!(
                "projected compressed state differs from its full-attention policy"
            ))),
        }
    }

    /// Inline selected compressed constructor/import controls. Tensor imports
    /// and isolated-copy operations retain their own context envelopes.
    pub fn projection_control_bytes() -> Option<usize> {
        let length = [CONSTRUCTOR_EXTENT, PROJECTED_GEOMETRY, PROJECTED_CAPACITY]
            .into_iter()
            .map(str::len)
            .max()?;
        let parts = [
            WorkspaceContext::metadata_error_bytes(length)?,
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<CompressedAttentionState<WorkspaceTensor>>(),
            std::mem::size_of::<(i32, i32)>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    /// Selects the exact local component geometry and native growth increment.
    /// No state tensors or native resources are created here.
    pub fn new(
        batch: NonZeroU32,
        latent_width: NonZeroU32,
        rotary_width: NonZeroU32,
        step: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let extent = |n: NonZeroU32| {
            i32::try_from(n.get())
                .map_err(|_| context.metadata_error(format_args!("{CONSTRUCTOR_EXTENT}")))
        };
        Ok(Self {
            context: context.clone(),
            batch: extent(batch)?,
            latent_width: extent(latent_width)?,
            rotary_width: extent(rotary_width)?,
            step: extent(step)?,
            capacity: 0,
            position: 0,
            storage: None,
            logical: None,
        })
    }
    /// Full allocated token capacity, not the logical token count.
    pub fn capacity(&self) -> i32 {
        self.capacity
    }
    /// Explicitly retained capacity increment.
    pub fn capacity_step(&self) -> i32 {
        self.step
    }
    /// Imports a completed resident state's capacity buffers and logical views.
    /// The provider must project all four arrays together so shared backing
    /// identities survive. Independently copied views remain separate roots.
    /// This records metadata only and never reconstructs state by replaying it.
    pub fn with_existing_state(
        mut self,
        storage: CompressedAttentionState<WorkspaceTensor>,
        logical: CompressedAttentionState<WorkspaceTensor>,
    ) -> Result<Self, Error> {
        let context = &self.context;
        self.context.validate_values([
            &storage.latent,
            &storage.rotary,
            &logical.latent,
            &logical.rotary,
        ])?;
        let validate = |pair: &CompressedAttentionState<WorkspaceTensor>| {
            let shape = pair.latent.shape();
            if shape.len() != 3
                || shape[0] != self.batch
                || shape[1] < 0
                || shape[2] != self.latent_width
                || pair.rotary.shape() != [self.batch, shape[1], self.rotary_width]
                || pair.latent.layout().dtype() != WorkspaceDtype::Float32
                || pair.rotary.layout().dtype() != WorkspaceDtype::Float32
            {
                return Err(context.metadata_error(format_args!("{PROJECTED_GEOMETRY}")));
            }
            Ok(shape[1])
        };
        let capacity = validate(&storage)?;
        let position = validate(&logical)?;
        if position > capacity {
            return Err(context.metadata_error(format_args!("{PROJECTED_CAPACITY}")));
        }
        self.capacity = capacity;
        self.position = position;
        self.storage = Some(storage);
        self.logical = Some(logical);
        Ok(self)
    }
    pub(in crate::working_memory) fn workspace_context(&self) -> &WorkspaceContext {
        &self.context
    }

    pub(in crate::working_memory) fn validate_context(
        &self,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        if !self.context.shares_trace(context) {
            return Err(context.metadata_error(format_args!(
                "compressed state belongs to another workspace trace"
            )));
        }
        Ok(())
    }
    fn copy_pair(
        pair: &Option<CompressedAttentionState<WorkspaceTensor>>,
        compact: bool,
        context: &WorkspaceContext,
    ) -> Result<Option<CompressedAttentionState<WorkspaceTensor>>, Error> {
        pair.as_ref()
            .map(|pair| {
                let copy = |value: &WorkspaceTensor| {
                    if compact {
                        value.contiguous(context)?.deep_copy(context)
                    } else {
                        value.deep_copy(context)
                    }
                };
                Ok(CompressedAttentionState {
                    latent: copy(&pair.latent)?,
                    rotary: copy(&pair.rotary)?,
                })
            })
            .transpose()
    }
    /// Traces independent copies of the backing arrays and their logical views.
    /// The copies have distinct storage identities; a retained view is not
    /// assumed to alias the independently copied capacity buffer.
    pub fn deep_copy_state(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        self.validate_context(context)?;
        let storage = Self::copy_pair(&self.storage, false, context)?;
        let logical = Self::copy_pair(&self.logical, false, context)?;
        Ok(Self {
            storage,
            logical,
            ..self.clone()
        })
    }
    /// Traces a compact isolated snapshot and keeps its logical arrays aliased
    /// to the copied stores. Later growth starts from this compact capacity.
    pub fn isolated_snapshot(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        self.validate_context(context)?;
        let storage = Self::copy_pair(&self.logical, true, context)?;
        Ok(Self {
            logical: storage.clone(),
            storage,
            capacity: self.position,
            ..self.clone()
        })
    }
    /// Borrows every retained array, including distinct backing copies that a
    /// restored logical view does not reach. The allocation ledger deduplicates
    /// aliases and therefore charges ordinary views only once.
    pub fn retained_arrays(&self) -> impl Iterator<Item = &WorkspaceTensor> {
        self.storage
            .iter()
            .chain(self.logical.iter())
            .flat_map(|pair| [&pair.latent, &pair.rotary])
    }
}

impl CompressedAttentionCache<WorkspaceTensor> for WorkspaceCompressedCache {
    type Checkpoint = Self;
    fn offset(&self) -> i32 {
        self.position
    }
    fn is_paged(&self) -> bool {
        false
    }
    fn append(
        &mut self,
        state: CompressedAttentionState<WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<CompressedAttentionView<WorkspaceTensor>, Error> {
        self.validate_context(context)?;
        context.validate_values([&state.latent, &state.rotary])?;
        let latent = state.latent.shape();
        let rotary = state.rotary.shape();
        if latent.len() != 3
            || rotary.len() != 3
            || latent[0] != self.batch
            || latent[1] <= 0
            || latent[2] != self.latent_width
            || rotary != [self.batch, latent[1], self.rotary_width]
            || state.latent.layout().dtype() != WorkspaceDtype::Float32
            || state.rotary.layout().dtype() != WorkspaceDtype::Float32
        {
            return Err(context.metadata_error(format_args!(
                "compressed append differs from selected floating state geometry"
            )));
        }
        let required = self.position.checked_add(latent[1]).ok_or_else(|| {
            context.metadata_error(format_args!("compressed token frontier overflow"))
        })?;
        let capacity = if required <= self.capacity {
            self.capacity
        } else {
            // Division before multiplication avoids spurious overflow at an
            // exact multiple near the native extent limit.
            let chunks = required / self.step + i32::from(required % self.step != 0);
            chunks.checked_mul(self.step).ok_or_else(|| {
                context.metadata_error(format_args!("compressed capacity overflow"))
            })?
        };
        let zeros =
            |tokens, width| WorkspaceTensor::full_f32(0.0, &[self.batch, tokens, width], context);
        let storage = match &self.storage {
            None if capacity == required => state,
            previous => {
                let padded = match previous {
                    None => CompressedAttentionState {
                        latent: zeros(capacity, self.latent_width)?,
                        rotary: zeros(capacity, self.rotary_width)?,
                    },
                    Some(previous) if capacity > self.capacity => {
                        let latent_padding = zeros(capacity - self.capacity, self.latent_width)?;
                        let rotary_padding = zeros(capacity - self.capacity, self.rotary_width)?;
                        CompressedAttentionState {
                            latent: WorkspaceTensor::concatenate(
                                &[previous.latent.clone(), latent_padding],
                                1,
                                context,
                            )?,
                            rotary: WorkspaceTensor::concatenate(
                                &[previous.rotary.clone(), rotary_padding],
                                1,
                                context,
                            )?,
                        }
                    }
                    Some(previous) => previous.clone(),
                };
                CompressedAttentionState {
                    latent: padded.latent.update_slice(
                        &state.latent,
                        &[0, self.position, 0],
                        context,
                    )?,
                    rotary: padded.rotary.update_slice(
                        &state.rotary,
                        &[0, self.position, 0],
                        context,
                    )?,
                }
            }
        };
        let logical = CompressedAttentionState {
            latent: storage
                .latent
                .index(&[Index::Full, Index::Range(0, required)], context)?,
            rotary: storage
                .rotary
                .index(&[Index::Full, Index::Range(0, required)], context)?,
        };
        self.storage = Some(storage);
        self.logical = Some(logical.clone());
        self.capacity = capacity;
        self.position = required;
        Ok(CompressedAttentionView::Resident(logical))
    }
    fn visit_blocks<F>(
        &mut self,
        _: i32,
        context: &WorkspaceContext,
        _: F,
    ) -> Result<CompressedAttentionScan, Error>
    where
        F: FnMut(CompressedAttentionBlock<WorkspaceTensor>) -> Result<u64, Error>,
    {
        Err(context.metadata_error(format_args!(
            "resident compressed mechanism has no paged scan"
        )))
    }
    fn checkpoint(&self) -> Self::Checkpoint {
        self.clone()
    }
    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        self.validate_context(context)?;
        checkpoint.validate_context(context)?;
        if (self.batch, self.latent_width, self.rotary_width, self.step)
            != (
                checkpoint.batch,
                checkpoint.latent_width,
                checkpoint.rotary_width,
                checkpoint.step,
            )
        {
            return Err(context.metadata_error(format_args!(
                "compressed checkpoint changed selected state geometry"
            )));
        }
        *self = checkpoint.deep_copy_state(context)?;
        Ok(())
    }
    fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn clear(&mut self) -> Result<(), Error> {
        self.storage = None;
        self.logical = None;
        self.position = 0;
        self.capacity = 0;
        Ok(())
    }
}

impl RuntimeLayerState<WorkspaceBackend> for WorkspaceCompressedCache {
    type RetainedValues<'a> =
        std::iter::Flatten<std::array::IntoIter<Option<&'a WorkspaceTensor>, 4>>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        [
            self.storage.as_ref().map(|s| &s.latent),
            self.storage.as_ref().map(|s| &s.rotary),
            self.logical.as_ref().map(|s| &s.latent),
            self.logical.as_ref().map(|s| &s.rotary),
        ]
        .into_iter()
        .flatten()
    }
}
impl ResettableRuntimeLayerState<WorkspaceBackend> for WorkspaceCompressedCache {
    fn reset(&mut self) -> Result<(), StateError> {
        self.clear().map_err(StateError::WorkspaceConstruction)
    }
}
