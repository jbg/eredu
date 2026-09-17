//! Typed resident state combining selected cache and fixed-component mechanisms.

use super::{WorkspaceCompressedCache, WorkspaceConcatLayerState, WorkspaceConcatStateFactory};
use crate::{
    ArchitectureStateFactory, DeviceState, ResettableRuntimeLayerState, RuntimeLayerState,
    RuntimeStateComponents, StateError, StateLayout,
};
use eredu_core::cache::{LayerCachePolicy, StateTensorRole};
use eredu_nn::{
    AttentionCache, AttentionRequest, AuxiliaryConvolutionState, CompressedAttentionBlock,
    CompressedAttentionCache, CompressedAttentionScan, CompressedAttentionState,
    CompressedAttentionView, Error,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceTensor},
};
use std::num::NonZeroU32;

mod pooling;
use super::{WorkspacePagedLayerState, WorkspacePagedValues, WorkspacePoolingLayerState};

/// Realizes architecture state with exact-concatenation ordinary attention and
/// capacity-rounded resident compressed attention. Composition supplies the
/// compressed growth increment from its retained native mechanism. Fixed slots
/// continue to be populated by the architecture's ordinary equations.
///
/// This is a metadata mechanism set, not a declaration that paged, quantized or
/// offloaded state has the same allocation behavior. All six ordinary state
/// access profiles can use its concrete layer type without a family branch.
#[derive(Debug, Clone)]
pub struct WorkspaceResidentStateFactory {
    ordinary: WorkspaceConcatStateFactory,
    batch: NonZeroU32,
    compressed_step: NonZeroU32,
    context: WorkspaceContext,
}
impl WorkspaceResidentStateFactory {
    /// Retains request batch, compressed capacity increment and cold facts.
    pub fn new(
        batch: NonZeroU32,
        compressed_step: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        i32::try_from(compressed_step.get()).map_err(|_| {
            context.metadata_error(format_args!(
                "compressed capacity step exceeds tensor extent"
            ))
        })?;
        Ok(Self {
            ordinary: WorkspaceConcatStateFactory::new(batch, context)?,
            batch,
            compressed_step,
            context: context.clone(),
        })
    }
}

mod selection;
pub use selection::validate_workspace_state_realization;

/// Exact access to ordinary/fixed or compressed resident state. A wrong cache
/// operation returns an error; the architecture's typed profile and StateLayout
/// determine which operation is valid for each execution unit.
#[derive(Debug, Clone)]
pub enum WorkspaceResidentLayerState {
    /// Exact-concatenation ordinary attention and/or fixed components.
    Ordinary(WorkspaceConcatLayerState),
    /// Exact ordered paged blocks/tail with separately retained native source pins.
    Paged(WorkspacePagedLayerState),
    /// Resident compressed attention with selected capacity rounding.
    Compressed(WorkspaceCompressedCache),
    /// Local key history, partial pooling windows and retained overlap.
    Pooling(WorkspacePoolingLayerState),
}

impl WorkspaceResidentLayerState {
    fn workspace_context(&self) -> &WorkspaceContext {
        match self {
            Self::Ordinary(state) => state.workspace_context(),
            Self::Paged(state) => state.workspace_context(),
            Self::Compressed(state) => state.workspace_context(),
            Self::Pooling(state) => state.workspace_context(),
        }
    }

    pub(crate) fn validate_workspace_context(
        &self,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        match self {
            Self::Ordinary(state) => state.validate_context(context),
            Self::Paged(state) => state.validate_context(context),
            Self::Pooling(state) => state.validate_context(context),
            Self::Compressed(state) => state.validate_context(context),
        }
    }

    /// Quotes the selected resident transaction checkpoint mechanism. Ordinary
    /// exact-concatenation and pooling caches retain immutable buffer handles;
    /// compressed caches copy both capacity buffers and logical views. Calling
    /// this on the checkpoint also bounds its rollback copies. This does not
    /// describe an isolated snapshot, whose copy policy is different.
    pub fn checkpoint_for_transaction(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        match self {
            Self::Ordinary(state) => {
                state.validate_context(context)?;
                Ok(self.clone())
            }
            Self::Pooling(state) => {
                state.validate_context(context)?;
                Ok(self.clone())
            }
            Self::Compressed(state) => state.deep_copy_state(context).map(Self::Compressed),
            Self::Paged(state) => state.checkpoint_for_transaction(context).map(Self::Paged),
        }
    }
}
impl ArchitectureStateFactory<WorkspaceBackend> for WorkspaceResidentStateFactory {
    type State = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;
    type Error = StateError;
    fn realize(&mut self, layout: &StateLayout) -> Result<Self::State, Self::Error> {
        let layout = crate::SharedStateLayout::copy_workspace(layout, &self.context)?;
        DeviceState::create_workspace_with_shared_layout_result(
            layout,
            &self.context,
            |layer, policy| match policy {
                LayerCachePolicy::CompressedLatentRotary {
                    latent_dim,
                    rotary_dim,
                    ..
                } => WorkspaceCompressedCache::new(
                    self.batch,
                    *latent_dim,
                    *rotary_dim,
                    self.compressed_step,
                    &self.context,
                )
                .map(WorkspaceResidentLayerState::Compressed)
                .map_err(StateError::WorkspaceConstruction),
                _ => self
                    .ordinary
                    .create_layer(layer, policy)
                    .map(WorkspaceResidentLayerState::Ordinary),
            },
        )
    }
}

/// Allocation-free borrowed iterator over the selected layer's retained arrays.
pub struct WorkspaceResidentValues<'a> {
    paged: Option<WorkspacePagedValues<'a>>,
    pooling: Option<super::WorkspacePoolingValues<'a>>,
    ordinary: Option<
        <WorkspaceConcatLayerState as RuntimeLayerState<WorkspaceBackend>>::RetainedValues<'a>,
    >,
    compressed: Option<
        <WorkspaceCompressedCache as RuntimeLayerState<WorkspaceBackend>>::RetainedValues<'a>,
    >,
}
impl<'a> Iterator for WorkspaceResidentValues<'a> {
    type Item = &'a WorkspaceTensor;
    fn next(&mut self) -> Option<Self::Item> {
        self.ordinary
            .as_mut()
            .and_then(Iterator::next)
            .or_else(|| self.compressed.as_mut().and_then(Iterator::next))
            .or_else(|| self.pooling.as_mut().and_then(Iterator::next))
            .or_else(|| self.paged.as_mut().and_then(Iterator::next))
    }
}
impl RuntimeLayerState<WorkspaceBackend> for WorkspaceResidentLayerState {
    type RetainedValues<'a> = WorkspaceResidentValues<'a>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        match self {
            Self::Ordinary(state) => WorkspaceResidentValues {
                paged: None,
                pooling: None,
                ordinary: Some(state.retained_values()),
                compressed: None,
            },
            Self::Compressed(state) => WorkspaceResidentValues {
                paged: None,
                pooling: None,
                ordinary: None,
                compressed: Some(state.retained_values()),
            },
            Self::Pooling(state) => WorkspaceResidentValues {
                paged: None,
                pooling: Some(state.retained_values()),
                ordinary: None,
                compressed: None,
            },
            Self::Paged(state) => WorkspaceResidentValues {
                paged: Some(state.retained_values()),
                pooling: None,
                ordinary: None,
                compressed: None,
            },
        }
    }
}
impl RuntimeStateComponents<WorkspaceBackend> for WorkspaceResidentLayerState {
    fn position(&self) -> i32 {
        match self {
            Self::Ordinary(s) => s.position(),
            Self::Paged(s) => s.position(),
            Self::Compressed(s) => s.offset(),
            Self::Pooling(s) => eredu_nn::PoolingAttentionCache::offset(s),
        }
    }
    fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<WorkspaceTensor>, StateError> {
        match self {
            Self::Ordinary(s) => s.fixed_component(role),
            Self::Paged(s) => s.fixed_component(role),
            Self::Compressed(_) | Self::Pooling(_) => Err(StateError::UnknownComponent { role }),
        }
    }
    fn advance_fixed(&mut self, tokens: i32) -> Result<(), StateError> {
        let context = self.workspace_context().clone();
        match self {
            Self::Ordinary(s) => s.advance_fixed(tokens),
            Self::Paged(s) => s.advance_fixed(tokens),
            Self::Compressed(_) | Self::Pooling(_) => Err(StateError::workspace_invalid_advance(
                &context,
                format_args!("attention layers advance through cache append"),
            )),
        }
    }
}
impl ResettableRuntimeLayerState<WorkspaceBackend> for WorkspaceResidentLayerState {
    fn reset(&mut self) -> Result<(), StateError> {
        match self {
            Self::Ordinary(s) => s.reset(),
            Self::Paged(s) => s.reset(),
            Self::Compressed(s) => s.reset(),
            Self::Pooling(s) => s.reset(),
        }
    }
}
impl AttentionCache<WorkspaceTensor> for WorkspaceResidentLayerState {
    fn uses_blockwise_attention(&self) -> bool {
        matches!(self, Self::Paged(_))
    }
    fn offset(&self) -> i32 {
        self.position()
    }
    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Ordinary(s) => s.max_size(),
            Self::Paged(s) => s.max_size(),
            Self::Compressed(_) | Self::Pooling(_) => None,
        }
    }
    fn update_for_attention(
        &mut self,
        keys: WorkspaceTensor,
        values: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(WorkspaceTensor, WorkspaceTensor), Error> {
        match self {
            Self::Ordinary(s) => s.update_for_attention(keys, values, context),
            Self::Paged(s) => s.update_for_attention(keys, values, context),
            Self::Compressed(_) | Self::Pooling(_) => Err(context
                .metadata_error(format_args!("layer uses a specialized attention mechanism"))),
        }
    }
    fn attention(
        &mut self,
        request: AttentionRequest<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        match self {
            Self::Ordinary(s) => s.attention(request, context),
            Self::Paged(s) => s.attention(request, context),
            Self::Compressed(_) | Self::Pooling(_) => Err(context
                .metadata_error(format_args!("layer uses a specialized attention mechanism"))),
        }
    }
    fn relative_attention<B: eredu_nn::NeuralBackend<Tensor = WorkspaceTensor>>(
        &mut self,
        request: eredu_nn::RelativeAttentionInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        match self {
            Self::Ordinary(state) => state.relative_attention::<B>(request, context),
            Self::Paged(state) => state.relative_attention::<B>(request, context),
            Self::Compressed(_) | Self::Pooling(_) => Err(context
                .metadata_error(format_args!("layer uses a specialized attention mechanism"))),
        }
    }
}
impl AuxiliaryConvolutionState<WorkspaceTensor> for WorkspaceResidentLayerState {
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<WorkspaceTensor>, Error> {
        match self {
            Self::Ordinary(state) => state.convolution_state(slot),
            Self::Paged(state) => state.convolution_state(slot),
            Self::Compressed(state) => Err(StateError::UnknownComponent {
                role: StateTensorRole::Convolution { slot },
            }
            .into_workspace_error(state.workspace_context())),
            Self::Pooling(state) => Err(StateError::UnknownComponent {
                role: StateTensorRole::Convolution { slot },
            }
            .into_workspace_error(state.workspace_context())),
        }
    }
}
impl CompressedAttentionCache<WorkspaceTensor> for WorkspaceResidentLayerState {
    type Checkpoint = Option<WorkspaceCompressedCache>;
    fn offset(&self) -> i32 {
        self.position()
    }
    fn is_paged(&self) -> bool {
        false
    }
    fn append(
        &mut self,
        state: CompressedAttentionState<WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<CompressedAttentionView<WorkspaceTensor>, Error> {
        match self {
            Self::Compressed(s) => s.append(state, context),
            Self::Ordinary(_) | Self::Pooling(_) | Self::Paged(_) => {
                Err(context.metadata_error(format_args!("layer has no compressed attention state")))
            }
        }
    }
    fn visit_blocks<F>(
        &mut self,
        query_tokens: i32,
        context: &WorkspaceContext,
        visitor: F,
    ) -> Result<CompressedAttentionScan, Error>
    where
        F: FnMut(CompressedAttentionBlock<WorkspaceTensor>) -> Result<u64, Error>,
    {
        match self {
            Self::Compressed(s) => s.visit_blocks(query_tokens, context, visitor),
            Self::Ordinary(_) | Self::Pooling(_) | Self::Paged(_) => {
                Err(context.metadata_error(format_args!("layer has no compressed attention state")))
            }
        }
    }
    fn checkpoint(&self) -> Self::Checkpoint {
        match self {
            Self::Compressed(s) => Some(s.checkpoint()),
            Self::Ordinary(_) | Self::Pooling(_) | Self::Paged(_) => None,
        }
    }
    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        match (self, checkpoint) {
            (Self::Compressed(s), Some(checkpoint)) => s.restore(checkpoint, context),
            _ => Err(context.metadata_error(format_args!(
                "checkpoint does not describe this compressed layer"
            ))),
        }
    }
    fn finalize(&mut self) -> Result<(), Error> {
        let context = self.workspace_context().clone();
        match self {
            Self::Compressed(s) => s.finalize(),
            Self::Ordinary(_) | Self::Pooling(_) | Self::Paged(_) => {
                Err(context.metadata_error(format_args!("layer has no compressed attention state")))
            }
        }
    }
    fn clear(&mut self) -> Result<(), Error> {
        let context = self.workspace_context().clone();
        match self {
            Self::Compressed(s) => s.clear(),
            Self::Ordinary(_) | Self::Pooling(_) | Self::Paged(_) => {
                Err(context.metadata_error(format_args!("layer has no compressed attention state")))
            }
        }
    }
}
