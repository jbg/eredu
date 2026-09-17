//! Metadata realization of exact-concatenation attention and fixed state.

#[cfg(test)]
mod tests;

use crate::{
    ArchitectureStateFactory, DeviceState, ResettableRuntimeLayerState, RuntimeLayerState,
    RuntimeStateComponents, StateError, StateLayout,
};
use eredu_core::cache::{LayerCachePolicy, StateTensorRole};
use eredu_nn::{
    AttentionCache, AttentionRequest, AuxiliaryConvolutionState, Error, Index, NeuralBackend,
    Tensor,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceTensor},
};
use std::num::NonZeroU32;

mod fixed;
use fixed::FixedSlots;

mod projection;
mod paged;
pub use paged::{WorkspacePagedLayerState, WorkspacePagedValues};

/// Traces an explicitly selected exact-concatenation cache mechanism. Sliding
/// history retains views of the complete append allocation; key-only history
/// shares its keys with the attention value input. Fixed slots are addressed by
/// architecture roles and populated by the ordinary equations.
///
/// This factory does not select a native cache or represent capacity-backed,
/// compressed, paged or quantized caches. Composition must match the retained
/// mechanism selection before using its trace in an admission estimate.
#[derive(Debug, Clone)]
pub struct WorkspaceConcatStateFactory {
    batch: i32,
    context: WorkspaceContext,
}

impl WorkspaceConcatStateFactory {
    /// One factory's inline controls and the exact batch conversion diagnostic.
    pub fn construction_control_bytes() -> Option<usize> {
        let parts = [
            WorkspaceContext::metadata_error_bytes(projection::BATCH_EXTENT.len())?,
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<std::num::TryFromIntError>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    /// Retains the request batch and cold allocation ledger without allocating
    /// native state or precharging uninitialized fixed components.
    pub fn new(batch: NonZeroU32, context: &WorkspaceContext) -> Result<Self, Error> {
        Ok(Self {
            batch: i32::try_from(batch.get()).map_err(|_| {
                context.metadata_error(format_args!("{}", projection::BATCH_EXTENT))
            })?,
            context: context.clone(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct AttentionGeometry {
    heads: i32,
    width: i32,
    value_width: i32,
    window: Option<i32>,
    key_only: bool,
}

/// Metadata state for the exact-concatenation mechanism. Cloning retains storage
/// identities and never refunds the trace ledger. Native isolated snapshots
/// require their own selected copy mechanism and reservation.
#[derive(Debug, Clone)]
pub struct WorkspaceConcatLayerState {
    batch: i32,
    context: WorkspaceContext,
    attention: Option<AttentionGeometry>,
    keys: Option<WorkspaceTensor>,
    values: Option<WorkspaceTensor>,
    fixed: FixedSlots,
    position: i32,
}

impl ArchitectureStateFactory<WorkspaceBackend> for WorkspaceConcatStateFactory {
    type State = DeviceState<WorkspaceBackend, WorkspaceConcatLayerState>;
    type Error = StateError;

    fn realize(&mut self, layout: &StateLayout) -> Result<Self::State, Self::Error> {
        let layout = crate::SharedStateLayout::copy_workspace(layout, &self.context)?;
        DeviceState::create_workspace_with_shared_layout_result(
            layout,
            &self.context,
            |layer, policy| self.create_layer(layer, policy),
        )
    }
}

impl WorkspaceConcatStateFactory {
    pub(super) fn create_layer(
        &self,
        layer: usize,
        policy: &LayerCachePolicy,
    ) -> Result<WorkspaceConcatLayerState, StateError> {
        let invalid = |reason: &str| {
            StateError::workspace_invalid_layer(&self.context, layer, format_args!("{reason}"))
        };
        let attention = match policy {
            LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => None,
            LayerCachePolicy::KeyValue {
                num_key_value_heads,
                head_dim,
                attention,
            }
            | LayerCachePolicy::KeyValueWithFixedState {
                num_key_value_heads,
                head_dim,
                attention,
                ..
            }
            | LayerCachePolicy::KeyOnly {
                num_key_heads: num_key_value_heads,
                head_dim,
                attention,
            }
            | LayerCachePolicy::KeyOnlyWithFixedState {
                num_key_heads: num_key_value_heads,
                head_dim,
                attention,
                ..
            } => Some(AttentionGeometry {
                heads: i32::try_from(num_key_value_heads.get())
                    .map_err(|_| invalid(projection::HEAD_COUNT))?,
                width: i32::try_from(head_dim.get())
                    .map_err(|_| invalid(projection::HEAD_WIDTH))?,
                value_width: i32::try_from(head_dim.get())
                    .map_err(|_| invalid(projection::VALUE_WIDTH))?,
                window: attention.sliding_window_i32().map_err(|e| {
                    StateError::workspace_invalid_layer(&self.context, layer, format_args!("{e}"))
                })?,
                key_only: matches!(
                    policy,
                    LayerCachePolicy::KeyOnly { .. }
                        | LayerCachePolicy::KeyOnlyWithFixedState { .. }
                ),
            }),
            LayerCachePolicy::CompressedLatentRotary { .. } => {
                return Err(invalid(projection::COMPRESSED_MECHANISM));
            }
        };
        Ok(WorkspaceConcatLayerState {
            batch: self.batch,
            context: self.context.clone(),
            attention,
            keys: None,
            values: None,
            fixed: FixedSlots::new(policy, &self.context)
                .map_err(StateError::WorkspaceConstruction)?,
            position: 0,
        })
    }

    /// Selected pooling realization: local keys use a separate one-channel
    /// persistence sentinel, whose retained allocation must also be traced.
    pub(super) fn create_pooling_local(
        &self,
        layer: usize,
        policy: &LayerCachePolicy,
    ) -> Result<WorkspaceConcatLayerState, StateError> {
        let mut state = self.create_layer(layer, policy)?;
        let Some(geometry) = state.attention.as_mut().filter(|geometry| {
            geometry.key_only && geometry.heads == 1 && geometry.window.is_some()
        }) else {
            return Err(StateError::workspace_invalid_layer(
                &self.context,
                layer,
                format_args!("{}", projection::POOLING_LOCAL),
            ));
        };
        geometry.key_only = false;
        geometry.value_width = 1;
        state.fixed.clear();
        Ok(state)
    }
}

impl WorkspaceConcatLayerState {
    pub(in crate::working_memory) fn workspace_context(&self) -> &WorkspaceContext {
        &self.context
    }

    pub(in crate::working_memory) fn validate_context(
        &self,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        if !self.context.shares_trace(context) {
            return Err(
                context.metadata_error(format_args!("workspace state belongs to another trace"))
            );
        }
        Ok(())
    }
}

impl AttentionCache<WorkspaceTensor> for WorkspaceConcatLayerState {
    fn offset(&self) -> i32 {
        self.position
    }
    fn max_size(&self) -> Option<i32> {
        self.attention.and_then(|a| a.window)
    }

    fn update_for_attention(
        &mut self,
        keys: WorkspaceTensor,
        values: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(WorkspaceTensor, WorkspaceTensor), Error> {
        self.validate_context(context)?;
        context.validate_values([&keys, &values])?;
        let geometry = self
            .attention
            .ok_or_else(|| context.metadata_error(format_args!("layer has no key/value state")))?;
        let shape = keys.shape();
        if shape.len() != 4
            || shape[0] != self.batch
            || shape[1] != geometry.heads
            || shape[2] <= 0
            || shape[3] != geometry.width
            || (!geometry.key_only
                && (values.shape() != [self.batch, geometry.heads, shape[2], geometry.value_width]
                    || values.layout().dtype() != keys.layout().dtype()))
            || self
                .keys
                .as_ref()
                .is_some_and(|old| old.layout().dtype() != keys.layout().dtype())
        {
            return Err(context.metadata_error(format_args!(
                "workspace cache append differs from selected state geometry"
            )));
        }
        let position = self.position.checked_add(shape[2]).ok_or_else(|| {
            context.metadata_error(format_args!("workspace cache position overflow"))
        })?;
        let append = |previous: &Option<WorkspaceTensor>, value: WorkspaceTensor| match previous {
            Some(previous) => WorkspaceTensor::concatenate(&[previous.clone(), value], 2, context),
            None => Ok(value),
        };
        let keys = append(&self.keys, keys)?;
        let values = if geometry.key_only {
            keys.clone()
        } else {
            append(&self.values, values)?
        };
        let tail = |value: &WorkspaceTensor| -> Result<Option<WorkspaceTensor>, Error> {
            let length = value.shape()[2];
            match geometry.window.map(|window| (window - 1).min(length)) {
                Some(0) => Ok(None),
                Some(keep) => value
                    .index(
                        &[
                            Index::Full,
                            Index::Full,
                            Index::Range(length - keep, length),
                        ],
                        context,
                    )
                    .map(Some),
                _ => Ok(Some(value.clone())),
            }
        };
        let retained_keys = tail(&keys)?;
        let retained_values = if geometry.key_only {
            None
        } else {
            tail(&values)?
        };
        // Publish only after every operation succeeds. Charges from successful
        // operations preceding a failure remain in the containing trace.
        self.keys = retained_keys;
        self.values = retained_values;
        self.position = position;
        Ok((keys, values))
    }

    fn attention(
        &mut self,
        request: AttentionRequest<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.validate_context(context)?;
        if self.attention.is_none() {
            return Err(context.metadata_error(format_args!("layer has no key/value state")));
        }
        WorkspaceBackend::attention_with_sinks(request, context)
    }
}

impl RuntimeLayerState<WorkspaceBackend> for WorkspaceConcatLayerState {
    type RetainedValues<'a> = std::iter::Chain<
        std::iter::Chain<
            std::option::Iter<'a, WorkspaceTensor>,
            std::option::Iter<'a, WorkspaceTensor>,
        >,
        std::iter::Flatten<std::slice::Iter<'a, Option<WorkspaceTensor>>>,
    >;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.keys
            .iter()
            .chain(self.values.iter())
            .chain(self.fixed.values().flatten())
    }
}
impl RuntimeStateComponents<WorkspaceBackend> for WorkspaceConcatLayerState {
    fn position(&self) -> i32 {
        self.position
    }
    fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<WorkspaceTensor>, StateError> {
        self.fixed
            .get_mut(&role, &self.context)
            .map_err(StateError::WorkspaceConstruction)?
            .ok_or(StateError::UnknownComponent { role })
    }
    fn advance_fixed(&mut self, tokens: i32) -> Result<(), StateError> {
        if self.attention.is_some() || tokens <= 0 {
            return Err(StateError::workspace_invalid_advance(
                &self.context,
                format_args!("fixed advancement requires a fixed-only layer and positive count"),
            ));
        }
        self.position = self.position.checked_add(tokens).ok_or_else(|| {
            StateError::workspace_invalid_advance(
                &self.context,
                format_args!("fixed-state token frontier overflowed"),
            )
        })?;
        Ok(())
    }
}
impl AuxiliaryConvolutionState<WorkspaceTensor> for WorkspaceConcatLayerState {
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<WorkspaceTensor>, Error> {
        let context = self.context.clone();
        self.fixed_component(StateTensorRole::Convolution { slot })
            .map_err(|error| error.into_workspace_error(&context))
    }
}
impl ResettableRuntimeLayerState<WorkspaceBackend> for WorkspaceConcatLayerState {
    fn reset(&mut self) -> Result<(), StateError> {
        self.fixed
            .values_mut(&self.context)
            .map_err(StateError::WorkspaceConstruction)?
            .iter_mut()
            .for_each(|slot| *slot = None);
        self.keys = None;
        self.values = None;
        self.position = 0;
        Ok(())
    }
}
