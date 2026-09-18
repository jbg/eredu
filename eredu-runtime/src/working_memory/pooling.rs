//! Exact resident or paged local-key and append-only pooling metadata mechanisms.

use super::{WorkspaceConcatLayerState, WorkspaceConcatStateFactory};
use crate::{
    ArchitectureStateFactory, DeviceState, ResettableRuntimeLayerState, RuntimeLayerState,
    StateError, StateLayout,
};
use eredu_core::cache::{
    LayerCachePolicy, PoolingStateComponent, StateTensorDimension, StateTensorDtype,
    StateTensorPolicy, StateTensorRole,
};
use eredu_nn::{
    AttentionCache, Error, Index, NeuralBackend, PoolingAttentionCache, PoolingOverlap,
    PoolingWindows, Tensor, workspace::*,
};
use std::{num::NonZeroU32, rc::Rc};

mod local;
use local::Local;
mod metadata;
mod projection;
use projection::ProjectionFailure;
mod stream;
use stream::WorkspacePoolingStream;
#[cfg(test)]
mod tests;

const COMPONENTS: [PoolingStateComponent; 5] = [
    PoolingStateComponent::PendingValues,
    PoolingStateComponent::PendingGates,
    PoolingStateComponent::Pooled,
    PoolingStateComponent::OverlapValues,
    PoolingStateComponent::OverlapGates,
];

/// Metadata realization of the selected resident pooling mechanism. It retains
/// local keys and their one-channel native sentinel, plus every partial, pooled
/// and overlap view. Native projection supplies actual backing capacities.
#[derive(Debug, Clone)]
pub struct WorkspacePoolingStateFactory {
    local: WorkspaceConcatStateFactory,
    batch: i32,
    context: WorkspaceContext,
}

impl WorkspacePoolingStateFactory {
    /// Selects exact-concatenation resident local state and pooling streams.
    pub fn new(batch: NonZeroU32, context: &WorkspaceContext) -> Result<Self, Error> {
        Ok(Self {
            local: WorkspaceConcatStateFactory::new(batch, context)?,
            batch: i32::try_from(batch.get()).map_err(|cause| context.metadata_source(cause))?,
            context: context.clone(),
        })
    }

    /// Realizes the pooling profile in the shared resident mechanism set.
    /// The caller selects this profile explicitly, including local-only layers.
    pub fn realize_resident(
        &self,
        layout: &StateLayout,
    ) -> Result<DeviceState<WorkspaceBackend, super::WorkspaceResidentLayerState>, StateError> {
        let layout = crate::SharedStateLayout::copy_workspace(layout, &self.context)?;
        DeviceState::create_workspace_with_shared_layout_result(
            layout,
            &self.context,
            |layer, policy| {
                self.create_layer(layer, policy)
                    .map(super::WorkspaceResidentLayerState::Pooling)
            },
        )
    }

    fn create_layer(
        &self,
        layer: usize,
        policy: &LayerCachePolicy,
    ) -> Result<WorkspacePoolingLayerState, StateError> {
        let geometry = crate::state::pooling_attention_geometry_plan(layer, policy, |args| {
            self.context.metadata_error(args)
        })
        .map_err(StateError::WorkspaceConstruction)?;
        let policy =
            metadata::policy(policy, &self.context).map_err(StateError::WorkspaceConstruction)?;
        let mut streams = self
            .context
            .metadata_vec(geometry.stream_ratios().len())
            .map_err(StateError::WorkspaceConstruction)?;
        for (stream, &ratio) in geometry.stream_ratios().iter().enumerate() {
            // Ordinary construction uses the first matching declaration. The
            // shared validation worker independently preserves its last-entry
            // duplicate rule; neither checkpoint cloning nor COW duplicates shapes.
            let declarations = std::array::from_fn(|slot| {
                policy.fixed_state().iter().position(|declaration| {
                    declaration.role
                        == StateTensorRole::Pooling {
                            stream: stream as u32,
                            component: COMPONENTS[slot],
                        }
                })
            });
            streams.push(WorkspacePoolingStream {
                ratio,
                position: 0,
                batch: self.batch,
                policy: policy.clone(),
                declarations,
                values: std::array::from_fn(|_| None),
            });
        }
        let streams = self
            .context
            .metadata_rc(streams)
            .map_err(|cause| StateError::WorkspaceConstruction(cause.into()))?;
        Ok(WorkspacePoolingLayerState {
            local: Local::Resident(self.local.create_pooling_local(layer, &policy)?),
            policy,
            streams,
            context: self.context.clone(),
            batch: self.batch,
        })
    }

    /// Imports the complete retained native layer without replay or native allocation.
    /// Host metadata is constructed through the context's cumulative allowance.
    /// The caller verifies the local mechanism and every stream's actual frontier.
    /// Every declared role must be supplied, including absent components.
    pub fn project_layer(
        &self,
        layer: usize,
        policy: &LayerCachePolicy,
        position: i32,
        keys: Option<WorkspaceTensor>,
        values: Option<WorkspaceTensor>,
        components: impl IntoIterator<Item = (StateTensorRole, Option<WorkspaceTensor>)>,
    ) -> Result<WorkspacePoolingLayerState, StateError> {
        let mut state = self.create_layer(layer, policy)?;
        state.local = Local::Resident(
            self.local
                .project_pooling_local(layer, policy, position, keys, values)?,
        );
        self.project_components(layer, policy, position, state, components)
    }

    /// Imports an actual key-only paged local member and its independently
    /// declared pooling streams. Native source/pin authority stays external.
    pub fn project_paged_layer(
        &self,
        layer: usize,
        policy: &LayerCachePolicy,
        paged: super::WorkspacePagedAppendState,
        components: impl IntoIterator<Item = (StateTensorRole, Option<WorkspaceTensor>)>,
    ) -> Result<WorkspacePoolingLayerState, StateError> {
        self.context
            .charge_metadata(std::mem::size_of::<(
                &Self,
                usize,
                &LayerCachePolicy,
                super::WorkspacePagedAppendState,
                WorkspacePoolingLayerState,
                Result<WorkspacePoolingLayerState, StateError>,
            )>())
            .map_err(|cause| StateError::WorkspaceConstruction(cause.into()))?;
        let local = Local::paged(layer, policy, self.batch, paged, &self.context)?;
        let position = local.offset();
        let mut state = self.create_layer(layer, policy)?;
        state.local = local;
        self.project_components(layer, policy, position, state, components)
    }

    fn component_control_bytes<
        I: IntoIterator<Item = (StateTensorRole, Option<WorkspaceTensor>)>,
    >() -> usize {
        std::mem::size_of::<(
            &Self,
            usize,
            &LayerCachePolicy,
            i32,
            WorkspacePoolingLayerState,
            I,
            I::IntoIter,
            Result<WorkspacePoolingLayerState, StateError>,
        )>()
    }

    fn project_components<I: IntoIterator<Item = (StateTensorRole, Option<WorkspaceTensor>)>>(
        &self,
        layer: usize,
        policy: &LayerCachePolicy,
        position: i32,
        mut state: WorkspacePoolingLayerState,
        components: I,
    ) -> Result<WorkspacePoolingLayerState, StateError> {
        self.context
            .charge_metadata(Self::component_control_bytes::<I>())
            .map_err(|cause| StateError::WorkspaceConstruction(cause.into()))?;
        let invalid = |failure: ProjectionFailure| {
            StateError::workspace_invalid_layer(
                &self.context,
                layer,
                format_args!("{}", failure.reason()),
            )
        };
        // Keep ingestion/duplicate rejection before declared-role validation and
        // unknown-role rejection last, following ordered-map traversal.
        let mut supplied = self
            .context
            .metadata_vec::<(StateTensorRole, Option<Option<WorkspaceTensor>>)>(
                policy.fixed_state().len(),
            )
            .map_err(StateError::WorkspaceConstruction)?;
        for (role, value) in components {
            if supplied.iter().any(|(known, _)| *known == role) {
                return Err(invalid(ProjectionFailure::Duplicate));
            }
            self.context
                .reserve_metadata_vec(&mut supplied, 1)
                .map_err(StateError::WorkspaceConstruction)?;
            supplied.push((role, Some(value)));
        }
        for stream in state
            .streams_mut()
            .map_err(StateError::WorkspaceConstruction)?
        {
            stream.position = position;
            for slot in 0..COMPONENTS.len() {
                let Some(index) = stream.declarations[slot] else {
                    continue;
                };
                let declaration = &stream.policy.fixed_state()[index];
                let value = supplied
                    .iter_mut()
                    .find(|(role, _)| *role == declaration.role)
                    .and_then(|(_, value)| value.take())
                    .ok_or_else(|| invalid(ProjectionFailure::Omitted))?;
                if value.is_some() != declaration.is_required_for(position as usize) {
                    return Err(invalid(ProjectionFailure::Presence));
                }
                if let Some(value) = &value {
                    let mut same_shape = true;
                    let mut dimensions = 0usize;
                    for dimension in
                        declaration.resolved_dimensions(self.batch as usize, position as usize)
                    {
                        let dimension = dimension.map_err(|cause| {
                            StateError::WorkspaceConstruction(self.context.metadata_source(cause))
                        })?;
                        same_shape &= value.shape().get(dimensions).copied() == Some(dimension);
                        dimensions += 1;
                    }
                    same_shape &= dimensions == value.shape().len();
                    if !same_shape {
                        return Err(invalid(ProjectionFailure::Shape));
                    }
                    stream
                        .validate(slot, value, &self.context)
                        .map_err(StateError::WorkspaceConstruction)?;
                }
                stream.values[slot] = value;
            }
        }
        if supplied.iter().any(|(_, value)| value.is_some()) {
            return Err(invalid(ProjectionFailure::Undeclared));
        }
        Ok(state)
    }
}

impl ArchitectureStateFactory<WorkspaceBackend> for WorkspacePoolingStateFactory {
    type State = DeviceState<WorkspaceBackend, WorkspacePoolingLayerState>;
    type Error = StateError;
    fn realize(&mut self, layout: &StateLayout) -> Result<Self::State, StateError> {
        let layout = crate::SharedStateLayout::copy_workspace(layout, &self.context)?;
        DeviceState::create_workspace_with_shared_layout_result(
            layout,
            &self.context,
            |layer, policy| self.create_layer(layer, policy),
        )
    }
}

/// Complete pooling layer with its actual local mechanism. Clones preserve metadata storage identities;
/// they are not independent native copies or new memory reservations.
#[derive(Debug, Clone)]
pub struct WorkspacePoolingLayerState {
    policy: Rc<LayerCachePolicy>,
    local: Local,
    streams: Rc<Vec<WorkspacePoolingStream>>,
    context: WorkspaceContext,
    batch: i32,
}

impl WorkspacePoolingLayerState {
    pub(in crate::working_memory) fn paged_geometry(
        &self,
    ) -> Option<super::WorkspacePagedGeometry> {
        match &self.local {
            Local::Resident(_) => None,
            Local::Paged(state) => Some(state.geometry()),
        }
    }

    /// Borrows the full retained policy of this actual projected layer.
    pub fn projected_policy(&self) -> &LayerCachePolicy {
        &self.policy
    }

    /// Validates complete declared policy and trace of this actual projected state.
    /// Empty cache arrays do not remove policy or source-context requirements.
    pub fn validate_projected_policy(
        &self,
        policy: &LayerCachePolicy,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        self.validate_context(context)?;
        if self.policy.as_ref() == policy {
            Ok(())
        } else {
            Err(context.metadata_error(format_args!(
                "projected pooling state differs from its declared policy"
            )))
        }
    }

    pub(in crate::working_memory) fn workspace_context(&self) -> &WorkspaceContext {
        &self.context
    }
    pub(in crate::working_memory) fn validate_context(
        &self,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        if self.context.shares_trace(context) {
            Ok(())
        } else {
            Err(context.metadata_error(format_args!(
                "pooling state belongs to another workspace trace"
            )))
        }
    }
    fn stream(&self, id: u32) -> Result<&WorkspacePoolingStream, Error> {
        self.streams.get(id as usize).ok_or_else(|| {
            self.context
                .metadata_error(format_args!("pooling stream is not declared"))
        })
    }
    fn stream_mut(&mut self, id: u32) -> Result<&mut WorkspacePoolingStream, Error> {
        if id as usize >= self.streams.len() {
            return Err(self
                .context
                .metadata_error(format_args!("pooling stream is not declared")));
        }
        Ok(&mut self.streams_mut()?[id as usize])
    }
}

impl PoolingAttentionCache<WorkspaceTensor> for WorkspacePoolingLayerState {
    type Checkpoint = Self;
    fn offset(&self) -> i32 {
        self.local.offset()
    }
    fn pooling_ratio(&self, stream: u32) -> Option<i32> {
        self.streams.get(stream as usize).map(|stream| stream.ratio)
    }
    fn append_local(
        &mut self,
        keys: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.validate_context(context)?;
        if keys.shape().len() != 3 || keys.shape()[0] != self.batch || keys.shape()[1] <= 0 {
            return Err(context.metadata_error(format_args!(
                "local pooling keys require positive [batch, tokens, width]"
            )));
        }
        let tokens = keys.shape()[1];
        let keys = keys.expand_dims(1, context)?;
        let sentinel = WorkspaceTensor::full_f32(
            0.,
            &[self.batch, 1, tokens, self.local.logical_value_width()],
            context,
        )?;
        let [keys, _] = self.local.update(keys, sentinel, context)?;
        keys.index(
            &[Index::Full, Index::At(0), Index::Full, Index::Full],
            context,
        )
    }
    fn local_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.validate_context(context)?;
        let window = self.local.window();
        if offset < 0 || query_tokens < 0 || offset.checked_add(query_tokens).is_none() {
            return Err(
                context.metadata_error(format_args!("invalid local pooling mask positions"))
            );
        }
        WorkspaceBackend::causal_mask(
            query_tokens,
            offset.min(window - 1),
            Some(window - 1),
            context,
        )
    }
    fn accumulate_pooling_windows(
        &mut self,
        stream: u32,
        values: WorkspaceTensor,
        gates: WorkspaceTensor,
        absolute_offset: i32,
        context: &WorkspaceContext,
    ) -> Result<PoolingWindows<WorkspaceTensor>, Error> {
        self.validate_context(context)?;
        self.stream_mut(stream)?
            .accumulate(values, gates, absolute_offset, context)
    }
    fn replace_pooling_overlap(
        &mut self,
        stream: u32,
        values: WorkspaceTensor,
        gates: WorkspaceTensor,
    ) -> Result<PoolingOverlap<WorkspaceTensor>, Error> {
        let context = self.context.clone();
        let stream = self.stream_mut(stream)?;
        stream.validate(3, &values, &context)?;
        stream.validate(4, &gates, &context)?;
        if values.shape()[1] != stream.ratio || gates.shape()[1] != stream.ratio {
            return Err(context.metadata_error(format_args!(
                "pooling overlap must retain one complete source group"
            )));
        }
        Ok(PoolingOverlap {
            values: stream.values[3].replace(values),
            gates: stream.values[4].replace(gates),
        })
    }
    fn append_pooled(
        &mut self,
        stream: u32,
        values: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.validate_context(context)?;
        self.stream_mut(stream)?.append(values, context)
    }
    fn pooling_mask(
        &self,
        stream: u32,
        query_tokens: i32,
        offset: i32,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        self.validate_context(context)?;
        let stream = self.stream(stream)?;
        let pooled = stream.values[2]
            .as_ref()
            .map_or(0, |value| value.shape()[1]);
        let geometry = eredu_nn::operation_geometry::PoolingMaskGeometry::new_fixed(
            query_tokens,
            pooled,
            offset,
            stream.ratio,
        )
        .map_err(|cause| context.metadata_source(cause))?;
        if pooled == 0 || query_tokens == 1 {
            return Ok(None);
        }
        let mut outputs = context.metadata_vec(1)?;
        outputs.push(context.layout(&[query_tokens, pooled], WorkspaceDtype::Bool)?);
        Ok(Some(
            context
                .execute(WorkspaceOperationKind::PoolingMask(geometry), &[], outputs)?
                .remove(0),
        ))
    }

    fn checkpoint(&self) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn restore(&mut self, checkpoint: &Self, context: &WorkspaceContext) -> Result<(), Error> {
        self.validate_context(context)?;
        checkpoint.validate_context(context)?;
        if self.batch != checkpoint.batch || self.policy != checkpoint.policy {
            return Err(context.metadata_error(format_args!(
                "pooling checkpoint differs from selected state geometry"
            )));
        }
        *self = checkpoint.clone();
        Ok(())
    }
    fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn clear(&mut self) -> Result<(), Error> {
        // Complete any COW allocation before changing the local frontier.
        self.streams_mut()?;
        self.local.reset(&self.context)?;
        for stream in Rc::get_mut(&mut self.streams).expect("unique cleared stream rows") {
            stream.position = 0;
            stream.values = std::array::from_fn(|_| None);
        }
        Ok(())
    }
}

type StreamValues<'a> = std::iter::Flatten<std::slice::Iter<'a, Option<WorkspaceTensor>>>;
/// Allocation-free traversal of all local, partial, pooled and overlap owners.
pub struct WorkspacePoolingValues<'a> {
    local: local::Values<'a>,
    streams: std::slice::Iter<'a, WorkspacePoolingStream>,
    current: Option<StreamValues<'a>>,
}
impl<'a> Iterator for WorkspacePoolingValues<'a> {
    type Item = &'a WorkspaceTensor;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(value) = self.local.next() {
            return Some(value);
        }
        loop {
            if let Some(value) = self.current.as_mut().and_then(Iterator::next) {
                return Some(value);
            }
            self.current = Some(self.streams.next()?.values.iter().flatten());
        }
    }
}
impl RuntimeLayerState<WorkspaceBackend> for WorkspacePoolingLayerState {
    type RetainedValues<'a> = WorkspacePoolingValues<'a>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        WorkspacePoolingValues {
            local: self.local.values(),
            streams: self.streams.iter(),
            current: None,
        }
    }
}
impl ResettableRuntimeLayerState<WorkspaceBackend> for WorkspacePoolingLayerState {
    fn reset(&mut self) -> Result<(), StateError> {
        self.clear().map_err(StateError::WorkspaceConstruction)
    }
}
