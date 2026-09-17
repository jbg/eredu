//! Cold projection of exact native local/pooling state, preserving all aliases.

use super::*;
use crate::backend::nn::workspace::{ExistingArrayProjection, ProjectedResidentState};
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceContext};
use eredu_runtime::working_memory::{WorkspacePoolingStateFactory, WorkspaceResidentLayerState};
use std::num::NonZeroU32;

mod complete;

#[cfg(test)]
mod tests;

impl MlxPoolingAttentionStateFactory {
    /// Imports current resident local keys, native persistence sentinels, partial
    /// windows, complete pooled history and overlap views. No prefix is replayed
    /// and no array is evaluated. Unknown backing remains unknown in the trace.
    pub fn project_resident_workspace(
        state: &MlxPoolingAttentionState,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>, ComputeError> {
        Self::project_resident_workspace_with_storage(state, batch, context)
            .map(|projected| projected.state)
    }

    /// Retains exact allocation witnesses for the same local/pooling traversal,
    /// including overlap aliases and native persistence sentinels.
    pub fn project_resident_workspace_with_storage(
        state: &MlxPoolingAttentionState,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<ProjectedResidentState, ComputeError> {
        if state
            .as_ref()
            .iter()
            .any(|layer| matches!(layer.local(), LiveKeyValueCache::Paged(_)))
        {
            return Self::project_complete_workspace_with_storage(state, batch, context)
                .map_err(|failure| failure.retire_unsubmitted(context));
        }
        let mut projection = ExistingArrayProjection::new(context);
        let state = match state.shared_layout() {
            Some(layout) => project_layers(
                layout,
                state.as_ref().len(),
                batch,
                context,
                |index| state.as_ref().get(index),
                |array| projection.project(array),
            )?,
            None if state
                .prepare_layer_copy_slots()
                .map_err(|cause| context.metadata_source(cause))?
                .is_none()
                && state.as_ref().is_empty() =>
            {
                DeviceState::stateless()
            }
            None => {
                return Err(context
                    .metadata_error(format_args!("pooling state lacks its actual shared layout")));
            }
        };
        Ok(ProjectedResidentState {
            state,
            storage: projection.into_storage(),
        })
    }
}

/// Shared native geometry validation for imports and closed copied destinations.
/// The caller selects the exact array mapping; no callback is treated as a byte
/// certificate and this metadata traversal grants no execution authority.
pub(super) fn project_layers<'a>(
    layout: &eredu_runtime::SharedStateLayout,
    len: usize,
    batch: NonZeroU32,
    context: &WorkspaceContext,
    mut layer: impl FnMut(usize) -> Option<&'a MlxPoolingAttentionCache>,
    mut project: impl FnMut(&'a Array) -> Result<eredu_nn::workspace::WorkspaceTensor, ComputeError>,
) -> Result<DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>, ComputeError> {
    if len != layout.layout().len() {
        return Err(context.metadata_error(format_args!(
            "pooling state inventory differs from its layout"
        )));
    }
    let factory = WorkspacePoolingStateFactory::new(batch, context)?;
    DeviceState::create_workspace_with_shared_layout(layout.clone(), context, |index, policy| {
        let cache = layer(index)
            .ok_or_else(|| context.metadata_error(format_args!("missing pooling source layer")))?;
        cache
            .project_workspace_layer(index, policy, &factory, context, &mut project)
            .map(WorkspaceResidentLayerState::Pooling)
    })
}

impl MlxPoolingAttentionCache {
    /// Projects this exact current member using architecture-declared policy.
    /// Shared with complete model-state traversal; supplied projection preserves
    /// aliases across prediction members. No native state is copied or replayed.
    pub(crate) fn project_workspace_layer<'a>(
        &'a self,
        index: usize,
        policy: &eredu_core::cache::LayerCachePolicy,
        factory: &WorkspacePoolingStateFactory,
        context: &WorkspaceContext,
        mut project: impl FnMut(&'a Array) -> Result<eredu_nn::workspace::WorkspaceTensor, ComputeError>,
    ) -> Result<eredu_runtime::working_memory::WorkspacePoolingLayerState, ComputeError> {
        let cache = self;
        let geometry =
            eredu_runtime::state::pooling_attention_geometry_plan(index, policy, |args| {
                context.metadata_error(args)
            })?;
        let (position, keys, values) = match cache.local() {
            LiveKeyValueCache::Resident(local) => local.project_workspace_pooling_local_with(
                geometry.sliding_window,
                context,
                &mut project,
            )?,
            LiveKeyValueCache::Paged(_) => {
                return Err(context.metadata_error(format_args!(
                    "resident pooling projection requires resident local backing"
                )));
            }
        };
        let components =
            self.project_components_with(index, policy, position, context, &mut project)?;
        factory
            .project_layer(index, policy, position, keys, values, components)
            .map_err(|cause| cause.into_workspace_error(context))
    }
    fn project_components_with<'a, F>(
        &'a self,
        index: usize,
        policy: &eredu_core::cache::LayerCachePolicy,
        position: i32,
        context: &WorkspaceContext,
        mut project: F,
    ) -> Result<
        Vec<(
            StateTensorRole,
            Option<eredu_nn::workspace::WorkspaceTensor>,
        )>,
        ComputeError,
    >
    where
        F: FnMut(&'a Array) -> Result<eredu_nn::workspace::WorkspaceTensor, ComputeError>,
    {
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            usize,
            &eredu_core::cache::LayerCachePolicy,
            i32,
            &WorkspaceContext,
            F,
            eredu_runtime::state::PoolingAttentionGeometryPlan,
            [Option<&Array>; 5],
            StateTensorRole,
            Option<&eredu_core::cache::StateTensorPolicy>,
            Vec<(
                StateTensorRole,
                Option<eredu_nn::workspace::WorkspaceTensor>,
            )>,
        )>())?;
        let cache = self;
        let geometry =
            eredu_runtime::state::pooling_attention_geometry_plan(index, policy, |args| {
                context.metadata_error(args)
            })?;
        let actual_streams = match cache {
            MlxPoolingAttentionCache::Local(_) => 0,
            MlxPoolingAttentionCache::Compressed { .. } => 1,
            MlxPoolingAttentionCache::Sparse { .. } => 2,
        };
        if actual_streams != geometry.stream_ratios().len() {
            return Err(context.metadata_error(format_args!(
                "native pooling stream inventory differs from its declaration"
            )));
        }
        // Only declared roles are emitted, at most once per stream/component.
        // Counting those same rows avoids capacity estimates from model size.
        let component_count = WorkspacePoolingStateFactory::projection_component_count(policy)
            .ok_or_else(|| {
                context.metadata_error(format_args!("pooling projection declaration changed"))
            })?;
        let mut components = context.metadata_vec(component_count)?;
        for (stream, ratio) in geometry.stream_ratios().iter().copied().enumerate() {
            let pool = cache.pool(stream as u32)?;
            if pool.ratio() != ratio || pool.processed_tokens() != position {
                return Err(context.metadata_error(format_args!(
                    "native pooling ratio or frontier differs from its declaration"
                )));
            }
            for (component, array) in POOLING_COMPONENTS.into_iter().zip(pool.state_arrays()) {
                let role = pooling_role(stream as u32, component);
                let declaration = policy
                    .fixed_state()
                    .iter()
                    .find(|declaration| declaration.role == role);
                if declaration.is_none() {
                    if array.is_some() {
                        return Err(context.metadata_error(format_args!(
                            "native pooling retains an undeclared component"
                        )));
                    }
                    continue;
                }
                if let Some(array) = array {
                    use eredu_core::cache::StateTensorDtype;
                    context.charge_metadata(
                        safemlx::Array::descriptor_control_bytes()
                            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
                    )?;
                    let descriptor = array
                        .try_descriptor()
                        .map_err(|cause| context.metadata_source(cause))?;
                    let dtype = descriptor.facts().dtype();
                    let accepted = match declaration.unwrap().dtype {
                        StateTensorDtype::Floating => matches!(
                            dtype,
                            safemlx::Dtype::Float32
                                | safemlx::Dtype::Float16
                                | safemlx::Dtype::Bfloat16
                        ),
                        StateTensorDtype::Float32 => dtype == safemlx::Dtype::Float32,
                        StateTensorDtype::Int32 => dtype == safemlx::Dtype::Int32,
                        StateTensorDtype::Uint32 => dtype == safemlx::Dtype::Uint32,
                    };
                    drop(descriptor);
                    if !accepted {
                        return Err(context.metadata_error(format_args!(
                            "native pooling dtype differs from its exact declaration"
                        )));
                    }
                }
                components.push((role, array.map(&mut project).transpose()?));
            }
        }
        Ok(components)
    }
}
