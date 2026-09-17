use super::*;
use crate::backend::nn::workspace::{ExistingArrayProjection, ProjectedResidentState};
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceContext};
use eredu_runtime::working_memory::{WorkspaceConcatStateFactory, WorkspaceResidentLayerState};
use std::num::NonZeroU32;
pub(super) mod complete;

#[cfg(test)]
mod tests;

impl MlxHybridState {
    /// Projects the retained resident attention and fixed tensors together.
    /// Architecture declarations validate geometry; the native owners supply
    /// storage identity, capacity, selected mechanism and actual frontier.
    /// No state is reconstructed by replaying a cached prefix.
    pub fn project_resident_workspace(
        &self,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<
        eredu_runtime::DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
        eredu_nn::Error,
    > {
        self.project_resident_workspace_with_storage(batch, context)
            .map(|projected| projected.state)
    }

    /// Includes the unique physical witnesses from this complete state traversal.
    /// The witnesses pin native handles only until quote binding finishes.
    pub fn project_resident_workspace_with_storage(
        &self,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<ProjectedResidentState, eredu_nn::Error> {
        if self.layers.slots().iter().any(|layer| {
            matches!(
                layer.attention.as_ref(),
                Some(MlxHybridAttentionState::KeyValue(
                    MlxKeyValueLayerState::Paged(_)
                ))
            )
        }) {
            return self
                .project_complete_workspace_with_storage(batch, context)
                .map_err(|failure| failure.retire_unsubmitted(context));
        }
        let mut projection = ExistingArrayProjection::new(context);
        let state = self.project_workspace_state(batch, &mut projection)?;
        Ok(ProjectedResidentState {
            state,
            storage: projection.into_storage(),
        })
    }

    /// Imports current state through the caller's shared storage identity set.
    pub(crate) fn project_workspace_state<'a>(
        &'a self,
        batch: NonZeroU32,
        projection: &mut ExistingArrayProjection<'a>,
    ) -> Result<
        eredu_runtime::DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
        eredu_nn::Error,
    > {
        let context = projection.context();
        if self.manager.is_some() || self.layers.len() != self.layout.layout().len() {
            return Err(context.metadata_error(format_args!(
                "resident state projection requires a complete resident layer inventory"
            )));
        }
        let factory = WorkspaceConcatStateFactory::new(batch, context)?;
        let state = eredu_runtime::DeviceState::create_workspace_with_shared_layout(
            self.layout.clone(),
            context,
            |index, policy| {
                let layer = &self.layers.slots()[index];
                if let Some(MlxHybridAttentionState::Compressed(cache)) = &layer.attention {
                    return project_compressed_with(
                        cache,
                        &layer.fixed,
                        policy,
                        batch,
                        context,
                        |array| projection.project(array),
                    );
                }
                project_ordinary_layer_with(
                    index,
                    policy,
                    layer.attention.as_ref(),
                    layer.fixed_offset,
                    layer.fixed.iter(),
                    &factory,
                    context,
                    |array| projection.project(array),
                )
            },
        )?;
        Ok(state)
    }
}

/// Same ordinary native geometry and role validation for source inspection and
/// copied-destination projection. Compressed attention stays in its separate
/// representation-specific path above.
pub(in crate::backend::runtime::cache::state::hybrid) fn project_ordinary_layer_with<'a>(
    index: usize,
    policy: &LayerCachePolicy,
    attention: Option<&'a MlxHybridAttentionState>,
    fixed_offset: i32,
    fixed: impl Iterator<Item = (&'a StateTensorRole, &'a Option<MlxTensor>)>,
    factory: &WorkspaceConcatStateFactory,
    context: &WorkspaceContext,
    mut project: impl FnMut(&'a Array) -> Result<eredu_nn::workspace::WorkspaceTensor, eredu_nn::Error>,
) -> Result<WorkspaceResidentLayerState, eredu_nn::Error> {
    let (position, keys, values) = match attention {
        Some(MlxHybridAttentionState::KeyValue(cache)) => {
            cache.project_workspace_attention_with(policy, context, &mut project)?
        }
        None if matches!(
            policy,
            LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. }
        ) =>
        {
            (fixed_offset, None, None)
        }
        None => {
            return Err(context.metadata_error(format_args!(
                "native layer lacks its declared attention mechanism"
            )));
        }
        Some(MlxHybridAttentionState::Compressed(_)) => {
            return Err(context.metadata_error(format_args!(
                "compressed attention requires its distinct copied projection"
            )));
        }
    };
    let fixed = project_fixed_with(policy, fixed, context, project)?;
    factory
        .project_layer(index, policy, position, keys, values, fixed)
        .map(WorkspaceResidentLayerState::Ordinary)
        .map_err(|error| error.into_workspace_error(context))
}

/// The same native fixed-role/dtype producer feeds ordinary and paged layers.
fn project_fixed_with<'a, I, F>(
    policy: &LayerCachePolicy,
    fixed: I,
    context: &WorkspaceContext,
    mut project: F,
) -> Result<
    Vec<(
        StateTensorRole,
        Option<eredu_nn::workspace::WorkspaceTensor>,
    )>,
    eredu_nn::Error,
>
where
    I: Iterator<Item = (&'a StateTensorRole, &'a Option<MlxTensor>)>,
    F: FnMut(&'a Array) -> Result<eredu_nn::workspace::WorkspaceTensor, eredu_nn::Error>,
{
    context.charge_metadata(std::mem::size_of::<(
        &LayerCachePolicy,
        I,
        &WorkspaceContext,
        F,
        Vec<(
            StateTensorRole,
            Option<eredu_nn::workspace::WorkspaceTensor>,
        )>,
    )>())?;
    let projected = fixed.map(|(role, tensor)| {
        context.charge_metadata(std::mem::size_of::<(
            &StateTensorRole,
            &Option<MlxTensor>,
            Option<&eredu_core::cache::StateTensorPolicy>,
            safemlx::Dtype,
            bool,
            (
                StateTensorRole,
                Option<eredu_nn::workspace::WorkspaceTensor>,
            ),
            Result<
                (
                    StateTensorRole,
                    Option<eredu_nn::workspace::WorkspaceTensor>,
                ),
                eredu_nn::Error,
            >,
        )>())?;
        if let Some(tensor) = tensor {
            let declaration = policy
                .fixed_state()
                .iter()
                .find(|value| value.role == *role)
                .ok_or_else(|| {
                    context
                        .metadata_error(format_args!("native fixed storage has no declared role"))
                })?;
            use eredu_core::cache::StateTensorDtype;
            use safemlx::Dtype;
            context.charge_metadata(
                safemlx::Array::descriptor_control_bytes()
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
            )?;
            let descriptor = tensor
                .as_array()
                .try_descriptor()
                .map_err(|cause| context.metadata_source(cause))?;
            let dtype = descriptor.facts().dtype();
            let accepted = match declaration.dtype {
                StateTensorDtype::Floating => matches!(
                    dtype,
                    Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 | Dtype::Float64
                ),
                StateTensorDtype::Float32 => dtype == Dtype::Float32,
                StateTensorDtype::Int32 => dtype == Dtype::Int32,
                StateTensorDtype::Uint32 => dtype == Dtype::Uint32,
            };
            if !accepted {
                return Err(context.metadata_error(format_args!(
                    "native fixed dtype differs from its exact declaration"
                )));
            }
        }
        Ok((
            *role,
            tensor
                .as_ref()
                .map(|tensor| project(tensor.as_array()))
                .transpose()?,
        ))
    });
    context.charge_metadata(std::mem::size_of_val(&projected))?;
    let mut fixed = context.metadata_vec(policy.fixed_state().len())?;
    for value in projected {
        let value = value?;
        context.reserve_metadata_vec(&mut fixed, 1)?;
        fixed.push(value);
    }
    Ok(fixed)
}

fn project_compressed_with<'a, F>(
    cache: &'a CompressedLatentCache,
    fixed: &FixedStateSlots,
    policy: &LayerCachePolicy,
    batch: NonZeroU32,
    context: &WorkspaceContext,
    project: F,
) -> Result<WorkspaceResidentLayerState, eredu_nn::Error>
where
    F: FnMut(&'a Array) -> Result<eredu_nn::workspace::WorkspaceTensor, eredu_nn::Error>,
{
    context.charge_metadata(std::mem::size_of::<(
        &CompressedLatentCache,
        &FixedStateSlots,
        &LayerCachePolicy,
        NonZeroU32,
        &WorkspaceContext,
        F,
        Result<WorkspaceResidentLayerState, eredu_nn::Error>,
    )>())?;
    let LayerCachePolicy::CompressedLatentRotary {
        latent_dim,
        rotary_dim,
        ..
    } = policy
    else {
        return Err(context.metadata_error(format_args!(
            "native compressed storage differs from declared state policy"
        )));
    };
    if !fixed.is_empty() {
        return Err(context.metadata_error(format_args!(
            "compressed state contains undeclared fixed storage"
        )));
    }
    cache
        .project_workspace_state_with(batch, *latent_dim, *rotary_dim, context, project)
        .map(WorkspaceResidentLayerState::Compressed)
}
