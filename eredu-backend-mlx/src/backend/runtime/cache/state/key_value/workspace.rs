use super::*;
use crate::backend::nn::workspace::{ExistingArrayProjection, ProjectedResidentState};
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceTensor};
use eredu_runtime::working_memory::{WorkspaceConcatStateFactory, WorkspaceResidentLayerState};
use std::num::NonZeroU32;
mod complete;
pub(crate) use complete::CompleteStateProjectionFailure;
pub(in crate::backend::runtime::cache::state) use complete::{project_complete, SourceCounts};

impl MlxKeyValueLayerState {
    pub(in crate::backend::runtime::cache::state) fn project_workspace_attention<'a>(
        &'a self,
        policy: &LayerCachePolicy,
        projection: &mut ExistingArrayProjection<'a>,
    ) -> Result<(i32, Option<WorkspaceTensor>, Option<WorkspaceTensor>), eredu_nn::Error> {
        self.project_workspace_attention_with(policy, projection.context(), |array| {
            projection.project(array)
        })
    }

    pub(in crate::backend::runtime::cache::state) fn project_workspace_attention_with<'a>(
        &'a self,
        policy: &LayerCachePolicy,
        context: &WorkspaceContext,
        project: impl FnMut(&'a safemlx::Array) -> Result<WorkspaceTensor, eredu_nn::Error>,
    ) -> Result<(i32, Option<WorkspaceTensor>, Option<WorkspaceTensor>), eredu_nn::Error> {
        match self {
            Self::Stateless if matches!(policy, LayerCachePolicy::NoState) => Ok((0, None, None)),
            Self::Stateless => Err(context.metadata_error(format_args!(
                "native stateless layer differs from declared state policy"
            ))),
            Self::Device(cache) => cache.project_workspace_attention_with(policy, context, project),
            Self::Paged(_) => Err(context.metadata_error(format_args!(
                "paged state workspace requires its complete manager/block inventory"
            ))),
        }
    }
}

impl MlxKeyValueState {
    /// Projects this actual resident state with aliases shared across all layers.
    /// The returned metadata owns no native state or submission authority.
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

    /// Projects the same state together with exact, temporarily pinned native
    /// allocation identities for binding already-registered decoder storage.
    pub fn project_resident_workspace_with_storage(
        &self,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<ProjectedResidentState, eredu_nn::Error> {
        if self.layers.slots().iter().any(|layer| matches!(layer, MlxKeyValueLayerState::Paged(_))) {
            return self.project_complete_workspace_with_storage(batch, context)
                .map_err(|failure| failure.retire_unsubmitted(context));
        }
        let mut projection = ExistingArrayProjection::new(context);
        let state = project_layers(
            &self.layout,
            self.layers.len(),
            batch,
            context,
            |index, policy| {
                self.layers.slots()[index].project_workspace_attention(policy, &mut projection)
            },
        )?;
        Ok(ProjectedResidentState {
            state,
            storage: projection.try_into_storage()?,
        })
    }
}

// One selected-layout and layer-geometry import for live and freshly copied
// projections. Only metadata tables are constructed here.
pub(in crate::backend::runtime::cache::state) fn project_layers(
    layout: &eredu_runtime::SharedStateLayout,
    count: usize,
    batch: NonZeroU32,
    context: &WorkspaceContext,
    mut project: impl FnMut(
        usize,
        &LayerCachePolicy,
    ) -> Result<
        (i32, Option<WorkspaceTensor>, Option<WorkspaceTensor>),
        eredu_nn::Error,
    >,
) -> Result<
    eredu_runtime::DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
    eredu_nn::Error,
> {
    if count != layout.layout().len() {
        return Err(context.metadata_error(format_args!(
            "native layer inventory differs from state layout"
        )));
    }
    let factory = WorkspaceConcatStateFactory::new(batch, context)?;
    eredu_runtime::DeviceState::create_workspace_with_shared_layout(
        layout.clone(),
        context,
        |index, policy| {
            let (position, keys, values) = project(index, policy)?;
            factory
                .project_layer(index, policy, position, keys, values, [])
                .map(WorkspaceResidentLayerState::Ordinary)
                .map_err(|error| error.into_workspace_error(context))
        },
    )
}
