//! Projection of the native exact-concatenation mechanism.

use super::ConcatKeyValueCache;
use crate::backend::nn::workspace::ExistingArrayProjection;
use eredu_core::cache::LayerCachePolicy;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceTensor},
};

impl ConcatKeyValueCache {
    /// Supplies actual retained attention buffers after checking the selected
    /// storage equation. Capacity-backed and legacy truncating caches need their
    /// own equations; they cannot be mislabeled as exact concatenation.
    pub(crate) fn project_workspace_attention<'a>(
        &'a self,
        policy: &LayerCachePolicy,
        projection: &mut ExistingArrayProjection<'a>,
    ) -> Result<(i32, Option<WorkspaceTensor>, Option<WorkspaceTensor>), Error> {
        self.project_workspace_attention_with(policy, projection.context(), |array| {
            projection.project(array)
        })
    }

    /// Applies the same exact-concatenation validation to a closed alternative
    /// projection of each stored operand, such as the isolated-copy program.
    pub(crate) fn project_workspace_attention_with<'a>(
        &'a self,
        policy: &LayerCachePolicy,
        context: &WorkspaceContext,
        project: impl FnMut(&'a safemlx::Array) -> Result<WorkspaceTensor, Error>,
    ) -> Result<(i32, Option<WorkspaceTensor>, Option<WorkspaceTensor>), Error> {
        let (attention, key_only) = match policy {
            LayerCachePolicy::KeyValue { attention, .. }
            | LayerCachePolicy::KeyValueWithFixedState { attention, .. } => (attention, false),
            LayerCachePolicy::KeyOnly { attention, .. }
            | LayerCachePolicy::KeyOnlyWithFixedState { attention, .. } => (attention, true),
            _ => {
                return Err(context.metadata_error(format_args!(
                    "selected layer has no ordinary attention mechanism"
                )));
            }
        };
        let window = attention
            .sliding_window_i32()
            .map_err(|cause| context.metadata_source(cause))?;
        self.project_exact_workspace(window, key_only, context, project)
    }

    /// Pooling local keys use the same append mechanism with a separate native
    /// one-channel value sentinel, despite the logical key-only declaration.
    pub(crate) fn project_workspace_pooling_local<'a>(
        &'a self,
        window: i32,
        projection: &mut ExistingArrayProjection<'a>,
    ) -> Result<(i32, Option<WorkspaceTensor>, Option<WorkspaceTensor>), Error> {
        self.project_workspace_pooling_local_with(window, projection.context(), |array| {
            projection.project(array)
        })
    }

    pub(crate) fn project_workspace_pooling_local_with<'a>(
        &'a self,
        window: i32,
        context: &WorkspaceContext,
        project: impl FnMut(&'a safemlx::Array) -> Result<WorkspaceTensor, Error>,
    ) -> Result<(i32, Option<WorkspaceTensor>, Option<WorkspaceTensor>), Error> {
        self.project_exact_workspace(Some(window), false, context, project)
    }

    fn project_exact_workspace<'a>(
        &'a self,
        window: Option<i32>,
        key_only: bool,
        context: &WorkspaceContext,
        mut project: impl FnMut(&'a safemlx::Array) -> Result<WorkspaceTensor, Error>,
    ) -> Result<(i32, Option<WorkspaceTensor>, Option<WorkspaceTensor>), Error> {
        if self.step > 1
            || self.max_size.is_some()
            || self.attention_window != window
            || self.key_only != key_only
            || self.offset < 0
        {
            return Err(context.metadata_error(format_args!(
                "native cache does not match selected exact-concatenation workspace mechanism"
            )));
        }
        let count = window.map_or(self.offset, |window| self.offset.min(window - 1));
        if self.length != count || self.capacity != count {
            return Err(context.metadata_error(format_args!(
                "native concatenation frontier and retained capacity disagree"
            )));
        }
        Ok((
            self.offset,
            self.keys.as_ref().map(&mut project).transpose()?,
            self.values.as_ref().map(&mut project).transpose()?,
        ))
    }
}
