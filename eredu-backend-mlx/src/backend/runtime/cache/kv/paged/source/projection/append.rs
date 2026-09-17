//! Closed pinned-source handoff into the shared paged append metadata worker.
use super::*;
use eredu_core::cache::LayerCachePolicy;
use eredu_runtime::working_memory::{
    WorkspacePagedAppendState, WorkspacePagedBlock, WorkspacePagedGeometry,
};
use std::num::NonZeroU32;

/// Symbolic state retires before the exact original native roots and pins.
pub(crate) struct ProjectedPagedAppendState {
    state: WorkspacePagedAppendState,
    source: ProjectedPagedCacheSource,
}
impl ProjectedPagedAppendState {
    pub(crate) fn state(&self) -> &WorkspacePagedAppendState {
        &self.state
    }
    pub(crate) fn state_mut(&mut self) -> &mut WorkspacePagedAppendState {
        &mut self.state
    }
    pub(crate) fn source(&self) -> &ProjectedPagedCacheSource {
        &self.source
    }
}

/// The complete source remains alive on any metadata construction refusal.
#[derive(thiserror::Error)]
#[error("{cause}")]
pub(crate) struct PagedAppendWorkspaceFailure {
    #[source]
    cause: Error,
    retained: ProjectedPagedCacheSource,
}
impl std::fmt::Debug for PagedAppendWorkspaceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedAppendWorkspaceFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl ProjectedPagedCacheSource {
    /// Consumes this actual source. Policy only validates declared geometry;
    /// native storage permission remains its original manager/pin inventory.
    pub(crate) fn into_append_workspace(
        self,
        policy: &LayerCachePolicy,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<ProjectedPagedAppendState, PagedAppendWorkspaceFailure> {
        let result = (|| {
            context.charge_metadata(std::mem::size_of::<(
                ProjectedPagedAppendState,
                PagedAppendWorkspaceFailure,
                WorkspacePagedGeometry,
                &Self,
                &LayerCachePolicy,
                NonZeroU32,
                &WorkspaceContext,
                Option<[WorkspaceTensor; 2]>,
                [WorkspaceTensor; 2],
            )>())?;
            if !context.shares_trace(&self._context) {
                return Err(context
                    .metadata_error(format_args!("paged source uses another metadata context")));
            }
            let geometry = append_geometry(&self.geometry, self.manager(), policy, batch, context)?;
            let blocks = self
                .geometry
                .blocks
                .iter()
                .zip(&self.blocks)
                .map(|(block, values)| {
                    // Every original source block is pinned by this retained owner.
                    WorkspacePagedBlock::new(block.id.start, block.id.end, values.clone(), true)
                });
            WorkspacePagedAppendState::project(geometry, blocks, self.tail.clone(), context)
        })();
        match result {
            Ok(state) => Ok(ProjectedPagedAppendState {
                state,
                source: self,
            }),
            Err(cause) => Err(PagedAppendWorkspaceFailure {
                cause,
                retained: self,
            }),
        }
    }
}

/// Shared actual source-to-declared append geometry; standalone and mixed
/// layer projections consume the same policy worker.
pub(super) fn append_geometry(
    geometry: &PagedCacheSourceGeometry,
    manager: &CacheResidencyManager,
    policy: &LayerCachePolicy,
    batch: NonZeroU32,
    context: &WorkspaceContext,
) -> Result<WorkspacePagedGeometry, Error> {
    context.charge_metadata(std::mem::size_of::<(
        &PagedCacheSourceGeometry,
        &CacheResidencyManager,
        &LayerCachePolicy,
        NonZeroU32,
        &WorkspaceContext,
        WorkspacePagedGeometry,
        (u32, u32, &eredu_core::AttentionPolicy, bool),
        [Result<i32, std::num::TryFromIntError>; 3],
        Option<i32>,
        Result<WorkspacePagedGeometry, Error>,
    )>())?;
    let (heads, width, attention, key_only) = match policy {
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
        } => (num_key_value_heads.get(), head_dim.get(), attention, false),
        LayerCachePolicy::KeyOnly {
            num_key_heads,
            head_dim,
            attention,
        }
        | LayerCachePolicy::KeyOnlyWithFixedState {
            num_key_heads,
            head_dim,
            attention,
            ..
        } => (num_key_heads.get(), head_dim.get(), attention, true),
        _ => {
            return Err(context.metadata_error(format_args!(
                "paged source lacks its declared attention policy"
            )));
        }
    };
    let window = attention
        .sliding_window_i32()
        .map_err(|cause| context.metadata_source(cause))?;
    if geometry.sliding_window != window || geometry.key_only != key_only {
        return Err(context.metadata_error(format_args!(
            "paged source differs from selected attention mechanism"
        )));
    }
    let dimensions = [batch.get(), heads, width].map(|value| i32::try_from(value));
    let [batch, heads, width] = dimensions;
    let output = WorkspacePagedGeometry {
        block_size: manager.options().block_size_tokens(),
        dimensions: [
            batch.map_err(|cause| context.metadata_source(cause))?,
            heads.map_err(|cause| context.metadata_source(cause))?,
            width.map_err(|cause| context.metadata_source(cause))?,
        ],
        offset: geometry.offset,
        tail_start: geometry.tail_start,
        window,
        prefix_tokens: geometry.prefix_tokens,
        key_only,
        retain_discarded: manager.options().retains_discarded_for_persistence(),
    };
    Ok(output)
}
