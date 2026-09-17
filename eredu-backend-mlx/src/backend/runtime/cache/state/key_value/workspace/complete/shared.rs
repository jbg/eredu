//! One native source projection constructor and failure/retirement boundary.
use super::*;

#[derive(Default)]
pub(in crate::backend::runtime::cache::state) struct SourceCounts {
    pub arrays: usize,
    pub pagers: usize,
}
enum LayerFailure {
    Source(CacheSourceFailure),
    Construction(eredu_nn::Error),
}
impl From<eredu_nn::Error> for LayerFailure {
    fn from(cause: eredu_nn::Error) -> Self {
        Self::Construction(cause)
    }
}

/// Storage equations and source discovery stay with each concrete state;
/// allocation, exact count enforcement, typed construction and prefix custody
/// use one shared worker. Neither callback can replace the native destination.
pub(in crate::backend::runtime::cache::state) fn project_complete<C, F>(
    layout: &eredu_runtime::SharedStateLayout,
    layers: usize,
    batch: NonZeroU32,
    context: &WorkspaceContext,
    counts: C,
    mut project: F,
) -> Result<ProjectedResidentState, CompleteStateProjectionFailure>
where
    C: FnOnce() -> Result<SourceCounts, CacheSourceFailure>,
    F: FnMut(
        usize,
        &LayerCachePolicy,
        &WorkspaceConcatStateFactory,
        &mut OwnedArrayProjection<'_>,
    ) -> Result<WorkspaceResidentLayerState, CacheSourceFailure>,
{
    let mut retained = None;
    let result = (|| {
        context
            .charge_metadata(
                controls::<C, F>().ok_or_else(|| {
                    CacheSourceFailure::source(CacheSourceError::Overflow, context)
                })?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        if layers != layout.layout().len() {
            return Err(CacheSourceFailure::source(
                CacheSourceError::Geometry,
                context,
            ));
        }
        let counts = counts()?;
        let mut projection = OwnedArrayProjection::prepare(&mut retained, context, counts.arrays)
            .map_err(|cause| CacheSourceFailure::projection(cause, context))?;
        projection
            .prepare_paged_sources(counts.pagers)
            .map_err(|cause| CacheSourceFailure::projection(cause, context))?;
        let factory = WorkspaceConcatStateFactory::new(batch, context)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        eredu_runtime::DeviceState::create_workspace_with_shared_layout_result(
            layout.clone(),
            context,
            |index, policy| {
                project(index, policy, &factory, &mut projection).map_err(LayerFailure::Source)
            },
        )
        .map_err(|cause| match cause {
            LayerFailure::Source(cause) => cause,
            LayerFailure::Construction(cause) => CacheSourceFailure::metadata(cause, context),
        })
    })();
    // Every manager loan and partially initialized metadata table is gone.
    match result {
        Ok(state) => Ok(ProjectedResidentState {
            state,
            storage: retained.expect("successful complete projection"),
        }),
        Err(cause) => Err(CompleteStateProjectionFailure { cause, retained }),
    }
}
fn controls<C, F>() -> Option<usize> {
    let controls = [
        size_of::<(
            &eredu_runtime::SharedStateLayout,
            usize,
            NonZeroU32,
            &WorkspaceContext,
            C,
            F,
        )>(),
        size_of::<(
            C,
            &mut F,
            &eredu_runtime::SharedStateLayout,
            usize,
            NonZeroU32,
            &WorkspaceContext,
            &mut Option<ProjectedNativeStorage>,
        )>(),
        size_of::<(
            &mut F,
            &WorkspaceConcatStateFactory,
            &mut OwnedArrayProjection<'_>,
        )>(),
        size_of::<SourceCounts>(),
        size_of::<WorkspaceConcatStateFactory>(),
        size_of::<Option<ProjectedNativeStorage>>(),
        size_of::<CompleteStateProjectionFailure>(),
        size_of::<ProjectedResidentState>(),
        size_of::<Result<ProjectedResidentState, CompleteStateProjectionFailure>>(),
        size_of::<OwnedArrayProjection<'_>>(),
        size_of::<LayerFailure>(),
        size_of::<
            Result<
                eredu_runtime::DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
                LayerFailure,
            >,
        >(),
        size_of::<
            Result<
                eredu_runtime::DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
                CacheSourceFailure,
            >,
        >(),
        size_of::<(usize, usize)>(),
        size_of::<Result<WorkspaceResidentLayerState, CacheSourceFailure>>(),
        WorkspaceContext::metadata_source_bytes::<CacheSourceFailure>()?,
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
