//! Exact current-state/root membership through the existing typed source visitor.
use super::*;
use crate::prediction_extension::{
    PredictionStateSourceFactory, workspace::WorkspacePredictionSequentialState,
};
use eredu_runtime::working_memory::WorkspacePoolingLayerState;

fn append<'a>(
    values: impl IntoIterator<Item = &'a WorkspaceTensor>,
    roots: &mut Vec<WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<(), Error> {
    for value in values {
        context.reserve_metadata_vec(roots, 1)?;
        roots.push(value.clone());
    }
    Ok(())
}
pub(super) fn append_target(
    state: &ResidentState,
    roots: &mut Vec<WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<(), Error> {
    append(
        state
            .as_ref()
            .iter()
            .flat_map(|layer| layer.retained_values()),
        roots,
        context,
    )
}
pub(super) fn append_observer(
    observer: &dyn InferenceWorkspaceObserver,
    roots: &mut Vec<WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<(), Error> {
    append_visited(|visitor| observer.visit_retained(visitor), roots, context)
}
pub(super) fn append_tails(
    tails: &dyn WorkspacePredictionEquationTails,
    roots: &mut Vec<WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<(), Error> {
    append_visited(|visitor| tails.visit_retained(visitor), roots, context)
}
fn append_visited<F>(
    visit: F,
    roots: &mut Vec<WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<(), Error>
where
    F: FnOnce(&mut dyn FnMut(&WorkspaceTensor)),
{
    let parts = [
        size_of::<F>(),
        size_of::<Option<Error>>(),
        size_of::<(
            &mut Vec<WorkspaceTensor>,
            &WorkspaceContext,
            &mut Option<Error>,
        )>(),
        size_of::<Result<(), Error>>(),
    ];
    context.charge_metadata(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    let mut failure = None;
    visit(&mut |value| {
        if failure.is_none() {
            if let Err(cause) = append([value], roots, context) {
                failure = Some(cause);
            }
        }
    });
    failure.map_or(Ok(()), Err)
}
pub(super) fn append_lane<A, P, E>(
    _extension: &E,
    state: &E::LaneState,
    roots: &mut Vec<WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<(), Error>
where
    P: WorkspacePredictionParameterSource,
    E: MaterializedPredictionExecutor<A, WorkspaceBackend, WorkspacePredictionMaterializer<P>>,
{
    context.charge_metadata(
        size_of::<Collector<'_>>()
            .checked_add(size_of::<Result<(), Error>>())
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    E::prepare_state(state, &mut Collector { roots, context })
}
struct Collector<'a> {
    roots: &'a mut Vec<WorkspaceTensor>,
    context: &'a WorkspaceContext,
}
impl<P: WorkspacePredictionParameterSource>
    PredictionStateSourceFactory<WorkspaceBackend, WorkspacePredictionMaterializer<P>>
    for Collector<'_>
{
    type Error = Error;
    type Prepared<T> = ();
    fn sequential(&mut self, source: &[WorkspacePredictionSequentialState]) -> Result<(), Error> {
        append(
            source.iter().flat_map(|state| state.retained_values()),
            self.roots,
            self.context,
        )
    }
    fn pooling(&mut self, source: &[WorkspacePoolingLayerState]) -> Result<(), Error> {
        append(
            source.iter().flat_map(|state| state.retained_values()),
            self.roots,
            self.context,
        )
    }
    fn model(&mut self, source: &ResidentState) -> Result<(), Error> {
        append_target(source, self.roots, self.context)
    }
}
