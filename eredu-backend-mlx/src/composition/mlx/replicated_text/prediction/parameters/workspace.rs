//! Actual selected module weights projected once, without populating placeholders.
use super::super::{MlxEmbeddedPredictionMaterializer, OwnedPredictionCache};
use super::{MlxPredictionModule, ResidencyManager};
use crate::{
    MlxTensor,
    backend::{
        nn::{
            shared::MlxNeuralBackend,
            workspace::{ExistingArrayProjection, ProjectedNativeStorage},
        },
        runtime::{
            cache::state::MlxPoolingAttentionCache, execution::generic::LayerwiseWorkspace,
            residency::manager::ResidentParameterSource,
        },
    },
};
use eredu_architectures::prediction_extension::workspace::{
    WorkspacePredictionInvocation, WorkspacePredictionInvocations,
    WorkspacePredictionParameterSource,
};
use eredu_architectures::prediction_extension::{
    MaterializedPredictionExecutor, PredictionResourceVisitor,
};
use eredu_nn::{
    Error, Parameterized,
    workspace::{
        WorkspaceContext, WorkspaceExistingStorage, WorkspaceMetadataError, WorkspaceTensor,
    },
};
use eredu_runtime::{
    LocalModelLayout, ReplicatedTextMaterializationTask,
    working_memory::{WorkspaceParameterLifetime, WorkspaceParameterOwner, WorkspaceParameterRows},
};
use safemlx::Array;
use std::{
    marker::PhantomData,
    mem::{size_of, size_of_val},
};
mod binding;
mod snapshot;

#[derive(Clone, Copy, Debug, thiserror::Error)]
enum SourceError {
    #[error("prediction parameter module source is absent or differs")]
    Module,
    #[error("prediction parameter source traversal changed")]
    Traversal,
    #[error("prediction parameter source has no borrowed metadata")]
    Metadata,
}
fn failure(context: &WorkspaceContext, cause: SourceError) -> Error {
    context.metadata_source(cause)
}
fn controls(context: &WorkspaceContext, parts: &[usize]) -> Result<(), Error> {
    context.charge_metadata(
        parts
            .iter()
            .copied()
            .try_fold(size_of_val(parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct SourceFailure {
    #[source]
    cause: Error,
    _funding: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
}
fn with_failure<T, F: FnOnce() -> Result<T, Error>>(
    context: &WorkspaceContext,
    work: F,
) -> Result<T, Error> {
    controls(
        context,
        &[
            size_of::<F>(),
            size_of::<Result<T, Error>>(),
            WorkspaceContext::metadata_source_bytes::<SourceFailure>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        ],
    )?;
    let funding = context.metadata_funding();
    work().map_err(|cause| {
        Error::backend_retained_source(SourceFailure {
            cause,
            _funding: funding,
        })
    })
}
/// Retained canonical native witnesses; values retire before their metadata owner.
pub(crate) struct PredictionParameterStorage {
    storage: ProjectedNativeStorage,
    invocations: WorkspacePredictionInvocations,
    _context: WorkspaceContext,
}
impl PredictionParameterStorage {
    pub(crate) fn invocations(&self) -> &WorkspacePredictionInvocations {
        &self.invocations
    }
    pub(crate) fn storage(&self) -> &ProjectedNativeStorage {
        &self.storage
    }
}

struct Row {
    name: String,
    value: WorkspaceTensor,
    native: Option<(safemlx::AllocationIdentity, u64)>,
}
struct Module {
    physical: usize,
    id: String,
    manager: ResidencyManager,
    rows: Vec<Row>,
}
/// A single cold materialization pass. Actual module/source borrows remain alive
/// until all corresponding metadata units have bound. This is not a native grant.
pub(crate) struct NativePredictionParameters<'a, A, P> {
    source: &'a P,
    layerwise: Option<LayerwiseWorkspace>,
    modules: Vec<Module>,
    next: usize,
    invocations: WorkspacePredictionInvocations,
    context: &'a WorkspaceContext,
    architecture: PhantomData<fn() -> A>,
}
pub(crate) struct MlxWorkspacePredictionParameterSource<A, P>(PhantomData<fn() -> (A, P)>);

/// Returns the parameter context and its separately retained native witnesses.
/// Keep the latter through source registration/admission. The same shared
/// projection covers every module and replacement, preserving native aliases.
/// The caller still authenticates actual execution/source revision at consumption.
pub(crate) fn prepare_prediction_parameters<'a, A, P>(
    source: &'a P,
    layerwise: Option<&'a LayerwiseWorkspace>,
    context: &'a WorkspaceContext,
) -> Result<
    (
        NativePredictionParameters<'a, A, P>,
        PredictionParameterStorage,
    ),
    Error,
>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    with_failure(context, || prepare_inner(source, layerwise, context))
}
fn prepare_inner<'a, A, P>(
    source: &'a P,
    layerwise: Option<&'a LayerwiseWorkspace>,
    context: &'a WorkspaceContext,
) -> Result<
    (
        NativePredictionParameters<'a, A, P>,
        PredictionParameterStorage,
    ),
    Error,
>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    controls(
        context,
        &[
            size_of::<NativePredictionParameters<'a, A, P>>(),
            size_of::<
                Result<
                    (
                        NativePredictionParameters<'a, A, P>,
                        PredictionParameterStorage,
                    ),
                    Error,
                >,
            >(),
            size_of::<snapshot::Counts>(),
            size_of::<ExistingArrayProjection<'_>>(),
        ],
    )?;
    let mut counts = snapshot::Counts::default();
    source
        .visit_retained_resources(&mut counts)
        .map_err(|cause| context.metadata_source(cause))?;
    if let Some(layerwise) = layerwise {
        // Validates the actual complete immutable owner closure before allocating
        // any prospective roots; no source map or placeholder is synthesized.
        layerwise
            .parameter_source()
            .count()
            .map_err(|cause| context.metadata_source(cause))?;
    }
    let mut collector = snapshot::Collect::new(counts, layerwise, context)?;
    source.visit_retained_resources(&mut collector)?;
    let (snapshots, module_layerwise) = collector.finish()?;
    let mut projection = ExistingArrayProjection::with_source_count(context, counts.native)
        .map_err(|cause| context.metadata_source(cause))?;
    let mut modules = context.metadata_vec(snapshots.len())?;
    for snapshot in &snapshots {
        let mut rows = context.metadata_vec(snapshot.rows.len())?;
        for row in &snapshot.rows {
            let value = match &row.value {
                snapshot::Value::Native(array) => projection
                    .project_prepared(array)
                    .map_err(|cause| context.metadata_source(cause))?,
                snapshot::Value::Prospective(value) => value.clone(),
            };
            let native = match &row.value {
                snapshot::Value::Native(array) => {
                    controls(
                        context,
                        &[
                            Array::descriptor_control_bytes()
                                .ok_or(WorkspaceMetadataError::Overflow)?,
                            size_of::<Option<(safemlx::AllocationIdentity, u64)>>(),
                        ],
                    )?;
                    let descriptor = array
                        .try_descriptor()
                        .map_err(|cause| context.metadata_source(cause))?;
                    let info = descriptor
                        .facts()
                        .allocation()
                        .ok_or_else(|| failure(context, SourceError::Module))?;
                    Some((
                        info.identity(),
                        u64::try_from(info.bytes())
                            .map_err(|_| WorkspaceMetadataError::Overflow)?,
                    ))
                }
                snapshot::Value::Prospective(_) => None,
            };
            rows.push(Row {
                name: context.metadata_string(format_args!("{}", row.name))?,
                value,
                native,
            });
        }
        modules.push(Module {
            physical: snapshot.physical,
            id: context.metadata_string(format_args!("{}", snapshot.id))?,
            manager: snapshot.manager.clone(),
            rows,
        });
    }
    let storage = projection.try_into_storage()?;
    let invocations = WorkspacePredictionInvocations::new(modules.len(), context)?;
    let invocation_source = invocations.alias()?;
    // Temporary snapshots retire only after the common projection owns the
    // canonical native witnesses; no borrowed reference survives their drop.
    drop(snapshots);
    Ok((
        NativePredictionParameters {
            source,
            layerwise: module_layerwise,
            modules,
            next: 0,
            invocations: invocation_source,
            context,
            architecture: PhantomData,
        },
        PredictionParameterStorage {
            storage,
            invocations,
            _context: context.clone(),
        },
    ))
}

impl<A: 'static, P: 'static> WorkspacePredictionParameterSource
    for MlxWorkspacePredictionParameterSource<A, P>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    type Context<'a> = NativePredictionParameters<'a, A, P>;
    fn bind<U: Parameterized<WorkspaceTensor>>(
        state: &mut Self::Context<'_>,
        dense_index: usize,
        source: &U,
        local: &mut U,
        tasks: &[ReplicatedTextMaterializationTask],
        layout: Option<&LocalModelLayout>,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        with_failure(context, || {
            if state.next != dense_index || !state.context.shares_trace(context) {
                return Err(failure(context, SourceError::Traversal));
            }
            let module = state
                .modules
                .get(dense_index)
                .ok_or_else(|| failure(context, SourceError::Traversal))?;
            controls(
                context,
                &[
                    size_of::<binding::Bind<'_, U>>(),
                    size_of::<Result<(), Error>>(),
                    size_of::<(
                        &U,
                        &mut U,
                        &[ReplicatedTextMaterializationTask],
                        Option<&LocalModelLayout>,
                    )>(),
                ],
            )?;
            let mut binder = binding::Bind::new(
                dense_index,
                state.modules.len(),
                module,
                source,
                local,
                tasks,
                layout,
                state.layerwise.as_ref(),
                context,
            );
            state.source.visit_retained_resources(&mut binder)?;
            binder.finish()?;
            state.next = state
                .next
                .checked_add(1)
                .ok_or(WorkspaceMetadataError::Overflow)?;
            Ok(())
        })
    }
    fn invocation(
        state: &mut Self::Context<'_>,
        index: usize,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspacePredictionInvocation>, Error> {
        with_failure(context, || {
            if state.next.checked_sub(1) != Some(index) || !state.context.shares_trace(context) {
                return Err(failure(context, SourceError::Traversal));
            }
            state.invocations.module(index, context).map(Some)
        })
    }
    fn finish(state: &mut Self::Context<'_>) -> Result<(), Error> {
        with_failure(state.context, || {
            if state.next != state.modules.len() {
                return Err(failure(state.context, SourceError::Traversal));
            }
            Ok(())
        })
    }
}
