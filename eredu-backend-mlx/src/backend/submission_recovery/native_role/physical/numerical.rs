//! The shared source-qualified numerical child of parameter and residency loans.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    nn::workspace::{ResidentExecutionMechanisms, SpeculativeNumericalRecipe},
};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};
use safemlx::{Array, OperationEvent, PreparedPipelineCachePlan};
use std::mem::size_of_val;
fn missing() -> Error {
    Error::OriginalSourceContract {
        stage: "standalone numerical control quotation",
        cause: eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
    }
}
pub(crate) fn execute_numerical<T>(
    report: &eredu_nn::workspace::WorkspaceTraceReport,
    inputs: &[&Array],
    output_roots: usize,
    mechanism: ResidentExecutionMechanisms,
    context: &WorkspaceContext,
    environment: &OriginalCopyEnvironment<'_>,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    host_controls: usize,
    operation: impl FnOnce() -> Result<T, Error>,
) -> Result<super::CompletedNumerical<T>, Error> {
    let allocator = environment
        .input_runtime()
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    execute_numerical_with_runtime(
        report,
        inputs,
        output_roots,
        mechanism,
        context,
        &allocator,
        environment.pool(),
        execution,
        host_controls,
        operation,
    )
}

/// The same numerical worker with an already authenticated allocator/ledger loan.
/// The caller retains its actual stream source and executable identity.
pub(crate) fn execute_numerical_with_runtime<T>(
    report: &eredu_nn::workspace::WorkspaceTraceReport,
    inputs: &[&Array],
    output_roots: usize,
    mechanism: ResidentExecutionMechanisms,
    context: &WorkspaceContext,
    allocator: &PreparedInputRuntime,
    pool: &MemoryLedger,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    host_controls: usize,
    operation: impl FnOnce() -> Result<T, Error>,
) -> Result<super::CompletedNumerical<T>, Error> {
    let recipe = SpeculativeNumericalRecipe::inspect_completed_outputs(
        report,
        output_roots,
        mechanism,
        context,
    )
    .map_err(Error::Neural)?;
    let population = OriginalBufferBudget::population_layout(
        allocator,
        usize::try_from(recipe.storage.mutable_bytes())
            .map_err(|_| Error::Neural(WorkspaceMetadataError::Overflow.into()))?,
        recipe.storage.maximum_births(),
    )
    .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    let capacity = NativeRoleCapacity {
        graph: recipe.graph_capacity,
        records: recipe.record_capacity,
        backing: population.capacity(),
    };
    let frames = [
        usize::try_from(
            report
                .host_workspace_bytes
                .ok_or(Error::OriginalSourceContract {
                    stage: "standalone numerical host workspace",
                    cause: WorkingMemoryError::UnknownBound,
                })?,
        )
        .map_err(|_| Error::Neural(WorkspaceMetadataError::Overflow.into()))?,
        usize::try_from(recipe.controls)
            .map_err(|_| Error::Neural(WorkspaceMetadataError::Overflow.into()))?,
        host_controls,
        OperationEvent::traversal_leaf_control_bytes()
            .and_then(|n| n.checked_mul(inputs.len()))
            .ok_or_else(missing)?,
        size_of::<SpeculativeNumericalRecipe>(),
        size_of::<NativeRoleCapacity>(),
        size_of::<Result<T, Error>>(),
        size_of::<safemlx::PreparedResidentGraph>(),
        size_of::<Result<safemlx::PreparedResidentGraph, safemlx::error::Exception>>(),
    ];
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or_else(|| Error::Neural(WorkspaceMetadataError::Overflow.into()))?,
        )
        .map_err(|cause| Error::Neural(cause.into()))?;
    let parent = safemlx::OriginalScopeObserver::try_current()?;
    let completed = super::Plan::new(
        allocator,
        capacity,
        Some(PreparedPipelineCachePlan::new(recipe.kernels)),
        (),
        |_, role| {
            for array in inputs {
                OperationEvent::validate_traversal_leaf(array, role.observer())?;
            }
            let mut graph =
                OperationEvent::prepare_resident_graph(recipe.completion.graph, role.observer())?;
            if recipe.completion.nested_completions != 0 {
                let traversal = recipe.completion.nested_traversal().ok_or_else(missing)?;
                graph.configure_nested_completions(
                    &traversal,
                    recipe.completion.nested_completions,
                )?;
            }
            let result = operation();
            drop(graph);
            Ok(result)
        },
    )
    .with_parent(parent)
    .run(pool, execution, context)
    .map_err(|cause| match cause {
        super::Failure::Admission(cause) => Error::PrefillControl(cause),
        super::Failure::Native(cause) => Error::StorageSource(cause),
    })?;
    Ok(super::CompletedNumerical {
        value: completed.value?,
        source: completed.source,
    })
}
