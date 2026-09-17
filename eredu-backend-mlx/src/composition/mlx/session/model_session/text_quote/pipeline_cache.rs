//! One finite precompiled pipeline source, installed before any original role.
use super::*;
use crate::backend::nn::workspace::ResidentNativeRecipe;
use eredu_runtime::working_memory::{OriginalTextControlGuard, OriginalTextMetadataCustody};
use safemlx::{
    PipelineCacheCause, PreparedPipelineCache, PreparedPipelineCachePlan, SubmissionGraphQuota,
};
use std::mem::size_of;

#[derive(Debug)]
enum FailureOwner {
    Raw(OriginalTextMetadataCustody),
    Cache(PreparedPipelineCache<OriginalTextMetadataCustody>),
}
#[derive(Debug, thiserror::Error)]
#[error("original pipeline source construction failed: {cause}")]
struct Failure {
    #[source]
    cause: PipelineCacheCause,
    // Either failed construction's unchanged raw custody or the same concrete
    // cache. Native storage is always destroyed before its raw-only owner.
    owner: FailureOwner,
}
fn failure(cause: PipelineCacheCause, owner: FailureOwner) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::new(
            eredu_core::BackendFailureKind::InvalidSession,
            Failure { cause, owner },
        ),
        false,
    )
}

pub(super) fn control_bytes(recipe: &ResidentNativeRecipe) -> Result<Option<u64>, Error> {
    let Some(attempts) = recipe.kernel_attempts() else {
        return Ok(None);
    };
    let plan = PreparedPipelineCachePlan::new(attempts);
    let layout = match plan.layout::<OriginalTextMetadataCustody>() {
        Ok(layout) => layout,
        Err(PipelineCacheCause::UnknownLayout) => return Ok(None),
        Err(PipelineCacheCause::Overflow) => return Err(memory(WorkingMemoryError::Overflow)),
        Err(_) => return Err(unknown()),
    };
    let source = eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let controls = [
        size_of::<Failure>(),
        size_of::<FailureOwner>(),
        size_of::<PreparedPipelineCachePlan>(),
        size_of::<safemlx::PipelineCacheLayout>(),
        size_of::<Option<usize>>(),
        size_of::<Result<Option<u64>, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<OriginalTextControlGuard>(),
        size_of::<OriginalTextMetadataCustody>(),
        source,
    ];
    let amount = controls
        .into_iter()
        .try_fold(
            layout
                .required_bytes()
                .and_then(|n| n.checked_add(std::mem::size_of_val(&controls)))
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            usize::checked_add,
        )
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    Ok(Some(
        u64::try_from(amount).map_err(|_| memory(WorkingMemoryError::Overflow))?,
    ))
}

pub(super) fn install(
    recipe: &ResidentNativeRecipe,
    controls: &OriginalTextControlGuard,
    graph: &SubmissionGraphQuota,
) -> Result<(), Error> {
    let attempts = recipe.kernel_attempts().ok_or_else(unknown)?;
    let cache = PreparedPipelineCachePlan::new(attempts)
        .realize(controls.metadata_custody())
        .map_err(|error| {
            let (cause, raw) = error.into_parts();
            failure(cause, FailureOwner::Raw(raw))
        })?;
    if let Err(cause) = cache.install(graph) {
        return Err(failure(cause, FailureOwner::Cache(cache)));
    }
    // Graph keeps its independent cache reference; the cache holds raw H only.
    // No request arena, Scope, Record or physical partition points back to it.
    drop(cache);
    Ok(())
}
