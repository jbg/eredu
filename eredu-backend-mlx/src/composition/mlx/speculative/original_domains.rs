//! Shared native domains for exact admitted speculative model invocations.
use crate::backend::{OriginalCopyEnvironment, OriginalCopyEnvironmentError};
use eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody;
use safemlx::{
    error::Exception, OriginalBufferBudget, OriginalBufferCause, PipelineCacheCause,
    PrefillFailureCause, PrefillRoots, PrefillRootsCause, PrefillRootsError, PrefillRootsRuntime,
    PreparedOriginalBufferBudget, PreparedPipelineCache, PreparedPipelineCachePlan,
    PreparedPrefillFailure, PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
    RetainedPrefillFailure, SubmissionGraphQuota, SubmissionGraphQuotaCause, SubmissionRecordQuota,
    SubmissionRecordQuotaCause,
};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
    rc::Rc,
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum OriginalSpeculativeDomainError {
    #[error(transparent)]
    Environment(#[from] OriginalCopyEnvironmentError),
    #[error(transparent)]
    Buffer(#[from] OriginalBufferCause),
    #[error(transparent)]
    Graph(#[from] SubmissionGraphQuotaCause),
    #[error(transparent)]
    Record(#[from] SubmissionRecordQuotaCause),
    #[error(transparent)]
    Pipeline(#[from] PipelineCacheCause),
    #[error(transparent)]
    Failure(#[from] PrefillFailureCause),
    #[error(transparent)]
    Roots(#[from] PrefillRootsError),
    #[error(transparent)]
    RootConstruction(#[from] PrefillRootsCause),
    #[error(transparent)]
    Native(#[from] Exception),
}
// Both outer owners carry role custody after this closed Rc. The last Rc shell
// therefore retires while the account is still held, including failed Scope cleanup.
#[derive(Clone)]
pub(crate) struct OriginalSpeculativeRoots(pub(crate) Rc<RefCell<PrefillRoots>>);
pub(crate) struct OriginalSpeculativeDomains {
    pub(crate) graph: SubmissionGraphQuota,
    pub(crate) record: SubmissionRecordQuota,
    pub(crate) buffer: OriginalBufferBudget,
    pub(crate) _failure: RetainedPrefillFailure,
    pub(crate) _pipeline: PreparedPipelineCache<OriginalSpeculativeBudgetCustody>,
}

/// Fixed new transport frames only. The actual consuming plan separately pays
/// every native layout, quota allocation, roots Rc and completion resource.
pub(crate) fn preparation_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<OriginalSpeculativeDomainError>(),
        size_of::<
            Result<
                (OriginalSpeculativeDomains, OriginalSpeculativeRoots),
                OriginalSpeculativeDomainError,
            >,
        >(),
        size_of::<OriginalSpeculativeBudgetCustody>(),
        size_of::<(
            &OriginalCopyEnvironment<'_>,
            &PrefillRootsRuntime,
            usize,
            usize,
            usize,
            &PreparedPipelineCachePlan,
            usize,
        )>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

/// Consumes the exact account alias after source binding and role admission.
/// It grants no source identity or invocation permission on its own. The caller
/// retains returned resources through terminal completion, failure or teardown.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_domains(
    environment: &OriginalCopyEnvironment<'_>,
    custody: OriginalSpeculativeBudgetCustody,
    runtime: &PrefillRootsRuntime,
    graph_capacity: usize,
    record_capacity: usize,
    physical_capacity: usize,
    pipeline_plan: &PreparedPipelineCachePlan,
    root_capacity: usize,
) -> Result<(OriginalSpeculativeDomains, OriginalSpeculativeRoots), OriginalSpeculativeDomainError>
{
    if environment.buffer_placement()?.as_ref() != custody.placement() {
        return Err(OriginalBufferCause::ForeignDomain.into());
    }
    let graph = PreparedSubmissionGraphQuota::try_new(graph_capacity, custody.clone())
        .map_err(|e| e.into_parts().0)?
        .try_allocate()
        .map_err(|e| e.into_parts().0)?;
    let record = PreparedSubmissionRecordQuota::try_new(record_capacity, custody.clone())
        .map_err(|e| e.into_parts().0)?
        .try_allocate()
        .map_err(|e| e.into_parts().0)?;
    let allocator = environment.input_runtime()?;
    let buffer =
        PreparedOriginalBufferBudget::try_new(&allocator, physical_capacity, custody.clone())
            .map_err(|e| e.into_parts().0)?
            .try_allocate()
            .map_err(|e| e.into_parts().0)?;
    let pipeline = pipeline_plan
        .realize(custody.clone())
        .map_err(|e| e.into_parts().0)?;
    pipeline.install(&graph)?;
    let failure = PreparedPrefillFailure::try_new(custody)
        .map_err(|e| e.into_parts().0)?
        .try_allocate()
        .map_err(|e| e.into_parts().0)?;
    let roots = OriginalSpeculativeRoots(Rc::new(RefCell::new(PrefillRoots::new_retained(
        runtime,
        root_capacity,
        &graph,
        &failure,
    )?)));
    Ok((
        OriginalSpeculativeDomains {
            graph,
            record,
            buffer,
            _failure: failure,
            _pipeline: pipeline,
        },
        roots,
    ))
}
