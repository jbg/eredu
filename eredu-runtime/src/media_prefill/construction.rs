//! The same cut/source table constructors with ordinary or counted destinations.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError, WorkspaceMetadataFunding},
};
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy)]
pub(super) struct Destination<'a>(pub(super) Option<&'a WorkspaceContext>);
impl Destination<'_> {
    fn vector<T>(&self, count: usize) -> Result<Vec<T>, Error> {
        match self.0 {
            Some(context) => context.metadata_vec(count),
            None => Ok(Vec::with_capacity(count)),
        }
    }
    pub(super) fn source<E: std::error::Error + Send + Sync + 'static>(&self, cause: E) -> Error {
        match self.0 {
            Some(context) => context.metadata_source(cause),
            None => Error::backend_source(cause),
        }
    }
    pub(super) fn missing_binding(&self) -> Error {
        const MESSAGE: &str = "original media equation source requires its actual bound source";
        match self.0 {
            Some(context) => context.metadata_error(format_args!("{MESSAGE}")),
            None => Error::backend(MESSAGE),
        }
    }
    pub(super) fn request(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<InferenceRequest, Error> {
        if let Some(context) = self.0 {
            let bytes = InferenceRequest::unbudgeted_control_bytes()
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or(WorkspaceMetadataError::Unqualified)?;
            context.charge_metadata(bytes)?;
        }
        InferenceRequest::without_memory_budget(execution, geometry)
            .map_err(|cause| self.source(cause))
    }
    pub(super) fn cut(
        &self,
        graph: ExecutionGraph,
        primary: &str,
    ) -> Result<CompositePrefillCut, Error> {
        CompositePrefillCut::new_with_storage(
            graph,
            primary,
            |count| self.vector(count),
            |cause| self.source(cause),
        )
    }
    pub(super) fn prepared<A, B, S>(
        &self,
        plan: A::IngressPlan,
        cut: CompositePrefillCut,
        request: InferenceRequest,
        execution: &InferenceExecutionIdentity,
        revision: InferenceStateRevision,
    ) -> Result<PreparedMediaPrefill<A, B, S>, Error>
    where
        B: NeuralBackend,
        S: RuntimeState<B>,
        A: PrefillIngressArchitecture<B, S>,
    {
        PreparedMediaPrefill::new_with_storage(
            plan,
            cut,
            request,
            execution,
            revision,
            |count| self.vector(count),
            |cause| self.source(cause),
        )
    }
}

// The error's erased Box and Arc retire before this payload; its independent
// funding alias then outlives the cause and every paid constructor error shell.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(super) struct SourceFailure {
    #[source]
    pub(super) cause: Error,
    pub(super) _funding: Option<WorkspaceMetadataFunding>,
}

pub(super) fn admit_source<A, S>(context: &WorkspaceContext) -> Result<(), Error>
where
    S: RuntimeState<eredu_nn::workspace::WorkspaceBackend>,
    A: PrefillIngressArchitecture<eredu_nn::workspace::WorkspaceBackend, S>,
{
    use eredu_nn::workspace::{WorkspaceBackend, WorkspaceTensor};
    let controls = [
        size_of::<workspace::MediaEquationSource<A, S>>(),
        size_of::<PreparedMediaPrefill<A, WorkspaceBackend, S>>(),
        size_of::<Result<workspace::MediaEquationSource<A, S>, Error>>(),
        size_of::<Result<PreparedMediaPrefill<A, WorkspaceBackend, S>, Error>>(),
        size_of::<CompositePrefillCut>(),
        size_of::<Result<CompositePrefillCut, Error>>(),
        size_of::<Vec<bool>>() * 2,
        size_of::<Vec<CutValue<WorkspaceTensor>>>(),
        size_of::<CutValue<WorkspaceTensor>>(),
        size_of::<A::IngressPlan>(),
        size_of::<InferenceRequest>(),
        size_of::<Result<InferenceRequest, eredu_core::CapabilityError>>(),
        size_of::<InferenceExecutionIdentity>(),
        size_of::<InferenceStateRevision>(),
        size_of::<InferenceGeometry>(),
        size_of::<Destination<'_>>(),
        size_of::<Option<WorkspaceMetadataFunding>>(),
        size_of::<SourceFailure>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Result<(), Error>>(),
    ];
    let bytes = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .and_then(|bytes| {
            bytes.checked_add(Error::retained_source_control_bytes::<SourceFailure>()?)
        })
        .ok_or(WorkspaceMetadataError::Overflow)?;
    context.charge_metadata(bytes)?;
    Ok(())
}

/// Same source storage worker, with the actual native types and request.
pub(crate) fn prepared_original<A, B, S>(
    plan: A::IngressPlan,
    cut: CompositePrefillCut,
    request: InferenceRequest,
    execution: &InferenceExecutionIdentity,
    revision: InferenceStateRevision,
    context: &WorkspaceContext,
) -> Result<PreparedMediaPrefill<A, B, S>, Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    let controls = [
        size_of::<PreparedMediaPrefill<A, B, S>>(),
        size_of::<Result<PreparedMediaPrefill<A, B, S>, Error>>(),
        size_of::<A::IngressPlan>(),
        size_of::<CompositePrefillCut>(),
        size_of::<Vec<CutValue<B::Tensor>>>(),
        size_of::<CutValue<B::Tensor>>(),
        size_of::<InferenceRequest>(),
        size_of::<InferenceStateRevision>(),
        size_of::<Option<WorkspaceContext>>(),
        size_of::<Destination<'_>>(),
    ];
    context.charge_metadata(
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    let mut source =
        Destination(Some(context)).prepared(plan, cut, request, execution, revision)?;
    source.metadata = Some(context.clone());
    Ok(source)
}

pub(crate) fn original_cut(
    graph: ExecutionGraph,
    primary: &str,
    context: &WorkspaceContext,
) -> Result<CompositePrefillCut, Error> {
    context.charge_metadata(size_of::<(
        CompositePrefillCut,
        Result<CompositePrefillCut, Error>,
        [Vec<bool>; 2],
        Destination<'_>,
    )>())?;
    Destination(Some(context)).cut(graph, primary)
}
