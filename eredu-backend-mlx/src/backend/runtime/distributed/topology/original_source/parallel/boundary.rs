//! One-shot model boundary source loans, separate from the native child role.
use super::*;
use crate::backend::nn::workspace::{ResidentExecutionMechanisms, PipelineBoundaryQuote};

/// Array-free recipe/source ownership. The account outlives the shared header;
/// no observer, request activation or parent model Q enters this shared owner.
#[derive(Clone)]
pub(crate) struct RetainedPipelineBoundary {
    value: Option<Rc<PipelineBoundaryQuote>>,
    funding: HostMetadataFunding,
}
impl RetainedPipelineBoundary {
    pub(crate) fn value(&self) -> &PipelineBoundaryQuote {
        self.value.as_deref().expect("live boundary recipe")
    }
}
impl Drop for RetainedPipelineBoundary {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            drop(Rc::into_inner(value));
        }
    }
}
/// The call keeps the actual invocation alive lexically. Native Recovery takes
/// only a clone of its array-free recipe, so it cannot retain the parent Q.
pub(crate) struct OriginalBoundaryCall {
    state: Option<Rc<State>>,
    quotes: Vec<RetainedPipelineBoundary>,
    funding: HostMetadataFunding,
}
impl OriginalBoundaryCall {
    pub(crate) fn quotes(&self) -> &[RetainedPipelineBoundary] {
        &self.quotes
    }
}
impl Drop for OriginalBoundaryCall {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            state.calling.set(false);
            drop(Rc::into_inner(state));
        }
    }
}
impl OriginalParallelSource {
    pub(super) fn prepare_boundary_occurrences(
        &self,
        operations: &[WorkspaceOperation],
        mechanism: Option<ResidentExecutionMechanisms>,
    ) -> Result<Vec<BoundaryOccurrence>, Error> {
        let count = operations
            .iter()
            .filter(|op| {
                matches!(
                    op.kind,
                    WorkspaceOperationKind::Collective(WorkspaceCollective::Boundary { .. })
                )
            })
            .count();
        reserve(
            self.funding(),
            &[
                size_of::<Vec<BoundaryOccurrence>>(),
                size_of::<BoundaryOccurrence>(),
                size_of::<Result<Vec<BoundaryOccurrence>, Error>>(),
                Layout::array::<BoundaryOccurrence>(count)
                    .map_err(|_| overflow())?
                    .size(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| failure(Cause::Resource, self.declaration_source(), self.funding()))?;
        for (ordinal, operation) in operations.iter().enumerate() {
            if !matches!(
                operation.kind,
                WorkspaceOperationKind::Collective(WorkspaceCollective::Boundary { .. })
            ) {
                continue;
            }
            let mechanism = mechanism.ok_or_else(|| {
                failure(Cause::Resource, self.declaration_source(), self.funding())
            })?;
            let quote = PipelineBoundaryQuote::prepare(self, operation.as_view(), mechanism)
                .map_err(Error::Neural)?;
            reserve(
                self.funding(),
                &[
                    size_of::<RetainedPipelineBoundary>(),
                    Layout::new::<[usize; 2]>()
                        .extend(Layout::new::<PipelineBoundaryQuote>())
                        .map_err(|_| overflow())?
                        .0
                        .pad_to_align()
                        .size(),
                ],
            )?;
            values.push(BoundaryOccurrence {
                ordinal,
                quote: RetainedPipelineBoundary {
                    value: Some(Rc::new(quote)),
                    funding: self.funding().clone(),
                },
            });
        }
        Ok(values)
    }
}
impl OriginalParallelBinding {
    pub(crate) fn claim_boundaries(
        &self,
        source: &OriginalParallelSource,
        frames: &eredu_runtime::PreparedBoundaryFrames<crate::MlxTensor>,
    ) -> Result<OriginalBoundaryCall, Error> {
        reserve(
            &self.funding,
            &[
                size_of::<OriginalBoundaryCall>(),
                size_of::<RetainedPipelineBoundary>(),
                size_of::<Result<OriginalBoundaryCall, Error>>(),
                size_of::<Option<Rc<State>>>(),
                size_of::<ControlSourceLoan>(),
                size_of::<(
                    &Self,
                    &OriginalParallelSource,
                    &eredu_runtime::PreparedBoundaryFrames<crate::MlxTensor>,
                )>(),
                size_of::<(usize, usize)>(),
                size_of::<Option<&BoundaryOccurrence>>(),
                Layout::array::<RetainedPipelineBoundary>(frames.values().len())
                    .map_err(|_| overflow())?
                    .size(),
                size_of::<Vec<RetainedPipelineBoundary>>(),
                size_of::<Result<(), std::collections::TryReserveError>>(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let state = self.state.upgrade().ok_or_else(|| {
            failure(
                Cause::Invocation(ParallelInvocationCause::Closed),
                &self.source,
                &self.funding,
            )
        })?;
        let loan = ControlSourceLoan(Some(state));
        let state = loan.0.as_ref().expect("live boundary claim");
        let failed = || failure(Cause::Identity, &self.source, &self.funding);
        if state.closed.get()
            || state.forward_done.get()
            || state.bound.get().is_none()
            || state.calling.get()
            || state.expert_active.get().is_some()
            || !state.quote_source.same_source(source)
            || state.authority.ensure_active().is_err()
            || frames.values().is_empty()
            || !frames.source().same_source(&state.source)
            || !frames.funding().same_account(&self.funding)
        {
            return Err(failed());
        }
        let mut quotes = Vec::new();
        quotes
            .try_reserve_exact(frames.values().len())
            .map_err(|_| failure(Cause::Resource, &self.source, &self.funding))?;
        for (ordinal, value) in frames.values().iter().enumerate() {
            let index = state.next_boundary.get();
            let row = state.boundaries.get(index).ok_or_else(failed)?;
            state
                .next_boundary
                .set(index.checked_add(1).ok_or_else(overflow)?);
            let quote = row.quote.value();
            let input = value.tensor().as_array();
            if state
                .occurrences
                .get(state.next.get())
                .is_some_and(|next| next.ordinal < row.ordinal)
                || state.expert_regions.get(state.next_expert.get()).is_some_and(|next| next.ordinal < row.ordinal)
                || !quote.matches_source(source)
                || quote.route != frames.route()
                || quote.ordinal != ordinal
                || quote.header_bytes != value.header().len()
                || quote.input.shape() != input.shape()
                || crate::backend::nn::workspace::byte_view::Dtype::from_layout(
                    quote.input.as_view(),
                )
                .is_none_or(|dtype| dtype.native() != input.dtype())
            {
                return Err(failed());
            }
            quotes.push(row.quote.clone());
        }
        state.calling.set(true);
        Ok(OriginalBoundaryCall {
            state: Some(state.clone()),
            quotes,
            funding: self.funding.clone(),
        })
    }
}
