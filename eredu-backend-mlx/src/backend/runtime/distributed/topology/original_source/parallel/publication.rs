//! The quoted publication continues the exact enclosing model invocation.
use super::super::operation_storage::{
    AcceptedCommunicationSource, OriginalCommunicationBacking,
    OriginalCommunicationCompletedOperation, OriginalCommunicationOperation,
};
use super::*;
use crate::backend::runtime::distributed::completion::prepared::ReadyCompletionResources;
use crate::backend::runtime::distributed::completion::OriginalCommunicationCompletion;
use crate::backend::submission_recovery::prefill::TransientRootsProjection;

/// The transient weak root control cannot escape its owning Q through an
/// invocation retained by the request's recipe. Every exit closes this lexical
/// view; the actual roots/observer remain in Q for completion or recovery.
struct RootLoan<'a>(&'a RefCell<Option<TransientRootsProjection>>);
impl Drop for RootLoan<'_> {
    fn drop(&mut self) {
        drop(self.0.borrow_mut().take());
    }
}

impl OriginalParallelInvocation {
    pub(crate) fn with_publication_context<T, E, F>(
        &self,
        observer: &OriginalScopeObserver,
        compute: &Stream,
        roots: TransientRootsProjection,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&Group, &WorkspaceMetadataFunding)>) -> Result<T, E>,
    {
        let state = self.state();
        reserve(
            &self.funding,
            &[
                size_of::<F>(),
                size_of::<T>(),
                size_of::<E>(),
                size_of::<Result<T, E>>(),
                size_of::<Result<Result<T, E>, Error>>(),
                size_of::<TransientRootsProjection>(),
                TransientRootsProjection::control_bytes().ok_or_else(overflow)?,
                size_of::<RootLoan<'_>>(),
                size_of::<std::cell::RefMut<'_, Option<TransientRootsProjection>>>(),
                size_of::<Result<(), TransientRootsProjection>>(),
                size_of::<(&Self, &OriginalScopeObserver, &Stream)>(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let fail = |cause| failure(cause, &state.source, &self.funding);
        let bound = state
            .bound
            .get()
            .ok_or_else(|| fail(Cause::Invocation(ParallelInvocationCause::Unbound)))?;
        reserve(
            &self.funding,
            &[bound
                .stream
                .source_comparison_control_bytes()
                .ok_or_else(overflow)?],
        )?;
        if !state.forward_done.get()
            || state.closed.get()
            || state.calling.get()
            || !observer.same_scope(&bound.observer)
            || !bound.stream.matches_source(compute)
            || state.publication_started.get()
        {
            return Err(fail(Cause::Identity));
        }
        let (group, publication, root) = state
            .quote_source
            .publication()
            .ok_or_else(|| fail(Cause::Identity))?;
        if !matches!(state.occurrences.get(state.next.get()).map(|row|&row.operation),
            Some(InvocationOperation::Broadcast{group,root:r}) if *group==publication.group&&*r==root)
        {
            return Err(fail(Cause::Identity));
        }
        state.publication_started.set(true);
        reserve(
            &self.funding,
            &[
                group.retention_copy_bytes().ok_or_else(overflow)?,
                size_of::<Group>(),
                size_of::<Result<Group, std::collections::TryReserveError>>(),
                size_of::<OriginalParallelBinding>(),
                neural_failure_controls().ok_or_else(overflow)?,
            ],
        )?;
        let context = group
            .try_copy_for_retention()
            .map_err(|_| fail(Cause::Resource))?
            .with_original_parallel(OriginalParallelBinding {
                state: Rc::downgrade(self.state.as_ref().expect("live invocation")),
                source: state.source.clone(),
                fallback: neural_failure(Cause::Resource, &state.source, &self.funding),
                funding: self.funding.clone(),
            });
        *state
            .publication_roots
            .try_borrow_mut()
            .map_err(|_| fail(Cause::Identity))? = Some(roots);
        let _roots = RootLoan(&state.publication_roots);
        let output = run(Some((&context, &self.funding)));
        if output.is_ok() {
            if state.next.get() != state.occurrences.len() || state.calling.get() {
                return Err(fail(Cause::Invocation(ParallelInvocationCause::Incomplete)));
            }
            state.closed.set(true);
        }
        // Failure never closes the observer or releases Q. The enclosing
        // transaction keeps the accepted graph through recovery/quarantine.
        Ok(output)
    }
}

impl OriginalParallelBinding {
    pub(crate) fn with_publication_group<T, E, F>(
        &self,
        group: &Group,
        prepared: &Group,
        funding: &WorkspaceMetadataFunding,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<&Group>) -> Result<T, E>,
    {
        reserve(
            &self.funding,
            &[
                size_of::<F>(),
                size_of::<T>(),
                size_of::<E>(),
                size_of::<Result<T, E>>(),
                size_of::<Result<Result<T, E>, Error>>(),
                size_of::<ControlSourceLoan>(),
                size_of::<(&Group, &Group, &WorkspaceMetadataFunding, &Stream)>(),
                CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let state = self
            .state
            .upgrade()
            .ok_or_else(|| Error::Neural(self.fallback.clone()))?;
        let loan = ControlSourceLoan(Some(state));
        let state = loan.0.as_deref().expect("active publication source");
        state.validate_publication(prepared, stream)?;
        let source = state.quote_source.communication_source()?;
        let (_, publication, _) = state
            .quote_source
            .publication()
            .ok_or_else(|| failure(Cause::Identity, &self.source, &self.funding))?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(publication.group, CommunicationOperation::Broadcast)
            .map_err(|cause| failure(Cause::Rank(cause), &self.source, &self.funding))?;
        if !funding.same_account(&self.funding) || !source.matches_group(selected.order(), group) {
            return Err(failure(Cause::Identity, &self.source, &self.funding));
        }
        Ok(run(Some(prepared)))
    }
    pub(crate) fn publish(
        &self,
        group: &Group,
        input: &Array,
        root: usize,
        stream: &Stream,
    ) -> Result<
        (
            OriginalCommunicationConstructed,
            OriginalCommunicationCompletion,
        ),
        Error,
    > {
        reserve(
            &self.funding,
            &[
                size_of::<ControlSourceLoan>(),
                size_of::<Option<Rc<State>>>(),
                size_of::<(&Self, &Group, &Array, usize, &Stream)>(),
                size_of::<
                    Result<
                        (
                            OriginalCommunicationConstructed,
                            OriginalCommunicationCompletion,
                        ),
                        Error,
                    >,
                >(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let state = self
            .state
            .upgrade()
            .ok_or_else(|| Error::Neural(self.fallback.clone()))?;
        let loan = ControlSourceLoan(Some(state));
        loan.0
            .as_deref()
            .expect("active publication")
            .publish(group, input, root, stream)
    }
}

impl State {
    fn validate_publication(&self, group: &Group, stream: &Stream) -> Result<(), Error> {
        reserve(
            &self.funding,
            &[
                size_of::<(&Self, &Group, &Stream)>(),
                size_of::<Result<(), Error>>(),
                OriginalScopeObserver::control_bytes().ok_or_else(overflow)?,
                size_of::<OriginalScopeObserver>(),
                size_of::<Result<OriginalScopeObserver, safemlx::error::Exception>>(),
                CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let fail = |cause| failure(cause, &self.source, &self.funding);
        let bound = self
            .bound
            .get()
            .ok_or_else(|| fail(Cause::Invocation(ParallelInvocationCause::Unbound)))?;
        reserve(
            &self.funding,
            &[bound
                .stream
                .source_comparison_control_bytes()
                .ok_or_else(overflow)?],
        )?;
        let current =
            OriginalScopeObserver::require_current().map_err(|cause| fail(Cause::Native(cause)))?;
        if self.closed.get()
            || !self.forward_done.get()
            || !self.publication_started.get()
            || self
                .publication_roots
                .try_borrow()
                .map_err(|_| fail(Cause::Identity))?
                .is_none()
            || !bound.stream.matches_source(stream)
            || !current.same_scope(&bound.observer)
            || self.authority.ensure_active().is_err()
        {
            return Err(fail(Cause::Identity));
        }
        let (_, publication, _) = self
            .quote_source
            .publication()
            .ok_or_else(|| fail(Cause::Identity))?;
        let source = self.quote_source.communication_source()?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(publication.group, CommunicationOperation::Broadcast)
            .map_err(|cause| fail(Cause::Rank(cause)))?;
        if !source.matches_group(selected.order(), group) {
            return Err(fail(Cause::Identity));
        }
        Ok(())
    }
    fn publish(
        &self,
        group: &Group,
        input: &Array,
        root: usize,
        stream: &Stream,
    ) -> Result<
        (
            OriginalCommunicationConstructed,
            OriginalCommunicationCompletion,
        ),
        Error,
    > {
        reserve(
            &self.funding,
            &[
                size_of::<Calling<'_>>(),
                size_of::<Array>(),
                size_of::<OriginalCommunicationSource<'_>>(),
                size_of::<Result<OriginalCommunicationSource<'_>, Error>>(),
                size_of::<OriginalCommunicationOperation<'_>>(),
                size_of::<Result<OriginalCommunicationOperation<'_>, Error>>(),
                size_of::<OriginalCommunicationCompletedOperation<'_>>(),
                size_of::<Result<OriginalCommunicationCompletedOperation<'_>, Error>>(),
                size_of::<OriginalCommunicationBacking<'_, '_>>(),
                size_of::<Result<OriginalCommunicationBacking<'_, '_>, Error>>(),
                size_of::<ReadyCompletionResources>(),
                size_of::<Result<ReadyCompletionResources, Error>>(),
                size_of::<AcceptedCommunicationSource<'_>>(),
                size_of::<Result<AcceptedCommunicationSource<'_>, Error>>(),
                size_of::<OriginalCommunicationConstructed>(),
                size_of::<OriginalCommunicationCompletion>(),
                size_of::<(Array, RetainedCommunicationSource, WorkspaceMetadataFunding)>(),
                size_of::<Result<(Array, OriginalCommunicationCompletion), Error>>(),
                size_of::<Option<Result<(Array, OriginalCommunicationCompletion), Error>>>(),
                size_of::<(
                    usize,
                    &Occurrence,
                    eredu_runtime::CommunicationGroupOperation<'_>,
                    &Stream,
                    u64,
                )>(),
                CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
                TransientRootsProjection::control_bytes().ok_or_else(overflow)?,
                TransientRootsProjection::publication_settlement_control_bytes().ok_or_else(overflow)?,
                size_of::<std::cell::Ref<'_, Option<TransientRootsProjection>>>(),
                size_of::<Option<TransientRootsProjection>>(),
                size_of::<Result<Array, safemlx::error::Exception>>(),
                size_of::<usize>(),
                size_of::<(&Self, &Group, &Array, usize, &Stream)>(),
                size_of::<
                    Result<
                        (
                            OriginalCommunicationConstructed,
                            OriginalCommunicationCompletion,
                        ),
                        Error,
                    >,
                >(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        self.validate_publication(group, stream)?;
        let fail = |cause| failure(cause, &self.source, &self.funding);
        if self.calling.replace(true) {
            return Err(fail(Cause::Invocation(ParallelInvocationCause::Recursive)));
        }
        let _calling = Calling(&self.calling);
        let index = self.next.get();
        let row = self
            .occurrences
            .get(index)
            .ok_or_else(|| fail(Cause::Invocation(ParallelInvocationCause::Exhausted)))?;
        let InvocationOperation::Broadcast {
            group: id,
            root: quoted_root,
        } = &row.operation
        else {
            return Err(fail(Cause::Identity));
        };
        let id = *id;
        if root != *quoted_root {
            return Err(fail(Cause::Identity));
        }
        // Consume before any native producer. Neither failure nor rollback may
        // reuse this occurrence or its contribution/input-settlement allowance.
        self.next.set(index.checked_add(1).ok_or_else(overflow)?);
        let contribution = if group.rank() == root {
            input.clone()
        } else {
            input
                .multiply(
                    Array::try_from_f32(0.0).map_err(|cause| fail(Cause::Native(cause)))?,
                    stream,
                )
                .map_err(|cause| fail(Cause::Native(cause)))?
        };
        let roots = self
            .publication_roots
            .try_borrow()
            .map_err(|_| fail(Cause::Identity))?
            .as_ref()
            .cloned()
            .ok_or_else(|| fail(Cause::Identity))?;
        roots.append(&contribution)?;
        roots.settle_publication(&contribution,stream)?;
        // The actual contribution is now a settled leaf. Revalidate its exact
        // prequoted layout/CPU population before the accepted Sum constructor.
        let native=row.native.as_ref().ok_or_else(||fail(Cause::Identity))?;
        reserve(&self.funding,&[native.binding_control_bytes().ok_or_else(overflow)?])?;
        let actual = native
            .bind_actual(&contribution)
            .map_err(|_| fail(Cause::Resource))?;
        drop(actual);
        let source = self.quote_source.communication_source()?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(id, CommunicationOperation::Broadcast)
            .map_err(|cause| fail(Cause::Rank(cause)))?;
        let operation = source.group_cpu_operation_storage(
            selected.order(),
            &contribution,
            safemlx::distributed::GroupWorkerOperation::Sum,
        )?;
        let backing = operation
            .backing_storage(self.quote_source.input_runtime())?
            .capacity();
        let quoted_backing = row
            .backing
            .output
            .checked_add(row.backing.scratch)
            .ok_or_else(overflow)?;
        if u64::try_from(backing)
            .ok()
            .is_none_or(|capacity| capacity > quoted_backing)
        {
            return Err(fail(Cause::Resource));
        }
        let operation = operation.with_completion()?;
        let ready = operation.prepare_resources(&source, Some(selected.order()))?;
        let bound = self.bound.get().expect("validated publication observer");
        let transport = group
            .retained_transport_stream()
            .ok_or_else(|| fail(Cause::Identity))?;
        let accepted = operation.construct_accepted(&source, &bound.observer, transport)?;
        if accepted.outputs()[0].shape() != input.shape()
            || accepted.outputs()[0].dtype() != input.dtype()
        {
            return Err(fail(Cause::Output));
        }
        accepted.submit(ready)
    }
}
