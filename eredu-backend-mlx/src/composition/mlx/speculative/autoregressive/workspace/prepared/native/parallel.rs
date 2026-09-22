//! Actual speculative completion lends the retained partition invocation.
use super::*;
use crate::backend::runtime::distributed::{
    Group,
    topology::original_source::{
        control::OriginalParallelControlProjection, parallel::OriginalParallelInvocation,
    },
};
use crate::backend::submission_recovery::prefill::{
    InvocationRootSource, TransientRootsProjection,
};

impl Active {
    pub(super) fn root_observer(&self) -> Result<OriginalScopeObserver, Error> {
        if let Some(cause) = self.observer.retained_failure() {
            return Err(cause.into());
        }
        if self.completed.get()
            || !self.graph_started.get()
            || self.observer.status().blocked()
            || !self
                .observer
                .same_scope(&OriginalScopeObserver::require_current()?)
        {
            return Err(failure(Cause::Source, &self.role));
        }
        Ok(self.observer.clone())
    }
}
impl InvocationRootSource for Active {
    fn source_custody(&self) -> Option<eredu_runtime::working_memory::OriginalHostSourceCustody> {
        Some(self.role.budget_custody().into())
    }
    fn scope_observer(&self) -> Result<OriginalScopeObserver, Error> {
        if let Some(cause) = self.observer.retained_failure() {
            return Err(cause.into());
        }
        if self.completed.get()
            || self.observer.status().blocked()
            || !self
                .observer
                .same_scope(&OriginalScopeObserver::require_current()?)
        {
            return Err(failure(Cause::Source, &self.role));
        }
        Ok(self.observer.clone())
    }
    fn metadata_funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    fn observer(&self) -> Result<OriginalScopeObserver, Error> {
        self.root_observer()
    }
    fn completion_recipe(&self) -> Result<ResidentCompletionRecipe, Error> {
        self.root_observer()?;
        Ok(self.completion)
    }
    fn append(&self, value: &Array) -> Result<(), Error> {
        self.root_observer()?;
        self.roots
            .0
            .try_borrow_mut()
            .map_err(|_| failure(Cause::Source, &self.role))?
            .append(value)
            .map_err(|cause| failure(cause.into(), &self.role))
    }
    fn retire_completed(&self, value: &Array) -> Result<(), Error> {
        self.root_observer()?;
        self.roots
            .0
            .try_borrow_mut()
            .map_err(|_| failure(Cause::Source, &self.role))?
            .retire_completed_current(value)
            .map_err(|cause| failure(cause.into(), &self.role))
    }
    fn close_construction(&self) -> Result<(), Error> {
        self.root_observer()?;
        let graph = self
            .graph
            .try_borrow_mut()
            .map_err(|_| failure(Cause::Source, &self.role))?
            .take();
        if graph.is_none() || self.publication_closed.replace(true) || self.readout_started.get() {
            return Err(failure(Cause::Source, &self.role));
        }
        drop(graph);
        Ok(())
    }
}
impl ActiveSpeculativeInvocation {
    pub(super) fn transient_roots(&self) -> TransientRootsProjection {
        TransientRootsProjection::from_invocation(self.inner())
    }
    pub(crate) fn with_parallel<T, E, F>(
        &self,
        context: &Stream,
        control: Option<&OriginalParallelControlProjection>,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&mut Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        self.authenticate(context)
            .map_err(|cause| failure(cause, self.role()))?;
        // The actual AR model owner is the sole request projection for this
        // role. A Text installation cannot be substituted into its group loan.
        if control.is_some() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let model = self.inner().model_control.as_ref();
        struct Close<'a>(Option<&'a ModelControl>);
        impl Drop for Close<'_> {
            fn drop(&mut self) {
                if let Some(model) = self.0 {
                    model.close();
                }
            }
        }
        let _close = Close(model);
        let result = match &self.inner().parallel {
            Some(parallel) => {
                let model = model.ok_or(Error::PrefillScopeUnavailable)?;
                parallel.prepare_lending::<F, Result<T, E>>()?;
                let neural = parallel.has_neural_context();
                parallel.with_model_context(
                    self.observer(),
                    context,
                    Some(&model.projection),
                    self.transient_roots(),
                    |parallel, funding| run(neural.then_some((parallel, funding))),
                )
            }
            None => Ok(run(None)),
        };
        if let Some(model) = model {
            let success = matches!(&result, Ok(Ok(_)));
            let closed = model.finish(success);
            if success {
                closed?;
            }
        }
        result
    }

    pub(crate) fn with_parallel_publication<T, E, F>(
        &self,
        context: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        self.authenticate(context)
            .map_err(|cause| failure(cause, self.role()))?;
        match &self.inner().parallel {
            Some(parallel) => {
                parallel.prepare_lending::<F, Result<T, E>>()?;
                parallel.with_publication_context(
                    self.observer(),
                    context,
                    self.transient_roots(),
                    run,
                )
            }
            None => Ok(run(None)),
        }
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Option<&ModelControl>>(),
        size_of::<Result<(), Error>>(),
        size_of::<bool>(),
        size_of::<OriginalParallelControlOwner>(),
        size_of::<(
            OriginalParallelControlInstallation,
            OriginalParallelControlProjection,
        )>(),
        size_of::<
            Result<
                (
                    OriginalParallelControlInstallation,
                    OriginalParallelControlProjection,
                ),
                Error,
            >,
        >(),
        TransientRootsProjection::control_bytes()?,
        TransientRootsProjection::boundary_traversal_control_bytes()?,
        TransientRootsProjection::publication_settlement_control_bytes()?,
        size_of::<Option<Rc<OriginalParallelInvocation>>>(),
        size_of::<
            Result<Rc<OriginalParallelInvocation>, eredu_nn::workspace::WorkspaceMetadataError>,
        >(),
        WorkspaceContext::metadata_rc_bytes::<OriginalParallelInvocation>()?,
        size_of::<(&Active, &Array)>(),
        size_of::<(&ActiveSpeculativeInvocation, &Stream)>(),
        size_of::<Result<OriginalScopeObserver, Error>>(),
        size_of::<std::cell::RefMut<'_, PrefillRoots>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
