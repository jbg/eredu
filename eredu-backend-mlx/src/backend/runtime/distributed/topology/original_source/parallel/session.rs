//! Actual shared session forward under the prepared per-equation Group.
use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use eredu_runtime::{
    LayeredArchitecture, ReplicatedTextExecutionStrategy, ReplicatedTextSession,
    ReplicatedTextSessionMechanisms,
};

impl OriginalParallelInvocation {
    /// The same forward/prefill/control callback executes with the source-bound
    /// Group. No native state is copied here. The enclosing role retains `self`
    /// until completion/quarantine, including when the callback fails.
    pub(crate) fn with_session<A, M, D, T, F>(
        &self,
        session: &mut ReplicatedTextSession<A, MlxNeuralBackend, M, D>,
        observer: &OriginalScopeObserver,
        compute: &Stream,
        run: F,
    ) -> Result<T, Error>
    where
        M: ReplicatedTextSessionMechanisms<A, MlxNeuralBackend>,
        A: LayeredArchitecture<MlxNeuralBackend, M::State>,
        D: ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                M::State,
                M::ResidentPolicy,
                M::BoundedPolicy,
            >,
        A::Error: std::fmt::Display,
        M::PolicyError: std::fmt::Display,
        M::Error: std::fmt::Display,
        F: FnOnce(&mut ReplicatedTextSession<A, MlxNeuralBackend, M, D>) -> Result<T, Error>,
    {
        let state = self.state();
        reserve(
            &self.funding,
            &[
                size_of::<F>(),
                size_of::<T>(),
                size_of::<Result<T, Error>>(),
                size_of::<
                    Result<
                        Result<T, Error>,
                        eredu_runtime::replicated_session::PreparedParallelContextFailure<Group>,
                    >,
                >(),
                size_of::<(
                    &Self,
                    &mut ReplicatedTextSession<A, MlxNeuralBackend, M, D>,
                    &OriginalScopeObserver,
                    &Stream,
                )>(),
                self.context.retention_copy_bytes().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        if !state.quote_source.has_neural_context() {
            return Err(failure(Cause::Identity, &state.source, &self.funding));
        }
        self.bind(observer, compute)?;
        let context = self
            .context
            .try_copy_for_retention()
            .map_err(|_| failure(Cause::Resource, &state.source, &self.funding))?;
        let output = session
            .with_prepared_parallel_context(context, &self.funding, run)
            .map_err(|cause| {
                let (cause, context) = cause.into_parts();
                // The uninstalled Group owns a weak alias only. Drop it before
                // returning the separately retained source/error shell.
                drop(context);
                failure(Cause::Context(cause), &state.source, &self.funding)
            })??;
        self.finish_construction()?;
        Ok(output)
    }
}

impl OriginalParallelInvocation {
    /// Binds the existing original observer, then lends a paid context alias
    /// to the shared session. This layer knows no execution strategy/runtime.
    pub(crate) fn with_context<T, E, F>(
        &self,
        observer: &OriginalScopeObserver,
        compute: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(&mut Group, &HostMetadataFunding) -> Result<T, E>,
    {
        self.with_context_control(observer, compute, None, run)
    }
    /// The request control view is authenticated before entering the shared
    /// executor. Each reached inner wave obtains a separate actual event claim.
    pub(crate) fn with_context_control<T, E, F>(
        &self,
        observer: &OriginalScopeObserver,
        compute: &Stream,
        control: Option<&super::super::control::OriginalParallelControlProjection>,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(&mut Group, &HostMetadataFunding) -> Result<T, E>,
    {
        let state = self.state();
        reserve(
            &self.funding,
            &[
                size_of::<F>(),
                size_of::<T>(),
                size_of::<E>(),
                size_of::<Option<&super::super::control::OriginalParallelControlProjection>>(),
                size_of::<Option<super::super::control::OriginalParallelControlProjection>>(),
                size_of::<Result<T, E>>(),
                size_of::<Result<Result<T, E>, Error>>(),
                size_of::<(
                    &Self,
                    &mut Group,
                    &OriginalScopeObserver,
                    &Stream,
                    &HostMetadataFunding,
                )>(),
                self.context.retention_copy_bytes().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        if let Some(control) = control {
            control.validate_model_source(self.source())?;
        }
        self.bind(observer, compute)?;
        let mut context = self
            .context
            .try_copy_for_retention()
            .map_err(|_| failure(Cause::Resource, &state.source, &self.funding))?;
        if let Some(control) = control {
            context.bind_original_control_request(control.retained()?);
        }
        let output = run(&mut context, &self.funding);
        if output.is_ok() {
            self.finish_construction()?;
        }
        // A failed forward leaves the observer/source under the same Q until
        // independent completion or quarantine. Context restoration is no fence.
        Ok(output)
    }
}

impl OriginalParallelInvocation {
    /// Whether this retained source selects the publication worker which closes
    /// the enclosing construction bank. This does not activate that worker.
    pub(crate) fn has_publication(&self) -> bool {
        self.state().quote_source.publication().is_some()
    }
    /// Descriptive architecture selection; this does not acquire or activate a Group.
    pub(crate) fn has_neural_context(&self) -> bool {
        self.state().quote_source.has_neural_context()
    }
}
