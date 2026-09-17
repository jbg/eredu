//! Native scope and completion for one already admitted Embedded equation.
//! The caller authenticates current state and parameter sources before admission.
use super::original_domains::{
    prepare_domains, OriginalSpeculativeDomainError, OriginalSpeculativeDomains,
    OriginalSpeculativeRoots,
};
use crate::backend::{
    error::Error,
    nn::{
        tensor::{TokenValidationIngress, TokenValidationScope},
        workspace::{EmbeddedEquationRecipe, ResidentCompletionRecipe},
    },
    submission_recovery::{PreparedRecovery, Retention, Status},
    OriginalCopyEnvironment,
};
use eredu_nn::workspace::WorkspaceMetadataFunding;
use eredu_runtime::working_memory::{
    OriginalEmbeddedSpeculativeRole, OriginalSpeculativeBudgetCustody,
};
use safemlx::{
    error::Exception, Array, OriginalBufferBudget, OriginalScopeObserver, PrefillRootsRuntime,
    PreparedPipelineCachePlan, Stream, SubmissionScope,
};
mod role;
pub(crate) use role::{ExternalEquationRecipe, ModelEquation, ModelRole};
use role::EquationIdentity;
use std::{
    cell::{Cell, RefCell},
    mem::{size_of, size_of_val},
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum EmbeddedNativeCause {
    #[error(transparent)]
    Domains(#[from] OriginalSpeculativeDomainError),
    #[error(transparent)]
    Native(#[from] Exception),
    #[error(transparent)]
    Backend(#[from] Error),
    #[error(transparent)]
    Scope(#[from] safemlx::SubmissionScopeOwnerCause),
    #[error(transparent)]
    Controls(#[from] safemlx::OriginalNativeControlError),
    #[error(transparent)]
    Roots(#[from] safemlx::PrefillRootsError),
    #[error("embedded invocation differs from its exact native scope, stream or equation")]
    Source,
    #[error("embedded native invocation has an incomplete bound")]
    Unknown,
    #[error("embedded native invocation has an incomplete {0} bound")]
    Missing(&'static str),
    #[error("embedded native invocation geometry overflow")]
    Overflow,
    #[error("embedded native observation unavailable: {0:?}")]
    Observation(safemlx::ScopedSubmissionProgress),
    #[error("embedded native invocation failed or did not establish completion")]
    Completion,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: EmbeddedNativeCause,
    custody: OriginalSpeculativeBudgetCustody,
}
fn failure(cause: EmbeddedNativeCause, role: &impl ModelRole) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(Failure {
            cause,
            custody: role.budget_custody(),
        })
        .with_operation("execute original embedded equation"),
        false,
    )
}

/// The native layout is constructed only from the retained equation reducer.
/// Source identity and all enclosing input/copy/output producers are separate.
pub(crate) struct EmbeddedNativeLayout {
    plan: eredu_runtime::working_memory::InferenceSpanWorkspacePlan,
    workspace: EquationIdentity,
    completion: ResidentCompletionRecipe,
    graph_capacity: usize,
    record_capacity: usize,
    physical_capacity: usize,
    pipeline: PreparedPipelineCachePlan,
    graph_bytes: u64,
    record_bytes: u64,
    native_controls: u64,
}
impl EmbeddedNativeLayout {
    pub(crate) fn inspect<E: ModelEquation>(
        recipe: &E,
        environment: &OriginalCopyEnvironment<'_>,
        preparation_graph_bytes: usize,
        preparation_mutable_bytes: usize,
    ) -> Result<Self, EmbeddedNativeCause> {
        let (completion, graph_capacity, record_capacity, kernels, query_controls) =
            recipe.equation_domains(preparation_graph_bytes)
                .map_err(|cause| cause.at_speculative_stage("embedded equation native domains"))?;
        let runtime = environment
            .input_runtime()
            .map_err(OriginalSpeculativeDomainError::from)?;
        let storage = recipe
            .record()
            .mutable_storage()
            .ok_or(EmbeddedNativeCause::Missing("native backing"))?;
        let population = OriginalBufferBudget::metal_population_layout(
            &runtime,
            usize::try_from(storage.mutable_bytes()).map_err(|_| EmbeddedNativeCause::Overflow)?,
            storage.maximum_births(),
        )
        .map_err(OriginalSpeculativeDomainError::from)?;
        let trace = recipe
            .plan()
            .records()
            .first()
            .ok_or(EmbeddedNativeCause::Missing("equation span"))?;
        let physical_capacity = population
            .capacity()
            .max(
                usize::try_from(
                    trace
                        .new_tensor_allocation_bytes()
                        .ok_or(EmbeddedNativeCause::Missing("equation tensor allocation"))?,
                )
                .map_err(|_| EmbeddedNativeCause::Overflow)?,
            )
            .checked_add(preparation_mutable_bytes)
            .ok_or(EmbeddedNativeCause::Overflow)?;
        let graph =
            safemlx::PreparedSubmissionGraphQuota::<OriginalSpeculativeBudgetCustody>::layout(
                graph_capacity,
            )
            .map_err(OriginalSpeculativeDomainError::from)?;
        let record =
            safemlx::PreparedSubmissionRecordQuota::<OriginalSpeculativeBudgetCustody>::layout(
                record_capacity,
            )
            .map_err(OriginalSpeculativeDomainError::from)?;
        let pipeline = PreparedPipelineCachePlan::new(kernels);
        let roots = safemlx::PrefillRoots::layout(completion.traversal.roots())
            .map_err(OriginalSpeculativeDomainError::from)?;
        let parts = [
            size_of::<Self>(),
            size_of::<u64>(),
            size_of::<Result<Self, EmbeddedNativeCause>>(),
            size_of::<OriginalSpeculativeDomains>(),
            super::original_domains::preparation_control_bytes()
                .ok_or(EmbeddedNativeCause::Overflow)?,
            size_of::<OriginalSpeculativeRoots>(),
            roots.host_bytes().ok_or(EmbeddedNativeCause::Overflow)?,
            rc_bytes::<RefCell<safemlx::PrefillRoots>>().ok_or(EmbeddedNativeCause::Overflow)?,
            population.control_bytes(),
            usize::try_from(recipe.validation_control_bytes()
                .map_err(|cause| cause.at_speculative_stage("embedded validation controls"))?)
                .map_err(|_| EmbeddedNativeCause::Overflow)?,
            safemlx::PreparedOriginalBufferBudget::<OriginalSpeculativeBudgetCustody>::layout(
                &runtime,
                physical_capacity,
            )
            .map_err(OriginalSpeculativeDomainError::from)?
            .total_owner_bytes()
            .ok_or(EmbeddedNativeCause::Overflow)?,
            safemlx::PreparedPrefillFailure::<OriginalSpeculativeBudgetCustody>::layout()
                .map_err(OriginalSpeculativeDomainError::from)?
                .total_bytes()
                .ok_or(EmbeddedNativeCause::Overflow)?,
            pipeline
                .layout::<OriginalSpeculativeBudgetCustody>()
                .map_err(OriginalSpeculativeDomainError::from)?
                .required_bytes()
                .ok_or(EmbeddedNativeCause::Overflow)?,
            safemlx::OriginalNativeControlLayout::inspect()?.fixed_control_bytes,
            safemlx::original_scoped_evaluation_control_bytes()
                .ok_or(EmbeddedNativeCause::Missing("scoped evaluation controls"))?,
            OriginalCopyEnvironment::control_bytes().ok_or(EmbeddedNativeCause::Missing("copy environment controls"))?,
            completion
                .graph
                .control_bytes()
                .ok_or(EmbeddedNativeCause::Missing("completion graph controls"))?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()
                .and_then(|n| n.checked_mul(2))
                .ok_or(EmbeddedNativeCause::Missing("failure retention controls"))?,
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(EmbeddedNativeCause::Overflow)?;
        let host_workspace = trace.host_workspace_bytes()
            .ok_or(EmbeddedNativeCause::Missing("equation host workspace"))?;
        let native_controls = controls
            .checked_add(query_controls)
            .and_then(|n| n.checked_add(host_workspace))
            .ok_or(EmbeddedNativeCause::Overflow)?;
        Ok(Self {
            plan: recipe.plan().clone(),
            workspace: recipe.identity(),
            completion,
            graph_capacity,
            record_capacity,
            physical_capacity,
            pipeline,
            graph_bytes: u64::try_from(graph.total_bytes().ok_or(EmbeddedNativeCause::Overflow)?)
                .map_err(|_| EmbeddedNativeCause::Overflow)?,
            record_bytes: u64::try_from(record.total_bytes().ok_or(EmbeddedNativeCause::Overflow)?)
                .map_err(|_| EmbeddedNativeCause::Overflow)?,
            native_controls,
        })
    }
    pub(crate) fn physical_bytes(&self) -> Option<u64> {
        u64::try_from(self.physical_capacity).ok()
    }
    pub(crate) fn graph_bytes(&self) -> u64 {
        self.graph_bytes
    }
    pub(crate) fn record_bytes(&self) -> u64 {
        self.record_bytes
    }
    /// Actual caller closures and result representations are included before
    /// role admission. Payload allocations and source/transfer banks are quoted
    /// by their producers; this accounts for their outer retained value.
    pub(crate) fn control_bytes<Q: 'static, W, T, B, F, G, H>(&self, handlers: &(F, G, H)) -> Option<u64> {
        self.control_bytes_for_role::<OriginalEmbeddedSpeculativeRole, Q, W, T, B, F, G, H>(handlers)
    }
    pub(crate) fn control_bytes_for_role<R: ModelRole, Q: 'static, W, T, B, F, G, H>(&self, _: &(F, G, H)) -> Option<u64> {
        let parts = [
            size_of::<R>(),
            size_of::<Retained<Q>>(),
            size_of::<ActiveEmbeddedNativeInvocation<'_, R>>(),
            size_of::<Construction<'_, R>>(),
            size_of::<B>(),
            size_of::<Result<B, Error>>(),
            size_of::<Result<T, Error>>(),
            size_of::<Result<T, Error>>(),
            size_of::<T>(),
            size_of::<(
                &mut W,
                &R::Equation,
                &OriginalCopyEnvironment<'_>,
                &PrefillRootsRuntime,
            )>(),
            size_of::<(F, G, H)>(),
            size_of::<(&mut W, &Q, &SubmissionScope)>(),
            size_of::<(&mut W, &Q, &ActiveEmbeddedNativeInvocation<'_, R>)>(),
            size_of::<EmbeddedNativeCause>(),
            size_of::<Failure>(),
            size_of::<Result<safemlx::SubmissionRetirement, Exception>>(),
            size_of::<Option<safemlx::PreparedResidentGraph>>(),
            size_of::<Result<safemlx::PreparedResidentGraph, Exception>>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<OriginalBufferBudget>(),
            size_of::<TokenValidationIngress>(),
            size_of::<Option<TokenValidationScope>>(),
            size_of::<
                Result<
                    (OriginalSpeculativeDomains, OriginalSpeculativeRoots),
                    OriginalSpeculativeDomainError,
                >,
            >(),
        ];
        let fixed = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)?;
        self.native_controls
            .checked_add(u64::try_from(fixed).ok()?)?
            .checked_add(PreparedRecovery::<
                Retained<Q>,
                OriginalSpeculativeBudgetCustody,
            >::control_bytes()?)
    }

    /// Scope mechanics only. The caller has bound actual source loans and
    /// accepted this exact role. The same prepared recovery engine retains all
    /// native work on failure/unwind, and publication follows terminal retirement.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run<Q: 'static, W, T, B, F, G, H, R: ModelRole>(
        self,
        recipe: &R::Equation,
        environment: &OriginalCopyEnvironment<'_>,
        roots_runtime: &PrefillRootsRuntime,
        role: R,
        payload: Q,
        funding: WorkspaceMetadataFunding,
        work: &mut W,
        handlers: (F, G, H),
    ) -> Result<T, Error>
    where
        F: FnOnce(&mut W, &Q, &SubmissionScope) -> Result<B, Error>,
        G: FnOnce(&mut W, &Q, &ActiveEmbeddedNativeInvocation<'_, R>) -> Result<T, Error>,
        H: FnOnce(
            &mut W,
            &T,
            &OriginalBufferBudget,
            &OriginalSpeculativeBudgetCustody,
        ) -> Result<(), Error>,
    {
        if !self.plan.same_plan(recipe.plan()) || self.workspace != recipe.identity() {
            return Err(failure(EmbeddedNativeCause::Source, &role));
        }
        role.validate_equation(recipe).map_err(|cause| failure(cause.into(), &role))?;
        if self.physical_bytes() != Some(role.physical_bytes())
            || self.graph_bytes != role.graph_bytes()
            || self.record_bytes != role.record_bytes()
        {
            return Err(failure(EmbeddedNativeCause::Source, &role));
        }
        let (domains, roots) = prepare_domains(
            environment,
            role.budget_custody(),
            roots_runtime,
            self.graph_capacity,
            self.record_capacity,
            self.physical_capacity,
            &self.pipeline,
            self.completion.traversal.roots(),
        )
        .map_err(|cause| failure(cause.into(), &role))?;
        let ingress = role.prepare_validations(recipe)
            .map_err(|cause| failure(cause.into(), &role))?;
        let graph = domains.graph.clone();
        let record = domains.record.clone();
        let buffer = domains.buffer.clone();
        let retained = Retained {
            roots: roots.clone(),
            domains,
            payload,
            funding,
            custody: role.budget_custody(),
        };
        let mut recovery = PreparedRecovery::new(retained, role.budget_custody())
            .map_err(|error| failure(error.cause.into(), &role))?
            .with_graph_quota(Some(graph))
            .with_record_quota(Some(record))
            .try_begin()
            .map_err(|error| failure(error.cause.into(), &role))?;
        recovery
            .configure_scope(|scope| -> Result<(), EmbeddedNativeCause> {
                scope
                    .enable_scoped_observation()
                    .map_err(EmbeddedNativeCause::Observation)?;
                scope.require_original_native_controls()?;
                roots.0.borrow_mut().bind_scope(scope).map_err(OriginalSpeculativeDomainError::from)?;
                scope.enable_original_native_controls()?;
                scope.bind_original_buffer_budget(&buffer).map_err(OriginalSpeculativeDomainError::from)?;
                Ok(())
            })
            .map_err(|cause| failure(cause, &role))?;
        let observer = OriginalScopeObserver::require_current()
            .map_err(|cause| failure(cause.into(), &role))?;
        let active = ActiveEmbeddedNativeInvocation {
            roots: &roots,
            observer: &observer,
            buffer: &buffer,
            role: &role,
            completion: self.completion,
            stream: environment.stream(),
            ingress: RefCell::new(ingress),
            validations: RefCell::new(None),
            graph: RefCell::new(None),
            started: Cell::new(false),
            completed: Cell::new(false),
        };
        let (bind, execute, publish) = handlers;
        let result = recovery.configure_scope_with_retention(|scope, retained| {
            let bank = bind(work, &retained.payload, scope)
                .map_err(|cause| failure(cause.into(), &role))?;
            let construction = Construction(&active);
            let result = execute(work, &retained.payload, &active)
                .map_err(|cause| failure(cause.into(), &role));
            drop(construction);
            drop(bank);
            let output = result?;
            if !active.completed.get() {
                return Err(failure(EmbeddedNativeCause::Completion, &role));
            }
            Ok(output)
        });
        recovery.seal();
        let output = result?;
        let status = recovery.finish();
        if !status.settled || status.failed || status.blocked {
            return Err(failure(EmbeddedNativeCause::Completion, &role));
        }
        if !matches!(
            observer
                .retire_completed_records()
                .map_err(|cause| failure(cause.into(), &role))?,
            safemlx::SubmissionRetirement::CompleteSnapshot
        ) {
            return Err(failure(EmbeddedNativeCause::Completion, &role));
        }
        publish(work, &output, &buffer, &role.budget_custody()).map_err(|cause| failure(cause.into(), &role))?;
        Ok(output)
    }
}
struct Retained<Q: 'static> {
    roots: OriginalSpeculativeRoots,
    domains: OriginalSpeculativeDomains,
    payload: Q,
    funding: WorkspaceMetadataFunding,
    custody: OriginalSpeculativeBudgetCustody,
}
impl<Q: 'static> Retention for Retained<Q> {
    fn observe(&self, _: Status) {}
}

/// Lexically borrowed from its actual scope, so it cannot outlive recovery.
pub(crate) struct ActiveEmbeddedNativeInvocation<'a, R: ModelRole = OriginalEmbeddedSpeculativeRole> {
    roots: &'a OriginalSpeculativeRoots,
    observer: &'a OriginalScopeObserver,
    buffer: &'a OriginalBufferBudget,
    role: &'a R,
    completion: ResidentCompletionRecipe,
    stream: &'a Stream,
    ingress: RefCell<TokenValidationIngress>,
    validations: RefCell<Option<TokenValidationScope>>,
    graph: RefCell<Option<safemlx::PreparedResidentGraph>>,
    started: Cell<bool>,
    completed: Cell<bool>,
}
impl<R: ModelRole> ActiveEmbeddedNativeInvocation<'_, R> {
    pub(crate) fn observer(&self) -> &OriginalScopeObserver {
        self.observer
    }
    pub(crate) fn budget(&self) -> &OriginalBufferBudget {
        self.buffer
    }
    pub(crate) fn role(&self) -> &R {
        self.role
    }
    fn authenticate(&self) -> Result<(), EmbeddedNativeCause> {
        if !self
            .observer
            .same_scope(&OriginalScopeObserver::require_current()?)
        {
            return Err(EmbeddedNativeCause::Source);
        }
        Ok(())
    }
    /// Reuses only this invocation's already installed prepared validation
    /// collector. An equal stream alone cannot admit an ordinary collector.
    pub(crate) fn validate_equation_scope(&self, stream: &Stream) -> Result<(), Error> {
        let result = (|| -> Result<(), EmbeddedNativeCause> {
            self.authenticate()?;
            if stream != self.stream || !self.started.get() || self.completed.get()
                || self.validations.try_borrow().map_err(|_| EmbeddedNativeCause::Source)?.is_none()
                || self.graph.try_borrow().map_err(|_| EmbeddedNativeCause::Source)?.is_none()
            {
                return Err(EmbeddedNativeCause::Source);
            }
            Ok(())
        })();
        result.map_err(|cause| failure(cause, self.role))
    }

    pub(crate) fn begin_equation(&self) -> Result<(), Error> {
        let result = (|| -> Result<(), EmbeddedNativeCause> {
            self.authenticate()?;
            if self.started.replace(true) {
                return Err(EmbeddedNativeCause::Source);
            }
            *self.validations.borrow_mut() = Some(self.ingress.borrow_mut().begin()?);
            let mut graph = safemlx::OperationEvent::prepare_resident_graph(
                self.completion.graph,
                self.observer,
            )?;
            if self.completion.nested_completions != 0 {
                graph.configure_nested_completions(
                    &self
                        .completion
                        .nested_traversal()
                        .ok_or(EmbeddedNativeCause::Unknown)?,
                    self.completion.nested_completions,
                )?;
            }
            *self.graph.borrow_mut() = Some(graph);
            Ok(())
        })();
        result.map_err(|cause| failure(cause, self.role))
    }
    /// Visits actual closing state and every output dependency without a new
    /// vector or native clone inventory. The caller includes capture side roots.
    pub(crate) fn complete(
        &self,
        visit: impl FnOnce(&mut dyn FnMut(&Array)) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let result = (|| -> Result<(), EmbeddedNativeCause> {
            self.authenticate()?;
            if self.completed.get() || !self.started.get() {
                return Err(EmbeddedNativeCause::Source);
            }
            drop(
                self.graph
                    .borrow_mut()
                    .take()
                    .ok_or(EmbeddedNativeCause::Source)?,
            );
            let mut roots = self.roots.0.borrow_mut();
            let mut appended = Ok(());
            visit(&mut |value| {
                if appended.is_ok() {
                    appended = roots.append(value);
                }
            })?;
            appended.map_err(OriginalSpeculativeDomainError::from)?;
            crate::backend::nn::tensor::append_active_token_validation_roots(&mut roots).map_err(OriginalSpeculativeDomainError::from)?;
            roots.complete_current_scope_on_stream_prepared(
                self.stream,
                &self.completion.traversal,
            )?;
            crate::backend::nn::tensor::validate_active_original_token_validations(self.observer)?;
            self.completed.set(true);
            Ok(())
        })();
        result.map_err(|cause| failure(cause, self.role))
    }
}
struct Construction<'a, R: ModelRole = OriginalEmbeddedSpeculativeRole>(&'a ActiveEmbeddedNativeInvocation<'a, R>);
impl<R: ModelRole> Drop for Construction<'_, R> {
    fn drop(&mut self) {
        drop(self.0.graph.borrow_mut().take());
        drop(self.0.validations.borrow_mut().take());
    }
}
fn rc_bytes<T>() -> Option<usize> {
    std::alloc::Layout::new::<[Cell<usize>; 2]>()
        .extend(std::alloc::Layout::new::<T>())
        .ok()
        .map(|layout| layout.0.pad_to_align().size())
}

mod addressable;
pub(crate) use addressable::AddressableModelSource;
