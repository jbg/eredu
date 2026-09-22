//! Source-bound native domain construction and synchronous invocation installation.
//! The public independent request stays gated until its surrounding copy and
//! sampling producers use the same request. No supplied Scope becomes a grant.
use super::*;
mod capture;
mod parallel;
use crate::backend::nn::tensor::{TokenValidationIngress, TokenValidationScope};
use crate::backend::nn::workspace::{PagedScopeRetention, ResidentCompletionRecipe};
use crate::backend::runtime::distributed::topology::original_source::control::CaptureSourceOwner;
use crate::backend::runtime::distributed::topology::original_source::control::speculative::{
    SpeculativeModelControlQuote, SpeculativeTransactionControl, SpeculativeTransactionQuote,
};
use crate::backend::runtime::distributed::topology::original_source::control::{
    OriginalParallelControlInstallation, OriginalParallelControlOwner,
    OriginalParallelControlProjection,
};
use crate::backend::submission_recovery::addressable::{
    AddressableExecutionRow, SpeculativeAddressableSources, SpeculativeAddressableSpan,
};
use crate::backend::submission_recovery::{PreparedRecovery, Retention, Status};
use crate::backend::{OriginalCopyEnvironment, OriginalCopyEnvironmentError};
use crate::composition::mlx::replicated_text::AutoregressiveStateRoots;
use crate::composition::mlx::speculative::autoregressive::input_readout::{
    AutoregressiveIoPlan, IoPreparationError, PreparedAutoregressiveIo,
};
use safemlx::{
    OriginalBufferBudget, OriginalBufferCause, OriginalNativeControlError, PipelineCacheCause,
    PrefillFailureCause, PrefillRoots, PrefillRootsCause, PrefillRootsError, PrefillRootsRuntime,
    PreparedOriginalBufferBudget, PreparedPipelineCache, PreparedPipelineCachePlan,
    PreparedPrefillFailure, PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
    RetainedPrefillFailure, SubmissionGraphQuota, SubmissionGraphQuotaCause, SubmissionRecordQuota,
    SubmissionRecordQuotaCause, SubmissionScopeOwnerCause,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Domains(#[from] OriginalSpeculativeDomainError),
    #[error("speculative native invocation differs from its exact source or active role")]
    Source,
    #[error("speculative native invocation has an incomplete producer bound")]
    Unknown,
    #[error("speculative native invocation geometry overflow")]
    Overflow,
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
    Scope(#[from] SubmissionScopeOwnerCause),
    #[error(transparent)]
    Controls(#[from] OriginalNativeControlError),
    #[error(transparent)]
    Roots(#[from] PrefillRootsError),
    #[error(transparent)]
    RootConstruction(#[from] PrefillRootsCause),
    #[error(transparent)]
    Native(#[from] Exception),
    #[error(transparent)]
    Backend(#[from] Error),
    #[error(transparent)]
    Io(#[from] IoPreparationError),
    #[error("speculative native observation unavailable: {0:?}")]
    Observation(safemlx::ScopedSubmissionProgress),
    #[error("speculative native invocation failed or was blocked")]
    Completion,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    role: OriginalSpeculativeRole,
}
fn failure(cause: Cause, role: &OriginalSpeculativeRole) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(Failure {
            cause,
            role: role.clone(),
        })
        .with_operation("execute original speculative span"),
        false,
    )
}

#[derive(Debug, thiserror::Error)]
#[error(
    "original speculative role {ordinal} {invocation:?} ({spans} spans; physical/graph/record/controls={bytes:?}, live domains={pool_used:?}): {cause}"
)]
struct RoleAdmissionFailure {
    #[source]
    cause: eredu_runtime::working_memory::SpeculativeRequestError,
    invocation: AutoregressiveInvocation,
    ordinal: usize,
    spans: usize,
    bytes: [u64; 4],
    pool_used: Option<eredu_runtime::working_memory::MemoryLedgerSnapshot>,
}

fn reserve_role(
    sources: &AutoregressiveSourcePair,
    claim: AutoregressiveOccurrenceClaim<'_>,
    requirements: SpeculativeInvocationRequirements,
    spans: usize,
    funding: &HostMetadataFunding,
    capture: Option<eredu_runtime::working_memory::AutoregressiveCaptureHostPlan<'_>>,
) -> Result<
    (
        OriginalSpeculativeRole,
        Option<eredu_runtime::working_memory::FundedAutoregressiveCaptureBank>,
    ),
    Error,
> {
    funding
        .reserve_metadata(size_of::<(
            Option<eredu_runtime::working_memory::AutoregressiveCaptureHostPlan<'_>>,
            Option<eredu_runtime::working_memory::PreparedAutoregressiveCapture<'_>>,
            Option<eredu_runtime::working_memory::FundedAutoregressiveCaptureBank>,
            Result<
                (
                    OriginalSpeculativeRole,
                    Option<eredu_runtime::working_memory::PreparedAutoregressiveCapture<'_>>,
                ),
                eredu_runtime::working_memory::SpeculativeRequestError,
            >,
            Result<
                (
                    OriginalSpeculativeRole,
                    Option<eredu_runtime::working_memory::FundedAutoregressiveCaptureBank>,
                ),
                Error,
            >,
        )>())
        .map_err(Error::WorkspacePlanning)?;
    funding
        .reserve_metadata(size_of::<(
            RoleAdmissionFailure,
            Result<OriginalSpeculativeRole, eredu_runtime::working_memory::SpeculativeRequestError>,
            Result<OriginalSpeculativeRole, Error>,
            (
                AutoregressiveInvocation,
                usize,
                usize,
                [u64; 4],
                Option<eredu_runtime::working_memory::MemoryLedgerSnapshot>,
            ),
        )>())
        .map_err(Error::WorkspacePlanning)?;
    let bytes = requirements.allocation_bytes();
    let invocation = claim.invocation();
    let ordinal = claim.ordinal();
    let ledger = sources.numerical_sources().pool();
    funding
        .reserve_metadata(
            ledger
                .topology()
                .len()
                .checked_mul(size_of::<eredu_runtime::working_memory::MemoryDomainSnapshot>())
                .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let pool_used = ledger.snapshot().ok();
    let admitted = match capture {
        Some(capture) => sources
            .request()
            .reserve_role_with_capture(claim, requirements, capture)
            .map(|(role, bank)| (role, Some(bank))),
        None => sources
            .request()
            .reserve_role(claim, requirements)
            .map(|role| (role, None)),
    }
    .map_err(|cause| {
        retain_planning_error(
            RoleAdmissionFailure {
                cause,
                invocation,
                ordinal,
                spans,
                bytes,
                pool_used,
            },
            funding.clone(),
        )
    })?;
    let (role, bank) = admitted;
    let bank = bank
        .map(|bank| {
            bank.construct()
                .map_err(|cause| retain_planning_error(cause, funding.clone()))
        })
        .transpose()?;
    Ok((role, bank))
}

use crate::composition::mlx::speculative::original_domains::{
    OriginalSpeculativeDomainError, OriginalSpeculativeDomains as Domains,
    OriginalSpeculativeRoots as Roots, preparation_control_bytes, prepare_domains,
};
enum RetainedEquation {
    Decode {
        media_source: Option<eredu_runtime::working_memory::RegisteredPreparedWorkspaceStorage<()>>,
        paged: Option<ProjectedPagedSources>,
        layerwise: Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        report: InferenceWorkspaceReport,
        context: WorkspaceContext,
    },
    Prefill(prefill::ActiveSpeculativePrefill),
}
impl RetainedEquation {
    fn paged(&self) -> Option<&ProjectedPagedSources> {
        match self {
            Self::Decode { paged, .. } => paged.as_ref(),
            Self::Prefill(value) => value.paged(),
        }
    }
    fn context(&self) -> &WorkspaceContext {
        match self {
            Self::Decode { context, .. } => context,
            Self::Prefill(value) => value.metadata_context(),
        }
    }
    fn layerwise(
        &self,
    ) -> Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace> {
        match self {
            Self::Decode { layerwise, .. } => layerwise.as_ref(),
            Self::Prefill(value) => value.layerwise(),
        }
    }
}
struct Retained {
    model_control: Option<OriginalParallelControlOwner>,
    roots: Roots,
    parallel: Option<Rc<crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation>>,
    capture: Option<Rc<capture::Capture>>,
    domains: Domains,
    equation: RetainedEquation,
    paged_scope: Option<PagedScopeRetention>,
    funding: HostMetadataFunding,
    role: OriginalSpeculativeRole,
}
impl Retention for Retained {
    fn observe(&self, status: Status) {
        if let Some(paged) = &self.paged_scope {
            paged.observe(status);
        }
    }
}

struct ModelControl {
    projection: OriginalParallelControlProjection,
    installation: OriginalParallelControlInstallation,
    owner: OriginalParallelControlOwner,
    finished: Cell<bool>,
}
impl ModelControl {
    fn finish(&self, success: bool) -> Result<(), Error> {
        self.installation.close();
        if self.finished.replace(true) {
            return Err(Error::PrefillScopeUnavailable);
        }
        self.owner.finish_model(success)
    }
    fn close(&self) {
        self.installation.close();
    }
}
struct Active {
    model_control: Option<ModelControl>,
    transaction: Option<SpeculativeTransactionControl>,
    metadata: WorkspaceContext,
    checkpoint: RefCell<Option<MlxPredictionTargetState>>,
    capture: Option<Rc<capture::Capture>>,
    selected: RefCell<Option<Array>>,
    roots: Roots,
    parallel: Option<Rc<crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation>>,
    io: RefCell<Option<PreparedAutoregressiveIo>>,
    ingress: RefCell<TokenValidationIngress>,
    validations: RefCell<Option<TokenValidationScope>>,
    observer: OriginalScopeObserver,
    buffer: OriginalBufferBudget,
    completion: ResidentCompletionRecipe,
    stream: safemlx::StreamCopyPlan<()>,
    graph: RefCell<Option<safemlx::PreparedResidentGraph>>,
    graph_started: Cell<bool>,
    readout: Option<safemlx::ResidentGraphLayout>,
    publication_closed: Cell<bool>,
    readout_started: Cell<bool>,
    completed: Cell<bool>,
    role: OriginalSpeculativeRole,
    funding: HostMetadataFunding,
}
/// Clone-only view of the actual installed role. It is not a Scope constructor
/// and cannot change the source, bank population or observer it authenticates.
pub(crate) struct ActiveSpeculativeInvocation(Option<Rc<Active>>);
impl Clone for ActiveSpeculativeInvocation {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(self.inner())))
    }
}
impl Drop for ActiveSpeculativeInvocation {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl std::fmt::Debug for ActiveSpeculativeInvocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ActiveSpeculativeInvocation")
    }
}
impl ActiveSpeculativeInvocation {
    fn inner(&self) -> &Rc<Active> {
        self.0.as_ref().expect("live role view")
    }
    pub(crate) fn metadata_context(&self) -> WorkspaceContext {
        self.inner().metadata.clone()
    }
    pub(crate) fn role(&self) -> &OriginalSpeculativeRole {
        &self.inner().role
    }
    pub(crate) fn metadata_funding(&self) -> HostMetadataFunding {
        self.inner().funding.clone()
    }
    pub(crate) fn observer(&self) -> &OriginalScopeObserver {
        &self.inner().observer
    }
    pub(crate) fn budget(&self) -> &OriginalBufferBudget {
        &self.inner().buffer
    }
    fn authenticate(&self, stream: &Stream) -> Result<(), Cause> {
        if !self.inner().stream.matches_source(stream)
            || !self
                .observer()
                .same_scope(&OriginalScopeObserver::require_current()?)
        {
            return Err(Cause::Source);
        }
        Ok(())
    }
    pub(crate) fn with_observer<T>(
        &self,
        stream: &Stream,
        run: impl FnOnce(
            &mut dyn eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error>,
        ) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.authenticate(stream)
            .map_err(|cause| failure(cause, self.role()))?;
        match &self.inner().capture {
            Some(capture) => capture.run(self, stream, run),
            None => run(&mut eredu_runtime::NoopObserver),
        }
    }
    fn complete_prefill_roots(
        &self,
        output: Option<&Array>,
        state: &dyn AutoregressiveStateRoots,
        span: &eredu_runtime::working_memory::OriginalSpeculativePrefillSpan,
        stream: &Stream,
    ) -> Result<(), Error> {
        let chunk = span.chunk();
        if self.inner().capture.is_some()
            && self.role().invocation().pass()
                == eredu_runtime::speculative::autoregressive::AutoregressivePass::TargetPrefill
            && chunk.input.end == self.role().invocation().positions() as u64
            && chunk.output == eredu_core::OutputDemand::Sequence
        {
            let value = output.ok_or_else(|| failure(Cause::Source, self.role()))?;
            let width = chunk
                .input
                .end
                .checked_sub(chunk.input.start)
                .ok_or_else(|| failure(Cause::Overflow, self.role()))?;
            let last = i32::try_from(
                width
                    .checked_sub(1)
                    .ok_or_else(|| failure(Cause::Source, self.role()))?,
            )
            .map_err(|_| failure(Cause::Overflow, self.role()))?;
            self.begin_readout_construction()?;
            *self.inner().selected.borrow_mut() =
                Some(value.try_index_device((.., last, ..), stream)?);
        }
        let selected = self.inner().selected.borrow();
        self.complete_sequence_roots(output.into_iter().chain(selected.iter()), state, stream)
    }
    fn take_selected(&self) -> Option<Array> {
        self.inner().selected.borrow_mut().take()
    }
    pub(crate) fn take_checkpoint(&self) -> Result<MlxPredictionTargetState, Error> {
        if !self.observer().same_scope(
            &OriginalScopeObserver::require_current()
                .map_err(|cause| failure(cause.into(), self.role()))?,
        ) {
            return Err(failure(Cause::Source, self.role()));
        }
        self.inner()
            .checkpoint
            .try_borrow_mut()
            .map_err(|_| failure(Cause::Source, self.role()))?
            .take()
            .ok_or_else(|| failure(Cause::Source, self.role()))
    }
    pub(crate) fn take_io(&self) -> Result<PreparedAutoregressiveIo, Error> {
        if !self.observer().same_scope(
            &OriginalScopeObserver::require_current()
                .map_err(|e| failure(e.into(), self.role()))?,
        ) {
            return Err(failure(Cause::Source, self.role()));
        }
        self.inner()
            .io
            .try_borrow_mut()
            .map_err(|_| failure(Cause::Source, self.role()))?
            .take()
            .ok_or_else(|| failure(Cause::Source, self.role()))
    }
    pub(crate) fn begin_equation_construction(&self) -> Result<(), Error> {
        let result = (|| -> Result<(), Cause> {
            if !self
                .observer()
                .same_scope(&OriginalScopeObserver::require_current()?)
                || self.inner().graph_started.replace(true)
            {
                return Err(Cause::Source);
            }
            let validations = self
                .inner()
                .ingress
                .try_borrow_mut()
                .map_err(|_| Cause::Source)?
                .begin()?;
            *self
                .inner()
                .validations
                .try_borrow_mut()
                .map_err(|_| Cause::Source)? = Some(validations);
            let mut graph = safemlx::OperationEvent::prepare_resident_graph(
                self.inner().completion.graph,
                self.observer(),
            )?;
            // An equation without nested evaluations has no nested program
            // to install. The native constructor requires a positive attempt
            // population and deliberately rejects a fabricated zero program.
            if self.inner().completion.nested_completions != 0 {
                graph.configure_nested_completions(
                    &self
                        .inner()
                        .completion
                        .nested_traversal()
                        .ok_or(Cause::Unknown)?,
                    self.inner().completion.nested_completions,
                )?;
            }
            *self
                .inner()
                .graph
                .try_borrow_mut()
                .map_err(|_| Cause::Source)? = Some(graph);
            Ok(())
        })();
        result.map_err(|cause| failure(cause, self.role()))
    }
    /// Reopen only the actual readout source after publication closed this role's
    /// bank. The suffix and model still share one final completion and recovery.
    pub(crate) fn begin_readout_construction(&self) -> Result<(), Error> {
        self.inner().root_observer()?;
        let result = (|| -> Result<(), Cause> {
            let source = self.inner().readout.ok_or(Cause::Source)?;
            if self.inner().readout_started.replace(true) {
                return Err(Cause::Source);
            }
            let mut graph = self
                .inner()
                .graph
                .try_borrow_mut()
                .map_err(|_| Cause::Source)?;
            if self.inner().publication_closed.get() {
                if graph.is_some()
                    || !self
                        .inner()
                        .parallel
                        .as_ref()
                        .is_some_and(|parallel| parallel.has_publication())
                {
                    return Err(Cause::Source);
                }
                *graph = Some(safemlx::OperationEvent::prepare_resident_graph(
                    source,
                    self.observer(),
                )?);
            } else if graph.is_none() {
                return Err(Cause::Source);
            }
            Ok(())
        })();
        result.map_err(|cause| failure(cause, self.role()))
    }
    fn end_equation_construction(&self) -> Result<(), Cause> {
        if !self.inner().graph_started.get() {
            return Err(Cause::Source);
        }
        let graph = self
            .inner()
            .graph
            .try_borrow_mut()
            .map_err(|_| Cause::Source)?
            .take();
        drop(graph);
        Ok(())
    }
    // Called by the lexical installation guard on both return and unwind.
    // No external loan of this RefCell exists; no native callback runs while
    // its short begin/end borrow is held.
    fn close_construction(&self) {
        if let Some(control) = &self.inner().model_control {
            control.close();
        }
        drop(self.inner().graph.borrow_mut().take());
        drop(self.inner().validations.borrow_mut().take());
    }
    pub(crate) fn complete_sequence_roots<'a>(
        &self,
        output: impl IntoIterator<Item = &'a Array>,
        state: &dyn AutoregressiveStateRoots,
        stream: &Stream,
    ) -> Result<(), Error> {
        let result = (|| -> Result<(), Cause> {
            self.authenticate(stream)?;
            if self.inner().completed.get() {
                return Err(Cause::Source);
            }
            self.end_equation_construction()?;
            let mut roots = self
                .inner()
                .roots
                .0
                .try_borrow_mut()
                .map_err(|_| Cause::Source)?;
            for value in output {
                roots.append(value)?;
            }
            let mut appended = Ok(());
            state.visit_roots(&mut |value| {
                if appended.is_ok() {
                    appended = roots.append(value);
                }
            })?;
            appended?;
            if let Some(capture) = &self.inner().capture {
                let mut appended = Ok(());
                capture.visit_retained(&mut |value| {
                    if appended.is_ok() {
                        appended = roots.append(value);
                    }
                })?;
                appended?;
            }
            crate::backend::nn::tensor::append_active_token_validation_roots(&mut roots)?;
            roots.complete_current_scope_on_stream_prepared(
                stream,
                &self.inner().completion.traversal,
            )?;
            // The prepared completion already performs scoped submit, wait
            // and validation. Ordinary collector APIs reject original owners.
            crate::backend::nn::tensor::validate_active_original_token_validations(
                self.observer(),
            )?;
            self.inner().completed.set(true);
            Ok(())
        })();
        result.map_err(|cause| failure(cause, self.role()))
    }
}
impl
    crate::composition::mlx::session::bounded_capture::funded_model::CaptureExecution<
        OriginalSpeculativeRole,
    > for ActiveSpeculativeInvocation
{
    fn validate_equation_scope(&self, stream: &Stream) -> Result<(), Error> {
        self.authenticate(stream)
            .map_err(|cause| failure(cause, self.role()))
    }
    fn role(&self) -> &OriginalSpeculativeRole {
        self.role()
    }
    fn observer(&self) -> &OriginalScopeObserver {
        self.observer()
    }
    fn activate_partition(&self, owner: &crate::backend::runtime::distributed::topology::original_source::control::speculative::SpeculativeCaptureOwner)
    -> Result<crate::backend::runtime::distributed::topology::original_source::control::speculative::CaptureActivation, Error>{
        owner.activate(self.role(), self.budget(), self.observer())
    }
}

struct Installation<'a> {
    source: &'a AutoregressiveSourcePair,
    active: ActiveSpeculativeInvocation,
}
impl Drop for Installation<'_> {
    fn drop(&mut self) {
        self.active.close_construction();
        let mut slot = self.source.active.borrow_mut();
        if slot
            .as_ref()
            .is_some_and(|active| active.role().same_role(self.active.role()))
        {
            let removed = slot.take();
            drop(slot);
            drop(removed);
        }
    }
}
impl AutoregressiveSourcePair {
    pub(crate) fn active_invocation(&self) -> Result<ActiveSpeculativeInvocation, Error> {
        self.active
            .try_borrow()
            .ok()
            .and_then(|slot| slot.clone())
            .ok_or_else(|| retain_planning_error(Cause::Source, self.metadata_funding().clone()))
    }
}

struct Plan {
    callback_funding: Option<HostMetadataFunding>,
    transaction: Option<SpeculativeTransactionQuote>,
    model_control: Option<SpeculativeModelControlQuote>,
    completion: ResidentCompletionRecipe,
    readout: Option<safemlx::ResidentGraphLayout>,
    graph_capacity: usize,
    record_capacity: usize,
    physical_capacity: usize,
    pipeline: PreparedPipelineCachePlan,
    root_capacity: usize,
    graph_bytes: u64,
    record_bytes: u64,
    controls: u64,
}
impl Plan {
    fn inspect<T, F>(
        recipe: &AutoregressiveEquationRecipe,
        state: &MlxAutoregressiveState,
        ordinal: usize,
        environment: &OriginalCopyEnvironment<'_>,
        funding: &HostMetadataFunding,
        input: Option<safemlx::OriginalPromptInputFacts>,
        io_controls: usize,
        transport_backing: usize,
        transaction: Option<SpeculativeTransactionQuote>,
        model_control: Option<SpeculativeModelControlQuote>,
        readout: Option<safemlx::ResidentGraphLayout>,
    ) -> Result<Self, Cause> {
        let post_publication = recipe
            .records()
            .get(ordinal)
            .ok_or(Cause::Source)?
            .parallel()
            .is_some_and(|parallel| parallel.has_publication());
        let (completion, graph_capacity, record_capacity, kernels, query_controls) = recipe
            .equation_domains_with_readout(
                ordinal,
                input.map_or(0, |input| input.graph_bytes()),
                readout.filter(|_| post_publication),
            )
            .map_err(|cause| {
                Cause::Backend(cause.at_speculative_stage("AR equation native domains"))
            })?;
        if !environment.pool().same_ledger(&state.pool)
            || !safemlx::StreamCopyPlan::<()>::capture(environment.stream())
                .map_err(|_| Cause::Source)?
                .matches_source(&state.stream)
        {
            return Err(Cause::Source);
        }
        let runtime = environment.input_runtime()?;
        let row = recipe.records().get(ordinal).ok_or(Cause::Unknown)?;
        let callback_funding = if row.addressable().is_some() {
            let bytes =
                usize::try_from(addressable_call_controls::<T, F>().ok_or(Cause::Overflow)?)
                    .map_err(|_| Cause::Overflow)?;
            Some(
                crate::backend::submission_recovery::addressable::prepare_wrapper_funding(
                    bytes, funding,
                )?,
            )
        } else {
            None
        };
        let storage = row.mutable_storage().ok_or(Cause::Unknown)?;
        let population = OriginalBufferBudget::population_layout(
            &runtime,
            usize::try_from(storage.mutable_bytes()).map_err(|_| Cause::Overflow)?,
            storage.maximum_births(),
        )?;
        let trace = recipe.plan().records().get(ordinal).ok_or(Cause::Unknown)?;
        let physical_capacity = population
            .capacity()
            .max(
                usize::try_from(trace.new_tensor_allocation_bytes().ok_or(Cause::Unknown)?)
                    .map_err(|_| Cause::Overflow)?,
            )
            .checked_add(input.map_or(0, |input| input.mutable_bytes()))
            .and_then(|n| n.checked_add(transport_backing))
            .and_then(|n| n.checked_add(transaction.as_ref().map_or(0, |t| t.backing_bytes())))
            .and_then(|n| n.checked_add(model_control.as_ref().map_or(0, |t| t.backing_bytes())))
            .ok_or(Cause::Overflow)?;
        let graph = PreparedSubmissionGraphQuota::<OriginalSpeculativeBudgetCustody>::layout(
            graph_capacity,
        )?;
        let record = PreparedSubmissionRecordQuota::<OriginalSpeculativeBudgetCustody>::layout(
            record_capacity,
        )?;
        let pipeline = PreparedPipelineCachePlan::new(kernels);
        let root_capacity = completion.traversal.roots();
        let roots = PrefillRoots::layout(root_capacity)?;
        let parts = [
            size_of::<Self>(),
            size_of::<Domains>(),
            preparation_control_bytes().ok_or(Cause::Overflow)?,
            size_of::<Retained>(),
            paged_call_control_bytes().ok_or(Cause::Overflow)?,
            size_of::<Active>(),
            parallel::control_bytes().ok_or(Cause::Unknown)?,
            size_of::<Installation<'_>>(),
            size_of::<Result<T, Error>>(),
            size_of::<Result<T, Error>>(),
            size_of::<F>(),
            size_of::<
                fn(
                    F,
                    &mut Executable,
                    &mut MlxAutoregressiveState,
                    &ActiveSpeculativeInvocation,
                ) -> Result<T, Error>,
            >(),
            size_of::<T>(),
            size_of::<MlxSpeculativeCompletion>(),
            size_of::<Result<MlxSpeculativeCompletion, Exception>>(),
            size_of::<Cause>(),
            size_of::<safemlx::SubmissionRetirement>(),
            size_of::<Result<safemlx::SubmissionRetirement, Exception>>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<OriginalBufferBudget>(),
            size_of::<Failure>(),
            size_of::<Result<Domains, Cause>>(),
            size_of::<Option<PreparedAutoregressiveIo>>(),
            size_of::<Option<safemlx::PreparedResidentGraph>>(),
            size_of::<Option<safemlx::ResidentGraphLayout>>(),
            size_of::<(safemlx::ResidentGraphLayout, bool, Result<(), Cause>)>(),
            size_of::<std::cell::RefMut<'_, Option<safemlx::PreparedResidentGraph>>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<safemlx::PreparedResidentGraph, Exception>>(),
            size_of::<ActiveSpeculativeInvocation>(),
            size_of::<Option<prefill::ActiveSpeculativePrefill>>(),
            size_of::<(
                &Option<prefill::ActiveSpeculativePrefill>,
                &HostMetadataFunding,
                &mut dyn FnMut(&Array),
            )>(),
            size_of::<Option<&eredu_runtime::working_memory::OriginalSpeculativePrefillSpan>>(),
            size_of::<(
                &mut Executable,
                &mut MlxAutoregressiveState,
                &ActiveSpeculativeInvocation,
            )>(),
            size_of::<Result<ActiveSpeculativeInvocation, Error>>(),
            size_of::<Result<Plan, Cause>>(),
            size_of::<
                Result<
                    OriginalSpeculativeRole,
                    eredu_runtime::working_memory::SpeculativeRequestError,
                >,
            >(),
            rc_bytes::<Active>().ok_or(Cause::Overflow)?,
            rc_bytes::<RefCell<PrefillRoots>>().ok_or(Cause::Overflow)?,
            roots.host_bytes().ok_or(Cause::Overflow)?,
            population.control_bytes(),
            io_controls,
            transaction.as_ref().map_or(0, |t| t.metadata_bytes()),
            usize::try_from(TokenValidationIngress::speculative_span_control_bytes(
                recipe, ordinal,
            )?)
            .map_err(|_| Cause::Overflow)?,
            PreparedOriginalBufferBudget::<OriginalSpeculativeBudgetCustody>::layout(
                &runtime,
                physical_capacity,
            )?
            .total_owner_bytes()
            .ok_or(Cause::Overflow)?,
            PreparedPrefillFailure::<OriginalSpeculativeBudgetCustody>::layout()?
                .total_bytes()
                .ok_or(Cause::Overflow)?,
            pipeline
                .layout::<OriginalSpeculativeBudgetCustody>()?
                .required_bytes()
                .ok_or(Cause::Overflow)?,
            safemlx::OriginalNativeControlLayout::inspect()?.fixed_control_bytes,
            safemlx::original_scoped_evaluation_control_bytes().ok_or(Cause::Unknown)?,
            OriginalCopyEnvironment::control_bytes().ok_or(Cause::Unknown)?,
            completion.graph.control_bytes().ok_or(Cause::Unknown)?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()
                .and_then(|n| n.checked_mul(2))
                .ok_or(Cause::Unknown)?,
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Cause::Overflow)?;
        let controls = u64::try_from(controls)
            .map_err(|_| Cause::Overflow)?
            .checked_add(query_controls)
            .and_then(|n| {
                n.checked_add(PreparedRecovery::<
                    Retained,
                    OriginalSpeculativeBudgetCustody,
                >::control_bytes()?)
            })
            .and_then(|n| n.checked_add(trace.host_workspace_bytes()?))
            .ok_or(Cause::Unknown)?;
        Ok(Self {
            callback_funding,
            transaction,
            model_control,
            completion,
            readout,
            graph_capacity,
            record_capacity,
            physical_capacity,
            pipeline,
            root_capacity,
            graph_bytes: u64::try_from(graph.total_bytes().ok_or(Cause::Overflow)?)
                .map_err(|_| Cause::Overflow)?,
            record_bytes: u64::try_from(record.total_bytes().ok_or(Cause::Overflow)?)
                .map_err(|_| Cause::Overflow)?,
            controls,
        })
    }
    fn prepare(
        &self,
        environment: &OriginalCopyEnvironment<'_>,
        role: &OriginalSpeculativeRole,
        runtime: &PrefillRootsRuntime,
    ) -> Result<(Domains, Roots), Cause> {
        prepare_domains(
            environment,
            role.budget_custody(),
            runtime,
            self.graph_capacity,
            self.record_capacity,
            self.physical_capacity,
            &self.pipeline,
            self.root_capacity,
        )
        .map_err(Into::into)
    }
}
fn rc_bytes<T>() -> Option<usize> {
    std::alloc::Layout::new::<[Cell<usize>; 2]>()
        .extend(std::alloc::Layout::new::<T>())
        .ok()
        .map(|layout| layout.0.pad_to_align().size())
}

impl PreparedAutoregressiveInvocation<'_> {
    /// Compiles and accepts the actual role before its first native constructor.
    /// The existing shared callback is invoked once, under the source-bound
    /// group bank. Error/unwind paths retire the bank before Scope recovery.
    pub(crate) fn run<T, F>(
        self,
        environment: &OriginalCopyEnvironment<'_>,
        run: F,
    ) -> Result<T, Error>
    where
        F: FnOnce(&mut Executable, &mut MlxAutoregressiveState) -> Result<T, Error>,
    {
        self.run_observed(environment, None, run)
    }
    pub(crate) fn run_observed<T, F>(
        mut self,
        environment: &OriginalCopyEnvironment<'_>,
        receiver: Option<&mut dyn eredu_runtime::ActivationObserver<MlxTensor, Error>>,
        run: F,
    ) -> Result<T, Error>
    where
        F: FnOnce(&mut Executable, &mut MlxAutoregressiveState) -> Result<T, Error>,
    {
        if let Some(input) = self.prepared_prefill.take() {
            return self.run_prefill(environment, input, run);
        }
        let cold_funding = self.funding.clone();
        let cold = |cause| retain_planning_error(cause, cold_funding.clone());
        let runtime = environment
            .input_runtime()
            .map_err(|cause| cold(Cause::Environment(cause)))?;
        let width = self
            .model
            .inference_blueprint()
            .ok_or_else(|| cold(Cause::Source))?
            .architecture()
            .text_output_width();
        let width = width.ok_or_else(|| cold(Cause::Source))?;
        let mechanism = self
            .model
            .resident_workspace_mechanisms()
            .ok_or_else(|| cold(Cause::Source))?;
        let io = AutoregressiveIoPlan::inspect(
            &runtime,
            self.claim.invocation(),
            width,
            mechanism,
            &self.context,
        )
        .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        self.recipe
            .bind_io(io.readout_recipe(), io.input_facts())
            .map_err(|cause| cold(Cause::Backend(cause)))?;
        let plan = Plan::inspect::<T, F>(
            &self.recipe,
            self.state,
            0,
            environment,
            &self.funding,
            io.input_facts(),
            io.control_bytes()
                .checked_add(self.transaction_controls)
                .ok_or_else(|| cold(Cause::Overflow))?,
            self.capture
                .as_ref()
                .map(|capture| capture.partition_backing(0))
                .transpose()?
                .unwrap_or(0),
            self.transactions
                .get_mut(0)
                .ok_or_else(|| cold(Cause::Source))?
                .take(),
            self.model_controls
                .get_mut(0)
                .ok_or_else(|| cold(Cause::Source))?
                .take(),
            io.readout_recipe()
                .map(|readout| readout.completion().graph),
        )
        .map_err(cold)?;
        let roots_runtime = self
            .model
            .erased()
            .prefill_roots_runtime()
            .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        // The rollback copy completes under its own source-bound account before
        // the equation Scope or resident Graph bank can become active.
        let checkpoint = self.state.native.copy_original(
            environment,
            &roots_runtime,
            mechanism.ordinary(),
            &self.funding,
            self.sources.request().limits().clone(),
        )?;
        let mut addressable = SpeculativeAddressableSources::prepare_with_paged(
            self.recipe.records(),
            self.group_source_facts,
            self.paged
                .as_ref()
                .map(|source| source.host_source_facts())
                .transpose()
                .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?
                .flatten(),
            &self.funding,
        )?;
        let source_controls = addressable.as_ref().map_or(0, |source| source.controls());
        let capture_controls = self
            .capture
            .as_ref()
            .map(|capture| {
                capture
                    .frame_controls(
                        0,
                        self.report
                            .span_workspace_plan()
                            .records()
                            .first()
                            .ok_or_else(|| cold(Cause::Source))?
                            .span(),
                        self.claim.invocation(),
                        self.report.geometry(),
                    )
                    .ok_or_else(|| cold(Cause::Unknown))
            })
            .transpose()?
            .unwrap_or(0);
        let requirements = SpeculativeInvocationRequirements::new(
            self.recipe.plan(),
            u64::try_from(plan.physical_capacity).ok(),
            Some(plan.graph_bytes),
            Some(plan.record_bytes),
            plan.controls
                .checked_add(self.group_control_bytes)
                .and_then(|n| n.checked_add(source_controls))
                .and_then(|n| n.checked_add(capture_controls)),
            environment
                .buffer_placement()
                .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?,
        )
        .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        let requirements = if let Some(source) = addressable.as_mut() {
            requirements
                .with_funded_host_source_spans(source.take_facts()?)
                .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?
        } else {
            match self.group_source_facts {
                Some(facts) => requirements
                    .with_host_source_constructions(facts)
                    .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?,
                None => requirements,
            }
        };
        let addressable = addressable
            .map(|source| source.take(0))
            .transpose()?
            .flatten();
        let (role, mut capture_bank) = reserve_role(
            self.sources,
            self.claim,
            requirements,
            self.recipe.records().len(),
            &self.funding,
            self.capture
                .as_ref()
                .map(|capture| capture.bank())
                .transpose()?,
        )?;
        // Canonical source pins now own the projection. Temporary native
        // witnesses must retire before the manager installs mutable catalogs.
        drop(self.projected);
        if let Some(paged) = &self.paged {
            paged.bind_speculative(&role, self.report.span_workspace_plan())?;
        }
        let frame = match (
            self.capture.take(),
            capture_bank.as_mut(),
            receiver.as_ref(),
        ) {
            (None, None, None) => None,
            (Some(capture), Some(bank), Some(receiver)) => {
                let descriptor = receiver
                    .original_speculative_capture()
                    .ok_or_else(|| failure(Cause::Source, &role))?;
                let value = bank
                    .begin_decode()
                    .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
                let roots = capture
                    .populations
                    .first()
                    .ok_or_else(|| failure(Cause::Source, &role))?
                    .retained_roots;
                let edits = capture
                    .edits
                    .first()
                    .ok_or_else(|| failure(Cause::Source, &role))?
                    .borrow_mut()
                    .take();
                Some(capture::Frame::prepare(
                    value,
                    roots,
                    edits,
                    &role,
                    &self.funding,
                    descriptor,
                    self.sources,
                    capture
                        .partitions
                        .first()
                        .ok_or_else(|| failure(Cause::Source, &role))?
                        .borrow_mut()
                        .take(),
                    0,
                )?)
            }
            _ => return Err(failure(Cause::Source, &role)),
        };
        let equation = RetainedEquation::Decode {
            media_source: self.media_source,
            paged: self.paged,
            layerwise: self.layerwise,
            report: self.report,
            context: self.context,
        };
        let result = run_span(
            self.sources,
            self.model,
            self.state,
            &self.recipe,
            &plan,
            None,
            environment,
            &roots_runtime,
            role,
            checkpoint,
            Some(io),
            equation,
            cold_funding,
            addressable,
            frame.as_ref().map(|frame| Rc::clone(&frame.value)),
            run,
            |run, model, state, _| run(model, state),
        );
        let delivery = match (frame, receiver) {
            (Some(frame), Some(receiver)) => frame.deliver(receiver, result.is_ok()),
            (None, None) => Ok(()),
            _ => Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)),
        };
        match result {
            Err(cause) => Err(cause),
            Ok(value) => delivery.map(|()| value),
        }
    }
}

// One native construction/Scope/recovery worker for decode and actual prefill
// rows. Each caller has already admitted its exact source-bound role. A failed
// or dropped span never restores its claim or invokes a later callback.
fn run_span<T, F>(
    sources: &AutoregressiveSourcePair,
    model: &mut Executable,
    state: &mut MlxAutoregressiveState,
    recipe: &AutoregressiveEquationRecipe,
    plan: &Plan,
    span: Option<&eredu_runtime::working_memory::OriginalSpeculativePrefillSpan>,
    environment: &OriginalCopyEnvironment<'_>,
    roots_runtime: &PrefillRootsRuntime,
    role: OriginalSpeculativeRole,
    checkpoint: MlxPredictionTargetState,
    io: Option<AutoregressiveIoPlan>,
    equation: RetainedEquation,
    cold_funding: HostMetadataFunding,
    mut addressable: Option<SpeculativeAddressableSpan>,
    capture: Option<Rc<capture::Capture>>,
    run: F,
    invoke: fn(
        F,
        &mut Executable,
        &mut MlxAutoregressiveState,
        &ActiveSpeculativeInvocation,
    ) -> Result<T, Error>,
) -> Result<T, Error> {
    let (domains, roots) = plan
        .prepare(environment, &role, &roots_runtime)
        .map_err(|cause| failure(cause, &role))?;
    let io = io
        .map(|io| io.prepare(role.clone()))
        .transpose()
        .map_err(|cause| failure(cause.into(), &role))?;
    let ingress = match span {
        Some(span) => TokenValidationIngress::prepare_speculative_span(recipe, span),
        None => TokenValidationIngress::prepare_speculative(recipe, &role),
    }
    .map_err(|cause| failure(cause.into(), &role))?;
    let graph_quota = domains.graph.clone();
    let record_quota = domains.record.clone();
    let buffer = domains.buffer.clone();
    let completed_budget = buffer.clone();
    // Completion includes retained ingress roots as well as current cache
    // roots. Preserve that same inventory after Recovery retirement: a
    // later span may install an ingress alias into the decoder state.
    let completed_prefill = match &equation {
        RetainedEquation::Prefill(source) => Some(source.clone()),
        RetainedEquation::Decode { .. } => None,
    };
    let metadata = equation.context().clone();
    let parallel = recipe
        .native_recipe()
        .records()
        .get(span.map_or(0, |span| span.ordinal()))
        .ok_or_else(|| failure(Cause::Source, &role))?
        .parallel()
        .map(|parallel| parallel.try_clone_for_retention())
        .transpose()?
        .map(|parallel| {
            metadata
                .metadata_rc(parallel)
                .map_err(|cause| Error::Neural(cause.into()))
        })
        .transpose()?;
    let retained = Retained {
        model_control: None,
        roots: roots.clone(),
        parallel: parallel.clone(),
        capture: capture.clone(),
        domains,
        equation,
        paged_scope: None,
        funding: cold_funding.clone(),
        role: role.clone(),
    };
    let pending = PreparedRecovery::new(retained, role.budget_custody())
        .map_err(|error| failure(error.cause.into(), &role))?
        .with_graph_quota(Some(graph_quota))
        .with_record_quota(Some(record_quota));
    let mut recovery = pending
        .try_begin()
        .map_err(|error| failure(error.cause.into(), &role))?;
    recovery
        .configure_scope(|scope| -> Result<(), Cause> {
            scope
                .enable_scoped_observation()
                .map_err(Cause::Observation)?;
            scope.require_original_native_controls()?;
            // The roots bind their retained failure exactly once, before
            // enabling original work; a second binding correctly refuses.
            roots.0.borrow_mut().bind_scope(scope)?;
            scope.enable_original_native_controls()?;
            scope.bind_original_buffer_budget(&buffer)?;
            Ok(())
        })
        .map_err(|cause| failure(cause, &role))?;
    let observer =
        OriginalScopeObserver::require_current().map_err(|cause| failure(cause.into(), &role))?;
    let retirement = observer.clone();
    let result = (|| -> Result<T, Error> {
        // Installation closes the actual host construction bank before
        // group-bank teardown and before the outside Recovery can seal.
        let mut bank = None;
        let mut initialized = None;
        recovery.configure_scope_with_retention(|scope, retained| {
            let source = retained.equation.layerwise();
            let paged = retained.equation.paged();
            let first_span = span.is_none_or(|span| span.ordinal() == 0);
            let has_partition = paged.is_some()
                || addressable.is_some()
                || recipe
                    .records()
                    .iter()
                    .any(|row| row.addressable().is_some());
            let mut partition = |root| {
                let target = match addressable.as_mut() {
                    Some(source) => source.accept(root)?,
                    None => root,
                };
                initialized = addressable
                    .as_ref()
                    .map(|source| source.take_initialized())
                    .transpose()?
                    .flatten();
                if first_span {
                    if let Some(paged) = paged {
                        let bank = addressable
                            .as_ref()
                            .map(|source| source.take_paged())
                            .transpose()?
                            .flatten();
                        paged
                            .construct_host_program(bank, &role.budget_custody().into())
                            .map_err(|cause| {
                                Error::Neural(retained.equation.context().metadata_source(cause))
                            })?;
                    }
                }
                Ok(target)
            };
            bank = match span {
                Some(span) => model.erased().prepare_speculative_span_neural_bank(
                    source,
                    recipe,
                    span,
                    scope,
                    has_partition.then_some(&mut partition),
                    Some(
                        crate::backend::runtime::residency::manager::OriginalMaterializedLoan {
                            budget: &buffer,
                            funding: &retained.funding,
                        },
                    ),
                )?,
                None => model.erased().prepare_speculative_neural_bank(
                    source,
                    recipe,
                    role.clone(),
                    scope,
                    has_partition.then_some(&mut partition),
                    Some(
                        crate::backend::runtime::residency::manager::OriginalMaterializedLoan {
                            budget: &buffer,
                            funding: &retained.funding,
                        },
                    ),
                )?,
            };
            Ok::<_, Error>(())
        })?;
        let addressable = addressable
            .filter(|source| source.has_addressable())
            .map(|source| {
                source.activate(
                    bank.as_ref()
                        .ok_or_else(|| failure(Cause::Source, &role))?
                        .selected_residency_access()?,
                    buffer.clone(),
                    &observer,
                    None,
                    &cold_funding,
                )
            })
            .transpose()?;
        let transaction = plan
            .transaction
            .as_ref()
            .map(|quote| {
                quote.activate(sources.request(), &role, &buffer, &observer, &cold_funding)
            })
            .transpose()?;
        let model_control = plan
            .model_control
            .as_ref()
            .map(|quote| {
                let owner =
                    quote.activate(sources.request(), &role, &buffer, &observer, &cold_funding)?;
                let owner = match initialized.take() {
                    Some(bank) => owner.with_token_sources(Some(bank))?,
                    None => owner,
                };
                let (installation, projection) = owner.install()?;
                recovery.retention_mut().model_control = Some(owner.retained());
                Ok::<_, Error>(ModelControl {
                    projection,
                    installation,
                    owner,
                    finished: Cell::new(false),
                })
            })
            .transpose()?;
        if initialized.is_some() {
            return Err(failure(Cause::Source, &role));
        }
        let active = ActiveSpeculativeInvocation(Some(Rc::new(Active {
            model_control,
            transaction,
            metadata,
            checkpoint: RefCell::new(Some(checkpoint)),
            capture,
            selected: RefCell::new(None),
            roots,
            parallel,
            io: RefCell::new(io),
            ingress: RefCell::new(ingress),
            validations: RefCell::new(None),
            observer,
            buffer,
            completion: plan.completion,
            stream: safemlx::StreamCopyPlan::capture(&state.stream)
                .map_err(|_| failure(Cause::Source, &role))?,
            graph: RefCell::new(None),
            graph_started: Cell::new(false),
            readout: plan.readout,
            publication_closed: Cell::new(false),
            readout_started: Cell::new(false),
            completed: Cell::new(false),
            role: role.clone(),
            funding: cold_funding.clone(),
        })));
        let paged_scope = recovery.configure_scope_with_retention(|scope, retained| {
            retained
                .equation
                .paged()
                .map(|paged| paged.enter_speculative(&role, span, scope, active.transient_roots()))
                .transpose()
        })?;
        recovery.retention_mut().paged_scope = paged_scope;
        {
            let mut slot = sources
                .active
                .try_borrow_mut()
                .map_err(|_| failure(Cause::Source, &role))?;
            if slot.is_some() {
                return Err(failure(Cause::Source, &role));
            }
            *slot = Some(active.clone());
        }
        let installed = Installation {
            source: sources,
            active: active.clone(),
        };
        let output = match addressable.as_ref() {
            Some(row) => invoke_addressable(
                row,
                run,
                model,
                state,
                &active,
                invoke,
                plan.callback_funding
                    .as_ref()
                    .ok_or_else(|| failure(Cause::Source, &role))?,
            ),
            None => invoke(run, model, state, &active),
        }
        .map_err(|cause| failure(cause.into(), &role));
        drop(installed);
        drop(addressable);
        drop(bank.take());
        let output = output?;
        if !active.inner().completed.get() {
            return Err(failure(Cause::Completion, &role));
        }
        Ok(output)
    })();
    recovery.seal();
    // Successful roots must settle the existing recovery owner before
    // exact Record retirement. A single progress poll is not completion.
    // Failed construction still follows Recovery's retained Drop path.
    let output = result?;
    let status = recovery
        .finish()
        .map_err(|cause| failure(cause.into_error().into(), &role))?;
    if !status.settled || status.failed || status.blocked {
        return Err(failure(Cause::Completion, &role));
    }
    if !matches!(
        retirement
            .retire_completed_records()
            .map_err(|cause| failure(cause.into(), &role))?,
        safemlx::SubmissionRetirement::CompleteSnapshot
    ) {
        return Err(failure(Cause::Completion, &role));
    }
    // Publish source facts only after exact completion and Recovery/Record
    // retirement. The next checkpoint authenticates these role births
    // alongside existing registered roots in the shared copy worker.
    state
        .native
        .publish_completed_original_source(
            &completed_budget,
            &role,
            &cold_funding,
            environment.stream(),
            |visitor| match &completed_prefill {
                Some(source) => source.visit_retained_media_roots(visitor, &cold_funding),
                None => Ok(()),
            },
        )
        .map_err(|cause| failure(cause.into(), &role))?;
    Ok(output)
}

mod prefill;
pub(crate) use prefill::ActiveSpeculativePrefill;

struct AddressableCall<'a, T, F> {
    run: Option<F>,
    model: &'a mut Executable,
    state: &'a mut MlxAutoregressiveState,
    active: &'a ActiveSpeculativeInvocation,
    invoke: fn(
        F,
        &mut Executable,
        &mut MlxAutoregressiveState,
        &ActiveSpeculativeInvocation,
    ) -> Result<T, Error>,
    output: Option<Result<T, Error>>,
}
impl<T, F> AddressableCall<'_, T, F> {
    fn invoke(&mut self) -> Result<bool, Error> {
        let run = self
            .run
            .take()
            .ok_or_else(|| failure(Cause::Source, &self.active.inner().role))?;
        let output = (self.invoke)(run, self.model, self.state, self.active);
        let success = output.is_ok();
        self.output = Some(output);
        Ok(success)
    }
}
fn paged_call_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Option<PagedScopeRetention>>(),
        size_of::<Result<PagedScopeRetention, Error>>(),
        size_of::<Result<Option<PagedScopeRetention>, Error>>(),
        size_of::<Option<&ProjectedPagedSources>>(),
        size_of::<(
            &mut Retained,
            &OriginalSpeculativeRole,
            &ActiveSpeculativeInvocation,
        )>(),
        size_of::<(
            &mut Option<SpeculativeAddressableSpan>,
            Option<&ProjectedPagedSources>,
            bool,
            &OriginalSpeculativeRole,
            &Retained,
        )>(),
        size_of::<eredu_runtime::working_memory::OriginalHostSourceCustody>(),
        size_of::<Option<eredu_runtime::working_memory::OriginalHostSourceBank>>() * 2,
        size_of::<Result<Option<eredu_runtime::working_memory::OriginalHostSourceBank>, Error>>()
            * 2,
        size_of::<&mut Option<eredu_runtime::working_memory::OriginalHostSourceBank>>(),
        size_of::<Result<OriginalParallelControlOwner, Error>>(),
        size_of::<Result<(), crate::backend::runtime::cache::residency::CacheSourceFailure>>(),
        size_of::<bool>() * 2,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

fn addressable_call_controls<T, F>() -> Option<u64> {
    let frames = [
        size_of::<AddressableCall<'_, T, F>>(),
        size_of::<&mut AddressableCall<'_, T, F>>(),
        size_of::<(
            &AddressableExecutionRow,
            F,
            &mut Executable,
            &mut MlxAutoregressiveState,
            &ActiveSpeculativeInvocation,
            fn(
                F,
                &mut Executable,
                &mut MlxAutoregressiveState,
                &ActiveSpeculativeInvocation,
            ) -> Result<T, Error>,
            &HostMetadataFunding,
        )>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<bool, Error>>(),
        size_of::<Option<Result<T, Error>>>(),
        size_of::<Result<T, Error>>(),
        size_of::<bool>(),
        size_of::<Option<SpeculativeAddressableSpan>>(),
        size_of::<Option<AddressableExecutionRow>>(),
        size_of::<
            Option<crate::backend::runtime::execution::generic::SpeculativeSourcePartition<'_>>,
        >(),
        size_of::<&mut Option<SpeculativeAddressableSpan>>(),
    ];
    u64::try_from(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?,
    )
    .ok()
}
fn invoke_addressable<T, F>(
    row: &AddressableExecutionRow,
    run: F,
    model: &mut Executable,
    state: &mut MlxAutoregressiveState,
    active: &ActiveSpeculativeInvocation,
    invoke: fn(
        F,
        &mut Executable,
        &mut MlxAutoregressiveState,
        &ActiveSpeculativeInvocation,
    ) -> Result<T, Error>,
    funding: &HostMetadataFunding,
) -> Result<T, Error> {
    funding
        .reserve_metadata(
            usize::try_from(
                addressable_call_controls::<T, F>()
                    .ok_or_else(|| failure(Cause::Overflow, &active.inner().role))?,
            )
            .map_err(|_| failure(Cause::Overflow, &active.inner().role))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let mut call = AddressableCall {
        run: Some(run),
        model,
        state,
        active,
        invoke,
        output: None,
    };
    let activation = row.during(&mut || call.invoke());
    match call.output.take() {
        Some(Err(cause)) => Err(cause),
        Some(Ok(value)) => activation.map(|()| value),
        None => Err(activation
            .err()
            .unwrap_or_else(|| failure(Cause::Source, &active.inner().role))),
    }
}

impl ActiveSpeculativeInvocation {
    pub(crate) fn with_transaction_control<T, E, F>(
        &self,
        event: eredu_runtime::replicated_session::ParallelControlEvent,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where
        F: FnOnce(
            Option<(
                &crate::backend::runtime::distributed::Group,
                &HostMetadataFunding,
            )>,
        ) -> Result<T, E>,
    {
        self.authenticate(stream)
            .map_err(|cause| failure(cause, self.role()).into_backend_failure())?;
        match &self.inner().transaction {
            Some(control) => control.run(event, stream, run),
            None => Ok(run(None)),
        }
    }
    pub(crate) fn finish_transaction(&self, success: bool) -> Result<(), Error> {
        self.inner()
            .transaction
            .as_ref()
            .map_or(Ok(()), |control| control.finish_transaction(success))
    }
    pub(crate) fn finish_prediction_loan(&self, success: bool) -> Result<(), Error> {
        self.inner()
            .transaction
            .as_ref()
            .map_or(Ok(()), |control| control.finish_loan(success))
    }
}
