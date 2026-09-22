//! Actual early/later control scopes use the accepted request-wide native bank.
use super::*;
use crate::backend::{
    runtime::residency::storage::native_storage::BankOwner,
    submission_recovery::{PreparedRecovery, Retention, Status},
};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, InferenceRequest, OriginalOperationMetadataCustody,
    OriginalTextControlGuard, OriginalTextMetadataCustody,
};
use safemlx::{
    OriginalBufferBudget, PreparedPrefillFailure, PreparedSubmissionGraphQuota,
    PreparedSubmissionRecordQuota, RegisteredThreadRuntimeHousekeeping, RetainedPrefillFailure,
};

#[derive(Clone, Debug)]
pub(crate) struct Custody {
    source: RetainedCommunicationSource,
    raw: OriginalOperationMetadataCustody,
    funding: HostMetadataFunding,
    model_funding: Option<HostMetadataFunding>,
}
impl Custody {
    fn model_funding(&self) -> &HostMetadataFunding {
        self.model_funding.as_ref().unwrap_or(&self.funding)
    }
    fn for_model(&self) -> Self {
        let mut custody = self.clone();
        custody.funding = self.model_funding().clone();
        custody
    }
}
struct Owner {
    request: RequestOwner,
    native: NativeOwner,
    token_sources: RefCell<Option<eredu_runtime::working_memory::OriginalHostSourceBank>>,
    running: Cell<bool>,
    failed: Cell<bool>,
    custody: Custody,
}
pub(crate) struct OriginalParallelControlOwner(Option<Rc<Owner>>);
impl std::fmt::Debug for OriginalParallelControlOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalParallelControlOwner")
            .field("installed", &self.0.is_some())
            .finish_non_exhaustive()
    }
}

impl Drop for OriginalParallelControlOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
impl OriginalParallelControlOwner {
    fn owner(&self) -> &Owner {
        self.0.as_deref().expect("live request control owner")
    }
    pub(crate) fn new(
        source: OriginalParallelSource,
        bank: BankOwner,
        execution: &InferenceRequest,
        controls: OriginalTextControlGuard,
    ) -> Result<Self, Error> {
        Self::from_source(ControlSource::Model(source), bank, execution, controls)
    }
    pub(crate) fn new_control(
        source: OriginalParallelControlSource,
        bank: BankOwner,
        execution: &InferenceRequest,
        controls: OriginalTextControlGuard,
    ) -> Result<Self, Error> {
        Self::from_source(ControlSource::Control(source), bank, execution, controls)
    }
    fn from_source(
        source: ControlSource,
        bank: BankOwner,
        execution: &InferenceRequest,
        controls: OriginalTextControlGuard,
    ) -> Result<Self, Error> {
        reserve(
            source.funding(),
            &[
                size_of::<Self>(),
                size_of::<Owner>(),
                size_of::<ControlSource>(),
                size_of::<Result<Self, Error>>(),
                size_of::<OriginalParallelControlProjection>(),
                size_of::<Option<Rc<Owner>>>(),
                size_of::<Custody>(),
                size_of::<BankOwner>(),
                size_of::<InferenceRequest>(),
                Layout::new::<[usize; 2]>()
                    .extend(Layout::new::<Owner>())
                    .map_err(|_| overflow())?
                    .0
                    .pad_to_align()
                    .size(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let actual = source.communication_source()?;
        controls
            .validate_reservation(execution.memory_reservation())
            .map_err(|cause| {
                control_error(
                    ControlCause::WorkingMemory(cause),
                    actual.source(),
                    source.funding(),
                )
            })?;
        let custody = Custody {
            source: actual.source().clone(),
            raw: controls.metadata_custody().into(),
            funding: source.funding().clone(),
            model_funding: None,
        };
        {
            let bank = bank.try_borrow().map_err(|_| {
                control_error(ControlCause::Identity, &custody.source, &custody.funding)
            })?;
            bank.budget_for_controls(&controls).map_err(|cause| {
                control_error(
                    ControlCause::WorkingMemory(cause),
                    &custody.source,
                    &custody.funding,
                )
            })?;
        }
        drop(actual);
        let request = OriginalParallelControlRequest::from_source(source)?;
        Ok(Self(Some(Rc::new(Owner {
            request: RequestOwner::Text(request),
            native: NativeOwner::Text {
                bank,
                execution: execution.clone(),
                controls,
            },
            token_sources: RefCell::new(None),
            running: Cell::new(false),
            failed: Cell::new(false),
            custody,
        }))))
    }
    fn installation_control_bytes() -> Option<usize> {
        metadata_bytes(&[
            size_of::<OriginalParallelControlProjection>(),
            size_of::<OriginalParallelControlInstallation>(),
            size_of::<Activation>(),
            OriginalParallelControlProjection::retirement_control_bytes()?,
            size_of::<
                Result<
                    (
                        OriginalParallelControlInstallation,
                        OriginalParallelControlProjection,
                    ),
                    Error,
                >,
            >(),
            Layout::new::<[usize; 2]>()
                .extend(Layout::new::<Activation>())
                .ok()?
                .0
                .pad_to_align()
                .size(),
        ])
    }
    pub(crate) fn install(
        &self,
    ) -> Result<
        (
            OriginalParallelControlInstallation,
            OriginalParallelControlProjection,
        ),
        Error,
    > {
        reserve_bytes(
            self.owner().custody.model_funding(),
            Self::installation_control_bytes(),
        )?;
        let active = Rc::new(Activation {
            active: Cell::new(true),
            custody: self.owner().custody.clone(),
        });
        let projection = OriginalParallelControlProjection {
            owner: Rc::downgrade(self.0.as_ref().expect("live owner")),
            active: Some(active.clone()),
            model_roots: None,
            fallback: self.owner().request.fallback.retained(),
            custody: self.owner().custody.clone(),
        };
        Ok((
            OriginalParallelControlInstallation(Some(active)),
            projection,
        ))
    }
}
struct Activation {
    active: Cell<bool>,
    custody: Custody,
}
/// Owned by the actual outer submission, closed at callback exit even if native
/// recovery must continue retaining its own sources after that boundary.
pub(crate) struct OriginalParallelControlInstallation(Option<Rc<Activation>>);
impl OriginalParallelControlInstallation {
    pub(crate) fn close(&self) {
        if let Some(active) = &self.0 {
            active.active.set(false);
        }
    }
}
impl Drop for OriginalParallelControlInstallation {
    fn drop(&mut self) {
        self.close();
        if let Some(active) = self.0.take() {
            drop(Rc::into_inner(active));
        }
    }
}
/// A lexical installation carries no native object or strong owner backedge.
pub(crate) struct OriginalParallelControlProjection {
    // Weak model view is lent only for the shared forward. Request authority
    // and native roots remain separately owned by their actual recoveries.
    model_roots: Option<Rc<boundary::ModelBoundaryContext>>,
    owner: Weak<Owner>,
    active: Option<Rc<Activation>>,
    fallback: eredu_core::SharedBackendFailure,
    custody: Custody,
}
impl Drop for OriginalParallelControlProjection {
    fn drop(&mut self) {
        if let Some(active) = self.active.take() {
            drop(Rc::into_inner(active));
        }
    }
}
// Group clones copy only these already-retained aliases. Their existing exact
// retention-copy producer includes alias_control_bytes; no native owner escapes.
impl Clone for OriginalParallelControlProjection {
    fn clone(&self) -> Self {
        Self {
            model_roots: self.model_roots.clone(),
            owner: self.owner.clone(),
            active: self.active.clone(),
            fallback: self.fallback.retained(),
            custody: self.custody.clone(),
        }
    }
}
impl OriginalParallelControlProjection {
    /// The actual outer operation closes this shared lexical installation at
    /// callback exit. This fact cannot certify a native scope or retire an
    /// escaped projection; it only identifies a stale session slot.
    fn is_active(&self) -> bool {
        self.active
            .as_ref()
            .is_some_and(|active| active.active.get())
    }
    // The installation pays its future fixed cleanup frames before publishing
    // the slot. Cleanup performs only reads/moves and never asks for late funds.
    fn retirement_control_bytes() -> Option<usize> {
        type Prefill = crate::backend::submission_recovery::prefill::PrefillControlProjection;
        type Removed = (Option<Prefill>, Option<OriginalParallelControlProjection>);
        type Loan = std::cell::RefMut<'static, Option<OriginalParallelControlProjection>>;
        let parts = [
            size_of::<Loan>(),
            size_of::<Result<Loan, std::cell::BorrowMutError>>(),
            size_of::<Option<Self>>(),           // removed parallel local
            size_of::<Option<Self>>(),           // take_inactive return
            size_of::<Removed>(),                // closure return transport
            size_of::<Result<Removed, Error>>(), // inspection result
            size_of::<
                Result<
                    Result<Removed, Error>,
                    eredu_runtime::replicated_session::RuntimeInspectionBoundary,
                >,
            >(), // runtime inspection boundary transport
            size_of::<Result<Result<Removed, Error>, Error>>(), // mapped boundary transport
            size_of::<Removed>(),                // owner outside all cell/session loans
            size_of::<&mut Option<Self>>(),
            size_of::<Option<&Self>>(), // optional slot predicate input
            size_of::<&Self>(),
            size_of::<bool>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn take_inactive(slot: &mut Option<Self>) -> Option<Self> {
        if slot.as_ref().is_some_and(|view| !view.is_active()) {
            slot.take()
        } else {
            None
        }
    }
    pub(crate) fn alias_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Weak<Owner>>(),
            size_of::<Option<Rc<Activation>>>(),
            size_of::<eredu_core::SharedBackendFailure>(),
            size_of::<Custody>(),
            size_of::<&Self>(),
            crate::backend::submission_recovery::prefill::TransientRootsProjection::control_bytes(
            )?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn retained(&self) -> Result<Self, Error> {
        reserve(
            &self.custody.funding,
            &[
                Self::alias_control_bytes().ok_or_else(overflow)?,
                size_of::<Result<Self, Error>>(),
            ],
        )?;
        Ok(self.clone())
    }
    pub(crate) fn validate_model_source(
        &self,
        source: &OriginalParallelSource,
    ) -> Result<(), Error> {
        reserve(
            &self.custody.funding,
            &[
                size_of::<(&Self, &OriginalParallelSource)>(),
                size_of::<OriginalParallelControlOwner>(),
                size_of::<Result<(), Error>>(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let owner = self
            .upgrade()
            .map_err(|cause| Error::with_original_control_source(cause, false))?;
        if !owner
            .owner()
            .request
            .source
            .model()
            .is_some_and(|actual| actual.same_source(source))
            || !owner.owner().request.funding.same_account(source.funding())
        {
            return Err(control_error(
                ControlCause::Identity,
                &self.custody.source,
                &self.custody.funding,
            ));
        }
        Ok(())
    }
    /// A separate control context never changes an executor's optional neural
    /// parallel field. The shared runtime restores this paid owner on all exits.
    pub(crate) fn with_request_context<T, E, F>(
        &self,
        roots: Option<crate::backend::submission_recovery::prefill::TransientRootsProjection>,
        binding: Option<super::super::parallel::OriginalParallelBinding>,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&mut Option<Box<Group>>, &HostMetadataFunding)>) -> Result<T, E>,
    {
        reserve(
            &self.custody.funding,
            &[
                size_of::<F>(),
                size_of::<T>(),
                size_of::<E>(),
                size_of::<Result<T, E>>(),
                size_of::<Result<Result<T, E>, Error>>(),
                size_of::<Option<(&mut Option<Box<Group>>, &HostMetadataFunding)>>(),
                size_of::<Option<Box<Group>>>(),
                size_of::<Box<Group>>(),
                Layout::new::<Group>().size(),
                size_of::<OriginalParallelControlOwner>(),
                size_of::<Result<Group, std::collections::TryReserveError>>(),
                CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let owner = self
            .upgrade()
            .map_err(|cause| Error::with_original_control_source(cause, false))?;
        let source = owner.owner().request.source.communication_source()?;
        let inputs = owner
            .owner()
            .request
            .source
            .agreement_inputs()
            .ok_or_else(|| {
                control_error(
                    ControlCause::Identity,
                    &self.custody.source,
                    &self.custody.funding,
                )
            })?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(inputs.group(), CommunicationOperation::FailureAgreement)
            .map_err(|cause| {
                failure(
                    Cause::Rank(cause),
                    &self.custody.source,
                    &self.custody.funding,
                )
            })?;
        let group = source
            .group(selected.order())
            .ok_or_else(|| {
                control_error(
                    ControlCause::Identity,
                    &self.custody.source,
                    &self.custody.funding,
                )
            })?
            .0;
        if group.has_original_parallel()
            || group.has_original_control()
            || group.original_control_request().is_some()
        {
            return Err(control_error(
                ControlCause::Identity,
                &self.custody.source,
                &self.custody.funding,
            ));
        }
        reserve(
            &self.custody.funding,
            &[group.retention_copy_bytes().ok_or_else(overflow)?],
        )?;
        let mut group = group
            .try_copy_for_retention()
            .map_err(|_| failure(Cause::Resource, &self.custody.source, &self.custody.funding))?;
        let mut projection = self.retained()?;
        let loan = boundary::ModelBoundaryLoan::new(roots, binding, &self.custody)?;
        projection.model_roots = loan.context();
        group.bind_original_control_request(projection);
        let mut context = Some(Box::new(group));
        Ok(run(Some((&mut context, &self.custody.funding))))
    }
    fn context_control_bytes<T, E, F>() -> Option<usize> {
        metadata_bytes(&[
            size_of::<F>(),
            size_of::<T>(),
            size_of::<E>(),
            size_of::<Result<T, E>>(),
            size_of::<Result<Result<T, E>, Error>>(),
            size_of::<Option<(&Group, &HostMetadataFunding)>>(),
            size_of::<(&Self, &Group, &HostMetadataFunding)>(),
            size_of::<(&Self, &Group, F)>(),
            size_of::<Result<Result<T, E>, Error>>(),
            size_of::<OriginalParallelControlOwner>(),
            failure_control_bytes()?,
        ])
    }
    pub(crate) fn with_context<T, E, F>(
        &self,
        prepared: &Group,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        self.with_context_funded(prepared, &self.custody.funding, run)
    }
    pub(crate) fn with_model_context<T, E, F>(
        &self,
        prepared: &Group,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        self.with_context_funded(prepared, self.custody.model_funding(), run)
    }
    fn with_context_funded<T, E, F>(
        &self,
        prepared: &Group,
        funding: &HostMetadataFunding,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        reserve_bytes(funding, Self::context_control_bytes::<T, E, F>())?;
        let owner = self
            .upgrade()
            .map_err(|cause| Error::with_original_control_source(cause, false))?;
        if prepared.has_original_parallel() {
            let source = owner.owner().request.source.model().ok_or_else(|| {
                control_error(ControlCause::Identity, &self.custody.source, funding)
            })?;
            prepared.validate_control_model_source_with_funding(source, funding)?;
        } else {
            let actual = owner
                .owner()
                .request
                .source
                .communication_source_funded(funding)?;
            let inputs = owner
                .owner()
                .request
                .source
                .agreement_inputs()
                .ok_or_else(|| {
                    control_error(ControlCause::Identity, &self.custody.source, funding)
                })?;
            reserve(
                funding,
                &[CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?],
            )?;
            let selected = actual
                .source()
                .manifest()
                .select_group_operation(inputs.group(), CommunicationOperation::FailureAgreement)
                .map_err(|cause| failure(Cause::Rank(cause), &self.custody.source, funding))?;
            if prepared.has_original_control() || !actual.matches_group(selected.order(), prepared)
            {
                return Err(control_error(
                    ControlCause::Identity,
                    &self.custody.source,
                    funding,
                ));
            }
        }
        Ok(run(Some((prepared, funding))))
    }
    /// Source/account loan only: authenticate the explicit model/control group
    /// and the actual stored route before neutral framing allocates anything.
    pub(crate) fn prepare_boundary_source(
        &self,
        prepared: &Group,
        route: &crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
    ) -> Result<eredu_runtime::PreparedBoundarySource, Error> {
        self.with_context(prepared, |bound| {
            if bound.is_none() {
                return Err(control_error(
                    ControlCause::Identity,
                    &self.custody.source,
                    &self.custody.funding,
                ));
            }
            reserve(
                &self.custody.funding,
                &[
                    size_of::<eredu_runtime::PreparedBoundarySource>(),
                    size_of::<Result<eredu_runtime::PreparedBoundarySource, Error>>(),
                    size_of::<OriginalParallelControlOwner>(),
                    size_of::<Option<usize>>(),
                    failure_control_bytes().ok_or_else(overflow)?,
                ],
            )?;
            let owner = self
                .upgrade()
                .map_err(|cause| Error::with_original_control_source(cause, false))?;
            let actual = owner.owner().request.source.communication_source()?;
            let order = actual
                .source()
                .manifest()
                .routes()
                .iter()
                .position(|entry| entry.id() == route.descriptor().id())
                .ok_or_else(|| {
                    control_error(
                        ControlCause::Identity,
                        &self.custody.source,
                        &self.custody.funding,
                    )
                })?;
            if !actual.matches_route(order, route) {
                return Err(control_error(
                    ControlCause::Identity,
                    &self.custody.source,
                    &self.custody.funding,
                ));
            }
            actual
                .source()
                .prepare_boundary_source(route.descriptor().id(), &self.custody.funding)
                .map_err(|cause| {
                    failure(
                        Cause::BoundaryFrame(cause),
                        &self.custody.source,
                        &self.custody.funding,
                    )
                })
        })?
    }
    /// Native header sources borrow this exact request's already initialized
    /// allocator. The lexical request/table loan survives every partial copy.
    pub(crate) fn prepare_boundary_headers(
        &self,
        prepared: &Group,
        frames: eredu_runtime::PreparedBoundaryFrames<crate::MlxTensor>,
        route: &crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
        observer: &OriginalScopeObserver,
    ) -> Result<super::super::OriginalBoundaryHeaders, Error> {
        self.with_context(prepared,|bound|{
            if bound.is_none(){return Err(control_error(ControlCause::Identity,&self.custody.source,&self.custody.funding));}
            reserve(&self.custody.funding,&[size_of::<OriginalParallelControlOwner>(),
                size_of::<Result<super::super::OriginalBoundaryHeaders,Error>>(),
                size_of::<Option<&agreement::OriginalAgreementInputs>>(),
                size_of::<(&Self,&Group,&crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
                    &OriginalScopeObserver)>(),failure_control_bytes().ok_or_else(overflow)?])?;
            let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
            let source=owner.owner().request.source.communication_source()?;
            let inputs=owner.owner().request.source.agreement_inputs()
                .ok_or_else(||control_error(ControlCause::Identity,&self.custody.source,&self.custody.funding))?;
            source.prepare_boundary_headers(frames,route,inputs.runtime(),observer)
        })?
    }
    /// Internal PP boundaries use the same request cursor and role producer as
    /// outer phases. This projection is supplied by the exact model Group loan.
    fn group_control_bytes<T, E, F>() -> Option<usize> {
        metadata_bytes(&[
            size_of::<F>(),
            size_of::<T>(),
            size_of::<E>(),
            size_of::<Result<T, E>>(),
            size_of::<Result<Result<T, E>, Error>>(),
            size_of::<Result<Result<Result<T, E>, Error>, eredu_core::BackendFailure>>(),
            size_of::<(ParallelControlEvent, &Group, &HostMetadataFunding, &Stream)>(),
            size_of::<Custody>(),
            size_of::<(&Self, &Custody)>(),
            failure_control_bytes()?,
        ])
    }
    pub(crate) fn with_group<T, E, F>(
        &self,
        event: ParallelControlEvent,
        group: &Group,
        funding: &HostMetadataFunding,
        executor: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<&Group>) -> Result<T, E>,
    {
        if !funding.same_account(self.custody.model_funding()) {
            return Err(control_error(
                ControlCause::Identity,
                &self.custody.source,
                &self.custody.funding,
            ));
        }
        reserve_bytes(funding, Self::group_control_bytes::<T, E, F>())?;
        self.run_group_funded(
            &self.custody.for_model(),
            event,
            Some(group),
            executor,
            model_callbacks::group_callback(self, event, group, funding, executor, run),
        )
        .map_err(|cause| Error::with_original_control_source(cause, false))?
    }
    fn upgrade(&self) -> Result<OriginalParallelControlOwner, eredu_core::BackendFailure> {
        if !self
            .active
            .as_ref()
            .is_some_and(|active| active.active.get())
        {
            return Err(self.fallback.retained().into_failure());
        }
        self.owner
            .upgrade()
            .map(|owner| OriginalParallelControlOwner(Some(owner)))
            .ok_or_else(|| self.fallback.retained().into_failure())
    }
    pub(crate) fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), Error> {
        reserve(
            &self.custody.funding,
            &[
                size_of::<&Self>(),
                size_of::<OriginalParallelControlOwner>(),
                size_of::<Result<(), Error>>(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let owner = self
            .upgrade()
            .map_err(|cause| Error::with_original_control_source(cause, false))?;
        owner
            .owner()
            .native
            .validate_execution(execution)
            .map_err(|cause| {
                control_error(
                    ControlCause::WorkingMemory(cause),
                    &self.custody.source,
                    &self.custody.funding,
                )
            })
    }
    pub(crate) fn run<T, E, F>(
        &self,
        event: ParallelControlEvent,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        self.run_group(event, None, stream, run)
    }
    fn run_group<T, E, F>(
        &self,
        event: ParallelControlEvent,
        target: Option<&Group>,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        self.run_group_funded(&self.custody, event, target, stream, run)
    }
    fn run_group_funded<T, E, F>(
        &self,
        custody: &Custody,
        event: ParallelControlEvent,
        target: Option<&Group>,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        let owner = self.upgrade()?;
        // The claim is spent before reentrancy, source or role admission can fail.
        let invocation =
            match owner
                .owner()
                .request
                .prepare_with_funding(event, target, &custody.funding)
            {
                Ok(value) => value,
                Err(cause) => {
                    owner.owner().failed.set(true);
                    return Err(cause.into_backend_failure());
                }
            };
        if let Err(cause) = owner
            .owner()
            .native
            .claim_model(&invocation, &custody.funding)
        {
            owner.owner().failed.set(true);
            return Err(cause.into_backend_failure());
        }
        if owner.owner().failed.get() || owner.owner().running.replace(true) {
            owner.owner().failed.set(true);
            return Err(self.fallback.retained().into_failure());
        }
        let _running = Running {
            running: &owner.owner().running,
            failed: &owner.owner().failed,
        };
        let result = run_role(
            invocation,
            &owner.owner().native,
            custody,
            stream,
            model_callbacks::option_callback(run),
        );
        if match &result {
            Err(_) => true,
            Ok(value) => value.is_err(),
        } {
            owner.owner().failed.set(true);
        }
        result
    }
}
struct Running<'a> {
    running: &'a Cell<bool>,
    failed: &'a Cell<bool>,
}
impl Drop for Running<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.failed.set(true);
        }
        self.running.set(false);
    }
}

fn run_role<T, E, F>(
    invocation: OriginalParallelControlInvocation,
    native: &NativeOwner,
    custody: &Custody,
    stream: &Stream,
    run: F,
) -> Result<Result<T, E>, eredu_core::BackendFailure>
where
    F: FnOnce(&Group, &HostMetadataFunding) -> Result<T, E>,
{
    let capacity = invocation.capacity();
    run_native_role(
        invocation,
        capacity,
        native,
        custody,
        model_callbacks::native_callback(stream, run),
    )
}

// A retained protocol invocation supplies its actual source-derived capacities.
// Both agreement and sampling keep the identical Scope/Recovery worker below.
fn run_native_role<I, T, E, F>(
    invocation: I,
    capacity: AgreementCapacity,
    native: &NativeOwner,
    custody: &Custody,
    run: F,
) -> Result<Result<T, E>, eredu_core::BackendFailure>
where
    I: 'static,
    F: FnOnce(&I, &OriginalScopeObserver) -> Result<Result<T, E>, Error>,
{
    reserve(
        &custody.funding,
        &[
            size_of::<I>(),
            size_of::<F>(),
            size_of::<AgreementCapacity>(),
            size_of::<(&NativeOwner, &Custody)>(),
            size_of::<Result<Result<T, E>, eredu_core::BackendFailure>>(),
        ],
    )
    .map_err(Error::into_backend_failure)?;
    run_native_role_with_pipeline(invocation, capacity, None, native, custody, run)
}
fn run_native_role_with_pipeline<I, T, E, F>(
    invocation: I,
    capacity: AgreementCapacity,
    pipeline: Option<safemlx::PreparedPipelineCachePlan>,
    native: &NativeOwner,
    custody: &Custody,
    run: F,
) -> Result<Result<T, E>, eredu_core::BackendFailure>
where
    I: 'static,
    F: FnOnce(&I, &OriginalScopeObserver) -> Result<Result<T, E>, Error>,
{
    let timeout = custody
        .source
        .manifest()
        .completion_policy()
        .ok_or_else(|| overflow().into_backend_failure())?
        .timeout();
    let capacity = crate::backend::submission_recovery::native_role::NativeRoleCapacity {
        graph: capacity.graph,
        records: capacity.records,
        backing: capacity.backing,
    };
    match native {
        NativeOwner::Text { bank, controls, .. } => {
            crate::backend::submission_recovery::native_role::run_with_context(
                invocation,
                capacity,
                pipeline,
                bank,
                controls,
                custody,
                &custody.funding,
                Some(timeout),
                model_callbacks::observer_callback(run),
            )
        }
        NativeOwner::Speculative {
            budget, observer, ..
        } => crate::backend::submission_recovery::native_role::run_with_prepared_budget(
            invocation,
            capacity,
            pipeline,
            budget,
            observer,
            custody,
            &custody.funding,
            Some(timeout),
            model_callbacks::observer_callback(run),
        ),
    }
}

mod model_callbacks;
mod speculative;
pub(crate) use speculative::SpeculativeModelRole;
use speculative::{NativeOwner, RequestOwner};

mod sampling;
pub(crate) use sampling::{BoundSamplingSource, OriginalSamplingSource};

mod boundary;

mod frame;

mod logical;

mod capture;
pub(crate) use capture::{CaptureSourceCustody, CaptureSourceOwner};
pub(crate) use capture::{CaptureTransportBinding, GatherRequirements, OriginalCaptureTransport};

#[path = "role/peer_counts.rs"]
mod peer_counts;
#[path = "role/provider.rs"]
mod provider;
mod variable;

#[path = "role/expert_input.rs"]
mod expert_input;

#[path = "role/expert_region.rs"]
mod expert_region;

#[path = "role/expert_movement.rs"]
mod expert_movement;
pub(crate) use expert_movement::OriginalExpertMovementSource;

#[cfg(test)]
mod retirement_tests;
