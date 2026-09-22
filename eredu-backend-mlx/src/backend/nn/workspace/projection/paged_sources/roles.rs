//! Exact accepted text or speculative role retains canonical page pins.
use super::*;
use crate::backend::{
    error::Error as NativeError,
    runtime::{
        cache::residency::{CacheSourceError, CacheSourceFailure},
        execution::generic::OriginalOperationRegistration,
    },
    submission_recovery::Status,
};
use eredu_runtime::{
    prefill::{PrefillControlRole, PrefillSpanControlPhase},
    working_memory::{
        InferenceRequest, InferenceSpanWorkspacePlan, InferenceWorkspaceSpan,
        OriginalHostSourceCustody, OriginalSpeculativePrefillSpan, OriginalSpeculativeRole,
        OriginalTextControlGuard, OriginalTextPrefillScopeSet, WorkingMemoryError,
    },
};
use safemlx::{OriginalScopeObserver, SubmissionScope};
use std::{cell::Cell, mem::size_of};

pub(super) struct RoleState {
    binding: Option<Binding>,
    pub(super) active: Option<Active>,
    next: usize,
    pub(super) failed: bool,
}
impl RoleState {
    pub(super) fn host_source_custody(
        &self,
        observer: &OriginalScopeObserver,
        ordinal: usize,
    ) -> Result<eredu_runtime::working_memory::OriginalHostSourceCustody, CacheSourceError> {
        let active = self.active.as_ref().ok_or(CacheSourceError::Identity)?;
        let binding = self.binding.as_ref().ok_or(CacheSourceError::Identity)?;
        if self.failed || active.ordinal != ordinal || !active.observer.same_scope(observer) {
            return Err(CacheSourceError::Identity);
        }
        Ok(binding.custody())
    }
    pub(super) fn new() -> Self {
        Self {
            binding: None,
            active: None,
            next: 0,
            failed: false,
        }
    }
}
enum Binding {
    Text {
        request: InferenceRequest,
        registration: OriginalOperationRegistration,
        controls: OriginalTextControlGuard,
    },
    Speculative(OriginalSpeculativeRole),
}
impl Binding {
    fn custody(&self) -> OriginalHostSourceCustody {
        match self {
            Self::Text { controls, .. } => controls.clone().into(),
            Self::Speculative(role) => role.budget_custody().into(),
        }
    }
}
pub(super) struct Active {
    pub(super) roots: crate::backend::submission_recovery::prefill::TransientRootsProjection,
    pub(super) append_active: bool,
    pub(super) observer: OriginalScopeObserver,
    pub(super) ordinal: usize,
}
fn identity() -> NativeError {
    NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn busy() -> NativeError {
    NativeError::PrefillScopeReentrant
}

impl ProjectedPagedSources {
    /// Called only with the same consumed prefill set and accepted native recipe.
    /// Catalog storage replacement is metadata-only; this does not enter a role.
    pub(crate) fn bind_request(
        &self,
        original: &OriginalTextPrefillScopeSet,
        request: &InferenceRequest,
        plan: &InferenceSpanWorkspacePlan,
        registration: OriginalOperationRegistration,
        controls: OriginalTextControlGuard,
    ) -> Result<(), NativeError> {
        original
            .validate_request(request)
            .map_err(NativeError::PrefillControl)?;
        controls
            .validate_reservation(request.memory_reservation())
            .map_err(NativeError::PrefillControl)?;
        if request.geometry() != plan.geometry() || !self.catalogs_match(plan) {
            return Err(identity());
        }
        self.bind(Binding::Text {
            request: request.clone(),
            registration,
            controls,
        })
    }
    /// The admitted invocation supplies its own account and exact source plan;
    /// no text request or independently fitted capacity can substitute for it.
    pub(crate) fn bind_speculative(
        &self,
        role: &OriginalSpeculativeRole,
        plan: &InferenceSpanWorkspacePlan,
    ) -> Result<(), NativeError> {
        role.validate_plan(plan)
            .map_err(NativeError::PrefillControl)?;
        let custody: OriginalHostSourceCustody = role.budget_custody().into();
        custody
            .validate_account(None)
            .map_err(NativeError::PrefillControl)?;
        if !self.catalogs_match(plan) {
            return Err(identity());
        }
        self.bind(Binding::Speculative(role.clone()))
    }
    fn bind(&self, binding: Binding) -> Result<(), NativeError> {
        {
            let state = self.inner.roles.try_borrow().map_err(|_| busy())?;
            if state.failed || state.binding.is_some() || state.active.is_some() {
                return Err(identity());
            }
        }
        // Recheck exact pinned blocks/tail before replacing metadata catalogs.
        // The lexical manager loan ends before any moved owner can retire.
        for source in &self.inner.sources {
            source
                .manager()
                .with_source_loan(source.selection(), &self.inner.context, |loan| {
                    source
                        .validate_catalog(&loan)
                        .map_err(|cause| CacheSourceFailure::source(cause, &self.inner.context))
                })
                .map_err(|cause| NativeError::Neural(self.inner.context.metadata_source(cause)))?;
        }
        let result = self.install_catalogs();
        if let Err(cause) = result {
            self.inner.roles.borrow_mut().failed = true;
            return Err(NativeError::Neural(
                self.inner.context.metadata_source(cause),
            ));
        }
        let mut state = self.inner.roles.try_borrow_mut().map_err(|_| busy())?;
        if state.failed || state.binding.is_some() || state.active.is_some() {
            return Err(identity());
        }
        state.binding = Some(binding);
        Ok(())
    }
    fn install_catalogs(&self) -> Result<(), CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, &self.inner.context);
        let mut catalogs = self
            .inner
            .catalogs
            .try_borrow_mut()
            .map_err(|_| fail(CacheSourceError::Busy))?;
        let catalogs = catalogs
            .as_mut()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        for index in 0..catalogs.entries.len() {
            let prepared = catalogs.entries[index]
                .take()
                .ok_or_else(|| fail(CacheSourceError::Identity))?;
            match prepared.install() {
                Ok(installed) => catalogs.installed.push(installed),
                Err(failure) => {
                    let (cause, prepared) = failure.into_parts();
                    catalogs.entries[index] = Some(prepared);
                    return Err(cause);
                }
            }
        }
        Ok(())
    }
    pub(crate) fn enter_prefill(
        &self,
        request: &InferenceRequest,
        role: PrefillControlRole,
        scope: &SubmissionScope,
        roots: crate::backend::submission_recovery::prefill::TransientRootsProjection,
    ) -> Result<PagedScopeRetention, NativeError> {
        self.enter(request, scope, roots, |span| match (role, span) {
            (
                PrefillControlRole::Span {
                    phase: PrefillSpanControlPhase::InputTransaction,
                    input_start,
                    input_end,
                    position,
                    output,
                },
                InferenceWorkspaceSpan::Prefill(chunk),
            ) => {
                input_start == chunk.input.start
                    && input_end == chunk.input.end
                    && position == chunk.position
                    && output == chunk.output
            }
            _ => false,
        })
    }
    pub(crate) fn enter_decode(
        &self,
        request: &InferenceRequest,
        attempt: u64,
        scope: &SubmissionScope,
        roots: crate::backend::submission_recovery::prefill::TransientRootsProjection,
    ) -> Result<PagedScopeRetention, NativeError> {
        self.enter(request, scope, roots, |span| {
            matches!(span,
            InferenceWorkspaceSpan::Decode { index, .. } if index.checked_add(1) == Some(attempt))
        })
    }
    fn enter(
        &self,
        request: &InferenceRequest,
        scope: &SubmissionScope,
        roots: crate::backend::submission_recovery::prefill::TransientRootsProjection,
        matches: impl FnOnce(&InferenceWorkspaceSpan) -> bool,
    ) -> Result<PagedScopeRetention, NativeError> {
        let (registration, ordinal) = {
            let state = self.inner.roles.try_borrow().map_err(|_| busy())?;
            if state.failed || state.active.is_some() {
                return Err(identity());
            }
            let Some(Binding::Text {
                request: bound_request,
                registration,
                controls,
            }) = state.binding.as_ref()
            else {
                return Err(identity());
            };
            bound_request
                .validate_same_request(request)
                .map_err(NativeError::PrefillControl)?;
            controls
                .validate_reservation(request.memory_reservation())
                .map_err(NativeError::PrefillControl)?;
            let catalogs = self.inner.catalogs.try_borrow().map_err(|_| busy())?;
            let span = catalogs
                .as_ref()
                .ok_or_else(identity)?
                .plan
                .generation_records()
                .ok_or_else(identity)?
                .nth(state.next)
                .ok_or_else(identity)?;
            if !matches(span.span()) {
                return Err(identity());
            }
            (registration.clone(), state.next)
        };
        let observer = registration.authenticate_scope(request, scope, &roots)?;
        self.activate(ordinal, observer, roots)
    }
    pub(crate) fn enter_speculative(
        &self,
        role: &OriginalSpeculativeRole,
        span: Option<&OriginalSpeculativePrefillSpan>,
        scope: &SubmissionScope,
        roots: crate::backend::submission_recovery::prefill::TransientRootsProjection,
    ) -> Result<PagedScopeRetention, NativeError> {
        let custody: OriginalHostSourceCustody = role.budget_custody().into();
        custody
            .validate_account(None)
            .map_err(NativeError::PrefillControl)?;
        let ordinal = {
            let state = self.inner.roles.try_borrow().map_err(|_| busy())?;
            if state.failed || state.active.is_some() {
                return Err(identity());
            }
            let Some(Binding::Speculative(bound)) = state.binding.as_ref() else {
                return Err(identity());
            };
            if !bound.same_role(role) {
                return Err(identity());
            }
            let catalogs = self.inner.catalogs.try_borrow().map_err(|_| busy())?;
            let plan = &catalogs.as_ref().ok_or_else(identity)?.plan;
            role.validate_plan(plan)
                .map_err(NativeError::PrefillControl)?;
            let record = plan
                .generation_records()
                .ok_or_else(identity)?
                .nth(state.next)
                .ok_or_else(identity)?;
            match (role.invocation().execution_pass(), span) {
                (eredu_runtime::ExpertPass::Prefill, Some(span)) => {
                    span.validate_plan(plan)
                        .map_err(NativeError::PrefillControl)?;
                    if !span.role().same_role(role)
                        || span.ordinal() != state.next
                        || span.record().span() != record.span()
                    {
                        return Err(identity());
                    }
                }
                (eredu_runtime::ExpertPass::Decode, None) => {
                    // An independent decode invocation has one equation record,
                    // represented by its exact input chunk rather than a text
                    // generation Decode row. Its native bank is one-use too.
                    if state.next != 0 || plan.records().len() != 1 {
                        return Err(identity());
                    }
                }
                _ => return Err(identity()),
            }
            state.next
        };
        let observer = roots.authenticate_source(scope, &custody)?;
        self.activate(ordinal, observer, roots)
    }
    fn activate(
        &self,
        ordinal: usize,
        observer: OriginalScopeObserver,
        roots: crate::backend::submission_recovery::prefill::TransientRootsProjection,
    ) -> Result<PagedScopeRetention, NativeError> {
        let mut state = self.inner.roles.try_borrow_mut().map_err(|_| busy())?;
        if state.failed || state.active.is_some() || state.next != ordinal {
            return Err(identity());
        }
        state.next = ordinal
            .checked_add(1)
            .ok_or(NativeError::PrefillControl(WorkingMemoryError::Overflow))?;
        state.active = Some(Active {
            roots,
            append_active: false,
            observer: observer.clone(),
            ordinal,
        });
        drop(state);
        // Only a weak equality lookup is installed; actual role/source
        // custody remains in the accepted Recovery, never in thread storage.
        super::append_claim::publish_current(self);
        Ok(PagedScopeRetention {
            source: self.clone(),
            observer,
            ordinal,
            finished: Cell::new(false),
        })
    }
}

/// The existing Recovery owns this source alias. A poll error or an early drop
/// never reports successful completion or refunds the consumed occurrence.
pub(crate) struct PagedScopeRetention {
    source: ProjectedPagedSources,
    observer: OriginalScopeObserver,
    ordinal: usize,
    finished: Cell<bool>,
}
impl PagedScopeRetention {
    pub(crate) fn observe(&self, status: Status) {
        if self.finished.get() {
            return;
        }
        let Ok(mut state) = self.source.inner.roles.try_borrow_mut() else {
            return;
        };
        let exact = state.active.as_ref().is_some_and(|active| {
            active.ordinal == self.ordinal && active.observer.same_scope(&self.observer)
        });
        if !exact {
            state.failed = true;
            return;
        }
        if status.failed || status.blocked {
            state.failed = true;
        }
        if status.settled {
            let complete = self
                .source
                .inner
                .catalogs
                .try_borrow()
                .ok()
                .is_some_and(|loan| {
                    loan.as_ref().is_some_and(|catalogs| {
                        catalogs.programs.iter().all(|program| {
                            program.as_ref().is_some_and(|program| {
                                program.ordinal != self.ordinal
                                    || (program.completed
                                        && program
                                            .scan
                                            .as_ref()
                                            .is_none_or(|scan| !scan.used || scan.completed)
                                        && program.visible.as_ref().is_none_or(|visible| {
                                            !visible.started || visible.completed
                                        }))
                            })
                        })
                    })
                });
            if !complete
                || state
                    .active
                    .as_ref()
                    .is_some_and(|active| active.append_active)
            {
                state.failed = true;
            }
        }
        let retired = if status.settled {
            self.finished.set(true);
            state.active.take()
        } else {
            None
        };
        drop(state);
        drop(retired);
        if status.settled {
            // Existing Recovery observation certifies this role's terminal
            // native boundary. Keep spent program metadata, retire payloads.
            if let Ok(mut loan) = self.source.inner.catalogs.try_borrow_mut() {
                if let Some(catalogs) = loan.as_mut() {
                    for program in catalogs.programs.iter_mut().flatten() {
                        if program.ordinal == self.ordinal {
                            program.failed_root = None;
                            if let Some(scan) = &mut program.scan {
                                scan.retire_settled();
                            }
                            if let Some(visible) = &mut program.visible {
                                visible.retire_settled();
                            }
                        }
                    }
                }
            }
        }
    }
}
impl Drop for PagedScopeRetention {
    fn drop(&mut self) {
        if !self.finished.get() {
            if let Ok(mut state) = self.source.inner.roles.try_borrow_mut() {
                state.failed = true;
            }
        }
    }
}

pub(super) fn control_bytes(forwards: usize) -> Option<usize> {
    let role_frames = [
        size_of::<PagedScopeRetention>(),
        size_of::<Option<PagedScopeRetention>>(),
        size_of::<Active>(),
        size_of::<Option<Active>>(),
        size_of::<Status>(),
        super::scan_program::retirement_control_bytes()?,
        super::visible_program::retirement_control_bytes()?,
        size_of::<std::cell::RefMut<'_, Option<super::catalogs::PreparedPagedCatalogs>>>(),
        size_of::<
            std::iter::Flatten<
                std::slice::IterMut<'_, Option<super::programs::PagedAppendProgram>>,
            >,
        >(),
        size_of::<(usize, OriginalScopeObserver, OriginalOperationRegistration)>(),
        size_of::<std::cell::RefMut<'_, RoleState>>(),
        size_of::<std::cell::Ref<'_, RoleState>>(),
        size_of::<Result<PagedScopeRetention, NativeError>>(),
        size_of::<(&ProjectedPagedSources, &InferenceRequest, &SubmissionScope)>(),
        size_of::<PrefillControlRole>(),
        size_of::<u64>(),
        size_of::<(&ProjectedPagedSources, &OriginalSpeculativeRole,
            Option<&OriginalSpeculativePrefillSpan>, &SubmissionScope)>(),
        size_of::<(&ProjectedPagedSources, usize, OriginalScopeObserver,
            crate::backend::submission_recovery::prefill::TransientRootsProjection)>(),
        size_of::<OriginalHostSourceCustody>(),
        size_of::<Option<&OriginalSpeculativePrefillSpan>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        crate::backend::submission_recovery::prefill::TransientRootsProjection::authenticate_source_control_bytes()?,
    ];
    let per_role = role_frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&role_frames), usize::checked_add)?;
    let fixed = [
        super::append_claim::lookup_control_bytes()?,
        size_of::<RoleState>(),
        size_of::<eredu_runtime::working_memory::OriginalHostSourceCustody>(),
        size_of::<Result<eredu_runtime::working_memory::OriginalHostSourceCustody, CacheSourceError>>(
        ),
        size_of::<Binding>(),
        size_of::<Option<Binding>>(),
        size_of::<(&ProjectedPagedSources, Binding)>(),
        size_of::<(
            &ProjectedPagedSources,
            &OriginalSpeculativeRole,
            &InferenceSpanWorkspacePlan,
        )>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Result<(), NativeError>>(),
        size_of::<Result<(), CacheSourceFailure>>(),
        WorkspaceContext::metadata_source_bytes::<CacheSourceFailure>()?,
    ];
    fixed.into_iter().try_fold(
        std::mem::size_of_val(&fixed).checked_add(per_role.checked_mul(forwards)?)?,
        usize::checked_add,
    )
}
