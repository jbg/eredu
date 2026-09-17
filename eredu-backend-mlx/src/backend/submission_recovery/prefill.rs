//! Source-bound prefill roles, using the existing final-node Recovery engine.
use super::{
    PreparedRecovery, PreparedRecoveryError, Recovery, RecoveryPreparationError, Retention, Status,
};
use crate::backend::error::Error;
use eredu_runtime::{
    prefill::{PrefillControlPlan, PrefillControlRole, PrefillSpanControlPhase},
    working_memory::{
        InferenceRequest, OriginalPrefillNativeCustody, OriginalPrefillRecoveryCustody,
        OriginalTextControlGuard, OriginalTextPrefillScopeSet, TextPrefillScopeFacts,
        WorkingMemoryError,
    },
};
use safemlx::{
    PrefillRoots, PrefillRootsRuntime, SubmissionGraphQuota, SubmissionRecordQuota,
    SubmissionScopeOwnerCause,
};
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    mem::size_of,
    rc::{Rc, Weak},
};
mod roots;
use roots::RootsOwner;
pub(crate) use roots::RootsProjection;
pub(crate) mod model_execution;
pub(crate) use model_execution::{
    ModelExecutionOwner, ModelExecutionPreparation, ModelExecutionProjection,
};

#[derive(Clone, Debug)]
pub(crate) enum PrefillControlProjection {
    Prefill(PrefillBankProjection),
    Model(ModelExecutionProjection),
}
impl PrefillControlProjection {
    pub(crate) fn is_live(&self) -> bool {
        match self {
            Self::Prefill(v) => v.is_live(),
            Self::Model(v) => v.is_live(),
        }
    }
    pub(crate) fn validate_execution(
        &self,
        execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    ) -> Result<(), Error> {
        match self {
            Self::Prefill(v) => v.validate_execution(execution),
            Self::Model(v) => v.validate_execution(execution),
        }
    }
}
pub(crate) enum NativeReservationGuard {
    Prefill(PrefillGuard),
    Model(model_execution::ModelReservationGuard),
}
pub(crate) struct ReservationGuard {
    // Lexical TLS authority must retire before asynchronous native recovery.
    validation: Option<crate::backend::nn::tensor::TokenValidationRole>,
    native: NativeReservationGuard,
}
impl ReservationGuard {
    pub(crate) fn new(native: NativeReservationGuard,
        parent: Option<crate::backend::nn::tensor::TokenValidationParent>) -> Result<Self, Error> {
        let validation = parent.map(crate::backend::nn::tensor::TokenValidationParent::bind).transpose()?;
        Ok(Self { validation, native })
    }
    pub(crate) fn into_native(self) -> NativeReservationGuard {
        let Self { validation, native } = self;
        drop(validation);
        native
    }
}

pub(crate) struct PrefillRetention {
    request: InferenceRequest,
    roots: Option<RootsOwner>,
    // Non-root roles may inspect metadata, prepare sources, index or agree;
    // each has its own prepaid carrier, without manufacturing a root collector.
    operation_failure: Option<safemlx::RetainedPrefillFailure>,
    registration: Option<crate::backend::runtime::execution::generic::RegisteredOriginalScope>,
    // Separate from optional group registration: even a single-group equation
    // must retire its terminal Record payload before advancing a reused bound.
    retirement: Option<safemlx::OriginalScopeObserver>,
    // Actual native/root nodes and request precede the final Rust-node custody.
    paged: Option<crate::backend::nn::workspace::PagedScopeRetention>,
    custody: Option<OriginalPrefillRecoveryCustody>,
}
impl Retention for PrefillRetention {
    fn observe(&self, status: Status) { if let Some(paged) = &self.paged { paged.observe(status); } }
}
pub(crate) type PrefillGuard = Recovery<PrefillRetention>;

/// Same completion driver, followed by exact successful-role alias retirement.
/// Root payload retirement precedes removing its observer from the request bank.
pub(crate) fn finish(mut guard: PrefillGuard) -> Result<Status, Error> {
    let registration = guard.retention_mut().registration.take();
    let retirement = guard.retention_mut().retirement.take();
    guard.seal();
    let status = guard.finish();
    if status.settled && !status.failed && !status.blocked {
        if let Some(observer) = &retirement {
            // Root/payload destruction above can expose the final native
            // wrappers. Keep this exact observer/Q live through their drain.
            super::retirement::complete(observer)?;
        }
    }
    if let Some(registration) = registration {
        registration.finish(status);
    }
    Ok(status)
}
/// A failed ordinary entry retains the exact request, without thread-local roots.
pub(crate) struct PrefillRequestRetention(pub(crate) InferenceRequest);
impl Retention for PrefillRequestRetention {
    fn observe(&self, _: Status) {}
}
type Ready = PreparedRecovery<PrefillRetention, OriginalPrefillNativeCustody>;
struct Slot {
    ready: Option<Ready>,
    roots: Option<RootsProjection>,
}
struct Bank {
    // Borrow only the currently entered input transaction's construction bank.
    // This weak header retires before the containing original request custody.
    capture: Option<roots::CaptureProjection>,
    native_storage: Option<crate::backend::runtime::residency::storage::native_storage::BankOwner>,
    operations: Option<crate::backend::runtime::execution::generic::OriginalOperationRegistration>,
    paged: Option<crate::backend::nn::workspace::ProjectedPagedSources>,
    slots: Vec<Slot>,
    next: usize,
    checked_out: bool,
    failed: bool,
    // Vector allocation, partial/prepared nodes and all source identity precede Q.
    original: OriginalTextPrefillScopeSet,
}
pub(crate) struct PrefillBankOwner(Option<Rc<RefCell<Bank>>>);
impl Drop for PrefillBankOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
/// Private weak installed view. It retains only closed original control custody,
/// never a native owner/session/quote backedge. Weak header retires before Q.
pub(crate) struct PrefillBankProjection {
    value: Weak<RefCell<Bank>>,
    controls: OriginalTextControlGuard,
}
impl Clone for PrefillBankProjection {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            controls: self.controls.clone(),
        }
    }
}
impl PrefillBankProjection {
    pub(crate) fn is_live(&self) -> bool {
        self.value.strong_count() != 0
    }
}
impl std::fmt::Debug for PrefillBankProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillBankProjection")
            .finish_non_exhaustive()
    }
}
fn memory(cause: WorkingMemoryError) -> Error {
    Error::PrefillControl(cause)
}
fn allocation() -> Error {
    Error::PrefillRoots(safemlx::PrefillRootsCause::Allocation.into())
}
fn completion_role(role: PrefillControlRole) -> bool {
    matches!(
        role,
        PrefillControlRole::Span {
            phase: PrefillSpanControlPhase::InputTransaction,
            ..
        }
    )
}
impl PrefillBankOwner {
    pub(crate) fn capture_observer(&self) -> Result<safemlx::OriginalScopeObserver, Error> {
        let projection = {
            let bank = self
                .0
                .as_ref()
                .ok_or(Error::PrefillScopeUnavailable)?
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            if bank.failed || bank.checked_out {
                return Err(Error::PrefillScopeUnavailable);
            }
            bank.capture.clone().ok_or(Error::PrefillScopeUnavailable)?
        };
        projection.capture_observer()
    }
    pub(crate) fn with_paged_sources(self, paged: Option<crate::backend::nn::workspace::ProjectedPagedSources>) -> Self {
        self.0.as_ref().expect("owned prefill bank").borrow_mut().paged = paged;
        self
    }
    pub(crate) fn with_native_storage(
        self,
        bank: Option<crate::backend::runtime::residency::storage::native_storage::BankOwner>,
    ) -> Self {
        self.0
            .as_ref()
            .expect("owned prefill bank")
            .borrow_mut()
            .native_storage = bank;
        self
    }
    pub(crate) fn with_operation_registration(
        self,
        registration: crate::backend::runtime::execution::generic::OriginalOperationRegistration,
    ) -> Self {
        self.0
            .as_ref()
            .expect("owned prefill bank")
            .borrow_mut()
            .operations = Some(registration);
        self
    }

    /// All concrete buffers, recovery nodes and native Scope owners exist before
    /// SourcePreparation begins. These are the same original Record/Graph arenas.
    pub(crate) fn new(
        mut original: OriginalTextPrefillScopeSet,
        request: &InferenceRequest,
        runtime: &PrefillRootsRuntime,
        record: Option<&SubmissionRecordQuota>,
        graph: &SubmissionGraphQuota,
        controls: OriginalTextControlGuard,
    ) -> Result<(Self, PrefillBankProjection), Error> {
        Self::new_with_resident_recipe(original, request, runtime, record, graph, controls, None)
    }
    pub(crate) fn new_with_resident_recipe(
        original: OriginalTextPrefillScopeSet,
        request: &InferenceRequest,
        runtime: &PrefillRootsRuntime,
        record: Option<&SubmissionRecordQuota>,
        graph: &SubmissionGraphQuota,
        controls: OriginalTextControlGuard,
        recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
    ) -> Result<(Self, PrefillBankProjection), Error> {
        Self::new_with_resident_sources(original, request, runtime, record, graph, controls, recipe, None)
    }
    pub(crate) fn new_with_resident_sources(
        mut original: OriginalTextPrefillScopeSet,
        request: &InferenceRequest,
        runtime: &PrefillRootsRuntime,
        record: Option<&SubmissionRecordQuota>,
        graph: &SubmissionGraphQuota,
        controls: OriginalTextControlGuard,
        recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
        addressable: Option<&crate::backend::submission_recovery::addressable::AddressableRequestOwner>,
    ) -> Result<(Self, PrefillBankProjection), Error> {
        if let Some(addressable) = addressable { addressable.validate_request(request)?; }
        if recipe.is_some_and(|recipe| recipe.plan().geometry() != request.geometry()) {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        original.validate_request(request).map_err(memory)?;
        controls
            .validate_reservation(
                request
                    .memory_reservation()
                    .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?,
            )
            .map_err(memory)?;
        if record.is_none() {
            return Err(safemlx::OriginalNativeControlError::MissingRecord.into());
        }
        let facts = original.facts();
        let count = usize::try_from(facts.plan().scope_count())
            .map_err(|_| memory(WorkingMemoryError::Overflow))?;
        let capacity = usize::try_from(facts.root_capacity())
            .map_err(|_| memory(WorkingMemoryError::Overflow))?;
        Layout::array::<Slot>(count).map_err(|_| memory(WorkingMemoryError::Overflow))?;
        let mut slots = Vec::new();
        slots.try_reserve_exact(count).map_err(|_| allocation())?;
        for ordinal in 0..facts.plan().scope_count() {
            let descriptor = facts.plan().role(ordinal).expect("checked role geometry");
            let role = original.take_next(descriptor).map_err(memory)?;
            let (native, recovery, root_custody, projection_custody) = role.into_custody();
            let mut operation_failure = None;
            let (roots, projection) = if completion_role(descriptor) {
                let (owner, projection) = RootsOwner::new(
                    runtime,
                    capacity,
                    Some(graph),
                    Some(root_custody),
                    Some(projection_custody),
                )?;
                let traversal = recipe
                    .map(|recipe| recipe.completion_for_prefill(descriptor))
                    .transpose()?;
                let parallel=recipe.map(|recipe|recipe.parallel_for_prefill(descriptor)).transpose()?.flatten();
                let row = recipe.map(|recipe| recipe.addressable_for_prefill(descriptor)).transpose()?.flatten()
                    .map(|(row, invocation)| addressable.ok_or(Error::PrefillScopeUnavailable)?
                        .row(row, 0, invocation)).transpose()?;
                let owner = owner.with_traversal(traversal)?.with_parallel(parallel)?.with_addressable(row)?;
                (Some(owner), Some(projection))
            } else {
                let prepared =
                    safemlx::PreparedPrefillFailure::try_new(root_custody).map_err(|error| {
                        let (cause, owner) = error.into_parts();
                        drop(owner);
                        Error::PrefillRoots(cause.into())
                    })?;
                operation_failure = Some(prepared.try_allocate().map_err(|error| {
                    let (cause, owner) = error.into_parts();
                    drop(owner);
                    Error::PrefillRoots(cause.into())
                })?);
                drop(projection_custody);
                (None, None)
            };
            let retention = PrefillRetention {
                request: request.clone(),
                paged: None,
                roots,
                operation_failure,
                registration: None,
                retirement: None,
                custody: Some(recovery),
            };
            let ready = match Ready::new(retention, native) {
                Ok(ready) => ready
                    .with_record_quota(record.cloned())
                    .with_graph_quota(Some(graph.clone())),
                Err(RecoveryPreparationError {
                    cause,
                    retention,
                    custody,
                }) => {
                    // No source/native work began; original and slots still own
                    // their full prefix while the actual failure locals retire.
                    drop(retention);
                    drop(custody);
                    return Err(Error::PrefillScope(cause));
                }
            };
            slots.push(Slot {
                ready: Some(ready),
                roots: projection,
            });
        }
        #[cfg(test)]
        test_trace::record(test_trace::Event::Prepared {
            plan: facts.plan(),
            roots: facts.root_capacity(),
            graph: facts.graph_bytes().map_err(memory)?,
        });
        let value = Rc::new(RefCell::new(Bank {
            capture: None,
            native_storage: None,
            operations: None,
            paged: None,
            slots,
            next: 0,
            checked_out: false,
            failed: false,
            original,
        }));
        let view = PrefillBankProjection {
            value: Rc::downgrade(&value),
            controls,
        };
        Ok((Self(Some(value)), view))
    }
}
struct Checkout<'a> {
    bank: &'a RefCell<Bank>,
    armed: bool,
}
impl Drop for Checkout<'_> {
    fn drop(&mut self) {
        if self.armed {
            // No native call holds this loan. Unwinding spends the run, not just
            // its slot; the extracted pending/active native owner retires apart.
            let mut bank = self.bank.borrow_mut();
            bank.failed = true;
            bank.checked_out = false;
        }
    }
}
impl PrefillBankProjection {
    pub(crate) fn validate_execution(
        &self,
        execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    ) -> Result<(), Error> {
        let owner = PrefillBankOwner(Some(
            self.value.upgrade().ok_or(Error::PrefillScopeUnavailable)?,
        ));
        let cell = owner.0.as_ref().expect("closed bank");
        let bank = cell
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        bank.original.validate_execution(execution).map_err(memory)
    }
    /// Missing or expired original installation never becomes ordinary begin.
    pub(crate) fn begin(
        &self,
        request: &InferenceRequest,
        role: PrefillControlRole,
    ) -> Result<(PrefillGuard, Option<RootsProjection>), Error> {
        let owner = PrefillBankOwner(Some(
            self.value.upgrade().ok_or(Error::PrefillScopeUnavailable)?,
        ));
        let cell = owner.0.as_ref().expect("closed upgraded bank");
        let (pending, previous_capture) = {
            let mut bank = cell
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            bank.original.validate_request(request).map_err(memory)?;
            self.controls
                .validate_reservation(
                    request
                        .memory_reservation()
                        .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?,
                )
                .map_err(memory)?;
            if bank.failed {
                return Err(Error::PrefillScopeUnavailable);
            }
            if bank.checked_out {
                return Err(Error::PrefillScopeReentrant);
            }
            if bank.original.facts().plan().role(bank.next as u64) != Some(role) {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            let index = bank.next;
            let pending = bank
                .slots
                .get_mut(index)
                .and_then(|slot| slot.ready.take())
                .ok_or(Error::PrefillScopeUnavailable)?;
            bank.checked_out = true;
            (pending, bank.capture.take())
        };
        // A prior role's last weak header never retires under the bank loan.
        drop(previous_capture);
        let mut checkout = Checkout {
            bank: cell,
            armed: true,
        };
        // All RefCell/source loans ended before actual native acceptance.
        match pending.try_begin() {
            Ok(mut active) => {
                // The same genuine claimed scope exists before this no-hooks
                // configuration. Refusal spends the bank and retains/retire-
                // closes the accepted empty scope; it never reconstructs Ready.
                let node = active
                    .node
                    .as_mut()
                    .expect("accepted prefill node")
                    .node_mut();
                let scope = node.probe.as_mut().expect("accepted native scope");
                scope
                    .enable_scoped_observation()
                    .map_err(|_| Error::PrefillScopeUnavailable)?;
                scope.require_original_native_controls()?;
                if let Some(roots) = &node.retention.roots {
                    roots.bind_scope(scope)?;
                } else if let Some(failure) = &node.retention.operation_failure {
                    failure.bind_original_scope(scope)?;
                }
                // Every role is original-required; absence of either carrier
                // is a fixed refusal, never an ordinary producer fallback.
                scope.enable_original_native_controls()?;
                let native_storage = cell
                    .try_borrow()
                    .map_err(|_| Error::PrefillScopeReentrant)?
                    .native_storage
                    .clone();
                if let Some(bank) = native_storage {
                    bank.bind_scope(scope, &self.controls)?;
                }
                let observer = safemlx::OriginalScopeObserver::require_current()?;
                if !observer.belongs_to(scope) {
                    return Err(Error::PrefillScopeUnavailable);
                }
                node.retention.retirement = Some(observer);
                let operations = cell
                    .try_borrow()
                    .map_err(|_| Error::PrefillScopeReentrant)?
                    .operations
                    .clone();
                if let Some(operations) = operations {
                    node.retention.registration =
                        operations.register_for_retirement(request, scope)?;
                }
                let capture = if let Some(roots) = &node.retention.roots {
                    roots.begin_host(scope)?;
                    Some(roots.capture_projection())
                } else {
                    None
                };
                if completion_role(role) {
                    let paged = cell.try_borrow().map_err(|_| Error::PrefillScopeReentrant)?.paged.clone();
                    if let Some(paged) = paged {
                        let transient = node.retention.roots.as_ref().ok_or(Error::PrefillScopeUnavailable)?.transient_projection()?;
                        node.retention.paged = Some(paged.enter_prefill(request, role, scope, transient)?);
                    }
                }
                let projection = {
                    let mut bank = cell.borrow_mut();
                    let index = bank.next;
                    let projection = bank.slots[index].roots.take();
                    bank.next += 1;
                    bank.checked_out = false;
                    bank.capture = capture;
                    projection
                };
                checkout.armed = false;
                #[cfg(test)]
                test_trace::record(test_trace::Event::Started(role));
                Ok((active, projection))
            }
            Err(PreparedRecoveryError { cause, pending }) => {
                // Only Busy preserves the identical node for an explicit retry.
                // Other refusals terminally fence this finite bank.
                let previous = {
                    let mut bank = cell.borrow_mut();
                    let index = bank.next;
                    let old = bank.slots[index].ready.replace(pending);
                    bank.checked_out = false;
                    bank.failed = cause != SubmissionScopeOwnerCause::RuntimeBusy;
                    old
                };
                checkout.armed = false;
                drop(previous);
                Err(Error::PrefillScope(cause))
            }
        }
    }
}
/// Ordinary compatibility uses the same Recovery retention and fixed root
/// mechanism. Its concrete owner remains the caller's ordinary operation.
pub(crate) fn begin_ordinary(
    request: InferenceRequest,
) -> Result<(PrefillGuard, Option<RootsProjection>), Error> {
    // Existing reserved components without C1 keep their existing completion
    // mechanism. They neither obtain C1 authority nor allocate a null-domain
    // collector while another original arena is active.
    let (roots, projection) = if request.memory_reservation().is_some() {
        (None, None)
    } else {
        let (roots, projection) = RootsOwner::ordinary();
        (Some(roots), Some(projection))
    };
    let guard = Recovery::try_begin(PrefillRetention {
        request,
        paged: None,
        roots,
        operation_failure: None,
        registration: None,
        retirement: None,
        custody: None,
    })
    .map_err(|error| {
        let (unused, cause) = error.into_parts();
        // No native work was accepted. Only the newly created empty roots
        // retire here on their owning thread. Preserve the exact request and
        // its charge in the error, independently of the caller's other aliases.
        let PrefillRetention {
            request,
            paged,
            roots,
            operation_failure,
            registration,
            retirement,
            custody,
        } = unused;
        drop(roots);
        drop(paged);
        drop(operation_failure);
        drop(registration);
        drop(retirement);
        drop(custody);
        Error::Other(Box::new(super::RecoveryBeginError {
            retention: PrefillRequestRetention(request),
            cause,
        }))
    })?;
    Ok((guard, projection))
}
/// Exact cold controls for each actual role and all coexistence. There is no
/// second numeric allowance: graph_bytes enters the one original arena plan.
pub(crate) fn facts(
    geometry: eredu_core::InferenceGeometry,
    root_capacity: u64,
) -> Result<TextPrefillScopeFacts, Error> {
    let plan = PrefillControlPlan::new(geometry, true).map_err(memory)?;
    let capacity =
        usize::try_from(root_capacity).map_err(|_| memory(WorkingMemoryError::Overflow))?;
    let layout = PrefillRoots::layout(capacity).map_err(|e| Error::PrefillRoots(e.into()))?;
    let native_controls = safemlx::OriginalNativeControlLayout::inspect()?;
    let synchronous = safemlx::original_scoped_evaluation_control_bytes()
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let role = Ready::control_bytes()
        .and_then(|n| n.checked_add(synchronous))
        .and_then(|n| n.checked_add(super::retirement::control_bytes()?))
        .and_then(|n| {
            n.checked_add(
                crate::backend::runtime::execution::generic::RegisteredOriginalScope::control_bytes()?,
            )
        })
        .and_then(|n| n.checked_add(u64::try_from(crate::backend::nn::tensor::TokenValidationRole::control_bytes()?).ok()?))
        .and_then(|n| n.checked_add(u64::try_from(size_of::<Slot>()).ok()?))
        .and_then(|n| n.checked_add(u64::try_from(size_of::<PrefillRetention>()).ok()?))
        .and_then(|n| n.checked_add(u64::try_from(native_controls.fixed_control_bytes).ok()?));
    let complete = role
        .and_then(|n| n.checked_add(roots::control_bytes()?))
        .and_then(|n| n.checked_add(u64::try_from(layout.host_bytes()?).ok()?));
    let failure = safemlx::PreparedPrefillFailure::<
        eredu_runtime::working_memory::OriginalPrefillRootCustody,
    >::layout()
    .map_err(|cause| Error::PrefillRoots(cause.into()))?;
    let operation = role.and_then(|n| n.checked_add(u64::try_from(failure.total_bytes()?).ok()?));
    let header = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .map_err(|_| memory(WorkingMemoryError::Overflow))?
        .pad_to_align();
    let allocation = header
        .extend(Layout::new::<RefCell<Bank>>())
        .map_err(|_| memory(WorkingMemoryError::Overflow))?
        .0
        .pad_to_align()
        .size();
    // Slot array storage itself is included once per actual role above.
    usize::try_from(plan.scope_count())
        .ok()
        .and_then(|n| Layout::array::<Slot>(n).ok())
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    // D2 replaces the four inspector boxes with fixed borrowed companions.
    // These are the actual outer-boundary/inner-callback return shapes; install
    // and retirement share the same projection shape. No callback is executed
    // here, and these controls do not certify full media workspace/completion.
    type Boundary = eredu_runtime::replicated_session::RuntimeInspectionBoundary;
    let bank =
        [
            size_of::<safemlx::PrefillRootsRuntime>(),
            size_of::<Option<safemlx::PrefillRuntimeBaseline>>(),
            size_of::<safemlx::OriginalNativeControlLayout>(),
            size_of::<
                Result<safemlx::OriginalNativeControlLayout, safemlx::OriginalNativeControlError>,
            >(),
            size_of::<Boundary>(),
            size_of::<Result<(), Boundary>>(),
            size_of::<Result<Result<Option<TextPrefillScopeFacts>, Error>, Boundary>>(),
            size_of::<Result<Result<safemlx::PrefillRootsRuntime, Error>, Boundary>>(),
            size_of::<Result<Result<Option<PrefillBankProjection>, Error>, Boundary>>(),
            size_of::<Result<Option<PrefillBankProjection>, Error>>(),
            size_of::<Bank>(),
            size_of::<Bank>(),
            size_of::<PrefillBankOwner>(),
            size_of::<PrefillBankProjection>(),
            size_of::<(PrefillBankOwner, PrefillBankProjection)>(),
            size_of::<Result<(PrefillBankOwner, PrefillBankProjection), Error>>(),
            size_of::<Rc<RefCell<Bank>>>(),
            size_of::<Weak<RefCell<Bank>>>(),
            size_of::<Option<Rc<RefCell<Bank>>>>(),
            size_of::<std::mem::ManuallyDrop<Rc<RefCell<Bank>>>>(),
            size_of::<Result<RefCell<Bank>, Rc<RefCell<Bank>>>>(),
            size_of::<Option<(PrefillBankOwner, PrefillBankProjection)>>(),
            size_of::<Result<Option<(PrefillBankOwner, PrefillBankProjection)>, Error>>(),
            size_of::<Option<PrefillBankProjection>>(),
            size_of::<std::cell::Ref<'static, Bank>>(),
            size_of::<PrefillControlProjection>(),
            size_of::<Option<PrefillControlProjection>>(),
            size_of::<Result<Option<PrefillControlProjection>, Error>>(),
            size_of::<std::cell::Ref<'static, Option<PrefillControlProjection>>>(),
            size_of::<std::cell::RefMut<'static, Option<PrefillControlProjection>>>(),
            size_of::<ReservationGuard>(),
            size_of::<Result<ReservationGuard, Error>>(),
            size_of::<
                std::cell::RefMut<
                    'static,
                    eredu_runtime::working_memory::OriginalTextPrefillScopes,
                >,
            >(),
            size_of::<Option<RefCell<Bank>>>(),
            size_of::<Checkout<'static>>(),
            size_of::<Option<roots::CaptureProjection>>(),
            size_of::<Option<roots::CaptureProjection>>(),
            size_of::<Vec<Slot>>(),
            size_of::<std::cell::RefMut<'static, Bank>>(),
            size_of::<Result<(PrefillGuard, Option<RootsProjection>), Error>>(),
        ]
        .into_iter()
        .try_fold(allocation, usize::checked_add)
        .and_then(|n| u64::try_from(n).ok());
    // Every cached decode retains one independently prepared collector until
    // its actual ModelExecution recovery releases. The first prefill keeps its
    // existing named role bank. Count the full possible retained population.
    let bank = bank.and_then(|n| {
        n.checked_add(
            model_execution::control_bytes(capacity)?
                .checked_mul(geometry.max_output_tokens.saturating_sub(1))?,
        )
    });
    TextPrefillScopeFacts::new(
        geometry,
        [
            operation, operation, operation, complete, operation, operation, operation,
        ],
        bank,
        root_capacity,
        u64::try_from(layout.graph_bytes).map_err(|_| memory(WorkingMemoryError::Overflow))?,
    )
    .map_err(memory)
}

#[cfg(test)]
pub(crate) mod test_trace {
    use super::{PrefillControlPlan, PrefillControlRole};
    use std::cell::RefCell;
    #[derive(Clone, Debug)]
    pub(crate) enum Event {
        ModelCompleted {
            roots: usize,
        },
        Prepared {
            plan: PrefillControlPlan,
            roots: u64,
            graph: u64,
        },
        Started(PrefillControlRole),
        Completed {
            roots: usize,
            original: bool,
        },
        Media {
            future: usize,
        },
    }
    thread_local! { static CURRENT: RefCell<Option<Vec<Event>>> = const { RefCell::new(None) }; }
    pub(crate) struct Trace;
    impl Trace {
        pub(crate) fn new() -> Self {
            CURRENT.with(|slot| assert!(slot.replace(Some(Vec::new())).is_none()));
            Self
        }
        pub(crate) fn events(&self) -> Vec<Event> {
            CURRENT.with(|slot| slot.borrow().as_ref().unwrap().clone())
        }
    }
    impl Drop for Trace {
        fn drop(&mut self) {
            CURRENT.with(|slot| drop(slot.take()));
        }
    }
    pub(crate) fn record(event: Event) {
        CURRENT.with(|slot| {
            if let Some(events) = slot.borrow_mut().as_mut() {
                events.push(event);
            }
        });
    }
}

pub(crate) mod nested;

pub(crate) use roots::TransientRootsProjection;
