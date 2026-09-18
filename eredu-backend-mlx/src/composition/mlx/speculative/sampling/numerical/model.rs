//! Model-phase logits cross retirement through one prepaid destination.
use super::*;
use crate::backend::runtime::cache::state::CompletedResidentSource;
use eredu_runtime::working_memory::{
    OriginalEmbeddedSpeculativeRole, OriginalSpeculativeSourceIdentity, WorkingMemoryError,
};
use safemlx::{OriginalBufferBudget, OriginalScopeObserver, PreparedStreamCopy, StreamCopyPlan};
use std::cell::{OnceCell, RefCell};

/// The immutable C stream wrapper and deferred retirement node keep their actual
/// host funding independently of the output value that borrowed the stream.
#[derive(Debug)]
pub(super) struct StreamOwner {
    _custody: StreamSource,
    _funding: HostMetadataFunding,
}
/// Exact retained source accounts share a native stream mechanism, without
/// treating a registered copy as a completed model or numerical allocation.
#[derive(Debug)]
enum StreamSource {
    Completed(eredu_runtime::working_memory::CompletedWorkspaceSourceAccount),
    Registered(eredu_runtime::working_memory::OriginalSpeculativeRegisteredSource),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure<E: std::error::Error + 'static> {
    #[source]
    cause: E,
    _funding: HostMetadataFunding,
}
pub(super) fn failed<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    funding: &HostMetadataFunding,
) -> Error {
    // Its exact shell was paid before the single constructor/fill/seal attempt.
    Error::StorageSource(eredu_core::BackendFailure::from_error(Failure {
        cause,
        _funding: funding.clone(),
    }))
}
fn invalid() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
pub(super) fn overflow() -> Error {
    Error::WorkspacePlanning(HostMetadataFundingError::Overflow)
}
struct Source {
    array: Option<Array>,
    budget: Option<OriginalBufferBudget>,
    stream: PreparedStreamCopy<StreamOwner>,
}
struct Destination {
    // Both completed and provisional native values retire before role and H.
    completed: OnceCell<OriginalNumericalValue>,
    source: RefCell<Option<Source>>,
    filled: Cell<bool>,
    sealed: Cell<bool>,
    role: OriginalEmbeddedSpeculativeRole,
    identity: OriginalSpeculativeSourceIdentity,
    funding: HostMetadataFunding,
}
/// No raw or incomplete value can enter the numerical compiler. Clones share
/// one fill and one seal; the returned carrier owns this destination even after
/// the model phase's Recovery payload retires.
pub(crate) struct PendingModelLogits(Option<Rc<Destination>>);
impl Clone for PendingModelLogits {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(self.0.as_ref().expect("live model logits"))))
    }
}
impl Drop for PendingModelLogits {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl std::fmt::Debug for PendingModelLogits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PendingModelLogits")
    }
}
impl PendingModelLogits {
    fn destination(&self) -> &Destination {
        self.0.as_deref().expect("live model logits")
    }
    pub(crate) fn prepare(
        role: &OriginalEmbeddedSpeculativeRole,
        sources: &OriginalSpeculativeNumericalSources,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let funding = sources.metadata_funding();
        let custody = role.budget_custody();
        if !SpeculativeNumericalSource::Model(&custody).belongs_to_request(sources.request()) {
            return Err(invalid());
        }
        let controls = [
            size_of::<Self>(),
            size_of::<Destination>(),
            size_of::<Source>(),
            size_of::<OriginalSpeculativeSourceIdentity>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Option<Source>>(),
            size_of::<Result<safemlx::StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
            size_of::<Failure<safemlx::StreamCopyCause>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<Failure<safemlx::StreamCopyCause>>().ok_or_else(overflow)?,
            rc_bytes::<Destination>().ok_or_else(overflow)?,
            value_control_bytes().ok_or_else(overflow)?,
            CompletedResidentSource::array_source_control_bytes().ok_or_else(overflow)?,
            size_of::<std::cell::RefMut<'_, Option<Source>>>(),
            size_of::<std::cell::Ref<'_, Option<Source>>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<super::super::logits::IndependentLogits, Error>>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<Result<OriginalScopeObserver, safemlx::error::Exception>>(),
            size_of::<Value>(),
            size_of::<Option<Array>>(),
            size_of::<Result<(), OriginalNumericalValue>>(),
        ];
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        // The same scalar query prices the later completed-source comparison;
        // it creates no stream or queue and does not itself establish completion.
        let comparison = safemlx::StreamCopyPlan::<()>::capture(stream)
            .map_err(|cause| failed(cause, funding))?;
        let compare_parts = [
            size_of::<safemlx::StreamCopyPlan<()>>(),
            comparison.source_comparison_control_bytes().ok_or_else(overflow)?,
        ];
        funding.reserve_metadata(compare_parts.into_iter().try_fold(size_of_val(&compare_parts),usize::checked_add).ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let stream = prepare_stream(custody, stream, funding)?;
        Ok(Self(Some(Rc::new(Destination {
            completed: OnceCell::new(),
            source: RefCell::new(Some(Source {
                array: None,
                budget: None,
                stream,
            })),
            filled: Cell::new(false),
            sealed: Cell::new(false),
            role: role.clone(),
            identity: sources.request().source_identity(),
            funding: funding.clone(),
        }))))
    }
    /// Called by the actual model readout after its already quoted static row
    /// primitive. The budget retains failure ownership; it is not completion.
    pub(crate) fn fill(
        &self,
        array: Array,
        role: &OriginalEmbeddedSpeculativeRole,
        budget: &OriginalBufferBudget,
        observer: &OriginalScopeObserver,
    ) -> Result<super::super::logits::IndependentLogits, Error> {
        let value = self.destination();
        if !value.role.same_role(role) || value.filled.replace(true) || value.sealed.get() {
            return Err(invalid());
        }
        if !observer.same_scope(&OriginalScopeObserver::require_current().map_err(Error::from)?) {
            return Err(invalid());
        }
        if array.ndim() != 2
            || array.dim(0) != 1
            || array.dim(1) <= 0
            || array.dtype() != safemlx::Dtype::Float32
        {
            return Err(invalid());
        }
        let mut source = value.source.try_borrow_mut().map_err(|_| invalid())?;
        let source = source.as_mut().ok_or_else(invalid)?;
        source.array = Some(array);
        source.budget = Some(budget.clone());
        Ok(super::super::logits::IndependentLogits::Pending(
            self.clone(),
        ))
    }
    pub(crate) fn visit_retained(&self, visitor: &mut dyn FnMut(&Array)) {
        let value = self.destination();
        if let Some(completed) = value.completed.get() {
            completed.visit_native_root(visitor);
        } else if let Some(array) = value
            .source
            .borrow()
            .as_ref()
            .and_then(|source| source.array.as_ref())
        {
            visitor(array);
        }
    }
    pub(crate) fn completed(&self) -> Option<&OriginalNumericalValue> {
        self.destination().completed.get()
    }
    /// The completion owner invokes this only after exact native settlement,
    /// Recovery finish and Record retirement. A failed seal is never retryable.
    pub(crate) fn seal(&self, completed: &CompletedResidentSource) -> Result<(), Error> {
        let destination = self.destination();
        if !destination.filled.get() || destination.sealed.replace(true) {
            return Err(invalid());
        }
        let mut source = destination.source.try_borrow_mut().map_err(|_| invalid())?;
        let pending = source.as_ref().ok_or_else(invalid)?;
        let array = pending.array.as_ref().ok_or_else(invalid)?;
        completed.validate_completed_stream(pending.stream.as_stream())?;
        let (budget, custody) = completed.array_source(array, &destination.funding)?;
        if !SpeculativeNumericalSource::Model(custody).belongs_to_identity(&destination.identity) {
            return Err(invalid());
        }
        let budget = budget.clone();
        let custody = custody.clone();
        let pending = source.take().expect("validated pending source");
        let value = OriginalNumericalValue(
            Some(Rc::new(Value {
                array: pending.array.expect("validated model row"),
                original_budget: Some(budget),
                stream: ValueStream::Embedded(pending.stream),
                meaning: Meaning::Logits,
                provenance: Provenance::Model(custody),
                funding: destination.funding.clone(),
                _copy: None,
                _snapshot_host: None,
                _readout_source: None,
            })),
            None,
        );
        // No allocation/funding query remains: the actual Rc destination was
        // charged before native entry. Its owner now carries source completion.
        destination.completed.set(value).map_err(|_| invalid())
    }
}
impl OriginalNumericalValue {
    pub(crate) fn visit_native_root(&self, visitor: &mut dyn FnMut(&Array)) {
        visitor(&self.value().array);
    }
}

/// Shared immutable source-stream owner for immediate and lazy model readouts.
pub(super) fn prepare_stream(
    custody: OriginalSpeculativeBudgetCustody,
    stream: &Stream,
    funding: &HostMetadataFunding,
) -> Result<PreparedStreamCopy<StreamOwner>, Error> {
    prepare_account_stream(custody.into(), stream, funding)
}
pub(super) fn prepare_account_stream(
    custody: eredu_runtime::working_memory::CompletedWorkspaceSourceAccount,
    stream: &Stream,
    funding: &HostMetadataFunding,
) -> Result<PreparedStreamCopy<StreamOwner>, Error> {
    prepare_source_stream(StreamSource::Completed(custody), stream, funding)
}
pub(super) fn prepare_registered_stream(
    source: eredu_runtime::working_memory::OriginalSpeculativeRegisteredSource,
    stream: &Stream,
    funding: &HostMetadataFunding,
) -> Result<PreparedStreamCopy<StreamOwner>, Error> {
    prepare_source_stream(StreamSource::Registered(source), stream, funding)
}
fn prepare_source_stream(
    custody: StreamSource,
    stream: &Stream,
    funding: &HostMetadataFunding,
) -> Result<PreparedStreamCopy<StreamOwner>, Error> {
    let parts =
        [
            size_of::<StreamSource>(),
            size_of::<StreamCopyPlan<StreamOwner>>(),
            size_of::<Result<StreamCopyPlan<StreamOwner>, safemlx::StreamCopyCause>>(),
            size_of::<Result<PreparedStreamCopy<StreamOwner>, safemlx::StreamCopyError<StreamOwner>>>(
            ),
            size_of::<StreamOwner>(),
            size_of::<Failure<safemlx::StreamCopyCause>>(),
            size_of::<Failure<safemlx::StreamCopyError<StreamOwner>>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<
                Failure<safemlx::StreamCopyCause>,
            >()
            .ok_or_else(overflow)?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<
                Failure<safemlx::StreamCopyError<StreamOwner>>,
            >()
            .ok_or_else(overflow)?,
        ];
    funding
        .reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(overflow)?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let plan =
        StreamCopyPlan::<StreamOwner>::capture(stream).map_err(|cause| failed(cause, funding))?;
    let shared = Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(plan.shared_body_layout())
        .map_err(|_| overflow())?
        .0
        .pad_to_align()
        .size();
    let controls = [
        plan.control_bytes().ok_or_else(overflow)?,
        shared,
        plan.owner_node_layout().size(),
        plan.native_wrapper_bytes(),
    ];
    funding
        .reserve_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or_else(overflow)?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let stream = plan
        .realize(StreamOwner {
            _custody: custody,
            _funding: funding.clone(),
        })
        .map_err(|cause| failed(cause, funding))?;
    Ok(stream)
}
