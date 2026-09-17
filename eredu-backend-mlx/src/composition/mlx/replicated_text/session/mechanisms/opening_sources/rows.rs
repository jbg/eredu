//! Original finite opening/end rows. This private composition does not activate
//! span execution, prove enclosing inventory, or authorize future acquisitions.
use super::capacity::registered::{descriptor_count, storage_entry};
use super::capacity::{NativeOpeningCapacityPlan, OpeningExecution, OpeningSlots};
use super::fixed::{FixedOpeningOwners, OpeningEntry, OpeningError};
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity as Key;
use eredu_runtime::{
    inspection::{PrefillOpeningExecution, SettledPrefillChunkRetention},
    prefill::PrefillChunk,
    working_memory::{
        BoundedPinAttempt, BoundedPublicationAttempt, BoundedPublishedAllocation,
        BoundedRegisteredStorage, CaptureSourceSegment, IncrementalInferenceQuote,
        InferenceExecutionIdentity, InferenceRequest, InferenceWorkspaceSpan,
        OriginalPrefillStoragePinSlots, OriginalPrefillStoragePublicationSlots,
        OriginalTextControlGuard, OwnedTextSpanWorkspace, PreparedCapturePlanPublication,
        PreparedPrefillStoragePinPlan, PreparedPrefillStoragePublicationPlan,
        PreparedTextControlWorkspace, TextHostControlFacts, WorkingMemoryFundingScope,
    },
    PreparedLayeredObservationPaths, PreparedObservationBindingIdentity,
    SharedLayeredObservationPaths,
};
use safemlx::{PreparedAllocationOwner, PreparedAllocationOwnerCause};
use std::{
    cell::{Cell, RefCell},
    mem::size_of,
};

mod owner;
pub(in crate::composition::mlx::replicated_text) use owner::InstalledOpeningRows;
pub(crate) use owner::NativeOpeningRowsOwner;

/// Created only by the exact quiescent paired native capacity inspection.
/// The current token is borrowed there; only its non-Clone fingerprint escapes.
pub(crate) struct NativeOpeningRowsPlan {
    slots: OpeningSlots,
    paths: SharedLayeredObservationPaths,
    identity: InferenceExecutionIdentity,
    binding: PreparedObservationBindingIdentity,
    selection: eredu_runtime::layered::PreparedCaptureSelection,
    geometry: eredu_core::InferenceGeometry,
}
impl NativeOpeningRowsPlan {
    pub(super) fn from_capacity<S: MlxStateMechanisms, E: OpeningExecution<S>>(
        plan: NativeOpeningCapacityPlan<'_, S, E>,
        identity: &InferenceExecutionIdentity,
    ) -> Result<Self, OpeningError> {
        plan.execution.validate_paths(plan.paths)?;
        let active = eredu_runtime::capture::CaptureObservationStep::new(
            plan.selection.selection().source().admission(),
            eredu_core::capture::CapturePhase::Prefill,
            0,
        )
        .map_err(|_| OpeningError::Identity)?;
        if !active.has_selected_prefill_hook() {
            return Err(OpeningError::Identity);
        }

        // This is an alias of the already architecture-bound selection, never
        // a new declaration binder or a tuple reconstructed from diagnostics.
        let selection = plan.selection.selection().clone();
        Ok(Self {
            slots: plan.slots,
            paths: plan.paths.source().clone(),
            identity: identity.clone(),
            binding: plan.paths.binding_identity(),
            selection,
            geometry: plan.selection.geometry(),
        })
    }

    /// Both inventories, all finite nodes and constructor/error overlap are
    /// included before the one original control seal. Nothing is refilled.
    fn control_bytes(&self, rows: usize) -> Result<u64, OpeningError> {
        let descriptors = descriptor_count(self.slots)?;
        let native = self
            .slots
            .arrays
            .checked_add(self.slots.hosts)
            .ok_or(OpeningError::Overflow)?;
        let layout = PreparedAllocationOwner::<Attachment>::layout();
        let node = layout
            .allocation_bytes()
            .ok_or(OpeningError::Overflow)?
            .checked_add(layout.native_list_bytes())
            .ok_or(OpeningError::Overflow)?
            .checked_add(layout.preparation_control_bytes())
            .ok_or(OpeningError::Overflow)?
            .checked_add(layout.preparation_failure_bytes())
            .ok_or(OpeningError::Overflow)?
            .checked_add(layout.attachment_failure_bytes())
            .ok_or(OpeningError::Overflow)?
            .checked_add(3 * size_of::<Option<PreparedAllocationOwner<Attachment>>>())
            .ok_or(OpeningError::Overflow)?;
        let row = FixedOpeningOwners::storage_bytes(self.slots)?
            .checked_mul(2)
            .ok_or(OpeningError::Overflow)?
            .checked_add(
                u64::try_from(native.checked_mul(node).ok_or(OpeningError::Overflow)?)
                    .map_err(|_| OpeningError::Overflow)?,
            )
            .ok_or(OpeningError::Overflow)?
            .checked_add(
                u64::try_from(
                    descriptors
                        .checked_mul(
                            size_of::<OwnerIndex>()
                                + size_of::<Option<BoundedPublishedAllocation<Key>>>(),
                        )
                        .ok_or(OpeningError::Overflow)?,
                )
                .map_err(|_| OpeningError::Overflow)?,
            )
            .ok_or(OpeningError::Overflow)?
            .checked_add((3 * size_of::<OpeningRow>()) as u64)
            .ok_or(OpeningError::Overflow)?;
        let controls = 3
            * (size_of::<Self>()
                + size_of::<SealedOpeningRows>()
                + size_of::<Rows>()
                + size_of::<RetiredOpeningRow>()
                + size_of::<RowFailure>()
                + size_of::<NativeRowFailure>()
                + size_of::<RowAllocationFailure>()
                + size_of::<RowError>());
        row.checked_mul(rows.try_into().map_err(|_| OpeningError::Overflow)?)
            .and_then(|n| n.checked_add(controls as u64))
            .and_then(|n| n.checked_add(owner::control_bytes()?))
            .and_then(|n| n.checked_add(PreparedObservationBindingIdentity::control_peak_bytes()?))
            .ok_or(OpeningError::Overflow)
    }

    pub(crate) fn seal(
        self,
        quote: IncrementalInferenceQuote,
        facts: TextHostControlFacts,
        publication: PreparedCapturePlanPublication<Key>,
    ) -> Result<(SealedOpeningRows, IncrementalInferenceQuote), Error> {
        self.seal_with_sequence(quote, facts, publication, None, None, None, None, None)
    }

    pub(crate) fn seal_with_sequence(
        self,
        quote: IncrementalInferenceQuote,
        facts: TextHostControlFacts,
        publication: PreparedCapturePlanPublication<Key>,
        claim: Option<&eredu_core::GenerationSequencePreparation<'_, '_>>,
        tracking: Option<eredu_runtime::working_memory::SubmissionTrackingFacts>,
        graph: Option<eredu_runtime::working_memory::GraphMetadataFacts>,
        prefill: Option<eredu_runtime::working_memory::TextPrefillScopeFacts>,
        native: Option<
            eredu_runtime::working_memory::PreparedNativeStoragePlan<
                crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage,
            >,
        >,
    ) -> Result<(SealedOpeningRows, IncrementalInferenceQuote), Error> {
        let result = || -> Result<_, OpeningError> {
            if quote.geometry() != self.geometry {
                return Err(OpeningError::Identity);
            }
            let span = quote.span_workspace().plan();
            let count = span
                .records()
                .iter()
                .filter(|r| matches!(r.span(), InferenceWorkspaceSpan::Prefill(_)))
                .count();
            let extra = self.control_bytes(count)?;
            let carrier = facts
                .carrier_bytes()
                .map(|n| n.checked_add(extra).ok_or(OpeningError::Overflow))
                .transpose()?;
            let controls = PreparedTextControlWorkspace::prepare(
                self.selection.source(),
                self.geometry,
                span,
                TextHostControlFacts::new(facts.admission_bytes(), carrier, facts.work_bytes()),
            )?
            .with_prefill_capture_selection(
                self.selection
                    .bind_geometry(self.geometry)
                    .map_err(|_| OpeningError::Identity)?,
            )?;
            let controls = controls.with_preparation_scopes(
                crate::composition::mlx::session::preparation_scope_facts()
                    .map_err(|_| OpeningError::Overflow)?,
            )?;
            let controls = controls.with_prediction_scopes(
                crate::composition::mlx::session::prediction_scope_facts()
                    .map_err(|_| OpeningError::Overflow)?,
            )?;
            let controls = match claim {
                Some(claim) => controls.with_generation_sequence(claim)?,
                None => controls,
            };
            let controls = match tracking {
                Some(facts) => controls.with_submission_tracking(facts)?,
                None => controls,
            };
            let controls = match graph {
                Some(facts) => controls.with_graph_metadata(facts)?,
                None => controls,
            };
            let controls = match prefill {
                Some(facts) => controls.with_prefill_scopes(facts)?,
                None => controls,
            };
            let controls = match native {
                Some(plan) => controls.with_native_storage(plan)?,
                None => controls,
            };
            let descriptors = descriptor_count(self.slots)?;
            let pins = PreparedPrefillStoragePinPlan::<Key>::prepare(span, |_| Some(descriptors))?;
            let batches =
                PreparedPrefillStoragePublicationPlan::<Key>::prepare(span, |_| Some(descriptors))?;
            Ok(controls
                .with_capture_plan_publication(publication)?
                .with_prefill_storage_pins(pins)?
                .with_prefill_storage_publications(batches)?)
        };
        let controls = result().map_err(row_error)?;
        let quote = quote
            .with_span_workspace_and_text_controls(controls.clone())
            .map_err(row_error)?;
        Ok((
            SealedOpeningRows {
                plan: self,
                controls,
            },
            quote,
        ))
    }
}

pub(crate) struct SealedOpeningRows {
    plan: NativeOpeningRowsPlan,
    controls: PreparedTextControlWorkspace,
}
impl SealedOpeningRows {
    pub(crate) fn allocate(
        self,
        accepted: &mut OwnedTextSpanWorkspace,
    ) -> Result<NativeOpeningRowsOwner, Error> {
        self.allocate_owned(accepted).map_err(row_error)
    }

    pub(crate) fn allocate_with_sequence(
        self,
        accepted: &mut OwnedTextSpanWorkspace,
    ) -> Result<NativeOpeningRowsOwner, eredu_core::BackendFailure> {
        self.allocate_owned(accepted).map_err(|cause| {
            eredu_core::BackendFailure::new(eredu_core::BackendFailureKind::InvalidSession, cause)
        })
    }

    pub(crate) fn sequence_error_control_bytes() -> Option<u64> {
        u64::try_from(eredu_core::BackendFailure::source_retention_peak_bytes::<
            RowAllocationFailure,
        >()?)
        .ok()
    }

    fn allocate_owned(
        self,
        accepted: &mut OwnedTextSpanWorkspace,
    ) -> Result<NativeOpeningRowsOwner, RowAllocationFailure> {
        let custody = accepted.control_guard();
        let result = (|| -> Result<_, RowError> {
            let actual = accepted
                .workspace()
                .text_controls()
                .ok_or(OpeningError::Identity)?;
            if !self.controls.same_binding(actual) {
                return Err(OpeningError::Identity.into());
            }
            custody
                .validate_reservation(accepted.reservation())
                .map_err(OpeningError::from)?;
            let request = InferenceRequest::from(accepted.reservation());
            request
                .validate(&self.plan.identity, self.plan.geometry)
                .map_err(OpeningError::from)?;
            let pins = accepted
                .take_prefill_storage_pins::<Key>()
                .map_err(OpeningError::from)?;
            let publications = accepted
                .take_prefill_storage_publications::<Key>()
                .map_err(OpeningError::from)?;
            let count = self
                .controls
                .plan()
                .records()
                .iter()
                .filter(|r| matches!(r.span(), InferenceWorkspaceSpan::Prefill(_)))
                .count();
            let mut rows = fixed_vec(count)?;
            for record in self.controls.plan().records() {
                if let InferenceWorkspaceSpan::Prefill(chunk) = record.span() {
                    rows.push(Some(OpeningRow::allocate(chunk.clone(), self.plan.slots)?));
                }
            }
            Ok((
                request,
                Rows {
                    rows,
                    pins,
                    publications,
                    current: None,
                    next: 0,
                },
            ))
        })();
        let (request, inner) = match result {
            Ok(value) => value,
            Err(cause) => {
                return Err(RowAllocationFailure {
                    cause,
                    plan: self,
                    custody,
                })
            }
        };
        Ok(NativeOpeningRowsOwner::new(NativeOpeningRows {
            inner: RefCell::new(inner),
            installed: Cell::new(false),
            #[cfg(test)]
            fixture_error_taken: Cell::new(false),
            request,
            paths: self.plan.paths,
            identity: self.plan.identity,
            binding: self.plan.binding,
            controls: self.controls,
            #[cfg(test)]
            retirement_probe: RefCell::new(None),
            custody,
        }))
    }
}

struct RowAllocationFailure {
    cause: RowError,
    plan: SealedOpeningRows,
    // Explicit accepted custody also covers an early foreign/unpromoted-plan rejection.
    custody: OriginalTextControlGuard,
}
impl std::fmt::Debug for RowAllocationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowAllocationFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Display for RowAllocationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("original native row allocation failed")
    }
}
impl std::error::Error for RowAllocationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

struct Attachment {
    allocation: Cell<Option<BoundedPublishedAllocation<Key>>>,
}
impl Attachment {
    fn empty() -> Self {
        Self {
            allocation: Cell::new(None),
        }
    }
    fn fill(
        &self,
        allocation: BoundedPublishedAllocation<Key>,
    ) -> Result<(), BoundedPublishedAllocation<Key>> {
        if let Some(old) = self.allocation.take() {
            self.allocation.set(Some(old));
            return Err(allocation);
        }
        self.allocation.set(Some(allocation));
        Ok(())
    }
}
#[derive(Clone, Copy)]
struct OwnerIndex {
    ordinal: usize,
    attachment: Option<usize>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Empty,
    RetainingOpening,
    Opening,
    Pinned,
    RetainingEnd,
    End,
    Publishing,
    Published,
}
struct OpeningRow {
    opening: FixedOpeningOwners,
    completed: FixedOpeningOwners,
    attachments: Vec<Option<PreparedAllocationOwner<Attachment>>>,
    indices: Vec<OwnerIndex>,
    outputs: Vec<Option<BoundedPublishedAllocation<Key>>>,
    rejected_key: Option<Key>,
    rejected_allocation: Option<BoundedPublishedAllocation<Key>>,
    pin: Option<BoundedPinAttempt<Key>>,
    group: Option<BoundedRegisteredStorage<Key>>,
    publication: Option<BoundedPublicationAttempt<Key>>,
    chunk: PrefillChunk,
    epoch: Option<eredu_core::DistributedCommitEpoch>,
    phase: Phase,
}
impl OpeningRow {
    fn allocate(chunk: PrefillChunk, slots: OpeningSlots) -> Result<Self, RowError> {
        let n = descriptor_count(slots)?;
        let native = slots
            .arrays
            .checked_add(slots.hosts)
            .ok_or(OpeningError::Overflow)?;
        let mut row = Self {
            opening: FixedOpeningOwners::empty(slots),
            completed: FixedOpeningOwners::empty(slots),
            attachments: fixed_vec(native)?,
            indices: fixed_vec(n)?,
            outputs: fixed_vec(n)?,
            rejected_key: None,
            rejected_allocation: None,
            pin: None,
            group: None,
            publication: None,
            chunk,
            epoch: None,
            phase: Phase::Empty,
        };
        row.opening.reserve()?;
        row.completed.reserve()?;
        for _ in 0..native {
            match PreparedAllocationOwner::try_new(Attachment::empty()) {
                Ok(node) => row.attachments.push(Some(node)),
                Err(error) => {
                    let (cause, owner) = error.into_parts();
                    drop(owner);
                    return Err(RowError::Attachment(cause));
                }
            }
        }
        Ok(row)
    }
}
struct Rows {
    rows: Vec<Option<OpeningRow>>,
    pins: OriginalPrefillStoragePinSlots<Key>,
    publications: OriginalPrefillStoragePublicationSlots<Key>,
    current: Option<usize>,
    next: usize,
}
/// Sole original bank; mechanisms keep only a Weak installation and work/quote
/// retain this Rc. No edge points back to the model or FundedWork.
pub(crate) struct NativeOpeningRows {
    inner: RefCell<Rows>,
    installed: Cell<bool>,
    #[cfg(test)]
    fixture_error_taken: Cell<bool>,
    request: InferenceRequest,
    paths: SharedLayeredObservationPaths,
    identity: InferenceExecutionIdentity,
    binding: PreparedObservationBindingIdentity,
    controls: PreparedTextControlWorkspace,
    #[cfg(test)]
    retirement_probe: RefCell<Option<owner::RetirementProbe>>,
    custody: OriginalTextControlGuard,
}
impl std::fmt::Debug for NativeOpeningRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeOpeningRows")
            .field("geometry", &self.controls.geometry())
            .finish_non_exhaustive()
    }
}
#[derive(Debug, thiserror::Error)]
pub(super) enum RowError {
    #[error(transparent)]
    Opening(#[from] OpeningError),
    #[error(transparent)]
    Pin(#[from] eredu_runtime::working_memory::BoundedPinError),
    #[error(transparent)]
    FailedPin(#[from] eredu_runtime::working_memory::FailedBoundedPinAttempt<Key>),
    #[error(transparent)]
    Publication(#[from] eredu_runtime::working_memory::BoundedPublicationError),
    #[error(transparent)]
    Attachment(#[from] PreparedAllocationOwnerCause),
    #[error("native opening row is borrowed")]
    Busy,
    #[error("completed non-native owner changed before acquisition was bound")]
    ChangedNonNative,
}
#[derive(Debug, thiserror::Error)]
#[error("original native opening row failed")]
struct RowFailure {
    #[source]
    cause: RowError,
}
#[derive(Debug, thiserror::Error)]
#[error("original native opening composition failed")]
struct NativeRowFailure {
    #[source]
    cause: Error,
}
enum RowHandoffFailure {
    Row(RowError),
    Native(eredu_runtime::working_memory::WorkingMemoryError),
}
fn row_error(e: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Other(Box::new(e))
}
fn fixed_vec<T>(count: usize) -> Result<Vec<T>, OpeningError> {
    let mut v = Vec::new();
    v.try_reserve_exact(count)?;
    if v.capacity() != count {
        return Err(OpeningError::Capacity);
    }
    Ok(v)
}

impl NativeOpeningRows {
    /// Fixed leaf plus one paired shared-inspection wrapper. Dynamic diagnostic
    /// buffers, independently escaped clones and arbitrary recursion are excluded.
    pub(crate) fn error_control_bytes() -> Option<usize> {
        type RuntimeError =
            eredu_runtime::ReplicatedTextSessionError<eredu_nn::Error, Error, Error>;
        let row = [
            size_of::<RowError>(),
            size_of::<RowFailure>(),
            size_of::<Box<RowFailure>>(),
            size_of::<NativeRowFailure>(),
            size_of::<Box<NativeRowFailure>>(),
            size_of::<RowHandoffFailure>(),
            size_of::<Result<(), RowHandoffFailure>>(),
            size_of::<eredu_runtime::working_memory::WorkingMemoryError>(),
            size_of::<Box<eredu_runtime::working_memory::WorkingMemoryError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        // Direct paired installation adds one shared inspector error Box. The
        // executing model's runtime wrapper is accounted at the outer boundary.
        row.checked_add(size_of::<RuntimeError>())?
            .checked_add(size_of::<Box<RuntimeError>>())?
            .checked_add(size_of::<Result<(), Error>>())
    }
    // The actual claimed operation or consumed installation owns the sole
    // error allowance. Diagnostics never clone that custody on rejection.
    pub(super) fn retain_error(&self, cause: Error) -> Error {
        row_error(NativeRowFailure { cause })
    }
    pub(super) fn fail(&self, cause: RowError) -> Error {
        row_error(RowFailure { cause })
    }
    pub(crate) fn validate_session(
        &self,
        identity: &InferenceExecutionIdentity,
        paths: &PreparedLayeredObservationPaths,
    ) -> Result<(), Error> {
        self.request
            .validate(identity, self.controls.geometry())
            .map_err(|e| self.fail(OpeningError::from(e).into()))?;
        if !paths.source().same_storage(&self.paths) || !self.binding.matches(paths) {
            return Err(self.fail(OpeningError::Identity.into()));
        }
        self.custody
            .validate_reservation(
                self.request
                    .memory_reservation()
                    .ok_or_else(|| self.fail(OpeningError::Identity.into()))?,
            )
            .map_err(|e| self.fail(OpeningError::from(e).into()))
    }
    pub(crate) fn validate_prepared_scope(
        &self,
        scope: &eredu_runtime::working_memory::PreparedWorkingMemoryFundingScope,
    ) -> Result<(), Error> {
        self.custody
            .validate_prepared_scope(scope)
            .map_err(|e| self.fail(OpeningError::from(e).into()))
    }
    pub(crate) fn validate_scope(&self, scope: &WorkingMemoryFundingScope) -> Result<(), Error> {
        self.custody
            .validate_native_scope(scope)
            .map_err(|e| self.fail(OpeningError::from(e).into()))
    }
    fn validate_view(
        &self,
        execution: &PrefillOpeningExecution<'_, MlxTensor>,
    ) -> Result<(), Error> {
        let paths = execution
            .prepared_paths()
            .ok_or_else(|| self.fail(OpeningError::Identity.into()))?;
        self.validate_session(&self.identity, paths)
    }
    pub(in super::super) fn collect<S: MlxStateMechanisms>(
        &self,
        state: &S,
        execution: &PrefillOpeningExecution<'_, MlxTensor>,
        context: &PrefillChunkRetentionContext<'_>,
        store: &dyn CheckpointSource,
        manager: &crate::backend::runtime::residency::manager::ResidencyManager,
    ) -> Result<(), Error> {
        self.validate_view(execution)?;
        self.request
            .validate_same_request(context.request())
            .map_err(|e| self.fail(OpeningError::from(e).into()))?;
        let result = || -> Result<(), RowError> {
            let mut rows = self.inner.try_borrow_mut().map_err(|_| RowError::Busy)?;
            if rows.current.is_some() {
                return Err(OpeningError::Used.into());
            }
            let index = rows.next;
            let row = rows
                .rows
                .get(index)
                .and_then(Option::as_ref)
                .ok_or(OpeningError::Used)?;
            if row.chunk != *context.chunk() || row.phase != Phase::Empty {
                return Err(OpeningError::Identity.into());
            }
            rows.current = Some(index);
            let row = rows.rows[index].as_mut().expect("installed row");
            row.epoch = Some(context.epoch());
            row.phase = Phase::RetainingOpening;
            collect_owners(
                &mut row.opening,
                state,
                execution,
                store,
                manager,
                context.chunk().position,
            )?;
            row.phase = Phase::Opening;
            Ok(())
        };
        result().map_err(|e| self.fail(e))
    }
    pub(in super::super) fn collect_completed<S: MlxStateMechanisms>(
        &self,
        state: &S,
        execution: &PrefillOpeningExecution<'_, MlxTensor>,
        ticket: &SettledPrefillChunkRetention,
        store: &dyn CheckpointSource,
        manager: &crate::backend::runtime::residency::manager::ResidencyManager,
    ) -> Result<(), Error> {
        self.validate_view(execution)?;
        let result = || -> Result<(), RowError> {
            let mut rows = self.inner.try_borrow_mut().map_err(|_| RowError::Busy)?;
            let index = rows.current.ok_or(OpeningError::Used)?;
            let row = rows.rows[index].as_mut().ok_or(OpeningError::Used)?;
            if row.chunk != *ticket.chunk()
                || row.epoch != Some(ticket.epoch())
                || row.phase != Phase::Pinned
            {
                return Err(OpeningError::Identity.into());
            }
            row.phase = Phase::RetainingEnd;
            let end = row
                .chunk
                .position
                .checked_add(row.chunk.input.end - row.chunk.input.start)
                .ok_or(OpeningError::Overflow)?;
            collect_owners(&mut row.completed, state, execution, store, manager, end)?;
            row.phase = Phase::End;
            Ok(())
        };
        result().map_err(|e| self.fail(e))
    }

    pub(crate) fn pin_opening(
        &self,
        context: &PrefillChunkRetentionContext<'_>,
        native: &mut WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
    ) -> Result<(), Error> {
        self.request
            .validate_same_request(context.request())
            .map_err(|e| self.fail(OpeningError::from(e).into()))?;
        let mut result = || -> Result<(), RowError> {
            self.custody
                .validate_native_scope(native)
                .map_err(OpeningError::from)?;
            let mut rows = self.inner.try_borrow_mut().map_err(|_| RowError::Busy)?;
            let index = rows.current.ok_or(OpeningError::Used)?;
            let Rows {
                rows: all,
                pins,
                publications,
                ..
            } = &mut *rows;
            let row = all[index].as_mut().ok_or(OpeningError::Used)?;
            if row.phase != Phase::Opening
                || row.chunk != *context.chunk()
                || row.epoch != Some(context.epoch())
            {
                return Err(OpeningError::Identity.into());
            }
            row.pin = Some(pins.begin(context, native, segment)?);
            row.publication = Some(publications.begin(context, native, segment)?);
            let attempt = row.pin.as_mut().expect("installed pin");
            row.opening.visit(&mut |entry| {
                if let Some((key, bytes)) = storage_entry(entry)? {
                    if let Err(key) = attempt.push_owned(key, bytes) {
                        row.rejected_key = Some(key);
                        return Err(OpeningError::Capacity);
                    }
                }
                Ok(())
            })?;
            let attempt = row.pin.take().expect("installed pin");
            row.group = Some(attempt.pin_registered(native, segment)?);
            segment.install_opening_group(native, context, &mut row.group)?;
            row.phase = Phase::Pinned;
            Ok(())
        };
        result().map_err(|e| self.fail(e))
    }
}

fn collect_owners<S: MlxStateMechanisms>(
    owners: &mut FixedOpeningOwners,
    state: &S,
    execution: &PrefillOpeningExecution<'_, MlxTensor>,
    store: &dyn CheckpointSource,
    manager: &crate::backend::runtime::residency::manager::ResidencyManager,
    frontier: u64,
) -> Result<(), OpeningError> {
    owners.begin()?;
    state
        .validate_text_frontier(frontier)
        .map_err(OpeningError::StateOwner)?;
    let mut first = None;
    let result = state.visit_all_retained_values(&mut |v| {
        if first.is_none() {
            first = owners.retain_array(v.as_array()).err();
        }
    });
    if let Some(e) = first {
        return Err(e);
    }
    result?;
    if let Some(layout) = state.shared_layout() {
        owners.retain_layout(layout)?;
    }
    let mut first = None;
    let result = state.visit_slot_metadata(&mut |v| {
        if first.is_none() {
            first = owners.retain_table(v).err();
        }
        Ok(())
    });
    if let Some(e) = first {
        return Err(e);
    }
    result.map_err(OpeningError::StateOwner)?;
    owners.retain_checkpoint(store)?;
    if !manager.try_visit_retained_storage(&mut |v| owners.retain(v))? {
        return Err(OpeningError::Incomplete);
    }
    let mut first = None;
    let complete = execution.visit(&mut |v| {
        if first.is_none() {
            first = owners.retain_array(v.as_array()).err();
        }
    });
    if let Some(e) = first {
        return Err(e);
    }
    if !complete {
        return Err(OpeningError::Incomplete);
    }
    owners.inspect_facts()
}

// Moved out only after successful native publication/attachment. Field ordering
// keeps captured owners and pending nodes ahead of original host custody.
pub(crate) struct RetiredOpeningRow {
    row: OpeningRow,
    custody: OriginalTextControlGuard,
}

impl NativeOpeningRows {
    /// Called only after the carrier's actual nonblocking completed-submission
    /// check and before it takes the old parcel. Failure leaves every prefix in
    /// this bank; the carrier's active marker remains untouched.
    pub(crate) fn publish_completed(
        &self,
        ticket: &SettledPrefillChunkRetention,
        native: &WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
    ) -> Result<(), Error> {
        let result = || -> Result<(), RowError> {
            self.custody
                .validate_native_scope(native)
                .map_err(OpeningError::from)?;
            segment
                .validate_settled_ticket(native, ticket)
                .map_err(OpeningError::from)?;
            let mut rows = self.inner.try_borrow_mut().map_err(|_| RowError::Busy)?;
            let index = rows.current.ok_or(OpeningError::Used)?;
            let row = rows.rows[index].as_mut().ok_or(OpeningError::Used)?;
            if row.phase != Phase::End
                || row.chunk != *ticket.chunk()
                || row.epoch != Some(ticket.epoch())
            {
                return Err(OpeningError::Identity.into());
            }
            row.phase = Phase::Publishing;
            let attempt = row.publication.as_mut().ok_or(OpeningError::Used)?;
            let mut ordinal = 0;
            let mut attachment = 0;
            row.completed.visit(&mut |entry| -> Result<(), RowError> {
                let native = matches!(&entry, OpeningEntry::Array(..) | OpeningEntry::Host(..));
                let owner_index = OwnerIndex {
                    ordinal,
                    attachment: native.then_some(attachment),
                };
                ordinal += 1;
                if native {
                    attachment += 1;
                }
                let Some((key, bytes)) = storage_entry(entry)? else {
                    return Ok(());
                };
                if !native {
                    // No new source/metadata acquisition is smuggled through
                    // the native allocation sidecar. The old group stays live.
                    let mut found = false;
                    row.opening
                        .visit(&mut |entry| -> Result<(), OpeningError> {
                            if let Some((old, capacity)) = storage_entry(entry)? {
                                found |= old == key && capacity == bytes;
                            }
                            Ok(())
                        })?;
                    if !found {
                        row.rejected_key = Some(key);
                        return Err(RowError::ChangedNonNative);
                    }
                }
                if row.indices.len() == row.indices.capacity() {
                    row.rejected_key = Some(key);
                    return Err(OpeningError::Capacity.into());
                }
                if let Err(key) = attempt.push_owned(key, bytes) {
                    row.rejected_key = Some(key);
                    return Err(OpeningError::Capacity.into());
                }
                row.indices.push(owner_index);
                row.outputs.push(None);
                Ok(())
            })?;
            attempt.publish(native, segment)?;
            for unique in 0..attempt.published_count() {
                let input = attempt
                    .published_input_index(unique)
                    .ok_or(OpeningError::Identity)?;
                let mapping = *row.indices.get(input).ok_or(OpeningError::Identity)?;
                let allocation = attempt
                    .take_allocation(unique)
                    .ok_or(OpeningError::Identity)?;
                // Keep an independent output in the external row BEFORE any
                // consuming native handoff; a panic cannot uncharge its backing.
                row.outputs[unique] = Some(allocation);
                let Some(slot) = mapping.attachment else {
                    continue;
                };
                let node = row
                    .attachments
                    .get_mut(slot)
                    .and_then(Option::as_mut)
                    .ok_or(OpeningError::Identity)?;
                if let Err(rejected) = node.owner().fill(
                    row.outputs[unique]
                        .as_ref()
                        .expect("installed output")
                        .clone(),
                ) {
                    row.rejected_allocation = Some(rejected);
                    return Err(OpeningError::Used.into());
                }
                #[cfg(test)]
                if ATTACH_BUSY_AFTER.get() == Some(unique) {
                    ATTACH_BUSY_AFTER.set(None);
                    return Err(RowError::Attachment(
                        PreparedAllocationOwnerCause::RuntimeBusy,
                    ));
                }
                // Validate the mapping before moving its node. Canonical alias
                // deduplication is owned entirely by the original batch.
                let entry = row.completed.entry_at(mapping.ordinal)?;
                let node = row.attachments[slot].take().expect("prepared native node");
                let attached = match entry {
                    OpeningEntry::Array(array, _) => node.try_attach(array),
                    OpeningEntry::Host(host, _) => host.try_attach_prepared_allocation_owner(node),
                    _ => {
                        row.attachments[slot] = Some(node);
                        return Err(OpeningError::Identity.into());
                    }
                };
                if let Err(error) = attached {
                    let (cause, node) = error.into_parts();
                    row.attachments[slot] = Some(node);
                    return Err(RowError::Attachment(cause));
                }
                #[cfg(test)]
                if PANIC_AFTER_ATTACH.replace(false) {
                    std::panic::panic_any(173_u32);
                }
            }
            row.phase = Phase::Published;
            Ok(())
        };
        result().map_err(|e| self.fail(e))
    }

    /// Acquire/validate the row loan before the canonical parcel take. After
    /// that take succeeds there are only infallible moves, so no rejected late
    /// borrow can retire the marker while this published prefix stays behind.
    pub(crate) fn retire_published<P>(
        &self,
        ticket: &SettledPrefillChunkRetention,
        take_parcel: impl FnOnce() -> Result<P, eredu_runtime::working_memory::WorkingMemoryError>,
    ) -> Result<(RetiredOpeningRow, P), Error> {
        #[cfg(test)]
        if HANDOFF_BUSY.replace(false) {
            return Err(self.fail(RowError::Busy));
        }
        let result = (|| -> Result<_, RowHandoffFailure> {
            let mut rows = self
                .inner
                .try_borrow_mut()
                .map_err(|_| RowHandoffFailure::Row(RowError::Busy))?;
            let index = rows
                .current
                .ok_or_else(|| RowHandoffFailure::Row(OpeningError::Used.into()))?;
            let row = rows.rows[index]
                .as_ref()
                .ok_or_else(|| RowHandoffFailure::Row(OpeningError::Used.into()))?;
            if row.phase != Phase::Published
                || row.chunk != *ticket.chunk()
                || row.epoch != Some(ticket.epoch())
            {
                return Err(RowHandoffFailure::Row(OpeningError::Identity.into()));
            }
            let next = rows
                .next
                .checked_add(1)
                .ok_or_else(|| RowHandoffFailure::Row(OpeningError::Overflow.into()))?;
            let custody = self.custody.clone();
            let parcel = take_parcel().map_err(RowHandoffFailure::Native)?;
            let row = rows.rows[index].take().expect("validated row");
            rows.current = None;
            rows.next = next;
            drop(rows);
            Ok((RetiredOpeningRow { row, custody }, parcel))
        })();
        result.map_err(|cause| match cause {
            RowHandoffFailure::Row(cause) => self.fail(cause),
            RowHandoffFailure::Native(cause) => self.retain_error(Error::Other(Box::new(cause))),
        })
    }
}

#[cfg(test)]
thread_local! {
    static PANIC_AFTER_ATTACH: Cell<bool> = const { Cell::new(false) };
    static HANDOFF_BUSY: Cell<bool> = const { Cell::new(false) };
    static ATTACH_BUSY_AFTER: Cell<Option<usize>> = const { Cell::new(None) };
}
#[cfg(test)]
impl NativeOpeningRows {
    pub(crate) fn observe_retirement_for_test(&self, drops: std::rc::Rc<Cell<usize>>) {
        assert!(self
            .retirement_probe
            .borrow_mut()
            .replace(owner::RetirementProbe(drops))
            .is_none());
    }
    /// Exact low-level fixture admission only; production uses its real claimed
    /// TextOperation/consumed InstalledCapture. No arbitrary guard or refill.
    pub(crate) fn claim_fixture_error_controls(
        &self,
        accepted: &OwnedTextSpanWorkspace,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<OriginalTextControlGuard, Error> {
        let actual = accepted
            .workspace()
            .text_controls()
            .ok_or_else(|| self.fail(OpeningError::Identity.into()))?;
        if !self.controls.same_binding(actual)
            || self.controls.source_identity() != Some(source.storage_identity())
        {
            return Err(self.fail(OpeningError::Identity.into()));
        }
        self.custody
            .validate_reservation(accepted.reservation())
            .map_err(|e| self.fail(OpeningError::from(e).into()))?;
        if self.fixture_error_taken.replace(true) {
            return Err(self.fail(OpeningError::Used.into()));
        }
        Ok(accepted.control_guard())
    }
    pub(crate) fn force_attachment_panic_for_test() {
        PANIC_AFTER_ATTACH.set(true);
    }
    pub(crate) fn force_handoff_busy_for_test() {
        HANDOFF_BUSY.set(true);
    }
    pub(crate) fn force_attachment_busy_after_for_test(prefix: usize) {
        ATTACH_BUSY_AFTER.set(Some(prefix));
    }
    pub(crate) fn snapshot_for_test(
        &self,
    ) -> (
        usize,
        Option<&'static str>,
        Vec<(Key, u64)>,
        Vec<(Key, u64)>,
    ) {
        let rows = self.inner.borrow();
        let Some(index) = rows.current else {
            return (rows.next, None, vec![], vec![]);
        };
        let row = rows.rows[index].as_ref().unwrap();
        let keys = |owners: &FixedOpeningOwners| {
            let mut keys = vec![];
            if owners
                .visit(&mut |entry| -> Result<(), OpeningError> {
                    if let Some(pair) = storage_entry(entry)? {
                        keys.push(pair);
                    }
                    Ok(())
                })
                .is_err()
            {
                keys.clear();
            }
            keys
        };
        let phase = match row.phase {
            Phase::Empty => "empty",
            Phase::RetainingOpening => "retaining opening",
            Phase::Opening => "opening",
            Phase::Pinned => "pinned",
            Phase::RetainingEnd => "retaining end",
            Phase::End => "end",
            Phase::Publishing => "publishing",
            Phase::Published => "published",
        };
        (
            rows.next,
            Some(phase),
            keys(&row.opening),
            keys(&row.completed),
        )
    }
}
