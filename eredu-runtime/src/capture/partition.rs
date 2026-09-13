//! Bounded native capture fragments retaining their original global admission.
//! Transport and all-rank completion must be established by the enclosing driver.

use super::*;

mod assembly;
mod sum;
pub use assembly::{assemble_reduced_fragments, assemble_tensor_fragments};
mod exchange;
mod observer;
mod receipt;
mod routed;
pub use routed::PartitionRoutedCaptureProducer;
mod vocabulary;
pub use observer::{
    PartitionCaptureLayout, PartitionCaptureObserver, PartitionCaptureObserverError,
    PartitionCapturePlacement, PartitionRoutedCapturePlacement, PartitionRoutedCaptureSource,
};
mod session;
pub use exchange::{
    PartitionCaptureExchange, PartitionCaptureExchangeError, PartitionCaptureExchangeStage,
    PartitionCaptureTransport,
};
pub use receipt::{
    PartitionCaptureDelivery, PartitionCaptureProducer, PartitionCaptureReceiptLimits,
    PartitionCaptureReceiptPlan, ReceivedPartitionCapture,
};
pub(super) use session::PartitionCaptureRun;
pub use session::{
    PartitionCaptureHookTransport, PartitionCaptureIdentity, PartitionCaptureNativeEstimate,
    SessionPartitionCapture, SessionPartitionCoordination, SessionPartitionHook,
    SessionPartitionIntervention, SessionPartitionSource,
};

/// Exact capture operation and validated native geometry for one fragment.
pub struct PartitionCaptureRequest<'a> {
    /// Original immutable global admission, never a rewritten local plan.
    pub plan: &'a AdmittedCapturePlan,
    /// Selection ordinal in that admission.
    pub selection_index: usize,
    /// Current committed forward phase.
    pub phase: CapturePhase,
    /// Run-relative prediction ordinal, distinct from tensor positions.
    pub prediction: u64,
    /// Explicit axes for independently shaped forwards; ordinary requests use None.
    pub invocation: Option<CaptureInvocationShape>,
    /// Validated global-to-local selection geometry.
    pub projection: &'a CaptureSlicePartition,
    /// Fragment ordinal within the local projection.
    pub fragment_index: usize,
    /// Retained producer rank, not inferred from tensor contents.
    pub producer_rank: usize,
}

/// A local result with explicit global identity and placement. It cannot be
/// mistaken for a complete global `CaptureRecord`; assembly checks coverage.
#[derive(Debug, Clone)]
pub struct CapturedPartitionFragment {
    combination: PartitionCaptureCombination,
    plan_identity: String,
    selection_index: usize,
    phase: CapturePhase,
    prediction: u64,
    invocation: Option<CaptureInvocationShape>,
    producer_rank: usize,
    axis: usize,
    global_shape: Vec<u64>,
    global_slice: ResolvedCaptureSlice,
    geometry: CaptureFragmentGeometry,
    record: CaptureRecord,
}

impl CapturedPartitionFragment {
    /// Original global capture admission identity.
    pub fn plan_identity(&self) -> &str {
        &self.plan_identity
    }
    /// Producer whose actual local tensor was transformed.
    pub const fn producer_rank(&self) -> usize {
        self.producer_rank
    }
    /// Global source shape, distinct from the record's native source shape.
    pub fn global_shape(&self) -> &[u64] {
        &self.global_shape
    }
    /// Validated local slice and exact global selected-result destination.
    pub const fn geometry(&self) -> &CaptureFragmentGeometry {
        &self.geometry
    }
    /// Local capture evidence, including actual source dtype and charged work.
    pub const fn record(&self) -> &CaptureRecord {
        &self.record
    }
}

/// Retained evidence for one producer's contribution to a global result.
#[derive(Debug, Clone)]
pub struct PartitionCaptureContribution {
    producer_rank: usize,
    geometry: CaptureFragmentGeometry,
    charged: CaptureUsage,
    routed: Option<RoutedUnitCaptureProvenance>,
}

impl PartitionCaptureContribution {
    /// Producer of this exact selected region.
    pub const fn producer_rank(&self) -> usize {
        self.producer_rank
    }
    /// Local-source and global-destination geometry retained after assembly.
    pub const fn geometry(&self) -> &CaptureFragmentGeometry {
        &self.geometry
    }
    /// Work reserved for this fragment before its native transformation.
    pub const fn charged(&self) -> CaptureUsage {
        self.charged
    }
    /// Retained sparse ownership, when this fragment contains participating routes.
    pub fn routed(&self) -> Option<&RoutedUnitCaptureProvenance> {
        self.routed.as_ref()
    }
}

/// Complete global tensor evidence, retaining producer coverage rather than
/// silently dropping partition provenance when local buffers are consumed.
#[derive(Debug, Clone)]
pub struct AssembledPartitionCapture {
    combination: PartitionCaptureCombination,
    plan_identity: String,
    phase: CapturePhase,
    prediction: u64,
    record: CaptureRecord,
    contributions: Vec<PartitionCaptureContribution>,
}

impl AssembledPartitionCapture {
    /// Whether regions cover disjoint coordinates or are measured additive terms.
    pub const fn combination(&self) -> PartitionCaptureCombination {
        self.combination
    }
    /// Original global admission identity.
    pub fn plan_identity(&self) -> &str {
        &self.plan_identity
    }
    /// Captured forward phase.
    pub const fn phase(&self) -> CapturePhase {
        self.phase
    }
    /// Captured prediction ordinal.
    pub const fn prediction(&self) -> u64 {
        self.prediction
    }
    /// Complete global record; publication still needs distributed agreement.
    pub const fn record(&self) -> &CaptureRecord {
        &self.record
    }
    /// Exact source regions retained from each producer, under `combination`.
    pub fn contributions(&self) -> &[PartitionCaptureContribution] {
        &self.contributions
    }
    /// Moves global data and producer evidence together into the delivery owner.
    pub fn into_parts(self) -> (CaptureRecord, Vec<PartitionCaptureContribution>) {
        (self.record, self.contributions)
    }
}

/// Executes one fragment through the ordinary reservation/transform mechanism.
/// The caller owns the shared ledger and must reserve distributed transport and
/// establish peer failure agreement separately. No fragment authorizes publication
/// of a complete global result.
pub fn capture_fragment<B: CaptureBackend>(
    backend: &mut B,
    tensor: &B::Tensor,
    request: PartitionCaptureRequest<'_>,
    ledger: &mut dyn CaptureReservation,
) -> Result<CapturedPartitionFragment, CaptureExecutionError<B::Error>> {
    capture_fragment_combined(
        backend,
        tensor,
        request,
        ledger,
        PartitionCaptureCombination::Disjoint,
    )
}

/// Captures an actual floating summand through a separately admitted sum receipt.
/// Nonlinear transforms are deferred until all selected native terms are assembled.
pub fn capture_sum_fragment<B: CaptureBackend>(
    backend: &mut B,
    tensor: &B::Tensor,
    request: PartitionCaptureRequest<'_>,
    ledger: &mut dyn CaptureReservation,
) -> Result<CapturedPartitionFragment, CaptureExecutionError<B::Error>> {
    capture_fragment_combined(
        backend,
        tensor,
        request,
        ledger,
        PartitionCaptureCombination::SumF64ToF32,
    )
}

fn validate_partition_request(request: &PartitionCaptureRequest<'_>) -> Result<(), CaptureError> {
    let PartitionCaptureRequest {
        plan,
        selection_index,
        phase,
        prediction,
        invocation,
        projection,
        fragment_index,
        ..
    } = *request;
    let selection = plan
        .plan()
        .selections
        .get(selection_index)
        .ok_or_else(|| CaptureError::Invalid("unknown partition capture selection".into()))?;
    let point = &plan.points()[selection_index];
    if prediction >= plan.request().max_predictions
        || !selection.schedule.includes(phase, prediction)
    {
        return Err(CaptureError::Invalid(
            "partition capture is outside its admitted schedule".into(),
        ));
    }
    vocabulary::validate_projection(&selection.transform, projection)?;
    plan.geometry_at(phase, prediction, invocation)?
        .validate_actual(point, projection.global_shape())?;
    if resolve_slice(point, selection, projection.global_shape())? != *projection.global_slice() {
        return Err(CaptureError::Invalid(
            "partition capture changed its admitted global slice".into(),
        ));
    }
    if fragment_index >= projection.fragments().len() {
        return Err(CaptureError::Invalid(
            "unknown partition capture fragment".into(),
        ));
    }
    Ok(())
}

fn capture_fragment_combined<B: CaptureBackend>(
    backend: &mut B,
    tensor: &B::Tensor,
    request: PartitionCaptureRequest<'_>,
    ledger: &mut dyn CaptureReservation,
    combination: PartitionCaptureCombination,
) -> Result<CapturedPartitionFragment, CaptureExecutionError<B::Error>> {
    validate_partition_request(&request)?;
    let PartitionCaptureRequest {
        plan,
        selection_index,
        phase,
        prediction,
        invocation,
        projection,
        fragment_index,
        producer_rank,
    } = request;
    let selection = plan
        .plan()
        .selections
        .get(selection_index)
        .ok_or_else(|| CaptureError::Invalid("unknown partition capture selection".into()))?;
    let point = &plan.points()[selection_index];
    if combination == PartitionCaptureCombination::SumF64ToF32 {
        sum::validate_projection(selection, point, projection)?;
        sum::validate_precision(backend.source_dtype(tensor).as_ref())?;
    }
    let geometry = projection
        .fragments()
        .get(fragment_index)
        .ok_or_else(|| CaptureError::Invalid("unknown partition capture fragment".into()))?;
    let shape = backend
        .shape(tensor)
        .map_err(CaptureExecutionError::Backend)?;
    if shape != projection.local_shape() {
        return Err(CaptureError::Invalid(
            "partition capture local source geometry differs from its projection".into(),
        )
        .into());
    }
    // Covers retained global/local/destination geometry and the original identity
    // before cloning host metadata. Native estimates cover each selected payload.
    let metadata = fragment_metadata_usage(selection, point, projection.global_shape().len())?
        .checked_add(if combination == PartitionCaptureCombination::SumF64ToF32 {
            metadata_reservation(selection, point)?
        } else {
            CaptureUsage::default()
        })?;
    ledger.reserve_quota(metadata)?;
    let mut record = CaptureRecord {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selection_id: selection.id.clone(),
        path: selection.path.clone(),
        node_id: point.node_id.clone(),
        position: point.position,
        source_shape: None,
        source_dtype: backend.source_dtype(tensor),
        selected_shape: None,
        outcome: CaptureOutcome::Missing,
        payload: None,
        charged: metadata,
    };
    let native_selection;
    let selection = if combination == PartitionCaptureCombination::SumF64ToF32 {
        native_selection = sum::native_selection(selection);
        &native_selection
    } else {
        selection
    };
    capture_resolved_value(
        backend,
        tensor,
        selection,
        &mut record,
        shape,
        geometry.local(),
        ledger,
    )?;
    Ok(CapturedPartitionFragment {
        invocation,
        combination,
        plan_identity: plan.identity().into(),
        selection_index,
        phase,
        prediction,
        producer_rank,
        axis: projection.axis(),
        global_shape: projection.global_shape().to_vec(),
        global_slice: projection.global_slice().clone(),
        geometry: geometry.clone(),
        record,
    })
}

fn fragment_metadata_usage(
    selection: &CaptureSelection,
    point: &eredu_core::ObservationPoint,
    rank: usize,
) -> Result<CaptureUsage, CaptureError> {
    metadata_reservation(selection, point)?.checked_add(CaptureUsage {
        host_bytes: add(1024, mul(rank as u64, 512)?)?,
        encoded_bytes: add(4096, mul(rank as u64, 320)?)?,
        ..Default::default()
    })
}

/// Captures a deferred native input with exactly the same geometry, reservation,
/// error preservation and source-precision semantics as an ordinary generated
/// observation. Each call reserves `source` before invoking its factory;
/// a rejected/skipped fragment creates no tensor. The prototype declares shape,
/// not the generated input's dtype. Distributed completion remains caller-owned.
#[allow(clippy::too_many_arguments)]
pub fn capture_generated_fragment<B: CaptureBackend, E>(
    backend: &mut B,
    prototype: &B::Tensor,
    source: &eredu_core::capture::GeneratedCaptureSource,
    generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    request: PartitionCaptureRequest<'_>,
    ledger: &mut dyn CaptureReservation,
    map_error: &impl Fn(CaptureExecutionError<B::Error>) -> E,
) -> Result<CapturedPartitionFragment, E> {
    capture_generated_fragment_combined(
        backend,
        prototype,
        source,
        generate,
        request,
        ledger,
        map_error,
        PartitionCaptureCombination::Disjoint,
    )
}

#[allow(clippy::too_many_arguments)]
fn capture_generated_fragment_combined<B: CaptureBackend, E>(
    backend: &mut B,
    prototype: &B::Tensor,
    source: &eredu_core::capture::GeneratedCaptureSource,
    generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    request: PartitionCaptureRequest<'_>,
    ledger: &mut dyn CaptureReservation,
    map_error: &impl Fn(CaptureExecutionError<B::Error>) -> E,
    combination: PartitionCaptureCombination,
) -> Result<CapturedPartitionFragment, E> {
    validate_partition_request(&request)
        .map_err(|error| map_error(CaptureExecutionError::Admission(error)))?;
    super::generated::capture_generated(backend, source, generate, map_error, |generated| {
        capture_fragment_combined(generated, prototype, request, ledger, combination)
    })
}

/// A global result cannot be published from these fragments.
#[derive(Debug, thiserror::Error)]
pub enum PartitionCaptureMergeError {
    /// One expected producer has not supplied its explicit receipt.
    #[error("partition capture producer {producer_rank} has no receipt")]
    MissingProducer {
        /// Missing world rank, including producers with no selected values.
        producer_rank: usize,
    },
    /// Geometry, identity, resource or collector-contract validation failed.
    #[error(transparent)]
    Capture(#[from] CaptureError),
    /// At least one selected global element has no producer.
    #[error(
        "partition capture is incomplete: {received_elements} of {expected_elements} elements"
    )]
    Incomplete {
        /// Complete requested value count.
        expected_elements: u64,
        /// Number covered without overlap.
        received_elements: u64,
    },
    /// Two fragments claim the same selected global element.
    #[error("partition capture has overlapping global coverage")]
    Overlap,
    /// A producer returned a non-value outcome; it must not be interpreted as zero.
    #[error("partition capture producer {producer_rank} returned {outcome:?}")]
    FragmentOutcome {
        /// Producer rank.
        producer_rank: usize,
        /// Original typed local outcome.
        outcome: CaptureOutcome,
    },
}
