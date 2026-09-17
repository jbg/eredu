//! Closed ordinary prediction and bound prefill capture equations on the original workspace context.
//! No session, frame, claim, native source, reservation or execution permission.
use super::intervention::{PreparedModelInterventions, PreparedTextInterventions};
use crate::backend::array_copy::{CaptureNativePopulation, CaptureTensorSelection};
use eredu_core::{InferenceGeometry, capture::*};
use eredu_nn::{
    Tensor,
    workspace::{WorkspaceContext, WorkspaceTensor, WorkspaceFloatingType},
};
use eredu_runtime::{
    ActivationObserver,
    capture::{CaptureObservationStep, CaptureProtocolError, CaptureRecordStatus},
    working_memory::{CaptureRunHostPlan, InferenceWorkspaceObserver, InferenceWorkspaceSpan},
};
use std::cell::{Cell, RefCell};

mod fragments;
mod partition;
pub(in crate::composition::mlx::session) use partition::estimate_geometry as estimate_partition_prefill;
pub(in crate::composition::mlx::session) use partition::trace_geometry as trace_partition_prefill_geometry;
pub(in crate::composition::mlx::session) use partition::invocation::trace_geometry as trace_partition_invocation_geometry;
pub(in crate::composition::mlx::session) use partition::invocation::estimate_geometry as estimate_partition_invocation;
pub(in crate::composition::mlx) use partition::source::validate as validate_partition_capture_source;
mod routed;

type Result<T> = std::result::Result<T, eredu_nn::Error>;
fn record_population(
    counter: Option<&Cell<CaptureNativePopulation>>,
    context: &WorkspaceContext,
    population: CaptureNativePopulation,
) -> Result<()> {
    if let Some(counter) = counter {
        let next = counter
            .get()
            .checked_add(population)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(std::mem::size_of::<(
            CaptureNativePopulation,
            Option<CaptureNativePopulation>,
            Result<()>,
        )>())?;
        counter.set(next);
    }
    Ok(())
}
// A logical Float32 layout is not an actual native scalar witness. Capture
// receipts preserve the selected source precision before the fixed F32 readout.
fn record_source(
    counter: Option<&[Cell<Option<WorkspaceFloatingType>>]>,
    index: usize,
    context: &WorkspaceContext,
    value: &WorkspaceTensor,
) -> Result<()> {
    record_scalar_source(counter,index,context,value.layout().representation().map(|source|source.dtype()))
}
fn record_scalar_source(
    counter:Option<&[Cell<Option<WorkspaceFloatingType>>]>,index:usize,
    context:&WorkspaceContext,dtype:Option<WorkspaceFloatingType>,
)->Result<()> {
    let Some(counters) = counter else { return Ok(()); };
    context.charge_metadata(std::mem::size_of::<(
        Option<WorkspaceFloatingType>, &WorkspaceTensor, Result<()>, usize,
        Option<&[Cell<Option<WorkspaceFloatingType>>]>, &Cell<Option<WorkspaceFloatingType>>,
        eredu_runtime::working_memory::WorkingMemoryError,
    )>())?;
    let counter = counters.get(index).ok_or_else(|| context.metadata_source(
        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?;
    let dtype = dtype
        .ok_or_else(|| context.metadata_source(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound))?;
    if counter.get().is_some_and(|prior| prior != dtype) {
        return Err(context.metadata_source(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
    }
    counter.set(Some(dtype));
    Ok(())
}
fn record_transfer(
    counter: Option<&Cell<CaptureNativePopulation>>,
    context: &WorkspaceContext,
) -> Result<()> {
    if counter.is_none() {
        return Ok(());
    }
    let population = CaptureNativePopulation::raw(1)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
    record_population(counter, context, population)
}

fn record_replica(
    counter: Option<&Cell<CaptureNativePopulation>>,
    context: &WorkspaceContext,
    roots: &mut Vec<WorkspaceTensor>,
    value: &WorkspaceTensor,
) -> Result<()> {
    context.charge_metadata(std::mem::size_of::<(
        Option<&Cell<CaptureNativePopulation>>, &WorkspaceContext,
        &mut Vec<WorkspaceTensor>, &WorkspaceTensor, CaptureNativePopulation, Result<()>,
    )>())?;
    let population = CaptureNativePopulation {
        publications: 0, completions: 1, retained_roots: 1,
        controls: super::model_session::text_funding::capture_replica_control_bytes()
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
    };
    let metadata = Metadata::new(context)?;
    metadata.reserve(roots, 1)?;
    roots.push(value.clone());
    record_population(counter, context, population)
}

/// The same completed replica source used by an ordinary projected hook.
/// Empty selected overlap remains an actual invocation with its own scalar.
pub(in crate::composition::mlx::session) fn trace_partition_replica(
    value:&WorkspaceTensor,context:&WorkspaceContext,roots:&mut Vec<WorkspaceTensor>,
)->Result<CaptureNativePopulation> {
    context.charge_metadata(std::mem::size_of::<(
        &WorkspaceTensor,&WorkspaceContext,&mut Vec<WorkspaceTensor>,
        Cell<CaptureNativePopulation>,Result<CaptureNativePopulation>,
    )>())?;
    let counter=Cell::new(CaptureNativePopulation::default());
    record_replica(Some(&counter),context,roots,value)?;
    Ok(counter.get())
}

// The same observer owns either ordinary metadata or paid Context destinations.
// This local context alias has no native source or execution authority.
#[derive(Debug, thiserror::Error)]
#[error("capture invocation geometry/authority mismatch at {location}")]
struct CaptureCoordinateFailure {
    location: &'static std::panic::Location<'static>,
    #[source]
    source: CaptureProtocolError,
}

struct Metadata(WorkspaceContext);
impl Metadata {
    fn new(context: &WorkspaceContext) -> Result<Self> {
        context.charge_metadata(std::mem::size_of::<(
            Self,
            Result<Self>,
            CaptureObservationStep<'_>,
            eredu_core::capture::CaptureSummaryGeometry<'_>,
            crate::backend::array_copy::PreparedCaptureSummary,
            eredu_core::capture::CaptureHistogramGeometry<'_>,
            crate::backend::array_copy::PreparedCaptureHistogram<'_>,
            eredu_core::capture::CapturePrefillTransformPlan<'_>,
            eredu_core::capture::CapturePrefillTransformFragment<'_, '_>,
            CaptureUsage,
            Option<CaptureInvocationWindow>,
            std::result::Result<Option<CaptureUsage>, CaptureError>,
            std::result::Result<Option<CaptureSkipReason>, CaptureError>,
            CaptureProtocolError,
            CaptureCoordinateFailure,
            Result<()>,
        )>())?;
        context.charge_metadata(
            CaptureTensorSelection::reduction_control_bytes()
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        context.charge_metadata(
            eredu_core::capture::CaptureSummaryGeometry::preparation_control_bytes()
                .and_then(|summary| {
                    summary.checked_add(
                        eredu_core::capture::CaptureHistogramGeometry::preparation_control_bytes()?,
                    )
                })
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        Ok(Self(context.clone()))
    }
    fn error<E: std::error::Error + Send + Sync + 'static>(&self, cause: E) -> eredu_nn::Error {
        if self.0.uses_checked_metadata() {
            self.0.metadata_source(cause)
        } else {
            eredu_nn::Error::backend_source(cause)
        }
    }
    #[track_caller]
    fn coordinate(&self) -> eredu_nn::Error {
        self.error(CaptureCoordinateFailure {
            location: std::panic::Location::caller(),
            source: CaptureProtocolError::Geometry,
        })
    }
    fn reserve<T>(&self, values: &mut Vec<T>, count: usize) -> Result<()> {
        if self.0.uses_checked_metadata() {
            self.0.reserve_metadata_vec(values, count)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Invocation<'a> {
    phase: CapturePhase,
    prediction: u64,
    shape: CaptureInvocationShape,
    window: Option<CaptureInvocationWindow>,
    selected: &'a [bool],
}

pub(in crate::composition::mlx) struct CaptureWorkspaceObserver<'a> {
    source: &'a SharedCapturePlan,
    invocation: Option<Invocation<'a>>,
    transfers: Option<&'a Cell<CaptureNativePopulation>>,
    scalar_source: Option<&'a [Cell<Option<WorkspaceFloatingType>>]>,
    placement: Option<(&'a eredu_architectures::component_partition::ComponentPartitionLayouts, usize)>,
    geometry: InferenceGeometry,
    context: WorkspaceContext,
    ledger: CaptureLedger,
    envelope_usage: Option<CaptureUsage>,
    prepared_prefix: Option<&'a eredu_runtime::capture::OriginalSpeculativeCapturePrefix>,
    interventions: Option<&'a RefCell<Option<PreparedModelInterventions>>>,
    text_interventions: Option<&'a RefCell<Option<PreparedTextInterventions>>>,
    partition_intervention_source: Option<&'a eredu_runtime::RetainedCommunicationSource>,
    statuses: Vec<CaptureRecordStatus>,
    roots: Vec<WorkspaceTensor>,
    first_prediction: u64,
    next_prediction: u64,
    prefill_end: u64,
    prefill_complete: bool,
    active: Option<u64>,
    bound: Option<eredu_runtime::layered::BoundCaptureSelection<'a>>,
    progress: Vec<eredu_runtime::capture::CapturePrefillRowProgress>,
    chunk: Option<eredu_runtime::prefill::PrefillChunk>,
    routed_selection: Option<usize>,
    routed_intervention_selection: Option<usize>,
    routed_active: bool,
    // Exact original provider extent of a cold addressable source. This is not
    // a delivered native batch or received-row count.
    routed_addressable_rows:Option<u64>,
    routed_tokens: [u64; 2],
    routed_native_rows: Option<u64>,
}
impl<'a> CaptureWorkspaceObserver<'a> {
    pub(in crate::composition::mlx) fn new(
        source: &'a SharedCapturePlan,
        geometry: InferenceGeometry,
        context: &WorkspaceContext,
    ) -> Result<(Self, CaptureRunHostPlan<'a>)> {
        Self::prepare(source, geometry, context, None, None, None, None)
    }
    /// Quote the same bound physical candidate used by original admission and
    /// native stamps. No target, claim, source registration or grant is created.
    pub(in crate::composition::mlx) fn with_prefill(
        bound: eredu_runtime::layered::BoundCaptureSelection<'a>,
        context: &WorkspaceContext,
    ) -> Result<(Self, CaptureRunHostPlan<'a>)> {
        Self::prepare(
            bound.selection().source(),
            bound.geometry(),
            context,
            Some(bound),
            None,
            None,
            None,
        )
    }
    /// The actual observer records accepted tensor transfers for the same span's
    /// native completion recipe. The borrowed counter has no schedule or grant.
    pub(in crate::composition::mlx) fn with_prefill_transfers(
        bound: eredu_runtime::layered::BoundCaptureSelection<'a>,
        context: &WorkspaceContext,
        transfers: &'a Cell<CaptureNativePopulation>,
    ) -> Result<(Self, CaptureRunHostPlan<'a>)> {
        Self::prepare(
            bound.selection().source(),
            bound.geometry(),
            context,
            Some(bound),
            Some(transfers),
            None,
            None,
        )
    }
    /// Lend the actual native source scalar alongside the same span's transfer
    /// population. Receivers validate their published tensor without tracing a
    /// fictitious local readout. No layout, source credit or permission is added.
    pub(in crate::composition::mlx) fn with_prefill_source_transfers(
        bound: eredu_runtime::layered::BoundCaptureSelection<'a>,
        context: &WorkspaceContext,
        transfers: &'a Cell<CaptureNativePopulation>,
        scalar: &'a [Cell<Option<WorkspaceFloatingType>>],
        placement: (&'a eredu_architectures::component_partition::ComponentPartitionLayouts, usize),
    ) -> Result<(Self, CaptureRunHostPlan<'a>)> {
        context.charge_metadata(std::mem::size_of::<(
            Self, Result<(Self, CaptureRunHostPlan<'a>)>,
            &[Cell<Option<WorkspaceFloatingType>>],
            (&eredu_architectures::component_partition::ComponentPartitionLayouts, usize),
        )>())?;
        let (mut observer, host) = Self::with_prefill_transfers(bound, context, transfers)?;
        observer.bind_partition_sources(scalar, placement)?;
        Ok((observer, host))
    }
    /// Borrow the exact source rows alongside either initial or saved capture
    /// state. No source or plan may be replaced after this observer is bound.
    pub(in crate::composition::mlx) fn bind_partition_sources(
        &mut self, scalar: &'a [Cell<Option<WorkspaceFloatingType>>],
        placement: (&'a eredu_architectures::component_partition::ComponentPartitionLayouts, usize),
    ) -> Result<()> {
        self.context.charge_metadata(std::mem::size_of::<(
            &mut Self, &[Cell<Option<WorkspaceFloatingType>>],
            (&eredu_architectures::component_partition::ComponentPartitionLayouts, usize), Result<()>,
        )>())?;
        if self.scalar_source.is_some() || self.placement.is_some()
            || scalar.len() != self.source.admission().plan().selections.len()
            || placement.1 >= placement.0.topology().world_size() {
            return Err(self.context.metadata_source(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
        self.scalar_source = Some(scalar);
        self.placement = Some(placement);
        Ok(())
    }
    /// The same capture worker at an immutable saved absolute frontier. Cold
    /// quotation starts from saved minimum usage; restoration may retain a
    /// larger live cumulative value and can therefore only perform less work.
    pub(in crate::composition::mlx) fn with_checkpoint_transfers(
        checkpoint: &'a eredu_runtime::capture::FundedCaptureCheckpoint,
        geometry: InferenceGeometry,
        bound: Option<eredu_runtime::layered::BoundCaptureSelection<'a>>,
        context: &WorkspaceContext,
        transfers: &'a Cell<CaptureNativePopulation>,
    ) -> Result<(Self, CaptureRunHostPlan<'a>)> {
        context.charge_metadata(std::mem::size_of::<(
            &eredu_runtime::capture::FundedCaptureCheckpoint,
            InferenceGeometry,
            Result<()>,
            std::result::Result<(), eredu_runtime::capture::FundedCaptureCheckpointError>,
            std::result::Result<
                CaptureRunHostPlan<'_>,
                eredu_runtime::capture::FundedCaptureCheckpointError,
            >,
            Option<eredu_runtime::layered::BoundCaptureSelection<'_>>,
        )>())?;
        Self::prepare(
            checkpoint.source(),
            geometry,
            context,
            bound,
            Some(transfers),
            Some(checkpoint),
            None,
        )
    }

    /// Quote one actual independent model invocation through the existing
    /// capture worker. The source, physical shape and scope mask stay borrowed;
    /// this constructor grants no model role, native scope or source permission.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::composition::mlx) fn with_invocation<'source: 'a>(
        source: &'source SharedCapturePlan,
        geometry: InferenceGeometry,
        context: &WorkspaceContext,
        transfers: &'a Cell<CaptureNativePopulation>,
        phase: CapturePhase,
        prediction: u64,
        shape: CaptureInvocationShape,
        selected: &'source [bool],
    ) -> Result<(Self, CaptureRunHostPlan<'source>)> {
        Self::with_invocation_window(
            source, geometry, context, transfers, phase, prediction, shape, selected, None,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(in crate::composition::mlx) fn with_invocation_window<'source: 'a>(
        source: &'source SharedCapturePlan,
        geometry: InferenceGeometry,
        context: &WorkspaceContext,
        transfers: &'a Cell<CaptureNativePopulation>,
        phase: CapturePhase,
        prediction: u64,
        shape: CaptureInvocationShape,
        selected: &'source [bool],
        window: Option<CaptureInvocationWindow>,
    ) -> Result<(Self, CaptureRunHostPlan<'source>)> {
        Self::prepare(
            source,
            geometry,
            context,
            None,
            Some(transfers),
            None,
            Some(Invocation {
                phase,
                prediction,
                shape,
                window,
                selected,
            }),
        )
    }

    fn prepare<'source: 'a>(
        source: &'source SharedCapturePlan,
        geometry: InferenceGeometry,
        context: &WorkspaceContext,
        bound: Option<eredu_runtime::layered::BoundCaptureSelection<'source>>,
        transfers: Option<&'a Cell<CaptureNativePopulation>>,
        checkpoint: Option<&'source eredu_runtime::capture::FundedCaptureCheckpoint>,
        invocation: Option<Invocation<'source>>,
    ) -> Result<(Self, CaptureRunHostPlan<'source>)> {
        let metadata = Metadata::new(context)?;
        geometry.validate().map_err(|cause| metadata.error(cause))?;
        let admission = source.admission();
        if bound.is_some_and(|bound| {
            bound.geometry() != geometry || !bound.selection().source().same_storage(source)
        }) {
            return Err(metadata.coordinate());
        }
        let first_prediction = invocation.map_or_else(
            || checkpoint.map_or(0, |source| source.next_prediction()),
            |source| source.prediction,
        );
        if let Some(invocation) = invocation {
            if bound.is_some()
                || checkpoint.is_some()
                || invocation.shape.batch != geometry.batch_size
                || invocation.shape.sequence != geometry.input_positions
                || invocation.selected.len() != admission.points().len()
                || invocation.shape.context.is_some_and(|actual| {
                    geometry
                        .cached_positions
                        .checked_add(geometry.input_positions)
                        != Some(actual)
                })
            {
                return Err(metadata.coordinate());
            }
        } else if let Some(checkpoint) = checkpoint {
            checkpoint
                .validate_continuation_geometry(geometry)
                .map_err(|cause| metadata.error(cause))?;
            if !checkpoint.source().same_storage(source)
                || (first_prediction != 0 && bound.is_some())
            {
                return Err(metadata.coordinate());
            }
        } else {
            let origin = admission
                .text_origin()
                .ok_or_else(|| metadata.error(CaptureProtocolError::Invocation))?;
            if admission.request().batch != geometry.batch_size
                || admission.request().prompt_tokens != geometry.input_positions
                || admission.request().max_predictions != geometry.max_output_tokens
                || origin.cached_positions != geometry.cached_positions
            {
                return Err(metadata.coordinate());
            }
        }
        let prefill = CaptureObservationStep::with_invocation(
            admission,
            invocation.map_or_else(
                || {
                    if first_prediction == 0 {
                        CapturePhase::Prefill
                    } else {
                        CapturePhase::Decode
                    }
                },
                |source| source.phase,
            ),
            first_prediction,
            invocation.map(|source| source.shape),
        )
        .and_then(|policy| policy.with_window(invocation.and_then(|source| source.window)))
        .map_err(|cause| metadata.error(cause))?;
        if invocation.is_none()
            && first_prediction == 0
            && bound.is_none()
            && prefill.requires_sequence_readout()
        {
            return Err(metadata.error(CaptureProtocolError::PrefillAttribution));
        }
        let mut progress = Vec::new();
        if let Some(bound) = bound {
            let policy = eredu_runtime::capture::CapturePrefillObservationPolicy::from_bound(bound)
                .map_err(|cause| metadata.error(cause))?;
            if context.uses_checked_metadata() {
                progress = context.metadata_vec(admission.plan().selections.len())?;
            } else {
                progress
                    .try_reserve_exact(admission.plan().selections.len())
                    .map_err(|cause| metadata.error(cause))?;
            }
            for index in 0..admission.plan().selections.len() {
                progress.push(
                    policy
                        .row(index)
                        .map_err(|cause| metadata.error(cause))?
                        .initial_progress(),
                );
            }
        }
        let host = match (invocation, checkpoint) {
            (Some(invocation), None) => CaptureRunHostPlan::prepare_invocation_window(
                source,
                invocation.phase,
                invocation.prediction,
                invocation.shape,
                invocation.selected,
                invocation.window,
            )
            .map_err(|cause| metadata.error(cause))?,
            (None, Some(checkpoint)) => checkpoint
                .continuation_host_plan_for(geometry)
                .map_err(|cause| metadata.error(cause))?,
            (None, None) => {
                CaptureRunHostPlan::prepare(source).map_err(|cause| metadata.error(cause))?
            }
            (Some(_), Some(_)) => return Err(metadata.coordinate()),
        };
        let ledger = match checkpoint {
            Some(checkpoint) => {
                CaptureLedger::with_inherited_usage(admission, checkpoint.inherited_usage())
                    .map_err(|cause| metadata.error(cause))?
            }
            None => CaptureLedger::new(admission),
        };
        context.charge_metadata(std::mem::size_of::<(
            Self,
            CaptureRunHostPlan<'_>,
            Result<(Self, CaptureRunHostPlan<'_>)>,
        )>())?;
        let statuses = if context.uses_checked_metadata() {
            let mut statuses = context.metadata_vec(admission.plan().selections.len())?;
            statuses.resize(
                admission.plan().selections.len(),
                CaptureRecordStatus::Skipped,
            );
            statuses
        } else {
            vec![CaptureRecordStatus::Skipped; admission.plan().selections.len()]
        };
        Ok((
            Self {
                source,
                invocation,
                transfers,
                scalar_source: None,
                placement: None,
                geometry,
                context: context.clone(),
                ledger,
                envelope_usage: None,
                prepared_prefix: None,
                interventions: None,
                text_interventions: None,
                partition_intervention_source: None,
                statuses,
                roots: Vec::new(),
                first_prediction,
                next_prediction: first_prediction,
                prefill_end: 0,
                prefill_complete: false,
                active: None,
                bound,
                progress,
                chunk: None,
                routed_selection: None,
                routed_intervention_selection: None,
                routed_active: false,
                routed_addressable_rows:None,
                routed_tokens: [0; 2],
                routed_native_rows: None,
            },
            host,
        ))
    }
    /// Use the same settled request usage as the forthcoming actual model role.
    pub(in crate::composition::mlx) fn with_inherited_usage(
        mut self,
        usage: CaptureUsage,
    ) -> Result<Self> {
        if self.invocation.is_none() || self.active.is_some() {
            return Err(self
                .context
                .metadata_source(CaptureProtocolError::Invocation));
        }
        self.ledger = CaptureLedger::with_inherited_usage(self.source.admission(), usage)
            .map_err(|cause| self.context.metadata_source(cause))?;
        Ok(self)
    }
    /// Resume the exact source-owned prefix without spending it again. The
    /// actual funded observer authenticates this same receipt and current usage.
    pub(in crate::composition::mlx) fn with_prepared_prefix(
        mut self,
        prefix: Option<&'a eredu_runtime::capture::OriginalSpeculativeCapturePrefix>,
    ) -> Result<Self> {
        if self.invocation.is_none() || self.active.is_some() || self.prepared_prefix.is_some() {
            return Err(self
                .context
                .metadata_source(CaptureProtocolError::Invocation));
        }
        if let Some(prefix) = prefix {
            prefix
                .resume(self.source, self.ledger.total())
                .map_err(|cause| self.context.metadata_source(cause))?;
        }
        self.prepared_prefix = prefix;
        Ok(self)
    }
    /// Same outer-envelope logical reservation as the real funded invocation.
    /// Must be attached before this one-use cold span begins.
    pub(in crate::composition::mlx) fn with_envelope_usage(
        mut self,
        usage: CaptureUsage,
    ) -> Result<Self> {
        if self.invocation.is_none() || self.active.is_some() || self.envelope_usage.is_some() {
            return Err(self
                .context
                .metadata_source(CaptureProtocolError::Invocation));
        }
        self.envelope_usage = Some(usage);
        Ok(self)
    }
    /// The cold and native consumers share one immutable per-hook edit plan.
    /// Only the trace mutates this caller-owned destination; no role is granted.
    pub(in crate::composition::mlx) fn with_model_interventions(
        mut self,
        edits: &'a RefCell<Option<PreparedModelInterventions>>,
    ) -> Result<Self> {
        if self.invocation.is_none()
            || self.active.is_some()
            || self.interventions.is_some()
            || self.text_interventions.is_some()
            || edits
                .try_borrow()
                .map_err(|cause| self.context.metadata_source(cause))?
                .is_none()
        {
            return Err(self
                .context
                .metadata_source(CaptureProtocolError::Invocation));
        }
        self.interventions = Some(edits);
        Ok(self)
    }
    /// One source-bound ordinary schedule uses this observer's existing ledger
    /// and native recorder. It cannot replace an explicit model invocation.
    pub(in crate::composition::mlx) fn with_text_interventions(
        mut self,
        edits: &'a RefCell<Option<PreparedTextInterventions>>,
    ) -> Result<Self> {
        let metadata = Metadata::new(&self.context)?;
        if self.invocation.is_some() || self.active.is_some()
            || self.interventions.is_some() || self.text_interventions.is_some()
            || self.first_prediction == 0 && self.bound.is_none()
        {
            return Err(metadata.coordinate());
        }
        {
            let borrowed = edits.try_borrow().map_err(|cause| metadata.error(cause))?;
            let prepared = borrowed.as_ref().ok_or_else(|| metadata.coordinate())?;
            if prepared.source().plan().admission().request() != self.source.admission().request() {
                return Err(metadata.coordinate());
            }
        }
        self.text_interventions = Some(edits);
        Ok(self)
    }
    /// Bind the same cold schedule to the actual retained partition source.
    pub(in crate::composition::mlx) fn with_partition_interventions(mut self,
        edits: &'a RefCell<Option<PreparedTextInterventions>>,
        communication: &'a eredu_runtime::RetainedCommunicationSource) -> Result<Self> {
        let metadata=Metadata::new(&self.context)?;
        let (layouts,rank)=self.placement.ok_or_else(||metadata.coordinate())?;
        if self.partition_intervention_source.is_some() {return Err(metadata.coordinate());}
        edits.try_borrow_mut().map_err(|cause|metadata.error(cause))?.as_mut()
            .ok_or_else(||metadata.coordinate())?.bind_partition(layouts,rank,communication,&self.context)?;
        self.partition_intervention_source=Some(communication);
        self.with_text_interventions(edits)
    }
    fn finish_text_interventions(&mut self) -> Result<()> {
        if let Some(edits) = self.text_interventions {
            edits.try_borrow_mut().map_err(|cause| self.context.metadata_source(cause))?
                .as_mut().ok_or_else(|| self.context.metadata_source(CaptureProtocolError::Transaction))?
                .finish(&self.context)?;
        }
        Ok(())
    }
    fn validate_invocation_span(&self, span: &InferenceWorkspaceSpan) -> Result<()> {
        let metadata = Metadata::new(&self.context)?;
        match span {
            InferenceWorkspaceSpan::Prefill(chunk)
                if chunk.input.start == 0
                    && chunk.input.end == self.geometry.input_positions
                    && chunk.position == self.geometry.cached_positions =>
            {
                Ok(())
            }
            _ => Err(metadata.coordinate()),
        }
    }
    fn phase(&self, prediction: u64) -> CapturePhase {
        self.invocation.map_or_else(
            || {
                if prediction == 0 {
                    CapturePhase::Prefill
                } else {
                    CapturePhase::Decode
                }
            },
            |source| source.phase,
        )
    }
    fn begin_prediction(&mut self, phase: CapturePhase, prediction: u64) -> Result<()> {
        let metadata = Metadata::new(&self.context)?;
        if self.active.is_some() || prediction != self.next_prediction || !self.roots.is_empty() {
            return Err(metadata.coordinate());
        }
        let policy = CaptureObservationStep::with_invocation(
            self.source.admission(),
            phase,
            prediction,
            self.invocation.map(|source| source.shape),
        )
        .and_then(|policy| policy.with_window(self.invocation.and_then(|source| source.window)))
        .map_err(|cause| metadata.error(cause))?;
        // A failed metadata reservation is still this attempted logical step.
        // Reusing this diagnostic owner cannot reset/retry it as fresh work.
        self.active = Some(prediction);
        if let Some(prefix) = self.prepared_prefix {
            self.ledger = prefix
                .resume(self.source, self.ledger.total())
                .map_err(|cause| metadata.error(cause))?;
        } else {
            self.ledger.begin_step();
            policy
                .reserve_invocation_metadata(&mut self.ledger)
                .map_err(|cause| metadata.error(cause))?;
            policy
                .reserve_metadata(&mut self.ledger)
                .map_err(|cause| metadata.error(cause))?;
        }
        if let Some(edits) = self.interventions {
            let mut edits = edits
                .try_borrow_mut()
                .map_err(|cause| metadata.error(cause))?;
            let edits = edits
                .as_mut()
                .ok_or_else(|| metadata.error(CaptureProtocolError::Invocation))?;
            match self.prepared_prefix {
                Some(_) => edits.begin_prepaid(phase, prediction, &self.context)?,
                None => edits.begin(phase, prediction, &mut self.ledger, &self.context)?,
            }
        }
        if let Some(edits) = self.text_interventions {
            edits.try_borrow_mut().map_err(|cause| metadata.error(cause))?
                .as_mut().ok_or_else(|| metadata.coordinate())?
                .begin(phase, prediction, &mut self.ledger, &self.context)?;
        }
        if let Some(usage) = self
            .envelope_usage
            .filter(|_| self.prepared_prefix.is_none())
        {
            if let Some(CaptureSkipReason::Limit { budget, cumulative }) = self
                .ledger
                .reserve(usage)
                .map_err(|cause| metadata.error(cause))?
            {
                return Err(metadata.error(CaptureError::Limit { budget, cumulative }));
            }
        }
        for (index, status) in self.statuses.iter_mut().enumerate() {
            *status = if self
                .invocation
                .is_some_and(|source| !source.selected[index])
            {
                CaptureRecordStatus::Skipped
            } else {
                policy
                    .initial_status(index)
                    .map_err(|cause| metadata.error(cause))?
            };
        }
        Ok(())
    }
    fn policy(&self) -> Result<CaptureObservationStep<'a>> {
        let metadata = Metadata::new(&self.context)?;
        let prediction = self
            .active
            .ok_or_else(|| metadata.error(CaptureProtocolError::Transaction))?;
        CaptureObservationStep::with_invocation(
            self.source.admission(),
            self.phase(prediction),
            prediction,
            self.invocation.map(|source| source.shape),
        )
        .and_then(|policy| policy.with_window(self.invocation.and_then(|source| source.window)))
        .map_err(|cause| metadata.error(cause))
    }
    fn observe_value(&mut self, path: &str, value: &WorkspaceTensor) -> Result<()> {
        self.observe_value_with_origin(path,value,true)
    }
    fn observe_value_with_origin(&mut self, path: &str, value: &WorkspaceTensor, mut local:bool) -> Result<()> {
        let metadata = Metadata::new(&self.context)?;
        let source_present = local;
        self.context.charge_metadata(std::mem::size_of::<(&mut Self,&str,&WorkspaceTensor,bool,bool,
            Option<eredu_architectures::component_partition::CompletePartitionCaptureSource>)>())?;
        // Inactive projected prefill selections do not acquire a new source
        // when the shared model emits the same path during cached decode.
        if self.active.is_some_and(|prediction|self.source.admission().plan().selections.iter()
            .any(|selection|selection.path==path&&selection.schedule.includes(self.phase(prediction),prediction))) {
            if let Some((layout, rank)) = self.placement {
                self.context.charge_metadata(
                    eredu_architectures::component_partition::ComponentPartitionLayouts::complete_capture_source_control_bytes()
                        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
                )?;
                if let Some(source)=layout.complete_capture_source(path) {
                    local &= source.producer()==rank;
                } else if (self.bound.is_some() && self.active==Some(0))
                    || (self.invocation.is_none() && self.active.is_some_and(|p|p>0)) {
                    self.context.charge_metadata(
                        eredu_architectures::component_partition::ContiguousPartitionCaptureSource::control_bytes()
                            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
                    let source=layout.contiguous_capture_source(path).map_err(|cause|metadata.error(cause))?;
                    local &= source.rank(rank).is_some_and(|rank|rank.produces);
                } else {return Err(metadata.coordinate());}
            }
        }
        if self.bound.is_some() && self.active == Some(0) {
            return self.observe_fragment_value_with_origin(path, value, local, source_present);
        }
        let policy = self.policy()?;
        for index in 0..self.statuses.len() {
            if !policy
                .select(index, self.statuses[index], path)
                .map_err(|cause| metadata.error(cause))?
            {
                continue;
            }
            // Presence and scalar type describe the reached model hook even
            // when cumulative capture allowance is exhausted. The later Host
            // source vote needs those facts to coordinate a valid skipped row;
            // recording them grants no native transform or retained root.
            if source_present && self.placement.is_some() {
                record_source(self.scalar_source, index, &self.context, value)?;
            }
            // A failed attempt is consumed, just like the real failed record.
            self.statuses[index] = CaptureRecordStatus::Consumed;
            if let Some(usage) = policy
                .window_metadata_usage(index)
                .map_err(|cause| metadata.error(cause))?
            {
                if policy
                    .reserve_value(&mut self.ledger, usage)
                    .map_err(|cause| metadata.error(cause))?
                    .is_some()
                {
                    self.statuses[index] = CaptureRecordStatus::Skipped;
                    continue;
                }
            }
            if let Some(placement)=self.placement.filter(|(layout,_)|
                layout.complete_capture_source(path).is_none()) {
                if self.invocation.is_some(){return Err(metadata.coordinate());}
                let prediction=self.active.ok_or_else(||metadata.coordinate())?;
                if partition::observer::observe_invocation(self.source,index,self.phase(prediction),prediction,
                    placement,source_present,value,&self.context,&mut self.roots,&mut self.ledger,
                    self.transfers,self.scalar_source)? {
                    self.statuses[index]=CaptureRecordStatus::Skipped;
                }
                continue;
            }
            if matches!(
                self.source.admission().plan().selections[index].transform,
                CaptureTransform::TopCandidates { .. }
            ) {
                let geometry = CaptureCandidateGeometry::prepare(
                    self.source.admission(),
                    index,
                    self.phase(self.active.ok_or_else(|| metadata.coordinate())?),
                    self.active.ok_or_else(|| metadata.coordinate())?,
                    self.invocation.map(|source| source.shape),
                )
                .map_err(|cause| metadata.error(cause))?;
                let program =
                    crate::backend::array_copy::CandidateExtraction::from_geometry(&geometry)
                        .map_err(|cause| metadata.error(cause))?;
                program.validate_workspace(value, &self.context)?;
                let usage = super::bounded_capture::estimate_candidates(&geometry)
                    .map_err(|cause| metadata.error(cause))?;
                if policy
                    .reserve_value(&mut self.ledger, usage)
                    .map_err(|cause| metadata.error(cause))?
                    .is_some()
                {
                    self.statuses[index] = CaptureRecordStatus::Skipped;
                    continue;
                }
                record_source(self.scalar_source,index,&self.context,value)?;
                if !local {
                    if source_present {record_replica(self.transfers,&self.context,&mut self.roots,value)?;}
                    continue;
                }
                metadata.reserve(&mut self.roots, 1)?;
                self.roots.push(value.clone());
                let outputs = program.trace(value, &self.context)?;
                metadata.reserve(&mut self.roots, outputs.len())?;
                self.roots.extend(outputs);
                if self.transfers.is_some() {
                    let population = if self.invocation.is_some() {
                        CaptureNativePopulation::within_candidates()
                    } else {
                        CaptureNativePopulation::candidates()
                    }
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                    record_population(self.transfers, &self.context, population)?;
                }
                continue;
            }
            if matches!(
                self.source.admission().plan().selections[index].transform,
                CaptureTransform::TokenScores { .. }
            ) {
                let geometry = CaptureTokenScoreGeometry::prepare(
                    self.source.admission(),
                    index,
                    self.phase(self.active.ok_or_else(|| metadata.coordinate())?),
                    self.active.ok_or_else(|| metadata.coordinate())?,
                    self.invocation.map(|source| source.shape),
                )
                .map_err(|cause| metadata.error(cause))?;
                let program =
                    crate::backend::array_copy::TokenScoreProgram::from_geometry(&geometry)
                        .map_err(|cause| metadata.error(cause))?;
                program
                    .validate_workspace(value, &self.context)
                    .map_err(|cause| metadata.error(cause))?;
                let usage = super::bounded_capture::estimate_token_scores(&geometry)
                    .map_err(|cause| metadata.error(cause))?;
                if policy
                    .reserve_value(&mut self.ledger, usage)
                    .map_err(|cause| metadata.error(cause))?
                    .is_some()
                {
                    self.statuses[index] = CaptureRecordStatus::Skipped;
                    continue;
                }
                record_source(self.scalar_source,index,&self.context,value)?;
                if !local {
                    if source_present {record_replica(self.transfers,&self.context,&mut self.roots,value)?;}
                    continue;
                }
                metadata.reserve(&mut self.roots, 1)?;
                self.roots.push(value.clone());
                program
                    .trace(value, &self.context, &mut self.roots)
                    .map_err(|cause| metadata.error(cause))?;
                if self.transfers.is_some() {
                    let population = if self.invocation.is_some() {
                        CaptureNativePopulation::within_token_scores(program)
                    } else {
                        CaptureNativePopulation::token_scores(program)
                    }
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                    record_population(self.transfers, &self.context, population)?;
                }
                continue;
            }
            if matches!(
                self.source.admission().plan().selections[index].transform,
                CaptureTransform::Summary
            ) {
                let geometry = policy
                    .summary_geometry(index)
                    .map_err(|cause| metadata.error(cause))?;
                let program =
                    crate::backend::array_copy::PreparedCaptureSummary::from_geometry(&geometry)
                        .map_err(|cause| metadata.error(cause))?;
                program
                    .validate_workspace_source(value, &self.context)
                    .map_err(|cause| metadata.error(cause))?;
                if policy.window().is_some() && geometry.elements() == 0 {
                    self.statuses[index] = CaptureRecordStatus::Skipped;
                    continue;
                }
                let usage = super::bounded_capture::estimate_summary(&geometry)
                    .map_err(|cause| metadata.error(cause))?;
                if policy
                    .reserve_value(&mut self.ledger, usage)
                    .map_err(|cause| metadata.error(cause))?
                    .is_some()
                {
                    self.statuses[index] = CaptureRecordStatus::Skipped;
                    continue;
                }
                record_source(self.scalar_source, index, &self.context, value)?;
                if !local {
                    if source_present { record_replica(self.transfers, &self.context, &mut self.roots, value)?; }
                    continue;
                }
                metadata.reserve(&mut self.roots, 1)?;
                self.roots.push(value.clone());
                program
                    .trace(value, &self.context, &mut self.roots)
                    .map_err(|cause| metadata.error(cause))?;
                if self.transfers.is_some() {
                    let population = if self.invocation.is_some() {
                        CaptureNativePopulation::within_summary(&program)
                    } else {
                        program.population()
                    }
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                    record_population(self.transfers, &self.context, population)?;
                }
                continue;
            }
            if matches!(
                self.source.admission().plan().selections[index].transform,
                CaptureTransform::Histogram { .. }
            ) {
                let geometry = policy
                    .histogram_geometry(index)
                    .map_err(|cause| metadata.error(cause))?;
                let program =
                    crate::backend::array_copy::PreparedCaptureHistogram::from_geometry(&geometry)
                        .map_err(|cause| metadata.error(cause))?;
                program
                    .validate_workspace_source(value, &self.context)
                    .map_err(|cause| metadata.error(cause))?;
                if policy.window().is_some() && geometry.elements() == 0 {
                    self.statuses[index] = CaptureRecordStatus::Skipped;
                    continue;
                }
                let usage = super::bounded_capture::estimate_histogram(&geometry)
                    .map_err(|cause| metadata.error(cause))?;
                if policy
                    .reserve_value(&mut self.ledger, usage)
                    .map_err(|cause| metadata.error(cause))?
                    .is_some()
                {
                    self.statuses[index] = CaptureRecordStatus::Skipped;
                    continue;
                }
                record_source(self.scalar_source, index, &self.context, value)?;
                if !local {
                    if source_present {record_replica(self.transfers,&self.context,&mut self.roots,value)?;}
                    continue;
                }
                metadata.reserve(&mut self.roots, 1)?;
                self.roots.push(value.clone());
                program
                    .trace(value, &self.context, &mut self.roots)
                    .map_err(|cause| metadata.error(cause))?;
                if self.transfers.is_some() {
                    let population = if self.invocation.is_some() {
                        CaptureNativePopulation::within_histogram(&program)
                    } else {
                        program.population()
                    }
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                    record_population(self.transfers, &self.context, population)?;
                }
                continue;
            }
            let geometry = policy
                .tensor_geometry(index)
                .map_err(|cause| metadata.error(cause))?;
            let program = CaptureTensorSelection::from_geometry(&geometry)
                .map_err(|cause| metadata.error(cause))?;
            program
                .validate_workspace_source(value, &self.context)
                .map_err(|cause| metadata.error(cause))?;
            record_source(self.scalar_source, index, &self.context, value)?;
            if policy.window().is_some()
                && geometry
                    .starts()
                    .iter()
                    .zip(geometry.ends())
                    .any(|(a, b)| a == b)
            {
                self.statuses[index] = CaptureRecordStatus::Skipped;
                continue;
            }
            let usage = super::bounded_capture::estimate_tensor_geometry(&geometry)
                .map_err(|cause| metadata.error(cause))?;
            if policy
                .reserve_value(&mut self.ledger, usage)
                .map_err(|cause| metadata.error(cause))?
                .is_some()
            {
                self.statuses[index] = CaptureRecordStatus::Skipped;
                continue;
            }
            // The complete receipt is charged on the receiver as well, but
            // only the selected producer owns the native transform graph.
            if !local {
                if source_present { record_replica(self.transfers, &self.context, &mut self.roots, value)?; }
                continue;
            }
            // The native work retains every source/intermediate through its
            // enclosing span. Context tracing retains all intermediate facts;
            // keep actual metadata roots until the same span's report is fixed.
            metadata.reserve(&mut self.roots, 1)?;
            self.roots.push(value.clone());
            if self.transfers.is_some() {
                // The original capture work keeps every selected intermediate
                // until its enclosing span settles. Trace the same retention.
                program
                    .trace_retained_within(value, &self.context, &mut self.roots)
                    .map_err(|cause| metadata.error(cause))?;
            } else {
                metadata.reserve(&mut self.roots, 1)?;
                self.roots.push(
                    program
                        .trace_within(value, &self.context)
                        .map_err(|cause| metadata.error(cause))?,
                );
            }
            if self.invocation.is_some() {
                let population = CaptureNativePopulation::within_raw()
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                record_population(self.transfers, &self.context, population)?;
            } else {
                record_transfer(self.transfers, &self.context)?;
            }
        }
        Ok(())
    }
}
impl ActivationObserver<WorkspaceTensor, eredu_nn::Error> for CaptureWorkspaceObserver<'_> {
    fn routed_unit_observer(&mut self, path: &str) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<WorkspaceTensor>>> {
        let index = self.source.admission().points().iter().position(|point|
            matches!(&point.value_type, eredu_core::ObservationValueType::RoutedUnits {routing,..} if routing == path));
        let edit = if let Some(edits) = self.interventions {
            edits.try_borrow().map_err(|cause| self.context.metadata_source(cause))?
                .as_ref().and_then(|edits| edits.routed_selection(path))
        } else { None };
        if index.is_none() && edit.is_none() { return Ok(None); }
        let metadata = Metadata::new(&self.context)?;
        if self.routed_active && (self.routed_selection != index || self.routed_intervention_selection != edit) {
            return Err(metadata.coordinate());
        }
        self.routed_selection = index;
        self.routed_intervention_selection = edit;
        Ok(Some(self))
    }
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn requires_sequence_readout(&self) -> bool {
        if self.text_interventions.is_some_and(|edits| {
            edits.try_borrow().map_or(true, |edits| edits.as_ref()
                .is_some_and(PreparedTextInterventions::requires_sequence_readout))
        }) { return true; }
        if self.interventions.is_some_and(|edits| {
            edits.try_borrow().map_or(true, |edits| {
                edits
                    .as_ref()
                    .is_some_and(|edits| edits.requires_sequence_readout())
            })
        }) {
            return true;
        }
        if let Some(bound) = self
            .bound
            .filter(|_| self.active.unwrap_or(self.first_prediction) == 0)
        {
            return bound.geometry().output == eredu_core::OutputDemand::Sequence;
        }
        let prediction = self.active.unwrap_or(self.first_prediction);
        CaptureObservationStep::with_invocation(
            self.source.admission(),
            self.phase(prediction),
            prediction,
            self.invocation.map(|source| source.shape),
        )
        .and_then(|policy| policy.with_window(self.invocation.and_then(|source| source.window)))
        .map_or(true, |policy| {
            self.invocation.map_or_else(
                || policy.requires_sequence_readout(),
                |source| {
                    policy
                        .requires_sequence_readout_for(source.selected)
                        .unwrap_or(true)
                },
            )
        })
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &WorkspaceTensor,
    ) -> Result<Option<WorkspaceTensor>> {
        if let Some(edits) = self.text_interventions {
            let (output, population) = edits.try_borrow_mut()
                .map_err(|cause| self.context.metadata_source(cause))?
                .as_mut().ok_or_else(|| self.context.metadata_source(CaptureProtocolError::Transaction))?
                .trace(path, value, &self.context, &mut self.ledger, &mut self.roots)?;
            record_population(self.transfers, &self.context, population)?;
            return Ok(output);
        }
        let Some(edits) = self.interventions else {
            return Ok(None);
        };
        let (output, population) = edits
            .try_borrow_mut()
            .map_err(|cause| self.context.metadata_source(cause))?
            .as_mut()
            .ok_or_else(|| {
                self.context
                    .metadata_source(CaptureProtocolError::Invocation)
            })?
            .trace(
                path,
                value,
                &self.context,
                &mut self.ledger,
                &mut self.roots,
            )?;
        record_population(self.transfers, &self.context, population)?;
        Ok(output)
    }
    fn observe(&mut self, path: &str, value: &WorkspaceTensor) -> Result<()> {
        let metadata = Metadata::new(&self.context)?;
        self.observe_value(path, value)
    }
    fn observe_replica(&mut self, path: &str, value: &WorkspaceTensor) -> Result<()> {
        if self.placement.is_none() { return Ok(()); }
        self.observe_value(path, value)
    }
    fn observe_generated_retained(
        &mut self,
        path: &str,
        prototype: &WorkspaceTensor,
        source: &GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<WorkspaceTensor, eredu_nn::Error>,
    ) -> Result<()> {
        let metadata = Metadata::new(&self.context)?;
        if self.bound.is_some() && self.active == Some(0) {
            return self.observe_fragment_generated(path, prototype, source, factory);
        }
        use eredu_nn::workspace::WorkspaceDtype;
        use eredu_nn::{GeneratedTensorProgram, GeneratedTensorSourceRole};
        let policy = self.policy()?;
        let mut generated = None;
        for index in 0..self.statuses.len() {
            if !policy
                .select(index, self.statuses[index], path)
                .map_err(|cause| metadata.error(cause))?
            {
                continue;
            }
            self.statuses[index] = CaptureRecordStatus::Consumed;
            if let Some(usage) = policy
                .window_metadata_usage(index)
                .map_err(|cause| metadata.error(cause))?
            {
                if policy
                    .reserve_value(&mut self.ledger, usage)
                    .map_err(|cause| metadata.error(cause))?
                    .is_some()
                {
                    self.statuses[index] = CaptureRecordStatus::Skipped;
                    continue;
                }
            }
            let geometry = policy
                .tensor_geometry(index)
                .map_err(|cause| metadata.error(cause))?;
            let selection = CaptureTensorSelection::from_geometry(&geometry)
                .map_err(|cause| metadata.error(cause))?;
            selection
                .validate_workspace_source(prototype, &self.context)
                .map_err(|cause| metadata.error(cause))?;
            policy
                .validate_generated(index, source, factory.program())
                .map_err(|cause| metadata.error(cause))?;
            if policy.window().is_some()
                && geometry
                    .starts()
                    .iter()
                    .zip(geometry.ends())
                    .any(|(a, b)| a == b)
            {
                self.statuses[index] = CaptureRecordStatus::Skipped;
                continue;
            }
            let usage = super::bounded_capture::estimate_tensor_geometry(&geometry)
                .map_err(|cause| metadata.error(cause))?;
            let usage = policy
                .generated_usage(usage, source)
                .map_err(|cause| metadata.error(cause))?;
            if policy
                .reserve_value(&mut self.ledger, usage)
                .map_err(|cause| metadata.error(cause))?
                .is_some()
            {
                self.statuses[index] = CaptureRecordStatus::Skipped;
                continue;
            }
            if self.invocation.is_some() {
                let population = CaptureNativePopulation::generated_preparation()
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                record_population(self.transfers, &self.context, population)?;
            }
            if policy
                .generated_slice_empty(index)
                .map_err(|cause| metadata.error(cause))?
            {
                continue;
            }
            if generated.is_none() {
                if self.invocation.is_some() {
                    let population = CaptureNativePopulation::generated_retention()
                        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                    record_population(self.transfers, &self.context, population)?;
                }
                let GeneratedTensorProgram::BlockFp8Input(plan) = factory.program();
                let (values_shape, scales_shape) = (plan.values_shape(), plan.scales_shape());
                if self.context.uses_checked_metadata() {
                    self.context.reserve_metadata_vec(&mut self.roots, 9)?;
                } else {
                    self.roots
                        .try_reserve_exact(9)
                        .map_err(|cause| metadata.error(cause))?;
                }
                let mut sources = 0;
                factory.visit_sources(&mut |role, value| {
                    let (expected, shape, dtype) = match sources {
                        0 => (
                            GeneratedTensorSourceRole::CompactValues,
                            values_shape,
                            WorkspaceDtype::Uint8,
                        ),
                        1 => (
                            GeneratedTensorSourceRole::BlockScales,
                            scales_shape,
                            WorkspaceDtype::Float32,
                        ),
                        _ => return Err(metadata.error(CaptureProtocolError::GeneratedSource)),
                    };
                    self.context.validate_values([value])?;
                    if role != expected || value.shape() != shape || value.layout().dtype() != dtype
                    {
                        return Err(metadata.coordinate());
                    }
                    metadata.reserve(&mut self.roots, 1)?;
                    self.roots.push(value.clone());
                    sources += 1;
                    Ok(())
                })?;
                if sources != 2 {
                    return Err(metadata.error(CaptureProtocolError::GeneratedSource));
                }
                let mut outputs = 0;
                let value = factory.generate(&mut |value| {
                    self.context.validate_values([value])?;
                    if outputs == 7 {
                        return Err(metadata.error(CaptureProtocolError::GeneratedSource));
                    }
                    metadata.reserve(&mut self.roots, 1)?;
                    self.roots.push(value.clone());
                    outputs += 1;
                    Ok(())
                })?;
                if outputs != 7 {
                    return Err(metadata.error(CaptureProtocolError::GeneratedSource));
                }
                generated = Some(value);
            }
            let value = generated
                .as_ref()
                .expect("one generated source for this hook");
            selection
                .validate_workspace_source(value, &self.context)
                .map_err(|cause| metadata.error(cause))?;
            // All earlier hooks' compact sources and intermediate roots remain
            // retained through the full enclosing equation span, including the
            // following projection finish. end_span is the only release point.
            if self.invocation.is_some() {
                // The original model capture keeps the generated input plus
                // each actual selection output through the same final frontier.
                metadata.reserve(&mut self.roots, 1)?;
                self.roots.push(value.clone());
                selection
                    .trace_retained_within(value, &self.context, &mut self.roots)
                    .map_err(|cause| metadata.error(cause))?;
                let population = CaptureNativePopulation::within_raw()
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                record_population(self.transfers, &self.context, population)?;
            } else {
                metadata.reserve(&mut self.roots, 1)?;
                self.roots.push(
                    selection
                        .trace_within(value, &self.context)
                        .map_err(|cause| metadata.error(cause))?,
                );
            }
        }
        Ok(())
    }
    fn observe_generated(
        &mut self,
        path: &str,
        _: &WorkspaceTensor,
        _: &GeneratedCaptureSource,
        _: &mut dyn FnMut() -> Result<WorkspaceTensor>,
    ) -> Result<()> {
        let metadata = Metadata::new(&self.context)?;
        let policy = self.policy()?;
        for (index, status) in self.statuses.iter_mut().enumerate() {
            if policy
                .select(index, *status, path)
                .map_err(|cause| metadata.error(cause))?
            {
                *status = CaptureRecordStatus::Consumed;
                return Err(metadata.error(CaptureProtocolError::GeneratedSource));
            }
        }
        Ok(())
    }
}
impl InferenceWorkspaceObserver for CaptureWorkspaceObserver<'_> {
    fn observe_remote_output(&mut self,value:&WorkspaceTensor,context:&WorkspaceContext)->Result<()> {
        let metadata=Metadata::new(&self.context)?;
        self.context.charge_metadata(std::mem::size_of::<(&mut Self,&WorkspaceTensor,&WorkspaceContext,bool)>())?;
        self.context.charge_metadata(
            eredu_architectures::component_partition::ComponentPartitionLayouts::complete_capture_source_control_bytes()
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        if !context.shares_trace(&self.context) || self.invocation.is_some() || self.interventions.is_some()
            || (self.text_interventions.is_some() && self.partition_intervention_source.is_none()) {
            return Err(metadata.error(eredu_runtime::working_memory::InferenceObservationError::RemoteOutput));
        }
        let placement=self.placement.ok_or_else(||metadata.error(
            eredu_runtime::working_memory::InferenceObservationError::RemoteOutput))?;
        if let Some(edits)=self.text_interventions {
            let communication=self.partition_intervention_source.ok_or_else(||metadata.coordinate())?;
            let mut edits=edits.try_borrow_mut().map_err(|cause|metadata.error(cause))?;
            let edits=edits.as_mut().ok_or_else(||metadata.coordinate())?;
            edits.validate_partition(placement.0,placement.1,communication).map_err(|cause|metadata.error(cause))?;
            edits.trace_remote_output(placement.0,placement.1,value,&self.context)?;
        }
        for selection in &self.source.admission().plan().selections {
            validate_partition_capture_source(selection,placement,&self.context)?;
        }
        // Non-publication selections without a local PP invocation advance at
        // end_fragment_span. The final logits value is not their tensor source.
        if !self.source.admission().plan().selections.iter()
            .any(|selection|selection.path==eredu_core::MODEL_LOGITS_OBSERVATION_PATH) {return Ok(());}
        // A real local replica already traversed its hook before publication.
        // Only ranks without that source need the remote logical progression.
        if self.placement.and_then(|(layouts, rank)| layouts.rank(rank))
            .and_then(|layout| layout.observation(eredu_core::MODEL_LOGITS_OBSERVATION_PATH))
            .is_some_and(|point| point.coordinates().is_some()) {
            return Ok(());
        }
        self.observe_value_with_origin(eredu_core::MODEL_LOGITS_OBSERVATION_PATH,value,false)
    }
    fn prefill_selection(&self) -> Option<eredu_runtime::layered::BoundCaptureSelection<'_>> {
        self.bound
    }
    fn begin_span(
        &mut self,
        geometry: InferenceGeometry,
        span: &InferenceWorkspaceSpan,
        prediction: u64,
        context: &WorkspaceContext,
    ) -> Result<bool> {
        let metadata = Metadata::new(&self.context)?;
        if geometry != self.geometry || !context.shares_trace(&self.context) {
            return Err(metadata.coordinate());
        }
        if let Some(transfers) = self.transfers {
            transfers.set(CaptureNativePopulation::default());
        }
        if let Some(scalars) = self.scalar_source {
            self.context.charge_metadata(std::mem::size_of::<(
                std::slice::Iter<'_, Cell<Option<WorkspaceFloatingType>>>,
                Option<&Cell<Option<WorkspaceFloatingType>>>,
            )>())?;
            for scalar in scalars { scalar.set(None); }
        }
        if let Some(invocation) = self.invocation {
            if self.prefill_complete || prediction != invocation.prediction {
                return Err(metadata.coordinate());
            }
            self.validate_invocation_span(span)?;
            self.begin_prediction(invocation.phase, prediction)?;
            return Ok(true);
        }
        if self.bound.is_some() {
            if let InferenceWorkspaceSpan::Prefill(chunk) = span {
                return self.begin_fragment_span(chunk, prediction);
            }
        }
        match span {
            InferenceWorkspaceSpan::Prefill(chunk) if self.first_prediction != 0 => {
                if prediction != 0
                    || self.prefill_complete
                    || self.prefill_end != 0
                    || chunk.input.start != 0
                    || chunk.input.end != geometry.input_positions
                    || chunk.position != geometry.cached_positions
                {
                    return Err(metadata.coordinate());
                }
                self.begin_prediction(CapturePhase::Decode, self.first_prediction)?;
                self.prefill_end = chunk.input.end;
                Ok(true)
            }
            InferenceWorkspaceSpan::Prefill(chunk) => {
                if geometry.cached_positions.checked_add(chunk.input.start) != Some(chunk.position)
                    || prediction != 0
                    || self.prefill_complete
                    || chunk.input.start != self.prefill_end
                    || chunk.input.end <= chunk.input.start
                    || chunk.input.end > geometry.input_positions
                {
                    return Err(metadata.coordinate());
                }
                if chunk.input.start == 0 {
                    self.begin_prediction(CapturePhase::Prefill, 0)?;
                }
                self.prefill_end = chunk.input.end;
                if self.prefill_end == geometry.input_positions {
                    self.active = None;
                    self.prefill_complete = true;
                    self.next_prediction = 1;
                }
                // p0 metadata is charged once, without a per-chunk frame/claim
                // or any whole-prompt tensor capture attribution.
                Ok(false)
            }
            InferenceWorkspaceSpan::Decode {
                index, position, ..
            } => {
                let expected = index.checked_add(1).ok_or_else(|| metadata.coordinate())?;
                let at = geometry
                    .cached_positions
                    .checked_add(geometry.input_positions)
                    .and_then(|p| p.checked_add(*index))
                    .ok_or_else(|| metadata.coordinate())?;
                if !self.prefill_complete || expected != prediction || at != *position {
                    return Err(metadata.coordinate());
                }
                let absolute = self
                    .first_prediction
                    .checked_add(prediction)
                    .ok_or_else(|| metadata.coordinate())?;
                self.begin_prediction(CapturePhase::Decode, absolute)?;
                Ok(true)
            }
        }
    }
    fn visit_retained(&self, visitor: &mut dyn FnMut(&WorkspaceTensor)) {
        for root in &self.roots {
            visitor(root);
        }
    }
    fn end_span(
        &mut self,
        span: &InferenceWorkspaceSpan,
        context: &WorkspaceContext,
    ) -> Result<()> {
        let metadata = Metadata::new(&self.context)?;
        if let Some(invocation) = self.invocation {
            if !context.shares_trace(&self.context)
                || self.active != Some(invocation.prediction)
                || self.prefill_complete
            {
                return Err(metadata.coordinate());
            }
            self.validate_invocation_span(span)?;
            if let Some(edits) = self.interventions {
                edits
                    .try_borrow_mut()
                    .map_err(|cause| metadata.error(cause))?
                    .as_mut()
                    .ok_or_else(|| metadata.error(CaptureProtocolError::Invocation))?
                    .finish(context)?;
            }
            self.active = None;
            self.prefill_complete = true;
            self.next_prediction = invocation
                .prediction
                .checked_add(1)
                .ok_or_else(|| metadata.coordinate())?;
            self.roots.clear();
            return Ok(());
        }
        if self.bound.is_some() {
            if let InferenceWorkspaceSpan::Prefill(chunk) = span {
                if !context.shares_trace(&self.context) {
                    return Err(metadata.coordinate());
                }
                return self.end_fragment_span(chunk);
            }
        }
        if self.first_prediction != 0 {
            if let InferenceWorkspaceSpan::Prefill(chunk) = span {
                if !context.shares_trace(&self.context)
                    || self.active != Some(self.first_prediction)
                    || self.prefill_complete
                    || chunk.input.start != 0
                    || chunk.input.end != self.geometry.input_positions
                    || chunk.position != self.geometry.cached_positions
                {
                    return Err(metadata.coordinate());
                }
                self.finish_absent_invocation_projections()?;
                self.finish_text_interventions()?;
                self.active = None;
                self.prefill_complete = true;
                self.next_prediction = self
                    .first_prediction
                    .checked_add(1)
                    .ok_or_else(|| metadata.coordinate())?;
                self.roots.clear();
                return Ok(());
            }
        }
        if !context.shares_trace(&self.context)
            || !matches!(span, InferenceWorkspaceSpan::Decode { .. })
        {
            return Err(metadata.coordinate());
        }
        let prediction = self.active.ok_or_else(|| metadata.coordinate())?;
        let InferenceWorkspaceSpan::Decode {
            index, position, ..
        } = span
        else {
            return Err(metadata.coordinate());
        };
        if index
            .checked_add(1)
            .and_then(|p| p.checked_add(self.first_prediction))
            != Some(prediction)
            || self
                .geometry
                .cached_positions
                .checked_add(self.geometry.input_positions)
                .and_then(|p| p.checked_add(*index))
                != Some(*position)
        {
            return Err(metadata.coordinate());
        }
        self.finish_absent_invocation_projections()?;
        self.finish_text_interventions()?;
        self.active = None;
        self.next_prediction = prediction
            .checked_add(1)
            .ok_or_else(|| metadata.coordinate())?;
        self.roots.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
