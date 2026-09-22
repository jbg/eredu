//! Actual hook geometry retained from the shared static activation trace.
use super::{NativeInterventionEstimator, PreparedStaticActivation};
use crate::backend::array_copy::{
    CaptureNativePopulation, CaptureTensorNativeError as Failure, PreparedCaptureTensor,
};
use eredu_core::{capture::*, intervention::*};
use eredu_nn::{
    Tensor,
    workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataError, WorkspaceTensor},
};
use eredu_runtime::{
    capture::{CaptureProtocolError, FundedCaptureError},
    intervention::{ActivationHook, activation_hook, reserve_envelope},
    working_memory::{
        CaptureInterventionClaim, ClaimedIntervention, OriginalInterventionSource,
        OriginalSpeculativeBudgetCustody, WorkingMemoryFundingScope,
    },
};
use safemlx::{Array, OriginalScopeObserver, Stream};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};
mod evidence;
mod partition;
mod prefill;
mod remote;
mod routed;
mod routing;
mod window;
#[derive(Clone, Copy)]
pub(in crate::composition::mlx::session) enum NativeCustody<'a> {
    Model(&'a OriginalSpeculativeBudgetCustody),
    Scheduled(&'a WorkingMemoryFundingScope),
}
impl NativeCustody<'_> {
    pub(in crate::composition::mlx::session) fn validate(
        self,
        claim: &CaptureInterventionClaim<'_>,
    ) -> Result<(), eredu_runtime::working_memory::WorkingMemoryError> {
        match self {
            Self::Model(custody) => claim.validate_model_custody(custody),
            Self::Scheduled(scope) => claim.validate_native_custody(scope),
        }
    }
}
struct Edit {
    shape: Vec<u64>,
    slice: ResolvedCaptureSlice,
    dtype: InterventionDtype,
    usage: CaptureUsage,
    projection: [CaptureUsage; 2],
    overlap: bool,
    evidence: [evidence::State; 2],
    payload: Option<eredu_runtime::intervention::PreparedWindowInterventionPayload>,
}
enum Row {
    Inactive,
    Pending,
    Ready(Edit),
    Routed(routed::Edit),
    Routing(routing::Edit),
    PartitionReady,
    PartitionAbsent,
    Remote(remote::Source),
}
/// One actual physical invocation. The immutable source and cold-owned shapes
/// survive until native construction; they confer no native or role authority.
pub(crate) struct PreparedModelInterventions {
    rows: Vec<Row>,
    phase: CapturePhase,
    prediction: u64,
    invocation: Option<CaptureInvocationShape>,
    window: Option<CaptureInvocationWindow>,
    scheduled_span: Option<eredu_runtime::intervention::InterventionPrefillWindow>,
    partition: Option<partition::Binding>,
    begun: bool,
    sealed: bool,
    evidence_skips: Option<Vec<[Option<CaptureSkipReason>; 2]>>,
    source: OriginalInterventionSource,
    // Row vectors and C retire before their actual preparation account.
    _funding: HostMetadataFunding,
}
impl PreparedModelInterventions {
    /// Actual sparse original worker profile; count/geometry qualification stays
    /// with its retained per-provider source and native claim validation.
    pub(crate) fn routed_mechanism_facts() -> InterventionMechanismFacts<'static> {
        routed::mechanism_facts()
    }

    pub(crate) fn prepare(
        source: &OriginalInterventionSource,
        selected: &[bool],
        phase: CapturePhase,
        prediction: u64,
        invocation: CaptureInvocationShape,
        window: Option<CaptureInvocationWindow>,
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        Self::prepare_with_evidence(
            source, selected, phase, prediction, invocation, window, None, context,
        )
    }
    pub(crate) fn prepare_with_evidence(
        source: &OriginalInterventionSource,
        selected: &[bool],
        phase: CapturePhase,
        prediction: u64,
        invocation: CaptureInvocationShape,
        window: Option<CaptureInvocationWindow>,
        evidence_skips: Option<&[[Option<CaptureSkipReason>; 2]]>,
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        Self::prepare_selected(
            source,
            selected,
            phase,
            prediction,
            Some(invocation),
            window,
            evidence_skips,
            context,
        )
    }
    /// Scheduled text retains its actual phase/index without inventing a model
    /// invocation. The same row resolver and static native recipe run below.
    pub(crate) fn prepare_scheduled(
        source: &OriginalInterventionSource,
        selected: &[bool],
        phase: CapturePhase,
        prediction: u64,
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        Self::prepare_scheduled_with_evidence(source, selected, phase, prediction, None, context)
    }
    pub(crate) fn prepare_scheduled_with_evidence(
        source: &OriginalInterventionSource,
        selected: &[bool],
        phase: CapturePhase,
        prediction: u64,
        evidence_skips: Option<&[[Option<CaptureSkipReason>; 2]]>,
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        Self::prepare_selected(
            source,
            selected,
            phase,
            prediction,
            None,
            None,
            evidence_skips,
            context,
        )
    }
    fn prepare_selected(
        source: &OriginalInterventionSource,
        selected: &[bool],
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        window: Option<CaptureInvocationWindow>,
        evidence_skips: Option<&[[Option<CaptureSkipReason>; 2]]>,
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        context.charge_metadata(
            Self::inspection_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let funding = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let plan = source.plan().admission();
        if selected.len() != plan.points().len()
            || selected.len() != plan.plan().operations.len()
            || evidence_skips.is_some_and(|rows| rows.len() != selected.len())
        {
            return Err(context.metadata_source(CaptureProtocolError::Geometry));
        }
        match (plan.invocation_bounds(), invocation) {
            (Some(bounds), Some(invocation)) => bounds
                .validate_fixed(invocation, prediction)
                .map_err(|cause| context.metadata_source(cause))?,
            (None, None) if window.is_none() => (),
            _ => return Err(context.metadata_source(CaptureProtocolError::Invocation)),
        }
        if let Some(window) = window {
            window
                .validate_fixed(
                    invocation
                        .ok_or_else(|| context.metadata_source(CaptureProtocolError::Invocation))?,
                )
                .map_err(|cause| context.metadata_source(cause))?;
        }
        let mut rows = context.metadata_vec(selected.len())?;
        for (index, ((operation, point), selected)) in plan
            .plan()
            .operations
            .iter()
            .zip(plan.points())
            .zip(selected)
            .enumerate()
        {
            if !(point.stage == InterventionStage::Activation
                || (point.stage == InterventionStage::RoutingBeforeDispatch
                    && point.routing.is_some())
                || (invocation.is_none() && point.stage == InterventionStage::LogitsBeforeSampling))
                || (point.routed_units.is_some()
                    && (operation.evidence != InterventionEvidence::None
                        || !routed::supports(&operation.action)))
                || (operation.action.dtype().is_none() && point.routing.is_none())
                || !matches!(
                    (&operation.evidence, source.plan().evidence(index)),
                    (InterventionEvidence::None, None)
                        | (
                            InterventionEvidence::Preview { .. } | InterventionEvidence::Summary,
                            Some(_)
                        )
                )
                || source.plan().evidence(index).is_some_and(|companion| {
                    companion.operation() != index
                        || companion.geometry_source().plan().selections.len()
                            != if point.routing.is_some() { 4 } else { 2 }
                        || evidence_skips.is_none()
                })
            {
                return Err(context.metadata_source(Failure::ClaimMismatch));
            }
            rows.push(
                if *selected && operation.schedule.includes(phase, prediction) {
                    Row::Pending
                } else {
                    Row::Inactive
                },
            );
        }
        let evidence_skips = evidence_skips
            .map(|rows| -> Result<_, eredu_nn::Error> {
                let mut copied = context.metadata_vec(rows.len())?;
                copied.extend_from_slice(rows);
                Ok(copied)
            })
            .transpose()?;
        Ok(Self {
            rows,
            evidence_skips,
            phase,
            prediction,
            invocation,
            window,
            scheduled_span: None,
            partition: None,
            begun: false,
            sealed: false,
            source: source.clone(),
            _funding: funding,
        })
    }
    pub(crate) fn begin(
        &mut self,
        phase: CapturePhase,
        prediction: u64,
        ledger: &mut CaptureLedger,
        context: &WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        self.begin_prepaid(phase, prediction, context)?;
        // Inactive rows and source-owned companions use the exact same metadata
        // worker as the actual frame and the original outer prefix.
        self.source
            .reserve_capture_metadata(ledger)
            .map_err(|cause| context.metadata_source(cause))?;
        Ok(())
    }
    // State transition shared with ordinary cold begin. Only a source-bound
    // prefix consumer may omit the already charged declaration metadata; hook
    // projection/native/evidence reservations still happen at each actual hook.
    pub(crate) fn begin_prepaid(
        &mut self,
        phase: CapturePhase,
        prediction: u64,
        context: &WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        context.charge_metadata(
            Self::inspection_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        if self.begun || self.sealed || (phase, prediction) != (self.phase, self.prediction) {
            return Err(context.metadata_source(CaptureProtocolError::Transaction));
        }
        self.begun = true;
        Ok(())
    }
    pub(crate) fn trace(
        &mut self,
        path: &str,
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
        ledger: &mut CaptureLedger,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<(Option<WorkspaceTensor>, CaptureNativePopulation), eredu_nn::Error> {
        self.trace_with_evidence_progress(path, value, context, ledger, retained, None)
    }
    pub(in super::super) fn trace_with_evidence_progress(
        &mut self,
        path: &str,
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
        ledger: &mut CaptureLedger,
        retained: &mut Vec<WorkspaceTensor>,
        mut evidence_progress: Option<
            &mut [Option<[eredu_runtime::capture::CapturePrefillRowProgress; 4]>],
        >,
    ) -> Result<(Option<WorkspaceTensor>, CaptureNativePopulation), eredu_nn::Error> {
        context.charge_metadata(
            Self::inspection_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        if !self.begun || self.sealed {
            return Err(context.metadata_source(CaptureProtocolError::Transaction));
        }
        context.validate_values([value])?;
        let mut effective = None;
        let mut population = CaptureNativePopulation::default();
        let plan = self.source.plan().admission();
        for (index, (operation, point)) in
            plan.plan().operations.iter().zip(plan.points()).enumerate()
        {
            let outcome = match &self.rows[index] {
                Row::Inactive => InterventionOutcome::Inactive,
                Row::Pending => InterventionOutcome::Missing,
                Row::Ready(_) => InterventionOutcome::Applied,
                Row::Routed(_) => InterventionOutcome::Missing,
                Row::Routing(edit) => {
                    if edit.complete {
                        InterventionOutcome::Applied
                    } else {
                        InterventionOutcome::Missing
                    }
                }
                Row::PartitionReady | Row::Remote(_) => InterventionOutcome::Applied,
                Row::PartitionAbsent => InterventionOutcome::Inactive,
            };
            if self
                .partition
                .as_ref()
                .is_some_and(|binding| partition::remote(binding, index))
            {
                continue;
            }
            match activation_hook(operation, point, &outcome, path) {
                ActivationHook::Unrelated => continue,
                ActivationHook::Repeated => {
                    return Err(context.metadata_source(CaptureProtocolError::Transaction));
                }
                ActivationHook::Active => (),
            }
            let input = effective.as_ref().unwrap_or(value);
            if self.partition.is_some() {
                let (output, native) = partition::trace(
                    self.partition.as_mut().expect("selected partition"),
                    index,
                    input,
                    context,
                    retained,
                )?;
                self.rows[index] = Row::PartitionReady;
                population = population
                    .checked_add(native)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                if output.is_some() {
                    effective = output;
                }
                continue;
            }
            if input.shape().len() != point.axes.len() || input.shape().len() > 32 {
                return Err(context.metadata_source(Failure::ShapeMismatch));
            }
            let span = self.scheduled_span;
            let physical = span.map(|s| s.physical()).or(self.invocation);
            let window = span.map(|s| s.window()).or(self.window);
            let metadata = if window.is_some() {
                let usage =
                    eredu_runtime::intervention::intervention_window_metadata(operation, point)
                        .map_err(|cause| context.metadata_source(cause))?;
                reserve_envelope(ledger, usage).map_err(|cause| context.metadata_source(cause))?;
                usage
            } else {
                CaptureUsage::default()
            };
            let mut shape = context.metadata_vec(input.shape().len())?;
            for axis in input.shape() {
                shape.push(u64::try_from(*axis).map_err(|cause| context.metadata_source(cause))?);
            }
            let dtype = operation
                .action
                .dtype()
                .ok_or_else(|| context.metadata_source(Failure::ShapeMismatch))?;
            PreparedStaticActivation::validate_workspace_source(input, dtype)
                .map_err(|cause| context.metadata_source(cause))?;
            let geometry = window::prepare(
                plan,
                index,
                self.phase,
                self.prediction,
                physical,
                window,
                span,
                &shape,
                dtype,
                metadata,
                context,
                ledger,
            )?;
            let action = geometry
                .payload
                .as_ref()
                .and_then(|payload| payload.projected_action())
                .unwrap_or(&operation.action);
            let mut usage = CaptureUsage::default();
            let mut native = CaptureNativePopulation {
                controls: Self::execution_control_bytes()
                    .ok_or(WorkspaceMetadataError::Overflow)?,
                ..Default::default()
            };
            if span.is_some() {
                native.controls = native
                    .controls
                    .checked_add(
                        prefill::execution_control_bytes()
                            .ok_or(WorkspaceMetadataError::Overflow)?,
                    )
                    .ok_or(WorkspaceMetadataError::Overflow)?;
            }
            let program = if geometry.overlap {
                let program = PreparedStaticActivation::new(action, &geometry.slice, &shape, dtype)
                    .map_err(|cause| context.metadata_source(cause))?;
                let actual = program
                    .population()
                    .map_err(|cause| context.metadata_source(cause))?;
                usage = NativeInterventionEstimator
                    .activation_usage(&shape, &geometry.slice, action)
                    .map_err(|cause| context.metadata_source(cause))?;
                if usage.captures != 0 || usage.encoded_bytes != 0 {
                    return Err(context.metadata_source(CaptureProtocolError::Transaction));
                }
                reserve_envelope(ledger, usage).map_err(|cause| context.metadata_source(cause))?;
                native.completions = actual.completions;
                native.retained_roots = actual.retained_roots;
                native.controls = native
                    .controls
                    .checked_add(actual.controls)
                    .and_then(|n| n.checked_add(actual.host_bytes))
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                Some(program)
            } else {
                None
            };
            // Preserve the shared driver's reservation and evaluation order.
            // After is lazy until its selected/flatten frontier settles all
            // dependencies inside this same model role and observer scope.
            let (before, before_population) = evidence::trace(
                self,
                index,
                InterventionEvidenceSide::Before,
                input,
                context,
                ledger,
                retained,
                evidence_progress
                    .as_deref_mut()
                    .and_then(|rows| rows[index].as_mut())
                    .map(|rows| &mut rows[0]),
            )?;
            let output = program
                .map(|program| {
                    program
                        .trace(input, context, retained)
                        .map_err(|cause| context.metadata_source(cause))
                })
                .transpose()?;
            let (after, after_population) = evidence::trace(
                self,
                index,
                InterventionEvidenceSide::After,
                output.as_ref().unwrap_or(input),
                context,
                ledger,
                retained,
                evidence_progress
                    .as_deref_mut()
                    .and_then(|rows| rows[index].as_mut())
                    .map(|rows| &mut rows[1]),
            )?;
            let next = population
                .checked_add(native)
                .and_then(|p| p.checked_add(before_population))
                .and_then(|p| p.checked_add(after_population))
                .ok_or(WorkspaceMetadataError::Overflow)?;
            self.rows[index] = Row::Ready(Edit {
                shape,
                slice: geometry.slice,
                dtype,
                usage,
                projection: geometry.projection,
                overlap: geometry.overlap,
                evidence: [before, after],
                payload: geometry.payload,
            });
            population = next;
            if let Some(output) = output {
                effective = Some(output);
            }
        }
        Ok((effective, population))
    }
    pub(crate) fn requires_sequence_readout(&self) -> bool {
        self.rows.iter().any(|row| !matches!(row, Row::Inactive))
    }
    pub(crate) fn finish(&mut self, context: &WorkspaceContext) -> Result<(), eredu_nn::Error> {
        if !self.begun
            || self.sealed
            || self.rows.iter().any(|row| {
                matches!(row, Row::Pending)
                    || matches!(row, Row::Routed(edit) if !edit.complete())
                    || matches!(row, Row::Routing(edit) if !edit.complete)
            })
        {
            return Err(context.metadata_source(CaptureProtocolError::Transaction));
        }
        self.sealed = true;
        Ok(())
    }
    fn checked<'a>(
        &'a self,
        value: &Array,
        claim: &CaptureInterventionClaim<'_>,
        custody: NativeCustody<'_>,
    ) -> Result<(&'a Edit, Option<PreparedStaticActivation<'a>>), Failure> {
        custody.validate(claim)?;
        claim.validate_source(&self.source)?;
        if !self.sealed
            || claim.coordinate() != (self.phase, self.prediction)
            || claim.invocation() != self.invocation
            || claim.invocation_window() != self.window
        {
            return Err(Failure::ClaimMismatch);
        }
        let Some(Row::Ready(edit)) = self.rows.get(claim.index()) else {
            return Err(Failure::ClaimMismatch);
        };
        let dtype = match value.dtype() {
            safemlx::Dtype::Float32 => InterventionDtype::Float32,
            safemlx::Dtype::Float16 => InterventionDtype::Float16,
            safemlx::Dtype::Bfloat16 => InterventionDtype::Bfloat16,
            other => return Err(Failure::UnsupportedDtype(other)),
        };
        if dtype != edit.dtype
            || value.shape().len() != edit.shape.len()
            || value
                .shape()
                .iter()
                .zip(&edit.shape)
                .any(|(a, b)| u64::try_from(*a).ok() != Some(*b))
        {
            return Err(Failure::ShapeMismatch);
        }
        let action = &self.source.plan().admission().plan().operations[claim.index()].action;
        Ok((
            edit,
            if edit.overlap {
                let action = edit
                    .payload
                    .as_ref()
                    .and_then(|payload| payload.projected_action())
                    .unwrap_or(action);
                Some(PreparedStaticActivation::new(
                    action,
                    &edit.slice,
                    &edit.shape,
                    edit.dtype,
                )?)
            } else {
                None
            },
        ))
    }
    pub(crate) fn usage(
        &self,
        value: &Array,
        claim: &CaptureInterventionClaim<'_>,
        custody: &OriginalSpeculativeBudgetCustody,
    ) -> Result<CaptureUsage, Failure> {
        self.checked(value, claim, NativeCustody::Model(custody))
            .map(|(edit, _)| edit.usage)
    }
    pub(crate) fn projection_usage(
        &self,
        value: &Array,
        claim: &CaptureInterventionClaim<'_>,
        custody: &OriginalSpeculativeBudgetCustody,
    ) -> Result<[CaptureUsage; 2], Failure> {
        self.checked(value, claim, NativeCustody::Model(custody))
            .map(|(edit, _)| edit.projection)
    }
    pub(crate) fn execute(
        &self,
        value: &Array,
        claim: CaptureInterventionClaim<'_>,
        charged: CaptureUsage,
        projection: [CaptureUsage; 2],
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        custody: &OriginalSpeculativeBudgetCustody,
        observer: &OriginalScopeObserver,
    ) -> Result<(Option<Array>, ClaimedIntervention), FundedCaptureError<Failure>> {
        self.execute_with(
            value,
            claim,
            charged,
            projection,
            stream,
            roots,
            NativeCustody::Model(custody),
            observer,
        )
    }
    pub(crate) fn usage_scheduled(
        &self,
        value: &Array,
        claim: &CaptureInterventionClaim<'_>,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<CaptureUsage, Failure> {
        self.checked(value, claim, NativeCustody::Scheduled(scope))
            .map(|(edit, _)| edit.usage)
    }
    pub(crate) fn projection_usage_scheduled(
        &self,
        value: &Array,
        claim: &CaptureInterventionClaim<'_>,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<[CaptureUsage; 2], Failure> {
        self.checked(value, claim, NativeCustody::Scheduled(scope))
            .map(|(edit, _)| edit.projection)
    }
    pub(crate) fn execute_scheduled(
        &self,
        value: &Array,
        claim: CaptureInterventionClaim<'_>,
        charged: CaptureUsage,
        projection: [CaptureUsage; 2],
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        scope: &WorkingMemoryFundingScope,
        observer: &OriginalScopeObserver,
    ) -> Result<(Option<Array>, ClaimedIntervention), FundedCaptureError<Failure>> {
        self.execute_with(
            value,
            claim,
            charged,
            projection,
            stream,
            roots,
            NativeCustody::Scheduled(scope),
            observer,
        )
    }
    fn execute_with(
        &self,
        value: &Array,
        claim: CaptureInterventionClaim<'_>,
        charged: CaptureUsage,
        projection: [CaptureUsage; 2],
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        custody: NativeCustody<'_>,
        observer: &OriginalScopeObserver,
    ) -> Result<(Option<Array>, ClaimedIntervention), FundedCaptureError<Failure>> {
        let (edit, program) = self
            .checked(value, &claim, custody)
            .map_err(FundedCaptureError::Backend)?;
        if charged != edit.usage || projection != edit.projection {
            return Err(FundedCaptureError::Backend(Failure::ClaimMismatch));
        }
        let output = Self::execute_program(program, value, stream, observer, roots)?;
        let charged = charged
            .checked_add(projection[0])?
            .checked_add(projection[1])?;
        Ok((output, claim.finish(charged)?))
    }
    fn execute_program(
        program: Option<PreparedStaticActivation<'_>>,
        value: &Array,
        stream: &Stream,
        observer: &OriginalScopeObserver,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<Option<Array>, FundedCaptureError<Failure>> {
        PreparedCaptureTensor::validate_stream(stream).map_err(FundedCaptureError::Backend)?;
        program
            .map(|program| program.execute_array(value, stream, observer, roots))
            .transpose()
            .map_err(FundedCaptureError::Backend)
    }
    pub(super) fn selected(&self, index: usize) -> bool {
        self.rows
            .get(index)
            .is_some_and(|row| !matches!(row, Row::Inactive))
    }
    pub(crate) fn inspection_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Row>(),
            routing::control_bytes()?,
            partition::control_bytes()?,
            size_of::<Edit>(),
            size_of::<Option<CaptureInvocationWindow>>(),
            size_of::<Option<eredu_runtime::intervention::InterventionPrefillWindow>>(),
            size_of::<Option<CaptureInvocationShape>>(),
            size_of::<[CaptureUsage; 2]>(),
            size_of::<Option<&mut [Option<[eredu_runtime::capture::CapturePrefillRowProgress; 4]>]>>(
            ),
            window::control_bytes()?,
            evidence::planning_controls()?,
            size_of::<Option<Vec<[Option<CaptureSkipReason>; 2]>>>(),
            size_of::<Option<&[[Option<CaptureSkipReason>; 2]]>>(),
            size_of::<Result<Vec<[Option<CaptureSkipReason>; 2]>, eredu_nn::Error>>(),
            size_of::<ResolvedCaptureSlice>(),
            size_of::<[Vec<u64>; 5]>(),
            size_of::<(CapturePhase, u64, CaptureInvocationShape)>(),
            size_of::<InterventionOutcome>(),
            size_of::<ActivationHook>(),
            size_of::<NativeInterventionEstimator>(),
            size_of::<CaptureNativePopulation>(),
            size_of::<Option<WorkspaceTensor>>(),
            size_of::<Option<PreparedStaticActivation<'static>>>(),
            size_of::<[(evidence::State, CaptureNativePopulation); 2]>(),
            size_of::<Result<Self, eredu_nn::Error>>(),
            size_of::<Result<(Option<WorkspaceTensor>, CaptureNativePopulation), eredu_nn::Error>>(
            ),
            size_of::<Result<(), eredu_nn::Error>>(),
            size_of::<(
                &Self,
                &str,
                &WorkspaceTensor,
                &WorkspaceContext,
                &mut CaptureLedger,
                &mut Vec<WorkspaceTensor>,
            )>(),
            PreparedStaticActivation::inspection_control_bytes()?,
            NativeInterventionEstimator::prepared_preflight_control_bytes()?,
            OriginalInterventionSource::validation_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    fn execution_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<NativeCustody<'static>>(),
            size_of::<(&NativeCustody<'_>, &CaptureInterventionClaim<'_>)>(),
            size_of::<(
                &Self,
                &Array,
                &CaptureInterventionClaim<'_>,
                &WorkingMemoryFundingScope,
            )>(),
            size_of::<(
                &Self,
                &Array,
                CaptureInterventionClaim<'_>,
                CaptureUsage,
                [CaptureUsage; 2],
                &Stream,
                &RefCell<Vec<Array>>,
                &WorkingMemoryFundingScope,
                &OriginalScopeObserver,
            )>(),
            size_of::<(
                &Self,
                &Array,
                CaptureInterventionClaim<'_>,
                CaptureUsage,
                [CaptureUsage; 2],
                &Stream,
                &RefCell<Vec<Array>>,
                NativeCustody<'_>,
                &OriginalScopeObserver,
            )>(),
            size_of::<CaptureInterventionClaim<'static>>(),
            size_of::<ClaimedIntervention>(),
            size_of::<PreparedStaticActivation<'static>>(),
            size_of::<(
                &Self,
                &Array,
                &CaptureInterventionClaim<'static>,
                &OriginalSpeculativeBudgetCustody,
            )>(),
            size_of::<(
                &Self,
                &Array,
                CaptureInterventionClaim<'static>,
                CaptureUsage,
                [CaptureUsage; 2],
                &Stream,
                &RefCell<Vec<Array>>,
                &OriginalSpeculativeBudgetCustody,
                &OriginalScopeObserver,
            )>(),
            size_of::<Result<(&Edit, Option<PreparedStaticActivation<'static>>), Failure>>(),
            size_of::<Result<[CaptureUsage; 2], Failure>>(),
            size_of::<Result<CaptureUsage, Failure>>(),
            size_of::<FundedCaptureError<Failure>>(),
            size_of::<Result<(Option<Array>, ClaimedIntervention), FundedCaptureError<Failure>>>(),
            size_of::<
                Result<
                    (Option<crate::MlxTensor>, ClaimedIntervention),
                    FundedCaptureError<Failure>,
                >,
            >(),
            size_of::<(&crate::MlxTensor, &CaptureInterventionClaim<'static>)>(),
            size_of::<(
                &crate::MlxTensor,
                CaptureInterventionClaim<'static>,
                CaptureUsage,
                [CaptureUsage; 2],
            )>(),
            size_of::<[Option<Array>; 2]>(),
            Stream::device_type_control_bytes()?,
            OriginalScopeObserver::control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
