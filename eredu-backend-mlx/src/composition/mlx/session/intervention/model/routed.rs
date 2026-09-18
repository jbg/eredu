//! One original operation source spans actual serial provider batches.
use super::*;
use super::super::static_activation::{SparseScalarEnvelope, PreparedSparseActivation};
use crate::backend::array_copy::CompletedRoutedCaptureSource;
use eredu_runtime::{RoutedUnitBatch, RoutedUnitInvocation,
    intervention::{PreparedRoutedInterventionRows, routed_intervention_full_component_count},
    working_memory::RoutedInterventionBatch};
use eredu_nn::workspace::WorkspaceDtype;

pub(super) fn mechanism_facts() -> InterventionMechanismFacts<'static> {
    InterventionMechanismFacts { routed_units: true,
        operations: &[InterventionKind::Zero, InterventionKind::Scale],
        dtypes: &[InterventionDtype::Float32], score_stages: &[] }
}
pub(super) fn supports(action: &InterventionAction) -> bool {
    let facts=mechanism_facts();
    facts.operations.contains(&action.kind())
        && action.dtype().is_some_and(|dtype|facts.dtypes.contains(&dtype))
}
struct Batch {
    range: [u64; 2],
    shape: [i32; 2],
    envelope: SparseScalarEnvelope,
    // Finite source-derived lowering/index/error storage, prepaid before the
    // enclosing original quote seals. The counter cannot create native credit.
    funding: HostMetadataFunding,
}
pub(super) struct Edit {
    bank: RoutedUnitGeometry,
    tokens: u64,
    logical_origin: u64,
    selected: ResolvedCaptureSlice,
    next: u64,
    finished: bool,
    usage: CaptureUsage,
    batches: Vec<Batch>,
}
impl Edit {
    pub(super) fn complete(&self) -> bool { self.finished && self.next == self.tokens }
}
impl PreparedModelInterventions {
    pub(crate) fn routed_selection(&self, path: &str) -> Option<usize> {
        self.source.plan().admission().points().iter().enumerate().position(|(index, point)|
            !matches!(self.rows[index], Row::Inactive)
                && point.routed_units.as_ref().is_some_and(|point| point.routing == path))
    }
    fn same_routed(&self, first: usize, index: usize) -> bool {
        let points = self.source.plan().admission().points();
        points.get(first).and_then(|p| p.routed_units.as_ref()).zip(
            points.get(index).and_then(|p| p.routed_units.as_ref()))
            .is_some_and(|(first, point)| first.routing == point.routing)
    }
    pub(crate) fn begin_routed(&mut self, first: usize, input: &RoutedUnitInvocation<'_, WorkspaceTensor>,
        context: &WorkspaceContext, ledger: &mut CaptureLedger) -> Result<(), eredu_nn::Error> {
        context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        let physical = self.invocation.ok_or_else(|| context.metadata_source(CaptureProtocolError::Invocation))?;
        let tokens = physical.batch.checked_mul(physical.sequence).ok_or(WorkspaceMetadataError::Overflow)?;
        if !self.begun || self.sealed || input.origins.is_some() || input.unit_coordinates.is_some()
            || input.input.shape().split_last().and_then(|(_, axes)| axes.iter().try_fold(1u64, |n, x|
                n.checked_mul(u64::try_from(*x).ok()?))) != Some(tokens) {
            return Err(context.metadata_source(CaptureProtocolError::Invocation));
        }
        let logical = self.window.map(|window| window.validate_fixed(physical)).transpose()
            .map_err(|cause| context.metadata_source(cause))?.unwrap_or(physical);
        let logical_tokens = logical.batch.checked_mul(logical.sequence).ok_or(WorkspaceMetadataError::Overflow)?;
        let plan = self.source.plan().admission();
        for index in 0..self.rows.len() {
            if !self.same_routed(first, index) || matches!(self.rows[index], Row::Inactive) { continue; }
            if !matches!(self.rows[index], Row::Pending) {
                return Err(context.metadata_source(CaptureProtocolError::Transaction));
            }
            let bank = plan.points()[index].routed_units.as_ref()
                .ok_or_else(|| context.metadata_source(CaptureProtocolError::Geometry))?.geometry;
            let components = bank.experts.checked_mul(bank.units_per_expert).ok_or(WorkspaceMetadataError::Overflow)?;
            let action = &plan.plan().operations[index].action;
            // The fixed resolver consumes initialized rank entries, not merely
            // paid backing capacity. Use the same ordinary row destination.
            let mut slice = super::window::slice(2, context)?;
            plan.resolve_prepared_invocation_at(index, self.phase, self.prediction, logical,
                &[logical_tokens, components], action.dtype().ok_or_else(|| context.metadata_source(Failure::ClaimMismatch))?,
                &mut slice).map_err(|cause| context.metadata_source(cause))?;
            // Complete expert components make the selected count independent
            // of expert IDs. Each actual chunk supplies its own token window;
            // native row coverage and actual lowered indices are checked later.
            let logical_origin = self.window.map_or(0, |window| window.start);
            let logical_end = logical_origin.checked_add(tokens).ok_or(WorkspaceMetadataError::Overflow)?;
            if routed_intervention_full_component_count(bank, &slice, [logical_origin, logical_end])
                .map_err(|cause| context.metadata_source(cause))?.is_none() {
                return Err(context.metadata_source(SourceFailure::SelectedCount));
            }
            let usage = NativeInterventionEstimator.window_routed_unit_usage(bank,
                &[logical_tokens, components], &[tokens, components], &slice, action)
                .map_err(|cause| context.metadata_source(cause))?;
            reserve_envelope(ledger, usage).map_err(|cause| context.metadata_source(cause))?;
            self.rows[index] = Row::Routed(Edit { bank, tokens, logical_origin, selected: slice, next: 0, finished: false, usage,
                batches: context.metadata_vec(0)? });
        }
        Ok(())
    }
    pub(crate) fn trace_routed(&mut self, first: usize, batch: &RoutedUnitBatch<'_, WorkspaceTensor>,
        context: &WorkspaceContext, retained: &mut Vec<WorkspaceTensor>)
        -> Result<(Option<WorkspaceTensor>, CaptureNativePopulation), eredu_nn::Error> {
        context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        if !self.begun || self.sealed || batch.origins.is_some() || batch.unit_coordinates.is_some() {
            return Err(context.metadata_source(CaptureProtocolError::Invocation));
        }
        let source = batch.capture_source().map_err(|cause| context.metadata_source(cause))?;
        let mut effective = None;
        let mut population = CaptureNativePopulation::default();
        for index in 0..self.rows.len() {
            if !self.same_routed(first, index) || matches!(self.rows[index], Row::Inactive) { continue; }
            let Row::Routed(edit) = &mut self.rows[index] else {
                return Err(context.metadata_source(CaptureProtocolError::Transaction));
            };
            let input = effective.as_ref().unwrap_or(source.values);
            let values = [input, source.token_indices, source.selection_indices, source.coefficients, source.source_groups];
            for (slot, value) in values.iter().enumerate() {
                let dtype = value.layout().dtype();
                if !(if matches!(slot, 0 | 3) { dtype == WorkspaceDtype::Float32 }
                    else { matches!(dtype, WorkspaceDtype::Int32 | WorkspaceDtype::Uint32) }) {
                    return Err(context.metadata_source(Failure::ShapeMismatch));
                }
            }
            context.validate_values(values)?;
            let (rows, chunk) = CompletedRoutedCaptureSource::validate_geometry(values.map(|v| v.shape()),
                source.token_offset, edit.bank, edit.tokens).map_err(|cause| context.metadata_source(cause))?;
            let end = source.token_offset.checked_add(chunk).ok_or(WorkspaceMetadataError::Overflow)?;
            if edit.finished || edit.next != source.token_offset || end > edit.tokens {
                return Err(context.metadata_source(CaptureProtocolError::Transaction));
            }
            if u64::try_from(rows).ok() != chunk.checked_mul(edit.bank.routes_per_token) {
                return Err(context.metadata_source(SourceFailure::ProviderRows));
            }
            let shape = [i32::try_from(rows).map_err(|cause| context.metadata_source(cause))?,
                i32::try_from(edit.bank.units_per_expert).map_err(|cause| context.metadata_source(cause))?];
            let action = &self.source.plan().admission().plan().operations[index].action;
            let logical_range = [source.token_offset, end].map(|token| token.checked_add(edit.logical_origin));
            let logical_range = [logical_range[0].ok_or(WorkspaceMetadataError::Overflow)?,
                logical_range[1].ok_or(WorkspaceMetadataError::Overflow)?];
            let count = routed_intervention_full_component_count(edit.bank, &edit.selected, logical_range)
                .map_err(|cause| context.metadata_source(cause))?
                .ok_or_else(|| context.metadata_source(SourceFailure::SelectedCount))?;
            let envelope = SparseScalarEnvelope::prepare(action, shape, count, context)
                .map_err(|cause| context.metadata_source(cause))?;
            let native = envelope.population().map_err(|cause| context.metadata_source(cause))?;
            let bytes = PreparedRoutedInterventionRows::required_bytes(&self.source, index, rows)
                .and_then(|n| n.checked_add(WorkspaceContext::metadata_vec_bytes::<i32>(envelope.index_count())?))
                .and_then(|n| n.checked_add(PreparedSparseActivation::inspection_control_bytes()?))
                .and_then(|n| n.checked_add(native_error_controls()?))
                .ok_or(WorkspaceMetadataError::Overflow)?;
            let funding = prepaid(bytes, context)?;
            context.reserve_metadata_vec(retained, 5)?;
            retained.extend(values.into_iter().cloned());
            let output = envelope.trace_bound(input, context, retained).map_err(|cause| context.metadata_source(cause))?;
            let controls = native.controls.checked_add(native.host_bytes)
                .and_then(|n| n.checked_add(CompletedRoutedCaptureSource::intervention_read_control_bytes()?))
                .and_then(|n| n.checked_add(control_bytes()?)).ok_or(WorkspaceMetadataError::Overflow)?;
            // Five routed source loans settle before the selected edit. The
            // shared Native worker also settles every emitted edit prefix.
            let current = CaptureNativePopulation { publications: 0,
                completions: native.completions.checked_add(5).ok_or(WorkspaceMetadataError::Overflow)?,
                retained_roots: native.retained_roots.checked_add(5).ok_or(WorkspaceMetadataError::Overflow)?, controls };
            population = population.checked_add(current).ok_or(WorkspaceMetadataError::Overflow)?;
            context.reserve_metadata_vec(&mut edit.batches, 1)?;
            edit.batches.push(Batch { range: [source.token_offset, end], shape, envelope, funding });
            edit.next = end;
            if output.is_some() { effective = output; }
        }
        Ok((effective, population))
    }
    pub(crate) fn finish_routed(&mut self, first: usize, success: bool, context: &WorkspaceContext) -> Result<(), eredu_nn::Error> {
        context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        for index in 0..self.rows.len() {
            if !self.same_routed(first, index) || matches!(self.rows[index], Row::Inactive) { continue; }
            let Row::Routed(edit) = &mut self.rows[index] else {
                return Err(context.metadata_source(CaptureProtocolError::Transaction));
            };
            if !success || edit.finished || edit.next != edit.tokens {
                return Err(context.metadata_source(CaptureProtocolError::Transaction));
            }
            edit.finished = true;
        }
        Ok(())
    }
    fn checked_routed(&self, source: &RoutedUnitCaptureSource<'_, Array>, claim: &CaptureInterventionClaim<'_>,
        custody: &OriginalSpeculativeBudgetCustody) -> Result<(&Edit, &Batch), Failure> {
        claim.validate_model_custody(custody)?;
        claim.validate_source(&self.source)?;
        if !self.sealed || claim.coordinate() != (self.phase, self.prediction)
            || claim.invocation() != self.invocation || claim.invocation_window() != self.window {
            return Err(Failure::ClaimMismatch);
        }
        let Some(Row::Routed(edit)) = self.rows.get(claim.index()) else { return Err(Failure::ClaimMismatch); };
        if !edit.complete() { return Err(Failure::ClaimMismatch); }
        let (rows, chunk) = CompletedRoutedCaptureSource::validate_borrowed(source, edit.bank, edit.tokens)?;
        let end = source.token_offset.checked_add(chunk).ok_or(Failure::GeometryOverflow)?;
        let batch = edit.batches.iter().find(|batch| batch.range == [source.token_offset, end])
            .ok_or(Failure::ClaimMismatch)?;
        if usize::try_from(batch.shape[0]).ok() != Some(rows) { return Err(Failure::ShapeMismatch); }
        Ok((edit, batch))
    }
    pub(crate) fn routed_range(&self, source: &RoutedUnitCaptureSource<'_, Array>, claim: &CaptureInterventionClaim<'_>,
        custody: &OriginalSpeculativeBudgetCustody) -> Result<[u64; 2], Failure> {
        self.checked_routed(source, claim, custody).map(|(_, batch)| batch.range)
    }
    pub(crate) fn routed_usage(&self, source: &RoutedUnitCaptureSource<'_, Array>, claim: &CaptureInterventionClaim<'_>,
        custody: &OriginalSpeculativeBudgetCustody) -> Result<CaptureUsage, Failure> {
        self.checked_routed(source, claim, custody).map(|(edit, _)| edit.usage)
    }
    pub(crate) fn execute_routed(&self, source: &RoutedUnitCaptureSource<'_, Array>, batch: RoutedInterventionBatch<'_, '_>,
        stream: &Stream, roots: &RefCell<Vec<Array>>, custody: &OriginalSpeculativeBudgetCustody,
        observer: &OriginalScopeObserver) -> Result<Option<Array>, FundedCaptureError<Failure>> {
        let (edit, selected) = self.checked_routed(source, batch.claim(), custody).map_err(FundedCaptureError::Backend)?;
        if batch.source_chunk() != (edit.tokens, selected.range) {
            return Err(FundedCaptureError::Backend(Failure::ClaimMismatch));
        }
        let error = |cause| FundedCaptureError::Backend(Failure::Workspace(selected.funding.metadata_source(cause)));
        let claim = batch.claim();
        let mut rows = PreparedRoutedInterventionRows::prepare(&self.source, claim.index(), self.phase, self.prediction,
            self.invocation, self.window, edit.tokens, selected.range, selected.shape[0] as usize, selected.funding.clone())
            .map_err(|cause| error(NativeFailure::Lowering(cause)))?;
        CompletedRoutedCaptureSource::settle_intervention_rows(source, &mut rows, stream, roots, observer)
            .map_err(|cause| error(NativeFailure::Read(cause)))?;
        let lowered = rows.finish().map_err(|cause| error(NativeFailure::Lowering(cause)))?;
        let dtype = self.source.plan().admission().plan().operations[claim.index()].action.dtype()
            .ok_or_else(|| error(NativeFailure::Native(Failure::ClaimMismatch)))?;
        let prepared = PreparedSparseActivation::prepare(lowered, selected.shape, dtype, selected.funding.clone())
            .map_err(|cause| error(NativeFailure::Activation(cause)))?;
        if prepared.lowered().indices().len() != selected.envelope.index_count() {
            return Err(error(NativeFailure::Native(Failure::SourceChanged)));
        }
        selected.envelope.validate_refinement(&prepared).map_err(|cause| error(NativeFailure::Native(cause)))?;
        let (output, prepared) = prepared.execute(source.values, stream, observer, roots)
            .map_err(|cause| error(NativeFailure::Activation(cause)))?;
        batch.finish(prepared.lowered()).map_err(FundedCaptureError::from)?;
        Ok(output)
    }
}
#[derive(Debug, thiserror::Error)]
enum SourceFailure {
    #[error("routed intervention requires a qualified source for its data-dependent selected index count")]
    SelectedCount,
    #[error("routed intervention provider rows do not cover the complete serial token range")]
    ProviderRows,
}
#[derive(Debug, thiserror::Error)]
enum NativeFailure {
    #[error(transparent)] Lowering(#[from] eredu_runtime::intervention::PreparedRoutedInterventionError),
    #[error(transparent)] Read(#[from] crate::backend::array_copy::RoutedInterventionReadError),
    #[error(transparent)] Activation(#[from] super::super::SparseActivationFailure),
    #[error(transparent)] Native(#[from] Failure),
}
fn native_error_controls() -> Option<usize> {
    eredu_core::BackendFailure::source_retention_peak_bytes::<NativeFailure>()?
        .checked_mul(4)?.checked_add(size_of::<NativeFailure>())?
        .checked_add(size_of::<FundedCaptureError<Failure>>())
}
fn prepaid(bytes: usize, context: &WorkspaceContext) -> Result<HostMetadataFunding, eredu_nn::Error> {
    let parent = context.metadata_funding().ok_or(WorkspaceMetadataError::Unqualified)?;
    let limit = bytes.checked_add(eredu_core::HostMetadataFunding::prepaid_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)
        .ok_or(WorkspaceMetadataError::Overflow)?;
    context.charge_metadata(limit.checked_add(eredu_core::HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
        .ok_or(WorkspaceMetadataError::Overflow)?).ok_or(WorkspaceMetadataError::Overflow)?)?;
    eredu_core::HostMetadataFunding::from_prepaid(limit, eredu_core::HostPreparationAuthority::retain(parent))
        .map(HostMetadataFunding::from).map_err(WorkspaceMetadataError::from).map_err(eredu_nn::Error::from)
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [size_of::<Edit>(), size_of::<Batch>(), size_of::<Option<Batch>>(),
        size_of::<[RoutedUnitGeometry; 2]>(), size_of::<RoutedUnitBatch<'_, WorkspaceTensor>>(),
        size_of::<RoutedUnitInvocation<'_, WorkspaceTensor>>(), size_of::<RoutedInterventionBatch<'_, '_>>(),
        size_of::<PreparedRoutedInterventionRows>(), size_of::<NativeFailure>(), size_of::<SourceFailure>(),
        size_of::<CaptureNativePopulation>(), size_of::<Result<Option<Array>, FundedCaptureError<Failure>>>(),
        size_of::<[&WorkspaceTensor; 5]>(), size_of::<[u64; 2]>() * 3,
        size_of::<[Option<u64>; 2]>(), size_of::<Option<usize>>(),
        size_of::<Result<Option<usize>, eredu_runtime::intervention::RoutedInterventionLoweringError>>(),
        size_of::<Result<(Option<WorkspaceTensor>, CaptureNativePopulation), eredu_nn::Error>>(),
        size_of::<Result<HostMetadataFunding, eredu_nn::Error>>(),
        SparseScalarEnvelope::control_bytes()?, native_error_controls()?,
        eredu_runtime::intervention::routed_intervention_full_component_count_control_bytes()?,
    ];
    frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
