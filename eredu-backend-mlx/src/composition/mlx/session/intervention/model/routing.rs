//! The ordinary selector's source projection retained for one original hook.
use super::*;
use eredu_nn::routing_intervention::GroupSelectionControl;
use eredu_runtime::{intervention::PreparedRoutingControl, RoutingDecision};
use std::cell::Cell;

pub(super) struct Edit {
    pub(super) rows: u64,
    pub(super) top_k: u32,
    pub(super) usage: CaptureUsage,
    pub(super) evidence: [evidence::State; 4],
    control: RefCell<Option<GroupSelectionControl>>,
    taken: Cell<bool>,
    pub(super) complete: bool,
    unmodified: bool,
}
impl PreparedModelInterventions {
    pub(crate) fn trace_routing_control(
        &mut self,
        path: &str,
        rows: u64,
        context: &WorkspaceContext,
        ledger: &mut CaptureLedger,
    ) -> Result<Option<GroupSelectionControl>, eredu_nn::Error> {
        context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        if !self.begun || self.sealed || self.partition.is_some() {
            return Err(context.metadata_source(Failure::ClaimMismatch));
        }
        let plan = self.source.plan().admission();
        let Some(index) = plan
            .plan()
            .operations
            .iter()
            .zip(plan.points())
            .enumerate()
            .find_map(|(index, (operation, point))| {
                (operation.target == path
                    && point.routing.is_some()
                    && !matches!(self.rows[index], Row::Inactive))
                .then_some(index)
            })
        else {
            return Ok(None);
        };
        if !matches!(self.rows[index], Row::Pending) {
            return Err(context.metadata_source(Failure::ClaimMismatch));
        }
        let physical = match self.scheduled_span {
            Some(span) => span.physical(),
            None => match self.invocation {
                Some(shape) => shape,
                None => plan
                    .geometry_at(self.phase, self.prediction, None)
                    .map_err(|cause| context.metadata_source(cause))?,
            },
        };
        if physical.batch.checked_mul(physical.sequence) != Some(rows) {
            return Err(context.metadata_source(Failure::ShapeMismatch));
        }
        let window = self
            .scheduled_span
            .map(|span| span.window())
            .or(self.window);
        let logical = window
            .map(|window| window.validate_fixed(physical))
            .transpose()
            .map_err(|cause| context.metadata_source(cause))?
            .unwrap_or(physical);
        let logical_rows = logical
            .batch
            .checked_mul(logical.sequence)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let operation = &plan.plan().operations[index];
        let policy = plan.points()[index]
            .routing
            .as_ref()
            .expect("selected routing source");
        let mut slice = window::slice(2, context)?;
        plan.resolve_prepared_routing_at(
            index,
            self.phase,
            self.prediction,
            self.invocation.map(|_| logical),
            &[logical_rows, u64::from(policy.top_k)],
            &mut slice,
        )
        .map_err(|cause| context.metadata_source(cause))?;
        let source = PreparedRoutingControl::inspect(operation, policy, rows, &slice, window)
            .map_err(|cause| context.metadata_source(cause))?;
        let mut usage = source
            .as_ref()
            .map(|source| source.usage())
            .transpose()
            .map_err(|cause| context.metadata_source(cause))?
            .unwrap_or_default();
        if source.is_some() && operation.evidence != InterventionEvidence::None {
            usage = usage
                .checked_add(
                    NativeInterventionEstimator
                        .original_route_usage(policy, rows)
                        .map_err(|cause| context.metadata_source(cause))?,
                )
                .map_err(|cause| context.metadata_source(cause))?;
        }
        reserve_envelope(ledger, usage).map_err(|cause| context.metadata_source(cause))?;
        // The quote consumes its own control. A separately paid exact copy is
        // moved once into the authenticated hot invocation; no hot allocation
        // or callback can obtain another source control from this row.
        let cold = source
            .as_ref()
            .map(|source| source.construct(&self._funding))
            .transpose()
            .map_err(|cause| context.metadata_source(cause))?;
        let hot = source
            .as_ref()
            .map(|source| source.construct(&self._funding))
            .transpose()
            .map_err(|cause| context.metadata_source(cause))?;
        self.rows[index] = Row::Routing(Edit {
            rows,
            top_k: policy.top_k,
            usage,
            evidence: [evidence::State::Inactive; 4],
            unmodified: source.is_none(),
            control: RefCell::new(hot),
            taken: Cell::new(false),
            complete: false,
        });
        Ok(cold)
    }
    pub(crate) fn routing_unmodified_interest(
        &self,
        path: &str,
    ) -> eredu_runtime::RoutingUnmodifiedInterest {
        if self.source.plan().admission().plan().operations.iter().enumerate().any(|(index,operation)|
            operation.target==path && matches!(&self.rows[index],Row::Routing(edit) if edit.unmodified && !edit.complete)) {
            eredu_runtime::RoutingUnmodifiedInterest::Metadata
        } else {eredu_runtime::RoutingUnmodifiedInterest::None}
    }
    pub(crate) fn trace_routing_result(
        &mut self,
        path: &str,
        original: Option<RoutingDecision<'_, WorkspaceTensor>>,
        effective: RoutingDecision<'_, WorkspaceTensor>,
        unmodified: bool,
        context: &WorkspaceContext,
        ledger: &mut CaptureLedger,
        roots: &mut Vec<WorkspaceTensor>,
        mut progress: Option<&mut [Option<[eredu_runtime::capture::CapturePrefillRowProgress; 4]>]>,
    ) -> Result<CaptureNativePopulation, eredu_nn::Error> {
        context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        let index = self
            .source
            .plan()
            .admission()
            .plan()
            .operations
            .iter()
            .position(|op| op.target == path)
            .ok_or_else(|| context.metadata_source(Failure::ClaimMismatch))?;
        let Some(Row::Routing(edit)) = self.rows.get(index) else {
            return Err(context.metadata_source(Failure::ClaimMismatch));
        };
        if edit.complete || edit.unmodified != unmodified {
            return Err(context.metadata_source(Failure::ClaimMismatch));
        }
        let shape = [
            i32::try_from(edit.rows).map_err(|cause| context.metadata_source(cause))?,
            i32::try_from(edit.top_k).map_err(|cause| context.metadata_source(cause))?,
        ];
        for decision in original
            .as_ref()
            .into_iter()
            .chain(std::iter::once(&effective))
        {
            context.validate_values([decision.ids, decision.coefficients])?;
            if decision.ids.shape() != shape
                || decision.coefficients.shape() != shape
                || decision.ids.layout().dtype() != eredu_nn::workspace::WorkspaceDtype::Uint32
                || decision.coefficients.layout().dtype()
                    != eredu_nn::workspace::WorkspaceDtype::Float32
            {
                return Err(context.metadata_source(Failure::ShapeMismatch));
            }
        }
        let mut population = CaptureNativePopulation {
            controls: control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
            ..Default::default()
        };
        let mut states = [evidence::State::Inactive; 4];
        if self.source.plan().evidence(index).is_some() {
            let original =
                original.ok_or_else(|| context.metadata_source(Failure::ClaimMismatch))?;
            for (ordinal, (side, field, value)) in [
                (
                    InterventionEvidenceSide::Before,
                    eredu_core::RoutingObservationField::SelectedExperts,
                    original.ids,
                ),
                (
                    InterventionEvidenceSide::Before,
                    eredu_core::RoutingObservationField::Coefficients,
                    original.coefficients,
                ),
                (
                    InterventionEvidenceSide::After,
                    eredu_core::RoutingObservationField::SelectedExperts,
                    effective.ids,
                ),
                (
                    InterventionEvidenceSide::After,
                    eredu_core::RoutingObservationField::Coefficients,
                    effective.coefficients,
                ),
            ]
            .into_iter()
            .enumerate()
            {
                let (state, current) = evidence::trace_field(
                    self,
                    index,
                    side,
                    Some(field),
                    value,
                    context,
                    ledger,
                    roots,
                    progress
                        .as_deref_mut()
                        .and_then(|rows| rows[index].as_mut())
                        .map(|rows| &mut rows[ordinal]),
                )?;
                states[ordinal] = state;
                population = population
                    .checked_add(current)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
            }
        }
        let Row::Routing(edit) = &mut self.rows[index] else {
            unreachable!()
        };
        edit.evidence = states;
        edit.complete = true;
        Ok(population)
    }
    fn checked_routing(
        &self,
        rows: u64,
        claim: &CaptureInterventionClaim<'_>,
        scope: &WorkingMemoryFundingScope,
        window: Option<eredu_runtime::intervention::InterventionPrefillWindow>,
    ) -> Result<&Edit, Failure> {
        claim.validate_native_custody(scope)?;
        claim.validate_source(&self.source)?;
        if !self.sealed
            || claim.coordinate() != (self.phase, self.prediction)
            || claim.invocation().is_some()
            || window != self.scheduled_span
        {
            return Err(Failure::ClaimMismatch);
        }
        let Some(Row::Routing(edit)) = self.rows.get(claim.index()) else {
            return Err(Failure::ClaimMismatch);
        };
        if !edit.complete || rows != edit.rows {
            return Err(Failure::ShapeMismatch);
        }
        Ok(edit)
    }
    pub(crate) fn routing_usage(
        &self,
        rows: u64,
        claim: &CaptureInterventionClaim<'_>,
        scope: &WorkingMemoryFundingScope,
        window: Option<eredu_runtime::intervention::InterventionPrefillWindow>,
    ) -> Result<CaptureUsage, Failure> {
        Ok(self.checked_routing(rows, claim, scope, window)?.usage)
    }
    pub(crate) fn take_routing_control(
        &self,
        rows: u64,
        claim: &CaptureInterventionClaim<'_>,
        scope: &WorkingMemoryFundingScope,
        window: Option<eredu_runtime::intervention::InterventionPrefillWindow>,
    ) -> Result<Option<GroupSelectionControl>, Failure> {
        let edit = self.checked_routing(rows, claim, scope, window)?;
        if edit.taken.replace(true) {
            return Err(Failure::ClaimMismatch);
        }
        edit.control
            .try_borrow_mut()
            .map_err(|_| Failure::CollectorBusy)
            .map(|mut value| value.take())
    }
    pub(crate) fn validate_routing_result(
        &self,
        rows: u64,
        original: Option<RoutingDecision<'_, Array>>,
        effective: RoutingDecision<'_, Array>,
        claim: &CaptureInterventionClaim<'_>,
        scope: &WorkingMemoryFundingScope,
        window: Option<eredu_runtime::intervention::InterventionPrefillWindow>,
    ) -> Result<(), Failure> {
        let edit = self.checked_routing(rows, claim, scope, window)?;
        if !edit.taken.get() {
            return Err(Failure::ClaimMismatch);
        }
        let shape = [
            i32::try_from(rows).map_err(|_| Failure::GeometryOverflow)?,
            edit.top_k as i32,
        ];
        for decision in original
            .as_ref()
            .into_iter()
            .chain(std::iter::once(&effective))
        {
            if decision.ids.shape() != shape
                || decision.coefficients.shape() != shape
                || decision.ids.dtype() != safemlx::Dtype::Uint32
                || decision.coefficients.dtype() != safemlx::Dtype::Float32
            {
                return Err(Failure::ShapeMismatch);
            }
        }
        Ok(())
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    [
        size_of::<Edit>(),
        size_of::<Option<PreparedRoutingControl<'_>>>(),
        size_of::<[CaptureUsage; 3]>(),
        size_of::<[evidence::State; 4]>(),
        size_of::<Option<GroupSelectionControl>>(),
        size_of::<(
            u64,
            ResolvedCaptureSlice,
            CaptureInvocationShape,
            Option<CaptureInvocationWindow>,
        )>(),
        size_of::<Result<Option<GroupSelectionControl>, Failure>>(),
        size_of::<CaptureNativePopulation>(),
        size_of::<[RoutingDecision<'_, Array>; 2]>(),
        size_of::<[RoutingDecision<'_, WorkspaceTensor>; 2]>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
