//! Source-bound ordinary prediction rows; the caller keeps the cumulative ledger.
use super::PreparedModelInterventions;
mod owner;
mod partition;
mod prefill;
mod remote;
use crate::backend::array_copy::{CaptureNativePopulation, CaptureTensorNativeError as Failure};
use eredu_core::capture::{CaptureLedger, CapturePhase, CaptureSkipReason};
use eredu_nn::workspace::{
    HostMetadataFunding, WorkspaceContext, WorkspaceMetadataError, WorkspaceTensor,
};
use eredu_runtime::{capture::CaptureProtocolError, working_memory::OriginalInterventionSource};
pub(crate) use owner::PreparedTextInterventionsOwner;
use std::mem::{size_of, size_of_val};

type Result<T> = std::result::Result<T, eredu_nn::Error>;

/// Exact scheduled declarations and native shapes, retained after the cold
/// driver seals all predictions. This owner neither issues claims nor resets
/// the caller's cumulative capture/intervention ledger.
pub(crate) struct PreparedTextInterventions {
    rows: Vec<PreparedModelInterventions>,
    prefill_rows: Vec<PreparedModelInterventions>,
    prefill_selected: Vec<bool>,
    prefill_active: Option<usize>,
    prefill_next: u64,
    prefill_evidence: Vec<Option<[eredu_runtime::capture::CapturePrefillRowProgress; 4]>>,
    partition: Option<partition::Binding>,
    first_prediction: u64,
    end_prediction: u64,
    next_prediction: u64,
    active: Option<usize>,
    source: OriginalInterventionSource,
    // Every row and source alias retires before this preparation account.
    funding: HostMetadataFunding,
}
impl PreparedTextInterventions {
    /// Select a contiguous subset of the exact admitted ordinary request. The
    /// source owns operation schedules; no copied per-row selection policy or
    /// independent ledger is constructed. Evidence begins without inherited
    /// skips and follows the existing cumulative ledger at each real hook.
    pub(crate) fn prepare(
        source: &OriginalInterventionSource,
        selected: &[bool],
        first_prediction: u64,
        predictions: u64,
        context: &WorkspaceContext,
    ) -> Result<Self> {
        context.charge_metadata(
            Self::control_bytes()
                .and_then(|bytes| bytes.checked_add(owner::control_bytes()?))
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let funding = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let plan = source.plan().admission();
        let end_prediction = first_prediction
            .checked_add(predictions)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        if plan.invocation_bounds().is_some()
            || end_prediction > plan.request().max_predictions
            || selected.len() != plan.plan().operations.len()
        {
            return Err(context.metadata_source(CaptureProtocolError::Invocation));
        }
        let count = usize::try_from(predictions).map_err(|cause| context.metadata_source(cause))?;
        let mut rows = context.metadata_vec(count)?;
        let mut evidence_skips = context.metadata_vec(selected.len())?;
        evidence_skips.resize(selected.len(), [None, None]);
        let mut prefill_selected = context.metadata_vec(selected.len())?;
        let mut terminal_selected = context.metadata_vec(selected.len())?;
        for (index, selected) in selected.iter().copied().enumerate() {
            let row = first_prediction == 0
                && selected
                && plan.plan().operations[index]
                    .schedule
                    .includes(CapturePhase::Prefill, 0)
                && eredu_runtime::intervention::InterventionPrefillWindow::row_axis(
                    &plan.points()[index],
                );
            if row {
                // This constructor retains descriptors only. Execution also
                // requires the original paid companion frame and native scope.
                let valid = if plan.plan().operations[index].evidence
                    != eredu_core::intervention::InterventionEvidence::None
                {
                    eredu_runtime::intervention::InterventionPrefillWindow::validate_evidence_operation(source,index)
                } else {
                    eredu_runtime::intervention::InterventionPrefillWindow::validate_operation(
                        plan, index,
                    )
                };
                valid.map_err(|cause| context.metadata_source(cause))?;
            }
            prefill_selected.push(row);
            terminal_selected.push(selected && !row);
        }
        for prediction in first_prediction..end_prediction {
            rows.push(PreparedModelInterventions::prepare_scheduled_with_evidence(
                source,
                if prediction == 0 {
                    &terminal_selected
                } else {
                    selected
                },
                phase(prediction),
                prediction,
                Some(&evidence_skips),
                context,
            )?);
        }
        Ok(Self {
            rows,
            prefill_rows: context.metadata_vec(0)?,
            prefill_selected,
            prefill_active: None,
            prefill_next: 0,
            prefill_evidence: context.metadata_vec(0)?,
            partition: None,
            first_prediction,
            end_prediction,
            next_prediction: first_prediction,
            active: None,
            source: source.clone(),
            funding,
        })
    }
    pub(crate) fn into_owner(
        self,
    ) -> std::result::Result<
        PreparedTextInterventionsOwner,
        eredu_runtime::working_memory::WorkingMemoryError,
    > {
        PreparedTextInterventionsOwner::new(self)
    }
    pub(crate) fn source(&self) -> &OriginalInterventionSource {
        &self.source
    }
    pub(crate) fn same_source(&self, source: &OriginalInterventionSource) -> bool {
        self.source.same_source(source)
    }
    pub(crate) fn validate_range(
        &self,
        source: &OriginalInterventionSource,
        first: u64,
        predictions: u64,
    ) -> std::result::Result<(), Failure> {
        if !self.same_source(source)
            || self.active.is_some()
            || self.prefill_active.is_some()
            || self.next_prediction != self.end_prediction
            || self.first_prediction != first
            || first.checked_add(predictions) != Some(self.end_prediction)
        {
            return Err(Failure::ClaimMismatch);
        }
        Ok(())
    }

    fn check_context(&self, context: &WorkspaceContext) -> Result<()> {
        context.charge_metadata(Self::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        if context
            .metadata_funding()
            .is_none_or(|funding| !funding.same_account(&self.funding))
        {
            return Err(context.metadata_source(CaptureProtocolError::Invocation));
        }
        Ok(())
    }
    pub(crate) fn begin(
        &mut self,
        actual_phase: CapturePhase,
        prediction: u64,
        ledger: &mut CaptureLedger,
        context: &WorkspaceContext,
    ) -> Result<()> {
        self.check_context(context)?;
        if self.active.is_some()
            || prediction != self.next_prediction
            || prediction >= self.end_prediction
            || actual_phase != phase(prediction)
        {
            return Err(context.metadata_source(CaptureProtocolError::Transaction));
        }
        let index = usize::try_from(prediction - self.first_prediction)
            .map_err(|cause| context.metadata_source(cause))?;
        // A failed declaration reservation remains this attempted row. No retry
        // can reset the source-owned cumulative spending or its row transaction.
        self.active = Some(index);
        self.rows[index].begin(actual_phase, prediction, ledger, context)
    }
    pub(crate) fn trace(
        &mut self,
        path: &str,
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
        ledger: &mut CaptureLedger,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<(Option<WorkspaceTensor>, CaptureNativePopulation)> {
        self.check_context(context)?;
        let index = self
            .active
            .ok_or_else(|| context.metadata_source(CaptureProtocolError::Transaction))?;
        let mut population = CaptureNativePopulation::default();
        let mut output = None;
        if let Some(fragment) = self.prefill_active {
            let (value, current) = self.prefill_rows[fragment].trace_with_evidence_progress(
                path,
                value,
                context,
                ledger,
                retained,
                Some(&mut self.prefill_evidence),
            )?;
            output = value;
            population = current;
        }
        let (last, current) = self.rows[index].trace(
            path,
            output.as_ref().unwrap_or(value),
            context,
            ledger,
            retained,
        )?;
        if last.is_some() {
            output = last;
        }
        Ok((
            output,
            population
                .checked_add(current)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        ))
    }
    fn routed_row(&self) -> Option<&PreparedModelInterventions> {
        self.prefill_active
            .map(|index| &self.prefill_rows[index])
            .or_else(|| self.active.map(|index| &self.rows[index]))
    }
    fn routed_row_mut(&mut self) -> Option<&mut PreparedModelInterventions> {
        match self.prefill_active {
            Some(index) => Some(&mut self.prefill_rows[index]),
            None => self.active.map(|index| &mut self.rows[index]),
        }
    }
    pub(crate) fn routed_selection(&self, path: &str) -> Option<usize> {
        self.routed_row()?.routed_selection(path)
    }
    pub(crate) fn begin_routed(
        &mut self,
        first: usize,
        input: &eredu_runtime::RoutedUnitInvocation<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
        ledger: &mut CaptureLedger,
    ) -> Result<()> {
        self.check_context(context)?;
        self.routed_row_mut()
            .ok_or_else(|| context.metadata_source(CaptureProtocolError::Transaction))?
            .begin_routed(first, input, context, ledger)
    }
    pub(crate) fn trace_routed(
        &mut self,
        first: usize,
        batch: &eredu_runtime::RoutedUnitBatch<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
        roots: &mut Vec<WorkspaceTensor>,
    ) -> Result<(Option<WorkspaceTensor>, CaptureNativePopulation)> {
        self.check_context(context)?;
        self.routed_row_mut()
            .ok_or_else(|| context.metadata_source(CaptureProtocolError::Transaction))?
            .trace_routed(first, batch, context, roots)
    }
    pub(crate) fn finish_routed(
        &mut self,
        first: usize,
        success: bool,
        context: &WorkspaceContext,
    ) -> Result<()> {
        self.check_context(context)?;
        self.routed_row_mut()
            .ok_or_else(|| context.metadata_source(CaptureProtocolError::Transaction))?
            .finish_routed(first, success, context)
    }
    pub(crate) fn prefill_routed_usage(
        &self,
        index: usize,
    ) -> std::result::Result<eredu_core::capture::CaptureUsage, Failure> {
        if self.active.is_some()
            || self.prefill_active.is_some()
            || self.next_prediction != self.end_prediction
            || self.first_prediction != 0
            || self.prefill_rows.is_empty()
        {
            return Err(Failure::ClaimMismatch);
        }
        self.prefill_rows.iter().try_fold(
            eredu_core::capture::CaptureUsage::default(),
            |sum, row| {
                sum.checked_add(row.retained_routed_usage(index)?)
                    .map_err(Failure::ActivationPolicy)
            },
        )
    }
    pub(crate) fn trace_routing_control(
        &mut self,
        path: &str,
        rows: u64,
        context: &WorkspaceContext,
        ledger: &mut CaptureLedger,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>> {
        self.check_context(context)?;
        let index = self
            .active
            .ok_or_else(|| context.metadata_source(CaptureProtocolError::Transaction))?;
        let row = match self.prefill_active {
            Some(fragment) => &mut self.prefill_rows[fragment],
            None => &mut self.rows[index],
        };
        row.trace_routing_control(path, rows, context, ledger)
    }
    pub(crate) fn routing_unmodified_interest(
        &self,
        path: &str,
    ) -> eredu_runtime::RoutingUnmodifiedInterest {
        let Some(index) = self.active else {
            return eredu_runtime::RoutingUnmodifiedInterest::None;
        };
        let row = match self.prefill_active {
            Some(fragment) => &self.prefill_rows[fragment],
            None => &self.rows[index],
        };
        row.routing_unmodified_interest(path)
    }
    pub(crate) fn trace_routing_result(
        &mut self,
        path: &str,
        original: Option<eredu_runtime::RoutingDecision<'_, WorkspaceTensor>>,
        effective: eredu_runtime::RoutingDecision<'_, WorkspaceTensor>,
        unmodified: bool,
        context: &WorkspaceContext,
        ledger: &mut CaptureLedger,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<CaptureNativePopulation> {
        self.check_context(context)?;
        let index = self
            .active
            .ok_or_else(|| context.metadata_source(CaptureProtocolError::Transaction))?;
        match self.prefill_active {
            Some(fragment) => self.prefill_rows[fragment].trace_routing_result(
                path,
                original,
                effective,
                unmodified,
                context,
                ledger,
                retained,
                Some(&mut self.prefill_evidence),
            ),
            None => self.rows[index].trace_routing_result(
                path, original, effective, unmodified, context, ledger, retained, None,
            ),
        }
    }
    pub(crate) fn finish(&mut self, context: &WorkspaceContext) -> Result<()> {
        self.check_context(context)?;
        let index = self
            .active
            .ok_or_else(|| context.metadata_source(CaptureProtocolError::Transaction))?;
        if self.prefill_active.is_some()
            || (self.first_prediction + index as u64 == 0
                && self.prefill_selected.iter().any(|v| *v)
                && self.prefill_next != self.source.plan().admission().request().prompt_tokens)
        {
            return Err(context.metadata_source(CaptureProtocolError::Transaction));
        }
        self.rows[index].finish(context)?;
        self.next_prediction = self
            .next_prediction
            .checked_add(1)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.active = None;
        Ok(())
    }
    /// The ordinary driver decides physical readout using the active row's
    /// existing predicate. In particular a logits sequence slice is not silently
    /// replaced by a terminal-token layout; the shared trace validates its axes.
    pub(crate) fn requires_sequence_readout(&self) -> bool {
        let index = self.active.or_else(|| {
            (self.next_prediction < self.end_prediction)
                .then(|| usize::try_from(self.next_prediction - self.first_prediction).ok())
                .flatten()
        });
        let Some(index) = index else {
            return false;
        };
        self.source
            .plan()
            .admission()
            .points()
            .iter()
            .enumerate()
            .any(|(operation, point)| {
                point.stage == eredu_core::intervention::InterventionStage::LogitsBeforeSampling
                    && eredu_runtime::intervention::InterventionPrefillWindow::row_axis(point)
                    && (self
                        .rows
                        .get(index)
                        .is_some_and(|row| row.selected(operation))
                        || self.first_prediction + index as u64 == 0
                            && self.prefill_selected[operation])
            })
    }
    /// Native consumers still authenticate their exact source, coordinate and
    /// scheduled funding scope through the unchanged model-row methods.
    pub(crate) fn row(
        &self,
        actual_phase: CapturePhase,
        prediction: u64,
    ) -> std::result::Result<&PreparedModelInterventions, Failure> {
        if self.active.is_some()
            || self.next_prediction != self.end_prediction
            || actual_phase != phase(prediction)
            || prediction < self.first_prediction
            || prediction >= self.end_prediction
        {
            return Err(Failure::ClaimMismatch);
        }
        let index = usize::try_from(prediction - self.first_prediction)
            .map_err(|_| Failure::GeometryOverflow)?;
        self.rows.get(index).ok_or(Failure::ClaimMismatch)
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self>>(),
            size_of::<Result<()>>(),
            size_of::<Option<usize>>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Vec<PreparedModelInterventions>>(),
            prefill::control_bytes()?,
            partition::control_bytes()?,
            size_of::<Vec<[Option<CaptureSkipReason>; 2]>>(),
            size_of::<(
                &OriginalInterventionSource,
                &[bool],
                u64,
                u64,
                &WorkspaceContext,
            )>(),
            size_of::<(
                &mut Self,
                CapturePhase,
                u64,
                &mut CaptureLedger,
                &WorkspaceContext,
            )>(),
            size_of::<(
                &mut Self,
                &str,
                &WorkspaceTensor,
                &WorkspaceContext,
                &mut CaptureLedger,
                &mut Vec<WorkspaceTensor>,
            )>(),
            size_of::<Result<(Option<WorkspaceTensor>, CaptureNativePopulation)>>(),
            size_of::<std::result::Result<&PreparedModelInterventions, Failure>>(),
            OriginalInterventionSource::validation_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
fn phase(prediction: u64) -> CapturePhase {
    if prediction == 0 {
        CapturePhase::Prefill
    } else {
        CapturePhase::Decode
    }
}

/// A borrowed source/row destination for the existing cold equation worker.
/// Rejected candidates release their previous row payload before tracing again.
#[derive(Clone, Copy)]
pub(in crate::composition::mlx) struct TextInterventionQuote<'a> {
    pub source: &'a OriginalInterventionSource,
    pub rows: &'a std::cell::RefCell<Option<PreparedTextInterventions>>,
}
impl TextInterventionQuote<'_> {
    pub(in crate::composition::mlx) fn prepare(&self, context: &WorkspaceContext) -> Result<()> {
        self.prepare_range(
            0,
            self.source.plan().admission().request().max_predictions,
            context,
        )
    }
    pub(in crate::composition::mlx) fn prepare_range(
        &self,
        first: u64,
        predictions: u64,
        context: &WorkspaceContext,
    ) -> Result<()> {
        context.charge_metadata(size_of::<(
            Self,
            Vec<bool>,
            std::cell::RefMut<'_, Option<PreparedTextInterventions>>,
            Result<PreparedTextInterventions>,
            u64,
            u64,
        )>())?;
        self.rows
            .try_borrow_mut()
            .map_err(|cause| context.metadata_source(cause))?
            .take();
        let count = self.source.plan().admission().points().len();
        let mut selected = context.metadata_vec(count)?;
        selected.resize(count, true);
        let prepared = PreparedTextInterventions::prepare(
            self.source,
            &selected,
            first,
            predictions,
            context,
        )?;
        *self
            .rows
            .try_borrow_mut()
            .map_err(|cause| context.metadata_source(cause))? = Some(prepared);
        Ok(())
    }
}
impl std::fmt::Debug for PreparedTextInterventions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedTextInterventions")
            .field("first_prediction", &self.first_prediction)
            .field("end_prediction", &self.end_prediction)
            .field("next_prediction", &self.next_prediction)
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
