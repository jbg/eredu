//! Original source composition of the existing online logical-window algebra.
use super::super::reductions::{
    WindowReductions,
    construction::{ConstructionError, Metadata},
};
use super::*;
use eredu_core::{
    capture::InterventionEvidenceSide,
    speculative::{PreparedSpeculativePrefillReductions, SpeculativePrefillReductionGeometry},
};

struct PrefixUpdate<'a> {
    ledger: CaptureLedger,
    guard: crate::working_memory::CaptureRunLedgerGuard<'a>,
}
impl Drop for PrefixUpdate<'_> {
    fn drop(&mut self) {
        self.guard.record(self.ledger.total());
    }
}

pub(super) struct Aggregate {
    pub(super) group: WindowReductions,
    // Unique empty shared destination, paid before the first physical equation.
    // Its account-only custody survives source/observer retirement after finish.
    delivery: PreparedSpeculativePrefillReductions,
}
impl Aggregate {
    fn prepare(
        owner: &OriginalSpeculativeCapture,
        active: Active,
        geometry: SpeculativePrefillReductionGeometry,
        ledger: &mut CaptureLedger,
    ) -> Result<Option<Self>, ConstructionError> {
        let metadata = Metadata::original(&owner.funding);
        let extra = match &owner.interventions {
            Some(edits) => WindowReductions::count_original_interventions(
                &edits.source,
                &edits.scopes,
                active.origin,
            )?,
            None => 0,
        };
        let Some(mut group) = WindowReductions::create_capture(
            owner.source.plan().admission(),
            ledger,
            &owner.scopes,
            geometry,
            active.origin,
            active.invocation,
            extra != 0,
            metadata,
        )?
        else {
            return Ok(None);
        };
        if let Some(edits) = &owner.interventions {
            group.create_original_interventions(
                &edits.source,
                ledger,
                &edits.scopes,
                active.origin,
                metadata,
            )?;
        }
        let controls = PreparedSpeculativePrefillReductions::retained_control_bytes::<
            HostMetadataFunding,
        >()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(ConstructionError::Overflow)?;
        metadata.controls(controls)?;
        let delivery = PreparedSpeculativePrefillReductions::retain(owner.funding.clone());
        Ok(Some(Self { group, delivery }))
    }
}
impl OriginalSpeculativeCapture {
    pub(super) fn aggregate_error(
        &self,
        cause: ConstructionError,
    ) -> OriginalSpeculativeCaptureError {
        self.reject(match cause {
            ConstructionError::Metadata(cause) => cause,
            cause => self.funding.metadata_source(cause),
        })
    }
    fn aggregate_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Aggregate>(),
            size_of::<Option<Aggregate>>(),
            size_of::<PrefixUpdate<'_>>(),
            size_of::<crate::working_memory::CaptureRunLedgerGuard<'_>>(),
            size_of::<Result<crate::working_memory::CaptureRunLedgerGuard<'_>, CaptureProtocolError>>(
            ),
            size_of::<Result<Option<Aggregate>, ConstructionError>>(),
            size_of::<PreparedSpeculativePrefillReductions>(),
            size_of::<Option<SpeculativeActivationCapture>>(),
            size_of::<SpeculativePrefillReductionGeometry>(),
            size_of::<crate::working_memory::OriginalEmbeddedCaptureLineage>(),
            size_of::<(
                &Self,
                Active,
                SpeculativePrefillReductionGeometry,
                &mut CaptureLedger,
            )>(),
            size_of::<Result<(), ConstructionError>>(),
            size_of::<Result<(), OriginalSpeculativeCaptureError>>(),
            size_of::<Option<CaptureSkipReason>>(),
            size_of::<[Option<CaptureSkipReason>; 2]>(),
            size_of::<InterventionEvidenceSide>(),
            OriginalSpeculativeCapturePrefix::control_bytes()?,
            WindowReductions::intervention_access_control_bytes()?,
            crate::working_memory::CaptureRunLedger::inspection_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(super) fn prepare_aggregate_invocation(
        &mut self,
        active: Active,
    ) -> Result<(), OriginalSpeculativeCaptureError> {
        if self.lineage.is_none() {
            if self.reduction_geometry.is_some() && active.span.is_some() {
                return Err(self.protocol(CaptureProtocolError::Invocation));
            }
            return Ok(());
        }
        self.funding
            .reserve_metadata(
                Self::aggregate_control_bytes()
                    .ok_or_else(|| self.reject(WorkspaceMetadataError::Overflow.into()))?,
            )
            .map_err(|cause| self.reject(WorkspaceMetadataError::from(cause).into()))?;
        let lineage = self.lineage.as_ref().expect("checked lineage").clone();
        let guard = lineage
            .ledger()
            .borrow()
            .map_err(|cause| self.protocol(cause))?;
        let ledger =
            CaptureLedger::with_inherited_usage(self.source.plan().admission(), guard.usage())
                .map_err(|cause| self.aggregate_error(cause.into()))?;
        let mut update = PrefixUpdate { ledger, guard };
        let result = self.prepare_aggregate_with_ledger(active, &mut update.ledger);
        let prefix = result.as_ref().ok().map(|_| {
            OriginalSpeculativeCapturePrefix::from_ledger(
                self.source.plan(),
                active,
                &update.ledger,
            )
        });
        // Every accepted prefix survives errors and unwinding. No source mutex
        // or cumulative active loan crosses the equation/observer callback.
        drop(update);
        result.map_err(|cause| self.aggregate_error(cause))?;
        self.prefix = prefix;
        Ok(())
    }
    fn prepare_aggregate_with_ledger(
        &mut self,
        active: Active,
        ledger: &mut CaptureLedger,
    ) -> Result<(), ConstructionError> {
        self.reserve_prefix(active, ledger)?;
        if let (Some(geometry), Some(span)) = (self.reduction_geometry, active.span) {
            if self.reductions.is_none() {
                self.reductions = Aggregate::prepare(self, active, geometry, ledger)?;
            }
            if let Some(aggregate) = &mut self.reductions {
                aggregate.group.begin_capture_window(
                    active.phase,
                    span,
                    active.origin,
                    geometry,
                )?;
                for entry in &aggregate.group.report.records {
                    if entry.phase == active.phase {
                        if let CaptureOutcome::Skipped { reason } = &entry.record.outcome {
                            self.selected[entry.selection_index] = false;
                            self.skipped[entry.selection_index] = Some(reason.clone());
                        }
                    }
                }
                if let Some(edits) = &mut self.interventions {
                    for (operation, skipped) in edits.evidence_skips.iter_mut().enumerate() {
                        for (side, destination) in [
                            InterventionEvidenceSide::Before,
                            InterventionEvidenceSide::After,
                        ]
                        .into_iter()
                        .zip(skipped)
                        {
                            *destination = aggregate
                                .group
                                .intervention_evidence_skip(active.phase, operation, side)
                                .cloned();
                        }
                    }
                }
            }
        }
        Ok(())
    }
    /// Actual target/prediction logical extents from the existing split driver.
    /// This scalar declaration performs no allocation or quota reservation.
    pub fn set_prefill_reduction_geometry(
        &mut self,
        geometry: SpeculativePrefillReductionGeometry,
    ) {
        self.reduction_geometry = Some(geometry);
    }
    /// Validate complete logical coverage before the shared fallible completion
    /// boundary closes. The already paid report remains provisional until finish.
    pub fn complete_prefill_reductions(&mut self) -> Result<(), OriginalSpeculativeCaptureError> {
        let result = match &mut self.reductions {
            Some(aggregate) => aggregate.group.seal_fixed(),
            None => Ok(()),
        };
        result.map_err(|cause| self.aggregate_error(cause))
    }
    /// Publish onto the last actual physical envelope after the shared driver
    /// settles prefill. This performs no allocation, native work, or refund and
    /// is safe on the existing failure/unwind path.
    pub fn finish_prefill_reductions(&mut self, success: bool) {
        self.checkpoint_ready &= success;
        let report = self
            .reductions
            .take()
            .map(|aggregate| aggregate.delivery.finish(aggregate.group.finish(success)));
        if let Some(mut last) = self.held_prefill.take() {
            last.prefill_reductions = report.map(Into::into);
            debug_assert!(self.records.len() < self.records.capacity());
            self.records.push(last);
        }
        self.reduction_geometry = None;
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
