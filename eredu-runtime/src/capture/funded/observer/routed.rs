//! Actual sparse provider batches share one original logical prefill row.
use super::fragments::progress_error;
use super::*;
mod interventions;
mod invocation;
mod partition;
use crate::capture::{CapturePrefillHookDecision, CapturePrefillObservationPolicy};
fn same_routing(source: &AdmittedCapturePlan, first: usize, index: usize) -> bool {
    crate::capture::partition::PartitionCaptureRoutedHooks::same_routing(source, first, index)
}
impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    fn routed_batch(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, T>,
        effective: bool,
    ) -> Result<(), FundedCaptureError<E>> {
        if self.routed_partition_hooks.is_some() {
            return self.partition_routed_batch(batch, effective);
        }
        if !self.routed_active || batch.origins.is_some() || batch.unit_coordinates.is_some() {
            return Err(CaptureProtocolError::Transaction.into());
        }
        let Some(first) = self.routed_selection else {
            return Ok(());
        };
        if self.prefill.is_none() {
            return self.routed_invocation_batch(batch, effective, first);
        }
        let chunk = self.current_fragment_chunk()?;
        let bound = self.bound.ok_or(CaptureProtocolError::Transaction)?;
        let policy = CapturePrefillObservationPolicy::from_bound(bound).map_err(progress_error)?;
        let source = batch.capture_source()?;
        for index in 0..self.session.plan.plan().selections.len() {
            if !same_routing(&self.session.plan, first, index)
                || (self.session.plan.points()[index].position
                    == eredu_core::ObservationPosition::AfterIntervention)
                    != effective
            {
                continue;
            }
            let row = policy.row(index).map_err(progress_error)?;
            let Some(plan) = row.routed_plan() else {
                continue;
            };
            let fragment = plan
                .fragment(chunk.input.start / bound.geometry().prefill_chunk_positions)
                .map_err(|cause| {
                    CaptureRunHostError::from(crate::working_memory::CapturePrefillHostError::from(
                        cause,
                    ))
                })?;
            let Frame::Active(frame) = &mut self.frame else {
                return Err(CaptureProtocolError::Transaction.into());
            };
            let decision = frame.begin_routed_prefill_batch(
                index,
                &chunk,
                &self.session.plan.plan().selections[index].path,
            )?;
            if decision == CapturePrefillHookDecision::Ignore {
                continue;
            }
            let started = std::time::Instant::now();
            let result = (|| {
                let actual = self
                    .backend
                    .validate_routed_prefill_source(&source, &fragment)?;
                if decision == CapturePrefillHookDecision::First {
                    let usage = self.backend.estimate_routed_prefill(plan.geometry())?;
                    if frame
                        .reserve_prefill_hook(index, &mut self.session.ledger, actual, usage)?
                        .is_some()
                    {
                        return Ok(());
                    }
                }
                let writer = frame.take_prefill_routed_fragment(index, &fragment)?;
                self.backend.transform_routed_prefill(&source, writer)
            })();
            self.session.capture_seconds += started.elapsed().as_secs_f64();
            if let Err(error) = result {
                frame.fail_prefill_hook(index);
                return Err(error);
            }
        }
        Ok(())
    }
    fn finish_routed_invocation(&mut self, success: bool) -> Result<(), FundedCaptureError<E>> {
        if !self.routed_active {
            return Err(CaptureProtocolError::Transaction.into());
        }
        if self.routed_partition_hooks.is_some() {
            return self.finish_partition_routed(success);
        }
        self.routed_active = false;
        self.finish_routed_interventions(success)?;
        let Some(first) = self.routed_selection else {
            return Ok(());
        };
        if self.prefill.is_none() {
            return self.finish_routed_ordinary_invocation(success, first);
        }
        let chunk = self.current_fragment_chunk()?;
        let bound = self.bound.ok_or(CaptureProtocolError::Transaction)?;
        let policy = CapturePrefillObservationPolicy::from_bound(bound).map_err(progress_error)?;
        let Frame::Active(frame) = &mut self.frame else {
            return Err(CaptureProtocolError::Transaction.into());
        };
        for index in 0..self.session.plan.plan().selections.len() {
            if !same_routing(&self.session.plan, first, index) {
                continue;
            }
            let row = policy.row(index).map_err(progress_error)?;
            let Some(plan) = row.routed_plan() else {
                continue;
            };
            if !success {
                frame.fail_prefill_hook(index);
                continue;
            }
            if matches!(
                frame.records()[index].outcome,
                CaptureOutcome::Skipped { .. }
            ) {
                continue;
            }
            let fragment = plan
                .fragment(chunk.input.start / bound.geometry().prefill_chunk_positions)
                .map_err(|cause| {
                    CaptureRunHostError::from(crate::working_memory::CapturePrefillHostError::from(
                        cause,
                    ))
                })?;
            frame.finish_routed_prefill_hook(index, &fragment)?;
        }
        Ok(())
    }
}
impl<T, E: std::error::Error + Send + Sync + 'static, N> crate::RoutedUnitObserver<T>
    for FundedCaptureObserver<'_, T, E, N>
{
    fn invocation_active(&self) -> bool {
        self.routed_active
    }
    fn begin_invocation(
        &mut self,
        invocation: &crate::RoutedUnitInvocation<'_, T>,
    ) -> Result<(), eredu_nn::Error> {
        let result = (|| {
            if self.routed_active
                || (self.routed_selection.is_none() && self.routed_intervention_selection.is_none())
            {
                return Err(CaptureProtocolError::Transaction.into());
            }
            if self.prefill.is_some() {
                self.current_fragment_chunk()?;
            } else if !matches!(self.frame, Frame::Active(_)) {
                return Err(CaptureProtocolError::Transaction.into());
            }
            if self.backend.partition_capture().is_some() {
                self.begin_partition_routed(invocation)?;
            } else if invocation.origins.is_some() || invocation.unit_coordinates.is_some() {
                return Err(CaptureProtocolError::Transaction.into());
            }
            self.routed_active = true;
            Ok(())
        })();
        result.map_err(|cause| self.backend.routed_error(cause))
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), eredu_nn::Error> {
        self.finish_routed_invocation(success)
            .map_err(|cause| self.backend.routed_error(cause))
    }
    fn observe(&mut self, batch: &crate::RoutedUnitBatch<'_, T>) -> Result<(), eredu_nn::Error> {
        self.routed_batch(batch, false)
            .map_err(|cause| self.backend.routed_error(cause))
    }
    fn intervene(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, T>,
    ) -> Result<Option<T>, eredu_nn::Error> {
        self.intervene_routed_batch(batch)
            .map_err(|cause| self.backend.routed_error(cause))
    }
    fn observe_effective(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, T>,
    ) -> Result<(), eredu_nn::Error> {
        self.routed_batch(batch, true)
            .map_err(|cause| self.backend.routed_error(cause))
    }
}
pub(in crate::capture::funded) fn control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<crate::RoutedUnitBatch<'_, ()>>(),
        size_of::<crate::RoutedUnitInvocation<'_, ()>>(),
        size_of::<RoutedUnitCaptureSource<'_, ()>>(),
        size_of::<crate::capture::CapturePrefillObservationRow<'_>>(),
        size_of::<crate::capture::CaptureObservationStep<'_>>(),
        size_of::<CaptureRoutedUnitsGeometry<'_>>(),
        size_of::<crate::working_memory::CaptureRoutedBatchWriter<'_, '_>>(),
        size_of::<(CaptureUsage, CaptureUsage, TensorDtype)>(),
        size_of::<crate::working_memory::RoutedInterventionCursor<'_>>(),
        size_of::<crate::working_memory::RoutedInterventionBatch<'_, '_>>(),
        size_of::<Option<usize>>(),
        size_of::<[u64; 2]>(),
        size_of::<Result<[u64; 2], FundedCaptureError<std::convert::Infallible>>>(),
        size_of::<CaptureRoutedPrefillFragment<'_, '_>>(),
        size_of::<Result<(), FundedCaptureError<std::convert::Infallible>>>(),
    ]
    .into_iter()
    .try_fold(
        CaptureRoutedUnitsGeometry::preparation_control_bytes()?,
        usize::checked_add,
    )?
    .checked_mul(3)
}
