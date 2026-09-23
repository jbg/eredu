use super::*;
use crate::api::forecast::{
    add_retention, continuation_forecast, estimate_continuation_memory, ContinuationForecast,
    ContinuationForecastBackend, GenerationForecastError, GenerationForecastOptions,
};
use eredu_runtime::execution_control::{SnapshotTokenController, TextSnapshotBackend};

impl<B: ContinuationForecastBackend + TextSnapshotBackend> ControlledGenerationSession<'_, B> {
    /// Forecasts further ordinary predictions at a settled decode boundary. This
    /// includes sampler, semantic pipeline, capture/trace and live snapshot/branch
    /// retention. It never advances, copies state, consumes budgets or changes the
    /// run's token limit. The horizon may exceed that limit as a what-if forecast.
    /// Prepared/terminal/in-flight boundaries and speculative sessions are not
    /// ordinary decode continuations. Unsupported storage retains an unknown bound.
    pub fn forecast_remaining_generation(
        &mut self,
        additional_tokens: u64,
        options: &GenerationForecastOptions,
    ) -> Result<ContinuationForecast, GenerationForecastError> {
        if !matches!(self.status(), GenerationStatus::Paused) {
            return Err(GenerationForecastError::UnsupportedContinuation(
                "controlled forecasts require a paused, advanced ordinary session".into(),
            ));
        }
        let predictions = self
            .next_prediction()
            .checked_add(additional_tokens)
            .ok_or(eredu_core::CapabilityError::ArithmeticOverflow {
                operation: "continuation prediction horizon",
            })?;
        let host = self
            .semantic_snapshot_bytes()
            .and_then(|n| n.checked_add(self.state.controller().snapshot_storage_bytes()?))
            .and_then(|n| n.checked_add(self.pipeline.continuation_storage_bytes(predictions)?))
            .and_then(|n| {
                n.checked_add(
                    self.state
                        .controller()
                        .inner()
                        .continuation_storage_bytes(predictions)?,
                )
            })
            // Semantic event vectors can contain many empty records; compact JSON
            // limits bound their count, so allow one event header per encoded byte.
            .and_then(|n| {
                n.checked_add(
                    self.delivery
                        .budget
                        .limits()
                        .total_bytes
                        .checked_mul(std::mem::size_of::<SemanticEvent>() as u64 + 1)?,
                )
            });
        let retained_snapshots = self
            .snapshot_usage()
            .map_or(0, |usage| usage.retained_bytes);
        let boundary = self
            .state
            .boundary(&mut self.driver)
            .map_err(|e| GenerationForecastError::UnsupportedContinuation(e.to_string()))?;
        let (runtime, state, pending) = boundary.parts();
        let mut result = continuation_forecast::<B>(
            runtime,
            state,
            pending,
            additional_tokens,
            options,
            Some(self.delivery.budget.limits()),
        )?;
        add_retention(
            &mut result.request,
            host,
            "current and future facade decoder/constraint/semantic storage envelope",
            true,
        )?;
        add_retention(&mut result.request, Some(retained_snapshots),
            "live snapshot/branch reserved retention; conservative per-pool upper allowance, never resident credit", false)?;
        let assumptions = result.estimate.assumptions;
        result.estimate = estimate_continuation_memory(&result.request, &result.continuation)?;
        for assumption in assumptions {
            if !result.estimate.assumptions.contains(&assumption) {
                result.estimate.assumptions.push(assumption);
            }
        }
        Ok(result)
    }
}
