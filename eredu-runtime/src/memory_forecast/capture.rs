//! Logical capture envelopes from admission, without reserving or executing work.

use super::*;
use crate::execution_control::{admitted_intervention_storage_bytes, storage, TraceLimits};
use eredu_core::{
    capture::{AdmittedCapturePlan, CaptureLimits, CaptureUsage},
    intervention::AdmittedInterventionPlan,
};

/// Adds admitted instrumentation to a loaded request's existing costs.
///
/// Native transforms and intervention evidence share the capture ledger. A
/// backend with step-scoped native lifetimes can use the smaller per-step cap;
/// otherwise the cumulative retained cap bounds overlapping steps. Host costs
/// allow retaining the entire run's records and one compact JSON trace. These
/// are logical planning envelopes, excluding private native workspace, allocator
/// overhead, application copies, snapshots and branches. Those need separate
/// allowances. No capture budget is consumed by this calculation.
pub fn apply_capture_memory_bound(
    request: &mut GenerationMemoryRequest,
    capture: &AdmittedCapturePlan,
    intervention: Option<&AdmittedInterventionPlan>,
    trace: TraceLimits,
    native_step_scoped: bool,
) -> Result<(), CapabilityError> {
    let plan_storage = plan_storage(capture, intervention);
    let (native, host) = bounds(
        &capture.plan().limits,
        plan_storage,
        trace,
        native_step_scoped,
    )?;
    apply(request, &native, &host)
}

/// Reproducible capture envelope. Geometry bounds are capped by admission limits;
/// missing source/transform coverage falls back to the admitted ceilings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureMemoryPlan {
    /// Retained backend transform costs and absolute schedules.
    pub projection: Option<crate::capture::CaptureUsageProjection>,
    /// First future prediction, independent of the native cache position.
    pub first_prediction: u64,
    /// Already charged history; restoration never refunds these reservations.
    pub inherited_usage: CaptureUsage,
    limits: CaptureLimits,
    trace: TraceLimits,
    native_step_scoped: bool,
    plan_storage: Option<u64>,
    baseline: Vec<(MemoryDomain, MemoryBytes)>,
}

impl CaptureMemoryPlan {
    /// Captures existing non-instrumentation costs for idempotent recomputation.
    pub fn new(
        request: &GenerationMemoryRequest,
        capture: &AdmittedCapturePlan,
        intervention: Option<&AdmittedInterventionPlan>,
        trace: TraceLimits,
        native_step_scoped: bool,
        projection: Option<crate::capture::CaptureUsageProjection>,
    ) -> Self {
        Self {
            projection,
            first_prediction: 0,
            inherited_usage: CaptureUsage::default(),
            limits: capture.plan().limits.clone(),
            trace,
            native_step_scoped,
            plan_storage: plan_storage(capture, intervention),
            baseline: request
                .domains
                .iter()
                .map(|d| (d.domain.clone(), d.retained_input.clone()))
                .collect(),
        }
    }

    /// Applies a horizon without charging admission or accumulating prior forecasts.
    pub fn apply(
        &self,
        request: &mut GenerationMemoryRequest,
        predictions: u64,
    ) -> Result<(), CapabilityError> {
        if request.domains.len() != self.baseline.len()
            || request
                .domains
                .iter()
                .zip(&self.baseline)
                .any(|(d, (id, _))| &d.domain != id)
        {
            return Err(CapabilityError::Observation(
                "capture forecast pool identities changed".into(),
            ));
        }
        if self
            .inherited_usage
            .exceeded(self.limits.cumulative)
            .is_some()
        {
            return Err(CapabilityError::Observation(
                "capture history exceeds admitted limits".into(),
            ));
        }
        let end = self.first_prediction.checked_add(predictions).ok_or(
            CapabilityError::ArithmeticOverflow {
                operation: "capture prediction horizon",
            },
        )?;
        let outlook = self
            .projection
            .as_ref()
            .map(|p| p.outlook(self.first_prediction, end))
            .transpose()
            .map_err(|e| CapabilityError::Observation(e.to_string()))?
            .flatten()
            .filter(|p| p.complete);
        let mut limits = self.limits.clone();
        if let Some(outlook) = &outlook {
            // Historical reservations bound records callers may still retain. Native
            // step-scoped transforms have been released, so only future peaks remain.
            let cap = |past: u64, future: u64, limit: u64| past + future.min(limit - past);
            limits.cumulative.host_bytes = cap(
                self.inherited_usage.host_bytes,
                outlook.total.host_bytes,
                limits.cumulative.host_bytes,
            );
            limits.cumulative.encoded_bytes = cap(
                self.inherited_usage.encoded_bytes,
                outlook.total.encoded_bytes,
                limits.cumulative.encoded_bytes,
            );
            if self.native_step_scoped {
                let peak = outlook
                    .phases
                    .iter()
                    .map(|p| p.retained_bytes)
                    .max()
                    .unwrap_or(0);
                limits.per_step.retained_bytes = peak.min(limits.per_step.retained_bytes);
                limits.cumulative.retained_bytes -= self.inherited_usage.retained_bytes;
            } else {
                limits.cumulative.retained_bytes = cap(
                    self.inherited_usage.retained_bytes,
                    outlook.total.retained_bytes,
                    limits.cumulative.retained_bytes,
                );
            }
        }
        let (mut native, mut host) = bounds(
            &limits,
            self.plan_storage,
            self.trace,
            self.native_step_scoped,
        )?;
        let detail = if outlook.is_some() {
            "selection geometry and absolute schedules, including per-step diagnostics and charged history, capped by admission limits"
        } else {
            "admitted-limit fallback: complete capture source/transform geometry, intervention costing, or horizon coverage unavailable"
        };
        native.detail = format!("{detail}; {}", native.detail);
        host.detail = format!("{detail}; {}", host.detail);
        let mut candidate = request.clone();
        for (domain, (_, baseline)) in candidate.domains.iter_mut().zip(&self.baseline) {
            domain.retained_input = baseline.clone();
        }
        apply(&mut candidate, &native, &host)?;
        *request = candidate;
        Ok(())
    }
}

fn plan_storage(
    capture: &AdmittedCapturePlan,
    intervention: Option<&AdmittedInterventionPlan>,
) -> Option<u64> {
    storage::heap_bytes(capture.plan())
        .and_then(|bytes| bytes.checked_add(storage::heap_bytes(capture.points())?))
        .and_then(|bytes| bytes.checked_add(capture.identity().len() as u64))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of_val(capture) as u64))
        .and_then(|bytes| {
            bytes.checked_add(match intervention {
                Some(plan) => admitted_intervention_storage_bytes(plan)?,
                None => 0,
            })
        })
}

fn bounds(
    limits: &CaptureLimits,
    plan_storage: Option<u64>,
    trace: TraceLimits,
    native_step_scoped: bool,
) -> Result<(MemoryBytes, MemoryBytes), CapabilityError> {
    let retained = if native_step_scoped {
        limits
            .per_step
            .retained_bytes
            .min(limits.cumulative.retained_bytes)
    } else {
        limits.cumulative.retained_bytes
    };
    let native = MemoryBytes::estimated(
        0,
        retained,
        if native_step_scoped {
            "admitted capture/intervention logical native storage: min(per-step, cumulative) retained limit; transforms complete before the next prediction"
        } else {
            "admitted capture/intervention logical native storage: cumulative retained limit; cross-step overlap allowed"
        },
    );
    // The trace contains the encoded capture records, so use an envelope rather
    // than charging two encodings. Keep the capture limit if it is larger: a
    // captured record can exist before trace delivery rejects it.
    let encoded = limits.cumulative.encoded_bytes.max(trace.total_bytes);
    let host = limits.cumulative.host_bytes.checked_add(encoded).ok_or(
        CapabilityError::ArithmeticOverflow {
            operation: "capture host and encoded envelope",
        },
    )?;
    let host = MemoryBytes::estimated(
        0,
        host,
        "cumulative capture/intervention host limit plus max(cumulative encoded limit, total trace limit); one retained record history and one compact JSON trace, excluding application copies and allocator overhead",
    );
    let plans = match plan_storage {
        Some(bytes) => MemoryBytes::estimated(
            0,
            bytes,
            "logical admitted capture/intervention plan storage, including intervention payloads",
        ),
        None => MemoryBytes::unknown(
            "admitted capture/intervention plan storage unavailable or overflowing",
        ),
    };
    Ok((native, sum(&host, &plans)?))
}

fn sum(a: &MemoryBytes, b: &MemoryBytes) -> Result<MemoryBytes, CapabilityError> {
    let mut result = a.add(b)?;
    result.detail = format!("{}; {}", a.detail, b.detail);
    Ok(result)
}

fn apply(
    request: &mut GenerationMemoryRequest,
    native: &MemoryBytes,
    host: &MemoryBytes,
) -> Result<(), CapabilityError> {
    // Compute first so overflow cannot leave a partially updated request.
    let retained = request
        .domains
        .iter()
        .map(|domain| {
            let mut cost = domain.retained_input.clone();
            if matches!(domain.domain, MemoryDomain::Host | MemoryDomain::Unified) {
                cost = sum(&cost, host)?;
            }
            if !domain.executions.is_empty() {
                cost = sum(&cost, native)?;
            }
            Ok(cost)
        })
        .collect::<Result<Vec<_>, CapabilityError>>()?;
    for (domain, retained) in request.domains.iter_mut().zip(retained) {
        domain.retained_input = retained;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
