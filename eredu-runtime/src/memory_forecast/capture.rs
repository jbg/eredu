//! Logical capture envelopes from admission, without reserving or executing work.

use super::*;
use crate::execution_control::{admitted_intervention_storage_bytes, storage, TraceLimits};
use eredu_core::{
    capture::{AdmittedCapturePlan, CaptureLimits},
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
    let plan_storage = storage::heap_bytes(capture.plan())
        .and_then(|bytes| bytes.checked_add(storage::heap_bytes(capture.points())?))
        .and_then(|bytes| bytes.checked_add(capture.identity().len() as u64))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of_val(capture) as u64))
        .and_then(|bytes| {
            bytes.checked_add(match intervention {
                Some(plan) => admitted_intervention_storage_bytes(plan)?,
                None => 0,
            })
        });
    let (native, host) = bounds(
        &capture.plan().limits,
        plan_storage,
        trace,
        native_step_scoped,
    )?;
    apply(request, &native, &host)
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
