//! MLX diagnostics for loaded facade models.

use std::time::Duration;

use super::ExpertCacheBenchmarkError;
use eredu_core::BackendFailure;

impl super::LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>> {
    /// Returns portable sparse expert-cache telemetry when available.
    pub fn expert_cache_telemetry(
        &self,
    ) -> Result<Option<crate::ExpertCacheTelemetry>, BackendFailure> {
        self.runtime
            .session()
            .parameter_bank_report()
            .map(|report| {
                report
                    .as_ref()
                    .map(eredu_backend_mlx::parameter_bank_telemetry)
            })
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("expert-cache telemetry")
            })
    }

    /// Returns portable weight-residency telemetry when available.
    pub fn residency_telemetry(&self) -> Result<Option<crate::ResidencyTelemetry>, BackendFailure> {
        self.runtime
            .session()
            .residency_report()
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("residency telemetry")
            })
            .map(|report| report.as_ref().map(eredu_runtime::residency_telemetry))
    }
}

/// One measured phase of a selected-backend expert-cache benchmark.
#[derive(Debug, Clone, Copy)]
pub struct LocalExpertCacheBenchmarkSample {
    /// End-to-end phase latency after exact completion.
    pub elapsed: Duration,
    /// Route rows requested by the router.
    pub requested_routes: u64,
    /// Distinct logical experts requested after coalescing.
    pub distinct_experts: u64,
    /// Duplicate requests eliminated before materialization.
    pub coalesced_duplicates: u64,
    /// Temporary compact banks built.
    pub compact_banks: u64,
    /// Temporary compact-bank bytes built.
    pub compact_bank_bytes: u64,
    /// Host-cache hits.
    pub host_hits: u64,
    /// Host-cache misses.
    pub host_misses: u64,
    /// Host-cache evictions.
    pub host_evictions: u64,
    /// Device-cache hits.
    pub device_hits: u64,
    /// Device-cache misses.
    pub device_misses: u64,
    /// Device-cache evictions.
    pub device_evictions: u64,
    /// Host-resident expert count after the phase.
    pub host_resident_experts: usize,
    /// Host-resident expert bytes after the phase.
    pub host_resident_bytes: u64,
    /// Device-resident expert count after the phase.
    pub device_resident_experts: usize,
    /// Device-resident expert bytes after the phase.
    pub device_resident_bytes: u64,
}

/// Cold prefill, repeated prefill, and cached decode measurements.
#[derive(Debug, Clone, Copy)]
pub struct LocalExpertCacheBenchmark {
    /// Prefill after resetting the model session.
    pub cold_prefill: LocalExpertCacheBenchmarkSample,
    /// A second prefill after resetting only model state.
    pub repeated_prefill: LocalExpertCacheBenchmarkSample,
    /// One decode using the state produced by repeated prefill.
    pub cached_decode: LocalExpertCacheBenchmarkSample,
}

#[derive(Clone, Copy)]
struct ExpertSnapshot {
    prefill: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    decode: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    host_resident_experts: usize,
    host_resident_bytes: u64,
    device_resident_experts: usize,
    device_resident_bytes: u64,
}

fn expert_snapshot(
    model: &super::LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
) -> Result<ExpertSnapshot, ExpertCacheBenchmarkError> {
    let report = model
        .runtime
        .session()
        .parameter_bank_report()
        .map_err(|error| {
            BackendFailure::from_error(error).with_operation("expert-cache telemetry")
        })?
        .ok_or(ExpertCacheBenchmarkError::ExpertCacheUnavailable)?;
    Ok(ExpertSnapshot {
        prefill: *report.bulk(),
        decode: *report.incremental(),
        host_resident_experts: report.host_resident_entries(),
        host_resident_bytes: report.host_resident_bytes(),
        device_resident_experts: report.device_resident_entries(),
        device_resident_bytes: report.device_resident_bytes(),
    })
}

fn benchmark_sample(
    elapsed: Duration,
    before: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    after: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    occupancy: ExpertSnapshot,
) -> LocalExpertCacheBenchmarkSample {
    LocalExpertCacheBenchmarkSample {
        elapsed,
        requested_routes: after
            .requested_selections()
            .saturating_sub(before.requested_selections()),
        distinct_experts: after
            .distinct_entries()
            .saturating_sub(before.distinct_entries()),
        coalesced_duplicates: after
            .coalesced_duplicates()
            .saturating_sub(before.coalesced_duplicates()),
        compact_banks: after.compact_banks().saturating_sub(before.compact_banks()),
        compact_bank_bytes: after
            .compact_bank_bytes()
            .saturating_sub(before.compact_bank_bytes()),
        host_hits: after.host().hits().saturating_sub(before.host().hits()),
        host_misses: after.host().misses().saturating_sub(before.host().misses()),
        host_evictions: after
            .host()
            .evictions()
            .saturating_sub(before.host().evictions()),
        device_hits: after.device().hits().saturating_sub(before.device().hits()),
        device_misses: after
            .device()
            .misses()
            .saturating_sub(before.device().misses()),
        device_evictions: after
            .device()
            .evictions()
            .saturating_sub(before.device().evictions()),
        host_resident_experts: occupancy.host_resident_experts,
        host_resident_bytes: occupancy.host_resident_bytes,
        device_resident_experts: occupancy.device_resident_experts,
        device_resident_bytes: occupancy.device_resident_bytes,
    }
}

fn validate_expert_cache_benchmark_prompt(
    token_ids: &[u32],
) -> Result<(), ExpertCacheBenchmarkError> {
    if token_ids.is_empty() {
        return Err(ExpertCacheBenchmarkError::EmptyPrompt);
    }
    Ok(())
}

/// Benchmarks selected-backend expert-cache reuse without exposing tensors or streams.
pub fn benchmark_local_expert_cache(
    model: &mut super::LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    token_ids: &[u32],
) -> Result<LocalExpertCacheBenchmark, ExpertCacheBenchmarkError> {
    validate_expert_cache_benchmark_prompt(token_ids)?;
    let prompt = <eredu_backend_mlx::backend::MlxBackend as eredu_core::TextGenerationBackend>::prepare_text_prompt(
        model.runtime.backend(),
        token_ids.to_vec(),
    )
    .map_err(|error| BackendFailure::from_error(error).with_operation("expert-cache prompt preparation"))?;

    let before_cold = expert_snapshot(model)?;
    model.runtime.session_mut().reset().map_err(|error| {
        BackendFailure::from_error(error).with_operation("expert-cache session reset")
    })?;
    let started = std::time::Instant::now();
    let logits = model
        .runtime
        .prefill(prompt.clone())
        .map_err(|error| {
            BackendFailure::from_error(error).with_operation("expert-cache prefill submission")
        })?
        .wait()
        .map_err(|error| {
            BackendFailure::from_error(error).with_operation("expert-cache prefill completion")
        })?
        .into_logits()
        .ok_or(ExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_cold = expert_snapshot(model)?;
    let cold_prefill = benchmark_sample(
        started.elapsed(),
        before_cold.prefill,
        after_cold.prefill,
        after_cold,
    );

    model.runtime.session_mut().reset().map_err(|error| {
        BackendFailure::from_error(error).with_operation("expert-cache session reset")
    })?;
    let started = std::time::Instant::now();
    let logits = model
        .runtime
        .prefill(prompt)
        .map_err(|error| {
            BackendFailure::from_error(error).with_operation("expert-cache prefill submission")
        })?
        .wait()
        .map_err(|error| {
            BackendFailure::from_error(error).with_operation("expert-cache prefill completion")
        })?
        .into_logits()
        .ok_or(ExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_repeated = expert_snapshot(model)?;
    let repeated_prefill = benchmark_sample(
        started.elapsed(),
        after_cold.prefill,
        after_repeated.prefill,
        after_repeated,
    );

    let started = std::time::Instant::now();
    let output = {
        let (backend, session) = model.runtime.parts_mut();
        session
            .submit_token_decode(backend, token_ids[token_ids.len() - 1])
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("expert-cache decode submission")
            })?
    }
    .wait()
    .map_err(|error| {
        BackendFailure::from_error(error).with_operation("expert-cache decode completion")
    })?;
    let logits = output
        .into_logits()
        .ok_or(ExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_decode = expert_snapshot(model)?;
    let cached_decode = benchmark_sample(
        started.elapsed(),
        after_repeated.decode,
        after_decode.decode,
        after_decode,
    );

    Ok(LocalExpertCacheBenchmark {
        cold_prefill,
        repeated_prefill,
        cached_decode,
    })
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::{validate_expert_cache_benchmark_prompt, ExpertCacheBenchmarkError};

    #[test]
    fn empty_benchmark_prompt_is_a_facade_input_error() {
        assert!(matches!(
            validate_expert_cache_benchmark_prompt(&[]),
            Err(ExpertCacheBenchmarkError::EmptyPrompt)
        ));
        validate_expert_cache_benchmark_prompt(&[1]).unwrap();
    }
}
