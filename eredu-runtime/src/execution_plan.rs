//! Portable execution-plan policy derivation before backend resource realization.

use eredu_core::{
    residency::OffloadConfig, ExecutionPlan, ResidencyPlan, WeightTransformationPlan,
};

use crate::{
    DenseDiskStreamLoadOptions, LayerwiseLoadOptions, NormalizedLoadRequest,
    NormalizedLoadRequestError, OrdinaryWeightResidency, ParallelLoadRequest,
    ParameterBankLoadOptions, WeightResidency,
};

/// Optional observations requested from the backend's residency mechanisms.
///
/// Deriving a request never samples memory or creates a native resource.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ResidencyDiagnostics {
    backend_memory: bool,
    process_memory: bool,
}

impl ResidencyDiagnostics {
    /// Selects backend allocator and process-memory observations independently.
    pub const fn new(backend_memory: bool, process_memory: bool) -> Self {
        Self {
            backend_memory,
            process_memory,
        }
    }
}

/// Portable failure deriving a load request; backend support is checked separately.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExecutionPlanLoadError {
    /// The serialized or caller-built plan violates its portable contract.
    #[error(transparent)]
    Plan(#[from] eredu_core::ExecutionPlanError),
    /// Complete rank, wire, invocation and completion policy is required.
    #[error("single-device automatic plans require a 1x1x1 parallel topology; distributed plans require an explicit parallel load request")]
    MissingParallelRequest,
    /// The supplied parallel request does not belong to the plan's topology.
    #[error("parallel load request topology differs from the execution plan")]
    ParallelTopologyMismatch,
    /// The plan selected policy not understood by this runtime.
    #[error("unsupported execution-plan {0}")]
    Unsupported(&'static str),
    /// An invalid offload policy cannot become a load request.
    #[error(transparent)]
    Offload(#[from] eredu_core::residency::OffloadError),
    /// Invalid ordinary or parameter-bank residency controls.
    #[error(transparent)]
    Residency(#[from] crate::WeightResidencyPolicyError),
    /// Invalid quantization, drafting, or completion controls.
    #[error(transparent)]
    Request(#[from] NormalizedLoadRequestError),
}

/// Derives the exact load-time transformation without backend capability policy.
pub fn execution_plan_quantization(
    transformation: WeightTransformationPlan,
) -> Result<Option<eredu_core::QuantizationRequest>, ExecutionPlanLoadError> {
    use eredu_core::QuantizationRequest;
    let request = match transformation {
        WeightTransformationPlan::PreserveCheckpoint => return Ok(None),
        WeightTransformationPlan::Affine { bits, group_size } => QuantizationRequest::Affine {
            group_size: u32::try_from(group_size).map_err(|_| {
                NormalizedLoadRequestError::Quantization(format!(
                    "group_size must be non-negative, got {group_size}"
                ))
            })?,
            bits: u8::try_from(bits).map_err(|_| {
                NormalizedLoadRequestError::Quantization(format!("bits must fit in u8, got {bits}"))
            })?,
        },
        WeightTransformationPlan::MxFp4 => QuantizationRequest::MxFp4,
        _ => return Err(ExecutionPlanLoadError::Unsupported("weight transformation")),
    };
    NormalizedLoadRequest::with_quantization(request).weight_quantization()?;
    Ok(Some(request))
}

impl NormalizedLoadRequest {
    /// Derives all portable load policy from a plan and optional complete rank request.
    ///
    /// Distributed plans supply their exact rank, wire, invocation limits and completion
    /// policy explicitly because an execution plan does not contain those values. Native
    /// device identity and realized resources never enter this conversion.
    pub fn from_execution_plan(
        plan: &ExecutionPlan,
        diagnostics: ResidencyDiagnostics,
        parallel: Option<ParallelLoadRequest>,
    ) -> Result<Self, ExecutionPlanLoadError> {
        plan.validate_structure()?;
        match parallel {
            Some(parallel) if parallel.rank().topology() != *plan.topology() => {
                return Err(ExecutionPlanLoadError::ParallelTopologyMismatch);
            }
            None if !plan.topology().is_replicated() => {
                return Err(ExecutionPlanLoadError::MissingParallelRequest);
            }
            _ => {}
        }
        let ordinary = match plan.residency() {
            ResidencyPlan::FullyResident => OrdinaryWeightResidency::FullyResident,
            ResidencyPlan::LayerwiseHost {
                device_layer_window,
                device_budget_bytes,
                host_budget_bytes,
            } => OrdinaryWeightResidency::LayerwiseHost(
                LayerwiseLoadOptions::new(OffloadConfig::new(
                    *device_budget_bytes,
                    *host_budget_bytes,
                    *device_layer_window,
                )?)
                .with_max_cached_shards(plan.max_cached_shards())
                .with_memory_sampling(diagnostics.backend_memory, diagnostics.process_memory),
            ),
            ResidencyPlan::DenseDiskStream {
                device_budget_bytes,
                host_budget_bytes,
                host_lookahead,
                background_queue,
            } => OrdinaryWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(
                    *device_budget_bytes,
                    *host_budget_bytes,
                    *host_lookahead,
                    *background_queue,
                )?
                .with_max_cached_shards(plan.max_cached_shards())
                .with_memory_sampling(diagnostics.backend_memory, diagnostics.process_memory),
            ),
            _ => return Err(ExecutionPlanLoadError::Unsupported("residency")),
        };
        let residency = match plan.expert_cache() {
            Some(bank) => WeightResidency::with_independent_parameter_banks(
                ordinary,
                ParameterBankLoadOptions::new(
                    OffloadConfig::new(bank.device_budget_bytes(), bank.host_budget_bytes(), 1)?
                        .with_eviction_policy(bank.eviction_policy()),
                    bank.scratch_bytes(),
                    bank.prefill_bank_bytes(),
                )?,
            ),
            None => WeightResidency::with_layers(ordinary.layers()),
        };
        let mut request = execution_plan_quantization(plan.weight_transformation())?
            .map_or_else(Self::default, Self::with_quantization)
            .with_weight_residency(residency)
            .with_max_cached_shards(
                std::num::NonZeroUsize::new(plan.max_cached_shards())
                    .expect("plan structure validates a positive reader limit"),
            )
            .with_required_session_capabilities(*plan.required_session_capabilities())
            .with_prompt_cache_persistence(plan.prompt_cache_persistence())
            .with_drafting_plan(plan.drafting())?;
        if let Some(parallel) = parallel {
            request = request.with_parallel_execution(parallel)?;
        }
        request.validate_model_preparation()?;
        Ok(request)
    }
}

#[cfg(test)]
mod tests;
