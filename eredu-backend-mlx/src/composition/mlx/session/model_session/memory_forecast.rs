use super::*;
use eredu_core::{CapabilityError, Observed, TextGenerationBackend};
use eredu_runtime::memory_estimation::LogitsWorkspace;
use eredu_runtime::memory_forecast::GenerationForecastError;
use eredu_runtime::memory_forecast::{
    ForecastExecutionContract, GenerationForecastBackend, LoadedMemoryProfile,
};

impl
    eredu_runtime::memory_forecast::SpeculativeForecastBackend<
        crate::composition::mlx::speculative::MlxDrafter,
    > for MlxBackend<'_>
{
    fn speculative_memory_profile(
        runtime: &ModelRuntime<Self>,
        drafting: &eredu_core::SpeculativeDraft<
            '_,
            crate::composition::mlx::speculative::MlxDrafter,
        >,
    ) -> Result<
        Option<eredu_runtime::memory_forecast::SpeculativeMemoryProfile>,
        GenerationForecastError,
    > {
        let eredu_core::SpeculativeDraft::External(drafter) = drafting else {
            // Embedded prediction retains architecture-specific feature and head
            // state beyond the ordinary target geometry. Do not guess its bound.
            return Ok(None);
        };
        let target = Self::loaded_memory_profile(runtime)?;
        if drafter.topology() == eredu_core::SpeculativeExecutionTopology::CrossDeviceSplit
            && target.parameters.physical_semantics != eredu_core::PhysicalMemorySemantics::Unified
        {
            // Cross-device copies need exact separate-pool identities/placement.
            return Ok(None);
        }
        let Some(draft) = drafter.autoregressive_memory_profile(target)? else {
            // Feature-conditioned assistants require their own architecture
            // projection, including the captured target features they retain.
            return Ok(None);
        };
        Ok(Some(eredu_runtime::memory_forecast::SpeculativeMemoryProfile {
            draft: Some(draft),
            auxiliary_bytes_per_position: eredu_runtime::memory_estimation::MemoryBytes::exact(0),
            sampling_bytes_per_vocabulary_entry: eredu_runtime::memory_estimation::MemoryBytes::estimated(
                0, 128,
                "MLX speculative sampling calibration v1: 32 float32/index rows per live distribution for logits processing, filtering, normalization and residual sampling; planning envelope, not a kernel allocation guarantee",
            ),
            proposal_capacity: drafter.selected().requirements().strategy().proposal_capacity().get() as u64,
            shared_allocator: true,
        }))
    }
}

impl GenerationForecastBackend for MlxBackend<'_> {
    fn capture_transforms_complete_per_step(_: &ModelRuntime<Self>) -> bool {
        // Bounded capture evaluates and transfers synchronously; no lazy capture
        // graphs or native views survive delivery of a completed prediction.
        true
    }

    fn loaded_memory_profile(
        runtime: &ModelRuntime<Self>,
    ) -> Result<LoadedMemoryProfile, GenerationForecastError> {
        let session = runtime.session();
        let mut geometry = session
            .capture_discovery
            .as_ref()
            .ok_or_else(|| {
                CapabilityError::Observation(
                    "loaded executable has no retained architecture memory projection".into(),
                )
            })?
            .generation_memory()?
            .clone();
        let offset = session
            .payload
            .model
            .erased()
            .forecast_state_offset()
            .map_err(eredu_core::BackendFailure::from_error)?;
        if offset != Some(0) {
            geometry.workspace = None;
            geometry.assumptions.push("existing or unavailable session state is not projected; reset before forecasting a fresh request".into());
        }
        let host_execution = runtime
            .backend()
            .stream()
            .get_device()
            .and_then(|d| d.get_type())
            .map_err(|e| {
                eredu_core::BackendFailure::from_error(e)
                    .with_operation("forecast execution device")
            })?
            == safemlx::DeviceType::Cpu;
        let allocator_cache_limit = match crate::allocator_cache_policy() {
            Ok(policy) => {
                let source = format!(
                    "MLX allocator-cache policy {:?}: {} bytes",
                    policy.source, policy.limit_bytes
                );
                geometry.assumptions.push(source.clone());
                Observed::exact(policy.limit_bytes, source)
            }
            Err(error) => Observed::unavailable(error.to_string()),
        };
        Ok(LoadedMemoryProfile {
            geometry,
            parameters: crate::composition::mlx::capability::static_model_memory(session)?,
            available: crate::composition::mlx::capability::available_memory()?,
            host_execution,
            allocator_cache_limit,
        })
    }

    fn forecast_execution_contract(
        runtime: &ModelRuntime<Self>,
        prompt: Option<&Self::Prompt>,
        instrumented: bool,
    ) -> ForecastExecutionContract {
        let full_pass_reason = if instrumented {
            Some("capture or intervention requires one complete prefill invocation".into())
        } else if let Err(reason) = Self::text_prefill_chunking_support(runtime) {
            Some(reason.into())
        } else if prompt.is_some_and(|prompt| prompt.chunkable_text_tokens().is_none()) {
            Some("prepared input structure requires a complete prefill pass".into())
        } else {
            None
        };
        ForecastExecutionContract {
            full_pass_reason,
            logits: if !instrumented
                && runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .projects_final_prefill_position()
            {
                LogitsWorkspace::FinalPosition
            } else {
                LogitsWorkspace::EveryPosition
            },
        }
    }
}
