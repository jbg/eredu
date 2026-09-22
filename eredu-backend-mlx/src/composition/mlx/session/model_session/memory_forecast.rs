use super::*;
use eredu_core::{CapabilityError, Observed, TextGenerationBackend};
use eredu_runtime::memory_estimation::LogitsWorkspace;
use eredu_runtime::memory_forecast::GenerationForecastError;
use eredu_runtime::memory_forecast::{
    ForecastExecutionContract, GenerationForecastBackend, LoadedMemoryProfile,
};

impl GenerationForecastBackend for MlxBackend<'_> {
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
        Ok(LoadedMemoryProfile {
            geometry,
            parameters: crate::composition::mlx::capability::static_model_memory(session)?,
            available: crate::composition::mlx::capability::available_memory()?,
            host_execution,
            allocator_cache_limit: match safemlx::memory::cache_limit() {
                Ok(limit) => Observed::exact(limit as u64, "current native allocator-cache limit"),
                Err(error) => Observed::unavailable(error.to_string()),
            },
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
