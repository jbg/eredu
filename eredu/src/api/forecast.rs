//! Request-aware forecasts shared by ordinary and controlled generation.
use super::*;
use eredu_core::{CapabilityError, InputTokenCount};
use eredu_runtime::memory_forecast::LoadedMemoryProfile;
pub use eredu_runtime::memory_forecast::{
    CaptureMemoryPlan, ConversionRetentionGroupMemoryPlan, ConversionRetentionMemoryPlan,
    EmbeddedPredictionMemoryPlan, ForecastCalibration, ForecastExecutionContract,
    GenerationForecastBackend, GenerationForecastError, SpeculativeForecastBackend,
    SpeculativeMemoryPlan,
};
use serde::{Deserialize, Serialize};
mod continuation;
pub use continuation::*;

pub use eredu_runtime::memory_forecast::GenerationForecastOptions;

/// Estimate plus the exact descriptive request and execution contract used for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationForecast {
    /// Selection geometry and admitted ceilings for capture horizon recomputation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture: Option<CaptureMemoryPlan>,
    /// Selected speculative resource facts for reproducible phase recomputation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative: Option<SpeculativeMemoryPlan>,
    /// Computed interval and fit verdict.
    pub estimate: GenerationMemoryEstimate,
    /// Descriptive request for recomputation and inspection.
    pub request: GenerationMemoryRequest,
    /// Actual full-pass and vocabulary projection contract.
    pub execution: ForecastExecutionContract,
    /// Requested maximum positions per prefill invocation.
    pub requested_chunk_tokens: u64,
}

impl GenerationForecast {
    /// Recomputes an output allowance, retaining both speculative models and
    /// their transaction/scheduler limits when present.
    pub fn with_max_output_tokens(&self, tokens: u64) -> Result<Self, GenerationForecastError> {
        let mut candidate = self.clone();
        candidate.request.max_output_tokens = Some(tokens);
        candidate.request.forecast_output_tokens = tokens;
        if let Some(draft) = candidate
            .speculative
            .as_mut()
            .and_then(|p| p.draft.as_mut())
        {
            draft.max_output_tokens = Some(tokens);
            draft.forecast_output_tokens = tokens;
        }
        candidate.reestimate()?;
        Ok(candidate)
    }

    /// Recomputes a chunk alternative only when the actual request can be chunked.
    pub fn with_prefill_chunk(&self, tokens: u64) -> Result<Self, GenerationForecastError> {
        if tokens == 0 {
            return Err(CapabilityError::InvalidConfiguration {
                field: "prefill_chunk_tokens",
                detail: "expected a positive chunk size".into(),
            }
            .into());
        }
        let mut candidate = self.clone();
        candidate.requested_chunk_tokens = tokens;
        if candidate.execution.full_pass_reason.is_none() {
            candidate.request.prefill_chunk_tokens = tokens;
        }
        candidate.reestimate()?;
        Ok(candidate)
    }

    fn reestimate(&mut self) -> Result<(), GenerationForecastError> {
        if let Some(plan) = &self.capture {
            let tokens = self
                .request
                .max_output_tokens
                .unwrap_or(self.request.forecast_output_tokens);
            plan.apply(&mut self.request, tokens)?;
        }
        let mut estimate = match &self.speculative {
            Some(plan) => {
                eredu_runtime::memory_forecast::estimate_speculative_memory(&self.request, plan)?
            }
            None => eredu_runtime::memory_estimation::estimate_generation_memory(&self.request)?,
        };
        for assumption in &self.estimate.assumptions {
            if !estimate.assumptions.contains(assumption) {
                estimate.assumptions.push(assumption.clone());
            }
        }
        self.estimate = estimate;
        Ok(())
    }
}

fn failure(error: impl std::fmt::Display) -> GenerationForecastError {
    CapabilityError::Observation(error.to_string()).into()
}
fn chunk_tokens(policy: eredu_core::PrefillChunkPolicy, positions: u64) -> u64 {
    match policy {
        eredu_core::PrefillChunkPolicy::Unchunked => positions.max(1),
        eredu_core::PrefillChunkPolicy::Bounded(n) => n.get() as u64,
    }
}

impl<B: GenerationForecastBackend> LoadedModel<B> {
    /// Forecasts the exact ordinary prepared request without consuming it,
    /// advancing state, invoking callbacks, or reopening its checkpoint.
    /// Start from fresh/reset state; backends must leave continuation peaks
    /// unbounded unless they project the existing state as well.
    pub fn forecast_prepared_generation<F>(
        &self,
        request: &PreparedChatGenerationRequest<'_, B, F>,
        options: &GenerationForecastOptions,
    ) -> Result<GenerationForecast, GenerationForecastError> {
        self.forecast_input(&request.input, request.settings, options, false)
    }

    /// Forecasts the selected target/drafter, rollback, verification and lookahead
    /// phases without consuming preparation. Uncovered components remain unknown.
    pub fn forecast_prepared_speculative_generation<D, F>(
        &self,
        request: &PreparedChatSpeculativeGenerationRequest<'_, B, D, F>,
        options: &GenerationForecastOptions,
    ) -> Result<GenerationForecast, GenerationForecastError>
    where
        B: SpeculativeForecastBackend<D>,
    {
        let mut forecast = self.forecast_input(&request.input, request.settings, options, false)?;
        self.project_speculation(&mut forecast, &request.drafting, request.options, options)?;
        Ok(forecast)
    }

    /// Forecasts a raw-token speculative lane with its actual selected drafter.
    /// Uses the same accounting as prepared ordinary and controlled requests.
    pub fn forecast_speculative_token_ids<D>(
        &self,
        token_ids: &[u32],
        settings: PreparedChatGenerationSettings,
        drafting: &eredu_core::SpeculativeDraft<'_, D>,
        speculative: PreparedChatSpeculativeGenerationOptions,
        options: &GenerationForecastOptions,
    ) -> Result<GenerationForecast, GenerationForecastError>
    where
        B: SpeculativeForecastBackend<D>,
    {
        let mut forecast = self.forecast_token_ids(token_ids, settings, options)?;
        self.project_speculation(&mut forecast, drafting, speculative, options)?;
        Ok(forecast)
    }

    fn project_speculation<D>(
        &self,
        forecast: &mut GenerationForecast,
        drafting: &eredu_core::SpeculativeDraft<'_, D>,
        speculative: PreparedChatSpeculativeGenerationOptions,
        options: &GenerationForecastOptions,
    ) -> Result<(), GenerationForecastError>
    where
        B: SpeculativeForecastBackend<D>,
    {
        speculative.scheduler.validate()?;
        let Some(profile) = B::speculative_memory_profile(&self.runtime, drafting)? else {
            return mark_speculative_forecast(forecast);
        };
        if speculative.max_draft_tokens.get() as u64 > profile.proposal_capacity {
            return Err(failure(
                "forecast proposal width exceeds the selected drafter capacity",
            ));
        }
        forecast.execution = ForecastExecutionContract {
            full_pass_reason: Some(
                "selected speculative transaction uses full-pass prefill".into(),
            ),
            logits: LogitsWorkspace::EveryPosition,
        };
        // Speculative lanes have isolated target state, even if an unrelated
        // ordinary session has advanced. Observe that exact lane selection.
        let target = loaded_forecast(
            B::speculative_target_memory_profile(&self.runtime)?,
            forecast.request.input,
            forecast
                .request
                .max_output_tokens
                .unwrap_or(forecast.request.forecast_output_tokens),
            eredu_core::PrefillChunkPolicy::Unchunked,
            forecast.execution.clone(),
            options,
        )?;
        forecast.request = target.request;
        forecast
            .estimate
            .assumptions
            .extend(target.estimate.assumptions);
        forecast.request.prefill_chunk_tokens = forecast.request.input.model_positions.max(1);
        for domain in &mut forecast.request.domains {
            for execution in &mut domain.executions {
                execution.logits = LogitsWorkspace::EveryPosition;
            }
        }
        let draft = profile
            .draft
            .map(|profile| {
                loaded_forecast(
                    profile,
                    forecast.request.input,
                    forecast
                        .request
                        .max_output_tokens
                        .unwrap_or(forecast.request.forecast_output_tokens),
                    eredu_core::PrefillChunkPolicy::Unchunked,
                    forecast.execution.clone(),
                    options,
                )
            })
            .transpose()?;
        if let Some(draft) = &draft {
            forecast
                .estimate
                .assumptions
                .extend(draft.estimate.assumptions.clone());
        }
        let mut embedded = profile
            .embedded
            .as_ref()
            .map(|topology| {
                eredu_runtime::memory_forecast::EmbeddedPredictionMemoryPlan::from_topology(
                    topology,
                    &forecast.request,
                    &options.calibration,
                )
            })
            .transpose()?;
        if let Some(embedded) = embedded.as_mut() {
            forecast.estimate.assumptions.extend(
                eredu_runtime::memory_forecast::apply_embedded_parameter_conversion_credit(
                    &mut forecast.request,
                    embedded,
                    profile.parameter_conversions.as_deref(),
                )?,
            );
        }
        forecast.speculative = Some(SpeculativeMemoryPlan {
            embedded,
            draft: draft.map(|f| f.request),
            auxiliary_bytes_per_position: profile.auxiliary_bytes_per_position,
            sampling_bytes_per_vocabulary_entry: profile.sampling_bytes_per_vocabulary_entry,
            max_draft_tokens: speculative.max_draft_tokens.get() as u64,
            scheduler: speculative.scheduler,
            shared_allocator: profile.shared_allocator,
        });
        forecast.reestimate()
    }

    fn forecast_input(
        &self,
        input: &PreparedChatInput<'_, B>,
        settings: PreparedChatGenerationSettings,
        options: &GenerationForecastOptions,
        instrumented: bool,
    ) -> Result<GenerationForecast, GenerationForecastError> {
        let count = match input {
            PreparedChatInput::RenderedPrompt(chat) => self.count_prepared_chat(chat)?,
            PreparedChatInput::TokenIds { token_ids, .. } => self.count_token_ids(token_ids)?,
            PreparedChatInput::PreparedBackendInput { prompt, .. } => {
                self.count_prepared_input(prompt)?
            }
        };
        self.forecast_count(
            count,
            input.backend_prompt(),
            settings,
            options,
            instrumented,
        )
    }

    /// Forecasts an encoded ordinary request using checkpoint-resolved output
    /// limits and the same settings as ordinary or controlled generation.
    ///
    /// Call on fresh/reset request state; use a controlled continuation forecast
    /// for an existing request. Reads current conversion claims without settling,
    /// allocating execution resources, trimming or consuming observation budgets.
    /// After trim, query again to remove released bindings' cast credit. Retained
    /// conversions are already part of resident parameters, while possible new
    /// retention is a subset of pending workspace. A retention ceiling does not
    /// clamp temporary casts or turn missing native facts into finite bounds.
    pub fn forecast_token_ids(
        &self,
        token_ids: &[u32],
        settings: PreparedChatGenerationSettings,
        options: &GenerationForecastOptions,
    ) -> Result<GenerationForecast, GenerationForecastError> {
        self.forecast_count(
            self.count_token_ids(token_ids)?,
            None,
            settings,
            options,
            false,
        )
    }

    /// Forecasts a prepared observed request before uninterrupted execution or
    /// `start_controlled_*`. Trace-only requests preserve ordinary chunking.
    /// Instrumented requests project admitted selections where native costs are known,
    /// falling back to admitted logical storage limits otherwise. Both allow one
    /// retained record history and one compact JSON trace. Extra application
    /// copies, snapshots and branches need separate allowances; unrelated unknown
    /// costs remain unknown. Forecasting does not consume capture/trace budgets.
    pub fn forecast_observed_generation(
        &self,
        prepared: &PreparedObservedGeneration,
        options: &GenerationForecastOptions,
    ) -> Result<GenerationForecast, GenerationForecastError> {
        if prepared.session_identity != self.session_identity {
            return Err(failure(
                "prepared forecast request belongs to a different loaded session",
            ));
        }
        let instrumented = !prepared.plan.is_empty() || prepared.intervention.is_some();
        let mut result = self.forecast_count(
            self.count_token_ids(&prepared.prompt_token_ids)?,
            None,
            prepared.settings,
            options,
            instrumented,
        )?;
        if instrumented {
            let projection = B::capture_memory_projection(
                &self.runtime,
                &prepared.plan,
                prepared.intervention.as_ref(),
                0,
            )?;
            result.capture = Some(CaptureMemoryPlan::new(
                &result.request,
                &prepared.plan,
                prepared.intervention.as_ref(),
                prepared.trace_limits,
                B::capture_transforms_complete_per_step(&self.runtime),
                projection,
            ));
            result.reestimate()?;
        }
        Ok(result)
    }

    fn forecast_count(
        &self,
        input: InputTokenCount,
        prompt: Option<&B::Prompt>,
        settings: PreparedChatGenerationSettings,
        options: &GenerationForecastOptions,
        instrumented: bool,
    ) -> Result<GenerationForecast, GenerationForecastError> {
        let (_, output) = self.resolve_text_generation_settings(settings)?;
        let profile = B::loaded_memory_profile(&self.runtime)?;
        let contract = B::forecast_execution_contract(&self.runtime, prompt, instrumented);
        loaded_forecast(
            profile,
            input,
            output.get() as u64,
            settings.prefill,
            contract,
            options,
        )
    }
}

/// Marks a target-only forecast incomplete when concurrent speculative resources
/// have not been projected. Used by planned raw-token consumers as well.
pub fn mark_speculative_forecast(
    forecast: &mut GenerationForecast,
) -> Result<(), GenerationForecastError> {
    mark_specialized(
        forecast,
        "speculative draft and verification state overlap is not projected",
    )
}

fn mark_specialized(
    forecast: &mut GenerationForecast,
    reason: &str,
) -> Result<(), GenerationForecastError> {
    forecast.execution.full_pass_reason = Some(reason.into());
    forecast.speculative = None;
    forecast.execution.logits = LogitsWorkspace::EveryPosition;
    forecast.request.prefill_chunk_tokens = forecast.request.input.model_positions.max(1);
    for domain in &mut forecast.request.domains {
        domain.staging.upper_bytes = None;
        domain.staging.detail = format!("{}; {reason}", domain.staging.detail);
        for execution in &mut domain.executions {
            execution.logits = LogitsWorkspace::EveryPosition;
        }
    }
    forecast.reestimate()?;
    Ok(())
}

pub(super) fn loaded_forecast(
    profile: LoadedMemoryProfile,
    input: InputTokenCount,
    output: u64,
    prefill: eredu_core::PrefillChunkPolicy,
    contract: ForecastExecutionContract,
    options: &GenerationForecastOptions,
) -> Result<GenerationForecast, GenerationForecastError> {
    let (request, assumptions) = eredu_runtime::memory_forecast::loaded_generation_request(
        profile,
        input,
        output,
        prefill,
        contract.clone(),
        options,
    )?;
    let mut estimate = eredu_runtime::memory_estimation::estimate_generation_memory(&request)?;
    estimate.assumptions.extend(assumptions);
    estimate.assumptions.push("Loaded request: loading is excluded; only declared resident parameter bytes are deducted, never process-global active allocation counters.".into());
    Ok(GenerationForecast {
        capture: None,
        speculative: None,
        request,
        estimate,
        execution: contract,
        requested_chunk_tokens: chunk_tokens(prefill, input.model_positions),
    })
}

#[cfg(test)]
mod tests;

/// Forecasts retained cold selection with the library's shared calibration.
pub fn forecast_inspected_generation(
    inspection: &eredu_architectures::ModelInspectionOutcome,
    options: &GenerationMemoryOptions,
    calibration: &ForecastCalibration,
) -> Result<GenerationForecast, GenerationForecastError> {
    let mut request = inspected_generation_memory_request(inspection, options)?;
    let selected = inspection
        .selected()
        .ok_or_else(|| failure("cold execution selection unavailable"))?;
    let full_pass_reason = selected
        .preparation()
        .prefill_chunking_support()
        .err()
        .map(str::to_owned)
        .or_else(|| {
            (options.input.model_positions != options.input.text_tokens)
                .then(|| "prepared media input requires a complete prefill pass".into())
        });
    let modeled_final_position = full_pass_reason.is_none()
        && request
            .domains
            .iter()
            .flat_map(|d| &d.executions)
            .all(|e| e.workspace.is_some());
    let execution = ForecastExecutionContract {
        full_pass_reason,
        logits: if modeled_final_position {
            LogitsWorkspace::FinalPosition
        } else {
            LogitsWorkspace::EveryPosition
        },
    };
    for domain in &mut request.domains {
        for e in &mut domain.executions {
            e.logits = execution.logits;
        }
    }
    calibration.apply(&mut request)?;
    let mut estimate = eredu_runtime::memory_estimation::estimate_generation_memory(&request)?;
    estimate.assumptions.extend(
        eredu_architectures::memory_estimation::generation_memory_geometry(
            selected.inspection().architecture_plan(),
        )?
        .assumptions,
    );
    Ok(GenerationForecast {
        capture: None,
        speculative: None,
        request,
        estimate,
        execution,
        requested_chunk_tokens: options.prefill_chunk_tokens,
    })
}
