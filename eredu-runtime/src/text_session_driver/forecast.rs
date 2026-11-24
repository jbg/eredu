//! Portable forecasting over installed executable facts and native memory observations.

use eredu_core::{
    capability::{NativeExecutionMemory, NativeWorkspaceEstimate, ParameterMemoryFacts},
    capture::CaptureWorkspace,
};

use super::TextExecutable;
use super::{
    capability::{
        installed_parameter_memory, static_memory_from_residency_and_components,
        TextCapabilityMechanisms,
    },
    InstalledTextState, TextPromptFor, TextSessionDriver,
};
use crate::executable_binding::ErasedTextExecutable;
use crate::memory_estimation::{LogitsWorkspace, MemoryBytes};
use crate::memory_forecast::*;
use eredu_core::{
    backend::TextOperationDriver, BackendFailure, CapabilityError, ModelCapabilitySession,
    ModelRuntime, TextGenerationBackend,
};

fn workspace_bytes(estimate: NativeWorkspaceEstimate) -> MemoryBytes {
    MemoryBytes::estimated(estimate.lower, estimate.upper, estimate.evidence)
}

/// Retained facts of a separately loaded speculative drafter.
pub trait DraftForecastFacts {
    /// Retained proposal limit.
    fn forecast_proposal_capacity(&self) -> u64;
    /// Projection for a conditioned assistant, when selected.
    fn conditioned_forecast(
        &self,
    ) -> Result<Option<ConditionedAssistantMemoryProfile>, GenerationForecastError>;
    /// Projection for an independent decoder, when selected.
    fn independent_forecast(&self) -> Result<Option<LoadedMemoryProfile>, GenerationForecastError>;
}

/// Projects an independent decoder from its own executable and retained queue.
pub(crate) fn independent_draft_profile<E: eredu_core::backend::TextExecutableTypes>(
    model: &dyn crate::speculative::IndependentDraftModel<E>,
    memory: ParameterMemoryFacts,
) -> Result<LoadedMemoryProfile, GenerationForecastError> {
    let model_exec = model.executable();
    let residency = model_exec
        .residency_report()
        .map_err(BackendFailure::from_error)?;
    let geometry = model.memory_geometry(residency.as_ref())?;
    let parameter_conversions = residency
        .as_ref()
        .and_then(|report| report.device_parameter_conversions().map(<[_]>::to_vec));
    let banks = super::capability::bank_memory_facts(
        model_exec
            .parameter_bank_report()
            .map_err(BackendFailure::from_error)?,
    );
    let parameters = static_memory_from_residency_and_components(
        residency,
        std::iter::empty(),
        banks,
        memory.native,
    )?;
    Ok(LoadedMemoryProfile {
        geometry,
        parameter_conversions,
        parameters,
        available: memory.available,
        placement: memory.placement,
        allocator_cache_limit: memory.allocator_cache_limit,
    })
}

impl<B, N, G> GenerationForecastSession<B> for TextSessionDriver<N::Resources, N, G>
where
    Self: TextOperationDriver<B, Prompt = TextPromptFor<B>>,
    B: TextGenerationBackend + eredu_core::input::PlainTextTokens<B::InputTensor>,
    N: TextCapabilityMechanisms<B>
        + NativeExecutionMemory<B>
        + CaptureWorkspace
        + super::TextTransactionMechanisms,
    Self: ModelCapabilitySession<B>,
{
    fn refinement_memory_plan(
        runtime: &ModelRuntime<B, Self>,
        policy: Option<eredu_core::refinement::RefinementPolicy>,
    ) -> Result<Option<crate::memory_forecast::RefinementMemoryPlan>, GenerationForecastError> {
        let session = runtime.session();
        if !session.supports_block_refinement() {
            return Ok(None);
        }
        let model = session.resources().model.as_ref();
        let mut workspace = N::refinement_workspace();
        if session.native().supports_host_refinement() {
            // Full host readback, retained rows, confidence, draws and the next score
            // matrix are simultaneously live during the portable refinement transition.
            workspace.upper = workspace.upper.saturating_add(64);
            workspace.evidence =
                "native score staging plus portable host refinement working storage";
        }
        Ok(model.block_refinement_geometry().map(|geometry| {
            crate::memory_forecast::RefinementMemoryPlan {
                canvas_positions: geometry.canvas_length,
                vocabulary_size: geometry.vocabulary_size,
                stability_history: policy.unwrap_or_default().stability_history_length(),
                bytes_per_vocabulary_entry: workspace_bytes(workspace),
            }
        }))
    }

    fn shared_backbone_memory_profile(
        runtime: &ModelRuntime<B, Self>,
    ) -> Result<Option<crate::memory_forecast::SpeculativeMemoryProfile>, GenerationForecastError>
    {
        let Some(policy) =
            N::capability(&runtime.session().resources().model).shared_backbone_draft()
        else {
            return Ok(None);
        };
        Ok(Some(crate::memory_forecast::SpeculativeMemoryProfile {
            conditioned: None,
            refinement: Self::refinement_memory_plan(runtime, None)?,
            shared_backbone: Some(
                crate::memory_forecast::SharedBackboneMemoryPlan::from_policy(policy),
            ),
            embedded: None,
            draft: None,
            parameter_conversions: None,
            auxiliary_bytes_per_position: crate::memory_estimation::MemoryBytes::exact(0),
            sampling_bytes_per_vocabulary_entry: workspace_bytes(N::sampling_workspace()),
            proposal_capacity: policy.proposal_capacity().get() as u64,
            shared_allocator: false,
        }))
    }

    fn capture_memory_projection(
        runtime: &ModelRuntime<B, Self>,
        capture: &eredu_core::capture::AdmittedCapturePlan,
        intervention: Option<&eredu_core::intervention::AdmittedInterventionPlan>,
        first_prediction: u64,
    ) -> Result<Option<crate::capture::CaptureUsageProjection>, GenerationForecastError> {
        // Partition transport and intervention evidence need their own complete
        // accounting. Deferred activation sources may allocate during creation;
        // ordinary logits are already materialized by generation's workspace.
        if N::control_transport(&runtime.session().resources().retained).is_some()
            || intervention.is_some_and(|p| !p.plan().operations.is_empty())
        {
            return Ok(None);
        }
        crate::capture::project_capture_usage(
            capture,
            first_prediction,
            |shape, selection, slice| {
                if selection.path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                    return Ok(None);
                }
                N::estimate_capture(shape, selection, slice).map(Some)
            },
        )
        .map_err(|e| CapabilityError::Observation(e.to_string()).into())
    }

    fn capture_transforms_complete_per_step(_: &ModelRuntime<B, Self>) -> bool {
        N::capture_completes_per_step()
    }

    fn loaded_memory_profile(
        runtime: &ModelRuntime<B, Self>,
    ) -> Result<LoadedMemoryProfile, GenerationForecastError> {
        memory_profile::<B, N, G>(runtime, false)
    }

    fn forecast_execution_contract(
        runtime: &ModelRuntime<B, Self>,
        prompt: Option<&TextPromptFor<B>>,
        instrumented: bool,
    ) -> ForecastExecutionContract {
        let model = &runtime.session().resources().model;
        let retains_cursor = model.as_ref().retains_prefill_cursor();
        let full_pass_reason = if retains_cursor {
            None
        } else if instrumented {
            Some("capture or intervention requires one complete prefill invocation".into())
        } else if let Err(reason) =
            TextOperationDriver::<B>::text_prefill_chunking_support(runtime.session())
        {
            Some(reason.into())
        } else if prompt.is_some_and(|prompt| {
            prompt
                .plain_token_tensor()
                .and_then(B::token_count)
                .is_none()
        }) {
            Some("prepared input structure requires a complete prefill pass".into())
        } else {
            None
        };
        ForecastExecutionContract {
            input_retention: if retains_cursor {
                crate::memory_forecast::PrefillInputRetention::PreparedProducts
            } else {
                crate::memory_forecast::PrefillInputRetention::TokenIds
            },
            full_pass_reason,
            logits: if !instrumented && model.as_ref().projects_final_prefill_position() {
                LogitsWorkspace::FinalPosition
            } else {
                LogitsWorkspace::EveryPosition
            },
        }
    }
}

fn memory_profile<B, N, G>(
    runtime: &ModelRuntime<B, TextSessionDriver<N::Resources, N, G>>,
    continuation: bool,
) -> Result<LoadedMemoryProfile, GenerationForecastError>
where
    B: TextGenerationBackend,
    N: TextCapabilityMechanisms<B> + NativeExecutionMemory<B> + super::TextTransactionMechanisms,
{
    let session = runtime.session();
    let mut geometry = session
        .model_discovery()
        .ok_or_else(|| {
            CapabilityError::Observation(
                "loaded executable has no retained architecture memory projection".into(),
            )
        })?
        .generation_memory()?
        .clone();
    let offset = session
        .resources()
        .model
        .as_ref()
        .forecast_state_offset()
        .map_err(BackendFailure::from_error)?;
    if !continuation && offset != Some(0) {
        geometry.workspace = None;
        geometry.execution_topology = None;
        geometry.assumptions.push("existing or unavailable session state is not projected; reset before forecasting a fresh request".into());
    }
    let memory = N::placement_memory(runtime.backend())?;
    let residency = session
        .residency_report()
        .map_err(BackendFailure::from_error)?;
    let parameters =
        installed_parameter_memory::<B, N, G>(session, residency.clone(), memory.native)?;
    geometry.apply_resident_storage(
        N::control_transport(&session.resources().retained).is_none()
            && session.parameter_overlay_identity().is_none(),
        residency.as_ref(),
    );
    let parameter_conversions =
        residency.and_then(|report| report.device_parameter_conversions().map(<[_]>::to_vec));
    Ok(LoadedMemoryProfile {
        parameter_conversions,
        geometry,
        parameters,
        available: memory.available,
        placement: memory.placement,
        allocator_cache_limit: memory.allocator_cache_limit,
    })
}

pub(super) fn automatic_prefill_chunk<B, N>(
    runtime: &ModelRuntime<B, TextSessionDriver<N::Resources, N>>,
    tokens: usize,
) -> std::num::NonZeroUsize
where
    B: TextGenerationBackend,
    N: TextCapabilityMechanisms<B> + NativeExecutionMemory<B> + super::TextTransactionMechanisms,
{
    use eredu_core::{InputTokenCount, PrefillChunkPolicy};
    let fallback = PrefillChunkPolicy::AUTOMATIC_FALLBACK;
    let session = runtime.session();
    // A rank-local observation cannot select a different collective schedule.
    if N::control_transport(&session.resources().retained).is_some() {
        return fallback;
    }
    let model = session.resources().model.as_ref();
    if model.forecast_state_offset().ok() != Some(Some(0)) {
        return fallback;
    }
    let Ok(profile) = memory_profile::<B, N, _>(runtime, false) else {
        return fallback;
    };
    let contract = ForecastExecutionContract {
        input_retention: if model.retains_prefill_cursor() {
            PrefillInputRetention::PreparedProducts
        } else {
            PrefillInputRetention::TokenIds
        },
        full_pass_reason: None,
        // Prefix invocations and instrumented requests can expose every row.
        logits: LogitsWorkspace::EveryPosition,
    };
    let Ok((request, _)) = loaded_generation_request(
        profile,
        InputTokenCount::text(tokens as u64),
        0,
        PrefillChunkPolicy::Automatic,
        contract,
        &GenerationForecastOptions::default(),
    ) else {
        return fallback;
    };
    crate::memory_estimation::select_prefill_chunk(&request).unwrap_or(fallback)
}
impl<B, N, G> ContinuationForecastSession<B> for TextSessionDriver<N::Resources, N, G>
where
    Self: TextOperationDriver<B, Prompt = TextPromptFor<B>>,
    B: TextGenerationBackend + eredu_core::input::PlainTextTokens<B::InputTensor>,
    N: TextCapabilityMechanisms<B>
        + NativeExecutionMemory<B>
        + CaptureWorkspace
        + super::TextTransactionMechanisms,
    Self: ModelCapabilitySession<B>,
    N::Model: InstalledTextState,
    Self: eredu_core::execution_control::NativeTextStateSession<B>,
{
    fn continuation_memory_profile(
        runtime: &ModelRuntime<B, Self>,
        additional_input_tokens: u64,
    ) -> Result<Option<crate::memory_forecast::ContinuationMemoryProfile>, GenerationForecastError>
    {
        use crate::memory_estimation::MemoryBytes;
        use eredu_core::execution_control::NativeTextStateSession;
        // This read-only native observation validates backend identity and idle authority.
        let Some(current) = Self::estimate_native_text_state(runtime, None)
            .map_err(eredu_core::BackendFailure::from_error)?
        else {
            return Ok(None);
        };
        let model = &runtime.session().resources().model;
        let Some(position) = model
            .as_ref()
            .forecast_state_offset()
            .map_err(eredu_core::BackendFailure::from_error)?
            .and_then(|n| u64::try_from(n).ok())
        else {
            return Ok(None);
        };
        // Include the current capacity envelope even for a zero-token forecast;
        // retained tensor views alone need not describe their backing capacity.
        let Some(state) = model.control_state() else {
            return Ok(None);
        };
        let current_capacity = state.estimate_installed_growth(0);
        let current_bound = current_capacity.and_then(|n| current.retained_bytes.checked_add(n));
        let growth = state.estimate_installed_growth(additional_input_tokens);
        let peak = growth.and_then(|n| current.retained_bytes.checked_add(n));
        Ok(Some(crate::memory_forecast::ContinuationMemoryProfile {
            loaded: memory_profile::<B, N, G>(runtime, true)?,
            plan: crate::memory_forecast::ContinuationMemoryPlan {
                current_positions: position,
                additional_input_tokens,
                query_tokens: model.as_ref().block_refinement_geometry()
                    .map_or(1, |geometry| u64::from(geometry.canvas_length.get())),
                current_state: match current_bound {
                    Some(n) => MemoryBytes::estimated(
                        0,
                        n,
                        "installed native logical state/capacity and materialization allowance; not measured distinct backing",
                    ),
                    None => MemoryBytes::unknown(
                        "installed native capacity allowance unavailable or overflowing",
                    ),
                },
                peak_state: match peak {
                    Some(n) => MemoryBytes::estimated(
                        0,
                        n,
                        "installed state plus native growth envelope, including allocation capacity and interior peaks",
                    ),
                    None => MemoryBytes::unknown(
                        "native continuation growth bound unavailable or overflowing",
                    ),
                },
            },
        }))
    }
}
impl<B, N, G, D> SpeculativeForecastSession<B, D> for TextSessionDriver<N::Resources, N, G>
where
    Self: TextOperationDriver<B, Prompt = TextPromptFor<B>>,
    B: TextGenerationBackend + eredu_core::input::PlainTextTokens<B::InputTensor>,
    N: TextCapabilityMechanisms<B>
        + NativeExecutionMemory<B>
        + CaptureWorkspace
        + super::TextTransactionMechanisms,
    Self: ModelCapabilitySession<B>,
    D: DraftForecastFacts,
{
    fn speculative_target_memory_profile(
        runtime: &ModelRuntime<B, Self>,
    ) -> Result<LoadedMemoryProfile, GenerationForecastError> {
        // Independent speculative lanes realize isolated caches. Retain selected
        // geometry even when an unrelated ordinary session has advanced.
        memory_profile::<B, N, G>(runtime, true)
    }

    fn speculative_memory_profile(
        runtime: &ModelRuntime<B, Self>,
        drafting: &eredu_core::SpeculativeDraft<'_, D>,
    ) -> Result<Option<crate::memory_forecast::SpeculativeMemoryProfile>, GenerationForecastError>
    {
        if matches!(drafting, eredu_core::SpeculativeDraft::Target) {
            if <_ as TextOperationDriver<B>>::decoding_capabilities(runtime.session()).speculation {
                return Self::shared_backbone_memory_profile(runtime);
            }
            return Ok(Some(crate::memory_forecast::SpeculativeMemoryProfile {
                conditioned: None,
                refinement: Self::refinement_memory_plan(runtime, None)?,
                shared_backbone: None,
                embedded: None,
                parameter_conversions: None,
                draft: None,
                auxiliary_bytes_per_position: crate::memory_estimation::MemoryBytes::exact(0),
                sampling_bytes_per_vocabulary_entry: workspace_bytes(N::sampling_workspace()),
                proposal_capacity: 0,
                shared_allocator: false,
            }));
        }
        if matches!(drafting, eredu_core::SpeculativeDraft::SharedBackbone) {
            return Self::shared_backbone_memory_profile(runtime);
        }
        let eredu_core::SpeculativeDraft::External(drafter) = drafting else {
            let Some(discovery) = runtime.session().model_discovery() else {
                return Ok(None);
            };
            let Some(topology) = discovery
                .embedded_prediction_topology()
                .map_err(|error| CapabilityError::Observation(error.to_string()))?
            else {
                return Ok(None);
            };
            return Ok(Some(crate::memory_forecast::SpeculativeMemoryProfile {
                conditioned: None,
                refinement: Self::refinement_memory_plan(runtime, None)?,
                shared_backbone: None,
                embedded: Some(topology.clone()),
                parameter_conversions: runtime
                    .session()
                    .residency_report()
                    .map_err(eredu_core::BackendFailure::from_error)?
                    .and_then(|report| report.device_parameter_conversions().map(<[_]>::to_vec)),
                draft: None,
                auxiliary_bytes_per_position: crate::memory_estimation::MemoryBytes::exact(0),
                sampling_bytes_per_vocabulary_entry: workspace_bytes(N::sampling_workspace()),
                proposal_capacity: topology.proposal_capacity as u64,
                shared_allocator: false,
            }));
        };
        let conditioned = drafter.conditioned_forecast()?;
        let draft = drafter.independent_forecast()?;
        if draft.is_none() && conditioned.is_none() {
            // This selected assistant has no retained architecture resource projection.
            return Ok(None);
        }
        let target = N::placement_memory(runtime.backend())?;
        let draft_placement = draft
            .as_ref()
            .map(|profile| &profile.placement)
            .or_else(|| conditioned.as_ref().map(|profile| &profile.placement));
        let shared_allocator = target.placement.allocator.as_ref().is_some_and(|id| {
            draft_placement.and_then(|placement| placement.allocator.as_ref()) == Some(id)
        });
        Ok(Some(crate::memory_forecast::SpeculativeMemoryProfile {
            conditioned,
            refinement: Self::refinement_memory_plan(runtime, None)?,
            shared_backbone: None,
            embedded: None,
            parameter_conversions: None,
            draft,
            auxiliary_bytes_per_position: crate::memory_estimation::MemoryBytes::exact(0),
            sampling_bytes_per_vocabulary_entry: workspace_bytes(N::sampling_workspace()),
            proposal_capacity: drafter.forecast_proposal_capacity(),
            shared_allocator,
        }))
    }
}
