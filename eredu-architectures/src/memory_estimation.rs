//! Cold projections reuse ordinary selected state and module-construction contracts.

use crate::{
    configuration::{GgufModelConfig, SafetensorsModelConfig},
    processor_plan::ArtifactArchitecturePlan,
};
use eredu_core::{CapabilityError, StateMemoryLayout};
use eredu_runtime::{execution_topology::*, memory_estimation::WorkspaceGeometry};

/// Architecture payload geometry; incomplete mechanisms never reject execution.
#[derive(Debug, Clone)]
pub struct GenerationMemoryGeometry {
    /// Exact persistent-state schedule from ordinary family capabilities.
    pub state_layout: StateMemoryLayout,
    /// Legacy workspace geometry, absent for topology-based production forecasts.
    pub workspace: Option<WorkspaceGeometry>,
    /// Reusable module topology from ordinary cold construction.
    pub execution_topology: Option<TextExecutionTopology>,
    /// Concrete missing invocation facts.
    pub assumptions: Vec<String>,
}

/// Retains exact selected topology and physical parameter formats without allocation.
pub fn selected_generation_memory_geometry(
    plan: &ArtifactArchitecturePlan,
    execution: &crate::SelectedExecution,
) -> Result<eredu_runtime::memory_forecast::LoadedMemoryGeometry, CapabilityError> {
    let mut geometry = generation_memory_geometry(plan)?;
    let text = execution.text_realization();
    geometry.execution_topology = text.requirements().execution_topology().cloned();
    // Routed construction can supply a topology even when ordinary unselected
    // admission cannot select that execution class. Retained selection is authoritative.
    geometry.assumptions.clear();
    if geometry.execution_topology.is_none() {
        geometry.assumptions.push(
            "ordinary module invocation topology is unavailable; persistent state remains described".into(),
        );
    }
    let state = text.state().layout();
    geometry.state_layout = StateMemoryLayout::new(
        state.layers().clone(),
        state.layer_prefix_offsets(),
        geometry.state_layout.hidden_size,
        geometry.state_layout.allocation_granularity,
        geometry.state_layout.completeness,
    )?;
    if execution.parallel_topology().is_some() {
        geometry.assumptions.push(
            "invocation workspace and logical state use an unpartitioned conservative envelope; selected parameter residency remains rank-local".into(),
        );
    }
    if !matches!(
        text.state().policy(),
        eredu_runtime::CacheResidencyPolicy::Device
    ) {
        geometry.assumptions.push(
            "logical cache payload excludes mechanism-specific offload staging and transport"
                .into(),
        );
    }
    if let Some(topology) = geometry.execution_topology.as_mut() {
        // Materialization selection, not source file/config defaults, determines
        // actual projection encoding (including requested load-time quantization).
        let formats = text
            .parameters()
            .iter()
            .map(|p| (p.name(), p.executable()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let update = |projection: &mut ProjectionTopology| {
            if let Some(format) = formats.get(projection.parameter.as_str()) {
                projection.format = *format;
            }
        };
        topology.visit_projections_mut(update);
        let nominal = text.state().floating_dtype().map(|d| d.bytes().get());
        let wider = text.materialization_tasks().iter().any(|task| {
            let dtype = task
                .derived_output()
                .map(|o| o.dtype().clone())
                .or_else(|| task.source_encoding().scalar_dtype().map(Into::into));
            matches!(dtype, Some(eredu_checkpoint::recipe::RecipeDtype::F32))
        });
        if nominal.is_some_and(|width| width < 4) && wider {
            let bytes = text
                .materialization_tasks()
                .iter()
                .try_fold(0u64, |total, task| {
                    let cast = task.logical_shape().iter().try_fold(4u64, |bytes, &dim| {
                        bytes
                            .checked_mul(dim as u64)
                            .ok_or(CapabilityError::ArithmeticOverflow {
                                operation: "mixed-precision parameter promotion",
                            })
                    })?;
                    topology
                        .selected_parameter_promotion_payloads
                        .insert(task.name().to_owned(), cast);
                    total
                        .checked_add(cast)
                        .ok_or(CapabilityError::ArithmeticOverflow {
                            operation: "mixed-precision parameter promotion",
                        })
                })?;
            topology.selected_parameter_promotion_bytes = Some(bytes);
        }
    }
    Ok(eredu_runtime::memory_forecast::LoadedMemoryGeometry {
        state_layout: geometry.state_layout,
        workspace: None,
        execution_topology: geometry.execution_topology,
        input_score_attention_mechanism: execution.input_score_attention_workspace(),
        scalar_bytes: text
            .state()
            .floating_dtype()
            .ok_or_else(|| {
                CapabilityError::Observation("selected floating state dtype unavailable".into())
            })?
            .bytes(),
        fully_resident: matches!(
            text.residency(),
            eredu_runtime::LayerWeightResidency::FullyResident
        ),
        assumptions: geometry.assumptions,
    })
}

/// Projects ordinary capabilities and construction topology before backend selection.
pub fn generation_memory_geometry(
    plan: &ArtifactArchitecturePlan,
) -> Result<GenerationMemoryGeometry, CapabilityError> {
    let capability = match (
        plan.safetensors_architecture().map(|p| p.model()),
        plan.gguf_plan().map(|p| p.model()),
    ) {
        (None, Some(GgufModelConfig::NemotronLabsDiffusion(plan))) => {
            crate::capability::nemotron_labs_diffusion(plan.text_config(), false)?
        }

        (Some(SafetensorsModelConfig::NemotronLabsDiffusion(args)), None) => {
            crate::capability::nemotron_labs_diffusion(args, args.vision.is_some())?
        }
        (Some(SafetensorsModelConfig::Llama(args)), None)
        | (None, Some(GgufModelConfig::Llama(args))) => crate::capability::llama(args)?,
        (Some(SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(GgufModelConfig::Qwen(args))) => crate::capability::qwen(args)?,
        (Some(SafetensorsModelConfig::Nanbeige(args)), None)
        | (None, Some(GgufModelConfig::Nanbeige(args))) => crate::capability::nanbeige(args)?,
        (Some(SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(GgufModelConfig::DeepSeekV3(args))) => crate::capability::deepseek_v3(args)?,
        (Some(SafetensorsModelConfig::DeepSeekV4(args)), None)
        | (None, Some(GgufModelConfig::DeepSeekV4(args))) => crate::capability::deepseek_v4(args)?,
        (Some(SafetensorsModelConfig::DiffusionGemma(args)), None)
        | (None, Some(GgufModelConfig::DiffusionGemma(args))) => {
            crate::capability::diffusion_gemma(args)?
        }
        (Some(SafetensorsModelConfig::Gemma4(args)), None)
        | (None, Some(GgufModelConfig::Gemma4(args))) => crate::capability::gemma4(args)?,
        (Some(SafetensorsModelConfig::Gemma2(args)), None)
        | (None, Some(GgufModelConfig::Gemma2(args))) => crate::capability::gemma2(args)?,
        (Some(SafetensorsModelConfig::GptOss(args)), None)
        | (None, Some(GgufModelConfig::GptOss(args))) => crate::capability::gpt_oss(args)?,
        (Some(SafetensorsModelConfig::Inkling(args)), None)
        | (None, Some(GgufModelConfig::Inkling(args))) => crate::capability::inkling(args)?,
        (Some(SafetensorsModelConfig::K2Horizon(args)), None)
        | (None, Some(GgufModelConfig::K2Horizon(args))) => crate::capability::k2_horizon(args)?,
        (Some(SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(GgufModelConfig::KimiLinear(args))) => crate::capability::kimi_linear(args)?,
        (Some(SafetensorsModelConfig::Muse(args)), None)
        | (None, Some(GgufModelConfig::Muse(args))) => crate::capability::muse(args)?,
        (Some(SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(GgufModelConfig::Lfm2(args))) => crate::capability::lfm2(args)?,
        (Some(SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(GgufModelConfig::NemotronH(args))) => crate::capability::nemotron_h(args)?,
        (Some(SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(GgufModelConfig::QwenHybrid(args))) => crate::capability::qwen_hybrid(args)?,
        (Some(SafetensorsModelConfig::QwenVl(args)), None) => crate::capability::qwen_vl(args)?,
        _ => {
            return Err(CapabilityError::Observation(
                "generation state geometry is unavailable for this architecture".into(),
            ))
        }
    };
    let execution_topology = crate::replicated_text::execution_topology(plan)
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let assumptions = if execution_topology.is_none() {
        vec!["ordinary module invocation topology is unavailable; persistent state remains described".into()]
    } else {
        Vec::new()
    };
    Ok(GenerationMemoryGeometry {
        state_layout: capability.into_parts().1,
        workspace: None,
        execution_topology,
        assumptions,
    })
}
