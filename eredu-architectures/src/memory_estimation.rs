//! Cold, architecture-owned projections for request memory planning.

use crate::{
    configuration::{GgufModelConfig, SafetensorsModelConfig},
    processor_plan::ArtifactArchitecturePlan,
};
use eredu_core::{CapabilityError, StateMemoryLayout};
use eredu_runtime::memory_estimation::WorkspaceGeometry;

/// Architecture payload geometry; missing workspace coverage never rejects execution.
#[derive(Debug, Clone)]
pub struct GenerationMemoryGeometry {
    /// Exact persistent-state schedule from the normalized family.
    pub state_layout: StateMemoryLayout,
    /// Dense workspace dimensions, when the equations are modeled.
    pub workspace: Option<WorkspaceGeometry>,
    /// Reasons workspace or additional graph coverage is incomplete.
    pub assumptions: Vec<String>,
}

fn dimension(value: i32) -> Result<u64, CapabilityError> {
    u64::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| CapabilityError::InvalidConfiguration {
            field: "workspace_dimension",
            detail: format!("expected a positive dimension, got {value}"),
        })
}

fn dense(
    hidden: i32,
    intermediate: i32,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    vocabulary: i32,
) -> Result<WorkspaceGeometry, CapabilityError> {
    let head_dim = dimension(head_dim)?;
    let width = |heads| {
        dimension(heads)?
            .checked_mul(head_dim)
            .ok_or(CapabilityError::ArithmeticOverflow {
                operation: "attention projection width",
            })
    };
    Ok(WorkspaceGeometry {
        hidden_size: dimension(hidden)?,
        intermediate_size: dimension(intermediate)?,
        query_width: width(heads)?,
        key_value_width: width(kv_heads)?,
        query_heads: dimension(heads)?,
        vocabulary_size: dimension(vocabulary)?,
    })
}

/// Projects validated metadata without allocating a device, tensor, or source store.
pub fn generation_memory_geometry(
    plan: &ArtifactArchitecturePlan,
) -> Result<GenerationMemoryGeometry, CapabilityError> {
    let (capability, workspace) = match (
        plan.safetensors_architecture().map(|p| p.model()),
        plan.gguf_plan().map(|p| p.model()),
    ) {
        (Some(SafetensorsModelConfig::Llama(args)), None)
        | (None, Some(GgufModelConfig::Llama(args))) => (
            crate::capability::llama(args)?,
            Some(dense(
                args.hidden_size,
                args.intermediate_size,
                args.num_attention_heads,
                args.num_key_value_heads,
                args.head_dim,
                args.vocab_size,
            )?),
        ),
        (Some(SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(GgufModelConfig::Qwen(args))) => (
            crate::capability::qwen(args)?,
            if args.is_moe() {
                None
            } else {
                Some(dense(
                    args.hidden_size,
                    args.intermediate_size,
                    args.num_attention_heads,
                    args.num_key_value_heads,
                    args.head_dim,
                    args.vocab_size,
                )?)
            },
        ),
        (Some(SafetensorsModelConfig::Nanbeige(args)), None)
        | (None, Some(GgufModelConfig::Nanbeige(args))) => {
            let dense_args = args.dense_config();
            (
                crate::capability::nanbeige(args)?,
                Some(dense(
                    dense_args.hidden_size,
                    dense_args.intermediate_size,
                    dense_args.num_attention_heads,
                    dense_args.num_key_value_heads,
                    dense_args.head_dim,
                    dense_args.vocab_size,
                )?),
            )
        }
        (Some(SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(GgufModelConfig::DeepSeekV3(args))) => {
            (crate::capability::deepseek_v3(args)?, None)
        }
        (Some(SafetensorsModelConfig::DeepSeekV4(args)), None)
        | (None, Some(GgufModelConfig::DeepSeekV4(args))) => {
            (crate::capability::deepseek_v4(args)?, None)
        }
        (Some(SafetensorsModelConfig::Gemma4(args)), None)
        | (None, Some(GgufModelConfig::Gemma4(args))) => (crate::capability::gemma4(args)?, None),
        (Some(SafetensorsModelConfig::Gemma2(args)), None)
        | (None, Some(GgufModelConfig::Gemma2(args))) => (crate::capability::gemma2(args)?, None),
        (Some(SafetensorsModelConfig::GptOss(args)), None)
        | (None, Some(GgufModelConfig::GptOss(args))) => (crate::capability::gpt_oss(args)?, None),
        (Some(SafetensorsModelConfig::Inkling(args)), None)
        | (None, Some(GgufModelConfig::Inkling(args))) => (crate::capability::inkling(args)?, None),
        (Some(SafetensorsModelConfig::K2Horizon(args)), None)
        | (None, Some(GgufModelConfig::K2Horizon(args))) => {
            (crate::capability::k2_horizon(args)?, None)
        }
        (Some(SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(GgufModelConfig::KimiLinear(args))) => {
            (crate::capability::kimi_linear(args)?, None)
        }
        (Some(SafetensorsModelConfig::MuseGlimmer(args)), None)
        | (None, Some(GgufModelConfig::MuseGlimmer(args))) => {
            (crate::capability::muse_glimmer(args)?, None)
        }
        (Some(SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(GgufModelConfig::Lfm2(args))) => (crate::capability::lfm2(args)?, None),
        (Some(SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(GgufModelConfig::NemotronH(args))) => {
            (crate::capability::nemotron_h(args)?, None)
        }
        (Some(SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(GgufModelConfig::QwenHybrid(args))) => {
            (crate::capability::qwen_hybrid(args)?, None)
        }
        (Some(SafetensorsModelConfig::QwenVl(args)), None) => {
            (crate::capability::qwen_vl(args)?, None)
        }
        _ => {
            return Err(CapabilityError::Observation(
                "generation memory geometry is unavailable for this architecture".into(),
            ))
        }
    };
    let assumptions = if workspace.is_none() {
        vec!["persistent state uses executable family geometry; recurrent, routed, media, or specialized activation workspace is not yet modeled".into()]
    } else {
        Vec::new()
    };
    Ok(GenerationMemoryGeometry {
        state_layout: capability.into_parts().1,
        workspace,
        assumptions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_llama_projection_reuses_executable_state_and_dtype() {
        use crate::preparation_selection::tests::{inspected_llama, BoundedIndependentAdapter};
        let (_root, artifact) = inspected_llama();
        let geometry = generation_memory_geometry(artifact.architecture_plan()).unwrap();
        let state = eredu_core::estimate_runtime_state(
            &geometry.state_layout,
            eredu_core::InputTokenCount::text(7),
            3,
            1,
            std::num::NonZeroU8::new(4).unwrap(),
        )
        .unwrap();
        // Two layers, K and V, one KV head, four scalars per head, ten positions.
        assert_eq!(state.context_state_bytes, 2 * 2 * 4 * 10 * 4);
        assert_eq!(geometry.workspace.unwrap().key_value_width, 4);
        let outcome = crate::inspect_selected_model(
            artifact,
            &eredu_runtime::NormalizedLoadRequest::default(),
            &BoundedIndependentAdapter::default(),
            eredu_core::MediaFeatureAvailability {
                image: false,
                audio: false,
            },
        );
        let selected = outcome.selected().expect("cold dense selection");
        assert_eq!(
            selected
                .preparation()
                .execution()
                .text_realization()
                .state()
                .floating_dtype()
                .unwrap()
                .bytes()
                .get(),
            4
        );
        // Native scratch is derived from metadata without constructing tensors.
        assert!(
            selected
                .memory_materialization_workspace()
                .value()
                .unwrap()
                .ordinary_recipe_peak_bytes
                > 0
        );
    }

    #[test]
    fn hybrid_retains_persistent_geometry_without_inventing_workspace() {
        use eredu_core::ModelConfigurationResolver;
        let config: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/configs/qwen3.5-0.8b-mlx-87e768fb.json"
        ))
        .unwrap();
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config["config"])
            .unwrap();
        let geometry = generation_memory_geometry(resolved.architecture_plan()).unwrap();
        assert!(geometry.workspace.is_none());
        assert!(!geometry.assumptions.is_empty());
        let state = eredu_core::estimate_runtime_state(
            &geometry.state_layout,
            eredu_core::InputTokenCount::text(10),
            4,
            1,
            std::num::NonZeroU8::new(2).unwrap(),
        )
        .unwrap();
        assert!(state.fixed_state_bytes > 0);
        assert!(state.context_state_bytes > 0);
    }

    #[test]
    fn dense_projection_preserves_grouped_query_widths() {
        let geometry = dense(128, 352, 8, 2, 16, 32000).unwrap();
        assert_eq!(geometry.query_width, 128);
        assert_eq!(geometry.key_value_width, 32);
        assert_eq!(geometry.intermediate_size, 352);
        assert!(dense(128, 352, 0, 2, 16, 32000).is_err());
    }
}
