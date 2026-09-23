//! Cold, architecture-owned projections for request memory planning.

use crate::{
    configuration::{GgufModelConfig, SafetensorsModelConfig},
    processor_plan::ArtifactArchitecturePlan,
};
use eredu_core::{CapabilityError, StateMemoryLayout};
use eredu_runtime::memory_estimation::{GatedConvolutionWorkspace, WorkspaceGeometry};

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

/// Projects the exact selected state geometry and retains explicit coverage gaps.
pub fn selected_generation_memory_geometry(
    plan: &ArtifactArchitecturePlan,
    execution: &crate::SelectedExecution,
) -> Result<eredu_runtime::memory_forecast::LoadedMemoryGeometry, CapabilityError> {
    let mut geometry = generation_memory_geometry(plan)?;
    if let Some(explicit) = geometry
        .workspace
        .as_mut()
        .and_then(|g| g.input_score_attention.as_mut())
    {
        explicit.mechanism = execution.input_score_attention_workspace();
    }
    let text = execution.text_realization();
    if execution.parallel_topology().is_some() {
        geometry.state_layout = eredu_core::StateMemoryLayout::new(
            eredu_core::LayerSchedule::empty(),
            Vec::new(),
            geometry.state_layout.hidden_size,
            1,
            geometry.state_layout.completeness,
        )?;
        geometry.workspace = None;
        geometry
            .assumptions
            .push("rank-local state and workspace projection unavailable".into());
    } else {
        let state = text.state().layout();
        geometry.state_layout = eredu_core::StateMemoryLayout::new(
            state.layers().clone(),
            state.layer_prefix_offsets(),
            geometry.state_layout.hidden_size,
            geometry.state_layout.allocation_granularity,
            geometry.state_layout.completeness,
        )?;
    }
    if !matches!(
        text.state().policy(),
        eredu_runtime::CacheResidencyPolicy::Device
    ) || !execution.bounded_residency_exclusions().is_empty()
    {
        geometry.workspace = None;
    }
    if let Some(workspace) = geometry.workspace.as_mut() {
        // Mixed-width parameters can promote activations and create implicit
        // projection-weight casts. Reserve the full potential F32 payload as a
        // conservative bound, without asserting every parameter is converted.
        // This follows selected task geometry and encodings for every covered
        // dense architecture, not the checkpoint format or a family name.
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
                    total
                        .checked_add(cast)
                        .ok_or(CapabilityError::ArithmeticOverflow {
                            operation: "mixed-precision parameter promotion",
                        })
                })?;
            workspace.mixed_precision_parameter_bytes = Some(bytes);
        }
    }
    Ok(eredu_runtime::memory_forecast::LoadedMemoryGeometry {
        state_layout: geometry.state_layout,
        workspace: geometry.workspace,
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
        gated_convolution: None,
        input_score_attention: None,
        mixed_precision_parameter_bytes: None,
    })
}

fn lfm2_workspace(
    args: &crate::lfm2::ModelArgs,
) -> Result<Option<WorkspaceGeometry>, CapabilityError> {
    use crate::lfm2::OperatorPolicy;
    if args.has_sparse_moe_layers() {
        return Ok(None);
    }
    let conv_layers = args
        .layer_schedule
        .iter()
        .filter(|p| matches!(p.operator, OperatorPolicy::CausalConvolution))
        .count() as u64;
    let has_attention = args
        .layer_schedule
        .iter()
        .any(|p| matches!(p.operator, OperatorPolicy::SelfAttention(_)));
    let mut geometry = dense(
        args.hidden_size,
        args.dense_intermediate_size,
        args.num_attention_heads,
        args.num_key_value_heads,
        args.hidden_size / args.num_attention_heads,
        args.vocab_size,
    )?;
    if !has_attention {
        geometry.query_width = 0;
        geometry.key_value_width = 0;
        geometry.query_heads = 0;
    }
    if has_attention {
        geometry.input_score_attention = Some(
            eredu_runtime::memory_estimation::InputScoreAttentionWorkspace {
                layers: args.layer_schedule.len() as u64 - conv_layers,
                mechanism: None,
            },
        );
    }
    if conv_layers > 0 {
        geometry.gated_convolution = Some(GatedConvolutionWorkspace {
            channels: dimension(args.hidden_size)?,
            kernel_size: dimension(args.conv_l_cache)?,
            layers: conv_layers,
        });
    }
    Ok(Some(geometry))
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
        | (None, Some(GgufModelConfig::Lfm2(args))) => {
            (crate::capability::lfm2(args)?, lfm2_workspace(args)?)
        }
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
    fn cold_dense_mixed_width_parameters_bound_potential_promotions() {
        use crate::preparation_selection::tests::{inspected_llama, BoundedIndependentAdapter};
        use safetensors::{
            tensor::{serialize_to_file, Dtype, TensorView},
            SafeTensors,
        };
        let (root, artifact) = inspected_llama();
        drop(artifact);
        let path = root.path().join("model.safetensors");
        let data = std::fs::read(&path).unwrap();
        let tensors = SafeTensors::deserialize(&data).unwrap();
        // Retain F32 normalization gains while using BF16 dense matrices. MLX
        // RMS normalization's result type promotes this mixed input to F32.
        let values = tensors
            .tensors()
            .into_iter()
            .map(|(name, tensor)| {
                let dtype = if tensor.shape().len() == 2 {
                    Dtype::BF16
                } else {
                    tensor.dtype()
                };
                let bytes =
                    vec![0u8; tensor.shape().iter().product::<usize>() * dtype.bitsize() / 8];
                (name, tensor.shape().to_vec(), dtype, bytes)
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            values.iter().map(|(name, shape, dtype, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            }),
            None,
            &path,
        )
        .unwrap();
        let artifact = crate::configuration::inspect_artifact(root.path()).unwrap();
        let plan = artifact.architecture_plan().clone();
        let outcome = crate::inspect_selected_model(
            artifact,
            &eredu_runtime::NormalizedLoadRequest::default(),
            &BoundedIndependentAdapter::with_half(),
            eredu_core::MediaFeatureAvailability {
                image: false,
                audio: false,
            },
        );
        let execution = outcome
            .selected()
            .unwrap_or_else(|| panic!("mixed dense selection failed: {outcome:?}"))
            .preparation()
            .execution();
        let geometry = selected_generation_memory_geometry(&plan, execution).unwrap();
        assert_eq!(geometry.scalar_bytes.get(), 2);
        let workspace = geometry.workspace.unwrap();
        assert!(workspace.gated_convolution.is_none());
        assert!(workspace.input_score_attention.is_none());
        let expected = values
            .iter()
            .map(|(_, shape, _, _)| shape.iter().product::<usize>() as u64 * 4)
            .sum::<u64>();
        assert_eq!(workspace.mixed_precision_parameter_bytes, Some(expected));
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
    fn lfm2_dense_workspace_uses_normalized_ffn_and_hybrid_schedule() {
        use eredu_core::ModelConfigurationResolver;
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/lfm2/released-config.json"))
                .unwrap();
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let geometry = generation_memory_geometry(resolved.architecture_plan()).unwrap();
        let workspace = geometry.workspace.unwrap();
        assert_eq!(workspace.hidden_size, 1024);
        assert_eq!(workspace.intermediate_size, 4608); // normalized SwiGLU 2/3 and 256 rounding
        assert_eq!(workspace.query_width, 1024);
        assert_eq!(workspace.key_value_width, 512);
        assert_eq!(
            workspace.gated_convolution,
            Some(GatedConvolutionWorkspace {
                channels: 1024,
                kernel_size: 3,
                layers: 10
            })
        );
        assert_eq!(workspace.input_score_attention.as_ref().unwrap().layers, 6);
        assert!(workspace
            .input_score_attention
            .as_ref()
            .unwrap()
            .mechanism
            .is_none());
        let mut args = crate::lfm2::model_args_from_config_value(&config).unwrap();
        let mut policies: Vec<_> = args.layer_schedule.iter().copied().collect();
        for policy in &mut policies {
            policy.operator = crate::lfm2::OperatorPolicy::CausalConvolution;
        }
        args.layer_schedule =
            eredu_core::LayerSchedule::new(policies.len(), policies.clone()).unwrap();
        let conv = lfm2_workspace(&args).unwrap().unwrap();
        assert_eq!(
            (conv.query_heads, conv.query_width, conv.key_value_width),
            (0, 0, 0)
        );
        assert_eq!(conv.gated_convolution.unwrap().layers, 16);
        assert!(conv.input_score_attention.is_none());
        policies[0].feed_forward = crate::lfm2::FeedForwardPolicy::SparseMoe;
        args.layer_schedule = eredu_core::LayerSchedule::new(policies.len(), policies).unwrap();
        assert!(lfm2_workspace(&args).unwrap().is_none());
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
