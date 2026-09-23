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
    if execution.parallel_topology().is_some() {
        geometry.state_layout = StateMemoryLayout::new(
            eredu_core::LayerSchedule::empty(),
            Vec::new(),
            geometry.state_layout.hidden_size,
            1,
            geometry.state_layout.completeness,
        )?;
        geometry.execution_topology = None;
        geometry
            .assumptions
            .push("rank-local invocation topology is unavailable".into());
    } else {
        let state = text.state().layout();
        geometry.state_layout = StateMemoryLayout::new(
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
        geometry.execution_topology = None;
        geometry.assumptions.push(
            "selected state or weight residency does not provide an invocation lifetime contract"
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
        update(&mut topology.output);
        for layer in &mut topology.layers {
            layer.input_projections.iter_mut().for_each(&update);
            match &mut layer.mixer {
                TokenMixerTopology::Attention { projections, .. }
                | TokenMixerTopology::GatedConvolution { projections, .. } => {
                    projections.iter_mut().for_each(&update)
                }
                TokenMixerTopology::Unknown { .. } => {}
            }
            match &mut layer.feed_forward {
                FeedForwardTopology::Gated { projections, .. } => {
                    projections.iter_mut().for_each(&update)
                }
                FeedForwardTopology::Routed {
                    projections,
                    router,
                    ..
                } => {
                    projections.iter_mut().for_each(&update);
                    update(router);
                }
                FeedForwardTopology::Unknown { .. } => {}
            }
        }
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
        (Some(SafetensorsModelConfig::MuseGlimmer(args)), None)
        | (None, Some(GgufModelConfig::MuseGlimmer(args))) => {
            crate::capability::muse_glimmer(args)?
        }
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
        let topology = geometry.execution_topology.unwrap();
        assert!(geometry.workspace.is_none());
        let TokenMixerTopology::Attention {
            kv_heads,
            key_width,
            ..
        } = topology.layers[0].mixer
        else {
            panic!("attention topology");
        };
        assert_eq!(kv_heads * key_width, 4);
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
        let topology = geometry.execution_topology.unwrap();
        assert!(geometry.workspace.is_none());
        let expected = values
            .iter()
            .map(|(_, shape, _, _)| shape.iter().product::<usize>() as u64 * 4)
            .sum::<u64>();
        assert_eq!(topology.selected_parameter_promotion_bytes, Some(expected));
        assert_eq!(
            topology
                .selected_parameter_promotion_payloads
                .values()
                .sum::<u64>(),
            expected
        );
        for task in execution.text_realization().materialization_tasks() {
            assert_eq!(
                topology
                    .selected_parameter_promotion_payloads
                    .get(task.name()),
                Some(&(task.logical_shape().iter().product::<usize>() as u64 * 4))
            );
        }
    }

    #[test]
    fn selected_topology_uses_load_time_quantization_without_loading() {
        use crate::preparation_selection::tests::{inspected_config, BoundedIndependentAdapter};
        let (_root, artifact) = inspected_config(serde_json::json!({
            "model_type":"llama", "hidden_size":32, "intermediate_size":64, "vocab_size":32,
            "num_hidden_layers":2, "num_attention_heads":4, "num_key_value_heads":2,
            "rms_norm_eps":0.00001, "head_dim":8, "max_position_embeddings":64, "tie_word_embeddings":false
        }));
        let plan = artifact.architecture_plan().clone();
        let request = eredu_runtime::NormalizedLoadRequest::with_quantization(
            eredu_core::QuantizationRequest::Affine {
                group_size: 16,
                bits: 4,
            },
        );
        let outcome = crate::inspect_selected_model(
            artifact,
            &request,
            &BoundedIndependentAdapter::with_transforms(),
            eredu_core::MediaFeatureAvailability {
                image: false,
                audio: false,
            },
        );
        let selected = outcome
            .selected()
            .unwrap_or_else(|| panic!("quantized selection failed: {outcome:?}"));
        let geometry =
            selected_generation_memory_geometry(&plan, selected.preparation().execution()).unwrap();
        let topology = geometry.execution_topology.unwrap();
        let TokenMixerTopology::Attention { projections, .. } = &topology.layers[0].mixer else {
            panic!("attention");
        };
        assert!(projections
            .iter()
            .all(|p| matches!(p.format, eredu_checkpoint::LinearFormat::Affine(_))));
        let wire = serde_json::to_value(&topology).unwrap();
        assert_eq!(
            serde_json::from_value::<TextExecutionTopology>(wire).unwrap(),
            topology
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
    fn lfm2_dense_workspace_uses_normalized_ffn_and_hybrid_schedule() {
        use eredu_core::ModelConfigurationResolver;
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/lfm2/released-config.json"))
                .unwrap();
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let geometry = generation_memory_geometry(resolved.architecture_plan()).unwrap();
        let topology = geometry.execution_topology.unwrap();
        assert_eq!(topology.hidden_size, 1024);
        assert_eq!(topology.layers.len(), 16);
        assert_eq!(
            topology
                .layers
                .iter()
                .filter(|l| matches!(
                    l.mixer,
                    TokenMixerTopology::GatedConvolution {
                        channels: 1024,
                        kernel: 3,
                        ..
                    }
                ))
                .count(),
            10
        );
        assert_eq!(
            topology
                .layers
                .iter()
                .filter(|l| matches!(
                    l.mixer,
                    TokenMixerTopology::Attention {
                        input_scores: true,
                        ..
                    }
                ))
                .count(),
            6
        );
        for layer in &topology.layers {
            let FeedForwardTopology::Gated {
                intermediate_size, ..
            } = layer.feed_forward
            else {
                panic!("dense block");
            };
            assert_eq!(intermediate_size, 4608);
        }
        let mut args = crate::lfm2::model_args_from_config_value(&config).unwrap();
        let mut policies: Vec<_> = args.layer_schedule.iter().copied().collect();
        for policy in &mut policies {
            policy.operator = crate::lfm2::OperatorPolicy::CausalConvolution;
        }
        args.layer_schedule =
            eredu_core::LayerSchedule::new(policies.len(), policies.clone()).unwrap();
        let conv = crate::lfm2::execution_topology(&args).unwrap();
        assert!(conv
            .layers
            .iter()
            .all(|l| matches!(l.mixer, TokenMixerTopology::GatedConvolution { .. })));
    }

    #[test]
    fn shared_decoder_topology_covers_previously_unmodeled_gemma2() {
        use eredu_core::ModelConfigurationResolver;
        let config = serde_json::json!({"model_type":"gemma2", "vocab_size":32, "hidden_size":16, "intermediate_size":32, "num_hidden_layers":2, "num_attention_heads":4, "num_key_value_heads":2, "head_dim":4, "max_position_embeddings":64, "sliding_window":16, "query_pre_attn_scalar":4, "attn_logit_softcapping":50.0, "final_logit_softcapping":30.0});
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let topology = generation_memory_geometry(resolved.architecture_plan())
            .unwrap()
            .execution_topology
            .unwrap();
        assert_eq!(topology.layers.len(), 2);
        assert!(topology
            .layers
            .iter()
            .all(|l| matches!(l.mixer, TokenMixerTopology::Attention { softcap: true, .. })));
        assert!(topology.layers.iter().all(|l| l.normalization_count == 4));
    }
}
