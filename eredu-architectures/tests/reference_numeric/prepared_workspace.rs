use super::*;
use eredu_core::{InferenceGeometry, OutputDemand, WorkspaceBound};
use eredu_nn::workspace::*;
use eredu_runtime::{working_memory::*, ArchitectureStateFactory};
use std::num::NonZeroU32;

#[path = "prepared_workspace/layerwise.rs"]
mod layerwise;

#[path = "prepared_workspace/destinations.rs"]
mod destinations;

#[derive(Debug, Clone, Default)]
struct Facts {
    operations: Arc<Mutex<Vec<WorkspaceOperation>>>,
    omit_host: bool,
    omit_copies: bool,
}
impl WorkspaceMechanisms for Facts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::memory_fixture::topology())
    }
    fn output_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory_fixture::placement())
    }
    fn scratch_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory_fixture::placement())
    }

    fn output_representation(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<WorkspaceRepresentation> {
        // This mock allocates every floating output as contiguous F32.
        (operation.outputs.get(output)?.dtype() == WorkspaceDtype::Float32).then_some(
            WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true),
        )
    }

    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        self.operations.lock().unwrap().push(operation.clone());
        if self.omit_copies && matches!(operation.kind, WorkspaceOperationKind::DeepCopy) {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "test mechanism allocates every output at its logical size".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok((!self.omit_host).then(|| WorkspaceHostBound {
            bytes: 0,
            assumptions: "test mechanism has no host workspace".into(),
        }))
    }
}
fn geometry(chunk: u64, output: OutputDemand) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 2,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 3,
        prefill_chunk_positions: chunk,
        output,
    }
}

#[test]
fn prepared_routed_workspace_shares_spans_across_gated_relu2_and_pooling_profiles() {
    let cases = [
        (config("qwen3_moe", false), false),
        (
            serde_json::json!({
                "model_type":"nemotron_h", "vocab_size":16, "hidden_size":8,
                "intermediate_size":12, "num_hidden_layers":2, "hybrid_override_pattern":"ME",
                "num_attention_heads":2, "num_key_value_heads":1, "head_dim":4,
                "mamba_num_heads":2, "n_groups":1, "mamba_head_dim":4, "ssm_state_size":3,
                "conv_kernel":3, "chunk_size":2, "n_routed_experts":4, "n_shared_experts":1,
                "moe_intermediate_size":6, "moe_shared_expert_intermediate_size":6,
                "num_experts_per_tok":2, "n_group":2, "topk_group":1,
                "num_nextn_predict_layers":0, "tie_word_embeddings":false
            }),
            false,
        ),
        (tiny_v4_config(), true),
    ];
    for (config, pooling) in cases {
        let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        for addressable in [false, true] {
            let plan = prepared_adapter::plan(None).with_expert_cache(addressable.then(|| {
                eredu_core::ExpertCachePlan::new(
                    Some(16 << 20),
                    Some(32 << 20),
                    1 << 20,
                    1 << 19,
                    eredu_core::residency::CacheEvictionPolicy::LeastRecentlyUsed,
                )
            }));
            let sources = prepared_adapter::prepare(
                &inspection,
                &plan,
                &prepared_adapter::NumericPreparationProvider { addressable },
            )
            .unwrap();
            let before = sources.target().source_diagnostics().unwrap();
            let input_positions = if pooling { 133 } else { 9 };
            for chunk in [3, input_positions - 1, input_positions] {
                for output in [
                    OutputDemand::StateOnly,
                    OutputDemand::LastPosition,
                    OutputDemand::Sequence,
                ] {
                    let facts = Facts::default();
                    let context = WorkspaceContext::new(facts.clone());
                    let layout = sources.selected().text_realization().state().layout();
                    let state = if pooling {
                        WorkspacePoolingStateFactory::new(NonZeroU32::new(2).unwrap(), &context)
                            .unwrap()
                            .realize_resident(layout)
                            .unwrap()
                    } else {
                        state(&sources, &context)
                    };
                    let geometry = InferenceGeometry {
                        input_positions,
                        ..geometry(chunk, output)
                    };
                    let report = sources
                        .inference_blueprint()
                        .quote_replicated_resident_text(geometry, &state, &context)
                        .unwrap_or_else(|error| {
                            panic!(
                                "{} addressable={addressable} {geometry:?}: {error}",
                                config["model_type"]
                            )
                        });
                    assert_eq!(report.geometry(), geometry);
                    assert!(report.transient().bytes().is_some());
                    assert!(report.first_gap().is_none());
                    assert_eq!(
                        facts.operations.lock().unwrap().iter().any(|operation| {
                            matches!(operation.kind, WorkspaceOperationKind::AddressableRegion(_))
                        }),
                        addressable,
                    );
                    for lane in state.as_ref() {
                        assert_eq!(lane.position(), 0);
                    }
                    assert!(facts.operations.lock().unwrap().iter().all(|operation| {
                        !matches!(operation.kind, WorkspaceOperationKind::DeepCopy)
                    }));
                }
            }
            let after = sources.target().source_diagnostics().unwrap();
            assert_eq!(before.physical_reads, after.physical_reads);
            assert_eq!(before.physical_read_bytes, after.physical_read_bytes);
            assert_eq!(before.payload_shard_paths, after.payload_shard_paths);
        }
    }
}

#[test]
fn prepared_composite_text_workspace_uses_shared_spans_and_exact_ingress_admission() {
    for config in [
        dense_muse_partition_fixture(),
        dense_inkling_partition_fixture(),
        conditional_qwen_partition_config(false),
        qwen_vl_partition_config(false),
        routed_muse_partition_fixture(),
        routed_inkling_partition_fixture(),
        conditional_qwen_partition_config(true),
        qwen_vl_partition_config(true),
    ] {
        let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let sources = prepared_adapter::prepare(
            &inspection,
            &prepared_adapter::plan(None),
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        let before = sources.target().source_diagnostics().unwrap();
        for chunk in [1, 3, 5] {
            for demand in [
                OutputDemand::StateOnly,
                OutputDemand::LastPosition,
                OutputDemand::Sequence,
            ] {
                let facts = Facts::default();
                let context = WorkspaceContext::new(facts.clone());
                let state = WorkspaceResidentStateFactory::new(
                    NonZeroU32::new(1).unwrap(),
                    NonZeroU32::new(256).unwrap(),
                    &context,
                )
                .unwrap()
                .realize(sources.selected().text_realization().state().layout())
                .unwrap();
                let g = InferenceGeometry {
                    batch_size: 1,
                    ..geometry(chunk, demand)
                };
                let report = sources
                    .inference_blueprint()
                    .quote_replicated_resident_text(g, &state, &context)
                    .unwrap_or_else(|error| panic!("{} {g:?}: {error}", config["model_type"]));
                assert_eq!(report.completed_spans(), 5u64.div_ceil(chunk) + 3);
                assert!(matches!(report.transient(), WorkspaceBound::Bounded { .. }));
                assert!(report.retained_peak_bytes().is_some());
                assert!(state.as_ref().iter().all(|lane| lane.position() == 0));
            }
        }
        let after = sources.target().source_diagnostics().unwrap();
        assert_eq!(before.physical_reads, after.physical_reads);
        assert_eq!(before.physical_read_bytes, after.physical_read_bytes);
        assert_eq!(before.payload_shard_paths, after.payload_shard_paths);
    }
}

#[test]
fn prepared_sampling_workspace_uses_selected_score_geometry_and_shared_policy() {
    let configs = [
        config("llama", true),
        heterogeneous_replicated_configs().remove(0),
    ];
    for mut config in configs {
        config["vocab_size"] = 37.into();
        let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let sources = prepared_adapter::prepare(
            &inspection,
            &prepared_adapter::plan(None),
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        let before = sources.target().source_diagnostics().unwrap();
        for strategy in 0..3 {
            for chunk in [1, 3, 5] {
                let facts = Facts::default();
                let context = WorkspaceContext::new(facts.clone());
                let state = WorkspaceResidentStateFactory::new(
                    NonZeroU32::new(1).unwrap(),
                    NonZeroU32::new(256).unwrap(),
                    &context,
                )
                .unwrap()
                .realize(sources.selected().text_realization().state().layout())
                .unwrap();
                let resolved = eredu_core::resolve_generation_config(
                    None,
                    eredu_core::GenerationConfigOverrides {
                        temperature: Some(if strategy == 0 { 0.0 } else { 0.7 }),
                        repetition_penalty: Some(1.2),
                        ..Default::default()
                    },
                )
                .unwrap();
                let request = eredu_core::TextGenerationConfig::new(resolved);
                let request = if strategy == 2 {
                    request.with_mirostat_v2(5.0, 0.1).unwrap()
                } else {
                    request
                };
                let geometry = InferenceGeometry {
                    batch_size: 1,
                    max_output_tokens: 9,
                    ..geometry(chunk, OutputDemand::LastPosition)
                };
                let report = sources
                    .inference_blueprint()
                    .quote_replicated_resident_text_with_sampling(
                        geometry,
                        &state,
                        &context,
                        request.clone(),
                        &eredu_core::TokenFilter::Allowed(vec![true, false, true]),
                    )
                    .unwrap();
                assert_eq!(report.equations.geometry(), geometry);
                assert_eq!(report.sampling.steps, 9);
                assert_eq!(report.sampling.final_history_bytes, 64);
                assert!(report.equations.transient().bytes().is_some());
                assert!(report.sampling.peak.bytes().is_some());
                assert!(state.as_ref().iter().all(|lane| lane.position() == 0));
                let operations = facts.operations.lock().unwrap();
                let sampled = operations
                    .iter()
                    .filter(|op| {
                        matches!(
                            op.kind,
                            WorkspaceOperationKind::Sampling(
                                WorkspaceSamplingOperation::Greedy
                                    | WorkspaceSamplingOperation::Categorical
                            )
                        )
                    })
                    .collect::<Vec<_>>();
                assert_eq!(sampled.len(), 9);
                assert!(sampled.iter().all(|op| op.inputs[0].shape() == [1, 1, 37]));
            }
        }
        assert_eq!(
            before.physical_reads,
            sources
                .target()
                .source_diagnostics()
                .unwrap()
                .physical_reads
        );
    }
}
#[derive(Debug)]
struct AliasedScoreFacts {
    base: Facts,
    score_capacity: u64,
}
impl WorkspaceMechanisms for AliasedScoreFacts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::memory_fixture::topology())
    }
    fn output_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory_fixture::placement())
    }
    fn scratch_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory_fixture::placement())
    }

    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let mut bound = self.base.operation_bound(operation)?.unwrap();
        if matches!(
            operation.kind,
            WorkspaceOperationKind::Sampling(
                WorkspaceSamplingOperation::Greedy | WorkspaceSamplingOperation::ReadToken
            )
        ) {
            bound.outputs[0] = WorkspaceOutputStorage::AliasInput(0);
        } else {
            for (layout, output) in operation.outputs.iter().zip(&mut bound.outputs) {
                if layout.shape() == [1, 1, 37] {
                    *output = WorkspaceOutputStorage::Allocate(self.score_capacity);
                }
            }
        }
        Ok(Some(bound))
    }
    fn host_workspace_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        self.base.host_workspace_bound(operation)
    }
}

#[test]
fn prepared_sampling_retains_actual_score_capacity_for_distinct_escaped_input_aliases() {
    let mut config = config("llama", true);
    config["vocab_size"] = 37.into();
    let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let request = eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.0),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
                repetition_penalty: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let quote = |score_capacity| {
        let context = WorkspaceContext::new(AliasedScoreFacts {
            base: Facts::default(),
            score_capacity,
        });
        let state = WorkspaceResidentStateFactory::new(
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(256).unwrap(),
            &context,
        )
        .unwrap()
        .realize(sources.selected().text_realization().state().layout())
        .unwrap();
        sources
            .inference_blueprint()
            .quote_replicated_resident_text_with_sampling(
                InferenceGeometry {
                    batch_size: 1,
                    ..geometry(3, OutputDemand::LastPosition)
                },
                &state,
                &context,
                request.clone(),
                &eredu_core::TokenFilter::All,
            )
            .unwrap()
            .sampling
    };
    let packed = quote(37 * 4);
    let padded = quote(2048);
    assert_eq!(packed.steps, 3);
    assert_eq!(
        padded.tensor_peak_bytes.unwrap() - packed.tensor_peak_bytes.unwrap(),
        2 * (2048 - 37 * 4)
    );
    assert_eq!(padded.host_peak_bytes, packed.host_peak_bytes);
    assert_eq!(padded.first_gap, None);
}

fn state(
    sources: &eredu_architectures::prepared_sources::PreparedModelSources,
    context: &WorkspaceContext,
) -> DeviceState<WorkspaceBackend, WorkspaceResidentLayerState> {
    WorkspaceResidentStateFactory::new(
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(256).unwrap(),
        context,
    )
    .unwrap()
    .realize(sources.selected().text_realization().state().layout())
    .unwrap()
}

#[test]
fn prepared_workspace_uses_all_resident_state_profiles_and_retained_source_graph() {
    let mut configs = heterogeneous_replicated_configs();
    let mut attention = configs[0].clone();
    attention["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
    configs.push(attention);
    let mut fixed = configs[0].clone();
    fixed["layer_types"] = serde_json::json!(["conv", "conv"]);
    configs.push(fixed);
    let mut compressed = configs[1].clone();
    compressed["linear_attn_config"]["kda_layers"] = serde_json::json!([]);
    compressed["linear_attn_config"]["full_attn_layers"] = serde_json::json!([1, 2]);
    configs.push(compressed);
    let mut stateless = configs[2].clone();
    stateless["hybrid_override_pattern"] = "--".into();
    configs.push(stateless);
    configs.push(config("llama", false));
    configs.push(config("llama", true));
    let mut trajectories = 0;
    for mut config in configs {
        config["vocab_size"] = 37.into();
        let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        for residency in [
            eredu_core::ResidencyPlan::FullyResident,
            eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: None,
                host_budget_bytes: None,
            },
            eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 16 << 20,
                host_budget_bytes: 32 << 20,
                host_lookahead: 1,
                background_queue: 1,
            },
        ] {
            let sources = prepared_adapter::prepare(
                &inspection,
                &prepared_adapter::plan(None).with_residency(residency),
                &prepared_adapter::NumericPreparationProvider { addressable: false },
            )
            .unwrap();
            let clone = sources.clone();
            assert!(std::ptr::eq(sources.graph(), clone.graph()));
            let blueprint = sources.inference_blueprint();
            let before = sources.target().source_diagnostics().unwrap();
            for chunk in [1, 3, 5] {
                for demand in [
                    OutputDemand::StateOnly,
                    OutputDemand::LastPosition,
                    OutputDemand::Sequence,
                ] {
                    let facts = Facts::default();
                    let context = WorkspaceContext::new(facts.clone());
                    let state = state(&sources, &context);
                    let g = geometry(chunk, demand);
                    let report = blueprint
                        .quote_replicated_resident_text(g, &state, &context)
                        .unwrap_or_else(|error| panic!("{} {g:?}: {error}", config["model_type"]));
                    assert_eq!(report.geometry(), g);
                    assert_eq!(report.completed_spans(), 5u64.div_ceil(chunk) + 3);
                    assert!(matches!(report.transient(), WorkspaceBound::Bounded { .. }));
                    assert!(report.retained_peak_bytes().is_some());
                    assert!(state.as_ref().iter().all(|lane| lane.position() == 0));
                    let vocabulary = config["vocab_size"].as_i64().unwrap() as i32;
                    let operations = facts.operations.lock().unwrap();
                    let compressed_layers = state
                        .as_ref()
                        .iter()
                        .filter(|lane| matches!(lane, WorkspaceResidentLayerState::Compressed(_)))
                        .count() as u64;
                    assert_eq!(
                        operations
                            .iter()
                            .filter(|op| matches!(op.kind, WorkspaceOperationKind::DeepCopy))
                            .count() as u64,
                        8 * compressed_layers * (report.completed_spans() - 1),
                        "{} {g:?}: checkpoint and rollback copies on every populated span",
                        config["model_type"]
                    );
                    let projections = operations
                        .iter()
                        .filter(|op| {
                            matches!(op.kind, WorkspaceOperationKind::Projection(_))
                                && op.outputs[0].shape().last() == Some(&vocabulary)
                        })
                        .collect::<Vec<_>>();
                    let prompt_rows = match demand {
                        OutputDemand::StateOnly => 0,
                        OutputDemand::LastPosition => 1,
                        OutputDemand::Sequence => 5,
                    };
                    assert_eq!(
                        projections
                            .iter()
                            .map(|op| op.inputs[0].shape()[1] as u64)
                            .sum::<u64>(),
                        prompt_rows + 3
                    );
                    trajectories += 1;
                }
            }
            let after = sources.target().source_diagnostics().unwrap();
            assert_eq!(before.physical_reads, after.physical_reads);
            assert_eq!(before.physical_read_bytes, after.physical_read_bytes);
            assert_eq!(before.payload_shard_paths, after.payload_shard_paths);
        }
    }
    assert_eq!(trajectories, 297);
}

#[test]
fn prepared_workspace_keeps_selected_quantization_and_rejects_unknown_host_prices() {
    let mut config = config("llama", false);
    config["hidden_size"] = 32.into();
    config["intermediate_size"] = 64.into();
    config["num_attention_heads"] = 8.into();
    let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    for packed in [false, true] {
        let sources = prepared_adapter::prepare(
            &inspection,
            &prepared_adapter::plan(packed.then_some(eredu_core::QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            })),
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        for omit_host in [false, true] {
            let facts = Facts {
                omit_host,
                ..Default::default()
            };
            let context = WorkspaceContext::new(facts.clone());
            let state = state(&sources, &context);
            let report = sources
                .inference_blueprint()
                .quote_replicated_resident_text(
                    geometry(3, OutputDemand::LastPosition),
                    &state,
                    &context,
                )
                .unwrap();
            assert_eq!(
                matches!(report.transient(), WorkspaceBound::Unknown { .. }),
                omit_host
            );
            assert_eq!(report.first_gap().is_some(), omit_host);
            let operations = facts.operations.lock().unwrap();
            let head = operations
                .iter()
                .find(|op| {
                    matches!(op.kind, WorkspaceOperationKind::Projection(_))
                        && op.outputs[0].shape().last() == Some(&17)
                })
                .unwrap();
            assert_eq!(
                head.inputs[1].shape(),
                if packed { vec![17, 4] } else { vec![17, 32] }
            );
        }
    }
}

#[test]
fn prepared_workspace_keeps_cached_transaction_copy_gaps_explicit() {
    let mut config = heterogeneous_replicated_configs().remove(1);
    config["linear_attn_config"]["kda_layers"] = serde_json::json!([]);
    config["linear_attn_config"]["full_attn_layers"] = serde_json::json!([1, 2]);
    let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let before = sources.target().source_diagnostics().unwrap();
    for (chunk, cached_positions) in [1, 3, 5]
        .into_iter()
        .flat_map(|chunk| [0, 4].map(|cached| (chunk, cached)))
    {
        for omit_copies in [false, true] {
            let facts = Facts {
                omit_copies,
                ..Default::default()
            };
            let context = WorkspaceContext::new(facts.clone());
            let mut state = state(&sources, &context);
            if cached_positions > 0 {
                let layout = state.layout().clone();
                for (index, lane) in state.as_mut().iter_mut().enumerate() {
                    let eredu_core::cache::LayerCachePolicy::CompressedLatentRotary {
                        latent_dim,
                        rotary_dim,
                        ..
                    } = layout.layer(index).unwrap()
                    else {
                        panic!("fixture must contain only compressed state");
                    };
                    let WorkspaceResidentLayerState::Compressed(cache) = lane else {
                        unreachable!()
                    };
                    let storage = |width: u32| {
                        WorkspaceExistingStorage::new(Some(2 * 7 * u64::from(width) * 4), &context)
                    };
                    let latent = storage(latent_dim.get());
                    let rotary = storage(rotary_dim.get());
                    let pair = |count| eredu_nn::CompressedAttentionState {
                        latent: WorkspaceTensor::existing_with_storage(
                            WorkspaceLayout::new(
                                &[2, count, latent_dim.get() as i32],
                                WorkspaceDtype::Float32,
                            )
                            .unwrap(),
                            &latent,
                            &context,
                        )
                        .unwrap(),
                        rotary: WorkspaceTensor::existing_with_storage(
                            WorkspaceLayout::new(
                                &[2, count, rotary_dim.get() as i32],
                                WorkspaceDtype::Float32,
                            )
                            .unwrap(),
                            &rotary,
                            &context,
                        )
                        .unwrap(),
                    };
                    *cache = cache
                        .clone()
                        .with_existing_state(pair(7), pair(cached_positions as i32))
                        .unwrap();
                }
            }
            let report = sources
                .inference_blueprint()
                .quote_replicated_resident_text(
                    InferenceGeometry {
                        cached_positions,
                        ..geometry(chunk, OutputDemand::LastPosition)
                    },
                    &state,
                    &context,
                )
                .unwrap();
            assert_eq!(report.transient().bytes().is_none(), omit_copies);
            assert_eq!(report.first_gap().is_some(), omit_copies);
            // A complete successful-state estimate cannot substitute for missing
            // transient rollback costs. The borrowed starting state is unchanged.
            assert!(report.retained_peak_bytes().is_some());
            assert!(state
                .as_ref()
                .iter()
                .all(|lane| lane.position() == cached_positions as i32));
            assert!(state.as_ref().iter().all(|lane| {
                let WorkspaceResidentLayerState::Compressed(cache) = lane else {
                    return false;
                };
                cache.capacity() == if cached_positions == 0 { 0 } else { 7 }
            }));
            assert!(state
                .as_ref()
                .iter()
                .all(|lane| lane.retained_values().next().is_none() == (cached_positions == 0)));
            assert_eq!(
                facts
                    .operations
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|op| matches!(op.kind, WorkspaceOperationKind::DeepCopy))
                    .count() as u64,
                8 * state.as_ref().len() as u64
                    * (report.completed_spans() - u64::from(cached_positions == 0))
            );
        }
    }
    assert_eq!(
        before.physical_reads,
        sources
            .target()
            .source_diagnostics()
            .unwrap()
            .physical_reads
    );
}

#[test]
fn prepared_workspace_starts_from_exact_cached_storage_without_replaying_history() {
    let (artifact, _) = prepared_adapter::payload_fixture_config(&config("llama", false), 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let blueprint = sources.inference_blueprint();
    for capacity in [Some(2048), None] {
        let facts = Facts::default();
        let context = WorkspaceContext::new(facts.clone());
        let storage = WorkspaceExistingStorage::new(capacity, &context);
        let keys = WorkspaceTensor::existing_with_storage(
            WorkspaceLayout::new(&[2, 1, 4, 4], WorkspaceDtype::Float32).unwrap(),
            &storage,
            &context,
        )
        .unwrap();
        let factory =
            WorkspaceConcatStateFactory::new(NonZeroU32::new(2).unwrap(), &context).unwrap();
        let state = DeviceState::<WorkspaceBackend, _>::create(
            sources
                .selected()
                .text_realization()
                .state()
                .layout()
                .clone(),
            |index, policy| {
                factory
                    .project_layer(index, policy, 4, Some(keys.clone()), Some(keys.clone()), [])
                    .map(WorkspaceResidentLayerState::Ordinary)
            },
        )
        .unwrap();
        let g = InferenceGeometry {
            cached_positions: 4,
            ..geometry(3, OutputDemand::LastPosition)
        };
        let report = blueprint
            .quote_replicated_resident_text(g, &state, &context)
            .unwrap();
        assert_eq!(report.completed_spans(), 5);
        assert_eq!(
            matches!(report.transient(), WorkspaceBound::Unknown { .. }),
            capacity.is_none()
        );
        if let Some(capacity) = capacity {
            assert!(report.tensor_transient_peak_bytes().unwrap() >= capacity);
        }
        assert!(state.as_ref().iter().all(|lane| lane.position() == 4));
        assert!(blueprint
            .quote_replicated_resident_text(
                InferenceGeometry {
                    cached_positions: 5,
                    ..g
                },
                &state,
                &context
            )
            .is_err());
        assert!(blueprint
            .quote_replicated_resident_text(g, &state, &WorkspaceContext::new(Facts::default()))
            .is_err());
    }
}

#[path = "prepared_workspace/observed.rs"]
mod observed;

#[path = "prepared_workspace/external_assistant.rs"]
mod external_assistant;
