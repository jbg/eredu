//! Nonzero shared-backbone causal/cache and transient-block conformance.
use super::*;
use eredu_architectures::nemotron_labs_diffusion::{
    FamilyConfig,
    model::{TextInput, TextModel},
};
use eredu_core::decoding::AttentionInvocation;

#[path = "nemotron_diffusion/shared_backbone.rs"]
mod shared_backbone;
#[path = "nemotron_diffusion/prepared_partitions.rs"]
pub(super) mod prepared_partitions;

fn family() -> FamilyConfig {
    FamilyConfig::from_hf_json(
        br#"{
        "model_type":"nemotron_labs_diffusion", "hidden_size":6, "head_dim":4,
        "num_attention_heads":2, "num_key_value_heads":1, "num_hidden_layers":2,
        "intermediate_size":7, "vocab_size":11, "rms_norm_eps":0.00001,
        "block_size":3, "mask_token_id":0, "hidden_act":"silu", "dlm_paradigm":"bidirectional",
        "tie_word_embeddings":false, "attention_bias":false, "mlp_bias":false,
        "max_position_embeddings":64, "sliding_window":null,
        "rope_parameters":{"rope_type":"yarn","factor":2.0,"rope_theta":10000.0,
            "original_max_position_embeddings":4,"beta_fast":32.0,"beta_slow":1.0,
            "mscale":1.0,"mscale_all_dim":1.0,"llama_4_scaling_beta":0.1}
    }"#,
    )
    .unwrap()
}

type State = DeviceState<NumericBackend, NumericCache>;
fn state(config: &FamilyConfig) -> State {
    DeviceState::create(decoder::state_layout(config).unwrap(), |_, _| {
        Ok::<_, Error>(NumericCache::new(None))
    })
    .unwrap()
}
fn input(tokens: &NumericTensor, invocation: AttentionInvocation) -> TextInput<'_, NumericTensor> {
    TextInput {
        tokens,
        mask: None,
        invocation,
        feedback: None,
    }
}

fn parameter_storage(
    runtime: &ResidentRuntime<TextModel<NumericBackend>, NumericBackend, State>,
) -> BTreeMap<String, usize> {
    struct Storage(BTreeMap<String, usize>);
    impl<'a> ParameterVisitor<'a, NumericTensor> for Storage {
        fn visit(&mut self, metadata: ParameterMetadata, tensor: &'a NumericTensor) {
            assert!(
                self.0
                    .insert(metadata.id.to_string(), tensor.data.as_ptr() as usize)
                    .is_none()
            );
        }
    }
    let mut storage = Storage(BTreeMap::new());
    runtime
        .architecture()
        .static_modules()
        .visit_parameters(&mut storage);
    for unit in runtime.units().iter().flatten() {
        unit.visit_parameters(&mut storage);
    }
    storage.0
}

#[test]
fn nemotron_diffusion_reads_the_causal_prefix_without_mutation_or_base_reallocation() {
    let config = family();
    let context = NumericContext::default();
    let mut cache = state(&config);
    let mut reference_cache = state(&config);
    let mut runtime = ResidentRuntime::new(
        TextModel::<NumericBackend>::new(config.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut reference = ResidentRuntime::new(
        TextModel::<NumericBackend>::new(config.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let storage = parameter_storage(&runtime);
    let prompt = NumericTensor::token_ids(&[1, 4, 2, 3]);
    let ordinary = runtime
        .forward(
            input(&prompt, AttentionInvocation::CAUSAL),
            &mut cache,
            &context,
        )
        .unwrap();
    let expected = reference
        .forward(
            input(&prompt, AttentionInvocation::CAUSAL),
            &mut reference_cache,
            &context,
        )
        .unwrap();
    assert_tensor_close(&ordinary, &expected, "ordinary prefill");
    for token in [5, 6, 7] {
        let before = cache.clone();
        let canvas = NumericTensor::token_ids(&[token, 0, 0]);
        let scores = runtime
            .forward(
                input(&canvas, AttentionInvocation::DIFFUSION),
                &mut cache,
                &context,
            )
            .unwrap();
        assert_eq!(scores.shape, [1, 3, 11]);
        assert!(scores.data.iter().any(|x| x.abs() > 0.001));
        for layer in 0..2 {
            assert_eq!(
                cache.layer(layer).unwrap().offset,
                before.clone().layer(layer).unwrap().offset
            );
            assert_eq!(
                cache.layer(layer).unwrap().keys.as_ref().unwrap().data,
                before
                    .clone()
                    .layer(layer)
                    .unwrap()
                    .keys
                    .as_ref()
                    .unwrap()
                    .data
            );
            assert_eq!(
                cache.layer(layer).unwrap().values.as_ref().unwrap().data,
                before
                    .clone()
                    .layer(layer)
                    .unwrap()
                    .values
                    .as_ref()
                    .unwrap()
                    .data
            );
        }
        assert_eq!(parameter_storage(&runtime), storage);
        let committed = NumericTensor::token_ids(&[token]);
        let actual = runtime
            .forward(
                input(&committed, AttentionInvocation::CAUSAL),
                &mut cache,
                &context,
            )
            .unwrap();
        let expected = reference
            .forward(
                input(&committed, AttentionInvocation::CAUSAL),
                &mut reference_cache,
                &context,
            )
            .unwrap();
        assert_tensor_close(
            &actual,
            &expected,
            "causal decode after transient refinement",
        );
    }
}

#[test]
fn nemotron_causal_and_transient_execution_match_rebuilt_layer_residency() {
    let config = family();
    let context = NumericContext::default();
    let mut resident_cache = state(&config);
    let mut bounded_cache = state(&config);
    let mut resident = ResidentRuntime::new(
        TextModel::<NumericBackend>::new(config.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut bounded = LayerwiseRuntime::new(
        TextModel::<NumericBackend>::new(config, &context).unwrap(),
        RebuildingUnitPolicy::default(),
    );
    for (tokens, invocation) in [
        (vec![1, 4, 2], AttentionInvocation::CAUSAL),
        (vec![5], AttentionInvocation::CAUSAL),
        (vec![5, 0, 0], AttentionInvocation::DIFFUSION),
        (vec![5, 7, 0], AttentionInvocation::DRAFT),
        (vec![7, 8], AttentionInvocation::CAUSAL),
        (vec![8, 0, 0], AttentionInvocation::DIFFUSION),
    ] {
        let tokens = NumericTensor::token_ids(&tokens);
        let actual = bounded
            .forward(input(&tokens, invocation), &mut bounded_cache, &context)
            .unwrap();
        let expected = resident
            .forward(input(&tokens, invocation), &mut resident_cache, &context)
            .unwrap();
        assert_tensor_close(&actual, &expected, "bounded shared-backbone invocation");
        assert_eq!(
            bounded_cache.layer(0).unwrap().offset,
            resident_cache.layer(0).unwrap().offset
        );
    }
}

#[test]
fn nemotron_query_position_gain_matches_the_independent_attention_equation() {
    let context = NumericContext::default();
    let mut config: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../fixtures/nemotron_labs_diffusion/3B-config.json"
    ))
    .unwrap();
    config["hidden_size"] = 6.into();
    config["num_attention_heads"] = 2.into();
    config["num_key_value_heads"] = 1.into();
    config["head_dim"] = 4.into();
    config["num_hidden_layers"] = 1.into();
    config["intermediate_size"] = 7.into();
    config["vocab_size"] = 101.into();
    let family = FamilyConfig::from_hf_value(&config).unwrap();
    let mut module = decoder::Attention::<NumericBackend>::new(&family, 0, &context).unwrap();
    let hidden = NumericTensor::new([1, 1, 6], vec![0.2, -0.3, 0.7, 0.1, -0.2, 0.9]);
    let mut prefix = NumericCache::new(None);
    prefix.offset = 16384;
    prefix.keys = Some(NumericTensor::new(
        [1, 1, 2, 4],
        vec![0.3, -0.2, 0.8, 0.1, -0.5, 0.7, 0.2, 0.6],
    ));
    prefix.values = Some(NumericTensor::new(
        [1, 1, 2, 4],
        vec![0.5, 0.8, -0.7, 0.3, -0.2, 0.4, 0.9, 0.6],
    ));
    let mut ordinary = prefix.clone();
    let actual = module
        .forward(
            decoder::AttentionInput {
                hidden: &hidden,
                mask: None,
                cache: Some(&mut prefix),
                selected_positions: None,
                allow_sliding_prefill: false,
                rotary_position: None,
            },
            &context,
        )
        .unwrap();
    // Independently apply the published gain to the query weight rows. Linear
    // projection commutes with this scalar for this single canonical position.
    let gain = 1.0 + 0.1 * 2.0_f32.ln();
    module.query_position_scaling = None;
    if let decoder::AttentionInputProjection::Split { query, .. } = &mut module.input_projection {
        for weight in &mut query.weight.data {
            *weight *= gain;
        }
    } else {
        panic!("published split projections");
    }
    let expected = module
        .forward(
            decoder::AttentionInput {
                hidden: &hidden,
                mask: None,
                cache: Some(&mut ordinary),
                selected_positions: None,
                allow_sliding_prefill: false,
                rotary_position: None,
            },
            &context,
        )
        .unwrap();
    assert_tensor_close(&actual, &expected, "position-dependent query gain");
}

#[test]
fn nemotron_draft_adapter_is_invocation_local_and_preserves_base_storage() {
    use eredu_architectures::nemotron_labs_diffusion::adapter::DraftAdapterConfig;
    let config = family();
    let context = NumericContext::default();
    let mut json: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::DRAFT_ADAPTER_CONFIG_JSON,
    )
    .unwrap();
    json["r"] = 2.into();
    json["lora_alpha"] = 8.into();
    let adapter = DraftAdapterConfig::from_json(
        &serde_json::to_vec(&json).unwrap(),
        "nvidia/Nemotron-Labs-Diffusion-3B",
    )
    .unwrap();
    let model =
        TextModel::<NumericBackend>::new_with_draft_adapter(config.clone(), adapter, &context)
            .unwrap();
    // Each base layer and both low-rank matrices have one exact residency owner.
    let topology = model.parameter_description(&context).unwrap();
    let adapter_members: Vec<_> = topology
        .groups()
        .iter()
        .flat_map(|g| g.group().members())
        .filter(|m| m.target().starts_with("draft_adapter."))
        .collect();
    assert_eq!(adapter_members.len(), 4);
    let mut runtime = ResidentRuntime::new(model, &context).unwrap();
    let mut reference = ResidentRuntime::new(
        TextModel::<NumericBackend>::new(config.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    for layer in runtime.units_mut().iter_mut().flatten() {
        let adapter = layer.draft_adapter.as_mut().unwrap();
        adapter.down.weight.data.fill(0.17);
        adapter
            .up
            .weight
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = (i as f32 * 0.71).sin() * 0.2);
    }
    let storage = parameter_storage(&runtime);
    let mut cache = state(&config);
    let mut reference_cache = state(&config);
    let prompt = NumericTensor::token_ids(&[1, 3, 2]);
    assert_tensor_exact(
        &runtime
            .forward(
                input(&prompt, AttentionInvocation::CAUSAL),
                &mut cache,
                &context,
            )
            .unwrap(),
        &reference
            .forward(
                input(&prompt, AttentionInvocation::CAUSAL),
                &mut reference_cache,
                &context,
            )
            .unwrap(),
        "adapter is disabled for causal context",
    );
    let canvas = NumericTensor::token_ids(&[2, 0, 0]);
    let ordinary = reference
        .forward(
            input(&canvas, AttentionInvocation::DIFFUSION),
            &mut reference_cache,
            &context,
        )
        .unwrap();
    for _ in 0..3 {
        let drafted = runtime
            .forward(
                input(&canvas, AttentionInvocation::DRAFT),
                &mut cache,
                &context,
            )
            .unwrap();
        assert!(
            drafted
                .data
                .iter()
                .zip(&ordinary.data)
                .any(|(a, b)| (a - b).abs() > 1e-4)
        );
        assert_tensor_exact(
            &runtime
                .forward(
                    input(&canvas, AttentionInvocation::DIFFUSION),
                    &mut cache,
                    &context,
                )
                .unwrap(),
            &ordinary,
            "adapter is disabled for ordinary diffusion",
        );
        assert_eq!(parameter_storage(&runtime), storage);
    }
    let next = NumericTensor::token_ids(&[4]);
    assert_tensor_exact(
        &runtime
            .forward(
                input(&next, AttentionInvocation::CAUSAL),
                &mut cache,
                &context,
            )
            .unwrap(),
        &reference
            .forward(
                input(&next, AttentionInvocation::CAUSAL),
                &mut reference_cache,
                &context,
            )
            .unwrap(),
        "draft-only adapter leaves canonical verification unchanged",
    );
}

#[test]
fn low_rank_tp_reduces_rank_activations_before_the_replicated_up_projection() {
    use eredu_nn::{ParameterSpec, low_rank::LowRankProjection};
    let input = [0.2f32, -0.4, 0.7, 0.1];
    let down = [0.3f32, 0.8, -0.2, 0.5, -0.6, 0.4, 0.9, -0.1];
    let up = [0.2f32, -0.7, 0.6, 0.3, -0.4, 0.8];
    let rank: Vec<f64> = down
        .chunks_exact(4)
        .map(|w| {
            w.iter()
                .zip(input)
                .map(|(w, x)| f64::from(*w) * f64::from(x))
                .sum()
        })
        .collect();
    let expected = NumericTensor::new(
        [1, 1, 3],
        up.chunks_exact(2)
            .map(|w| {
                (w.iter()
                    .zip(&rank)
                    .map(|(w, x)| f64::from(*w) * x)
                    .sum::<f64>()
                    * 4.0) as f32
            })
            .collect(),
    );
    let world = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|shard| {
                let world = Arc::clone(&world);
                scope.spawn(move || {
                    let context = NumericContext::default();
                    let mut adapter = LowRankProjection::<NumericBackend>::new(
                        2,
                        3,
                        2,
                        4.0,
                        ParameterSpec::trainable("adapter.A").unwrap(),
                        ParameterSpec::trainable("adapter.B").unwrap(),
                        &context,
                    )
                    .unwrap();
                    adapter.down.weight.data = down
                        .chunks_exact(4)
                        .flat_map(|w| w[shard * 2..shard * 2 + 2].iter().copied())
                        .collect();
                    adapter.up.weight.data = up.to_vec();
                    let local =
                        NumericTensor::new([1, 1, 2], input[shard * 2..shard * 2 + 2].to_vec());
                    adapter
                        .forward(
                            &local,
                            Some(&NumericParallelContext::new(shard, world)),
                            &context,
                        )
                        .unwrap()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    for actual in outputs {
        assert_tensor_close(&actual, &expected, "independent low-rank TP equation");
    }
}

#[test]
fn nemotron_tp2_alternates_causal_diffusion_and_adapted_draft_on_one_base() {
    use eredu_architectures::decoder::PartitionedConfig;
    use eredu_architectures::nemotron_labs_diffusion::adapter::DraftAdapterConfig;
    let context = NumericContext::default();
    let mut source: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../fixtures/nemotron_labs_diffusion/3B-config.json"
    ))
    .unwrap();
    for (key, value) in [
        ("hidden_size", 6),
        ("num_hidden_layers", 2),
        ("num_attention_heads", 2),
        ("num_key_value_heads", 2),
        ("head_dim", 4),
        ("intermediate_size", 8),
        ("vocab_size", 101),
    ] {
        source[key] = value.into();
    }
    let config = FamilyConfig::from_hf_value(&source).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::DRAFT_ADAPTER_CONFIG_JSON,
    )
    .unwrap();
    json["r"] = 2.into();
    json["lora_alpha"] = 8.into();
    let adapter = DraftAdapterConfig::from_json(
        &serde_json::to_vec(&json).unwrap(),
        "nvidia/Nemotron-Labs-Diffusion-3B",
    )
    .unwrap();
    let model = TextModel::<NumericBackend>::new_with_draft_adapter(
        config.clone(),
        adapter.clone(),
        &context,
    )
    .unwrap();
    let groups: Vec<_> = model
        .parameter_description(&context)
        .unwrap()
        .groups()
        .iter()
        .map(|g| g.group().clone())
        .collect();
    let mut runtime = ResidentRuntime::new(model, &context).unwrap();
    let mut cache = state(&config);
    let invocations = [
        (vec![1, 2, 3], AttentionInvocation::CAUSAL),
        (vec![3, 100, 100], AttentionInvocation::DIFFUSION),
        (vec![3, 100, 100], AttentionInvocation::DRAFT),
        (vec![4], AttentionInvocation::CAUSAL),
        (vec![4, 100, 100], AttentionInvocation::DRAFT),
        (vec![5], AttentionInvocation::CAUSAL),
    ];
    let expected: Vec<_> = invocations
        .iter()
        .map(|(ids, invocation)| {
            runtime
                .forward(
                    input(&NumericTensor::token_ids(ids), *invocation),
                    &mut cache,
                    &context,
                )
                .unwrap()
        })
        .collect();
    let world = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let world = Arc::clone(&world);
                let config = config.clone();
                let adapter = adapter.clone();
                let groups = &groups;
                let invocations = &invocations;
                scope.spawn(move || {
                    let layout = numeric_local_layout(groups, 2, rank).unwrap();
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry =
                        decoder::local_geometry(&config, &layout, |config, index, layout| {
                            config.local_block_config(index, layout).map_err(|e| {
                                eredu_runtime::ParallelPlanError::InvalidGroup(e.to_string())
                            })
                        })
                        .unwrap();
                    let mut cache = State::create(geometry.state_layout().clone(), |_, _| {
                        Ok::<_, Error>(NumericCache::new(None))
                    })
                    .unwrap();
                    let model = TextModel::<NumericBackend>::new_parallel_with_draft_adapter(
                        config, geometry, adapter, &context,
                    )
                    .unwrap();
                    let units = (0..2)
                        .map(|index| model.construct_unit(index, &context).unwrap())
                        .collect();
                    let mut runtime = LayerwiseRuntime::<_, NumericBackend, State, _>::new(
                        model,
                        ResidentUnitWindow::new(units),
                    );
                    let parallel = NumericParallelContext::new(rank, world);
                    invocations
                        .iter()
                        .map(|(ids, invocation)| {
                            runtime
                                .forward_parallel(
                                    input(&NumericTensor::token_ids(ids), *invocation),
                                    &mut cache,
                                    &parallel,
                                    &context,
                                )
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    for rank in outputs {
        for (actual, expected) in rank.iter().zip(&expected) {
            assert_tensor_close(actual, expected, "shared-base adapted TP2 invocation");
        }
    }
}

#[test]
fn nemotron_pp2_and_tp2_pp2_transport_modes_frontiers_and_complete_canvases() {
    use eredu_architectures::nemotron_labs_diffusion::{
        adapter::DraftAdapterConfig,
        parallel,
        partitioned::{PartitionedTextModel, TextBoundarySchema},
    };
    use eredu_runtime::ArchitectureBoundary;
    let context = NumericContext::default();
    let mut source: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../fixtures/nemotron_labs_diffusion/3B-config.json"
    ))
    .unwrap();
    for (key, value) in [
        ("hidden_size", 6),
        ("num_hidden_layers", 2),
        ("num_attention_heads", 2),
        ("num_key_value_heads", 2),
        ("head_dim", 4),
        ("intermediate_size", 8),
        ("vocab_size", 101),
    ] {
        source[key] = value.into();
    }
    let mut json: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::DRAFT_ADAPTER_CONFIG_JSON,
    )
    .unwrap();
    json["r"] = 2.into();
    json["lora_alpha"] = 8.into();
    let config = FamilyConfig::from_hf_value(&source)
        .unwrap()
        .with_draft_adapter(
            DraftAdapterConfig::from_json(
                &serde_json::to_vec(&json).unwrap(),
                "nvidia/Nemotron-Labs-Diffusion-3B",
            )
            .unwrap(),
        );
    let description = parallel::parameter_description(&config).unwrap();
    let mut resident = ResidentRuntime::new(
        TextModel::<NumericBackend>::new(config.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut cache = state(&config);
    let invocations = [
        (vec![1, 2, 3], AttentionInvocation::CAUSAL),
        (vec![3, 100, 100], AttentionInvocation::DIFFUSION),
        (vec![3, 100, 100], AttentionInvocation::DRAFT),
        (vec![4], AttentionInvocation::CAUSAL),
        (vec![4, 100, 100], AttentionInvocation::DRAFT),
        (vec![4, 100, 100], AttentionInvocation::DIFFUSION),
        (vec![5], AttentionInvocation::CAUSAL),
    ];
    let expected: Vec<_> = invocations
        .iter()
        .enumerate()
        .map(|(step, (ids, invocation))| {
            let tokens = NumericTensor::token_ids(ids);
            let mask = (step == 1).then(|| {
                NumericTensor::new(
                    [3, 6],
                    vec![
                        -1e9, 0.0, 0.0, 0.0, 0.0, 0.0, -1e9, 0.0, 0.0, 0.0, 0.0, 0.0, -1e9, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                    ],
                )
            });
            resident
                .forward(
                    TextInput {
                        tokens: &tokens,
                        mask: mask.as_ref(),
                        invocation: *invocation,
                        feedback: None,
                    },
                    &mut cache,
                    &context,
                )
                .unwrap()
        })
        .collect();
    for tp in [1, 2] {
        let worlds = [NumericParallelGroup::new(tp), NumericParallelGroup::new(tp)];
        let outputs = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..tp)
                .map(|rank| {
                    let worlds = worlds.clone();
                    let config = config.clone();
                    let description = description.clone();
                    let invocations = &invocations;
                    scope.spawn(move || {
                        let make_stage = |stage| {
                            let topology = ParallelRankTopology::new(
                                ParallelTopology::new(tp, 2, 1, 1).unwrap(),
                                stage * tp + rank,
                            )
                            .unwrap();
                            let layout = eredu_runtime::derive_partitioned_local_layout(
                                &description,
                                topology,
                            )
                            .unwrap();
                            let context = NumericContext::with_local_layout(layout.clone());
                            let ownership = if stage == 0 {
                                PartitionOwnership::new(true, false, ["embedding"]).unwrap()
                            } else {
                                PartitionOwnership::new(false, true, ["norm", "output"]).unwrap()
                            };
                            let partition = parallel::partition(
                                &config,
                                &description,
                                &layout,
                                stage..stage + 1,
                                ownership,
                            )
                            .unwrap();
                            let model = PartitionedTextModel::<NumericBackend>::new(
                                config.clone(),
                                &description,
                                &partition,
                                &context,
                            )
                            .unwrap();
                            let unit = model.construct_unit(stage, &context).unwrap();
                            let state = State::create(
                                partition.state().unwrap().layout().clone(),
                                |_, _| Ok::<_, Error>(NumericCache::new(None)),
                            )
                            .unwrap();
                            let driver =
                                LayeredPartitionDriver::new(&partition, 0, stage..stage + 1)
                                    .unwrap();
                            (
                                model,
                                unit,
                                state,
                                driver,
                                NumericParallelContext::new(rank, Arc::clone(&worlds[stage])),
                                context,
                            )
                        };
                        let (
                            mut first_model,
                            mut first_unit,
                            mut first_state,
                            first_driver,
                            first_parallel,
                            first_context,
                        ) = make_stage(0);
                        let (
                            mut last_model,
                            mut last_unit,
                            mut last_state,
                            last_driver,
                            last_parallel,
                            last_context,
                        ) = make_stage(1);
                        let mut outputs = Vec::new();
                        for (step, (ids, invocation)) in invocations.iter().enumerate() {
                            let tokens = NumericTensor::token_ids(ids);
                            let mask = (step == 1).then(|| {
                                NumericTensor::new(
                                    [3, 6],
                                    vec![
                                        -1e9, 0.0, 0.0, 0.0, 0.0, 0.0, -1e9, 0.0, 0.0, 0.0, 0.0,
                                        0.0, -1e9, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    ],
                                )
                            });
                            let first_tp = (tp > 1).then_some(&first_parallel);
                            let last_tp = (tp > 1).then_some(&last_parallel);
                            let prepared = first_model
                                .prepare_partition_input(
                                    TextInput {
                                        tokens: &tokens,
                                        mask: mask.as_ref(),
                                        invocation: *invocation,
                                        feedback: None,
                                    },
                                    &mut first_state,
                                    first_tp,
                                    &first_context,
                                )
                                .unwrap();
                            let mut first = first_driver
                                .begin(
                                    &mut first_model,
                                    prepared.input,
                                    prepared.mask.as_ref(),
                                    &mut first_state,
                                    first_tp,
                                    &first_context,
                                )
                                .unwrap();
                            first.hidden = if tp > 1 {
                                first_model
                                    .forward_unit_parallel(
                                        0,
                                        0,
                                        &mut first_unit,
                                        &first.hidden,
                                        &mut first_state,
                                        &mut first.context,
                                        &first_parallel,
                                        &first_context,
                                    )
                                    .unwrap()
                            } else {
                                first_model
                                    .forward_unit(
                                        0,
                                        0,
                                        &mut first_unit,
                                        &first.hidden,
                                        &mut first_state,
                                        &mut first.context,
                                        &first_context,
                                    )
                                    .unwrap()
                            };
                            let LayeredPartitionOutput::Boundary { hidden, auxiliary } =
                                first_driver
                                    .finish(
                                        &mut first_model,
                                        &first.hidden,
                                        &mut first_state,
                                        &mut first.context,
                                        first_tp,
                                        &first_context,
                                    )
                                    .unwrap()
                            else {
                                panic!("first owner returns a full activation boundary");
                            };
                            let schema = TextBoundarySchema::new(6);
                            let wire = schema.encode(auxiliary).unwrap();
                            let auxiliary = schema
                                .decode(wire.into_iter().map(|v| v.into_parts().1).collect())
                                .unwrap();
                            let mut last = last_driver
                                .begin(
                                    &mut last_model,
                                    LayeredPartitionInput::Hidden { hidden, auxiliary },
                                    prepared.mask.as_ref(),
                                    &mut last_state,
                                    last_tp,
                                    &last_context,
                                )
                                .unwrap();
                            last.hidden = if tp > 1 {
                                last_model
                                    .forward_unit_parallel(
                                        0,
                                        1,
                                        &mut last_unit,
                                        &last.hidden,
                                        &mut last_state,
                                        &mut last.context,
                                        &last_parallel,
                                        &last_context,
                                    )
                                    .unwrap()
                            } else {
                                last_model
                                    .forward_unit(
                                        0,
                                        1,
                                        &mut last_unit,
                                        &last.hidden,
                                        &mut last_state,
                                        &mut last.context,
                                        &last_context,
                                    )
                                    .unwrap()
                            };
                            let LayeredPartitionOutput::Final { output, .. } = last_driver
                                .finish(
                                    &mut last_model,
                                    &last.hidden,
                                    &mut last_state,
                                    &mut last.context,
                                    last_tp,
                                    &last_context,
                                )
                                .unwrap()
                            else {
                                panic!("last owner returns complete position logits");
                            };
                            assert_eq!(
                                first_state.layer(0).unwrap().offset(),
                                last_state.layer(0).unwrap().offset()
                            );
                            outputs.push(output);
                        }
                        outputs
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        for rank in outputs {
            for (actual, expected) in rank.iter().zip(&expected) {
                assert_tensor_close(actual, expected, "shared-base PP/TP+PP invocation");
            }
        }
    }
}
