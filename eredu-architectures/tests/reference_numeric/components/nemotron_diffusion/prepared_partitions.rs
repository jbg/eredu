//! Exact cold selection, source preparation and total TP/PP construction.
use super::*;

#[test]
fn nemotron_standard_prepared_partitions_preserve_cached_and_complete_canvas_equations() {
    prepared_partition_equations(false);
}

#[test]
fn prepared_draft_adapter_is_scoped_and_sharded_without_duplicating_base_weights() {
    prepared_partition_equations(true);
}

fn prepared_partition_equations(with_adapter: bool) {
    let config_json = config_json();
    let (artifact, mut values) =
        prepared_adapter::payload_fixture_config_with(&config_json, 1.0, |name, shape| {
            let seed = name
                .bytes()
                .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(b.into()));
            Some(NumericTensor::new(
                shape.to_vec(),
                (0..shape.iter().product::<i32>() as usize)
                    .map(|index| {
                        let delta = ((index * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                        if name.contains("norm") {
                            1.0 + delta * 0.003
                        } else {
                            delta * 0.01
                        }
                    })
                    .collect(),
            ))
        });
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let mut config = FamilyConfig::from_hf_value(&config_json).unwrap();
    let adapter = with_adapter.then(|| draft_adapter_fixture(&config, &mut values));
    if let Some((_, adapter_config)) = &adapter {
        config = config.with_draft_adapter(adapter_config.clone());
    }

    let parameters =
        eredu_architectures::nemotron_labs_diffusion::parallel::parameter_description(&config)
            .unwrap();
    let calls = [
        (vec![1, 4, 2], AttentionInvocation::CAUSAL, 3),
        (vec![5, 0, 0], AttentionInvocation::DIFFUSION, 3),
        (vec![5, 7, 0], AttentionInvocation::DRAFT, 3),
        (vec![5, 7, 8], AttentionInvocation::CAUSAL, 6),
        (vec![2, 0, 0], AttentionInvocation::DRAFT, 6),
        (vec![2, 3, 0], AttentionInvocation::DIFFUSION, 6),
        (vec![2], AttentionInvocation::CAUSAL, 7),
        (vec![3, 0, 0], AttentionInvocation::DRAFT, 7),
        (vec![3, 4], AttentionInvocation::CAUSAL, 9),
    ];
    let context = NumericContext::default();
    let mut reference = ResidentRuntime::new(
        TextModel::<NumericBackend>::new(config.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    struct Bind<'a>(&'a prepared_adapter::ParameterBits);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            let name = metadata.id.as_str();
            let source = if name == "lm_head.weight" {
                "diffusion_head.weight".into()
            } else {
                name.replacen("model.", "encoder.", 1)
            };
            let (shape, bits) = self
                .0
                .get(&source)
                .or_else(|| self.0.get(name))
                .expect("exact released source alias");
            assert_eq!(shape, &value.shape);
            *value = NumericTensor::new(
                shape.clone(),
                bits.iter().map(|&bits| f32::from_bits(bits)).collect(),
            );
        }
    }
    reference
        .architecture_mut()
        .static_modules_mut()
        .visit_parameters_mut(&mut Bind(&values));
    for unit in reference.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut Bind(&values));
    }
    if with_adapter {
        let prompt = NumericTensor::token_ids(&[1, 4, 2]);
        let canvas = NumericTensor::token_ids(&[5, 0, 0]);
        let mut scoped_cache = state(&config);
        reference
            .forward(
                input(&prompt, AttentionInvocation::CAUSAL),
                &mut scoped_cache,
                &context,
            )
            .unwrap();
        let before = parameter_storage(&reference);
        let ordinary = reference
            .forward(
                input(&canvas, AttentionInvocation::DIFFUSION),
                &mut scoped_cache,
                &context,
            )
            .unwrap();
        let drafted = reference
            .forward(
                input(&canvas, AttentionInvocation::DRAFT),
                &mut scoped_cache,
                &context,
            )
            .unwrap();
        assert!(
            ordinary
                .data
                .iter()
                .zip(&drafted.data)
                .any(|(a, b)| (a - b).abs() > 1e-5),
            "nonzero adapter must affect drafting"
        );
        let after = reference
            .forward(
                input(&canvas, AttentionInvocation::DIFFUSION),
                &mut scoped_cache,
                &context,
            )
            .unwrap();
        assert_eq!(
            ordinary.data, after.data,
            "adapter state must never leak across invocations"
        );
        assert_eq!(parameter_storage(&reference), before);
        let causal_with_adapter = reference
            .forward(
                input(&canvas, AttentionInvocation::CAUSAL),
                &mut scoped_cache,
                &context,
            )
            .unwrap();
        let mut baseline = ResidentRuntime::new(
            TextModel::<NumericBackend>::new(
                FamilyConfig::from_hf_value(&config_json).unwrap(),
                &context,
            )
            .unwrap(),
            &context,
        )
        .unwrap();
        baseline
            .architecture_mut()
            .static_modules_mut()
            .visit_parameters_mut(&mut Bind(&values));
        for unit in baseline.units_mut().iter_mut().flatten() {
            unit.visit_parameters_mut(&mut Bind(&values));
        }
        let mut base_cache = state(&config);
        baseline
            .forward(
                input(&prompt, AttentionInvocation::CAUSAL),
                &mut base_cache,
                &context,
            )
            .unwrap();
        let base_causal = baseline
            .forward(
                input(&canvas, AttentionInvocation::CAUSAL),
                &mut base_cache,
                &context,
            )
            .unwrap();
        assert_eq!(causal_with_adapter.data, base_causal.data);
    }
    let mut cache = state(&config);
    let expected = calls
        .iter()
        .map(|(tokens, invocation, frontier)| {
            let value = reference
                .forward(
                    input(&NumericTensor::token_ids(tokens), *invocation),
                    &mut cache,
                    &context,
                )
                .unwrap();
            assert_eq!(cache.layer(0).unwrap().offset, *frontier);
            value
        })
        .collect::<Vec<_>>();
    for (tp, pp) in [(2, 1), (1, 2), (2, 2)] {
        let topology = ParallelTopology::new(tp, pp, 1, 1).unwrap();
        for residency in [
            eredu_core::ResidencyPlan::FullyResident,
            eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: Some(1 << 20),
                host_budget_bytes: Some(1 << 20),
            },
            eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 1 << 20,
                host_budget_bytes: 1 << 20,
                host_lookahead: 1,
                background_queue: 1,
            },
        ] {
            let mut plan = prepared_adapter::plan(None)
                .with_topology(topology)
                .with_residency(residency);
            if let Some((root, _)) = &adapter {
                plan = plan.with_draft_adapter(eredu_core::DraftAdapterSource::new(
                    root.path(),
                    "nvidia/Nemotron-Labs-Diffusion-3B",
                ));
            }

            let world = Arc::new(NumericPartitionWorld::default());
            let results = std::thread::scope(|scope| {
                let handles = (0..topology.world_size())
                    .map(|rank| {
                        let (inspection, plan, parameters, calls) =
                            (&inspection, &plan, &parameters, &calls);
                        let world = Arc::clone(&world);
                        scope.spawn(move || {
                            let sources = partitioned_adapter::prepare_plan(
                                inspection,
                                plan,
                                rank,
                                std::time::Duration::from_secs(30),
                            )
                            .unwrap();
                            let layout = eredu_runtime::derive_partitioned_local_layout(
                                parameters,
                                ParallelRankTopology::new(topology, rank).unwrap(),
                            )
                            .unwrap();
                            let mut context = NumericContext::with_partition(layout, rank, world);
                            context.bind_checkpoint_values = true;
                            let mut executable =
                                partitioned_adapter::dense(sources, &context).unwrap();
                            calls
                                .iter()
                                .map(|(tokens, invocation, frontier)| {
                                    let output = executable
                                        .attention(&NumericTensor::token_ids(tokens), *invocation)
                                        .unwrap();
                                    assert!(
                                        executable
                                            .positions()
                                            .unwrap()
                                            .iter()
                                            .all(|offset| offset == frontier)
                                    );
                                    output
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for rank in results {
                for (actual, expected) in rank.iter().zip(&expected) {
                    assert_tensor_close(
                        actual,
                        expected,
                        "standard prepared TP/PP cached and full-canvas invocation",
                    );
                }
            }
        }
    }
}

/// Nonzero independent PEFT source with published names and ordinary rank equations.
pub(crate) fn draft_adapter_fixture(
    config: &FamilyConfig,
    values: &mut prepared_adapter::ParameterBits,
) -> (
    tempfile::TempDir,
    eredu_architectures::nemotron_labs_diffusion::adapter::DraftAdapterConfig,
) {
    use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
    let root = tempfile::tempdir().unwrap();
    let mut json: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../fixtures/nemotron_labs_diffusion/3B-adapter-config.json"
    ))
    .unwrap();
    json["r"] = 2.into();
    json["lora_alpha"] = 4.into();
    let bytes = serde_json::to_vec(&json).unwrap();
    std::fs::write(root.path().join("adapter_config.json"), &bytes).unwrap();
    let adapter =
        eredu_architectures::nemotron_labs_diffusion::adapter::DraftAdapterConfig::from_json(
            &bytes,
            "nvidia/Nemotron-Labs-Diffusion-3B",
        )
        .unwrap();
    let schema =
        eredu_architectures::nemotron_labs_diffusion::adapter::safetensors_plan(config, &adapter)
            .unwrap();
    let tensors = schema
        .common_tensors
        .iter()
        .map(|constraint| {
            let elements = constraint.shape.iter().product::<usize>();
            let data = (0..elements)
                .map(|index| (((index * 11 + 7) % 31) as f32 - 15.0) * 0.03)
                .collect::<Vec<_>>();
            let shape = constraint.shape.iter().map(|&size| size as i32).collect();
            values.insert(
                constraint.key.clone(),
                (shape, data.iter().map(|x| x.to_bits()).collect()),
            );
            (
                constraint.aliases[0].clone(),
                constraint.shape.clone(),
                data.iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        tensors.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &root.path().join("adapter_model.safetensors"),
    )
    .unwrap();
    (root, adapter)
}

fn config_json() -> serde_json::Value {
    serde_json::json!({
        "model_type":"nemotron_labs_diffusion", "hidden_size":6, "head_dim":4,
        "num_attention_heads":4, "num_key_value_heads":2, "num_hidden_layers":2,
        "intermediate_size":8, "vocab_size":12, "rms_norm_eps":0.00001,
        "block_size":3, "mask_token_id":0, "hidden_act":"silu", "dlm_paradigm":"bidirectional",
        "tie_word_embeddings":false, "attention_bias":false, "mlp_bias":false,
        "max_position_embeddings":64, "sliding_window":null,
        "rope_parameters":{"rope_type":"yarn","factor":2.0,"rope_theta":10000.0,
            "original_max_position_embeddings":4,"beta_fast":32.0,"beta_slow":1.0,
            "mscale":1.0,"mscale_all_dim":1.0,"llama_4_scaling_beta":0.1}
    })
}

#[test]
fn cold_adapted_partition_reports_exact_local_parameters_and_kv_without_base_replicas() {
    let json = config_json();
    let (root, mut values) = prepared_adapter::payload_fixture_config_with(&json, 1.0, |_, _| None);
    let config = FamilyConfig::from_hf_value(&json).unwrap();
    let (adapter_root, _) = draft_adapter_fixture(&config, &mut values);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let topology = ParallelTopology::new(2, 2, 1, 1).unwrap();
    let plan = prepared_adapter::plan(None)
        .with_topology(topology)
        .with_draft_adapter(eredu_core::DraftAdapterSource::new(
            adapter_root.path(),
            "nvidia/Nemotron-Labs-Diffusion-3B",
        ));
    let mut total = 0;
    for rank in 0..4 {
        let parallel = eredu_runtime::ParallelLoadRequest::new(
            ParallelRankTopology::new(topology, rank).unwrap(),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            8,
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(2),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        )
        .unwrap();
        let request = eredu_runtime::NormalizedLoadRequest::from_execution_plan(
            &plan,
            eredu_runtime::ResidencyDiagnostics::new(false, false),
            Some(parallel),
        )
        .unwrap();
        let outcome = eredu_architectures::inspect_selected_model(
            inspection.clone(),
            &request,
            &prepared_adapter::NumericPreparationProvider { addressable: false },
            eredu_core::MediaFeatureAvailability {
                image: false,
                audio: false,
            },
        );
        let resources = &outcome.report().resources;
        let local = resources
            .selected_rank
            .value()
            .unwrap_or_else(|| panic!("rank resource projection missing: {:?}", outcome.report()));
        // Matrices shard once; norms and each rank's B matrix remain replicated.
        // First owner: one layer (256 scalars) + 36 embedding scalars.
        // Last owner: one layer + 36 readout scalars + six final RMS gains.
        assert_eq!(
            local.materialized_parameter_bytes,
            if rank < 2 { 1168 } else { 1192 }
        );
        assert_eq!(local.cache_bytes_per_position, 32);
        assert_eq!(resources.materialized_parameter_bytes.value(), Some(&4504));
        assert!(
            outcome
                .report()
                .architecture_descriptor
                .as_ref()
                .unwrap()
                .node("draft_adapter.layers.1.o_proj.down")
                .is_some()
        );
        total += local.materialized_parameter_bytes;
    }
    assert_eq!(total, 4720);
}
