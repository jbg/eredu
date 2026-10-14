//! Standard prepared construction through independently owned media/text groups.
use super::*;

#[test]
fn pixtral_standard_prepared_partitions_keep_image_context_across_transient_invocations() {
    pixtral_prepared_partition_equations(false);
}

#[test]
fn pixtral_prepared_draft_adapter_keeps_media_context_and_exact_rank_collective_schedule() {
    pixtral_prepared_partition_equations(true);
}

fn pixtral_prepared_partition_equations(with_adapter: bool) {
    use eredu_architectures::composite_execution::CompositeArchitecture;
    let fixture: serde_json::Value =
        serde_json::from_str(eredu_evaluation::nemotron_labs_diffusion::PIXTRAL_REFERENCE_JSON)
            .unwrap();
    let mut config_json = fixture["config"].clone();
    config_json["num_attention_heads"] = 4.into();
    config_json["num_key_value_heads"] = 2.into();
    config_json["block_size"] = 3.into();
    // Joint placement gives the first stage two vision layers; this differs
    // from balancing the tower independently across the five stages.
    config_json["vision_config"]["num_hidden_layers"] = 3.into();
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
    let adapter = with_adapter.then(|| {
        super::super::nemotron_diffusion::prepared_partitions::draft_adapter_fixture(
            &config,
            &mut values,
        )
    });
    if let Some((_, adapter_config)) = &adapter {
        config = config.with_draft_adapter(adapter_config.clone());
    }

    let description =
        eredu_architectures::nemotron_labs_diffusion::parallel::composite_parameter_description(
            &config,
        )
        .unwrap();
    let inputs = [
        prepared_prompt(&fixture, false),
        prepared_text(&[13]),
        prepared_text(&[14, 0, 0]),
        prepared_text(&[14, 15, 0]),
        prepared_text(&[14, 15]),
        prepared_text(&[16, 0, 0]),
        prepared_text(&[16, 17, 0]),
        prepared_text(&[16]),
    ];
    let invocations = [
        AttentionInvocation::CAUSAL,
        AttentionInvocation::CAUSAL,
        AttentionInvocation::DIFFUSION,
        AttentionInvocation::DRAFT,
        AttentionInvocation::CAUSAL,
        AttentionInvocation::DRAFT,
        AttentionInvocation::DIFFUSION,
        AttentionInvocation::CAUSAL,
    ];
    let context = NumericContext::default();
    let mut reference =
        ResidentRuntime::new(Composite::new(config.clone(), &context).unwrap(), &context).unwrap();
    struct Bind<'a>(&'a prepared_adapter::ParameterBits);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            let name = metadata.id.as_str();
            let source = if name == "lm_head.weight" {
                "diffusion_head.weight".into()
            } else {
                name.replacen("model.", "encoder.", 1)
            };
            let (_, bits) = self
                .0
                .get(&source)
                .or_else(|| self.0.get(name))
                .unwrap_or_else(|| panic!("missing admitted source {source}"));
            assert_eq!(bits.len(), value.data.len());
            *value = NumericTensor::new(
                value.shape.clone(),
                bits.iter().map(|&value| f32::from_bits(value)).collect(),
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
    let mut cache = composite_state(&config);
    let mut frontier = 0;
    let expected = inputs
        .iter()
        .zip(invocations)
        .map(|(input, invocation)| {
            let admitted =
                <Composite as CompositeArchitecture<NumericBackend, State>>::admit_prepared_input(
                    &config,
                    input,
                    &NumericInputInspector,
                )
                .unwrap();
            if invocation == AttentionInvocation::CAUSAL {
                frontier += admitted.decoder_positions() as i32;
            }
            let paired = PreparedCompositeInput::new(input, &admitted)
                .unwrap()
                .with_attention_invocation(invocation);
            let output = reference.forward(paired, &mut cache, &context).unwrap();
            assert_eq!(cache.layer(0).unwrap().offset, frontier);
            (output, frontier)
        })
        .collect::<Vec<_>>();
    for (tp, pp) in [(2, 1), (1, 5), (2, 5)] {
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
                        let (inspection, plan, description, inputs, expected) =
                            (&inspection, &plan, &description, &inputs, &expected);
                        let world = Arc::clone(&world);
                        // Generic prepared construction needs the same bounded worker
                        // stack as the DiffusionGemma partition fixture on Linux.
                        std::thread::Builder::new()
                            .name(format!("pixtral-prepared-rank-{rank}"))
                            .stack_size(8 << 20)
                            .spawn_scoped(scope, move || {
                            let sources =
                                partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(
                                    inspection,
                                    plan,
                                    rank,
                                    std::time::Duration::from_secs(30),
                                    None,
                                    16,
                                )
                                .unwrap();
                            let layout = eredu_runtime::derive_partitioned_local_layout(
                                description,
                                ParallelRankTopology::new(topology, rank).unwrap(),
                            )
                            .unwrap();
                            let mut context = NumericContext::with_partition(layout, rank, world);
                            context.bind_checkpoint_values = true;
                            let mut executable =
                                partitioned_adapter::composite(sources, &context).unwrap();
                            inputs
                                .iter()
                                .zip(invocations)
                                .zip(expected)
                                .map(|((input, invocation), (_, frontier))| {
                                    let output = executable.attention(input, invocation).unwrap_or_else(|error| panic!("TP{tp}/PP{pp} rank {rank}, {invocation:?}: {error:?}"));
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
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for rank in results {
                for (actual, (expected, _)) in rank.iter().zip(&expected) {
                    assert_tensor_close(
                        actual,
                        expected,
                        "prepared Pixtral tower/projector/decoder ownership and retained image context",
                    );
                }
            }
        }
    }
}
