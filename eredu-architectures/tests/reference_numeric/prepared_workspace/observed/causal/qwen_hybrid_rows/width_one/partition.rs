//! Actual rank-local consumers of the same history-free layout and mixer.
use super::*;

#[test]
fn width_one_direct_parallel_consumers_preserve_local_recurrence_and_kv() {
    for kind in ["qwen3_next", "qwen3_5_text"] {
        let mut value = configuration(kind, false, false);
        value["linear_conv_kernel_dim"] = 1.into();
        let config = hybrid::model_args_from_config_value(&value).unwrap().text;
        let context = NumericContext::default();
        let architecture = HybridModel::new(config.clone(), &context).unwrap();
        let parameters = architecture.parameter_description(&context).unwrap();
        let groups = parameters
            .groups()
            .iter()
            .map(|group| group.group().clone())
            .collect::<Vec<_>>();
        let group = NumericParallelGroup::new(2);
        std::thread::scope(|scope| {
            let workers = (0..2)
                .map(|rank| {
                    let config = &config;
                    let groups = &groups;
                    let group = Arc::clone(&group);
                    scope.spawn(move || {
                        let layout = numeric_local_layout(groups, 2, rank).unwrap();
                        let context = NumericContext::with_local_layout(layout.clone());
                        let geometry = hybrid::local_geometry(config, &layout).unwrap();
                        let state_layout = geometry.state_layout().clone();
                        let architecture =
                            HybridModel::new_parallel(config.clone(), geometry, &context).unwrap();
                        let mut model = ResidentRuntime::new(architecture, &context).unwrap();
                        load_recurrent_parameters(&mut model, 4);
                        let (architecture, units) = model.into_parts();
                        let mut model =
                            LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                        let parallel = NumericParallelContext::new(rank, group);
                        let mut full = HybridState::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                        let mut forward = |state: &mut HybridState, ids: &[usize]| {
                            model
                                .forward_parallel(
                                    hybrid::EmbeddedInput::target(
                                        &NumericTensor::token_ids(ids),
                                        None,
                                    ),
                                    state,
                                    &parallel,
                                    &context,
                                )
                                .unwrap()
                        };
                        forward(&mut full, &[4, 2]);
                        let prefix = full.clone();
                        let expected = forward(&mut full, &[1, 3, 5, 2, 6]);
                        let mut chunked = prefix.clone();
                        let mut consumed = Vec::new();
                        let mut outputs = Vec::new();
                        for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
                            outputs.push(forward(&mut chunked, ids));
                            consumed.extend_from_slice(ids);
                            let mut reference = prefix.clone();
                            forward(&mut reference, &consumed);
                            assert_hybrid_state(&chunked, &reference);
                        }
                        assert_tensor_close(
                            &NumericTensor::concatenate(&outputs, 1, &context).unwrap(),
                            &expected,
                            "parallel width-one scores",
                        );
                        assert_hybrid_state(&chunked, &full);
                        for layer in full.as_ref() {
                            assert!(!layer
                                .fixed
                                .contains_key(&StateTensorRole::Convolution { slot: 0 }));
                            if let Some(Some(value)) = layer.fixed.get(&StateTensorRole::Recurrent)
                            {
                                assert!(value.data.iter().any(|v| v.abs() > 1e-9));
                            }
                        }
                        for id in [1, 3, 2] {
                            let a = forward(&mut full, &[id]);
                            let b = forward(&mut chunked, &[id]);
                            assert_tensor_close(&a, &b, "parallel continued decode");
                            assert_hybrid_state(&full, &chunked);
                        }
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
    }
}

fn parameters(name: &str, shape: &[i32]) -> Option<NumericTensor> {
    let (base, step) = recurrent_parameter_pattern(name)?;
    let count: usize = shape.iter().map(|n| usize::try_from(*n).unwrap()).product();
    Some(NumericTensor::new(
        shape.to_vec(),
        (0..count).map(|i| base + step * (i % 7) as f32).collect(),
    ))
}

#[test]
fn width_one_conditional_prepared_partitions_preserve_owned_state_across_all_residencies() {
    for routed in [false, true] {
        let mut config = conditional_qwen_partition_config(routed);
        config["text_config"]["linear_conv_kernel_dim"] = 1.into();
        let (artifact, values) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, parameters);
        assert!(values
            .values()
            .all(|(_, bits)| bits.iter().any(|v| f32::from_bits(*v).abs() > 1e-9)));
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let description = numeric_composite_parameter_description(&config);
        let mut topologies = vec![(2, 1, 1), (1, 2, 1), (2, 2, 1)];
        if routed {
            topologies.push((2, 2, 2));
        }
        for (tp, pp, ep) in topologies {
            let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
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
                let plan = prepared_adapter::plan(None)
                    .with_topology(topology)
                    .with_residency(residency);
                let world = Arc::new(NumericPartitionWorld::default());
                std::thread::scope(|scope| {
                    let workers = (0..topology.world_size()).map(|rank| {
                        let (inspection, description, plan) = (&inspection, &description, &plan);
                        let world = Arc::clone(&world);
                        scope.spawn(move || {
                            let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                            let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(description, rank_topology).unwrap();
                            let mut context = NumericContext::with_partition(layout, rank, world);
                            context.bind_checkpoint_values = true;
                            let make = || {
                                let sources = partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(inspection, plan, rank, std::time::Duration::from_secs(30), None, 8).unwrap();
                                partitioned_adapter::composite(sources, &context).unwrap()
                            };
                            let mut chunked = make();
                            chunked.forward(&numeric_text_prepared_input(&[4, 2]), true).unwrap();
                            let mut consumed = Vec::new();
                            let mut final_reference = None;
                            for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
                                let output = chunked.forward(&numeric_text_prepared_input(ids), true).unwrap();
                                consumed.extend_from_slice(ids);
                                // Fresh genuine sessions compare each frontier against one full
                                // continuation without rewriting cache state or reset counters.
                                let mut reference = make();
                                reference.forward(&numeric_text_prepared_input(&[4, 2]), true).unwrap();
                                let expected = reference.forward(&numeric_text_prepared_input(&consumed), true).unwrap();
                                assert_tensor_close(&output, &expected, "prepared conditional last-row output");
                                let actual = chunked.snapshot().unwrap();
                                assert_hybrid_state(&actual, &reference.snapshot().unwrap());
                                for layer in actual.as_ref() {
                                    assert!(!layer.fixed.contains_key(&StateTensorRole::Convolution { slot: 0 }));
                                    if let Some(Some(value)) = layer.fixed.get(&StateTensorRole::Recurrent) {
                                        assert!(value.data.iter().all(|v| v.is_finite()));
                                        assert!(value.data.iter().any(|v| v.abs() > 1e-9));
                                        assert_eq!(layer.position(), 2 + consumed.len() as i32);
                                    }
                                }
                                final_reference = Some(reference);
                            }
                            let mut reference = final_reference.unwrap();
                            for id in [1, 3, 2] {
                                let input = numeric_text_prepared_input(&[id]);
                                let a = chunked.forward(&input, false).unwrap();
                                let b = reference.forward(&input, false).unwrap();
                                assert_tensor_close(&a, &b, "prepared conditional decode");
                                assert_hybrid_state(&chunked.snapshot().unwrap(), &reference.snapshot().unwrap());
                            }
                        })
                    }).collect::<Vec<_>>();
                    for worker in workers {
                        worker.join().unwrap();
                    }
                });
            }
        }
    }
}
