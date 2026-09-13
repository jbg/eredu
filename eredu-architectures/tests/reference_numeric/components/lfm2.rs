//! Exact prepared-source ownership under quantized mixed-state execution.
use super::*;

#[test]
fn lfm2_prepared_transforms_retain_source_layout_across_residency_and_parallelism() {
    use eredu_architectures::lfm2 as family;
    for intermediate in [96, 128] {
        for routed in [false, true] {
            let config = serde_json::json!({
                "model_type": if routed { "lfm2_moe" } else { "lfm2" },
                "vocab_size": 64, "hidden_size": 64, "intermediate_size": intermediate,
                "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "max_position_embeddings": 32,
                "layer_types": ["conv", "full_attention"], "conv_L_cache": 3,
                "block_multiple_of": 32, "block_ffn_dim_multiplier": 1.0,
                "block_auto_adjust_ff_dim": false, "num_dense_layers": if routed { 1 } else { 0 },
                "moe_intermediate_size": intermediate,
                "num_experts": if routed { 2 } else { 0 }, "num_experts_per_tok": 1,
                "tie_word_embeddings": false
            });
            let (artifact, _) =
                prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                    let seed = name
                        .bytes()
                        .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(b.into()));
                    Some(NumericTensor::new(
                        shape.to_vec(),
                        (0..shape.iter().product::<i32>() as usize)
                            .map(|index| {
                                let delta = ((index * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                                if name.contains("norm") && name.ends_with("weight") {
                                    1.0 + delta * 0.003
                                } else if name.ends_with("A_log") {
                                    -0.5 + delta * 0.01
                                } else {
                                    delta * 0.01
                                }
                            })
                            .collect(),
                    ))
                });
            let inspection =
                eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
            let args = family::model_args_from_config_value(&config).unwrap();
            let parameters = family::LayeredModel::<NumericBackend>::new(
                args.clone(),
                &NumericContext::default(),
            )
            .unwrap()
            .parameter_description(&NumericContext::default())
            .unwrap();
            let mut topologies = vec![(2, 1, 1), (1, 2, 1), (2, 2, 1)];
            if routed {
                topologies.extend([(1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2)]);
            }
            let mut reference: Option<Vec<NumericTensor>> = None;
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
                    let plan =
                        prepared_adapter::plan(Some(eredu_core::QuantizationRequest::Affine {
                            group_size: 32,
                            bits: 4,
                        }))
                        .with_topology(topology)
                        .with_residency(residency);
                    let world = Arc::new(NumericPartitionWorld::default());
                    let results = std::thread::scope(|scope| {
                        let handles = (0..topology.world_size()).map(|rank| {
                        let (inspection, plan, args, parameters) = (&inspection, &plan, &args, &parameters);
                        let world = world.clone();
                        scope.spawn(move || {
                            let sources = partitioned_adapter::prepare_plan(inspection, plan, rank, std::time::Duration::from_secs(30)).unwrap();
                            let selected = sources.selected().execution().text_realization();
                            assert!(selected.parameters().iter().any(|p| matches!(p.lowering(), eredu_runtime::WeightLoweringKind::Transform | eredu_runtime::WeightLoweringKind::DerivedTransform)));
                            let target_args = family::with_checkpoint_formats(args, selected.parameters().iter().filter_map(|p| p.executable().weight_quantization().map(|format| (p.name().to_owned(), format))).collect()).unwrap();
                            let target = family::LayeredModel::<NumericBackend>::new(target_args, &NumericContext::default()).unwrap();
                            let target_parameters = target.parameter_description(&NumericContext::default()).unwrap();
                            let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                            let source = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters, rank_topology).unwrap();
                            let encoded = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(&target_parameters, rank_topology).unwrap();
                            // Numeric operators retain unpacked values, with the same
                            // semantic cuts selected for the encoded target.
                            let mut layout = eredu_runtime::derive_transform_source_layout(&source, &encoded).unwrap();
                            if intermediate == 96 && tp == 2 {
                                let ordinary = source.tensor("model.layers.0.feed_forward.w2.weight").unwrap();
                                let mapped = layout.tensor("model.layers.0.feed_forward.w2.weight").unwrap();
                                assert_eq!(ordinary.local_shape()[1], 48);
                                assert_eq!(mapped.local_shape()[1], if rank_topology.tensor_parallel_rank() == 0 { 64 } else { 32 });
                            }
                            for (name, companion) in encoded.tensors() { if !layout.contains(name) { layout.insert(name.to_owned(), companion.clone()); } }
                            let mut context = NumericContext::with_partition(layout, rank, world);
                            context.bind_checkpoint_values = true;
                            let mut executable = if routed {
                                partitioned_adapter::routed(sources, &context, Arc::new(AtomicUsize::new(0)), None)
                            } else { partitioned_adapter::dense(sources, &context) }.unwrap();
                            [&[1_usize, 3, 2][..], &[4][..], &[5][..]].into_iter().enumerate().map(|(step, tokens)| {
                                executable.forward_observed(&NumericTensor::token_ids(tokens), step == 0, &mut Components::strict()).unwrap()
                            }).collect::<Vec<_>>()
                        })
                    }).collect::<Vec<_>>();
                        handles
                            .into_iter()
                            .map(|handle| handle.join().unwrap())
                            .collect::<Vec<_>>()
                    });
                    if reference.is_none() {
                        reference = Some(results[0].clone());
                    }
                    for result in results {
                        for (actual, expected) in result.iter().zip(reference.as_ref().unwrap()) {
                            assert_tensor_close(
                                actual,
                                expected,
                                "LFM2 exact transform-source placement and cached execution",
                            );
                        }
                    }
                }
            }
        }
    }
}
