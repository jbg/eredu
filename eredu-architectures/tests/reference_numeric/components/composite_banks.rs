//! Nonzero media and cached text parity through independently acquired expert banks.
use super::*;

struct Trace<'a> {
    values: BTreeMap<String, NumericTensor>,
    complete: &'a std::collections::BTreeSet<String>,
    provider_local: &'a std::collections::BTreeSet<String>,
}
impl ActivationObserver<NumericTensor, Error> for Trace<'_> {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        if self.complete.contains(path)
            || (!self.provider_local.contains(path)
                && (path.ends_with(".output")
                    || path.ends_with("attention.residual")
                    || path.ends_with("feed_forward.residual")))
        {
            assert!(self.values.insert(path.to_owned(), value.clone()).is_none());
        }
        Ok(())
    }
    fn observe_replica(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.observe(path, value)
    }
}

#[test]
fn composite_independent_banks_preserve_media_writes_and_cached_decode() {
    for (family, config, prefill) in [
        (
            "Muse",
            routed_muse_partition_fixture(),
            muse_partition_image_input(),
        ),
        (
            "Qwen3-VL",
            qwen_vl_partition_config(true),
            qwen_partition_image_input(),
        ),
        (
            "Gemma4",
            serde_json::json!({
                "model_type":"gemma4_unified", "tie_word_embeddings":false,
                "text_config":{
                    "model_type":"gemma4_text", "hidden_size":8, "num_hidden_layers":2,
                    "intermediate_size":12, "num_attention_heads":2, "num_key_value_heads":2,
                    "head_dim":4, "rms_norm_eps":0.00001, "vocab_size":19,
                    "max_position_embeddings":64, "attention_k_eq_v":false,
                    "num_kv_shared_layers":0, "layer_types":["sliding_attention","full_attention"],
                    "sliding_window":4, "enable_moe_block":true, "num_experts":4,
                    "top_k_experts":2, "moe_intermediate_size":6, "final_logit_softcapping":7.0
                }
            }),
            numeric_text_prepared_input(&[1, 2, 3]),
        ),
    ] {
        let (artifact, _) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                let seed = name
                    .bytes()
                    .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
                let count = shape.iter().map(|n| *n as usize).product();
                Some(NumericTensor::new(
                    shape.to_vec(),
                    (0..count)
                        .map(|i| {
                            let delta = ((i * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                            if name.contains("norm") && name.ends_with("weight") {
                                1.0 + delta * 0.002
                            } else {
                                delta * 0.008
                            }
                        })
                        .collect(),
                ))
            });
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let descriptor = inspection.architecture_plan().architecture_descriptor();
        let mut complete_writes = descriptor
            .components
            .iter()
            .filter_map(|group| group.output.clone())
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(readout) = &descriptor.component_readout {
            for term in &readout.other_writes {
                complete_writes.insert(term.output.clone());
                complete_writes.insert(term.effective_output.clone());
            }
        }
        // Provider-local MoE diagnostics are intentionally absent before TP's
        // architecture-owned reduction. Compare declared complete residual
        // writes, unit boundaries and media outputs across these placements.
        let provider_local = descriptor
            .observations
            .points
            .iter()
            .filter(|point| {
                point.path.ends_with(".output")
                    && !complete_writes.contains(&point.path)
                    && descriptor
                        .nodes
                        .iter()
                        .find(|node| node.id == point.node_id)
                        .unwrap()
                        .kind
                        == eredu_core::ArchitectureNodeKind::MixtureOfExperts
            })
            .map(|point| point.path.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let requirements =
            eredu_architectures::replicated_text::composite_text_requirements(&inspection).unwrap();
        let routed = requirements.routed_execution().unwrap();
        let member_bytes = routed
            .bank(eredu_runtime::RoutedBankId::new(0))
            .unwrap()
            .catalog()
            .units()
            .iter()
            .filter_map(ExpertResidencyUnit::byte_len)
            .max()
            .unwrap();
        let bank_budget = member_bytes * routed.routes_per_token() as u64;
        let parameters = numeric_composite_parameter_description(&config);
        let inputs = [
            prefill,
            numeric_text_prepared_input(&[4]),
            numeric_text_prepared_input(&[5]),
        ];
        let mut reference: Option<Vec<NumericTensor>> = None;
        let mut reference_captures: Option<BTreeMap<(usize, String), NumericTensor>> = None;
        for independent in [false, true] {
            for (tp, pp, ep) in [
                (2, 1, 1),
                (1, 2, 1),
                (1, 1, 2),
                (2, 2, 1),
                (2, 1, 2),
                (1, 2, 2),
                (2, 2, 2),
            ] {
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
                    eprintln!("{family} independent={independent} {topology:?} {residency:?}");
                    let plan = prepared_adapter::plan(None)
                        .with_topology(topology)
                        .with_residency(residency);
                    let world = Arc::new(NumericPartitionWorld::default());
                    let results = std::thread::scope(|scope| {
                        let workers = (0..topology.world_size()).map(|rank| {
                            let (inspection, parameters, inputs, plan, complete, provider_local) = (&inspection, &parameters, &inputs, &plan, &complete_writes, &provider_local);
                            let world = world.clone();
                            std::thread::Builder::new()
                                .name(format!("reference-composite-banks-{rank}"))
                                .stack_size(32 * 1024 * 1024)
                                .spawn_scoped(scope, move || {
                                let sources = partitioned_adapter::prepare_plan_with_banks(
                                    inspection, plan, rank, std::time::Duration::from_secs(30),
                                    independent.then(|| ParameterBankLoadOptions::new(
                                        eredu_core::residency::OffloadConfig::new(Some(bank_budget), Some(1 << 20), 1).unwrap(), bank_budget, bank_budget).unwrap()),
                                ).unwrap();
                                let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters, ParallelRankTopology::new(topology, rank).unwrap()).unwrap();
                                let mut context = NumericContext::with_partition(layout, rank, world);
                                context.bind_checkpoint_values = true;
                                reset_reference_stage_evidence("SafeTensors");
                                let mut executable = partitioned_adapter::composite(sources, &context).unwrap();
                                let mut captures = BTreeMap::new();
                                let outputs = inputs.iter().enumerate().map(|(step, input)| {
                                    let mut trace = Trace { values:BTreeMap::new(),complete,provider_local };
                                    let output = executable.forward_observed(input, step == 0, &mut trace).unwrap();
                                    captures.extend(trace.values.into_iter().map(|(path, value)| ((step, path), value)));
                                    output
                                }).collect::<Vec<_>>();
                                if independent {
                                    let evidence = last_reference_stage_evidence();
                                    assert!(!evidence.bank_acquisitions.is_empty());
                                    assert!(evidence.bank_completions > 0);
                                    assert!(evidence.peak_bank_bytes <= bank_budget);
                                    if tp == 1 && pp == 1 { assert!(evidence.bank_evictions > 0); }
                                }
                                (outputs, captures)
                            }).expect("spawn reference composite-banks rank worker")
                        }).collect::<Vec<_>>();
                        workers
                            .into_iter()
                            .map(|worker| worker.join().unwrap())
                            .collect::<Vec<_>>()
                    });
                    let captures = results
                        .iter()
                        .flat_map(|(_, captures)| {
                            captures
                                .iter()
                                .map(|(key, value)| (key.clone(), value.clone()))
                        })
                        .collect::<BTreeMap<_, _>>();
                    assert!(!captures.is_empty());
                    if let Some(expected) = &reference_captures {
                        assert_eq!(
                            captures.keys().collect::<Vec<_>>(),
                            expected.keys().collect::<Vec<_>>()
                        );
                        for (key, value) in &captures {
                            assert_tensor_close(
                                value,
                                &expected[key],
                                &format!("{family} {key:?}"),
                            );
                        }
                    } else {
                        reference_captures = Some(captures);
                    }
                    if reference.is_none() {
                        reference = Some(results[0].0.clone());
                    }
                    for (outputs, _) in results {
                        for (actual, expected) in outputs.iter().zip(reference.as_ref().unwrap()) {
                            assert!(actual.data.iter().any(|value| value.abs() > 1e-5));
                            assert_tensor_close(
                                actual,
                                expected,
                                &format!("{family} independent bank output"),
                            );
                        }
                    }
                }
            }
        }
    }
}
