//! Real cold selection distinguishes shared provider addresses from decoder owners.
use super::*;

#[test]
fn inkling_selected_component_placement_preserves_shared_and_routed_invocations() {
    for shared in [1, 4] {
        let mut config = routed_inkling_partition_fixture();
        config.as_object_mut().unwrap().remove("vision_config");
        config["text_config"]["model_max_length"] = 128.into();
        config["text_config"]["n_shared_experts"] = shared.into();
        let (artifact, _) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                let seed = name
                    .bytes()
                    .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
                let count = shape.iter().map(|value| *value as usize).product();
                let values = (0..count)
                    .map(|index| {
                        let delta = ((index * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                        if name.ends_with("global_scale") {
                            1.3
                        } else if name.contains("norm") && name.ends_with("weight") {
                            1.0 + delta * 0.002
                        } else {
                            delta * 0.025
                        }
                    })
                    .collect();
                Some(NumericTensor::new(shape.to_vec(), values))
            });
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let descriptor = inspection.architecture_plan().architecture_descriptor();
        let parameters = numeric_composite_parameter_description(&config);
        assert_eq!(descriptor.routed_components.len(), 4);
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
            let sources = partitioned_adapter::prepare_with_timeout(
                &inspection,
                topology,
                0,
                false,
                std::time::Duration::from_secs(30),
            )
            .unwrap();
            let discovery = sources
                .prepare_discovery(Default::default(), Default::default())
                .bind_partition_parameters(Some(Arc::new(parameters.clone())))
                .unwrap();
            let layouts = discovery.component_partition_layouts(64).unwrap().unwrap();
            assert!(!discovery.identity_is_resolved());
            for rank in 0..topology.world_size() {
                let local = layouts.rank(rank).unwrap();
                let rank = ParallelRankTopology::new(topology, rank).unwrap();
                for component in &descriptor.routed_components {
                    let observation = local.routed_observation(&component.activation).unwrap();
                    let shared_branch = component.routing.contains("shared.routing");
                    let experts = if shared_branch { shared } else { 4 };
                    assert_eq!(observation.geometry().experts, experts);
                    assert_eq!(
                        observation.geometry().routes_per_token,
                        if shared_branch { shared } else { 2 }
                    );
                    let owned = pp == 1 || component.layer_index == rank.pipeline_parallel_rank();
                    assert_eq!(observation.ownership().is_some(), owned);
                    if let Some(ownership) = observation.ownership() {
                        let expert_range = if shared_branch {
                            0..shared as usize
                        } else {
                            let width = 4 / ep;
                            rank.expert_parallel_rank() * width
                                ..(rank.expert_parallel_rank() + 1) * width
                        };
                        let actual = &ownership.coordinates;
                        assert_eq!(
                            (0..actual.experts().local_count())
                                .map(|i| actual.experts().local_to_global(i).unwrap())
                                .collect::<Vec<_>>(),
                            expert_range.collect::<Vec<_>>()
                        );
                        assert_eq!(
                            actual.units().contiguous_range(),
                            Some(
                                rank.tensor_parallel_rank() * 6 / tp
                                    ..(rank.tensor_parallel_rank() + 1) * 6 / tp
                            )
                        );
                        assert_eq!(
                            ownership.source_peer,
                            (!shared_branch && ep > 1).then_some(0)
                        );
                        assert_eq!(
                            ownership.source_peers,
                            if shared_branch { 1 } else { ep as u64 }
                        );
                    }
                    assert_eq!(
                        observation.ownership(),
                        local
                            .routed_observation(&component.effective_activation)
                            .unwrap()
                            .ownership()
                    );
                }
            }
            // A shared bank with a foreign write or route count cannot reuse the
            // routed sibling's valid owner to obtain capture authority.
            for bad_routes in [false, true] {
                let mut wrong = descriptor.clone();
                let component = &mut wrong.routed_components[1];
                if bad_routes {
                    component.routes_per_token = Some(99);
                } else {
                    let eredu_core::component::RoutedComponentParameter::Packed { name } =
                        &mut component.write_weight
                    else {
                        panic!("packed fixture")
                    };
                    name.parameter.push_str(".foreign");
                }
                assert!(sources
                    .selected()
                    .execution()
                    .component_partition_layout(&wrong, &parameters)
                    .is_err());
            }
        }
    }
}

struct PlacedUnits {
    layout: eredu_architectures::component_partition::ComponentPartitionLayout,
    path: String,
    active: bool,
    last_token: usize,
    mask_shared: Option<bool>,
    original: Option<NumericTensor>,
    rows: usize,
    edits: usize,
    fail_shared: bool,
    failure_seen: bool,
}
impl ActivationObserver<NumericTensor, Error> for PlacedUnits {
    fn observe(&mut self, _: &str, _: &NumericTensor) -> Result<(), Error> {
        Ok(())
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<NumericTensor>>, Error> {
        assert!(!self.active);
        self.path = path.into();
        assert!(self
            .layout
            .routed_observation(&format!("{path}.units"))
            .unwrap()
            .ownership()
            .is_some());
        Ok(Some(self))
    }
}
impl PlacedUnits {
    fn evidence(&self, batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>) -> Vec<usize> {
        let declared = self
            .layout
            .routed_observation(&format!("{}.units", self.path))
            .unwrap();
        let ownership = declared.ownership().unwrap();
        assert_eq!(batch.unit_coordinates, Some(ownership.coordinates.units()));
        let source_width = *batch.source_groups.shape.last().unwrap() as usize;
        let local_width = *batch.units.coefficients.shape.last().unwrap() as usize;
        (0..batch.units.values.shape[0] as usize)
            .map(|row| {
                let native_token = batch.units.token_indices.data[row] as usize;
                let slot = batch.units.selection_indices.data[row] as usize % local_width;
                let source_token = batch.source_token(native_token).unwrap();
                let expert = batch
                    .global_group(
                        batch.source_groups.data[source_token * source_width + slot] as usize,
                    )
                    .unwrap();
                assert!(
                    ownership
                        .coordinates
                        .experts()
                        .global_to_local(expert)
                        .is_some(),
                    "foreign expert {expert} at {}",
                    self.path
                );
                let origin = batch.route_origin(native_token, slot).unwrap();
                assert_eq!(
                    origin.source_peer.is_some(),
                    ownership.source_peer.is_some()
                );
                assert!(origin
                    .source_peer
                    .is_none_or(|peer| peer < ownership.source_peers as usize));
                assert!(origin.token <= self.last_token);
                origin.token
            })
            .collect()
    }
    fn edited(
        &self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Option<NumericTensor> {
        if self.mask_shared != Some(self.path.contains("shared.routing")) {
            return None;
        }
        let column = batch.unit_coordinates.unwrap().global_to_local(1)?;
        let mut value = batch.units.values.clone();
        let width = value.shape[1] as usize;
        for (row, token) in self.evidence(batch).into_iter().enumerate() {
            if token == self.last_token {
                value.data[row * width + column] = 0.0;
            }
        }
        Some(value)
    }
}
impl eredu_runtime::RoutedUnitObserver<NumericTensor> for PlacedUnits {
    fn begin_invocation(
        &mut self,
        _: &eredu_runtime::RoutedUnitInvocation<'_, NumericTensor>,
    ) -> Result<(), Error> {
        assert!(!std::mem::replace(&mut self.active, true));
        Ok(())
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), Error> {
        assert!(std::mem::replace(&mut self.active, false));
        if success {
            assert!(self.original.is_none());
        } else {
            self.original.take();
        }
        Ok(())
    }
    fn invocation_active(&self) -> bool {
        self.active
    }
    fn observe(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        self.rows += self.evidence(batch).len();
        assert!(self.original.replace(batch.units.values.clone()).is_none());
        Ok(())
    }
    fn intervene(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<Option<NumericTensor>, Error> {
        if self.fail_shared && self.path == "model.layers.0.shared.routing" {
            self.failure_seen = true;
            return Err(Error::backend(
                "injected shared provider observation failure",
            ));
        }
        let edited = self.edited(batch);
        if let Some(value) = &edited {
            self.edits += value
                .data
                .iter()
                .zip(&batch.units.values.data)
                .filter(|(a, b)| a != b)
                .count();
        }
        Ok(edited)
    }
    fn observe_effective(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        let original = self.original.take().unwrap();
        let original_batch = eredu_runtime::RoutedUnitBatch {
            units: batch.units.with_values(&original),
            ..*batch
        };
        let expected = self.edited(&original_batch).unwrap_or(original);
        assert_tensor_exact(
            batch.units.values,
            &expected,
            "shared/routed effective coordinates",
        );
        Ok(())
    }
}

#[test]
fn inkling_selected_provider_components_match_cold_placement_and_masked_execution() {
    for shared in [1, 4] {
        let mut config = routed_inkling_partition_fixture();
        config.as_object_mut().unwrap().remove("vision_config");
        config["text_config"]["model_max_length"] = 128.into();
        config["text_config"]["n_shared_experts"] = shared.into();
        let (artifact, _) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                let seed = name
                    .bytes()
                    .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
                let count = shape.iter().map(|value| *value as usize).product();
                let values = (0..count)
                    .map(|index| {
                        let delta = ((index * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                        if name.ends_with("global_scale") {
                            1.3
                        } else if name.contains("norm") && name.ends_with("weight") {
                            1.0 + delta * 0.002
                        } else {
                            delta * 0.025
                        }
                    })
                    .collect();
                Some(NumericTensor::new(shape.to_vec(), values))
            });
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let parameters = numeric_composite_parameter_description(&config);
        let mut reference: Option<Vec<Vec<NumericTensor>>> = None;
        for (tp, pp, ep) in [(2, 1, 1), (1, 1, 2), (2, 2, 2)] {
            let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
            let mut trials = Vec::new();
            for trial in 0..4 {
                let world = Arc::new(NumericPartitionWorld::default());
                let results = std::thread::scope(|scope| {
                    (0..topology.world_size()).map(|rank| {
                        let inspection = &inspection;
                        let parameters = &parameters;
                        let world = world.clone();
                        scope.spawn(move || {
                            let sources = partitioned_adapter::prepare_with_timeout(inspection, topology, rank, false, std::time::Duration::from_secs(30)).unwrap();
                            let discovery = sources.prepare_discovery(Default::default(), Default::default()).bind_partition_parameters(Some(Arc::new(parameters.clone()))).unwrap();
                            let retained = discovery.component_partition_layouts(64).unwrap().unwrap();
                            let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                            let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters, rank_topology).unwrap();
                            let mut context = NumericContext::with_partition(layout, rank, world);
                            context.bind_checkpoint_values = true;
                            let mut executable = partitioned_adapter::composite(sources, &context).unwrap();
                            let mut observer = PlacedUnits { layout: retained.rank(rank).unwrap().clone(), path: String::new(), active: false, last_token: 0, mask_shared: match trial { 2 => Some(false), 3 => Some(true), _ => None }, original: None, rows: 0, edits: 0, fail_shared: false, failure_seen: false };
                            let outputs = [&[1_usize,2,5][..], &[3][..], &[4][..]].into_iter().enumerate().map(|(step, tokens)| {
                                observer.last_token = tokens.len()-1;
                                let input = numeric_text_prepared_input(tokens);
                                if trial == 0 { executable.forward(&input, step == 0).unwrap() }
                                else { executable.forward_observed(&input, step == 0, &mut observer).unwrap() }
                            }).collect::<Vec<_>>();
                            assert!(!observer.active);
                            (outputs, observer.rows, observer.edits)
                        })
                    }).collect::<Vec<_>>().into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>()
                });
                if trial > 0 {
                    assert!(results.iter().map(|result| result.1).sum::<usize>() > 0);
                }
                if trial > 1 {
                    assert!(results.iter().map(|result| result.2).sum::<usize>() > 0);
                }
                for result in &results {
                    for (actual, expected) in result.0.iter().zip(&results[0].0) {
                        assert_tensor_exact(actual, expected, "all ranks publish the same scores");
                    }
                }
                trials.push(results[0].0.clone());
            }
            for (a, b) in trials[0].iter().zip(&trials[1]) {
                assert_tensor_exact(a, b, "Inkling no-op instrumentation");
            }
            for masked in &trials[2..] {
                assert!(
                    masked.iter().zip(&trials[0]).any(|(a, b)| a
                        .data
                        .iter()
                        .zip(&b.data)
                        .any(|(a, b)| (a - b).abs() > 1e-7)),
                    "nonzero masks change actual scores (shared={shared}, tp={tp}, pp={pp}, ep={ep})"
                );
            }
            if let Some(reference) = &reference {
                for (trial, expected) in trials.iter().zip(reference) {
                    for (actual, expected) in trial.iter().zip(expected) {
                        assert_eq!(actual.shape, expected.shape);
                        assert!(
                            actual
                                .data
                                .iter()
                                .zip(&expected.data)
                                .all(|(a, b)| (a - b).abs() < 2e-5),
                            "parallel shared/routed numerical parity"
                        );
                    }
                }
            } else {
                reference = Some(trials);
            }
        }
    }
}

#[test]
fn inkling_selected_shared_provider_failure_rejects_every_replica_before_commit() {
    for (tp, pp, ep) in [(1, 1, 2), (2, 2, 2)] {
        let mut config = routed_inkling_partition_fixture();
        config.as_object_mut().unwrap().remove("vision_config");
        config["text_config"]["model_max_length"] = 128.into();
        config["text_config"]["n_shared_experts"] = 4.into();
        let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let parameters = numeric_composite_parameter_description(&config);
        let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
        let world = Arc::new(NumericPartitionWorld::default());
        let results = std::thread::scope(|scope| {
            (0..topology.world_size()).map(|rank| {
                let inspection = &inspection;
                let parameters = &parameters;
                let world = world.clone();
                scope.spawn(move || {
                    let sources = partitioned_adapter::prepare_with_timeout(inspection, topology, rank, false, std::time::Duration::from_secs(10)).unwrap();
                    let discovery = sources.prepare_discovery(Default::default(), Default::default()).bind_partition_parameters(Some(Arc::new(parameters.clone()))).unwrap();
                    let retained = discovery.component_partition_layouts(64).unwrap().unwrap();
                    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters, ParallelRankTopology::new(topology, rank).unwrap()).unwrap();
                    let mut context = NumericContext::with_partition(layout, rank, world);
                    context.bind_checkpoint_values = true;
                    let mut executable = partitioned_adapter::composite(sources, &context).unwrap();
                    let mut observer = PlacedUnits { layout: retained.rank(rank).unwrap().clone(), path: String::new(), active: false, last_token: 2, mask_shared: None, original: None, rows: 0, edits: 0, fail_shared: rank == 0, failure_seen: false };
                    let before = executable.positions().unwrap();
                    let result = executable.forward_observed(&numeric_text_prepared_input(&[1,2,5]), true, &mut observer);
                    assert!(result.is_err(), "peer failure must reject rank {rank}");
                    assert_eq!(executable.positions().unwrap(), before, "failed shared bank must not advance mutable state");
                    assert!(!observer.active);
                    observer.failure_seen
                })
            }).collect::<Vec<_>>().into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>()
        });
        assert_eq!(results.iter().filter(|failed| **failed).count(), 1);
        assert!(
            world.lifecycle_counts().3.is_empty(),
            "provider rejection precedes commit"
        );
    }
}

#[test]
fn inkling_selected_replicated_shared_banks_keep_their_global_ids_when_addressable() {
    // More shared experts than routed experts proves that provider key lookup
    // cannot use the routed bank's identity vector for this invocation.
    non_mlx_inkling_composite_with_shared_banks(4);
}

#[test]
fn inkling_selected_partition_transforms_retain_source_geometry_and_cached_execution() {
    inkling_partition_bank_trials(
        false,
        Some(eredu_core::QuantizationRequest::Affine {
            group_size: 16,
            bits: 4,
        }),
    );
}

#[test]
fn inkling_partition_independent_shared_banks_match_resident_and_bound_eviction() {
    for quantization in [
        None,
        Some(eredu_core::QuantizationRequest::Affine {
            group_size: 16,
            bits: 4,
        }),
    ] {
        inkling_partition_bank_trials(true, quantization);
    }
}

fn inkling_partition_bank_trials(
    independent: bool,
    quantization: Option<eredu_core::QuantizationRequest>,
) {
    let bank_budget = 49152;
    #[derive(Default)]
    struct Trace(Vec<(String, NumericTensor)>);
    impl ActivationObserver<NumericTensor, Error> for Trace {
        fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
            if path.ends_with("attention.input")
                || path.ends_with("attention.residual")
                || path.ends_with("feed_forward.input")
                || path.ends_with("feed_forward.residual")
                || path == "readout.embedding"
            {
                self.0.push((path.to_owned(), value.clone()));
            }
            Ok(())
        }
    }
    for routed in [false, true] {
        if independent && !routed {
            continue;
        }
        let mut config = routed_inkling_partition_fixture();
        config.as_object_mut().unwrap().remove("vision_config");
        for (field, value) in [
            ("hidden_size", 32),
            ("vocab_size", 32),
            ("unpadded_vocab_size", 30),
            ("num_attention_heads", 4),
            ("num_key_value_heads", 2),
            ("head_dim", 8),
            ("d_rel", 16),
            ("intermediate_size", 32),
            ("dense_intermediate_size", 32),
            ("moe_intermediate_size", 32),
            ("n_shared_experts", 4),
            ("n_routed_experts", 2),
        ] {
            config["text_config"][field] = value.into();
        }
        if quantization.is_some() {
            // TP2 keeps the packed format's minimum 32 input columns per rank.
            for field in [
                "hidden_size",
                "intermediate_size",
                "dense_intermediate_size",
                "moe_intermediate_size",
            ] {
                config["text_config"][field] = 64.into();
            }
            config["text_config"]["head_dim"] = 16.into();
            config["text_config"]["d_rel"] = 32.into();
        }
        config["text_config"]["mlp_layer_types"] = if routed {
            serde_json::json!(["moe", "moe"])
        } else {
            serde_json::json!(["dense", "dense"])
        };
        let (artifact, _) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                let seed = name
                    .bytes()
                    .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
                let count = shape.iter().map(|value| *value as usize).product();
                Some(NumericTensor::new(
                    shape.to_vec(),
                    (0..count)
                        .map(|index| {
                            let delta = ((index * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                            if name.ends_with("global_scale") {
                                1.3
                            } else if name.contains("norm") && name.ends_with("weight") {
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
        let args = eredu_architectures::inkling::ModelArgs::from_hf_json(
            &serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let parameters = numeric_composite_parameter_description(&config);
        let plan = prepared_adapter::plan(quantization);
        let mut reference: Option<Vec<NumericTensor>> = None;
        let mut reference_trace: Option<BTreeMap<(usize, String), NumericTensor>> = None;
        let mut topologies = vec![(2, 1, 1), (1, 2, 1), (2, 2, 1)];
        if routed {
            topologies.extend([(1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2)]);
        }
        for independent_banks in if independent {
            vec![false, true]
        } else {
            vec![false]
        } {
            for &(tp, pp, ep) in &topologies {
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
                    eprintln!("Inkling transform routed={routed} tp={tp} pp={pp} ep={ep} residency={residency:?}");
                    let world = Arc::new(NumericPartitionWorld::default());
                    let results = std::thread::scope(|scope| {
                        let handles = (0..topology.world_size()).map(|rank| {
                        let plan = plan.clone().with_topology(topology).with_residency(residency.clone());
                        let inspection = &inspection;
                        let parameters = &parameters;
                        let args = &args;
                        let world = world.clone();
                        scope.spawn(move || {
                            let sources = partitioned_adapter::prepare_plan_with_banks(inspection, &plan, rank, std::time::Duration::from_secs(30), independent_banks.then(|| ParameterBankLoadOptions::new(eredu_core::residency::OffloadConfig::new(Some(bank_budget), Some(1 << 20), 1).unwrap(), bank_budget, bank_budget).unwrap())).unwrap();
                            let selected = sources.selected().execution().text_realization();
                            assert_eq!(selected.parameters().iter().any(|parameter| matches!(parameter.lowering(), eredu_runtime::WeightLoweringKind::Transform | eredu_runtime::WeightLoweringKind::DerivedTransform)), quantization.is_some());
                            let routers = selected.parameters().iter().filter(|parameter| parameter.name().ends_with(".moe.router.weight")).collect::<Vec<_>>();
                            assert_eq!(routers.len(), if routed { 2 } else { 0 });
                            assert!(routers.iter().all(|parameter| !matches!(parameter.lowering(), eredu_runtime::WeightLoweringKind::Transform | eredu_runtime::WeightLoweringKind::DerivedTransform)));
                            let target_args = eredu_architectures::inkling::with_checkpoint_formats(args, selected.parameters().iter().filter_map(|parameter| parameter.executable().weight_quantization().map(|format| (parameter.name().to_owned(), format))).collect()).unwrap();
                            let target = eredu_architectures::inkling::LayeredModel::<NumericBackend>::new(target_args, &NumericContext::default()).unwrap();
                            let target_parameters = target.parameter_description(&NumericContext::default()).unwrap().into_owned();
                            let topology_rank = ParallelRankTopology::new(topology, rank).unwrap();
                            let source = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters, topology_rank).unwrap();
                            let encoded = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(&target_parameters, topology_rank).unwrap();
                            let mut layout = eredu_runtime::derive_transform_source_layout(&source, &encoded).unwrap();
                            for (name, companion) in encoded.tensors() {
                                if !layout.contains(name) { layout.insert(name.to_owned(), companion.clone()); }
                            }
                            let mut context = NumericContext::with_partition(layout, rank, world);
                            context.bind_checkpoint_values = true;
                            reset_reference_stage_evidence("SafeTensors");
                            let mut executable = partitioned_adapter::composite(sources, &context).unwrap();
                            let mut trace = Trace::default();
                            let mut captures = BTreeMap::new();
                            let outputs = [&[1_usize,2,5][..], &[3][..], &[4][..]].into_iter().enumerate().map(|(step,tokens)| {
                                let output = executable.forward_observed(&numeric_text_prepared_input(tokens), step == 0, &mut trace).unwrap();
                                captures.extend(trace.0.drain(..).map(|(path, value)| ((step, path), value)));
                                output
                            }).collect::<Vec<_>>();
                            if independent_banks {
                                let evidence = last_reference_stage_evidence();
                                assert!(!evidence.bank_acquisitions.is_empty());
                                assert!(evidence.bank_acquisitions.iter().any(|key| key.member() == 3), "all four shared identities must be acquired despite only two routed experts");
                                assert!(evidence.bank_completions > 0);
                                assert!(evidence.peak_bank_bytes <= bank_budget);
                                if tp == 1 && pp == 1 { assert!(evidence.bank_evictions > 0); }
                            }
                            (outputs, captures)
                        })
                    }).collect::<Vec<_>>();
                        handles
                            .into_iter()
                            .map(|handle| handle.join().unwrap())
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
                    if let Some(expected) = &reference_trace {
                        for (key, value) in &captures {
                            assert_tensor_close(
                                value,
                                &expected[key],
                                &format!("Inkling transform trace {key:?}"),
                            );
                        }
                    } else {
                        reference_trace = Some(captures);
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
                                "Inkling affine source/target partition",
                            );
                        }
                    }
                }
            }
        }
    }
}
