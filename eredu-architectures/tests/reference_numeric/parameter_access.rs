//! Real partition owners, source-backed loans and parameter-sensitive execution.
use super::*;
use eredu_core::{capture::*, parameters::*};
use eredu_runtime::parameter_operations::{LayeredParameterOwner, PreparedParameterLocation};
use eredu_runtime::StaticParameterVisitorMut;
use std::convert::Infallible;

struct Budget(CaptureUsage);

fn cold_retained_values(executable: &NumericPartitionExecutable) -> Vec<NumericTensor> {
    let before = last_reference_stage_evidence();
    let values = (executable.retained_parameter_values)().unwrap();
    assert_eq!(
        last_reference_stage_evidence(),
        before,
        "cold traversal must not read or materialize checkpoint data"
    );
    assert!(!values.is_empty());
    assert!(values
        .iter()
        .any(|value| value.data.iter().any(|value| *value != 0.0)));
    values
}

fn retained_bits(values: &[NumericTensor]) -> Vec<(Vec<i32>, Vec<u32>)> {
    let mut values = values
        .iter()
        .map(|value| {
            (
                value.shape.clone(),
                value.data.iter().map(|value| value.to_bits()).collect(),
            )
        })
        .collect::<Vec<_>>();
    values.sort();
    values
}
impl CaptureReservation for Budget {
    fn reserve(&mut self, usage: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        self.0 = self.0.checked_add(usage)?;
        assert!(self.0.host_bytes < 1 << 24);
        Ok(None)
    }
}

fn offsets(region: &ParameterRegion, shape: &[u64]) -> Vec<usize> {
    let count: u64 = region.shape.iter().product();
    (0..count)
        .map(|mut index| {
            let mut destination = 0;
            let mut stride = 1;
            for axis in (0..shape.len()).rev() {
                destination += (region.starts[axis] + index % region.shape[axis]) * stride;
                index /= region.shape[axis];
                stride *= shape[axis];
            }
            destination as usize
        })
        .collect()
}

fn local_values(global: &NumericTensor, coordinates: &ParameterCoordinateMap) -> NumericTensor {
    let count: u64 = coordinates.local_shape().iter().product();
    let values = (0..count)
        .map(|mut index| {
            let mut source = 0;
            let mut stride = 1;
            for axis in (0..coordinates.axes().len()).rev() {
                let coordinate = index % coordinates.local_shape()[axis];
                index /= coordinates.local_shape()[axis];
                source += coordinates.axes()[axis]
                    .local_to_global(coordinate as usize)
                    .unwrap()
                    * stride;
                stride *= coordinates.global_shape()[axis] as usize;
            }
            global.data[source]
        })
        .collect();
    NumericTensor::new(
        coordinates
            .local_shape()
            .iter()
            .map(|n| *n as i32)
            .collect::<Vec<_>>(),
        values,
    )
}

fn ordinary(
    config: &serde_json::Value,
    values: &BTreeMap<String, NumericTensor>,
    inputs: &[NumericTensor],
) -> Vec<NumericTensor> {
    struct Bind<'a>(&'a BTreeMap<String, NumericTensor>);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
        fn visit_mut(
            &mut self,
            metadata: eredu_nn::ParameterMetadata,
            value: &'a mut NumericTensor,
        ) {
            let id = metadata.alias_of.as_ref().unwrap_or(&metadata.id).as_str();
            if let Some(expected) = self.0.get(id) {
                *value = expected.clone();
            }
        }
    }
    impl StaticParameterVisitorMut<NumericBackend> for Bind<'_> {
        type Error = Infallible;
        fn visit_mut<M: Parameterized<NumericTensor>>(
            &mut self,
            _: &str,
            module: &mut M,
        ) -> Result<(), Self::Error> {
            module.visit_parameters_mut(self);
            Ok(())
        }
    }
    let context = NumericContext::default();
    let args = llama::model_args_from_config_value(config).unwrap();
    let address = decoder::dense_parameter_description(&args)
        .unwrap()
        .unit_layout()
        .address(0)
        .unwrap();
    let mut architecture = decoder::LayeredModel::<NumericBackend, _>::new(args, &context).unwrap();
    architecture
        .visit_static_parameters_mut(&mut Bind(values))
        .unwrap();
    let mut state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let units = (0..2)
        .map(|index| {
            let mut unit = architecture.construct_unit(index, &context).unwrap();
            unit.visit_parameters_mut(&mut Bind(values));
            unit
        })
        .collect();
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let mut retained = 0;
    assert!(runtime.visit_retained_values(&mut |value| {
        retained += value.data.len();
    }));
    assert!(retained > 0);
    let loan = LayerwisePolicy::<NumericBackend, _>::acquire::<Error, _>(
        runtime.policy_mut(),
        0,
        address,
        |_| panic!("resident inspection cannot construct a unit"),
        &context,
    )
    .unwrap();
    let mut partial = 0;
    assert!(!runtime.visit_retained_values(&mut |value| {
        partial += value.data.len();
    }));
    assert!(partial > 0, "idle static and unloaned units remain visible");
    assert!(partial < retained, "the active unit has no idle inventory");
    LayerwisePolicy::<NumericBackend, _>::abort(
        runtime.policy_mut(),
        Some((0, address, loan)),
        &context,
    );
    assert!(runtime.visit_retained_values(&mut |_| {}));
    inputs
        .iter()
        .map(|tokens| {
            let output = runtime
                .forward(
                    decoder::LayeredInput { tokens, mask: None },
                    &mut state,
                    &context,
                )
                .unwrap();
            numeric_text_output(output).unwrap()
        })
        .collect()
}

#[test]
fn prepared_partition_parameter_loans_and_replacements_match_independent_execution() {
    let inputs = [
        NumericTensor::token_ids(&[1, 2, 5]),
        NumericTensor::token_ids(&[3]),
        NumericTensor::token_ids(&[4]),
    ];
    for tied in [false, true] {
        let config = serde_json::json!({"model_type":"llama", "hidden_size":8, "intermediate_size":16,
            "num_hidden_layers":2, "num_attention_heads":4, "num_key_value_heads":2, "head_dim":2,
            "vocab_size":15, "rms_norm_eps":1e-5, "max_position_embeddings":32, "tie_word_embeddings":tied});
        let (root, bits) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let parameters = decoder::dense_parameter_description(
            &llama::model_args_from_config_value(&config).unwrap(),
        )
        .unwrap();
        let baseline = bits
            .iter()
            .map(|(id, (shape, values))| {
                (
                    id.clone(),
                    NumericTensor::new(
                        shape.clone(),
                        values.iter().map(|v| f32::from_bits(*v)).collect(),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let edits = [
            (
                "model.layers.0.self_attn.o_proj.weight",
                ParameterRegion {
                    starts: vec![1, 1],
                    shape: vec![2, 3],
                },
                ParameterUpdate::Add {
                    values: vec![0.4, -0.3, 0.7, -0.2, 0.1, -0.5],
                },
            ),
            (
                "model.layers.1.mlp.down_proj.weight",
                ParameterRegion {
                    starts: vec![0, 2],
                    shape: vec![3, 5],
                },
                ParameterUpdate::Replace {
                    values: (0..15).map(|i| (i as f32 - 7.) * 0.31).collect(),
                },
            ),
            (
                "model.norm.weight",
                ParameterRegion {
                    starts: vec![2],
                    shape: vec![2],
                },
                ParameterUpdate::Add {
                    values: vec![0.15, -0.25],
                },
            ),
            (
                "model.embed_tokens.weight",
                ParameterRegion {
                    starts: vec![1, 2],
                    shape: vec![3, 2],
                },
                ParameterUpdate::Replace {
                    values: vec![0.7, -0.2, 0.4, 0.1, -0.3, 0.8],
                },
            ),
        ];
        let mut changed = baseline.clone();
        // Independent reference edits apply directly in global row-major storage.
        for (id, region, update) in &edits {
            let tensor = changed.get_mut(*id).unwrap();
            let shape = tensor.shape.iter().map(|n| *n as u64).collect::<Vec<_>>();
            for (index, value) in offsets(region, &shape).into_iter().zip(update.values()) {
                tensor.data[index] = match update {
                    ParameterUpdate::Add { .. } => tensor.data[index] + value,
                    ParameterUpdate::Replace { .. } => *value,
                };
            }
        }
        let ordinary_baseline = ordinary(&config, &baseline, &inputs);
        let ordinary_changed = ordinary(&config, &changed, &inputs);
        assert!(ordinary_baseline
            .iter()
            .zip(&ordinary_changed)
            .any(|(a, b)| a
                .data
                .iter()
                .zip(&b.data)
                .any(|(a, b)| (a - b).abs() > 1e-3)));
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
            for topology in [
                ParallelTopology::new(2, 1, 1, 1).unwrap(),
                ParallelTopology::new(1, 2, 1, 1).unwrap(),
                ParallelTopology::new(2, 2, 1, 1).unwrap(),
            ] {
                let world = Arc::new(NumericPartitionWorld::default());
                std::thread::scope(|scope| {
                    let handles = (0..topology.world_size()).map(|rank| {
                        let world = Arc::clone(&world); let residency = residency.clone();
                        let inspection = &inspection; let parameters = &parameters; let inputs = &inputs;
                        let baseline = &baseline; let changed = &changed; let edits = &edits;
                        let ordinary_baseline = &ordinary_baseline; let ordinary_changed = &ordinary_changed;
                        scope.spawn(move || {
                            let topology_rank = ParallelRankTopology::new(topology, rank).unwrap();
                            let plan = prepared_adapter::plan(None).with_topology(topology).with_residency(residency);
                            let sources = partitioned_adapter::prepare_plan(inspection, &plan, rank, std::time::Duration::from_secs(10)).unwrap();
                            let discovery = sources.prepare_discovery(Default::default(), Default::default()).bind_partition_parameters(Some(Arc::new(parameters.clone()))).unwrap();
                            let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters, topology_rank).unwrap();
                            let mut context = NumericContext::with_partition(layout, rank, world);
                            context.bind_checkpoint_values = true;
                            let mut executable = partitioned_adapter::dense(sources, &context).unwrap();
                            let retained_before = retained_bits(&cold_retained_values(&executable));
                            let mut original = BTreeMap::new(); let mut replacement = BTreeMap::new();
                            let mut locations = BTreeMap::new();
                            let mut budget = Budget(Default::default());
                            for (id, region, update) in edits {
                                let layout = discovery.parameter_partition_layout_for_rank(id, rank, &mut budget).unwrap().unwrap();
                                let Some(coordinates) = layout.coordinates() else { continue; };
                                let owner = parameters.groups().iter().find(|group| group.members().iter().any(|member| member.target() == layout.target())).unwrap().owner();
                                let location = match owner {
                                    eredu_runtime::ParameterGroupOwner::ExecutionUnit { global_unit, .. } => PreparedParameterLocation::Unit {
                                        ordinal: *global_unit - if topology.pipeline() == 2 { topology_rank.pipeline_parallel_rank() } else {0},
                                        address: parameters.unit_layout().address(*global_unit).unwrap(),
                                    },
                                    eredu_runtime::ParameterGroupOwner::StaticRole(role) => PreparedParameterLocation::Static { role: role.clone() },
                                    eredu_runtime::ParameterGroupOwner::StaticAnyOf(roles) => PreparedParameterLocation::Static { role: roles[0].clone() },
                                    _ => panic!("fixture owner"),
                                };
                                assert!((executable.parameters)(&location, true).is_err());
                                let values = (executable.parameters)(&location, false).unwrap();
                                let projection = coordinates.project_region(region, 32, &mut budget).unwrap();
                                let mut aliases = 0;
                                for (slot, tensor) in values {
                                    let Ok(Some(alias)) = discovery.parameter_partition_layout_for_rank(&slot, rank, &mut budget) else { continue; };
                                    if alias.target() != layout.target() { continue; }
                                    assert_tensor_exact(&tensor, &local_values(&baseline[*id], coordinates), "actual local prepared values");
                                    let mut next = tensor.clone();
                                    for (index, fragment) in projection.fragments().iter().enumerate() {
                                        let update = projection.project_update(index, update, &mut budget).unwrap();
                                        for (index, value) in offsets(fragment.local(), coordinates.local_shape()).into_iter().zip(update.values()) {
                                            next.data[index] = match update { ParameterUpdate::Add { .. } => next.data[index] + value, ParameterUpdate::Replace { .. } => *value };
                                        }
                                    }
                                    assert_tensor_exact(&next, &local_values(&changed[*id], coordinates), "independent global replacement");
                                    locations.insert(slot.clone(), location.clone());
                                    original.insert(slot.clone(), tensor); replacement.insert(slot, next); aliases += 1;
                                }
                                assert!(aliases > 0, "local parameter has no actual loaded slot: {id} rank {rank}");
                            }
                            for (step, tokens) in inputs.iter().enumerate() { assert_tensor_close(&executable.forward(tokens, step == 0).unwrap(), &ordinary_baseline[step], "baseline after failed and successful parameter loans"); }
                            executable.reset().unwrap();
                            assert!((executable.publish_parameters)(&replacement, true).unwrap());
                            let retained = cold_retained_values(&executable);
                            for expected in replacement.values() {
                                assert!(retained.iter().any(|value| value.shape == expected.shape && value.data == expected.data), "cold inventory must include current replacements or stored reload overrides");
                            }
                            for (slot, expected) in &replacement { assert_tensor_exact(&(executable.parameters)(&locations[slot], false).unwrap()[slot], expected, "published values survive bounded reload"); }
                            for (step, tokens) in inputs.iter().enumerate() { assert_tensor_close(&executable.forward(tokens, step == 0).unwrap(), &ordinary_changed[step], "partition edits match independently edited ordinary model"); }
                            executable.reset().unwrap();
                            assert!((executable.publish_parameters)(&original, false).unwrap());
                            assert_eq!(retained_bits(&cold_retained_values(&executable)), retained_before, "restoration removes override ownership and restores resident values");
                            for (step, tokens) in inputs.iter().enumerate() { assert_tensor_close(&executable.forward(tokens, step == 0).unwrap(), &ordinary_baseline[step], "restoration preserves baseline"); }
                        })
                    }).collect::<Vec<_>>();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            }
        }
    }
}
