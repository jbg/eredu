//! Actual prepared providers, including exchange, TP columns and pipeline ownership.
use super::*;
use eredu_core::{
    capture::{ResolvedCaptureSlice, RoutedUnitGeometry},
    component::{ComponentCoordinateMap, RoutedComponentCoordinateMap},
    intervention::{InterventionAction, InterventionDtype, RoutedUnitLocation},
};
use eredu_runtime::ActivationObserver;
#[path = "partition/invocation.rs"]
mod invocation;
#[path = "partition/observer.rs"]
mod observer;
#[path = "partition/placement.rs"]
mod placement;

type Key = (String, Option<usize>, usize, usize, usize, usize);
type Values = BTreeMap<Key, f32>;
struct Observe {
    path: String,
    topology: Option<ParallelRankTopology>,
    geometry: BTreeMap<String, RoutedUnitGeometry>,
    source_tokens: usize,
    edit: bool,
    original: Values,
    effective: Values,
    requested: Vec<String>,
    started: BTreeMap<String, usize>,
    active: Option<(usize, usize)>,
    retained: Option<eredu_architectures::component_partition::ComponentPartitionLayout>,
}
impl Observe {
    fn new(
        descriptor: &eredu_core::ArchitectureDescriptor,
        topology: Option<ParallelRankTopology>,
        source_tokens: usize,
        edit: bool,
    ) -> Self {
        Self {
            path: String::new(),
            topology,
            source_tokens,
            edit,
            requested: Vec::new(),
            started: BTreeMap::new(),
            active: None,
            retained: None,
            geometry: descriptor
                .routed_components
                .iter()
                .map(|c| {
                    (
                        c.routing.clone(),
                        RoutedUnitGeometry {
                            experts: c.expert_count as u64,
                            units_per_expert: c.units_per_expert as u64,
                            routes_per_token: 2,
                        },
                    )
                })
                .collect(),
            original: BTreeMap::new(),
            effective: BTreeMap::new(),
        }
    }
    fn coordinates(
        &self,
        batch: &RoutedUnitBatch<'_, NumericTensor>,
    ) -> RoutedComponentCoordinateMap {
        let geometry = self.geometry[&self.path];
        let experts = match self.topology {
            Some(rank) => eredu_core::balanced_contiguous_range(
                geometry.experts as usize,
                rank.expert_parallel_size(),
                rank.expert_parallel_rank(),
                false,
            )
            .unwrap(),
            None => 0..geometry.experts as usize,
        };
        let expected_units = match self.topology {
            Some(rank) => {
                let width = geometry.units_per_expert as usize / rank.tensor_parallel_size();
                rank.tensor_parallel_rank() * width..(rank.tensor_parallel_rank() + 1) * width
            }
            None => 0..geometry.units_per_expert as usize,
        };
        assert_eq!(batch.unit_coordinates.is_some(), self.topology.is_some());
        let units = batch.unit_coordinates.cloned().unwrap_or_else(|| {
            ComponentCoordinateMap::range(
                geometry.units_per_expert as usize,
                expected_units.clone(),
            )
            .unwrap()
        });
        assert_eq!(units.contiguous_range(), Some(expected_units));
        assert_eq!(units.global_count(), geometry.units_per_expert as usize);
        assert_eq!(units.local_count(), batch.units.values.shape[1] as usize);
        let coordinates = RoutedComponentCoordinateMap::new(
            ComponentCoordinateMap::range(geometry.experts as usize, experts).unwrap(),
            units,
        );
        if let Some(retained) = &self.retained {
            let declared = retained
                .routed_observation(&format!("{}.units", self.path))
                .unwrap();
            let selected = &declared.ownership().unwrap().coordinates;
            assert_eq!(
                selected.units(),
                coordinates.units(),
                "cold selected placement must match actual prepared provider columns"
            );
            assert_eq!(
                selected.experts().local_count(),
                coordinates.experts().local_count()
            );
            for local in 0..selected.experts().local_count() {
                assert_eq!(
                    selected.experts().local_to_global(local),
                    coordinates.experts().local_to_global(local)
                );
            }
        }
        coordinates
    }
    fn collect(&mut self, batch: &RoutedUnitBatch<'_, NumericTensor>, effective: bool) {
        if self.topology.is_some() {
            let (rows, end) = self
                .active
                .as_mut()
                .expect("actual local invocation precedes chunks");
            if !effective {
                assert_eq!(batch.source_token(0).unwrap(), *end);
                *end += batch.units.coefficients.shape[0] as usize;
                assert!(*end <= *rows);
            }
        }
        let coordinates = self.coordinates(batch);
        let width = coordinates.units().local_count();
        let output = if effective {
            &mut self.effective
        } else {
            &mut self.original
        };
        for (row, (peer, token, slot, expert)) in Capture::keys(batch).into_iter().enumerate() {
            assert_eq!(
                peer.is_some(),
                self.topology
                    .is_some_and(|rank| rank.expert_parallel_size() > 1)
            );
            assert!(coordinates.experts().global_to_local(expert).is_some());
            for local in 0..width {
                let unit = coordinates.units().local_to_global(local).unwrap();
                let key = (self.path.clone(), peer, token, slot, expert, unit);
                let value = batch.units.values.data[row * width + local];
                assert!(
                    output.insert(key, value).is_none(),
                    "duplicate local routed value"
                );
            }
        }
    }
}
impl ActivationObserver<NumericTensor, Error> for Observe {
    fn observe(&mut self, _path: &str, _value: &NumericTensor) -> Result<(), Error> {
        Ok(())
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<NumericTensor>>, Error> {
        assert!(self.geometry.contains_key(path));
        self.path = path.into();
        self.requested.push(path.into());
        Ok(Some(self))
    }
}
impl RoutedUnitObserver<NumericTensor> for Observe {
    fn begin_invocation(
        &mut self,
        invocation: &eredu_runtime::RoutedUnitInvocation<'_, NumericTensor>,
    ) -> Result<(), Error> {
        let topology = self.topology.expect("partition provider lifecycle");
        if let Some(retained) = &self.retained {
            let declared = retained
                .routed_observation(&format!("{}.units", self.path))
                .unwrap();
            let ownership = declared
                .ownership()
                .expect("only the selected PP stage invokes this bank");
            assert_eq!(
                declared.input_width(),
                *invocation.input.shape.last().unwrap() as u64
            );
            assert_eq!(
                ownership.source_peers,
                topology.expert_parallel_size() as u64
            );
            if let Some(units) = invocation.unit_coordinates {
                assert_eq!(units, ownership.coordinates.units());
            }
        }
        assert!(self.active.is_none());
        let rows = invocation.input.shape[..invocation.input.shape.len() - 1]
            .iter()
            .map(|n| *n as usize)
            .product();
        if topology.expert_parallel_size() > 1 {
            let origins = invocation.origins.expect("actual received tags");
            assert_eq!(origins.capture_coordinates().row_count(), rows);
            assert_eq!(
                origins.capture_coordinates().peer_count(),
                topology.expert_parallel_size()
            );
        } else {
            assert!(invocation.origins.is_none());
            assert!(invocation.unit_coordinates.is_some());
            assert_eq!(rows, self.source_tokens);
        }
        assert!(
            self.started.insert(self.path.clone(), rows).is_none(),
            "one start per actual invocation"
        );
        self.active = Some((rows, 0));
        Ok(())
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), Error> {
        assert!(success);
        let (rows, end) = self
            .active
            .take()
            .expect("explicit start, including idle owner");
        assert_eq!(end, rows, "all actual chunks precede provider completion");
        Ok(())
    }
    fn observe(&mut self, batch: &RoutedUnitBatch<'_, NumericTensor>) -> Result<(), Error> {
        self.collect(batch, false);
        Ok(())
    }
    fn observe_effective(
        &mut self,
        batch: &RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        self.collect(batch, true);
        Ok(())
    }
    fn intervene(
        &mut self,
        batch: &RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<Option<NumericTensor>, Error> {
        if !self.edit {
            return Ok(None);
        }
        let geometry = self.geometry[&self.path];
        let coordinates = self.coordinates(batch);
        let keys = Capture::keys(batch);
        let mut values = batch.units.values.clone();
        if self.topology.is_none() {
            // Independent full-bank reference: keep a different unit in each expert.
            let width = values.shape[1] as usize;
            for (row, (_, _, _, expert)) in keys.iter().enumerate() {
                for unit in 0..width {
                    if unit != expert % width {
                        values.data[row * width + unit] = 0.0;
                    }
                }
            }
        } else {
            let rows = keys
                .iter()
                .map(|(peer, token, slot, expert)| RoutedUnitLocation {
                    source_peer: peer.map(|p| p as u64),
                    token: *token as u64,
                    slot: *slot as u64,
                    expert: *expert as u64,
                })
                .collect::<Vec<_>>();
            let components = geometry.components().unwrap();
            let slice = ResolvedCaptureSlice {
                starts: vec![0, 0],
                ends: vec![self.source_tokens as u64, components],
                strides: vec![1, 1],
                shape: vec![self.source_tokens as u64, components],
            };
            let action = InterventionAction::MaskComponents {
                dtype: InterventionDtype::Float32,
                indices: (0..geometry.experts)
                    .map(|e| (e * geometry.units_per_expert + e % geometry.units_per_expert) as u32)
                    .collect(),
                keep_selected: true,
            };
            let lowered = eredu_runtime::intervention::lower_partition_routed_intervention(
                geometry,
                self.source_tokens as u64,
                &rows,
                &coordinates,
                &slice,
                &action,
            )
            .unwrap();
            for index in lowered.indices {
                values.data[index as usize] = 0.0;
            }
        }
        Ok(Some(values))
    }
}

fn ordinary(
    args: &qwen::ModelArgs,
    fixture: &prepared_adapter::ParameterBits,
    descriptor: &eredu_core::ArchitectureDescriptor,
    inputs: &[NumericTensor],
    edit: bool,
) -> Vec<(NumericTensor, Values, Values)> {
    struct Populate<'a>(&'a prepared_adapter::ParameterBits);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            let (shape, bits) = &self.0[metadata.id.as_str()];
            assert_eq!(&value.shape, shape);
            value.data = bits.iter().map(|bits| f32::from_bits(*bits)).collect();
        }
    }
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut architecture =
        qwen::RoutedLayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    architecture
        .static_modules_mut()
        .visit_parameters_mut(&mut Populate(fixture));
    let mut state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let units = (0..2)
        .map(|layer| {
            let mut unit = architecture.construct_unit(layer, &context).unwrap();
            unit.visit_parameters_mut(&mut Populate(fixture));
            unit
        })
        .collect();
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    inputs
        .iter()
        .enumerate()
        .map(|(step, tokens)| {
            let mut observer = Observe::new(descriptor, None, tokens.data.len(), edit);
            let output = runtime
                .forward_with_provider_and_observer(
                    decoder::LayeredInput { tokens, mask: None },
                    &mut state,
                    if step == 0 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    },
                    &mut eredu_runtime::ResidentExpertProvider,
                    &context,
                    &mut observer,
                )
                .unwrap();
            assert!(!observer.original.is_empty());
            let width = *output.shape.last().unwrap() as usize;
            (
                NumericTensor::new(
                    vec![1, 1, width as i32],
                    output.data[output.data.len() - width..].to_vec(),
                ),
                observer.original,
                observer.effective,
            )
        })
        .collect()
}

#[test]
fn selected_partition_providers_preserve_sparse_units_and_edits_across_tp_ep_pp_residency() {
    let config = serde_json::json!({"model_type":"qwen3_moe", "vocab_size":16, "hidden_size":8, "intermediate_size":0,
        "moe_intermediate_size":6, "num_hidden_layers":2, "num_attention_heads":4, "num_key_value_heads":2, "head_dim":2,
        "max_position_embeddings":64, "rms_norm_eps":1e-5, "num_experts":5, "num_experts_per_tok":2,
        "norm_topk_prob":true, "tie_word_embeddings":false});
    let args = qwen::model_args_from_config_value(&config).unwrap();
    let (root, fixture) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let descriptor = inspection.architecture_plan().architecture_descriptor();
    let description =
        qwen::RoutedLayeredModel::<NumericBackend>::new(args.clone(), &NumericContext::default())
            .unwrap()
            .parameter_description(&NumericContext::default())
            .unwrap();
    let inputs = [
        NumericTensor::token_ids(&[1, 2, 5]),
        NumericTensor::token_ids(&[3]),
        NumericTensor::token_ids(&[4]),
    ];
    let expected = [
        ordinary(&args, &fixture, &descriptor, &inputs, false),
        ordinary(&args, &fixture, &descriptor, &inputs, true),
    ];
    assert!(expected[0].iter().zip(&expected[1]).any(|(a, b)| a
        .0
        .data
        .iter()
        .zip(&b.0.data)
        .any(|(a, b)| (a - b).abs() > 1e-5)));
    verify_partitions(&inspection, &description, &descriptor, &inputs, &expected);
}

fn verify_partitions(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    description: &eredu_runtime::ArchitectureParameterDescription,
    descriptor: &eredu_core::ArchitectureDescriptor,
    inputs: &[NumericTensor],
    expected: &[Vec<(NumericTensor, Values, Values)>; 2],
) {
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
        for (tp, pp, ep) in [
            (2, 1, 1),
            (1, 2, 1),
            (2, 2, 1),
            (1, 1, 2),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2),
        ] {
            let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
            let world = Arc::new(NumericPartitionWorld::default());
            let reference_plan = prepared_adapter::plan(None)
                .with_topology(topology)
                .with_residency(residency.clone());
            let reference_sources = partitioned_adapter::prepare_plan(
                &inspection,
                &reference_plan,
                0,
                std::time::Duration::from_secs(10),
            )
            .unwrap();
            let reference_layouts = placement::layouts(&reference_sources, &description);
            placement::verify_rejections(&reference_sources, &description);
            let capture_world = Arc::new(observer::World::default());
            let results = std::thread::scope(|scope| {
                (0..topology.world_size()).map(|rank| {
                    let (inspection,description,inputs,descriptor) = (&inspection,&description,&inputs,&descriptor);
                    let reference_layouts = &reference_layouts;
                    let capture_world = capture_world.clone();
                    let expected = &expected[0];
                    let world = world.clone(); let residency = residency.clone();
                    scope.spawn(move || {
                        let rank_topology = ParallelRankTopology::new(topology,rank).unwrap();
                        let plan = prepared_adapter::plan(None).with_topology(topology).with_residency(residency);
                        let sources = partitioned_adapter::prepare_plan(inspection,&plan,rank,std::time::Duration::from_secs(10)).unwrap();
                        let retained = placement::layouts(&sources, description);
                        assert_eq!(&retained, reference_layouts, "all cold ranks agree on the actual selected placement");
                        placement::verify(&retained, descriptor);
                        let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(description,rank_topology).unwrap();
                        let mut context = NumericContext::with_partition(layout,rank,world);
                        context.bind_checkpoint_values = true;
                        let mut executable = partitioned_adapter::routed(sources,&context,Arc::new(AtomicUsize::new(0)),None).unwrap();
                        let transport = observer::Transport::new(capture_world,rank,topology.world_size());
                        observer::run(&mut executable,&retained,descriptor,&transport,inputs,Some(expected),None);
                        [false,true].map(|edit| {
                            executable.reset().unwrap();
                            inputs.iter().enumerate().map(|(step,tokens)| {
                                let mut observer = Observe::new(descriptor,Some(rank_topology),tokens.data.len(),edit);
                                observer.retained = retained.rank(rank).cloned();
                                let output = executable.forward_observed(tokens,step==0,&mut observer).unwrap();
                                assert!(observer.active.is_none());
                                assert_eq!(observer.started.len(), observer.requested.len(), "idle local invocations must also finish");
                                (output,observer.original,observer.effective)
                            }).collect::<Vec<_>>()
                        })
                    })
                }).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
            });
            for trial in 0..2 {
                for step in 0..inputs.len() {
                    for rank in &results {
                        assert_eq!(rank[trial][step].0.shape, expected[trial][step].0.shape);
                        for (a, b) in rank[trial][step]
                            .0
                            .data
                            .iter()
                            .zip(&expected[trial][step].0.data)
                        {
                            assert!(
                                (a - b).abs() < 2e-5,
                                "logits {tp}/{pp}/{ep} trial{trial} step{step}: {a}/{b}"
                            );
                        }
                    }
                    for effective in [false, true] {
                        let expected_values = if effective {
                            &expected[trial][step].2
                        } else {
                            &expected[trial][step].1
                        };
                        let mut global = BTreeMap::new();
                        for rank in &results {
                            let values = if effective {
                                &rank[trial][step].2
                            } else {
                                &rank[trial][step].1
                            };
                            for ((path, peer, token, slot, expert, unit), value) in values {
                                let key = (path.clone(), None, *token, *slot, *expert, *unit);
                                let reference = expected_values[&key];
                                assert!(
                                    (value - reference).abs() < 5e-6,
                                    "unit {tp}/{pp}/{ep} {key:?}: {value}/{reference}"
                                );
                                if peer.is_none() || *peer == Some(0) {
                                    assert!(
                                        global.insert(key, *value).is_none(),
                                        "duplicate global unit coverage"
                                    );
                                }
                            }
                        }
                        assert_eq!(
                            global.keys().collect::<Vec<_>>(),
                            expected_values.keys().collect::<Vec<_>>(),
                            "missing global units {tp}/{pp}/{ep}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn k2_selected_partition_providers_preserve_sparse_units_and_edits_across_tp_ep_pp_residency() {
    use eredu_architectures::k2_horizon as family;
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/k2_horizon/reference.json")).unwrap();
    for name in ["moe", "mova"] {
        let config = &fixtures[name]["config"];
        let args = family::model_args_from_config_value(config).unwrap();
        let (root, _) = super::super::k2_horizon::checkpoint_fixture(config, 1.0);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let descriptor = inspection.architecture_plan().architecture_descriptor();
        let description = family::parameter_description(&args).unwrap();
        let inputs = [
            NumericTensor::token_ids(&[1, 2, 5]),
            NumericTensor::token_ids(&[3]),
            NumericTensor::token_ids(&[4]),
        ];
        let expected = [false, true].map(|edit| {
            let context = NumericContext::default();
            let architecture =
                family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
            let mut state = DeviceState::<NumericBackend, _>::create(
                architecture.state_layout().unwrap(),
                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
            )
            .unwrap();
            let units = (0..args.num_hidden_layers as usize)
                .map(|layer| architecture.construct_unit(layer, &context).unwrap())
                .collect();
            let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
            inputs
                .iter()
                .enumerate()
                .map(|(step, tokens)| {
                    let mut observer = Observe::new(&descriptor, None, tokens.data.len(), edit);
                    let output = runtime
                        .forward_with_provider_and_observer(
                            decoder::LayeredInput { tokens, mask: None },
                            &mut state,
                            if step == 0 {
                                ExpertPass::Prefill
                            } else {
                                ExpertPass::Decode
                            },
                            &mut eredu_runtime::ResidentExpertProvider,
                            &context,
                            &mut observer,
                        )
                        .unwrap();
                    assert!(!observer.original.is_empty());
                    let width = *output.shape.last().unwrap() as usize;
                    (
                        NumericTensor::new(
                            vec![1, 1, width as i32],
                            output.data[output.data.len() - width..].to_vec(),
                        ),
                        observer.original,
                        observer.effective,
                    )
                })
                .collect::<Vec<_>>()
        });
        assert!(expected[0].iter().zip(&expected[1]).any(|(a, b)| a
            .0
            .data
            .iter()
            .zip(&b.0.data)
            .any(|(a, b)| (a - b).abs() > 1e-7)));
        verify_partitions(&inspection, &description, &descriptor, &inputs, &expected);
    }
}
