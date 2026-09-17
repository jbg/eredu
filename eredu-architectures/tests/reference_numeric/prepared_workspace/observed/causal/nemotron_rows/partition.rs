//! Prepared checkpoint execution with a stateless embedding-owning PP stage.
use super::*;
use eredu_architectures::partitioned_execution::derive_partitioned_local_layout;

struct BodyRows {
    paths: Vec<String>,
    values: BTreeMap<String, Vec<NumericTensor>>,
}
impl BodyRows {
    fn new(paths: &[String]) -> Self {
        Self {
            paths: paths.to_vec(),
            values: BTreeMap::new(),
        }
    }
}
impl ActivationObserver<NumericTensor, Error> for BodyRows {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        if self.paths.iter().any(|selected| selected == path) {
            self.values
                .entry(path.into())
                .or_default()
                .push(value.clone());
        }
        Ok(())
    }
}

// The same official-source mapping as the existing Nemotron component fixture.
// The ordinary reference consumes these exact payloads, not its initialization.
struct Populate<'a>(&'a prepared_adapter::ParameterBits);
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        let canonical = metadata.id.as_str();
        let source = if let Some(rest) = canonical.strip_prefix("model.layers.") {
            let (layer, rest) = rest.split_once('.').unwrap();
            let rest = if rest.starts_with("norm.") {
                rest.to_string()
            } else {
                format!("mixer.{}", rest.split_once('.').unwrap().1)
            };
            format!("backbone.layers.{layer}.{rest}")
        } else if let Some(rest) = canonical.strip_prefix("model.") {
            format!("backbone.{rest}")
        } else {
            canonical.to_string()
        };
        let (shape, bits) = self
            .0
            .get(&source)
            .unwrap_or_else(|| panic!("source {source} for {canonical}"));
        assert_eq!(
            value.shape.iter().product::<i32>(),
            shape.iter().product::<i32>()
        );
        if !canonical.ends_with(".conv1d.weight") {
            assert_eq!(&value.shape, shape, "{canonical}");
        }
        value.data = bits
            .iter()
            .map(|bits| {
                let v = f32::from_bits(*bits);
                if canonical.ends_with(".A_log") {
                    (-v).ln()
                } else {
                    v
                }
            })
            .collect();
    }
}

#[test]
fn nemotron_prepared_stateless_leading_tp_pp_uses_downstream_cached_attention() {
    for pattern in ["--M*", "E-M*"] {
        let config = configuration(pattern, 1, false);
        let args = nemotron_h::model_args_from_config_value(&config).unwrap();
        let (artifact, payload) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                name.ends_with(".A_log").then(|| {
                    NumericTensor::new(
                        shape.to_vec(),
                        (0..shape.iter().product::<i32>())
                            .map(|i| -1.0 - 0.1 * i as f32)
                            .collect(),
                    )
                })
            });
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let context = NumericContext {
            bind_checkpoint_values: true,
            ..Default::default()
        };
        let mut architecture = HybridModel::new(args.clone(), &context).unwrap();
        let parameters = architecture.parameter_description(&context).unwrap();
        let declarations = <HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::prefill_observation_declarations(&architecture).unwrap();
        let paths = declarations
            .iter()
            .filter(|d| d.readout_stage() == Stage::BeforeReadout)
            .map(|d| d.path().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(paths.len(), 18);
        architecture
            .static_modules_mut()
            .visit_parameters_mut(&mut Populate(&payload));
        let mut state =
            HybridState::create(nemotron_h::state_layout(&args).unwrap(), |_, policy| {
                Ok::<_, Error>(NumericHybridLayerState::new(policy))
            })
            .unwrap();
        let units = (0..4)
            .map(|index| {
                let mut unit = architecture.construct_unit(0, index, &context).unwrap();
                unit.visit_parameters_mut(&mut Populate(&payload));
                unit
            })
            .collect();
        let mut reference = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
        // The second and fourth calls have several rows behind a cached prefix.
        // An offset-zero mask on the first stateless rank would fail this parity.
        let inputs = [
            NumericTensor::token_ids(&[4, 2]),
            NumericTensor::token_ids(&[1, 3]),
            NumericTensor::token_ids(&[5]),
            NumericTensor::token_ids(&[2, 6]),
            NumericTensor::token_ids(&[7]),
            NumericTensor::token_ids(&[8]),
            NumericTensor::token_ids(&[9]),
        ];
        let mut frontier = 0;
        let expected = inputs
            .iter()
            .map(|tokens| {
                let mut rows = BodyRows::new(&paths);
                let output = reference
                    .forward_with_observer(
                        nemotron_h::EmbeddedInput::target(tokens, None),
                        &mut state,
                        &context,
                        &mut rows,
                    )
                    .unwrap();
                frontier += tokens.shape[1];
                assert_populated(&state, &args, frontier);
                let vocabulary = *output.shape.last().unwrap() as usize;
                let final_scores = NumericTensor::new(
                    vec![1, 1, vocabulary as i32],
                    output.data[output.data.len() - vocabulary..].to_vec(),
                );
                assert_eq!(rows.values.len(), 18);
                for values in rows.values.values() {
                    assert_eq!(values.len(), 1);
                    assert_eq!(values[0].shape[1], tokens.shape[1]);
                    assert!(values[0].data.iter().any(|x| x.abs() > 1e-6));
                }
                (final_scores, rows, frontier)
            })
            .collect::<Vec<_>>();
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
                let results = std::thread::scope(|scope| {
                    let handles = (0..topology.world_size())
                        .map(|rank| {
                            let world = Arc::clone(&world);
                            let residency = residency.clone();
                            let inspection = &inspection;
                            let parameters = &parameters;
                            let inputs = &inputs;
                            let paths = &paths;
                            scope.spawn(move || -> Result<_, String> {
                                let rank_topology =
                                    ParallelRankTopology::new(topology, rank).unwrap();
                                let plan = prepared_adapter::plan(None)
                                    .with_topology(topology)
                                    .with_residency(residency);
                                let sources = partitioned_adapter::prepare_plan(
                                    inspection,
                                    &plan,
                                    rank,
                                    std::time::Duration::from_secs(10),
                                )?;
                                let descriptor = sources.architecture().architecture_descriptor();
                                let placement = sources
                                    .selected()
                                    .execution()
                                    .component_partition_layout(&descriptor, parameters)
                                    .map_err(|e| e.to_string())?
                                    .unwrap();
                                let owned = (0..4)
                                    .map(|layer| {
                                        // Mamba is a whole residual write. Sparse units
                                        // also have child component invocations: all must
                                        // agree with their enclosing write's placement.
                                        let mut owners = descriptor
                                            .components
                                            .iter()
                                            .filter(|c| c.layer_index == layer)
                                            .map(|component| {
                                                placement
                                                    .point(&component.activation)
                                                    .expect("component placement")
                                                    .coordinates()
                                                    .is_some()
                                            })
                                            .chain(
                                                descriptor
                                                    .component_readout
                                                    .as_ref()
                                                    .unwrap()
                                                    .other_writes
                                                    .iter()
                                                    .filter(|write| write.layer_index == layer)
                                                    .map(|write| {
                                                        placement
                                                            .observation(&write.output)
                                                            .expect(
                                                                "whole-write observation placement",
                                                            )
                                                            .coordinates()
                                                            .is_some()
                                                    }),
                                            );
                                        let owned = owners.next().expect("declared layer write");
                                        assert!(owners.all(|other| other == owned));
                                        owned
                                    })
                                    .collect::<Vec<_>>();
                                let local =
                                    derive_partitioned_local_layout(parameters, rank_topology)
                                        .map_err(|e| e.to_string())?;
                                let mut context =
                                    NumericContext::with_partition(local, rank, world);
                                context.bind_checkpoint_values = true;
                                let mut executable = if pattern.contains('E') {
                                    partitioned_adapter::routed(
                                        sources,
                                        &context,
                                        Arc::new(AtomicUsize::new(0)),
                                        None,
                                    )?
                                } else {
                                    partitioned_adapter::dense(sources, &context)?
                                };
                                let mut actual = Vec::new();
                                for (step, tokens) in inputs.iter().enumerate() {
                                    let mut rows = BodyRows::new(paths);
                                    let scores = executable
                                        .forward_observed(tokens, step < 4, &mut rows)
                                        .map_err(|e| e.to_string())?;
                                    let positions =
                                        executable.positions().map_err(|e| e.to_string())?;
                                    actual.push((scores, rows, positions));
                                }
                                Ok((owned, actual))
                            })
                        })
                        .collect::<Vec<_>>();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap())
                        .collect::<Vec<_>>()
                });
                // All bounded rank work joins before any numerical/owner assertion.
                for (rank, result) in results.into_iter().enumerate() {
                    let (owned, actual) = result.unwrap();
                    assert_eq!(actual.len(), expected.len());
                    if topology.pipeline() == 2 {
                        assert!(
                            owned == [true, true, false, false]
                                || owned == [false, false, true, true]
                        );
                    }
                    for (step, ((scores, rows, positions), (wanted, reference, frontier))) in
                        actual.iter().zip(&expected).enumerate()
                    {
                        assert_tensor_close(
                            scores,
                            wanted,
                            &format!("{pattern} rank {rank} step {step}"),
                        );
                        let expected_positions = owned
                            .iter()
                            .enumerate()
                            .filter(|(_, own)| **own)
                            .map(|(index, _)| if index < 2 { 0 } else { *frontier })
                            .collect::<Vec<_>>();
                        assert_eq!(positions, &expected_positions);
                        let embedding_owned = owned[0];
                        assert_eq!(
                            rows.values.len(),
                            owned.iter().filter(|own| **own).count() * 4
                                + usize::from(embedding_owned) * 2
                        );
                        for path in &paths {
                            let owner = if path.starts_with("readout.embedding") {
                                embedding_owned
                            } else {
                                let index = path
                                    .strip_prefix("model.layers.")
                                    .unwrap()
                                    .split_once('.')
                                    .unwrap()
                                    .0
                                    .parse::<usize>()
                                    .unwrap();
                                owned[index]
                            };
                            let value = rows.values.get(path);
                            if owner {
                                let value =
                                    value.unwrap_or_else(|| panic!("rank {rank} missing {path}"));
                                assert_eq!(value.len(), 1, "rank {rank} {path}");
                                assert_eq!(value[0].shape[1], inputs[step].shape[1]);
                                assert_tensor_close(&value[0], &reference.values[path][0], path);
                            } else {
                                assert!(value.is_none(), "rank {rank} inactive {path}");
                            }
                        }
                    }
                }
            }
        }
    }
}
