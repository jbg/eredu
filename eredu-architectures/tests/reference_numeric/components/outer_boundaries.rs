//! Actual looped invocation boundaries through retained prepared partition routes.
use super::*;

#[derive(Debug, thiserror::Error)]
#[error("effective outer input fixture failure")]
struct EffectiveInputFailure(Arc<()>);

struct OuterObserver {
    paths: Vec<String>,
    target: Option<&'static str>,
    fail: Option<Arc<()>>,
    events: Vec<(String, bool)>, // false: observe, true: intervene
    values: BTreeMap<String, Vec<NumericTensor>>,
}
impl OuterObserver {
    fn new(layers: usize, target: Option<&'static str>) -> Self {
        let paths = (0..layers)
            .flat_map(|layer| {
                ["input", "output"].into_iter().flat_map(move |boundary| {
                    let original = format!("model.layers.{layer}.{boundary}");
                    [original.clone(), format!("{original}.effective")]
                })
            })
            .collect();
        Self {
            paths,
            target,
            fail: None,
            events: Vec::new(),
            values: BTreeMap::new(),
        }
    }
}
impl ActivationObserver<NumericTensor, Error> for OuterObserver {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        if self.paths.iter().any(|p| p == path) {
            self.events.push((path.into(), false));
            self.values
                .entry(path.into())
                .or_default()
                .push(value.clone());
            if path == "model.layers.0.input.effective" {
                if let Some(cause) = self.fail.take() {
                    return Err(Error::backend_source(EffectiveInputFailure(cause)));
                }
            }
        }
        Ok(())
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        if self.paths.iter().any(|p| p == path) {
            self.events.push((path.into(), true));
        }
        Ok((self.target == Some(path)).then(|| {
            NumericTensor::new(value.shape.clone(), vec![1.25; value.data.len()])
                .with_dtype(value.dtype.clone())
        }))
    }
}

fn residencies() -> [eredu_core::ResidencyPlan; 3] {
    [
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
    ]
}

#[test]
fn prepared_looped_outer_replacements_are_once_only_on_actual_tp_pp_owners() {
    use eredu_architectures::partitioned_execution::derive_partitioned_local_layout;
    let mut config = super::super::nanbeige::tiny_config(false);
    config["head_dim"] = 2.into();
    let args = eredu_architectures::nanbeige::model_args_from_config_value(&config).unwrap();
    let layers = args.state_layer_count();
    let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let parameters = decoder::dense_parameter_description(&args).unwrap();
    let inputs = [
        NumericTensor::token_ids(&[1, 2, 5]),
        NumericTensor::token_ids(&[3]),
        NumericTensor::token_ids(&[4]),
    ];
    let targets = [
        None,
        Some("model.layers.0.input"),
        Some("model.layers.2.output"),
    ];
    let reference = targets
        .iter()
        .map(|&target| {
            let context = NumericContext::default();
            let architecture = eredu_architectures::nanbeige::LayeredModel::<NumericBackend>::new(
                args.clone(),
                &context,
            )
            .unwrap();
            let mut state = DeviceState::<NumericBackend, _>::create(
                architecture.state_layout().unwrap(),
                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
            )
            .unwrap();
            let units = (0..layers)
                .map(|index| {
                    let mut unit = architecture.construct_unit(index, &context).unwrap();
                    unit.visit_parameters_mut(&mut super::super::nanbeige::FixtureAliases);
                    unit
                })
                .collect();
            let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
            inputs
                .iter()
                .map(|tokens| {
                    let mut observer = OuterObserver::new(layers, target);
                    let output = runtime
                        .forward_with_observer(
                            decoder::LayeredInput { tokens, mask: None },
                            &mut state,
                            &context,
                            &mut observer,
                        )
                        .unwrap();
                    let vocabulary = *output.shape.last().unwrap() as usize;
                    let output = NumericTensor::new(
                        vec![1, 1, vocabulary as i32],
                        output.data[output.data.len() - vocabulary..].to_vec(),
                    );
                    for values in observer.values.values() {
                        assert_eq!(values.len(), 1);
                    }
                    assert_eq!(observer.values.len(), layers * 4);
                    if let Some(path) = target {
                        assert!(observer.values[path][0]
                            .data
                            .iter()
                            .any(|&v| (v - 1.25).abs() > 1e-4));
                        assert!(observer.values[&format!("{path}.effective")][0]
                            .data
                            .iter()
                            .all(|&v| v == 1.25));
                    } else {
                        assert!(observer
                            .values
                            .values()
                            .all(|v| v[0].data.iter().any(|&x| x != 0.0)));
                    }
                    (output, observer)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for residency in residencies() {
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
                        scope.spawn(move || -> Result<_, String> {
                            let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
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
                            // Every logical Nanbeige unit owns its component invocations,
                            // even when later loops reuse the same physical parameters.
                            // The component capture map need not expose every outer hook.
                            let owned_units = (0..layers)
                                .map(|layer| {
                                    let component = descriptor
                                        .components
                                        .iter()
                                        .find(|c| c.layer_index == layer)
                                        .unwrap();
                                    placement
                                        .point(&component.activation)
                                        .unwrap()
                                        .coordinates()
                                        .is_some()
                                })
                                .collect::<Vec<_>>();
                            let local = derive_partitioned_local_layout(parameters, rank_topology)
                                .map_err(|e| e.to_string())?;
                            let mut context = NumericContext::with_partition(local, rank, world);
                            context.bind_checkpoint_values = true;
                            let mut executable = partitioned_adapter::dense(sources, &context)?;
                            let mut trials = Vec::new();
                            for target in targets {
                                executable.reset().map_err(|e| e.to_string())?;
                                let mut steps = Vec::new();
                                for (step, tokens) in inputs.iter().enumerate() {
                                    let mut observer = OuterObserver::new(layers, target);
                                    let output = executable
                                        .forward_observed(tokens, step == 0, &mut observer)
                                        .map_err(|e| e.to_string())?;
                                    steps.push((output, observer));
                                }
                                trials.push(steps);
                            }
                            Ok((placement, owned_units, trials))
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });
            // No rank-local assertion/barrier can strand a peer: all ranks join
            // before checking values, callback counts or source-derived ownership.
            for (rank, result) in results.into_iter().enumerate() {
                let (placement, owned_units, trials) = result.unwrap();
                for (trial, steps) in trials.into_iter().enumerate() {
                    for (step, (output, observer)) in steps.into_iter().enumerate() {
                        assert_tensor_close(
                            &output,
                            &reference[trial][step].0,
                            "outer replacement cached logits",
                        );
                        let mut expected_events = Vec::new();
                        for layer in 0..layers {
                            for boundary in ["input", "output"] {
                                let original = format!("model.layers.{layer}.{boundary}");
                                let effective = format!("{original}.effective");
                                let owned = owned_units[layer];
                                if let Some(mapped) = placement.observation(&original) {
                                    assert_eq!(mapped.coordinates().is_some(), owned);
                                    assert_eq!(
                                        placement
                                            .observation(&effective)
                                            .unwrap()
                                            .coordinates()
                                            .is_some(),
                                        owned
                                    );
                                }
                                for path in [&original, &effective] {
                                    let actual = observer.values.get(path);
                                    if owned {
                                        let actual = actual.unwrap_or_else(|| {
                                            panic!("rank {rank} missing {path}")
                                        });
                                        assert_eq!(actual.len(), 1, "rank {rank} duplicate {path}");
                                        assert_tensor_close(
                                            &actual[0],
                                            &reference[trial][step].1.values[path][0],
                                            path,
                                        );
                                    } else {
                                        assert!(actual.is_none(), "rank {rank} inactive {path}");
                                    }
                                }
                                if owned {
                                    expected_events.extend([
                                        (original.clone(), false),
                                        (original, true),
                                        (effective, false),
                                    ]);
                                }
                            }
                        }
                        assert_eq!(
                            observer.events, expected_events,
                            "rank {rank} actual invocation order"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn prepared_effective_input_failure_precedes_state_and_releases_partition_policy_loan() {
    use eredu_architectures::partitioned_execution::derive_partitioned_local_layout;
    let mut config = super::super::nanbeige::tiny_config(false);
    config["head_dim"] = 2.into();
    let args = eredu_architectures::nanbeige::model_args_from_config_value(&config).unwrap();
    let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let parameters = decoder::dense_parameter_description(&args).unwrap();
    for residency in residencies() {
        let topology = ParallelTopology::new(1, 2, 1, 1).unwrap();
        let world = Arc::new(NumericPartitionWorld::default());
        let identity = Arc::new(());
        let results = std::thread::scope(|scope| {
            let handles = (0..topology.world_size())
                .map(|rank| {
                    let world = Arc::clone(&world);
                    let identity = identity.clone();
                    let residency = residency.clone();
                    let inspection = &inspection;
                    let parameters = &parameters;
                    scope.spawn(move || -> Result<_, String> {
                        let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                        let plan = prepared_adapter::plan(None)
                            .with_topology(topology)
                            .with_residency(residency);
                        let sources = partitioned_adapter::prepare_plan(
                            inspection,
                            &plan,
                            rank,
                            std::time::Duration::from_secs(10),
                        )?;
                        let local = derive_partitioned_local_layout(parameters, rank_topology)
                            .map_err(|e| e.to_string())?;
                        let mut context = NumericContext::with_partition(local, rank, world);
                        context.bind_checkpoint_values = true;
                        let mut executable = partitioned_adapter::dense(sources, &context)?;
                        let before = executable.positions().map_err(|e| e.to_string())?;
                        let tokens = NumericTensor::token_ids(&[1, 2, 5]);
                        let mut observer = OuterObserver::new(4, Some("model.layers.0.input"));
                        observer.fail = Some(identity);
                        let error = executable
                            .forward_observed(&tokens, true, &mut observer)
                            .err();
                        let after = executable.positions().map_err(|e| e.to_string())?;
                        // The original local observer rejection is agreed negative;
                        // a healthy retry exercises return of the policy lease.
                        let retry = executable.forward(&tokens, true).map_err(|e| e.to_string());
                        Ok((before, after, observer, error, retry))
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        for (rank, result) in results.into_iter().enumerate() {
            let (before, after, observer, error, retry) = result.unwrap();
            assert_eq!(after, before, "no state work after effective input failure");
            let error = error.expect("actual observer rejection must propagate");
            if rank == 0 {
                let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&error);
                let mut original = None;
                while let Some(error) = source {
                    if let Some(found) = error.downcast_ref::<EffectiveInputFailure>() {
                        original = Some(found);
                        break;
                    }
                    source = error.source();
                }
                assert!(Arc::ptr_eq(
                    &original.expect("original typed observer cause").0,
                    &identity
                ));
                assert_eq!(
                    observer.events,
                    vec![
                        ("model.layers.0.input".into(), false),
                        ("model.layers.0.input".into(), true),
                        ("model.layers.0.input.effective".into(), false)
                    ]
                );
            } else {
                assert!(observer.events.is_empty());
            }
            let output = retry.expect("agreed rejection returns the policy loan");
            assert!(output.data.iter().any(|&v| v != 0.0));
        }
    }
}
