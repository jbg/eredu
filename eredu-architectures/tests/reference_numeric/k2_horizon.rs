use super::*;
use eredu_architectures::k2_horizon as family;

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("../fixtures/k2_horizon/reference.json")).unwrap()
}

// Serialize the exact logical oracle matrices into the publisher's individual
// expert layout. The packed oracle name supplies a single reproducible matrix.
fn checkpoint_fixture(
    config: &serde_json::Value,
    scale: f32,
) -> (tempfile::TempDir, prepared_adapter::ParameterBits) {
    prepared_adapter::payload_fixture_config_with(config, scale, |name, _| {
        let args = family::model_args_from_config_value(config).unwrap();
        if name.ends_with(".mlp.gate.bias") || name.ends_with(".self_attn.v_router.bias") {
            let count = if name.ends_with(".mlp.gate.bias") {
                args.num_experts
            } else {
                args.mova_num_experts
            };
            return Some(parameter(
                &ParameterSpec::trainable(name).unwrap(),
                vec![count],
                true,
            ));
        }
        let packed =
            |name: String, shape| parameter(&ParameterSpec::trainable(name).unwrap(), shape, false);
        if let Some((root, suffix)) = name.split_once(".mlp.experts.") {
            let (expert, projection) = suffix.split_once('.')?;
            let expert = expert.parse::<usize>().ok()?;
            let width = args.moe_intermediate_size;
            let (matrix, range) = match projection {
                "gate_proj.weight" => (
                    packed(
                        format!("{root}.mlp.experts.gate_up_proj"),
                        vec![args.num_experts, 2 * width, args.hidden_size],
                    ),
                    Some(0..width as usize),
                ),
                "up_proj.weight" => (
                    packed(
                        format!("{root}.mlp.experts.gate_up_proj"),
                        vec![args.num_experts, 2 * width, args.hidden_size],
                    ),
                    Some(width as usize..2 * width as usize),
                ),
                "down_proj.weight" => (
                    packed(
                        format!("{root}.mlp.experts.down_proj"),
                        vec![args.num_experts, args.hidden_size, width],
                    ),
                    None,
                ),
                _ => return None,
            };
            let mut matrix = matrix.axis_slice(0, expert, expert + 1);
            if let Some(range) = range {
                matrix = matrix.axis_slice(1, range.start, range.end);
            }
            return Some(NumericTensor::new(matrix.shape[1..].to_vec(), matrix.data));
        }
        if let Some((root, suffix)) = name.split_once(".self_attn.v_experts.") {
            let expert = suffix.strip_suffix(".weight")?.parse::<usize>().ok()?;
            let matrix = packed(
                format!("{root}.self_attn.v_experts.weight"),
                vec![
                    args.mova_num_experts,
                    args.num_key_value_heads * args.head_dim,
                    args.hidden_size,
                ],
            )
            .axis_slice(0, expert, expert + 1);
            return Some(NumericTensor::new(matrix.shape[1..].to_vec(), matrix.data));
        }
        None
    })
}

#[test]
fn k2_registered_mova_executes_distinct_resident_and_addressable_banks() {
    let fixture = fixture();
    let case = &fixture["mova"];
    let context = NumericContext::default();
    let tokens = NumericTensor::token_ids(&[1, 3, 2]);
    let resident = execute_numeric_routed_visitor(&case["config"], &context, &tokens, false);
    let bounded_runs = [
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
    .into_iter()
    .map(|residency| {
        execute_numeric_routed_visitor_with_policy(
            &case["config"],
            &context,
            &tokens,
            true,
            None,
            residency,
        )
    })
    .collect::<Vec<_>>();
    let expected = case["logits"]
        .as_array()
        .unwrap()
        .iter()
        .take(5)
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
        })
        .collect::<Vec<_>>();
    for run in std::iter::once(&resident).chain(&bounded_runs) {
        let actual = run
            .outputs
            .iter()
            .flat_map(|x| x.data.iter())
            .copied()
            .collect::<Vec<_>>();
        // Sessions return only the last prefill position, followed by decode.
        let expected = [
            &expected[2 * 17..3 * 17],
            &expected[3 * 17..4 * 17],
            &expected[4 * 17..5 * 17],
        ]
        .concat();
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() <= 1e-5 + 1e-4 * e.abs(), "{a} != {e}");
        }
    }
    for bounded in &bounded_runs {
        assert_state_exact(
            &resident.state,
            &bounded.state,
            3,
            "registered K2 ordinary KV",
        );
        assert_eq!(bounded.bank_reports.len(), 2);
        for (id, report) in &bounded.bank_reports {
            assert!(report.acquisitions > 2, "{id:?}");
            assert_eq!(report.acquisitions, report.completions);
            assert!(report.evictions > 0, "{id:?}");
            assert!(
                report.peak_entries
                    <= if *id == family::ExpertBank::AttentionValue.id() {
                        3
                    } else {
                        2
                    }
            );
        }
    }
}

#[test]
fn k2_dense_prepared_payload_tp_pp_combined_and_bounded_match_four_decode_steps() {
    let fixture = fixture();
    for name in ["dense", "grouped"] {
        let case = &fixture[name];
        let args = family::model_args_from_config_value(&case["config"]).unwrap();
        let (root, _) = checkpoint_fixture(&case["config"], 1.0);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let inputs = [
            NumericTensor::token_ids(&[1, 3, 2]),
            NumericTensor::token_ids(&[4]),
            NumericTensor::token_ids(&[5]),
            NumericTensor::token_ids(&[6]),
            NumericTensor::token_ids(&[7]),
        ];
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
                ParallelTopology::new(1, 3, 1, 1).unwrap(),
            ] {
                let world = Arc::new(NumericPartitionWorld::default());
                let actual = std::thread::scope(|scope| {
                    let threads = (0..topology.world_size()).map(|rank| {
                        let world = Arc::clone(&world);
                        let args = &args;
                        let inspection = &inspection;
                        let inputs = &inputs;
                        let residency = residency.clone();
                        scope.spawn(move || {
                            use eredu_architectures::partitioned_execution::derive_partitioned_local_layout;
                            let plan = prepared_adapter::plan(None).with_topology(topology).with_residency(residency);
                            let sources = partitioned_adapter::prepare_plan(inspection, &plan, rank,
                                std::time::Duration::from_secs(10)).unwrap();
                            let description = decoder::dense_parameter_description(args).unwrap();
                            let layout = derive_partitioned_local_layout(&description,
                                ParallelRankTopology::new(topology, rank).unwrap()).unwrap();
                            let mut context = NumericContext::with_partition(layout, rank, world);
                            context.bind_checkpoint_values = true;
                            let mut executable = partitioned_adapter::dense(sources, &context).unwrap();
                            let output = inputs.iter().enumerate().map(|(step, input)|
                                executable.forward(input, step == 0).unwrap()).collect::<Vec<_>>();
                            assert!(executable.positions().unwrap().iter().all(|p| *p == 7));
                            executable.reset().unwrap();
                            assert_tensor_exact(&executable.forward(&inputs[0], true).unwrap(), &output[0], "K2 partition reset");
                            output
                        })
                    }).collect::<Vec<_>>();
                    threads
                        .into_iter()
                        .map(|thread| thread.join().unwrap())
                        .collect::<Vec<_>>()
                });
                for rank in actual {
                    for (step, output) in rank.iter().enumerate() {
                        let expected = case["logits"][step + 2].as_array().unwrap();
                        assert_eq!(output.data.len(), expected.len());
                        for (&a, e) in output.data.iter().zip(expected) {
                            let e = e.as_f64().unwrap() as f32;
                            assert!(
                                (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
                                "{name} {topology:?} {residency:?} step {step}: {a} != {e}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn k2_dense_grouped_moe_mova_prefill_chunking_and_four_cached_steps_match_independent_oracle() {
    let fixture = fixture();
    let context = NumericContext::default();
    for name in ["dense", "grouped", "moe", "mova"] {
        let case = &fixture[name];
        let args = family::model_args_from_config_value(&case["config"]).unwrap();
        let layout = decoder::state_layout(&args).unwrap();
        let state = || {
            DeviceState::<NumericBackend, _>::create(layout.clone(), |_, policy| {
                Ok::<_, Error>(NumericHybridLayerState::new(policy))
            })
            .unwrap()
        };
        let expected = NumericTensor::new(
            vec![1, 7, 17],
            case["logits"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap() as f32)
                })
                .collect(),
        );
        let mut runtime = ResidentRuntime::new(
            family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
            &context,
        )
        .unwrap();
        for chunks in [vec![7], vec![3, 1, 1, 1, 1], vec![2, 1, 2, 2]] {
            let mut state = state();
            let mut output = Vec::new();
            let mut start = 0;
            for length in chunks {
                let tokens =
                    NumericTensor::token_ids(&[1, 3, 2, 4, 5, 6, 7][start..start + length]);
                output.extend(
                    runtime
                        .forward(
                            decoder::LayeredInput {
                                tokens: &tokens,
                                mask: None,
                            },
                            &mut state,
                            &context,
                        )
                        .unwrap()
                        .data,
                );
                start += length;
                for layer in 0..layout.len() {
                    assert_eq!(state.layer(layer).unwrap().position(), start as i32);
                }
            }
            let actual = NumericTensor::new(vec![1, 7, 17], output);
            assert_eq!(actual.shape, expected.shape);
            for (i, (&a, &e)) in actual.data.iter().zip(&expected.data).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
                    "{name} logit {i}: {a} != {e}"
                );
            }
        }
    }
}

#[derive(Default, Debug)]
struct BankRecorder {
    calls: Vec<(usize, NumericTensor, NumericTensor, NumericTensor)>,
}
impl BankRecorder {
    fn record(&mut self, request: &RoutedExpertRequest<'_, NumericTensor>) {
        self.calls.push((
            request.layer,
            request.routes.group_indices().clone(),
            request.routes.selected_scores().clone(),
            request.routes.coefficients().clone(),
        ));
    }
}
impl RoutedExpertProvider<NumericBackend> for BankRecorder {
    type Error = Error;
    fn forward_grouped(
        &mut self,
        bank: &mut NumericExpertBank,
        request: RoutedExpertRequest<'_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.record(&request);
        bank.forward_grouped(request.input, request.routes, context)
    }
    fn forward_linear_routed(
        &mut self,
        bank: &mut grouped_linear::NumericLinearGroups,
        request: RoutedExpertRequest<'_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.record(&request);
        bank.forward_grouped(request.input, request.routes, context)
    }
    fn forward_relu2_routed(
        &mut self,
        bank: &mut NumericRelu2Groups,
        request: RoutedExpertRequest<'_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.record(&request);
        bank.forward_grouped(request.input, request.routes, context)
    }
}
#[derive(Default)]
struct RoutingObserver(Vec<(String, i32)>);
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for RoutingObserver {
    fn observe(&mut self, _: &str, _: &NumericTensor) -> Result<(), Error> {
        Ok(())
    }
    fn observe_routing(
        &mut self,
        event: eredu_runtime::RoutingObservation<'_, NumericTensor>,
    ) -> Result<(), Error> {
        self.0.push((event.path.to_owned(), event.expert_count));
        Ok(())
    }
}

#[test]
fn k2_observed_provider_separates_value_and_feed_forward_routes_and_coefficients() {
    let fixture = fixture();
    let case = &fixture["mova"];
    let context = NumericContext::default();
    let args = family::model_args_from_config_value(&case["config"]).unwrap();
    let layout = decoder::state_layout(&args).unwrap();
    let mut state = DeviceState::<NumericBackend, _>::create(layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let architecture = family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let units = (0..3)
        .map(|layer| family::new_block::<NumericBackend>(&args, layer, &context).unwrap())
        .collect();
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let mut providers = eredu_runtime::RoutedBankProviders::new([
        (
            family::ExpertBank::FeedForward.id(),
            BankRecorder::default(),
        ),
        (
            family::ExpertBank::AttentionValue.id(),
            BankRecorder::default(),
        ),
    ])
    .unwrap();
    let mut observer = RoutingObserver::default();
    let tokens = NumericTensor::token_ids(&[1, 3, 2, 4, 5, 6, 7]);
    runtime
        .forward_with_provider_and_observer(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut state,
            ExpertPass::Prefill,
            &mut providers,
            &context,
            &mut observer,
        )
        .unwrap();
    for bank in [
        family::ExpertBank::AttentionValue,
        family::ExpertBank::FeedForward,
    ] {
        let records = &providers.bank(bank.id()).unwrap().calls;
        assert_eq!(records.len(), 2);
        for (layer, ids, scores, coefficients) in records {
            let expected = &case["routes"][format!("{layer}:{}", bank.id().value())];
            for (field, actual) in [
                ("ids", ids),
                ("scores", scores),
                ("coefficients", coefficients),
            ] {
                let expected = expected[field]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| {
                        row.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap() as f32)
                    })
                    .collect::<Vec<_>>();
                for (&a, &e) in actual.data.iter().zip(&expected) {
                    assert!(
                        (a - e).abs() < 1e-6,
                        "{bank:?} layer {layer} {field}: {a} != {e}"
                    );
                }
            }
        }
    }
    assert_eq!(
        observer.0,
        vec![
            ("model.layers.1.self_attn.values".into(), 3),
            ("model.layers.1.mlp".into(), 5),
            ("model.layers.2.self_attn.values".into(), 3),
            ("model.layers.2.mlp".into(), 5)
        ]
    );
}

#[test]
fn k2_tensor_parallel_keeps_complete_value_projections_and_local_kv_state() {
    for name in ["dense", "grouped", "moe", "mova"] {
        let fixture = fixture();
        let args = family::model_args_from_config_value(&fixture[name]["config"]).unwrap();
        let context = NumericContext::default();
        let architecture =
            family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
        let cold = family::parameter_description(&args).unwrap();
        let constructed = architecture.parameter_description(&context).unwrap();
        let members = |description: &eredu_runtime::ArchitectureParameterDescription| {
            description
                .groups()
                .iter()
                .flat_map(|owned| {
                    owned.group().members().iter().map(move |member| {
                        (
                            member.target().to_owned(),
                            (owned.owner().clone(), owned.group().role(), member.clone()),
                        )
                    })
                })
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        assert_eq!(
            members(&cold),
            members(&constructed),
            "{name}: cold and constructed physical topology"
        );
        let mut groups = decoder::static_parallel_parameter_groups::<NumericBackend>(
            &architecture.static_modules().embeddings,
            &architecture.static_modules().norm,
            architecture.static_modules().lm_head.as_ref(),
            "model",
        )
        .unwrap();
        for layer in 0..3 {
            let block = family::new_block::<NumericBackend>(&args, layer, &context).unwrap();
            groups.extend(family::block_parameter_groups(&block, &args, layer).unwrap());
        }
        let collective = NumericParallelGroup::new(2);
        let results = std::thread::scope(|scope| {
            let handles = (0..2)
                .map(|rank| {
                    let layout = numeric_local_layout(&groups, 2, rank).unwrap();
                    let collective = Arc::clone(&collective);
                    let args = args.clone();
                    scope.spawn(move || {
                        let context = NumericContext::with_local_layout(layout.clone());
                        let geometry =
                            decoder::local_geometry(&args, &layout, family::local_block_args)
                                .unwrap();
                        let state_layout = geometry.state_layout().clone();
                        if args.is_mova_layer(1) {
                            let value = layout
                                .tensor("model.layers.1.self_attn.v_experts.weight")
                                .unwrap();
                            assert_eq!(value.local_shape(), [3, 4, 8]);
                        }
                        let architecture = family::LayeredModel::<NumericBackend>::new_parallel(
                            args.clone(),
                            geometry,
                            &context,
                        )
                        .unwrap();
                        let units = (0..3)
                            .map(|layer| {
                                <family::LayeredModel<NumericBackend> as LayeredArchitecture<
                                    NumericBackend,
                                    DeviceState<NumericBackend, NumericHybridLayerState>,
                                >>::build_unit(
                                    &architecture, 0, layer, &context
                                )
                                .unwrap()
                            })
                            .collect();
                        let mut runtime =
                            LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                        let mut state =
                            DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                                Ok::<_, Error>(NumericHybridLayerState::new(policy))
                            })
                            .unwrap();
                        let parallel = NumericParallelContext::new(rank, collective);
                        let inputs = [1, 3, 2, 4, 5, 6, 7];
                        let mut logits = Vec::new();
                        for tokens in std::iter::once(&inputs[..3]).chain(inputs[3..].chunks(1)) {
                            let tokens = NumericTensor::token_ids(tokens);
                            logits.extend(
                                runtime
                                    .forward_parallel(
                                        decoder::LayeredInput {
                                            tokens: &tokens,
                                            mask: None,
                                        },
                                        &mut state,
                                        &parallel,
                                        &context,
                                    )
                                    .unwrap()
                                    .data,
                            );
                        }
                        for layer in 0..3 {
                            assert_eq!(state.layer(layer).unwrap().position(), 7);
                        }
                        logits
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        let expected = fixture[name]["logits"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
            })
            .collect::<Vec<_>>();
        for (rank, actual) in results.iter().enumerate() {
            assert_eq!(actual.len(), expected.len());
            for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
                    "{name} TP rank {rank} logit {i}: {a} != {e}"
                );
            }
        }
    }
}

#[test]
fn k2_pipeline_cuts_before_and_after_dense_to_mova_transition_match_cached_reference() {
    use eredu_runtime::{LayeredPartitionDriver, LayeredPartitionInput, LayeredPartitionOutput};
    let fixture = fixture();
    let args = family::model_args_from_config_value(&fixture["mova"]["config"]).unwrap();
    let context = NumericContext::default();
    let model = family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let description = model.parameter_description(&context).unwrap();
    let topology = ParallelTopology::new(1, 2, 1, 1).unwrap();
    let rank = ParallelRankTopology::new(topology, 0).unwrap();
    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
        &description,
        rank,
    )
    .unwrap();
    let context = NumericContext::with_local_layout(layout.clone());
    for cut in [1, 2] {
        let mut stages = [0..cut, cut..3]
            .into_iter()
            .enumerate()
            .map(|(stage, owned)| {
                let geometry =
                    decoder::partition_local_geometry(&args, &layout, owned.clone()).unwrap();
                let state_layout = geometry.complete_state_layout().clone();
                let state_plan = eredu_runtime::ArchitectureStatePartitionPlan::new([
                    eredu_runtime::ArchitectureStatePartitionRule::group_units(0, 0..3),
                ]);
                let roles = if stage == 0 {
                    vec!["embedding"]
                } else {
                    vec!["norm", "output"]
                };
                let partition = ArchitecturePartition::from_description(
                    &description,
                    [(decoder::TEXT_DECODER_EXECUTION_GROUP, owned.clone())],
                    eredu_runtime::PartitionOwnership::new(stage == 0, stage == 1, roles).unwrap(),
                    &state_layout,
                    &state_plan,
                    geometry,
                    eredu_runtime::NoAuxiliaryBoundarySchema::new(args.hidden_size),
                )
                .unwrap();
                let architecture =
                    family::PartitionedLayeredModel::<NumericBackend>::from_partition(
                        args.clone(),
                        &description,
                        &partition,
                        &context,
                    )
                    .unwrap();
                let units = owned
                    .clone()
                    .map(|layer| architecture.construct_unit(layer, &context).unwrap())
                    .collect::<Vec<_>>();
                let state = DeviceState::<NumericBackend, _>::create(
                    partition.state().unwrap().layout().clone(),
                    |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
                )
                .unwrap();
                let driver = LayeredPartitionDriver::new(&partition, 0, owned).unwrap();
                (architecture, units, state, driver)
            })
            .collect::<Vec<_>>();
        let inputs = [1, 3, 2, 4, 5, 6, 7];
        let mut actual = Vec::new();
        for tokens in std::iter::once(&inputs[..3]).chain(inputs[3..].chunks(1)) {
            let tokens = NumericTensor::token_ids(tokens);
            let mut boundary = None;
            for (stage, (architecture, units, state, driver)) in stages.iter_mut().enumerate() {
                let input = if stage == 0 {
                    LayeredPartitionInput::Tokens(&tokens)
                } else {
                    LayeredPartitionInput::Hidden {
                        hidden: boundary.take().unwrap(),
                        auxiliary: eredu_runtime::NoAuxiliaryBoundary,
                    }
                };
                let mut forward = driver
                    .begin::<NumericBackend, _, _>(architecture, input, None, state, None, &context)
                    .unwrap();
                for (layer, unit) in driver.range().zip(units) {
                    forward.hidden = architecture
                        .forward_unit(
                            0,
                            layer,
                            unit,
                            &forward.hidden,
                            state,
                            &mut forward.context,
                            &context,
                        )
                        .unwrap();
                }
                match driver
                    .finish::<NumericBackend, _, _>(
                        architecture,
                        &forward.hidden,
                        state,
                        &mut forward.context,
                        None,
                        &context,
                    )
                    .unwrap()
                {
                    LayeredPartitionOutput::Boundary { hidden, .. } => boundary = Some(hidden),
                    LayeredPartitionOutput::Final { output, .. } => actual.extend(output.data),
                }
            }
        }
        let expected = fixture["mova"]["logits"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(&expected) {
            assert!(
                (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
                "pipeline cut {cut}: {a} != {e}"
            );
        }
    }
}

struct LinearBankMechanism {
    source: grouped_linear::NumericLinearGroups,
    resident: Vec<ParameterBankKey>,
    report: Arc<std::sync::Mutex<NumericBankReport>>,
}
impl AddressableGroupedBank<NumericBackend> for LinearBankMechanism {
    type Acquisition = Vec<usize>;
    type Report = NumericBankReport;
    type Error = Error;
    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        (key.unit() == 1 && key.member() < 3).then_some(100)
    }
    fn acquire(
        &mut self,
        request: ParameterBankAcquisition<'_>,
        _: &NumericContext,
    ) -> Result<Self::Acquisition, Error> {
        assert!(request.entries().len() <= 2);
        let mut report = self.report.lock().unwrap();
        report.acquisitions += 1;
        report.peak_entries = report.peak_entries.max(request.entries().len());
        for &(key, _) in request.entries() {
            if !self.resident.contains(&key) {
                if self.resident.len() == 2 {
                    self.resident.remove(0);
                    report.evictions += 1;
                }
                self.resident.push(key);
            }
        }
        report.peak_resident = report.peak_resident.max(self.resident.len());
        Ok(request
            .entries()
            .iter()
            .map(|(key, _)| key.member())
            .collect())
    }
    fn gated_product_groups(
        &mut self,
        _: &Self::Acquisition,
        _: &GroupedGatedProductSpec,
        _: &NumericContext,
    ) -> Result<NumericExpertBank, Error> {
        Err(Error::backend("linear acquisition cannot construct SwiGLU"))
    }
    fn relu2_groups(
        &mut self,
        _: &Self::Acquisition,
        _: &GroupedRelu2Spec,
        _: &NumericContext,
    ) -> Result<NumericRelu2Groups, Error> {
        Err(Error::backend(
            "linear acquisition cannot construct ReLU squared",
        ))
    }
    fn linear_groups(
        &mut self,
        ids: &Self::Acquisition,
        spec: &eredu_nn::GroupedLinearSpec,
        _: &NumericContext,
    ) -> Result<grouped_linear::NumericLinearGroups, Error> {
        self.source.selected(spec, ids)
    }
    fn complete(
        &mut self,
        _: Self::Acquisition,
        _: &NumericTensor,
        _: &NumericContext,
    ) -> Result<(), Error> {
        self.report.lock().unwrap().completions += 1;
        Ok(())
    }
    fn report(&self) -> Result<Self::Report, Error> {
        Ok(*self.report.lock().unwrap())
    }
}

#[test]
fn bounded_value_bank_chunks_prefill_union_and_reacquires_with_complete_projection_outputs() {
    let fixture = fixture();
    let args = family::model_args_from_config_value(&fixture["mova"]["config"]).unwrap();
    let context = NumericContext::default();
    let spec = family::value_expert_spec(&args, 1)
        .unwrap()
        .partition_output(0..4)
        .unwrap();
    let mut resident = NumericBackend::grouped_linear_bank(spec.clone(), &context).unwrap();
    let owner = ExecutionGroupId::new("text_decoder").unwrap();
    let plan = ExpertRealizationPlan::balanced(
        3,
        ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
        BTreeMap::from([((owner.clone(), 1), spec.clone())]),
    )
    .unwrap();
    let catalog = ExpertResidencyCatalog::new((0..3).map(|expert| {
        ExpertResidencyUnit::new(
            ParameterBankKey::new(0, 1, expert),
            owner.clone(),
            1,
            "model.layers.1.self_attn.values",
            ExpertResidencyDistribution::ExpertParallel,
            [ExpertParameterRecipe::new(
                "weight",
                "model.layers.1.self_attn.v_experts.weight",
                eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                    format!("expert.{expert}.weight"),
                    eredu_checkpoint::store::TensorSelection::Full,
                ),
                ExpertParameterRole::Preserved,
            )
            .unwrap()],
        )
        .unwrap()
        .with_byte_len(100)
        .unwrap()
    }))
    .unwrap();
    let bytes = (0..3)
        .map(|expert| (ParameterBankKey::new(0, 1, expert), 100))
        .collect();
    let report = Arc::new(std::sync::Mutex::new(NumericBankReport::default()));
    let provider =
        eredu_architectures::routed_text::PlannedAddressableLinear::<NumericBackend, _, _>::new(
            owner,
            plan,
            catalog,
            bytes,
            LinearBankMechanism {
                source: resident.clone(),
                resident: Vec::new(),
                report: Arc::clone(&report),
            },
            NumericIndexedMovement,
            ParameterBankLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(300), Some(200), 1).unwrap(),
                200,
                200,
            )
            .unwrap(),
            2,
        )
        .unwrap();
    let mut providers = eredu_runtime::RoutedBankProviders::new([(
        family::ExpertBank::AttentionValue.id(),
        Box::new(provider)
            as Box<
                dyn RoutedExpertProvider<
                    NumericBackend,
                    Error = eredu_architectures::routed_text::RoutedTextExecutionError,
                >,
            >,
    )])
    .unwrap();
    let input = NumericTensor::new(
        vec![3, 8],
        (0..24).map(|i| (i as f32 - 8.0) / 9.0).collect(),
    );
    let routes = GroupSelection::new(
        NumericTensor::new(vec![3, 2], vec![0., 1., 2., 0., 1., 2.]),
        NumericTensor::new(vec![3, 2], vec![0.6, 0.4, 0.7, 0.3, 0.8, 0.2]),
        NumericTensor::new(vec![3, 2], vec![0.6, 0.4, 0.7, 0.3, 0.8, 0.2]),
    );
    let expected = resident.forward_grouped(&input, &routes, &context).unwrap();
    assert_eq!(expected.shape, [3, 4]);
    for _ in 0..2 {
        let actual = providers
            .forward_linear_routed(
                &mut resident,
                RoutedExpertRequest {
                    bank: family::ExpertBank::AttentionValue.id(),
                    layer: 1,
                    input: &input,
                    routes: &routes,
                    pass: ExpertPass::Prefill,
                },
                &context,
            )
            .unwrap();
        assert_tensor_close(&actual, &expected, "bounded activated values");
    }
    let report = report.lock().unwrap();
    assert_eq!(report.acquisitions, report.completions);
    assert!(report.acquisitions >= 4);
    assert!(report.evictions > 0);
    assert_eq!(report.peak_resident, 2);
    assert!(report.peak_entries <= 2);
}

#[test]
fn k2_mova_collective_waves_keep_bank_order_ownership_and_value_output_width() {
    use eredu_architectures::decoder::PartitionedConfig;
    use eredu_architectures::partitioned_execution::{
        RoutedExpertCollectiveWaveSchedule, RoutedExpertUnitWave, RoutedExpertWaveOperation,
    };
    let args = family::model_args_from_config_value(&fixture()["mova"]["config"]).unwrap();
    let context = NumericContext::default();
    let model = family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let description = model.parameter_description(&context).unwrap();
    let owner = ExecutionGroupId::new("text_decoder").unwrap();
    for stages in [2, 3] {
        let topology = ParallelTopology::new(2, stages, 2, 1).unwrap();
        for ordinal in 0..topology.world_size() {
            let rank = ParallelRankTopology::new(topology, ordinal).unwrap();
            let layout =
                eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                    &description,
                    rank,
                )
                .unwrap();
            let banks = family::expert_realization_plans(&args, rank, Some(&layout)).unwrap();
            assert_eq!(
                banks[&family::ExpertBank::FeedForward.id()].global_group_count(),
                5
            );
            assert_eq!(
                banks[&family::ExpertBank::AttentionValue.id()].global_group_count(),
                3
            );
            let mut blocks = vec![vec![RoutedExpertUnitWave::ordinary(0, 8, (1, 1))]];
            for layer in 1..3 {
                blocks.push(
                    args.routed_bank_order(layer)
                        .into_iter()
                        .map(|bank| {
                            RoutedExpertUnitWave::routed(
                                bank,
                                layer,
                                &owner,
                                &banks[&bank],
                                2,
                                args.routed_bank_tensor_reductions(layer, bank).unwrap(),
                            )
                            .unwrap()
                        })
                        .collect(),
                );
            }
            let schedule = RoutedExpertCollectiveWaveSchedule::from_block_waves(
                blocks,
                2,
                rank.tensor_parallel_rank(),
                stages,
                17,
            )
            .unwrap();
            let waves = (0..stages)
                .flat_map(|stage| schedule.stage(stage).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(waves.len(), 5);
            assert_eq!(waves[0].bank(), None);
            for pair in waves[1..].chunks(2) {
                assert_eq!(
                    pair[0].bank(),
                    Some(family::ExpertBank::AttentionValue.id())
                );
                assert_eq!(pair[1].bank(), Some(family::ExpertBank::FeedForward.id()));
                assert_eq!((pair[0].hidden_width(), pair[0].output_width()), (8, 4));
                assert_eq!((pair[1].hidden_width(), pair[1].output_width()), (8, 8));
                assert_eq!(pair[0].unit(), pair[1].unit());
                assert!(!pair[0]
                    .operations()
                    .contains(&RoutedExpertWaveOperation::ReversePostReduceBias));
            }
            if stages == 3 {
                assert_eq!(schedule.stage(0).unwrap().len(), 1);
                assert_eq!(schedule.stage(1).unwrap().len(), 2);
                assert_eq!(schedule.stage(2).unwrap().len(), 2);
            }
        }
    }
}

#[test]
fn k2_routed_prepared_payload_tp_pp_ep_and_bounded_match_four_decode_steps() {
    let fixture = fixture();
    for name in ["moe", "mova"] {
        let case = &fixture[name];
        let args = family::model_args_from_config_value(&case["config"]).unwrap();
        let (root, _) = checkpoint_fixture(&case["config"], 1.0);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let inputs = [
            NumericTensor::token_ids(&[1, 3, 2]),
            NumericTensor::token_ids(&[4]),
            NumericTensor::token_ids(&[5]),
            NumericTensor::token_ids(&[6]),
            NumericTensor::token_ids(&[7]),
        ];
        for addressable in [false, true] {
            let bank_options = addressable.then(|| {
                ParameterBankLoadOptions::new(
                    eredu_core::residency::OffloadConfig::new(Some(1152), Some(1 << 20), 1)
                        .unwrap(),
                    1152,
                    1152,
                )
                .unwrap()
            });
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
                    ParallelTopology::new(1, 1, 2, 1).unwrap(),
                    ParallelTopology::new(2, 1, 2, 1).unwrap(),
                    ParallelTopology::new(1, 3, 2, 1).unwrap(),
                    ParallelTopology::new(2, 3, 2, 1).unwrap(),
                ] {
                    let world = Arc::new(NumericPartitionWorld::default());
                    let actual = std::thread::scope(|scope| {
                        let threads = (0..topology.world_size()).map(|rank| {
                        let world = Arc::clone(&world);
                        let args = &args; let inspection = &inspection; let inputs = &inputs; let residency = residency.clone();
                        scope.spawn(move || {
                            let plan = prepared_adapter::plan(None).with_topology(topology).with_residency(residency.clone());
                            let sources = partitioned_adapter::prepare_plan_with_banks(inspection, &plan, rank, std::time::Duration::from_secs(10), bank_options).unwrap_or_else(|e| panic!("{name} {topology:?} {residency:?} prepare rank {rank}: {e}"));
                            let context = NumericContext::default();
                            let description = family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap().parameter_description(&context).unwrap();
                            let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(&description, ParallelRankTopology::new(topology, rank).unwrap()).unwrap();
                            let mut context = NumericContext::with_partition(layout, rank, world); context.bind_checkpoint_values = true;
                            reset_reference_stage_evidence("SafeTensors");
                            let mut executable = partitioned_adapter::routed(sources, &context, Arc::new(AtomicUsize::new(0)), None).unwrap_or_else(|e| panic!("{name} {topology:?} {residency:?} construct rank {rank}: {e}"));
                            let outputs = inputs.iter().enumerate().map(|(step, input)| executable.forward(input, step == 0).unwrap_or_else(|e| panic!("{name} {topology:?} {residency:?} step {step} rank {rank}: {e}"))).collect::<Vec<_>>();
                            assert!(executable.positions().unwrap().iter().all(|p| *p == 7));
                            let evidence = last_reference_stage_evidence();
                            if !matches!(residency, eredu_core::ResidencyPlan::FullyResident) {
                                assert!(evidence.bounded_unit_acquisitions.len() >= 5);
                                assert_eq!(evidence.peak_bound_units, 1);
                            }
                            let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                            let owned = balanced_rank_range(args.num_hidden_layers as usize, topology.pipeline(), rank_topology.pipeline_parallel_rank());
                            if addressable && owned.into_iter().any(|layer| args.is_sparse_layer(layer)) {
                                assert!(!evidence.bank_acquisitions.is_empty());
                                assert!(evidence.peak_bank_bytes <= 1152);
                                assert!(evidence.bank_completions > 0);
                                if topology.expert() == 1 && topology.pipeline() == 1 {
                                    assert!(evidence.bank_evictions > 0);
                                }
                                if name == "mova" {
                                    assert_eq!(evidence.bank_acquisitions.iter().map(|key| key.bank()).collect::<BTreeSet<_>>(), BTreeSet::from([0, 1]));
                                }
                            }
                            executable.reset().unwrap();
                            assert_tensor_exact(&executable.forward(&inputs[0], true).unwrap(), &outputs[0], "K2 routed partition reset");
                            outputs
                        })
                    }).collect::<Vec<_>>();
                        threads
                            .into_iter()
                            .map(|t| t.join().unwrap())
                            .collect::<Vec<_>>()
                    });
                    for rank in actual {
                        for (step, output) in rank.iter().enumerate() {
                            let expected = case["logits"][step + 2].as_array().unwrap();
                            assert_eq!(output.data.len(), expected.len());
                            for (&a, e) in output.data.iter().zip(expected) {
                                let e = e.as_f64().unwrap() as f32;
                                assert!(
                                    (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
                                    "{name} {topology:?} {residency:?} step {step}: {a} != {e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
