//! Actual EP provider boundaries, including idle owners and failure before return exchange.
use super::*;

#[derive(Default)]
struct Vote {
    size: usize,
    values: Mutex<BTreeMap<usize, bool>>,
    ready: Condvar,
}
impl Vote {
    fn agree(&self, rank: usize, success: bool) -> Result<bool, Error> {
        let mut values = self.values.lock().unwrap();
        assert!(
            values.insert(rank, success).is_none(),
            "one invocation completion per rank"
        );
        self.ready.notify_all();
        let (values, timeout) = self
            .ready
            .wait_timeout_while(values, std::time::Duration::from_secs(5), |values| {
                values.len() < self.size
            })
            .unwrap();
        if timeout.timed_out() {
            return Err(Error::backend("missing local invocation completion"));
        }
        Ok(values.values().all(|success| *success))
    }
}

struct Observer {
    rank: usize,
    fault: Option<&'static str>,
    vote: Arc<Vote>,
    world: Arc<NumericPartitionWorld>,
    rows: Option<usize>,
    callbacks: usize,
    finished: bool,
    injected: bool,
    exchanges_at_finish: usize,
    ownership: eredu_core::capture::RoutedUnitCaptureOwnership,
}
impl Observer {
    fn fail(&mut self, stage: &str) -> bool {
        let owner = if self.rows == Some(0) {
            "idle"
        } else {
            "active"
        };
        let injected = self
            .fault
            .is_some_and(|fault| fault == format!("{stage}-{owner}"));
        self.injected |= injected;
        injected
    }
}
impl ActivationObserver<NumericTensor, Error> for Observer {
    fn observe(&mut self, _: &str, _: &NumericTensor) -> Result<(), Error> {
        Ok(())
    }
    fn routed_unit_observer(
        &mut self,
        _: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<NumericTensor>>, Error> {
        Ok(Some(self))
    }
}
impl RoutedUnitObserver<NumericTensor> for Observer {
    fn begin_invocation(
        &mut self,
        invocation: &eredu_runtime::RoutedUnitInvocation<'_, NumericTensor>,
    ) -> Result<(), Error> {
        assert!(self.rows.is_none());
        let origins = invocation.origins.expect("actual received origins");
        let rows = origins.capture_coordinates().row_count();
        assert_eq!(invocation.input.shape[0] as usize, rows);
        assert_eq!(origins.capture_coordinates().peer_count(), 2);
        assert_eq!(self.ownership.source_peers, 2);
        if let Some(units) = invocation.unit_coordinates {
            assert_eq!(units, self.ownership.coordinates.units());
        }
        self.rows = Some(rows);
        if self.fail("start") {
            return Err(Error::backend_retained_source(UnitSentinel));
        }
        Ok(())
    }
    fn observe(&mut self, batch: &RoutedUnitBatch<'_, NumericTensor>) -> Result<(), Error> {
        assert!(self.rows.is_some_and(|rows| rows > 0));
        assert_eq!(
            batch.unit_coordinates,
            Some(self.ownership.coordinates.units())
        );
        for (_, _, _, expert) in Capture::keys(batch) {
            assert!(self
                .ownership
                .coordinates
                .experts()
                .global_to_local(expert)
                .is_some());
        }
        self.callbacks += 1;
        if self.fail("units") {
            return Err(Error::backend_retained_source(UnitSentinel));
        }
        Ok(())
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), Error> {
        assert!(self.rows.is_some());
        assert!(!self.finished);
        self.finished = true;
        let fail = self.fail("finish");
        let accepted = self.vote.agree(self.rank, success && !fail)?;
        self.exchanges_at_finish = self
            .world
            .trace()
            .iter()
            .filter(|event| event.operation == NumericOpaqueOperation::VariableAllToAll)
            .count();
        if fail {
            return Err(Error::backend_retained_source(UnitSentinel));
        }
        if !accepted {
            return Err(Error::backend("another local invocation failed"));
        }
        Ok(())
    }
}

#[test]
fn prepared_gated_and_relu2_idle_ep_owners_finish_before_reverse_exchange_on_failure() {
    for gated in [true, false] {
        let config = if gated {
            serde_json::json!({"model_type":"qwen3_moe", "vocab_size":16, "hidden_size":8, "intermediate_size":0,
                "moe_intermediate_size":6, "num_hidden_layers":1, "num_attention_heads":4, "num_key_value_heads":2, "head_dim":2,
                "max_position_embeddings":64, "rms_norm_eps":1e-5, "num_experts":3, "num_experts_per_tok":1,
                "norm_topk_prob":true, "tie_word_embeddings":false})
        } else {
            serde_json::json!({"model_type":"nemotron_h", "vocab_size":16, "hidden_size":8,
                "intermediate_size":10, "num_hidden_layers":1, "hybrid_override_pattern":"E",
                "num_attention_heads":2, "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":2,
                "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3, "conv_kernel":3, "chunk_size":2,
                "n_routed_experts":3, "n_shared_experts":1, "moe_intermediate_size":6,
                "moe_shared_expert_intermediate_size":6, "num_experts_per_tok":1, "n_group":1, "topk_group":1,
                "num_nextn_predict_layers":0, "tie_word_embeddings":false, "residual_in_fp32":true})
        };
        // Equal router rows select the same expert for every token. Whichever
        // tie order the family uses, the other EP owner receives exactly zero rows.
        let (root, _) =
            prepared_adapter::payload_fixture_config_with(&config, 1., |name, shape| {
                name.ends_with(".gate.weight").then(|| {
                    NumericTensor::new(
                        shape.to_vec(),
                        vec![0.; shape.iter().map(|n| *n as usize).product()],
                    )
                })
            });
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let description = if gated {
            qwen::RoutedLayeredModel::<NumericBackend>::new(
                qwen::model_args_from_config_value(&config).unwrap(),
                &NumericContext::default(),
            )
            .unwrap()
            .parameter_description(&NumericContext::default())
            .unwrap().into_owned()
        } else {
            nemotron_h::LayeredModel::<NumericBackend>::new(
                nemotron_h::model_args_from_config_value(&config).unwrap(),
                &NumericContext::default(),
            )
            .unwrap()
            .parameter_description(&NumericContext::default())
            .unwrap().into_owned()
        };
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
            for (tp, fault) in [1, 2].into_iter().flat_map(|tp| {
                [
                    None,
                    Some("start-active"),
                    Some("start-idle"),
                    Some("units-active"),
                    Some("finish-active"),
                    Some("finish-idle"),
                ]
                .into_iter()
                .map(move |fault| (tp, fault))
            }) {
                let topology = ParallelTopology::new(tp, 1, 2, 1).unwrap();
                let world = Arc::new(NumericPartitionWorld::default());
                let capture_worlds = Arc::new(
                    (0..4)
                        .map(|_| {
                            (
                                Arc::new(NumericPartitionWorld::default()),
                                Arc::new(observer::World::default()),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                let vote = Arc::new(Vote {
                    size: topology.world_size(),
                    ..Default::default()
                });
                let outputs = std::thread::scope(|scope| {
                    (0..topology.world_size()).map(|rank| {
                        let world = world.clone(); let vote = vote.clone(); let residency = residency.clone();
                        let capture_worlds = capture_worlds.clone();
                        let (inspection, description) = (&inspection, &description);
                        scope.spawn(move || {
                            let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                            let plan = prepared_adapter::plan(None).with_topology(topology).with_residency(residency);
                            let sources = partitioned_adapter::prepare_plan(inspection, &plan, rank, std::time::Duration::from_secs(10)).unwrap();
                            let descriptor = sources.architecture().architecture_descriptor();
                            let retained = placement::layouts(&sources, description);
                            placement::verify(&retained, &descriptor);
                            let ownership = retained.rank(rank).unwrap().routed_observation(&descriptor.routed_components[0].activation).unwrap().ownership().unwrap().clone();
                            let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(description, rank_topology).unwrap();
                            let mut context = NumericContext::with_partition(layout, rank, world.clone());
                            context.bind_checkpoint_values = true;
                            let mut executable = partitioned_adapter::routed(sources, &context, Arc::new(AtomicUsize::new(0)), None).unwrap();
                            let mut observer = Observer { rank, fault, vote, world, rows: None, callbacks: 0, finished: false, injected: false, exchanges_at_finish: 0, ownership };
                            let before = executable.positions().unwrap();
                            let result = executable.forward_observed(&NumericTensor::token_ids(&[1, 2, 5]), true, &mut observer);
                            assert_eq!(result.is_ok(), fault.is_none(), "{fault:?}: {result:?}");
                            assert!(observer.finished);
                            if observer.rows == Some(0) { assert_eq!(observer.callbacks, 0); }
                            if fault.is_some() {
                                assert_eq!(executable.positions().unwrap(), before, "failed observation rolls back model state");
                                if observer.injected { assert_unit_source(result.as_ref().unwrap_err()); }
                            } else {
                                executable.reset().unwrap();
                                let baseline = executable.forward(&NumericTensor::token_ids(&[1, 2, 5]), true).unwrap();
                                assert_tensor_close(result.as_ref().unwrap(), &baseline, "local invocation instrumentation preserves ordinary execution");
                            }
                            let capture_usage = if fault.is_none() {
                                let inputs = [NumericTensor::token_ids(&[1,2,5]), NumericTensor::token_ids(&[3]), NumericTensor::token_ids(&[4])];
                                Some([None, Some("source-active"), Some("source-idle"), Some("collector")].into_iter().enumerate().map(|(case, capture_fault)| {
                                    let (model_world, capture_world) = &capture_worlds[case];
                                    let sources = partitioned_adapter::prepare_plan(inspection, &plan, rank, std::time::Duration::from_secs(10)).unwrap();
                                    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(description, rank_topology).unwrap();
                                    let mut context = NumericContext::with_partition(layout, rank, model_world.clone());
                                    context.bind_checkpoint_values = true;
                                    let mut model = partitioned_adapter::routed(sources, &context, Arc::new(AtomicUsize::new(0)), None).unwrap();
                                    let transport = observer::Transport::new(capture_world.clone(),rank,topology.world_size());
                                    observer::run(&mut model,&retained,&descriptor,&transport,&inputs,None,capture_fault)
                                }).collect::<Vec<_>>())
                            } else { None };
                            (observer.rows.unwrap(), observer.exchanges_at_finish, observer.injected, capture_usage)
                        })
                    }).collect::<Vec<_>>().into_iter().map(|worker| worker.join().unwrap()).collect::<Vec<_>>()
                });
                assert_eq!(
                    outputs.iter().filter(|output| output.0 == 0).count(),
                    tp,
                    "one real idle EP owner across all its TP peers"
                );
                assert!(
                    outputs.windows(2).all(|pair| pair[0].3 == pair[1].3),
                    "shared capture budgets agree across active and idle owners"
                );
                if let Some(usage) = &outputs[0].3 {
                    assert!(
                        usage[1..].windows(2).all(|pair| pair[0] == pair[1]),
                        "failure stage refunds no prepaid work"
                    );
                }
                if fault.is_some() {
                    assert_eq!(outputs.iter().filter(|output| output.2).count(), tp);
                    let exchanges = world
                        .trace()
                        .iter()
                        .filter(|event| event.operation == NumericOpaqueOperation::VariableAllToAll)
                        .count();
                    assert!(exchanges > 0, "real forward route exchange occurred");
                    assert!(
                        outputs.iter().all(|output| output.1 == exchanges),
                        "failed local capture must stop reverse route exchange"
                    );
                    assert!(world
                        .lifecycle_counts()
                        .3
                        .values()
                        .all(|commits| *commits == 0));
                }
            }
        }
    }
}
