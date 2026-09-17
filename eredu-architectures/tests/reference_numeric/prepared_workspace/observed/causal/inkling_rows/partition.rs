use super::*;

fn compare_partition_state(
    actual: &State,
    expected: &State,
    f: &Fixture,
    identity: &eredu_core::cache::PromptCacheModelIdentity,
    rank: ParallelRankTopology,
) {
    let range = identity.global_layer_start()..identity.global_layer_end();
    assert_eq!(actual.as_ref().len(), range.len());
    assert_eq!(actual.layout().len(), range.len());
    for (local, a) in actual.as_ref().iter().enumerate() {
        let global = range.start + local;
        let policy = f.args.text_config.layer_policy(global).unwrap();
        let sliding = policy.attention.window().is_some();
        let heads = eredu_core::balanced_contiguous_range(
            f.args.text_config.key_value_heads(sliding) as usize,
            rank.tensor_parallel_size(),
            rank.tensor_parallel_rank(),
            false,
        )
        .unwrap();
        let dim = f.args.text_config.attention_head_dim(sliding) as usize;
        // Actual placement shards KV heads and their two pre-normalization
        // convolution histories. Hidden-width residual histories are replicated.
        let mut b = expected.as_ref()[global].clone();
        let cache = b.attention.as_mut().unwrap();
        for value in [&mut cache.keys, &mut cache.values] {
            *value = value
                .as_ref()
                .map(|v| v.axis_slice(1, heads.start, heads.end));
        }
        for slot in [0, 1] {
            if let Some(value) = b.fixed.get_mut(&StateTensorRole::Convolution { slot }) {
                *value = value
                    .as_ref()
                    .map(|v| v.axis_slice(2, heads.start * dim, heads.end * dim));
            }
        }
        compare_layer(a, &b);
        let declared = actual.layout().layers().get(local).unwrap();
        assert_eq!(declared.fixed_state().len(), a.fixed.len());
        assert!(declared.attention().is_some());
        for value in a.fixed.values() {
            nonzero(value.as_ref().unwrap());
        }
        let cache = a.attention.as_ref().unwrap();
        nonzero(cache.keys.as_ref().unwrap());
        nonzero(cache.values.as_ref().unwrap());
    }
}

// The snapshot is the existing typed-session test seam added by the pinned
// Gemma4 package. No public accessor or fabricated local state is introduced.
#[test]
fn inkling_group_two_rows_and_complete_histories_cross_prepared_tp_pp_residencies() {
    for sparse in [false, true] {
        for kernel in [1, 3] {
            let f = fixture(configuration(sparse, kernel, false));
            let inspection =
                eredu_architectures::configuration::inspect_artifact(f.artifact.path()).unwrap();
            let descriptor = inspection.architecture_plan().architecture_descriptor();
            let parameters = numeric_composite_parameter_description(&f.value);
            let context = NumericContext {
                bind_checkpoint_values: true,
                ..Default::default()
            };
            let (mut model, declarations, paths) = make_model(&f, &context);
            let inputs = vec![
                vec![4, 2],
                vec![1, 3],
                vec![5],
                vec![2, 6],
                vec![7],
                vec![8],
                vec![9],
            ];
            let mut direct_state = state(&f);
            let reference = inputs
                .iter()
                .map(|ids| {
                    let mut rows = Rows::new(&declarations);
                    let scores = run(
                        &mut model,
                        &f,
                        ids,
                        &mut direct_state,
                        &context,
                        &mut rows,
                        &paths,
                        OutputDemand::Sequence,
                    )
                    .unwrap();
                    rows.complete(ids.len());
                    (scores, rows.values, direct_state.clone())
                })
                .collect::<Vec<_>>();
            for (tp, pp) in [(2, 1), (1, 2), (2, 2)] {
                let topology = ParallelTopology::new(tp, pp, 1, 1).unwrap();
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
                    // Collect all rank results before numerical assertions so a
                    // failed comparison cannot strand a peer at the next call.
                    let results = std::thread::scope(|scope| {
                        let workers=(0..topology.world_size()).map(|rank| {
                            let (inspection,parameters,descriptor,plan,inputs,declarations)=(&inspection,&parameters,&descriptor,&plan,&inputs,&declarations);
                            let world=Arc::clone(&world);
                            scope.spawn(move || {
                                let rank_topology=ParallelRankTopology::new(topology,rank).unwrap();
                                let sources=partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(inspection,plan,rank,std::time::Duration::from_secs(30),None,8).unwrap();
                                let components=sources.selected().execution().component_partition_layout(descriptor,parameters).unwrap().unwrap();
                                let owned=declarations.iter().map(|d| (d.path().to_owned(),components.observation(d.path()).unwrap().coordinates().is_some())).collect::<BTreeMap<_,_>>();
                                let layout=eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters,rank_topology).unwrap();
                                let mut context=NumericContext::with_partition(layout,rank,Arc::clone(&world));context.bind_checkpoint_values=true;
                                let mut executable=partitioned_adapter::composite(sources,&context).unwrap();
                                let identity=world.prompt_cache_identities().remove(&rank).unwrap();
                                let mut calls=Vec::new();
                                for (step,ids) in inputs.iter().enumerate() {
                                    let mut rows=Rows::new(declarations);rows.prepared=false;
                                    let output=executable.forward_observed(&numeric_text_prepared_input(ids),step<4,&mut rows).unwrap();
                                    calls.push((output,rows.values,executable.snapshot().unwrap(),executable.positions().unwrap()));
                                }
                                (rank_topology,identity,owned,calls)
                            })
                        }).collect::<Vec<_>>();
                        workers
                            .into_iter()
                            .map(|w| w.join().unwrap())
                            .collect::<Vec<_>>()
                    });
                    for (rank, identity, owned, calls) in results {
                        if pp == 2 {
                            assert_eq!(
                                identity.global_layer_start(),
                                rank.pipeline_parallel_rank()
                            );
                            assert_eq!(
                                identity.global_layer_end(),
                                identity.global_layer_start() + 1
                            );
                        }
                        assert_eq!(calls.len(), inputs.len());
                        for (step, (output, rows, snapshot, positions)) in
                            calls.into_iter().enumerate()
                        {
                            let n = inputs[step].len();
                            assert_tensor_close(
                                &output,
                                &reference[step].0.axis_slice(1, n - 1, n),
                                "selected actual global prediction",
                            );
                            for d in &declarations {
                                let path = d.path();
                                assert_eq!(
                                    rows.contains_key(path),
                                    owned[path],
                                    "actual partition ownership {path}"
                                );
                                if let Some(actual) = rows.get(path) {
                                    nonzero(actual);
                                    // Observation uses every physical row before
                                    // public output selects its final position.
                                    assert_tensor_close(actual, &reference[step].1[path], path);
                                }
                            }
                            compare_partition_state(
                                &snapshot,
                                &reference[step].2,
                                &f,
                                &identity,
                                rank,
                            );
                            assert_eq!(
                                positions,
                                snapshot
                                    .as_ref()
                                    .iter()
                                    .map(|l| l.position())
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
        }
    }
}
