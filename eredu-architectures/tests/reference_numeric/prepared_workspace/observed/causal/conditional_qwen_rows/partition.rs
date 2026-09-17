//! Original prepared TP/PP selection, real callbacks and complete local state.
use super::*;

fn project(
    value: &NumericTensor,
    axis: usize,
    map: &eredu_core::component::ComponentCoordinateMap,
) -> NumericTensor {
    let mut shape = value.shape.clone();
    shape[axis] = map.local_count() as i32;
    let mut out = NumericTensor::zeros(shape);
    out.dtype = value.dtype.clone();
    for i in 0..out.data.len() {
        let mut c = unravel(i, &out.shape);
        c[axis] = map.local_to_global(c[axis]).unwrap();
        out.data[i] = value.data[offset(&c, &value.shape)];
    }
    out
}
fn run(f: &Fixture, topology: ParallelTopology, residency: eredu_core::ResidencyPlan) {
    let inspection =
        eredu_architectures::configuration::inspect_artifact(f.artifact.path()).unwrap();
    let descriptor = inspection.architecture_plan().architecture_descriptor();
    let description = numeric_composite_parameter_description(&f.value);
    let c = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let (mut model, d, paths) = f.model(&c);
    let mut state = f.state();
    let ids = [
        vec![4, 2],
        vec![1, 3],
        vec![5],
        vec![2, 6],
        vec![1],
        vec![3],
        vec![2],
    ];
    let global = ids
        .iter()
        .map(|ids| {
            let mut rows = Rows::new(&d);
            let scores = model
                .run(
                    ids,
                    &mut state,
                    &c,
                    &mut rows,
                    &paths,
                    OutputDemand::Sequence,
                )
                .unwrap()
                .unwrap();
            rows.complete(ids.len());
            (scores, rows.values)
        })
        .collect::<Vec<_>>();
    let plan = prepared_adapter::plan(None)
        .with_topology(topology)
        .with_residency(residency);
    let world = Arc::new(NumericPartitionWorld::default());
    std::thread::scope(|scope| {
        let workers = (0..topology.world_size())
            .map(|rank| {
                let (inspection, descriptor, description, plan, ids, global, d) =
                    (&inspection, &descriptor, &description, &plan, &ids, &global, &d);
                let world = Arc::clone(&world);
                scope.spawn(move || {
                    let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                        description, rank_topology,
                    ).unwrap();
                    let mut context = NumericContext::with_partition(layout, rank, world);
                    context.bind_checkpoint_values = true;
                    let prepare = || {
                        partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(
                            inspection, plan, rank, std::time::Duration::from_secs(30), None, 8,
                        ).unwrap()
                    };
                    let source = prepare();
                    let components = source.selected().execution()
                        .component_partition_layout(descriptor, description).unwrap().unwrap();
                    let discovery = source.prepare_discovery(
                        ObservationMechanisms {
                            activation_tensors: true,
                            floating_to_f32: true,
                            ..Default::default()
                        },
                        CaptureCapabilities::default(),
                    );
                    let capture = discovery.capture().unwrap();
                    let catalog = &capture.catalog;
                    let mut chunked = partitioned_adapter::composite(source, &context).unwrap();
                    let mut consumed = Vec::new();
                    let mut final_reference = None;
                    for (step, input_ids) in ids.iter().enumerate() {
                        let mut rows = Rows::new(d);
                        rows.prepared = false;
                        let out = chunked.forward_observed(
                            &numeric_text_prepared_input(input_ids), step < 4, &mut rows,
                        ).unwrap();
                        assert_tensor_close(
                            &out,
                            &global[step].0.axis_slice(1, input_ids.len() - 1, input_ids.len()),
                            "prepared partition global last row",
                        );
                        for declaration in d {
                            let path = declaration.path();
                            let site = components.observation(path)
                                .unwrap_or_else(|| panic!("missing actual placement {path}"));
                            assert_eq!(
                                rows.values.contains_key(path), site.coordinates().is_some(),
                                "rank {rank} {path}",
                            );
                            if let Some(actual) = rows.values.get(path) {
                                nonzero(actual);
                                // All observations retain the agreed Sequence rows; only
                                // the separately checked public return selects its last row.
                                let expected = &global[step].1[path];
                                let point = catalog.get(path).unwrap();
                                let axis = point.axes.as_ref().unwrap().iter()
                                    .position(|a| a.name == site.axis()).unwrap();
                                let expected = project(expected, axis, site.coordinates().unwrap());
                                assert_tensor_close(actual, &expected, &format!("rank {rank} {path}"));
                            }
                        }
                        let snapshot = chunked.snapshot().unwrap();
                        populated(&snapshot, ids[..=step].iter().map(|i| i.len() as i32).sum());
                        if (1..=3).contains(&step) {
                            consumed.extend_from_slice(input_ids);
                            let mut reference = partitioned_adapter::composite(prepare(), &context).unwrap();
                            reference.forward(&numeric_text_prepared_input(&ids[0]), true).unwrap();
                            let expected = reference.forward(
                                &numeric_text_prepared_input(&consumed), true,
                            ).unwrap();
                            assert_tensor_close(&out, &expected, "full versus chunked original partition output");
                            same_state(&snapshot, &reference.snapshot().unwrap());
                            final_reference = Some(reference);
                        } else if step >= 4 {
                            let reference = final_reference.as_mut().unwrap();
                            let expected = reference.forward(
                                &numeric_text_prepared_input(input_ids), false,
                            ).unwrap();
                            assert_tensor_close(&out, &expected, "partition continued decode");
                            same_state(&snapshot, &reference.snapshot().unwrap());
                        }
                        assert_eq!(
                            chunked.positions().unwrap(),
                            snapshot.as_ref().iter().map(|s| s.position()).collect::<Vec<_>>(),
                        );
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
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
fn conditional_qwen_prepared_rows_and_complete_state_cross_tp_pp_and_residency() {
    for hybrid in [false, true] {
        for routed in [false, true] {
            let f = Fixture::new(configuration(hybrid, routed, false, false), hybrid);
            for (tp, pp) in [(2, 1), (1, 2), (2, 2)] {
                for residency in residencies() {
                    run(&f, ParallelTopology::new(tp, pp, 1, 1).unwrap(), residency);
                }
            }
        }
    }
}
#[test]
fn conditional_qwen_prepared_routed_rows_keep_actual_tp_pp_ep_ownership() {
    for hybrid in [false, true] {
        let f = Fixture::new(configuration(hybrid, true, false, false), hybrid);
        for residency in residencies() {
            run(&f, ParallelTopology::new(2, 2, 2, 1).unwrap(), residency);
        }
    }
}
#[test]
fn qwen_vl_zero_sections_cross_real_prepared_tp_pp_with_all_residencies() {
    for routed in [false, true] {
        let f = Fixture::new(configuration(false, routed, true, false), false);
        for residency in residencies() {
            run(&f, ParallelTopology::new(2, 2, 1, 1).unwrap(), residency);
        }
    }
}

#[path = "partition/prefill.rs"]
mod prefill;
