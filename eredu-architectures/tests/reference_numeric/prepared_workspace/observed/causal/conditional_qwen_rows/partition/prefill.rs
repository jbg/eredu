//! Actual selected SessionPrefill boundaries, not a hand-split forward loop.
use super::*;
use bounded_readout::parallel_boundary::{BoundaryReport, BoundaryTrace};
use eredu_runtime::prefill::{PrefillChunk, PrefillOutcome};

type RankResult = (State, Option<NumericTensor>, BoundaryTrace);

fn vocabulary(projections: &[(String, Vec<i32>)]) -> Vec<Vec<i32>> {
    projections
        .iter()
        .filter(|(name, _)| name == "lm_head.weight")
        .map(|(_, shape)| shape.clone())
        .collect()
}

fn assert_boundary(report: &BoundaryReport, rank: usize, output_rank: bool, cancelled: bool) {
    let expected = if cancelled {
        vec![PrefillChunk {
            input: 0..2,
            position: 0,
            output: OutputDemand::StateOnly,
        }]
    } else {
        vec![
            PrefillChunk {
                input: 0..2,
                position: 0,
                output: OutputDemand::StateOnly,
            },
            PrefillChunk {
                input: 2..4,
                position: 2,
                output: OutputDemand::StateOnly,
            },
            PrefillChunk {
                input: 4..5,
                position: 4,
                output: OutputDemand::LastPosition,
            },
        ]
    };
    let trace = &report.trace;
    assert_eq!(
        report.outcome,
        if cancelled {
            PrefillOutcome::Cancelled
        } else {
            PrefillOutcome::Complete
        }
    );
    assert_eq!(
        report.requested_locally,
        cancelled && rank == 0,
        "only rank zero requested cancellation"
    );
    assert_eq!(
        report.terminal_cancelled, cancelled,
        "agreement cancels every rank's token"
    );
    assert_eq!(
        trace.prepared, expected,
        "no source preparation after the agreed boundary"
    );
    assert_eq!(trace.announced, expected);
    assert_eq!(
        trace.delivered, expected,
        "one completed, state-committed delivery per span"
    );
    assert!(
        !trace.observations.is_empty(),
        "actual local model callbacks"
    );
    assert!(
        !trace.projections_after_delivery.last().unwrap().is_empty(),
        "actual nonzero model projection work"
    );
    assert_eq!(trace.transactions_prepared, expected.len());
    assert_eq!(trace.transactions_coordinated, expected.len());
    assert_eq!(trace.transactions_completed, expected.len());
    assert_eq!(trace.transactions_finished, vec![true; expected.len()]);
    assert_eq!(trace.projections_after_delivery.len(), expected.len());
    for (span, projections) in expected.iter().zip(&trace.projections_after_delivery) {
        let vocabulary = vocabulary(projections);
        if span.output == OutputDemand::StateOnly || !output_rank {
            assert!(
                vocabulary.is_empty(),
                "rank {rank} projected vocabulary for {span:?}: {vocabulary:?}"
            );
        } else {
            assert_eq!(
                vocabulary.len(),
                1,
                "only the final span projects vocabulary"
            );
            assert_eq!(
                vocabulary[0],
                [1, 1, 8],
                "select the hidden position before projecting"
            );
        }
    }
    assert_eq!(report.output.is_none(), cancelled);
}

fn run_boundary(
    fixture: &Fixture,
    topology: ParallelTopology,
    residency: eredu_core::ResidencyPlan,
    stepped: bool,
    cancelled: bool,
) -> Vec<RankResult> {
    let inspection =
        eredu_architectures::configuration::inspect_artifact(fixture.artifact.path()).unwrap();
    let description = numeric_composite_parameter_description(&fixture.value);
    let plan = prepared_adapter::plan(None)
        .with_topology(topology)
        .with_residency(residency);
    let world = Arc::new(NumericPartitionWorld::default());
    std::thread::scope(|scope| {
        let workers = (0..topology.world_size()).map(|rank| {
            let (inspection, description, plan) = (&inspection, &description, &plan);
            let world = Arc::clone(&world);
            scope.spawn(move || {
                let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                    description, rank_topology,
                ).unwrap();
                let mut context = NumericContext::with_partition(layout, rank, world);
                context.bind_checkpoint_values = true;
                let prepare = || {
                    let source = partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(
                        inspection, plan, rank, std::time::Duration::from_secs(30), None, 8,
                    ).unwrap();
                    assert!(matches!((plan.residency(), source.selected().text_realization().residency()),
                        (eredu_core::ResidencyPlan::FullyResident, LayerWeightResidency::FullyResident) |
                        (eredu_core::ResidencyPlan::LayerwiseHost { .. }, LayerWeightResidency::LayerwiseHost(_)) |
                        (eredu_core::ResidencyPlan::DenseDiskStream { .. }, LayerWeightResidency::DenseDiskStream(_))
                    ), "retain the actual selected residency");
                    source
                };
                let prompt = [4, 2, 1, 3, 5];
                // Independent state and ordinary unsplit execution, with the
                // exact same selected rank layout and real residency mechanism.
                let mut reference = partitioned_adapter::composite(prepare(), &context).unwrap();
                let prefix = if cancelled { &prompt[..2] } else { &prompt[..] };
                let expected = reference.forward(&numeric_text_prepared_input(prefix), true).unwrap();
                nonzero(&expected);
                assert_eq!(expected.shape, [1, 1, 7]);
                assert_eq!(vocabulary(&context.projections.lock().unwrap()).len(),
                    usize::from(rank_topology.owns_output_head()),
                    "the exact head selector observes real unsplit vocabulary work");
                let mut actual = partitioned_adapter::composite(prepare(), &context).unwrap();
                context.projections.lock().unwrap().clear();
                let report = actual.prefill_boundary(
                    &numeric_text_prepared_input(&prompt), stepped, cancelled && rank == 0,
                ).unwrap();
                assert_boundary(&report, rank, rank_topology.owns_output_head(), cancelled);
                let state = actual.snapshot().unwrap();
                populated(&state, prefix.len() as i32);
                same_state(&state, &reference.snapshot().unwrap());
                assert_eq!(actual.positions().unwrap(), state.as_ref().iter().map(|s| s.position()).collect::<Vec<_>>());
                if let Some(output) = &report.output {
                    nonzero(output);
                    assert_tensor_close(output, &expected, "shared parallel prefill final scores");
                    // Multiple real cached decodes expose any lost recurrent,
                    // convolution, attention, or pipeline-receiver state.
                    for (step, token) in [2, 6, 1].into_iter().enumerate() {
                        let input = numeric_text_prepared_input(&[token]);
                        let expected = reference.forward(&input, false).unwrap();
                        let output = actual.forward(&input, false).unwrap();
                        nonzero(&output);
                        assert_tensor_close(&output, &expected, "shared parallel prefill cached decode");
                        let state = actual.snapshot().unwrap();
                        populated(&state, 6 + step as i32);
                        same_state(&state, &reference.snapshot().unwrap());
                    }
                }
                (actual.snapshot().unwrap(), report.output, report.trace)
            })
        }).collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect()
    })
}

fn compare_modes(cancelled: bool) {
    let fixture = Fixture::new(configuration(true, false, false, false), true);
    for (tp, pp) in [(2, 1), (1, 2), (2, 2)] {
        let topology = ParallelTopology::new(tp, pp, 1, 1).unwrap();
        for residency in residencies() {
            let run = run_boundary(&fixture, topology, residency.clone(), false, cancelled);
            let steps = run_boundary(&fixture, topology, residency, true, cancelled);
            assert_eq!(run.len(), topology.world_size());
            assert_eq!(steps.len(), run.len());
            for ((state, output, trace), (step_state, step_output, step_trace)) in
                run.iter().zip(&steps)
            {
                same_state(state, step_state);
                assert_eq!(
                    trace, step_trace,
                    "run and step share every source/model boundary"
                );
                match (output, step_output) {
                    (Some(a), Some(b)) => assert_tensor_close(a, b, "run/step final scores"),
                    (None, None) => assert!(cancelled),
                    _ => panic!("run/step output presence differs"),
                }
            }
        }
    }
}

#[test]
fn selected_qwen_hybrid_parallel_prefill_run_and_step_preserve_uneven_spans_and_full_state() {
    compare_modes(false);
}

#[test]
fn selected_qwen_hybrid_parallel_prefill_one_rank_cancel_stops_all_sources_and_model_work() {
    compare_modes(true);
}
