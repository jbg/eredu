use super::*;
use eredu_runtime::{ExecutionGraph, ExecutionGroupSpec, GroupSubmissionMechanism};

fn geometry(input: u64, chunk: u64, outputs: u64) -> eredu_core::InferenceGeometry {
    eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: input,
        max_output_tokens: outputs,
        prefill_chunk_positions: chunk,
        output: eredu_core::OutputDemand::LastPosition,
    }
}
fn graphs() -> Vec<ExecutionGraph> {
    vec![
        ExecutionGraph::chain(["single"]).unwrap(),
        ExecutionGraph::chain(["first", "middle", "last"]).unwrap(),
        ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("image"),
                ExecutionGroupSpec::root("audio"),
                ExecutionGroupSpec::with_dependencies("text", ["image", "audio"]),
            ],
            "text",
        )
        .unwrap(),
    ]
}
#[test]
fn manual_group_traversal_retains_only_actual_policy_final_submissions() {
    for graph in graphs() {
        let layout =
            ExecutionUnitLayout::new(&graph, std::iter::repeat_n(3, graph.groups().len())).unwrap();
        for g in [geometry(5, 2, 4), geometry(1, 1, 1), geometry(5, 5, 2)] {
            let forwards =
                g.input_positions.div_ceil(g.prefill_chunk_positions) + g.max_output_tokens - 1;
            for final_submission in [false, true] {
                let population = NeuralPopulation::from_execution(
                    &layout,
                    g,
                    eredu_runtime::GroupSubmissionMechanism::PolicyOnly,
                    final_submission,
                )
                .unwrap();
                assert_eq!(
                    population.submissions,
                    usize::try_from(forwards).unwrap() * usize::from(final_submission)
                );
                assert_eq!(population.per_forward, usize::from(final_submission));
                assert_eq!(population.shape.consumers(), 0);
                let live = Rc::new(Cell::new(0));
                let controls = Custody::new(&live);
                let mut bank = PreparedOperationBank::try_new(
                    population.submissions,
                    factory::<Custody, NeverObserved>(population.shape, Some(controls.clone())),
                )
                .unwrap();
                let mut completed = Vec::new();
                for _ in 0..forwards {
                    if final_submission {
                        completed.push(bank.checkout().unwrap());
                    }
                }
                assert!(
                    bank.checkout().is_err(),
                    "manual traversal has no group submission"
                );
                drop(bank);
                drop(controls);
                assert_eq!(
                    live.get() == 0,
                    completed.is_empty(),
                    "issued final-completion slots retain their original custody"
                );
                drop(completed);
                retire_to(&live, 0);
            }
        }
    }
}

#[test]
fn neural_request_population_follows_graph_boundaries_and_actual_prefill_decode_calls() {
    for graph in graphs() {
        for units in [0, 3] {
            let layout =
                ExecutionUnitLayout::new(&graph, std::iter::repeat_n(units, graph.groups().len()))
                    .unwrap();
            for g in [geometry(5, 2, 4), geometry(1, 1, 1), geometry(5, 5, 2)] {
                let population = NeuralPopulation::from_execution(
                    &layout,
                    g,
                    GroupSubmissionMechanism::LayeredGraph,
                    true,
                )
                .unwrap();
                let mut invocations = 0;
                let mut position = 0;
                while position < g.input_positions {
                    invocations += 1;
                    position = position
                        .saturating_add(g.prefill_chunk_positions)
                        .min(g.input_positions);
                }
                for _ in 1..g.max_output_tokens {
                    invocations += 1;
                }
                let mut actual_calls = 0;
                for _ in 0..invocations {
                    if graph.groups().len() > 1 {
                        actual_calls += 1; // Actual initial hook call.
                        for _ in graph.execution_order() {
                            actual_calls += 1;
                        }
                    }
                    actual_calls += 1; // Concrete MLX final-output submission.
                }
                assert_eq!(population.submissions, actual_calls);
                assert_eq!(population.shape.arrays(), 1);
                assert_eq!(population.shape.host_resources(), 0);
                let mut per_producer = vec![0; graph.groups().len() + 1];
                if graph.groups().len() > 1 {
                    for &group in graph.execution_order() {
                        let deps = graph.dependencies(group).unwrap();
                        if deps.is_empty() {
                            per_producer[0] += 1;
                        }
                        for &dependency in deps {
                            per_producer[dependency + 1] += 1;
                        }
                    }
                    per_producer[graph.output() + 1] += 1;
                }
                assert_eq!(
                    population.shape.consumers(),
                    *per_producer.iter().max().unwrap()
                );
                let native = population.native_requirements().unwrap();
                assert_eq!(native.root_handles, actual_calls);
                assert_eq!(
                    native.consumer_waits,
                    actual_calls * population.shape.consumers()
                );
                assert_eq!(
                    native.recovery_observers,
                    actual_calls * (population.shape.consumers() + 1)
                );
            }
        }
    }
}

#[test]
fn neural_uniform_bank_prices_all_slots_not_only_reached_waits_and_rejects_overflow() {
    let graph = graphs().pop().unwrap();
    let layout = ExecutionUnitLayout::new(&graph, [0, 1, 0]).unwrap();
    let population = NeuralPopulation::from_execution(
        &layout,
        geometry(1, 1, 1),
        GroupSubmissionMechanism::LayeredGraph,
        true,
    )
    .unwrap();
    assert_eq!(
        (population.submissions, population.shape.consumers()),
        (5, 2)
    );
    assert_eq!(population.native_requirements().unwrap().consumer_waits, 10);
    assert_eq!(layout.submission_geometry().consumer_waits_per_forward(), 5);
    let fit = NeuralProducerFit::pending(population).unwrap();
    assert!(fit.additional_control_bytes().is_none());
    let pending = fit.requirement();
    assert_eq!(
        (pending.roots_per_submission, pending.waits_per_submission),
        (1, 2)
    );
    assert_eq!(pending.populations.consumer_waits, 10);
    assert_eq!(
        pending.wait_records,
        safemlx::OperationEvent::wait_record_layout(10)
    );
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(
            pending.wait_records.is_some(),
            "pinned validation requires the qualified wait producer"
        );
    }
    if let Some(receipts) = pending.wait_records {
        assert_eq!(receipts.wait_count(), 10);
        assert_eq!(receipts.total_record_allocations(), 30);
        assert_eq!(receipts.capture_slots(), 8);
        assert_eq!(receipts.stream_receipts(), 1);
        assert_eq!(
            receipts.total_record_requested_bytes(),
            10 * receipts.requested_bytes_per_wait()
        );
    } else {
        for count in [0, 1, 10, usize::MAX] {
            assert!(safemlx::OperationEvent::wait_record_layout(count).is_none());
        }
    }
    assert_eq!(fit.additional_control_bytes(), None); // partial facts never activate total fit

    assert_eq!(
        pending.missing(),
        [
            NeuralFitContribution::ReachableEvaluationDagInRoleGraph,
            NeuralFitContribution::EvaluationAndWaitReceiptsInRoleRecord,
            NeuralFitContribution::OutsideArenaStreamEventAndQueueOwners,
        ]
    );
    let zero = NeuralPopulation {
        submissions: 0,
        ..population
    };
    let two = NeuralPopulation {
        submissions: 2,
        ..population
    };
    let each = PreparedSlot::control_bytes(population.shape).unwrap()
        + u64::try_from(size_of::<Option<PreparedNeuralSubmission>>()).unwrap()
        + u64::try_from(size_of::<OperationControls>()).unwrap();
    assert_eq!(
        two.rust_control_bytes().unwrap() - zero.rust_control_bytes().unwrap(),
        2 * each
    );
    assert!(NeuralPopulation {
        submissions: usize::MAX,
        ..population
    }
    .rust_control_bytes()
    .is_none());
    // Keep the geometry valid so the five-submission population itself
    // overflows, rather than failing the earlier context-frontier check.
    assert!(matches!(
        NeuralPopulation::from_execution(
            &layout,
            geometry(1, 1, u64::MAX - 1),
            GroupSubmissionMechanism::LayeredGraph,
            true
        ),
        Err(Error::PrefillControl(WorkingMemoryError::Overflow))
    ));
    assert!(matches!(
        NeuralPopulation::from_execution(
            &layout,
            geometry(1, 1, u64::MAX),
            GroupSubmissionMechanism::LayeredGraph,
            true
        ),
        Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
    ));
    assert!(matches!(
        NeuralPopulation::from_execution(
            &layout,
            geometry(1, 0, 1),
            GroupSubmissionMechanism::LayeredGraph,
            true
        ),
        Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
    ));
}

struct Custody {
    live: Rc<Cell<usize>>,
    identity: usize,
}
impl Custody {
    fn new(live: &Rc<Cell<usize>>) -> Self {
        live.set(live.get() + 1);
        Self {
            live: live.clone(),
            identity: 713,
        }
    }
}
impl Clone for Custody {
    fn clone(&self) -> Self {
        assert_eq!(self.identity, 713);
        self.live.set(self.live.get() + 1);
        Self {
            live: self.live.clone(),
            identity: self.identity,
        }
    }
}
impl Drop for Custody {
    fn drop(&mut self) {
        self.live.set(self.live.get() - 1);
    }
}
struct NeverObserved;
impl Observer for NeverObserved {
    type Error = Infallible;
    fn observe(
        &self,
    ) -> Result<crate::backend::submission_recovery::observed::Observation, Infallible> {
        panic!("never-started neural slots must not observe or submit native work")
    }
}
type Prepared = PreparedNeuralSubmission<Custody, NeverObserved>;
fn retire_to(live: &Rc<Cell<usize>>, expected: usize) {
    crate::backend::submission_recovery::wait_for_retirement(|| live.get() == expected);
}
#[test]
fn neural_prepared_bank_zero_exact_and_one_extra_preserve_existing_custody() {
    for count in [0, 3] {
        let live = Rc::new(Cell::new(0));
        let controls = Custody::new(&live);
        let mut bank = PreparedOperationBank::try_new(
            count,
            factory::<Custody, NeverObserved>(
                NeuralSubmissionShape::new(1, 2).unwrap(),
                Some(controls.clone()),
            ),
        )
        .unwrap();
        assert_eq!((bank.len(), bank.remaining()), (count, count));
        let before_checkout = live.get();
        let mut checked_out = Vec::new();
        for index in 0..count {
            checked_out.push(bank.checkout().unwrap());
            assert_eq!(bank.remaining(), count - index - 1);
            assert_eq!(
                live.get(),
                before_checkout,
                "checkout moves existing owners only"
            );
        }
        assert_eq!(bank.checkout().unwrap_err().prepared, count);
        assert_eq!(bank.checkout().unwrap_err().prepared, count);
        drop(bank);
        if count > 0 {
            assert!(live.get() > 1, "moved slots still retain original custody");
        }
        drop(checked_out);
        retire_to(&live, 1);
        drop(controls);
        assert_eq!(live.get(), 0);
    }
}

#[test]
fn neural_bank_slot_reserve_failure_retains_real_error_and_successful_prefix() {
    let live = Rc::new(Cell::new(0));
    let controls = Custody::new(&live);
    let shape = NeuralSubmissionShape::new(1, 2).unwrap();
    let mut actual_factory = factory::<Custody, NeverObserved>(shape, Some(controls.clone()));
    let calls = Cell::new(0);
    let error = PreparedOperationBank::try_new(3, |ordinal| {
        calls.set(calls.get() + 1);
        if ordinal == 1 {
            Prepared::fail_next_reserve_for_test(1);
        }
        actual_factory(ordinal)
    })
    .unwrap_err();
    assert_eq!(calls.get(), 2, "no later slot after actual failed reserve");
    assert_eq!(error.prepared_len(), 1);
    let (cause, prefix, factory) = error.into_parts();
    let BankPreparationCause::Slot { ordinal, cause } = cause else {
        panic!("actual slot reserve failure")
    };
    assert_eq!(ordinal, 1);
    assert!(
        matches!(&cause.cause, SubmissionPreparationCause::Reserve { site: "cleanup slots", cause } if cause.to_string().contains("capacity"))
    );
    assert!(cause.pending.is_some());
    drop(prefix);
    drop(factory);
    drop(actual_factory);
    assert!(
        live.get() > 1,
        "failed slot still owns its source and pending buffers"
    );
    drop(cause);
    retire_to(&live, 1);
    drop(controls);
    assert_eq!(live.get(), 0);
}

#[test]
fn neural_outer_bank_overflow_never_invokes_slot_factory() {
    let live = Rc::new(Cell::new(0));
    let controls = Custody::new(&live);
    let mut actual_factory = factory::<Custody, NeverObserved>(
        NeuralSubmissionShape::new(1, 0).unwrap(),
        Some(controls.clone()),
    );
    let calls = Cell::new(0);
    let error = PreparedOperationBank::try_new(usize::MAX, |ordinal| {
        calls.set(calls.get() + 1);
        actual_factory(ordinal)
    })
    .unwrap_err();
    assert_eq!(calls.get(), 0);
    assert_eq!(error.prepared_len(), 0);
    assert!(matches!(&error.cause, BankPreparationCause::Overflow));
    drop(error);
    drop(actual_factory);
    assert_eq!(live.get(), 1);
    drop(controls);
    assert_eq!(live.get(), 0);
}

#[test]
fn realtime_bounded_bank_retains_final_submission_after_actual_group_boundaries() {
    for graph in graphs() {
        let layout =
            ExecutionUnitLayout::new(&graph, std::iter::repeat_n(1, graph.groups().len())).unwrap();
        let bounded = NeuralPopulation::single_forward(&layout, true).unwrap();
        let resident = NeuralPopulation::single_forward(&layout, false).unwrap();
        assert_eq!(
            bounded,
            NeuralPopulation::from_execution(
                &layout,
                geometry(1, 1, 1),
                GroupSubmissionMechanism::LayeredGraph,
                true
            )
            .unwrap()
        );
        assert_eq!(
            resident,
            NeuralPopulation::from_execution(
                &layout,
                geometry(1, 1, 1),
                GroupSubmissionMechanism::LayeredGraph,
                false
            )
            .unwrap()
        );
        let live = Rc::new(Cell::new(0));
        let controls = Custody::new(&live);
        let mut bank = PreparedOperationBank::try_new(
            bounded.submissions,
            factory::<Custody, NeverObserved>(bounded.shape, Some(controls.clone())),
        )
        .unwrap();
        // Follow the actual graph driver: a multi-group pass submits its
        // initial hidden value, then one result at each semantic group exit.
        // No unit count or family name determines this traversal population.
        let mut reached = Vec::new();
        if graph.groups().len() > 1 {
            reached.push(bank.checkout().unwrap());
            for _ in graph.execution_order() {
                reached.push(bank.checkout().unwrap());
            }
        }
        assert_eq!(reached.len(), resident.submissions);
        assert_eq!(
            bank.remaining(),
            1,
            "bounded finish must still own its exact final slot"
        );
        let before = live.get();
        reached.push(bank.checkout().unwrap());
        assert_eq!(live.get(), before, "checkout moves existing source custody");
        assert_eq!(bank.remaining(), 0);
        assert_eq!(bank.checkout().unwrap_err().prepared, bounded.submissions);
        drop(reached);
        drop(bank);
        drop(controls);
        retire_to(&live, 0);
    }
}
