use super::*;
use crate::backend::nn::grouped::{TopKGroupScoring, TopKGroupSelector, TopKGroupSelectorConfig};
use safemlx::{Array, OperationEvent};

#[test]
fn original_router_keeps_global_cutoff_ties_and_gpu_no_tie_order_without_eager_eval() {
    const CHILD: &str = "EREDU_ROUTER_COMPONENT_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                concat!(
                    "composition::mlx::session::model_session::text_quote::prefill_tests::router_tests::",
                    "original_router_keeps_global_cutoff_ties_and_gpu_no_tie_order_without_eager_eval"
                ),
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{output}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(output.contains("ROUTER_COMPONENT_OK"), "{output}");
        return;
    }
    let required = std::env::var_os("EREDU_REQUIRE_ROUTER_QUALIFICATION").is_some()
        || std::env::var_os("EREDU_REQUIRE_CPU_WORKER_STARTUP_QUALIFICATION").is_some();
    // Fresh-process preparation precedes any ordinary stream/Scheduler. Only
    // explicit unsupported producer layouts may skip a general-platform test.
    use eredu_runtime::working_memory::SharedNativeInitializationCustody as Custody;
    macro_rules! qualified {
        ($plan:expr, $unknown:pat) => {
            match $plan {
                Ok(_) => {}
                Err($unknown) => {
                    assert!(
                        !required,
                        "selected router validation requires native qualification"
                    );
                    println!("ROUTER_COMPONENT_OK: typed unknown producer");
                    return;
                }
                Err(error) => panic!("unexpected router layout refusal: {error:?}"),
            }
        };
    }
    qualified!(
        safemlx::PreparedMetalDevice::<Custody>::layout(),
        safemlx::MetalDeviceCause::UnknownLayout
    );
    qualified!(
        safemlx::PreparedScheduler::<Custody>::layout(),
        safemlx::SchedulerCause::UnknownLayout
    );
    qualified!(
        safemlx::PreparedCpuStream::<Custody>::layout(),
        safemlx::StreamRegistrationCause::UnknownLayout
    );
    qualified!(
        safemlx::PreparedCpuWorker::<Custody>::layout(),
        safemlx::CpuWorkerCause::UnknownLayout
    );
    crate::backend::managed_memory::input_allocator::prepare_admitted(
        &crate::backend::managed_memory::domain(),
    )
    .unwrap();
    // Create all cold source/runtime owners before either request retains R.
    let prepared = [
        OriginalOperationFixture::prepare(true),
        OriginalOperationFixture::prepare(true),
    ];
    let stream = prepared[0].stream.clone();
    let config = TopKGroupSelectorConfig::new(
        2,
        4,
        3,
        TopKGroupScoring::SelectedSoftmax,
        false,
        0.0,
        1.0,
        1,
        1,
        false,
        false,
        None,
        false,
        false,
    )
    .unwrap();
    let mut selector = TopKGroupSelector::new_with_quantization(config, None, &stream).unwrap();
    assert!(
        crate::backend::managed_memory::router::is_ready(),
        "qualified selector construction must retain its actual admitted CPU worker"
    );
    let weights = Array::from_slice(
        &[
            3.0f32, 5.0, 4.0, 3.0, 1.0, 3.0, 3.0, 4.0, 2.0, 1.0, 2.0, 1.0,
        ],
        &[4, 3],
    );
    weights.evaluated().unwrap();
    selector.weight.value = weights.clone();
    // First input has a cutoff tie in row zero and distinct scores in row one.
    // The second has no cutoff ties anywhere. The predicate is global, not rowwise.
    for (input, mut prepared) in [
        [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    ]
    .into_iter()
    .zip(prepared)
    {
        let stream = prepared.stream.clone();
        let input = Array::from_slice(&input, &[2, 3]);
        input.evaluated().unwrap();
        let expected = selector
            .select_with_selection_bias(&input, None, &stream)
            .unwrap();
        let expected_ids = expected
            .indices
            .evaluated()
            .unwrap()
            .try_to_vec::<u32>()
            .unwrap();
        let expected_scores = expected
            .scores
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let expected_weights = expected
            .weights
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        drop(expected);
        // Reference evaluation can attach completed ordinary events to leaves.
        // Settle them before crossing into the new original domain.
        for leaf in [&input, &weights] {
            leaf.evaluated().unwrap();
        }
        let baseline_unquoted = prepared.pool.unquoted_owner_count().unwrap();
        // Exact selected worker population: 68 potential primitives, one seed,
        // two completed inputs and three outputs, plus the Eval Synchronizer.
        let graph = OperationEvent::resident_graph_layout(68, 1, 2).unwrap();
        let base = OperationEvent::eval_record_layout(69, 2, 69).unwrap();
        let traversal =
            OperationEvent::eval_traversal_layout(safemlx::OperationEvalTraversalLimits {
                roots: 3,
                arrays: 1 + 68 + 1 + 2 + 3,
                tape_entries: 69,
                input_edges: 77 + 3,
                output_slots: 69,
                streams: 2,
                captures: base.capture_slots().max(1),
            })
            .unwrap();
        let cpu = OperationEvent::cpu_argpartition_layout(true).unwrap();
        let receipts = OperationEvent::router_receipt_layout().unwrap();
        let controls = [
            graph.control_bytes().unwrap(),
            base.query_control_bytes().unwrap(),
            traversal.query_control_bytes().unwrap(),
            cpu.control_bytes().unwrap(),
            receipts.control_bytes().unwrap(),
            crate::backend::managed_memory::router::control_bytes().unwrap(),
            std::mem::size_of::<[&Array; 3]>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .unwrap();
        POINTWISE_CONTROLS.with(|slot| {
            assert!(
                slot.replace(Some(u64::try_from(controls).unwrap()))
                    .is_none()
            )
        });
        let reset = PointwiseControlsReset;
        let (output, pool, custody) = with_prepared_original_operation_controls(
            None,
            &mut prepared,
            |controls, observer, pool, bank| {
                assert!(bank.is_none());
                for leaf in [&input, &weights] {
                    OperationEvent::validate_traversal_leaf(leaf, observer).unwrap();
                }
                assert!(!observer.status().has_work());
                let host = OperationEvent::prepare_resident_graph(graph, observer).unwrap();
                let output = selector
                    .select_with_selection_bias(&input, None, &stream)
                    .unwrap();
                assert!(
                    !observer.status().has_work(),
                    "router construction evaluated eagerly"
                );
                drop(host);
                let roots = [&output.indices, &output.scores, &output.weights];
                let event = safemlx::transforms::async_eval_with_original_prepared_traversal(
                    roots.iter().copied(),
                    observer,
                    &stream,
                    &traversal,
                )
                .unwrap();
                event.synchronize().unwrap();
                assert!(
                    roots
                        .iter()
                        .all(|root| root.completed_in_original_scope(observer).is_ok())
                );
                // Borrow the completed data under its authentic Scope; a second
                // ordinary Eval would create unrelated lifetime owners.
                assert_eq!(
                    output
                        .indices
                        .completed_in_original_scope(observer)
                        .unwrap()
                        .try_as_slice::<u32>()
                        .unwrap(),
                    expected_ids.as_slice()
                );
                assert_eq!(
                    output
                        .scores
                        .completed_in_original_scope(observer)
                        .unwrap()
                        .try_as_slice::<f32>()
                        .unwrap(),
                    expected_scores.as_slice()
                );
                assert_eq!(
                    output
                        .weights
                        .completed_in_original_scope(observer)
                        .unwrap()
                        .try_as_slice::<f32>()
                        .unwrap(),
                    expected_weights.as_slice()
                );
                // Persistent source descriptors may also carry this completed
                // Event. Authenticate their handoff before the original Scope
                // retires, while both exact external owners are still borrowed.
                for leaf in [&input, &weights] {
                    observer.validate_completed_array(leaf).unwrap();
                }
                safemlx::try_with_submission_retirement(|| drop(event)).unwrap();
                (output, pool.clone(), controls.metadata_custody())
            },
        );
        drop(reset);
        assert!(
            expected_weights
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
        // This single observer retains only raw host custody. Every request
        // Graph/Record arena, native budget, Scope/Roots/recovery and publication
        // owner retains the same raw owner until its own storage retires.
        assert!(
            !custody.is_sole_owner(),
            "live router output must retain its actual request custody"
        );
        drop(output);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            disk::reclaim();
            custody.is_sole_owner()
        });
        assert_eq!(pool.unquoted_owner_count().unwrap(), baseline_unquoted);
        drop(custody);
        prepared.finish();
    }
    println!("ROUTER_COMPONENT_OK: actual tie and no-tie execution");
}
