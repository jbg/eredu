use super::*;
mod legacy;
#[derive(Debug)]
struct Unpriced;
impl WorkspaceMechanisms for Unpriced {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
fn tensor(context: &WorkspaceContext, storage: Rc<Storage>) -> WorkspaceTensor {
    WorkspaceTensor {
        layout: WorkspaceLayout::new(&[0], WorkspaceDtype::Float32).unwrap(),
        storage,
        context: context.identity.clone(),
        imported_existing: false,
    }
}
fn scalar_view(v: &WorkspaceTraceReport) -> impl fmt::Debug + PartialEq {
    (
        v.total_bytes,
        v.retained_bytes,
        v.transient_bytes,
        (
            v.tensor_buffers.total_bytes,
            v.tensor_buffers.retained_bytes,
            v.tensor_buffers.transient_bytes,
        ),
        v.host_workspace_bytes,
        v.state
            .as_ref()
            .map(|x| (x.retained_bytes, x.displaced_bytes, x.transient_bytes)),
        v.residual.as_ref().map(|x| {
            (
                x.total_bytes,
                x.retained_bytes,
                x.displaced_bytes,
                x.transient_bytes,
                x.opening_storage,
                x.closing_storage,
            )
        }),
        v.unpriced_operations.clone(),
        v.unpriced_host_operations.clone(),
        v.assumptions.clone(),
    )
}
#[test]
fn fixed_report_matches_independent_legacy_union_unknown_and_displacement_order() {
    for mode in 0..192usize {
        let context = WorkspaceContext::new(Unpriced);
        let bytes = match mode % 4 {
            0 => Some(7),
            1 => Some(u64::MAX),
            2 => None,
            _ => Some(0),
        };
        let a = context.new_storage(bytes, vec![]);
        let mut b = context.new_storage(Some(11), vec![]);
        // A collapsed source envelope can represent multiple actual backings.
        Rc::get_mut(&mut b).unwrap().maximum_allocations = if mode & 32 != 0 { 3 } else { 1 };
        let c = context.new_storage(Some(0), vec![a.clone(), b.clone(), a.clone()]);
        let d = context.new_storage(Some(13), vec![c.clone()]);
        let all = [a, b, c, d];
        let retained = if mode & 4 == 0 {
            vec![
                tensor(&context, all[3].clone()),
                tensor(&context, all[1].clone()),
                tensor(&context, all[3].clone()),
            ]
        } else {
            vec![tensor(&context, all[1].clone())]
        };
        if mode & 8 != 0 {
            let roots = if mode & 16 != 0 {
                vec![WorkspaceExistingStorage {
                    storage: all[1].clone(),
                    context: context.identity.clone(),
                }]
            } else {
                vec![]
            };
            context
                .set_borrowed_storage(WorkspaceBorrowedStorage::new(&context, &roots).unwrap())
                .unwrap();
        }
        let opening = if mode % 3 == 0 {
            None
        } else if mode % 3 == 1 {
            Some(vec![])
        } else {
            Some(vec![all[0].clone(), all[2].clone()])
        };
        *context.trace.borrow_mut() = Trace {
            report_finished: false,
            opening_state: opening,
            allocations: if mode & 32 == 0 {
                vec![all[0].clone(), all[1].clone()]
            } else {
                vec![all[1].clone(), all[0].clone()]
            },
            scratch: 1,
            scratch_overflow: false,
            placed_scratch: vec![],
            scratch_sources: vec![],
            host_workspace: 3,
            host_staging_incomplete: mode & 128 != 0,
            operations: vec![WorkspaceOperation {
                kind: WorkspaceOperationKind::Contiguous,
                inputs: vec![],
                outputs: vec![],
            }],
            missing: if mode & 64 != 0 { vec![0] } else { vec![] },
            missing_host: if mode & 128 != 0 { vec![0] } else { vec![] },
            assumptions: vec!["independent fixture".into()],
        };
        let old = legacy::report(&context, &retained);
        let new = context.report(&retained);
        match (old, new) {
            (Ok(old), Ok(new)) => {
                assert_eq!(scalar_view(&old), scalar_view(&new), "mode {mode}");
                assert_eq!(old.closing_storage, new.closing_storage, "mode {mode}");
                assert_eq!(old.opening_storage, new.opening_storage, "mode {mode}");
                assert_eq!(
                    format!("{:?}", old.operations),
                    format!("{:?}", new.operations)
                );
                if let (Some(a), Some(b)) = (&old.residual, &new.residual) {
                    assert!(a.borrowed_storage.same_identity(&b.borrowed_storage));
                }
            }
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "mode {mode}"),
            (a, b) => panic!("mode {mode}: legacy={a:?}, fixed={b:?}"),
        }
    }
}
fn inputs<'a>(
    opening: Option<&'a [usize]>,
    allocations: &'a [usize],
    closing: &'a [usize],
    borrowed: Option<&'a [usize]>,
) -> WorkspaceReportInputs<'a> {
    WorkspaceReportInputs {
        opening,
        allocations,
        closing,
        borrowed,
        scratch: 0,
        host_workspace: Some(0),
        tensor_complete: true,
    }
}
#[test]
fn fixed_report_duplicate_roots_cycles_and_each_exact_destination() {
    let nodes = [
        WorkspaceReportNode {
            bytes: Some(3),
            alias_start: 0,
            alias_count: 2,
        },
        WorkspaceReportNode {
            bytes: Some(5),
            alias_start: 2,
            alias_count: 1,
        },
    ];
    let graph = WorkspaceReportGraph::new(&nodes, &[1, 1, 0]).unwrap();
    let repeated = [0usize; 65];
    let mut exact =
        WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(2, 3).unwrap()).unwrap();
    let v = exact
        .report(
            graph,
            inputs(Some(&repeated), &[], &repeated, Some(&repeated)),
        )
        .unwrap();
    assert_eq!(v.state.unwrap().retained_bytes, Some(8));
    assert_eq!(v.residual.unwrap().retained_bytes, Some(5));
    for layout in [
        WorkspaceReportLayout::new(1, 3).unwrap(),
        WorkspaceReportLayout::new(2, 2).unwrap(),
    ] {
        let mut short = WorkspaceReportWorkspace::new(layout).unwrap();
        assert_eq!(
            short
                .report(graph, inputs(Some(&[]), &[], &repeated, None))
                .unwrap_err(),
            WorkspaceReportError::Capacity
        );
    }
    for destination in 0..5 {
        let mut short =
            WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(2, 3).unwrap()).unwrap();
        short.scratch.shorten_for_test(destination);
        let before = format!("{short:?}");
        assert_eq!(
            short
                .report(
                    graph,
                    inputs(Some(&repeated), &[], &repeated, Some(&repeated))
                )
                .unwrap_err(),
            WorkspaceReportError::Capacity
        );
        assert_eq!(format!("{short:?}"), before);
    }
    // Mark/work destinations each have the same checked N capacity. All five
    // actual reserve sites are failed independently below, not inferred from N.
    let empty = WorkspaceReportGraph::new(&[], &[]).unwrap();
    let mut zero =
        WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(0, 0).unwrap()).unwrap();
    assert_eq!(
        zero.report(empty, inputs(Some(&[]), &[], &[], Some(&[])))
            .unwrap()
            .residual
            .unwrap()
            .total_bytes,
        Some(0)
    );
}
#[test]
fn fixed_report_preserves_unknown_before_and_after_checked_overflow() {
    let nodes = [
        WorkspaceReportNode {
            bytes: None,
            alias_start: 0,
            alias_count: 0,
        },
        WorkspaceReportNode {
            bytes: Some(u64::MAX),
            alias_start: 0,
            alias_count: 0,
        },
        WorkspaceReportNode {
            bytes: Some(1),
            alias_start: 0,
            alias_count: 0,
        },
    ];
    let graph = WorkspaceReportGraph::new(&nodes, &[]).unwrap();
    let mut w = WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(3, 0).unwrap()).unwrap();
    assert_eq!(
        w.report(graph, inputs(Some(&[]), &[0, 1, 2], &[], None))
            .unwrap()
            .total_bytes,
        None
    );
    assert_eq!(
        w.report(graph, inputs(Some(&[]), &[1, 2, 0], &[], None))
            .unwrap_err(),
        WorkspaceReportError::Overflow
    );
    // Closing roots are visited in reverse order, exactly as the old stack.
    assert_eq!(
        w.report(graph, inputs(None, &[], &[2, 1, 0], None))
            .unwrap()
            .total_bytes,
        Some(0)
    );
    assert_eq!(
        w.report(graph, inputs(None, &[], &[0, 2, 1], None))
            .unwrap_err(),
        WorkspaceReportError::Overflow
    );
}
#[test]
fn fixed_report_real_reserve_failures_retain_each_preceding_payload() {
    let layout = WorkspaceReportLayout::new(4, 7).unwrap();
    let expected = [
        0,
        Layout::array::<Node<usize>>(4).unwrap().size(),
        Layout::array::<Node<usize>>(4).unwrap().size() + Layout::array::<usize>(7).unwrap().size(),
        Layout::array::<Node<usize>>(4).unwrap().size()
            + Layout::array::<usize>(7).unwrap().size()
            + Layout::array::<Frame>(4).unwrap().size(),
        Layout::array::<Node<usize>>(4).unwrap().size()
            + Layout::array::<usize>(7).unwrap().size()
            + Layout::array::<Frame>(4).unwrap().size()
            + Layout::array::<u8>(4).unwrap().size(),
    ];
    for (site, bytes) in expected.into_iter().enumerate() {
        let (scratch, cause) = Scratch::<usize>::new(layout, Some(site)).unwrap_err();
        let e = WorkspaceReportConstructionError { scratch, cause };
        assert_eq!(e.destination(), Some(site));
        assert_eq!(e.retained_heap_bytes(), bytes);
        assert!(std::error::Error::source(&e).is_some());
    }
    fn send_sync<T: Send + Sync>() {}
    send_sync::<WorkspaceReportWorkspace>();
    send_sync::<WorkspaceReportConstructionError>();
}
#[test]
fn fixed_report_invalid_sources_and_overflow_do_not_mutate_destinations() {
    assert_eq!(
        WorkspaceReportLayout::new(usize::MAX, 0).unwrap_err(),
        WorkspaceReportError::Overflow
    );
    assert_eq!(
        WorkspaceReportGraph::new(
            &[WorkspaceReportNode {
                bytes: Some(0),
                alias_start: usize::MAX,
                alias_count: 1
            }],
            &[]
        )
        .unwrap_err(),
        WorkspaceReportError::Overflow
    );
    let graph = WorkspaceReportGraph::new(
        &[WorkspaceReportNode {
            bytes: Some(4),
            alias_start: 0,
            alias_count: 0,
        }],
        &[],
    )
    .unwrap();
    let mut w = WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(1, 0).unwrap()).unwrap();
    w.report(graph, inputs(None, &[0], &[], None)).unwrap();
    let before = format!("{:?}", w);
    assert_eq!(
        w.report(graph, inputs(None, &[0, 0], &[], None))
            .unwrap_err(),
        WorkspaceReportError::Source
    );
    assert_eq!(format!("{:?}", w), before);
    assert_eq!(
        w.report(graph, inputs(None, &[], &[1], None)).unwrap_err(),
        WorkspaceReportError::Source
    );
    assert_eq!(format!("{:?}", w), before);
}
#[test]
fn ordinary_lifecycle_counts_retired_spans_and_total_constructor_exhaustion() {
    let context = WorkspaceContext::new(Unpriced);
    let clone = context.clone();
    for n in 1..=5 {
        let a = WorkspaceExistingStorage::new(Some(7), &context);
        drop(a);
        context.begin_span();
        assert_eq!(clone.report_workspace_layout().unwrap().nodes(), n);
        assert_eq!(context.report(&[]).unwrap().total_bytes, Some(0));
    }
    context.tracing_started.counts.set(Some((usize::MAX, 0)));
    let root = WorkspaceExistingStorage::new(Some(1), &context); // stays total
    assert_eq!(
        context.report_workspace_layout().unwrap_err(),
        WorkspaceReportError::Overflow
    );
    drop(root);
    assert!(context.report(&[]).is_err());
}

#[derive(Debug)]
struct CopyMechanism;
impl WorkspaceMechanisms for CopyMechanism {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let outputs = op
            .outputs
            .iter()
            .map(|layout| {
                Ok(if matches!(op.kind, WorkspaceOperationKind::Contiguous) {
                    WorkspaceOutputStorage::AliasInput(0)
                } else {
                    WorkspaceOutputStorage::Allocate(layout.bytes()?)
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Some(WorkspaceOperationBound {
            outputs,
            scratch_bytes: 0,
            assumptions: "fixed copy fixture".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "fixed copy fixture has no host work".into(),
        }))
    }
}
#[test]
fn isolated_copy_private_report_counts_rebased_roots_and_each_actual_copy() {
    let context = WorkspaceContext::new(CopyMechanism);
    let root = WorkspaceExistingStorage::new(Some(16), &context);
    let borrowed = WorkspaceBorrowedStorage::new(&context, [&root, &root]).unwrap();
    let a = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[2], WorkspaceDtype::Float32).unwrap(),
        &root,
        &context,
    )
    .unwrap();
    let plan = WorkspaceIsolatedCopyPlan::prepare(&context, &borrowed, &[a.clone(), a]).unwrap();
    assert_eq!(plan.incremental_bytes(), Some(16));
    assert_eq!(
        plan.report().state.as_ref().unwrap().retained_bytes,
        Some(32)
    );
    assert_eq!(context.report_workspace_layout().unwrap().nodes(), 1);
    assert!(plan.source_storage().same_identity(&borrowed));
}

#[test]
fn ordinary_report_checks_foreign_context_before_exhausted_scratch() {
    let context = WorkspaceContext::new(Unpriced);
    let foreign = WorkspaceContext::new(Unpriced);
    let value = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap(),
        &foreign,
    )
    .unwrap();
    context.tracing_started.counts.set(None);
    let old = legacy::report(&context, &[value.clone()]).unwrap_err();
    let new = context.report(&[value]).unwrap_err();
    assert_eq!(old.to_string(), new.to_string());
    assert!(!std::error::Error::source(&new).is_some_and(|e| e.is::<WorkspaceReportError>()));
}

#[test]
fn fixed_report_n_deep_chain_cycle_and_borrowed_duplicates_use_exact_buffers() {
    const N: usize = 97;
    for cyclic in [false, true] {
        let edges: Vec<_> = (1..N).chain(cyclic.then_some(0)).collect();
        let nodes: Vec<_> = (0..N)
            .map(|i| WorkspaceReportNode {
                bytes: Some((i + 1) as u64),
                alias_start: i.min(edges.len()),
                alias_count: usize::from(i + 1 < N || cyclic),
            })
            .collect();
        let graph = WorkspaceReportGraph::new(&nodes, &edges).unwrap();
        let roots = vec![0; N * 3];
        let allocations: Vec<_> = (0..N).collect();
        let mut workspace =
            WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(N, edges.len()).unwrap())
                .unwrap();
        let before = workspace.retained_heap_bytes();
        let report = workspace
            .report(
                graph,
                inputs(Some(&roots), &allocations, &roots, Some(&roots)),
            )
            .unwrap();
        let sum = (N * (N + 1) / 2) as u64;
        assert_eq!(report.total_bytes, Some(sum));
        assert_eq!(report.retained_bytes, Some(sum));
        assert_eq!(report.transient_bytes, Some(0));
        let state = report.state.unwrap();
        assert_eq!(state.retained_bytes, Some(sum));
        assert_eq!(state.displaced_bytes, Some(0));
        let residual = report.residual.unwrap();
        assert_eq!(residual.total_bytes, Some(sum - 1));
        assert_eq!(residual.retained_bytes, Some(sum - 1));
        assert_eq!(workspace.retained_heap_bytes(), before);
    }
}

#[test]
fn fixed_flat_residual_uses_source_identity_order_after_allocation_order() {
    let mut workspace =
        WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(3, 2).unwrap()).unwrap();
    // Main allocation is unknown in both cases. The residual visits the union
    // in source-index order (the flat counterpart of the legacy pointer map).
    for unknown_first in [true, false] {
        let unknown = if unknown_first { 0 } else { 2 };
        let values = if unknown_first {
            [None, Some(u64::MAX), Some(1)]
        } else {
            [Some(u64::MAX), Some(1), None]
        };
        let nodes = std::array::from_fn::<_, 3, _>(|i| WorkspaceReportNode {
            bytes: values[i],
            alias_start: 0,
            alias_count: if i == unknown { 2 } else { 0 },
        });
        let edges = if unknown_first { [1, 2] } else { [0, 1] };
        let graph = WorkspaceReportGraph::new(&nodes, &edges).unwrap();
        let result = workspace.report(graph, inputs(Some(&[]), &[unknown], &[], Some(&[])));
        if unknown_first {
            let report = result.unwrap();
            assert_eq!(report.total_bytes, None);
            assert_eq!(report.residual.unwrap().total_bytes, None);
        } else {
            assert_eq!(result.unwrap_err(), WorkspaceReportError::Overflow);
        }
    }
}

thread_local! {
    static RETIRE_PROBE: Cell<Option<bool>> = const { Cell::new(None) };
}
pub(super) fn probe_retirement(context: &WorkspaceContext) {
    RETIRE_PROBE.with(|probe| {
        if probe.get().is_some() {
            let trace = context.trace.try_borrow_mut();
            let borrowed = context.borrowed.try_borrow_mut();
            probe.set(Some(trace.is_ok() && borrowed.is_ok()));
        }
    });
}
#[test]
fn ordinary_scratch_retires_after_both_loans_on_success_and_overflow() {
    for overflow in [false, true] {
        let context = WorkspaceContext::new(Unpriced);
        let a = context.new_storage(Some(if overflow { u64::MAX } else { 7 }), vec![]);
        let b = context.new_storage(Some(1), vec![]);
        let retained = [tensor(&context, a.clone()), tensor(&context, b.clone())];
        context.trace.borrow_mut().allocations = vec![a.clone(), b.clone()];
        let before = (Rc::strong_count(&a), Rc::strong_count(&b));
        RETIRE_PROBE.with(|p| p.set(Some(false)));
        let result = context.report(&retained);
        assert_eq!(result.is_err(), overflow);
        assert_eq!(RETIRE_PROBE.with(Cell::take), Some(true));
        assert_eq!((Rc::strong_count(&a), Rc::strong_count(&b)), before);
        context.begin_span();
        drop(retained);
        assert_eq!(Rc::strong_count(&a), 1);
        assert_eq!(Rc::strong_count(&b), 1);
    }
    assert!(
        WorkspaceContext::report_lifecycle_bytes().unwrap()
            > Layout::new::<[Cell<usize>; 2]>()
                .extend(Layout::new::<Cell<bool>>())
                .unwrap()
                .0
                .pad_to_align()
                .size()
    );
}
