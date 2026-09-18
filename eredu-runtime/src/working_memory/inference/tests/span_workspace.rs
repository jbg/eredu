use super::*;

#[derive(Debug)]
struct SpanFacts;
impl WorkspaceMechanisms for SpanFacts {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 11,
            assumptions: "exact fixture outputs and per-operation scratch".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 7,
            assumptions: "exact fixture disjoint host scratch".into(),
        }))
    }
}
#[test]
fn actual_schedule_retains_new_state_and_nonzero_operations_without_opening_credit() {
    let g = geometry();
    let context = WorkspaceContext::new(SpanFacts);
    let root = WorkspaceExistingStorage::new(Some(4096), &context);
    let mut current = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap(),
        &root,
        &context,
    )
    .unwrap();
    let mut actual = Vec::new();
    let report = quote_inference_workspace(g, |span| {
        actual.push(span.clone());
        context.begin_state_span([&current])?;
        let n = match span {
            InferenceWorkspaceSpan::Sampling(_) => unreachable!("model-only traversal fixture"),
            InferenceWorkspaceSpan::Prefill(c) => c.input.end - c.input.start,
            InferenceWorkspaceSpan::Decode { .. } => 1,
        };
        let value = WorkspaceTensor::full_f32(1.25, &[n as i32], &context)?;
        current = value.square(&context)?;
        context.report(&[current.clone()])
    })
    .unwrap();
    let plan = report.span_workspace_plan();
    assert_eq!(plan.geometry(), g);
    assert_eq!(plan.generation_forward_count(), Some(5));
    assert_eq!(
        plan.generation_records().unwrap().map(|record| record.span()).collect::<Vec<_>>(),
        actual[..5].iter().collect::<Vec<_>>(),
    );
    assert_eq!(
        plan.records()
            .iter()
            .map(|r| r.span().clone())
            .collect::<Vec<_>>(),
        actual
    );
    assert_eq!(
        plan.records()
            .iter()
            .map(|r| r.new_allocation_bytes())
            .collect::<Vec<_>>(),
        vec![Some(60), Some(60), Some(44), Some(44), Some(44), Some(44)]
    );
    assert_eq!(
        plan.records()
            .iter()
            .map(|r| (r.new_tensor_allocation_bytes(), r.host_workspace_bytes()))
            .collect::<Vec<_>>(),
        vec![
            (Some(46), Some(14)),
            (Some(46), Some(14)),
            (Some(30), Some(14)),
            (Some(30), Some(14)),
            (Some(30), Some(14)),
            (Some(30), Some(14)),
        ]
    );
    assert!(
        report.transient().bytes().unwrap() > 4096,
        "full diagnostics keep displaced original backing"
    );
    let InferenceWorkspaceSpan::Prefill(last) = plan.records()[2].span() else {
        panic!("third prefill span")
    };
    assert_eq!(last.input, 6..7);
    assert_eq!(last.position, 8);
    assert_eq!(last.output, OutputDemand::LastPosition);
    assert!(matches!(
        plan.records()[5].span(),
        InferenceWorkspaceSpan::Decode {
            index: 2,
            position: 11,
            ..
        }
    ));
    assert!(plan.same_plan(report.clone().span_workspace_plan()));
}
#[test]
fn missing_host_or_state_remains_unknown_in_span_diagnostics() {
    let unknown_host =
        quote_inference_workspace(geometry(), |_| Ok::<_, Error>(trace(8, None, true))).unwrap();
    assert!(unknown_host
        .span_workspace_plan()
        .records()
        .iter()
        .all(|r| r.new_allocation_bytes().is_none()));
    assert!(unknown_host
        .span_workspace_plan()
        .records()
        .iter()
        .all(|r| r.new_tensor_allocation_bytes() == Some(8) && r.host_workspace_bytes().is_none()));
    let no_state = quote_inference_workspace(geometry(), |_| {
        let context = WorkspaceContext::new(SpanFacts);
        context.begin_span();
        let x = WorkspaceTensor::full_f32(2.5, &[3], &context)?;
        let report = context.report(&[x])?;
        assert!(report.total_bytes.is_some());
        Ok::<_, Error>(report)
    })
    .unwrap();
    assert!(no_state
        .span_workspace_plan()
        .records()
        .iter()
        .all(|r| r.new_allocation_bytes().is_none()));
    assert!(no_state
        .span_workspace_plan()
        .records()
        .iter()
        .all(
            |r| r.new_tensor_allocation_bytes() == Some(23) && r.host_workspace_bytes() == Some(7)
        ));
}

#[test]
fn domain_bounds_keep_their_actual_span_instead_of_combining_unrelated_maxima() {
    let mut index = 0;
    let report = quote_inference_workspace(geometry(), |_| {
        index += 1;
        Ok::<_, Error>(match index {
            1 => trace(100, Some(10), false),
            2 => trace(4, Some(80), false),
            _ => trace(8, Some(5), false),
        })
    })
    .unwrap();
    let records = report.span_workspace_plan().records();
    assert_eq!(records[0].new_tensor_allocation_bytes(), Some(100));
    assert_eq!(records[0].host_workspace_bytes(), Some(10));
    assert_eq!(records[0].new_allocation_bytes(), Some(110));
    assert_eq!(records[1].new_tensor_allocation_bytes(), Some(4));
    assert_eq!(records[1].host_workspace_bytes(), Some(80));
    assert_eq!(records[1].new_allocation_bytes(), Some(84));
    assert_eq!(report.transient().bytes(), Some(110));
}
#[test]
fn plan_identity_and_actual_spare_capacity_are_preserved_without_deep_clone() {
    let make = || {
        quote_inference_workspace(geometry(), |_| Ok::<_, Error>(trace(8, Some(5), false))).unwrap()
    };
    let a = make();
    let b = make();
    let plan = a.span_workspace_plan();
    assert!(!plan.same_plan(b.span_workspace_plan()));
    assert!(plan.same_plan(&plan.clone()));
    assert!(
        plan.capacity_bytes().unwrap()
            >= (plan.records().len() * std::mem::size_of::<InferenceSpanWorkspaceRecord>()) as u64
    );
}
