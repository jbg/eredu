use super::*;
use crate::{BlockwiseAttentionBackend, BlockwiseAttentionSpec};

#[test]
fn rounded_capped_biased_blocks_preserve_two_pass_state_and_refuse_foreign_bias() {
    use crate::{AttentionArithmetic, BlockwiseAttentionOptions};
    let context = context();
    let q = tensor(&[2, 4, 3, 16], WorkspaceDtype::Float32, &context);
    let options = BlockwiseAttentionOptions {
        arithmetic: AttentionArithmetic::InputScores,
        softcap: Some(1.75),
    };
    let mut accumulator =
        WorkspaceBackend::begin_blockwise_attention_with_options(spec(&q), options, &context)
            .unwrap();
    assert!(WorkspaceBackend::begin_blockwise_value_pass(&mut accumulator, &context).is_err());
    for pass in 0..options.passes() {
        if pass == 1 {
            WorkspaceBackend::begin_blockwise_value_pass(&mut accumulator, &context).unwrap();
            assert!(
                WorkspaceBackend::begin_blockwise_value_pass(&mut accumulator, &context).is_err()
            );
        }
        for (start, end) in [(0, 2), (5, 9), (9, 11)] {
            let k = tensor(&[2, 2, end - start, 16], WorkspaceDtype::Float32, &context);
            let v = tensor(&[2, 2, end - start, 24], WorkspaceDtype::Float32, &context);
            let bias = tensor(&[3, end - start], WorkspaceDtype::Float32, &context);
            if start == 0 {
                let foreign = WorkspaceContext::new(AllocatingMechanism);
                let wrong = tensor(&[3, end - start], WorkspaceDtype::Float32, &foreign);
                let count = context.report(&[]).unwrap().operations.len();
                assert!(
                    WorkspaceBackend::accumulate_blockwise_attention_with_bias(
                        &mut accumulator,
                        start as i64,
                        end as i64,
                        k.clone(),
                        v.clone(),
                        Some(&wrong),
                        &context
                    )
                    .is_err()
                );
                assert_eq!(context.report(&[]).unwrap().operations.len(), count);
            }
            WorkspaceBackend::accumulate_blockwise_attention_with_bias(
                &mut accumulator,
                start as i64,
                end as i64,
                k,
                v,
                Some(&bias),
                &context,
            )
            .unwrap();
        }
    }
    let output = WorkspaceBackend::finish_blockwise_attention(accumulator, &context).unwrap();
    let report = context.report(&[output]).unwrap();
    assert_eq!(report.operations.len(), 8);
    for (index, op) in report.operations.iter().enumerate() {
        let WorkspaceOperationKind::BlockwiseAttention { policy, stage } = &op.kind else {
            panic!("shared blockwise worker")
        };
        assert_eq!(policy.options, options);
        if let WorkspaceBlockwiseStage::Accumulate {
            previous,
            value_pass,
            bias,
            ..
        } = stage
        {
            assert!(*bias);
            assert_eq!(*value_pass, index >= 4);
            assert_eq!(*previous, index != 1 && index != 4);
            assert_eq!(op.outputs.len(), if *value_pass { 1 } else { 3 });
        }
    }
    assert_eq!(report.operations.last().unwrap().inputs.len(), 1);
}

fn tensor(shape: &[i32], dtype: WorkspaceDtype, context: &WorkspaceContext) -> WorkspaceTensor {
    WorkspaceTensor::existing(WorkspaceLayout::new(shape, dtype).unwrap(), context).unwrap()
}
fn spec(queries: &WorkspaceTensor) -> BlockwiseAttentionSpec<'_, WorkspaceTensor> {
    BlockwiseAttentionSpec {
        queries,
        scale: 0.25,
        mask: None,
        query_start: 8,
        context_end: 11,
        sliding_window: Some(4),
        prefix_tokens: 2,
        sinks: None,
    }
}

#[test]
fn gapped_prefix_and_window_blocks_preserve_absolute_policy_and_online_state() {
    let context = context();
    let q = tensor(&[2, 4, 3, 16], WorkspaceDtype::Float32, &context);
    let mask = tensor(&[3, 16], WorkspaceDtype::Float32, &context);
    let sinks = tensor(&[4], WorkspaceDtype::Float32, &context);
    let mut policy = spec(&q);
    policy.mask = Some(&mask);
    policy.sinks = Some(&sinks);
    let mut accumulator = WorkspaceBackend::begin_blockwise_attention(policy, &context).unwrap();
    for (start, end) in [(0, 2), (5, 9), (9, 11)] {
        let k = tensor(&[2, 2, end - start, 16], WorkspaceDtype::Float32, &context);
        let v = tensor(&[2, 2, end - start, 24], WorkspaceDtype::Float32, &context);
        let bytes = WorkspaceBackend::accumulate_blockwise_attention(
            &mut accumulator,
            start as i64,
            end as i64,
            k,
            v,
            &context,
        )
        .unwrap();
        assert_eq!(bytes, (2 * 2 * (end - start) * (16 * 4 + 24 * 4)) as u64);
    }
    let output = WorkspaceBackend::finish_blockwise_attention(accumulator, &context).unwrap();
    assert_eq!(output.shape(), [2, 4, 3, 24]);
    assert_eq!(output.layout().dtype(), WorkspaceDtype::Float32);
    let report = context.report(&[]).unwrap();
    assert_eq!(report.operations.len(), 5);
    for (index, op) in report.operations.iter().enumerate() {
        let WorkspaceOperationKind::BlockwiseAttention { policy, stage } = &op.kind else {
            panic!("blockwise stage expected")
        };
        assert_eq!(
            (policy.query_start, policy.context_end, policy.prefix_tokens),
            (8, 11, 2)
        );
        assert_eq!(policy.mask_origin, Some(-5));
        assert_eq!(policy.sliding_window, Some(4));
        if index == 0 {
            assert_eq!(*stage, WorkspaceBlockwiseStage::Begin);
            assert_eq!(op.outputs[0].dtype(), WorkspaceDtype::Float32);
            assert_eq!(op.outputs[1].shape(), [2, 4, 3, 16]);
        } else if index < 4 {
            assert!(
                matches!(stage, WorkspaceBlockwiseStage::Accumulate { previous, .. } if *previous == (index > 1))
            );
            assert_eq!(op.inputs.len(), if index == 1 { 5 } else { 8 });
            assert_eq!(op.outputs[0].shape(), [2, 4, 3, 1]);
            assert_eq!(op.outputs[2].shape(), [2, 4, 3, 24]);
        } else {
            assert_eq!(*stage, WorkspaceBlockwiseStage::Finish);
        }
    }
    assert!(report.total_bytes.unwrap() > output.layout().bytes().unwrap());
}

#[test]
fn invalid_ranges_geometry_and_context_do_not_advance_the_accumulator() {
    let context = context();
    let foreign = WorkspaceContext::new(AllocatingMechanism);
    let q = tensor(&[2, 4, 3, 16], WorkspaceDtype::Float32, &context);
    let mut accumulator = WorkspaceBackend::begin_blockwise_attention(spec(&q), &context).unwrap();
    let k = tensor(&[2, 2, 2, 16], WorkspaceDtype::Float32, &context);
    let v = tensor(&[2, 2, 2, 24], WorkspaceDtype::Float32, &context);
    WorkspaceBackend::accumulate_blockwise_attention(
        &mut accumulator,
        0,
        2,
        k.clone(),
        v.clone(),
        &context,
    )
    .unwrap();
    let before = context.report(&[]).unwrap();
    for (start, end) in [(1, 3), (-1, 1), (2, 2), (10, 12), (2, 5)] {
        assert!(
            WorkspaceBackend::accumulate_blockwise_attention(
                &mut accumulator,
                start,
                end,
                k.clone(),
                v.clone(),
                &context
            )
            .is_err()
        );
    }
    for bad in [
        tensor(&[2, 3, 2, 16], WorkspaceDtype::Float32, &context),
        tensor(&[2, 2, 2, 16], WorkspaceDtype::Float32, &foreign),
    ] {
        assert!(
            WorkspaceBackend::accumulate_blockwise_attention(
                &mut accumulator,
                2,
                4,
                bad,
                v.clone(),
                &context
            )
            .is_err()
        );
    }
    let bad_v = tensor(&[2, 2, 2, 25], WorkspaceDtype::Float32, &context);
    assert!(
        WorkspaceBackend::accumulate_blockwise_attention(
            &mut accumulator,
            2,
            4,
            k.clone(),
            bad_v,
            &context
        )
        .is_err()
    );
    assert_eq!(
        context.report(&[]).unwrap().operations.len(),
        before.operations.len()
    );
    assert_eq!(context.report(&[]).unwrap().total_bytes, before.total_bytes);
    WorkspaceBackend::accumulate_blockwise_attention(&mut accumulator, 2, 4, k, v, &context)
        .unwrap();
    assert_eq!(
        WorkspaceBackend::finish_blockwise_attention(accumulator, &context)
            .unwrap()
            .shape(),
        [2, 4, 3, 24]
    );
}

#[test]
fn mask_coverage_empty_history_and_invalid_begin_fail_before_dependent_work() {
    let context = context();
    let q = tensor(&[2, 4, 3, 16], WorkspaceDtype::Float32, &context);
    for invalid in [0, -1] {
        let mut policy = spec(&q);
        policy.sliding_window = Some(invalid);
        assert!(WorkspaceBackend::begin_blockwise_attention(policy, &context).is_err());
    }
    let mut policy = spec(&q);
    policy.query_start = i64::MAX;
    assert!(WorkspaceBackend::begin_blockwise_attention(policy, &context).is_err());
    let mask = tensor(&[5, 3, 11], WorkspaceDtype::Float32, &context);
    let mut policy = spec(&q);
    policy.mask = Some(&mask);
    assert!(WorkspaceBackend::begin_blockwise_attention(policy, &context).is_err());
    assert!(context.report(&[]).unwrap().operations.is_empty());
    let empty = WorkspaceBackend::begin_blockwise_attention(spec(&q), &context).unwrap();
    assert!(WorkspaceBackend::finish_blockwise_attention(empty, &context).is_err());
    let mask = tensor(&[3, 4], WorkspaceDtype::Float32, &context);
    let mut policy = spec(&q);
    policy.mask = Some(&mask);
    policy.prefix_tokens = 128;
    let mut accumulator = WorkspaceBackend::begin_blockwise_attention(policy, &context).unwrap();
    let k = tensor(&[2, 2, 2, 16], WorkspaceDtype::Float32, &context);
    let v = tensor(&[2, 2, 2, 24], WorkspaceDtype::Float32, &context);
    assert!(
        WorkspaceBackend::accumulate_blockwise_attention(
            &mut accumulator,
            5,
            7,
            k.clone(),
            v.clone(),
            &context
        )
        .is_err()
    );
    assert_eq!(context.report(&[]).unwrap().operations.len(), 2);
    WorkspaceBackend::accumulate_blockwise_attention(&mut accumulator, 7, 9, k, v, &context)
        .unwrap();
}

#[test]
fn missing_blockwise_native_facts_remain_unknown_through_finish() {
    #[derive(Debug)]
    struct Missing;
    impl WorkspaceMechanisms for Missing {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            Ok(None)
        }
    }
    let context = WorkspaceContext::new(Missing);
    let q = tensor(&[2, 4, 3, 16], WorkspaceDtype::Float32, &context);
    let mut accumulator = WorkspaceBackend::begin_blockwise_attention(spec(&q), &context).unwrap();
    let k = tensor(&[2, 2, 2, 16], WorkspaceDtype::Float32, &context);
    let v = tensor(&[2, 2, 2, 24], WorkspaceDtype::Float32, &context);
    assert_eq!(
        WorkspaceBackend::accumulate_blockwise_attention(&mut accumulator, 0, 2, k, v, &context)
            .unwrap(),
        1280
    );
    let output = WorkspaceBackend::finish_blockwise_attention(accumulator, &context).unwrap();
    let report = context.report(&[output]).unwrap();
    assert_eq!(report.total_bytes, None);
    assert_eq!(report.unpriced_operations, [0, 1, 2]);
    assert_eq!(report.unpriced_host_operations, [0, 1, 2]);
}

#[test]
fn failed_mechanism_fact_preserves_previous_online_state_and_can_retry_the_same_range() {
    use std::{cell::Cell, rc::Rc};
    #[derive(Debug)]
    struct Fail {
        enabled: Rc<Cell<bool>>,
    }
    impl WorkspaceMechanisms for Fail {
        fn host_workspace_bound(
            &self,
            op: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            AllocatingMechanism.host_workspace_bound(op)
        }
        fn operation_bound(
            &self,
            op: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            if self.enabled.get()
                && matches!(
                    op.kind,
                    WorkspaceOperationKind::BlockwiseAttention {
                        stage: WorkspaceBlockwiseStage::Accumulate { previous: true, .. },
                        ..
                    }
                )
            {
                return Err(Error::backend("injected blockwise fact failure"));
            }
            AllocatingMechanism.operation_bound(op)
        }
    }
    let enabled = Rc::new(Cell::new(true));
    let context = WorkspaceContext::new(Fail {
        enabled: enabled.clone(),
    });
    let q = tensor(&[2, 4, 3, 16], WorkspaceDtype::Float32, &context);
    let mut accumulator = WorkspaceBackend::begin_blockwise_attention(spec(&q), &context).unwrap();
    let k = tensor(&[2, 2, 2, 16], WorkspaceDtype::Float32, &context);
    let v = tensor(&[2, 2, 2, 24], WorkspaceDtype::Float32, &context);
    WorkspaceBackend::accumulate_blockwise_attention(
        &mut accumulator,
        0,
        2,
        k.clone(),
        v.clone(),
        &context,
    )
    .unwrap();
    let before = context.report(&[]).unwrap();
    assert!(
        WorkspaceBackend::accumulate_blockwise_attention(
            &mut accumulator,
            2,
            4,
            k.clone(),
            v.clone(),
            &context
        )
        .is_err()
    );
    assert_eq!(context.report(&[]).unwrap().total_bytes, before.total_bytes);
    assert_eq!(context.report(&[]).unwrap().operations.len(), 2);
    enabled.set(false);
    WorkspaceBackend::accumulate_blockwise_attention(&mut accumulator, 2, 4, k, v, &context)
        .unwrap();
    let output = WorkspaceBackend::finish_blockwise_attention(accumulator, &context).unwrap();
    assert_eq!(output.shape(), [2, 4, 3, 24]);
    assert_eq!(context.report(&[]).unwrap().operations.len(), 4);
    assert!(context.report(&[]).unwrap().total_bytes.unwrap() > before.total_bytes.unwrap());
}
