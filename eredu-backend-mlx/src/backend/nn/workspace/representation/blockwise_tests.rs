use eredu_nn::Tensor;
use super::*;
use eredu_nn::workspace::WorkspaceBlockwiseStage;
use eredu_nn::{BlockwiseAttentionBackend, BlockwiseAttentionOptions, BlockwiseAttentionSpec};

fn source(shape: &[i32], dtype: Option<F>, context: &WorkspaceContext) -> WorkspaceTensor {
    WorkspaceTensor::existing(
        context
            .layout(shape, WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(dtype.map(|dtype| WorkspaceRepresentation::new(dtype, false))),
        context,
    )
    .unwrap()
}
#[test]
fn blockwise_scalar_roundtrip_preserves_original_query_and_host_store_qualification() {
    let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for dtype in [Some(F::Float32), Some(F::Float16), Some(F::Bfloat16), None] {
        for arithmetic in [
            eredu_nn::AttentionArithmetic::Fused,
            eredu_nn::AttentionArithmetic::InputScores,
        ] {
            let context = WorkspaceContext::new(facts);
            let q = source(&[1, 2, 2, 4], dtype, &context);
            let k = source(&[1, 1, 2, 4], dtype, &context);
            let v = source(&[1, 1, 2, 3], dtype, &context);
            let options = BlockwiseAttentionOptions {
                arithmetic,
                softcap: Some(1.75),
            };
            let mut accumulator = WorkspaceBackend::begin_blockwise_attention_with_options(
                BlockwiseAttentionSpec {
                    queries: &q,
                    scale: 0.5,
                    mask: None,
                    query_start: 0,
                    context_end: 2,
                    sliding_window: None,
                    prefix_tokens: 0,
                    sinks: None,
                },
                options,
                &context,
            )
            .unwrap();
            for pass in 0..options.passes() {
                if pass == 1 {
                    WorkspaceBackend::begin_blockwise_value_pass(&mut accumulator, &context)
                        .unwrap();
                }
                WorkspaceBackend::accumulate_blockwise_attention(
                    &mut accumulator,
                    0,
                    2,
                    k.clone(),
                    v.clone(),
                    &context,
                )
                .unwrap();
            }
            let output =
                WorkspaceBackend::finish_blockwise_attention(accumulator, &context).unwrap();
            assert_eq!(
                output
                    .layout()
                    .representation()
                    .map(WorkspaceRepresentation::dtype),
                dtype
            );
            assert!(
                output
                    .layout()
                    .representation()
                    .is_none_or(|value| !value.row_contiguous())
            );
            let report = context.report(&[output.clone()]).unwrap();
            for operation in &report.operations {
                if let WorkspaceOperationKind::BlockwiseAttention { policy, stage } = operation.kind
                {
                    assert_eq!(policy.output_type, dtype);
                    let expected = if stage == WorkspaceBlockwiseStage::Finish {
                        dtype
                    } else {
                        Some(F::Float32)
                    };
                    for layout in &operation.outputs {
                        assert_eq!(
                            layout.representation().map(WorkspaceRepresentation::dtype),
                            expected
                        );
                    }
                }
            }
            // The next layer's sealed floating page can now describe its real
            // Host store. Unknown original query type remains unknown at Finish.
            if let Some(dtype) = dtype {
                let stored = context.store_host_value(&output, dtype).unwrap();
                assert_eq!(stored.floating_type(), dtype);
                assert_eq!(
                    stored
                        .load(&context)
                        .unwrap()
                        .layout()
                        .representation()
                        .unwrap()
                        .dtype(),
                    dtype
                );
            }
        }
    }
}

#[test]
fn metal_sampling_filters_retain_only_known_f32_source_representation(){
    let mechanism=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for shape in [&[64][..],&[1,64][..],&[1,1,64][..]]{
        for dtype in [Some(F::Float32),Some(F::Float16),Some(F::Bfloat16),None]{
            let context=WorkspaceContext::new(mechanism);
            let input=source(shape,dtype,&context);
            for kind in [WorkspaceSamplingOperation::TokenFilter,WorkspaceSamplingOperation::OptionalTokenFilter,
                WorkspaceSamplingOperation::TopK {keep:40},WorkspaceSamplingOperation::TopP,WorkspaceSamplingOperation::MinP]{
                let layout=context.layout(shape,WorkspaceDtype::Float32).unwrap();
                let result=context.execute(WorkspaceOperationKind::Sampling(kind),&[&input],vec![layout]).unwrap().remove(0);
                assert_eq!(result.layout().representation().map(|r|r.dtype()),dtype.filter(|d|*d==F::Float32));
            }
        }
    }
}

#[test]
fn causal_padding_preserves_actual_scalar_through_convolution_and_history() {
    let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for dtype in [Some(F::Float32), Some(F::Float16), Some(F::Bfloat16), None] {
        let context = WorkspaceContext::new(facts);
        let input = source(&[1, 2, 16], dtype, &context);
        let weight = source(&[16, 3, 1], Some(F::Float32), &context);
        let padded = WorkspaceTensor::pad(&input, &[(0,0),(2,0),(0,0)],
            eredu_nn::PadMode::Constant, &context).unwrap();
        let output = WorkspaceTensor::conv1d(&padded, &weight, 1, 0, 1, 16, &context).unwrap();
        assert_eq!(padded.layout().representation().map(|value|value.dtype()), dtype);
        assert_eq!(output.layout().representation().map(|value|value.dtype()),
            dtype.map(|value|promote(value,F::Float32)));
        let history = padded.index(&[eredu_nn::Index::Full,eredu_nn::Index::Range(2,4),
            eredu_nn::Index::Full], &context).unwrap();
        assert_eq!(history.layout().representation().map(|value|value.dtype()),dtype);
    }
}
