use super::*;
use crate::ops::{
    all_close,
    indexing::{take_axis, IndexOp, TryIndexOp},
    matmul, reshape,
};

#[test]
fn reduced_precision_cpu_groups_preserve_unsorted_rows_and_reject_invalid_ids() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for dtype in [Dtype::Bfloat16, Dtype::Float16] {
        let inputs = Array::from_slice(&[1_f32, 2., 3., 4., 5., 6., 7., 8.], &[4, 2])
            .as_dtype(dtype, &stream)
            .unwrap();
        let weights = Array::from_slice(
            &[
                1_f32, 0., 2., 0., 1., 3., 2., 1., 0., 1., 2., 1., 3., 2., 1., 2., 3., 2.,
            ],
            &[3, 2, 3],
        )
        .as_dtype(dtype, &stream)
        .unwrap();
        let ids = Array::from_slice(&[2_i32, 0, 2, 1], &[4]);
        let result = grouped_matmul(&inputs, &weights, &ids, false, &stream).unwrap();
        assert_eq!(result.dtype(), dtype);
        assert_eq!(
            result
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[7., 8., 5., 3., 4., 18., 27., 28., 17., 22., 23., 8.]
        );
        let invalid = Array::from_slice(&[0_u64, 1, 2, 1_u64 << 32], &[4]);
        assert!(grouped_matmul(&inputs, &weights, invalid, false, &stream).is_err());
        let empty_inputs = inputs.try_index_device((0..0, ..), &stream).unwrap();
        let empty_ids = ids.try_index_device(0..0, &stream).unwrap();
        assert_eq!(
            grouped_matmul(empty_inputs, &weights, empty_ids, true, &stream)
                .unwrap()
                .shape(),
            &[0, 3]
        );
    }
}

#[test]
fn test_group_by_id_topk_plan() {
    let stream = crate::test_stream();
    let groups = Array::from_slice(&[2i32, 0, 1, 2, 0, 1], &[3, 2]);
    let plan = topk_group_plan(&groups, stream).unwrap();

    assert_eq!(
        crate::array::eval_vec::<i32>(&plan.sorted_group_ids),
        &[0, 0, 1, 1, 2, 2]
    );
    assert_eq!(
        crate::array::eval_vec::<i32>(&plan.selection_indices),
        &[1, 4, 2, 5, 0, 3]
    );
    assert_eq!(
        crate::array::eval_vec::<i32>(&plan.token_indices),
        &[0, 2, 1, 2, 0, 1]
    );
}

#[test]
fn test_grouped_matmul_matches_gathered_reference() {
    let stream = crate::test_stream();
    let inputs = reshape(
        Array::arange::<_, f32>(0.0, 12.0, None, stream).unwrap(),
        &[4, 3],
        stream,
    )
    .unwrap();
    let weights = reshape(
        Array::arange::<_, f32>(0.0, 18.0, None, stream).unwrap(),
        &[3, 3, 2],
        stream,
    )
    .unwrap();
    let group_ids = Array::from_slice(&[2i32, 0, 1, 2], &[4]);
    let plan = group_by_id(&group_ids, stream).unwrap();
    let sorted_inputs = take_axis(&inputs, &plan.token_indices, 0, stream).unwrap();
    let grouped = grouped_matmul(
        &sorted_inputs,
        &weights,
        &plan.sorted_group_ids,
        true,
        stream,
    )
    .unwrap();
    let selected_weights = take_axis(&weights, &plan.sorted_group_ids, 0, stream).unwrap();
    let expected = matmul(
        sorted_inputs.index_device((.., crate::ops::indexing::NewAxis, ..), stream),
        selected_weights,
        stream,
    )
    .unwrap()
    .reshape(&[4, 2], stream)
    .unwrap();

    assert!(all_close(&grouped, &expected, 1e-5, 1e-5, None, stream)
        .unwrap()
        .item::<bool>(&stream));
}
