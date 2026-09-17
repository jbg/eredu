//! Actual pooled-cache integer mask under a report-derived original operation.
use super::*;
use crate::backend::runtime::cache::kv::PoolingCache;
use eredu_nn::{operation_geometry::PoolingMaskGeometry, workspace::WorkspaceOperationKind};

#[test]
fn original_pooling_mask_preserves_chunk_visibility_and_large_integer_coordinates() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    let one = MlxTensor::from_array(Array::from_slice(&[1.0f32], &[]));
    let zero = MlxTensor::from_array(Array::from_slice(&[0.0f32], &[]));
    for (queries, pooled, offset, ratio) in [
        (5, 2, 3, 4),
        (4, 3, 9, 6),
        (3, 2, (1 << 24) - 1, (1 << 24) + 1),
    ] {
        let mut cache = PoolingCache::new(ratio).unwrap();
        let values: Vec<f32> = (0..pooled).map(|i| (i + 1) as f32 / 7.0).collect();
        let source = cache
            .update_and_fetch(Array::from_slice(&values, &[1, pooled, 1]), &stream)
            .unwrap();
        source.evaluated().unwrap();
        let geometry =
            PoolingMaskGeometry::new_fixed(queries, cache.pooled_tokens(), offset, cache.ratio())
                .unwrap();
        let expected: Vec<f32> = (0..queries)
            .flat_map(|query| {
                (0..pooled).map(move |column| {
                    if (i64::from(column) + 1) * i64::from(ratio)
                        <= i64::from(offset) + i64::from(query) + 1
                    {
                        1.0
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        assert!(expected.contains(&0.0) && expected.contains(&1.0));
        let invoke = || {
            let mask =
                MlxTensor::from_array(cache.make_mask(queries, offset, &stream).unwrap().unwrap());
            MlxTensor::where_condition(&mask, &one, &zero, &stream)
                .unwrap()
                .into_array()
        };
        let ordinary = invoke().evaluated().unwrap().try_to_vec::<f32>().unwrap();
        assert_eq!(ordinary, expected);
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let yes = WorkspaceTensor::existing(
            context.layout(&[], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let no = WorkspaceTensor::existing(
            context.layout(&[], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        context.begin_span();
        // Same atomic descriptor emitted by WorkspacePoolingLayerState from
        // this exact retained pooled count; the physical and recipe emitters
        // independently validate its Bool output and absence of native inputs.
        let mask = context
            .execute(
                WorkspaceOperationKind::PoolingMask(geometry),
                &[],
                vec![
                    context
                        .layout(&[queries, pooled], WorkspaceDtype::Bool)
                        .unwrap(),
                ],
            )
            .unwrap()
            .remove(0);
        let output = WorkspaceTensor::where_condition(&mask, &yes, &no, &context).unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        exercise_component(
            &mut prepared,
            &plan,
            &[one.as_array(), zero.as_array()],
            &expected,
            invoke,
        );
        assert!(cache.make_mask(1, offset, &stream).unwrap().is_none());
    }
    drop((one, zero, stream));
    prepared.finish();
}
