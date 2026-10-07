//! Native mixed-channel observation through the public backend mechanism.

use eredu_backend_mlx::{backend::runtime::realtime::MlxRealtimeChannelObserver, MlxTensor};
use eredu_core::{
    RealtimeChannelCadence, RealtimeChannelDeclaration, RealtimeChannelId, RealtimeChannelLayout,
    RealtimeChannelPayload, RealtimeSlotCoordinate,
};
use eredu_runtime::{
    validate_completed_realtime_host_frame, CompletedRealtimeFrame, RealtimeFrameChannel,
    RealtimeFrameHostObserver, RealtimeHostChannelValues,
};
use safemlx::{Array, Dtype, Stream};

#[test]
fn mixed_realtime_observation_preserves_values_and_rejects_reshaped_payloads() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let dense_id = RealtimeChannelId::new(7);
    let category_id = RealtimeChannelId::new(19);
    let layout = RealtimeChannelLayout::new(
        "mixed-output-observation".into(),
        [
            (
                dense_id,
                RealtimeChannelPayload::Dense { shape: vec![2, 2] },
            ),
            (
                category_id,
                RealtimeChannelPayload::Categorical {
                    width: 1,
                    cardinality: 16,
                },
            ),
        ]
        .into_iter()
        .map(|(id, payload)| RealtimeChannelDeclaration {
            id,
            payload,
            cadence: RealtimeChannelCadence {
                period: 1,
                phase: 0,
            },
            delay: 0,
            feedback: vec![],
        })
        .collect(),
    )
    .unwrap();
    let dense = Array::from_slice(&[0.25f32, -0.5, 2.0, 3.5], &[1, 2, 2])
        .as_dtype(Dtype::Float16, &stream)
        .unwrap();
    let category = Array::from_slice(&[13u32], &[1, 1]).copy(&stream).unwrap();
    let aligned = MlxTensor::from_array(Array::from_slice(&[-1.0f32, 0.0, 0.5, 4.0], &[1, 2, 2]));
    let logits = (0..16).map(|n| n as f32 * 0.25 - 2.0).collect::<Vec<_>>();
    let diagnostic = MlxTensor::from_array(
        Array::from_slice(&logits, &[1, 16])
            .as_dtype(Dtype::Float16, &stream)
            .unwrap(),
    );
    let coordinate = RealtimeSlotCoordinate::new(3, dense_id);
    let make_frame = |dense: Array, category: Array| {
        CompletedRealtimeFrame::<_, MlxTensor>::new(
            layout.clone(),
            1,
            vec![
                RealtimeFrameChannel::new(dense_id, None, MlxTensor::from_array(dense)),
                RealtimeFrameChannel::new(category_id, None, MlxTensor::from_array(category)),
            ],
            vec![RealtimeFrameChannel::new(
                dense_id,
                Some(coordinate),
                aligned.clone(),
            )],
            vec![category_id],
            vec![diagnostic.clone()],
        )
        .unwrap()
    };
    let frame = make_frame(dense.clone(), category.clone());
    let mut observer = MlxRealtimeChannelObserver::new(&stream);
    let host = observer.observe(&frame).unwrap();
    validate_completed_realtime_host_frame(&host).unwrap();
    assert_eq!(
        host.value(dense_id),
        Some(&RealtimeHostChannelValues::Dense(vec![
            0.25, -0.5, 2.0, 3.5
        ]))
    );
    assert_eq!(
        host.value(category_id),
        Some(&RealtimeHostChannelValues::Categorical(vec![13]))
    );

    assert_eq!(host.aligned()[0].coordinate(), Some(coordinate));
    assert_eq!(
        host.aligned()[0].payload(),
        &RealtimeHostChannelValues::Dense(vec![-1.0, 0.0, 0.5, 4.0])
    );
    assert_eq!(host.decision_channels(), &[category_id]);
    assert_eq!(host.diagnostics()[0].prediction(), 0);
    assert_eq!(host.diagnostics()[0].shape(), &[1, 16]);
    assert_eq!(host.diagnostics()[0].logits(), logits);

    // Equal scalar counts do not make a flattened dense tensor admissible.
    let wrong_shape = make_frame(dense.reshape(&[1, 4], &stream).unwrap(), category.clone());
    assert!(observer.observe(&wrong_shape).is_err());
    // Categorical identity must never pass through a floating conversion.
    let wrong_kind = make_frame(dense, Array::from_slice(&[13.0f32], &[1, 1]));
    assert!(observer.observe(&wrong_kind).is_err());
    let integer_dense = make_frame(Array::from_slice(&[1i32, 2, 3, 4], &[1, 2, 2]), category);
    assert!(observer.observe(&integer_dense).is_err());
}
