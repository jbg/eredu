use super::*;

#[test]
fn lane_proposal_width_cannot_exceed_selected_realization() {
    let config = SpeculativeConfig {
        max_draft_tokens: 5,
        ..SpeculativeConfig::default()
    };
    let error = validate_lane_proposal_capacity(&config, 4).unwrap_err();
    assert!(error.to_string().contains("admits at most 4"));
    validate_lane_proposal_capacity(&config, 5).unwrap();
}

#[test]
fn fused_rows_are_selected_by_backend_mechanisms_without_family_policy() {
    let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let value = MlxTensor::from_array(Array::from_slice(
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1, 2, 3],
    ));
    assert_eq!(
        MlxEmbeddedPredictionMechanisms::sequence_len(&value).unwrap(),
        2
    );
    let row = MlxEmbeddedPredictionMechanisms::fused_logits_row(
        &value,
        1,
        SpeculativeExecutionStreams::single(&stream),
    )
    .unwrap();
    assert_eq!(row.evaluated().unwrap().as_slice::<f32>(), &[4.0, 5.0, 6.0]);
}
