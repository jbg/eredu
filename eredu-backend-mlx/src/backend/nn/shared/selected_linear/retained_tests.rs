use super::*;

#[test]
fn retained_selected_linear_values_are_complete_with_metadata_only_specification() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let projection = eredu_nn::GroupedProjectionSpec::new(
        ParameterSpec::trainable("bank.weight").unwrap(),
        None,
        LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
    )
    .unwrap();
    let mut bank = MlxGroupedLinear::new(
        GroupedLinearSpec::new(2, 2, 2, GroupedLinearActivation::Identity, projection).unwrap(),
        &stream,
    )
    .unwrap();
    let value = Array::from_slice(&[1_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])
        .square(&stream)
        .unwrap();
    bank.weight.replace(MlxTensor::from_array(value.clone()));
    let before = eredu_nn::validate_parameter_topology(&bank).unwrap();
    let mut values = Vec::new();
    assert!(bank.visit_retained_values(&mut |value| values.push(value.clone())));
    assert_eq!(values.len(), 1);
    assert_eq!(value.allocation_info().unwrap(), None);
    assert_eq!(values[0].as_array().allocation_info().unwrap(), None);
    assert_eq!(
        eredu_nn::validate_parameter_topology(&bank).unwrap(),
        before
    );
    assert_eq!(
        values[0].as_array().evaluated().unwrap().as_slice::<f32>(),
        &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0],
    );
    assert_eq!(
        value.allocation_info().unwrap(),
        values[0].as_array().allocation_info().unwrap()
    );
}
