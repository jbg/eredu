use super::*;

#[test]
fn physical_linear_bound_covers_real_optional_companions_before_population() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let mut linear = common::linear::PhysicalLinear::unloaded(
        2,
        2,
        false,
        eredu_checkpoint::LinearFormat::Dense,
        &stream,
    )
    .unwrap();
    assert_eq!(linear.native_retained_value_slot_bound(), Some(5));
    let mut visited = 0;
    assert!(linear.visit_native_retained_values(&mut |_| visited += 1));
    assert_eq!(visited, 1);
    linear.weight.value = Array::from_slice(&[2_f32, 3., 5., 7.], &[2, 2]);
    linear.weight_scale_inv.value = Some(Array::from_slice(&[11_f32], &[1]));
    linear.scales.value = Some(Array::from_slice(&[13_f32], &[1]));
    linear.biases.value = Some(Array::from_slice(&[17_f32], &[1]));
    linear.bias.value = Some(Array::from_slice(&[19_f32, 23.], &[2]));
    let mut values = Vec::new();
    assert!(
        linear.visit_native_retained_values(&mut |v| values.push(v.as_array().shape().to_vec()))
    );
    assert_eq!(values, [vec![2, 2], vec![1], vec![1], vec![1], vec![2]]);
    assert_eq!(linear.native_retained_value_slot_bound(), Some(5));
}

#[test]
fn native_unknown_leaf_never_gains_a_bound_through_wrapper() {
    let module = MlxNamedModule::with_exact_topology(
        OpaqueNativeLeaf {
            weight: PhysicalParam::new(Array::from_slice(&[3_f32], &[1])),
            hidden: Array::from_slice(&[5_f32], &[1]),
        },
        [("weight", ParameterSpec::trainable("module.weight").unwrap())],
    )
    .unwrap();
    assert_eq!(module.retained_value_slot_bound(), None);
    assert_eq!(MlxModule::new(module).retained_value_slot_bound(), None);
}
