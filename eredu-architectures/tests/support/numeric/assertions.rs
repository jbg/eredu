fn assert_finite_values(values: &[f32], label: &str) {
    for (index, value) in values.iter().enumerate() {
        assert!(value.is_finite(), "{label}[{index}] is not finite: {value}");
    }
}

fn assert_tensor_close(actual: &NumericTensor, expected: &NumericTensor, label: &str) {
    assert_eq!(actual.shape, expected.shape, "{label} shape");
    assert_eq!(
        actual.data.len(),
        expected.data.len(),
        "{label} data length"
    );
    assert_finite_values(&actual.data, &format!("{label} actual"));
    assert_finite_values(&expected.data, &format!("{label} expected"));
    for (index, (actual, expected)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (*actual - *expected).abs() <= 2.0e-4,
            "{label}[{index}]: expected {expected}, got {actual}"
        );
    }
}

fn assert_tensor_exact(actual: &NumericTensor, expected: &NumericTensor, label: &str) {
    assert_eq!(actual.shape, expected.shape, "{label} shape");
    assert_eq!(actual.data, expected.data, "{label} values");
}
