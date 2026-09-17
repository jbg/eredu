use super::*;
use eredu_nn::{RotaryAlgorithm, RotaryArithmetic};

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

fn retained(module: &impl Parameterized<MlxTensor>) -> (bool, Vec<MlxTensor>) {
    assert_borrowed_operator_sources(module);
    let mut values = Vec::new();
    let complete = module.visit_retained_values(&mut |value| values.push(value.clone()));
    (complete, values)
}

fn topology(module: &impl Parameterized<MlxTensor>) -> Vec<eredu_nn::ParameterMetadata> {
    let mut parameters = eredu_nn::validate_parameter_topology(module).unwrap();
    parameters.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
    parameters
}

fn rotary(algorithm: RotaryAlgorithm, arithmetic: RotaryArithmetic, stream: &Stream) -> MlxRotary {
    MlxNeuralBackend::rotary(
        RotarySpec {
            dimensions: 8,
            base: 10_000.0,
            traditional: false,
            algorithm,
            arithmetic,
        },
        stream,
    )
    .unwrap()
}

#[test]
fn retained_rotary_values_cover_lazy_scaled_and_explicit_roots_without_parameters() {
    let stream = stream();
    let mut operator = rotary(
        RotaryAlgorithm::Llama3 {
            factor: 8.0,
            low_frequency_factor: 1.0,
            high_frequency_factor: 4.0,
            original_max_positions: 8192,
        },
        RotaryArithmetic::InputProducts,
        &stream,
    );
    let topology = eredu_nn::validate_parameter_topology(&operator).unwrap();
    assert!(topology.is_empty());
    let (complete, values) = retained(&operator);
    assert!(complete);
    assert_eq!(
        values.len(),
        2,
        "denominators and explicit inverse frequencies"
    );
    for value in &values {
        assert_eq!(value.as_array().allocation_info().unwrap(), None);
        assert_eq!(value.shape(), &[4]);
    }
    operator.set_trainable(true);
    assert_eq!(
        eredu_nn::validate_parameter_topology(&operator).unwrap(),
        topology
    );

    // The cold visit did not evaluate either graph. Settling the returned roots
    // now reveals two distinct allocations, whose values are reciprocal.
    let denominator = values[0].as_array().evaluated().unwrap();
    let inverse = values[1].as_array().evaluated().unwrap();
    for (&a, &b) in denominator
        .as_slice::<f32>()
        .iter()
        .zip(inverse.as_slice::<f32>())
    {
        assert!(a.is_finite() && a > 0.0 && b.is_finite() && b > 0.0);
        assert!((a * b - 1.0).abs() < 1e-6);
    }
    assert_ne!(
        values[0]
            .as_array()
            .allocation_info()
            .unwrap()
            .unwrap()
            .identity(),
        values[1]
            .as_array()
            .allocation_info()
            .unwrap()
            .unwrap()
            .identity(),
    );
    let (complete, aliases) = retained(&operator.clone());
    assert!(complete);
    for (value, alias) in values.iter().zip(aliases.iter()) {
        assert_eq!(
            value.as_array().allocation_info().unwrap(),
            alias.as_array().allocation_info().unwrap()
        );
    }
}

#[test]
fn retained_rotary_values_include_all_selected_variants_and_keep_default_empty() {
    let stream = stream();
    for algorithm in [
        RotaryAlgorithm::Default,
        RotaryAlgorithm::Linear { factor: 2.0 },
        RotaryAlgorithm::Proportional {
            factor: 2.0,
            rotary_fraction: 0.5,
        },
        RotaryAlgorithm::Yarn {
            factor: 4.0,
            original_max_positions: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            amplitude: 1.2,
            truncate: true,
        },
    ] {
        let native_count = usize::from(matches!(
            algorithm,
            RotaryAlgorithm::Proportional { .. } | RotaryAlgorithm::Yarn { .. }
        ));
        for arithmetic in [RotaryArithmetic::Native, RotaryArithmetic::InputProducts] {
            let operator = rotary(algorithm, arithmetic, &stream);
            let (complete, values) = retained(&operator);
            assert!(complete);
            assert_eq!(
                values.len(),
                native_count + usize::from(arithmetic == RotaryArithmetic::InputProducts)
            );
            assert!(eredu_nn::validate_parameter_topology(&operator)
                .unwrap()
                .is_empty());
            for value in values {
                assert_eq!(value.shape(), &[4]);
                let evaluated = value.as_array().evaluated().unwrap();
                let values = evaluated.as_slice::<f32>();
                assert!(values.iter().all(|v| v.is_finite() && *v >= 0.0));
                assert!(values.iter().any(|v| *v > 0.0));
            }
        }
    }
}

#[test]
fn retained_linear_values_preserve_shared_backing_and_editable_topology() {
    let stream = stream();
    let mut operator = MlxNeuralBackend::linear(
        LinearSpec {
            input: 2,
            output: 2,
            weight: ParameterSpec::trainable("projection.weight").unwrap(),
            bias: Some(ParameterSpec::trainable("projection.bias").unwrap()),
            format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        },
        &stream,
    )
    .unwrap();
    let base = Array::from_slice(&[1.25_f32, 2.5, 3.75, 5.0], &[2, 2]);
    let bias = base.as_strided(&[2][..], &[1][..], 0, &stream).unwrap();
    operator.module.weight.value = base.clone();
    operator.module.bias.value = Some(bias.clone());
    let before = topology(&operator);
    let (complete, values) = retained(&operator);
    assert!(complete);
    assert_eq!(values.len(), 2);
    assert_eq!(
        bias.allocation_info().unwrap(),
        None,
        "traversal stays cold"
    );
    assert_eq!(topology(&operator), before);
    let mut backing = BTreeMap::new();
    for value in values {
        value.as_array().evaluated().unwrap();
        let info = value.as_array().allocation_info().unwrap().unwrap();
        backing.insert(info.identity(), info.bytes());
    }
    let base_info = base.allocation_info().unwrap().unwrap();
    assert_eq!(
        backing,
        BTreeMap::from([(base_info.identity(), base_info.bytes())])
    );
    assert_eq!(bias.evaluated().unwrap().as_slice::<f32>(), &[1.25, 2.5]);
}

#[test]
fn retained_quantized_embedding_includes_packed_owner_without_changing_parameters() {
    let stream = stream();
    let module = nn::QuantizedEmbedding::unloaded_with_mode(
        2,
        32,
        32,
        4,
        safemlx::ops::QuantizationMode::Affine,
        &stream,
    )
    .unwrap();
    let format = LinearFormatSpec::affine(
        LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap()),
        ParameterSpec::trainable("embedding.scales").unwrap(),
        ParameterSpec::trainable("embedding.biases").unwrap(),
    )
    .unwrap();
    let module = common::linear::PhysicalEmbedding::Quantized(module);
    let parameters = parameter_topology(
        &module,
        ParameterSpec::trainable("embedding.weight").unwrap(),
        None,
        &format,
    )
    .unwrap();
    let mut operator = MlxEmbedding {
        module,
        topology: parameters,
        vocabulary: 2,
        vocabulary_range: None,
    };
    let before = topology(&operator);
    let (complete, visible) = retained(&operator);
    assert!(complete);
    assert_eq!(visible.len(), 3);

    let native = common::native_quantization::NativeQuantizedTensor::from_iq_array(
        Array::from_slice(&[3_u8; 68], &[2, 34]),
        &[2, 32],
        eredu_gguf::GgmlType::Q8_0,
        eredu_gguf::Endian::Little,
    )
    .unwrap();
    let common::linear::PhysicalEmbedding::Quantized(module) = &mut operator.module else {
        unreachable!()
    };
    native.retained_packed_array().evaluated().unwrap();
    let packed = native
        .retained_packed_array()
        .allocation_info()
        .unwrap()
        .unwrap();
    module.native = Some(native);
    let (complete, still_visible) = retained(&operator);
    assert!(complete);
    assert_eq!(still_visible.len(), visible.len() + 1);
    assert_eq!(
        still_visible
            .iter()
            .filter(|value| value.as_array().allocation_info().unwrap() == Some(packed))
            .count(),
        1
    );
    assert_eq!(operator.retained_value_slot_bound(), Some(4));
    assert_eq!(topology(&operator), before);
    let common::linear::PhysicalEmbedding::Quantized(module) = &mut operator.module else {
        unreachable!()
    };
    module.native = None;
    assert!(retained(&operator).0);
}

fn assert_borrowed_operator_sources(module: &impl Parameterized<MlxTensor>) {
    struct Rows<'a> {
        named: Vec<eredu_nn::ParameterMetadata>,
        values: Vec<&'a MlxTensor>,
    }
    impl<'a> eredu_nn::ParameterSourceVisitor<'a, MlxTensor> for Rows<'a> {
        fn parameter(
            &mut self,
            metadata: eredu_nn::ParameterMetadataView<'a>,
            value: &'a MlxTensor,
        ) {
            self.named.push(metadata.to_owned());
            self.values.push(value);
        }
        fn retained(&mut self, value: &'a MlxTensor) {
            self.values.push(value);
        }
    }
    let mut rows = Rows {
        named: Vec::new(),
        values: Vec::new(),
    };
    module.visit_parameter_sources(&mut rows).unwrap();
    let mut expected = Vec::new();
    assert!(module.visit_retained_values(&mut |value| expected.push(value as *const MlxTensor)));
    assert_eq!(
        rows.values
            .iter()
            .map(|value| *value as *const MlxTensor)
            .collect::<Vec<_>>(),
        expected
    );
    rows.named.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
    assert_eq!(rows.named, topology(module));
}
