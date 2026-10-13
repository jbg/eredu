fn parameter(name: &str) -> ParameterSpec {
    ParameterSpec::trainable(name).unwrap()
}

fn test_format(weight: &str, format: LinearFormat) -> eredu_nn::LinearFormatSpec {
    let prefix = weight.strip_suffix(".weight").unwrap_or(weight);
    match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => {
            eredu_nn::LinearFormatSpec::unscaled(format).unwrap()
        }
        LinearFormat::E4M3BlockFp8(_) => eredu_nn::LinearFormatSpec::scaled(
            format,
            parameter(&format!("{prefix}.weight_scale_inv")),
        )
        .unwrap(),
        LinearFormat::MxFp4 => {
            eredu_nn::LinearFormatSpec::scaled(format, parameter(&format!("{prefix}.scales")))
                .unwrap()
        }
        LinearFormat::Affine(_) => eredu_nn::LinearFormatSpec::affine(
            format,
            parameter(&format!("{prefix}.scales")),
            parameter(&format!("{prefix}.biases")),
        )
        .unwrap(),
    }
}

fn affine_group_projection(weight: &str, scales: &str, biases: &str) -> GroupedProjectionSpec {
    GroupedProjectionSpec::new(
        parameter(weight),
        None,
        eredu_nn::LinearFormatSpec::affine(
            LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap()),
            parameter(scales),
            parameter(biases),
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_quantized_topologies_use_literal_neutral_identities() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = execution.stream();
    let affine = LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap());
    let linear = <MlxNeuralBackend as NeuralBackend>::linear(
        LinearSpec {
            input: 32,
            output: 32,
            weight: parameter("arbitrary.linear.matrix"),
            bias: None,
            format: eredu_nn::LinearFormatSpec::affine(
                affine,
                parameter("arbitrary.linear.scale"),
                parameter("arbitrary.linear.affine"),
            )
            .unwrap(),
        },
        stream,
    )
    .unwrap();
    let linear_ids = eredu_nn::validate_parameter_topology::<MlxTensor, _>(&linear)
        .unwrap()
        .into_iter()
        .map(|parameter| parameter.id.as_str().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        linear_ids,
        [
            "arbitrary.linear.affine",
            "arbitrary.linear.matrix",
            "arbitrary.linear.scale",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    );

    let embedding = <MlxNeuralBackend as NeuralBackend>::embedding(
        EmbeddingSpec {
            vocabulary: 32,
            dimensions: 32,
            weight: parameter("arbitrary.embedding.table"),
            format: eredu_nn::LinearFormatSpec::affine(
                affine,
                parameter("arbitrary.embedding.scale"),
                parameter("arbitrary.embedding.affine"),
            )
            .unwrap(),
        },
        stream,
    )
    .unwrap();
    let embedding_ids = eredu_nn::validate_parameter_topology::<MlxTensor, _>(&embedding)
        .unwrap()
        .into_iter()
        .map(|parameter| parameter.id.as_str().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        embedding_ids,
        [
            "arbitrary.embedding.affine",
            "arbitrary.embedding.scale",
            "arbitrary.embedding.table",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    );

    let gate = affine_group_projection(
        "arbitrary.gated.matrix_a",
        "arbitrary.gated.scale_a",
        "arbitrary.gated.affine_a",
    );
    let down = affine_group_projection(
        "arbitrary.gated.matrix_b",
        "arbitrary.gated.scale_b",
        "arbitrary.gated.affine_b",
    );
    let gated = <MlxNeuralBackend as GroupedNeuralBackend>::grouped_gated_product(
        GroupedGatedProductSpec::new(
            2,
            32,
            32,
            32,
            eredu_nn::GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: gate,
                down,
            },
        )
        .unwrap(),
        stream,
    )
    .unwrap();
    let gated_ids = eredu_nn::validate_parameter_topology::<MlxTensor, _>(&gated)
        .unwrap()
        .into_iter()
        .map(|parameter| parameter.id.as_str().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        gated_ids,
        [
            "arbitrary.gated.affine_a",
            "arbitrary.gated.affine_b",
            "arbitrary.gated.matrix_a",
            "arbitrary.gated.matrix_b",
            "arbitrary.gated.scale_a",
            "arbitrary.gated.scale_b",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    );

    let relu = <MlxNeuralBackend as GroupedNeuralBackend>::grouped_relu2(
        GroupedRelu2Spec::new(
            2,
            32,
            32,
            affine_group_projection(
                "arbitrary.relu.matrix_a",
                "arbitrary.relu.scale_a",
                "arbitrary.relu.affine_a",
            ),
            affine_group_projection(
                "arbitrary.relu.matrix_b",
                "arbitrary.relu.scale_b",
                "arbitrary.relu.affine_b",
            ),
        )
        .unwrap(),
        stream,
    )
    .unwrap();
    let relu_ids = eredu_nn::validate_parameter_topology::<MlxTensor, _>(&relu)
        .unwrap()
        .into_iter()
        .map(|parameter| parameter.id.as_str().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        relu_ids,
        [
            "arbitrary.relu.affine_a",
            "arbitrary.relu.affine_b",
            "arbitrary.relu.matrix_a",
            "arbitrary.relu.matrix_b",
            "arbitrary.relu.scale_a",
            "arbitrary.relu.scale_b",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    );
}

fn bind_linear(
    linear: &mut MlxLinear,
    weight: Array,
    format: LinearFormat,
    stream: &safemlx::Stream,
) {
    match format.weight_quantization() {
        None => linear.module.weight.value = weight,
        Some(quantization) => {
            let mode = match quantization {
                WeightQuantization::Affine(_) => QuantizationMode::Affine,
                WeightQuantization::MxFp4 => QuantizationMode::MxFp4,
                WeightQuantization::GgufIQuant { .. } => {
                    panic!("test helper does not synthesize GGUF blocks")
                }
            };
            let arrays = quantize_with_mode(
                &weight,
                quantization.group_size(),
                quantization.bits(),
                mode,
                stream,
            )
            .unwrap();
            linear.module.weight.value = arrays.weight;
            linear.module.scales.value = Some(arrays.scales);
            linear.module.biases.value = arrays.biases;
        }
    }
}

fn linear(
    name: &str,
    input: i32,
    output: i32,
    format: LinearFormat,
    weight: &[f32],
    stream: &safemlx::Stream,
) -> MlxLinear {
    let mut linear = <MlxNeuralBackend as NeuralBackend>::linear(
        LinearSpec {
            input,
            output,
            weight: parameter(name),
            bias: None,
            format: test_format(name, format),
        },
        stream,
    )
    .unwrap();
    bind_linear(
        &mut linear,
        Array::from_slice(weight, &[output, input]),
        format,
        stream,
    );
    linear
}

fn bind_embedding(embedding: &mut MlxEmbedding, weight: Array, stream: &safemlx::Stream) {
    match &mut embedding.module {
        PhysicalEmbedding::Dense(embedding) => embedding.weight.value = weight,
        PhysicalEmbedding::Quantized(embedding) => {
            let arrays = quantize_with_mode(
                &weight,
                embedding.group_size,
                embedding.bits,
                embedding.mode,
                stream,
            )
            .unwrap();
            embedding.inner.weight.value = arrays.weight;
            embedding.scales.value = Some(arrays.scales);
            embedding.biases.value = arrays.biases;
        }
    }
}

fn embedding(
    name: &str,
    vocabulary: i32,
    dimensions: i32,
    quantization: Option<WeightQuantization>,
    weight: &[f32],
    stream: &safemlx::Stream,
) -> MlxEmbedding {
    let mut embedding = <MlxNeuralBackend as NeuralBackend>::embedding(
        EmbeddingSpec {
            vocabulary,
            dimensions,
            weight: parameter(name),
            format: test_format(name, quantization.into()),
        },
        stream,
    )
    .unwrap();
    bind_embedding(
        &mut embedding,
        Array::from_slice(weight, &[vocabulary, dimensions]),
        stream,
    );
    embedding
}

fn supported_embedding_formats() -> [Option<WeightQuantization>; 3] {
    [
        None,
        Some(WeightQuantization::Affine(
            AffineQuantization::new(32, 4).unwrap(),
        )),
        Some(WeightQuantization::MxFp4),
    ]
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_explicit_projection_precision_rounds_input_and_restores_f32_output() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = execution.stream();
    let native = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let group = crate::backend::runtime::distributed::Group::uncontracted(&native);
    for (element, dtype, expected) in [
        (
            eredu_nn::TensorElementType::F16,
            Dtype::Float16,
            vec![1.0, -2.003_906_3],
        ),
        (
            eredu_nn::TensorElementType::Bf16,
            Dtype::Bfloat16,
            vec![1.0, -2.0],
        ),
    ] {
        let mut projection = <MlxNeuralBackend as NeuralBackend>::linear(
            LinearSpec {
                input: 2,
                output: 2,
                weight: parameter("precision.weight"),
                bias: None,
                format: test_format("precision.weight", LinearFormat::Dense)
                    .with_compute_element(Some(element))
                    .unwrap(),
            },
            stream,
        )
        .unwrap();
        bind_linear(
            &mut projection,
            Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2])
                .as_dtype(dtype, stream)
                .unwrap(),
            LinearFormat::Dense,
            stream,
        );
        let input = MlxTensor::from_array(Array::from_slice(&[1.0003f32, -2.003], &[1, 2]));
        let output = projection.forward(&input, stream).unwrap();
        assert_eq!(
            output.element_type(),
            Some(eredu_nn::TensorElementType::F32)
        );
        assert_eq!(output.to_f32_vec(stream).unwrap(), expected);
        let observed = projection
            .forward_with_input_observer(&input, stream, None)
            .unwrap();
        assert_eq!(observed.to_f32_vec(stream).unwrap(), expected);
        let parallel =
            MlxNeuralBackend::row_parallel_linear(&mut projection, &input, &group, stream).unwrap();
        assert_eq!(
            parallel.element_type(),
            Some(eredu_nn::TensorElementType::F32)
        );
        assert_eq!(parallel.to_f32_vec(stream).unwrap(), expected);
        assert_eq!(input.to_f32_vec(stream).unwrap(), vec![1.0003, -2.003]);
    }
}
