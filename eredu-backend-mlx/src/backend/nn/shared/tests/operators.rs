fn close(actual: &MlxTensor, expected: &[f32], tolerance: f32) {
    let actual = actual.as_array().evaluated().unwrap();
    assert_eq!(actual.as_slice::<f32>().len(), expected.len());
    assert!(actual
        .as_slice::<f32>()
        .iter()
        .zip(expected)
        .all(|(left, right)| (left - right).abs() <= tolerance));
}

#[test]
fn mlx_portable_normalization_geometry_preserves_additive_l2() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    for (values, epsilon) in [
        ([1.0_f32, -1.0], 4.0),
        ([0.0, 0.0], 0.25),
        ([0.01, -0.02], 0.1),
    ] {
        let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 2]));
        let denominator = (values[0] * values[0] + values[1] * values[1] + epsilon).sqrt();
        close(
            &MlxNeuralBackend::l2_normalize(&input, epsilon, stream).unwrap(),
            &values.map(|value| value / denominator),
            1e-6,
        );
        for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(MlxNeuralBackend::l2_normalize(&input, epsilon, stream).is_err());
            assert!(MlxNeuralBackend::rms_norm_without_weight(&input, epsilon, stream).is_err());
        }
    }
    let input = MlxTensor::from_array(Array::from_slice(&[1.0_f32; 4], &[1, 4]));
    let gate = MlxTensor::from_array(Array::from_slice(&[0.5_f32; 4], &[1, 4]));
    let weight = MlxTensor::from_array(Array::from_slice(&[1.0_f32; 4], &[4]));
    for (groups, epsilon) in [(0, 1e-5), (-1, 1e-5), (3, 1e-5), (2, f32::NAN)] {
        assert!(MlxNeuralBackend::gated_group_rms_norm(
            &input, &gate, &weight, groups, epsilon, stream
        )
        .is_err());
        assert!(MlxNeuralBackend::silu_gated_group_rms_norm(
            &input, &gate, &weight, groups, epsilon, stream
        )
        .is_err());
    }
}

#[test]
fn mlx_portable_causal_geometry_preserves_inclusive_distance() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    for distance in [None, Some(0), Some(1), Some(2), Some(i32::MAX)] {
        let mask = MlxNeuralBackend::causal_mask(3, 2, distance, stream).unwrap();
        assert_eq!(mask.shape(), [3, 5]);
        let mask = mask.as_array().evaluated().unwrap();
        let values = mask.as_slice::<bool>();
        for query in 0..3 {
            for key in 0..5 {
                let position = query + 2;
                let expected =
                    key <= position && distance.is_none_or(|distance| key >= position - distance);
                assert_eq!(values[(query * 5 + key) as usize], expected);
            }
        }
    }
    for (sequence, offset, distance) in [
        (-1, 0, None),
        (1, -1, None),
        (1, 0, Some(-1)),
        (1, i32::MAX, None),
    ] {
        assert!(MlxNeuralBackend::causal_mask(sequence, offset, distance, stream).is_err());
    }
}

#[test]
fn mlx_attention_preserves_output_dtype_and_mask_semantics() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    for (query_dtype, key_dtype, value_dtype, output_dtype) in [
        (
            Dtype::Bfloat16,
            Dtype::Bfloat16,
            Dtype::Bfloat16,
            Dtype::Bfloat16,
        ),
        (
            Dtype::Float16,
            Dtype::Float16,
            Dtype::Float16,
            Dtype::Float16,
        ),
        (
            Dtype::Float32,
            Dtype::Float32,
            Dtype::Float32,
            Dtype::Float32,
        ),
        (
            Dtype::Bfloat16,
            Dtype::Float32,
            Dtype::Bfloat16,
            Dtype::Float32,
        ),
        (
            Dtype::Bfloat16,
            Dtype::Bfloat16,
            Dtype::Float32,
            Dtype::Float32,
        ),
    ] {
        let tensor = |data: &[f32], shape: &[i32], dtype| {
            MlxTensor::from_array(
                Array::from_slice(data, shape)
                    .as_dtype(dtype, stream)
                    .unwrap(),
            )
        };
        let queries = tensor(&[0.0; 4], &[1, 2, 2, 1], query_dtype);
        let keys = tensor(&[0.0; 4], &[1, 1, 4, 1], key_dtype);
        let values = tensor(&[4.0, 8.0, 16.0, 32.0], &[1, 1, 4, 1], value_dtype);
        // Two committed keys plus a bidirectional proposal block, with the
        // oldest committed key falling outside the second query's window.
        let additive = tensor(
            &[0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0],
            &[1, 1, 2, 4],
            Dtype::Float32,
        );
        let boolean = MlxTensor::from_array(Array::from_slice(
            &[true, true, true, true, false, true, true, true],
            &[1, 1, 2, 4],
        ));
        let bias = tensor(&[0.0, 0.0, 0.0, 2.0_f32.ln()], &[4], Dtype::Float32);
        for (mask, expected) in [
            (Some(&additive), [15.0, 56.0 / 3.0, 15.0, 56.0 / 3.0]),
            (Some(&boolean), [15.0, 56.0 / 3.0, 15.0, 56.0 / 3.0]),
            (Some(&bias), [92.0 / 5.0; 4]),
            (None, [15.0; 4]),
        ] {
            let output = MlxNeuralBackend::attention(
                queries.clone(),
                keys.clone(),
                values.clone(),
                1.0,
                mask,
                stream,
            )
            .unwrap();
            assert_eq!(output.as_array().dtype(), output_dtype);
            assert_eq!(output.shape(), [1, 2, 2, 1]);
            let output = output.as_array().as_dtype(Dtype::Float32, stream).unwrap();
            let tolerance = match output_dtype {
                Dtype::Bfloat16 => 0.15,
                Dtype::Float16 => 0.02,
                _ => 1e-5,
            };
            close(&MlxTensor::from_array(output), &expected, tolerance);
        }
    }
}

#[test]
#[ignore = "explicit MLX dtype regression; run outside the sandbox"]
fn mlx_weighted_rms_norm_preserves_bfloat16_input_dtype() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = execution.stream();
    let input = MlxTensor::from_array(
        Array::from_slice(&[1.0_f32, -2.0, 3.0, -4.0], &[1, 4])
            .as_dtype(Dtype::Bfloat16, stream)
            .unwrap(),
    );
    let weight = MlxTensor::from_array(Array::from_slice(&[1.0_f32; 4], &[4]));
    let output =
        <MlxNeuralBackend as NeuralBackend>::rms_norm_with_weight(&input, &weight, 1e-5, stream)
            .unwrap();
    assert_eq!(output.as_array().dtype(), Dtype::Bfloat16);
    output.as_array().evaluated().unwrap();
}

#[test]
fn hot_path_api_construction_keeps_values_backend_native() {
    fn lookup(
        embedding: &mut MlxEmbedding,
        tokens: &MlxTensor,
        stream: &safemlx::Stream,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        embedding.lookup(tokens, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
    }
    fn project(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        stream: &safemlx::Stream,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        linear.forward(input, stream)
    }
    fn sum(
        embeddings: &mut MultiTableEmbedding<MlxNeuralBackend>,
        tokens: &[&MlxTensor],
        stream: &safemlx::Stream,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        embeddings.forward(tokens, stream)
    }

    let _: fn(
        &mut MlxEmbedding,
        &MlxTensor,
        &safemlx::Stream,
    ) -> Result<MlxTensor, eredu_nn::Error> = lookup;
    let _: fn(&mut MlxLinear, &MlxTensor, &safemlx::Stream) -> Result<MlxTensor, eredu_nn::Error> =
        project;
    let _: fn(
        &mut MultiTableEmbedding<MlxNeuralBackend>,
        &[&MlxTensor],
        &safemlx::Stream,
    ) -> Result<MlxTensor, eredu_nn::Error> = sum;
}

fn assert_fused_split_equivalence(format: LinearFormat) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = execution.stream();
    let input_width = 32;
    let segment_width = 2;
    let output_width = 6;
    let input = (0..input_width)
        .map(|index| (index as f32 - 15.0) / 16.0)
        .collect::<Vec<_>>();
    let weight = (0..output_width * input_width)
        .map(|index| ((index % 19) as f32 - 9.0) / 32.0)
        .collect::<Vec<_>>();
    let input_array = MlxTensor::from_array(Array::from_slice(&input, &[1, input_width]));
    let mut fused = linear(
        "fused.weight",
        input_width,
        output_width,
        format,
        &weight,
        stream,
    );
    let fused_output = fused.forward(&input_array, stream).unwrap();

    let mut split_outputs = Vec::new();
    for segment in 0..3 {
        let start = segment * segment_width * input_width;
        let end = start + segment_width * input_width;
        let mut split = linear(
            &format!("split.{segment}.weight"),
            input_width,
            segment_width,
            format,
            &weight[start as usize..end as usize],
            stream,
        );
        split_outputs.push(split.forward(&input_array, stream).unwrap());
    }
    let split_output = safemlx::ops::concatenate_axis(
        &split_outputs
            .iter()
            .map(MlxTensor::as_array)
            .collect::<Vec<_>>(),
        -1,
        stream,
    )
    .unwrap();
    let fused_evaluated = fused_output.as_array().evaluated().unwrap();
    let split_evaluated = split_output.evaluated().unwrap();
    assert_eq!(
        fused_evaluated.as_slice::<f32>(),
        split_evaluated.as_slice::<f32>()
    );

    let qkv = FusedProjectionLayout::new([
        FusedProjectionSegment::new("query", 2).unwrap(),
        FusedProjectionSegment::new("key", 2).unwrap(),
        FusedProjectionSegment::new("value", 2).unwrap(),
    ])
    .unwrap();
    assert_eq!(qkv.split(&fused_output, stream).unwrap().len(), 3);
    let gate_up = FusedProjectionLayout::new([
        FusedProjectionSegment::new("gate", 3).unwrap(),
        FusedProjectionSegment::new("up", 3).unwrap(),
    ])
    .unwrap();
    assert_eq!(gate_up.split(&fused_output, stream).unwrap().len(), 2);

    if format == LinearFormat::Dense {
        let expected = weight
            .chunks_exact(input_width as usize)
            .map(|row| {
                row.iter()
                    .zip(&input)
                    .map(|(left, right)| left * right)
                    .sum()
            })
            .collect::<Vec<f32>>();
        close(&fused_output, &expected, 1e-6);
    }
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_dense_fused_projection_equivalence() {
    assert_fused_split_equivalence(LinearFormat::Dense);
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_affine_fused_projection_equivalence() {
    assert_fused_split_equivalence(LinearFormat::Affine(
        AffineQuantization::new(32, 4).unwrap(),
    ));
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_mxfp4_fused_projection_equivalence() {
    assert_fused_split_equivalence(LinearFormat::MxFp4);
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_sentinel_embedding_validation() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = execution.stream();
    let dimensions = 32;
    let vocabulary = 4;
    let weight = (0..vocabulary * dimensions)
        .map(|index| (index as f32 + 1.0) / 64.0)
        .collect::<Vec<_>>();
    for quantization in supported_embedding_formats() {
        let mut embedding = embedding(
            "embedding.weight",
            vocabulary,
            dimensions,
            quantization,
            &weight,
            stream,
        );
        let valid = MlxTensor::from_array(Array::from_slice(&[-1_i32, 0, 3], &[3]));
        let scope = TokenValidationScope::begin().unwrap();
        let output = embedding
            .lookup(&valid, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
            .unwrap();
        let output = output.as_array().as_type::<f32>(stream).unwrap();
        let validations = scope.finish();
        let event =
            async_eval_with_event(std::iter::once(&output).chain(validations.arrays())).unwrap();
        event.synchronize().unwrap();
        validations.validate_completed().unwrap();
        let output = output.evaluated().unwrap();
        assert!(output.as_slice::<f32>()[..dimensions as usize]
            .iter()
            .all(|value| value.to_bits() == 0));

        for invalid in [-2_i32, vocabulary] {
            let tokens = MlxTensor::from_array(Array::from_slice(&[invalid], &[1]));
            let scope = TokenValidationScope::begin().unwrap();
            let output = embedding
                .lookup(&tokens, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
                .expect("lazy lookup must not synchronize while building the graph");
            let validations = scope.finish();
            let event = async_eval_with_event(
                std::iter::once(output.as_array()).chain(validations.arrays()),
            )
            .unwrap();
            event.synchronize().unwrap();
            assert!(
                validations.validate_completed().is_err(),
                "embedding accepted invalid token {invalid} under {quantization:?}"
            );
        }
    }
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_multi_table_embedding_sum_is_ordered_and_sentinel_safe() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = execution.stream();
    let dimensions = 32;
    for quantization in supported_embedding_formats() {
        let specs = (0..3)
            .map(|table| NamedEmbeddingSpec {
                name: format!("stream-{table}"),
                embedding: EmbeddingSpec {
                    vocabulary: 4,
                    dimensions,
                    weight: parameter(&format!("tables.{table}.weight")),
                    format: test_format(&format!("tables.{table}.weight"), quantization.into()),
                },
                lookup: EmbeddingLookupPolicy::ZeroSentinel(-1),
            })
            .collect::<Vec<_>>();
        let mut embeddings = MultiTableEmbedding::<MlxNeuralBackend>::new(specs, stream).unwrap();
        for (table, named) in embeddings.tables.iter_mut().enumerate() {
            let weight = (0..4 * dimensions)
                .map(|index| (table as f32 + 1.0) * (index as f32 + 1.0) / 128.0)
                .collect::<Vec<_>>();
            bind_embedding(
                &mut named.embedding,
                Array::from_slice(&weight, &[4, dimensions]),
                stream,
            );
        }
        let first = MlxTensor::from_array(Array::from_slice(&[0_i32], &[1]));
        let sentinel = MlxTensor::from_array(Array::from_slice(&[-1_i32], &[1]));
        let third = MlxTensor::from_array(Array::from_slice(&[2_i32], &[1]));
        let scope = TokenValidationScope::begin().unwrap();
        let output = embeddings
            .forward(&[&first, &sentinel, &third], stream)
            .unwrap();
        assert_eq!(output.shape(), &[1, dimensions]);
        let output = output.as_array().as_type::<f32>(stream).unwrap();
        let validations = scope.finish();
        let event =
            async_eval_with_event(std::iter::once(&output).chain(validations.arrays())).unwrap();
        event.synchronize().unwrap();
        validations.validate_completed().unwrap();
        let output = output.evaluated().unwrap();
        assert!(output
            .as_slice::<f32>()
            .iter()
            .all(|value| value.is_finite()));
    }
}

#[test]
fn relative_attention_gathers_causal_distance_profiles() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let queries = MlxTensor::from_array(Array::from_slice(&[0.0_f32], &[1, 1, 1, 1]));
    let keys = MlxTensor::from_array(Array::from_slice(&[1.0_f32, 2.0], &[1, 1, 2, 1]));
    let values = MlxTensor::from_array(Array::from_slice(&[10.0_f32, 20.0], &[1, 1, 2, 1]));
    let profiles =
        MlxTensor::from_array(Array::from_slice(&[0.0_f32, 3.0_f32.ln()], &[1, 1, 1, 2]));
    let output = <MlxNeuralBackend as NeuralBackend>::relative_attention(
        RelativeAttentionInput {
            queries: &queries,
            keys: &keys,
            values: &values,
            profiles: &profiles,
            query_offset: 1,
            key_offset: 0,
            window: None,
            log_scaling_floor: None,
            log_scaling_alpha: 0.0,
        },
        stream,
    )
    .unwrap();
    let output = output.as_array().evaluated().unwrap();
    assert!((output.as_slice::<f32>()[0] - 12.5).abs() < 1e-5);
}

#[test]
fn joint_group_selection_selects_with_bias_but_weights_unbiased_logits() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let hidden = MlxTensor::from_array(Array::from_slice(&[1.0_f32], &[1, 1]));
    let weight = MlxTensor::from_array(Array::from_slice(&[0.0_f32, 1.0, 2.0], &[3, 1]));
    let correction = MlxTensor::from_array(Array::from_slice(&[10.0_f32, 0.0], &[2]));
    let global = MlxTensor::from_array(Array::from_slice(&[0.5_f32], &[1]));
    let selections = <MlxNeuralBackend as GroupedNeuralBackend>::joint_group_selection(
        JointGroupSelectionInput::new(
            &hidden,
            &weight,
            &correction,
            &global,
            JointGroupSelectionSpec::new(2, 1, 1, 2.0).unwrap(),
        )
        .unwrap(),
        stream,
    )
    .unwrap();
    let ids = selections.primary_indices().as_array().evaluated().unwrap();
    let grouped = selections
        .primary_coefficients()
        .as_array()
        .evaluated()
        .unwrap();
    let shared = selections
        .always_on_coefficients()
        .as_array()
        .evaluated()
        .unwrap();
    assert_eq!(ids.as_slice::<u32>(), &[0]);
    let expected = 0.5 / (0.5 + 1.0 / (1.0 + (-2.0_f32).exp()));
    assert!((grouped.as_slice::<f32>()[0] - expected).abs() < 1e-5);
    assert!((grouped.as_slice::<f32>()[0] + shared.as_slice::<f32>()[0] - 1.0).abs() < 1e-5);
}

#[test]
#[ignore = "explicit MLX normalization parity; run outside the sandbox"]
fn mlx_general_normalization_matches_scalar_references() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let values = [1.0_f32, -2.0, 3.0, -4.0];
    let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 4]));
    let epsilon = 1e-5;

    let mut normalization = <MlxNeuralBackend as NeuralBackend>::normalization(
        NormalizationConstructionSpec {
            groups: None,
            dimensions: 4,
            epsilon,
            scale: NormalizationScale::Unit,
        },
        stream,
    )
    .unwrap();
    let rms = (values.iter().map(|value| value * value).sum::<f32>() / 4.0 + epsilon).sqrt();
    let expected_rms = values.map(|value| value / rms);
    close(
        &MlxTensor::from_array(normalization.forward(input.as_array(), stream).unwrap()),
        &expected_rms,
        1e-5,
    );

    let l2 = (values.iter().map(|value| value * value).sum::<f32>() + epsilon).sqrt();
    let expected_l2 = values.map(|value| value / l2);
    close(
        &<MlxNeuralBackend as NeuralBackend>::l2_normalize(&input, epsilon, stream).unwrap(),
        &expected_l2,
        1e-5,
    );
}

#[test]
#[ignore = "explicit MLX grouped-normalization parity; run outside the sandbox"]
fn mlx_silu_gated_group_norm_matches_scalar_reference() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let values = [1.0_f32, -2.0, 3.0, -4.0];
    let gates = [0.5_f32, -1.0, 1.5, -0.25];
    let weights = [1.0_f32, 2.0, 0.5, -1.0];
    let epsilon = 1e-5;
    let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 4]));
    let gate = MlxTensor::from_array(Array::from_slice(&gates, &[1, 4]));
    let weight = MlxTensor::from_array(Array::from_slice(&weights, &[4]));
    let mut expected = [0.0_f32; 4];
    for group in 0..2 {
        let start = group * 2;
        let variance =
            (values[start] * values[start] + values[start + 1] * values[start + 1]) / 2.0;
        let scale = (variance + epsilon).sqrt().recip();
        for index in start..start + 2 {
            let silu_gate = gates[index] / (1.0 + (-gates[index]).exp());
            expected[index] = values[index] * scale * weights[index] * silu_gate;
        }
    }
    let actual = <MlxNeuralBackend as NeuralBackend>::silu_gated_group_rms_norm(
        &input, &gate, &weight, 2, epsilon, stream,
    )
    .unwrap();
    close(&actual, &expected, 1e-5);
}

#[test]
#[ignore = "explicit MLX head-expansion parity; run outside the sandbox"]
fn mlx_head_expansion_matches_scalar_reference() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 2, 2]));
    let expansion = HeadExpansion {
        axis: 1,
        source_heads: 2,
        target_heads: 4,
    };
    let actual =
        <MlxNeuralBackend as NeuralBackend>::expand_heads(&input, expansion, stream).unwrap();
    let (expected, shape) = reference_expand_heads(&values, &[1, 2, 2], 1, 4).unwrap();
    let shape = shape
        .into_iter()
        .map(|dimension| i32::try_from(dimension).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(actual.shape(), shape.as_slice());
    close(&actual, &expected, 1e-5);
}

#[test]
#[ignore = "explicit MLX segmented-attention parity; run outside the sandbox"]
fn mlx_segmented_attention_matches_scalar_reference() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let queries = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0];
    let keys = [1.0_f32, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0];
    let values = [1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let query = MlxTensor::from_array(Array::from_slice(&queries, &[4, 1, 2]));
    let key = MlxTensor::from_array(Array::from_slice(&keys, &[4, 1, 2]));
    let value = MlxTensor::from_array(Array::from_slice(&values, &[4, 1, 2]));
    let segments = [2, 2];
    let scale = 2.0_f32.sqrt().recip();
    let actual = <MlxNeuralBackend as NeuralBackend>::segmented_attention(
        SegmentedAttentionInput {
            queries: &query,
            keys: &key,
            values: &value,
            segment_lengths: &segments,
            scale,
        },
        stream,
    )
    .unwrap();
    let expected =
        reference_segmented_attention(4, 1, 2, 2, &queries, &keys, &values, &segments, scale)
            .unwrap();
    close(&actual, &expected, 2e-5);
}

#[test]
#[ignore = "explicit native activation precision conformance"]
fn scaled_softplus_preserves_bfloat16_and_rounds_once() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let values = [-20.0_f32, -2.296875, -0.69921875, 0.0, 0.5, 4.0, 25.0];
    let input = MlxTensor::from_array(
        Array::from_slice(&values, &[7])
            .as_dtype(Dtype::Bfloat16, stream)
            .unwrap(),
    );
    let beta = std::f32::consts::LN_2;
    let actual = MlxNeuralBackend::softplus(input, beta, stream).unwrap();
    assert_eq!(actual.as_array().dtype(), Dtype::Bfloat16);
    let expected = values.map(|x| {
        let wide = f64::from(beta) * f64::from(x);
        let result = ((wide.max(0.0) + (-wide.abs()).exp().ln_1p()) / f64::from(beta)) as f32;
        let bits = result.to_bits();
        f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
    });
    let actual = MlxTensor::from_array(actual.as_array().as_dtype(Dtype::Float32, stream).unwrap());
    close(&actual, &expected, 0.0);
}
