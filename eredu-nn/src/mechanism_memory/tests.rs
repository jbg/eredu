use super::*;
use crate::{
    CausalDepthwiseConvolutionSpec, ConvolutionActivation, GroupedLinearActivation,
    GroupedLinearSpec, GroupedProjectionSpec, LinearFormatSpec, LinearSpec, ParameterSpec,
};
fn parameter(name: &str) -> ParameterSpec {
    ParameterSpec::trainable(name).unwrap()
}

#[test]
fn projection_geometry_reuses_format_and_does_not_recharge_parameters() {
    let spec = LinearSpec {
        input: 64,
        output: 96,
        weight: parameter("weight"),
        bias: None,
        format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
    };
    let request = spec
        .memory_invocation(7, TensorElementType::Bf16, Some(TensorElementType::Bf16))
        .unwrap();
    let logical = request.logical_values().unwrap();
    assert_eq!(logical[0].logical_bytes().unwrap(), 7 * 64 * 2);
    assert_eq!(logical[1].logical_bytes().unwrap(), 7 * 96 * 2);
    assert_eq!(logical.len(), 2);
    let unknown = MechanismMemoryContract::unknown(&request).unwrap();
    assert!(unknown.storage.is_empty());
    assert!(!unknown.missing.is_empty());
}

#[test]
fn grouped_projection_uses_selected_routes_and_local_output_partition() {
    let projection = GroupedProjectionSpec::new(
        parameter("experts"),
        None,
        LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
    )
    .unwrap();
    let spec = GroupedLinearSpec::new(16, 64, 128, GroupedLinearActivation::Silu, projection)
        .unwrap()
        .partition_output(32..64)
        .unwrap();
    let values = spec
        .memory_invocation(3, 4, TensorElementType::F32)
        .unwrap()
        .logical_values()
        .unwrap();
    assert_eq!(values.last().unwrap().shape, [3, 32]);
    assert_eq!(values[1].logical_bytes().unwrap(), 3 * 4 * 4);
    assert!(spec
        .memory_invocation(3, 17, TensorElementType::F32)
        .is_err());
}

#[test]
fn convolution_history_is_fixed_by_receptive_field_not_prompt_length() {
    let mut spec = CausalDepthwiseConvolutionSpec {
        dilation: 1,
        channels: 32,
        kernel_size: 4,
        weight: parameter("conv"),
        bias: None,
        activation: ConvolutionActivation::Silu,
    };
    for dilation in [1, 3, 7] {
        spec.dilation = dilation;
        for tokens in [1, 9, 1024] {
            let values = spec
                .memory_invocation(2, tokens, TensorElementType::Bf16)
                .unwrap()
                .logical_values()
                .unwrap();
            assert_eq!(
                values[2].logical_bytes().unwrap(),
                2 * 3 * dilation as u64 * 32 * 2
            );
            assert_eq!(values[2].kind, LogicalValueKind::State);
        }
    }
    spec.kernel_size = 1;
    assert_eq!(
        spec.memory_invocation(2, 3, TensorElementType::F32)
            .unwrap()
            .logical_values()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn attention_geometry_tracks_grouped_keys_and_distinct_value_width() {
    let request = MechanismInvocation::Attention {
        batch: 2,
        query_heads: 8,
        kv_heads: 2,
        queries: 7,
        keys: 91,
        key_width: 32,
        value_width: 48,
        element: TensorElementType::Bf16,
        arithmetic: crate::AttentionArithmetic::InputScores,
        softcap: false,
        sinks: true,
    };
    let values = request.logical_values().unwrap();
    assert_eq!(values[1].logical_bytes().unwrap(), 2 * 2 * 91 * 32 * 2);
    assert_eq!(values[3].logical_bytes().unwrap(), 2 * 8 * 7 * 48 * 2);
    let MechanismInvocation::Attention { mut kv_heads, .. } = request.clone() else {
        unreachable!()
    };
    kv_heads += 1;
    let mut bad = request;
    if let MechanismInvocation::Attention { kv_heads: head, .. } = &mut bad {
        *head = kv_heads;
    }
    assert!(bad.logical_values().is_err());
}

#[test]
fn recurrent_state_is_fp32_and_independent_of_scan_length() {
    for kind in [
        RecurrentKind::GatedDelta,
        RecurrentKind::SelectiveStateSpace,
    ] {
        for tokens in [1, 400] {
            let request = MechanismInvocation::Recurrent {
                kind,
                batch: 2,
                tokens,
                heads: 8,
                value_width: 64,
                state_width: 16,
                element: TensorElementType::Bf16,
                chunk_size: Some(64),
            };
            let values = request.logical_values().unwrap();
            assert_eq!(values[2].logical_bytes().unwrap(), 2 * 8 * 64 * 16 * 4);
        }
    }
}

#[test]
fn invalid_extents_and_overflow_never_turn_into_zero_or_finite_defaults() {
    let request = MechanismInvocation::CacheUpdate {
        batch: 1,
        heads: 2,
        previous: u64::MAX,
        appended: 1,
        key_width: 8,
        value_width: 8,
        element: TensorElementType::F32,
    };
    assert!(request.logical_values().is_err());
    let request = MechanismInvocation::Sampling {
        rows: 1,
        vocabulary: 0,
        history: 0,
        element: TensorElementType::F32,
        mode: SamplingMode::Greedy,
        top_k: 0,
        top_p: false,
        min_p: false,
        penalties: false,
    };
    assert!(request.logical_values().is_err());
    assert!(MechanismBytes {
        lower: 8,
        upper: Some(7)
    }
    .validate()
    .is_err());
    assert_eq!(MechanismBytes::unknown(0).upper, None);
}

#[test]
fn quantized_projection_keeps_packed_format_and_companion_contract() {
    let format = LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(64, 4).unwrap());
    let spec = LinearSpec {
        input: 128,
        output: 64,
        weight: parameter("packed"),
        bias: None,
        format: LinearFormatSpec::affine(format, parameter("scale"), parameter("affine_bias"))
            .unwrap(),
    };
    let request = spec
        .memory_invocation(9, TensorElementType::Bf16, None)
        .unwrap();
    assert!(
        matches!(request, MechanismInvocation::Projection { format: LinearFormat::Affine(config), .. } if config.bits == 4)
    );
    let values = request.logical_values().unwrap();
    assert_eq!(values[1].logical_bytes().unwrap(), 9 * 64 * 2);
    assert_eq!(
        values.len(),
        2,
        "companion parameters are borrowed through prepared bindings"
    );
}

#[test]
fn convolution_rejects_invalid_or_overflowing_receptive_fields() {
    let mut spec = CausalDepthwiseConvolutionSpec {
        channels: 1,
        kernel_size: 4,
        dilation: 0,
        weight: parameter("conv"),
        bias: None,
        activation: ConvolutionActivation::Identity,
    };
    assert!(spec.validate().is_err());
    spec.dilation = -1;
    assert!(spec.validate().is_err());
    spec.dilation = i32::MAX;
    assert!(spec.validate().is_err());
    spec.kernel_size = 2;
    assert!(spec.validate().is_err());
    spec.kernel_size = 1;
    assert_eq!(spec.history_len().unwrap(), 0);
    spec.validate().unwrap();
    spec.kernel_size = 0;
    assert!(spec.history_len().is_err());
    let invocation = MechanismInvocation::Convolution {
        batch: 1,
        tokens: 1,
        channels: 1,
        kernel: u64::MAX,
        dilation: 2,
        element: TensorElementType::F32,
    };
    assert!(invocation.logical_values().is_err());
}

#[test]
fn rotary_geometry_preserves_prefix_and_outputs_float32_for_all_layouts() {
    use crate::multimodal::{MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec};
    let mut spec = MultiAxisRotarySpec {
        axes: vec![
            RotaryAxisSpec {
                dimensions: 4,
                position_offset: -2,
            },
            RotaryAxisSpec {
                dimensions: 6,
                position_offset: 1,
            },
        ],
        base: 100.0,
        minimum_position: 0,
        layout: MultiAxisRotaryLayout::IndependentAxes,
    };
    for layout in [
        MultiAxisRotaryLayout::IndependentAxes,
        MultiAxisRotaryLayout::SplitHalves,
        MultiAxisRotaryLayout::RoundRobinSections,
    ] {
        spec.layout = layout;
        let values = spec
            .memory_invocation(&[2, 3, 2], TensorElementType::I32)
            .unwrap()
            .logical_values()
            .unwrap();
        assert_eq!(values[0].shape, [2, 3, 2]);
        assert_eq!(values[0].logical_bytes().unwrap(), 48);
        for output in &values[1..] {
            assert_eq!(output.shape, [2, 3, 10]);
            assert_eq!(output.element, TensorElementType::F32);
            assert_eq!(output.logical_bytes().unwrap(), 240);
        }
    }
    for bad_shape in [vec![], vec![2], vec![2, 3], vec![0, 2], vec![u64::MAX, 2]] {
        assert!(spec
            .memory_invocation(&bad_shape, TensorElementType::I32)
            .is_err());
    }
    spec.axes[0].dimensions = 3;
    assert!(spec
        .memory_invocation(&[2, 2], TensorElementType::I32)
        .is_err());
    spec.axes[0].dimensions = 4;
    spec.base = f32::NAN;
    assert!(spec
        .memory_invocation(&[2, 2], TensorElementType::I32)
        .is_err());
}

#[test]
fn projection_bias_representation_requires_a_present_floating_parameter() {
    let invocation = |bias, bias_element| MechanismInvocation::Projection {
        rows: 3,
        input: 5,
        output: 7,
        format: eredu_checkpoint::LinearFormat::Dense,
        element: TensorElementType::Bf16,
        weight_element: Some(TensorElementType::Bf16),
        bias,
        bias_element,
    };
    for (bias, element) in [
        (false, Some(TensorElementType::F32)),
        (true, Some(TensorElementType::I32)),
        (true, Some(TensorElementType::Bool)),
    ] {
        assert!(invocation(bias, element).logical_values().is_err());
    }
    for (bias, element) in [
        (false, None),
        (true, None),
        (true, Some(TensorElementType::F32)),
    ] {
        let values = invocation(bias, element).logical_values().unwrap();
        assert_eq!(values[0].shape, [3, 5]);
        assert_eq!(values[1].shape, [3, 7]);
        // Geometry is portable; selected native arithmetic establishes promotion.
        assert_eq!(values[1].element, TensorElementType::Bf16);
    }
}

#[test]
fn allocation_capacity_intervals_contain_payload_and_preserve_unknown_upper() {
    for (payload, capacity) in [
        (0, MechanismBytes::exact(0)),
        (0, MechanismBytes::exact(64)),
        (
            65,
            MechanismBytes {
                lower: 128,
                upper: Some(256),
            },
        ),
        (65, MechanismBytes::unknown(65)),
        (u64::MAX, MechanismBytes::exact(u64::MAX)),
        (u64::MAX, MechanismBytes::unknown(u64::MAX)),
    ] {
        capacity.validate_for_payload(payload).unwrap();
    }
    let unknown = MechanismBytes::unknown(65);
    unknown.validate_for_payload(65).unwrap();
    assert_eq!(unknown.upper, None);
}

#[test]
fn allocation_capacity_rejects_inverted_or_insufficient_intervals() {
    for (payload, capacity) in [
        (65, MechanismBytes::exact(64)),
        (
            65,
            MechanismBytes {
                lower: 0,
                upper: Some(128),
            },
        ),
        (65, MechanismBytes::unknown(64)),
        (
            65,
            MechanismBytes {
                lower: 128,
                upper: Some(127),
            },
        ),
        (
            0,
            MechanismBytes {
                lower: 1,
                upper: Some(0),
            },
        ),
    ] {
        assert!(capacity.validate_for_payload(payload).is_err());
    }
}

#[test]
fn layer_normalization_geometry_preserves_nominal_dtype_and_borrowed_affine_parameters() {
    for (weight, weight_element, bias, bias_element) in [
        (false, None, false, None),
        (true, None, true, None),
        (
            true,
            Some(TensorElementType::F32),
            true,
            Some(TensorElementType::F64),
        ),
        (false, None, true, Some(TensorElementType::F32)),
    ] {
        let invocation = MechanismInvocation::LayerNormalization {
            rows: 3,
            width: 7,
            element: TensorElementType::Bf16,
            weight,
            weight_element,
            bias,
            bias_element,
        };
        let values = invocation.logical_values().unwrap();
        assert_eq!(
            values.len(),
            2,
            "affine parameters are borrowed, not workspace"
        );
        assert_eq!(values[0].name, "input");
        assert_eq!(values[1].name, "output");
        for value in values {
            assert_eq!(value.shape, [3, 7]);
            assert_eq!(value.element, TensorElementType::Bf16);
            assert_eq!(value.logical_bytes().unwrap(), 42);
        }
    }
}

#[test]
fn layer_normalization_geometry_rejects_bad_presence_types_and_extents() {
    let make = |rows, width, element, weight, weight_element, bias, bias_element| {
        MechanismInvocation::LayerNormalization {
            rows,
            width,
            element,
            weight,
            weight_element,
            bias,
            bias_element,
        }
    };
    for invocation in [
        make(0, 7, TensorElementType::F32, false, None, false, None),
        make(3, 0, TensorElementType::F32, false, None, false, None),
        make(
            u64::MAX,
            7,
            TensorElementType::F32,
            false,
            None,
            false,
            None,
        ),
        make(3, 7, TensorElementType::I32, false, None, false, None),
        make(
            3,
            7,
            TensorElementType::F32,
            false,
            Some(TensorElementType::F32),
            false,
            None,
        ),
        make(
            3,
            7,
            TensorElementType::F32,
            false,
            None,
            false,
            Some(TensorElementType::F32),
        ),
        make(
            3,
            7,
            TensorElementType::F32,
            true,
            Some(TensorElementType::Bool),
            false,
            None,
        ),
        make(
            3,
            7,
            TensorElementType::F32,
            false,
            None,
            true,
            Some(TensorElementType::Complex64),
        ),
    ] {
        assert!(invocation.logical_values().is_err());
    }
}
