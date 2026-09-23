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
fn convolution_history_is_fixed_by_kernel_not_prompt_length() {
    let mut spec = CausalDepthwiseConvolutionSpec {
        channels: 32,
        kernel_size: 4,
        weight: parameter("conv"),
        bias: None,
        activation: ConvolutionActivation::Silu,
    };
    for tokens in [1, 9, 1024] {
        let values = spec
            .memory_invocation(2, tokens, TensorElementType::Bf16)
            .unwrap()
            .logical_values()
            .unwrap();
        assert_eq!(values[2].logical_bytes().unwrap(), 2 * 3 * 32 * 2);
        assert_eq!(values[2].kind, LogicalValueKind::State);
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
