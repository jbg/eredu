/// Verifies Qwen3-Next linear and full-attention state through distributed
/// prefill, decode, persistence, and bounded streaming.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_next_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Qwen3Next);
}

/// Verifies that Qwen3.5 uses the same hybrid pipeline contract without being
/// treated as a Qwen3-Next checkpoint alias.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen35_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Qwen35);
}

/// Verifies output-owned Qwen Hybrid prediction state is persisted and restored
/// with the target prompt cache instead of being truncated to the decoder range.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen35_prompt_cache_round_trip_includes_mtp() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen35,
        WorkerMode::QwenHybridPromptCache,
    );
}

/// Verifies Qwen3-Next TP=2 + PP=2 across recurrent and full-attention stages,
/// including rank-local state, persistence, and synchronized generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_next_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3Next, "tp-pp");
}

/// Proves the unsupported direct Qwen3-Next prediction route is rejected at
/// neutral TP admission; embedded prediction requires its selected adapter.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen3-Next fixture"]
fn ring_two_process_qwen3_next_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Next,
        "tp",
        WorkerMode::OpaqueUnsupportedDirectPartition,
    );
}

/// Proves the unsupported direct Qwen3.5 prediction route is rejected at
/// neutral TP admission; embedded prediction requires its selected adapter.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen3.5 fixture"]
fn ring_two_process_qwen35_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35,
        "tp",
        WorkerMode::OpaqueUnsupportedDirectPartition,
    );
}

/// Covers pure TP loading and text execution through the conditional Qwen3.5
/// graph, including its vision-aware static parameter topology.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic multimodal Qwen3.5 fixture"]
fn ring_two_process_qwen35_multimodal_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35Multimodal,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Covers the prediction-free conditional Qwen graph through the generic
/// composite partition binder. Prediction-bearing variants are classified as
/// separate neutral prediction targets before composite admission.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic conditional Qwen fixture"]
fn ring_two_process_qwen35_zero_prediction_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35ZeroPrediction,
        "tp",
        WorkerMode::OpaqueQwenConditionalMedia,
    );
}

/// Covers the prediction-free conditional Qwen media graph across two neutral
/// pipeline owners without constructing a family pipeline shell.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic conditional Qwen fixture"]
fn ring_two_process_qwen35_zero_prediction_pipeline_neutral_composite_session() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen35ZeroPrediction,
        WorkerMode::OpaqueQwenConditionalMedia,
    );
}

/// Covers pure TP loading and multimodal execution for Qwen3-VL through the
/// neutral composite partition binder.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen3-VL fixture"]
fn ring_two_process_qwen3_vl_multimodal_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Vl,
        "tp",
        WorkerMode::OpaqueQwen3VlMedia,
    );
}

/// Proves Qwen3-VL media continuation traverses the ordinary neutral PP session.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_vl_pipeline_neutral_composite_session() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen3Vl,
        WorkerMode::OpaqueQwen3VlMedia,
    );
}

/// Verifies Qwen3.5-MoE TP=2 + PP=2 with tensor-sharded routed/shared
/// intermediates and corresponding-coordinate pipeline transport.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen35_moe_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen35Moe, "tp-pp");
}

/// Verifies scheduled typed image ingress, TP-sharded vision/text blocks,
/// corresponding-coordinate PP transport, cached decode, and persistence.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen35_multimodal_tensor_pipeline() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen35Multimodal, "tp-pp");
}

/// Verifies Qwen3-VL DeepStack vision, mRoPE text, and position-delta state
/// through tensor- and pipeline-parallel prefill and decode.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_vl_tensor_pipeline() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3Vl, "tp-pp");
}

/// Exercises Qwen3-VL's vision unit and decoder layers through bounded
/// checkpoint streaming under TP=2 x PP=2, with resident-reference logits.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Qwen3-VL media fixture"]
fn ring_four_process_qwen3_vl_streamed_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3Vl, "tp-pp");
}

/// Exercises the same Qwen3-VL TP+PP graph with host-resident media/decoder
/// units and a one-unit device window.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Qwen3-VL host-layerwise media fixture"]
fn ring_four_process_qwen3_vl_layerwise_host_tensor_pipeline() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3Vl,
        "tp-pp",
        WorkerMode::Standard,
    );
}

/// Exercises routed Qwen3-VL across TP=2 x PP=2 x EP=2 with the neutral
/// shared vision tower and stage-local expert ownership.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen3_vl_moe_triple_axis() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3VlMoe,
        "tp-pp-ep",
        WorkerMode::OpaqueQwen3VlMedia,
    );
}

/// Combines bounded Qwen3-VL media/decoder streaming with independent cached
/// routed experts across TP=2 x PP=2 x EP=2.
#[test]
#[ignore = "requires the MLX Ring backend, eight loopback CPU ranks, and the synthetic Qwen3-VL-MoE media fixture"]
fn ring_eight_process_qwen3_vl_moe_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3VlMoe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Combines host-layerwise Qwen3-VL media/decoder units with independently
/// cached routed experts across all three parallel axes.
#[test]
#[ignore = "requires the MLX Ring backend, eight loopback CPU ranks, and the synthetic Qwen3-VL-MoE host-layerwise media fixture"]
fn ring_eight_process_qwen3_vl_moe_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3VlMoe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies multimodal Qwen3.5-MoE across all Cartesian axes with bounded
/// media/decoder reads and independently cached routed experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen35_moe_multimodal_streamed_triple_axis() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen35MoeMultimodal,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies the same typed ingress and expert-storage semantics with host-backed
/// vision and decoder windows.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen35_moe_multimodal_layerwise_host_triple_axis() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen35MoeMultimodal,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies Qwen3.5-MoE PP=2 + EP=2 with stage-local packed experts,
/// recurrent/full-attention state, persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen35_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen35Moe, "pp-ep");
}

/// Verifies the Qwen3-Next checkpoint specialization uses the same PP+EP
/// ownership and transport contract as Qwen3.5-MoE.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_next_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3NextMoe, "pp-ep");
}

/// Verifies resident Qwen3-Next-MoE execution across all Cartesian axes,
/// including recurrent state, routed/shared TP projections, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen3_next_moe_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3NextMoe, "tp-pp-ep");
}

/// Verifies Qwen3.5-MoE dense streaming composes with stage/EP-local expert
/// caches, bounded reads, prompt-cache reload, and synchronized decode.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen35_moe_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen35Moe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises host-backed hybrid non-expert layers with independent expert
/// caching across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen3_next_moe_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3NextMoe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Covers independent hybrid expert caching when PP is active without EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen35_moe_pipeline_parameter_bank() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen35Moe,
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies cached Qwen hybrid expert execution through one session.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen35_moe_pipeline_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35Moe,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

// These use the existing public capture, global-mask, parameter, overlay and
// controlled-replay harness with the hybrid family's actual component points.
fn run_qwen_hybrid_component_matrix(family: FixtureFamily, routed: bool) {
    let topologies: &[&str] = if routed {
        &["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"]
    } else {
        &["tp", "pp", "tp-pp"]
    };
    for axes in topologies {
        eprintln!("Qwen hybrid components: {family:?} {axes} resident");
        run_ring_cartesian_pipeline_mode(false, family, axes, WorkerMode::OpaqueComponentCapture);
        eprintln!("Qwen hybrid components: {family:?} {axes} host");
        run_ring_layerwise_host_cartesian_pipeline_mode(
            family,
            axes,
            WorkerMode::OpaqueComponentCapture,
        );
        eprintln!("Qwen hybrid components: {family:?} {axes} disk");
        run_ring_cartesian_pipeline_mode(true, family, axes, WorkerMode::OpaqueComponentCapture);
    }
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_public_component_capture_qwen_next_hybrid_matrix() {
    run_qwen_hybrid_component_matrix(FixtureFamily::Qwen3Next, false);
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_public_component_capture_qwen_35_hybrid_matrix() {
    run_qwen_hybrid_component_matrix(FixtureFamily::Qwen35, false);
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_public_component_capture_qwen_next_hybrid_moe_matrix() {
    run_qwen_hybrid_component_matrix(FixtureFamily::Qwen3NextMoe, true);
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_public_component_capture_qwen_35_hybrid_moe_matrix() {
    run_qwen_hybrid_component_matrix(FixtureFamily::Qwen35Moe, true);
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen_next_hybrid_moe_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3NextMoe,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_public_component_capture_qwen_35_conditional_matrix() {
    run_qwen_hybrid_component_matrix(FixtureFamily::Qwen35Multimodal, false);
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_public_component_capture_qwen_35_conditional_moe_matrix() {
    run_qwen_hybrid_component_matrix(FixtureFamily::Qwen35MoeMultimodal, true);
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_qwen_35_conditional_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35Multimodal,
        "tp-pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen_35_conditional_pipeline_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen35Multimodal,
        "pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires eight local MLX Ring ranks"]
fn ring_public_component_capture_qwen_35_conditional_moe_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35MoeMultimodal,
        "tp-pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns local CPU Ring processes and opens loopback sockets; run explicitly"]
fn ring_qwen_text_prediction_components_tensor_parallel() {
    for family in [
        FixtureFamily::Qwen3Next,
        FixtureFamily::Qwen3NextMoe,
        FixtureFamily::Qwen35,
        FixtureFamily::Qwen35Moe,
    ] {
        eprintln!("Qwen prediction component family={family:?} mode=tp");
        run_ring_cartesian_pipeline_mode(false, family, "tp", WorkerMode::OpaqueQwenHybridMtp);
    }
}

#[test]
#[ignore = "spawns local CPU Ring processes and opens loopback sockets; run explicitly"]
fn ring_qwen_text_prediction_components_pipeline_parallel() {
    for family in [
        FixtureFamily::Qwen3Next,
        FixtureFamily::Qwen3NextMoe,
        FixtureFamily::Qwen35,
        FixtureFamily::Qwen35Moe,
    ] {
        eprintln!("Qwen prediction component family={family:?} mode=pp");
        run_ring_cartesian_pipeline_mode(false, family, "pp", WorkerMode::OpaqueQwenHybridMtp);
    }
}

fn run_qwen_prediction_components(family: FixtureFamily, routed: bool) {
    run_component_matrix_with_mode(family, routed, WorkerMode::OpaqueQwenHybridMtp);
}

fn run_component_matrix_with_mode(family: FixtureFamily, routed: bool, mode: WorkerMode) {
    let topologies: &[&str] = if routed {
        &["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"]
    } else {
        &["tp", "pp", "tp-pp"]
    };
    for axes in topologies {
        eprintln!("Components: {family:?} {axes} resident");
        run_ring_cartesian_pipeline_mode(false, family, axes, mode);
        eprintln!("Components: {family:?} {axes} host");
        run_ring_layerwise_host_cartesian_pipeline_mode(family, axes, mode);
        eprintln!("Components: {family:?} {axes} disk");
        run_ring_cartesian_pipeline_mode(true, family, axes, mode);
    }
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_composite_independent_bank_components_tensor_parallel() {
    for family in [
        FixtureFamily::Inkling,
        FixtureFamily::InklingGguf,
        FixtureFamily::Qwen3NextMoe,
        FixtureFamily::Qwen35Moe,
        FixtureFamily::Qwen35MoeMultimodal,
    ] {
        eprintln!("Independent component TP2: {family:?}");
        run_ring_cartesian_pipeline_mode(
            false,
            family,
            "tp",
            WorkerMode::OpaqueComponentCaptureAddressableBank,
        );
    }
    eprintln!("Independent component TP2: Inkling affine");
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "tp",
        WorkerMode::OpaqueComponentCaptureAddressableBankRequantize,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_independent_bank_components_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::Inkling,
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_affine_independent_bank_components_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::Inkling,
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBankRequantize,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_gguf_independent_bank_components_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::InklingGguf,
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_independent_bank_components_matrix() {
    for family in [
        FixtureFamily::Qwen3NextMoe,
        FixtureFamily::Qwen35Moe,
        FixtureFamily::Qwen35MoeMultimodal,
    ] {
        run_component_matrix_with_mode(
            family,
            true,
            WorkerMode::OpaqueComponentCaptureAddressableBank,
        );
    }
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_inkling_target_components_tensor_parallel() {
    for family in [FixtureFamily::InklingDense, FixtureFamily::Inkling] {
        run_ring_cartesian_pipeline_mode(false, family, "tp", WorkerMode::OpaqueComponentCapture);
    }
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_inkling_gguf_target_components_tensor_parallel() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::InklingGguf,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_gguf_target_components_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::InklingGguf,
        true,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns two to four local MLX Ring ranks; run explicitly"]
fn ring_inkling_target_components_dense_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::InklingDense,
        false,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_target_components_routed_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::Inkling,
        true,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_inkling_target_quantized_components_tensor_parallel() {
    for mode in [
        WorkerMode::OpaqueComponentCaptureRequantize,
        WorkerMode::OpaqueComponentCaptureMxFp4,
    ] {
        for family in [FixtureFamily::InklingDense, FixtureFamily::Inkling] {
            eprintln!("Inkling target components: {family:?} {mode:?} tp");
            run_ring_cartesian_pipeline_mode(false, family, "tp", mode);
        }
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_target_affine_components_matrix() {
    for (family, routed) in [
        (FixtureFamily::InklingDense, false),
        (FixtureFamily::Inkling, true),
    ] {
        run_component_matrix_with_mode(
            family,
            routed,
            WorkerMode::OpaqueComponentCaptureRequantize,
        );
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_target_mxfp4_components_matrix() {
    for (family, routed) in [
        (FixtureFamily::InklingDense, false),
        (FixtureFamily::Inkling, true),
    ] {
        run_component_matrix_with_mode(family, routed, WorkerMode::OpaqueComponentCaptureMxFp4);
    }
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_inkling_prediction_quantized_components_tensor_parallel() {
    for mode in [
        WorkerMode::OpaqueInklingComponentsRequantize,
        WorkerMode::OpaqueInklingComponentsMxFp4,
    ] {
        for family in [FixtureFamily::InklingDense, FixtureFamily::Inkling] {
            eprintln!("Inkling prediction components: {family:?} {mode:?} tp");
            run_ring_cartesian_pipeline_mode(false, family, "tp", mode);
        }
    }
}

#[test]
#[ignore = "spawns four local MLX Ring ranks; run explicitly"]
fn ring_inkling_prediction_quantized_components_tensor_expert_parallel() {
    for mode in [
        WorkerMode::OpaqueInklingComponentsRequantize,
        WorkerMode::OpaqueInklingComponentsMxFp4,
    ] {
        run_ring_cartesian_pipeline_mode(false, FixtureFamily::Inkling, "tp-ep", mode);
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_prediction_affine_components_matrix() {
    for (family, routed) in [
        (FixtureFamily::InklingDense, false),
        (FixtureFamily::Inkling, true),
    ] {
        run_component_matrix_with_mode(
            family,
            routed,
            WorkerMode::OpaqueInklingComponentsRequantize,
        );
    }
}

#[test]
#[ignore = "spawns eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_prediction_affine_components_three_axes() {
    let family = FixtureFamily::Inkling;
    let mode = WorkerMode::OpaqueInklingComponentsRequantize;
    run_ring_cartesian_pipeline_mode(false, family, "tp-pp-ep", mode);
    run_ring_layerwise_host_cartesian_pipeline_mode(family, "tp-pp-ep", mode);
    run_ring_cartesian_pipeline_mode(true, family, "tp-pp-ep", mode);
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_prediction_mxfp4_components_matrix() {
    for (family, routed) in [
        (FixtureFamily::InklingDense, false),
        (FixtureFamily::Inkling, true),
    ] {
        run_component_matrix_with_mode(family, routed, WorkerMode::OpaqueInklingComponentsMxFp4);
    }
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_inkling_prediction_components_tensor_parallel() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "tp",
        WorkerMode::OpaqueInklingComponents,
    );
}

#[test]
#[ignore = "spawns two to four local MLX Ring ranks; run explicitly"]
fn ring_inkling_prediction_components_dense_target_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::InklingDense,
        false,
        WorkerMode::OpaqueInklingComponents,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_inkling_prediction_components_routed_target_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::Inkling,
        true,
        WorkerMode::OpaqueInklingComponents,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_prediction_components_next_matrix() {
    run_qwen_prediction_components(FixtureFamily::Qwen3Next, false);
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_prediction_components_next_moe_matrix() {
    run_qwen_prediction_components(FixtureFamily::Qwen3NextMoe, true);
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_prediction_components_35_matrix() {
    run_qwen_prediction_components(FixtureFamily::Qwen35, false);
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_prediction_components_35_moe_matrix() {
    run_qwen_prediction_components(FixtureFamily::Qwen35Moe, true);
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_prediction_components_35_conditional_matrix() {
    run_qwen_prediction_components(FixtureFamily::Qwen35Multimodal, false);
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_prediction_components_35_conditional_moe_matrix() {
    run_qwen_prediction_components(FixtureFamily::Qwen35MoeMultimodal, true);
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_qwen_prediction_fp8_components_tensor_parallel() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35MoeMultimodal,
        "tp",
        WorkerMode::OpaqueQwenHybridMtpFp8,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_prediction_fp8_components_conditional_moe_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::Qwen35MoeMultimodal,
        true,
        WorkerMode::OpaqueQwenHybridMtpFp8,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_qwen_prediction_fp8_components_expert_parallel() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35MoeMultimodal,
        "ep",
        WorkerMode::OpaqueQwenHybridMtpFp8,
    );
}

#[test]
#[ignore = "spawns two to four local MLX Ring ranks; run explicitly"]
fn ring_nemotron_prediction_components_dense_target_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::NemotronH,
        false,
        WorkerMode::OpaqueNemotronHMtp,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_nemotron_prediction_components_routed_tensor_parallel() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::NemotronH,
        "tp",
        WorkerMode::OpaqueNemotronHMtpRouted,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_nemotron_prediction_components_routed_target_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::NemotronH,
        true,
        WorkerMode::OpaqueNemotronHMtpRouted,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_nemotron_prediction_mxfp4_components_tensor_parallel() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::NemotronH,
        "tp",
        WorkerMode::OpaqueNemotronHMtpRoutedMxFp4,
    );
}

#[test]
#[ignore = "spawns two to four local MLX Ring ranks; run explicitly"]
fn ring_nemotron_prediction_mxfp4_components_dense_target_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::NemotronH,
        false,
        WorkerMode::OpaqueNemotronHMtpMxFp4,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_nemotron_prediction_mxfp4_components_routed_target_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::NemotronH,
        true,
        WorkerMode::OpaqueNemotronHMtpRoutedMxFp4,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_muse_components_tensor_parallel() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::MuseGlimmer,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns two to four local MLX Ring ranks; run explicitly"]
fn ring_muse_components_dense_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::MuseGlimmer,
        false,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_muse_components_routed_tensor_parallel() {
    for mode in [
        WorkerMode::OpaqueComponentCapture,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    ] {
        run_ring_cartesian_pipeline_mode(false, FixtureFamily::MuseGlimmerMoe, "tp", mode);
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_muse_components_routed_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::MuseGlimmerMoe,
        true,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_muse_components_independent_bank_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::MuseGlimmerMoe,
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_muse_components_quantized_tensor_parallel() {
    for family in [FixtureFamily::MuseGlimmer, FixtureFamily::MuseGlimmerMoe] {
        for mode in [
            WorkerMode::OpaqueComponentCaptureRequantize,
            WorkerMode::OpaqueComponentCaptureMxFp4,
        ] {
            eprintln!("Muse quantized TP: {family:?} {mode:?}");
            run_ring_cartesian_pipeline_mode(false, family, "tp", mode);
        }
    }
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::MuseGlimmerMoe,
        "tp",
        WorkerMode::OpaqueComponentCaptureAddressableBankRequantize,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_muse_components_affine_matrix() {
    for (family, routed) in [
        (FixtureFamily::MuseGlimmer, false),
        (FixtureFamily::MuseGlimmerMoe, true),
    ] {
        run_component_matrix_with_mode(
            family,
            routed,
            WorkerMode::OpaqueComponentCaptureRequantize,
        );
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_muse_components_mxfp4_matrix() {
    for (family, routed) in [
        (FixtureFamily::MuseGlimmer, false),
        (FixtureFamily::MuseGlimmerMoe, true),
    ] {
        run_component_matrix_with_mode(family, routed, WorkerMode::OpaqueComponentCaptureMxFp4);
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_muse_components_affine_independent_bank_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::MuseGlimmerMoe,
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBankRequantize,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_muse_components_gguf_tensor_parallel() {
    for routed in [false, true] {
        eprintln!("Muse GGUF TP: sparse={routed} ordinary banks");
        run_ring_cartesian_pipeline_mode(
            false,
            FixtureFamily::MuseGlimmerGguf(routed),
            "tp",
            WorkerMode::OpaqueComponentCapture,
        );
    }
    eprintln!("Muse GGUF TP: sparse=true independent banks");
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::MuseGlimmerGguf(true),
        "tp",
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_muse_components_gguf_matrix() {
    for routed in [false, true] {
        run_component_matrix_with_mode(
            FixtureFamily::MuseGlimmerGguf(routed),
            routed,
            WorkerMode::OpaqueComponentCapture,
        );
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_muse_components_gguf_independent_bank_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::MuseGlimmerGguf(true),
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

#[test]
#[ignore = "spawns eight local MLX Ring ranks; run explicitly"]
fn ring_muse_components_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::MuseGlimmerMoe,
        "tp-pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_components_tensor_parallel() {
    for family in [FixtureFamily::Qwen3Vl, FixtureFamily::Qwen3VlMoe] {
        eprintln!("Qwen3-VL component TP: {family:?}");
        run_ring_cartesian_pipeline_mode(false, family, "tp", WorkerMode::OpaqueComponentCapture);
    }
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3VlMoe,
        "tp",
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_components_matrix() {
    for (family, routed) in [
        (FixtureFamily::Qwen3Vl, false),
        (FixtureFamily::Qwen3VlMoe, true),
    ] {
        run_component_matrix_with_mode(family, routed, WorkerMode::OpaqueComponentCapture);
    }
}

#[test]
#[ignore = "spawns eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_components_cartesian_replay() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3VlMoe,
        "tp-pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_components_independent_bank_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::Qwen3VlMoe,
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_components_quantized_tensor_parallel() {
    for family in [FixtureFamily::Qwen3Vl, FixtureFamily::Qwen3VlMoe] {
        for mode in [
            WorkerMode::OpaqueComponentCaptureRequantize,
            WorkerMode::OpaqueComponentCaptureMxFp4,
        ] {
            eprintln!("Qwen3-VL quantized TP: {family:?} {mode:?}");
            run_ring_cartesian_pipeline_mode(false, family, "tp", mode);
        }
    }
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3VlMoe,
        "tp",
        WorkerMode::OpaqueComponentCaptureAddressableBankRequantize,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_components_affine_matrix() {
    for (family, routed) in [
        (FixtureFamily::Qwen3Vl, false),
        (FixtureFamily::Qwen3VlMoe, true),
    ] {
        run_component_matrix_with_mode(
            family,
            routed,
            WorkerMode::OpaqueComponentCaptureRequantize,
        );
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_components_mxfp4_matrix() {
    for (family, routed) in [
        (FixtureFamily::Qwen3Vl, false),
        (FixtureFamily::Qwen3VlMoe, true),
    ] {
        run_component_matrix_with_mode(family, routed, WorkerMode::OpaqueComponentCaptureMxFp4);
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_components_affine_independent_bank_matrix() {
    run_component_matrix_with_mode(
        FixtureFamily::Qwen3VlMoe,
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBankRequantize,
    );
}

fn run_gemma4_component_case(
    sparse: bool,
    axes: &'static str,
    residency: WorkerResidency,
    mode: WorkerMode,
) {
    eprintln!(
        "Gemma4 component case sparse={sparse} axes={axes} residency={residency:?} mode={mode:?}"
    );
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_component_fixture(
        checkpoint.path(),
        sparse,
        mode.is_quantized_component_capture(),
    );
    let path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        residency,
        FixtureFamily::Gemma,
        mode,
        checkpoint,
        path,
        Some(axes),
    );
}

fn run_gemma4_component_matrix(sparse: bool, mode: WorkerMode) {
    let axes: &[&str] = if sparse {
        &["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"]
    } else {
        &["tp", "pp", "tp-pp"]
    };
    for axes in axes {
        for (label, residency) in [
            ("resident", WorkerResidency::FullyResident),
            ("host", WorkerResidency::LayerwiseHost),
            ("disk", WorkerResidency::DenseDiskStream),
        ] {
            eprintln!("Gemma4 components sparse={sparse} {axes} {label}");
            run_gemma4_component_case(sparse, axes, residency, mode);
        }
    }
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_gemma4_components_tensor_parallel() {
    for sparse in [false, true] {
        run_gemma4_component_case(
            sparse,
            "tp",
            WorkerResidency::FullyResident,
            WorkerMode::OpaqueComponentCapture,
        );
    }
    run_gemma4_component_case(
        true,
        "tp",
        WorkerResidency::FullyResident,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_gemma4_components_matrix() {
    for sparse in [false, true] {
        run_gemma4_component_matrix(sparse, WorkerMode::OpaqueComponentCapture);
    }
}

#[test]
#[ignore = "spawns eight local MLX Ring ranks; run explicitly"]
fn ring_gemma4_components_triple_axis() {
    for residency in [
        WorkerResidency::FullyResident,
        WorkerResidency::LayerwiseHost,
        WorkerResidency::DenseDiskStream,
    ] {
        run_gemma4_component_case(
            true,
            "tp-pp-ep",
            residency,
            WorkerMode::OpaqueComponentCapture,
        );
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_gemma4_components_independent_bank_matrix() {
    run_gemma4_component_matrix(true, WorkerMode::OpaqueComponentCaptureAddressableBank);
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_gemma4_components_quantized_tensor_parallel() {
    for sparse in [false, true] {
        for mode in [
            WorkerMode::OpaqueComponentCaptureRequantize,
            WorkerMode::OpaqueComponentCaptureMxFp4,
        ] {
            run_gemma4_component_case(sparse, "tp", WorkerResidency::FullyResident, mode);
        }
    }
    run_gemma4_component_case(
        true,
        "tp",
        WorkerResidency::FullyResident,
        WorkerMode::OpaqueComponentCaptureAddressableBankRequantize,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_gemma4_components_affine_matrix() {
    for sparse in [false, true] {
        run_gemma4_component_matrix(sparse, WorkerMode::OpaqueComponentCaptureRequantize);
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_gemma4_components_mxfp4_matrix() {
    for sparse in [false, true] {
        run_gemma4_component_matrix(sparse, WorkerMode::OpaqueComponentCaptureMxFp4);
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_gemma4_components_affine_independent_bank_matrix() {
    run_gemma4_component_matrix(
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBankRequantize,
    );
}

fn run_gemma4_gguf_component_case(
    sparse: bool,
    axes: &'static str,
    residency: WorkerResidency,
    mode: WorkerMode,
) {
    eprintln!(
        "Gemma4 GGUF components sparse={sparse} axes={axes} residency={residency:?} mode={mode:?}"
    );
    let checkpoint = tempfile::tempdir().unwrap();
    let path = checkpoint.path().join("model.gguf");
    write_gemma4_gguf_component_fixture(&path, sparse);
    run_ring_pipeline_processes(
        residency,
        FixtureFamily::Gemma,
        mode,
        checkpoint,
        path,
        Some(axes),
    );
}

fn run_gemma4_gguf_component_matrix(sparse: bool, mode: WorkerMode) {
    let axes: &[&str] = if sparse {
        &["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"]
    } else {
        &["tp", "pp", "tp-pp"]
    };
    for axes in axes {
        for residency in [
            WorkerResidency::FullyResident,
            WorkerResidency::LayerwiseHost,
            WorkerResidency::DenseDiskStream,
        ] {
            run_gemma4_gguf_component_case(sparse, axes, residency, mode);
        }
    }
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_gemma4_gguf_components_tensor_parallel() {
    for sparse in [false, true] {
        run_gemma4_gguf_component_case(
            sparse,
            "tp",
            WorkerResidency::FullyResident,
            WorkerMode::OpaqueComponentCapture,
        );
    }
    run_gemma4_gguf_component_case(
        true,
        "tp",
        WorkerResidency::FullyResident,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_gemma4_gguf_components_matrix() {
    for sparse in [false, true] {
        run_gemma4_gguf_component_matrix(sparse, WorkerMode::OpaqueComponentCapture);
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_gemma4_gguf_components_independent_bank_matrix() {
    run_gemma4_gguf_component_matrix(true, WorkerMode::OpaqueComponentCaptureAddressableBank);
}

fn run_qwen_vl_gguf_component_case(
    routed: bool,
    axes: &'static str,
    residency: WorkerResidency,
    mode: WorkerMode,
) {
    run_qwen_vl_gguf_encoded_component_case(routed, axes, residency, mode, false);
}

fn run_qwen_vl_gguf_encoded_component_case(
    routed: bool,
    axes: &'static str,
    residency: WorkerResidency,
    mode: WorkerMode,
    packed: bool,
) {
    eprintln!(
        "Qwen3-VL GGUF components routed={routed} packed={packed} {axes} {residency:?} {mode:?}"
    );
    let checkpoint = tempfile::tempdir().unwrap();
    let path = checkpoint.path().join("model.gguf");
    if packed {
        write_qwen_vl_gguf_component_fixture_encoded(&path, routed, true);
    } else {
        write_qwen_vl_gguf_component_fixture(&path, routed);
    }
    run_ring_pipeline_processes(
        residency,
        if routed {
            FixtureFamily::Qwen3VlMoe
        } else {
            FixtureFamily::Qwen3Vl
        },
        mode,
        checkpoint,
        path,
        Some(axes),
    );
}

#[test]
#[ignore = "spawns local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_gguf_components_tensor_parallel() {
    for routed in [false, true] {
        run_qwen_vl_gguf_component_case(
            routed,
            "tp",
            WorkerResidency::FullyResident,
            WorkerMode::OpaqueComponentCapture,
        );
    }
    run_qwen_vl_gguf_component_case(
        true,
        "tp",
        WorkerResidency::FullyResident,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

fn run_qwen_vl_gguf_component_matrix(routed: bool, mode: WorkerMode) {
    let axes: &[&str] = if routed {
        &["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"]
    } else {
        &["tp", "pp", "tp-pp"]
    };
    for &axes in axes {
        for residency in [
            WorkerResidency::FullyResident,
            WorkerResidency::LayerwiseHost,
            WorkerResidency::DenseDiskStream,
        ] {
            run_qwen_vl_gguf_component_case(routed, axes, residency, mode);
        }
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_gguf_components_matrix() {
    for routed in [false, true] {
        run_qwen_vl_gguf_component_matrix(routed, WorkerMode::OpaqueComponentCapture);
    }
}

#[test]
#[ignore = "spawns two to eight local MLX Ring ranks; run explicitly"]
fn ring_qwen_vl_gguf_components_independent_bank_matrix() {
    run_qwen_vl_gguf_component_matrix(true, WorkerMode::OpaqueComponentCaptureAddressableBank);
}

#[test]
#[ignore = "spawns local CPU Ring processes and opens loopback sockets; run explicitly"]
fn ring_qwen_prediction_preparation_failures() {
    let mode = WorkerMode::OpaqueQwenHybridMtpPreparationFailure;
    let family = FixtureFamily::Qwen3Next;
    run_ring_cartesian_pipeline_mode(false, family, "tp", mode);
    run_ring_layerwise_host_cartesian_pipeline_mode(family, "tp", mode);
    run_ring_cartesian_pipeline_mode(true, family, "pp", mode);
    run_ring_cartesian_pipeline_mode(false, family, "tp-pp", mode);
}

#[test]
#[ignore = "spawns local CPU Ring processes and opens loopback sockets; run explicitly"]
fn ring_qwen_prediction_visitor_preparation_failures() {
    for mode in [
        WorkerMode::OpaqueQwenHybridMtpSchedulerFailure,
        WorkerMode::OpaqueQwenHybridMtpControlSetupFailure,
    ] {
        let family = FixtureFamily::Qwen3Next;
        run_ring_cartesian_pipeline_mode(false, family, "tp", mode);
        run_ring_layerwise_host_cartesian_pipeline_mode(family, "tp", mode);
        run_ring_cartesian_pipeline_mode(true, family, "pp", mode);
        run_ring_cartesian_pipeline_mode(false, family, "tp-pp", mode);
    }
}

#[test]
#[ignore = "spawns local CPU Ring processes and opens loopback sockets; run explicitly"]
fn ring_qwen_prediction_coordinated_control_delivery() {
    for case in [
        "cancel_prefill",
        "cancel_pending",
        "record",
        "caller",
        "terminal",
        "sampling",
    ] {
        eprintln!("speculative coordinated delivery case={case}");
        let mode = WorkerMode::OpaqueQwenHybridMtpControlDelivery(case);
        let family = FixtureFamily::Qwen3Next;
        run_ring_cartesian_pipeline_mode(false, family, "tp", mode);
        run_ring_layerwise_host_cartesian_pipeline_mode(family, "tp", mode);
        run_ring_cartesian_pipeline_mode(true, family, "pp", mode);
        run_ring_cartesian_pipeline_mode(false, family, "tp-pp", mode);
    }
}

#[test]
#[ignore = "spawns local CPU Ring processes with a packed DeepStack projector; run explicitly"]
fn ring_qwen_vl_packed_gguf_components_focused() {
    for (routed, axes, residency, mode) in [
        (
            false,
            "tp",
            WorkerResidency::FullyResident,
            WorkerMode::OpaqueComponentCapture,
        ),
        (
            true,
            "pp",
            WorkerResidency::DenseDiskStream,
            WorkerMode::OpaqueComponentCapture,
        ),
        (
            true,
            "tp-pp-ep",
            WorkerResidency::LayerwiseHost,
            WorkerMode::OpaqueComponentCaptureAddressableBank,
        ),
    ] {
        run_qwen_vl_gguf_encoded_component_case(routed, axes, residency, mode, true);
    }
}

fn run_qwen_vl_packed_gguf_component_matrix(routed: bool, mode: WorkerMode) {
    let axes: &[&str] = if routed {
        &["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"]
    } else {
        &["tp", "pp", "tp-pp"]
    };
    for &axes in axes {
        for residency in [
            WorkerResidency::FullyResident,
            WorkerResidency::LayerwiseHost,
            WorkerResidency::DenseDiskStream,
        ] {
            run_qwen_vl_gguf_encoded_component_case(routed, axes, residency, mode, true);
        }
    }
}

#[test]
#[ignore = "spawns local CPU Ring processes with a packed DeepStack projector; run explicitly"]
fn ring_qwen_vl_packed_gguf_components_matrix() {
    for routed in [false, true] {
        run_qwen_vl_packed_gguf_component_matrix(routed, WorkerMode::OpaqueComponentCapture);
    }
}

#[test]
#[ignore = "spawns local CPU Ring processes with a packed DeepStack projector; run explicitly"]
fn ring_qwen_vl_packed_gguf_components_independent_bank_matrix() {
    run_qwen_vl_packed_gguf_component_matrix(
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}
