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
