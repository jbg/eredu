/// Verifies DeepSeek MLA paged-prefix persistence across two pipeline stages.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_pipeline_persistence() {
    run_ring_pipeline(false, FixtureFamily::DeepSeek);
}

/// Proves prediction-free DeepSeek-V3 pure TP uses one neutral routed session.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v3_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves a public DeepSeek-V3 artifact containing embedded MTP weights builds
/// one neutral ordinary TP target and retains its typed prediction extension,
/// without constructing a complete-TP or pipeline target shell.
#[test]
#[ignore = "spawns two local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v3_mtp_target_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp",
        WorkerMode::OpaqueDeepSeekMtpTarget,
    );
}

/// Proves prediction-free DeepSeek-V3 pure PP uses typed MLA boundaries.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v3_pipeline_opaque_session() {
    run_ring_pipeline_mode(false, FixtureFamily::DeepSeek, WorkerMode::OpaqueSession);
}

/// Proves prediction-free DeepSeek-V3 pure EP uses exact compact expert banks.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v3_expert_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves prediction-free DeepSeek-V3 TP x PP remains on the neutral driver.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v3_tensor_pipeline_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-pp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves prediction-free DeepSeek-V3 TP x EP consumes compound local banks.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v3_tensor_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves prediction-free DeepSeek-V3 PP x EP uses all-stage expert waves.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v3_pipeline_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves prediction-free DeepSeek-V3 TP x PP x EP uses one neutral session.
#[test]
#[ignore = "spawns eight local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_v3_triple_axis_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Verifies DeepSeek TP=2 + PP=2 across dense and routed-MoE stages with
/// tensor-sharded MLA, compressed caches, bounded reads, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::DeepSeek, "tp-pp");
}

/// Verifies DeepSeek PP=2 + EP=2 across a dense-to-MoE stage boundary with
/// stage-local expert ownership, compressed cache persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::DeepSeek, "pp-ep");
}

/// Proves resident DeepSeek TP=2 x PP=2 x EP=2 execution across compressed
/// MLA state, TP-sharded shared projections, and EP-owned routed experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::DeepSeek, "tp-pp-ep");
}

/// Verifies DeepSeek V4 local/compressed attention, hyper-connections, hash
/// routing, prompt persistence, and embedded MTP across two pipeline stages.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_pipeline_persistence_and_mtp() {
    run_ring_pipeline(false, FixtureFamily::DeepSeekV4);
}

/// Ensures the backend-generic speculative capability query delegates to the
/// distributed session instead of assuming a replicated complete model.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_prepared_speculative_capability() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        WorkerMode::OpaquePreparedSpeculativeCapability,
    );
}

/// Proves sequential DeepSeek-V4 MTP uses the neutral pooling-state target and
/// extension-only MLX units without a complete-TP or pipeline target shell.
#[test]
#[ignore = "spawns two local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_mtp_target_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "tp",
        WorkerMode::OpaqueDeepSeekMtpTarget,
    );
}

/// Proves admitted DeepSeek-V4 DSpark executes fused proposals, target
/// verification, and transactional commit through the public neutral scheduler.
#[test]
#[ignore = "spawns two local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_dspark_tensor_parallel_neutral_scheduler() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "tp",
        WorkerMode::OpaqueDeepSeekDsparkTarget,
    );
}

/// Proves a prediction-bearing V4 artifact reuses its neutral pure-EP target
/// when the prepared speculative capability is queried.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_expert_prepared_speculative_capability() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "ep",
        WorkerMode::OpaquePreparedSpeculativeCapability,
    );
}

/// Verifies V4 output-group/head sharding and pipeline transport together.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v4_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::DeepSeekV4, "tp-pp");
}

/// Verifies V4 token/learned routing through stage-local expert ownership.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v4_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::DeepSeekV4, "pp-ep");
}

/// Proves the prediction-free V4 target takes the neutral pooling-state route
/// through exact TP and EP manifest groups without constructing a duplicate model.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v4_prediction_free_tensor_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "tp-ep",
        WorkerMode::OpaqueSessionPredictionFree,
    );
}

/// Exercises V4 TP, PP, EP, streamed non-experts, and independent expert
/// caching in the full admitted Cartesian topology.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_v4_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeekV4,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Keeps DeepSeek non-expert stage units resident while routed experts remain
/// independently cached across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_resident_nonexpert_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises dense-streamed DeepSeek non-experts and independent expert
/// caching across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Compares the cache-backed neutral DeepSeek V3 session under TP=2 x EP=2
/// with the replicated model, covering rank-local expert geometry and exact-once TP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_cached_tensor_expert_model_parity() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Applies the same model-level TP+EP parity check to DeepSeek V4, including
/// its hyper-connection block and independently cached routed expert bank.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v4_cached_tensor_expert_model_parity() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "tp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Exercises host-layerwise DeepSeek MLA blocks and independent expert caches
/// across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Covers DeepSeek2 GGUF recipes, bounded reads, and independent expert
/// caching across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_gguf_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeekGguf,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Covers independent DeepSeek expert caching for TP+PP with EP inactive.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_tensor_pipeline_parameter_bank_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeek,
        "tp-pp",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies cached DeepSeek schedule failure reaches consensus without leaving
/// compressed MLA state reusable.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}
