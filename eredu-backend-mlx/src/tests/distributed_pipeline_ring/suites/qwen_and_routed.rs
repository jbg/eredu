/// Verifies dependency-safe Gemma placement, auxiliary-state transport, shared
/// KV decode state, and prompt-cache restoration across two pipeline stages.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gemma_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Gemma);
}

/// Exercises Gemma 4's canonical vision, audio, and decoder unit traversal
/// through bounded checkpoint streaming across two pipeline stages.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_gemma4_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Gemma);
}

/// Verifies biased GQA, mixed full/sliding cache persistence, and two-stage
/// execution for a Qwen2 decoder.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen2_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Qwen2);
}

/// Compares Qwen2 TP=2 prefill and decode logits with the fully resident
/// single-rank public-loader result.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen2 fixture"]
fn ring_two_process_qwen2_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_qwen_fixture(checkpoint.path(), "qwen2");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Qwen2,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves indexed SafeTensors follows the same neutral Qwen2 TP constructor.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen2_indexed_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_indexed_qwen_fixture(checkpoint.path(), "qwen2");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Qwen2,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves Qwen2 pure PP uses one neutral manifest and the public cache lifecycle.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen2_pipeline_resident_reference() {
    run_ring_pipeline_mode(false, FixtureFamily::Qwen2, WorkerMode::OpaqueSession);
}

/// Compares dense Qwen3 TP=2 prefill and decode logits with the fully
/// resident single-rank public-loader result.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic dense Qwen3 fixture"]
fn ring_two_process_qwen3_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_qwen_fixture(checkpoint.path(), "qwen3");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Qwen3,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves one-rank MLX cache preparation failure rolls back the peer shard,
/// propagates its causal phase, and fences retry before further native work.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_prompt_cache_prepare_failure_rolls_back_and_fences() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_qwen_fixture(checkpoint.path(), "qwen3");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Qwen3,
        WorkerMode::PromptCachePrepareFailure,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves Qwen3 pure PP uses the family-blind neutral resident driver.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_pipeline_resident_reference() {
    run_ring_pipeline_mode(false, FixtureFamily::Qwen3, WorkerMode::OpaqueSession);
}

/// Proves dense Qwen2 GGUF enters the neutral TP path without a complete shell.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen2_gguf_tensor_parallel_resident_reference() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen2Gguf,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves dense Qwen3 GGUF enters the neutral pure-PP path.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_gguf_pipeline_resident_reference() {
    run_ring_pipeline_mode(false, FixtureFamily::Qwen3Gguf, WorkerMode::OpaqueSession);
}

/// Compares Qwen2 TP=2 x PP=2 prefill, decode, and prompt-cache continuity
/// with the fully resident single-rank public-loader result.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Qwen2 fixture"]
fn ring_four_process_qwen2_tensor_pipeline_resident_reference() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen2,
        "tp-pp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves Qwen3 TP=2 x PP=2 numeric/cache parity through neutral construction.
#[test]
#[ignore = "requires the MLX Ring backend and four loopback CPU ranks"]
fn ring_four_process_qwen3_tensor_pipeline_resident_reference() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3,
        "tp-pp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves bounded host-local Qwen2 units retain neutral TP execution.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen2_layerwise_host_tensor_parallel_reference() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen2,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves disk-streamed Qwen3 units retain neutral pure-PP execution.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_dense_stream_pipeline_reference() {
    run_ring_pipeline_mode(true, FixtureFamily::Qwen3, WorkerMode::OpaqueSession);
}

/// Proves the architecture-selected affine transform materializes once for TP.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_transformed_tensor_parallel_reference() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3,
        "tp",
        WorkerMode::OpaqueSessionRequantize,
    );
}

/// Proves the architecture-selected affine transform materializes once for PP.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_transformed_pipeline_reference() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen3,
        WorkerMode::OpaqueSessionRequantize,
    );
}

/// Proves transformed Qwen3 TP=2 x PP=2 uses one neutral construction.
#[test]
#[ignore = "requires the MLX Ring backend and four loopback CPU ranks"]
fn ring_four_process_qwen3_transformed_tensor_pipeline_reference() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3,
        "tp-pp",
        WorkerMode::OpaqueSessionRequantize,
    );
}

/// Proves routed Qwen3-MoE GGUF uses the neutral partitioned session.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_moe_gguf_tensor_parallel_neutral_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3MoeGguf,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Verifies Q/K-normalized Qwen3 execution through streamed local layers.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Qwen3);
}

/// Verifies Qwen3 routed-expert ownership, paged cache persistence, and
/// rank-synchronized two-stage execution through the shared Qwen stage.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_moe_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Qwen3Moe);
}

/// Verifies GPT-OSS native MXFP4 experts and mixed full/sliding state across
/// two pipeline ranks.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_pipeline() {
    run_ring_pipeline(false, FixtureFamily::GptOss);
}

/// Proves public Qwen3-MoE TP uses the typed neutral partition constructor.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_moe_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves Qwen3-MoE pure PP uses routed local units and typed boundaries.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_moe_pipeline_opaque_session() {
    run_ring_pipeline_mode(false, FixtureFamily::Qwen3Moe, WorkerMode::OpaqueSession);
}

/// Proves Qwen3-MoE TP=2 x PP=2 stays on the neutral routed driver.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_tensor_pipeline_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-pp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves public GPT-OSS TP uses the typed neutral partition constructor.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves GPT-OSS pure PP uses the neutral routed pipeline strategy.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_pipeline_opaque_session() {
    run_ring_pipeline_mode(false, FixtureFamily::GptOss, WorkerMode::OpaqueSession);
}

/// Proves GPT-OSS TP=2 x PP=2 uses the neutral routed pipeline strategy.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_gpt_oss_tensor_pipeline_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp-pp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves Qwen3-MoE PP=2 x EP=2 follows the architecture-selected collective
/// wave on inactive pipeline stages through one neutral public session.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_pipeline_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves GPT-OSS PP=2 x EP=2 preserves its biased reverse wave through one
/// neutral public session.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_gpt_oss_pipeline_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves GPT-OSS TP=2 x PP=2 x EP=2 executes the architecture-selected
/// world-wide expert waves through the neutral public session.
#[test]
#[ignore = "spawns eight local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_triple_axis_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves public Qwen3-MoE EP uses neutral owner/count exchange.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_moe_expert_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves public GPT-OSS EP uses neutral owner/count exchange.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_expert_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves public Qwen3-MoE TP=2 x EP=2 uses one neutral manifest and the
/// consensus-proven overlapping logical-subgroup exchange wave.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_tensor_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves public GPT-OSS TP=2 x EP=2 retains its biased expert semantics over
/// one neutral manifest and the consensus-proven logical-subgroup wave.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_gpt_oss_tensor_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves TP+EP uses the selected rank-local addressable Qwen bank and exposes
/// eviction/reload telemetry through the neutral public session.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_tensor_expert_addressable_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-ep",
        WorkerMode::OpaqueSessionEvictingAddressableParameterBank,
    );
}

/// Proves PP+EP uses one neutral session with independent Qwen expert banks.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_pipeline_expert_addressable_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Proves GPT-OSS TP+PP+EP composes bounded ordinary storage with independent
/// expert banks while preserving post-reduction bias exactly once.
#[test]
#[ignore = "spawns eight local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_streamed_triple_axis_addressable_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Proves the ReLU-squared routed equation uses the same exact addressable
/// session while inactive PP stages retain an empty local bank catalog.
#[test]
#[ignore = "spawns eight local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_nemotron_h_streamed_triple_axis_addressable_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::NemotronH,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Proves GPT-OSS resident TP+PP+EP execution against the single-rank model.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::GptOss, "tp-pp-ep");
}

/// Exercises GPT-OSS triple-axis dense streaming with independent expert caches.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises GPT-OSS host-backed non-expert layers with independent expert
/// caching across TP=2 x PP=2 x EP=2.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises canonical type-39 GPT-OSS GGUF across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_gguf_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::GptOssGguf,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Covers independent GPT-OSS expert caching when PP is active without EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_pipeline_parameter_bank() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies opaque-session execution with GPT-OSS cached experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_gpt_oss_pipeline_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}
