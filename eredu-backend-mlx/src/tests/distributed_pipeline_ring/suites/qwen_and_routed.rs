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

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_qwen3_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3,
        "tp-pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen3_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen3_pipeline_streamed() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::Qwen3,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen3_tensor_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3MoeGguf,
        "tp-pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3MoeGguf,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_pipeline_streamed() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::Qwen3MoeGguf,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_tensor_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3MoeGguf,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3MoeGguf,
        "ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_tensor_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3MoeGguf,
        "tp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires eight local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3MoeGguf,
        "tp-pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3MoeGguf,
        "pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_pipeline_expert_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3MoeGguf,
        "pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_pipeline_expert_disk() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3MoeGguf,
        "pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires eight local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_tensor_pipeline_expert_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3MoeGguf,
        "tp-pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires eight local MLX Ring ranks"]
fn ring_public_component_capture_qwen_moe_packed_gguf_tensor_pipeline_expert_disk() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3MoeGguf,
        "tp-pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_qwen3_moe_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires eight local MLX Ring ranks"]
fn ring_public_component_capture_qwen3_moe_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_gpt_oss_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires eight local MLX Ring ranks"]
fn ring_public_component_capture_gpt_oss_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_lfm2_moe_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Lfm2Moe,
        "pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires eight local MLX Ring ranks"]
fn ring_public_component_capture_lfm2_moe_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Lfm2Moe,
        "tp-pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_lfm2_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Lfm2,
        "tp-pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_lfm2_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Lfm2,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_lfm2_pipeline_streamed() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::Lfm2,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_lfm2_tensor_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Lfm2,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_lfm2_moe_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Lfm2Moe,
        "tp-pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_lfm2_moe_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Lfm2Moe,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_lfm2_moe_pipeline_streamed() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::Lfm2Moe,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_lfm2_moe_tensor_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Lfm2Moe,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_nanbeige_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Nanbeige,
        "tp-pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_nanbeige_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Nanbeige,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_nanbeige_pipeline_streamed() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::Nanbeige,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_nanbeige_tensor_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Nanbeige,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen3_moe_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_qwen3_moe_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen3_moe_tensor_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3Moe,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_qwen3_moe_pipeline_streamed() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::Qwen3Moe,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_gpt_oss_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires four local MLX Ring ranks"]
fn ring_public_component_capture_gpt_oss_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp-pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_gpt_oss_tensor_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::GptOss,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks"]
fn ring_public_component_capture_gpt_oss_pipeline_streamed() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::GptOss,
        WorkerMode::OpaqueComponentCapture,
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

#[test]
#[ignore = "spawns local CPU ranks and opens loopback sockets; run explicitly"]
fn ring_k2_dense_and_mova_partition_matrix() {
    for family in [FixtureFamily::K2Dense, FixtureFamily::K2Mova] {
        for axes in ["tp", "tp-pp"] {
            run_ring_cartesian_pipeline_mode(false, family, axes, WorkerMode::OpaqueSession);
        }
        run_ring_pipeline_mode(false, family, WorkerMode::OpaqueSession);
        run_ring_layerwise_host_cartesian_pipeline_mode(family, "tp", WorkerMode::OpaqueSession);
        run_ring_pipeline_mode(true, family, WorkerMode::OpaqueSession);
    }
    for axes in ["ep", "tp-ep", "pp-ep", "tp-pp-ep"] {
        run_ring_cartesian_pipeline_mode(
            false,
            FixtureFamily::K2Mova,
            axes,
            WorkerMode::OpaqueSession,
        );
    }
}

#[test]
#[ignore = "spawns local CPU ranks and opens loopback sockets; run explicitly"]
fn ring_k2_mova_bounded_bank_partition_matrix() {
    for axes in ["tp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"] {
        run_ring_cartesian_pipeline_mode(
            false,
            FixtureFamily::K2Mova,
            axes,
            WorkerMode::OpaqueSessionAddressableParameterBank,
        );
    }
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::K2Mova,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::K2Mova,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

#[test]
#[ignore = "spawns local CPU ranks and opens loopback sockets; run explicitly"]
fn ring_k2_mova_quantized_banks_tensor_pipeline_expert() {
    for format in [GgmlType::Q4_0, GgmlType::MxFp4, GgmlType::IQ4NL] {
        for residency in [
            WorkerResidency::FullyResident,
            WorkerResidency::LayerwiseHost,
            WorkerResidency::DenseDiskStream,
        ] {
            let checkpoint = crate::tests::support::k2_horizon::gguf(format);
            let path = checkpoint.path().join("packed.gguf");
            run_ring_pipeline_processes(
                residency,
                FixtureFamily::K2Mova,
                WorkerMode::OpaqueSessionAddressableParameterBank,
                checkpoint,
                path,
                Some("tp-pp-ep"),
            );
        }
    }
}

#[test]
#[ignore = "spawns local CPU ranks and opens loopback sockets; run explicitly"]
fn ring_k2_mova_fp8_banks_tensor_pipeline_expert() {
    for residency in [
        WorkerResidency::FullyResident,
        WorkerResidency::LayerwiseHost,
        WorkerResidency::DenseDiskStream,
    ] {
        let (_dense, checkpoint) = crate::tests::support::k2_horizon::fp8();
        let path = checkpoint.path().to_owned();
        run_ring_pipeline_processes(
            residency,
            FixtureFamily::K2Mova,
            WorkerMode::OpaqueSessionAddressableParameterBank,
            checkpoint,
            path,
            Some("tp-pp-ep"),
        );
    }
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_dense_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::K2Dense,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_dense_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::K2Dense,
        "tp-pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_dense_tensor_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::K2Dense,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_dense_pipeline_disk() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::K2Dense,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_mova_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::K2Mova,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_mova_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::K2Mova,
        "tp-pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_mova_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::K2Mova,
        "ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_mova_tensor_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::K2Mova,
        "tp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_mova_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::K2Mova,
        "pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_mova_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::K2Mova,
        "tp-pp-ep",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_mova_tensor_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::K2Mova,
        "tp",
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "requires local MLX Ring ranks"]
fn ring_public_component_capture_k2_mova_pipeline_disk() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::K2Mova,
        WorkerMode::OpaqueComponentCapture,
    );
}

// The mixed Mamba/dense-ReLU²/routed-ReLU²/attention fixture exercises native
// component capture and edits across actual architecture and pipeline boundaries.
macro_rules! nemotron_component_matrix {
    ($name:ident, $residency:ident, $gguf:expr) => {
        #[test]
        #[ignore = "requires native MLX Ring with up to eight loopback processes"]
        fn $name() {
            for axes in [
                None,
                Some("tp"),
                Some("ep"),
                Some("tp-ep"),
                Some("tp-pp"),
                Some("pp-ep"),
                Some("tp-pp-ep"),
            ] {
                let checkpoint = tempfile::tempdir().unwrap();
                let (family, path) = if $gguf {
                    let path = checkpoint.path().join("model.gguf");
                    write_nemotron_h_moe_gguf_fixture_with_values(&path, true);
                    (FixtureFamily::NemotronHGguf, path)
                } else {
                    write_nemotron_component_fixture(checkpoint.path());
                    (FixtureFamily::NemotronH, checkpoint.path().to_owned())
                };
                eprintln!("Nemotron components {}, axes={axes:?}", stringify!($residency));
                run_ring_pipeline_processes(
                    WorkerResidency::$residency,
                    family,
                    WorkerMode::OpaqueComponentCapture,
                    checkpoint,
                    path,
                    axes,
                );
            }
        }
    };
}
nemotron_component_matrix!(ring_public_component_capture_nemotron_resident, FullyResident, false);
nemotron_component_matrix!(ring_public_component_capture_nemotron_host, LayerwiseHost, false);
nemotron_component_matrix!(ring_public_component_capture_nemotron_disk, DenseDiskStream, false);

nemotron_component_matrix!(ring_public_component_capture_nemotron_gguf_resident, FullyResident, true);
nemotron_component_matrix!(ring_public_component_capture_nemotron_gguf_host, LayerwiseHost, true);
nemotron_component_matrix!(ring_public_component_capture_nemotron_gguf_disk, DenseDiskStream, true);

macro_rules! k2_fp8_component_matrix {
    ($name:ident, $residency:ident, $mode:ident) => {
        #[test]
        #[ignore = "requires native MLX Ring with up to eight loopback processes"]
        fn $name() {
            for partial in [false, true] {
                for axes in [None, Some("tp"), Some("ep"), Some("tp-ep"),
                    Some("tp-pp"), Some("pp-ep"), Some("tp-pp-ep")] {
                    let checkpoint = tempfile::tempdir().unwrap();
                    write_k2_fp8_fixture(checkpoint.path(), partial);
                    let path = checkpoint.path().to_owned();
                    eprintln!("K2 grouped FP8 {}, partial={partial}, axes={axes:?}", stringify!($residency));
                    run_ring_pipeline_processes(WorkerResidency::$residency,
                        FixtureFamily::K2Fp8(partial), WorkerMode::$mode,
                        checkpoint, path, axes);
                }
            }
        }
    };
}
k2_fp8_component_matrix!(ring_public_component_capture_k2_fp8_resident, FullyResident, OpaqueComponentCapture);
k2_fp8_component_matrix!(ring_public_component_capture_k2_fp8_host, LayerwiseHost, OpaqueComponentCapture);
k2_fp8_component_matrix!(ring_public_component_capture_k2_fp8_disk, DenseDiskStream, OpaqueComponentCapture);

k2_fp8_component_matrix!(ring_public_component_capture_k2_fp8_cached_resident, FullyResident, OpaqueComponentCaptureAddressableBank);

k2_fp8_component_matrix!(ring_public_component_capture_k2_fp8_cached_host, LayerwiseHost, OpaqueComponentCaptureAddressableBank);

k2_fp8_component_matrix!(ring_public_component_capture_k2_fp8_cached_disk, DenseDiskStream, OpaqueComponentCaptureAddressableBank);

// These fixtures cover packed columns, ordinary biases, ReLU², and companion
// storage through the same public independently cached parameter lifecycle.
macro_rules! cached_component_bank_matrix {
    ($name:ident, $residency:ident) => {
        #[test]
        #[ignore = "requires native MLX Ring with up to eight loopback processes"]
        fn $name() {
            for family in [FixtureFamily::GptOss, FixtureFamily::Qwen3MoeGguf,
                FixtureFamily::NemotronH, FixtureFamily::NemotronHGguf] {
                for axes in [None, Some("tp"), Some("ep"), Some("tp-ep"),
                    Some("tp-pp"), Some("pp-ep"), Some("tp-pp-ep")] {
                    let checkpoint = tempfile::tempdir().unwrap();
                    let path = match family {
                        FixtureFamily::GptOss => {
                            write_gpt_oss_fixture_with_patterns(checkpoint.path(), true);
                            checkpoint.path().to_owned()
                        }
                        FixtureFamily::Qwen3MoeGguf => {
                            let path = checkpoint.path().join("model.gguf");
                            write_qwen3_moe_gguf_fixture(&path, true);
                            path
                        }
                        FixtureFamily::NemotronH => {
                            write_nemotron_component_fixture(checkpoint.path());
                            checkpoint.path().to_owned()
                        }
                        FixtureFamily::NemotronHGguf => {
                            let path = checkpoint.path().join("model.gguf");
                            write_nemotron_h_moe_gguf_fixture_with_values(&path, true);
                            path
                        }
                        _ => unreachable!(),
                    };
                    eprintln!("Cached bank components {family:?}, {}, axes={axes:?}", stringify!($residency));
                    run_ring_pipeline_processes(WorkerResidency::$residency, family,
                        WorkerMode::OpaqueComponentCaptureAddressableBank, checkpoint, path, axes);
                }
            }
        }
    };
}
cached_component_bank_matrix!(ring_public_component_capture_cached_banks_resident, FullyResident);
cached_component_bank_matrix!(ring_public_component_capture_cached_banks_host, LayerwiseHost);
cached_component_bank_matrix!(ring_public_component_capture_cached_banks_disk, DenseDiskStream);

#[test]
#[ignore = "requires native MLX Ring tensor-parallel workers"]
fn ring_public_component_capture_fp8_partition_tail_tensor() {
    let checkpoint = tempfile::tempdir().unwrap();
    write_k2_fp8_fixture_with_dense_tail(checkpoint.path(), true, true);
    let path = checkpoint.path().to_owned();
    run_ring_pipeline_processes(WorkerResidency::FullyResident, FixtureFamily::K2Fp8(true),
        WorkerMode::OpaqueComponentCapture, checkpoint, path, Some("tp"));
}

#[test]
#[ignore = "requires native MLX Ring tensor-parallel workers"]
fn ring_public_component_capture_fp8_fused_tail_tensor() {
    let checkpoint = tempfile::tempdir().unwrap();
    write_k2_fp8_fixture_with_tails(checkpoint.path(), true, true, true);
    let path = checkpoint.path().to_owned();
    run_ring_pipeline_processes(WorkerResidency::FullyResident, FixtureFamily::K2Fp8(true),
        WorkerMode::OpaqueComponentCapture, checkpoint, path, Some("tp"));
}

macro_rules! fp8_partition_tail_matrix {
    ($name:ident, $residency:ident, $mode:ident) => {
        #[test]
        #[ignore = "requires native MLX Ring with up to eight loopback processes"]
        fn $name() {
            for axes in [None, Some("tp"), Some("ep"), Some("tp-ep"),
                Some("tp-pp"), Some("pp-ep"), Some("tp-pp-ep")] {
                let checkpoint = tempfile::tempdir().unwrap();
                write_k2_fp8_fixture_with_dense_tail(checkpoint.path(), true, true);
                let path = checkpoint.path().to_owned();
                eprintln!("FP8 sharded tail {}, {}, axes={axes:?}", stringify!($residency), stringify!($mode));
                run_ring_pipeline_processes(WorkerResidency::$residency,
                    FixtureFamily::K2Fp8(true), WorkerMode::$mode, checkpoint, path, axes);
            }
        }
    };
}
fp8_partition_tail_matrix!(ring_public_component_capture_fp8_partition_tail_matrix_resident, FullyResident, OpaqueComponentCapture);
fp8_partition_tail_matrix!(ring_public_component_capture_fp8_partition_tail_matrix_host, LayerwiseHost, OpaqueComponentCapture);
fp8_partition_tail_matrix!(ring_public_component_capture_fp8_partition_tail_matrix_disk, DenseDiskStream, OpaqueComponentCapture);
fp8_partition_tail_matrix!(ring_public_component_capture_fp8_partition_tail_matrix_cached_resident, FullyResident, OpaqueComponentCaptureAddressableBank);
fp8_partition_tail_matrix!(ring_public_component_capture_fp8_partition_tail_matrix_cached_host, LayerwiseHost, OpaqueComponentCaptureAddressableBank);
fp8_partition_tail_matrix!(ring_public_component_capture_fp8_partition_tail_matrix_cached_disk, DenseDiskStream, OpaqueComponentCaptureAddressableBank);

macro_rules! fp8_fused_tail_matrix {
    ($name:ident, $residency:ident, $mode:ident) => {
        #[test]
        #[ignore = "requires native MLX Ring with up to eight loopback processes"]
        fn $name() {
            for axes in [None, Some("tp"), Some("ep"), Some("tp-ep"),
                Some("tp-pp"), Some("pp-ep"), Some("tp-pp-ep")] {
                let checkpoint = tempfile::tempdir().unwrap();
                write_k2_fp8_fixture_with_tails(checkpoint.path(), true, true, true);
                let path = checkpoint.path().to_owned();
                eprintln!("FP8 fused tail {}, {}, axes={axes:?}", stringify!($residency), stringify!($mode));
                run_ring_pipeline_processes(WorkerResidency::$residency,
                    FixtureFamily::K2Fp8(true), WorkerMode::$mode, checkpoint, path, axes);
            }
        }
    };
}
fp8_fused_tail_matrix!(ring_public_component_capture_fp8_fused_tail_matrix_resident, FullyResident, OpaqueComponentCapture);
fp8_fused_tail_matrix!(ring_public_component_capture_fp8_fused_tail_matrix_host, LayerwiseHost, OpaqueComponentCapture);
fp8_fused_tail_matrix!(ring_public_component_capture_fp8_fused_tail_matrix_disk, DenseDiskStream, OpaqueComponentCapture);
fp8_fused_tail_matrix!(ring_public_component_capture_fp8_fused_tail_matrix_cached_resident, FullyResident, OpaqueComponentCaptureAddressableBank);
fp8_fused_tail_matrix!(ring_public_component_capture_fp8_fused_tail_matrix_cached_host, LayerwiseHost, OpaqueComponentCaptureAddressableBank);
fp8_fused_tail_matrix!(ring_public_component_capture_fp8_fused_tail_matrix_cached_disk, DenseDiskStream, OpaqueComponentCaptureAddressableBank);

#[test]
#[ignore = "requires native MLX Ring tensor-parallel workers"]
fn ring_public_component_capture_fp8_attention_tail_tensor() {
    let checkpoint = tempfile::tempdir().unwrap();
    write_k2_fp8_fixture_with_attention_tails(checkpoint.path());
    let path = checkpoint.path().to_owned();
    run_ring_pipeline_processes(WorkerResidency::FullyResident, FixtureFamily::K2Fp8(true),
        WorkerMode::OpaqueComponentCapture, checkpoint, path, Some("tp"));
}

macro_rules! fp8_attention_tail_matrix {
    ($name:ident, $residency:ident, $mode:ident) => {
        fp8_attention_tail_matrix!($name, $residency, $mode, write_k2_fp8_fixture_with_attention_tails);
    };
    ($name:ident, $residency:ident, $mode:ident, $write:ident) => {
        #[test]
        #[ignore = "requires native MLX Ring with up to eight loopback processes"]
        fn $name() {
            for axes in [None, Some("tp"), Some("ep"), Some("tp-ep"),
                Some("tp-pp"), Some("pp-ep"), Some("tp-pp-ep")] {
                let checkpoint = tempfile::tempdir().unwrap();
                $write(checkpoint.path());
                let path = checkpoint.path().to_owned();
                eprintln!("FP8 attention tail {}, {}, axes={axes:?}", stringify!($residency), stringify!($mode));
                run_ring_pipeline_processes(WorkerResidency::$residency,
                    FixtureFamily::K2Fp8(true), WorkerMode::$mode, checkpoint, path, axes);
            }
        }
    };
}
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_attention_tail_matrix_resident, FullyResident, OpaqueComponentCapture);
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_attention_tail_matrix_host, LayerwiseHost, OpaqueComponentCapture);
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_attention_tail_matrix_disk, DenseDiskStream, OpaqueComponentCapture);
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_attention_tail_matrix_cached_resident, FullyResident, OpaqueComponentCaptureAddressableBank);
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_attention_tail_matrix_cached_host, LayerwiseHost, OpaqueComponentCaptureAddressableBank);
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_attention_tail_matrix_cached_disk, DenseDiskStream, OpaqueComponentCaptureAddressableBank);

fp8_attention_tail_matrix!(ring_public_component_capture_fp8_non_f32_scales_resident, FullyResident, OpaqueComponentCapture, write_k2_fp8_fixture_with_non_f32_scales);
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_non_f32_scales_host, LayerwiseHost, OpaqueComponentCapture, write_k2_fp8_fixture_with_non_f32_scales);
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_non_f32_scales_disk, DenseDiskStream, OpaqueComponentCapture, write_k2_fp8_fixture_with_non_f32_scales);
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_non_f32_scales_cached_resident, FullyResident, OpaqueComponentCaptureAddressableBank, write_k2_fp8_fixture_with_non_f32_scales);
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_non_f32_scales_cached_host, LayerwiseHost, OpaqueComponentCaptureAddressableBank, write_k2_fp8_fixture_with_non_f32_scales);
fp8_attention_tail_matrix!(ring_public_component_capture_fp8_non_f32_scales_cached_disk, DenseDiskStream, OpaqueComponentCaptureAddressableBank, write_k2_fp8_fixture_with_non_f32_scales);
