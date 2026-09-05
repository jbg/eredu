/// Run with:
/// `cargo test -p eredu-backend-mlx --lib tests::distributed_pipeline_ring::ring_two_process_pipeline -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Llama);
}

#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_pipeline_inspection_uses_canonical_paths() {
    run_ring_pipeline_mode(false, FixtureFamily::Llama, WorkerMode::OpaqueInspection);
}

#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_pipeline_final_output_intervention_is_uniform_across_families() {
    for family in [
        FixtureFamily::Llama,
        FixtureFamily::Qwen3,
        FixtureFamily::DeepSeek,
        FixtureFamily::KimiLinear,
    ] {
        run_ring_pipeline_mode(false, family, WorkerMode::FinalOutputIntervention);
    }
}

/// Runs backend-generic text generation across a pipeline whose non-output
/// rank legitimately produces no local logits.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_pipeline_generic_text_generation() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Llama,
        WorkerMode::OpaqueTextGeneration,
    );
}

/// Compares the public Llama TP=2 session's prefill and decode logits with a
/// fully resident single-rank reference built from the identical checkpoint.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Llama fixture"]
fn ring_two_process_llama_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Compares a two-rank PP=2 neutral Llama session with the same resident reference.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Llama fixture"]
fn ring_two_process_llama_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Compares a four-rank TP=2, PP=2 neutral session with the same resident reference.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Llama fixture"]
fn ring_four_process_llama_tensor_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp-pp"),
    );
}

/// Proves selected affine transformation stays inside the neutral TP session.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Llama fixture"]
fn ring_two_process_llama_transformed_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSessionRequantize,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves a nonzero PP-local source unit is transformed into its exact target unit.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Llama fixture"]
fn ring_two_process_llama_transformed_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSessionRequantize,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Runs the same TP=2 resident-reference oracle through the Mistral
/// specialization of the shared neutral decoder.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Mistral fixture"]
fn ring_two_process_mistral_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_mistral_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Compares a two-rank PP=2 neutral Mistral session with the same resident reference.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Mistral fixture"]
fn ring_two_process_mistral_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_mistral_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Compares a four-rank TP=2, PP=2 neutral Mistral session with the same resident reference.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Mistral fixture"]
fn ring_four_process_mistral_tensor_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_mistral_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp-pp"),
    );
}

/// Proves the public neutral TP loader consumes a single unindexed Llama
/// SafeTensors payload without selecting the complete-model bridge.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_llama_unindexed_safetensors_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_unindexed_llama_compatible_fixture(checkpoint.path(), "llama");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves the public neutral PP loader consumes a single unindexed Mistral
/// SafeTensors payload and matches the same-artifact resident reference.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_mistral_unindexed_safetensors_pipeline_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_unindexed_llama_compatible_fixture(checkpoint.path(), "mistral");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Proves the public neutral PP loader consumes an admitted Llama GGUF store
/// without instantiating a backend-owned pipeline model.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_llama_gguf_pipeline_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = checkpoint.path().join("model.gguf");
    write_llama_compatible_gguf(&checkpoint_path, "llama");
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Proves the public neutral TP loader consumes an admitted Mistral GGUF store
/// and matches the same-artifact resident reference.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_mistral_gguf_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = checkpoint.path().join("model.gguf");
    write_llama_compatible_gguf(&checkpoint_path, "mistral");
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves the public generic loader and architecture-erased model session own
/// pipeline loading, prefill, repeated decode, cache state, and communication.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_opaque_model_session() {
    run_ring_pipeline_mode(false, FixtureFamily::Llama, WorkerMode::OpaqueSession);
}

/// Verifies fair multi-request scheduling, independent request caches, exact
/// schedule consensus, variable prompt shapes, decode parity, EOS, and cancel.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_pipeline_opaque_session_repeated_decode() {
    run_ring_pipeline_mode(false, FixtureFamily::Llama, WorkerMode::OpaqueSession);
}

/// Verifies the public Llama PP session genuinely selects disk-streamed local
/// layers while preserving numeric output, cache isolation, and neutral ownership.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_dense_stream_opaque_session() {
    run_ring_pipeline_mode(true, FixtureFamily::Llama, WorkerMode::OpaqueSession);
}

/// Verifies public Mistral TP execution genuinely selects host-layerwise
/// traversal while retaining the neutral partitioned constructor.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_mistral_layerwise_host_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_mistral_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::LayerwiseHost,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Verifies that divergent rank-local schedules fail before point-to-point
/// Exercises paged cache selection through the opaque session lifecycle.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_pipeline_opaque_session_cache_policy() {
    run_ring_pipeline_mode(false, FixtureFamily::Llama, WorkerMode::OpaqueSession);
}

/// Run with:
/// `cargo test -p eredu-backend-mlx --lib tests::distributed_pipeline_ring::ring_two_process_dense_stream_pipeline -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Llama);
}

/// Verifies stage-local 4-bit materialization precedes TP+PP dense-stream
/// residency and preserves synchronized prefill, decode, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_requantized_dense_stream_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(true, FixtureFamily::Qwen3, "tp-pp", WorkerMode::Requantize);
}

/// Verifies the same packed stage overlay feeds host-layerwise residency before
/// bounded device promotion under TP+PP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_requantized_layerwise_host_tensor_pipeline() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3,
        "tp-pp",
        WorkerMode::Requantize,
    );
}

/// Verifies host-resident stage layers and bounded device promotion compose
/// with TP-sharded attention, cache state, and pipeline transport.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_layerwise_host_tensor_pipeline() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3,
        "tp-pp",
        WorkerMode::Standard,
    );
}
