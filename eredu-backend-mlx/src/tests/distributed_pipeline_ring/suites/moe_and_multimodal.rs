/// Verifies arbitrary Cartesian composition with TP-sharded Qwen3-MoE
/// projections, stage-local EP ownership, corresponding-coordinate pipeline
/// transport, cache persistence, and globally synchronized generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3Moe, "tp-pp-ep");
}

/// Verifies triple-axis ownership and generation with a tied output head.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_tied_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3MoeTied, "tp-pp-ep");
}

/// Exercises the same triple-axis semantics with bounded rank-local layer
/// materialization.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_streamed_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3Moe, "tp-pp-ep");
}

/// Proves complete Qwen3-MoE expert banks use the shared atomic packed overlay
/// under PP+EP without constructing an independent expert cache.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_fully_resident_load_time_quantization() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::Requantize,
    );
}

/// Verifies resident non-expert parameters plus stage/EP-local independent
/// expert caches across PP=2 x EP=2, including persistence and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_resident_nonexpert_pipeline_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies TP-sharded cached experts compose with dense-streamed non-experts
/// and corresponding-coordinate pipeline lanes in an eight-rank topology.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_streamed_tensor_pipeline_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3Moe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises host-backed non-expert layers, bounded device windows, and
/// independent expert caching for Qwen3-MoE across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_layerwise_host_tensor_pipeline_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3Moe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Proves the same host-layerwise path reads canonical GGUF and preserves
/// stage-local ownership in a PP=2 x EP=2 topology.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_layerwise_host_pipeline_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3MoeGguf,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises stage-local cached expert selections and bounded reads from GGUF.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_pipeline_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3MoeGguf,
        "pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Verifies one session owns both pipeline communication and expert caches.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_pipeline_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies PP-only stages cache all of their local layers' experts without
/// constructing an EP communicator. Prefill, decode, prompt persistence, and
/// synchronized generation are exercised by the shared worker.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_pipeline_parameter_bank_without_ep() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies TP-sharded cached experts and dense-streamed non-experts compose
/// across TP=2 x PP=2 while EP remains inactive.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_streamed_tensor_pipeline_parameter_bank_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3Moe,
        "tp-pp",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises PP-only cache ownership and bounded reads from canonical GGUF.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_pipeline_parameter_bank_without_ep() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::Qwen3MoeGguf,
        WorkerMode::AddressableParameterBank,
    );
}

/// Executes the Qwen3-MoE triple-axis path from a canonical GGUF and verifies
/// rank-local ownership, bounded reads, cache persistence, and parity.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3MoeGguf, "tp-pp-ep");
}

/// Exercises the canonical GGUF path with rank-local dense disk streaming.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_streamed_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3MoeGguf, "tp-pp-ep");
}

/// Verifies opaque model-session execution across every triple-axis rank.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_tensor_pipeline_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Exercises the public complete-model loader and opaque session through the
/// explicit Inkling tensor-parallel bridge without realizing a neutral manifest.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Exercises an admitted dense, prediction-free Inkling through the generic
/// composite partition binder without constructing a duplicate model shell.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_dense_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::InklingDense,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves dense Inkling's active image and audio roots traverse the same
/// neutral composite TP session used by routed Inkling variants.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_dense_multimodal_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::InklingDenseMultimodal,
        "tp",
        WorkerMode::OpaqueInklingMedia,
    );
}

/// Exercises the public complete-model loader and sparse neutral Inkling
/// decoder on a pure two-rank expert-parallel topology without PP or TP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_expert_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "ep",
        WorkerMode::OpaqueSession,
    );
}

/// Exercises ordered projected image embeddings plus the native dMel tower
/// through the public neutral Inkling TP session and prompt-cache lifecycle.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_multimodal_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::InklingMultimodal,
        "tp",
        WorkerMode::OpaqueInklingMedia,
    );
}

/// Exercises the neutral embedded predictor through the complete public TP
/// loader, rank-synchronized scheduler, sharded vocabulary, and MTP state.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_inkling_mtp_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "tp",
        WorkerMode::OpaqueInklingMtp,
    );
}

/// Exercises routed prediction units through authoritative expert providers
/// and banks on a pure expert-parallel target, without TP or PP ownership.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_inkling_mtp_expert_parallel_neutral_visitor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "ep",
        WorkerMode::OpaqueInklingMtp,
    );
}

/// Proves routed prediction reuses the target's addressable expert bank while
/// extension weights remain a separately materialized resident component.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes an addressable expert cache"]
fn ring_two_process_inkling_mtp_expert_parallel_addressable_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "ep",
        WorkerMode::OpaqueInklingMtpAddressableParameterBank,
    );
}

/// Exercises pipeline MTP through the same neutral visitor lifecycle used by
/// facade prepared-chat generation on every pipeline rank.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_inkling_mtp_pipeline_neutral_visitor() {
    run_ring_pipeline_mode(false, FixtureFamily::Inkling, WorkerMode::OpaqueInklingMtp);
}

/// Exercises conditional Qwen hybrid MTP over one neutral TP target and extension-only state.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_qwen35_mtp_tensor_parallel_neutral_visitor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35Multimodal,
        "tp",
        WorkerMode::OpaqueQwenHybridMtp,
    );
}

/// Exercises conditional Qwen hybrid MTP over the neutral two-stage target session.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_qwen35_mtp_pipeline_neutral_visitor() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen35Multimodal,
        WorkerMode::OpaqueQwenHybridMtp,
    );
}

/// Exercises composite Qwen hybrid prediction over the admitted Cartesian
/// tensor-by-pipeline target using the same public speculative scheduler.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_four_process_qwen35_mtp_tensor_pipeline_neutral_visitor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35Multimodal,
        "tp-pp",
        WorkerMode::OpaqueQwenHybridMtp,
    );
}

/// Exercises patterned Nemotron-H MTP over one neutral TP target and
/// extension-only state, with no family model/session fallback.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_nemotron_h_mtp_tensor_parallel_neutral_visitor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::NemotronH,
        "tp",
        WorkerMode::OpaqueNemotronHMtp,
    );
}

/// Exercises the architecture-owned Gemma 4 composite TP partition through
/// the public loader and neutral manifest runtime.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic untied Gemma 4 text fixture"]
fn ring_two_process_gemma4_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Exercises the architecture-owned Gemma 4 composite across two pipeline
/// stages with public-session cache continuation and publication.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic untied Gemma 4 text fixture"]
fn ring_two_process_gemma4_pipeline_neutral_composite_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    // This exact PP proof uses independent per-layer KV state. Gemma 4
    // checkpoints whose later layers consume a pass-local shared KV
    // publication still require that typed publication in the neutral wire
    // boundary and therefore remain outside this first production slice.
    write_gemma4_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Exercises Gemma 4 Unified's neutral image and audio towers, ordered media
/// assembly, per-layer inputs, TP decoder, and prompt-cache continuation.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic untied Gemma 4 Unified media fixture"]
fn ring_two_process_gemma4_multimodal_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_multimodal_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueGemma4Media,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Exercises Gemma 4 Unified media ingress, optional roots, merge, and decoder
/// continuation across two neutral pipeline owners.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic untied Gemma 4 Unified media fixture"]
fn ring_two_process_gemma4_multimodal_pipeline_neutral_composite_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_multimodal_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueGemma4Media,
        checkpoint,
        checkpoint_path,
        None,
    );
}

#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic untied Gemma 4 Unified media fixture"]
fn ring_two_process_gemma4_multimodal_inspection_uses_canonical_paths() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_multimodal_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueGemma4MediaInspection,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Compares tied Gemma 4 Unified image/audio prefill and decode with the
/// single-rank resident model while the TP path projects through embeddings.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic tied Gemma 4 Unified media fixture"]
fn ring_two_process_gemma4_tied_multimodal_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_tied_multimodal_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueGemma4Media,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Verifies that a tied Gemma 4 checkpoint without an `lm_head` is bound and
/// projected through the rank-local vocabulary embedding by the public path.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic tied Gemma 4 text fixture"]
fn ring_two_process_gemma4_tied_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_tensor_parallel_fixture_with_tied_embeddings(checkpoint.path(), true);
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Exercises the neutral Muse-Glimmer text binder through the public loader
/// and opaque session on a pure two-rank tensor-parallel topology.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Muse-Glimmer text fixture"]
fn ring_two_process_muse_glimmer_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_muse_glimmer_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::MuseGlimmer,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Exercises Muse-Glimmer's neutral vision tower and media assembly through
/// the public loader on a pure two-rank tensor-parallel topology, including a
/// paged prompt-cache save/open/continue round trip.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Muse-Glimmer image fixture"]
fn ring_two_process_muse_glimmer_image_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_muse_glimmer_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::MuseGlimmer,
        WorkerMode::OpaqueMuseImage,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Exercises Muse-Glimmer's image root and decoder continuation across two
/// neutral pipeline owners without constructing a family pipeline shell.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Muse-Glimmer image fixture"]
fn ring_two_process_muse_glimmer_image_pipeline_neutral_composite_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_muse_glimmer_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::MuseGlimmer,
        WorkerMode::OpaqueMuseImage,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Exercises Muse-Glimmer's canonical vision and decoder unit traversal
/// through bounded checkpoint streaming across two pipeline stages.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_muse_glimmer_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::MuseGlimmer);
}

/// Verifies Inkling's uneven 2+1 stage placement and combined KV/convolution
/// state against the resident text decoder.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Inkling);
}

/// Verifies Inkling TP=2 + PP=2 across full/sliding attention, dense/sparse
/// transitions, rank-local KV/convolution state, persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Inkling, "tp-pp");
}

/// Exercises Inkling TP+PP rank-local recipes and bounded reads from a
/// representative canonical text GGUF.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_gguf_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::InklingGguf, "tp-pp");
}

/// Verifies Inkling PP=2 + EP=2 with stage-local routed experts, shared
/// experts, matching-EP transport, persistence, and bounded layer reads.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Inkling, "pp-ep");
}

/// Exercises Inkling PP+EP expert selection and stage-local state from a
/// representative canonical text GGUF.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_gguf_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::InklingGguf, "pp-ep");
}

/// Proves resident Inkling TP=2 x PP=2 x EP=2 execution across uneven stage
/// placement, full/sliding attention, short-convolution state, and routed/shared
/// expert ownership.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Inkling, "tp-pp-ep");
}

/// Exercises dense-streamed Inkling non-experts and stage-local independent
/// expert caches across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Inkling,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises host-layerwise Inkling attention/convolution state with
/// independently cached routed experts across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Inkling,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises canonical Inkling GGUF recipes, bounded reads, and independent
/// expert caching across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_gguf_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::InklingGguf,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Proves TP+PP can independently cache every stage-local Inkling expert bank
/// without constructing an EP communicator.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_tensor_pipeline_parameter_bank_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Inkling,
        "tp-pp",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies each Inkling stage's expert cache remains session-owned.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Proves bounded affine materialization feeds stage-local Inkling expert
/// caches under PP+EP, including persistence and synchronized decode.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_quantized_pipeline_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Inkling,
        "pp-ep",
        WorkerMode::AddressableParameterBankRequantize,
    );
}

/// Proves a PP+EP stage can bounded-quantize and pin its complete rank-local
/// Inkling routed/shared banks without introducing an independent expert cache.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_fully_resident_load_time_quantization() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "pp-ep",
        WorkerMode::Requantize,
    );
}

/// Runs scheduled Inkling audio/image ingress through TP-sharded placed groups,
/// matching-TP pipeline transport, persistence, decode, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_multimodal_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::InklingMultimodal, "tp-pp");
}

/// Runs scheduled Inkling audio/image ingress through stage-local expert
/// ownership and matching-EP pipeline lanes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_multimodal_pipeline_expert() {
    run_ring_cartesian_pipeline(false, FixtureFamily::InklingMultimodal, "pp-ep");
}

/// Proves scheduled multimodal ingress composes with all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_multimodal_triple_axis() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::InklingMultimodal,
        "tp-pp-ep",
        WorkerMode::OpaqueInklingMedia,
    );
}

/// Proves multimodal ingress composes with streamed non-experts and stage-local
/// independent expert caches across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_multimodal_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::InklingMultimodal,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Proves multimodal ingress composes with host-layerwise non-experts and
/// independent expert caches across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_multimodal_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::InklingMultimodal,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}
