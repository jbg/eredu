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

/// Public capture, component masks and branch replay across the MLA pipeline cut.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v3_pipeline_components() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueComponentCapture,
    );
    run_ring_pipeline_mode(
        true,
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueComponentCapture,
    );
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::DeepSeek,
        "pp",
        WorkerMode::OpaqueComponentCapture,
    );
}

/// Dense MLA uses direct construction and ordinary TP reductions at every residency.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_dense_partition_components() {
    run_deepseek_v3_partition_components(
        FixtureFamily::DeepSeekDense,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mixed_partition_components() {
    run_deepseek_v3_partition_components(
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mixed_gguf_partition_components() {
    run_deepseek_v3_partition_components(
        FixtureFamily::DeepSeekGguf,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mixed_affine_partition_components() {
    run_deepseek_v3_partition_components(
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueComponentCaptureRequantize,
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_dense_gguf_partition_components() {
    run_deepseek_v3_partition_components(
        FixtureFamily::DeepSeekDenseGguf,
        WorkerMode::OpaqueComponentCapture,
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_dense_affine_partition_components() {
    run_deepseek_v3_partition_components(
        FixtureFamily::DeepSeekDense,
        WorkerMode::OpaqueComponentCaptureRequantize,
    );
}

fn run_deepseek_v3_partition_components(family: FixtureFamily, mode: WorkerMode) {
    run_deepseek_v3_component_axes(family, mode, &["tp", "pp", "tp-pp"]);
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mixed_expert_partition_components() {
    run_deepseek_v3_component_axes(
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueComponentCapture,
        &["ep", "tp-ep", "pp-ep", "tp-pp-ep"],
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mixed_expert_gguf_partition_components() {
    run_deepseek_v3_component_axes(
        FixtureFamily::DeepSeekGguf,
        WorkerMode::OpaqueComponentCapture,
        &["ep", "tp-ep", "pp-ep", "tp-pp-ep"],
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mixed_expert_affine_partition_components() {
    run_deepseek_v3_component_axes(
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueComponentCaptureRequantize,
        &["ep", "tp-ep", "pp-ep", "tp-pp-ep"],
    );
}

/// Independently cached experts must retain every public component lifecycle.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mixed_cached_partition_components() {
    run_deepseek_v3_component_axes(
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
        &["tp", "pp", "tp-pp", "ep", "tp-ep", "pp-ep", "tp-pp-ep"],
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mixed_cached_gguf_partition_components() {
    run_deepseek_v3_component_axes(
        FixtureFamily::DeepSeekGguf,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
        &["tp", "pp", "tp-pp", "ep", "tp-ep", "pp-ep", "tp-pp-ep"],
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mixed_cached_affine_partition_components() {
    run_deepseek_v3_component_axes(
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueComponentCaptureAddressableBankRequantize,
        &["tp", "pp", "tp-pp", "ep", "tp-ep", "pp-ep", "tp-pp-ep"],
    );
}

fn run_deepseek_v3_component_axes(family: FixtureFamily, mode: WorkerMode, axes: &[&'static str]) {
    for &axes in axes {
        for streamed in [false, true] {
            eprintln!(
                "V3 component case: {} {axes}, streamed={streamed}",
                family.name()
            );
            run_ring_cartesian_pipeline_mode(streamed, family, axes, mode);
        }
        eprintln!(
            "V3 component case: {} {axes}, layerwise host",
            family.name()
        );
        run_ring_layerwise_host_cartesian_pipeline_mode(family, axes, mode);
    }
}

/// Public internal activation admission compares each selected partition with
/// ordinary local execution, including sparse units and causal channel masks.
#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mtp_component_capture_residency_and_parallel_matrix() {
    run_deepseek_prediction_components(
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueDeepSeekMtpTarget,
    );
}

#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mtp_affine_components_and_parameters_matrix() {
    run_deepseek_prediction_components(
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueDeepSeekMtpTargetRequantize,
    );
}

#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v3_mtp_mxfp4_components_and_parameters_matrix() {
    run_deepseek_prediction_components(
        FixtureFamily::DeepSeek,
        WorkerMode::OpaqueDeepSeekMtpTargetMxFp4,
    );
}

#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v4_mtp_components_and_parameters_matrix() {
    run_deepseek_prediction_components(
        FixtureFamily::DeepSeekV4,
        WorkerMode::OpaqueDeepSeekMtpTarget,
    );
}

#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v4_mtp_affine_components_and_parameters_matrix() {
    run_deepseek_prediction_components(
        FixtureFamily::DeepSeekV4,
        WorkerMode::OpaqueDeepSeekMtpTargetRequantize,
    );
}

#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v4_mtp_mxfp4_components_and_parameters_matrix() {
    run_deepseek_prediction_components(
        FixtureFamily::DeepSeekV4,
        WorkerMode::OpaqueDeepSeekMtpTargetMxFp4,
    );
}

#[test]
#[ignore = "spawns two local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_mtp_components_pipeline() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "pp",
        WorkerMode::OpaqueDeepSeekMtpTarget,
    );
}

#[test]
#[ignore = "spawns two local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_mtp_affine_components_host() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::DeepSeekV4,
        "tp",
        WorkerMode::OpaqueDeepSeekMtpTargetRequantize,
    );
}

#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v4_target_components_and_parameters_matrix() {
    run_deepseek_v4_target_components(WorkerMode::OpaqueComponentCapture);
}

#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v4_target_affine_components_and_parameters_matrix() {
    run_deepseek_v4_target_components(WorkerMode::OpaqueComponentCaptureRequantize);
}

#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v4_target_mxfp4_components_and_parameters_matrix() {
    run_deepseek_v4_target_components(WorkerMode::OpaqueComponentCaptureMxFp4);
}

fn run_deepseek_v4_target_components(mode: WorkerMode) {
    for axes in ["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"] {
        for residency in ["resident", "host", "disk"] {
            eprintln!("V4 target component and parameter trial: {mode:?} {axes} {residency}");
            if residency == "host" {
                run_ring_layerwise_host_cartesian_pipeline_mode(
                    FixtureFamily::DeepSeekV4,
                    axes,
                    mode,
                );
            } else {
                run_ring_cartesian_pipeline_mode(
                    residency == "disk",
                    FixtureFamily::DeepSeekV4,
                    axes,
                    mode,
                );
            }
        }
    }
}

fn run_deepseek_prediction_components(family: FixtureFamily, mode: WorkerMode) {
    for axes in ["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"] {
        for residency in ["resident", "host", "disk"] {
            eprintln!(
                "{} prediction components and parameters: {mode:?} {axes} {residency}",
                family.name()
            );
            if residency == "host" {
                run_ring_layerwise_host_cartesian_pipeline_mode(family, axes, mode);
            } else {
                run_ring_cartesian_pipeline_mode(residency == "disk", family, axes, mode);
            }
        }
    }
}

#[test]
#[ignore = "spawns two local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v3_mtp_target_pipeline_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "pp",
        WorkerMode::OpaqueDeepSeekMtpTarget,
    );
}

/// Multi-block fused context/proposal captures, selected-position interventions,
/// effective parameter queries and atomic overlays across all native CPU axes.
#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v4_dspark_components_and_parameters_matrix() {
    run_deepseek_prediction_components(
        FixtureFamily::DeepSeekV4,
        WorkerMode::OpaqueDeepSeekDsparkTarget,
    );
}

#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v4_dspark_affine_components_and_parameters_matrix() {
    run_deepseek_prediction_components(
        FixtureFamily::DeepSeekV4,
        WorkerMode::OpaqueDeepSeekDsparkTargetRequantize,
    );
}

#[test]
#[ignore = "spawns two to eight local processes and opens loopback sockets; run explicitly"]
fn ring_deepseek_v4_dspark_mxfp4_components_and_parameters_matrix() {
    run_deepseek_prediction_components(
        FixtureFamily::DeepSeekV4,
        WorkerMode::OpaqueDeepSeekDsparkTargetMxFp4,
    );
}

fn run_v4_fp8_component_case(
    dspark: bool,
    ue8m0: bool,
    axes: &'static str,
    residency: WorkerResidency,
) {
    let checkpoint = tempfile::tempdir().unwrap();
    write_deepseek_v4_fp8_fixture(checkpoint.path(), dspark, ue8m0);
    let path = checkpoint.path().to_owned();
    eprintln!("V4 block-FP8 dspark={dspark} ue8m0={ue8m0} {axes} {residency:?}");
    run_ring_pipeline_processes(
        residency,
        FixtureFamily::DeepSeekV4,
        if dspark {
            WorkerMode::OpaqueDeepSeekDsparkTarget
        } else {
            WorkerMode::OpaqueDeepSeekMtpTarget
        },
        checkpoint,
        path,
        Some(axes),
    );
}

#[test]
#[ignore = "requires native MLX and local Ring ranks; run explicitly"]
fn ring_deepseek_v4_fp8_components_focused() {
    for dspark in [false, true] {
        for ue8m0 in [false, true] {
            run_v4_fp8_component_case(dspark, ue8m0, "tp", WorkerResidency::FullyResident);
        }
    }
}

macro_rules! v4_fp8_component_matrix {
    ($name:ident, $dspark:expr, $ue8m0:expr) => {
        #[test]
        #[ignore = "requires native MLX and up to eight local Ring ranks; run explicitly"]
        fn $name() {
            for axes in ["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"] {
                for residency in [
                    WorkerResidency::FullyResident,
                    WorkerResidency::LayerwiseHost,
                    WorkerResidency::DenseDiskStream,
                ] {
                    run_v4_fp8_component_case($dspark, $ue8m0, axes, residency);
                }
            }
        }
    };
}
v4_fp8_component_matrix!(
    ring_deepseek_v4_mtp_fp8_float_components_matrix,
    false,
    false
);
v4_fp8_component_matrix!(
    ring_deepseek_v4_mtp_fp8_ue8m0_components_matrix,
    false,
    true
);
v4_fp8_component_matrix!(
    ring_deepseek_v4_dspark_fp8_float_components_matrix,
    true,
    false
);
v4_fp8_component_matrix!(
    ring_deepseek_v4_dspark_fp8_ue8m0_components_matrix,
    true,
    true
);

fn run_deepseek_additional_fp8_case(
    v3: bool,
    dspark: bool,
    axes: &'static str,
    residency: WorkerResidency,
) {
    let checkpoint = tempfile::tempdir().unwrap();
    if v3 {
        write_deepseek_v3_fp8_fixture(checkpoint.path());
    } else {
        write_deepseek_v4_fp8_fixture_kind(checkpoint.path(), dspark, true, true);
    }
    let path = checkpoint.path().to_owned();
    eprintln!(
        "DeepSeek published encoding v3={v3} mixed_v4={} dspark={dspark} {axes} {residency:?}",
        !v3
    );
    run_ring_pipeline_processes(
        residency,
        if v3 {
            FixtureFamily::DeepSeek
        } else {
            FixtureFamily::DeepSeekV4
        },
        if dspark {
            WorkerMode::OpaqueDeepSeekDsparkTarget
        } else {
            WorkerMode::OpaqueDeepSeekMtpTarget
        },
        checkpoint,
        path,
        Some(axes),
    );
}

#[test]
#[ignore = "requires native MLX and local Ring ranks; run explicitly"]
fn ring_deepseek_additional_fp8_components_focused() {
    for (v3, dspark) in [(true, false), (false, false), (false, true)] {
        run_deepseek_additional_fp8_case(v3, dspark, "tp", WorkerResidency::FullyResident);
    }
}

macro_rules! additional_deepseek_fp8_matrix {
    ($name:ident, $v3:expr, $dspark:expr) => {
        #[test]
        #[ignore = "requires native MLX and up to eight local Ring ranks; run explicitly"]
        fn $name() {
            for axes in ["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"] {
                for residency in [
                    WorkerResidency::FullyResident,
                    WorkerResidency::LayerwiseHost,
                    WorkerResidency::DenseDiskStream,
                ] {
                    run_deepseek_additional_fp8_case($v3, $dspark, axes, residency);
                }
            }
        }
    };
}
additional_deepseek_fp8_matrix!(ring_deepseek_v3_fp8_components_matrix, true, false);
additional_deepseek_fp8_matrix!(
    ring_deepseek_v4_mtp_mixed_fp8_components_matrix,
    false,
    false
);
additional_deepseek_fp8_matrix!(
    ring_deepseek_v4_dspark_mixed_fp8_components_matrix,
    false,
    true
);

/// Exercise the shared stream/readout oracle across its ordinary and fused callers.
#[test]
#[ignore = "requires native MLX and local Ring ranks; run explicitly"]
fn ring_deepseek_v4_reconstruction_regressions() {
    for mode in [
        WorkerMode::OpaqueComponentCapture,
        WorkerMode::OpaqueComponentCaptureRequantize,
        WorkerMode::OpaqueComponentCaptureMxFp4,
        WorkerMode::OpaqueDeepSeekDsparkTarget,
        WorkerMode::OpaqueDeepSeekDsparkTargetRequantize,
        WorkerMode::OpaqueDeepSeekDsparkTargetMxFp4,
    ] {
        for (axes, disk) in [("tp", false), ("pp", true)] {
            eprintln!("V4 stream reconstruction regression {mode:?} {axes} disk={disk}");
            run_ring_cartesian_pipeline_mode(disk, FixtureFamily::DeepSeekV4, axes, mode);
        }
    }
}
