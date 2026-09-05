//! Unified independent-backend production-constructor conformance.
//!
//! `numeric::NumericBackend` and its generic session mechanisms are the shared
//! tensor backend and construction harness for replicated, routed, composite,
//! and partitioned execution. The stage evidence below is emitted inside that
//! harness at the causal boundaries, rather than inferred from a green result.
//!
//! Realtime and speculative execution deliberately add the shape-only
//! `ReferenceBackend` lifecycle mechanisms: realtime requires frame tensor and
//! host-token materializers, while speculative execution requires extension
//! materializers and output sinks. They still enter the same production
//! architecture constructors, but are not represented as NumericBackend
//! executions because those lifecycle-specific mechanism traits are distinct.

use std::collections::BTreeSet;

#[allow(
    dead_code,
    unused_imports,
    reason = "the unified harness compiles only its explicitly shared support paths"
)]
#[path = "reference_numeric.rs"]
mod numeric;
#[allow(
    dead_code,
    unused_imports,
    reason = "the unified harness compiles only its explicitly shared support paths"
)]
#[path = "reference_realtime_construction.rs"]
mod realtime;
#[allow(
    dead_code,
    unused_imports,
    reason = "the unified harness compiles only its explicitly shared support paths"
)]
#[path = "reference_speculative_execution.rs"]
mod speculative;

fn assert_exact_payload_handoff(evidence: &numeric::ReferenceStageEvidence) {
    assert!(evidence.materialization_tasks > 0);
    assert!(evidence.payload_reads.len() >= evidence.materialization_tasks);
    assert!(evidence
        .payload_reads
        .iter()
        .all(|read| read.physically_bounded && read.encoded_bytes > 0));
    assert!(evidence
        .payload_reads
        .iter()
        .all(|read| !read.task.is_empty()
            && !read.source.is_empty()
            && !read.output_shape.is_empty()));
}

fn conformance_replicated_families_formats_and_materialization() {
    let safetensors = numeric::run_reference_conformance_replicated_safetensors_families();
    let expected_families = BTreeSet::from([
        "deepseek_v3",
        "kimi_linear",
        "lfm2",
        "llama",
        "nemotron_h",
        "qwen2",
        "qwen3",
        "qwen3_5_text",
        "qwen3_next",
    ]);
    assert_eq!(
        safetensors
            .iter()
            .map(|evidence| evidence.family.as_str())
            .collect::<BTreeSet<_>>(),
        expected_families
    );
    for evidence in &safetensors {
        assert_eq!(evidence.format, "SafeTensors");
        assert_exact_payload_handoff(evidence);
        assert_eq!(
            evidence.stages,
            [
                "construction_started",
                "typed_architecture",
                "materialization",
                "session_constructed",
                "prefill",
                "decode",
                "decode",
                "report",
            ]
        );
    }
    let gguf = numeric::run_reference_conformance_gguf_replicated_production();
    assert_eq!(gguf.family, "llama");
    assert_eq!(gguf.format, "Gguf");
    assert_exact_payload_handoff(&gguf);
    let transformed = numeric::run_reference_conformance_transformed_replicated_production();
    assert_exact_payload_handoff(&transformed);
    assert!(transformed
        .lowering_kinds
        .iter()
        .any(|kind| kind == "Transform"));
    assert!(transformed.generated_companions > 0);
    realtime::load_time_affine_transform_reaches_typed_construction_and_frame_execution();
}

fn conformance_routed_resident_and_addressable_production() {
    assert_eq!(
        numeric::run_reference_conformance_routed_safetensors_families()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "deepseek_v3".to_owned(),
            "deepseek_v4".to_owned(),
            "gpt_oss".to_owned(),
            "kimi_linear".to_owned(),
            "lfm2_moe".to_owned(),
            "nemotron_h".to_owned(),
            "qwen3_moe".to_owned(),
        ])
    );
    numeric::non_mlx_session_executes_resident_and_addressable_routing_through_one_driver();
    numeric::non_mlx_session_executes_relu2_routing_through_the_same_driver();
    let transformed = numeric::run_reference_conformance_transformed_addressable_route();
    assert_exact_payload_handoff(&transformed);
    assert_eq!(transformed.family, "qwen3_moe");
    assert!(transformed
        .lowering_kinds
        .iter()
        .any(|kind| kind == "Transform"));
    assert!(transformed.generated_companions > 0);
}

fn conformance_composite_production_and_family_dispatch() {
    assert_eq!(
        numeric::run_reference_conformance_composite_families()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "gemma4",
            "inkling",
            "muse_glimmer",
            "qwen3_5",
            "qwen3_vl",
            "qwen3_vl_moe",
        ])
    );
    numeric::routed_muse_and_inkling_use_the_ordinary_partition_session_across_admitted_axes();
    numeric::placed_routed_composite_observation_intervention_is_transactional();
}

fn conformance_partitioned_production() {
    numeric::authoritative_partitioned_numeric_sessions_match_tp_pp_and_tp_pp_reference();
    numeric::prediction_free_deepseek_v3_v4_routed_sessions_cover_cartesian_state_cache_and_failure(
    );
}

fn conformance_typed_extension_without_backend_family_dispatch() {
    numeric::new_architecture_uses_production_selection_materialization_and_session_without_backend_case();
}

fn conformance_state_rollback_completion_and_failure_timing() {
    numeric::compressed_cache_growth_boundaries_and_rollback_are_backend_neutral();
    numeric::dense_stream_acquisition_failure_is_atomic_before_first_real_unit();
    realtime::realtime_observer_failure_discards_the_caller_owned_transaction();
}

fn conformance_speculative_production() {
    speculative::sequential_embedded_runs_the_inspected_materialized_scheduler_path();
    speculative::fused_dspark_runs_the_inspected_materialized_scheduler_path();
    speculative::gemma_runs_the_inspected_materialized_scheduler_path();
    speculative::released_muse_dflash_runs_the_inspected_materialized_scheduler_path();
    speculative::exact_completion_retains_resources_and_failure_rolls_back_before_publication();
}

fn conformance_realtime_production() {
    realtime::native_moshi_resident_and_bounded_use_the_same_reference_frame_scheduler();
    realtime::personaplex_uses_released_inspection_and_the_common_reference_scheduler();
}

fn main() {
    let cases: [(&str, fn()); 8] = [
        (
            "replicated_families_formats_and_materialization",
            conformance_replicated_families_formats_and_materialization,
        ),
        (
            "routed_resident_and_addressable_production",
            conformance_routed_resident_and_addressable_production,
        ),
        (
            "composite_production_and_family_dispatch",
            conformance_composite_production_and_family_dispatch,
        ),
        ("partitioned_production", conformance_partitioned_production),
        (
            "typed_extension_without_backend_family_dispatch",
            conformance_typed_extension_without_backend_family_dispatch,
        ),
        (
            "state_rollback_completion_and_failure_timing",
            conformance_state_rollback_completion_and_failure_timing,
        ),
        ("speculative_production", conformance_speculative_production),
        ("realtime_production", conformance_realtime_production),
    ];
    let selected_case = std::env::var("EREDU_REFERENCE_CASE").ok();
    let mut failures = Vec::new();
    println!("running {} reference conformance cases", cases.len());
    for (name, case) in cases {
        if selected_case
            .as_deref()
            .is_some_and(|selected| selected != name)
        {
            continue;
        }
        match std::panic::catch_unwind(case) {
            Ok(()) => println!("case {name} ... ok"),
            Err(_) => {
                println!("case {name} ... FAILED");
                failures.push(name);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "reference conformance failures: {}",
        failures.join(", ")
    );
}
