use super::*;
use eredu_core::{
    AvailableMemory, EstimationCompleteness, LayerSchedule, ObservationKind, StateMemoryLayout,
    StaticMemoryReport,
};
use eredu_core::{Observed, PhysicalMemorySemantics};
use eredu_runtime::memory_estimation::WorkspaceGeometry;
use eredu_runtime::memory_forecast::LoadedMemoryGeometry;
use std::num::NonZeroU8;

fn profile(cached: Observed<u64>, resident: u64) -> LoadedMemoryProfile {
    LoadedMemoryProfile {
        geometry: LoadedMemoryGeometry {
            execution_topology: None,
            input_score_attention_mechanism: None,
            state_layout: StateMemoryLayout::new(
                LayerSchedule::new(
                    2,
                    vec![
                        eredu_core::cache::LayerCachePolicy::key_only(
                            eredu_core::AttentionPolicy::Full,
                            1,
                            4
                        )
                        .unwrap();
                        2
                    ],
                )
                .unwrap(),
                vec![0; 2],
                16,
                4,
                EstimationCompleteness::Complete,
            )
            .unwrap(),
            workspace: Some(WorkspaceGeometry {
                hidden_size: 16,
                intermediate_size: 32,
                query_width: 16,
                key_value_width: 4,
                query_heads: 4,
                vocabulary_size: 32,
                gated_convolution: None,
                input_score_attention: None,
                mixed_precision_parameter_bytes: Some(1024),
            }),
            scalar_bytes: NonZeroU8::new(2).unwrap(),
            fully_resident: true,
            assumptions: vec![],
        },
        parameters: StaticMemoryReport {
            logical_parameter_bytes: Observed::exact(1024, "original parameters"),
            current_host_resident_bytes: Observed::exact(0, "device only"),
            current_device_resident_bytes: Observed::exact(
                resident,
                "originals plus retained conversions",
            ),
            current_device_parameter_conversion_bytes: cached,
            planned_disk_backed_bytes: Observed::exact(0, "resident"),
            backend_active_allocation_bytes: Observed::exact(999999, "unrelated global activity"),
            backend_allocator_cache_bytes: Observed::exact(0, "disabled"),
            physical_semantics: PhysicalMemorySemantics::Unified,
            currently_cached_shards: Observed::exact(0, "none"),
        },
        available: AvailableMemory {
            physical_memory_bytes: Observed::exact(1 << 30, "fixture"),
            available_memory_bytes: Observed::exact(1 << 30, "fixture"),
            physical_semantics: PhysicalMemorySemantics::Unified,
        },
        host_execution: false,
        allocator_cache_limit: Observed::exact(0, "disabled"),
    }
}

fn forecast(profile: LoadedMemoryProfile) -> Result<GenerationForecast, GenerationForecastError> {
    loaded_forecast(
        profile,
        InputTokenCount::text(17),
        3,
        eredu_core::PrefillChunkPolicy::Unchunked,
        ForecastExecutionContract {
            full_pass_reason: None,
            logits: LogitsWorkspace::FinalPosition,
        },
        &GenerationForecastOptions::default(),
    )
}

#[test]
fn resident_parameter_conversions_are_counted_once_and_only_pending_casts_are_workspace() {
    let cold = forecast(profile(Observed::exact(0, "empty"), 1024)).unwrap();
    for retained in [256, 1024] {
        let warm = forecast(profile(
            Observed::exact(retained, "owned conversion"),
            1024 + retained,
        ))
        .unwrap();
        let domain = &warm.request.domains[0];
        assert_eq!(domain.resident_parameters.lower_bytes, 1024 + retained);
        assert_eq!(
            domain.resident_parameters.upper_bytes,
            Some(1024 + retained)
        );
        assert_eq!(domain.already_resident_bytes, 1024 + retained);
        assert_eq!(
            domain.executions[0]
                .workspace
                .as_ref()
                .unwrap()
                .mixed_precision_parameter_bytes,
            Some(1024 - retained)
        );
        // Two layers plus one overlap copy: pending cast scratch falls by 1.5x
        // the reused payload. The resident conversion itself is not charged twice.
        assert_eq!(
            cold.estimate.domains[0]
                .additional_generation_peak
                .upper_bytes
                .unwrap()
                - warm.estimate.domains[0]
                    .additional_generation_peak
                    .upper_bytes
                    .unwrap(),
            retained + retained / 2,
        );
        assert_eq!(
            warm.estimate.domains[0].phases[0].persistent_state,
            cold.estimate.domains[0].phases[0].persistent_state
        );
    }
}

#[test]
fn unavailable_conversion_observation_keeps_scratch_and_invalid_subset_is_rejected() {
    let unknown = forecast(profile(Observed::unavailable("not observed"), 1280)).unwrap();
    assert_eq!(
        unknown.request.domains[0].executions[0]
            .workspace
            .as_ref()
            .unwrap()
            .mixed_precision_parameter_bytes,
        Some(1024)
    );
    assert_eq!(
        unknown.request.domains[0].resident_parameters.lower_bytes,
        1280
    );
    assert!(forecast(profile(Observed::exact(2048, "invalid subset"), 1024)).is_err());
    // The counter is independent of observation quality of process allocations.
    let mut host = profile(Observed::exact(256, "owned conversion"), 1280);
    host.parameters.backend_active_allocation_bytes = Observed::Available {
        value: u64::MAX,
        kind: ObservationKind::Observational,
        source: "unrelated".into(),
    };
    host.host_execution = true;
    host.parameters.physical_semantics = PhysicalMemorySemantics::SeparateTiers;
    let host = forecast(host).unwrap();
    assert_eq!(host.request.domains[0].domain, MemoryDomain::Host);
    assert_eq!(host.request.domains[0].already_resident_bytes, 1280);
}

#[test]
fn generic_topology_credits_resident_conversions_without_losing_promotion() {
    use eredu_runtime::execution_topology::*;
    let generic_profile = |retained| {
        let mut profile = profile(
            Observed::exact(retained, "owned conversion"),
            1024 + retained,
        );
        let projection = |input, output| ProjectionTopology {
            input,
            output,
            format: eredu_checkpoint::LinearFormat::Dense,
            bias: false,
            parameter: format!("projection-{input}-{output}"),
        };
        profile.geometry.workspace = None;
        profile.geometry.execution_topology = Some(TextExecutionTopology {
            hidden_size: 16,
            vocabulary_size: 32,
            output: projection(16, 32),
            output_invocations: 1,
            output_softcap: false,
            selected_parameter_promotion_bytes: Some(1024),
            selected_parameter_promotion_payloads: Default::default(),
            missing: vec![],
            layers: vec![
                TextLayerTopology {
                    input_projections: vec![],
                    mixer: TokenMixerTopology::Attention {
                        output_gate: false,
                        query_heads: 4,
                        kv_heads: 1,
                        key_width: 4,
                        value_width: 4,
                        input_scores: false,
                        softcap: false,
                        sinks: false,
                        projections: vec![
                            projection(16, 16),
                            projection(16, 4),
                            projection(16, 4),
                            projection(16, 16)
                        ],
                        query_key_normalization: false,
                        rotary: true,
                    },
                    feed_forward: FeedForwardTopology::Gated {
                        intermediate_size: 32,
                        projections: vec![
                            projection(16, 32),
                            projection(16, 32),
                            projection(32, 16)
                        ],
                    },
                    normalization_count: 2,
                };
                2
            ],
        });
        profile
    };
    let cold = forecast(generic_profile(0)).unwrap();
    for retained in [256, 1024] {
        let warm = forecast(generic_profile(retained)).unwrap();
        let domain = &warm.request.domains[0];
        assert_eq!(domain.already_resident_bytes, 1024 + retained);
        assert_eq!(
            domain.executions[0]
                .execution_topology
                .as_ref()
                .unwrap()
                .selected_parameter_promotion_bytes,
            Some(1024 - retained)
        );
        assert_eq!(
            cold.estimate.domains[0]
                .additional_generation_peak
                .upper_bytes
                .unwrap()
                - warm.estimate.domains[0]
                    .additional_generation_peak
                    .upper_bytes
                    .unwrap(),
            retained + retained / 2
        );
    }
}
