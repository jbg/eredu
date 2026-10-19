use super::*;
use eredu_runtime::memory_forecast::{
    ConditionedAssistantMemoryProfile, ConditionedPredictionBody, ConditionedPredictionScoring,
    ConditionedPredictionTopology, SpeculativeForecastBackend, SpeculativeMemoryProfile,
};

struct ConditionedFixture {
    scratch: MemoryBytes,
}

impl SpeculativeForecastBackend<ConditionedFixture> for MockBackend {
    fn speculative_memory_profile(
        _: &ModelRuntime<Self>,
        drafting: &eredu_core::SpeculativeDraft<'_, ConditionedFixture>,
    ) -> Result<Option<SpeculativeMemoryProfile>, GenerationForecastError> {
        let eredu_core::SpeculativeDraft::External(assistant) = drafting else {
            panic!("external fixture required")
        };
        Ok(Some(SpeculativeMemoryProfile {
            conditioned: Some(fixture_conditioned_profile(assistant.scratch.clone())),
            refinement: None,
            shared_backbone: None,
            embedded: None,
            draft: None,
            parameter_conversions: None,
            auxiliary_bytes_per_position: MemoryBytes::exact(0),
            sampling_bytes_per_vocabulary_entry: MemoryBytes::exact(128),
            proposal_capacity: 4,
            shared_allocator: true,
        }))
    }
}

fn fixture_conditioned_profile(scratch: MemoryBytes) -> ConditionedAssistantMemoryProfile {
    let mut base = fixture_profile();
    base.parameters.parameter_conversion_retention =
        Observed::exact(vec![], "fixture has no conversion owners");
    ConditionedAssistantMemoryProfile {
        topology: ConditionedPredictionTopology {
            body: ConditionedPredictionBody {
                hidden: 8,
                captures: 2,
                layers: 2,
                query_heads: 2,
                head_width: 4,
                key_value_width: 4,
                intermediate: 16,
                context_window: 3,
                block_positions: 8,
                convolution: None,
            },
            scoring: ConditionedPredictionScoring::CandidateSelector {
                vocabulary: 128,
                rank: 3,
                candidates: 5,
            },
        },
        parameters: eredu_core::speculative::SpeculativeParameterMemoryObservation {
            parameters: base.parameters,
            parameter_conversions: Some(vec![]),
            available: base.available,
            allocator_cache_limit: base.allocator_cache_limit,
        },
        residency: eredu_runtime::LayerWeightResidency::FullyResident,
        parameter_bounds: None,
        parameter_loading: Some(
            eredu_runtime::memory_forecast::ParameterLoadingMemoryProfile {
                source_payload: MemoryBytes::exact(0),
                host_workspace: MemoryBytes::exact(0),
                execution_workspace: MemoryBytes::exact(0),
            },
        ),
        host_execution: false,
        native_workspace: scratch,
    }
}

#[test]
fn conditioned_facade_forecast_preserves_both_owners_and_fixed_block_on_recompute() {
    let (model, _, settings) = setup();
    let mut assistant = ConditionedFixture {
        scratch: MemoryBytes::exact(0),
    };
    let settings = eredu::api::PreparedChatGenerationSettings {
        decoding_mode: Some(eredu_core::decoding::DecodingMode::SelfSpeculation(
            Default::default(),
        )),
        ..settings
    };
    let forecast = model
        .forecast_speculative_token_ids(
            &[1; 17],
            settings,
            &eredu_core::SpeculativeDraft::External(&mut assistant),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
    let plan = forecast.speculative.as_ref().unwrap();
    assert_eq!(
        plan.conditioned
            .as_ref()
            .unwrap()
            .topology
            .body
            .block_positions,
        8
    );
    assert!(plan
        .draft
        .as_ref()
        .unwrap()
        .domains
        .iter()
        .all(|d| d.executions.is_empty()));
    for phase in forecast.estimate.domains[0]
        .phases
        .iter()
        .filter(|p| p.phase != eredu::api::MemoryPhase::Loading)
    {
        assert_eq!(phase.parameters.lower_bytes, 8192);
        assert_eq!(phase.parameters.upper_bytes, Some(8192));
    }
    let shorter = forecast.with_max_output_tokens(2).unwrap();
    assert_eq!(
        shorter.speculative.as_ref().unwrap().conditioned,
        plan.conditioned
    );
    assert_eq!(
        shorter
            .speculative
            .as_ref()
            .unwrap()
            .draft
            .as_ref()
            .unwrap()
            .max_output_tokens,
        Some(2)
    );
    let encoded = serde_json::to_value(&shorter).unwrap();
    let decoded: eredu::api::GenerationForecast = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded.speculative, shorter.speculative);
    assistant.scratch = MemoryBytes::unknown("fixture unobserved native scratch");
    let unknown = model
        .forecast_speculative_token_ids(
            &[1; 17],
            settings,
            &eredu_core::SpeculativeDraft::External(&mut assistant),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(unknown.estimate.fit, MemoryFit::InsufficientInformation);
    assert!(unknown
        .estimate
        .uncertainties
        .iter()
        .any(|s| s == "fixture unobserved native scratch"));
}

#[test]
fn conditioned_parameter_pools_use_capacity_and_cache_limits_without_hiding_unknowns() {
    use eredu_core::{residency::OffloadConfig, PhysicalMemorySemantics};
    use eredu_runtime::memory_forecast::ParameterResidencyMemoryBounds;
    use eredu_runtime::{LayerWeightResidency, LayerwiseLoadOptions};

    let residency = LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
        OffloadConfig::new(Some(250), Some(1000), 1).unwrap(),
    ));
    let bounds = ParameterResidencyMemoryBounds::from_unit_capacities(
        [(100, 128), (200, 256), (300, 512)],
        residency,
    )
    .unwrap();
    // The host buffer capacities total 896, not the 600-byte payload. Device
    // cache occupancy can reach its 250-byte budget despite a one-unit window.
    assert_eq!(bounds.host_capacity_bytes(), 896);
    assert_eq!(bounds.device_payload_bytes(), 250);
    let make = |semantics, host_execution, observed_host| {
        let mut profile = fixture_conditioned_profile(MemoryBytes::unknown("native scratch"));
        profile.residency = residency;
        profile.parameter_bounds = Some(bounds.clone());
        profile.parameter_loading = None;
        profile.host_execution = host_execution;
        let p = &mut profile.parameters.parameters;
        p.physical_semantics = semantics;
        p.logical_parameter_bytes = Observed::exact(600, "fixture unit payloads");
        p.current_host_resident_bytes = Observed::exact(observed_host, "fixture managed host");
        p.current_device_resident_bytes = Observed::exact(116, "100 parameters plus 16 converted");
        p.current_device_parameter_conversion_bytes = Observed::exact(16, "fixture conversion");
        profile
    };
    let prepare = |profile: ConditionedAssistantMemoryProfile| {
        profile.into_request(eredu_core::InputTokenCount::text(1), 3, &Default::default())
    };
    let (unified, _) = prepare(make(PhysicalMemorySemantics::Unified, false, 100)).unwrap();
    assert_eq!(unified.domains.len(), 1);
    assert_eq!(unified.domains[0].resident_parameters.lower_bytes, 116);
    assert_eq!(
        unified.domains[0].resident_parameters.upper_bytes,
        Some(1162)
    );
    assert!(
        unified.domains[0].staging.upper_bytes.is_none(),
        "managed pool limits do not bound transfers or source caches"
    );
    let (separate, _) = prepare(make(PhysicalMemorySemantics::SeparateTiers, false, 100)).unwrap();
    assert_eq!(
        separate.domains[0].resident_parameters.upper_bytes,
        Some(896)
    );
    assert_eq!(
        separate.domains[1].resident_parameters.upper_bytes,
        Some(266)
    );
    let (host, _) = prepare(make(PhysicalMemorySemantics::SeparateTiers, true, 100)).unwrap();
    assert_eq!(host.domains.len(), 1);
    assert_eq!(host.domains[0].domain, eredu::api::MemoryDomain::Host);
    assert_eq!(host.domains[0].resident_parameters.upper_bytes, Some(1162));
    let mut missing = make(PhysicalMemorySemantics::Unified, false, 100);
    missing.parameter_bounds = None;
    assert!(prepare(missing).unwrap().0.domains[0]
        .resident_parameters
        .upper_bytes
        .is_none());
    // Source backing belongs to the host pool; native intermediates belong to
    // execution. Unified and host execution combine both contributions once.
    for (semantics, host_execution, expected) in [
        (
            PhysicalMemorySemantics::SeparateTiers,
            false,
            vec![170, 131],
        ),
        (PhysicalMemorySemantics::Unified, false, vec![301]),
        (PhysicalMemorySemantics::SeparateTiers, true, vec![301]),
    ] {
        let mut profile = make(semantics, host_execution, 100);
        profile.parameter_loading = Some(
            eredu_runtime::memory_forecast::ParameterLoadingMemoryProfile {
                source_payload: MemoryBytes::estimated(0, 71, "retained source backing"),
                host_workspace: MemoryBytes::estimated(0, 99, "host transfer graph"),
                execution_workspace: MemoryBytes::estimated(0, 131, "native transfer graph"),
            },
        );
        let (request, _) = prepare(profile).unwrap();
        assert_eq!(
            request
                .domains
                .iter()
                .map(|d| d.staging.upper_bytes.unwrap())
                .collect::<Vec<_>>(),
            expected,
        );
    }
    // A producer cannot make an under-bound look valid by clamping it upward.
    assert!(prepare(make(PhysicalMemorySemantics::SeparateTiers, false, 897)).is_err());
    let mut wrong_policy = make(PhysicalMemorySemantics::Unified, false, 100);
    wrong_policy.residency = LayerWeightResidency::FullyResident;
    assert!(prepare(wrong_policy).is_err());

    let direct = LayerWeightResidency::DenseDiskStream(
        eredu_runtime::DenseDiskStreamLoadOptions::new(400, 0, 0, 0).unwrap(),
    );
    let direct = ParameterResidencyMemoryBounds::from_unit_capacities(
        [(100, 128), (200, 256), (300, 512)],
        direct,
    )
    .unwrap();
    assert_eq!(direct.host_capacity_bytes(), 0);
    assert_eq!(direct.device_payload_bytes(), 400);
    assert!(ParameterResidencyMemoryBounds::from_unit_capacities([(2, 1)], residency).is_err());
    assert!(ParameterResidencyMemoryBounds::from_unit_capacities(
        [(u64::MAX, u64::MAX), (1, 1)],
        residency
    )
    .is_err());
}
