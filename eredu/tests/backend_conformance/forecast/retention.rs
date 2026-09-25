use super::*;
use eredu_core::{residency::*, resources::*};
use eredu_runtime::execution_topology::{
    FeedForwardTopology, ProjectionTopology, TextExecutionTopology, TextLayerTopology,
    TokenMixerTopology,
};

/// Observe the fixture's actual claims, including their stable backing identities.
pub(super) fn profile(runtime: &ModelRuntime<MockBackend>) -> LoadedMemoryProfile {
    let mut profile = fixture_profile();
    let report = runtime.session().retention.report();
    let owner = ResourceIdentity {
        scope: report.group.0.scope.clone(),
        key: "parameters".into(),
    };
    let conversions = runtime
        .session()
        .retention_claims
        .lock()
        .unwrap()
        .iter()
        .filter(|claim| claim.is_active())
        .map(|claim| ResidentParameterConversion {
            allocation: ResourceAllocation {
                identity: claim.allocation().clone(),
                uses: vec![ResourceUse {
                    owner: owner.clone(),
                    role: ResourceRole::Parameters,
                }],
                placement: Observed::unavailable("fixture allocation"),
                size: ResourceSize::Fixed {
                    extent: ResourceExtent {
                        payload: ResourceByteBounds::exact(claim.payload_bytes()),
                        capacity: ResourceByteBounds::exact(claim.payload_bytes()),
                    },
                },
            },
            bindings: vec![ResidentParameterConversionBinding {
                retention_group: Some(claim.group().clone()),
                owner: owner.clone(),
                unit: OffloadUnitId::new("output").unwrap(),
                name: "output.weight".into(),
                logical_target: Some("output.weight".into()),
            }],
        })
        .collect::<Vec<_>>();
    let retained = report.usage.value().unwrap().retained_payload_bytes;
    profile.parameters.current_device_resident_bytes = Observed::exact(4096 + retained, "fixture");
    profile.parameters.current_device_parameter_conversion_bytes =
        Observed::exact(retained, "fixture");
    profile.parameters.parameter_conversion_retention =
        Observed::exact(vec![report], "live ledger");
    profile.parameter_conversions = Some(conversions);
    let projection = |name: &str| ProjectionTopology {
        input: 2,
        output: 2,
        format: eredu_checkpoint::LinearFormat::Dense,
        bias: false,
        parameter: name.into(),
    };
    profile.geometry.execution_topology = Some(TextExecutionTopology {
        hidden_size: 2,
        vocabulary_size: 2,
        layers: vec![TextLayerTopology {
            input_projections: vec![],
            normalization_count: 2,
            mixer: TokenMixerTopology::Attention {
                query_heads: 1,
                kv_heads: 1,
                key_width: 2,
                value_width: 2,
                input_scores: false,
                softcap: false,
                sinks: false,
                output_gate: false,
                query_key_normalization: false,
                rotary: true,
                projections: ["query", "key", "value", "attention_output"]
                    .map(projection)
                    .into(),
            },
            feed_forward: FeedForwardTopology::Gated {
                intermediate_size: 2,
                projections: ["gate", "up", "down"].map(projection).into(),
            },
        }],
        output: ProjectionTopology {
            input: 2,
            output: 2,
            format: eredu_checkpoint::LinearFormat::Dense,
            bias: false,
            parameter: "output.weight".into(),
        },
        output_invocations: 1,
        output_softcap: false,
        selected_parameter_promotion_bytes: Some(16),
        selected_parameter_promotion_payloads: [("output.weight".into(), 16)].into(),
        projection_storage: Default::default(),
        projection_input_normalizations: Default::default(),
        f32_rms_normalization_gains: Default::default(),
        missing: vec![],
    });
    profile
}

fn pending(request: &eredu::api::GenerationMemoryRequest) -> u64 {
    request.domains[0].executions[0]
        .execution_topology
        .as_ref()
        .unwrap()
        .selected_parameter_promotion_bytes
        .unwrap()
}

#[test]
fn controlled_forecasts_refresh_trimmed_claims_without_changing_request_or_budgets() {
    use eredu::api::TraceLimits;
    use eredu_core::execution_control::SnapshotLimits;
    use std::ops::ControlFlow;

    let mut runtime = ModelRuntime::prepare(MockBackend, Default::default()).unwrap();
    runtime.session_mut().retention_forecast = true;
    super::super::conversion_retention::retain_fixture(&runtime);
    let mut model = unicode_model_with_runtime(None, 512, QWEN_TEMPLATE, runtime);
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
            tools: vec![serde_json::json!({"type":"function", "function":{
                "name":"weather", "description":"Weather",
                "parameters":{"type":"object", "properties":{}}
            }})],
            tool_choice: eredu::runtime::chat::ToolChoice::None,
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(4),
            ..Default::default()
        },
        ..Default::default()
    };
    let loaded = model
        .forecast_token_ids(&[1, 2], settings, &Default::default())
        .unwrap();
    assert_eq!(pending(&loaded.request), 0);
    assert_eq!(
        loaded.request.domains[0].resident_parameters.lower_bytes,
        4112
    );
    let prepared = model
        .prepare_observed_chat(
            &chat,
            settings,
            eredu_core::capture::CapturePlan::none(),
            TraceLimits {
                per_record_bytes: 65536,
                total_bytes: 1 << 20,
            },
        )
        .unwrap();
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 0,
        retained_bytes: 32 << 20,
        cumulative_copy_bytes: 64 << 20,
    })
    .unwrap();
    run.step(|_| ControlFlow::Continue(())).unwrap();
    let saved = run.snapshot(|_| ControlFlow::Continue(())).unwrap();
    let tokens = run.token_ids().to_vec();
    let checkpoint = run.output_checkpoint();
    let sampling = run.sampling_state().unwrap();
    let usage = run.snapshot_usage();
    let emitted = run.emitted_bytes();
    let before = run
        .forecast_remaining_generation(2, &Default::default())
        .unwrap();
    let repeated = run
        .forecast_remaining_generation(2, &Default::default())
        .unwrap();
    assert_eq!(before.request, repeated.request);
    assert_eq!(before.estimate, repeated.estimate);
    assert_eq!(pending(&before.request), 0);

    assert_eq!(
        run.trim_parameter_conversions().unwrap()[0].released_payload_bytes,
        16
    );
    let after = run
        .forecast_remaining_generation(2, &Default::default())
        .unwrap();
    assert_eq!(pending(&after.request), 16);
    assert_eq!(
        after.request.domains[0].resident_parameters.lower_bytes,
        4096
    );
    assert!(
        after.estimate.domains[0].phases[1].workspace.upper_bytes
            > before.estimate.domains[0].phases[1].workspace.upper_bytes
    );
    let retention = after
        .request
        .parameter_conversion_retention
        .as_ref()
        .unwrap();
    assert_eq!(
        retention.retained_conversions.as_deref(),
        Some([].as_slice())
    );
    assert_eq!(
        retention.groups[0].additional_admission_payload.upper_bytes,
        Some(16)
    );
    assert_eq!(after.continuation, before.continuation);
    assert_eq!(run.token_ids(), tokens);
    assert_eq!(run.output_checkpoint(), checkpoint);
    assert_eq!(run.sampling_state().unwrap(), sampling);
    assert_eq!(run.snapshot_usage(), usage);
    assert_eq!(run.emitted_bytes(), emitted);

    run.restore(&saved, |_| ControlFlow::Continue(())).unwrap();
    let restored = run
        .forecast_remaining_generation(2, &Default::default())
        .unwrap();
    assert_eq!(
        pending(&restored.request),
        16,
        "restoring request state cannot restore released retention credit"
    );
}
