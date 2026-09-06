use super::*;
use eredu_core::{
    residency::CacheEvictionPolicy, DevicePlan, DraftPlacementPlan, DraftingPlan, ExpertCachePlan,
    ParallelRankTopology, ParallelTopology, QuantizationRequest, SessionCapabilities,
};

fn plan() -> ExecutionPlan {
    ExecutionPlan::fully_resident(DevicePlan::new("independent", "client:7").unwrap())
}

fn normalize(plan: &ExecutionPlan) -> Result<NormalizedLoadRequest, ExecutionPlanLoadError> {
    NormalizedLoadRequest::from_execution_plan(plan, ResidencyDiagnostics::new(true, false), None)
}

#[test]
fn prompt_cache_persistence_is_explicit_and_independent_of_decode_state() {
    let ordinary = plan();
    let default_request = normalize(&ordinary).unwrap();
    assert!(!ordinary.prompt_cache_persistence());
    assert!(!default_request.prompt_cache_persistence());
    let persistent = ordinary.clone().with_prompt_cache_persistence(true);
    let request = normalize(&persistent).unwrap();
    assert!(request.prompt_cache_persistence());
    assert_eq!(request.state_residency(), default_request.state_residency());
    assert_eq!(
        request.required_session_capabilities(),
        default_request.required_session_capabilities()
    );
    assert_ne!(request, default_request);
    assert_eq!(
        normalize(&persistent.with_prompt_cache_persistence(false)).unwrap(),
        default_request
    );
}

#[test]
fn transformation_and_residency_policies_preserve_independent_controls() {
    let residencies = [
        ResidencyPlan::FullyResident,
        ResidencyPlan::LayerwiseHost {
            device_layer_window: 3,
            device_budget_bytes: Some(4096),
            host_budget_bytes: Some(8192),
        },
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 4096,
            host_budget_bytes: 8192,
            host_lookahead: 3,
            background_queue: 2,
        },
    ];
    for (transform, expected) in [
        (WeightTransformationPlan::PreserveCheckpoint, None),
        (
            WeightTransformationPlan::Affine {
                bits: 4,
                group_size: 64,
            },
            Some(QuantizationRequest::Affine {
                bits: 4,
                group_size: 64,
            }),
        ),
        (
            WeightTransformationPlan::MxFp4,
            Some(QuantizationRequest::MxFp4),
        ),
    ] {
        for residency in &residencies {
            for banked in [false, true] {
                let plan = plan()
                    .with_weight_transformation(transform)
                    .with_residency(residency.clone())
                    .with_max_cached_shards(7)
                    .with_required_session_capabilities(SessionCapabilities::new(true, false, true))
                    .with_expert_cache(banked.then(|| {
                        ExpertCachePlan::new(
                            Some(1024),
                            Some(2048),
                            128,
                            64,
                            CacheEvictionPolicy::LeastRecentlyUsed,
                        )
                    }));
                let request = normalize(&plan).unwrap();
                assert_eq!(request.quantization(), expected);
                assert_eq!(request.max_cached_shards(), 7);
                assert_eq!(
                    request.required_session_capabilities(),
                    SessionCapabilities::new(true, false, true)
                );
                assert_eq!(
                    request.weight_residency().parameter_bank_cache().is_some(),
                    banked
                );
                let layers = request.weight_residency().layers();
                assert_eq!(
                    layers.is_fully_resident(),
                    matches!(residency, ResidencyPlan::FullyResident)
                );
                if !layers.is_fully_resident() {
                    assert!(layers.sample_backend_memory());
                    assert!(!layers.sample_process_memory());
                    assert_eq!(layers.max_cached_shards(), 7);
                    assert_eq!(layers.offload().unwrap().device_budget_bytes(), Some(4096));
                    assert_eq!(layers.offload().unwrap().host_budget_bytes(), Some(8192));
                }
                if let Some(bank) = request.weight_residency().parameter_bank_cache() {
                    assert_eq!(bank.compact_bank_scratch_bytes(), 128);
                    assert_eq!(bank.prefill_compact_bank_target_bytes(), 64);
                    assert_eq!(bank.offload().device_budget_bytes(), Some(1024));
                    assert_eq!(bank.offload().host_budget_bytes(), Some(2048));
                }
            }
        }
    }
}

#[test]
fn drafting_intent_is_checked_without_native_placement() {
    assert_eq!(
        normalize(&plan()).unwrap().drafting(),
        crate::DraftingLoadRequest::Disabled
    );
    let embedded = plan().with_drafting(DraftingPlan::Embedded {
        max_draft_tokens: 5,
        lookahead: true,
        adaptive_lookahead: false,
    });
    assert_eq!(
        normalize(&embedded)
            .unwrap()
            .drafting()
            .embedded_capacity()
            .unwrap()
            .get(),
        5
    );
    let external = plan().with_drafting(DraftingPlan::External {
        model: "assistant".into(),
        placement: DraftPlacementPlan::Device {
            device: DevicePlan::new("foreign", "queue:2").unwrap(),
        },
        max_draft_tokens: 6,
        lookahead: false,
        adaptive_lookahead: false,
    });
    assert_eq!(
        normalize(&external).unwrap().drafting(),
        crate::DraftingLoadRequest::ExternalTarget
    );
    let invalid = external.with_drafting(DraftingPlan::External {
        model: "assistant".into(),
        placement: DraftPlacementPlan::Target,
        max_draft_tokens: 0,
        lookahead: false,
        adaptive_lookahead: false,
    });
    assert!(matches!(
        normalize(&invalid),
        Err(ExecutionPlanLoadError::Plan(
            eredu_core::ExecutionPlanError::ZeroDraftTokens
        ))
    ));
}

#[test]
fn distributed_plan_requires_exact_portable_rank_contract() {
    let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
    let plan = plan().with_topology(topology);
    assert!(matches!(
        normalize(&plan),
        Err(ExecutionPlanLoadError::MissingParallelRequest)
    ));
    let parallel = ParallelLoadRequest::new(
        ParallelRankTopology::new(topology, 1).unwrap(),
        crate::PipelineWireContract::new(crate::PipelineActivationDtype::Float32),
        2,
        128,
        crate::CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(2),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap(),
    )
    .unwrap();
    let request = NormalizedLoadRequest::from_execution_plan(
        &plan,
        ResidencyDiagnostics::default(),
        Some(parallel),
    )
    .unwrap();
    assert_eq!(request.parallel_execution(), Some(parallel));
    assert_eq!(
        request.preparation_policy().unwrap().topology(),
        Some(topology)
    );
    let wrong = plan.with_topology(ParallelTopology::new(1, 2, 1, 1).unwrap());
    assert!(matches!(
        NormalizedLoadRequest::from_execution_plan(
            &wrong,
            ResidencyDiagnostics::default(),
            Some(parallel)
        ),
        Err(ExecutionPlanLoadError::ParallelTopologyMismatch)
    ));
}

#[test]
fn invalid_plans_are_distinct_from_unavailable_backend_mechanisms() {
    assert!(matches!(
        normalize(&plan().with_max_cached_shards(0)),
        Err(ExecutionPlanLoadError::Plan(
            eredu_core::ExecutionPlanError::ZeroMappedShards
        ))
    ));
    for (bits, group_size) in [(-1, 64), (300, 64), (4, -1), (4, 0)] {
        assert!(matches!(
            normalize(
                &plan().with_weight_transformation(WeightTransformationPlan::Affine {
                    bits,
                    group_size
                })
            ),
            Err(ExecutionPlanLoadError::Request(
                NormalizedLoadRequestError::Quantization(_)
            ))
        ));
    }
    assert!(matches!(
        normalize(&plan().with_expert_cache(Some(ExpertCachePlan::new(
            None,
            None,
            0,
            1,
            CacheEvictionPolicy::LeastRecentlyUsed
        )))),
        Err(ExecutionPlanLoadError::Residency(
            crate::WeightResidencyPolicyError::ZeroParameterBankScratchLimit
        ))
    ));
    assert!(matches!(
        normalize(&plan().with_residency(ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1,
            host_budget_bytes: 0,
            host_lookahead: 1,
            background_queue: 1
        })),
        Err(ExecutionPlanLoadError::Residency(
            crate::WeightResidencyPolicyError::HostDisabledControls
        ))
    ));
    // Device-family support is a backend fact. Portable derivation must preserve
    // its policy without trying to discover or construct that device.
    assert!(
        normalize(&plan().with_device(DevicePlan::new("foreign", "unavailable:99").unwrap()))
            .is_ok()
    );
}
