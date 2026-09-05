use eredu_core::{DraftingPlan, ParallelRankTopology, ParallelTopology, QuantizationRequest};
use eredu_runtime::LayerwiseLoadOptions;

use super::MlxLoadRequest;
use crate::backend::DeviceAssignment;
use eredu_runtime::WeightResidency;

fn topology(rank: usize, tp: usize, pp: usize, ep: usize) -> ParallelRankTopology {
    ParallelRankTopology::new(ParallelTopology::new(tp, pp, ep, 1).unwrap(), rank).unwrap()
}

#[test]
fn preparation_policy_preserves_quantized_nonresident_request() {
    let options = MlxLoadRequest::with_quantization(QuantizationRequest::MxFp4)
        .with_weight_residency(WeightResidency::layerwise_host(
            LayerwiseLoadOptions::default(),
        ));
    let policy = options.preparation_policy().unwrap();
    assert_eq!(
        policy.quantization(),
        Some(eredu_core::QuantizationRequest::MxFp4)
    );
    assert_eq!(
        policy.residency(),
        eredu_core::ResidencyRequest::LayerwiseHost
    );
}

#[test]
fn preparation_policy_rejects_invalid_affine_geometry() {
    let error = MlxLoadRequest::with_quantization(QuantizationRequest::Affine {
        group_size: 17,
        bits: 4,
    })
    .preparation_policy()
    .unwrap_err();

    assert!(matches!(
        error,
        crate::backend::error::Error::Quantization(message)
            if message.contains("group_size")
    ));
}

#[test]
fn preparation_policy_preserves_exact_parallel_topology() {
    let topology = topology(5, 2, 3, 2);
    let device = DeviceAssignment::new(safemlx::DeviceType::Cpu, 0);
    let policy = MlxLoadRequest::with_parallel(
        topology,
        device,
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        128,
        MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap()
    .preparation_policy()
    .unwrap();
    assert_eq!(policy.topology(), Some(topology.topology()));
}

#[test]
fn preparation_policies_distinguish_parallel_axes() {
    let device = DeviceAssignment::new(safemlx::DeviceType::Cpu, 0);
    let tensor_pipeline = topology(0, 2, 3, 1);
    let tensor_expert = topology(0, 2, 1, 3);
    let tensor_pipeline_policy = MlxLoadRequest::with_parallel(
        tensor_pipeline,
        device,
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        128,
        MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap()
    .preparation_policy()
    .unwrap();
    let tensor_expert_policy = MlxLoadRequest::with_parallel(
        tensor_expert,
        device,
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        128,
        MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap()
    .preparation_policy()
    .unwrap();

    assert_ne!(tensor_pipeline_policy, tensor_expert_policy);
}

#[test]
fn parallel_policy_rejects_nonpositive_invocation_limits() {
    let topology = topology(0, 2, 1, 1);
    let device = DeviceAssignment::new(safemlx::DeviceType::Cpu, 0);
    let error = MlxLoadRequest::with_parallel(
        topology,
        device,
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        0,
        128,
        MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("partitioned invocation limits must be positive"));
}

#[test]
fn portable_drafting_plan_is_fixed_before_payload_selection() {
    let disabled = MlxLoadRequest::default()
        .with_drafting_plan(&DraftingPlan::Disabled)
        .unwrap();
    assert_eq!(
        disabled.checked_normalized().unwrap().0.drafting(),
        eredu_runtime::DraftingLoadRequest::Disabled
    );

    let embedded = MlxLoadRequest::default()
        .with_drafting_plan(&DraftingPlan::Embedded {
            max_draft_tokens: 3,
            lookahead: false,
            adaptive_lookahead: false,
        })
        .unwrap();
    assert_eq!(
        embedded.checked_normalized().unwrap().0.drafting(),
        eredu_runtime::DraftingLoadRequest::Embedded {
            max_draft_tokens: std::num::NonZeroUsize::new(3).unwrap()
        }
    );

    assert!(MlxLoadRequest::default()
        .with_drafting_plan(&DraftingPlan::Embedded {
            max_draft_tokens: 0,
            lookahead: false,
            adaptive_lookahead: false,
        })
        .is_err());
}

#[test]
fn adapter_translation_preserves_the_exact_normalized_request() {
    let capabilities = eredu_core::SessionCapabilities::new(true, false, true);
    let residency = WeightResidency::layerwise_host(LayerwiseLoadOptions::default());
    let options = MlxLoadRequest::with_quantization(QuantizationRequest::MxFp4)
        .with_weight_residency(residency)
        .with_required_session_capabilities(capabilities)
        .with_drafting_plan(&DraftingPlan::Disabled)
        .unwrap();
    let expected =
        eredu_runtime::NormalizedLoadRequest::with_quantization(QuantizationRequest::MxFp4)
            .with_weight_residency(residency)
            .with_required_session_capabilities(capabilities)
            .with_drafting(eredu_runtime::DraftingLoadRequest::Disabled);

    let (normalized, rank) = options.checked_normalized().unwrap();
    assert_eq!(normalized, &expected);
    assert!(rank.is_none());
}

#[test]
fn native_device_is_not_part_of_normalized_request_equality() {
    let topology = topology(0, 2, 1, 1);
    let completion = MlxLoadRequest::test_communication_completion_policy();
    let wire =
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32);
    let cpu = MlxLoadRequest::with_parallel(
        topology,
        DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        wire,
        1,
        128,
        completion,
    )
    .unwrap();
    let gpu = MlxLoadRequest::with_parallel(
        topology,
        DeviceAssignment::new(safemlx::DeviceType::Gpu, 0),
        wire,
        1,
        128,
        completion,
    )
    .unwrap();

    let (cpu_normalized, cpu_rank) = cpu.checked_normalized().unwrap();
    let (gpu_normalized, gpu_rank) = gpu.checked_normalized().unwrap();
    assert_eq!(cpu_normalized, gpu_normalized);
    assert_ne!(cpu_rank, gpu_rank);
    assert_ne!(cpu, gpu);
}

#[test]
fn checked_translation_rejects_unpaired_parallel_halves() {
    let topology = topology(0, 2, 1, 1);
    let parallel = eredu_runtime::ParallelLoadRequest::new(
        topology,
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        128,
        MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap();
    let normalized = eredu_runtime::NormalizedLoadRequest::default()
        .with_parallel_execution(parallel)
        .unwrap();
    let missing_device = MlxLoadRequest {
        normalized,
        parallel_device: None,
    };
    assert!(missing_device.checked_normalized().is_err());

    let orphaned_device = MlxLoadRequest {
        normalized: eredu_runtime::NormalizedLoadRequest::default(),
        parallel_device: Some(DeviceAssignment::new(safemlx::DeviceType::Cpu, 0)),
    };
    assert!(orphaned_device.checked_normalized().is_err());
}
