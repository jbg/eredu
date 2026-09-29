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
    let options = MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::with_quantization(QuantizationRequest::MxFp4)
            .with_weight_residency(WeightResidency::layerwise_host(
                LayerwiseLoadOptions::default(),
            )),
    );
    let policy = options.normalized().preparation_policy().unwrap();
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
    let error = MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::with_quantization(QuantizationRequest::Affine {
            group_size: 17,
            bits: 4,
        }),
    )
    .normalized()
    .preparation_policy()
    .unwrap_err();

    assert!(matches!(
        error,
        eredu_runtime::NormalizedLoadRequestError::Quantization(message)
            if message.contains("group_size")
    ));
}

#[test]
fn portable_drafting_plan_is_fixed_before_payload_selection() {
    let disabled = NormalizedLoadRequest::default()
        .with_drafting_plan(&DraftingPlan::Disabled)
        .unwrap();
    assert_eq!(
        MlxLoadRequest::from_normalized(disabled)
            .checked_normalized()
            .unwrap()
            .0
            .drafting(),
        eredu_runtime::DraftingLoadRequest::Disabled
    );

    let embedded = NormalizedLoadRequest::default()
        .with_drafting_plan(&DraftingPlan::Embedded {
            max_draft_tokens: 3,
            lookahead: false,
            adaptive_lookahead: false,
        })
        .unwrap();
    assert_eq!(
        MlxLoadRequest::from_normalized(embedded)
            .checked_normalized()
            .unwrap()
            .0
            .drafting(),
        eredu_runtime::DraftingLoadRequest::Embedded {
            max_draft_tokens: std::num::NonZeroUsize::new(3).unwrap()
        }
    );

    assert!(NormalizedLoadRequest::default()
        .with_drafting_plan(&DraftingPlan::Embedded {
            max_draft_tokens: 0,
            lookahead: false,
            adaptive_lookahead: false,
        })
        .is_err());
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
use eredu_runtime::NormalizedLoadRequest;
