use super::{device_capabilities, MlxBackend, MlxDeviceIdentity, MlxModel};
use crate::backend::ExecutionContext;
use crate::tests::support::path_instrumentation;
use eredu_core::BackendProvider as _;
use safemlx::{Device, DeviceType};

#[test]
fn collective_capability_requires_an_attached_world() {
    assert!(!device_capabilities(false).collectives());
    assert!(device_capabilities(true).collectives());
}

#[test]
fn ordinary_backend_device_report_is_fail_closed_for_collectives() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let backend = MlxBackend::new(execution.stream(), execution.stream());
    let devices = backend.devices().unwrap();
    assert_eq!(devices.len(), 1);
    assert!(!devices[0].1.collectives());
}

#[test]
fn planned_backend_rejects_a_stream_that_differs_from_its_realized_device() {
    let realized = Device::new(DeviceType::Cpu, 0);
    let identity = MlxDeviceIdentity::from_realized_device(&realized, None).unwrap();
    let other = ExecutionContext::new(Device::new(DeviceType::Cpu, 1));
    let backend = MlxBackend::for_execution_plan(other.stream(), other.stream(), identity);

    let error = backend.devices().unwrap_err();
    assert!(error
        .to_string()
        .contains("does not match backend stream device"));
}

#[test]
fn missing_world_rejects_before_payload_or_architecture_construction() {
    path_instrumentation::reset();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let backend = MlxBackend::new(execution.stream(), execution.stream());
    let manifest = eredu_runtime::CommunicationManifest::new(2, 0, Vec::new(), Vec::new())
        .unwrap()
        .with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(1),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    let rank =
        super::MlxRankContext::new(2, 0, super::DeviceAssignment::new(DeviceType::Cpu, 0)).unwrap();

    let error = match backend.materialize_after_communication(
        eredu_core::SessionCapabilities::default(),
        Some(&manifest),
        Some(rank),
        |_| -> Result<MlxModel, super::Error> {
            path_instrumentation::payload_open();
            path_instrumentation::architecture_construction();
            unreachable!("missing world must reject before materialization")
        },
    ) {
        Ok(_) => panic!("missing world unexpectedly reached materialization"),
        Err(error) => error,
    };

    assert!(error
        .to_string()
        .contains("distributed model preparation requires native::distributed_backend"));
    assert_eq!(
        path_instrumentation::communication_realization_attempts(),
        1
    );
    assert_eq!(path_instrumentation::snapshot(), Default::default());
}

#[test]
fn mismatched_world_rejects_before_payload_or_architecture_construction() {
    path_instrumentation::reset();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(world.size(), 1);
    let backend =
        MlxBackend::with_distributed_world(execution.stream(), execution.stream(), &world);
    let manifest = eredu_runtime::CommunicationManifest::new(2, 0, Vec::new(), Vec::new())
        .unwrap()
        .with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(1),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    let rank =
        super::MlxRankContext::new(2, 0, super::DeviceAssignment::new(DeviceType::Cpu, 0)).unwrap();

    let error = match backend.materialize_after_communication(
        eredu_core::SessionCapabilities::default(),
        Some(&manifest),
        Some(rank),
        |_| -> Result<MlxModel, super::Error> {
            path_instrumentation::payload_open();
            path_instrumentation::architecture_construction();
            unreachable!("mismatched world must reject before materialization")
        },
    ) {
        Ok(_) => panic!("mismatched world unexpectedly reached materialization"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("communication projection has 1 manifests, expected 2"),
        "unexpected world mismatch: {error}"
    );
    assert_eq!(
        path_instrumentation::communication_realization_attempts(),
        1
    );
    assert_eq!(path_instrumentation::snapshot(), Default::default());
}

#[test]
fn selected_manifest_rejects_before_payload_or_architecture_construction() {
    path_instrumentation::reset();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(world.size(), 1);
    let backend =
        MlxBackend::with_distributed_world(execution.stream(), execution.stream(), &world);
    let manifest = eredu_runtime::CommunicationManifest::new(2, 0, Vec::new(), Vec::new())
        .unwrap()
        .with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(1),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );

    let error = match backend.materialize_after_communication(
        eredu_core::SessionCapabilities::default(),
        Some(&manifest),
        None,
        |_| -> Result<MlxModel, super::Error> {
            path_instrumentation::payload_open();
            path_instrumentation::architecture_construction();
            unreachable!("mismatched manifest world must reject before materialization")
        },
    ) {
        Ok(_) => panic!("mismatched manifest world unexpectedly reached materialization"),
        Err(error) => error,
    };

    assert!(error
        .to_string()
        .contains("communication manifest has no MLX rank/device context"));
    assert_eq!(
        path_instrumentation::communication_realization_attempts(),
        0
    );
    assert_eq!(path_instrumentation::snapshot(), Default::default());
}
