use super::{device_capabilities, MlxBackend, MlxDeviceIdentity, MlxModel};
use crate::backend::ExecutionContext;
use crate::tests::support::path_instrumentation;
use eredu_core::BackendProvider as _;
use safemlx::{Device, DeviceType};

#[test]
fn materialization_reclaims_retired_owners_before_allocating_the_next_model() {
    use crate::backend::ordinary_retirement::OrdinaryRetirement;
    use std::{cell::Cell, rc::Rc};
    struct PreviousModel(Rc<Cell<bool>>);
    impl Drop for PreviousModel {
        fn drop(&mut self) {
            assert!(safemlx::can_reclaim_submission_resources());
            self.0.set(true);
        }
    }
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let backend = MlxBackend::new(execution.stream(), execution.stream());
    let released = Rc::new(Cell::new(false));
    drop(OrdinaryRetirement::new(PreviousModel(Rc::clone(&released))));
    assert!(!released.get());
    let error = backend
        .materialize_after_communication(
            eredu_core::SessionCapabilities::default(),
            None,
            None,
            |_| {
                assert!(released.get(), "old owner outlived next model allocation");
                Err(super::Error::ArchitectureModel(
                    "materialization sentinel".into(),
                ))
            },
        )
        .err()
        .expect("sentinel stops before allocating a new model");
    assert!(error.to_string().contains("materialization sentinel"));
}

#[test]
fn prepared_model_rejects_another_native_device_before_session_reset() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let other = ExecutionContext::new(Device::new(DeviceType::Cpu, 1));
    let backend = MlxBackend::new(execution.stream(), execution.stream());
    let replacement = MlxBackend::new(other.stream(), other.stream());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    path_instrumentation::reset();
    let error = replacement
        .create_session(model)
        .err()
        .expect("a prepared CPU0 model cannot be paired with CPU1");
    assert!(error.to_string().contains("different native devices"));
    assert_eq!(path_instrumentation::session_reset_attempts(), 0);
    assert_eq!(path_instrumentation::session_input_creation_attempts(), 0);
    assert_eq!(path_instrumentation::snapshot(), Default::default());
}

#[test]
fn prepared_session_allows_another_stream_but_rejects_another_device_before_work() {
    use eredu_core::{BackendSession as _, Completion as _, InspectableBackendSession as _};
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let second = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let other = ExecutionContext::new(Device::new(DeviceType::Cpu, 1));
    let backend = MlxBackend::new(execution.stream(), execution.stream());
    let compatible = MlxBackend::new(second.stream(), second.stream());
    let replacement = MlxBackend::new(other.stream(), other.stream());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    let mut session = compatible.create_session(model).unwrap();
    session
        .submit_token_decode(&compatible, 1)
        .unwrap()
        .completion
        .wait()
        .unwrap();
    let before = session
        .neutral_prediction_target_mut()
        .unwrap()
        .fixed_numeric_state_snapshot()
        .unwrap();
    let input = crate::composition::mlx::MlxModelInput::from(
        crate::backend::runtime::media::input::ModelInput::new(&[]),
    );
    let token = safemlx::Array::from_slice(&[2_u32], &[1, 1]);
    let descriptor = eredu_core::cache::PromptCacheDescriptor::from_model_identity(
        session.prompt_cache_model_identity().unwrap(),
        "fixture",
        "tokens:1",
        1,
    )
    .unwrap();
    let cache = tempfile::tempdir().unwrap();
    let destination = cache.path().join("must-not-exist");
    path_instrumentation::reset();
    let reject =
        |error: super::Error| assert!(error.to_string().contains("different native devices"));
    reject(session.submit_token_decode(&replacement, 2).err().unwrap());
    reject(session.prefill(&replacement, input.clone()).err().unwrap());
    reject(session.decode(&replacement, token.clone()).err().unwrap());
    reject(
        session
            .inspect_prefill(
                &replacement,
                input.clone(),
                &eredu_core::ObservationRequest::all(),
            )
            .err()
            .unwrap(),
    );
    reject(
        session
            .inspect_decode(&replacement, token, &eredu_core::ObservationRequest::all())
            .err()
            .unwrap(),
    );
    reject(
        session
            .save_prompt_cache(
                &replacement,
                &destination,
                descriptor.clone(),
                &[1],
                &eredu_core::cache::PromptCacheOptions::default(),
            )
            .unwrap_err(),
    );
    reject(
        session
            .load_prompt_cache(&replacement, &destination, &descriptor, &[1])
            .unwrap_err(),
    );
    reject(
        session
            .load_prompt_cache_for_input(&replacement, &destination, &descriptor, &[1], &input)
            .unwrap_err(),
    );
    assert!(!destination.exists());
    assert_eq!(path_instrumentation::session_reset_attempts(), 0);
    assert_eq!(path_instrumentation::session_input_creation_attempts(), 0);
    assert_eq!(path_instrumentation::snapshot(), Default::default());
    assert_eq!(
        session
            .neutral_prediction_target_mut()
            .unwrap()
            .fixed_numeric_state_snapshot()
            .unwrap(),
        before
    );
    session
        .submit_token_decode(&compatible, 2)
        .unwrap()
        .completion
        .wait()
        .unwrap();
}

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
