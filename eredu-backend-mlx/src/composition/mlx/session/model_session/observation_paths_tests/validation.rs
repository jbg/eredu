//! Read-only erasure forwards actual stored token validation, never a rebind.
use super::*;
use crate::tests::support::path_instrumentation;

struct Slots;
impl eredu_nn::ParameterSlotVisitor<crate::MlxTensor> for Slots {
    fn visit_slot(&mut self, _: eredu_nn::ParameterMetadata, _: &mut crate::MlxTensor) {}
}
fn binding_error(error: &Error) {
    let mut cause: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(error) = cause.downcast_ref::<eredu_runtime::PreparedSessionObservationError>()
        {
            assert_eq!(
                *error,
                eredu_runtime::PreparedSessionObservationError::BindingMismatch
            );
            return;
        }
        cause = cause.source().expect("original typed path-binding cause");
    }
}

#[test]
fn replicated_and_composite_erasure_validate_actual_path_owner_and_stale_token_coldly() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let source_stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for family in ["llama", "gemma4"] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let artifact = if family == "llama" {
            crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true)
        } else {
            super::super::host_layerwise_tests::family_artifact(family)
        };
        let backend = MlxBackend::new(&stream, &source_stream).with_memory_pool(pool.clone());
        let mut first =
            eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
                .unwrap();
        let mut second =
            eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
                .unwrap();
        let a = first
            .executable_mut()
            .erased()
            .shared_observation_paths()
            .unwrap()
            .clone();
        let b = second
            .executable_mut()
            .erased()
            .shared_observation_paths()
            .unwrap()
            .clone();
        assert_eq!(strings(&a), strings(&b));
        assert!(!a.same_storage(&b));
        let group = (0..a.group_count())
            .find(|&group| a.unit_count(group).is_some_and(|count| count > 0))
            .expect("actual decoder unit in the prepared traversal");
        let original_path = a.unit_paths(group, 0).unwrap().0.as_ptr();
        let before = path_instrumentation::snapshot();
        let used = pool.used_bytes().unwrap();
        let owners = pool.unquoted_owner_count().unwrap();
        first
            .executable_mut()
            .erased()
            .validate_prepared_observation_paths(&a)
            .unwrap();
        second
            .executable_mut()
            .erased()
            .validate_prepared_observation_paths(&b)
            .unwrap();
        binding_error(
            &first
                .executable_mut()
                .erased()
                .validate_prepared_observation_paths(&b)
                .unwrap_err(),
        );
        assert_eq!(path_instrumentation::snapshot(), before);
        assert_eq!(pool.used_bytes().unwrap(), used);
        assert_eq!(pool.unquoted_owner_count().unwrap(), owners);
        // Mutable parameter-owner access invalidates the existing runtime token,
        // even though this visitor changes no tensor or semantic declaration.
        let _ = first
            .executable_mut()
            .erased_mut()
            .visit_loaded_parameters(&mut Slots);
        let stale_before = path_instrumentation::snapshot();
        binding_error(
            &first
                .executable_mut()
                .erased()
                .validate_prepared_observation_paths(&a)
                .unwrap_err(),
        );
        assert_eq!(path_instrumentation::snapshot(), stale_before);
        assert_eq!(a.unit_paths(group, 0).unwrap().0.as_ptr(), original_path);
        assert!(a.same_storage(
            first
                .executable_mut()
                .erased()
                .shared_observation_paths()
                .unwrap()
        ));
        drop((first, second, a, b));
        stream.synchronize().unwrap();
        source_stream.synchronize().unwrap();
        // Completion alone does not retire native graph owners. This existing
        // empty event is cleanup after all unchanged cold-operation assertions.
        safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
            .unwrap()
            .synchronize()
            .unwrap();
        settle(&pool, 0);
    }
}
