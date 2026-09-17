use super::*;
use crate::MlxLoadRequest;
thread_local! { static COUNT_HOOKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
fn count_hook() {
    COUNT_HOOKS.with(|value| value.set(value.get() + 1));
}
struct CountHook;
impl Drop for CountHook {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(count_hook);
    }
}

#[test]
fn completed_native_static_source_reuses_real_owner_across_residencies_without_checkpoint_reads() {
    let (stream, weights_stream) = execution_streams();
    let host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
    );
    for model_type in ["llama", "qwen3_moe"] {
        for residency in [
            eredu_runtime::WeightResidency::fully_resident(),
            eredu_runtime::WeightResidency::layerwise_host(host),
            eredu_runtime::WeightResidency::dense_disk_stream(Default::default()),
        ] {
            let root = tiny_safetensors_artifact(model_type, false, false, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let options = MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
            );
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.normalized().preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
            let mut executable = model.into_executable();
            let checkpoint = root.path().join("model.safetensors");
            let hidden_root = tempfile::tempdir().unwrap();
            let hidden = hidden_root.path().join("artifact");
            let reads = executable
                .erased()
                .residency_report()
                .unwrap()
                .unwrap()
                .weight_store()
                .physical_reads;
            // Hide the directory without changing the admitted file's ctime.
            // Renaming the file itself invalidates the later streamed read.
            std::fs::rename(root.path(), &hidden).unwrap();
            assert!(!checkpoint.exists());
            let mut guard = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
                .unwrap()
                .enter()
                .unwrap();
            safemlx::register_thread_runtime_housekeeping(count_hook);
            let hook_owner = CountHook;
            COUNT_HOOKS.with(|value| value.set(0));
            let counted = NativeParameterOwnerSource::new(executable.erased())
                .count(&mut guard)
                .unwrap();
            assert!(std::ptr::addr_eq(
                counted.source().owner(),
                executable.erased()
            ));
            let first = counted.counts();
            assert!(
                first
                    .role(ParameterOwnerRole::Static)
                    .parameters
                    .named_slots
                    >= 3
            );
            assert_eq!(first.prediction_modules, 0);
            assert_eq!(first.pooling_prototypes, 0);
            assert_eq!(first.model_prototypes, 0);
            assert_eq!(
                NativeParameterOwnerSource::new(executable.erased())
                    .count(&mut guard)
                    .unwrap()
                    .counts(),
                first
            );
            drop(counted);
            assert_eq!(COUNT_HOOKS.with(std::cell::Cell::get), 0);
            drop(hook_owner);
            drop(guard);
            assert_eq!(
                executable
                    .erased()
                    .residency_report()
                    .unwrap()
                    .unwrap()
                    .weight_store()
                    .physical_reads,
                reads
            );
            std::fs::rename(&hidden, root.path()).unwrap();
            let tokens = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
            let parts = [input::token_ids_part(&tokens).unwrap()];
            drop(
                executable
                    .erased_mut()
                    .prefill(input::ModelInput::new(&parts), &stream)
                    .unwrap(),
            );
            let mut guard = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
                .unwrap()
                .enter()
                .unwrap();
            assert_eq!(
                NativeParameterOwnerSource::new(executable.erased())
                    .count(&mut guard)
                    .unwrap()
                    .counts(),
                first
            );
        }
    }
}
