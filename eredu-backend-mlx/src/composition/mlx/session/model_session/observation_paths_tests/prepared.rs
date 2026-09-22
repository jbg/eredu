//! Native adapter forwards the prepared requirement through actual equations.
use super::*;

struct BorrowedPaths {
    source: SharedLayeredObservationPaths,
    boundaries: usize,
    logits: usize,
}
impl eredu_runtime::ActivationObserver<crate::MlxTensor, Error> for BorrowedPaths {
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn observe(&mut self, path: &str, _: &crate::MlxTensor) -> Result<(), Error> {
        for group in 0..self.source.group_count() {
            for unit in 0..self.source.unit_count(group).unwrap() {
                let (input, output) = self.source.unit_paths(group, unit).unwrap();
                for expected in [input, output] {
                    if path == expected {
                        assert_eq!(path.as_ptr(), expected.as_ptr());
                        self.boundaries += 1;
                    }
                }
            }
        }
        if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
            self.logits += 1;
        }
        Ok(())
    }
}

#[test]
fn actual_native_prepared_traversal_preserves_full_logits_and_cached_decode_values() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let source_stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &source_stream).with_memory_ledger(pool);
    let mut ordinary =
        eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
            .unwrap();
    let mut prepared =
        eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
            .unwrap();
    let source = prepared
        .executable_mut()
        .erased()
        .shared_observation_paths()
        .unwrap()
        .clone();
    let mut observer = BorrowedPaths {
        source,
        boundaries: 0,
        logits: 0,
    };
    for ids in [vec![2_u32, 7, 5], vec![3], vec![4]] {
        let mut values = Vec::new();
        for (model, use_prepared) in [(&mut ordinary, false), (&mut prepared, true)] {
            // Raw diagnostic calls remain under the model's original unquoted
            // loading/recovery owner. This does not claim managed capture admission.
            let memory = model.memory_owner().unwrap().clone();
            let output = crate::backend::submission_recovery::detached_retained(memory, || {
                let tokens = Array::from_slice(&ids, &[1, ids.len() as i32]);
                let executable = model.executable_mut();
                let output = if use_prepared {
                    let mut arrays =
                        crate::composition::mlx::session::observation::ArrayObserverAdapter {
                            inner: &mut observer,
                            routed_path: None,
                            routed_invocation_active: false,
                            allocation_authority: None,
                        };
                    executable.erased_mut().forward_with_observer(
                        &tokens,
                        None,
                        &stream,
                        &mut arrays,
                    )?
                } else {
                    executable.erased_mut().forward_with_observer(
                        &tokens,
                        None,
                        &stream,
                        &mut eredu_runtime::NoopObserver,
                    )?
                };
                let roots = executable
                    .erased()
                    .retained_decoder_state_storage()?
                    .into_retained_arrays()
                    .map_err(|(cause, _storage)| cause)?;
                crate::backend::MlxCompletion::submission_retaining(output, roots)?.wait()
            })
            .unwrap();
            assert_eq!(output.shape()[1], ids.len() as i32);
            values.push(output.evaluated().unwrap().as_slice::<f32>().to_vec());
        }
        assert_eq!(values[0], values[1]);
        assert!(values[1].iter().any(|value| value.abs() > 1e-6));
    }
    assert_eq!(observer.logits, 3);
    assert!(observer.boundaries > 0);
    assert!(observer.source.same_storage(
        prepared
            .executable_mut()
            .erased()
            .shared_observation_paths()
            .unwrap()
    ));
}
