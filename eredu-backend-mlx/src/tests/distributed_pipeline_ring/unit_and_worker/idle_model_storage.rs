mod idle_model_storage_tests {
    use super::*;
    use crate::backend::runtime::residency::storage::RetainedStorage;
    use eredu_runtime::working_memory::MemoryLedger;

    #[test]
    fn enclosing_coverage_and_decoder_growth_remain_separate() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream)
            .with_memory_ledger(crate::memory_fixture::ledger(u64::MAX, 0).unwrap());
        let checkpoint =
            crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
        let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default())
            .unwrap()
            .into_inner();
        // Only the executable survives this move. The fixture has no retained
        // outer processor, communicator, overlay or independently owned input.
        let mut executable = model.into_executable();
        let before = executable.erased().state_snapshot();
        let idle = executable
            .retained_idle_storage(Some(RetainedStorage::default()))
            .unwrap();
        let nonstate = idle.nonstate_bytes().unwrap().unwrap();
        assert!(nonstate > 0);
        assert_eq!(idle.decoder_state_bytes().unwrap(), Some(0));
        assert!(idle.has_empty_decoder_storage().unwrap());
        let retention = executable.erased().retained_inference_authority().unwrap();
        assert_eq!(retention.requests().len(), 0);
        assert_eq!(executable.erased().state_snapshot(), before);

        let unknown = executable.retained_idle_storage(None).unwrap();
        assert_eq!(unknown.nonstate_bytes().unwrap(), None);
        assert_eq!(unknown.decoder_state_bytes().unwrap(), Some(0));
        assert_eq!(executable.erased().state_snapshot(), before);
        drop((idle, unknown, retention));

        let tokens = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
        let parts = [text_input_part(&tokens)];
        executable
            .prefill(
                crate::backend::runtime::media::input::ModelInput::new(&parts),
                &stream,
            )
            .unwrap()
            .evaluated()
            .unwrap();
        executable
            .erased_mut()
            .decode(&Array::from_slice(&[2_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        let frontier = executable.erased().state_snapshot();
        let populated = executable
            .retained_idle_storage(Some(RetainedStorage::default()))
            .unwrap();
        assert!(!populated.has_empty_decoder_storage().unwrap());
        let decoder = populated.decoder_state_bytes().unwrap().unwrap();
        assert!(decoder > 0);
        assert_eq!(populated.nonstate_bytes().unwrap(), Some(nonstate));
        assert_eq!(executable.erased().state_snapshot(), frontier);

        let (mut all, state) = populated.into_parts();
        all.merge(state).unwrap();
        assert_eq!(all.byte_bound().unwrap(), Some(nonstate + decoder));
        // State inspection remains complete even when outer coverage is missing.
        let unknown = executable.retained_idle_storage(None).unwrap();
        assert_eq!(unknown.nonstate_bytes().unwrap(), None);
        assert_eq!(unknown.decoder_state_bytes().unwrap(), Some(decoder));
    }

    #[test]
    fn retained_blueprint_source_aliases_count_once() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream)
            .with_memory_ledger(crate::memory_fixture::ledger(u64::MAX, 0).unwrap());
        let checkpoint = tempfile::tempdir().unwrap();
        let path = checkpoint.path().join("model.gguf");
        write_llama_compatible_gguf(&path, "llama");
        let executable = load_model(&backend, &path, MlxLoadRequest::default())
            .unwrap()
            .into_inner()
            .into_executable();
        let source = executable
            .inference_blueprint()
            .unwrap()
            .source_storage()
            .unwrap()
            .unwrap();
        // Unlike the weak SafeTensors payload cache, this real GGUF source
        // retains a positive reader-buffer ceiling with a stable shared owner.
        let source_bytes = source.bytes().unwrap();
        assert!(source_bytes > 0);
        assert_eq!(source.owner_count(), 1);
        let source_entries = source.capacities().collect::<Vec<_>>();
        // Loading settles numerical roots but does not certify arbitrary
        // custom backing. Metal may retain the GGUF host buffer with its
        // managed deleter, leaving that native capacity explicitly unknown.
        let before = executable.erased().state_snapshot();
        let unaliased = executable
            .retained_idle_storage(Some(RetainedStorage::default()))
            .unwrap();
        let nonstate_bytes = unaliased.nonstate_bytes().unwrap();
        assert!(unaliased.has_empty_decoder_storage().unwrap());
        let (unaliased_nonstate, _) = unaliased.into_parts();
        assert_eq!(
            unaliased_nonstate
                .source_storage()
                .capacities()
                .collect::<Vec<_>>(),
            source_entries
        );
        match nonstate_bytes {
            Some(bytes) => assert!(bytes > source_bytes),
            None => assert!(!unaliased_nonstate.unknown_arrays().is_empty()),
        }
        let mut baseline = RetainedStorage::default();
        baseline.include_sources(Some(source.clone())).unwrap();
        let partial_bytes = baseline.byte_bound().unwrap().unwrap();
        assert_eq!(partial_bytes, source_bytes);
        let mut enclosing_alias = RetainedStorage::default();
        enclosing_alias
            .include_sources(Some(source.clone()))
            .unwrap();
        enclosing_alias.include_sources(Some(source)).unwrap();
        baseline.merge(enclosing_alias).unwrap();
        assert_eq!(baseline.byte_bound().unwrap(), Some(partial_bytes));
        let aliased = executable.retained_idle_storage(Some(baseline)).unwrap();
        // The executable already includes this blueprint owner. Additional
        // retained source aliases cannot charge its reader ceiling again.
        assert_eq!(aliased.nonstate_bytes().unwrap(), nonstate_bytes);
        assert!(aliased.has_empty_decoder_storage().unwrap());
        let (aliased_nonstate, _) = aliased.into_parts();
        assert_eq!(
            aliased_nonstate
                .source_storage()
                .capacities()
                .collect::<Vec<_>>(),
            source_entries
        );
        assert_eq!(
            aliased_nonstate.unknown_arrays().len(),
            unaliased_nonstate.unknown_arrays().len()
        );
        if nonstate_bytes.is_none() {
            assert!(!aliased_nonstate.unknown_arrays().is_empty());
        }
        // Missing enclosing coverage remains unknown even with settled roots.
        let unknown = executable.retained_idle_storage(None).unwrap();
        assert_eq!(unknown.nonstate_bytes().unwrap(), None);
        assert!(unknown.has_empty_decoder_storage().unwrap());
        assert_eq!(executable.erased().state_snapshot(), before);
    }

    #[test]
    fn dormant_prediction_is_included_and_erased_observer_payload_stays_unknown() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream)
            .with_memory_ledger(crate::memory_fixture::ledger(u64::MAX, 0).unwrap());
        let checkpoint = tempfile::tempdir().unwrap();
        write_deepseek_fixture_with_prediction(checkpoint.path(), 2, 1);
        let mut executable = load_model(&backend, checkpoint.path(), MlxLoadRequest::default())
            .unwrap()
            .into_inner()
            .into_executable();
        let mut expected = executable
            .erased()
            .retained_target_nonstate_storage()
            .unwrap();
        let target_bytes = expected.byte_bound().unwrap().unwrap();
        let prediction = executable.erased().retained_prediction_storage().unwrap();
        assert!(prediction.byte_bound().unwrap().unwrap() > 0);
        expected.merge(prediction).unwrap();
        // An idle session retains its fixed state metadata and prepared
        // observation paths even before it owns any decoder arrays.
        let auxiliary = executable
            .erased()
            .retained_idle_auxiliary_storage()
            .unwrap();
        let paths = executable.erased().shared_observation_paths().unwrap();
        let path_bytes = paths.capacity_bytes().unwrap();
        assert!(path_bytes > 0);
        assert!(auxiliary.byte_bound().unwrap().unwrap() >= path_bytes);
        expected.merge(auxiliary).unwrap();
        expected
            .include_sources(
                executable
                    .inference_blueprint()
                    .unwrap()
                    .source_storage()
                    .unwrap(),
            )
            .unwrap();
        let idle = executable
            .retained_idle_storage(Some(RetainedStorage::default()))
            .unwrap();
        assert!(expected.byte_bound().unwrap().unwrap() > target_bytes);
        assert_eq!(
            idle.nonstate_bytes().unwrap(),
            expected.byte_bound().unwrap()
        );
        assert!(idle.has_empty_decoder_storage().unwrap());
        struct RetainedObserver {
            _payload: Array,
        }
        impl eredu_runtime::ActivationObserver<MlxTensor, crate::backend::error::Error> for RetainedObserver {
            fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), crate::backend::error::Error> {
                Ok(())
            }
        }
        let payload = Array::from_slice(&[2_f32, 3.0, 5.0, 7.0], &[4]);
        assert!(payload.allocation_info().unwrap().unwrap().bytes() > 0);
        assert!(executable.install_embedded_prediction_observers(
            eredu_architectures::speculative_execution::EmbeddedPredictionObservers::new(
                RetainedObserver { _payload: payload },
                eredu_runtime::NoopObserver,
            ),
        ));
        let unknown = executable
            .retained_idle_storage(Some(RetainedStorage::default()))
            .unwrap();
        assert_eq!(unknown.nonstate_bytes().unwrap(), None);
        assert_eq!(unknown.decoder_state_bytes().unwrap(), Some(0));
        assert_eq!(
            executable
                .erased()
                .retained_idle_auxiliary_storage()
                .unwrap()
                .byte_bound()
                .unwrap(),
            None,
        );
    }
}
