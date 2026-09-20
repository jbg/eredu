use super::*;

#[test]
fn retained_target_parameters_track_replacements_without_loading_or_double_counting() {
    let (stream, weights_stream) = execution_streams();
    let host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
    );
    let disk = eredu_runtime::DenseDiskStreamLoadOptions::default();
    for (tied, packed) in [(false, false), (true, false), (false, true)] {
        for residency in [
            eredu_runtime::WeightResidency::fully_resident(),
            eredu_runtime::WeightResidency::layerwise_host(host),
            eredu_runtime::WeightResidency::dense_disk_stream(disk),
        ] {
            let label = format!("tied={tied} packed={packed} {residency:?}");
            let root = tiny_safetensors_artifact("llama", tied, false, packed);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let options = crate::MlxLoadRequest::from_normalized(
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
            let target = executable.erased_mut();
            assert_eq!(
                target
                    .retained_prediction_storage()
                    .unwrap()
                    .byte_bound()
                    .unwrap(),
                Some(0)
            );
            let initial = target.retained_target_storage().unwrap();
            let initial_bytes = initial
                .byte_bound()
                .unwrap()
                .unwrap_or_else(|| panic!("idle target storage is settled: {label}: {initial:?}"));
            let tokens = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
            let parts = [input::token_ids_part(&tokens).unwrap()];
            drop(
                target
                    .prefill(input::ModelInput::new(&parts), &stream)
                    .unwrap(),
            );
            for token in [4_u32, 5] {
                drop(
                    target
                        .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                        .unwrap(),
                );
                let current = target.retained_target_storage().unwrap();
                let bound = current.byte_bound().unwrap();
                assert!(
                    bound.unwrap_or_else(|| {
                        panic!("completed decode storage is settled: {label}: {current:?}")
                    }) > initial_bytes,
                    "cached state adds retained payload"
                );
            }
            let key = "model.layers.0.input_layernorm.weight";
            let location = target
                .prepared_parameter_slots()
                .iter()
                .find(|slot| slot.parameter.id.as_str() == key)
                .unwrap()
                .location
                .clone();
            struct Read<'a> {
                key: &'a str,
                value: Option<MlxTensor>,
            }
            impl eredu_nn::ParameterSlotVisitor<MlxTensor> for Read<'_> {
                fn visit_slot(
                    &mut self,
                    metadata: eredu_nn::ParameterMetadataView<'_>,
                    value: &MlxTensor,
                ) {
                    if metadata.id().as_str() == self.key {
                        self.value = Some(value.clone());
                    }
                }
            }
            let mut read = Read { key, value: None };
            assert!(target
                .with_parameter_slots(
                    &location,
                    &std::collections::BTreeSet::from([key.to_owned()]),
                    &mut |visit| {
                        visit(&mut read);
                        Ok(())
                    },
                    &stream
                )
                .unwrap());
            let original = read.value.unwrap();
            original.as_array().evaluated().unwrap();
            let replacement = MlxTensor::from(Array::from_slice(
                &vec![1.25f32; original.as_array().size()],
                original.as_array().shape(),
            ));
            let replacement_bytes = replacement
                .as_array()
                .allocation_info()
                .unwrap()
                .unwrap()
                .bytes() as u64;
            let source_reads = target
                .residency_report()
                .unwrap()
                .unwrap()
                .weight_store()
                .physical_reads;
            let checkpoint = root.path().join("model.safetensors");
            let hidden_checkpoint = root.path().join("held.safetensors");
            std::fs::rename(&checkpoint, &hidden_checkpoint).unwrap();
            let mut before = target.retained_target_module_storage().unwrap();
            let full_before = target.retained_target_storage().unwrap();
            let full_before_bytes = full_before.byte_bound().unwrap().unwrap();
            let before_bytes = before.byte_bound().unwrap().unwrap_or_else(|| {
                panic!("idle target parameters have settled backing: {label}: {before:?}")
            });
            assert!(before_bytes > 0);
            assert!(target
                .publish_parameter_replacements(
                    &std::collections::BTreeMap::from([(key.to_owned(), replacement.clone())]),
                    true
                )
                .unwrap());
            let mut after = target.retained_target_module_storage().unwrap();
            let mut full_after = target.retained_target_storage().unwrap();
            assert_eq!(
                full_after.byte_bound().unwrap(),
                Some(full_before_bytes + replacement_bytes)
            );
            full_after
                .merge(target.retained_target_module_storage().unwrap())
                .unwrap();
            assert_eq!(
                full_after.byte_bound().unwrap(),
                Some(full_before_bytes + replacement_bytes),
                "target parameters alias the manager and replacement inventories"
            );
            let after_bytes = after.byte_bound().unwrap().unwrap();
            after.include_array(replacement.as_array()).unwrap();
            assert_eq!(
                after.byte_bound().unwrap(),
                Some(after_bytes),
                "current replacement is already inventoried"
            );
            before.include_array(replacement.as_array()).unwrap();
            assert_eq!(
                before.byte_bound().unwrap(),
                Some(before_bytes + replacement_bytes),
                "old snapshot retains the original owners"
            );
            assert!(target
                .publish_parameter_replacements(
                    &std::collections::BTreeMap::from([(key.to_owned(), original)]),
                    false
                )
                .unwrap());
            assert_eq!(
                target
                    .retained_target_module_storage()
                    .unwrap()
                    .byte_bound()
                    .unwrap(),
                Some(before_bytes)
            );
            assert_eq!(
                target
                    .retained_target_storage()
                    .unwrap()
                    .byte_bound()
                    .unwrap(),
                Some(full_before_bytes)
            );
            assert_eq!(
                target
                    .residency_report()
                    .unwrap()
                    .unwrap()
                    .weight_store()
                    .physical_reads,
                source_reads
            );
            std::fs::rename(hidden_checkpoint, checkpoint).unwrap();
            drop(executable);
            assert_eq!(after.byte_bound().unwrap(), Some(after_bytes));
        }
    }
}
