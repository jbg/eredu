#[test]
fn retained_prediction_storage_covers_family_owners_and_unloaded_replacements() {
    use eredu_runtime::parameter_operations::PreparedParameterLocation;

    fn write_nonzero_inkling(path: &Path) {
        write_inkling_mtp_fixture(path);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let file = path.join("model.safetensors");
        let tensors = Array::load_safetensors(&file, &stream).unwrap();
        let tensors = tensors
            .into_iter()
            .map(|(name, original)| {
                let values = (0..original.size())
                    .map(|i| {
                        if name.contains("norm") {
                            0.95 + (i % 7) as f32 * 0.01
                        } else {
                            0.01 + (i % 11) as f32 * 0.002
                        }
                    })
                    .collect::<Vec<_>>();
                let value = Array::from_slice(&values, original.shape())
                    .as_dtype(original.dtype(), &stream)
                    .unwrap();
                (name, value)
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            tensors.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            &file,
        )
        .unwrap();
    }

    let fixtures: [(&str, fn(&Path)); 7] = [
        ("v3", |path| {
            write_deepseek_fixture_with_prediction(path, 2, 1)
        }),
        ("v4", |path| write_deepseek_v4_fixture(path, 2)),
        ("dspark", |path| {
            write_deepseek_v4_dspark_fixture(path, false)
        }),
        ("inkling", write_nonzero_inkling),
        ("qwen", |path| write_qwen35_multimodal_fixture(path, false)),
        ("qwen-moe", |path| {
            write_qwen35_multimodal_fixture(path, true)
        }),
        ("nemotron", write_nemotron_mtp_fixture),
    ];
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let backend = crate::native::backend(&stream, &stream);
    let host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 26), Some(1 << 26), 1).unwrap(),
    );
    for (family, write) in fixtures {
        for residency in [
            eredu_runtime::WeightResidency::fully_resident(),
            eredu_runtime::WeightResidency::layerwise_host(host),
            eredu_runtime::WeightResidency::dense_disk_stream(Default::default()),
        ] {
            let label = format!("{family} {residency:?}");
            eprintln!("checking {label}");
            let root = tempfile::tempdir().unwrap();
            let checkpoint = root.path().join("checkpoint");
            std::fs::create_dir(&checkpoint).unwrap();
            write(&checkpoint);
            let request = MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
            );
            let model = load_model(&backend, &checkpoint, request)
                .unwrap_or_else(|error| panic!("{label}: {error}"))
                .into_inner();
            let mut executable = model.into_executable();
            let target = executable.erased_mut();
            assert!(target.has_embedded_prediction(), "{label}");
            let slot = target
                .prepared_parameter_slots()
                .iter()
                .find(|slot| {
                    matches!(slot.location, PreparedParameterLocation::Prediction { .. })
                        && slot.parameter.id.as_str().ends_with("weight")
                        && !slot.parameter.id.as_str().contains("norm")
                })
                .expect("prediction projection parameter");
            let key = slot.parameter.id.as_str().to_owned();
            let location = slot.location.clone();
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
            let mut read = Read {
                key: &key,
                value: None,
            };
            assert!(target
                .with_parameter_slots(
                    &location,
                    &std::collections::BTreeSet::from([key.clone()]),
                    &mut |visit| {
                        visit(&mut read);
                        Ok(())
                    },
                    &stream,
                )
                .unwrap());
            let original = read.value.unwrap();
            assert!(
                original
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .iter()
                    .any(|v| *v != 0.),
                "{label}: loan must bind nonzero checkpoint parameter {key}"
            );
            let replacement = MlxTensor::from_array(Array::from_slice(
                &vec![1.25f32; original.as_array().size()],
                original.as_array().shape(),
            ));
            let replacement_bytes = replacement
                .as_array()
                .allocation_info()
                .unwrap()
                .unwrap()
                .bytes() as u64;
            let reads = target
                .residency_report()
                .unwrap()
                .unwrap()
                .weight_store()
                .physical_reads;
            std::fs::rename(&checkpoint, root.path().join("moved")).unwrap();
            let before = target.retained_prediction_storage().unwrap();
            let bytes = before
                .byte_bound()
                .unwrap()
                .unwrap_or_else(|| panic!("{label}: {before:?}"));
            assert!(bytes > 0, "{label}");
            let mut combined = target.retained_target_storage().unwrap();
            combined
                .merge(target.retained_prediction_storage().unwrap())
                .unwrap();
            let combined_bytes = combined
                .byte_bound()
                .unwrap()
                .unwrap_or_else(|| panic!("{label}: {combined:?}"));
            combined
                .merge(target.retained_prediction_storage().unwrap())
                .unwrap();
            assert_eq!(
                combined.byte_bound().unwrap(),
                Some(combined_bytes),
                "{label}: aliases count once"
            );
            let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(
                combined_bytes + replacement_bytes,
                0,
            )
            .unwrap();
            let target_charge = target
                .retained_target_storage()
                .unwrap()
                .register(&pool)
                .unwrap();
            let prediction_charge = target
                .retained_prediction_storage()
                .unwrap()
                .register(&pool)
                .unwrap();
            assert_eq!(
                pool.used_bytes().unwrap(),
                combined_bytes,
                "{label}: separate target and prediction registrations share physical charges"
            );
            assert!(target
                .publish_parameter_replacements(
                    &std::collections::BTreeMap::from([(key.clone(), replacement.clone())]),
                    true,
                )
                .unwrap());
            let mut combined_after = target.retained_target_storage().unwrap();
            combined_after
                .merge(target.retained_prediction_storage().unwrap())
                .unwrap();
            assert_eq!(
                combined_after.byte_bound().unwrap(),
                Some(combined_bytes + replacement_bytes),
                "{label}: target and prediction share the replacement owner"
            );
            let replacement_charge = combined_after.register(&pool).unwrap();
            assert_eq!(
                pool.used_bytes().unwrap(),
                combined_bytes + replacement_bytes
            );
            let mut after = target.retained_prediction_storage().unwrap();
            assert_eq!(
                after.byte_bound().unwrap(),
                Some(bytes + replacement_bytes),
                "{label}: unloaded replacement retained"
            );
            after.include_array(replacement.as_array()).unwrap();
            assert_eq!(
                after.byte_bound().unwrap(),
                Some(bytes + replacement_bytes),
                "{label}: replacement alias"
            );
            let lazy = replacement.as_array().square(&stream).unwrap();
            assert!(target
                .publish_parameter_replacements(
                    &std::collections::BTreeMap::from([(
                        key.clone(),
                        MlxTensor::from_array(lazy.clone())
                    )]),
                    true,
                )
                .unwrap());
            let unknown = target.retained_prediction_storage().unwrap();
            assert!(target
                .retained_prediction_storage()
                .unwrap()
                .register(&pool)
                .is_err());
            assert_eq!(
                pool.used_bytes().unwrap(),
                combined_bytes + replacement_bytes
            );
            assert_eq!(
                unknown.byte_bound().unwrap(),
                None,
                "{label}: unevaluated override is unknown"
            );
            assert_eq!(
                lazy.allocation_info().unwrap(),
                None,
                "{label}: inspection cannot evaluate"
            );
            assert!(target
                .publish_parameter_replacements(
                    &std::collections::BTreeMap::from([(key, original)]),
                    false,
                )
                .unwrap());
            assert_eq!(
                target
                    .retained_prediction_storage()
                    .unwrap()
                    .byte_bound()
                    .unwrap(),
                Some(bytes),
                "{label}: original owners restored"
            );
            assert_eq!(
                target
                    .residency_report()
                    .unwrap()
                    .unwrap()
                    .weight_store()
                    .physical_reads,
                reads,
                "{label}: no hidden checkpoint reads"
            );
            drop(executable);
            drop(replacement);
            assert_eq!(
                after.byte_bound().unwrap(),
                Some(bytes + replacement_bytes),
                "{label}: snapshot retains physical owners"
            );
            assert_eq!(unknown.byte_bound().unwrap(), None);
            drop((before, after, combined, unknown, lazy));
            assert_eq!(pool.used_bytes().unwrap(), combined_bytes + replacement_bytes,
                "{label}: registrations retain their physical inventories after executable destruction");
            drop(replacement_charge);
            crate::backend::ordinary_retirement::reclaim_all();
            assert_eq!(pool.used_bytes().unwrap(), combined_bytes);
            drop(prediction_charge);
            crate::backend::ordinary_retirement::reclaim_all();
            assert_eq!(pool.used_bytes().unwrap(), target_charge.bytes());
            drop(target_charge);
            crate::backend::ordinary_retirement::reclaim_all();
            assert_eq!(pool.used_bytes().unwrap(), 0);
            assert_eq!(
                pool.peak_bytes().unwrap(),
                combined_bytes + replacement_bytes
            );
        }
    }
}
