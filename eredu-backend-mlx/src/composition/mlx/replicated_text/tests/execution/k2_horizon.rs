#[test]
fn k2_mova_native_prepared_banks_residency_paged_cache_and_rollback_match() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eredu-architectures/tests/fixtures/k2_horizon/reference.json"
    )))
    .unwrap();
    let root = tiny_heterogeneous_artifact(fixture["mova"]["config"].clone());
    let (stream, weights_stream) = execution_streams();
    let host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
    );
    let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 20, 2 << 20, 1, 1).unwrap();
    let bank_options = eredu_runtime::ParameterBankLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1152), Some(1 << 20), 1).unwrap(),
        1152,
        1152,
    )
    .unwrap();
    let mut baseline = None;
    for (ordinary, addressable) in [
        (eredu_runtime::OrdinaryWeightResidency::FullyResident, false),
        (eredu_runtime::OrdinaryWeightResidency::FullyResident, true),
        (
            eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(host),
            true,
        ),
        (
            eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(disk),
            true,
        ),
    ] {
        let residency = if addressable {
            eredu_runtime::WeightResidency::with_independent_parameter_banks(ordinary, bank_options)
        } else {
            eredu_runtime::WeightResidency::fully_resident()
        };
        let options = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default()
                .with_weight_residency(residency)
                .with_state_residency(CacheResidencyPolicy::Paged(
                    PagedCacheOptions::new(2, 768, 1 << 20, 1)
                        .unwrap()
                        .with_full_attention(true),
                )),
        );
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.normalized().preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let prefix = [1_u32, 3, 2];
        let prompt = Array::from_slice(&prefix, &[1, 3]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        let prefill = generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let saved = generic.state_snapshot();
        assert!(saved.iter().all(|(position, _)| *position == 3));
        let continuation = Array::from_slice(&[4_u32], &[1, 1]);
        let probe = generic
            .checkpoint_restore_probe(&continuation, &stream)
            .unwrap();
        assert_eq!(probe.0, probe.2);
        assert_eq!(probe.3, probe.5);
        assert_eq!(generic.state_snapshot(), saved);
        let descriptor = PromptCacheDescriptor::from_model_identity(
            generic.prompt_cache_model_identity().clone(),
            "k2-fixture",
            "tokens:1,3,2",
            1,
        )
        .unwrap();
        let cache_root = tempfile::tempdir().unwrap();
        let destination = cache_root.path().join("cache");
        generic
            .save_prompt_cache(
                &destination,
                descriptor.clone(),
                &prefix,
                &PromptCacheOptions::default(),
            )
            .unwrap();
        let mut outputs = vec![prefill];
        for token in [4_u32, 5, 6, 7] {
            outputs.push(
                generic
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec(),
            );
        }
        assert!(generic
            .state_snapshot()
            .iter()
            .all(|(position, _)| *position == 7));
        let cache_report = generic.cache_residency_report().unwrap().unwrap();
        assert!(cache_report.peak_device_bytes <= 768);
        assert!(cache_report.host_demotions > 0);
        assert!(cache_report.host_promotions > 0);
        generic.reset_cache().unwrap();
        generic
            .load_prompt_cache(&destination, &descriptor, &prefix)
            .unwrap();
        let restored = generic
            .decode(&continuation, &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(restored, outputs[1]);
        if addressable {
            let report = generic.parameter_bank_report().unwrap().unwrap();
            assert_eq!(report.banks().len(), 2);
            assert!(report.peak_device_resident_bytes() <= 1152);
            assert!(report.device_resident_bytes() <= 1152);
            assert!(report.peak_host_resident_bytes() <= 1 << 20);
            assert!(report.incremental().device().evictions() > 0);
            for (id, bank) in report.banks() {
                assert!(bank.bulk().distinct_entries() > 0);
                assert!(bank.incremental().distinct_entries() > 0);
                assert!(bank
                    .placements()
                    .iter()
                    .all(|(key, _)| key.bank() == id.value() as usize));
            }
        }
        assert!(outputs.iter().flatten().all(|x| x.is_finite()));
        assert!(outputs.iter().flatten().any(|x| *x != 0.0));
        if let Some(expected) = &baseline {
            let expected: &Vec<Vec<f32>> = expected;
            for (actual, expected) in outputs.iter().flatten().zip(expected.iter().flatten()) {
                assert!((actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs());
            }
        } else {
            baseline = Some(outputs);
        }
    }
}

#[test]
fn k2_mova_native_control_forks_and_bank_interventions_preserve_state_and_accounting() {
    use eredu_nn::routing_intervention::{GroupSelectionAction, GroupSelectionControl};
    use std::collections::BTreeMap;

    struct Observer {
        stream: Stream,
        control: Option<(String, GroupSelectionControl)>,
        decisions: BTreeMap<String, (Vec<u32>, Vec<f32>)>,
        applied: Vec<String>,
    }
    impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
        fn observe(&mut self, _: &str, _: &Array) -> Result<(), Exception> {
            Ok(())
        }
        fn routing_control(
            &mut self,
            path: &str,
            rows: u64,
        ) -> Result<Option<GroupSelectionControl>, Exception> {
            assert_eq!(rows, 1);
            Ok(self
                .control
                .as_ref()
                .filter(|(target, _)| target == path)
                .map(|(_, c)| c.clone()))
        }
        fn routing_applied(
            &mut self,
            path: &str,
            original: Option<eredu_runtime::RoutingDecision<'_, Array>>,
            _: eredu_runtime::RoutingDecision<'_, Array>,
        ) -> Result<(), Exception> {
            assert!(original.is_some());
            self.applied.push(path.to_owned());
            Ok(())
        }
        fn observe_routing(
            &mut self,
            routing: eredu_runtime::RoutingObservation<'_, Array>,
        ) -> Result<(), Exception> {
            let ids = routing
                .selected_experts
                .as_dtype(Dtype::Uint32, &self.stream)?
                .contiguous(false, &self.stream)?
                .evaluated()?
                .as_slice::<u32>()
                .to_vec();
            let weights = routing
                .coefficients
                .as_dtype(Dtype::Float32, &self.stream)?
                .contiguous(false, &self.stream)?
                .evaluated()?
                .as_slice::<f32>()
                .to_vec();
            assert!(weights.iter().all(|w| w.is_finite() && *w > 0.0));
            self.decisions
                .insert(routing.path.to_owned(), (ids, weights));
            Ok(())
        }
    }
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eredu-architectures/tests/fixtures/k2_horizon/reference.json"
    )))
    .unwrap();
    let config = fixture["mova"]["config"].clone();
    let args = eredu_architectures::k2_horizon::model_args_from_config_value(&config).unwrap();
    let root = tiny_heterogeneous_artifact(config);
    let (stream, weights_stream) = execution_streams();
    let bank_options = eredu_runtime::ParameterBankLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1152), Some(1 << 20), 1).unwrap(),
        1152,
        1152,
    )
    .unwrap();
    let mut baseline = None;
    for (addressable, paged) in [(false, false), (true, false), (false, true), (true, true)] {
        let residency = if addressable {
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                bank_options,
            )
        } else {
            eredu_runtime::WeightResidency::fully_resident()
        };
        let mut request =
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency);
        if paged {
            request = request.with_state_residency(CacheResidencyPolicy::Paged(
                PagedCacheOptions::new(2, 4096, 1 << 20, 1)
                    .unwrap()
                    .with_full_attention(true),
            ));
        }
        let options = crate::MlxLoadRequest::from_normalized(request);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.normalized().preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let prompt = Array::from_slice(&[1_u32, 3, 2], &[1, 3]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        assert!(matches!(
            generic.native_control_support(),
            eredu_core::execution_control::ControlSupport::Supported
        ));
        let mut saved = generic.capture_native_control_state().unwrap();
        let mut fork = generic.copy_native_control_state(saved.as_ref()).unwrap();
        let estimate = generic
            .estimate_native_control_state(Some(saved.as_ref()))
            .unwrap()
            .unwrap();
        if paged {
            assert!(
                estimate.retained_bytes >= 65536,
                "sealed block catalog is included"
            );
        }
        let mut outputs = Vec::new();
        let mut decisions = None;
        for token in [4_u32, 5, 6, 7] {
            let mut observer = Observer {
                stream: stream.clone(),
                control: None,
                decisions: BTreeMap::new(),
                applied: vec![],
            };
            let output = generic
                .forward_with_observer(
                    &Array::from_slice(&[token], &[1, 1]),
                    None,
                    &stream,
                    &mut observer,
                )
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            assert_eq!(observer.decisions.len(), 4);
            assert!(observer.applied.is_empty());
            if decisions.is_none() {
                decisions = Some(observer.decisions);
            }
            outputs.push(output);
        }
        let before_restore = generic
            .parameter_bank_report()
            .unwrap()
            .map(|r| r.incremental().device().requests());
        generic
            .exchange_native_control_state(saved.as_mut())
            .unwrap();
        assert!(generic
            .state_snapshot()
            .iter()
            .all(|(position, _)| *position == 3));
        assert_eq!(
            before_restore,
            generic
                .parameter_bank_report()
                .unwrap()
                .map(|r| r.incremental().device().requests())
        );
        let restored_prefix = generic
            .decode(&Array::from_slice(&[4_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(
            restored_prefix, outputs[0],
            "immediate restored prefix, paged={paged}, addressable={addressable}"
        );
        for (path, bank, count) in [
            (
                "model.layers.1.self_attn.values",
                eredu_architectures::k2_horizon::ExpertBank::AttentionValue,
                args.mova_num_experts,
            ),
            (
                "model.layers.1.mlp",
                eredu_architectures::k2_horizon::ExpertBank::FeedForward,
                args.num_experts,
            ),
        ] {
            // Each trial starts from its own copy of the same prefix. Force a
            // distinct expert set, using the bank's actual normalization policy.
            let mut trial = generic.copy_native_control_state(fork.as_ref()).unwrap();
            generic
                .exchange_native_control_state(trial.as_mut())
                .unwrap();
            let ids = &decisions.as_ref().unwrap()[path].0;
            let absent = (0..count as u32).find(|id| !ids.contains(id)).unwrap();
            let forced = vec![absent, ids[0]];
            let control = GroupSelectionControl {
                expected: args.routing_spec(bank).unwrap(),
                learned_coefficient_scale: false,
                first_row: 0,
                end_row: 1,
                row_stride: 1,
                action: GroupSelectionAction::Force(forced.clone()),
                capture_original: true,
            };
            let mut observer = Observer {
                stream: stream.clone(),
                control: Some((path.into(), control)),
                decisions: BTreeMap::new(),
                applied: vec![],
            };
            generic
                .forward_with_observer(
                    &Array::from_slice(&[4_u32], &[1, 1]),
                    None,
                    &stream,
                    &mut observer,
                )
                .unwrap()
                .evaluated()
                .unwrap();
            assert_eq!(observer.applied, [path]);
            let mut actual = observer.decisions[path].0.clone();
            actual.sort_unstable();
            let mut expected = forced;
            expected.sort_unstable();
            assert_eq!(actual, expected);
            assert_eq!(observer.decisions.len(), 4);
            if matches!(
                bank,
                eredu_architectures::k2_horizon::ExpertBank::AttentionValue
            ) {
                let sum: f32 = observer.decisions[path].1.iter().sum();
                assert!((sum - args.router_scaling_factor.unwrap_or(1.0)).abs() < 1e-5);
            }
        }
        let after_interventions = generic
            .parameter_bank_report()
            .unwrap()
            .map(|r| r.incremental().device().requests());
        generic
            .exchange_native_control_state(fork.as_mut())
            .unwrap();
        assert_eq!(
            after_interventions,
            generic
                .parameter_bank_report()
                .unwrap()
                .map(|r| r.incremental().device().requests())
        );
        if addressable {
            assert!(after_interventions.unwrap() > before_restore.unwrap());
        }
        for (index, token) in [4_u32, 5, 6, 7].into_iter().enumerate() {
            let restored = generic
                .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            assert_eq!(
                restored, outputs[index],
                "fork decode {index}, paged={paged}, addressable={addressable}"
            );
        }
        if let Some(expected) = &baseline {
            let expected: &Vec<Vec<f32>> = expected;
            for (a, b) in outputs.iter().flatten().zip(expected.iter().flatten()) {
                assert!((a - b).abs() <= 1e-5 + 1e-4 * b.abs());
            }
        } else {
            baseline = Some(outputs);
        }
    }
}

#[test]
fn k2_fp8_banks_and_excluded_parameters_match_publisher_dynamic_activation_reference() {
    let (stream, weights_stream) = execution_streams();
    assert_fp8_publisher_reference(stream, weights_stream);
}

#[test]
fn k2_fp8_cpu_banks_match_publisher_dynamic_activation_reference() {
    let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    assert_fp8_publisher_reference(
        Stream::new_with_device(&device),
        Stream::new_with_device(&device),
    );
}

fn assert_fp8_publisher_reference(stream: Stream, weights_stream: Stream) {
    let (_dense, encoded) = crate::tests::support::k2_horizon::fp8();

    let host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
    );
    let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 24, 1 << 24, 1, 1).unwrap();
    let bank_options = eredu_runtime::ParameterBankLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(393312), Some(1 << 24), 1).unwrap(),
        393312,
        393312,
    )
    .unwrap();
    let reference: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/validation/k2_fp8_reference.json"
    )))
    .unwrap();
    use sha2::Digest;
    let bytes = std::fs::read(encoded.path().join("model.safetensors")).unwrap();
    assert_eq!(
        sha2::Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
        reference["artifact_sha256"].as_str().unwrap()
    );
    let expected: Vec<Vec<f32>> = serde_json::from_value(reference["logits"].clone()).unwrap();
    for (root, residency) in [
        (&encoded, eredu_runtime::WeightResidency::fully_resident()),
        (
            &encoded,
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                bank_options,
            ),
        ),
        (
            &encoded,
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(host),
                bank_options,
            ),
        ),
        (
            &encoded,
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(disk),
                bank_options,
            ),
        ),
    ] {
        let options = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
        );
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.normalized().preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let prompt = Array::from_slice(&[1_u32, 3, 2], &[1, 3]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        let mut outputs = vec![generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec()];
        for token in [4_u32, 5, 6, 7] {
            outputs.push(
                generic
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec(),
            );
        }
        if let Some(report) = generic.parameter_bank_report().unwrap() {
            assert_eq!(report.banks().len(), 2);
            assert!(report.peak_device_resident_bytes() <= 393312);
            assert!(report.incremental().device().evictions() > 0);
        }
        assert_eq!(outputs.len(), expected.len());
        for (actual, expected) in outputs.iter().zip(&expected) {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                    "{residency:?}: {actual} != {expected}"
                );
            }
        }
    }
}
