// Nonzero format fixtures use explicit block encodings and an independent
// arithmetic expansion. No Eredu quantizer/dequantizer constructs the oracle.
#[test]
fn k2_packed_gguf_banks_match_expanded_weights_after_eviction() {
    use eredu_gguf::GgmlType;
    let (stream, weights_stream) = execution_streams();
    for format in [GgmlType::Q4_0, GgmlType::MxFp4, GgmlType::IQ4NL] {
        let root = crate::tests::support::k2_horizon::gguf(format);
        let packed_path = root.path().join("packed.gguf");
        let expanded_path = root.path().join("expanded.gguf");
        let host = eredu_runtime::LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
        );
        let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 20, 1 << 20, 1, 1).unwrap();
        let banks = eredu_runtime::ParameterBankLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(16384), Some(1 << 20), 1).unwrap(),
            16384,
            16384,
        )
        .unwrap();
        let mut baseline: Option<Vec<Vec<f32>>> = None;
        for (path, residency) in [
            (
                &expanded_path,
                eredu_runtime::WeightResidency::fully_resident(),
            ),
            (
                &packed_path,
                eredu_runtime::WeightResidency::fully_resident(),
            ),
            (
                &packed_path,
                eredu_runtime::WeightResidency::with_independent_parameter_banks(
                    eredu_runtime::OrdinaryWeightResidency::FullyResident,
                    banks,
                ),
            ),
            (
                &packed_path,
                eredu_runtime::WeightResidency::with_independent_parameter_banks(
                    eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(host),
                    banks,
                ),
            ),
            (
                &packed_path,
                eredu_runtime::WeightResidency::with_independent_parameter_banks(
                    eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(disk),
                    banks,
                ),
            ),
        ] {
            let options = crate::MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
            );
            let inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
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
                assert!(report.peak_device_resident_bytes() <= 16384);
                assert!(report.incremental().device().evictions() > 0);
            }
            if let Some(expected) = &baseline {
                for (actual, expected) in outputs.iter().flatten().zip(expected.iter().flatten()) {
                    assert!(
                        (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                        "{format:?} {residency:?}: {actual} != {expected}"
                    );
                }
            } else {
                baseline = Some(outputs);
            }
        }
    }
}
