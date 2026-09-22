#[test]
fn retained_prediction_storage_covers_family_owners_and_unloaded_replacements() {
    use eredu_core::{
        capture::CaptureUsage,
        parameters::{ParameterBackend, ParameterRegion},
    };
    use eredu_runtime::parameter_operations::PreparedParameterLocation;
    if !crate::tests::support::native_process::enter("prepared prediction parameter storage") {
        return;
    }
    let physical = crate::tests::support::test_utils::initialize_original_sources();

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
    for (family, write) in fixtures {
        for residency in 0..3 {
            let label = format!("{family} residency={residency}");
            eprintln!("checking {label}");
            let root = tempfile::tempdir().unwrap();
            let checkpoint = root.path().join("checkpoint");
            std::fs::create_dir(&checkpoint).unwrap();
            write(&checkpoint);
            let placement = match residency {
                0 => eredu_core::ResidencyPlan::FullyResident,
                1 => eredu_core::ResidencyPlan::LayerwiseHost {
                    device_layer_window: 1,
                    device_budget_bytes: Some(1 << 30),
                    host_budget_bytes: Some(1 << 30),
                },
                2 => eredu_core::ResidencyPlan::DenseDiskStream {
                    device_budget_bytes: 1 << 30,
                    host_budget_bytes: 0,
                    host_lookahead: 0,
                    background_queue: 0,
                },
                _ => unreachable!(),
            };
            let plan = eredu_core::ExecutionPlan::fully_resident(
                eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap(),
            )
            .with_residency(placement)
            .with_drafting(eredu_core::DraftingPlan::Embedded {
                max_draft_tokens: 1,
                lookahead: false,
                adaptive_lookahead: false,
            });
            let inspection =
                eredu_architectures::configuration::inspect_artifact_with_prepared_gguf_headers(
                    &checkpoint,
                )
                .unwrap();
            let factory = crate::MlxBackendFactory::default();
            let selected =
                eredu_core::select_execution_plan_target(&factory, &plan, inspection).unwrap();
            let target =
                eredu_core::realize_execution_plan_target(&factory, &plan, selected).unwrap();
            let mut runtime = target.into_runtime().unwrap();
            let target = runtime.session().original_model_source().unwrap().erased();
            assert!(target.has_embedded_prediction(), "{label}");
            let key = target
                .prepared_parameter_slots()
                .iter()
                .find(|slot| {
                    matches!(slot.location, PreparedParameterLocation::Prediction { .. })
                        && slot.parameter.id.as_str().ends_with("weight")
                        && !slot.parameter.id.as_str().contains("norm")
                })
                .expect("prediction projection parameter")
                .parameter
                .id
                .as_str()
                .to_owned();
            let discovery = MlxBackend::parameter_discovery(&mut runtime).unwrap();
            let row = discovery
                .parameters
                .iter()
                .find(|row| row.id == key)
                .unwrap();
            let region = ParameterRegion {
                starts: vec![0; row.shape.len()],
                shape: row.shape.clone(),
            };
            let limits = CaptureUsage {
                captures: u64::MAX,
                retained_bytes: u64::MAX,
                host_bytes: u64::MAX,
                encoded_bytes: u64::MAX,
            };
            let queried = MlxBackend::query_parameter(
                &mut runtime,
                &discovery.identity,
                &key,
                region,
                limits,
            )
            .unwrap_or_else(|cause| panic!("{label} {key}: {cause:#?}"));
            assert!(
                queried.values.iter().any(|&value| value != 0.),
                "{label}: nonzero source loan"
            );
            let alias = queried.clone();
            let shape = queried
                .region
                .shape
                .iter()
                .map(|&n| i32::try_from(n).unwrap())
                .collect::<Vec<_>>();
            let original = MlxTensor::from_array(Array::from_slice(&queried.values, &shape));
            drop(queried);
            assert!(alias.values.iter().any(|&value| value != 0.));
            let (backend, session) = runtime.parts_mut();
            let stream = backend.stream();
            let funding = physical
                .prepare_workspace_metadata(
                    session
                        .original_model_source()
                        .unwrap()
                        .erased()
                        .inference_execution_identity(),
                    physical.configured_limits().clone(),
                )
                .unwrap();
            let (
                after,
                combined,
                unknown,
                lazy,
                replacement,
                replacement_charge,
                prediction_charge,
                target_charge,
                pool,
                bytes,
                combined_bytes,
                replacement_bytes,
                combined_physical,
                replacement_physical,
            ) = session
                .with_model_operation_funded(funding, |executable| {
                    let target = executable.erased_mut();
                    let replacement = MlxTensor::from_array(Array::from_slice(
                        &vec![1.25f32; original.as_array().size()],
                        original.as_array().shape(),
                    ));
                    let replacement_facts =
                        replacement.as_array().allocation_info().unwrap().unwrap();
                    let replacement_bytes = u64::try_from(replacement_facts.bytes()).unwrap();
                    let replacement_physical = replacement_bytes
                        .checked_add(u64::try_from(replacement_facts.host_control_bytes()).unwrap())
                        .unwrap();
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
                    // This ownership fixture includes publication and native
                    // controls in a finite physical ceiling. Exact-fit refusal
                    // is covered by the domain admission conformance matrix.
                    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
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
                    let separate_charge = pool.fixture_host_charge().unwrap();
                    let mut union = target.retained_target_storage().unwrap();
                    union
                        .merge(target.retained_prediction_storage().unwrap())
                        .unwrap();
                    let union = union.register(&pool).unwrap();
                    let combined_physical = union.bytes().unwrap();
                    assert_eq!(
                        separate_charge, combined_physical,
                        "{label}: separate inventories share backing and native controls"
                    );
                    assert_eq!(
                        pool.fixture_host_charge().unwrap(),
                        combined_physical,
                        "{label}: registering the complete union adds no physical charge"
                    );
                    assert!(combined_physical >= combined_bytes);
                    drop(union);
                    crate::backend::ordinary_retirement::reclaim_all();
                    crate::memory_fixture::publish_model_parameters(
                        target,
                        [(key.clone(), replacement.clone())],
                        true,
                    );
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
                        pool.fixture_host_charge().unwrap(),
                        combined_physical + replacement_physical
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
                    crate::memory_fixture::publish_model_parameters(
                        target,
                        [(key.clone(), MlxTensor::from_array(lazy.clone()))],
                        true,
                    );
                    let unknown = target.retained_prediction_storage().unwrap();
                    assert!(target
                        .retained_prediction_storage()
                        .unwrap()
                        .register(&pool)
                        .is_err());
                    assert_eq!(
                        pool.fixture_host_charge().unwrap(),
                        combined_physical + replacement_physical
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
                    crate::memory_fixture::publish_model_parameters(
                        target,
                        [(key, original)],
                        false,
                    );
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
                    drop(before);
                    Ok((
                        after,
                        combined,
                        unknown,
                        lazy,
                        replacement,
                        replacement_charge,
                        prediction_charge,
                        target_charge,
                        pool,
                        bytes,
                        combined_bytes,
                        replacement_bytes,
                        combined_physical,
                        replacement_physical,
                    ))
                })
                .unwrap();
            drop(runtime);
            drop(replacement);
            assert_eq!(
                after.byte_bound().unwrap(),
                Some(bytes + replacement_bytes),
                "{label}: snapshot retains physical owners"
            );
            assert_eq!(unknown.byte_bound().unwrap(), None);
            drop((after, combined, unknown, lazy));
            assert_eq!(pool.fixture_host_charge().unwrap(), combined_physical + replacement_physical,
                "{label}: registrations retain their physical inventories after executable destruction");
            let peak = pool.fixture_host_peak().unwrap();
            assert!(
                peak > combined_physical + replacement_physical,
                "{label}: historical peak includes publication bookkeeping"
            );
            drop(replacement_charge);
            crate::backend::ordinary_retirement::reclaim_all();
            assert_eq!(pool.fixture_host_charge().unwrap(), combined_physical);
            drop(prediction_charge);
            crate::backend::ordinary_retirement::reclaim_all();
            assert_eq!(
                pool.fixture_host_charge().unwrap(),
                target_charge.bytes().unwrap()
            );
            drop(target_charge);
            crate::backend::ordinary_retirement::reclaim_all();
            assert_eq!(pool.fixture_host_charge().unwrap(), 0);
            assert_eq!(pool.fixture_host_peak().unwrap(), peak);
        }
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
