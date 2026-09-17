//! Real sparse callback, transport and delivery through the public shared driver.
use super::*;
use eredu_core::checkpoint::TensorDtype;
fn fixture_source() -> Fixture {
    crate::routed_components::routed_fixture()
}
fn selected(paths: &[String; 2], world: usize) -> CapturePlan {
    let mut plan = CapturePlan::none();
    for (index, path) in paths.iter().enumerate() {
        plan.selections.push(CaptureSelection {
            id: format!("distributed sparse {index}"),
            path: path.clone(),
            schedule: CaptureSchedule::default(),
            transform: CaptureTransform::RoutedUnits,
            slices: vec![CaptureSlice {
                axis: "component".into(),
                start: 1,
                end: 6,
                stride: 2,
            }],
        });
    }
    let factor = (world as u64).checked_mul(world as u64).unwrap();
    plan.limits.per_step = CaptureUsage {
        captures: 2 * (2 + world as u64),
        retained_bytes: 4 * factor << 20,
        host_bytes: 32 * factor << 20,
        encoded_bytes: factor << 20,
    };
    plan.limits.cumulative = CaptureUsage {
        captures: plan.limits.per_step.captures * 4,
        retained_bytes: plan.limits.per_step.retained_bytes * 4,
        host_bytes: plan.limits.per_step.host_bytes * 4,
        encoded_bytes: plan.limits.per_step.encoded_bytes * 4,
    };
    plan
}
fn loaded<const TP: usize, const PP: usize, const EP: usize>(
    mode: &str,
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
    partitioned: bool,
) -> serde_json::Value {
    let descriptor = inspect_architecture(&root.0).unwrap();
    let bank = &descriptor.routed_components[0];
    let paths = [bank.activation.clone(), bank.effective_activation.clone()];
    let world = TP * PP * EP;
    capture_loaded_plan_with_capacity(
        mode,
        model,
        root,
        partitioned,
        Some(16u64 << 30),
        |discovery| {
            for path in &paths {
                let point = discovery
                    .catalog
                    .points
                    .iter()
                    .find(|p| &p.path == path)
                    .unwrap();
                assert!(matches!(
                    point.value_type,
                    eredu_core::ObservationValueType::RoutedUnits { .. }
                ));
                assert!(point
                    .axes
                    .as_ref()
                    .unwrap()
                    .iter()
                    .any(|axis| axis.name == "component"));
            }
            selected(&paths, world)
        },
        |ids, text, frames, partitioned| evaluate(ids, text, frames, partitioned, EP),
    )
}
fn ordinary<const TP: usize, const PP: usize, const EP: usize>(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
) -> serde_json::Value {
    loaded::<TP, PP, EP>("ordinary", model, root, true)
}
fn managed<const TP: usize, const PP: usize, const EP: usize>(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
) -> serde_json::Value {
    loaded::<TP, PP, EP>("managed", model, root, true)
}
fn controlled<const TP: usize, const PP: usize, const EP: usize>(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
) -> serde_json::Value {
    loaded::<TP, PP, EP>("controlled", model, root, true)
}
fn run<const TP: usize, const PP: usize, const EP: usize>(mode: &str) -> serde_json::Value {
    if mode == "serial" {
        let root = managed_fixture(fixture_source());
        let execution =
            ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
        let (model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap()
                .into_parts();
        return loaded::<TP, PP, EP>("ordinary", model, root, false);
    }
    let worker = match mode {
        "ordinary" => ordinary::<TP, PP, EP>,
        "managed" => managed::<TP, PP, EP>,
        "controlled" => controlled::<TP, PP, EP>,
        _ => panic!("capture mode"),
    };
    run_partitioned_with_fixture(
        mode,
        eredu_core::ParallelTopology::new(TP, PP, EP, 1).unwrap(),
        Some(worker),
        fixture_source,
    )
}
fn evaluate(
    ids: &[u32],
    text: &str,
    frames: &[&CapturedStep],
    partitioned: bool,
    ep: usize,
) -> serde_json::Value {
    assert_eq!(ids.len(), 4);
    assert_eq!(frames.len(), 4);
    let mut output = Vec::new();
    for (prediction, frame) in frames.iter().enumerate() {
        assert_eq!(frame.outcome, CaptureStepOutcome::Committed);
        assert_eq!(frame.prediction_index, prediction as u64);
        assert_eq!(frame.records.len(), 2);
        assert_eq!(frame.partitions.len(), if partitioned { 2 } else { 0 });
        for (index, evidence) in frame.partitions.iter().enumerate() {
            evidence.context.validate().unwrap();
            assert_eq!(evidence.context.selection_index, index);
            assert_eq!(evidence.context.prediction, prediction as u64);
            assert_eq!(evidence.combination, PartitionCaptureCombination::Disjoint);
            assert!(!evidence.producers.is_empty());
            assert!(!evidence.contributions.is_empty());
            for contribution in &evidence.contributions {
                assert!(evidence.producers.contains(&contribution.producer_rank));
                let routed = contribution
                    .routed
                    .as_ref()
                    .expect("actual sparse ownership and native ranges");
                assert_eq!(routed.ownership.source_peer.is_some(), ep > 1);
                let mut end = 0;
                for &[start, next] in &routed.source_token_ranges {
                    assert_eq!(start, end);
                    assert!(next > start);
                    end = next;
                }
                if ep == 1 {
                    assert_eq!(end, if prediction == 0 { 5 } else { 1 });
                }
            }
        }
        let mut rows = Vec::new();
        for record in &frame.records {
            assert!(matches!(record.outcome, CaptureOutcome::Captured));
            assert_eq!(record.source_dtype, Some(TensorDtype::F32));
            let Some(CapturePayload::RoutedUnits(payload)) = &record.payload else {
                panic!("sparse payload")
            };
            assert_eq!(payload.rows.len(), if prediction == 0 { 20 } else { 4 });
            if partitioned {
                assert!(payload.source_token_ranges.is_empty());
            }
            let mut ordered: Vec<_> = payload.rows.iter().collect();
            ordered.sort_unstable_by_key(|row| (row.token, row.slot, row.expert));
            let mut identities = Vec::new();
            let mut values = Vec::new();
            for row in ordered {
                identities.push([
                    row.token,
                    row.slot,
                    row.expert,
                    row.unit_start,
                    row.unit_stride,
                ]);
                assert_eq!((row.unit_start, row.unit_stride), (1, 2));
                let TensorObservationData::F32(units) = row.values.data() else {
                    panic!("floating sparse values")
                };
                assert_eq!(units.len(), 3);
                assert!(units.iter().all(|value| value.is_finite()));
                assert!(units.iter().any(|value| value.abs() > 1e-6));
                values.push(row.coefficient);
                values.extend_from_slice(units);
            }
            rows.push(
                serde_json::json!({"id":record.selection_id,"shape":record.selected_shape,
                "payload_shape":null,"outcome":record.outcome,"counts":identities,"values":values}),
            );
        }
        output.push(serde_json::json!({"prediction":prediction,"rows":rows}));
    }
    serde_json::json!({"ids":ids,"text":text,"frames":output})
}

#[test]
#[ignore = "requires Metal and 2 local Ring processes"]
fn native_routed_tensor_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by("managed_plain::parallel::capture::routed::native_routed_tensor_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_ROUTED_TENSOR_MODE","PUBLIC_ROUTED_TENSOR_RESULT:",
        "sparse tensor capture",2,&["ordinary","managed","controlled"],run::<2,1,1>,super::compare)
}

#[test]
#[ignore = "requires Metal and 2 local Ring processes"]
fn native_routed_pipeline_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by("managed_plain::parallel::capture::routed::native_routed_pipeline_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_ROUTED_PIPELINE_MODE","PUBLIC_ROUTED_PIPELINE_RESULT:",
        "sparse pipeline capture",2,&["ordinary","managed","controlled"],run::<1,2,1>,super::compare)
}

#[test]
#[ignore = "requires Metal and 4 local Ring processes"]
fn native_routed_combined_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by("managed_plain::parallel::capture::routed::native_routed_combined_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_ROUTED_COMBINED_MODE","PUBLIC_ROUTED_COMBINED_RESULT:",
        "sparse combined capture",4,&["ordinary","managed","controlled"],run::<2,2,1>,super::compare)
}

#[test]
#[ignore = "requires Metal and 2 local Ring processes"]
fn native_routed_expert_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by("managed_plain::parallel::capture::routed::native_routed_expert_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_ROUTED_EXPERT_MODE","PUBLIC_ROUTED_EXPERT_RESULT:",
        "sparse expert capture",2,&["ordinary","managed","controlled"],run::<1,1,2>,super::compare)
}

#[test]
#[ignore = "requires Metal and 4 local Ring processes"]
fn native_routed_tensor_expert_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by("managed_plain::parallel::capture::routed::native_routed_tensor_expert_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_ROUTED_TENSOR_EXPERT_MODE","PUBLIC_ROUTED_TENSOR_EXPERT_RESULT:",
        "sparse tensor_expert capture",4,&["ordinary","managed","controlled"],run::<2,1,2>,super::compare)
}

#[test]
#[ignore = "requires Metal and 4 local Ring processes"]
fn native_routed_pipeline_expert_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by("managed_plain::parallel::capture::routed::native_routed_pipeline_expert_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_ROUTED_PIPELINE_EXPERT_MODE","PUBLIC_ROUTED_PIPELINE_EXPERT_RESULT:",
        "sparse pipeline_expert capture",4,&["ordinary","managed","controlled"],run::<1,2,2>,super::compare)
}

#[test]
#[ignore = "requires Metal and 8 local Ring processes"]
fn native_routed_combined_expert_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by("managed_plain::parallel::capture::routed::native_routed_combined_expert_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_ROUTED_COMBINED_EXPERT_MODE","PUBLIC_ROUTED_COMBINED_EXPERT_RESULT:",
        "sparse combined_expert capture",8,&["ordinary","managed","controlled"],run::<2,2,2>,super::compare)
}
