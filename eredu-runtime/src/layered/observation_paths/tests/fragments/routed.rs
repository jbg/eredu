//! Original sparse batches through the actual shared funded observer.
use super::*;
fn sparse_source() -> SharedCapturePlan {
    sparse_source_with_decode(false)
}
fn sparse_source_with_decode(decode: bool) -> SharedCapturePlan {
    let ordinary = admitted(5, 100, CaptureLimitPolicy::Fail);
    let mut point = ordinary.admission().points()[0].clone();
    point.value_type = ObservationValueType::RoutedUnits {
        routing: "route".into(),
        geometry: RoutedUnitGeometry {
            experts: 7,
            units_per_expert: 5,
            routes_per_token: 3,
        },
    };
    point.axes = Some(vec![
        TensorAxis {
            name: "token".into(),
            dimension: SymbolicDimension::TokenRows,
        },
        TensorAxis {
            name: "route".into(),
            dimension: SymbolicDimension::Known(3),
        },
        TensorAxis {
            name: "unit".into(),
            dimension: SymbolicDimension::Known(5),
        },
    ]);
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let mut plan = ordinary.admission().plan().clone();
    plan.selections[0].schedule.decode = decode;
    plan.selections.truncate(1);
    plan.selections[0].transform = CaptureTransform::RoutedUnits;
    plan.selections[0].slices = vec![
        CaptureSlice {
            axis: "token".into(),
            start: 0,
            end: if decode { 1 } else { 3 },
            stride: 2,
        },
        CaptureSlice {
            axis: "route".into(),
            start: 0,
            end: 3,
            stride: 2,
        },
        CaptureSlice {
            axis: "unit".into(),
            start: 1,
            end: 5,
            stride: 2,
        },
    ];
    SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &catalog,
            &support,
            &CaptureCapabilities {
                transformations: vec![CaptureTransformKind::RoutedUnits],
                ..Default::default()
            },
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 3,
            },
            CaptureTextOrigin {
                cached_positions: 7,
            },
        )
        .unwrap(),
    )
}
fn scalar(shape: &[i32], values: Vec<f32>) -> Value {
    Value {
        shape: shape.into(),
        values: Arc::new(values),
    }
}
#[test]
fn original_routed_observer_charges_once_across_provider_batches_and_empty_selected_batches() {
    let source = sparse_source();
    let mut paths = super::super::source();
    Arc::get_mut(&mut paths.0).unwrap().prefill =
        vec![PrefillObservationDeclaration::causal_routed_units(
            "layer.0.output".into(),
        )]
        .into_boxed_slice();
    let selected = paths.prepare_capture_selection(&source).unwrap();
    let bound = selected.bind_geometry(geometry()).unwrap();
    let (pool, r, run) = fresh(&source, 0);
    let mut bank = bank(&source, &r, &run);
    let mut native = backend(&run);
    let used = pool.used_bytes().unwrap();
    bank.with_prefill_observer(&mut native, bound, &Error::Capture, |o| {
        for k in 0..2 {
            enter(o, k);
            let tokens = (3 - k * 2).min(2);
            let groups = scalar(
                &[tokens as i32, 3],
                (0..tokens * 3).map(|i| (i % 7) as f32).collect(),
            );
            let input = scalar(&[tokens as i32, 5], vec![1.; tokens as usize * 5]);
            let routed = o.routed_unit_observer("route").unwrap().unwrap();
            routed
                .begin_invocation(&crate::RoutedUnitInvocation {
                    input: &input,
                    origins: None,
                    unit_coordinates: None,
                })
                .unwrap();
            for physical in 0..tokens {
                let token = k * 2 + physical;
                let values = scalar(
                    &[3, 5],
                    (0..15)
                        .map(|i| token as f32 * 100. + i as f32 + 0.125)
                        .collect(),
                );
                let indices = scalar(&[3], vec![0., 0., 0.]);
                let slots = scalar(&[3], vec![0., 1., 2.]);
                let coefficients = scalar(&[1, 3], vec![0.25, 0.5, 0.75]);
                let batch = crate::RoutedUnitBatch {
                    units: eredu_nn::GroupedUnitBatch {
                        values: &values,
                        group_indices: &slots,
                        selection_indices: &slots,
                        token_indices: &indices,
                        coefficients: &coefficients,
                        token_offset: physical as usize,
                        total_token_count: tokens as usize,
                        group_count: 7,
                    },
                    source_groups: &groups,
                    global_groups: None,
                    provider_token_offset: 0,
                    origins: None,
                    unit_coordinates: None,
                };
                routed.observe(&batch).unwrap();
                routed.observe_effective(&batch).unwrap();
            }
            routed.finish_invocation(true).unwrap();
            commit(o, k);
        }
        o.finish_prefill(true);
    })
    .unwrap();
    assert_eq!(native.transforms, 3);
    assert_eq!(bank.usage().captures, 1);
    assert_eq!(bank.spent_steps(), 1);
    let delivery = bank.take_shared_step().unwrap().unwrap();
    assert_eq!(delivery.outcome(), CaptureStepOutcome::Committed);
    let Some(CapturePayload::RoutedUnits(units)) = &delivery.records()[0].payload else {
        panic!("sparse result");
    };
    assert_eq!(
        units
            .rows
            .iter()
            .map(|r| (r.token, r.slot))
            .collect::<Vec<_>>(),
        [(0, 0), (0, 2), (2, 0), (2, 2)]
    );
    for row in &units.rows {
        let TensorObservationData::F32(values) = row.values.data() else {
            panic!("float rows");
        };
        assert_eq!(
            values,
            &[
                row.token as f32 * 100. + row.slot as f32 * 5. + 1.125,
                row.token as f32 * 100. + row.slot as f32 * 5. + 3.125
            ]
        );
        assert_eq!(row.coefficient, if row.slot == 0 { 0.25 } else { 0.75 });
    }
    assert_eq!(delivery.records()[0].charged.captures, 1);
    native.scope.certify().unwrap();
    drop(bank);
    drop(run);
    drop(r);
    assert_eq!(pool.used_bytes().unwrap(), used);
    drop(delivery);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_routed_observer_keeps_one_charge_for_ordinary_prefill_and_cached_decodes() {
    let source = sparse_source_with_decode(true);
    let (pool, reservation, run) = fresh(&source, 0);
    let mut bank = bank(&source, &reservation, &run);
    let mut native = backend(&run);
    let mut deliveries = Vec::new();
    for prediction in 0..3_u64 {
        let tokens = if prediction == 0 { 3 } else { 1 };
        let pass = if prediction == 0 { ExpertPass::Prefill } else { ExpertPass::Decode };
        bank.with_observer(&mut native, prediction, &Error::Capture, |observer| {
            observer.prepare_transaction(epoch(prediction), pass).unwrap();
            let groups = scalar(&[tokens as i32, 3],
                (0..tokens * 3).map(|i| (i % 7) as f32).collect());
            let input = scalar(&[tokens as i32, 5], vec![1.; tokens as usize * 5]);
            let routed = observer.routed_unit_observer("route").unwrap().unwrap();
            routed.begin_invocation(&crate::RoutedUnitInvocation {
                input: &input, origins: None, unit_coordinates: None,
            }).unwrap();
            for token in 0..tokens {
                let values = scalar(&[3, 5], (0..15)
                    .map(|i| prediction as f32 * 1000. + token as f32 * 100. + i as f32 + 0.125)
                    .collect());
                let indices = scalar(&[3], vec![0., 0., 0.]);
                let slots = scalar(&[3], vec![0., 1., 2.]);
                let coefficients = scalar(&[1, 3], vec![0.25, 0.5, 0.75]);
                let batch = crate::RoutedUnitBatch {
                    units: eredu_nn::GroupedUnitBatch {
                        values: &values, group_indices: &slots, selection_indices: &slots,
                        token_indices: &indices, coefficients: &coefficients,
                        token_offset: token as usize, total_token_count: tokens as usize,
                        group_count: 7,
                    },
                    source_groups: &groups, global_groups: None, provider_token_offset: 0,
                    origins: None, unit_coordinates: None,
                };
                routed.observe(&batch).unwrap();
                routed.observe_effective(&batch).unwrap();
            }
            routed.finish_invocation(true).unwrap();
            observer.complete_transaction(epoch(prediction)).unwrap();
            observer.finish_transaction(epoch(prediction), true);
        }).unwrap();
        let delivery = bank.take_shared_step().unwrap().unwrap();
        assert_eq!(delivery.outcome(), CaptureStepOutcome::Committed);
        assert_eq!(delivery.records()[0].charged.captures, 1);
        let Some(CapturePayload::RoutedUnits(units)) = &delivery.records()[0].payload else {
            panic!("sparse delivery");
        };
        assert_eq!(units.rows.len(), 2);
        for row in &units.rows {
            let TensorObservationData::F32(values) = row.values.data() else { panic!("float row"); };
            assert_eq!(row.token, 0);
            let base = prediction as f32 * 1000. + row.slot as f32 * 5.;
            assert_eq!(values, &[base + 1.125, base + 3.125]);
            assert_eq!(row.coefficient, if row.slot == 0 { 0.25 } else { 0.75 });
        }
        deliveries.push(delivery);
    }
    assert_eq!(native.transforms, 5);
    assert_eq!(bank.usage().captures, 3);
    native.scope.certify().unwrap();
    drop((bank, run, reservation));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(deliveries);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod partition;
