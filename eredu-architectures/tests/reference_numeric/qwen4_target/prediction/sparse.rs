//! Selected-route capture and editing through the ordinary speculative ledger.
use super::*;

fn locations(source: &RoutedUnitCaptureSource<'_, NumericTensor>) -> RoutedUnitLocations {
    let routes = source.coefficients.shape[1] as usize;
    let source_routes = source.source_groups.shape[1] as usize;
    let groups = source
        .source_groups
        .exact_i32
        .as_ref()
        .expect("exact routed group IDs");
    RoutedUnitLocations {
        source_token_range: [
            source.token_offset,
            source.token_offset + source.coefficients.shape[0] as u64,
        ],
        rows: (0..source.values.shape[0] as usize)
            .map(|row| {
                let token = source.token_offset + source.token_indices.data[row] as u64;
                let slot = source.selection_indices.data[row] as usize % routes;
                let group = groups[token as usize * source_routes + slot] as usize;
                RoutedUnitLocation {
                    source_peer: None,
                    token,
                    slot: slot as u64,
                    expert: source.global_groups.map_or(group, |ids| ids[group]) as u64,
                }
            })
            .collect(),
    }
}
pub(super) fn cost(shape: &[u64], slice: &ResolvedCaptureSlice) -> CaptureUsage {
    let rows = slice.shape[0] * slice.shape[1];
    let values = slice.shape.iter().product::<u64>();
    CaptureUsage {
        captures: 1,
        retained_bytes: shape.iter().product::<u64>() * 4 + 4096,
        host_bytes: 4096 + rows * 256 + values * 16,
        encoded_bytes: 4096 + rows * 512 + values * 32,
    }
}
pub(super) fn capture(
    source: &RoutedUnitCaptureSource<'_, NumericTensor>,
    geometry: RoutedUnitGeometry,
    slice: &ResolvedCaptureSlice,
) -> RoutedUnitCapture {
    TRANSFORMS.with(|n| n.set(n.get() + 1));
    let locations = locations(source);
    let width = geometry.units_per_expert as usize;
    let selected = |axis: usize, x: u64| {
        x >= slice.starts[axis]
            && x < slice.ends[axis]
            && (x - slice.starts[axis]).is_multiple_of(slice.strides[axis])
    };
    let rows = locations
        .rows
        .iter()
        .enumerate()
        .filter_map(|(row, location)| {
            if !selected(0, location.token) || !selected(1, location.slot) {
                return None;
            }
            let values = (0..slice.shape[2])
                .map(|unit| {
                    source.values.data
                        [row * width + (slice.starts[2] + unit * slice.strides[2]) as usize]
                })
                .collect::<Vec<_>>();
            Some(RoutedUnitCaptureRow {
                source_peer: None,
                token: location.token,
                slot: location.slot,
                expert: location.expert,
                coefficient: source.coefficients.data[source.selection_indices.data[row] as usize],
                unit_start: slice.starts[2],
                unit_stride: slice.strides[2],
                values: eredu_core::TensorObservation::new(
                    vec![values.len()],
                    eredu_core::TensorObservationData::F32(values),
                )
                .unwrap(),
            })
        })
        .collect();
    RoutedUnitCapture {
        geometry,
        source_token_ranges: vec![locations.source_token_range],
        rows,
    }
}
pub(super) fn unit_locations(
    source: &RoutedUnitCaptureSource<'_, NumericTensor>,
) -> RoutedUnitLocations {
    TRANSFORMS.with(|n| n.set(n.get() + 1));
    locations(source)
}
pub(super) fn edit_cost(geometry: RoutedUnitGeometry, source: &[u64]) -> CaptureUsage {
    // Only selected rows are materialized: source[1] is a virtual expert-dense extent.
    let rows = source[0] * geometry.routes_per_token;
    let values = rows * geometry.units_per_expert;
    CaptureUsage {
        retained_bytes: 4096 + values * 32,
        host_bytes: 4096 + rows * 128 + values * 32,
        ..Default::default()
    }
}
fn floats(value: &eredu_core::TensorObservation) -> &[f32] {
    let eredu_core::TensorObservationData::F32(values) = value.data() else {
        panic!()
    };
    values
}
fn payload<'a>(invocation: &'a SpeculativeActivationCapture, path: &str) -> &'a RoutedUnitCapture {
    let record = invocation
        .captures
        .records
        .iter()
        .find(|r| r.path == path)
        .unwrap();
    let Some(CapturePayload::RoutedUnits(payload)) = &record.payload else {
        panic!("missing {path}")
    };
    payload
}
pub(super) fn validate(invocation: &SpeculativeActivationCapture, record: &CaptureRecord) -> bool {
    let Some(CapturePayload::RoutedUnits(units)) = &record.payload else {
        panic!()
    };
    assert_eq!(
        units.geometry,
        RoutedUnitGeometry {
            experts: 3,
            units_per_expert: 2,
            routes_per_token: 2
        }
    );
    assert_eq!(units.rows.len(), 2);
    let root = record
        .path
        .strip_suffix(".units.effective")
        .or_else(|| record.path.strip_suffix(".units"))
        .unwrap();
    let tensor = |field: eredu_core::RoutingObservationField| {
        let path = field.path(root);
        let Some(CapturePayload::Tensor(value)) = &invocation
            .captures
            .records
            .iter()
            .find(|r| r.path == path)
            .unwrap()
            .payload
        else {
            panic!()
        };
        value
    };
    let eredu_core::TensorObservationData::I64(ids) =
        tensor(eredu_core::RoutingObservationField::SelectedExperts).data()
    else {
        panic!()
    };
    let coefficients = floats(tensor(eredu_core::RoutingObservationField::Coefficients));
    for (slot, row) in units.rows.iter().enumerate() {
        assert_eq!(
            (
                row.source_peer,
                row.token,
                row.slot,
                row.unit_start,
                row.unit_stride
            ),
            (None, 0, slot as u64, 0, 1)
        );
        assert_eq!(row.expert as i64, ids[slot]);
        assert_eq!(row.coefficient, coefficients[slot]);
        assert_eq!(row.values.shape(), &[2]);
        assert!(floats(&row.values).iter().all(|v| v.is_finite()));
    }
    assert_eq!(
        payload(invocation, &format!("{root}.units")),
        payload(invocation, &format!("{root}.units.effective"))
    );
    units
        .rows
        .iter()
        .any(|row| floats(&row.values).iter().any(|v| *v != 0.))
}

pub(super) fn checks(
    selected: &Selected,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    discovery: &SpeculativeActivationDiscovery,
    plain: &Outcome,
) {
    collector_check(ctx);
    selected_component_checks(selected, prepared, ctx, discovery);
    let mut unavailable = discovery.clone();
    unavailable
        .captures
        .support
        .capture
        .transformations
        .retain(|t| *t != CaptureTransformKind::RoutedUnits);
    assert!(plan(&unavailable).admit(&unavailable).is_err());
    for root in ["model.layers.0.mlp", "mtp.layers.0.mlp", "mtp.layers.1.mlp"] {
        let target = format!("{root}.units");
        let scope = discovery
            .bindings
            .iter()
            .find(|b| b.node_id == discovery.captures.catalog.get(&target).unwrap().node_id)
            .unwrap()
            .scope;
        for identity in [true, false] {
            let mut request = plan(discovery);
            request
                .interventions
                .operations
                .push(InterventionOperation {
                    id: "sparse-edit".into(),
                    target: target.clone(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![],
                    action: InterventionAction::Scale {
                        dtype: InterventionDtype::Float32,
                        factor: if identity { 1. } else { 0. },
                    },
                    evidence: InterventionEvidence::None,
                });
            let mut readonly = request.clone();
            readonly.interventions.operations[0]
                .target
                .push_str(".effective");
            assert!(readonly.admit(discovery).is_err());
            let mut options = super::super::options();
            options.activations = Some(request.admit(discovery).unwrap());
            let mut reached = false;
            let result = run(selected, prepared, ctx, options, |session| {
                while let Some(step) = session.step()? {
                    for invocation in &step.activations {
                        let edit = &invocation.captures.interventions[0];
                        if !scope.applies(invocation.phase) {
                            assert_eq!(edit.outcome, InterventionOutcome::Inactive);
                            continue;
                        }
                        reached = true;
                        assert_eq!(edit.outcome, InterventionOutcome::Applied);
                        let receipt = edit.routed_units.as_ref().unwrap();
                        assert_eq!(receipt.completed_tokens, receipt.source_tokens);
                        assert_eq!(receipt.affected_values, receipt.source_tokens * 4);
                        let before = payload(invocation, &target);
                        let after = payload(invocation, &format!("{target}.effective"));
                        if identity {
                            assert_eq!(before, after);
                        } else {
                            assert!(before
                                .rows
                                .iter()
                                .any(|r| floats(&r.values).iter().any(|v| *v != 0.)));
                            assert!(after
                                .rows
                                .iter()
                                .all(|r| floats(&r.values).iter().all(|v| *v == 0.)));
                            let path = eredu_core::RoutingObservationField::RoutedOutput.path(root);
                            let Some(CapturePayload::Tensor(output)) = &invocation
                                .captures
                                .records
                                .iter()
                                .find(|r| r.path == path)
                                .unwrap()
                                .payload
                            else {
                                panic!()
                            };
                            assert!(
                                floats(output).iter().all(|v| *v == 0.),
                                "down projection consumes edited units"
                            );
                        }
                    }
                    if !identity {
                        break;
                    }
                }
                Ok(())
            });
            assert!(reached);
            if identity {
                assert_eq!(result.tokens, plain.tokens);
                exact(&result.target, &plain.target, "sparse identity target");
                exact(
                    &result.prediction,
                    &plain.prediction,
                    "sparse identity prediction",
                );
            }
        }
    }
}

// Independent collector fixture: native rows are expert-sorted, selection rows
// are token/slot ordered, and source IDs use a compact-to-global mapping.
fn collector_check(ctx: &NumericContext) {
    let values = NumericTensor::new(
        vec![4, 4],
        vec![
            30., 31., 32., 33., 10., 11., 12., 13., 20., 21., 22., 23., 40., 41., 42., 43.,
        ],
    );
    let tokens = NumericTensor::from_i32_slice(&[0, 0, 1, 1], &[4], ctx).unwrap();
    let selections = NumericTensor::from_i32_slice(&[1, 0, 2, 3], &[4], ctx).unwrap();
    let groups = NumericTensor::from_i32_slice(&[1, 0, 1, 2], &[2, 2], ctx).unwrap();
    let coefficients = NumericTensor::new(vec![2, 2], vec![0.1, 0.3, 0.2, 0.4]);
    let source = RoutedUnitCaptureSource {
        values: &values,
        token_indices: &tokens,
        selection_indices: &selections,
        coefficients: &coefficients,
        source_groups: &groups,
        token_offset: 0,
        global_groups: Some(&[2, 7, 11]),
    };
    let geometry = RoutedUnitGeometry {
        experts: 12,
        units_per_expert: 4,
        routes_per_token: 2,
    };
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 1, 1],
        ends: vec![2, 2, 4],
        strides: vec![1, 1, 2],
        shape: vec![2, 1, 2],
    };
    let mut result = capture(&source, geometry, &slice);
    result.finish_ordinary(&slice, 2).unwrap();
    assert_eq!(
        result
            .rows
            .iter()
            .map(|r| (
                r.token,
                r.slot,
                r.expert,
                r.coefficient,
                floats(&r.values).to_vec()
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 1, 2, 0.3, vec![31., 33.]),
            (1, 1, 11, 0.4, vec![41., 43.])
        ]
    );
    assert_eq!(
        locations(&source)
            .rows
            .iter()
            .map(|r| (r.token, r.slot, r.expert))
            .collect::<Vec<_>>(),
        vec![(0, 1, 2), (0, 0, 7), (1, 0, 7), (1, 1, 11)]
    );
    let shifted_groups = NumericTensor::from_i32_slice(&[2, 2, 1, 0, 1, 2], &[3, 2], ctx).unwrap();
    let shifted = RoutedUnitCaptureSource {
        source_groups: &shifted_groups,
        token_offset: 1,
        ..source
    };
    let shifted_slice = ResolvedCaptureSlice {
        starts: vec![1, 1, 1],
        ends: vec![3, 2, 4],
        ..slice.clone()
    };
    let shifted = capture(&shifted, geometry, &shifted_slice);
    shifted.validate_rows(&shifted_slice).unwrap();
    assert_eq!(shifted.source_token_ranges, vec![[1, 3]]);
    for (a, b) in shifted.rows.iter().zip(&result.rows) {
        assert_eq!(a.token, b.token + 1);
        assert_eq!(
            (a.slot, a.expert, a.coefficient, &a.values),
            (b.slot, b.expert, b.coefficient, &b.values)
        );
    }
    let duplicate_groups = NumericTensor::from_i32_slice(&[1, 1, 1, 2], &[2, 2], ctx).unwrap();
    let duplicate = RoutedUnitCaptureSource {
        source_groups: &duplicate_groups,
        ..source
    };
    let all = ResolvedCaptureSlice {
        starts: vec![0, 0, 0],
        ends: vec![2, 2, 4],
        strides: vec![1, 1, 1],
        shape: vec![2, 2, 4],
    };
    let mut duplicate = capture(&duplicate, geometry, &all);
    duplicate.finish_ordinary(&all, 2).unwrap();
    assert_eq!((duplicate.rows[0].expert, duplicate.rows[1].expert), (7, 7));
    assert_eq!((duplicate.rows[0].slot, duplicate.rows[1].slot), (0, 1));
    assert_ne!(duplicate.rows[0].values, duplicate.rows[1].values);
    let large = RoutedUnitGeometry {
        experts: 1_000_000,
        ..geometry
    };
    assert_eq!(
        edit_cost(geometry, &[2, 48]),
        edit_cost(large, &[2, 4_000_000]),
        "virtual expert width must not reserve dense values"
    );
}

fn selected_component_checks(
    selected: &Selected,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    discovery: &SpeculativeActivationDiscovery,
) {
    let root = "mtp.layers.0.mlp";
    let target = format!("{root}.units");
    let mut options = super::super::options();
    options.activations = Some(plan(discovery).admit(discovery).unwrap());
    let mut ordinary = None;
    run(selected, prepared, ctx, options, |session| {
        ordinary = session
            .step()?
            .unwrap()
            .activations
            .into_iter()
            .find(|a| a.phase == SpeculativeActivationPhase::PredictionPrefill);
        Ok(())
    });
    let ordinary = ordinary.unwrap();
    let before = payload(&ordinary, &target);
    let active = before.rows[0].expert;
    let inactive = (0..3)
        .find(|id| before.rows.iter().all(|r| r.expert != *id))
        .unwrap();
    for expert in [active, inactive] {
        let mut request = plan(discovery);
        request
            .interventions
            .operations
            .push(InterventionOperation {
                id: "one-expert-unit".into(),
                target: target.clone(),
                schedule: CaptureSchedule::default(),
                slices: vec![
                    CaptureSlice {
                        axis: "token".into(),
                        start: 0,
                        end: 1,
                        stride: 1,
                    },
                    CaptureSlice {
                        axis: "component".into(),
                        start: expert * 2 + 1,
                        end: expert * 2 + 2,
                        stride: 1,
                    },
                ],
                action: InterventionAction::Zero {
                    dtype: InterventionDtype::Float32,
                },
                evidence: InterventionEvidence::None,
            });
        let mut options = super::super::options();
        options.activations = Some(request.admit(discovery).unwrap());
        run(selected, prepared, ctx, options, |session| {
            let step = session.step()?.unwrap();
            let invocation = step
                .activations
                .iter()
                .find(|a| a.phase == SpeculativeActivationPhase::PredictionPrefill)
                .unwrap();
            assert_eq!(before, payload(invocation, &target));
            let after = payload(invocation, &format!("{target}.effective"));
            for (before, after) in before.rows.iter().zip(&after.rows) {
                assert_eq!(
                    (before.token, before.slot, before.expert, before.coefficient),
                    (after.token, after.slot, after.expert, after.coefficient)
                );
                let mut expected = floats(&before.values).to_vec();
                if before.expert == expert {
                    expected[1] = 0.;
                }
                assert_eq!(floats(&after.values), expected);
            }
            let receipt = invocation.captures.interventions[0]
                .routed_units
                .as_ref()
                .unwrap();
            assert_eq!(receipt.affected_values, u64::from(expert == active));
            assert_eq!(receipt.completed_tokens, receipt.source_tokens);
            Ok(())
        });
    }
}
