use super::*;
use eredu_core::component::ComponentCoordinateMap;

mod sum;
mod original;

fn admitted(action: InterventionAction, partial: bool) -> AdmittedInterventionPlan {
    admitted_at(action, partial, 1, 1)
}

fn admitted_at(
    action: InterventionAction,
    partial: bool,
    sequence_start: u64,
    sequence_stride: u64,
) -> AdmittedInterventionPlan {
    let logits = matches!(action, InterventionAction::MaskLogits { .. });
    let axis = if logits { "vocabulary" } else { "component" };
    let point = InterventionPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        stage: if logits {
            InterventionStage::LogitsBeforeSampling
        } else {
            InterventionStage::Activation
        },
        axes: vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: axis.into(),
                dimension: SymbolicDimension::Known(8),
            },
        ],
        dtypes: vec![action.dtype().unwrap()],
        operations: vec![action.kind()],
        score_stages: vec![],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
        routed_units: None,
        routing: None,
    };
    let mut slices = vec![CaptureSlice {
        axis: "sequence".into(),
        start: sequence_start,
        end: 3,
        stride: sequence_stride,
    }];
    if partial {
        slices.push(CaptureSlice {
            axis: axis.into(),
            start: 1,
            end: 8,
            stride: 2,
        });
    }
    InterventionPlan {
        schema_version: 1,
        operations: vec![InterventionOperation {
            id: "edit".into(),
            target: point.path.clone(),
            schedule: CaptureSchedule {
                decode: false,
                ..Default::default()
            },
            slices,
            action,
            evidence: InterventionEvidence::None,
        }],
    }
    .admit(
        &InterventionDiscovery {
            schema_version: 1,
            artifact_identity: "source".into(),
            session_identity: Some("native-session".into()),
            points: vec![point],
        },
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions: 3,
        },
        "run",
    )
    .unwrap()
}

struct Budget {
    limit: CaptureUsage,
    used: CaptureUsage,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            limit: CaptureUsage {
                captures: 0,
                retained_bytes: 1 << 20,
                host_bytes: 1 << 20,
                encoded_bytes: 0,
            },
            used: CaptureUsage::default(),
        }
    }
}
impl CaptureReservation for Budget {
    fn reserve(&mut self, usage: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        let total = self.used.checked_add(usage)?;
        if let Some(budget) = total.exceeded(self.limit) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: true,
            });
        }
        self.used = total;
        Ok(None)
    }
}

fn local(value: &Value, axis: usize, map: &ComponentCoordinateMap) -> Value {
    let mut shape = value.shape.clone();
    shape[axis] = map.local_count() as u64;
    let mut data = vec![];
    for row in 0..shape[0] as usize {
        for column in 0..shape[1] as usize {
            let r = if axis == 0 {
                map.local_to_global(row).unwrap()
            } else {
                row
            };
            let c = if axis == 1 {
                map.local_to_global(column).unwrap()
            } else {
                column
            };
            data.push(value.data[r * value.shape[1] as usize + c]);
        }
    }
    Value { shape, data }
}

#[test]
fn projected_activation_actions_preserve_global_values_positions_and_payload_order() {
    let dtype = InterventionDtype::Float32;
    let tensor = || InterventionTensor {
        shape: vec![2, 4],
        values: InterventionValues::Float32((0..8).map(|i| 20.0 - i as f32 * 1.7).collect()),
    };
    let actions = [
        (InterventionAction::Zero { dtype }, true),
        (
            InterventionAction::Scale {
                dtype,
                factor: -0.75,
            },
            true,
        ),
        (
            InterventionAction::Mask {
                dtype,
                shape: vec![2, 4],
                keep: vec![true, false, true, true, false, true, false, true],
            },
            true,
        ),
        (InterventionAction::Replace { tensor: tensor() }, true),
        (InterventionAction::Add { tensor: tensor() }, true),
        (
            InterventionAction::MaskComponents {
                dtype,
                indices: vec![6, 1, 4],
                keep_selected: true,
            },
            false,
        ),
        (
            InterventionAction::MaskComponents {
                dtype,
                indices: vec![6, 1, 4],
                keep_selected: false,
            },
            false,
        ),
        (
            InterventionAction::MaskLogits {
                dtype,
                token_ids: vec![4, 5, 6, 7],
            },
            false,
        ),
    ];
    let input = Value {
        shape: vec![3, 8],
        data: (0..24).map(|i| i as f32 * 0.3 - 2.7).collect(),
    };
    for (action, partial) in actions {
        let plan = admitted(action, partial);
        let global_slice = plan
            .validate_actual(0, CapturePhase::Prefill, 0, &input.shape, Some(dtype))
            .unwrap();
        let expected = apply_activation(
            &mut Backend::default(),
            &input,
            &plan.plan().operations[0].action,
            &global_slice,
        )
        .unwrap();
        for (axis, map) in [
            (1, ComponentCoordinateMap::range(8, 0..4).unwrap()),
            (1, ComponentCoordinateMap::range(8, 4..8).unwrap()),
            (1, ComponentCoordinateMap::range(8, 0..8).unwrap()),
            (1, ComponentCoordinateMap::range(8, 0..0).unwrap()),
            (
                1,
                ComponentCoordinateMap::indices(8, vec![6, 1, 4]).unwrap(),
            ),
            (
                1,
                ComponentCoordinateMap::indices(8, vec![7, 5, 3, 1]).unwrap(),
            ),
            (0, ComponentCoordinateMap::indices(3, vec![2, 0]).unwrap()),
        ] {
            let mut backend = Backend::default();
            let mut budget = Budget::default();
            let projection = PartitionActivationProjection::new(
                &plan,
                0,
                CapturePhase::Prefill,
                0,
                &input.shape,
                axis,
                &map,
                16,
            )
            .unwrap();
            let work = projection.reserve(&mut budget, &Estimates).unwrap();
            assert_eq!(work.plan_identity(), plan.identity());
            assert_eq!(work.intent_identity(), plan.intent_identity());
            assert_eq!(work.operation_index(), 0);
            assert_eq!(work.prediction(), 0);
            assert_eq!(work.phase(), CapturePhase::Prefill);
            assert_eq!(work.charged(), budget.used);
            assert_eq!(backend.applications, 0);
            let source = local(&input, axis, &map);
            let actual = work.apply(&mut backend, &source).unwrap().unwrap_or(source);
            let expected = local(&expected, axis, &map);
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(
                actual.data,
                expected.data,
                "{:?} map={map:?}",
                plan.plan().operations[0].action.kind()
            );
        }
    }
}

#[test]
fn compact_projected_masks_do_not_expand_permuted_components_into_operations() {
    let plan = admitted(
        InterventionAction::MaskComponents {
            dtype: InterventionDtype::Float32,
            indices: vec![7],
            keep_selected: true,
        },
        false,
    );
    let map = ComponentCoordinateMap::indices(8, vec![7, 1, 6, 2, 5, 3, 4, 0]).unwrap();
    let projection =
        PartitionActivationProjection::new(&plan, 0, CapturePhase::Prefill, 0, &[3, 8], 1, &map, 1)
            .unwrap();
    assert_eq!(projection.region_count(), 1);
    let zero = admitted(
        InterventionAction::Zero {
            dtype: InterventionDtype::Float32,
        },
        true,
    );
    assert!(PartitionActivationProjection::new(
        &zero,
        0,
        CapturePhase::Prefill,
        0,
        &[3, 8],
        1,
        &map,
        1
    )
    .is_err());
}

#[test]
fn partition_activation_preparation_precedes_native_work_and_never_refunds() {
    let plan = admitted(
        InterventionAction::Add {
            tensor: InterventionTensor {
                shape: vec![2, 4],
                values: InterventionValues::Float32(vec![1.5; 8]),
            },
        },
        true,
    );
    let map = ComponentCoordinateMap::indices(8, vec![7, 5, 3, 1]).unwrap();
    let project = || {
        PartitionActivationProjection::new(&plan, 0, CapturePhase::Prefill, 0, &[3, 8], 1, &map, 8)
            .unwrap()
    };
    let mut budget = Budget::default();
    budget.limit.host_bytes = 1;
    assert!(project().reserve(&mut budget, &Estimates).is_err());
    assert_eq!(budget.used, CaptureUsage::default());
    budget = Budget::default();
    budget.limit.retained_bytes = 0;
    assert!(project().reserve(&mut budget, &Estimates).is_err());
    assert!(
        budget.used.host_bytes > 0,
        "payload projection was prepaid before the native estimate was rejected"
    );
    assert_eq!(budget.used.retained_bytes, 0);
    budget = Budget::default();
    let work = project().reserve(&mut budget, &Estimates).unwrap();
    let spent = budget.used;
    drop(work);
    assert_eq!(budget.used, spent);
    let work = project().reserve(&mut budget, &Estimates).unwrap();
    assert_eq!(budget.used, spent.checked_mul(2).unwrap());
    let mut backend = Backend {
        fail_application: Some(2),
        ..Default::default()
    };
    let input = Value {
        shape: vec![3, 4],
        data: vec![2.0; 12],
    };
    assert!(work.apply(&mut backend, &input).is_err());
    assert_eq!(input.data, vec![2.0; 12]);
    assert_eq!(budget.used, spent.checked_mul(2).unwrap());
    let work = project().reserve(&mut budget, &Estimates).unwrap();
    backend.applications = 0;
    let wrong = Value {
        shape: vec![3, 3],
        data: vec![2.0; 9],
    };
    assert!(work.apply(&mut backend, &wrong).is_err());
    assert_eq!(backend.applications, 0);
    let work = project().reserve(&mut budget, &Estimates).unwrap();
    backend.dtype = Some(InterventionDtype::Bfloat16);
    assert!(work.apply(&mut backend, &input).is_err());
    assert_eq!(backend.applications, 0);
    for (index, phase, prediction, shape, axis, limit) in [
        (1, CapturePhase::Prefill, 0, vec![3, 8], 1, 8),
        (0, CapturePhase::Decode, 1, vec![1, 8], 1, 8),
        (0, CapturePhase::Prefill, 0, vec![3, 7], 1, 8),
        (0, CapturePhase::Prefill, 0, vec![3, 8], 0, 8),
        (0, CapturePhase::Prefill, 0, vec![3, 8], 1, 0),
    ] {
        assert!(PartitionActivationProjection::new(
            &plan, index, phase, prediction, &shape, axis, &map, limit
        )
        .is_err());
    }
}
