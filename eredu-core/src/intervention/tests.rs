use super::*;
use crate::{ObservationSupportStatus as S, SymbolicDimension as D};

fn request() -> CaptureRequestShape {
    CaptureRequestShape {
        batch: 1,
        prompt_tokens: 3,
        max_predictions: 4,
    }
}

fn discovery(routing: bool) -> InterventionDiscovery {
    let mut point = InterventionPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        stage: InterventionStage::Activation,
        axes: [
            ("batch", D::Batch),
            ("sequence", D::Sequence),
            ("hidden", D::Known(2)),
        ]
        .into_iter()
        .map(|(name, dimension)| TensorAxis {
            name: name.into(),
            dimension,
        })
        .collect(),
        dtypes: vec![
            InterventionDtype::Float32,
            InterventionDtype::Float16,
            InterventionDtype::Bfloat16,
        ],
        operations: vec![
            InterventionKind::Zero,
            InterventionKind::Scale,
            InterventionKind::Mask,
            InterventionKind::Replace,
            InterventionKind::Add,
        ],
        score_stages: vec![],
        prefill: S::Supported,
        decode: S::Supported,
        conditions: vec![],
        routing: None,
    };
    if routing {
        point.path = "block.moe".into();
        point.stage = InterventionStage::RoutingBeforeDispatch;
        point.axes = vec![
            TensorAxis {
                name: "token".into(),
                dimension: D::TokenRows,
            },
            TensorAxis {
                name: "selected_expert".into(),
                dimension: D::Known(2),
            },
        ];
        point.dtypes.clear();
        point.operations = vec![
            InterventionKind::ExcludeExperts,
            InterventionKind::ZeroExpertContribution,
            InterventionKind::BiasRoutingScores,
            InterventionKind::ForceExperts,
        ];
        point.score_stages = vec![RoutingScoreStage::RankingScores];
        point.routing = Some(InterventionRoutingPolicy {
            expert_count: 4,
            top_k: 2,
            scoring: RoutingScoring::Softmax,
            normalize_selected: true,
            normalization_epsilon: 0.0,
            coefficient_scale: 1.5,
            groups: 1,
            selected_groups: 1,
            learned_coefficient_scale: false,
            shared_experts: 1,
        });
    }
    InterventionDiscovery {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        artifact_identity: "source-a".into(),
        session_identity: Some("backend-session".into()),
        points: vec![point],
    }
}

fn operation(action: InterventionAction) -> InterventionOperation {
    InterventionOperation {
        id: "operation".into(),
        target: if action.dtype().is_some() {
            "block.output"
        } else {
            "block.moe"
        }
        .into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        action,
        evidence: InterventionEvidence::None,
    }
}

fn plan(operations: Vec<InterventionOperation>) -> InterventionPlan {
    InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations,
    }
}

fn admit(operation: InterventionOperation) -> Result<AdmittedInterventionPlan, CaptureError> {
    let routing = operation.action.dtype().is_none();
    plan(vec![operation]).admit(&discovery(routing), request(), "session-a")
}

fn zero() -> InterventionOperation {
    operation(InterventionAction::Zero {
        dtype: InterventionDtype::Float32,
    })
}

#[test]
fn plan_roundtrip_and_identity_bind_source_session_geometry_and_order() {
    let original = plan(vec![zero()]);
    let decoded: InterventionPlan =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    let a = original
        .clone()
        .admit(&discovery(false), request(), "session-a")
        .unwrap();
    assert_eq!(
        a.identity(),
        decoded
            .admit(&discovery(false), request(), "session-a")
            .unwrap()
            .identity()
    );
    assert_ne!(
        a.identity(),
        original
            .clone()
            .admit(&discovery(false), request(), "session-b")
            .unwrap()
            .identity()
    );
    let mut other = discovery(false);
    other.artifact_identity = "source-b".into();
    assert_ne!(
        a.identity(),
        original
            .clone()
            .admit(&other, request(), "session-a")
            .unwrap()
            .identity()
    );
    let mut r = request();
    r.prompt_tokens = 4;
    assert_ne!(
        a.identity(),
        original
            .admit(&discovery(false), r, "session-a")
            .unwrap()
            .identity()
    );
    let mut scale = operation(InterventionAction::Scale {
        dtype: InterventionDtype::Float32,
        factor: 2.0,
    });
    scale.id = "scale".into();
    assert_ne!(
        plan(vec![zero(), scale.clone()])
            .admit(&discovery(false), request(), "s")
            .unwrap()
            .identity(),
        plan(vec![scale, zero()])
            .admit(&discovery(false), request(), "s")
            .unwrap()
            .identity()
    );
}

#[test]
fn replacement_selects_prefill_rows_independently_of_prediction_schedule() {
    let mut op = operation(InterventionAction::Replace {
        tensor: InterventionTensor {
            shape: vec![1, 1, 2],
            values: InterventionValues::Float32(vec![7.0, 8.0]),
        },
    });
    op.slices.push(CaptureSlice {
        axis: "sequence".into(),
        start: 1,
        end: 2,
        stride: 1,
    });
    op.schedule.decode = false;
    let admitted = admit(op.clone()).unwrap();
    let slice = admitted
        .validate_actual(
            0,
            CapturePhase::Prefill,
            0,
            &[1, 3, 2],
            Some(InterventionDtype::Float32),
        )
        .unwrap();
    assert_eq!(slice.starts, [0, 1, 0]);
    assert_eq!(slice.shape, [1, 1, 2]);
    assert!(admitted
        .validate_actual(
            0,
            CapturePhase::Decode,
            1,
            &[1, 1, 2],
            Some(InterventionDtype::Float32)
        )
        .is_err());
    op.schedule.decode = true;
    assert!(
        admit(op).is_err(),
        "decode has only one row; selecting row one is invalid at admission"
    );
}

#[test]
fn exact_shapes_prevent_broadcast_and_partial_replacement() {
    for (shape, values) in [
        (vec![2], vec![1.0, 2.0]),
        (vec![1, 3, 2], vec![1.0; 5]),
        (vec![1, 1, 2], vec![1.0; 2]),
    ] {
        assert!(admit(operation(InterventionAction::Replace {
            tensor: InterventionTensor {
                shape,
                values: InterventionValues::Float32(values)
            }
        }))
        .is_err());
    }
    assert!(admit(operation(InterventionAction::Mask {
        dtype: InterventionDtype::Float32,
        shape: vec![1, 3, 2],
        keep: vec![true; 5]
    }))
    .is_err());
}

#[test]
fn floating_payloads_are_finite_and_runtime_dtypes_are_exact() {
    for values in [
        InterventionValues::Float32(vec![f32::NAN; 6]),
        InterventionValues::Float16(vec![0x7c00; 6]),
        InterventionValues::Bfloat16(vec![0x7f80; 6]),
    ] {
        let mut op = operation(InterventionAction::Replace {
            tensor: InterventionTensor {
                shape: vec![1, 3, 2],
                values,
            },
        });
        op.schedule.decode = false;
        assert!(admit(op).is_err());
    }
    assert!(admit(operation(InterventionAction::Scale {
        dtype: InterventionDtype::Float16,
        factor: 65536.0
    }))
    .is_err());
    let admitted = admit(zero()).unwrap();
    assert!(admitted
        .validate_actual(
            0,
            CapturePhase::Prefill,
            0,
            &[1, 3, 2],
            Some(InterventionDtype::Float16)
        )
        .is_err());
    assert!(admitted
        .validate_actual(
            0,
            CapturePhase::Prefill,
            0,
            &[1, 3, 4],
            Some(InterventionDtype::Float32)
        )
        .is_err());
}

#[test]
fn unknown_targets_duplicate_ids_axes_and_unsupported_phases_fail() {
    let mut op = zero();
    op.target = "read.only.tensor".into();
    assert!(admit(op).is_err());
    assert!(plan(vec![zero(), zero()])
        .admit(&discovery(false), request(), "s")
        .is_err());
    let mut op = zero();
    op.slices = vec![CaptureSlice {
        axis: "missing".into(),
        start: 0,
        end: 1,
        stride: 1,
    }];
    assert!(admit(op).is_err());
    let mut op = zero();
    op.slices = vec![
        CaptureSlice {
            axis: "sequence".into(),
            start: 0,
            end: 1,
            stride: 1
        };
        2
    ];
    assert!(admit(op).is_err());
    for status in [
        S::Unsupported("partitioned".into()),
        S::Unverified("native support".into()),
        S::Conditional("media required".into()),
    ] {
        let mut d = discovery(false);
        d.points[0].prefill = status;
        assert!(plan(vec![zero()]).admit(&d, request(), "s").is_err());
    }
}

#[test]
fn bounded_plans_reject_oversized_payloads_metadata_and_operations() {
    let payload =
        InterventionValues::Float32(vec![0.0; MAX_INTERVENTION_PAYLOAD_BYTES as usize / 4 + 1]);
    assert!(admit(operation(InterventionAction::Replace {
        tensor: InterventionTensor {
            shape: vec![payload.len() as u64],
            values: payload
        }
    }))
    .is_err());
    let mut op = zero();
    op.id = "x".repeat(129);
    assert!(admit(op).is_err());
    assert!(plan(vec![zero(); MAX_INTERVENTION_OPERATIONS + 1])
        .admit(&discovery(false), request(), "s")
        .is_err());
    let mut op = zero();
    op.evidence = InterventionEvidence::Preview { max_elements: 4097 };
    assert!(admit(op).is_err());
}

#[test]
fn routing_validates_namespace_duplicates_count_and_stage() {
    for ids in [vec![4], vec![1, 1], vec![0, 1, 2]] {
        assert!(admit(operation(InterventionAction::ExcludeExperts {
            expert_ids: ids
        }))
        .is_err());
    }
    assert!(admit(operation(InterventionAction::ExcludeExperts {
        expert_ids: vec![0, 2]
    }))
    .is_ok());
    assert!(admit(operation(InterventionAction::ZeroExpertContribution {
        expert_ids: vec![0, 1, 2, 3]
    }))
    .is_ok());
    for (stage, biases) in [
        (RoutingScoreStage::RawLogits, vec![1.0]),
        (RoutingScoreStage::RankingScores, vec![f32::INFINITY]),
        (RoutingScoreStage::RankingScores, vec![]),
    ] {
        assert!(admit(operation(InterventionAction::BiasRoutingScores {
            stage,
            expert_ids: vec![1],
            biases
        }))
        .is_err());
    }
    assert!(admit(operation(InterventionAction::BiasRoutingScores {
        stage: RoutingScoreStage::RankingScores,
        expert_ids: vec![1],
        biases: vec![1.0]
    }))
    .is_ok());
}

#[test]
fn forced_routes_are_complete_per_token_without_duplicate_ids() {
    let mut op = operation(InterventionAction::ForceExperts {
        shape: [3, 2],
        expert_ids: vec![0, 1, 2, 3, 1, 0],
    });
    op.schedule.decode = false;
    assert!(admit(op.clone()).is_ok());
    op.schedule.decode = true;
    assert!(admit(op).is_err());
    let mut op = operation(InterventionAction::ForceExperts {
        shape: [3, 2],
        expert_ids: vec![0, 0, 2, 3, 1, 0],
    });
    op.schedule.decode = false;
    assert!(admit(op).is_err());
    let mut op = operation(InterventionAction::ForceExperts {
        shape: [1, 2],
        expert_ids: vec![3, 1],
    });
    op.slices.push(CaptureSlice {
        axis: "token".into(),
        start: 0,
        end: 1,
        stride: 1,
    });
    assert!(admit(op).is_ok());
}

#[test]
fn grouped_exclusion_and_forcing_preserve_group_constraints() {
    let mut d = discovery(true);
    d.points[0].routing.as_mut().unwrap().groups = 2;
    let op = operation(InterventionAction::ExcludeExperts {
        expert_ids: vec![0],
    });
    assert!(plan(vec![op]).admit(&d, request(), "s").is_err());
    let mut op = operation(InterventionAction::ForceExperts {
        shape: [3, 2],
        expert_ids: vec![0, 2, 0, 1, 2, 3],
    });
    op.schedule.decode = false;
    assert!(plan(vec![op]).admit(&d, request(), "s").is_err());
}

#[test]
fn routing_overlaps_fail_but_disjoint_schedules_are_admitted() {
    let a = operation(InterventionAction::ExcludeExperts {
        expert_ids: vec![1],
    });
    let mut b = operation(InterventionAction::ForceExperts {
        shape: [1, 2],
        expert_ids: vec![1, 2],
    });
    b.id = "force".into();
    b.schedule.prefill = false;
    assert!(plan(vec![a.clone(), b.clone()])
        .admit(&discovery(true), request(), "s")
        .is_err());
    let mut a = a;
    a.schedule.decode = false;
    assert!(plan(vec![a, b])
        .admit(&discovery(true), request(), "s")
        .is_ok());
}

#[test]
fn empty_plans_and_inactive_schedules_have_explicit_semantics() {
    assert!(InterventionPlan::none()
        .admit(&discovery(false), request(), "s")
        .unwrap()
        .is_empty());
    let mut op = zero();
    op.schedule.first_prediction = 4;
    let p = admit(op).unwrap();
    assert!(!p.is_empty());
    assert!(p
        .validate_actual(
            0,
            CapturePhase::Decode,
            4,
            &[1, 1, 2],
            Some(InterventionDtype::Float32)
        )
        .is_err());
}
