use super::*;
use crate::*;

fn fixture() -> (SpeculativeActivationPlan, SpeculativeActivationDiscovery) {
    let point = ObservationPoint {
        path: "opaque.units".into(),
        node_id: "opaque.node".into(),
        meaning: "actual pre-projection units".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "unit".into(),
                dimension: SymbolicDimension::Known(2),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![
            ObservationRequirement::ActivationHooks,
            ObservationRequirement::PredictionExecution,
        ],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let edited = InterventionPoint {
        path: point.path.clone(),
        node_id: point.node_id.clone(),
        stage: InterventionStage::Activation,
        axes: point.axes.clone().unwrap(),
        dtypes: vec![InterventionDtype::Float32],
        operations: vec![InterventionKind::Scale],
        score_stages: vec![],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
        routing: None,
        routed_units: None,
    };
    let discovery = SpeculativeActivationDiscovery {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        execution_identity: "selected-with-overlay-7".into(),
        captures: CaptureDiscovery {
            artifact_identity: "exact-source".into(),
            catalog: ObservationCatalog {
                schema_version: DISCOVERY_SCHEMA_VERSION,
                points: vec![point.clone()],
                completeness: DescriptionCompleteness::Complete,
            },
            support: ObservationSupportReport {
                schema_version: DISCOVERY_SCHEMA_VERSION,
                capture: CaptureCapabilities {
                    transformations: vec![CaptureTransformKind::Preview],
                    ..Default::default()
                },
                points: vec![ObservationSupport {
                    path: point.path.clone(),
                    prefill: ObservationSupportStatus::Supported,
                    decode: ObservationSupportStatus::Supported,
                    floating_to_f32: true,
                }],
            },
        },
        interventions: InterventionDiscovery {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            artifact_identity: "exact-source".into(),
            session_identity: Some("realized-11".into()),
            points: vec![edited],
        },
        bindings: vec![SpeculativeCaptureBinding {
            node_id: point.node_id.clone(),
            scope: SpeculativeCaptureScope::Prediction { depth: 2 },
        }],
    };
    let budget = CaptureUsage {
        captures: 32,
        retained_bytes: 1 << 20,
        host_bytes: 1 << 20,
        encoded_bytes: 1 << 20,
    };
    let plan = SpeculativeActivationPlan {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        captures: CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: vec![CaptureSelection {
                id: "units".into(),
                path: point.path.clone(),
                schedule: Default::default(),
                slices: vec![],
                transform: CaptureTransform::Preview { max_elements: 64 },
            }],
            limits: CaptureLimits {
                per_step: budget,
                cumulative: budget.checked_mul(8).unwrap(),
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        },
        interventions: InterventionPlan {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations: vec![InterventionOperation {
                id: "scale".into(),
                target: point.path,
                schedule: Default::default(),
                slices: vec![],
                action: InterventionAction::Scale {
                    dtype: InterventionDtype::Float32,
                    factor: 0.5,
                },
                evidence: InterventionEvidence::Preview { max_elements: 64 },
            }],
        },
        bounds: CaptureInvocationBounds {
            batch: 1,
            max_sequence: 4,
            max_context: None,
            max_predictions: 8,
        },
    };
    (plan, discovery)
}

#[test]
fn admitted_internal_plan_binds_geometry_scopes_and_loaded_provenance() {
    let (plan, discovery) = fixture();
    assert_eq!(
        serde_json::from_value::<SpeculativeActivationDiscovery>(
            serde_json::to_value(&discovery).unwrap()
        )
        .unwrap(),
        discovery
    );
    let restored: SpeculativeActivationPlan =
        serde_json::from_value(serde_json::to_value(&plan).unwrap()).unwrap();
    assert_eq!(restored, plan);
    let admitted = restored.admit(&discovery).unwrap();
    assert!(!admitted.is_empty());
    assert_eq!(admitted.captures().invocation_bounds(), Some(plan.bounds));
    assert_eq!(
        admitted.interventions().invocation_bounds(),
        Some(plan.bounds)
    );
    assert_eq!(
        admitted.capture_scopes(),
        &[SpeculativeCaptureScope::Prediction { depth: 2 }]
    );
    assert_eq!(admitted.capture_scopes(), admitted.intervention_scopes());
    admitted.validate(&discovery).unwrap();
    for changed in 0..4 {
        let mut other = discovery.clone();
        match changed {
            0 => other.execution_identity.push_str("-new-overlay"),
            1 => {
                other.captures.artifact_identity.push_str("-other");
                other.interventions.artifact_identity = other.captures.artifact_identity.clone();
            }
            2 => other.interventions.session_identity = Some("reloaded-12".into()),
            _ => other.bindings[0].scope = SpeculativeCaptureScope::Target,
        }
        assert!(admitted.validate(&other).is_err());
        assert_ne!(
            plan.clone().admit(&other).unwrap().identity(),
            admitted.identity()
        );
    }
    let mut shorter = plan.clone();
    shorter.bounds.max_sequence = 3;
    assert_ne!(
        shorter.admit(&discovery).unwrap().identity(),
        admitted.identity()
    );
}

#[test]
fn unresolved_scope_support_missing_bindings_and_invalid_geometry_cannot_admit() {
    let (plan, discovery) = fixture();
    for invalid in 0..6 {
        let mut other = discovery.clone();
        match invalid {
            0 => other.interventions.session_identity = None,
            1 => other.bindings.clear(),
            2 => other.bindings.push(other.bindings[0].clone()),
            3 => other.interventions.artifact_identity = "different".into(),
            4 => {
                other.captures.support.points[0].decode =
                    ObservationSupportStatus::Unverified("prediction hook unresolved".into())
            }
            _ => {
                other.interventions.points[0].prefill =
                    ObservationSupportStatus::Unsupported("call path absent".into())
            }
        }
        assert!(plan.clone().admit(&other).is_err());
    }
    for invalid in 0..3 {
        let mut other = plan.clone();
        match invalid {
            0 => other.schema_version = 9,
            1 => other.bounds.max_sequence = 0,
            _ => other.interventions.operations[0].slices.push(CaptureSlice {
                axis: "sequence".into(),
                start: 0,
                end: 5,
                stride: 1,
            }),
        }
        assert!(other.admit(&discovery).is_err());
    }
    let mut empty = plan;
    empty.captures.selections.clear();
    empty.interventions.operations.clear();
    assert!(empty.admit(&discovery).unwrap().is_empty());
}

#[test]
fn conditional_scope_support_preserves_the_admitted_invocation_binding() {
    let (plan, mut discovery) = fixture();
    discovery.captures.support.points[0].decode =
        ObservationSupportStatus::Conditional("prediction group must execute".into());
    discovery.interventions.points[0].prefill =
        ObservationSupportStatus::Conditional("prediction group must execute".into());
    let admitted = plan.admit(&discovery).unwrap();
    assert_eq!(
        admitted.capture_scopes(),
        &[SpeculativeCaptureScope::Prediction { depth: 2 }]
    );
    assert_eq!(admitted.capture_scopes(), admitted.intervention_scopes());
    assert_eq!(
        admitted.interventions().points()[0].prefill,
        discovery.interventions.points[0].prefill
    );
    admitted.validate(&discovery).unwrap();
}

#[test]
fn context_and_fused_scopes_have_distinct_phases_and_admitted_identities() {
    use SpeculativeActivationPhase as P;
    use SpeculativeCaptureScope as S;
    let phases = [
        P::TargetPrefill,
        P::PredictionPrefill,
        P::Proposal { depth: 0 },
        P::Proposal { depth: 1 },
        P::FusedProposal,
        P::Verification,
        P::PredictionReplay,
        P::TargetReplay,
    ];
    for (scope, expected) in [
        (
            S::Target,
            [true, false, false, false, false, true, false, true],
        ),
        (
            S::Prediction { depth: 1 },
            [false, true, false, true, false, false, true, false],
        ),
        (
            S::PredictionContext,
            [false, true, false, false, false, false, true, false],
        ),
        (
            S::FusedProposal,
            [false, false, false, false, true, false, false, false],
        ),
    ] {
        assert_eq!(phases.map(|phase| scope.applies(phase)), expected);
        let (plan, mut discovery) = fixture();
        discovery.bindings[0].scope = scope;
        let decoded = serde_json::from_slice::<SpeculativeActivationDiscovery>(
            &serde_json::to_vec(&discovery).unwrap(),
        )
        .unwrap();
        let admitted = plan.clone().admit(&decoded).unwrap();
        assert_eq!(admitted.capture_scopes(), &[scope]);
        for other in [
            S::Target,
            S::Prediction { depth: 0 },
            S::PredictionContext,
            S::FusedProposal,
        ] {
            if scope != other {
                discovery.bindings[0].scope = other;
                assert!(admitted.validate(&discovery).is_err());
                assert_ne!(
                    plan.clone().admit(&discovery).unwrap().identity(),
                    admitted.identity()
                );
            }
        }
        let mut old_plan = plan;
        old_plan.schema_version = 1;
        assert!(old_plan.admit(&decoded).is_err());
    }
}
