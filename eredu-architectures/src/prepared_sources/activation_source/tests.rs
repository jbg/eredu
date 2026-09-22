use super::*;
use crate::prepared_sources::PreparedPredictionDiscovery;
use eredu_core::{capture::*, intervention::*, speculative::*, *};
use std::sync::Arc;

fn fixture() -> (
    PreparedModelDiscovery,
    SpeculativeActivationExecution,
    SpeculativeActivationPlan,
) {
    // The ordinary family descriptor supplies real target and prediction nodes,
    // axes, requirements and invocation bindings; no path-derived scopes.
    let configuration = crate::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&serde_json::json!({
            "model_type":"deepseek_v3", "hidden_size":8, "vocab_size":16,
            "num_hidden_layers":2, "num_attention_heads":2,
            "intermediate_size":10, "moe_intermediate_size":4,
            "q_lora_rank":3, "kv_lora_rank":3, "qk_nope_head_dim":2,
            "qk_rope_head_dim":2, "v_head_dim":3, "first_k_dense_replace":1,
            "n_routed_experts":2, "n_shared_experts":1, "num_experts_per_tok":1,
            "n_group":1, "topk_group":1, "max_position_embeddings":64,
            "num_nextn_predict_layers":2
        }))
        .unwrap();
    let architecture = configuration.architecture_plan();
    let descriptor = architecture.architecture_descriptor();
    let context = eredu_runtime::inspection::ObservationExecutionContext {
        activation_inspection: true,
        prediction_inspection: false,
        partitioned: false,
        selected: true,
        mechanisms: ObservationMechanisms {
            activation_tensors: true,
            routing_tensors: true,
            routed_unit_tensors: true,
            floating_to_f32: true,
        },
    };
    let mut support =
        eredu_runtime::inspection::observation_support(&descriptor.observations, context);
    support.capture = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Preview],
        ..Default::default()
    };
    let identity = artifact::fingerprint_artifact(
        "activation-validation",
        [artifact::ArtifactMemberIdentity::new("fixture", 1, [7; 32])],
    )
    .unwrap();
    let source = PreparedModelDiscovery {
        identity: artifact::DeferredArtifactIdentity::ready(identity),
        execution_identity: "selected-execution".into(),
        descriptor: descriptor.clone(),
        partition_selection: None,
        partition_parameters: None,
        partition_parameter_index: None,
        partition_hooks: None,
        support,
        observation_context: context,
        intervention_points: architecture.intervention_points(),
        prediction: Some(PreparedPredictionDiscovery {
            placement: Arc::new(Default::default()),
            descriptor,
            intervention_points: architecture.intervention_points(),
        }),
    };
    let execution = SpeculativeActivationExecution {
        depth: 2,
        strategy: eredu_runtime::SpeculativeStrategyClass::EmbeddedSequential,
    };
    let points = &source
        .prediction
        .as_ref()
        .unwrap()
        .descriptor
        .observations
        .points;
    let selections = [
        SpeculativeCaptureScope::Target,
        SpeculativeCaptureScope::Prediction { depth: 1 },
    ]
    .into_iter()
    .enumerate()
    .map(|(index, scope)| {
        let point = points
            .iter()
            .find(|point| {
                point.prefill
                    && point.decode
                    && point.axes.is_some()
                    && matches!(point.value_type, ObservationValueType::Tensor)
                    && SpeculativeActivationExecution::retained_scope(
                        &source.prediction.as_ref().unwrap().descriptor,
                        &point.node_id,
                    ) == Ok(scope)
            })
            .unwrap();
        CaptureSelection {
            id: format!("selected-{index}"),
            path: point.path.clone(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::Preview { max_elements: 8 },
        }
    })
    .collect();
    let allowance = CaptureUsage {
        captures: 32,
        retained_bytes: 1 << 20,
        host_bytes: 1 << 20,
        encoded_bytes: 1 << 20,
    };
    let plan = SpeculativeActivationPlan {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        captures: CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections,
            limits: CaptureLimits {
                per_step: allowance,
                cumulative: allowance.checked_mul(8).unwrap(),
                on_limit: CaptureLimitPolicy::Fail,
            },
        },
        interventions: InterventionPlan {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations: vec![],
        },
        bounds: CaptureInvocationBounds {
            batch: 1,
            max_sequence: 4,
            max_context: Some(16),
            max_predictions: 8,
        },
    };
    (source, execution, plan)
}
fn admit(
    source: &PreparedModelDiscovery,
    execution: &SpeculativeActivationExecution,
    plan: SpeculativeActivationPlan,
) -> AdmittedSpeculativeActivations {
    let discovery = source
        .speculative_activations(
            execution,
            &InterventionMechanisms::default(),
            "session",
            Some("overlay"),
        )
        .unwrap();
    plan.admit(&discovery).unwrap()
}
#[test]
fn independent_target_capture_uses_ordinary_declarations_without_prediction_catalog() {
    let (mut source, _, mut plan) = fixture();
    source.prediction = None;
    let execution = SpeculativeActivationExecution::ordinary_target();
    let discovery = source
        .autoregressive_activations(
            &InterventionMechanisms::default(),
            "session",
            Some("overlay"),
        )
        .unwrap();
    assert!(!discovery.captures.catalog.points.is_empty());
    assert!(
        discovery
            .bindings
            .iter()
            .all(|binding| binding.scope == SpeculativeCaptureScope::Target)
    );
    assert!(
        plan.clone().admit(&discovery).is_err(),
        "prediction hooks require their selected executor"
    );
    plan.captures.selections.truncate(1);
    let admitted = plan.admit(&discovery).unwrap();
    source
        .validate_original_speculative_activations(
            &admitted,
            &execution,
            "session",
            Some("overlay"),
        )
        .unwrap();
    let path = &admitted.captures().points()[0].path;
    source
        .descriptor
        .observations
        .points
        .iter_mut()
        .find(|point| point.path == *path)
        .unwrap()
        .meaning
        .push_str(" stale");
    assert_eq!(
        source.validate_original_speculative_activations(
            &admitted,
            &execution,
            "session",
            Some("overlay"),
        ),
        Err(E::Declaration)
    );
}

#[test]
fn original_activation_validation_borrows_exact_ordinary_catalog_and_scopes() {
    let (source, execution, plan) = fixture();
    let admitted = admit(&source, &execution, plan);
    let points = admitted.captures().points().as_ptr();
    let identity = admitted.identity().as_ptr();
    for _ in 0..3 {
        source
            .validate_original_speculative_activations(
                &admitted,
                &execution,
                "session",
                Some("overlay"),
            )
            .unwrap();
        let ordinary = source
            .speculative_activations(
                &execution,
                &InterventionMechanisms::default(),
                "session",
                Some("overlay"),
            )
            .unwrap();
        admitted.validate(&ordinary).unwrap();
        assert_eq!(admitted.captures().points().as_ptr(), points);
        assert_eq!(admitted.identity().as_ptr(), identity);
    }
    assert!(
        PreparedModelDiscovery::original_speculative_activation_validation_control_bytes().unwrap()
            > 0
    );
    for (session, overlay) in [
        ("foreign", Some("overlay")),
        ("session", None),
        ("session", Some("foreign")),
    ] {
        assert_eq!(
            source
                .validate_original_speculative_activations(&admitted, &execution, session, overlay),
            Err(E::Identity)
        );
    }
}
#[test]
fn original_activation_validation_rejects_changed_declarations_scopes_and_hooks() {
    let (source, execution, plan) = fixture();
    let admitted = admit(&source, &execution, plan);
    let path = &admitted.captures().points()[1].path;
    for mutation in 0..9 {
        let mut changed = source.clone();
        let prediction = changed.prediction.as_mut().unwrap();
        match mutation {
            0 => prediction
                .descriptor
                .observations
                .points
                .iter_mut()
                .find(|p| p.path == *path)
                .unwrap()
                .meaning
                .push_str(" stale"),
            1 => prediction
                .descriptor
                .observations
                .points
                .iter_mut()
                .find(|p| p.path == *path)
                .unwrap()
                .axes
                .as_mut()
                .unwrap()[0]
                .name
                .push_str(" stale"),
            2 => {
                let binding = SpeculativeCaptureBinding {
                    node_id: prediction.descriptor.component_scopes[0].node_id.clone(),
                    scope: SpeculativeCaptureScope::Prediction { depth: 0 },
                };
                prediction
                    .descriptor
                    .speculative_invocations
                    .extend([binding.clone(), binding]);
            }
            3 => changed.observation_context.activation_inspection = false,
            4 => changed.observation_context.mechanisms.activation_tensors = false,
            5 => changed.observation_context.selected = false,
            6 => changed.support.capture.transformations.clear(),
            7 => {
                prediction
                    .descriptor
                    .observations
                    .points
                    .iter_mut()
                    .find(|p| p.path == *path)
                    .unwrap()
                    .decode = false
            }
            8 => changed.support.schema_version = 0,
            _ => unreachable!(),
        }
        assert_eq!(
            changed.validate_original_speculative_activations(
                &admitted,
                &execution,
                "session",
                Some("overlay")
            ),
            Err(E::Declaration),
            "mutation {mutation}"
        );
    }
    for strategy in [
        eredu_runtime::SpeculativeStrategyClass::EmbeddedSequential,
        eredu_runtime::SpeculativeStrategyClass::EmbeddedFused,
    ] {
        let other = SpeculativeActivationExecution { depth: 1, strategy };
        assert_eq!(
            source.validate_original_speculative_activations(
                &admitted,
                &other,
                "session",
                Some("overlay")
            ),
            Err(E::Declaration)
        );
    }
}
#[test]
fn original_internal_edits_require_the_exact_static_producer() {
    validate_internal_edit_evidence(InterventionEvidence::None);
}
#[test]
fn original_internal_preview_summary_keep_exact_static_source_phase_and_scope_checks() {
    validate_internal_edit_evidence(InterventionEvidence::Preview { max_elements: 3 });
    validate_internal_edit_evidence(InterventionEvidence::Summary);
}
fn validate_internal_edit_evidence(evidence: InterventionEvidence) {
    let (source, execution, mut plan) = fixture();
    let mechanisms = InterventionMechanisms {
        operations: vec![InterventionKind::Scale],
        dtypes: vec![InterventionDtype::Float32],
        ..Default::default()
    };
    let discovery = source
        .speculative_activations(&execution, &mechanisms, "session", Some("overlay"))
        .unwrap();
    let point = discovery
        .interventions
        .points
        .iter()
        .find(|point| {
            point.operations.contains(&InterventionKind::Scale)
                && point.dtypes.contains(&InterventionDtype::Float32)
                && matches!(
                    point.prefill,
                    ObservationSupportStatus::Supported | ObservationSupportStatus::Conditional(_)
                )
                && matches!(
                    point.decode,
                    ObservationSupportStatus::Supported | ObservationSupportStatus::Conditional(_)
                )
                && point.routing.is_none()
                && point.routed_units.is_none()
        })
        .unwrap();
    plan.interventions.operations.push(InterventionOperation {
        id: "edit".into(),
        target: point.path.clone(),
        schedule: Default::default(),
        slices: vec![],
        action: InterventionAction::Scale {
            dtype: InterventionDtype::Float32,
            factor: 0.5,
        },
        evidence,
    });
    let admitted = plan.admit(&discovery).unwrap();
    assert_eq!(
        source.validate_original_speculative_activations(
            &admitted,
            &execution,
            "session",
            Some("overlay")
        ),
        Err(E::UnqualifiedIntervention)
    );
    let validate = |source: &PreparedModelDiscovery, facts| {
        source.validate_original_speculative_activations_with_interventions(
            &admitted,
            &execution,
            "session",
            Some("overlay"),
            facts,
        )
    };
    let points = admitted.interventions().points().as_ptr();
    validate(&source, mechanisms.borrowed()).unwrap();
    assert_eq!(admitted.interventions().points().as_ptr(), points);
    let mut changed = source.clone();
    changed
        .prediction
        .as_mut()
        .unwrap()
        .intervention_points
        .iter_mut()
        .find(|point| point.path == admitted.interventions().points()[0].path)
        .unwrap()
        .axes[0]
        .name
        .push_str("foreign");
    assert_eq!(
        validate(&changed, mechanisms.borrowed()),
        Err(E::Declaration)
    );
    let unavailable = InterventionMechanisms {
        dtypes: vec![],
        ..mechanisms.clone()
    };
    assert_eq!(
        validate(&source, unavailable.borrowed()),
        Err(E::Declaration)
    );
    let mut changed = source.clone();
    changed.observation_context.activation_inspection = false;
    assert_eq!(
        validate(&changed, mechanisms.borrowed()),
        Err(E::Declaration)
    );
}

#[test]
fn original_sparse_internal_source_requires_exact_producer_facts_and_retained_geometry() {
    let (source, execution, mut plan) = fixture();
    let mechanisms = InterventionMechanisms {
        routed_units: true,
        operations: vec![InterventionKind::Scale],
        dtypes: vec![InterventionDtype::Float32],
        ..Default::default()
    };
    let discovery = source
        .speculative_activations(&execution, &mechanisms, "session", Some("overlay"))
        .unwrap();
    let point = discovery
        .interventions
        .points
        .iter()
        .find(|point| {
            point.routed_units.is_some()
                && point.operations.contains(&InterventionKind::Scale)
                && point.dtypes.contains(&InterventionDtype::Float32)
                && matches!(
                    point.prefill,
                    ObservationSupportStatus::Supported | ObservationSupportStatus::Conditional(_)
                )
                && matches!(
                    point.decode,
                    ObservationSupportStatus::Supported | ObservationSupportStatus::Conditional(_)
                )
        })
        .unwrap();
    plan.interventions.operations.push(InterventionOperation {
        id: "sparse-edit".into(),
        target: point.path.clone(),
        schedule: Default::default(),
        slices: vec![],
        action: InterventionAction::Scale {
            dtype: InterventionDtype::Float32,
            factor: -0.5,
        },
        evidence: InterventionEvidence::None,
    });
    let admitted = plan.admit(&discovery).unwrap();
    assert_eq!(
        source.validate_original_speculative_activations_with_interventions(
            &admitted,
            &execution,
            "session",
            Some("overlay"),
            mechanisms.borrowed()
        ),
        Err(E::UnqualifiedIntervention)
    );
    let validate = |actual: &PreparedModelDiscovery, facts| {
        actual.validate_original_speculative_activations_with_routed_interventions(
            &admitted,
            &execution,
            "session",
            Some("overlay"),
            mechanisms.borrowed(),
            facts,
        )
    };
    let points = admitted.interventions().points().as_ptr();
    validate(&source, mechanisms.borrowed()).unwrap();
    assert_eq!(points, admitted.interventions().points().as_ptr());
    let wrong_dtype = InterventionMechanisms {
        dtypes: vec![InterventionDtype::Float16],
        ..mechanisms.clone()
    };
    assert_eq!(
        validate(&source, wrong_dtype.borrowed()),
        Err(E::UnqualifiedIntervention)
    );
    let wrong_action = InterventionMechanisms {
        operations: vec![InterventionKind::Zero],
        ..mechanisms.clone()
    };
    assert_eq!(
        validate(&source, wrong_action.borrowed()),
        Err(E::UnqualifiedIntervention)
    );
    let mut changed = source.clone();
    changed
        .prediction
        .as_mut()
        .unwrap()
        .intervention_points
        .iter_mut()
        .find(|point| point.path == admitted.interventions().points()[0].path)
        .unwrap()
        .routed_units
        .as_mut()
        .unwrap()
        .geometry
        .units_per_expert += 1;
    assert_eq!(
        validate(&changed, mechanisms.borrowed()),
        Err(E::Declaration)
    );
}
