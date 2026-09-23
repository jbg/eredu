use crate::execution_resources::*;
use crate::parameter_operations::{PreparedParameterLocation, PreparedParameterSlot};
use crate::{
    select_replicated_text_realization, CacheResidencyPolicy, LayerWeightResidency,
    ReplicatedTextSelectionRequest,
};
use eredu_checkpoint::recipe::{RecipeDtype, RecipeMetadata};
use eredu_core::{resources::*, Observed};
use eredu_nn::{ParameterMetadata, ParameterSpec};

fn query() -> PreparedResourceQuery {
    PreparedResourceQuery {
        scope: ResourceIdentity {
            scope: "test-instance".into(),
            key: "target".into(),
        },
        batch_size: 1,
        prefix_positions: 5,
        additional_positions: 3,
        device_pool: Observed::Unavailable {
            reason: "no native pool selected by neutral fixture".into(),
        },
    }
}

fn selection() -> crate::SelectedReplicatedTextRealization {
    select_replicated_text_realization(
        &super::requirements(),
        &ReplicatedTextSelectionRequest::new(
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        ),
        &super::capabilities(),
    )
    .unwrap()
}

fn slot(name: &str, backing: Option<&str>, role: &str) -> PreparedParameterSlot {
    PreparedParameterSlot {
        parameter: ParameterMetadata::from_spec(&ParameterSpec::trainable(name).unwrap(), false),
        materialized: RecipeMetadata {
            shape: vec![4, 8],
            dtype: RecipeDtype::F16,
            byte_len: 64,
        },
        backing: backing.map(str::to_owned),
        location: PreparedParameterLocation::Static { role: role.into() },
    }
}

#[test]
fn resources_use_binding_sharing_and_keep_replicas_and_companions_distinct() {
    let selected = selection();
    let mut tied = slot("output.weight", Some("embedding"), "output");
    tied.parameter.alias_of = Some(eredu_nn::ParameterId::new("embedding.weight").unwrap());
    let slots = vec![
        slot("embedding.weight", Some("embedding"), "embedding"),
        tied,
        slot("copied.weight", Some("copied"), "output"),
        slot("output.scales", Some("scales"), "output"),
    ];
    let description =
        describe_prepared_resources(&selected, Some(selected.state()), &slots, &[], &query())
            .unwrap();
    let parameters = description
        .allocations
        .iter()
        .filter(|a| a.uses[0].role == ResourceRole::Parameters)
        .collect::<Vec<_>>();
    assert_eq!(parameters.len(), 3);
    assert_eq!(parameters.iter().map(|a| a.uses.len()).sum::<usize>(), 4);
    assert_eq!(
        parameters
            .iter()
            .find(|a| a.uses.len() == 2)
            .unwrap()
            .identity
            .key,
        "parameter/static/embedding"
    );
    assert!(parameters.iter().all(|a| matches!(&a.size, ResourceSize::Fixed { extent } if extent.payload == ResourceByteBounds::exact(64) && extent.capacity.upper_bytes.is_none())));
    let again =
        describe_prepared_resources(&selected, Some(selected.state()), &slots, &[], &query())
            .unwrap();
    assert_eq!(description, again);
    let mut other = query();
    other.scope.key = "independent-copy".into();
    let other = describe_prepared_resources(&selected, Some(selected.state()), &slots, &[], &other)
        .unwrap();
    assert_ne!(
        description.allocations[0].identity,
        other.allocations[0].identity
    );
}

#[test]
fn resources_do_not_infer_backing_from_logical_ties_or_aggregate_banks() {
    let selected = selection();
    let mut tied = slot("output.weight", None, "output");
    tied.parameter.alias_of = Some(eredu_nn::ParameterId::new("embedding.weight").unwrap());
    let mut bank = slot("bank.weight", Some("aggregate"), "bank");
    bank.location = PreparedParameterLocation::Bank { bank: 0, unit: 1 };
    let slots = [
        slot("embedding.weight", Some("embedding"), "embedding"),
        tied,
        bank,
    ];
    let description = describe_prepared_resources(&selected, None, &slots, &[], &query()).unwrap();
    assert_eq!(description.allocations.len(), 1);
    let ResourceCoverage::Partial { reasons } = &description.coverage else {
        panic!("incomplete mechanism facts cannot imply coverage")
    };
    assert!(reasons
        .iter()
        .any(|r| r.contains("output.weight") && r.contains("backing")));
    assert!(reasons
        .iter()
        .any(|r| r.contains("bank 0") && r.contains("per-member")));
    assert!(reasons
        .iter()
        .any(|r| r.contains("decoder") && r.contains("unit 0")));
}

#[test]
fn conflicting_backing_geometry_invalid_pool_and_overflow_are_rejected() {
    let selected = selection();
    let first = slot("a", Some("shared"), "embedding");
    let mut second = slot("b", Some("shared"), "output");
    second.materialized.byte_len = 128;
    assert!(describe_prepared_resources(&selected, None, &[first, second], &[], &query()).is_err());
    let mut q = query();
    q.prefix_positions = u64::MAX;
    assert!(selected.describe_prepared_resources(&q).is_err());
    q = query();
    q.device_pool = Observed::Available {
        value: ResourceIdentity {
            scope: "machine".into(),
            key: "memory".into(),
        },
        kind: eredu_core::ObservationKind::Estimated,
        source: "guess".into(),
    };
    assert!(selected.describe_prepared_resources(&q).is_err());
}

#[test]
fn selected_state_payload_uses_selected_dtype_and_horizon_without_capacity_claim() {
    let selected = selection();
    let description = selected.describe_prepared_resources(&query()).unwrap();
    assert_eq!(description.allocations.len(), 2);
    for allocation in &description.allocations {
        let ResourceSize::ContextDependent {
            current,
            horizon_peak,
        } = &allocation.size
        else {
            panic!("state is context dependent")
        };
        assert_eq!(
            current.payload,
            ResourceByteBounds {
                detail: "ordinary selected component shape, presence and storage dtype".into(),
                ..ResourceByteBounds::exact(80)
            }
        );
        assert_eq!(horizon_peak.payload.upper_bytes, Some(128));
        assert_eq!(current.capacity.upper_bytes, None);
        assert_eq!(horizon_peak.capacity.upper_bytes, None);
    }
    let mut q = query();
    q.additional_positions = 0;
    let zero = selected.describe_prepared_resources(&q).unwrap();
    for allocation in zero.allocations {
        let ResourceSize::ContextDependent {
            current,
            horizon_peak,
        } = allocation.size
        else {
            panic!("state")
        };
        assert_eq!(current, horizon_peak);
    }
    let selected = select_replicated_text_realization(
        &super::requirements(),
        &super::request(LayerWeightResidency::FullyResident),
        &super::capabilities(),
    )
    .unwrap();
    let paged = selected.describe_prepared_resources(&query()).unwrap();
    assert!(paged.allocations.is_empty());
    assert!(
        matches!(paged.coverage, ResourceCoverage::Partial { reasons } if reasons.iter().any(|r| r.contains("paged block")))
    );
}

#[test]
fn resources_preserve_segment_offsets_and_require_frame_horizons() {
    let mut selected = selection();
    selected.state.layout = crate::StateLayout::segmented(
        selected.state.layout.layers().clone(),
        [crate::StateSegmentSpec::new(
            "delayed",
            0..1,
            crate::StateSegmentLifetime::Persistent,
            -2,
        )
        .unwrap()],
    )
    .unwrap();
    let description = selected.describe_prepared_resources(&query()).unwrap();
    for allocation in description.allocations {
        assert!(allocation.uses[0].owner.key.contains("delayed"));
        let ResourceSize::ContextDependent {
            current,
            horizon_peak,
        } = allocation.size
        else {
            panic!("state")
        };
        assert_eq!(current.payload.upper_bytes, Some(48));
        assert_eq!(horizon_peak.payload.upper_bytes, Some(96));
    }
    selected.state.layout = crate::StateLayout::segmented(
        selected.state.layout.layers().clone(),
        [
            crate::StateSegmentSpec::new("frame", 0..1, crate::StateSegmentLifetime::FrameLocal, 0)
                .unwrap(),
        ],
    )
    .unwrap();
    let description = selected.describe_prepared_resources(&query()).unwrap();
    assert!(description.allocations.is_empty());
    assert!(
        matches!(description.coverage, ResourceCoverage::Partial { reasons } if reasons.iter().any(|r| r.contains("frame-local") && r.contains("frame horizon")))
    );
}

fn prediction_topology() -> crate::prediction_resources::EmbeddedPredictionTopology {
    use crate::{SpeculativeCaptureEntry, SpeculativeCaptureSchema, SpeculativeIdentity};
    let id = |name: &str| SpeculativeIdentity::new(name).unwrap();
    crate::prediction_resources::EmbeddedPredictionTopology {
        mode: crate::prediction_resources::PredictionExecutionMode::Sequential,
        proposal_capacity: 2,
        nodes: vec![],
        edges: vec![],
        parameters: vec![],
        invocations: vec![],
        state: vec![],
        target_features: SpeculativeCaptureSchema::new(
            id("features"),
            ["a", "b"].map(|name| {
                SpeculativeCaptureEntry::new(
                    id(name),
                    vec![1, 8, 4],
                    id("target"),
                    id(&format!("observation/{name}")),
                )
                .unwrap()
                .with_bounded_dimension(1)
                .unwrap()
            }),
        )
        .unwrap(),
    }
}

#[test]
fn prediction_resources_preserve_physical_parameter_owners_state_offsets_and_feature_views() {
    use crate::prediction_resources::*;
    let selected = selection();
    let topology = prediction_topology();
    let mut prediction = slot("prediction.weight", Some("weights"), "unused");
    prediction.location = PreparedParameterLocation::Prediction { module: 0 };
    let target = slot("target.weight", Some("weights"), "embedding");
    // Same local backing key in separate materialization batches is not sharing.
    let mut description = describe_prepared_resources(
        &selected,
        None,
        &[target.clone(), prediction.clone()],
        &[],
        &query(),
    )
    .unwrap();
    assert_eq!(description.allocations.len(), 2);
    let mut target_output = description.allocations[0].clone();
    target_output.identity.key = "actual-target-output".into();
    let backings = std::collections::BTreeMap::from([
        ("a".into(), target_output.clone()),
        ("b".into(), target_output),
    ]);
    let resource_query = PredictionResourceQuery {
        prepared: query(),
        floating_state_bytes: Some(2),
        feature_scalar_bytes: Some(2),
        feature_shapes: vec![vec![1, 4, 4], vec![1, 4, 4]],
        feature_backings: backings,
    };
    let state = PredictionStateLayer {
        layer: 0,
        policy: eredu_core::cache::LayerCachePolicy::key_value(
            eredu_core::AttentionPolicy::Full,
            1,
            4,
        )
        .unwrap(),
        processed_token_offset: -1,
    };
    let module = PreparedPredictionModule {
        ordinal: 0,
        residency_owner: None,
        shared: false,
        parameters: vec![prediction.parameter.clone()],
    };
    description = describe_prediction_resources(
        &topology,
        &selected,
        Some(&[state]),
        &[module],
        &[target, prediction],
        None,
        &resource_query,
    )
    .unwrap();
    let state = description
        .allocations
        .iter()
        .filter(|a| a.uses[0].role == ResourceRole::MutableState)
        .collect::<Vec<_>>();
    assert_eq!(state.len(), 2);
    for allocation in state {
        let ResourceSize::ContextDependent {
            current,
            horizon_peak,
        } = &allocation.size
        else {
            panic!("state")
        };
        assert_eq!(current.payload.lower_bytes, 32); // (5 - 1) positions * 4 * f16
        assert_eq!(horizon_peak.payload.upper_bytes, Some(56));
        assert_eq!(horizon_peak.capacity.upper_bytes, None);
    }
    let output = description
        .allocations
        .iter()
        .filter(|a| a.identity.key == "actual-target-output")
        .collect::<Vec<_>>();
    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0]
            .uses
            .iter()
            .filter(|usage| usage.role == ResourceRole::RetainedTensor)
            .count(),
        2
    );
    assert!(matches!(
        description.coverage,
        ResourceCoverage::Partial { .. }
    ));
    assert_eq!(
        PredictionExecutionMode::Sequential.prefill_sequence_len(0),
        0
    );
    assert_eq!(
        PredictionExecutionMode::Sequential.prefill_sequence_len(5),
        4
    );
    assert_eq!(PredictionExecutionMode::Fused.prefill_sequence_len(5), 5);
}

#[test]
fn feature_binding_rejects_bad_shapes_conflicts_and_duplicate_input_without_mutation() {
    use crate::prediction_resources::*;
    let selected = selection();
    let topology = prediction_topology();
    let mut description = describe_prepared_resources(
        &selected,
        None,
        &[slot("target", Some("target"), "embedding")],
        &[],
        &query(),
    )
    .unwrap();
    let original = description.clone();
    assert!(include_target_features(
        &mut description,
        &topology,
        vec![vec![1, 9, 4], vec![1, 4, 4]],
        2,
        &Default::default()
    )
    .is_err());
    assert_eq!(description, original);
    let mut conflicting = description.allocations[0].clone();
    conflicting.placement = Observed::unavailable("another pool");
    let backings = std::collections::BTreeMap::from([
        ("a".into(), description.allocations[0].clone()),
        ("b".into(), conflicting),
    ]);
    assert!(include_target_features(
        &mut description,
        &topology,
        vec![vec![1, 4, 4]; 2],
        2,
        &backings
    )
    .is_err());
    assert_eq!(description, original);
    description
        .allocations
        .push(description.allocations[0].clone());
    let invalid = description.clone();
    assert!(include_target_features(
        &mut description,
        &topology,
        vec![vec![1, 4, 4]; 2],
        2,
        &Default::default()
    )
    .is_err());
    assert_eq!(description, invalid);
}

#[test]
fn prediction_resource_slots_must_match_physical_module_declarations() {
    use crate::prediction_resources::*;
    let selected = selection();
    let topology = prediction_topology();
    let mut parameter = slot("shared-name", Some("backing"), "unused");
    parameter.location = PreparedParameterLocation::Prediction { module: 0 };
    let first = PreparedPredictionModule {
        ordinal: 0,
        residency_owner: None,
        shared: false,
        parameters: vec![parameter.parameter.clone()],
    };
    let mut second = first.clone();
    second.ordinal = 1;
    let query = PredictionResourceQuery {
        prepared: query(),
        floating_state_bytes: None,
        feature_scalar_bytes: None,
        feature_shapes: vec![],
        feature_backings: Default::default(),
    };
    let describe = |modules: &[PreparedPredictionModule], slots: &[PreparedParameterSlot]| {
        describe_prediction_resources(
            &topology,
            &selected,
            Some(&[]),
            modules,
            slots,
            None,
            &query,
        )
    };
    let description = describe(&[first.clone(), second.clone()], &[parameter.clone()]).unwrap();
    let ResourceCoverage::Partial { reasons } = description.coverage else {
        panic!("missing module")
    };
    assert!(reasons
        .iter()
        .any(|reason| reason.contains("prediction module 1 parameter \"shared-name\"")));
    assert!(!reasons
        .iter()
        .any(|reason| reason.contains("prediction module 0 parameter \"shared-name\"")));
    assert!(describe(&[first.clone(), first.clone()], &[]).is_err());
    let mut wrong = parameter.clone();
    wrong.location = PreparedParameterLocation::Prediction { module: 2 };
    assert!(describe(&[first.clone(), second], &[wrong]).is_err());
    let mut wrong = parameter.clone();
    wrong.parameter.alias_of = Some(eredu_nn::ParameterId::new("another-owner").unwrap());
    assert!(describe(std::slice::from_ref(&first), &[wrong]).is_err());
    assert!(describe(&[first], &[parameter.clone(), parameter]).is_err());
}

#[test]
fn prediction_sliding_state_keeps_window_minimum_and_full_prefix_upper_end() {
    use crate::prediction_resources::*;
    let selected = selection();
    let state = PredictionStateLayer {
        layer: 0,
        policy: eredu_core::cache::LayerCachePolicy::key_value(
            eredu_core::AttentionPolicy::Sliding {
                window: std::num::NonZeroU32::new(2).unwrap(),
            },
            1,
            4,
        )
        .unwrap(),
        processed_token_offset: -1,
    };
    let query = PredictionResourceQuery {
        prepared: query(),
        floating_state_bytes: Some(2),
        feature_scalar_bytes: None,
        feature_shapes: vec![],
        feature_backings: Default::default(),
    };
    let description = describe_prediction_resources(
        &prediction_topology(),
        &selected,
        Some(&[state]),
        &[],
        &[],
        None,
        &query,
    )
    .unwrap();
    assert_eq!(description.allocations.len(), 2);
    for allocation in description.allocations {
        let ResourceSize::ContextDependent {
            current,
            horizon_peak,
        } = allocation.size
        else {
            panic!("state")
        };
        assert_eq!(
            (current.payload.lower_bytes, current.payload.upper_bytes),
            (16, Some(32))
        );
        assert_eq!(
            (
                horizon_peak.payload.lower_bytes,
                horizon_peak.payload.upper_bytes
            ),
            (16, Some(56))
        );
        assert_eq!(current.payload.kind, eredu_core::ObservationKind::Estimated);
        assert_eq!(
            horizon_peak.payload.kind,
            eredu_core::ObservationKind::Estimated
        );
        assert_eq!(current.capacity.lower_bytes, 16);
    }
}
