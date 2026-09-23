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
