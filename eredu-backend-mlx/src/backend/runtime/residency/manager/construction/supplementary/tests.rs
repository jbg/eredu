use super::*;
use eredu_checkpoint::store::TensorSelection;

#[test]
fn supplementary_inventory_preserves_physical_ids_and_singleton_repeated_requests() {
    let id = |name| OffloadUnitId::new(name).unwrap();
    let units: Vec<_> = [
        "target.0",
        "prediction.module.00001",
        "prediction.module.00003",
        "target.1",
    ]
    .into_iter()
    .map(|name| {
        OffloadUnit::new(
            id(name),
            vec![WeightBinding::new("weight", name, TensorSelection::Full, 16).unwrap()],
        )
        .unwrap()
    })
    .collect();
    let selected = [id("target.0"), id("target.1")];
    let source = SupplementarySourcePlan::prepare(&units, &selected)
        .unwrap()
        .unwrap();
    assert_eq!(
        source.ids,
        [id("prediction.module.00001"), id("prediction.module.00003")]
    );
    assert_eq!(source.definitions, [units[1].clone(), units[2].clone()]);
    // Dense call indices are separate from physical module ordinals, and the
    // source inventory never turns repeats into extra source IDs.
    let calls = [1usize, 0, 1, 1];
    let resolved: Vec<_> = calls
        .into_iter()
        .map(|dense| {
            let range = source
                .layout
                .window_range(dense, NonZeroUsize::new(1).unwrap())
                .unwrap();
            assert_eq!(range.len(), 1);
            source.ids[range.start].as_str()
        })
        .collect();
    assert_eq!(
        resolved,
        [
            "prediction.module.00003",
            "prediction.module.00001",
            "prediction.module.00003",
            "prediction.module.00003"
        ]
    );
    assert!(!source.ids.contains(&id("foreign")));
    assert!(
        source
            .layout
            .window_range(2, NonZeroUsize::new(1).unwrap())
            .is_none()
    );
    assert!(
        SupplementarySourcePlan::prepare(
            &units,
            &units.iter().map(|u| u.id().clone()).collect::<Vec<_>>()
        )
        .unwrap()
        .is_none()
    );
}
