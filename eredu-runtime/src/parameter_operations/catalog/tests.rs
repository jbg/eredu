use super::*;
use eredu_core::{
    component::ComponentCoordinateMap,
    intervention::InterventionDtype,
    parameters::{ParameterAccess, ProjectionInputTransform},
};

#[derive(Default)]
struct Budget {
    usage: CaptureUsage,
    denied: bool,
}
impl CaptureReservation for Budget {
    fn reserve(&mut self, cost: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        if self.denied {
            return Err(CaptureError::Limit {
                budget: CaptureBudget::Host,
                cumulative: true,
            });
        }
        self.usage = self.usage.checked_add(cost)?;
        Ok(None)
    }
}
fn ranks() -> Vec<Vec<LocalParameterFacts>> {
    (0..4)
        .map(|rank| {
            vec![LocalParameterFacts {
                parameter: LoadedParameter {
                    id: if rank < 2 { "weight" } else { "tied" }.into(),
                    shared_id: "weight".into(),
                    shape: vec![4, 6],
                    dtype: Some(InterventionDtype::Float32),
                    supported: false,
                    access: Some(ParameterAccess {
                        query: true,
                        projection: true,
                        replacement: false,
                    }),
                    condition: "Actual F32 slot; public reads available".into(),
                    input_transform: ProjectionInputTransform::Identity,
                },
                local_shape: vec![4, 3],
            }]
        })
        .collect()
}
fn coordinate(
    rank: usize,
    _: &LoadedParameter,
    _: &mut Budget,
) -> Result<Option<ParameterCoordinateMap>, ParameterError> {
    Ok(Some(ParameterCoordinateMap::new(
        vec![4, 6],
        vec![
            ComponentCoordinateMap::range(4, 0..4).unwrap(),
            ComponentCoordinateMap::range(6, (rank % 2) * 3..(rank % 2 + 1) * 3).unwrap(),
        ],
    )?))
}
#[test]
fn actual_tied_pipeline_slots_form_complete_catalogue_and_roundtrip_bounded_wire() {
    let mut budget = Budget::default();
    let decoded = ranks()
        .iter()
        .map(|facts| {
            let words = encode_parameter_catalog(facts, 4096, &mut budget).unwrap();
            let decoded = decode_parameter_catalog(&words, 4096, &mut budget).unwrap();
            assert_eq!(&decoded, facts);
            decoded
        })
        .collect::<Vec<_>>();
    let catalog = PartitionParameterCatalog::new(&decoded, coordinate, &mut budget).unwrap();
    assert_eq!(
        catalog
            .parameters()
            .iter()
            .map(|p| p.id.as_str())
            .collect::<Vec<_>>(),
        ["tied", "weight"]
    );
    for p in catalog.parameters() {
        assert!(p.access().query && p.access().projection);
        assert!(!p.access().replacement && !p.supported);
        assert_eq!(
            catalog
                .coordinates(&p.id)
                .unwrap()
                .iter()
                .filter(|p| p.is_some())
                .count(),
            2
        );
    }
    assert!(budget.usage.host_bytes > 0);
}
#[test]
fn partial_missing_duplicate_or_foreign_loaded_reports_cannot_become_complete_facts() {
    for fault in 0..5 {
        let mut ranks = ranks();
        match fault {
            0 => ranks[1].clear(),
            1 => {
                let extra = ranks[0][0].clone();
                ranks[0].push(extra);
            }
            2 => ranks[1][0].local_shape[1] = 2,
            3 => ranks[1][0].parameter.shared_id = "foreign".into(),
            _ => ranks[1][0].parameter.shape[1] = 9,
        }
        assert!(
            PartitionParameterCatalog::new(&ranks, coordinate, &mut Budget::default()).is_err(),
            "fault {fault}"
        );
    }
    let mut called = false;
    assert!(PartitionParameterCatalog::new(
        &ranks(),
        |_, _, _| {
            called = true;
            unreachable!()
        },
        &mut Budget {
            denied: true,
            ..Default::default()
        }
    )
    .is_err());
    assert!(!called);
}
#[test]
fn weaker_native_support_and_inconsistent_dtypes_are_exposed_without_false_admission() {
    let mut facts = ranks();
    facts[1][0].parameter.access = Some(ParameterAccess::default());
    let catalog =
        PartitionParameterCatalog::new(&facts, coordinate, &mut Budget::default()).unwrap();
    assert_eq!(
        catalog
            .parameters()
            .iter()
            .find(|p| p.id == "weight")
            .unwrap()
            .access(),
        ParameterAccess::default()
    );
    facts[1][0].parameter.dtype = Some(InterventionDtype::Float16);
    let catalog =
        PartitionParameterCatalog::new(&facts, coordinate, &mut Budget::default()).unwrap();
    assert!(catalog
        .parameters()
        .iter()
        .find(|p| p.id == "weight")
        .unwrap()
        .dtype
        .is_none());
}
#[test]
fn malformed_or_excessive_wire_rejects_before_metadata_publication() {
    let mut budget = Budget::default();
    assert!(encode_parameter_catalog(&ranks()[0], 8, &mut budget).is_err());
    let words = encode_parameter_catalog(&ranks()[0], 4096, &mut budget).unwrap();
    assert!(decode_parameter_catalog(&words, 1, &mut budget).is_err());
    let mut changed = words.clone();
    changed[0] = u32::MAX;
    assert!(decode_parameter_catalog(&changed, 4096, &mut budget).is_err());
    let mut changed = words;
    changed.push(0);
    assert!(decode_parameter_catalog(&changed, 4096, &mut budget).is_err());
}

#[test]
fn empty_owners_do_not_downgrade_the_nonempty_native_realization() {
    let mut reports = ranks();
    let mut empty = reports[0][0].clone();
    empty.local_shape[1] = 0;
    empty.parameter.access = Some(ParameterAccess::default());
    empty.parameter.dtype = None;
    reports.insert(0, vec![empty]);
    let catalog = PartitionParameterCatalog::new(
        &reports,
        |rank, p, budget| {
            if rank == 0 {
                Ok(Some(ParameterCoordinateMap::new(
                    vec![4, 6],
                    vec![
                        ComponentCoordinateMap::range(4, 0..4).unwrap(),
                        ComponentCoordinateMap::range(6, 0..0).unwrap(),
                    ],
                )?))
            } else {
                coordinate(rank - 1, p, budget)
            }
        },
        &mut Budget::default(),
    )
    .unwrap();
    let weight = catalog
        .parameters()
        .iter()
        .find(|p| p.id == "weight")
        .unwrap();
    assert!(weight.access().query);
    assert_eq!(weight.dtype, Some(InterventionDtype::Float32));
}
