use super::*;
use std::collections::VecDeque;

fn controller(edges: &[Vec<usize>]) -> (ResidencyController, Vec<OffloadUnitId>) {
    let ids = (0..edges.len())
        .map(|i| OffloadUnitId::new(format!("unit.{i:04}")).unwrap())
        .collect::<Vec<_>>();
    let catalog = Catalog(
        (0..edges.len())
            .map(|i| {
                let key = format!("physical.{i}");
                (key.clone(), metadata(&key, vec![1]))
            })
            .collect(),
    );
    let units = edges
        .iter()
        .enumerate()
        .rev()
        .map(|(i, owners)| {
            let mut bindings = vec![WeightBinding::new(
                "owner",
                format!("physical.{i}"),
                TensorSelection::Full,
                4,
            )
            .unwrap()
            .with_logical_target(format!("logical.{i}"))
            .unwrap()];
            for (j, owner) in owners.iter().enumerate() {
                bindings.push(
                    WeightBinding::alias(format!("alias.{j}"), format!("logical.{owner}"), 4)
                        .unwrap()
                        .with_logical_target(format!("alias.{i}.{j}"))
                        .unwrap(),
                );
            }
            OffloadUnit::new(ids[i].clone(), bindings).unwrap()
        })
        .collect::<Vec<_>>();
    let plan = OffloadPlan::new(
        OffloadConfig::default(),
        ids.iter().map(|id| {
            OffloadUnitSpec::new(id.clone(), 4, ResidencyPolicy::Windowed, MemoryTier::Disk)
                .unwrap()
        }),
    )
    .unwrap();
    (
        ResidencyController::new(&catalog, plan, units).unwrap(),
        ids,
    )
}

#[test]
fn canonical_source_closure_accepts_real_cross_unit_cycles_and_preserves_identity() {
    // Each alias resolves directly to a physical owner: no tensor alias cycle.
    // Runtime unit recursion used to cycle A -> B -> A before checking hits.
    let (control, ids) = controller(&[vec![1], vec![0], vec![]]);
    let mut scratch = vec![ResidencyClosureSlot::default(); control.units().len()];
    let closure = control.operation_closure(&ids[..1], &mut scratch).unwrap();
    assert_eq!(closure.len(), 2);
    assert_eq!(
        closure.units().map(OffloadUnit::id).collect::<Vec<_>>(),
        vec![&ids[0], &ids[1]]
    );
    for unit in closure.units() {
        assert!(std::ptr::eq(unit, control.unit(unit.id()).unwrap()));
        for binding in unit.bindings() {
            let ordinary = control.binding_owner(unit.id(), binding);
            let borrowed = control.binding_owner_borrowed(unit.id(), binding);
            assert_eq!(
                ordinary.map(|(id, b)| (id, b.name())),
                borrowed.map(|(id, b)| (id, b.name()))
            );
            if let (Some((ordinary_id, ordinary)), Some((borrowed_id, borrowed))) =
                (ordinary, borrowed)
            {
                assert!(std::ptr::eq(ordinary_id, borrowed_id));
                assert!(std::ptr::eq(ordinary, borrowed));
            }
        }
    }
}

#[test]
fn canonical_source_closure_matches_independent_queue_for_diamonds_duplicates_and_cycles() {
    let cases = [
        vec![vec![1, 1, 2], vec![3], vec![3], vec![]],
        vec![vec![0], vec![2], vec![1], vec![]],
        vec![vec![1], vec![2], vec![0, 3], vec![]],
        vec![vec![], vec![], vec![], vec![]],
    ];
    for edges in cases {
        let (control, ids) = controller(&edges);
        for root in 0..ids.len() {
            let mut expected = BTreeSet::new();
            let mut queue = VecDeque::from([root]);
            while let Some(next) = queue.pop_front() {
                if expected.insert(next) {
                    queue.extend(edges[next].iter().copied());
                }
            }
            // More root entries than the N-slot destination cannot overflow it.
            let roots = vec![ids[root].clone(); 3 * ids.len() + 1];
            let mut scratch = vec![ResidencyClosureSlot::default(); ids.len()];
            let closure = control.operation_closure(&roots, &mut scratch).unwrap();
            assert_eq!(
                closure.units().map(OffloadUnit::id).collect::<Vec<_>>(),
                expected.iter().map(|i| &ids[*i]).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn closure_destination_and_unknown_root_refuse_before_writes_and_empty_reuses_scratch() {
    let (control, ids) = controller(&[vec![1], vec![]]);
    let mut exact = vec![ResidencyClosureSlot::default(); 2];
    assert_eq!(
        control
            .operation_closure(&ids[..1], &mut exact)
            .unwrap()
            .len(),
        2
    );
    let populated = exact.clone();
    for length in [1, 3] {
        let mut destination = vec![populated[0]; length];
        let before = destination.clone();
        assert!(matches!(
            control.operation_closure(&ids, &mut destination),
            Err(ResidencyClosureError::DestinationLength)
        ));
        assert_eq!(destination, before);
    }
    let roots = [ids[0].clone(), OffloadUnitId::new("absent").unwrap()];
    assert!(matches!(
        control.operation_closure(&roots, &mut exact),
        Err(ResidencyClosureError::UnknownRoot)
    ));
    assert_eq!(exact, populated);
    let closure = control.operation_closure(&[], &mut exact).unwrap();
    assert!(closure.is_empty());
    assert_eq!(closure.units().count(), 0);
    assert!(ResidencyClosureSlot::layout(usize::MAX).is_none());
}

#[test]
fn deep_reverse_order_closure_has_no_recursive_source_stack() {
    let n = 96;
    let edges = (0..n)
        .map(|i| if i == 0 { vec![] } else { vec![i - 1] })
        .collect::<Vec<_>>();
    let (control, ids) = controller(&edges);
    let mut scratch = vec![ResidencyClosureSlot::default(); n];
    let closure = control
        .operation_closure(&ids[n - 1..], &mut scratch)
        .unwrap();
    assert_eq!(closure.len(), n);
    assert_eq!(
        closure.units().map(OffloadUnit::id).collect::<Vec<_>>(),
        ids.iter().collect::<Vec<_>>()
    );
}
