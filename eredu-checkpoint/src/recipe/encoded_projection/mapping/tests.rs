use super::*;
use {
    EncodedRecipeMapping as Mapping, EncodedRecipeMappingInput as MappingInput,
    EncodedRecipeMappingPlan as MappingPlan,
};

fn coords(mapping: &Mapping) -> Vec<(Range<usize>, Range<usize>)> {
    mapping
        .ranges
        .iter()
        .map(|row| (row.source.clone(), row.destination.clone()))
        .collect()
}
fn source(range: Range<usize>) -> Mapping {
    let length = range.len();
    Mapping {
        ranges: vec![EncodedRange {
            source: range,
            destination: 0..length,
        }],
        length,
    }
}

#[test]
fn counted_sources_and_selections_keep_coalescing_and_refuse_wrong_destinations() {
    let empty = MappingPlan::new(MappingInput::Source(12..12)).unwrap();
    assert_eq!((empty.count, empty.length, empty.layout.size()), (0, 0, 0));
    assert!(empty.build().unwrap().ranges.is_empty());
    let plan = MappingPlan::new(MappingInput::Source(100..104)).unwrap();
    assert_eq!((plan.count, plan.length), (1, 4));
    assert_eq!(coords(&plan.build().unwrap()), vec![(100..104, 0..4)]);
    let input = Mapping {
        ranges: vec![
            EncodedRange {
                source: 100..104,
                destination: 0..4,
            },
            EncodedRange {
                source: 200..204,
                destination: 4..8,
            },
        ],
        length: 8,
    };
    let selections = [1..7, 0..2, 2..4];
    let plan = MappingPlan::new(MappingInput::Selected {
        input: &input,
        ranges: &selections,
    })
    .unwrap();
    assert_eq!((plan.count, plan.length), (3, 10));
    assert_eq!(plan.layout, Layout::array::<EncodedRange>(3).unwrap());
    for count in [2, 4] {
        let mut destination = vec![
            EncodedRange {
                source: 91..97,
                destination: 31..37
            };
            count
        ];
        assert!(matches!(
            plan.fill_into(&mut destination),
            Err(RecipeError::ArithmeticOverflow(_))
        ));
        assert!(
            destination
                .iter()
                .all(|row| row.source == (91..97) && row.destination == (31..37))
        );
    }
    let mapping = plan.build().unwrap();
    assert_eq!(
        coords(&mapping),
        vec![(101..104, 0..3), (200..203, 3..6), (100..104, 6..10)]
    );
    assert_eq!(mapping.length, 10);

    let huge = source(0..4_000_000_000);
    let selected = [2_000_000_000..2_000_000_004];
    let plan = MappingPlan::new(MappingInput::Selected {
        input: &huge,
        ranges: &selected,
    })
    .unwrap();
    assert_eq!((plan.count, plan.length), (1, 4));
    assert_eq!(
        coords(&plan.build().unwrap()),
        vec![(2_000_000_000..2_000_000_004, 0..4)]
    );
}

#[test]
fn interleaved_children_preserve_row_order_and_cross_child_adjacency() {
    let children = [source(0..6), source(100..104)];
    let refs = [&children[0], &children[1]];
    let plan = MappingPlan::new(MappingInput::Interleaved {
        children: Children::Borrowed(&refs, &[3, 2]),
        outer: 2,
    })
    .unwrap();
    assert_eq!((plan.count, plan.length), (4, 10));
    assert_eq!(
        coords(&plan.build().unwrap()),
        vec![
            (0..3, 0..3),
            (100..102, 3..5),
            (3..6, 5..8),
            (102..104, 8..10)
        ]
    );
    let adjacent = [source(0..4), source(4..8)];
    let refs = [&adjacent[0], &adjacent[1]];
    let plan = MappingPlan::new(MappingInput::Interleaved {
        children: Children::Borrowed(&refs, &[4, 4]),
        outer: 1,
    })
    .unwrap();
    assert_eq!((plan.count, plan.length), (1, 8));
    assert_eq!(coords(&plan.build().unwrap()), vec![(0..8, 0..8)]);
}

#[test]
fn mapping_count_refuses_bad_geometry_and_output_length_overflow() {
    assert!(MappingPlan::new(MappingInput::Source(4..2)).is_err());
    let child = source(0..8);
    for selected in [3..2, 0..9] {
        assert!(
            MappingPlan::new(MappingInput::Selected {
                input: &child,
                ranges: &[selected]
            })
            .is_err()
        );
    }
    let children = [child];
    for (chunks, outer) in [(vec![], 1), (vec![3], 2), (vec![2], usize::MAX)] {
        assert!(
            MappingPlan::new(MappingInput::Interleaved {
                children: Children::Borrowed(&[&children[0]], &chunks),
                outer
            })
            .is_err()
        );
    }
    let huge = source(0..usize::MAX);
    assert!(matches!(
        MappingPlan::new(MappingInput::Selected {
            input: &huge,
            ranges: &[0..usize::MAX, 0..1]
        }),
        Err(RecipeError::ArithmeticOverflow(_))
    ));
}
