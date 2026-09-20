use super::*;

fn coordinates(spans: &[ReadSpan]) -> Vec<(Range<u64>, Range<usize>)> {
    spans
        .iter()
        .map(|span| (span.source.clone(), span.destination.clone()))
        .collect()
}

#[test]
fn counted_projection_preserves_repeated_intersections_and_exact_destinations() {
    let spans = [
        ReadSpan {
            source: 100..104,
            destination: 0..4,
        },
        ReadSpan {
            source: 500..504,
            destination: 8..12,
        },
    ];
    let ranges = [
        EncodedRange {
            source: 2..10,
            destination: 0..8,
        },
        EncodedRange {
            source: 0..4,
            destination: 8..12,
        },
        EncodedRange {
            source: 2..4,
            destination: 12..14,
        },
    ];
    let plan = SpanProjectionPlan::new(&spans, &ranges).unwrap();
    assert_eq!(plan.count, 4);
    assert_eq!(plan.covered, 10);
    assert_eq!(
        plan.layout,
        std::alloc::Layout::array::<ReadSpan>(4).unwrap()
    );
    for length in [3, 5] {
        let mut destination = vec![
            ReadSpan {
                source: 91..97,
                destination: 31..37
            };
            length
        ];
        assert!(
            matches!(plan.fill_into(&mut destination), Err(EncodedProjectionError::Destination { expected: 4, actual }) if actual == length)
        );
        assert_eq!(coordinates(&destination), vec![(91..97, 31..37); length]);
    }
    let mut destination = vec![
        ReadSpan {
            source: 0..0,
            destination: 0..0
        };
        4
    ];
    plan.fill_into(&mut destination).unwrap();
    assert!(
        destination
            .windows(2)
            .all(|pair| pair[0].source.start <= pair[1].source.start)
    );
    let mut actual = coordinates(&destination);
    actual.sort_by_key(|(source, destination)| (source.start, destination.start));
    assert_eq!(
        actual,
        vec![
            (100..104, 8..12),
            (102..104, 0..2),
            (102..104, 12..14),
            (500..502, 6..8)
        ]
    );

    let source = [ReadSpan {
        source: 0..4_000_000_000,
        destination: 0..4_000_000_000,
    }];
    let selected = [EncodedRange {
        source: 2_000_000_000..2_000_000_004,
        destination: 0..4,
    }];
    let plan = SpanProjectionPlan::new(&source, &selected).unwrap();
    assert_eq!((plan.count, plan.covered), (1, 4));
    let mut destination = [ReadSpan {
        source: 0..0,
        destination: 0..0,
    }];
    plan.fill_into(&mut destination).unwrap();
    assert_eq!(
        coordinates(&destination),
        vec![(2_000_000_000..2_000_000_004, 0..4)]
    );
    let empty = SpanProjectionPlan::new(&source, &[]).unwrap();
    assert_eq!((empty.count, empty.covered, empty.layout.size()), (0, 0, 0));
    empty.fill_into(&mut []).unwrap();
}

#[test]
fn projection_plan_rejects_invalid_original_spans() {
    let ranges = [EncodedRange {
        source: 0..4,
        destination: 0..4,
    }];
    for spans in [
        vec![ReadSpan {
            source: 10..9,
            destination: 0..4,
        }],
        vec![ReadSpan {
            source: 10..12,
            destination: 0..4,
        }],
        vec![ReadSpan {
            source: 10..14,
            destination: 4..0,
        }],
        vec![
            ReadSpan {
                source: 10..14,
                destination: 0..4,
            },
            ReadSpan {
                source: 20..24,
                destination: 3..7,
            },
        ],
        vec![
            ReadSpan {
                source: 10..14,
                destination: 4..8,
            },
            ReadSpan {
                source: 20..24,
                destination: 0..4,
            },
        ],
    ] {
        assert!(matches!(
            SpanProjectionPlan::new(&spans, &ranges),
            Err(EncodedProjectionError::Invalid)
        ));
    }
}

#[test]
fn projected_file_and_memory_batches_retain_sources_and_exact_output() {
    use crate::store::{CheckpointSource, MemoryWeightStore, SafetensorsWeightStore};
    use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weights.safetensors");
    let a = [1, 2, 3, 4];
    let b = [11, 12, 13, 14];
    serialize_to_file(
        [
            ("a", TensorView::new(Dtype::U8, vec![4], &a).unwrap()),
            ("b", TensorView::new(Dtype::U8, vec![4], &b).unwrap()),
        ],
        None,
        &path,
    )
    .unwrap();
    let sources: Vec<Box<dyn CheckpointSource>> = vec![
        Box::new(SafetensorsWeightStore::open(&path).unwrap()),
        Box::new(
            MemoryWeightStore::from_safetensors([
                ("a".into(), Dtype::U8, vec![4], a.to_vec()),
                ("b".into(), Dtype::U8, vec![4], b.to_vec()),
            ])
            .unwrap(),
        ),
    ];
    let ranges = [
        EncodedRange {
            source: 6..8,
            destination: 0..2,
        },
        EncodedRange {
            source: 0..3,
            destination: 2..5,
        },
        EncodedRange {
            source: 6..7,
            destination: 5..6,
        },
    ];
    for source in sources {
        let batch = source
            .prepare_encoded_read(&["a".into(), "b".into()])
            .unwrap()
            .unwrap();
        let projected = batch.project_ranges(&ranges, 6).unwrap();
        assert_eq!(projected.byte_len(), 6);
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
        drop(source);
        let mut output = [0; 6];
        projected.read_into(&mut output).unwrap();
        assert_eq!(output, [13, 14, 1, 2, 3, 13]);
    }
}

fn prepared_memory(custody: Arc<()>) -> PreparedEncodedRead<Arc<()>> {
    let store = MemoryWeightStore::from_safetensors([
        ("a".into(), Dtype::U8, vec![4], vec![1, 2, 3, 4]),
        ("b".into(), Dtype::U8, vec![4], vec![11, 12, 13, 14]),
    ]).unwrap();
    MemoryEncodedReadPlan::new(&store, &["a".into(), "b".into()]).unwrap().construct(custody).unwrap()
}

#[test]
fn invalid_projection_keeps_the_original_read_and_custody() {
    let custody = Arc::new(());
    let alive = Arc::downgrade(&custody);
    let read = prepared_memory(custody);
    let mapping = crate::recipe::EncodedRecipeMappingPlan::source(0..9).unwrap().build().unwrap();
    let failure = match read.project(&mapping) {
        Ok(_) => panic!("mapping exceeds the original read"),
        Err(error) => error,
    };
    drop(mapping);
    assert!(matches!(failure.cause(), EncodedProjectionError::Invalid));
    assert_eq!(failure.retained_tensors().len(), 2);
    assert!(alive.upgrade().is_some());
    drop(failure);
    assert!(alive.upgrade().is_none());
}

#[test]
fn later_projection_failure_retains_replaced_prefix_and_both_custodies() {
    let original = Arc::new(());
    let source_alive = Arc::downgrade(&original);
    let projected = Arc::new(());
    let projection_alive = Arc::downgrade(&projected);
    let read = prepared_memory(original);
    let full = crate::recipe::EncodedRecipeMappingPlan::source(0..8).unwrap().build().unwrap();
    let mapping = crate::recipe::EncodedRecipeMappingPlan::selected(&full, &[6..8, 0..3, 6..7]).unwrap().build().unwrap();
    let mut plan = read.project(&mapping).unwrap();
    // Inject a later invalid span to exercise prefix ownership after the first
    // source has been replaced. Public plans expose no mutable source geometry.
    plan.read.batch.memory[1].spans[0].source = 9..8;
    let failure = plan.construct(projected).unwrap_err();
    drop((mapping, full));
    assert!(matches!(failure.cause(), EncodedProjectionError::Invalid));
    assert_eq!(coordinates(&failure.partial.batch.memory[0].spans), [(0..3, 2..5)]);
    assert_eq!(failure.retained_tensors().len(), 2);
    assert!(source_alive.upgrade().is_some());
    assert!(projection_alive.upgrade().is_some());
    assert!(std::error::Error::source(&failure).unwrap().is::<EncodedProjectionError>());
    drop(failure);
    assert!(source_alive.upgrade().is_none());
    assert!(projection_alive.upgrade().is_none());
}
