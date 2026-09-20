use super::*;
use crate::store::{
    CheckpointSource, LeaseControlBorrowError, SafetensorsReadPlan, SafetensorsWeightStore,
    SourceMetadataBorrowError, checked_elements, invalid_selection,
};
use safetensors::tensor::{TensorView, serialize_to_file};

fn push_coalesced_range(ranges: &mut Vec<Range<usize>>, range: Range<usize>) {
    if let Some(previous) = ranges.last_mut() {
        if previous.end == range.start {
            previous.end = range.end;
            return;
        }
    }
    ranges.push(range);
}

fn old_plan_safetensors_reads(
    key: &str,
    dtype: Dtype,
    shape: &[usize],
    payload_len: usize,
    selection: &TensorSelection,
    output_shape: &[usize],
    policy: ReadPolicy,
) -> Result<SafetensorsReadPlan, StoreError> {
    let bounded = matches!(policy, ReadPolicy::RequireBounded);
    if matches!(selection, TensorSelection::Full) {
        return Ok(SafetensorsReadPlan::single(0..payload_len, true));
    }
    let bits = dtype.bitsize();
    let scalar_bytes = bits.checked_div(8).filter(|_| bits.is_multiple_of(8));
    if let (
        Some(scalar_bytes),
        TensorSelection::Contiguous {
            offset_elements,
            shape,
        },
    ) = (scalar_bytes, selection)
    {
        let start =
            offset_elements
                .checked_mul(scalar_bytes)
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("contiguous byte start for {key:?}"),
                })?;
        let end = checked_elements(key, shape)?
            .checked_mul(scalar_bytes)
            .and_then(|length| start.checked_add(length))
            .ok_or_else(|| StoreError::Overflow {
                context: format!("contiguous byte end for {key:?}"),
            })?;
        if end > payload_len {
            return Err(invalid_selection(
                key,
                "contiguous byte span outside payload",
            ));
        }
        return Ok(SafetensorsReadPlan::single(start..end, true));
    }
    if let (
        Some(_),
        TensorSelection::Range {
            axis: 0,
            start,
            end,
        },
    ) = (scalar_bytes, selection)
    {
        let row_bytes = payload_len
            .checked_div(shape[0])
            .filter(|_| payload_len.is_multiple_of(shape[0]))
            .ok_or_else(|| invalid_selection(key, "payload is not row divisible"))?;
        let byte_start = start
            .checked_mul(row_bytes)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("row selection byte start for {key:?}"),
            })?;
        let byte_end = end
            .checked_mul(row_bytes)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("row selection byte end for {key:?}"),
            })?;
        return Ok(SafetensorsReadPlan::single(byte_start..byte_end, true));
    }
    if !bounded {
        return Ok(SafetensorsReadPlan::single(0..payload_len, false));
    }
    let (axis, indices): (usize, Vec<usize>) = match selection {
        TensorSelection::Range { axis, start, end } => (*axis, (*start..*end).collect()),
        TensorSelection::Indices { axis, indices } => (*axis, indices.clone()),
        TensorSelection::Contiguous { .. } => {
            return Err(StoreError::BoundedSelectionUnavailable {
                key: key.into(),
                message: "packed contiguous selection is not byte aligned".into(),
            });
        }
        TensorSelection::Full => unreachable!(),
    };
    let axis_len = shape[axis];
    let outer = shape[..axis].iter().product::<usize>();
    let inner = shape[axis + 1..].iter().product::<usize>();
    let output_bits = checked_elements(key, output_shape)?
        .checked_mul(bits)
        .ok_or_else(|| StoreError::Overflow {
            context: format!("selected bit length for {key:?}"),
        })?;
    if !output_bits.is_multiple_of(8) {
        return Err(StoreError::BoundedSelectionUnavailable {
            key: key.into(),
            message: "selected packed payload is not byte aligned".into(),
        });
    }
    let block_bytes = if bits == 4 {
        if !inner.is_multiple_of(2)
            || indices
                .iter()
                .any(|index| !(index * inner).is_multiple_of(2))
        {
            return Err(StoreError::BoundedSelectionUnavailable {
                key: key.into(),
                message: "FP4 selection crosses a nibble boundary".into(),
            });
        }
        inner / 2
    } else {
        inner
            .checked_mul(
                scalar_bytes.ok_or_else(|| StoreError::BoundedSelectionUnavailable {
                    key: key.into(),
                    message: "stored scalar width is not byte aligned".into(),
                })?,
            )
            .ok_or_else(|| StoreError::Overflow {
                context: format!("selection block bytes for {key:?}"),
            })?
    };
    let mut ranges = Vec::new();
    for outer_index in 0..outer {
        for index in &indices {
            let start = outer_index
                .checked_mul(axis_len)
                .and_then(|value| value.checked_add(*index))
                .and_then(|value| value.checked_mul(block_bytes))
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("selection byte start for {key:?}"),
                })?;
            let end = start
                .checked_add(block_bytes)
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("selection byte end for {key:?}"),
                })?;
            if end > payload_len {
                return Err(invalid_selection(key, "selection exceeds payload"));
            }
            push_coalesced_range(&mut ranges, start..end);
        }
    }
    if ranges.is_empty() {
        return Err(invalid_selection(
            key,
            "selection produced no physical ranges",
        ));
    }
    Ok(SafetensorsReadPlan {
        ranges,
        physically_bounded: true,
    })
}

fn describe(
    result: Result<SafetensorsReadPlan, StoreError>,
) -> Result<(Vec<Range<usize>>, bool), String> {
    result
        .map(|p| (p.ranges, p.physically_bounded))
        .map_err(|e| e.to_string())
}

#[test]
fn fixed_read_count_fill_matches_original_ranges_and_errors_for_actual_sources() {
    let cases = [
        (Dtype::U8, vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]),
        (Dtype::F4, vec![2, 4], vec![0x12, 0x34, 0x56, 0x78]),
        (Dtype::U8, vec![0, 4], vec![]),
    ];
    for (dtype, shape, payload) in cases {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("source.safetensors");
        serialize_to_file(
            [(
                "weight",
                TensorView::new(dtype, shape.clone(), &payload).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let opened = SafetensorsWeightStore::open(&path);
        if shape.contains(&0) {
            // Admission rejects empty dimensions before exposing a read source.
            // The separate direct-worker regression covers zero-stride overflow.
            assert!(matches!(opened, Err(StoreError::Io { message, .. })
                if message == "tensor \"weight\" has a zero dimension"));
            continue;
        }
        let store = opened.unwrap();
        let loan = store.source_lease_controls("weight").unwrap();
        let source = loan.safetensors_reads().unwrap();
        let selections = [
            TensorSelection::Full,
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 1,
            },
            TensorSelection::Range {
                axis: 1,
                start: 1,
                end: 3,
            },
            TensorSelection::Indices {
                axis: 1,
                indices: vec![0, 3],
            },
            TensorSelection::Indices {
                axis: 1,
                indices: vec![1, 1],
            },
            TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0],
            },
            TensorSelection::Contiguous {
                offset_elements: 1,
                shape: vec![2],
            },
        ];
        for selection in &selections {
            for policy in [ReadPolicy::RequireBounded, ReadPolicy::AllowFullTensorRead] {
                let output = crate::store::validate_selection("weight", &shape, selection);
                let expected = output.as_ref().map_err(|e| e.to_string()).and_then(|out| {
                    describe(old_plan_safetensors_reads(
                        "weight",
                        dtype,
                        &shape,
                        payload.len(),
                        selection,
                        out,
                        policy,
                    ))
                });
                let ordinary = output.as_ref().map_err(|e| e.to_string()).and_then(|out| {
                    describe(crate::store::plan_safetensors_reads(
                        "weight",
                        dtype,
                        &shape,
                        payload.len(),
                        selection,
                        out,
                        policy,
                    ))
                });
                assert_eq!(ordinary, expected);
                let mut initial = vec![99; shape.len()];
                let mut replacement = match selection {
                    TensorSelection::Contiguous { shape, .. } => vec![88; shape.len()],
                    _ => vec![],
                };
                let actual = source
                    .plan(selection, policy, &mut initial, &mut replacement)
                    .map_err(|e| {
                        assert_eq!(e.to_string(), StoreError::from(e).to_string());
                        e.to_string()
                    })
                    .map(|plan| {
                        assert_eq!(
                            plan.range_layout(),
                            Layout::array::<Range<usize>>(plan.range_count()).unwrap()
                        );
                        let mut destination = vec![97..98; plan.range_count()];
                        let filled = plan.fill_into(&mut destination).unwrap();
                        (filled.ranges().to_vec(), filled.physically_bounded())
                    });
                assert_eq!(
                    actual, expected,
                    "{dtype:?} {shape:?} {selection:?} {policy:?}"
                );
            }
        }
        assert_eq!(store.source_diagnostics().unwrap().physical_reads, 0);
    }
}

#[test]
fn fixed_ranges_preserve_duplicates_cross_outer_coalescing_and_exact_destination() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.safetensors");
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::U8, vec![2, 4], &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(&path).unwrap();
    let source = store
        .source_lease_controls("weight")
        .unwrap()
        .safetensors_reads()
        .unwrap();
    let selection = TensorSelection::Indices {
        axis: 1,
        indices: vec![0, 3],
    };
    let mut shape = [0; 2];
    let plan = source
        .plan(&selection, ReadPolicy::RequireBounded, &mut shape, &mut [])
        .unwrap();
    assert_eq!(plan.range_count(), 3);
    for len in [2, 4] {
        let mut destination = vec![70..80; len];
        assert!(plan.fill_into(&mut destination).is_err());
        assert_eq!(destination, vec![70..80; len]);
    }
    std::fs::remove_file(&path).unwrap(); // fill uses only already-admitted immutable loans
    let mut ranges = [0..0, 0..0, 0..0];
    let address = ranges.as_ptr();
    let filled = plan.fill_into(&mut ranges).unwrap();
    assert_eq!(filled.ranges(), &[0..1, 3..5, 7..8]);
    let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let selected = filled
        .ranges()
        .iter()
        .flat_map(|range| payload[range.clone()].iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(selected, [1, 4, 5, 8]);

    assert_eq!(filled.ranges().as_ptr(), address);
    assert!(filled.physically_bounded());
    let duplicates = TensorSelection::Indices {
        axis: 1,
        indices: vec![1, 1],
    };
    let mut output = [0; 2];
    let plan = source
        .plan(
            &duplicates,
            ReadPolicy::RequireBounded,
            &mut output,
            &mut [],
        )
        .unwrap();
    let mut ranges = [0..0, 0..0, 0..0, 0..0];
    assert_eq!(
        plan.fill_into(&mut ranges).unwrap().ranges(),
        &[1..2, 1..2, 5..6, 5..6]
    );
    assert_eq!(store.source_diagnostics().unwrap().physical_reads, 0);
}

#[test]
fn fixed_read_source_refuses_unprepared_header_and_follows_prepared_owner() {
    use crate::store::{
        MemoryWeightStore, PreparedCheckpointSource, PreparedTensorSource, SharedCheckpointSource,
    };
    use std::{collections::BTreeMap, sync::Arc};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("shard.safetensors");
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::U8, vec![2], &[7, 9]).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    std::fs::write(
        directory.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"weight":"shard.safetensors"}}"#,
    )
    .unwrap();
    let source: SharedCheckpointSource =
        Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    assert!(matches!(
        source.source_lease_controls("weight"),
        Err(LeaseControlBorrowError::Source(
            SourceMetadataBorrowError::HeaderUnavailable
        ))
    ));
    let metadata = source.source_metadata("weight").unwrap();
    let prepared = PreparedCheckpointSource::new(
        source.clone(),
        BTreeMap::from([(
            "weight".into(),
            PreparedTensorSource {
                metadata,
                provenance: source.source_provenance("weight").unwrap(),
            },
        )]),
    )
    .unwrap();
    let first = source.source_lease_controls("weight").unwrap();
    let second = prepared.source_lease_controls("weight").unwrap();
    assert!(first.same_entry(&second));
    assert_eq!(second.prepared_request_clones(), 1);
    assert!(second.safetensors_reads().is_some());
    let memory =
        MemoryWeightStore::from_safetensors([("weight".into(), Dtype::U8, vec![2], vec![7, 9])])
            .unwrap();
    assert!(
        memory
            .source_lease_controls("weight")
            .unwrap()
            .safetensors_reads()
            .is_none()
    );
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
}

#[test]
fn fixed_geometry_checks_legacy_slice_overflow_without_allocating_or_panicking() {
    // A leading zero permits the complete source product, but the legacy
    // planner's separate inner product can overflow. Only fixed policy changes.
    let shape = [0, 2, usize::MAX, usize::MAX];
    let source = SafetensorsReadSource::new("empty", &shape, &shape, Dtype::U8, (0, 0));
    let selection = TensorSelection::Range {
        axis: 1,
        start: 0,
        end: 1,
    };
    let mut output = [0; 4];
    let error = match source.plan(&selection, ReadPolicy::RequireBounded, &mut output, &mut []) {
        Ok(_) => panic!("inner product must refuse"),
        Err(error) => error,
    };
    assert!(matches!(
        error.cause,
        Cause::Geometry("inner stride overflow")
    ));
    assert_eq!(output, [0, 1, usize::MAX, usize::MAX]);
    let selection = TensorSelection::Full;
    let mut output = [0; 4];
    let plan = source
        .plan(&selection, ReadPolicy::RequireBounded, &mut output, &mut [])
        .unwrap();
    let mut range = [9..10];
    assert_eq!(plan.fill_into(&mut range).unwrap().ranges(), &[0..0]);
}

#[test]
fn ordinary_read_plan_keeps_original_single_and_gathered_vec_growth() {
    let shape = [2, 32];
    let selections = [
        TensorSelection::Full,
        TensorSelection::Range {
            axis: 0,
            start: 0,
            end: 1,
        },
        TensorSelection::Indices {
            axis: 1,
            indices: vec![1; 17],
        },
        TensorSelection::Indices {
            axis: 1,
            indices: vec![0, 31],
        },
    ];
    for selection in &selections {
        let output = crate::store::validate_selection("tensor", &shape, selection).unwrap();
        let old = old_plan_safetensors_reads(
            "tensor",
            Dtype::U8,
            &shape,
            64,
            selection,
            &output,
            ReadPolicy::RequireBounded,
        )
        .unwrap();
        let new = crate::store::plan_safetensors_reads(
            "tensor",
            Dtype::U8,
            &shape,
            64,
            selection,
            &output,
            ReadPolicy::RequireBounded,
        )
        .unwrap();
        assert_eq!(new.ranges, old.ranges);
        assert_eq!(new.ranges.capacity(), old.ranges.capacity());
        assert_eq!(new.physically_bounded, old.physically_bounded);
    }
}

#[test]
fn shared_read_driver_preserves_index_range_error_and_retirement_order() {
    use std::{cell::RefCell, rc::Rc};
    type Log = Rc<RefCell<Vec<&'static str>>>;
    struct IndexVec {
        values: Vec<usize>,
        log: Log,
    }
    impl Indices for IndexVec {
        fn len(&self) -> usize {
            self.values.len()
        }
        fn get(&self, i: usize) -> usize {
            self.values[i]
        }
    }
    impl Drop for IndexVec {
        fn drop(&mut self) {
            self.log.borrow_mut().push("indices retired");
        }
    }
    struct RangeVec {
        values: Vec<Range<usize>>,
        log: Log,
    }
    impl Drop for RangeVec {
        fn drop(&mut self) {
            self.log.borrow_mut().push("ranges retired");
        }
    }
    struct Trace(Log);
    impl Drop for Trace {
        fn drop(&mut self) {
            self.0.borrow_mut().push("policy retired");
        }
    }
    impl<'a> Policy<'a> for Trace {
        type Indices = IndexVec;
        type Ranges = RangeVec;
        type Output = RangeVec;
        type Error = SafetensorsReadError<'a>;
        fn indices(&self, selection: &'a TensorSelection) -> IndexVec {
            self.0.borrow_mut().push("indices created");
            IndexVec {
                values: Ordinary.indices(selection),
                log: self.0.clone(),
            }
        }
        fn product(&self, _: &'a str, values: &[usize], _: bool) -> Result<usize, Self::Error> {
            Ok(values.iter().product())
        }
        fn index_offset(
            &self,
            _: &'a str,
            index: usize,
            inner: usize,
        ) -> Result<usize, Self::Error> {
            Ok(index * inner)
        }
        fn initial(&mut self) -> RangeVec {
            self.0.borrow_mut().push("ranges created");
            RangeVec {
                values: vec![],
                log: self.0.clone(),
            }
        }
        fn last(ranges: &mut RangeVec) -> Option<&mut Range<usize>> {
            ranges.values.last_mut()
        }
        fn push(
            &self,
            _: &'a str,
            ranges: &mut RangeVec,
            value: Range<usize>,
        ) -> Result<(), Self::Error> {
            ranges.values.push(value);
            Ok(())
        }
        fn is_empty(ranges: &RangeVec) -> bool {
            ranges.values.is_empty()
        }
        fn single(self, range: Range<usize>, _: bool) -> RangeVec {
            RangeVec {
                values: vec![range],
                log: self.0.clone(),
            }
        }
        fn finish(self, ranges: RangeVec, _: bool) -> RangeVec {
            ranges
        }
        fn error(&self, key: &'a str, cause: Cause<'a>) -> Self::Error {
            self.0.borrow_mut().push("error created");
            SafetensorsReadError { key, cause }
        }
    }
    let log = Log::default();
    let selection = TensorSelection::Range {
        axis: 1,
        start: 0,
        end: 2,
    };
    assert!(
        run(
            "tensor",
            Dtype::F4.bitsize(),
            &[2, 3],
            3,
            &selection,
            &[2, 2],
            ReadPolicy::RequireBounded,
            Trace(log.clone())
        )
        .is_err()
    );
    assert_eq!(
        &*log.borrow(),
        &[
            "indices created",
            "error created",
            "indices retired",
            "policy retired"
        ]
    );
    log.borrow_mut().clear();
    assert!(
        run(
            "tensor",
            Dtype::U8.bitsize(),
            &[0, 4],
            0,
            &selection,
            &[0, 2],
            ReadPolicy::RequireBounded,
            Trace(log.clone())
        )
        .is_err()
    );
    assert_eq!(
        &*log.borrow(),
        &[
            "indices created",
            "ranges created",
            "error created",
            "ranges retired",
            "indices retired",
            "policy retired"
        ]
    );
    log.borrow_mut().clear();
    let result = run(
        "tensor",
        Dtype::U8.bitsize(),
        &[2, 4],
        8,
        &selection,
        &[2, 2],
        ReadPolicy::RequireBounded,
        Trace(log.clone()),
    )
    .unwrap();
    assert_eq!(result.values, [0..2, 4..6]);
    assert_eq!(
        &*log.borrow(),
        &[
            "indices created",
            "ranges created",
            "policy retired",
            "indices retired"
        ]
    );
    drop(result);
    assert_eq!(log.borrow().last(), Some(&"ranges retired"));
}

#[test]
fn encoded_projection_counts_exact_ranges_and_refuses_wrong_destinations() {
    let cases = [
        (8, vec![2, 4], 8, TensorSelection::Full, vec![0..8]),
        (
            8,
            vec![2, 4],
            8,
            TensorSelection::Range {
                axis: 1,
                start: 1,
                end: 3,
            },
            vec![1..3, 5..7],
        ),
        (
            8,
            vec![2, 4],
            8,
            TensorSelection::Indices {
                axis: 1,
                indices: vec![3, 0, 0, 1],
            },
            vec![3..4, 0..1, 0..2, 7..8, 4..5, 4..6],
        ),
        (
            8,
            vec![2, 4],
            8,
            TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0, 1],
            },
            vec![4..8, 0..8],
        ),
        (
            4,
            vec![2, 4],
            4,
            TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0, 1],
            },
            vec![2..4, 0..4],
        ),
        (
            32,
            vec![2, 3],
            24,
            TensorSelection::Indices {
                axis: 1,
                indices: vec![2, 0],
            },
            vec![8..12, 0..4, 20..24, 12..16],
        ),
        (
            8,
            vec![2, 4],
            8,
            TensorSelection::Contiguous {
                offset_elements: 2,
                shape: vec![3],
            },
            vec![2..5],
        ),
        (8, vec![0, 4], 0, TensorSelection::Full, vec![0..0]),
        (
            8,
            vec![1_000_000_000, 4],
            4_000_000_000,
            TensorSelection::Range {
                axis: 0,
                start: 500_000_000,
                end: 500_000_001,
            },
            vec![2_000_000_000..2_000_000_004],
        ),
    ];
    for (bits, shape, payload_len, selection, expected) in cases {
        let output = crate::store::validate_selection("tensor", &shape, &selection).unwrap();
        let plan = encoded_selection_plan("tensor", bits, &shape, payload_len, &selection, &output)
            .unwrap();
        assert_eq!(plan.range_count(), expected.len());
        assert_eq!(
            plan.range_layout(),
            Layout::array::<Range<usize>>(expected.len()).unwrap()
        );
        for length in [expected.len() - 1, expected.len() + 1] {
            let mut destination = vec![91..97; length];
            let error = match plan.fill_into(&mut destination) {
                Ok(_) => panic!("wrong destination accepted"),
                Err(error) => error,
            };
            assert!(
                matches!(error.cause, Cause::Destination { expected: count, actual } if count == expected.len() && actual == length)
            );
            assert_eq!(destination, vec![91..97; length]);
        }
        let mut destination = vec![91..97; expected.len()];
        let result = plan.fill_into(&mut destination).unwrap();
        assert_eq!(result.ranges(), expected);
        assert!(result.physically_bounded());
        // The same bound inputs can fill a fresh destination without re-inference.
        let mut second = vec![0..0; expected.len()];
        assert_eq!(plan.fill_into(&mut second).unwrap().ranges(), expected);
    }
}

#[test]
fn encoded_projection_range_plan_retains_typed_alignment_and_overflow_errors() {
    for (shape, selection, output, expected) in [
        (
            vec![2, 4],
            TensorSelection::Indices {
                axis: 1,
                indices: vec![1, 3],
            },
            vec![2, 2],
            "FP4 selection crosses a nibble boundary",
        ),
        (
            vec![2, 4],
            TensorSelection::Contiguous {
                offset_elements: 1,
                shape: vec![2],
            },
            vec![2],
            "packed contiguous selection is not byte aligned",
        ),
    ] {
        let error = match encoded_selection_plan("packed", 4, &shape, 4, &selection, &output) {
            Ok(_) => panic!("unaligned selection accepted"),
            Err(error) => error,
        };
        assert!(matches!(error.cause, Cause::Bounded(message) if message == expected));
        assert!(
            matches!(StoreError::from(error), StoreError::BoundedSelectionUnavailable { key, message } if key == "packed" && message == expected)
        );
    }
    let shape = [0, 2, usize::MAX, usize::MAX];
    let selection = TensorSelection::Range {
        axis: 1,
        start: 0,
        end: 1,
    };
    let output = [0, 1, usize::MAX, usize::MAX];
    let error = match encoded_selection_plan("empty", 8, &shape, 0, &selection, &output) {
        Ok(_) => panic!("inner stride overflow accepted"),
        Err(error) => error,
    };
    assert!(matches!(
        error.cause,
        Cause::Geometry("inner stride overflow")
    ));
}
