use super::*;
use crate::{schema::*, validation::resolve_safetensors_plan};
use std::sync::atomic::{AtomicBool, Ordering};

fn old_selection(
    key: &str,
    dtype: Dtype,
    shape: &[usize],
    data: &[u8],
    selection: &TensorSelection,
    output_shape: &[usize],
    policy: ReadPolicy,
) -> Result<(Range<usize>, Option<Vec<u8>>), StoreError> {
    if matches!(selection, TensorSelection::Full) {
        return Ok((0..data.len(), None));
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
        return data
            .get(start..end)
            .map(|_| (start..end, None))
            .ok_or_else(|| invalid_selection(key, "contiguous byte span outside payload"));
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
        let row_bytes = data
            .len()
            .checked_div(shape[0])
            .filter(|_| data.len().is_multiple_of(shape[0]))
            .ok_or_else(|| invalid_selection(key, "payload is not row divisible"))?;
        let start = start * row_bytes;
        let end = end * row_bytes;
        return Ok((start..end, None));
    }
    if matches!(policy, ReadPolicy::AllowFullTensorRead) {
        return Ok((0..data.len(), None));
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
    let mut output = Vec::with_capacity(output_bits / 8);
    if bits == 4 {
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
        let block_bytes = inner / 2;
        for outer_index in 0..outer {
            for index in &indices {
                let start = (outer_index * axis_len + index) * block_bytes;
                output.extend_from_slice(
                    data.get(start..start + block_bytes)
                        .ok_or_else(|| invalid_selection(key, "selection exceeds payload"))?,
                );
            }
        }
    } else {
        let scalar_bytes = scalar_bytes.ok_or_else(|| StoreError::BoundedSelectionUnavailable {
            key: key.into(),
            message: "stored scalar width is not byte aligned".into(),
        })?;
        let block_bytes = inner * scalar_bytes;
        for outer_index in 0..outer {
            for index in &indices {
                let start = (outer_index * axis_len + index) * block_bytes;
                output.extend_from_slice(
                    data.get(start..start + block_bytes)
                        .ok_or_else(|| invalid_selection(key, "selection exceeds payload"))?,
                );
            }
        }
    }
    Ok((0..output.len(), Some(output)))
}

impl MemoryWeightStore {
    fn old_acquire(&self, request: TensorReadRequest) -> Result<MemoryLease, StoreError> {
        let tensor =
            self.tensors
                .get(&request.key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor {
                    key: request.key.clone(),
                })?;
        let output_shape = validate_selection(
            &request.key,
            &tensor.metadata.logical_shape,
            &request.selection,
        )?;
        let (span, selected_bytes) = old_selection(
            &request.key,
            tensor.dtype,
            &tensor.metadata.logical_shape,
            &tensor.bytes,
            &request.selection,
            &output_shape,
            request.policy,
        )?;
        let length = selected_bytes
            .as_ref()
            .map_or(span.len(), |bytes| bytes.len());
        let full_selection = matches!(request.selection, TensorSelection::Full);
        Ok(MemoryLease {
            tensor,
            selection: request.selection,
            output_shape,
            proof: BoundedReadProof {
                physically_bounded: matches!(request.policy, ReadPolicy::RequireBounded)
                    || full_selection,
                offset_bytes: u64::try_from(span.start).map_err(|_| StoreError::Overflow {
                    context: "in-memory selection byte offset".into(),
                })?,
                length_bytes: u64::try_from(length).map_err(|_| StoreError::Overflow {
                    context: "in-memory selection byte length".into(),
                })?,
                physical_reads: 0,
                physical_read_bytes: 0,
            },
            span,
            selected_bytes: selected_bytes.map(Arc::from),
        })
    }
}

fn memory(dtype: Dtype, shape: Vec<usize>, bytes: Vec<u8>) -> Arc<MemoryWeightStore> {
    Arc::new(MemoryWeightStore::from_safetensors([("weight".into(), dtype, shape, bytes)]).unwrap())
}
fn request(selection: TensorSelection, policy: ReadPolicy) -> TensorReadRequest {
    TensorReadRequest {
        key: "weight".into(),
        selection,
        policy,
    }
}
fn proof(lease: &impl EncodedTensorLease) -> (bool, u64, u64, u64, u64) {
    let p = lease.bounded_read_proof();
    (
        p.physically_bounded,
        p.offset_bytes,
        p.length_bytes,
        p.physical_reads,
        p.physical_read_bytes,
    )
}
fn selections() -> Vec<TensorSelection> {
    vec![
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
            axis: 0,
            indices: vec![1, 0, 1],
        },
        TensorSelection::Indices {
            axis: 1,
            indices: vec![0, 2, 3, 1],
        },
        TensorSelection::Indices {
            axis: 1,
            indices: vec![0, 1, 1, 2],
        },
        TensorSelection::Contiguous {
            offset_elements: 2,
            shape: vec![2, 2],
        },
        TensorSelection::Range {
            axis: 4,
            start: 0,
            end: 1,
        },
        TensorSelection::Range {
            axis: 0,
            start: 0,
            end: 3,
        },
        TensorSelection::Indices {
            axis: 0,
            indices: vec![],
        },
        TensorSelection::Contiguous {
            offset_elements: 7,
            shape: vec![2],
        },
    ]
}

#[test]
fn actual_memory_matrix_matches_untouched_acquire_and_preserves_borrow_or_final_buffer() {
    for (dtype, bytes) in [
        (Dtype::U8, vec![1, 2, 3, 4, 5, 6, 7, 8]),
        (Dtype::F4, vec![0x12, 0x34, 0x56, 0x78]),
    ] {
        let source = memory(dtype, vec![2, 4], bytes);
        for policy in [ReadPolicy::RequireBounded, ReadPolicy::AllowFullTensorRead] {
            for selection in selections() {
                let read = request(selection, policy);
                let old = source.old_acquire(read.clone());
                let ordinary = source.acquire(read.clone());
                let prepared = Prepared::prepare(&source, &read);
                match old {
                    Ok(old) => {
                        let ordinary = ordinary.unwrap();
                        let prepared = prepared.unwrap();
                        assert!(prepared.matches(&source, &read));
                        assert!(prepared.staging.is_empty());
                        let selected_ptr = prepared.selected_bytes.as_ref().map(|bytes| {
                            assert!(
                                bytes.iter().all(|b| *b == 0),
                                "preparation must not copy selected payload"
                            );
                            bytes.as_ptr()
                        });
                        let source_ptr = prepared
                            .tensor
                            .bytes
                            .get(prepared.span.clone())
                            .map(|bytes| bytes.as_ptr());
                        let lease = prepared.acquire(&read.key).unwrap();
                        assert_eq!(
                            lease.encoded_bytes(),
                            old.encoded_bytes(),
                            "{dtype:?} {read:?}"
                        );
                        assert_eq!(ordinary.encoded_bytes(), old.encoded_bytes());
                        assert_eq!(lease.metadata(), old.metadata());
                        assert_eq!(lease.selection(), old.selection());
                        assert_eq!(lease.output_shape(), old.output_shape());
                        assert_eq!(proof(&lease), proof(&old));
                        assert_eq!(proof(&ordinary), proof(&old));
                        assert_eq!(lease.backing_path(), None);
                        let CheckpointLease::Memory(inner) = &lease else {
                            panic!("real Memory lease")
                        };
                        assert!(inner.tensor.same(&source.tensors["weight"]));
                        assert_eq!(inner.selected_bytes.is_some(), old.selected_bytes.is_some());
                        assert_eq!(
                            lease.encoded_bytes().unwrap().as_ptr(),
                            selected_ptr.or(source_ptr).unwrap()
                        );
                        if policy == ReadPolicy::AllowFullTensorRead
                            && !matches!(read.selection, TensorSelection::Full)
                        {
                            assert!(
                                !lease.bounded_read_proof().physically_bounded,
                                "ordinary declaration retained even for borrowed leading span"
                            );
                        }
                    }
                    Err(old) => {
                        assert_eq!(ordinary.unwrap_err().to_string(), old.to_string());
                        assert_eq!(
                            prepared.unwrap_err().store_error().unwrap().to_string(),
                            old.to_string()
                        );
                    }
                }
            }
        }
    }
    for (shape, bytes) in [(vec![], vec![29]), (vec![0, 4], vec![])] {
        let source = memory(Dtype::U8, shape, bytes);
        let read = request(TensorSelection::Full, ReadPolicy::RequireBounded);
        let old = source.old_acquire(read.clone()).unwrap();
        let lease = Prepared::prepare(&source, &read)
            .unwrap()
            .acquire(&read.key)
            .unwrap();
        assert_eq!(lease.encoded_bytes(), old.encoded_bytes());
        assert_eq!(lease.output_shape(), old.output_shape());
        assert_eq!(proof(&lease), proof(&old));
    }
}

#[test]
fn shared_ordinary_selection_preserves_old_vec_capacity_errors_and_read_policy_order() {
    for (dtype, bytes) in [
        (Dtype::U8, vec![1, 2, 3, 4, 5, 6, 7, 8]),
        (Dtype::F4, vec![0x12, 0x34, 0x56, 0x78]),
    ] {
        for policy in [ReadPolicy::RequireBounded, ReadPolicy::AllowFullTensorRead] {
            for selection in selections() {
                let Ok(output) = validate_selection("weight", &[2, 4], &selection) else {
                    continue;
                };
                let old = old_selection(
                    "weight",
                    dtype,
                    &[2, 4],
                    &bytes,
                    &selection,
                    &output,
                    policy,
                );
                let new = ordinary_selection(
                    "weight",
                    dtype,
                    &[2, 4],
                    &bytes,
                    &selection,
                    &output,
                    policy,
                );
                match old {
                    Ok((span, bytes)) => {
                        let (newspan, newbytes) = new.unwrap();
                        assert_eq!(span, newspan);
                        assert_eq!(bytes, newbytes);
                        assert_eq!(
                            bytes.as_ref().map(Vec::capacity),
                            newbytes.as_ref().map(Vec::capacity)
                        );
                    }
                    Err(error) => assert_eq!(new.unwrap_err().to_string(), error.to_string()),
                }
            }
        }
    }
}

struct Traced {
    events: Arc<Mutex<Vec<String>>>,
}
struct TracedBuffer {
    bytes: Vec<u8>,
    events: Arc<Mutex<Vec<String>>>,
}
impl Drop for TracedBuffer {
    fn drop(&mut self) {
        self.events
            .lock()
            .unwrap()
            .push(format!("drop {}", self.bytes.len()));
    }
}
impl<'a> Storage<'a> for Traced {
    type Buffer = TracedBuffer;
    fn range(&mut self, range: Range<usize>) -> Indices<'a> {
        self.events.lock().unwrap().push("range".into());
        Indices::Owned(range.collect())
    }
    fn indices(&mut self, indices: &'a Vec<usize>) -> Indices<'a> {
        self.events.lock().unwrap().push("indices".into());
        Indices::Owned(indices.clone())
    }
    fn output(&mut self, capacity: usize) -> Result<Self::Buffer, StoreError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("allocate {capacity}"));
        Ok(TracedBuffer {
            bytes: Vec::with_capacity(capacity),
            events: self.events.clone(),
        })
    }
    fn append(output: &mut Self::Buffer, bytes: &[u8]) -> Result<(), StoreError> {
        output
            .events
            .lock()
            .unwrap()
            .push(format!("copy {}", bytes.len()));
        output.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn len(output: &Self::Buffer) -> usize {
        output.bytes.len()
    }
}
#[test]
fn worker_preserves_output_before_nibble_check_and_drops_written_prefix_on_later_error() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let selected = TensorSelection::Indices {
        axis: 1,
        indices: vec![0, 1],
    };
    let error = select(
        "weight",
        Dtype::F4,
        &[2, 2],
        &[0x12, 0x34],
        &selected,
        &[2, 2],
        ReadPolicy::RequireBounded,
        Traced {
            events: events.clone(),
        },
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("nibble boundary"));
    assert_eq!(*events.lock().unwrap(), ["indices", "allocate 2", "drop 0"]);
    events.lock().unwrap().clear();
    let selected = TensorSelection::Indices {
        axis: 0,
        indices: vec![0, 1],
    };
    let error = select(
        "weight",
        Dtype::U8,
        &[2, 2],
        &[1, 2, 3],
        &selected,
        &[2, 2],
        ReadPolicy::RequireBounded,
        Traced {
            events: events.clone(),
        },
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("selection exceeds payload"));
    assert_eq!(
        *events.lock().unwrap(),
        ["indices", "allocate 4", "copy 2", "drop 2"]
    );
    events.lock().unwrap().clear();
    let selected = TensorSelection::Indices {
        axis: 0,
        indices: vec![0],
    };
    let error = select(
        "weight",
        Dtype::F4,
        &[2, 1],
        &[0x12],
        &selected,
        &[1, 1],
        ReadPolicy::RequireBounded,
        Traced {
            events: events.clone(),
        },
    )
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("selected packed payload is not byte aligned")
    );
    assert_eq!(
        *events.lock().unwrap(),
        ["indices"],
        "encoded-length rejection precedes output allocation"
    );
}

#[test]
fn fixed_gather_rejects_short_and_long_destinations_without_writing_and_reuses_exact_slice() {
    let selected = TensorSelection::Indices {
        axis: 0,
        indices: vec![1, 0, 1],
    };
    let data = [1, 2, 3, 4, 5, 6];
    for size in [8, 10] {
        let mut output = vec![0xab; size];
        assert!(
            select(
                "weight",
                Dtype::U8,
                &[2, 3],
                &data,
                &selected,
                &[3, 3],
                ReadPolicy::RequireBounded,
                Fixed {
                    output: Some(&mut output)
                }
            )
            .is_err()
        );
        assert_eq!(output, vec![0xab; size]);
    }
    let mut output = [0xab; 9];
    let pointer = output.as_ptr();
    let (span, filled) = select(
        "weight",
        Dtype::U8,
        &[2, 3],
        &data,
        &selected,
        &[3, 3],
        ReadPolicy::RequireBounded,
        Fixed {
            output: Some(&mut output),
        },
    )
    .unwrap();
    assert_eq!(span, 0..9);
    assert_eq!(filled.as_ref().unwrap().output.as_ptr(), pointer);
    drop(filled);
    assert_eq!(output, [4, 5, 6, 1, 2, 3, 4, 5, 6]);
}

fn snapshot(source: SharedCheckpointSource) -> PreparedCheckpointSource {
    let catalog = source
        .source_keys()
        .into_iter()
        .map(|key| {
            let entry = PreparedTensorSource {
                metadata: source.source_metadata(&key).unwrap(),
                provenance: source.source_provenance(&key).unwrap(),
            };
            (key, entry)
        })
        .collect();
    PreparedCheckpointSource::new(source, catalog).unwrap()
}
#[test]
fn closed_memory_route_preserves_wrapper_authorization_actual_tensor_and_final_owner() {
    let source = memory(Dtype::U8, vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let tensor = source.tensors["weight"].ordinary_weak();
    let restricted: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            source.clone(),
            "selected",
            BTreeSet::from(["weight".into()]),
        )
        .unwrap(),
    );
    let composed: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new([restricted.clone()]).unwrap());
    let prepared: SharedCheckpointSource = Arc::new(snapshot(composed));
    let contract = SafetensorsCheckpointPlan::new(
        "memory",
        vec![SafetensorsTensorConstraint::required(
            "weight",
            vec![2, 4],
            StoredDtypeConstraint::Exact(StoredDtype::U8),
        )],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .unwrap();
    let contract = resolve_safetensors_plan(prepared.as_ref(), &contract).unwrap();
    let root: SharedCheckpointSource = Arc::new(ResolvedCheckpointSource::new(prepared, contract));
    let weak = Arc::downgrade(&root);
    let read = request(
        TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        },
        ReadPolicy::RequireBounded,
    );
    let ordinary = root.acquire_lease(read.clone()).unwrap();
    let destination = PreparedCheckpointAcquisition::prepare(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    assert!(destination.matches_request(&root, &read));
    for denied in [root.clone(), restricted.clone()] {
        let mut bad = read.clone();
        bad.key = "absent".into();
        let error =
            PreparedCheckpointAcquisition::prepare(denied.clone(), bad.clone()).unwrap_err();
        assert_eq!(
            error.to_string(),
            denied.acquire_lease(bad).unwrap_err().to_string()
        );
    }
    drop((source, root, restricted));
    assert!(weak.upgrade().is_some());
    let lease = destination.acquire().unwrap();
    assert!(weak.upgrade().is_none());
    assert_eq!(lease.encoded_bytes(), ordinary.encoded_bytes());
    assert_eq!(proof(&lease), proof(&ordinary));
    drop(ordinary);
    assert!(tensor.upgrade().is_some());
    drop(lease);
    assert!(tensor.upgrade().is_none());
}

struct Switching {
    first: Arc<MemoryWeightStore>,
    second: Arc<MemoryWeightStore>,
    switched: AtomicBool,
}
impl CheckpointSource for Switching {
    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        if self.switched.load(Ordering::SeqCst) {
            self.second.prepared_acquisition_source()
        } else {
            self.first.prepared_acquisition_source()
        }
    }
    fn source_keys(&self) -> Vec<String> {
        self.first.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.first.source_metadata(key)
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("no ordinary fallback")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        panic!("no diagnostics fallback")
    }
}
#[test]
fn equal_bytes_foreign_tensor_refusal_keeps_destination_source_until_error_retirement() {
    let first = memory(Dtype::U8, vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let second = memory(Dtype::U8, vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let tensor = first.tensors["weight"].ordinary_weak();
    let read = request(
        TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0],
        },
        ReadPolicy::RequireBounded,
    );
    let direct = Prepared::prepare(&first, &read).unwrap();
    assert!(!direct.matches(&second, &read));
    drop(direct);
    let owner = Arc::new(Switching {
        first,
        second,
        switched: AtomicBool::new(false),
    });
    let weak = Arc::downgrade(&owner);
    let root: SharedCheckpointSource = owner.clone();
    let prepared = PreparedCheckpointAcquisition::prepare(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    owner.switched.store(true, Ordering::SeqCst);
    let error = prepared.acquire().unwrap_err();
    assert_eq!(
        error.to_string(),
        "prepared checkpoint acquisition source changed"
    );
    assert!(error.matches_request(&root, &read));
    assert!(!error.retains_rejected_lease());
    drop((root, owner));
    assert!(weak.upgrade().is_some());
    assert!(tensor.upgrade().is_some());
    drop(error);
    assert!(weak.upgrade().is_none());
    assert!(tensor.upgrade().is_none());
}

#[test]
fn invalid_packed_preparation_and_failed_copy_retain_actual_tensor_and_written_prefix() {
    let source = memory(Dtype::F4, vec![2, 4], vec![0x12, 0x34, 0x56, 0x78]);
    let tensor = source.tensors["weight"].ordinary_weak();
    let read = request(
        TensorSelection::Indices {
            axis: 1,
            indices: vec![0, 1],
        },
        ReadPolicy::RequireBounded,
    );
    let failure = Prepared::prepare(&source, &read).unwrap_err();
    assert!(failure.to_string().contains("nibble boundary"));
    assert!(failure.pending.as_ref().unwrap().selected_bytes.is_none());
    drop(source);
    assert!(tensor.upgrade().is_some());
    drop(failure);
    assert!(tensor.upgrade().is_none());
    // The private fixed worker reports a real failing range after retaining its
    // previously copied prefix; actual immutable admitted tensors cannot shrink.
    let selected = TensorSelection::Indices {
        axis: 0,
        indices: vec![0, 1],
    };
    let mut output = [0xaa; 4];
    let error = select(
        "weight",
        Dtype::U8,
        &[2, 2],
        &[7, 8, 9],
        &selected,
        &[2, 2],
        ReadPolicy::RequireBounded,
        Fixed {
            output: Some(&mut output),
        },
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("selection exceeds payload"));
    assert_eq!(output, [7, 8, 0xaa, 0xaa]);
}
