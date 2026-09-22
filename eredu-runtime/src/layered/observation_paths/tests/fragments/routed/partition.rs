//! The actual funded observer lends and returns a retained sparse program.
use super::*;
use crate::capture::partition::*;
use eredu_core::component::{ComponentCoordinateMap, RoutedComponentCoordinateMap};
use eredu_core::consensus::{BoundedConsensusTransport, ConsensusTransport};
use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use std::{
    cell::RefCell,
    convert::Infallible,
    sync::atomic::{AtomicUsize, Ordering},
};
#[derive(Debug)]
struct Metadata(Arc<AtomicUsize>);
impl HostMetadataAccount for Metadata {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.0.fetch_add(bytes, Ordering::SeqCst);
        Ok(())
    }
}
struct Done;
impl Completion for Done {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Infallible> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Infallible> {
        Ok(())
    }
}
impl BoundedCompletion for Done {
    fn wait_bounded(
        self,
        _: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Infallible> {
        Ok(BoundedCompletionOutcome::Completed)
    }
}
struct LocalTransport {
    metadata: HostMetadataFunding,
    calls: RefCell<Vec<PartitionCaptureFrameKind>>,
}
impl ConsensusTransport for LocalTransport {
    type Error = Infallible;
    fn participant_count(&self) -> usize {
        1
    }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, Infallible> {
        panic!("untyped transport")
    }
}
impl BoundedConsensusTransport for LocalTransport {
    type Completion = Done;
    type GatherOutput = ();
    fn submit_all_gather_words(&self, _: &[u32]) -> Result<Submission<(), Done>, Infallible> {
        panic!("untyped transport")
    }
    fn resolve_all_gather_words(&self, _: ()) -> Result<Vec<u32>, Infallible> {
        panic!("untyped transport")
    }
}
impl PartitionCaptureTransport for LocalTransport {
    fn capture_rank(&self) -> usize {
        0
    }
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError> {
        Ok(BoundedCompletionWait::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap())
    }
    fn ensure_capture_active(&self) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn estimate_capture_gather(&self, _: usize) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage::default())
    }
    fn fail_capture_exchange(&self, _: &PartitionCaptureExchangeError) {}
    fn capture_word_destination(
        &self,
        n: usize,
    ) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        Ok(PartitionCaptureBuffer::funded(n, &self.metadata)?)
    }
    fn capture_byte_destination(
        &self,
        n: usize,
    ) -> Result<PartitionCaptureBuffer<u8>, PartitionCaptureExchangeError> {
        Ok(PartitionCaptureBuffer::funded(n, &self.metadata)?)
    }
    fn gather_capture_frame(
        &self,
        frame: &PartitionCaptureFrame<'_>,
        _: BoundedCompletionWait,
    ) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        self.calls.borrow_mut().push(frame.kind());
        let mut out = PartitionCaptureBuffer::funded(frame.gathered_words(), &self.metadata)?;
        out.extend_from_slice(frame.words())?;
        Ok(out)
    }
}
fn native_usage() -> CaptureUsage {
    CaptureUsage {
        captures: 1,
        retained_bytes: 65536,
        host_bytes: 65536,
        encoded_bytes: 65536,
    }
}
struct PartitionBackend<'a> {
    inner: Backend,
    program: PreparedPartitionCaptureProgram<'a, LocalTransport>,
    fail: bool,
    completed: usize,
}
impl ScheduledCaptureBackend for PartitionBackend<'_> {
    type Tensor = Value;
    type Error = Native;
    fn partition_capture(&mut self) -> Option<&mut (dyn ScheduledPartitionCapture + '_)> {
        Some(&mut self.program)
    }
    fn validate_source(
        &self,
        t: &Value,
        g: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Native> {
        self.inner.validate_source(t, g)
    }
    fn estimate(
        &self,
        t: &Value,
        g: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        self.inner.estimate(t, g)
    }
    fn transform(
        &mut self,
        t: &Value,
        c: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Native> {
        self.inner.transform(t, c)
    }
    fn complete_partition_source(&mut self, t: &Value) -> Result<(), FundedCaptureError<Native>> {
        self.completed += 1;
        self.inner.roots.push(t.clone());
        Ok(())
    }
    fn validate_partition_routed_invocation(
        &self,
        i: &crate::RoutedUnitInvocation<'_, Value>,
        g: &PartitionRoutedUnitCaptureLayout<'_>,
    ) -> Result<(u64, TensorDtype), FundedCaptureError<Native>> {
        if i.input.shape.len() != 2 || i.input.shape[0] < 0 || i.input.shape[1] != 5 {
            return Err(FundedCaptureError::Backend(Native::Shape));
        }
        let rows = i.input.shape[0] as u64;
        g.validate_input_invocation(
            rows,
            i.unit_coordinates,
            i.origins.map(|origins| origins.capture_coordinates()),
        )?;
        Ok((rows, TensorDtype::F32))
    }
    fn estimate_partition_routed(
        &self,
        _: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(native_usage())
    }
    fn validate_partition_routed_batch_source(
        &self,
        s: &PartitionRoutedUnitCaptureSource<'_, Value>,
        g: &PartitionRoutedUnitCaptureLayout<'_>,
        rows: u64,
    ) -> Result<(TensorDtype, u64), FundedCaptureError<Native>> {
        g.validate_invocation(rows, s.unit_coordinates, s.origins)?;
        if s.source.values.shape != [3, 5]
            || s.source.token_indices.shape != [3]
            || s.source.selection_indices.shape != [3]
            || s.source.coefficients.shape != [1, 3]
            || s.source.source_groups.shape != [rows as i32, 3]
        {
            return Err(FundedCaptureError::Backend(Native::Shape));
        }
        Ok((TensorDtype::F32, 1))
    }
    fn validate_partition_routed_source(
        &self,
        s: &PartitionRoutedUnitCaptureSource<'_, Value>,
        r: &PartitionRoutedUnitCaptureRequest<'_>,
        tokens: u64,
        rows: u64,
    ) -> Result<TensorDtype, FundedCaptureError<Native>> {
        let mut layout = r.layout();
        layout.source_tokens = tokens;
        self.validate_partition_routed_batch_source(s, &layout, rows)
            .map(|(dtype, _)| dtype)
    }
    fn transform_partition_routed_batch(
        &mut self,
        s: &PartitionRoutedUnitCaptureSource<'_, Value>,
        mut w: CapturePartitionRoutedWriter<'_, '_>,
    ) -> Result<(), FundedCaptureError<Native>> {
        w.validate_native_scope(&self.inner.scope)
            .map_err(CaptureRunHostError::from)?;
        self.inner.transforms += 1;
        let physical = s.source.token_offset;
        let logical = w
            .logical_token(physical)
            .ok_or(FundedCaptureError::Backend(Native::Shape))?;
        let r = w.request();
        let bounds: [[u64; 3]; 3] = [
            r.slice.starts.as_slice().try_into().unwrap(),
            r.slice.ends.as_slice().try_into().unwrap(),
            r.slice.strides.as_slice().try_into().unwrap(),
        ];
        let selected = |x: u64, axis: usize| {
            x >= bounds[0][axis]
                && x < bounds[1][axis]
                && (x - bounds[0][axis]) % bounds[2][axis] == 0
        };
        if selected(logical, 0) {
            for slot in [0_u64, 2] {
                if !selected(slot, 1) {
                    continue;
                }
                let expert =
                    s.source.source_groups.values[physical as usize * 3 + slot as usize] as u64;
                w.begin_row(
                    None,
                    logical,
                    slot,
                    expert,
                    s.source.coefficients.values[slot as usize],
                )
                .map_err(CaptureRunHostError::from)?;
                for unit in [1, 3] {
                    w.push_f32(s.source.values.values[slot as usize * 5 + unit])
                        .map_err(CaptureRunHostError::from)?;
                }
                if self.fail {
                    return Err(FundedCaptureError::Backend(Native::Shape));
                }
                w.finish_row().map_err(CaptureRunHostError::from)?;
            }
        }
        w.source_chunk(physical, physical + 1)
            .map_err(CaptureRunHostError::from)?;
        w.finish().map_err(CaptureRunHostError::from)?;
        Ok(())
    }
}
#[test]
fn original_partition_routed_observer_returns_sparse_owners_after_uneven_chunks_and_retains_failed_batch()
 {
    for fail in [false, true] {
        let source = sparse_source();
        let metadata_used = Arc::new(AtomicUsize::new(0));
        let metadata = HostMetadataFunding::new(Metadata(metadata_used.clone())).unwrap();
        let transport = LocalTransport {
            metadata: metadata.clone(),
            calls: RefCell::new(vec![]),
        };
        let global = CaptureRoutedUnitsGeometry::prepare(
            source.admission(),
            0,
            CapturePhase::Prefill,
            0,
            None,
        )
        .unwrap();
        let ownership = RoutedUnitCaptureOwnership {
            coordinates: RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::range(7, 0..7).unwrap(),
                ComponentCoordinateMap::range(5, 0..5).unwrap(),
            ),
            source_peer: None,
            source_peers: 1,
        };
        let producers = [PartitionCaptureRoutedProducerSource {
            rank: 0,
            ownership: &ownership,
        }];
        let context = PartitionCaptureContext {
            artifact_identity: "fixture".into(),
            execution_identity: "local-routed".into(),
            run_identity: "run".into(),
            overlay_identity: None,
            capture_plan_identity: source.admission().identity().into(),
            selection_index: 0,
            phase: CapturePhase::Prefill,
            prediction: 0,
            forward_epoch: epoch(0).value(),
            invocation: None,
            invocation_window: None,
        };
        let limits = PartitionCaptureReceiptLimits {
            max_producers: 1,
            max_fragments: 8,
            max_record_bytes: 64 << 10,
        };
        let mut quota = CaptureLedger::new(source.admission());
        quota.begin_step();
        let prototype = PartitionCaptureReceiptPlan::new_routed_shared_funded(
            &source, &context, &producers, 1, limits, &metadata, &mut quota,
        )
        .unwrap();
        let slices: Vec<_> = (0..prototype.producer(0).unwrap().fragments().len())
            .map(|i| {
                let a =
                    CaptureRoutedUnitsGeometry::fragment_axes(prototype.producer(0).unwrap(), i)
                        .unwrap();
                ResolvedCaptureSlice {
                    starts: a[0].to_vec(),
                    ends: a[1].to_vec(),
                    strides: a[2].to_vec(),
                    shape: a[3].to_vec(),
                }
            })
            .collect();
        let native: Vec<_> = slices
            .iter()
            .enumerate()
            .map(|(fragment, slice)| PartitionCaptureRoutedFragmentGeometry {
                producer: 0,
                fragment,
                request: PartitionRoutedUnitCaptureRequest {
                    geometry: global.bank(),
                    source_tokens: 3,
                    ownership: &ownership,
                    slice,
                },
                estimate: PartitionCaptureNativeEstimate {
                    capture: native_usage(),
                    generated_creation_bytes: 0,
                },
            })
            .collect();
        let retained = PreparedPartitionRoutedSource::new_local_prefill(
            &source,
            0,
            &producers,
            &native,
            Some(PartitionCaptureRoutedLocalSource {
                producer: 0,
                layout: PartitionRoutedUnitCaptureLayout {
                    geometry: global.bank(),
                    source_tokens: 3,
                    ownership: &ownership,
                },
                dtype: Some(TensorDtype::F32),
            }),
            geometry(),
            &metadata,
        )
        .unwrap();
        let h = CaptureRunHostPlan::prepare(&source)
            .unwrap()
            .initialization_peak_bytes()
            + PartitionFragmentHostPlan::prepare_prefill(&prototype, geometry())
                .unwrap()
                .initialization_peak_bytes();
        let (pool, reservation, run) = fresh_capacity(h, 0);
        let host = run
            .prepare_partition_fragment_host(&reservation, prototype, Some(geometry()))
            .unwrap();
        let program = PreparedPartitionCaptureProgram::new_selected(
            &transport,
            &source,
            &context,
            &[PreparedPartitionCaptureRow::Routed],
            limits,
            &metadata,
        )
        .unwrap()
        .with_routed_row(0, retained, host)
        .unwrap();
        let mut native = PartitionBackend {
            inner: backend(&run),
            program,
            fail,
            completed: 0,
        };
        let mut bank = bank(&source, &reservation, &run);
        let mut paths = super::super::super::source();
        Arc::get_mut(&mut paths.0).unwrap().prefill =
            vec![PrefillObservationDeclaration::causal_routed_units(
                "layer.0.output".into(),
            )]
            .into_boxed_slice();
        let selected = paths.prepare_capture_selection(&source).unwrap();
        let bound = selected.bind_geometry(geometry()).unwrap();
        let result = bank.with_prefill_observer(&mut native, bound, &Error::Capture, |observer| {
            for k in 0..2 {
                enter(observer, k);
                observer.coordinate_transaction(epoch(k)).unwrap();
                let tokens = (3 - k * 2).min(2);
                let input = scalar(&[tokens as i32, 5], vec![1.; tokens as usize * 5]);
                let groups = scalar(
                    &[tokens as i32, 3],
                    (0..tokens * 3).map(|i| (i % 7) as f32).collect(),
                );
                let routed = observer.routed_unit_observer("route").unwrap().unwrap();
                routed
                    .begin_invocation(&crate::RoutedUnitInvocation {
                        input: &input,
                        origins: None,
                        unit_coordinates: Some(ownership.coordinates.units()),
                    })
                    .unwrap();
                for token in 0..tokens {
                    let values = scalar(
                        &[3, 5],
                        (0..15)
                            .map(|i| (k * 2 + token) as f32 * 100. + i as f32 + 0.125)
                            .collect(),
                    );
                    let indices = scalar(&[3], vec![0., 0., 0.]);
                    let slots = scalar(&[3], vec![0., 1., 2.]);
                    let coefficients = scalar(&[1, 3], vec![0.25, 0.5, 0.75]);
                    let batch = crate::RoutedUnitBatch {
                        units: eredu_nn::GroupedUnitBatch {
                            values: &values,
                            group_indices: &slots,
                            selection_indices: &slots,
                            token_indices: &indices,
                            coefficients: &coefficients,
                            token_offset: token as usize,
                            total_token_count: tokens as usize,
                            group_count: 7,
                        },
                        source_groups: &groups,
                        global_groups: None,
                        provider_token_offset: 0,
                        origins: None,
                        unit_coordinates: Some(ownership.coordinates.units()),
                    };
                    if let Err(error) = routed.observe(&batch) {
                        return Some(error);
                    }
                    routed.observe_effective(&batch).unwrap();
                }
                routed.finish_invocation(true).unwrap();
                commit(observer, k);
            }
            observer.finish_prefill(true);
            None
        });
        assert_eq!(
            result
                .as_ref()
                .ok()
                .and_then(|value| value.as_ref())
                .is_some()
                || result.is_err(),
            fail
        );
        assert!(metadata_used.load(Ordering::SeqCst) > 0);
        if !fail {
            assert_eq!(native.completed, 2);
            assert_eq!(native.inner.transforms, 3);
            let delivery = bank.take_shared_step().unwrap().unwrap();
            let Some(CapturePayload::RoutedUnits(units)) = &delivery.records()[0].payload else {
                panic!("actual sparse final delivery")
            };
            assert_eq!(
                units
                    .rows
                    .iter()
                    .map(|r| (r.token, r.slot))
                    .collect::<Vec<_>>(),
                [(0, 0), (0, 2), (2, 0), (2, 2)]
            );
            for row in &units.rows {
                let TensorObservationData::F32(values) = row.values.data() else {
                    panic!("F32 rows")
                };
                assert_eq!(
                    values,
                    &[
                        row.token as f32 * 100. + row.slot as f32 * 5. + 1.125,
                        row.token as f32 * 100. + row.slot as f32 * 5. + 3.125
                    ]
                );
            }
            assert!(
                transport
                    .calls
                    .borrow()
                    .contains(&PartitionCaptureFrameKind::Delivery)
            );
            let PartitionBackend { inner, program, .. } = native;
            let Backend { scope, roots, .. } = inner;
            scope.certify().unwrap();
            drop((roots, program));
            drop(bank);
            drop(run);
            drop(reservation);
            assert!(pool.payload_used_bytes().unwrap() > 0);
            drop(delivery);
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        } else {
            assert_eq!(native.inner.transforms, 1);
            let PartitionBackend { inner, program, .. } = native;
            let Backend { scope, roots, .. } = inner;
            // This fixture's source completion and scalar transform are fully
            // synchronous. Explicitly settle that real mock scope after its
            // failed callback; dropping an uncertified scope must quarantine.
            scope.certify().unwrap();
            drop((roots, program));
            drop(bank);
            drop(run);
            drop(reservation);
            assert!(
                pool.payload_used_bytes().unwrap() > 0,
                "failed callback owns its partial Host destination"
            );
            drop(result);
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        }
    }
}
