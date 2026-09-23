use super::super::observer::HookTransport;
use super::*;
use crate::{ActivationObserver, RoutedUnitBatch, RoutedUnitInvocation, RoutedUnitObserver};
use std::collections::BTreeMap;

struct Layout;
impl PartitionCaptureLayout for Layout {
    fn capture_placement(
        &self,
        _: &AdmittedCapturePlan,
        _: usize,
        _: CapturePhase,
        _: u64,
        _: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        panic!("sparse selections must not use dense source geometry")
    }
    fn routed_capture_placement(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        _: CapturePhase,
        _: u64,
        _: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionRoutedCapturePlacement, CaptureError> {
        let producers = fixture::producers(plan, true);
        let mut sources = producers
            .iter()
            .map(|producer| PartitionRoutedCaptureSource {
                rank: producer.rank,
                ownership: producer.ownership.clone(),
                input_width: 4,
            })
            .collect::<Vec<_>>();
        // A replica participates in source work and votes but exports no duplicate.
        sources.push(PartitionRoutedCaptureSource {
            rank: 7,
            ..sources[2].clone()
        });
        Ok(PartitionRoutedCapturePlacement {
            routing: "experts".into(),
            effective: index == 1,
            producers,
            sources,
        })
    }
}

#[derive(Default)]
struct ObserverBackend {
    inner: SparseBackend,
    calls: Arc<[AtomicUsize; 2]>,
    source_failure: bool,
    source_extra: u64,
}
impl CaptureBackend for ObserverBackend {
    type Tensor = Value;
    type Error = std::io::Error;
    fn shape(&self, tensor: &Value) -> Result<Vec<u64>, Self::Error> {
        self.inner.shape(tensor)
    }
    fn source_dtype(&self, tensor: &Value) -> Option<TensorDtype> {
        self.inner.source_dtype(tensor)
    }
    fn estimate(
        &self,
        _: &Value,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        unreachable!()
    }
    fn transform(
        &mut self,
        _: &Value,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        unreachable!()
    }
    fn estimate_partition_routed_units(
        &self,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        self.inner.estimate_partition_routed_units(request)
    }
    fn capture_partition_routed_units(
        &mut self,
        source: &PartitionRoutedUnitCaptureSource<'_, Value>,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Option<Result<RoutedUnitCapture, Self::Error>> {
        self.calls[1].fetch_add(1, Ordering::SeqCst);
        self.inner.capture_partition_routed_units(source, request)
    }
    fn estimate_partition_source(
        &self,
        shape: &[u64],
        _: BoundedCompletionWait,
    ) -> Result<CaptureUsage, CaptureError> {
        assert_eq!(
            shape,
            [18, 4],
            "prepay worst-case receive rows, not original rows or selected scalar output"
        );
        Ok(CaptureUsage {
            retained_bytes: shape.iter().product::<u64>() * 4 + 128 + self.source_extra,
            ..Default::default()
        })
    }
    fn prepare_partition_source(
        &mut self,
        tensor: &Value,
        _: BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
        assert_eq!(tensor.shape.last(), Some(&4));
        self.calls[0].fetch_add(1, Ordering::SeqCst);
        if self.source_failure {
            return Err(std::io::Error::other("source completion sentinel"));
        }
        Ok(eredu_core::BoundedCompletionOutcome::Completed)
    }
}

fn invoke(
    observer: &mut dyn RoutedUnitObserver<Value>,
    owned: &RoutedUnitCaptureOwnership,
    fault: Option<&str>,
) -> Result<(), eredu_nn::Error> {
    let mut tags = vec![];
    let mut experts = vec![];
    for token in 0..3usize {
        for slot in 0..2 {
            let expert = if token == 0 { 1 } else { (token + slot) % 5 };
            if owned
                .coordinates
                .experts()
                .global_to_local(expert)
                .is_some()
            {
                tags.push(token * 2 + slot);
                experts.push(expert as u64);
            }
        }
    }
    let count = tags.len();
    tags.extend_from_within(..);
    experts.extend_from_within(..);
    let counts = if fault == Some("origins") {
        vec![count * 2]
    } else {
        vec![count, 0, count]
    };
    let origins = crate::RoutedUnitOrigins::new(&counts, &tags, 2).unwrap();
    let rows = tags.len() as u64;
    let input = Value {
        shape: vec![
            1,
            if fault == Some("rows") { 19 } else { rows },
            if fault == Some("width") { 5 } else { 4 },
        ],
        data: if fault == Some("dtype") {
            TensorObservationData::U64(vec![])
        } else {
            TensorObservationData::F32(vec![])
        },
    };
    let wrong_units = ComponentCoordinateMap::range(7, 0..7).unwrap();
    let invocation = RoutedUnitInvocation {
        input: &input,
        origins: Some(origins),
        unit_coordinates: (fault == Some("units")).then_some(&wrong_units),
    };
    crate::with_routed_unit_invocation(
        Some(observer),
        invocation,
        |observer| {
            let observer = observer.unwrap();
            if fault == Some("provider") {
                return Err(eredu_nn::Error::backend("provider sentinel"));
            }
            let groups = Value {
                shape: vec![rows, 1],
                data: TensorObservationData::U64(experts),
            };
            let units = owned.coordinates.units();
            for start in (0..rows).step_by(2) {
                let end = (start + 2).min(rows);
                if fault == Some("missing chunk") && end == rows {
                    break;
                }
                let values = Value {
                    shape: vec![end - start, units.local_count() as u64],
                    data: TensorObservationData::F32(
                        (start..end)
                            .flat_map(|row| {
                                let origin = origins.resolve(row as usize).unwrap();
                                (0..units.local_count()).map(move |unit| {
                                    scalar(
                                        origin.token,
                                        origin.slot,
                                        units.local_to_global(unit).unwrap(),
                                    ) + origin.source_peer.unwrap() as f32 * 1000.
                                })
                            })
                            .collect(),
                    ),
                };
                let coefficients = Value {
                    shape: vec![end - start, 1],
                    data: TensorObservationData::F32(vec![]),
                };
                let batch = |values| RoutedUnitBatch {
                    units: eredu_nn::GroupedUnitBatch {
                        values,
                        group_indices: &groups,
                        token_indices: &groups,
                        selection_indices: &groups,
                        coefficients: &coefficients,
                        token_offset: 0,
                        total_token_count: rows as usize,
                        group_count: 5,
                    },
                    source_groups: &groups,
                    global_groups: None,
                    provider_token_offset: start as usize,
                    origins: Some(origins),
                    unit_coordinates: Some(units),
                };
                observer.observe(&batch(&values))?;
                let TensorObservationData::F32(data) = &values.data else {
                    unreachable!()
                };
                let effective = Value {
                    shape: values.shape.clone(),
                    data: TensorObservationData::F32(
                        data.iter().map(|value| value * -0.5).collect(),
                    ),
                };
                observer.observe_effective(&batch(&effective))?;
            }
            Ok(())
        },
        |error| error,
    )
}

#[test]
fn shared_routed_observer_prepays_receives_and_votes_before_provider_returns() {
    let mut usage_by_empty = BTreeMap::new();
    for (empty, committed, fault, failed_rank) in [
        (false, true, None, 2),
        (true, true, None, 2),
        (false, false, None, 2),
        (false, false, Some("width"), 2),
        (false, false, Some("width"), 0),
        (false, false, Some("width"), 7),
        (false, false, Some("rows"), 2),
        (false, false, Some("origins"), 2),
        (false, false, Some("dtype"), 2),
        (false, false, Some("units"), 2),
        (false, false, Some("provider"), 2),
        (false, false, Some("provider"), 0),
        (false, false, Some("missing chunk"), 2),
        (false, false, Some("corrupt chunk"), 2),
    ] {
        let all = world(9);
        let active = world(8);
        let outputs = std::thread::scope(|scope| {
            (0..9)
                .map(|rank| {
                    let all = Arc::clone(&all);
                    let active = Arc::clone(&active);
                    scope.spawn(move || {
                        let transport = HookTransport {
                            transport: transport(all, rank, Fault::None),
                            hook: (rank < 8).then(|| transport(active, rank, Fault::None)),
                            members: (0..8).collect(),
                        };
                        let plan = fixture::plan_with_effective(empty, true);
                        let ownership = Layout
                            .routed_capture_placement(
                                &plan,
                                0,
                                CapturePhase::Prefill,
                                0,
                                fixture::limits(),
                            )
                            .unwrap()
                            .sources
                            .into_iter()
                            .find(|source| source.rank == rank)
                            .map(|source| source.ownership);
                        let mut session = configured(plan, 9);
                        let calls = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
                        let backend = ObserverBackend {
                            calls: Arc::clone(&calls),
                            inner: SparseBackend {
                                corrupt: rank == failed_rank && fault == Some("corrupt chunk"),
                                ..Default::default()
                            },
                            ..Default::default()
                        };
                        let epoch = DistributedCommitEpoch::FIRST;
                        let mut observer = PartitionCaptureObserver::for_step(
                            &mut session,
                            backend,
                            &transport,
                            &Layout,
                            0,
                            fixture::limits(),
                            estimate,
                            |error| error,
                        );
                        observer
                            .prepare_transaction(epoch, crate::ExpertPass::Prefill)
                            .unwrap();
                        observer.coordinate_transaction(epoch).unwrap();
                        if let Some(owned) = &ownership {
                            let unit_observer =
                                observer.routed_unit_observer("experts").unwrap().unwrap();
                            let result = invoke(
                                unit_observer,
                                owned,
                                if rank == failed_rank { fault } else { None },
                            );
                            assert_eq!(
                                result.is_ok(),
                                fault.is_none(),
                                "rank {rank}, {fault:?}: {result:?}"
                            );
                            // Returning from invoke establishes all final votes. Capture
                            // remains staged; no model reverse exchange has been called.
                            assert!(!observer.invocation_active());
                            assert_eq!(transport.hook.as_ref().unwrap().calls.load(Ordering::SeqCst), 4, "both source votes and both final votes finish before provider return");
                        }
                        let source_calls = calls[0].load(Ordering::SeqCst);
                        let collector_calls = calls[1].load(Ordering::SeqCst);
                        let delivered = observer.complete_transaction(epoch);
                        assert_eq!(
                            delivered.is_ok(),
                            fault.is_none(),
                            "{fault:?}: {delivered:?}"
                        );
                        observer.finish_transaction(epoch, committed);
                        drop(observer);
                        let step = session.take_step().unwrap();
                        let prepaid = step.cumulative_usage;
                        if committed {
                            for (index, record) in step.records.iter().enumerate() {
                                assert_eq!(record.outcome, CaptureOutcome::Captured);
                                let Some(CapturePayload::RoutedUnits(payload)) = &record.payload
                                else {
                                    panic!("routed payload");
                                };
                                assert_eq!(payload.rows.len(), if empty { 0 } else { 4 });
                                for row in &payload.rows {
                                    let TensorObservationData::F32(values) = row.values.data()
                                    else {
                                        panic!("float rows");
                                    };
                                    let multiplier = if index == 0 { 1. } else { -0.5 };
                                    for (unit, value) in values.iter().enumerate() {
                                        assert_eq!(
                                            *value,
                                            scalar(
                                                row.token as usize,
                                                row.slot as usize,
                                                1 + 2 * unit
                                            ) * multiplier
                                        );
                                    }
                                }
                            }
                            assert_eq!(step.partitions.len(), 2);
                        } else {
                            assert!(step.records.iter().all(|record| record.payload.is_none()));
                        }
                        if rank == 8 || matches!(fault, Some("width" | "rows" | "origins" | "dtype" | "units")) {
                            assert_eq!(source_calls, 0);
                        } else {
                            assert_eq!(source_calls, 2);
                        }
                        if empty || !(2..6).contains(&rank) {
                            assert_eq!(collector_calls, 0);
                        }
                        prepaid
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(outputs.windows(2).all(|pair| pair[0] == pair[1]));
        if let Some(previous) = usage_by_empty.insert(empty, outputs[0]) {
            assert_eq!(
                previous, outputs[0],
                "failure and abort never refund preparation"
            );
        }
    }
}

#[test]
fn routed_observer_source_budget_rejects_before_source_or_collection_work() {
    let transport = HookTransport {
        transport: transport(world(9), 0, Fault::None),
        hook: None,
        members: (0..8).collect(),
    };
    let calls = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
    let backend = ObserverBackend {
        calls: Arc::clone(&calls),
        source_extra: 1_000_000_000,
        ..Default::default()
    };
    let mut session = configured(fixture::plan_with_effective(false, true), 9);
    let mut observer = PartitionCaptureObserver::for_step(
        &mut session,
        backend,
        &transport,
        &Layout,
        0,
        fixture::limits(),
        estimate,
        |error| error,
    );
    let error = observer
        .prepare_transaction(DistributedCommitEpoch::FIRST, crate::ExpertPass::Prefill)
        .unwrap_err();
    assert!(matches!(
        error,
        PartitionCaptureObserverError::Exchange(PartitionCaptureExchangeError::Capture(
            CaptureError::Limit {
                budget: CaptureBudget::Retention,
                ..
            }
        ))
    ));
    assert!(calls.iter().all(|calls| calls.load(Ordering::SeqCst) == 0));
    assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0);
    observer.finish_transaction(DistributedCommitEpoch::FIRST, false);
}

struct ChangedLayout(&'static str);
impl PartitionCaptureLayout for ChangedLayout {
    fn capture_placement(
        &self,
        _: &AdmittedCapturePlan,
        _: usize,
        _: CapturePhase,
        _: u64,
        _: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        unreachable!()
    }
    fn routed_capture_placement(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionRoutedCapturePlacement, CaptureError> {
        let mut placement =
            Layout.routed_capture_placement(plan, index, phase, prediction, limits)?;
        match self.0 {
            "invocation" => placement.routing.push_str(".unrelated"),
            "timing" => placement.effective = !placement.effective,
            "source" => placement.sources[0].ownership = placement.sources[2].ownership.clone(),
            _ => unreachable!(),
        }
        Ok(placement)
    }
}

#[test]
fn routed_observer_rejects_changed_invocation_timing_and_source_before_native_work() {
    for changed in ["invocation", "timing", "source"] {
        let transport = HookTransport {
            transport: transport(world(9), 0, Fault::None),
            hook: None,
            members: (0..8).collect(),
        };
        let calls = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let backend = ObserverBackend {
            calls: Arc::clone(&calls),
            ..Default::default()
        };
        let mut session = configured(fixture::plan_with_effective(false, true), 9);
        let layout = ChangedLayout(changed);
        let mut observer = PartitionCaptureObserver::for_step(
            &mut session,
            backend,
            &transport,
            &layout,
            0,
            fixture::limits(),
            estimate,
            |error| error,
        );
        assert!(observer
            .prepare_transaction(DistributedCommitEpoch::FIRST, crate::ExpertPass::Prefill)
            .is_err());
        assert!(calls.iter().all(|calls| calls.load(Ordering::SeqCst) == 0));
        assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0);
        observer.finish_transaction(DistributedCommitEpoch::FIRST, false);
    }
}
