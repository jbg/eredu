//! Real host-to-device miss, warm lease return and independent request custody.
use super::*;
use crate::backend::runtime::checkpoint::store::{
    OriginalMaterializationSlots, PreparedMaterializationObservation, PreparedPendingWeight,
    PreparedWeightMaterialization,
};
use crate::backend::submission_recovery::observed::bank::PreparedOperationBank;
use eredu_runtime::{residency::ResidencyClosureSlot, working_memory::OriginalTextControlGuard};
use safemlx::OriginalScopeObserver;

pub(crate) struct LeaseReturnFixture {
    manager: ResidencyManager,
    foreign: ResidencyManager,
    aliases: AliasClosureFixture,
    controller_failure: std::cell::RefCell<Option<PreparedAdmissionFailure>>,
    _native_runtime: [safemlx::PrefillRootsRuntime; 2],
    _directory: tempfile::TempDir,
}
impl LeaseReturnFixture {
    pub(crate) fn new(pool: &eredu_runtime::working_memory::WorkingMemoryPool) -> Self {
        let (directory, source) = fixture_store();
        let ids = [id("first"), id("second")];
        let graph = eredu_runtime::ExecutionGraph::new(
            vec![eredu_runtime::ExecutionGroupSpec::root("only")],
            "only",
        )
        .unwrap();
        let layout = eredu_runtime::execution::ExecutionUnitLayout::new(&graph, [2]).unwrap();
        let make = || {
            let source_stream = cpu_stream();
            let device_stream = cpu_stream();
            let native_runtime =
                safemlx::PrefillRootsRuntime::prepare_for_stream(&device_stream, &source_stream)
                    .unwrap();
            let plan = OffloadPlan::new(
                OffloadConfig::new(Some(16), Some(fixture_host_capacity(2)), 1).unwrap(),
                [
                    spec("first", 8, ResidencyPolicy::Cacheable, MemoryTier::Host),
                    spec("second", 8, ResidencyPolicy::Cacheable, MemoryTier::Host),
                ],
            )
            .unwrap();
            let manager = ResidencyManager::prepare_original_host(
                source.clone().into(),
                BTreeMap::new(),
                &plan,
                &[single("first", "a"), single("second", "b")],
                &["only".into()],
                &ids,
                &layout,
                2,
                &std::collections::BTreeSet::new(),
                None,
                &source_stream,
                &device_stream,
                pool,
            )
            .unwrap()
            .expect("admitted immutable host sources for both requested units");
            manager.inner.validate_pool(pool).unwrap();
            (manager, native_runtime)
        };
        let (manager, native_runtime) = make();
        let (foreign, foreign_runtime) = make();
        let aliases = AliasClosureFixture::new(source, pool);
        Self {
            manager,
            foreign,
            aliases,
            controller_failure: std::cell::RefCell::new(None),
            _native_runtime: [native_runtime, foreign_runtime],
            _directory: directory,
        }
    }

    fn acquire(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        warm: bool,
        refusal: u8,
    ) -> Option<ResidentTransfer> {
        let roots = [id("first"), id("second")];
        let mut scratch = [ResidencyClosureSlot::default(); 2];
        let catalog = self
            .manager
            .prepare_name_catalog(&roots, &mut scratch, controls.metadata_custody().into())
            .unwrap_or_else(|error| panic!("actual lease catalog: {}", error.into_source()));
        let mut expected_ids = Vec::new();
        let mut transfers = PreparedOperationBank::try_new(2, |index| {
            let ready = PreparedResidentTransfer::try_new_for_window(
                controls.metadata_custody().into(),
                TransferPayloadShape {
                    leases: 2,
                    lease_id_bytes: roots.iter().map(|id| id.as_str().len()).sum(),
                    unit_ids: 2,
                    prepared_units: 2,
                    unit_id_bytes: roots.iter().map(|id| id.as_str().len()).sum(),
                    retained_arrays: 6,
                    retained_host: 2,
                    retained_events: 2,
                    ..Default::default()
                },
                &self.manager,
                &catalog,
                &roots,
                &mut scratch,
            )
            .unwrap_or_else(|_| panic!("prepare exact lease return slot"));
            if index == 0 {
                expected_ids = ready.prepared_lease_id_addresses();
            }
            Ok::<_, std::convert::Infallible>(ready)
        })
        .unwrap();
        let mut observations = bank(0, || PreparedTransferObservation::new(controls.metadata_custody().into()));
        let mut host = bank(0, || PreparedHostMaterialization::new(controls.metadata_custody().into()));
        let mut pending = bank(0, || PreparedPendingWeight::new(controls.clone()));
        let mut weights = bank(0, || PreparedWeightMaterialization::new(controls.clone()));
        let mut consumers = bank(0, || {
            PreparedMaterializationObservation::new(controls.clone())
        });
        let mut requests = [(id("first"), 1), (id("second"), 1)];
        if refusal == 2 {
            requests.swap(0, 1);
        }
        let manager = if refusal == 1 {
            &self.foreign
        } else {
            &self.manager
        };
        let mut closure_ids = Some(
            self.manager
                .prepare_closure_ids(
                    &roots,
                    &mut scratch,
                    roots.len(),
                    roots.iter().map(|id| id.as_str().len()).sum(),
                    controls.metadata_custody().into(),
                )
                .unwrap_or_else(|_| panic!("prepare actual closure IDs")),
        );
        let closure_pointers = closure_ids.as_ref().unwrap().source_addresses();
        super::super::closure_ids::last_use();
        let mut controller = Some(self.manager.prepare_controller_for_test(
            &roots,
            &mut scratch,
            controls.clone(),
        ));
        let controller_addresses = controller.as_ref().unwrap().source_addresses();
        super::super::controller_attempt::last_use();
        let mut acquisitions = self
            .manager
            .prepare_source_acquisitions(&roots, &mut scratch, controls.clone())
            .unwrap();
        let before = self.pin_counts(observer);
        let _graph = (refusal == 0).then(|| copy_bank(observer, 2, 2));
        let result = manager.acquire_many_with_original_transfer(
            &requests,
            MemoryTier::Device,
            &mut OriginalResidencySlots {
            background_host: None,
                controller: &mut controller,
                closure_ids: &mut closure_ids,
                closure: &mut scratch,
                transfers: &mut transfers,
                observations: &mut observations,
                host_materializations: &mut host,
                reservation: None,
                foreground_disk: &mut super::super::PreparedForegroundDiskSlots::unavailable(),
                materialization: OriginalMaterializationSlots {
                    acquisitions: &mut acquisitions,
                    pending_weights: &mut pending,
                    weight_materializations: &mut weights,
                    observations: &mut consumers,
                },
            },
            observer,
        );
        if refusal != 0 {
            assert!(
                controller.is_some(),
                "earlier identity refusal preserves admission destination"
            );
            assert!(super::super::controller_attempt::last_use().is_none());
            assert!(super::super::closure_ids::last_use().is_none());
            assert!(
                closure_ids.is_some(),
                "earlier request refusal preserves closure storage"
            );
            assert!(matches!(
                result,
                Err(ResidencyError::OriginalNamedDestination(
                    NamedArrayError::InvalidSource
                ))
            ));
            assert_eq!(transfers.remaining(), 1, "only the warm slot was consumed");
            assert_eq!(
                self.pin_counts(observer),
                before,
                "refusal precedes ledger pinning"
            );
            let foreign = self.foreign.lock_original(observer).unwrap();
            assert!(foreign.storage.values().all(|unit| unit.device.is_none()));
            return None;
        }
        assert_eq!(
            super::super::closure_ids::last_use(),
            Some(closure_pointers),
            "actual warm/miss worker used the cold closure Vec and both String allocations"
        );
        assert!(
            closure_ids.is_none(),
            "successful acquisition consumes its own closure IDs"
        );
        let (a, b, c, d) = controller_addresses;
        assert_eq!(
            super::super::controller_attempt::last_use(),
            Some((a, b, c, d, 2, if warm { 0 } else { 2 })),
            "real warm/miss execution uses final prepaid buffers and initial flags"
        );
        assert!(
            controller.is_none(),
            "one controller owner consumed per actual attempt"
        );
        let mut transfer = result
            .unwrap_or_else(|error| panic!("actual lease acquire: {error}; debug: {error:?}"));
        assert_eq!(transfers.remaining(), usize::from(warm));
        transfer.synchronize().unwrap();
        assert_eq!(transfer.leases().len(), 2);
        for (index, lease) in transfer.leases().iter().enumerate() {
            assert_eq!(lease.id(), &roots[index]);
            assert_eq!(
                lease.id().as_str().as_ptr() as usize,
                expected_ids[index],
                "the actual cold ID allocation moved into the returned lease"
            );
            let expected: &[i32] = if index == 0 { &[1, 2] } else { &[3, 4] };
            assert_eq!(
                lease
                    .device_value("weight")
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<i32>(),
                expected
            );
        }
        assert_eq!(self.pin_counts(observer), before.map(|pins| pins + 1));
        Some(transfer)
    }

    fn pin_counts(&self, observer: &OriginalScopeObserver) -> [u64; 2] {
        let state = self.manager.lock_original(observer).unwrap();
        [id("first"), id("second")].map(|id| {
            state
                .control
                .ledger()
                .copy_status(&id, MemoryTier::Device)
                .unwrap()
                .map_or(0, |copy| copy.pins())
        })
    }

    pub(crate) fn first_request(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
    ) -> ResidentTransfer {
        self.manager
            .exercise_closure_destination_checks(&self.foreign, controls);
        *self.controller_failure.borrow_mut() = Some(
            self.manager
                .exercise_controller_destination_checks(&self.foreign, controls),
        );
        self.aliases.exercise(controls, observer);
        self.acquire(controls, observer, false, 1);
        self.acquire(controls, observer, false, 2);
        self.acquire(controls, observer, false, 0).unwrap()
    }

    pub(crate) fn settle_old_collection(&self) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut previous = [3, 2];
        loop {
            // Both original roles have ended. Retire the old application and
            // then its same cold collection node outside every manager lock.
            crate::backend::ordinary_retirement::reclaim();
            let state = self.manager.lock().unwrap();
            let pins = [id("first"), id("second")].map(|id| {
                state
                    .control
                    .ledger()
                    .copy_status(&id, MemoryTier::Device)
                    .unwrap()
                    .unwrap()
                    .pins()
            });
            drop(state);
            if pins == [1, 1] {
                break;
            }
            // The old two-lease transfer and the new request's one-lease
            // partial collection can retire in separate queue snapshots.
            assert!((1..=previous[0]).contains(&pins[0]));
            assert!((1..=previous[1]).contains(&pins[1]));
            previous = pins;
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
    }

    pub(crate) fn second_request(
        &self,
        old: ResidentTransfer,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
    ) -> ResidentTransfer {
        let old_failure = self
            .controller_failure
            .borrow_mut()
            .take()
            .expect("actual old request refusal");
        assert_eq!(old_failure.to_string(), "unknown residency unit: alien");
        assert!(old_failure.capacity().is_none());
        // Neutral buffers/source and their old custody can retire on a foreign
        // thread; no thread-affine observer was moved into this error owner.
        std::thread::spawn(move || drop(old_failure))
            .join()
            .unwrap();
        let current = self.acquire(controls, observer, true, 0).unwrap();
        let partial = self.partial_collection(controls, observer);
        let state = self.manager.lock_original(observer).unwrap();
        drop(partial);
        drop(old);
        for (id, expected) in [(id("first"), 3), (id("second"), 2)] {
            assert_eq!(
                state
                    .control
                    .ledger()
                    .copy_status(&id, MemoryTier::Device)
                    .unwrap()
                    .unwrap()
                    .pins(),
                expected,
                "actual lease cleanup is deferred while the manager lock is held"
            );
        }
        drop(state);
        current
    }

    fn partial_collection(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
    ) -> transfer::ResidentLeaseCollection {
        let roots = [id("first"), id("second")];
        let ready = transfer::PreparedLeaseCollection::try_new(
            &self.manager,
            &roots,
            TransferPayloadShape {
                leases: roots.len(),
                lease_id_bytes: roots.iter().map(|id| id.as_str().len()).sum(),
                ..Default::default()
            },
            controls.metadata_custody().into(),
        )
        .unwrap_or_else(|_| panic!("prepare actual partial lease collection"));
        ready
            .validate(
                &self.manager.inner,
                &[(roots[0].clone(), 37), (roots[1].clone(), 41)],
                MemoryTier::Device,
            )
            .unwrap();
        let ids = ready.requested_ids().unwrap();
        let buffer = ids.as_ptr();
        let second_cell = &ids[1] as *const OffloadUnitId;
        let string_addresses = ids
            .iter()
            .map(|id| id.as_str().as_ptr())
            .collect::<Vec<_>>();
        let mut builder = transfer::LeaseBuilder::new(Some(ready));
        let transfer::LeaseBuilder::Original(prepared) = &builder else {
            panic!("original return destination")
        };
        assert!(matches!(
            prepared.requested_ids(),
            Err(ResidencyError::OriginalNamedDestination(
                NamedArrayError::InvalidSource
            ))
        ));
        assert_eq!(prepared.remaining_ids_for_test().as_ptr(), buffer);

        let mut state = self.manager.lock_original(observer).unwrap();
        let before = roots.each_ref().map(|id| {
            state
                .control
                .ledger()
                .copy_status(id, MemoryTier::Device)
                .unwrap()
                .unwrap()
                .pins()
        });
        let storage = state
            .storage
            .get(&roots[0])
            .unwrap()
            .device
            .as_ref()
            .unwrap()
            .clone();
        builder
            .acquire(
                state.control.ledger_mut(),
                &roots[0],
                MemoryTier::Device,
                37,
                ResidentLeaseStorage::Device(storage.clone()),
                self.manager.inner.downgrade(),
            )
            .unwrap();
        // A wrong second ID must leave the next String, iterator buffer and
        // second unit pin untouched while retaining the completed first lease.
        let error = builder
            .acquire(
                state.control.ledger_mut(),
                &roots[0],
                MemoryTier::Device,
                41,
                ResidentLeaseStorage::Device(storage),
                self.manager.inner.downgrade(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ResidencyError::OriginalNamedDestination(NamedArrayError::InvalidSource)
        ));
        let transfer::LeaseBuilder::Original(prepared) = &builder else {
            unreachable!()
        };
        assert!(prepared.requested_ids().is_err());
        let remaining = prepared.remaining_ids_for_test();
        assert_eq!(remaining, &roots[1..]);
        assert_eq!(remaining.as_ptr(), second_cell);
        assert_eq!(remaining[0].as_str().as_ptr(), string_addresses[1]);
        for (index, id) in roots.iter().enumerate() {
            assert_eq!(
                state
                    .control
                    .ledger()
                    .copy_status(id, MemoryTier::Device)
                    .unwrap()
                    .unwrap()
                    .pins(),
                before[index] + u64::from(index == 0)
            );
        }
        let collection = builder.finish();
        assert_eq!(collection.as_slice().len(), 1);
        assert_eq!(collection.as_slice()[0].id(), &roots[0]);
        assert_eq!(
            collection.as_slice()[0].id().as_str().as_ptr(),
            string_addresses[0]
        );
        drop(state);
        collection
    }
}

fn bank<S>(count: usize, mut make: impl FnMut() -> S) -> PreparedOperationBank<S> {
    PreparedOperationBank::try_new(count, |_| Ok::<_, std::convert::Infallible>(make())).unwrap()
}

fn copy_bank(
    observer: &OriginalScopeObserver,
    copies: usize,
    bindings: usize,
) -> safemlx::PreparedResidentGraph {
    use safemlx::{OperationEvalTraversalLimits, OperationEvent};
    // Each physical source constructs one leaf and one copy; each published
    // binding can clone one native handle. The same admitted Graph/Record
    // quotas cover the copy frontiers and the final alias-expanded aggregate.
    let layout =
        OperationEvent::resident_graph_layout_with_shells(copies, copies, 1, 4, bindings)
            .unwrap();
    let roots = bindings.max(2);
    let traversal = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
        roots,
        arrays: roots + 1,
        tape_entries: 2,
        input_edges: roots,
        output_slots: 2,
        streams: 1,
        captures: OperationEvent::eval_record_layout(2, 1, 2)
            .unwrap()
            .capture_slots(),
    })
    .unwrap();
    let mut graph = OperationEvent::prepare_resident_graph(layout, observer).unwrap();
    graph
        .configure_nested_completions(&traversal, copies + 1)
        .unwrap();
    graph
}

// The actual admitted fixture invokes this under the same genuine controls and
// observer as the miss/warm lease test. Its manager and ordinary reference are
// built before the original role, including all host source materialization.
// The original manager admits its immutable host sources through the same pool.
struct AliasClosureFixture {
    manager: ResidencyManager,
    _native_runtime: safemlx::PrefillRootsRuntime,
    window: WindowPopulation,
    expected: [[i32; 2]; 4],
}
impl AliasClosureFixture {
    fn new(
        source: Arc<SafetensorsWeightStore>,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Self {
        let graph = eredu_runtime::ExecutionGraph::new(
            vec![eredu_runtime::ExecutionGroupSpec::root("only")],
            "only",
        )
        .unwrap();
        let layout = eredu_runtime::execution::ExecutionUnitLayout::new(&graph, [1]).unwrap();
        let build = |original| {
            let source_stream = cpu_stream();
            let device_stream = cpu_stream();
            let runtime =
                safemlx::PrefillRootsRuntime::prepare_for_stream(&device_stream, &source_stream)
                    .unwrap();
            let tier = if original {
                MemoryTier::Host
            } else {
                MemoryTier::Disk
            };
            let plan = OffloadPlan::new(
                OffloadConfig::new(Some(16), Some(fixture_host_capacity(2)), 1).unwrap(),
                [
                    spec("first", 8, ResidencyPolicy::Cacheable, tier),
                    spec("second", 8, ResidencyPolicy::Cacheable, tier),
                ],
            )
            .unwrap();
            let units = [
                unit(
                    "first",
                    [
                        binding("own_a", "a", TensorSelection::Full, 8)
                            .with_logical_target("first.owner")
                            .unwrap(),
                        WeightBinding::alias("other_b", "second.owner", 8).unwrap(),
                    ],
                ),
                unit(
                    "second",
                    [
                        binding("own_b", "b", TensorSelection::Full, 8)
                            .with_logical_target("second.owner")
                            .unwrap(),
                        WeightBinding::alias("other_a", "first.owner", 8).unwrap(),
                    ],
                ),
            ];
            let manager = if original {
                ResidencyManager::prepare_original_host(
                    source.clone().into(),
                    BTreeMap::new(),
                    &plan,
                    &units,
                    &["only".into()],
                    &[id("first")],
                    &layout,
                    1,
                    &std::collections::BTreeSet::new(),
                    None,
                    &source_stream,
                    &device_stream,
                    pool,
                )
                .unwrap()
                .expect("admitted immutable host sources for the alias closure")
            } else {
                let manager = ResidencyManager::new(
                    source.clone(), plan, units, source_stream, device_stream,
                )
                .unwrap();
                manager.initialize().unwrap();
                manager
            };
            if original {
                manager.inner.validate_pool(pool).unwrap();
            }
            let host = manager.acquire(&id("first"), MemoryTier::Host).unwrap();
            assert_eq!(host_i32(&host, "own_a"), [1, 2]);
            assert_eq!(host_i32(&host, "other_b"), [3, 4]);
            drop(host);
            (manager, runtime)
        };
        let (ordinary, ordinary_runtime) = build(false);
        let mut reference = ordinary
            .acquire_many_with_transfer(&[(id("first"), 1)], MemoryTier::Device)
            .unwrap();
        reference.synchronize().unwrap();
        let expected = Self::snapshot(&ordinary.lock().unwrap(), &reference);
        assert_eq!(expected, [[1, 2], [3, 4], [3, 4], [1, 2]]);
        drop(reference);
        drop(ordinary);
        drop(ordinary_runtime);
        let (manager, runtime) = build(true);
        let source = manager
            .prepare_original_operation_source(&[id("first")], &layout, 1)
            .unwrap();
        let window = source.windows()[0];
        assert_eq!(window.requested, 1);
        assert_eq!(
            window.units, 2,
            "the real alias cycle expands the requested root"
        );
        assert_eq!(window.unit_id_bytes, "first".len() + "second".len());
        Self {
            manager,
            _native_runtime: runtime,
            window,
            expected,
        }
    }
    fn snapshot(state: &ManagerState, transfer: &ResidentTransfer) -> [[i32; 2]; 4] {
        assert_eq!(
            transfer.leases().len(),
            1,
            "closure owners do not become extra requested leases"
        );
        let first = &transfer.leases()[0];
        assert_eq!(first.id(), &id("first"));
        let second = state
            .storage
            .get(&id("second"))
            .unwrap()
            .device
            .as_ref()
            .unwrap();
        let arrays = [
            first.device_value("own_a").unwrap(),
            first.device_value("other_b").unwrap(),
            second.arrays.get("own_b").unwrap(),
            second.arrays.get("other_a").unwrap(),
        ];
        let values = arrays.map(|array| {
            array
                .evaluated()
                .unwrap()
                .as_slice::<i32>()
                .try_into()
                .unwrap()
        });
        assert_eq!(
            arrays[0].allocation_info().unwrap().unwrap().identity(),
            arrays[3].allocation_info().unwrap().unwrap().identity()
        );
        assert_eq!(
            arrays[1].allocation_info().unwrap().unwrap().identity(),
            arrays[2].allocation_info().unwrap().unwrap().identity()
        );
        values
    }
    fn exercise(&self, controls: &OriginalTextControlGuard, observer: &OriginalScopeObserver) {
        for warm in [false, true] {
            let roots = [id("first")];
            let mut scratch = [ResidencyClosureSlot::default(); 2];
            let catalog = self
                .manager
                .prepare_name_catalog(&roots, &mut scratch, controls.metadata_custody().into())
                .unwrap_or_else(|e| panic!("actual alias catalog: {}", e.into_source()));
            let shape = TransferPayloadShape::window(self.window).unwrap();
            let mut transfers = PreparedOperationBank::try_new(2, |_| {
                let ready = PreparedResidentTransfer::try_new_for_window(
                    controls.metadata_custody().into(),
                    shape,
                    &self.manager,
                    &catalog,
                    &roots,
                    &mut scratch,
                )
                .unwrap_or_else(|_| panic!("prepare actual alias window"));
                Ok::<_, std::convert::Infallible>(ready)
            })
            .unwrap();
            let mut observations = bank(0, || PreparedTransferObservation::new(controls.metadata_custody().into()));
            let mut host = bank(0, || PreparedHostMaterialization::new(controls.metadata_custody().into()));
            let mut pending = bank(0, || PreparedPendingWeight::new(controls.clone()));
            let mut weights = bank(0, || PreparedWeightMaterialization::new(controls.clone()));
            let mut consumers = bank(0, || {
                PreparedMaterializationObservation::new(controls.clone())
            });
            let mut closure_ids = Some(
                self.manager
                    .prepare_closure_ids(
                        &roots,
                        &mut scratch,
                        self.window.units,
                        self.window.unit_id_bytes,
                        controls.metadata_custody().into(),
                    )
                    .unwrap_or_else(|_| panic!("prepare source-expanded closure IDs")),
            );
            let addresses = closure_ids.as_ref().unwrap().source_addresses();
            assert_eq!(addresses.1, 2);
            super::super::closure_ids::last_use();
            let mut controller = Some(self.manager.prepare_controller_for_test(
                &roots,
                &mut scratch,
                controls.clone(),
            ));
            let mut acquisitions = self
                .manager
                .prepare_source_acquisitions(&roots, &mut scratch, controls.clone())
                .unwrap();
            let _graph = copy_bank(observer, self.window.physical_bindings, self.window.bindings);
            let mut transfer = self
                .manager
                .acquire_many_with_original_transfer(
                    &[(id("first"), 1)],
                    MemoryTier::Device,
                    &mut OriginalResidencySlots {
            background_host: None,
                        controller: &mut controller,
                        closure_ids: &mut closure_ids,
                        closure: &mut scratch,
                        transfers: &mut transfers,
                        observations: &mut observations,
                        host_materializations: &mut host,
                        reservation: None,
                        foreground_disk:
                            &mut super::super::PreparedForegroundDiskSlots::unavailable(),
                        materialization: OriginalMaterializationSlots {
                            acquisitions: &mut acquisitions,
                            pending_weights: &mut pending,
                            weight_materializations: &mut weights,
                            observations: &mut consumers,
                        },
                    },
                    observer,
                )
                .unwrap_or_else(|e| panic!("actual alias-expanded acquire: {e}; {e:?}"));
            assert_eq!(super::super::closure_ids::last_use(), Some(addresses));
            assert!(closure_ids.is_none());
            assert_eq!(transfers.remaining(), usize::from(warm));
            transfer.synchronize().unwrap();
            let state = self.manager.lock_original(observer).unwrap();
            assert_eq!(Self::snapshot(&state, &transfer), self.expected);
            drop(state);
            drop(transfer);
        }
    }
}
