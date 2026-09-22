use super::*;
use crate::backend::runtime::checkpoint::store::{cache, PreparedSourceAcquisitions};
use crate::backend::runtime::residency::manager::acquisition_destinations::SelectedSourceOccurrence;
use eredu_checkpoint::store::{SelectedGgufConversionPlan, SharedCheckpointSource};
use eredu_runtime::working_memory::OriginalHostMetadataCustody;
use std::alloc::Layout;

fn qualified() -> bool {
    let result = OriginalHostMetadataCustody::fresh_clone_storage_bytes(Layout::new::<u8>());
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_G4").is_some() {
        assert!(result.is_ok(), "mandatory pinned G4 producer");
    }
    match result {
        Ok(1) => true,
        Err(WorkingMemoryError::UnknownBound) => false,
        other => panic!("unexpected qualification {other:?}"),
    }
}
fn selected(
    fixture: &OriginalGgufMissFixture,
    count: usize,
) -> (SharedCheckpointSource, Vec<SelectedSourceOccurrence>, u64) {
    let root: SharedCheckpointSource = Arc::new(fixture.source.clone());
    let plan = SelectedGgufConversionPlan::query(root.clone(), OriginalGgufMissFixture::request())
        .unwrap()
        .unwrap();
    let acquisition = plan.acquisition_storage().unwrap().unwrap();
    let rows = vec![SelectedSourceOccurrence {
        plan,
        multiplicity: count,
        acquisition,
    }];
    let route_layout = rows[0].plan.acquisition_route_shared_layout().unwrap();
    let route_header = OriginalHostMetadataCustody::shared_storage_bytes(route_layout).unwrap()
        - route_layout.size() as u64;
    let existing = route_header
        + super::cache_metadata::cache_baseline(fixture)
        + OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<NeutralGgufWeightStore>())
            .unwrap()
        + Layout::array::<SelectedSourceOccurrence>(rows.capacity())
            .unwrap()
            .size() as u64
        + (rows[0].plan.metadata_bytes().unwrap() - size_of::<SelectedGgufConversionPlan>()) as u64;
    (root, rows, existing)
}

#[test]
fn g4_exact_tickets_outlive_bank_and_repeated_exhaustion_owns_no_custody() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<crate::backend::runtime::checkpoint::store::PreparedSourceAcquisitionFailure>();
    if !qualified() {
        return;
    }
    let fixture = OriginalGgufMissFixture::new();
    let (root, rows, existing) = selected(&fixture, 1);
    let bytes = PreparedSourceAcquisitions::selected_storage_bytes(&rows, 1).unwrap();
    let reads = fixture.source_physical_reads();
    let (lease, errors, pool) = component_destinations_with_baseline(
        bytes,
        1,
        0,
        None,
        existing,
        |_| 0,
        |controls, _, pool, mut host| {
            let mut bank = PreparedSourceAcquisitions::prepare_selected(
                &rows,
                1,
                bytes,
                host.take_source_constructions().unwrap(),
                controls.clone(),
            )
            .unwrap();
            let lease = bank
                .acquire(root.as_ref(), "bank.weight", &TensorSelection::Full)
                .unwrap();
            assert_eq!(bank.remaining(), 0);
            let errors = (0..32)
                .map(|_| {
                    bank.acquire(root.as_ref(), "bank.weight", &TensorSelection::Full)
                        .unwrap_err()
                })
                .collect::<Vec<_>>();
            for error in &errors {
                let actual = std::error::Error::source(error)
                    .unwrap()
                    .downcast_ref::<eredu_checkpoint::store::PreparedAcquisitionRefusal>()
                    .unwrap();
                assert!(matches!(
                    actual,
                    eredu_checkpoint::store::PreparedAcquisitionRefusal::NoDestination
                ));
            }
            drop(bank);
            assert_eq!(fixture.source_physical_reads(), reads);
            (lease, errors, pool.clone())
        },
    );
    assert!(pool.fixture_host_charge().unwrap() > existing);
    drop((root, rows, fixture, lease));
    settle_pool_at(&pool, existing);
    assert_eq!(errors.len(), 32, "retained fixed refusals do not pin G4");
    drop(errors);
}

#[test]
fn g4_debit_refusal_is_typed_and_exhaustion_has_no_custody() {
    if !qualified() {
        return;
    }
    for exhausted in [false, true] {
        let fixture = OriginalGgufMissFixture::new();
        let (root, rows, existing) = selected(&fixture, 2);
        let bytes = PreparedSourceAcquisitions::selected_storage_bytes(&rows, 2).unwrap();
        let reads = fixture.source_physical_reads();
        let (error, pool) = component_destinations_with_baseline(
            bytes - 1,
            1,
            0,
            None,
            existing,
            |_| 0,
            |controls, _, pool, mut host| {
                let mut source = host.take_source_constructions().unwrap();
                if exhausted {
                    drop(source.try_debit(0).unwrap());
                }
                let error = match PreparedSourceAcquisitions::prepare_selected(
                    &rows,
                    2,
                    bytes,
                    source,
                    controls.clone(),
                ) {
                    Err(error) => error,
                    Ok(_) => panic!("must refuse before tickets"),
                };
                let actual = std::error::Error::source(&error)
                    .unwrap()
                    .downcast_ref::<eredu_runtime::working_memory::OriginalHostSourceError>()
                    .unwrap();
                if exhausted {
                    assert!(matches!(actual.cause(), HostDestinationCause::Attempts));
                    assert!(!actual.retains_receipt());
                } else {
                    assert!(
                        matches!(actual.cause(),HostDestinationCause::Capacity { required,remaining } if *required==bytes && *remaining==bytes-1)
                    );
                    assert!(actual.retains_receipt());
                }
                assert_eq!(fixture.source_physical_reads(), reads);
                (error, pool.clone())
            },
        );
        drop((root, rows, fixture));
        if exhausted {
            settle_pool_at(&pool, existing);
        } else {
            assert!(pool.fixture_host_charge().unwrap() > existing);
        }
        drop(error);
        settle_pool_at(&pool, existing);
    }
}

fn pending_controls() -> u64 {
    // Actual direct constructor and its typed result; reuse the owning slot
    // layout without inventing a bank factory which this fixture never calls.
    let transport = size_of::<PreparedPendingWeight>()
        + size_of::<OriginalTextControlGuard>()
        + size_of::<Rc<safemlx::PreparedInputRuntime>>()
        + size_of::<OriginalHostDestinationBank>()
        + size_of::<Result<PreparedPendingWeight, (OriginalHostDestinationBank, WorkingMemoryError)>>(
        );
    PreparedPendingWeight::slot_control_bytes().unwrap() + transport as u64
}
#[test]
fn g4_same_box_survives_g1_refusal_or_successful_native_source_alias() {
    if !qualified() {
        return;
    }
    for refuse_raw in [true, false] {
        let fixture = OriginalGgufMissFixture::new();
        let (root, rows, existing) = selected(&fixture, 1);
        let g4 = PreparedSourceAcquisitions::selected_storage_bytes(&rows, 1).unwrap();
        let destinations =
            PreparedPendingWeight::host_destination_requests(rows[0].plan.physical()).unwrap();
        let copy =
            fixture.source_storage_layouts().iter().sum::<u64>() + cache::control_bytes().unwrap();
        let copy_attempts = rows[0].plan.physical().conversion().outputs().len() + 1;
        // The actual current-miss cache key is prepared before G1. Fund its
        // exact buffers so this case reaches the intended raw-read refusal.
        let destination_bytes = if refuse_raw {
            rows[0]
                .plan
                .physical()
                .identity()
                .cache_storage_requests()
                .unwrap()
                .bytes() as u64
        } else {
            destinations.bytes() as u64
        };
        let reads = fixture.source_physical_reads();
        let raw_bytes = rows[0].plan.physical().requested_read_bytes();
        let pending = pending_controls();
        let (error, array, context, pool, address) = component_destinations_with_baseline(
            g4 + pending + copy,
            2 + copy_attempts,
            3,
            Some((destination_bytes, destinations.calls(), 3)),
            existing,
            |_| 0,
            |controls, observer, pool, mut host| {
                let mut g4bank = host.split(0, 0, Some((g4, 1))).unwrap();
                let mut acquisitions = PreparedSourceAcquisitions::prepare_selected(
                    &rows,
                    1,
                    g4,
                    g4bank.take_source_constructions().unwrap(),
                    controls.clone(),
                )
                .unwrap();
                let acquired = acquisitions
                    .acquire(root.as_ref(), "bank.weight", &TensorSelection::Full)
                    .unwrap();
                let address = acquired.address();
                drop(acquisitions);
                drop(g4bank);
                let mut pending_control_bank = host.split(0, 0, Some((pending, 1))).unwrap();
                let receipt = pending_control_bank
                    .take_source_constructions()
                    .unwrap()
                    .try_debit(pending)
                    .unwrap();
                let pending_bank = host
                    .split(
                        destination_bytes,
                        destinations.calls(),
                        Some((copy, copy_attempts)),
                    )
                    .unwrap();
                let ready = fixture.admitted_ready(controls, pending_bank);
                let result =
                    acquired.prepare_fixture(&fixture.context, &fixture.stream, ready, observer);
                drop(receipt);
                match result {
                    Err(CheckpointMaterializationError::PreparedGgufAdmitted(error))
                        if refuse_raw =>
                    {
                        assert_eq!(std::ptr::from_ref(error.lease()), address);
                        assert!(matches!(
                            error.host_destination_cause(),
                            Some(HostDestinationCause::Capacity { required, remaining: 0 })
                                if *required == raw_bytes
                        ));
                        assert_eq!(fixture.source_physical_reads(), reads);
                        (
                            Some(error),
                            None,
                            fixture.context.clone(),
                            pool.clone(),
                            address,
                        )
                    }
                    Ok(pending) if !refuse_raw => {
                        assert_eq!(OriginalGgufMissFixture::address(pending.lease()), address);
                        fixture.settle(&pending, observer);
                        assert_eq!(fixture.source_physical_reads(), reads + 1);
                        assert_eq!(
                            super::super::values(pending.output(), Some(observer)),
                            fixture.expected["bank.weight"]
                        );
                        let array = pending.output().clone();
                        OriginalGgufMissFixture::finish(pending);
                        (
                            None,
                            Some(array),
                            fixture.context.clone(),
                            pool.clone(),
                            address,
                        )
                    }
                    Err(cause) => panic!("unexpected component failure: {cause:?}"),
                    Ok(_) => panic!("raw destination refusal unexpectedly succeeded"),
                }
            },
        );
        drop((root, rows, fixture));
        assert!(pool.fixture_host_charge().unwrap() > existing);
        if let Some(error) = error {
            assert_eq!(std::ptr::from_ref(error.lease()), address);
            drop(error);
        }
        drop(array);
        crate::backend::ordinary_retirement::reclaim_all();
        let retired = {
            let mut rows = context.converted_groups.lock().unwrap();
            rows.extract_if(|_, row| row.stale())
        };
        drop((retired, context));
        settle_pool_at(&pool, existing);
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
