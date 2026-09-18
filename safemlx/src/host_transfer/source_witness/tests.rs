use super::*;
use crate::{
    Device, DeviceType, Dtype, HostTransferBuffer, HostTransferPolicy, HostTransferStorageKind,
    PreparedHostTransferPlan, PreparedInputArena, PreparedInputRuntime,
    PreparedSubmissionGraphQuota, Stream, host_transfer_memory_stats,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Probe(Arc<AtomicUsize>);
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn immutable_host_witness_preserves_constructor_provenance_and_shared_custody() {
    const CHILD: &str = "SAFEMLX_IMMUTABLE_HOST_WITNESS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                concat!(
                    "host_transfer::source_witness::tests::",
                    "immutable_host_witness_preserves_constructor_provenance_and_shared_custody"
                ),
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed; 0 failed;"),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let baseline = host_transfer_memory_stats(HostTransferStorageKind::MetalShared).unwrap();
    let values = [1.25f32, -2.5, 7.0, 13.0];
    for prepared in [false, true] {
        for shared in [false, true] {
            let mut host = if prepared {
                let plan = PreparedHostTransferPlan::new(
                    &runtime,
                    &[4],
                    Dtype::Float32,
                    usize::from(shared),
                )
                .unwrap();
                let quota =
                    PreparedSubmissionGraphQuota::try_new(plan.metadata_bytes(), ()).unwrap();
                let arena = PreparedInputArena::try_allocate(quota).unwrap();
                plan.construct(&arena).unwrap()
            } else {
                HostTransferBuffer::new(&[4], Dtype::Float32, HostTransferPolicy::Transfer).unwrap()
            };
            for (bytes, value) in host.as_bytes_mut().unwrap().chunks_exact_mut(4).zip(values) {
                bytes.copy_from_slice(&value.to_ne_bytes());
            }
            let host = host.freeze();
            let before = host_transfer_memory_stats(HostTransferStorageKind::MetalShared).unwrap();
            let witness = host.inspect_original_source().unwrap();
            assert_eq!(witness.is_prepared_source(), prepared);
            let allocation = witness.allocation();
            assert_eq!(allocation, host.allocation_info().unwrap());
            let drops = Arc::new(AtomicUsize::new(0));
            let (owner, mut retirement) =
                PreparedAllocationOwner::try_new_with_retirement(Probe(drops.clone())).unwrap();
            witness.try_attach(owner).unwrap();
            let alias = shared.then(|| {
                let alias = if prepared {
                    host.try_prepared_source_array().unwrap()
                } else {
                    host.copy_to_array(&stream).unwrap().synchronize().unwrap()
                };
                alias.evaluated().unwrap();
                let observed = alias.inspect_host_transfer_alias().unwrap().unwrap();
                assert_eq!(observed.is_prepared_source(), prepared);
                assert_eq!(observed.allocation(), allocation);
                assert_eq!(
                    alias.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
                    &values
                );
                alias
            });
            let after = host_transfer_memory_stats(HostTransferStorageKind::MetalShared).unwrap();
            assert_eq!(after.active_bytes, before.active_bytes);
            assert_eq!(after.active_allocations, before.active_allocations);
            assert_eq!(
                host.inspect_original_source().unwrap().allocation(),
                allocation
            );
            assert!(!retirement.try_reclaim());
            drop(host);
            if shared {
                assert!(
                    !retirement.try_reclaim(),
                    "shared native backing keeps its exact owner"
                );
                assert_eq!(drops.load(Ordering::SeqCst), 0);
                assert_eq!(
                    alias
                        .as_ref()
                        .unwrap()
                        .evaluated()
                        .unwrap()
                        .try_as_slice::<f32>()
                        .unwrap(),
                    &values
                );
            }
            drop(alias);
            stream.synchronize().unwrap();
            retirement.try_reclaim();
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            crate::reclaim_allocation_owners();
            let final_stats =
                host_transfer_memory_stats(HostTransferStorageKind::MetalShared).unwrap();
            assert_eq!(final_stats.active_bytes, baseline.active_bytes);
            assert_eq!(final_stats.active_allocations, baseline.active_allocations);
        }
    }
}

#[test]
fn immutable_host_witness_rejects_changed_facts_without_consuming_owner() {
    let host = HostTransferBuffer::new(&[4], Dtype::Float32, HostTransferPolicy::Transfer)
        .unwrap()
        .freeze();
    let witness = host.inspect_original_source().unwrap();
    let original = witness.facts;
    for field in 0..3 {
        let mut changed = original;
        match field {
            0 => changed.backing.identity = changed.backing.identity.checked_add(1).unwrap(),
            1 => {
                changed.backing.charged_bytes =
                    changed.backing.charged_bytes.checked_add(1).unwrap()
            }
            _ => changed.prepared_source = !changed.prepared_source,
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = PreparedAllocationOwner::try_new(Probe(drops.clone())).unwrap();
        let error = owner
            .attach_immutable_host(host.buffer.raw, changed)
            .unwrap_err();
        let (cause, owner) = error.into_parts();
        assert_eq!(cause, OriginalBufferCause::BirthChanged);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(owner);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(
            host.inspect_original_source().unwrap().allocation(),
            witness.allocation()
        );
    }
}
