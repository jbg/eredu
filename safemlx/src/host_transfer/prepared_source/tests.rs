use super::*;
use crate::{PreparedAllocationOwner, PreparedSubmissionGraphQuota};
use std::sync::Arc;

#[test]
fn prepared_host_page_boundaries_preserve_actual_storage_and_escaped_custody() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let page = runtime.raw().page_size;
    let (kind, header) = match runtime.raw().storage_kind {
        CPU_STORAGE => (HostTransferStorageKind::Cpu, size_of::<usize>()),
        METAL_STORAGE => (HostTransferStorageKind::MetalShared, 0),
        other => panic!("unsupported prepared host allocator {other}"),
    };
    for (bytes, capacity) in [
        (1, page),
        (page - header, page),
        (page - header + 1, 2 * page),
        (2 * page - header, 2 * page),
    ] {
        let shape = [i32::try_from(bytes).unwrap()];
        let plan = PreparedHostTransferPlan::new(&runtime, &shape, Dtype::Uint8, 1).unwrap();
        assert_eq!(plan.logical_bytes(), bytes);
        assert_eq!(plan.backing_bytes(), capacity);
        let quota = PreparedSubmissionGraphQuota::try_new(plan.metadata_bytes(), ()).unwrap();
        let arena = PreparedInputArena::try_allocate(quota).unwrap();
        let mut buffer = plan.construct(&arena).unwrap();
        // Releasing the arena handle cannot release the source's own metadata.
        drop(arena);
        let expected: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
        buffer.as_bytes_mut().unwrap().copy_from_slice(&expected);
        let host = buffer.freeze();
        let metadata = host.prepared_metadata().unwrap();
        assert_eq!(metadata.storage_kind(), kind);
        assert_eq!(host.storage_kind().unwrap(), kind);
        assert_eq!(metadata.nbytes(), bytes);
        assert_eq!(metadata.allocation().bytes(), capacity);
        assert_eq!(metadata.allocation(), host.allocation_info().unwrap());
        let witness = host.inspect_original_source().unwrap();
        assert!(witness.is_prepared_source());
        assert_eq!(witness.allocation(), metadata.allocation());
        let custody = Arc::new(());
        let weak = Arc::downgrade(&custody);
        let (owner, mut retirement) =
            PreparedAllocationOwner::try_new_with_retirement(custody).unwrap();
        witness.try_attach(owner).unwrap();
        let alias = host.try_prepared_source_array().unwrap();
        assert!(host.try_prepared_source_array().is_err());
        let alias_witness = alias.inspect_host_transfer_alias().unwrap().unwrap();
        assert!(alias_witness.is_prepared_source());
        assert_eq!(alias_witness.allocation(), metadata.allocation());
        drop(host);
        assert!(!retirement.try_reclaim());
        assert!(weak.upgrade().is_some());
        assert_eq!(
            alias.evaluated().unwrap().try_as_slice::<u8>().unwrap(),
            expected
        );
        drop(alias);
        assert!(retirement.try_reclaim());
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn prepared_host_layout_refusals_leave_output_unchanged() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let original = runtime.raw();
    for case in 0..8 {
        let mut facts = original;
        let mut shape = [1_i32; 10];
        let rank = match case {
            0 => {
                facts.page_size = 0;
                1
            }
            1 => {
                facts.page_size = 3;
                1
            }
            2 => {
                facts.storage_kind = safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CUDA_PINNED;
                1
            }
            3 => {
                facts.maximum = 0;
                1
            }
            4 => {
                shape[0] = 0;
                1
            }
            5 => {
                shape[0] = -1;
                1
            }
            6 => {
                shape.fill(i32::MAX);
                10
            }
            _ => 0,
        };
        let mut layout = safemlx_sys::mlx_prepared_host_transfer_layout {
            metadata_bytes: 11,
            backing_bytes: 13,
            logical_bytes: 17,
            controls: 19,
        };
        // Query only: synthetic facts are never passed to an allocator.
        let status = unsafe {
            safemlx_sys::mlx_prepared_host_transfer_layout_for(
                &mut layout,
                facts,
                shape.as_ptr(),
                rank,
                Dtype::Uint8.into(),
                0,
            )
        };
        assert_ne!(status, 0, "case {case}");
        assert_eq!(
            (
                layout.metadata_bytes,
                layout.backing_bytes,
                layout.logical_bytes,
                layout.controls
            ),
            (11, 13, 17, 19)
        );
    }
    let plan = PreparedHostTransferPlan::new(&runtime, &[7], Dtype::Uint8, 0).unwrap();
    let short = PreparedSubmissionGraphQuota::try_new(plan.metadata_bytes() / 2, ()).unwrap();
    let arena = PreparedInputArena::try_allocate(short).unwrap();
    assert!(plan.construct(&arena).is_err());
}
