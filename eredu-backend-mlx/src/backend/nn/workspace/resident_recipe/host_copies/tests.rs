use super::*;
use crate::backend::runtime::residency::manager::WindowPopulation;

fn transfers(attempts: usize) -> HostTransfers {
    let window = WindowPopulation {
        requested: 1,
        units: 2,
        physical_bindings: 3,
        bindings: 4,
        physical_bytes: 96,
        ..Default::default()
    };
    HostTransfers::from_windows(1, 2 * attempts, 4 * attempts, 3, attempts, &[window]).unwrap()
}
fn prepare(device: DeviceType, attempts: usize) -> PreparedSourceCopies {
    let mut layout = CopyLayout::default();
    for dtype in [safemlx::Dtype::Float32, safemlx::Dtype::Float16, safemlx::Dtype::Bfloat16] {
        for rank in 1..=4 { layout.include(rank, dtype, device).unwrap(); }
    }
    PreparedSourceCopies::prepare(layout, transfers(attempts), 0, device).unwrap()
}
#[test]
fn source_copy_recipes_keep_finite_windows_and_price_the_selected_cpu_or_metal_workers() {
    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        let once = prepare(device, 1);
        let twice = prepare(device, 2);
        assert_eq!(once.transfers.copies, 3);
        assert_eq!(twice.transfers.copies, 6);
        assert_eq!(twice.transfers.binding_shells, 8);
        assert_eq!(twice.transfers.bytes, 192);
        assert_eq!(twice.transfers.roots, 3);
        assert_eq!(twice.copies.traversal.limits().arrays, 3);
        assert_eq!(twice.copies.traversal.limits().tape_entries, 2);
        assert_eq!(twice.copies.aggregate_traversal.limits().arrays, 4);
        let dispatch = twice.copies.dispatch;
        let aggregate = twice.copies.aggregate_dispatch;
        match device {
            DeviceType::Cpu => {
                let cpu = dispatch.cpu_model.unwrap();
                let mut expected = CpuPopulation::default();
                expected.copy(OperationEvent::cpu_host_transfer_layout(
                    safemlx::Dtype::Float32, 4, false, false).unwrap(), 1).unwrap();
                assert_eq!(cpu.primitives, 1);
                assert_eq!(cpu.births, 1);
                assert_eq!(cpu.extents, expected.extents);
                assert_eq!(cpu.hidden_leaves, 0);
                assert_eq!(dispatch.cpu_entries, 2);
                assert_eq!(dispatch.gpu_entries, 0);
                assert_eq!(dispatch.kernel_attempts, 0);
                assert_eq!(aggregate.cpu_entries, 1);
                assert_eq!(aggregate.cpu_model.unwrap().primitives, 0);
                assert_eq!(aggregate.cpu_model.unwrap().births, 0);
            }
            DeviceType::Gpu => {
                assert!(dispatch.cpu_model.is_none());
                assert_eq!(dispatch.gpu_entries, 2);
                assert_eq!(dispatch.gpu_births, 1);
                assert_eq!(dispatch.cpu_entries, 0);
                assert!(dispatch.worker_graph_extents > 0);
                assert_eq!(aggregate.gpu_entries, 1);
                assert_eq!(aggregate.gpu_births, 0);
            }
        }
        let mut first = graph_capacity::ResidentGraphStorage::default();
        first.include_source_copies(once.copies, once.transfers, once.copies.dispatch).unwrap();
        let mut second = graph_capacity::ResidentGraphStorage::default();
        second.include_source_copies(twice.copies, twice.transfers, twice.copies.dispatch).unwrap();
        assert_eq!(second.known_constructor_bytes, first.known_constructor_bytes * 2,
            "all actual copies, aggregate completions and waits preserve finite attempts");
        assert!(second.full_capacity.unwrap() > first.full_capacity.unwrap());
    }
}
#[test]
fn source_copy_frontiers_preserve_cpu_population_and_reject_cross_device_or_unknown_rows() {
    let cpu = prepare(DeviceType::Cpu, 1);
    let gpu = prepare(DeviceType::Gpu, 1);
    let mut limits = cpu.copies.traversal.limits();
    limits.arrays += 6;
    limits.input_edges += 3;
    let prior = cpu.copies.dispatch;
    let (expanded, _) = cpu.expand_equation_dispatch(prior, limits, 4).unwrap();
    assert_eq!(expanded.cpu_model.unwrap().extents, prior.cpu_model.unwrap().extents);
    assert_eq!(expanded.cpu_entries, prior.cpu_entries);
    assert_eq!(expanded.cpu_input_edges, prior.cpu_input_edges);
    assert_eq!(expanded.worker_graph_extents, 0);
    assert_eq!(expanded.kernel_attempts, 0);
    assert!(matches!(gpu.expand_equation_dispatch(prior, limits, 4),
        Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))));
    assert!(matches!(cpu.expand_equation_dispatch(gpu.copies.dispatch,
        gpu.copies.traversal.limits(), 4),
        Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))));
    for (rank, dtype) in [(0,safemlx::Dtype::Float32),(5,safemlx::Dtype::Float32),
        (2,safemlx::Dtype::Uint32)] {
        assert!(CopyLayout::default().include(rank,dtype,DeviceType::Cpu).is_none());
    }
}
