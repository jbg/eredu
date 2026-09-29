//! Isolated physical host-staging conformance; counters are process-wide.
use eredu_backend_mlx::backend::{
    error::Error,
    runtime::media::{
        input::{input_part, InputPayload, ModelInput},
        PreparedModelInput,
    },
};
use eredu_core::{InputMetadataKey, InputModality};
use eredu_runtime::input_transfer::{InputTransferBudget, InputTransferError};
use safemlx::{ops::indexing::TryIndexOp, Array, Device, DeviceType, Stream};

#[test]
fn serial_prepared_input_transfer_bounds_physical_staging_with_views_and_aliases() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let values: Vec<f32> = (0..128 * 256)
        .map(|n| (n as f32 - 12000.0) / 17.0)
        .collect();
    let backing = Array::from_slice(&values, &[128, 256]);
    let view = backing
        .try_index_device((0..32, ..), &stream)
        .unwrap()
        .transpose(&stream)
        .unwrap();
    let grid = Array::from_slice(&[1_i32, 16, 16], &[1, 3]);
    let tokens = Array::from_slice(&[16_777_217_u32, u32::MAX], &[1, 2]);
    safemlx::transforms::eval([&view, &grid, &tokens]).unwrap();
    let mut parts =
        vec![input_part(InputModality::Text, InputPayload::TokenIds(tokens), [], []).unwrap()];
    for _ in 0..3 {
        parts.push(
            input_part(
                InputModality::Image,
                InputPayload::Tensor(view.clone()),
                [(InputMetadataKey::PatchGrid, grid.clone())],
                [],
            )
            .unwrap(),
        );
    }
    let source = PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap();
    let probe = safemlx::HostTransferBuffer::new(
        &[1],
        safemlx::Dtype::Uint8,
        safemlx::HostTransferPolicy::Transfer,
    )
    .unwrap();
    let kind = probe.storage_kind().unwrap();
    drop(probe);
    safemlx::reset_host_transfer_peak_memory(kind).unwrap();
    let baseline = safemlx::host_transfer_memory_stats(kind).unwrap();
    let resources = source.transfer_resources().unwrap();
    assert_eq!(resources.tensor_count, 7);
    assert_eq!(resources.payload_bytes, 8 + 3 * (32 * 256 * 4 + 12));
    assert_eq!(
        resources.staging_capacity_bytes,
        safemlx::host_transfer_capacity_upper_bound(
            32 * 256 * 4,
            safemlx::HostTransferPolicy::Transfer
        )
        .unwrap() as u64
    );
    assert_eq!(safemlx::host_transfer_memory_stats(kind).unwrap(), baseline);
    let denied = InputTransferBudget {
        staging_capacity_bytes: resources.staging_capacity_bytes - 1,
        ..resources.into_budget()
    };
    assert!(matches!(
        source.transfer_to_stream(&stream, denied),
        Err(Error::InputTransfer(InputTransferError::Exhausted {
            resource: "staging capacity bytes",
            ..
        }))
    ));
    assert_eq!(safemlx::host_transfer_memory_stats(kind).unwrap(), baseline);
    let start = std::time::Instant::now();
    let mut transfer_elapsed = std::time::Duration::ZERO;
    for _ in 0..16 {
        let transfer_start = std::time::Instant::now();
        let (copied, report) = source
            .transfer_to_stream(&stream, resources.into_budget())
            .unwrap();
        transfer_elapsed += transfer_start.elapsed();
        assert_eq!(report, resources);
        assert_eq!(copied.identity(), source.identity());
        let arrays = copied.wire_arrays();
        assert_eq!(
            arrays[0].evaluated().unwrap().as_slice::<u32>(),
            &[16_777_217, u32::MAX]
        );
        for image in 0..3 {
            let actual = arrays[1 + image * 2].evaluated().unwrap();
            for column in 0..256 {
                for row in 0..32 {
                    assert_eq!(
                        actual.as_slice::<f32>()[column * 32 + row],
                        values[row * 256 + column]
                    );
                }
            }
            assert_eq!(
                arrays[2 + image * 2].evaluated().unwrap().as_slice::<i32>(),
                &[1, 16, 16]
            );
        }
        let stats = safemlx::host_transfer_memory_stats(kind).unwrap();
        assert_eq!(
            stats.active_bytes, baseline.active_bytes,
            "completed destinations must not retain staging"
        );
        assert_eq!(stats.active_allocations, baseline.active_allocations);
        assert!(
            stats.peak_bytes - baseline.active_bytes <= resources.staging_capacity_bytes as usize
        );
        assert!(stats.peak_allocations - baseline.active_allocations <= 1);
    }
    let stats = safemlx::host_transfer_memory_stats(kind).unwrap();
    eprintln!(
        "prepared transfer: copies=16 wire_bytes={} staging_bound={} physical_peak={} transfer_elapsed={:?} elapsed={:?}",
        resources.payload_bytes,
        resources.staging_capacity_bytes,
        stats.peak_bytes - baseline.active_bytes,
        transfer_elapsed,
        start.elapsed()
    );
}
