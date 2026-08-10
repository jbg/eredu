#[test]
fn host_transfer_handles_obey_c_ownership_contract() {
    unsafe {
        let device = safemlx_sys::mlx_device_new_type(safemlx_sys::mlx_device_type__MLX_CPU, 0);
        let stream = safemlx_sys::mlx_stream_new_device(device);
        let source_data = [1.0f32, 2.0, 3.0, 4.0];
        let source = safemlx_sys::mlx_array_new_data(
            source_data.as_ptr().cast(),
            [2, 2].as_ptr(),
            2,
            safemlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        let mut buffer = safemlx_sys::mlx_host_transfer_buffer {
            ctx: std::ptr::null_mut(),
        };
        let mut event = safemlx_sys::mlx_event_new();

        assert_eq!(
            safemlx_sys::mlx_copy_to_host(
                &mut buffer,
                &mut event,
                source,
                safemlx_sys::mlx_host_transfer_policy__MLX_HOST_TRANSFER_POLICY_TRANSFER,
                stream,
            ),
            0
        );
        assert!(!buffer.ctx.is_null());
        assert_eq!(safemlx_sys::mlx_event_synchronize(event), 0);

        let mut bytes = 0;
        assert_eq!(
            safemlx_sys::mlx_host_transfer_buffer_nbytes(&mut bytes, buffer),
            0
        );
        assert_eq!(bytes, 4 * size_of::<f32>());
        let mut capacity = 0;
        assert_eq!(
            safemlx_sys::mlx_host_transfer_buffer_capacity(&mut capacity, buffer),
            0
        );
        let mut bound = 0;
        assert_eq!(
            safemlx_sys::mlx_host_transfer_capacity_upper_bound(
                &mut bound,
                bytes,
                safemlx_sys::mlx_host_transfer_policy__MLX_HOST_TRANSFER_POLICY_TRANSFER,
            ),
            0
        );
        assert_eq!(capacity, bound);
        let mut kind = 0;
        assert_eq!(
            safemlx_sys::mlx_host_transfer_buffer_storage_kind(&mut kind, buffer),
            0
        );
        let mut live = safemlx_sys::mlx_host_transfer_memory_stats {
            active_bytes: 0,
            peak_bytes: 0,
            active_allocations: 0,
            peak_allocations: 0,
        };
        assert_eq!(
            safemlx_sys::mlx_host_transfer_memory_stats_get(&mut live, kind),
            0
        );
        assert!(live.active_bytes >= capacity);
        assert!(live.active_allocations >= 1);
        let mut data = std::ptr::null();
        assert_eq!(
            safemlx_sys::mlx_host_transfer_buffer_data(&mut data, buffer),
            0
        );
        assert_eq!(
            std::slice::from_raw_parts(data.cast::<f32>(), 4),
            &source_data
        );

        let mut round_trip = safemlx_sys::mlx_array_new();
        let mut round_trip_event = safemlx_sys::mlx_event_new();
        assert_eq!(
            safemlx_sys::mlx_copy_from_host(&mut round_trip, &mut round_trip_event, buffer, stream,),
            0
        );
        assert_eq!(safemlx_sys::mlx_event_synchronize(round_trip_event), 0);

        assert_eq!(safemlx_sys::mlx_array_free(round_trip), 0);
        assert_eq!(safemlx_sys::mlx_event_free(round_trip_event), 0);
        assert_eq!(safemlx_sys::mlx_host_transfer_buffer_free(buffer), 0);
        let mut released = safemlx_sys::mlx_host_transfer_memory_stats {
            active_bytes: 0,
            peak_bytes: 0,
            active_allocations: 0,
            peak_allocations: 0,
        };
        assert_eq!(
            safemlx_sys::mlx_host_transfer_memory_stats_get(&mut released, kind),
            0
        );
        assert_eq!(released.active_bytes, live.active_bytes - capacity);
        assert_eq!(released.active_allocations, live.active_allocations - 1);
        assert_eq!(safemlx_sys::mlx_event_free(event), 0);
        assert_eq!(safemlx_sys::mlx_array_free(source), 0);
        assert_eq!(safemlx_sys::mlx_stream_free(stream), 0);
        assert_eq!(safemlx_sys::mlx_device_free(device), 0);
    }
}
