#[test]
fn default_event_obeys_c_ownership_and_status_contract() {
    unsafe {
        let event = safemlx_sys::mlx_event_new();
        assert!(!event.ctx.is_null());

        let mut complete = false;
        assert_eq!(safemlx_sys::mlx_event_query(&mut complete, event), 0);
        assert!(complete);
        assert_eq!(safemlx_sys::mlx_event_synchronize(event), 0);
        assert_eq!(safemlx_sys::mlx_event_synchronize(event), 0);

        let mut has_device = true;
        assert_eq!(safemlx_sys::mlx_event_has_device(&mut has_device, event), 0);
        assert!(!has_device);

        let mut backend = u32::MAX;
        assert_eq!(safemlx_sys::mlx_event_get_backend(&mut backend, event), 0);
        assert_eq!(
            backend,
            safemlx_sys::mlx_event_backend__MLX_EVENT_BACKEND_NONE
        );
        assert_eq!(safemlx_sys::mlx_event_free(event), 0);
    }
}
