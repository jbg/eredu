use std::{ffi::CStr, sync::Mutex};

unsafe extern "C" fn capture_error(message: *const std::ffi::c_char, data: *mut std::ffi::c_void) {
    let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let captured = unsafe { &*(data.cast::<Mutex<String>>()) };
    *captured.lock().unwrap() = message.into_owned();
}

unsafe extern "C" fn drop_capture(data: *mut std::ffi::c_void) {
    drop(unsafe { Box::from_raw(data.cast::<Mutex<String>>()) });
}

#[test]
fn all_to_all_v_retains_inputs_and_reports_validation_errors() {
    unsafe {
        std::env::set_var("DEVICE", "cpu");
        let captured = Box::into_raw(Box::new(Mutex::new(String::new())));
        safemlx_sys::mlx_set_error_handler(
            Some(capture_error),
            captured.cast(),
            Some(drop_capture),
        );

        let cpu = safemlx_sys::mlx_device_new_type(safemlx_sys::mlx_device_type__MLX_CPU, 0);
        assert_eq!(safemlx_sys::mlx_set_default_device(cpu), 0);

        let mut group = safemlx_sys::mlx_distributed_group_new();
        assert_eq!(
            safemlx_sys::mlx_distributed_init(&mut group, false, c"ring".as_ptr()),
            0
        );
        assert_eq!(safemlx_sys::mlx_distributed_group_size(group), 1);
        let stream = safemlx_sys::mlx_default_cpu_stream_new();
        let mut stream_device = safemlx_sys::mlx_device_new();
        assert_eq!(
            safemlx_sys::mlx_stream_get_device(&mut stream_device, stream),
            0
        );
        let mut stream_device_type = u32::MAX;
        assert_eq!(
            safemlx_sys::mlx_device_get_type(&mut stream_device_type, stream_device),
            0
        );
        assert_eq!(stream_device_type, safemlx_sys::mlx_device_type__MLX_CPU);
        assert_eq!(safemlx_sys::mlx_device_free(stream_device), 0);
        let mut input = safemlx_sys::mlx_array_new();
        assert_eq!(
            safemlx_sys::mlx_arange(
                &mut input,
                1.0,
                5.0,
                1.0,
                safemlx_sys::mlx_dtype__MLX_INT32,
                stream,
            ),
            0
        );
        assert!(!input.ctx.is_null(), "{}", (*captured).lock().unwrap());
        let mut output = safemlx_sys::mlx_array_new();
        let status = safemlx_sys::mlx_distributed_all_to_all_v(
            &mut output,
            input,
            [4i64].as_ptr(),
            1,
            [4i64].as_ptr(),
            1,
            group,
            stream,
        );
        assert_eq!(status, 0, "{}", (*captured).lock().unwrap());
        assert_eq!(safemlx_sys::mlx_array_free(input), 0);
        assert_eq!(
            safemlx_sys::mlx_array_eval(output),
            0,
            "{}",
            (*captured).lock().unwrap()
        );
        assert_eq!(safemlx_sys::mlx_array_size(output), 4);
        assert_eq!(
            std::slice::from_raw_parts(safemlx_sys::mlx_array_data_int32(output), 4),
            &[1, 2, 3, 4]
        );
        assert_eq!(safemlx_sys::mlx_array_free(output), 0);

        let scalar = safemlx_sys::mlx_array_new_int(1);
        output = safemlx_sys::mlx_array_new();
        assert_eq!(
            safemlx_sys::mlx_distributed_all_to_all_v(
                &mut output,
                scalar,
                [1i64].as_ptr(),
                1,
                [1i64].as_ptr(),
                1,
                group,
                stream,
            ),
            1
        );
        assert!((*captured)
            .lock()
            .unwrap()
            .contains("[all_to_all_v] Input must have a leading row dimension."));
        assert_eq!(safemlx_sys::mlx_array_free(output), 0);
        assert_eq!(safemlx_sys::mlx_array_free(scalar), 0);
        assert_eq!(safemlx_sys::mlx_stream_free(stream), 0);
        assert_eq!(safemlx_sys::mlx_distributed_group_free(group), 0);
        assert_eq!(safemlx_sys::mlx_device_free(cpu), 0);
        safemlx_sys::mlx_set_error_handler(None, std::ptr::null_mut(), None);
    }
}
