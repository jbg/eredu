use std::ffi::CStr;

use crate::{
    device::Device,
    error::Result,
    utils::{guard::Guarded, runtime_lock, SUCCESS},
};

/// A stream of evaluation attached to a particular device.
///
/// Typically, this is used via the `stream:` parameter on MLX operations.
pub struct Stream {
    pub(crate) c_stream: safemlx_sys::mlx_stream,
}

// SAFETY: the owned MLX stream handle may move between threads. Every safemlx
// operation that touches MLX runtime-global state enters the runtime guard.
unsafe impl Send for Stream {}

impl AsRef<Stream> for Stream {
    fn as_ref(&self) -> &Stream {
        self
    }
}

impl Clone for Stream {
    fn clone(&self) -> Self {
        let _guard = runtime_lock::enter();
        Stream::try_from_op(|res| unsafe { safemlx_sys::mlx_stream_set(res, self.c_stream) })
            .expect("Failed to clone stream")
    }
}

impl Stream {
    /// Borrows MLX's default CPU execution stream through an owned handle.
    /// MLX retains one default per host thread; acquiring another handle must
    /// not register another native execution stream for every synchronous copy.
    pub(crate) fn try_default_cpu() -> Result<Stream> {
        crate::error::ensure_mlx_error_handler();
        let _guard = runtime_lock::enter();
        let c_stream = unsafe { safemlx_sys::mlx_default_cpu_stream_new() };
        if c_stream.ctx.is_null() {
            return Err(crate::error::get_and_clear_last_mlx_error()
                .expect("MLX default CPU stream initialization failed but no error was set")
                .into());
        }
        Ok(Stream { c_stream })
    }

    /// Tries to create a new stream on the given device.
    #[track_caller]
    pub fn try_new_with_device(device: &Device) -> Result<Stream> {
        crate::error::ensure_mlx_error_handler();
        let _guard = runtime_lock::enter();
        let c_stream = unsafe { safemlx_sys::mlx_stream_new_device(device.c_device) };
        if c_stream.ctx.is_null() {
            return Err(crate::error::get_and_clear_last_mlx_error()
                .expect("MLX stream initialization failed but no error was set")
                .into());
        }
        Ok(Stream { c_stream })
    }

    /// Creates a new stream on the given device.
    ///
    /// # Panics
    ///
    /// Panics with the underlying MLX error when stream initialization fails.
    #[track_caller]
    pub fn new_with_device(device: &Device) -> Stream {
        Self::try_new_with_device(device).expect("Failed to initialize stream")
    }

    /// Get the underlying C pointer.
    pub fn as_ptr(&self) -> safemlx_sys::mlx_stream {
        self.c_stream
    }

    /// Get the index of the stream.
    pub fn get_index(&self) -> Result<i32> {
        i32::try_from_op(|res| unsafe { safemlx_sys::mlx_stream_get_index(res, self.c_stream) })
    }

    /// Synchronize with the stream.
    pub fn synchronize(&self) -> Result<()> {
        let _guard = runtime_lock::enter();
        <() as Guarded>::try_from_op(|_| unsafe { safemlx_sys::mlx_synchronize(self.c_stream) })
    }

    /// Order subsequently submitted work on this stream after `event`.
    ///
    /// This is a backend-ordered dependency and does not block the host. The
    /// event's producer device must equal this stream's device. Since MLX is
    /// lazy, the dependency applies to consumer work submitted after this call;
    /// constructing a graph alone does not submit it.
    pub fn wait_event(&self, event: &crate::Event) -> Result<()> {
        let _guard = runtime_lock::enter();
        <() as Guarded>::try_from_op(|_| unsafe {
            safemlx_sys::mlx_stream_wait_event(self.c_stream, event.c_event)
        })
    }

    /// Concrete parameter/result controls for the fixed native device comparison.
    pub fn device_comparison_control_bytes() -> Option<usize> {
        let native = unsafe { safemlx_sys::mlx_prepared_input_target_controls() };
        [
            std::mem::size_of::<&Self>(),
            std::mem::size_of::<&Device>(),
            std::mem::size_of::<bool>(),
        ]
        .into_iter()
        .try_fold(native, usize::checked_add)
    }

    /// Compares the two existing immutable native device values without creating
    /// a Device, entering housekeeping, or publishing an error. Nulls reject.
    pub fn matches_device(&self, device: &Device) -> bool {
        // SAFETY: both owned wrappers remain borrowed throughout the scalar
        // native comparison; no pointer or mutable/native object escapes.
        unsafe {
            safemlx_sys::mlx_prepared_input_target_matches(self.c_stream, device.c_device) != 0
        }
    }

    /// Read this existing stream's device kind without allocating a Device shell.
    pub fn device_type(&self) -> Result<crate::DeviceType> {
        match crate::StreamCopyPlan::<()>::capture(self) {
            Ok(plan) => Ok(plan.device_type()),
            Err(cause) => match crate::OriginalScopeObserver::try_current()? {
                Some(observer) => Err(observer.error(2)),
                None => Err(crate::error::Exception::from_source(cause)),
            },
        }
    }

    /// Concrete controls of the scalar device query; no Stream or Device owner.
    pub fn device_type_control_bytes() -> Option<usize> {
        let mut native = safemlx_sys::mlx_stream_copy_layout::default();
        // SAFETY: the existing pure query writes this fixed local layout.
        if unsafe { safemlx_sys::mlx_stream_copy_layout_for(&mut native) } != 0 {
            return None;
        }
        [
            std::mem::size_of::<crate::StreamCopyPlan<()>>(),
            std::mem::size_of::<Result<crate::DeviceType>>(),
            std::mem::size_of::<
                std::result::Result<crate::StreamCopyPlan<()>, crate::StreamCopyCause>,
            >(),
            std::mem::size_of::<&Self>(),
            crate::OriginalScopeObserver::control_bytes()?,
        ]
        .into_iter()
        .try_fold(native.controls, usize::checked_add)
    }

    /// Get the device associated with the stream.
    pub fn get_device(&self) -> Result<Device> {
        Device::try_from_op(|res| unsafe { safemlx_sys::mlx_stream_get_device(res, self.c_stream) })
    }

    fn describe(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let _guard = runtime_lock::enter();
        unsafe {
            let mut mlx_str = safemlx_sys::mlx_string_new();
            let result =
                match safemlx_sys::mlx_stream_tostring(&mut mlx_str as *mut _, self.c_stream) {
                    SUCCESS => {
                        let ptr = safemlx_sys::mlx_string_data(mlx_str);
                        let c_str = CStr::from_ptr(ptr);
                        write!(f, "{}", c_str.to_string_lossy())
                    }
                    _ => Err(std::fmt::Error),
                };
            safemlx_sys::mlx_string_free(mlx_str);
            result
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _guard = runtime_lock::enter();
        unsafe { safemlx_sys::mlx_stream_free(self.c_stream) };
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl std::fmt::Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        unsafe { safemlx_sys::mlx_stream_equal(self.c_stream, other.c_stream) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cpu_stream_handles_reuse_native_execution_stream() {
        let initial = Stream::try_default_cpu().unwrap();
        assert_eq!(
            initial.get_device().unwrap().get_type().unwrap(),
            crate::DeviceType::Cpu
        );
        for _ in 0..64 {
            let stream = Stream::try_default_cpu().unwrap();
            assert_eq!(stream.get_index().unwrap(), initial.get_index().unwrap());
            let values = crate::Array::zeros::<f32>(&[3], &stream).unwrap();
            assert_eq!(values.evaluated().unwrap().as_slice::<f32>(), &[0.0; 3]);
        }
    }

    #[test]
    fn test_stream_clone() {
        let stream = crate::test_stream().clone();
        let cloned_stream = stream.clone();
        assert_eq!(stream, cloned_stream);
    }

    #[test]
    #[cfg_attr(
        not(any(feature = "metal", feature = "cuda")),
        ignore = "requires a GPU backend"
    )]
    fn test_cpu_gpu_stream_not_equal() {
        let cpu_stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
        let gpu_stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0));

        // Assert that CPU and GPU streams are not equal
        assert_ne!(cpu_stream, gpu_stream);
    }

    #[test]
    fn cpu_stream_creation_is_concurrent_safe() {
        std::thread::scope(|scope| {
            for _ in 0..crate::test_concurrency() {
                scope.spawn(|| {
                    for _ in 0..64 {
                        let stream =
                            Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
                        let x = crate::Array::zeros::<f32>(&[1], &stream).unwrap();
                        x.evaluated().unwrap();
                    }
                });
            }
        });
    }

    #[test]
    fn streams_can_move_between_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<Stream>();
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    #[test]
    fn gpu_stream_initialization_returns_the_original_error() {
        let error = Stream::try_new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0))
            .unwrap_err();
        assert!(error
            .what()
            .contains("Cannot make gpu stream without gpu backend"));
    }
}
