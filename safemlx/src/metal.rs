//! Metal backend configuration.

use crate::error::{self, Exception, Result};
use std::{ffi::CString, path::Path};

/// Path to the uncompressed `mlx.metallib` exported for this build.
///
/// The compressed library is embedded automatically, so normal applications
/// do not need this build-machine path. It is retained for diagnostics and
/// explicit custom-library overrides.
pub const BUILT_METALLIB_PATH: Option<&str> = option_env!("SAFEMLX_METALLIB_PATH");

fn check_status(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(error::get_and_clear_last_mlx_error()
            .expect("MLX Metal operation failed but no error was set")
            .into())
    }
}

/// Returns whether the MLX Metal backend is available on the current device.
pub fn is_available() -> Result<bool> {
    error::ensure_mlx_error_handler();
    let mut available = false;
    check_status(unsafe { safemlx_sys::mlx_metal_is_available(&mut available) })?;
    Ok(available)
}

/// Overrides the path from which MLX loads its default Metal library.
///
/// Call this before creating arrays or performing any other MLX operation. The
/// explicit path takes precedence over the library embedded by `safemlx`.
pub fn set_metallib_path(path: impl AsRef<Path>) -> Result<()> {
    error::ensure_mlx_error_handler();
    let path = path
        .as_ref()
        .to_str()
        .ok_or_else(|| Exception::custom("metallib path is not valid UTF-8"))?;
    let path = CString::new(path).map_err(|error| Exception::custom(error.to_string()))?;
    check_status(unsafe { safemlx_sys::mlx_metal_set_metallib_path(path.as_ptr()) })
}

static CAPTURE_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Process-wide Metal capture to a `.gputrace` destination. This is diagnostic
/// instrumentation, not an ordinary timing mode. Set `MTL_CAPTURE_ENABLED=1`
/// before starting the process. Settle pre-existing work before beginning and
/// complete captured work before finishing; this guard does not synchronize streams.
/// The guard is thread-affine and prevents overlapping captures through this API.
#[derive(Debug)]
pub struct Capture {
    active: bool,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Capture {
    /// Begin export; rejects invalid paths and capture nesting before calling Metal.
    pub fn begin(path: impl AsRef<Path>) -> Result<Self> {
        use std::sync::atomic::Ordering;
        let path = path.as_ref();
        if path.exists() {
            return Err(Exception::custom(
                "Metal capture destination already exists",
            ));
        }
        let path = path
            .to_str()
            .ok_or_else(|| Exception::custom("Metal capture path is not valid UTF-8"))?;
        let path = CString::new(path).map_err(|error| Exception::custom(error.to_string()))?;
        CAPTURE_ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Exception::custom("Metal capture is already active or cleanup failed"))?;
        let _guard = crate::utils::runtime_lock::enter();
        error::ensure_mlx_error_handler();
        // SAFETY: the NUL-terminated path remains alive throughout this synchronous call.
        if let Err(error) =
            check_status(unsafe { safemlx_sys::mlx_metal_start_capture(path.as_ptr()) })
        {
            CAPTURE_ACTIVE.store(false, Ordering::Release);
            return Err(error);
        }
        Ok(Self {
            active: true,
            _thread: std::marker::PhantomData,
        })
    }
    /// Stop export explicitly so errors are observable. Drop also attempts cleanup.
    pub fn finish(mut self) -> Result<()> {
        self.stop()
    }
    fn stop(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let _guard = crate::utils::runtime_lock::enter();
        error::ensure_mlx_error_handler();
        // SAFETY: this guard exclusively owns the capture through this wrapper.
        check_status(unsafe { safemlx_sys::mlx_metal_stop_capture() })?;
        self.active = false;
        CAPTURE_ACTIVE.store(false, std::sync::atomic::Ordering::Release);
        Ok(())
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
