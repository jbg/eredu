//! Explicit execution ownership for MLX backend operations.

use safemlx::{Device, DeviceType, Stream};

use super::{error::Error, MlxAcceleratorFamily};

/// Device and stream selected for a backend execution domain.
#[derive(Debug)]
pub struct ExecutionContext {
    device: Device,
    stream: Stream,
}

impl ExecutionContext {
    /// Creates a CPU execution domain without requiring native device construction by callers.
    pub fn cpu() -> Result<Self, Error> {
        Self::try_new(Device::new(DeviceType::Cpu, 0))
    }

    /// Creates an execution domain on the compiled accelerator at a local index.
    /// Rejects unavailable accelerator support before creating a stream.
    pub fn accelerator(index: usize) -> Result<Self, Error> {
        let index = i32::try_from(index)
            .map_err(|_| Error::AutomaticPlanning("device index does not fit in i32".into()))?;
        let family = [MlxAcceleratorFamily::Metal, MlxAcceleratorFamily::Cuda]
            .into_iter()
            .find(|family| family.is_compiled())
            .ok_or_else(|| Error::AutomaticPlanning("no MLX accelerator is compiled".into()))?;
        if !family.is_available()? {
            return Err(Error::AutomaticPlanning(format!(
                "MLX {} accelerator is unavailable",
                family.as_str()
            )));
        }
        Self::try_new(Device::new(DeviceType::Gpu, index))
    }

    /// Creates a context and propagates native stream construction failures.
    pub fn try_new(device: Device) -> Result<Self, Error> {
        let stream = Stream::try_new_with_device(&device)?;
        Ok(Self { device, stream })
    }

    /// Creates a context with a new stream on `device`.
    pub fn new(device: Device) -> Self {
        let stream = Stream::new_with_device(&device);
        Self { device, stream }
    }

    /// Returns the selected device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the selected stream.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

impl AsRef<Stream> for ExecutionContext {
    fn as_ref(&self) -> &Stream {
        &self.stream
    }
}
