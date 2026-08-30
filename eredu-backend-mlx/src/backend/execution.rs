//! Explicit execution ownership for MLX backend operations.

use safemlx::{Device, Stream};

/// Device and stream selected for a backend execution domain.
#[derive(Debug)]
pub struct ExecutionContext {
    device: Device,
    stream: Stream,
}

impl ExecutionContext {
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
