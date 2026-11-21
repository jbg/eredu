//! Process-lifetime transfer queues, matching MLX's process-lifetime registry.

use super::*;

#[derive(Default)]
struct TransferStreams(Vec<(CacheTransferDevice, CacheTransferLane, Stream)>);

impl TransferStreams {
    fn get_or_create(
        &mut self,
        device: CacheTransferDevice,
        role: CacheTransferLane,
        create: impl FnOnce() -> Result<Stream, BackendFailure>,
    ) -> Result<Stream, BackendFailure> {
        if let Some((_, _, stream)) = self
            .0
            .iter()
            .find(|(key, lane, _)| *key == device && *lane == role)
        {
            return Ok(stream.clone());
        }
        let stream = create()?;
        self.0.push((device, role, stream.clone()));
        Ok(stream)
    }
}

pub(super) fn cache_transfer_stream(
    device: CacheTransferDevice,
    role: CacheTransferLane,
) -> Result<Stream, BackendFailure> {
    static STREAMS: Mutex<TransferStreams> = Mutex::new(TransferStreams(Vec::new()));
    // Serialize lookup and creation, including concurrent model loads. Release
    // this lock before submitting any work or waiting for a completion.
    STREAMS
        .lock()
        .map_err(|_| {
            BackendFailure::from_error(std::io::Error::other(
                "cache transfer stream registry poisoned",
            ))
        })?
        .get_or_create(device, role, || {
            Stream::try_new_with_device(&Device::new(device.device_type, device.index))
                .map_err(BackendFailure::from_error)
        })
}

// Streams are shared ordered submission lanes, not reusable storage. Each
// transfer retains its own arrays, event and budget reservation through the
// existing cache lifecycle. Reuse neither synchronizes the host nor retires a
// failed transfer. Keep promotion and demotion on separate lanes: promotion may
// wait for a demotion while its own manager is under device-budget pressure.
