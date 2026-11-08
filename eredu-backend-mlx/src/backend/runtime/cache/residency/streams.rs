//! Process-lifetime transfer queues, matching MLX's process-lifetime registry.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CacheTransferRole {
    Promotion,
    Demotion,
    Persistence,
}

#[derive(Default)]
struct TransferStreams(Vec<(CacheTransferDevice, CacheTransferRole, Stream)>);

impl TransferStreams {
    fn get_or_create(
        &mut self,
        device: CacheTransferDevice,
        role: CacheTransferRole,
        create: impl FnOnce() -> Result<Stream, CacheResidencyError>,
    ) -> Result<Stream, CacheResidencyError> {
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
    role: CacheTransferRole,
) -> Result<Stream, CacheResidencyError> {
    static STREAMS: Mutex<TransferStreams> = Mutex::new(TransferStreams(Vec::new()));
    // Serialize lookup and creation, including concurrent model loads. Release
    // this lock before submitting any work or waiting for a completion.
    STREAMS
        .lock()
        .map_err(|_| {
            CacheResidencyError::Runtime("cache transfer stream registry poisoned".into())
        })?
        .get_or_create(device, role, || {
            Stream::try_new_with_device(&Device::new(device.device_type, device.index))
                .map_err(|source| transfer_error("create cache transfer stream", source))
        })
}

// Streams are shared ordered submission lanes, not reusable storage. Each
// transfer retains its own arrays, event and budget reservation through the
// existing cache lifecycle. Reuse neither synchronizes the host nor retires a
// failed transfer. Keep promotion and demotion on separate lanes: promotion may
// wait for a demotion while its own manager is under device-budget pressure.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_queue_creation_failure_preserves_existing_lanes_and_allows_retry() {
        let mut registry = TransferStreams::default();
        let device = CacheTransferDevice::CPU;
        let create = || {
            Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0))
                .map_err(|source| transfer_error("create cache transfer stream", source))
        };
        let promotion = registry
            .get_or_create(device, CacheTransferRole::Promotion, create)
            .unwrap();
        let error = registry
            .get_or_create(device, CacheTransferRole::Demotion, || {
                Err(transfer_error(
                    "create cache transfer stream",
                    safemlx::error::Exception::custom("injected queue exhaustion"),
                ))
            })
            .unwrap_err();
        assert!(error.to_string().contains("injected queue exhaustion"));
        let demotion = registry
            .get_or_create(device, CacheTransferRole::Demotion, create)
            .unwrap();
        for _ in 0..1024 {
            for (role, expected) in [
                (CacheTransferRole::Promotion, &promotion),
                (CacheTransferRole::Demotion, &demotion),
            ] {
                let stream = registry
                    .get_or_create(device, role, || {
                        panic!("created another queue for an existing lane")
                    })
                    .unwrap();
                assert_eq!(stream.get_index().unwrap(), expected.get_index().unwrap());
            }
        }
    }
}
