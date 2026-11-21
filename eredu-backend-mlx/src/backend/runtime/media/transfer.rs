//! Native prepared-input copies with one retained host staging buffer.

use super::input::MlxInputInspector;
use crate::backend::error::Error;
use safemlx::{Array, Stream};

impl eredu_core::input::InputTensorTransfer for MlxInputInspector {
    type Tensor = Array;
    type Context = Stream;
    type Error = Error;

    fn host_staging_capacity(&self, payload_bytes: u64) -> Result<u64, Error> {
        let bytes =
            usize::try_from(payload_bytes).map_err(eredu_core::BackendFailure::from_error)?;
        safemlx::host_transfer_capacity_upper_bound(bytes, safemlx::HostTransferPolicy::Transfer)
            .map(|bytes| bytes as u64)
            .map_err(Into::into)
    }

    fn copy_input_tensors(
        &self,
        values: Vec<&Array>,
        stream: &Stream,
        staging_capacity_bytes: u64,
    ) -> Result<Vec<Array>, Error> {
        let recovery = Recovery::begin(InputTransferRetention {
            arrays: values.into_iter().cloned().collect(),
            staging: None,
        })?;
        copy_with_recovery(stream, staging_capacity_bytes, recovery)
    }
}
use crate::backend::submission_recovery::{Recovery, Retention, Status};

// Source roots, every destination and current staging remain owned across
// errors and unwind. Staging retires only after both transfer events complete.
struct InputTransferRetention {
    arrays: Vec<Array>,
    staging: Option<safemlx::ImmutableHostTransferBuffer>,
}
impl Retention for InputTransferRetention {
    fn observe(&self, _: Status) {}
}

fn copy_with_recovery(
    stream: &safemlx::Stream,
    staging_capacity_bytes: u64,
    mut recovery: Recovery<InputTransferRetention>,
) -> Result<Vec<Array>, Error> {
    let count = recovery.retention().arrays.len();
    for index in 0..count {
        let (staging, downloaded) = safemlx::HostTransferBuffer::copy_from_array(
            &recovery.retention().arrays[index],
            safemlx::HostTransferPolicy::Transfer,
            stream,
        )?
        .into_parts();
        recovery.retention_mut().staging = Some(staging.freeze());
        let actual = recovery
            .retention()
            .staging
            .as_ref()
            .expect("registered staging")
            .capacity()? as u64;
        if actual > staging_capacity_bytes {
            return Err(Error::ArchitectureModel(
                "native transfer staging exceeded its admitted capacity bound".into(),
            ));
        }
        downloaded.synchronize()?;
        let (copied, uploaded) = recovery
            .retention()
            .staging
            .as_ref()
            .expect("registered staging")
            .copy_to_array(stream)?
            .into_parts();
        recovery.retention_mut().arrays.push(copied);
        uploaded.synchronize()?;
        // Both events settled; no later copy can overlap this staging.
        recovery.retention_mut().staging = None;
    }
    let arrays = recovery.retention().arrays[count..].to_vec();
    recovery.seal();
    let status = recovery.finish();
    if status.failed || status.blocked {
        return Err(Error::ArchitectureModel(
            "native prepared-input transfer failed or is unobservable".into(),
        ));
    }
    Ok(arrays)
}
