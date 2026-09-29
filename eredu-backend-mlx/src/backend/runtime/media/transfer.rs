//! One serial transfer loop shared by native execution and completion-fault tests.

use super::{Array, Error, PreparedModelInput};
use crate::backend::submission_recovery::{Probe, Recovery, Retention, Status};
use eredu_runtime::input_transfer::InputTransferResources;

/// Retained with the resources it waits on. The native implementation has no
/// state; tests substitute completion results without replacing the copy loop.
trait Completion: 'static {
    fn synchronize(&self, event: &safemlx::Event) -> Result<(), Error>;
    fn observe(&self, _: Status, _: usize, _: bool) {}
}

struct NativeCompletion;
impl Completion for NativeCompletion {
    fn synchronize(&self, event: &safemlx::Event) -> Result<(), Error> {
        Ok(event.synchronize()?)
    }
}

// Source roots, every destination and current staging remain owned across
// errors and unwind. Staging retires only after both transfer events complete.
struct InputTransferRetention<C> {
    arrays: Vec<Array>,
    staging: Option<safemlx::ImmutableHostTransferBuffer>,
    completion: C,
}
impl<C: Completion> Retention for InputTransferRetention<C> {
    fn observe(&self, status: Status) {
        // Host metadata only: observing retirement must not submit native work.
        self.completion
            .observe(status, self.arrays.len(), self.staging.is_some());
    }
}

pub(super) fn copy(
    source: &PreparedModelInput,
    stream: &safemlx::Stream,
    resources: InputTransferResources,
) -> Result<(PreparedModelInput, InputTransferResources), Error> {
    let recovery = Recovery::begin(InputTransferRetention {
        arrays: source.wire_arrays(),
        staging: None,
        completion: NativeCompletion,
    })?;
    copy_with_recovery(source, stream, resources, recovery)
}

fn copy_with_recovery<C: Completion, P: Probe>(
    source: &PreparedModelInput,
    stream: &safemlx::Stream,
    resources: InputTransferResources,
    mut recovery: Recovery<InputTransferRetention<C>, P>,
) -> Result<(PreparedModelInput, InputTransferResources), Error> {
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
        if actual > resources.staging_capacity_bytes {
            return Err(Error::ArchitectureModel(
                "native transfer staging exceeded its admitted capacity bound".into(),
            ));
        }
        recovery.retention().completion.synchronize(&downloaded)?;
        let (copied, uploaded) = recovery
            .retention()
            .staging
            .as_ref()
            .expect("registered staging")
            .copy_to_array(stream)?
            .into_parts();
        recovery.retention_mut().arrays.push(copied);
        recovery.retention().completion.synchronize(&uploaded)?;
        // Both events settled; no later copy can overlap this staging.
        recovery.retention_mut().staging = None;
    }
    let arrays = recovery.retention().arrays[count..].to_vec();
    let mut transferred = PreparedModelInput::from_identity_wire_arrays(source.identity(), arrays)?;
    transferred.cache_identity = source.cache_identity.clone();
    recovery.seal();
    let status = recovery.finish();
    if status.failed || status.blocked {
        return Err(Error::ArchitectureModel(
            "native prepared-input transfer failed or is unobservable".into(),
        ));
    }
    Ok((transferred, resources))
}

#[cfg(test)]
mod tests;
