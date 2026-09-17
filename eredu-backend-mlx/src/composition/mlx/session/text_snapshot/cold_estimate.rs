//! Pre-grant logical facts from immutable native descriptors and source owners.
use super::*;

pub(super) fn inspect(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    sampling: &MlxTextSamplingState,
    pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
) -> Option<[SnapshotEstimate; 3]> {
    let decoder = runtime
        .session()
        .original_snapshot_estimate(runtime.backend())?;
    let sampler = sampling_storage(sampling, array_bytes)?;
    let pending = match pending {
        None => SnapshotEstimate {
            retained_bytes: 0,
            copy_bytes: 0,
        },
        Some(PendingTextInput::Decode(token)) => {
            let descriptor = token.value.try_descriptor().ok()?;
            // Match the ordinary scalar-size guard without a native getter.
            let elements = descriptor.shape().iter().try_fold(1usize, |n, dimension| {
                n.checked_mul(usize::try_from(*dimension).ok()?)
            })?;
            if elements != 1 {
                return None;
            }
            let bytes = logical_bytes(descriptor.facts())?;
            SnapshotEstimate {
                retained_bytes: bytes,
                copy_bytes: bytes,
            }
        }
        Some(PendingTextInput::Prefill(prompt)) => {
            let source = model_session::saved_array_copy::pending_input::PromptCopySource::prepare(
                sampling, prompt,
            )
            .ok()?;
            pending_input::PromptCopyPlan::original_plain_estimate(
                prompt,
                source.positions(),
                array_bytes,
            )?
        }
    };
    Some([decoder, sampler, pending])
}

fn array_bytes(array: &Array) -> Option<u64> {
    let descriptor = array.try_descriptor().ok()?;
    logical_bytes(descriptor.facts())
}
fn logical_bytes(facts: safemlx::ArrayDescriptorFacts) -> Option<u64> {
    // These are the existing logical SnapshotBudget allowances. The separate
    // saved-components preparation/copy fit must still qualify every physical
    // constructor, arena and completion owner before original activation.
    crate::backend::runtime::cache::state::snapshot_estimate::array_bytes(
        facts.logical_bytes(),
        facts.rank(),
    )
}

pub(super) fn resume(
    source: model_session::saved_array_copy::decoder::LogicalResumeSource<'_>,
) -> Option<SnapshotEstimate> {
    let mut decoder =
        eredu_runtime::replicated_session::ReplicatedTextControlState::logical_snapshot_estimate(
            source.decoder.logical_snapshot_estimate()?,
            source.control_input,
        )?;
    let native_metadata = NativeMemoryRetention::singleton_metadata_bytes()
        .checked_add(std::mem::size_of::<MlxNativeTextState>() as u64)?;
    decoder.retained_bytes = decoder.retained_bytes.checked_add(native_metadata)?;
    decoder.copy_bytes = decoder.copy_bytes.checked_add(native_metadata)?;
    let sampler = sampling_storage_parts(
        source.history_bytes,
        std::mem::size_of::<eredu_runtime::working_memory::InferenceRequest>() as u64,
        0,
        source.key,
        array_bytes,
    )?;
    // The shared pending program returns a packed uint32 [1,N] matrix for both
    // scalar decode and saved prefill. Use that destination's logical geometry,
    // never the source's scalar rank or a physical copy-account byte total.
    let pending = match (source.pending, source.media) {
        (Some(pending), None) => {
            let (pending_bytes, pending_rank) = pending.numerical().logical_output();
            pending_input::PromptCopyPlan::one_part_estimate(
                crate::backend::runtime::cache::state::snapshot_estimate::array_bytes(
                    pending_bytes,
                    pending_rank,
                )?,
                pending.logical_metadata_bytes()?,
            )?
        }
        (None, Some(media)) => {
            pending_input::PromptCopyPlan::completed_media_estimate(media, array_bytes)?
        }
        _ => return None,
    };
    [sampler, pending]
        .into_iter()
        .try_fold(decoder, |total, part| {
            Some(SnapshotEstimate {
                retained_bytes: total.retained_bytes.checked_add(part.retained_bytes)?,
                copy_bytes: total.copy_bytes.checked_add(part.copy_bytes)?,
            })
        })
}
