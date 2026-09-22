//! Borrowed logical sizing for independently copied pending input.
use super::*;
use std::cell::RefCell;

/// Saved continuation growth consumes the exact source-authenticated decoder
/// extent. Raw media descriptors alone never establish this attribution.
pub(super) fn decoder_positions(prompt: &MlxModelInput) -> Option<u64> {
    if prompt.has_original_input_custody() {
        let source = <MlxBackend<'_> as eredu_runtime::input::OriginalModelInputBackend>
            ::original_model_input_semantics(prompt)?;
        return u64::try_from(source.layout().positions())
            .ok()
            .filter(|positions| *positions != 0);
    }
    prompt
        .controlled_decoder_positions()
        .or_else(|| with_text_prompt_array(prompt, |a| u64::try_from(a.shape()[1]).ok()).flatten())
        .filter(|positions| *positions != 0)
}

/// Scalar inventory borrowed from one immutable input; preparation allocates
/// neither source handles nor a second descriptor/slot list.
pub(super) struct PromptCopyPlan<'a> {
    _source: &'a MlxModelInput,
    estimate: SnapshotEstimate,
}
impl<'a> PromptCopyPlan<'a> {
    pub(super) fn prepare(source: &'a MlxModelInput) -> Option<Self> {
        Self::prepare_with(source, decoder_positions(source)?, array_storage)
    }

    pub(in crate::composition::mlx::session) fn original_plain_estimate(
        source: &'a MlxModelInput,
        positions: u64,
        array_bytes: impl Fn(&Array) -> Option<u64>,
    ) -> Option<SnapshotEstimate> {
        Self::prepare_with(source, positions, array_bytes).map(|plan| plan.estimate())
    }

    fn prepare_with(
        source: &'a MlxModelInput,
        positions: u64,
        array_bytes: impl Fn(&Array) -> Option<u64>,
    ) -> Option<Self> {
        if positions == 0 {
            return None;
        }
        source.with_borrowed(|input| {
            let (_, estimate) =
                Self::estimate_parts(input.parts, source.copied_metadata_bytes()?, array_bytes)?;
            Some(Self {
                _source: source,
                estimate,
            })
        })
    }

    fn estimate_parts(
        parts: &[input::InputPart],
        source_metadata: u64,
        array_bytes: impl Fn(&Array) -> Option<u64>,
    ) -> Option<(usize, SnapshotEstimate)> {
        let mut slots = 0usize;
        let mut arrays = 0u64;
        let mut metadata = 0u64;
        let mut extents = 0u64;
        for part in parts {
            // Unknown future payload variants stay unsupported before work.
            match part.payload() {
                input::InputPayload::TokenIds(_)
                | input::InputPayload::Tensor(_)
                | input::InputPayload::Embeddings(_) => {}
                _ => return None,
            }
            slots = slots.checked_add(1)?.checked_add(part.metadata().len())?;
            arrays = arrays.checked_add(array_bytes(part.payload().value())?)?;
            for value in part.metadata().values() {
                arrays = arrays.checked_add(array_bytes(value)?)?;
            }
            metadata =
                metadata.checked_add(u64::try_from(part.metadata().len()).ok()?.checked_mul(
                    std::mem::size_of::<(eredu_core::InputMetadataKey, Array)>() as u64,
                )?)?;
            extents = extents.checked_add(
                u64::try_from(part.extents().len())
                    .ok()?
                    .checked_mul(std::mem::size_of::<eredu_core::InputExtent>() as u64)?,
            )?;
        }
        let estimate = Self::copy_estimate(
            slots,
            parts.len(),
            arrays,
            metadata,
            extents,
            source_metadata,
        )?;
        Some((slots, estimate))
    }
    /// Same logical SnapshotBudget descriptor policy for the immutable B owner.
    /// This estimate is not a native copy recipe or source-storage credit.
    pub(super) fn completed_media_estimate(
        source: &crate::composition::mlx::CompletedOriginalModelInput,
        array_bytes: impl Fn(&Array) -> Option<u64>,
    ) -> Option<SnapshotEstimate> {
        let (parts, cache) = source.pending_parts();
        Self::estimate_parts(parts, cache.as_ref().logical_metadata_bytes()?, array_bytes)
            .map(|(_, estimate)| estimate)
    }
    /// Same logical copy policy for the saved resume worker's single packed
    /// text part. It carries no attribution/media metadata or old request.
    pub(super) fn one_part_estimate(
        array_bytes: u64,
        source_metadata: u64,
    ) -> Option<SnapshotEstimate> {
        Self::copy_estimate(1, 1, array_bytes, 0, 0, source_metadata)
    }

    fn copy_estimate(
        slots: usize,
        parts: usize,
        arrays: u64,
        metadata: u64,
        extents: u64,
        source_metadata: u64,
    ) -> Option<SnapshotEstimate> {
        let roots = slots.checked_mul(2)?;
        let part_bytes = u64::try_from(parts)
            .ok()?
            .checked_mul(std::mem::size_of::<input::InputPart>() as u64)?;
        let controls = part_bytes.checked_add(metadata)?.checked_add(extents)?;
        let retained_bytes = arrays
            .checked_add(std::mem::size_of::<MlxModelInput>() as u64)?
            .checked_add(controls)?
            .checked_add(source_metadata)?;
        // Ordinary logical copy accounting includes the source-control clone,
        // temporary metadata staging and both intermediate/destination roots.
        // Existing array_storage allowances are not original physical bounds.
        let copy_bytes = retained_bytes
            .checked_add(controls)?
            .checked_add(metadata)?
            .checked_add(
                u64::try_from(roots)
                    .ok()?
                    .checked_mul(std::mem::size_of::<Array>() as u64)?,
            )?
            .checked_add(std::mem::size_of::<MlxModelInput>() as u64)?
            .checked_add(std::mem::size_of::<RefCell<Vec<Array>>>() as u64)?
            .checked_add(2 * std::mem::size_of::<usize>() as u64)?;
        Some(SnapshotEstimate {
            retained_bytes,
            copy_bytes,
        })
    }

    pub(super) fn estimate(&self) -> SnapshotEstimate {
        self.estimate
    }
}
