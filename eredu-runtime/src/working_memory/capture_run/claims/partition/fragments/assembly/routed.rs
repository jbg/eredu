//! Merge sorted participating routes using paid fragment cursors and one bitmap.
use super::*;
fn key(row: &RoutedUnitCaptureRow) -> (Option<u64>, u64, u64) {
    (row.source_peer, row.token, row.slot)
}
impl PreparedPartitionFragmentDestinations {
    pub(super) fn assemble_routed(
        &mut self,
        receipt: &PartitionCaptureReceiptPlan,
        claim: CaptureRoutedClaim<'_, '_>,
        dtype: &TensorDtype,
    ) -> Result<(PartitionFragmentValue, CaptureUsage), AssemblyCause> {
        self.custody.validate()?;
        claim.partition_identity().custody.validate()?;
        if !self.matches(receipt)
            || !self.complete()
            || receipt.combination() != PartitionCaptureCombination::Disjoint
            || !matches!(
                dtype,
                TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32 | TensorDtype::F64
            )
        {
            return Err(AssemblyCause::Source);
        }
        let Some((_, projection)) = receipt.producers().next() else {
            return Err(AssemblyCause::Source);
        };
        let geometry = claim.geometry();
        let global = projection.global_slice();
        if !std::ptr::eq(geometry.admission(), self.source.admission())
            || geometry.selection_index() != receipt.context().selection_index
            || geometry.phase() != receipt.context().phase
            || geometry.prediction() != receipt.context().prediction
            || geometry
                .source_shape()
                .iter()
                .map(|n| *n as u64)
                .ne(projection.global_shape().iter().copied())
            || geometry.starts() != global.starts
            || geometry.ends() != global.ends
            || geometry.strides() != global.strides
            || geometry.shape()[2] != self.routed_merge.len()
        {
            return Err(AssemblyCause::Source);
        }
        for (index, slot) in self.rows.iter().enumerate() {
            if slot.assembly_cursor != 0
                || slot.value.is_some()
                || !self
                    .allowance
                    .fragment_source(slot.producer, slot.fragment)
                    .is_some_and(|(actual, _)| actual == dtype)
            {
                return Err(AssemblyCause::Source);
            }
            let value = slot.routed_value.as_ref().ok_or(AssemblyCause::Source)?;
            value.identity().custody.validate()?;
            if value.observation().geometry != geometry.bank() {
                return Err(AssemblyCause::Source);
            }
            if let Some(previous) = self.rows[..index]
                .iter()
                .find(|previous| previous.producer == slot.producer)
            {
                if previous
                    .routed_value
                    .as_ref()
                    .ok_or(AssemblyCause::Source)?
                    .observation()
                    .source_token_ranges
                    != value.observation().source_token_ranges
                {
                    return Err(AssemblyCause::Source);
                }
            }
        }
        let usage = receipt
            .fragment_assembly_usage()?
            .ok_or(AssemblyCause::Source)?;
        self.allowance.quota_mut().reserve_quota(usage)?;
        let charged = self.allowance.assembly_record_charge(receipt)?;
        let mut output = claim.prepare()?;
        loop {
            let first = self
                .rows
                .iter()
                .filter_map(|slot| {
                    slot.routed_value
                        .as_ref()?
                        .observation()
                        .rows
                        .get(slot.assembly_cursor)
                })
                .min_by_key(|row| key(row));
            let Some(first) = first else {
                break;
            };
            let current = key(first);
            output.begin_assembled_row(first, &mut self.routed_merge)?;
            let mut copied = 0u64;
            for slot in &mut self.rows {
                let rows = &slot
                    .routed_value
                    .as_ref()
                    .ok_or(AssemblyCause::Source)?
                    .observation()
                    .rows;
                let Some(row) = rows
                    .get(slot.assembly_cursor)
                    .filter(|row| key(row) == current)
                else {
                    continue;
                };
                let fragment = receipt
                    .producer(slot.producer)
                    .and_then(|projection| projection.fragments().get(slot.fragment))
                    .ok_or(AssemblyCause::Source)?;
                copied = copied
                    .checked_add(output.merge_assembled_row(
                        row,
                        fragment.destination(),
                        &mut self.routed_merge,
                    )?)
                    .ok_or(WorkingMemoryError::Overflow)?;
                slot.assembly_cursor += 1;
            }
            if copied != self.routed_merge.len() as u64
                || self.routed_merge.iter().any(|seen| !*seen)
            {
                return Err(AssemblyCause::Source);
            }
            output.finish_row()?;
        }
        Ok((
            PartitionFragmentValue::Routed(output.finish_partition()?),
            charged,
        ))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<CaptureRoutedClaim<'_, '_>>(),
        size_of::<ScheduledCaptureRoutedUnits<'_, '_>>(),
        size_of::<ClaimedAssembledRoutedUnits>(),
        size_of::<Result<ClaimedAssembledRoutedUnits, CaptureRoutedFailure>>(),
        size_of::<CaptureRoutedFailure>(),
        size_of::<RoutedUnitAssemblyError>(),
        size_of::<RoutedUnitRowIdentity>(),
        size_of::<Option<&RoutedUnitCaptureRow>>(),
        size_of::<(Option<u64>, u64, u64)>(),
        size_of::<u64>(),
        size_of::<(
            &mut PreparedPartitionFragmentDestinations,
            &PartitionCaptureReceiptPlan,
            CaptureRoutedClaim<'_, '_>,
            &TensorDtype,
        )>(),
        size_of::<(
            &RoutedUnitCaptureRow,
            u64,
            f32,
            &ResolvedCaptureSlice,
            &mut [f32],
            &mut [bool],
        )>(),
        size_of::<Result<u64, RoutedUnitAssemblyError>>(),
        size_of::<Result<u64, CaptureRoutedHostError>>(),
        size_of::<Result<(PartitionFragmentValue, CaptureUsage), AssemblyCause>>(),
        size_of::<std::slice::IterMut<'_, Slot>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
