//! Paid immutable aliases and canonical pins for prompt-cache serialization.
use super::*;
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use safemlx::PreparedArrayClone;

/// Descriptor aliases and canonical pins outlive the synchronous file writer.
/// All partial handles remain outside the manager loan on every failure path.
pub(crate) struct PromptCacheSourceRows {
    rows: Vec<PromptCacheSourceRow>,
    clones: Vec<Option<[PreparedArrayClone; 2]>>,
    context: WorkspaceContext,
}
pub(crate) struct PromptCacheSourceRow {
    pub(crate) arrays: [Option<Array>; 2],
    pub(crate) host: Option<[Arc<ImmutableHostTransferBuffer>; 2]>,
    pub(crate) disk: Option<DiskLocation>,
    pub(crate) id: CacheBlockId,
    pub(crate) shapes: [Vec<i32>; 2],
    pub(crate) dtypes: [String; 2],
    pub(crate) bytes: u64,
    pin: Option<PinnedCacheBlock>,
}
impl PromptCacheSourceRows {
    pub(crate) fn rows(&self) -> &[PromptCacheSourceRow] {
        &self.rows
    }
    fn prepare(
        &mut self,
        mut source: CacheBlockSourceLoan<'_>,
        layers: std::ops::Range<usize>,
    ) -> Result<(), CacheSourceFailure> {
        let context = &self.context;
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let count = source
            .all_blocks()
            .filter(|row| layers.contains(&row.id().global_layer))
            .count();
        self.rows = context
            .metadata_vec(count)
            .map_err(|e| CacheSourceFailure::metadata(e, context))?;
        self.clones = context
            .metadata_vec(count)
            .map_err(|e| CacheSourceFailure::metadata(e, context))?;
        for block in source
            .all_blocks()
            .filter(|row| layers.contains(&row.id().global_layer))
        {
            context
                .charge_metadata(row_controls().ok_or_else(|| fail(CacheSourceError::Overflow))?)
                .map_err(|e| CacheSourceFailure::metadata(e.into(), context))?;
            if source
                .lifecycle
                .is_device_leased(block.id())
                .map_err(|e| fail(e.into()))?
            {
                return Err(fail(CacheSourceError::Identity));
            }
            let mut shapes = [
                context
                    .metadata_vec(block.shapes()[0].len())
                    .map_err(|e| CacheSourceFailure::metadata(e, context))?,
                context
                    .metadata_vec(block.shapes()[1].len())
                    .map_err(|e| CacheSourceFailure::metadata(e, context))?,
            ];
            for index in 0..2 {
                shapes[index].extend_from_slice(block.shapes()[index]);
            }
            let dtypes = [
                context
                    .metadata_string(format_args!("{}", block.dtypes()[0]))
                    .map_err(|e| CacheSourceFailure::metadata(e, context))?,
                context
                    .metadata_string(format_args!("{}", block.dtypes()[1]))
                    .map_err(|e| CacheSourceFailure::metadata(e, context))?,
            ];
            // Slots carry no source descriptor until filled in the second pass.
            let clones = if block.device().is_some() {
                context
                    .charge_metadata(
                        PreparedArrayClone::control_bytes()
                            .and_then(|n| n.checked_add(Array::inspection_clone_handle_bytes()))
                            .and_then(|n| n.checked_mul(2))
                            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
                    )
                    .map_err(|e| CacheSourceFailure::metadata(e.into(), context))?;
                Some([
                    PreparedArrayClone::try_prepare_for_inspection().map_err(|e| fail(e.into()))?,
                    PreparedArrayClone::try_prepare_for_inspection().map_err(|e| fail(e.into()))?,
                ])
            } else {
                None
            };
            self.clones.push(clones);
            self.rows.push(PromptCacheSourceRow {
                arrays: [None, None],
                host: None,
                disk: block.record.disk().cloned(),
                id: block.id().clone(),
                shapes,
                dtypes,
                bytes: block.logical_bytes(),
                pin: None,
            });
        }
        for (index, row) in self.rows.iter_mut().enumerate() {
            let selection = CacheBlockSelection::new(
                row.id.global_layer,
                row.id.representation,
                row.id.start,
                row.id.end,
                0,
            );
            row.pin = Some(
                source
                    .selected(selection)
                    .pin_prepared_block(&row.id, context.metadata_funding())
                    .map_err(fail)?,
            );
            let record = source
                .records
                .get(&row.id)
                .ok_or_else(|| fail(CacheSourceError::Identity))?;
            if let Some(host) = record.physical.host_resource() {
                row.host = Some(match host {
                    HostCacheBlock::KeyValue { keys, values } => [keys.clone(), values.clone()],
                    HostCacheBlock::CompressedLatentRotary { latent, rotary_key } => {
                        [latent.clone(), rotary_key.clone()]
                    }
                });
            } else if let Some(device) = record.physical.device_resource() {
                for (slot, array) in device.arrays().into_iter().enumerate() {
                    row.arrays[slot] = Some(
                        self.clones[index]
                            .as_mut()
                            .ok_or_else(|| fail(CacheSourceError::Identity))?[slot]
                            .fill_for_inspection(array)
                            .map_err(|e| fail(e.into()))?,
                    );
                }
            } else if row.disk.is_none() {
                return Err(fail(CacheSourceError::PendingStorage));
            }
        }
        Ok(())
    }
}
fn row_controls() -> Option<usize> {
    let frames = [
        size_of::<PromptCacheSourceRow>(),
        size_of::<Option<[PreparedArrayClone; 2]>>(),
        PinnedCacheBlock::fixed_controls()?,
        size_of::<CacheBlockSelection>(),
        size_of::<[Option<Array>; 2]>(),
        size_of::<Result<Array, safemlx::PreparedArrayCloneCause>>(),
        size_of::<Result<bool, CacheLifecycleError>>(),
        size_of::<std::ops::Range<usize>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
impl CacheResidencyManager {
    /// Retains exact stable block owners through serialization without evaluating,
    /// sealing tails, changing residency or acquiring a demand lease.
    pub(crate) fn prompt_cache_sources(
        &self,
        layers: std::ops::Range<usize>,
        context: &WorkspaceContext,
    ) -> Result<PromptCacheSourceRows, CacheSourceFailure> {
        context
            .charge_metadata(size_of::<(
                PromptCacheSourceRows,
                Result<(), CacheSourceFailure>,
                std::ops::Range<usize>,
            )>())
            .map_err(|e| CacheSourceFailure::metadata(e.into(), context))?;
        let mut pending = PromptCacheSourceRows {
            rows: Vec::new(),
            clones: Vec::new(),
            context: context.clone(),
        };
        self.with_source_loan(
            CacheBlockSelection::new(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0),
            context,
            |source| pending.prepare(source, layers),
        )?;
        Ok(pending)
    }
}
