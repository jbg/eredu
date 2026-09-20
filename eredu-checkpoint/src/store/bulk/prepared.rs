//! Move-only encoded read records with their original caller custody.
use super::*;

/// Exact file or memory read records. All metadata and source handles retire
/// before caller custody. Memory reads use the direct copy worker; file reads
/// use the existing borrowed worker with separately admitted read scratch.
///
/// ```compile_fail
/// use eredu_checkpoint::store::PreparedEncodedRead;
/// fn copy(read: PreparedEncodedRead<()>) { let _ = read.clone(); }
/// ```
pub struct PreparedEncodedRead<C> {
    pub(crate) batch: EncodedReadBatch,
    pub(crate) _custody: C,
}
impl<C> PreparedEncodedRead<C> {
    /// Metadata in original occurrence order, including repeated sources.
    pub fn tensors(&self) -> &[TensorMetadata] {
        self.batch.tensors()
    }
    /// Exact caller-owned payload destination length.
    pub fn byte_len(&self) -> usize {
        self.batch.byte_len()
    }
    /// Separate scratch contribution for a read over these completed records.
    pub fn read_layout(&self) -> Option<EncodedReadLayout> {
        EncodedReadLayout::inspect(std::iter::once(&self.batch))
    }
    /// Fill the exact caller destination without staging a payload. Memory-only
    /// reads allocate no scratch. File reads validate original source identity
    /// and update the same finite diagnostics through the borrowed worker.
    pub fn read_into(&self, destination: &mut [u8]) -> Result<(), EncodedReadFailure> {
        if self.batch.shards.is_empty() {
            if destination.len() != self.byte_len() {
                return Err(EncodedReadFailure {
                    batch: None,
                    shard: None,
                    completed_shards: 0,
                    cause: EncodedReadFailureCause::DestinationLengths,
                });
            }
            for source in &self.batch.memory {
                source.copy_into(destination)?;
            }
            Ok(())
        } else {
            borrowed::read_many(std::iter::once(&self.batch), &mut [destination], || {})
        }
    }
    /// Move additional caller custody alongside the existing read owner. This
    /// allocates nothing, changes no source identity and reserves no new bytes.
    pub fn with_custody<D>(self, custody: D) -> PreparedEncodedRead<(C, D)> {
        PreparedEncodedRead {
            batch: self.batch,
            _custody: (self._custody, custody),
        }
    }
    /// Consume the read into a counted projection over its exact source spans.
    /// Planning reads no payload and retains the original read on refusal.
    pub fn project<'a>(
        self,
        mapping: &'a crate::recipe::EncodedRecipeMapping,
    ) -> Result<EncodedReadProjectionPlan<'a, C>, EncodedProjectionBuildError<C>> {
        EncodedReadProjectionPlan::new(self, mapping)
    }
}
impl<C> std::fmt::Debug for PreparedEncodedRead<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedEncodedRead")
            .field("tensors", &self.batch.tensors.len())
            .field("byte_len", &self.byte_len())
            .finish_non_exhaustive()
    }
}
