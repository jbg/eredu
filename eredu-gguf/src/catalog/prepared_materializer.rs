//! Exact fresh coordinate/scratch construction from the actual owned checkpoint.
use super::*;
use std::{
    alloc::Layout,
    any::Any,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Debug)]
struct SharedBody<C> {
    checkpoint: Checkpoint,
    custody: C,
}
trait SharedStorage: Send + Sync + fmt::Debug {
    fn checkpoint(&self) -> &Checkpoint;
    fn retire(self: Arc<Self>);
}
impl<C: Any + Send + Sync + fmt::Debug> SharedStorage for SharedBody<C> {
    fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
#[derive(Debug)]
pub(super) struct SharedCheckpoint(Option<Arc<dyn SharedStorage>>);
impl SharedCheckpoint {
    #[cfg(test)]
    pub(super) fn test_alive(&self) -> impl Fn() -> bool {
        let weak = Arc::downgrade(self.0.as_ref().expect("shared checkpoint"));
        move || weak.strong_count() != 0
    }
    pub(super) fn new<C: Any + Send + Sync + fmt::Debug>(
        checkpoint: Checkpoint,
        custody: C,
    ) -> Self {
        Self(Some(Arc::new(SharedBody {
            checkpoint,
            custody,
        })))
    }
}
impl Clone for SharedCheckpoint {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for SharedCheckpoint {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}
impl std::ops::Deref for SharedCheckpoint {
    type Target = Checkpoint;
    fn deref(&self) -> &Checkpoint {
        self.0.as_ref().expect("shared checkpoint").checkpoint()
    }
}

/// Concrete fresh constructor requests. Original checkpoint payload is retained,
/// not copied, and must already belong to the caller's source baseline.
#[derive(Clone, Copy, Debug)]
pub struct PreparedMaterializerStorage {
    coordinates: usize,
    scratch: usize,
    shared: Layout,
    controls: usize,
}
impl PreparedMaterializerStorage {
    /// Exact coordinate and scratch vector backing requested before construction.
    pub fn requested_buffer_bytes(&self) -> Option<usize> {
        Layout::array::<TensorLocation>(self.coordinates)
            .ok()?
            .size()
            .checked_add(self.scratch)
    }
    /// One shared immutable checkpoint body; caller qualifies Arc counters/padding.
    pub fn shared_body(&self) -> Layout {
        self.shared
    }
    /// Fixed constructor, failure, reserve and nonrecursive ordering transports.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Actual parser scratch requested by this materializer.
    pub fn scratch_bytes(&self) -> usize {
        self.scratch
    }
}

/// Actual failed fresh destinations and original checkpoint. No owning value or
/// custody is extracted; the complete prefix retires before constructor custody.
#[derive(Debug)]
pub struct PreparedMaterializerFailure<C> {
    checkpoint: Checkpoint,
    coordinates: Vec<TensorLocation>,
    scratch: Vec<u8>,
    cause: TryReserveError,
    custody: C,
}
impl<C> PreparedMaterializerFailure<C> {
    /// Same source input, retained without cloning or reopening.
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }
    /// Actual allocator failure at the reached destination.
    pub fn cause(&self) -> &TryReserveError {
        &self.cause
    }
}
impl<C: fmt::Debug> fmt::Display for PreparedMaterializerFailure<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl<C: fmt::Debug> std::error::Error for PreparedMaterializerFailure<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl Checkpoint {
    /// Owns every request of the exact fresh coordinate materializer worker.
    /// Qualification and pre-admission are the consuming runtime's obligation.
    pub fn prepared_materializer_storage<C>(&self) -> Option<PreparedMaterializerStorage> {
        let coordinates = self
            .shards
            .iter()
            .try_fold(0usize, |n, s| n.checked_add(s.tensors.len()))?;
        let scratch = self
            .shards
            .iter()
            .filter_map(|s| s.prepared_header.as_ref())
            .map(|h| h.scratch_len())
            .max()
            .unwrap_or(0);
        Layout::array::<TensorLocation>(coordinates).ok()?;
        Layout::array::<u8>(scratch).ok()?;
        let controls = [
            size_of::<TensorMaterializer>(),
            size_of::<SharedBody<C>>(),
            size_of::<Option<SharedBody<C>>>(),
            size_of::<SharedCheckpoint>(),
            size_of::<Arc<SharedBody<C>>>(),
            size_of::<Arc<dyn SharedStorage>>(),
            size_of::<Option<Arc<dyn SharedStorage>>>(),
            size_of::<&SharedCheckpoint>(),
            size_of::<PreparedMaterializerStorage>(),
            size_of::<Option<PreparedMaterializerStorage>>(),
            size_of::<PreparedMaterializerFailure<C>>(),
            size_of::<std::result::Result<TensorMaterializer, PreparedMaterializerFailure<C>>>(),
            size_of::<std::result::Result<(), TryReserveError>>(),
            size_of::<&Checkpoint>(),
            size_of::<&mut Vec<TensorLocation>>(),
            size_of::<&mut Vec<u8>>(),
            size_of::<CoordinateSift>(),
            size_of::<std::iter::Rev<std::ops::Range<usize>>>(),
            size_of::<&mut [TensorLocation]>(),
            size_of::<TensorLocation>(), // comparator a
            size_of::<TensorLocation>(), // comparator b
            size_of::<&Checkpoint>(),    // comparator source
            size_of::<std::iter::Enumerate<std::slice::Iter<'static, CatalogShard>>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<std::cmp::Ordering>(),
        ];
        Some(PreparedMaterializerStorage {
            coordinates,
            scratch,
            shared: Layout::new::<SharedBody<C>>(),
            controls: controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)?,
        })
    }
    /// Same actual source and pooled-reader semantics, using only fresh exact
    /// reserves. The consumer must qualify these requests before calling.
    pub fn try_into_prepared_materializer<C: Any + Send + Sync + fmt::Debug>(
        self,
        custody: C,
    ) -> std::result::Result<TensorMaterializer, PreparedMaterializerFailure<C>> {
        let count = self
            .shards
            .iter()
            .try_fold(0usize, |n, s| n.checked_add(s.tensors.len()))
            // TensorLocation is nonzero-sized; the reached reserve produces
            // its typed capacity-overflow error while retaining this source.
            .unwrap_or(usize::MAX);
        let scratch_len = self
            .shards
            .iter()
            .filter_map(|s| s.prepared_header.as_ref())
            .map(|h| h.scratch_len())
            .max()
            .unwrap_or(0);
        let mut coordinates = Vec::new();
        let mut scratch = Vec::new();
        let result = (|| {
            coordinates.try_reserve_exact(count)?;
            for (shard_index, shard) in self.shards.iter().enumerate() {
                for tensor_index in 0..shard.tensors.len() {
                    coordinates.push(TensorLocation {
                        shard_index,
                        tensor_index,
                    });
                }
            }
            order_coordinates(&mut coordinates, &self);
            scratch.try_reserve_exact(scratch_len)?;
            scratch.resize(scratch_len, 0);
            Ok(())
        })();
        if let Err(cause) = result {
            return Err(PreparedMaterializerFailure {
                checkpoint: self,
                coordinates,
                scratch,
                cause,
                custody,
            });
        }
        Ok(TensorMaterializer {
            locations: MaterializerIndex::Coordinates(coordinates),
            reader: None,
            header_scratch: scratch,
            reader_storage: ReaderStorage::Pooled(None),
            checkpoint: MaterializerCheckpoint::Shared(SharedCheckpoint::new(self, custody)),
        })
    }
}
// Iterative heapsort uses fixed scalar controls and no scratch/recursion. Equal
// names retain the ordinary source-order winner through coordinate tie-breaks.
struct CoordinateSift {
    root: usize,
    end: usize,
    child: usize,
}
fn order_coordinates(values: &mut [TensorLocation], checkpoint: &Checkpoint) {
    fn cmp(a: TensorLocation, b: TensorLocation, c: &Checkpoint) -> std::cmp::Ordering {
        c.shards[a.shard_index].tensors[a.tensor_index]
            .descriptor
            .name
            .cmp(
                &c.shards[b.shard_index].tensors[b.tensor_index]
                    .descriptor
                    .name,
            )
            .then_with(|| a.shard_index.cmp(&b.shard_index))
            .then_with(|| a.tensor_index.cmp(&b.tensor_index))
    }
    fn sift(values: &mut [TensorLocation], root: usize, end: usize, c: &Checkpoint) {
        let mut frame = CoordinateSift {
            root,
            end,
            child: 0,
        };
        while frame.root < frame.end / 2 {
            frame.child = frame.root * 2 + 1;
            if frame.child + 1 < frame.end
                && cmp(values[frame.child], values[frame.child + 1], c).is_lt()
            {
                frame.child += 1;
            }
            if !cmp(values[frame.root], values[frame.child], c).is_lt() {
                break;
            }
            values.swap(frame.root, frame.child);
            frame.root = frame.child;
        }
    }
    for root in (0..values.len() / 2).rev() {
        sift(values, root, values.len(), checkpoint);
    }
    for end in (1..values.len()).rev() {
        values.swap(0, end);
        sift(values, 0, end, checkpoint);
    }
}
