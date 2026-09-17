//! Physical-name lookup using either the original map or retained coordinates.
use super::*;

pub(super) enum MaterializerIndex {
    Ordinary(HashMap<String, TensorLocation>),
    Coordinates(Vec<TensorLocation>),
}
impl From<HashMap<String, TensorLocation>> for MaterializerIndex {
    fn from(index: HashMap<String, TensorLocation>) -> Self {
        Self::Ordinary(index)
    }
}
impl MaterializerIndex {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Ordinary(index) => index.len(),
            Self::Coordinates(index) => index.len(),
        }
    }
    pub(super) fn get(&self, name: &str, checkpoint: &Checkpoint) -> Option<TensorLocation> {
        match self {
            Self::Ordinary(index) => index.get(name).copied(),
            Self::Coordinates(index) => {
                // Exact physical names only. Last equal coordinate agrees with
                // the ordinary map's source-order insertion even for a private
                // duplicate fixture; public parsing rejects duplicates earlier.
                let end =
                    index.partition_point(|location| physical_name(checkpoint, *location) <= name);
                let location = *index.get(end.checked_sub(1)?)?;
                (physical_name(checkpoint, location) == name).then_some(location)
            }
        }
    }
    fn prepare(checkpoint: &Checkpoint) -> Self {
        let mut coordinates = Vec::with_capacity(checkpoint.physical_tensor_count);
        for (shard_index, shard) in checkpoint.shards.iter().enumerate() {
            for tensor_index in 0..shard.tensors.len() {
                coordinates.push(TensorLocation {
                    shard_index,
                    tensor_index,
                });
            }
        }
        coordinates.sort_unstable_by(|a, b| {
            physical_name(checkpoint, *a)
                .cmp(physical_name(checkpoint, *b))
                .then_with(|| a.shard_index.cmp(&b.shard_index))
                .then_with(|| a.tensor_index.cmp(&b.tensor_index))
        });
        Self::Coordinates(coordinates)
    }
}
fn physical_name(checkpoint: &Checkpoint, location: TensorLocation) -> &str {
    &checkpoint.shards[location.shard_index].tensors[location.tensor_index]
        .descriptor
        .name
}
impl Checkpoint {
    /// Move this catalog into a cold materializer whose physical-name index owns
    /// only finite coordinates. Lookup borrows exact retained names and is
    /// logarithmic; ordinary constructors preserve their original hash map.
    /// This constructor does not qualify cold allocator or source-control costs.
    pub fn into_coordinate_materializer(self) -> TensorMaterializer {
        let locations = MaterializerIndex::prepare(&self);
        TensorMaterializer {
            header_scratch: self.header_scratch(),
            checkpoint: self.into(),
            locations,
            reader: None,
            reader_storage: ReaderStorage::Ordinary,
        }
    }
    /// Coordinate-index variant of the existing shared catalog constructor.
    /// The same catalog moves into one Arc; names/descriptors are not cloned.
    pub fn into_shared_coordinate_materializer(self) -> TensorMaterializer {
        self.into_coordinate_materializer().share_catalog()
    }
}
impl TensorMaterializer {
    pub(super) fn share_catalog(mut self) -> Self {
        self.checkpoint = match self.checkpoint {
            MaterializerCheckpoint::Owned(checkpoint) => MaterializerCheckpoint::Shared(
                prepared_materializer::SharedCheckpoint::new(checkpoint, ()),
            ),
            shared => shared,
        };
        self
    }
    /// Actual coordinate-index backing, excluding reader and header scratch,
    /// surrounding inline owners, allocator charge and cold construction overlap.
    /// `None` preserves the unqualified private layout of an ordinary hash map.
    pub fn prepared_index_layout(&self) -> Option<std::alloc::Layout> {
        match &self.locations {
            MaterializerIndex::Ordinary(_) => None,
            MaterializerIndex::Coordinates(index) => {
                std::alloc::Layout::array::<TensorLocation>(index.capacity()).ok()
            }
        }
    }
}
