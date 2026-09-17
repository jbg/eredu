//! Pre-G4 constructor/close oracle; method names and the new empty ordinary scratch field are adapted; the inline checkpoint adapter adds no allocation.
use super::*;
impl Checkpoint {
    pub fn old_materializer(&self) -> TensorMaterializer {
        let locations: HashMap<String, TensorLocation> = self
            .shards
            .iter()
            .enumerate()
            .flat_map(|(shard_index, shard)| {
                shard
                    .tensors
                    .iter()
                    .enumerate()
                    .map(move |(tensor_index, tensor)| {
                        (
                            tensor.descriptor.name.clone(),
                            TensorLocation {
                                shard_index,
                                tensor_index,
                            },
                        )
                    })
            })
            .collect();
        TensorMaterializer {
            checkpoint: self.clone().into(),
            locations: locations.into(),
            reader: None,
            header_scratch: Vec::new(),
            reader_storage: ReaderStorage::Ordinary,
        }
    }
}
impl TensorMaterializer {
    pub fn old_close_reader(&mut self) -> Option<PathBuf> {
        let (index, _) = self.reader.take()?;
        Some(self.checkpoint.shards[index].path.clone())
    }
}
