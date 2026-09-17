//! Reader-buffer ownership policy; no source, selection or cache policy.
use super::*;
use crate::ReaderBuffer;

pub(super) enum ReaderStorage {
    Ordinary,
    Local([Option<ReaderBuffer>; 2]),
    Pooled(Option<ReaderBuffer>),
}
impl ReaderStorage {
    pub(super) fn take(&mut self) -> Result<Option<ReaderBuffer>> {
        let buffer = match self {
            Self::Ordinary => return Ok(None),
            Self::Local(buffers) => buffers.iter_mut().find_map(Option::take),
            Self::Pooled(buffer) => buffer.take(),
        };
        buffer.map(Some).ok_or(Error::PreparedReaderStorage {
            resource: "available cold reader buffer",
        })
    }
    pub(super) fn recycle(&mut self, buffer: ReaderBuffer) {
        let slot = match self {
            Self::Ordinary => unreachable!("ordinary readers do not own prepared backing"),
            Self::Local(buffers) => buffers.iter_mut().find(|slot| slot.is_none()),
            Self::Pooled(slot) => slot.is_none().then_some(slot),
        };
        *slot.expect("closed reader returns to its vacated destination") = Some(buffer);
    }
    pub(super) fn retire(&mut self, reader: CatalogReader<FileReader>) {
        if let Some(buffer) = reader.into_buffer() {
            self.recycle(buffer);
        }
    }
}
impl Checkpoint {
    /// Move this catalog into a direct materializer with two already prepared
    /// buffers. The second preserves opening-before-closing on replacement.
    /// Failure returns the unchanged source and both actual supplied owners.
    pub fn into_materializer_with_reader_buffers(
        self,
        buffers: [ReaderBuffer; 2],
    ) -> std::result::Result<TensorMaterializer, (Error, Self, [ReaderBuffer; 2])> {
        if !buffers.iter().all(ReaderBuffer::is_prepared) {
            return Err((
                Error::PreparedReaderStorage {
                    resource: "two complete direct reader buffers",
                },
                self,
                buffers,
            ));
        }
        let mut materializer = self.into_coordinate_materializer();
        materializer.reader_storage = ReaderStorage::Local(buffers.map(Some));
        Ok(materializer)
    }
    /// Iterate with one prepared buffer, closing each shard before its next open.
    /// The ordinary iterator constructor retains its original lazy allocation.
    pub fn converted_tensors_with_reader_buffer(
        &self,
        buffer: ReaderBuffer,
    ) -> std::result::Result<ConvertedTensorIter<'_>, (Error, ReaderBuffer)> {
        if !buffer.is_prepared() {
            return Err((
                Error::PreparedReaderStorage {
                    resource: "complete iterator reader buffer",
                },
                buffer,
            ));
        }
        let mut iterator = self.converted_tensors();
        iterator.reader_storage = ReaderStorage::Local([Some(buffer), None]);
        Ok(iterator)
    }
}
impl TensorMaterializer {
    /// Require a pooled reader destination for future opens, without allocating.
    /// This cold transition rejects an already-open or already-configured owner.
    pub fn with_pooled_reader_storage(mut self) -> std::result::Result<Self, Self> {
        if self.reader.is_some() || !matches!(self.reader_storage, ReaderStorage::Ordinary) {
            return Err(self);
        }
        self.reader_storage = ReaderStorage::Pooled(None);
        Ok(self)
    }
    /// Move one actual source-bank buffer into an empty pooled reader slot.
    /// Refusal returns that same buffer; it never selects another storage policy.
    pub fn supply_reader_buffer(
        &mut self,
        buffer: ReaderBuffer,
    ) -> std::result::Result<(), ReaderBuffer> {
        if self.reader.is_none() && buffer.is_prepared() {
            if let ReaderStorage::Pooled(slot @ None) = &mut self.reader_storage {
                *slot = Some(buffer);
                return Ok(());
            }
        }
        Err(buffer)
    }
    /// Return an idle/recycled buffer to its owning source bank. An open reader's
    /// buffer cannot be extracted, and ordinary/local storage is unaffected.
    pub fn take_idle_reader_buffer(&mut self) -> Option<ReaderBuffer> {
        match &mut self.reader_storage {
            ReaderStorage::Pooled(buffer) => buffer.take(),
            _ => None,
        }
    }
}

pub(super) fn open_with_reader_storage(
    shard: &CatalogShard,
    limits: Limits,
    scratch: &mut [u8],
    storage: &mut ReaderStorage,
) -> Result<CatalogReader<FileReader>> {
    let buffer = storage.take()?;
    let reader = match buffer {
        None => open_reopened_reader(shard, limits, scratch)?,
        Some(buffer) => match CatalogReader::open_with_buffer(
            &shard.path,
            limits,
            shard.prepared_header.as_ref(),
            scratch,
            buffer,
        ) {
            Ok(reader) => reader,
            Err((source, buffer)) => {
                storage.recycle(buffer);
                return Err(Error::Shard {
                    path: shard.path.to_path_buf(),
                    source: Box::new(source),
                });
            }
        },
    };
    if let Err(error) = validate_reopened_shard_ref(&reader, shard) {
        storage.retire(reader);
        return Err(error);
    }
    Ok(reader)
}
