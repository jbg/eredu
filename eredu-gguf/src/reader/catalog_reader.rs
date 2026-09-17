//! Catalog-only reader that can retain the immutable cold final header.
use super::*;
struct PreparedReader<R> {
    // Current file/buffer retires before the shared final header.
    inner: R,
    endian: Endian,
    version: u32,
    alignment: u64,
    limits: Limits,
    header: Arc<PreparedHeader>,
}
enum Storage<R> {
    Ordinary(Reader<R>),
    Prepared(PreparedReader<R>),
}
pub(crate) struct CatalogReader<R> {
    storage: Storage<R>,
}
impl<R: Read + Seek> CatalogReader<R> {
    pub(crate) fn ordinary(reader: Reader<R>) -> Self {
        Self {
            storage: Storage::Ordinary(reader),
        }
    }
    pub(super) fn captured(mut reader: Reader<R>) -> Result<Self> {
        let header = reader
            .prepared_header
            .as_mut()
            .ok_or_else(|| storage::refused("captured header owner"))?;
        let unique =
            Arc::get_mut(header).ok_or_else(|| storage::refused("unique cold header owner"))?;
        unique.parsed = Some(storage::FinalHeader::new(
            std::mem::take(&mut reader.metadata),
            std::mem::take(&mut reader.tensors),
        ));
        let Reader {
            inner,
            endian,
            version,
            alignment,
            limits,
            prepared_header,
            ..
        } = reader;
        Ok(Self {
            storage: Storage::Prepared(PreparedReader {
                inner,
                endian,
                version,
                alignment,
                limits,
                header: prepared_header.ok_or_else(|| storage::refused("captured header move"))?,
            }),
        })
    }
    pub(super) fn reopened(
        inner: R,
        limits: Limits,
        header: &Arc<PreparedHeader>,
        scratch: &mut [u8],
    ) -> Result<Self> {
        let parsed = parse::parse(
            inner,
            limits,
            HeaderPolicy::compare(header),
            Some(header),
            Some(scratch),
        )?;
        // Borrowed results are the actual sealed owners. They are never converted to owning clones.
        if !matches!(parsed.metadata, Cow::Borrowed(_))
            || !matches!(parsed.tensors, Cow::Borrowed(_))
        {
            return Err(storage::refused("borrowed header result"));
        }
        Ok(Self {
            storage: Storage::Prepared(PreparedReader {
                inner: parsed.inner,
                endian: parsed.endian,
                version: parsed.version,
                alignment: parsed.alignment,
                limits: parsed.limits,
                header: parsed
                    .prepared_header
                    .ok_or_else(|| storage::refused("prepared header move"))?,
            }),
        })
    }
    pub(crate) fn captured_header(&self) -> Option<&Arc<PreparedHeader>> {
        match &self.storage {
            Storage::Ordinary(r) => r.captured_header(),
            Storage::Prepared(r) => Some(&r.header),
        }
    }
    pub(crate) fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        match &self.storage {
            Storage::Ordinary(r) => r.metadata(),
            Storage::Prepared(r) => {
                &r.header
                    .parsed
                    .as_ref()
                    .expect("closed captured/reopened construction")
                    .metadata
            }
        }
    }
    pub(crate) fn tensors(&self) -> &[TensorDescriptor] {
        match &self.storage {
            Storage::Ordinary(r) => r.tensors(),
            Storage::Prepared(r) => {
                &r.header
                    .parsed
                    .as_ref()
                    .expect("closed captured/reopened construction")
                    .tensors
            }
        }
    }
    pub(crate) fn version(&self) -> u32 {
        match &self.storage {
            Storage::Ordinary(r) => r.version(),
            Storage::Prepared(r) => r.version,
        }
    }
    pub(crate) fn endian(&self) -> Endian {
        match &self.storage {
            Storage::Ordinary(r) => r.endian(),
            Storage::Prepared(r) => r.endian,
        }
    }
    pub(crate) fn alignment(&self) -> u64 {
        match &self.storage {
            Storage::Ordinary(r) => r.alignment(),
            Storage::Prepared(r) => r.alignment,
        }
    }
    fn payload(&mut self) -> payload::Payload<'_, R> {
        match &mut self.storage {
            Storage::Ordinary(r) => r.payload(),
            Storage::Prepared(r) => payload::Payload {
                inner: &mut r.inner,
                endian: r.endian,
                limits: &r.limits,
            },
        }
    }
    pub(crate) fn read_raw(&mut self, tensor: &TensorDescriptor) -> Result<Vec<u8>> {
        self.payload()
            .read_raw_with_storage(tensor.view(), RawStorage::Ordinary)
            .map(RawBuffer::into_owned)
            .map_err(ReadDestinationError::ordinary)
    }
    pub(crate) fn read_tensor(&mut self, tensor: &TensorDescriptor) -> Result<ConvertedTensor> {
        self.payload()
            .read_tensor_with_storage(tensor.view(), RawStorage::Ordinary, None)
            .map_err(ReadDestinationError::ordinary)
    }
    #[cfg(test)]
    pub(crate) fn read_tensor_with_storage<C: read_destination::ConversionDestination>(
        &mut self,
        tensor: &TensorDescriptor,
        storage: RawStorage<'_>,
        conversion: C,
    ) -> std::result::Result<C::Output, ReadDestinationError> {
        self.payload()
            .read_tensor_with_storage(tensor.view(), storage, conversion)
    }
    pub(crate) fn read_tensor_view_with_storage<C: read_destination::ConversionDestination>(
        &mut self,
        tensor: crate::TensorDescriptorView<'_>,
        storage: RawStorage<'_>,
        conversion: C,
    ) -> std::result::Result<C::Output, ReadDestinationError> {
        self.payload()
            .read_tensor_with_storage(tensor, storage, conversion)
    }
    pub(crate) fn read_tensor_plan_with_storage<C: read_destination::ConversionDestination>(
        &mut self,
        plan: plan_storage::AxisView<'_>,
        storage: RawStorage<'_>,
        conversion: C,
    ) -> std::result::Result<C::Output, ReadDestinationError> {
        self.payload()
            .read_tensor_plan_with_storage(plan, storage, conversion)
    }
    pub(crate) fn read_dense_tensor_span_with_storage<
        C: read_destination::ConversionDestination,
    >(
        &mut self,
        plan: plan_storage::SpanView<'_>,
        storage: RawStorage<'_>,
        conversion: C,
    ) -> std::result::Result<C::Output, ReadDestinationError> {
        self.payload()
            .read_dense_tensor_span_with_storage(plan, storage, conversion)
    }
}
impl CatalogReader<BufReader<File>> {
    pub(crate) fn open_captured(path: &Path, limits: Limits) -> Result<Self> {
        Self::captured(Reader::open_captured(path, limits)?)
    }
    pub(crate) fn open_prepared(
        path: &Path,
        limits: Limits,
        header: &Arc<PreparedHeader>,
        scratch: &mut [u8],
    ) -> Result<Self> {
        let file = File::open(path).map_err(|source| Error::Io { offset: 0, source })?;
        Self::reopened(
            BufReader::with_capacity(Reader::<BufReader<File>>::file_buffer_capacity(), file),
            limits,
            header,
            scratch,
        )
    }
}

impl CatalogReader<BufReader<File>> {
    pub(crate) fn into_file_reader(self) -> CatalogReader<FileReader> {
        let storage = match self.storage {
            Storage::Ordinary(reader) => {
                let Reader {
                    inner,
                    endian,
                    version,
                    alignment,
                    metadata,
                    tensors,
                    limits,
                    prepared_header,
                } = reader;
                Storage::Ordinary(Reader {
                    inner: FileReader::Ordinary(inner),
                    endian,
                    version,
                    alignment,
                    metadata,
                    tensors,
                    limits,
                    prepared_header,
                })
            }
            Storage::Prepared(reader) => {
                let PreparedReader {
                    inner,
                    endian,
                    version,
                    alignment,
                    limits,
                    header,
                } = reader;
                Storage::Prepared(PreparedReader {
                    inner: FileReader::Ordinary(inner),
                    endian,
                    version,
                    alignment,
                    limits,
                    header,
                })
            }
        };
        CatalogReader { storage }
    }
}
impl CatalogReader<FileReader> {
    #[cfg(test)]
    pub(crate) fn buffer_address(&self) -> Option<usize> {
        match &self.storage {
            Storage::Ordinary(reader) => reader.inner.buffer_address(),
            Storage::Prepared(reader) => reader.inner.buffer_address(),
        }
    }
    pub(crate) fn into_buffer(self) -> Option<ReaderBuffer> {
        match self.storage {
            Storage::Ordinary(reader) => reader.inner.into_buffer(),
            Storage::Prepared(reader) => reader.inner.into_buffer(),
        }
    }
    pub(crate) fn open_with_buffer(
        path: &Path,
        limits: Limits,
        header: Option<&Arc<PreparedHeader>>,
        scratch: &mut [u8],
        buffer: ReaderBuffer,
    ) -> std::result::Result<Self, (Error, ReaderBuffer)> {
        if !buffer.is_prepared() {
            return Err((
                Error::PreparedReaderStorage {
                    resource: "complete initialized buffer",
                },
                buffer,
            ));
        }
        let file = match File::open(path) {
            Ok(file) => file,
            Err(source) => return Err((Error::Io { offset: 0, source }, buffer)),
        };
        let mut inner = FileReader::prepared(file, buffer);
        let policy = header.map_or(HeaderPolicy::Ordinary, |header| {
            HeaderPolicy::compare(header)
        });
        // Borrow the owner through the original parser so failure cannot destroy
        // its scratch before the caller can recycle it. No parser is replaced.
        let parsed = parse::parse(
            &mut inner,
            limits,
            policy,
            header.map(AsRef::as_ref),
            header.map(|_| scratch),
        );
        let storage = match parsed {
            Err(error) => return Err((error, inner.into_buffer().expect("prepared file owner"))),
            Ok(parsed) => {
                let parse::Parsed {
                    inner: _,
                    endian,
                    version,
                    alignment,
                    limits,
                    metadata,
                    tensors,
                    prepared_header,
                } = parsed;
                if header.is_some() {
                    if !matches!(metadata, Cow::Borrowed(_)) || !matches!(tensors, Cow::Borrowed(_))
                    {
                        return Err((
                            storage::refused("borrowed header result"),
                            inner.into_buffer().expect("prepared file owner"),
                        ));
                    }
                    let Some(header) = prepared_header else {
                        return Err((
                            storage::refused("prepared header move"),
                            inner.into_buffer().expect("prepared file owner"),
                        ));
                    };
                    Storage::Prepared(PreparedReader {
                        inner,
                        endian,
                        version,
                        alignment,
                        limits,
                        header,
                    })
                } else {
                    // The ordinary parse produced these actual owners. into_owned
                    // moves them; no retained-header clone or new parse occurs.
                    Storage::Ordinary(Reader {
                        inner,
                        endian,
                        version,
                        alignment,
                        limits,
                        metadata: metadata.into_owned(),
                        tensors: tensors.into_owned(),
                        prepared_header,
                    })
                }
            }
        };
        Ok(Self { storage })
    }
}
