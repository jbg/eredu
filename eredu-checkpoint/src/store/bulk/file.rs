//! Original file-read metadata over already prepared immutable shard headers.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of};

/// Fixed category and source occurrence of an encoded file inspection refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetensorsEncodedReadPlanErrorKind {
    /// This source does not contain the requested occurrence.
    UnknownTensor { index: usize },
    /// An enclosing concrete view excludes this occurrence.
    UnauthorizedTensor { index: usize },
    /// The source has not prepared this header.
    HeaderUnavailable { index: usize },
    /// The source retains a failed header construction.
    Header { index: usize },
    /// Checked geometry cannot be represented.
    Overflow(&'static str),
}

enum InspectionCause {
    Fixed(SafetensorsEncodedReadPlanErrorKind),
    Unauthorized {
        index: usize,
        owner: acquisition::PreparedAcquisitionOwner,
    },
    Header {
        index: usize,
        owner: Arc<AdmittedShard>,
    },
}

/// Inspection refusal owning the actual failed header or authorization owner.
/// Retaining existing handles allocates no diagnostic strings and cannot retry
/// header construction. The source's original error/custody remains unchanged.
pub struct SafetensorsEncodedReadPlanError(InspectionCause);
impl SafetensorsEncodedReadPlanError {
    pub(in crate::store) fn unknown(index: usize) -> Self {
        Self(InspectionCause::Fixed(
            SafetensorsEncodedReadPlanErrorKind::UnknownTensor { index },
        ))
    }
    fn unavailable(index: usize) -> Self {
        Self(InspectionCause::Fixed(
            SafetensorsEncodedReadPlanErrorKind::HeaderUnavailable { index },
        ))
    }
    fn overflow(context: &'static str) -> Self {
        Self(InspectionCause::Fixed(
            SafetensorsEncodedReadPlanErrorKind::Overflow(context),
        ))
    }
    pub(in crate::store) fn unauthorized(
        index: usize,
        owner: acquisition::PreparedAcquisitionOwner,
    ) -> Self {
        Self(InspectionCause::Unauthorized { index, owner })
    }
    /// Allocation-free category, retaining the original ordered occurrence.
    pub fn kind(&self) -> SafetensorsEncodedReadPlanErrorKind {
        match &self.0 {
            InspectionCause::Fixed(kind) => *kind,
            InspectionCause::Unauthorized { index, .. } => {
                SafetensorsEncodedReadPlanErrorKind::UnauthorizedTensor { index: *index }
            }
            InspectionCause::Header { index, .. } => {
                SafetensorsEncodedReadPlanErrorKind::Header { index: *index }
            }
        }
    }
    /// Exact retained contract name, without a clone or new routing callback.
    pub fn contract(&self) -> Option<&str> {
        match &self.0 {
            InspectionCause::Unauthorized { owner, .. } => owner.encoded_contract(),
            _ => None,
        }
    }
    fn header_error(&self) -> Option<&StoreError> {
        match &self.0 {
            InspectionCause::Header { owner, .. } => {
                owner.header.get().and_then(|result| result.as_ref().err())
            }
            _ => None,
        }
    }
    pub(super) fn into_ordinary(self, keys: &[String]) -> StoreError {
        use SafetensorsEncodedReadPlanErrorKind as K;
        match self.kind() {
            K::UnknownTensor { index } => StoreError::UnknownTensor {
                key: keys[index].clone(),
            },
            K::UnauthorizedTensor { index } => StoreError::UnauthorizedTensor {
                key: keys[index].clone(),
                contract: self
                    .contract()
                    .expect("retained authorization owner")
                    .into(),
            },
            K::Header { .. } => self
                .header_error()
                .expect("retained header failure")
                .clone(),
            K::HeaderUnavailable { .. } => {
                StoreError::Internal("encoded source header is not prepared".into())
            }
            K::Overflow(context) => StoreError::Overflow {
                context: context.into(),
            },
        }
    }
}
impl fmt::Debug for SafetensorsEncodedReadPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SafetensorsEncodedReadPlanError")
            .field("kind", &self.kind())
            .field("contract", &self.contract())
            .field("source", &self.header_error())
            .finish()
    }
}
impl fmt::Display for SafetensorsEncodedReadPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SafetensorsEncodedReadPlanErrorKind as K;
        match self.kind() {
            K::UnknownTensor { index } => {
                write!(f, "encoded file source occurrence {index} is absent")
            }
            K::UnauthorizedTensor { index } => write!(
                f,
                "encoded file source occurrence {index} is not authorized by {:?}",
                self.contract().expect("retained authorization owner")
            ),
            K::HeaderUnavailable { index } => write!(
                f,
                "encoded file source occurrence {index} has no prepared header"
            ),
            K::Header { index } => write!(
                f,
                "encoded file source occurrence {index} has a retained header failure: {}",
                self.header_error().expect("retained header failure")
            ),
            K::Overflow(context) => write!(f, "encoded file read overflow: {context}"),
        }
    }
}
impl std::error::Error for SafetensorsEncodedReadPlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.header_error()
            .map(|cause| cause as &(dyn std::error::Error + 'static))
    }
}

struct Entry<'a> {
    path: &'a Path,
    admitted: &'a Arc<AdmittedFile>,
    source: Range<u64>,
    destination: Range<usize>,
}

/// Original constructor bound to the actual source, prepared headers and keys.
/// Header, source-cache, telemetry and key storage keep their separate admission.
/// This plan copies no payloads and creates no cache or diagnostic entries.
pub struct SafetensorsEncodedReadPlan<'a> {
    store: &'a SafetensorsWeightStore,
    keys: &'a [String],
    byte_len: usize,
    max_groups: usize,
    backing_bytes: usize,
}
impl<'a> SafetensorsEncodedReadPlan<'a> {
    /// Authenticate and borrow an actual file source through retained built-in
    /// views, preserving their ordered batch authorization. The root and keys
    /// stay borrowed through inspection/construction; completed batches own their
    /// file identities. Unsupported or forwarded routes return no plan and never
    /// invoke ordinary read preparation. Header failures retain their actual source-owned diagnostics.
    ///
    /// ```compile_fail
    /// use eredu_checkpoint::store::{RetainedCheckpointSource, SafetensorsEncodedReadPlan};
    /// fn retire_early(root: RetainedCheckpointSource, keys: &[String]) {
    ///     let plan = SafetensorsEncodedReadPlan::from_source(&root, keys).unwrap().unwrap();
    ///     drop(root);
    ///     let _ = plan.construct(());
    /// }
    /// ```
    pub fn from_source(
        source: &'a RetainedCheckpointSource,
        keys: &'a [String],
    ) -> Result<Option<Self>, SafetensorsEncodedReadPlanError> {
        let Some(store) = acquisition::retained_route::encoded_file_source(source, keys)? else {
            return Ok(None);
        };
        Self::new(store, keys).map(Some)
    }

    /// Inspect only headers already retained by this exact source. Missing headers
    /// refuse without an ordinary fallback or a new header-initialization charge.
    pub fn new(
        store: &'a SafetensorsWeightStore,
        keys: &'a [String],
    ) -> Result<Self, SafetensorsEncodedReadPlanError> {
        let overflow = SafetensorsEncodedReadPlanError::overflow;
        let max_groups = keys.len().min(store.shards.payload_paths().len());
        let mut plan = Self {
            store,
            keys,
            byte_len: 0,
            max_groups,
            backing_bytes: 0,
        };
        let mut path_max = 0usize;
        let mut metadata_bytes = 0usize;
        for index in 0..keys.len() {
            let (metadata, row) = plan.entry(index, plan.byte_len)?;
            plan.byte_len = row.destination.end;
            path_max = path_max.max(row.path.as_os_str().as_encoded_bytes().len());
            metadata_bytes = metadata_bytes
                .checked_add(
                    MetadataCloneLayout::of(metadata)
                        .and_then(|layout| layout.payload_bytes())
                        .ok_or(overflow("bulk metadata layout"))?,
                )
                .ok_or(overflow("bulk metadata storage"))?;
        }
        // Group capacity is bounded by both occurrences and admitted shards.
        // One maximum selected path per possible group is conservative; no
        // payload-length multiplier or postconstruction clone quote is used.
        let paths = max_groups
            .checked_mul(path_max)
            .ok_or(overflow("bulk path storage"))?;
        plan.backing_bytes = [
            Layout::array::<TensorMetadata>(keys.len())
                .map_err(|_| overflow("bulk tensor metadata"))?
                .size(),
            Layout::array::<Entry<'_>>(keys.len())
                .map_err(|_| overflow("bulk source ordering"))?
                .size(),
            Layout::array::<(PathBuf, ReadShard)>(max_groups)
                .map_err(|_| overflow("bulk shard metadata"))?
                .size(),
            Layout::array::<ReadSpan>(keys.len())
                .map_err(|_| overflow("bulk span metadata"))?
                .size(),
            metadata_bytes,
            paths,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(overflow("bulk constructor storage"))?;
        Ok(plan)
    }

    fn entry(
        &self,
        index: usize,
        destination: usize,
    ) -> Result<(&'a TensorMetadata, Entry<'a>), SafetensorsEncodedReadPlanError> {
        use SafetensorsEncodedReadPlanError as E;
        let key = &self.keys[index];
        let entry = self.store.catalog.get(key).ok_or(E::unknown(index))?;
        let admission = self.store.shards.admission(&entry.shard);
        let header = admission
            .header
            .get()
            .ok_or(E::unavailable(index))?
            .as_ref()
            .map_err(|_| {
                E(InspectionCause::Header {
                    index,
                    owner: Arc::clone(admission),
                })
            })?;
        let info = header.metadata.info(key).ok_or(E::unknown(index))?;
        let metadata = header.tensors.get(key).ok_or(E::unknown(index))?;
        let start = header
            .payload_offset
            .checked_add(info.data_offsets.0)
            .ok_or(E::overflow("bulk tensor offset"))?;
        let length = usize::try_from(metadata.encoded_byte_len)
            .map_err(|_| E::overflow("bulk tensor length"))?;
        let end = destination
            .checked_add(length)
            .ok_or(E::overflow("bulk output length"))?;
        let file_start = u64::try_from(start).map_err(|_| E::overflow("bulk file offset"))?;
        let file_end = file_start
            .checked_add(metadata.encoded_byte_len)
            .ok_or(E::overflow("bulk file end"))?;
        Ok((
            metadata,
            Entry {
                path: &entry.shard,
                admitted: &admission.file,
                source: file_start..file_end,
                destination: destination..end,
            },
        ))
    }

    /// Requested fresh backing and named constructor/error controls. Group path
    /// storage uses a conservative selected-path maximum; allocator-private and
    /// OS storage, existing headers/diagnostics and later read scratch are excluded.
    pub fn required_bytes<C>(&self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<PreparedEncodedRead<C>>(),
            size_of::<SafetensorsEncodedReadBuildError<C>>(),
            size_of::<Result<PreparedEncodedRead<C>, SafetensorsEncodedReadBuildError<C>>>(),
            size_of::<Vec<Entry<'_>>>(),
            size_of::<Entry<'_>>(),
            size_of::<TensorMetadata>(),
            size_of::<ReadShard>(),
            size_of::<ReadSpan>(),
            size_of::<Vec<ReadSpan>>(),
            size_of::<MutexGuard<'_, CacheState>>(),
            size_of::<Result<(), SafetensorsEncodedReadBuildCause>>(),
        ]
        .into_iter()
        .try_fold(self.backing_bytes, usize::checked_add)
    }

    /// Construct after admission, retaining the actual prefix on reserve or
    /// diagnostic failure. Source headers are never initialized by this worker.
    pub fn construct<C>(
        self,
        custody: C,
    ) -> Result<PreparedEncodedRead<C>, SafetensorsEncodedReadBuildError<C>> {
        let mut owner = PreparedEncodedRead {
            batch: EncodedReadBatch {
                tensors: Vec::new(),
                shards: Vec::new(),
                memory: Vec::new(),
                byte_len: 0,
                telemetry: Some(Arc::clone(&self.store.read_telemetry)),
                cache: Some(Arc::clone(&self.store.cache)),
            },
            _custody: custody,
        };
        let result = (|| {
            let mut entries = Vec::new();
            entries.try_reserve_exact(self.keys.len())?;
            owner.batch.tensors.try_reserve_exact(self.keys.len())?;
            owner.batch.shards.try_reserve_exact(self.max_groups)?;
            for index in 0..self.keys.len() {
                let (metadata, entry) = self
                    .entry(index, owner.batch.byte_len)
                    .expect("inspected immutable headers");
                owner.batch.byte_len = entry.destination.end;
                owner.batch.tensors.push(metadata.clone());
                entries.push(entry);
            }
            entries.sort_unstable_by(|a, b| {
                a.path
                    .cmp(b.path)
                    .then(a.destination.start.cmp(&b.destination.start))
            });
            let mut start = 0;
            while start < entries.len() {
                let first = &entries[start];
                let mut end = start + 1;
                while end < entries.len() && entries[end].path == first.path {
                    end += 1;
                }
                let mut spans = Vec::new();
                spans.try_reserve_exact(end - start)?;
                spans.extend(entries[start..end].iter().map(|row| ReadSpan {
                    source: row.source.clone(),
                    destination: row.destination.clone(),
                }));
                spans.sort_unstable_by_key(|span| span.source.start);
                spans.dedup_by(|later, earlier| {
                    if earlier.source.end == later.source.start
                        && earlier.destination.end == later.destination.start
                    {
                        earlier.source.end = later.source.end;
                        earlier.destination.end = later.destination.end;
                        true
                    } else {
                        false
                    }
                });
                if !self
                    .store
                    .cache
                    .lock()
                    .map_err(|_| SafetensorsEncodedReadBuildCause::CachePoisoned)?
                    .paths
                    .mark_touched(first.path)
                {
                    return Err(SafetensorsEncodedReadBuildCause::DiagnosticIdentity);
                }
                owner.batch.shards.push((
                    first.path.to_path_buf(),
                    ReadShard {
                        admitted: Arc::clone(first.admitted),
                        spans,
                    },
                ));
                start = end;
            }
            debug_assert_eq!(owner.batch.byte_len, self.byte_len);
            Ok(())
        })();
        match result {
            Ok(()) => Ok(owner),
            Err(cause) => Err(SafetensorsEncodedReadBuildError {
                cause,
                partial: owner,
            }),
        }
    }
}

/// Fixed construction refusal, independent of source/header error storage.
#[derive(Debug, thiserror::Error)]
pub enum SafetensorsEncodedReadBuildCause {
    /// A fresh destination could not be reserved.
    #[error("encoded file metadata reserve failed: {0}")]
    Reserve(#[from] TryReserveError),
    /// The source's existing diagnostics cannot be accessed.
    #[error("checkpoint shard cache is poisoned")]
    CachePoisoned,
    /// A path does not belong to the source's admitted diagnostic universe.
    #[error("encoded file diagnostic identity mismatch")]
    DiagnosticIdentity,
}
/// Retains the actual constructed prefix and custody through error retirement.
pub struct SafetensorsEncodedReadBuildError<C> {
    cause: SafetensorsEncodedReadBuildCause,
    partial: PreparedEncodedRead<C>,
}
impl<C> SafetensorsEncodedReadBuildError<C> {
    /// The fixed underlying reserve or diagnostic refusal.
    pub fn cause(&self) -> &SafetensorsEncodedReadBuildCause {
        &self.cause
    }
    /// Number of completed metadata records retained by this error.
    pub fn completed_tensors(&self) -> usize {
        self.partial.batch.tensors.len()
    }
}
impl<C> fmt::Debug for SafetensorsEncodedReadBuildError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SafetensorsEncodedReadBuildError")
            .field("cause", &self.cause)
            .field("partial", &self.partial)
            .finish()
    }
}
impl<C> fmt::Display for SafetensorsEncodedReadBuildError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<C> std::error::Error for SafetensorsEncodedReadBuildError<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[cfg(test)]
mod route_tests;
#[cfg(test)]
mod tests;
