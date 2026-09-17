//! Final lease storage using the actual shared SafeTensors cache.
use super::*;
use std::{alloc::Layout, collections::TryReserveError};

/// Actual selected source/key loan; only source routing constructs this view.
#[derive(Clone, Copy, Debug)]
pub struct SafetensorsLeaseSource<'a> {
    store: &'a SafetensorsWeightStore,
    key: &'a str,
    metadata: &'a TensorMetadata,
}
impl<'a> SafetensorsLeaseSource<'a> {
    pub(super) fn new(
        store: &'a SafetensorsWeightStore,
        key: &'a str,
        metadata: &'a TensorMetadata,
    ) -> Self {
        Self {
            store,
            key,
            metadata,
        }
    }
    /// Prepare owned final fields and a unique payload destination. This performs
    /// no shard-cache acquisition or payload I/O. The caller must fund these
    /// allocations; neither this source view nor its destination grants a budget.
    pub fn prepare(
        self,
        selection: TensorSelection,
        policy: ReadPolicy,
    ) -> Result<PreparedSafetensorsLease, PreparedSafetensorsLeaseFailure> {
        self.prepare_with_payload(selection, policy, true)
    }
    pub(crate) fn prepare_with_payload(
        self,
        selection: TensorSelection,
        policy: ReadPolicy,
        allocate_payload: bool,
    ) -> Result<PreparedSafetensorsLease, PreparedSafetensorsLeaseFailure> {
        let entry = &self.store.catalog[self.key];
        let mut destination = Destination {
            key: self.key.to_owned(),
            metadata: self.metadata.clone(),
            selection,
            policy,
            output_shape: Vec::new(),
            ranges: Vec::new(),
            bytes: None,
            geometry: None,
            source: SourceOwner {
                cache: Arc::clone(&self.store.cache),
                maximum: self.store.max_cached_shards,
                admission: Arc::clone(self.store.shards.admission(&entry.shard)),
                telemetry: Arc::clone(&self.store.read_telemetry),
            },
        };
        match destination.prepare(allocate_payload) {
            Ok(()) => Ok(PreparedSafetensorsLease(destination)),
            Err(cause) => Err(PreparedSafetensorsLeaseFailure {
                cause,
                destination,
                shard: None,
                payload: None,
            }),
        }
    }
}

#[derive(Debug)]
struct SourceOwner {
    cache: Arc<Mutex<CacheState>>,
    maximum: usize,
    admission: Arc<AdmittedShard>,
    telemetry: Arc<SafetensorsReadTelemetry>,
}
#[derive(Clone, Copy, Debug)]
struct Geometry {
    header_payload_start: usize,
    payload_start: usize,
    tensor_len: usize,
    byte_layout: Layout,
    physically_bounded: bool,
    complete_tensor: bool,
}
#[derive(Debug)]
struct Destination {
    key: String,
    metadata: TensorMetadata,
    selection: TensorSelection,
    policy: ReadPolicy,
    output_shape: Vec<usize>,
    ranges: Vec<Range<usize>>,
    bytes: Option<Arc<Vec<u8>>>,
    geometry: Option<Geometry>,
    source: SourceOwner,
}
impl Destination {
    fn path(&self) -> &PathBuf {
        self.metadata
            .backing_shard
            .as_ref()
            .expect("actual SafeTensors metadata")
    }
    fn prepare(&mut self, allocate_payload: bool) -> Result<(), FailureCause> {
        // The source loan already required this retained header to exist.
        let header = self.source.admission.header(self.path())?;
        let info = header.metadata.info(&self.key).ok_or_else(|| {
            io_error(
                self.path(),
                format!("shard does not contain tensor {:?}", self.key),
            )
        })?;
        self.output_shape =
            validate_selection(&self.key, &self.metadata.logical_shape, &self.selection)?;
        let payload_start = header
            .payload_offset
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("payload start for {:?}", self.key),
            })?;
        let tensor_len = info
            .data_offsets
            .1
            .checked_sub(info.data_offsets.0)
            .ok_or_else(|| io_error(self.path(), "tensor payload offsets descend"))?;
        let read = plan_safetensors_reads(
            &self.key,
            info.dtype,
            &info.shape,
            tensor_len,
            &self.selection,
            &self.output_shape,
            self.policy,
        )?;
        self.ranges = read.ranges;
        let length = self.ranges.iter().try_fold(0usize, |total, range| {
            total
                .checked_add(range.len())
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("selected payload length for {}", self.path().display()),
                })
        })?;
        let byte_layout = Layout::array::<u8>(length).map_err(|_| StoreError::Overflow {
            context: format!("selected payload length for {}", self.path().display()),
        })?;
        self.geometry = Some(Geometry {
            header_payload_start: header.payload_offset,
            payload_start,
            tensor_len,
            byte_layout,
            physically_bounded: read.physically_bounded,
            complete_tensor: self.ranges.len() == 1
                && self.ranges[0].start == 0
                && self.ranges[0].end == tensor_len,
        });
        if allocate_payload {
            Self::ensure_bytes(&mut self.bytes, byte_layout.size())?;
        }
        Ok(())
    }
    fn ensure_bytes(bytes: &mut Option<Arc<Vec<u8>>>, length: usize) -> Result<(), FailureCause> {
        if bytes.is_some() {
            return Ok(());
        }
        *bytes = Some(Arc::new(Vec::new()));
        let bytes = Arc::get_mut(bytes.as_mut().unwrap()).expect("unpublished final buffer");
        bytes
            .try_reserve_exact(length)
            .map_err(FailureCause::Reserve)?;
        bytes.resize(length, 0);
        Ok(())
    }
}

/// A move-only final destination tied to the exact existing cache/admission.
/// Cache maps, path diagnostics and OS ownership still need their own managed
/// memory bounds. This type does not report complete original readiness.
#[derive(Debug)]
pub struct PreparedSafetensorsLease(Destination);
impl PreparedSafetensorsLease {
    pub(crate) fn retained_payload_capacity(&self) -> Option<(usize, usize)> {
        use super::acquisition::storage::{metadata, selection, vector};
        let bytes = self
            .0
            .key
            .capacity()
            .checked_add(metadata(&self.0.metadata)?)?
            .checked_add(selection(&self.0.selection)?)?
            .checked_add(vector::<usize>(self.0.output_shape.capacity())?)?
            .checked_add(vector::<Range<usize>>(self.0.ranges.capacity())?)?
            .checked_add(self.0.bytes.as_ref().map_or(0, |bytes| bytes.capacity()))?
            .checked_add(if self.0.bytes.is_some() {
                std::mem::size_of::<Vec<u8>>()
            } else {
                0
            })?;
        Some((bytes, usize::from(self.0.bytes.is_some())))
    }

    /// Verify the actual routed source and request without cloning or reading.
    pub fn matches_request(
        &self,
        source: SafetensorsLeaseSource<'_>,
        selection: &TensorSelection,
        policy: ReadPolicy,
    ) -> bool {
        Arc::ptr_eq(&self.0.source.cache, &source.store.cache)
            && self.0.key == source.key
            && self.0.selection == *selection
            && matches!(
                (self.0.policy, policy),
                (ReadPolicy::RequireBounded, ReadPolicy::RequireBounded)
                    | (
                        ReadPolicy::AllowFullTensorRead,
                        ReadPolicy::AllowFullTensorRead
                    )
            )
    }
    /// Actual prepared payload layout, excluding the cache and source owners.
    pub fn destination_layout(&self) -> Layout {
        self.0.geometry.unwrap().byte_layout
    }
    /// Consume into the existing format-erased checkpoint lease used by native
    /// materialization. Format dispatch remains in this neutral source owner.
    pub fn acquire_checkpoint(self) -> Result<CheckpointLease, PreparedSafetensorsLeaseFailure> {
        self.acquire().map(CheckpointLease::Safetensors)
    }
    /// Execute against the same shared cache and return the existing genuine
    /// lease type. Cache allocations are explicit existing operations; no second
    /// cache, retry, or allocating payload fallback is introduced.
    pub fn acquire(self) -> Result<SafetensorsLease, PreparedSafetensorsLeaseFailure> {
        let mut destination = self.0;
        let mut shard = None;
        let mut payload = None;
        let result = (|| -> Result<BoundedReadProof, FailureCause> {
            let geometry = destination.geometry.expect("prepared geometry");
            shard = Some(cache_policy::acquire_shard(
                &destination.source.cache,
                destination.source.maximum,
                destination.path(),
                || Arc::clone(&destination.source.admission),
            )?);
            let shard = shard.as_ref().unwrap();
            // Prepared metadata is the same immutable admitted header value.
            // Its ordinary cache touch retains its original relative position.
            cache_policy::touch_metadata(&destination.source.cache, destination.path())?;
            let cached = cache_policy::lookup(shard, &destination.key)?;
            let cache_hit = cached.is_some();
            let cached = cached
                .map(cache_policy::CachedPayload::validate)
                .transpose()?;
            if let Some(cached) = cached {
                if geometry.complete_tensor {
                    payload = Some(cached.into_bytes());
                } else {
                    // A failed selected copy retains its actual source payload
                    // alongside the final partial destination.
                    payload = Some(cached.retained_bytes());
                    Destination::ensure_bytes(&mut destination.bytes, geometry.byte_layout.size())?;
                    let plan = SafetensorsBytePlan::new(
                        read_bytes::ReadFileSource {
                            path: destination.metadata.backing_shard.as_ref().unwrap(),
                            admitted: &shard.admitted_file,
                            header_payload_start: geometry.header_payload_start,
                            telemetry: &destination.source.telemetry,
                        },
                        &destination.key,
                        geometry.payload_start,
                        geometry.tensor_len,
                        &destination.ranges,
                        geometry.byte_layout,
                        geometry.physically_bounded,
                    );
                    let bytes = Arc::get_mut(destination.bytes.as_mut().unwrap())
                        .expect("unpublished final buffer");
                    let copied = plan
                        .copy_validated_cache(&cached, bytes)
                        .map_err(|error| FailureCause::Byte(error.into_owned()))?;
                    drop(copied);
                    payload = Some(Arc::clone(destination.bytes.as_ref().unwrap()));
                }
            } else {
                Destination::ensure_bytes(&mut destination.bytes, geometry.byte_layout.size())?;
                let plan = SafetensorsBytePlan::new(
                    read_bytes::ReadFileSource {
                        path: destination.metadata.backing_shard.as_ref().unwrap(),
                        admitted: &shard.admitted_file,
                        header_payload_start: geometry.header_payload_start,
                        telemetry: &destination.source.telemetry,
                    },
                    &destination.key,
                    geometry.payload_start,
                    geometry.tensor_len,
                    &destination.ranges,
                    geometry.byte_layout,
                    geometry.physically_bounded,
                );
                let bytes = Arc::get_mut(destination.bytes.as_mut().unwrap())
                    .expect("unpublished final buffer");
                let read = plan
                    .read_into(bytes)
                    .map_err(|error| FailureCause::Byte(error.into_owned()))?;
                drop(read);
                payload = Some(Arc::clone(destination.bytes.as_ref().unwrap()));
                if geometry.complete_tensor {
                    cache_policy::publish_full(shard, &destination.key, payload.as_ref().unwrap())?;
                }
            }
            let length = u64::try_from(payload.as_ref().unwrap().len()).map_err(|_| {
                StoreError::Overflow {
                    context: format!("physical read length for {:?}", destination.key),
                }
            })?;
            cache_policy::publish_payload(&destination.source.cache, shard)?;
            Ok(BoundedReadProof {
                physically_bounded: geometry.physically_bounded,
                offset_bytes: u64::try_from(destination.ranges[0].start).map_err(|_| {
                    StoreError::Overflow {
                        context: "selection byte offset".into(),
                    }
                })?,
                length_bytes: length,
                physical_reads: if cache_hit {
                    0
                } else {
                    u64::try_from(destination.ranges.len()).unwrap_or(u64::MAX)
                },
                physical_read_bytes: if cache_hit { 0 } else { length },
            })
        })();
        match result {
            Ok(proof) => Ok(SafetensorsLease {
                metadata: destination.metadata,
                selection: destination.selection,
                output_shape: destination.output_shape,
                proof,
                shard: shard.unwrap(),
                bytes: payload.unwrap(),
            }),
            Err(cause) => Err(PreparedSafetensorsLeaseFailure {
                cause,
                destination,
                shard,
                payload,
            }),
        }
    }
}

#[derive(Debug)]
enum FailureCause {
    Store(StoreError),
    Reserve(TryReserveError),
    Byte(read_bytes::OwnedByteFailure),
}
impl From<StoreError> for FailureCause {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}
/// Failed preparation or fill retaining the final partial destination and actual
/// shared cache/source/shard/payload owners. Drop is the only release operation;
/// there is no implicit retry or success conversion.
#[derive(Debug)]
pub struct PreparedSafetensorsLeaseFailure {
    cause: FailureCause,
    destination: Destination,
    shard: Option<Arc<CachedShard>>,
    payload: Option<Arc<Vec<u8>>>,
}
impl PreparedSafetensorsLeaseFailure {
    /// Actual original ordinary store cause, if the shared cache rejected work.
    pub fn store_error(&self) -> Option<&StoreError> {
        match &self.cause {
            FailureCause::Store(error) => Some(error),
            _ => None,
        }
    }
    /// Failed final storage; these bytes are not a successfully acquired tensor.
    pub fn destination(&self) -> &[u8] {
        self.destination.bytes.as_deref().map_or(&[], Vec::as_slice)
    }
    /// Whole read ranges completed before a byte failure; final Changed errors
    /// invalidate even this prefix.
    pub fn completed_bytes(&self) -> usize {
        match &self.cause {
            FailureCause::Byte(error) => error.completed_bytes(),
            _ => 0,
        }
    }
    /// Extent initialized by byte execution; a failing read_exact may modify an
    /// unknown subset of its final attempted range.
    pub fn initialized_bytes(&self) -> usize {
        match &self.cause {
            FailureCause::Byte(error) => error.initialized_bytes(),
            _ => 0,
        }
    }
    /// A byte-read failure retains the actual opened file until error retirement.
    pub fn retains_file(&self) -> bool {
        matches!(&self.cause, FailureCause::Byte(error) if error.retains_file())
    }
    /// Whether acquisition already pinned its actual shared cached shard.
    pub fn retains_shard(&self) -> bool {
        self.shard.is_some()
    }
    /// Whether a completed or reused payload remains retained by a later failure.
    pub fn retains_payload(&self) -> bool {
        self.payload.is_some()
    }
}
impl std::fmt::Display for PreparedSafetensorsLeaseFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cause {
            FailureCause::Store(error) => std::fmt::Display::fmt(error, f),
            FailureCause::Reserve(error) => std::fmt::Display::fmt(error, f),
            FailureCause::Byte(error) => {
                error.fmt_at(self.destination.path(), &self.destination.key, f)
            }
        }
    }
}
impl std::error::Error for PreparedSafetensorsLeaseFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            FailureCause::Store(error) => Some(error),
            FailureCause::Reserve(error) => Some(error),
            FailureCause::Byte(error) => error.source(),
        }
    }
}

#[cfg(test)]
mod tests;
