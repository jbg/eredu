//! One immutable source/header layout for ordinary and paid live-cache I/O.
use super::*;
use eredu_checkpoint::safetensors::{SafetensorsHeaderError, SafetensorsHeaderPlan};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError, WorkspaceMetadataFunding};
use safetensors::tensor::{Dtype, TensorInfo};
use std::mem::{size_of, size_of_val};

/// Unchanged format, host-metadata or filesystem failure from the shared worker.
#[derive(Debug, thiserror::Error)]
pub enum CacheShardError {
    /// Exact source geometry, ordering or encoded-header failure.
    #[error(transparent)]
    Format(#[from] SafetensorsHeaderError),
    /// Original source/header metadata was not admitted.
    #[error(transparent)]
    Metadata(#[from] WorkspaceMetadataError),
    /// Payload lengths differ from the immutable source declarations.
    #[error("cache shard payload length differs from its source layout")]
    Payload,
    /// File bytes no longer have the exact retained writer's header/extent.
    #[error("cache shard bytes differ from the retained writer layout")]
    Header,
    /// The actual writer failed; the caller retains its destination/source.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
/// Actual tensor declarations, before their encoded header destination is built.
/// Source shape vectors retire before their optional original metadata account.
pub struct CacheShardMetadata {
    entries: [(&'static str, TensorInfo); 2],
    order: [usize; 2],
    funding: Option<WorkspaceMetadataFunding>,
}
#[derive(Debug)]
struct Layout {
    entries: [(&'static str, TensorInfo); 2],
    order: [usize; 2],
    header: Vec<u8>,
    file_bytes: usize,
}
/// Immutable actual writer layout. Every shared alias retires its bytes and
/// declarations before the metadata accounts; this supplies no I/O/native grant.
#[derive(Clone, Debug)]
pub struct CacheShardLayout {
    inner: Arc<Layout>,
    source_funding: Option<WorkspaceMetadataFunding>,
    header_funding: Option<WorkspaceMetadataFunding>,
}
/// Borrowed exact tensor payload selected by a retained writer layout.
pub struct CacheShardTensor<'a> {
    info: &'a TensorInfo,
    data: &'a [u8],
}
impl CacheShardTensor<'_> {
    /// Actual stored scalar encoding.
    pub fn dtype(&self) -> Dtype {
        self.info.dtype
    }
    /// Immutable stored shape, without copying a tensor view.
    pub fn shape(&self) -> &[usize] {
        &self.info.shape
    }
    /// Exact source byte span after validating the complete retained header.
    pub fn data(&self) -> &[u8] {
        self.data
    }
}
/// Canonical live/prompt-cache names for the portable cache representation.
pub fn cache_shard_tensor_names(representation: CacheRepresentation) -> [&'static str; 2] {
    match representation {
        CacheRepresentation::KeyValue => ["keys", "values"],
        CacheRepresentation::CompressedLatentRotary => ["latent", "rotary_key"],
    }
}
impl CacheShardMetadata {
    /// Exact owned source metadata contribution, before either shape allocation.
    /// Encoded header/Arc storage is queried from this completed immutable source.
    pub fn control_bytes(shapes: [&[usize]; 2]) -> Option<usize> {
        let frames = [
            shapes[0].len().checked_mul(size_of::<usize>())?,
            shapes[1].len().checked_mul(size_of::<usize>())?,
            size_of::<Self>(),
            size_of::<Result<Self, CacheShardError>>(),
            size_of::<CacheShardError>(),
            size_of::<[Vec<usize>; 2]>(),
            size_of::<[TensorInfo; 2]>(),
            size_of::<[usize; 2]>(),
            size_of::<(
                [&[usize]; 2],
                [Dtype; 2],
                [usize; 2],
                Option<&WorkspaceContext>,
            )>(),
            size_of::<std::iter::Zip<std::slice::Iter<'_, usize>, std::slice::Iter<'_, usize>>>(),
            SafetensorsHeaderPlan::control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Prepares source declarations under the actual original metadata account.
    /// This allocates no encoded file/header or numerical buffer.
    pub fn prepare(
        representation: CacheRepresentation,
        shapes: [&[usize]; 2],
        dtypes: [Dtype; 2],
        bytes: [usize; 2],
        context: &WorkspaceContext,
    ) -> Result<Self, CacheShardError> {
        if context.metadata_funding().is_none() {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        Self::create(representation, shapes, dtypes, bytes, Some(context))
    }
    /// Ordinary source construction, using the same geometry and ordering worker.
    pub fn ordinary(
        representation: CacheRepresentation,
        shapes: [&[usize]; 2],
        dtypes: [Dtype; 2],
        bytes: [usize; 2],
    ) -> Result<Self, CacheShardError> {
        Self::create(representation, shapes, dtypes, bytes, None)
    }
    fn create(
        representation: CacheRepresentation,
        shapes: [&[usize]; 2],
        dtypes: [Dtype; 2],
        bytes: [usize; 2],
        context: Option<&WorkspaceContext>,
    ) -> Result<Self, CacheShardError> {
        // Shape allocation follows the exact component charge. The encoded
        // destination remains separate until these source declarations exist.
        if let Some(context) = context {
            context.charge_metadata(
                Self::control_bytes(shapes).ok_or(WorkspaceMetadataError::Overflow)?,
            )?;
        }
        let names = cache_shard_tensor_names(representation);
        let mut order = [0, 1];
        if dtypes[1]
            .cmp(&dtypes[0])
            .then(names[0].cmp(names[1]))
            .is_gt()
        {
            order.swap(0, 1);
        }
        let [first, second] = order;
        let first_end = bytes[first];
        let end = first_end
            .checked_add(bytes[second])
            .ok_or(SafetensorsHeaderError::Overflow)?;
        let entries = [
            (
                names[first],
                TensorInfo {
                    dtype: dtypes[first],
                    shape: shapes[first].to_vec(),
                    data_offsets: (0, first_end),
                },
            ),
            (
                names[second],
                TensorInfo {
                    dtype: dtypes[second],
                    shape: shapes[second].to_vec(),
                    data_offsets: (first_end, end),
                },
            ),
        ];
        SafetensorsHeaderPlan::prepare(&entries)?;
        Ok(Self {
            entries,
            order,
            funding: context.and_then(WorkspaceContext::metadata_funding),
        })
    }
    /// Exact second-stage header destination/owner contribution from this source.
    pub fn layout_control_bytes(&self) -> Result<usize, CacheShardError> {
        let plan = SafetensorsHeaderPlan::prepare(&self.entries)?;
        let frames = [
            plan.header_bytes(),
            WorkspaceContext::metadata_arc_bytes::<Layout>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
            SafetensorsHeaderPlan::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
            size_of::<CacheShardLayout>(),
            size_of::<Layout>(),
            size_of::<Vec<u8>>(),
            size_of::<Result<CacheShardLayout, CacheShardError>>(),
            size_of::<Result<(), CacheShardError>>(),
            size_of::<Result<(), std::io::Error>>(),
            size_of::<[&[u8]; 2]>(),
            size_of::<Option<WorkspaceMetadataFunding>>(),
            size_of::<[CacheShardTensor<'_>; 2]>(),
            size_of::<Result<[CacheShardTensor<'_>; 2], CacheShardError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or_else(|| WorkspaceMetadataError::Overflow.into())
    }
    /// Builds the exact encoded header under a separately admitted account.
    /// Both original metadata owners remain in every resulting layout alias.
    pub fn into_prepared_layout(
        self,
        context: &WorkspaceContext,
    ) -> Result<CacheShardLayout, CacheShardError> {
        if self.funding.is_none() || context.metadata_funding().is_none() {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        self.into_layout(Some(context))
    }
    /// Ordinary construction through the same encoder; original sources cannot
    /// silently drop their required prepared-header producer.
    pub fn into_ordinary_layout(self) -> Result<CacheShardLayout, CacheShardError> {
        if self.funding.is_some() {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        self.into_layout(None)
    }
    fn into_layout(
        self,
        context: Option<&WorkspaceContext>,
    ) -> Result<CacheShardLayout, CacheShardError> {
        if let Some(context) = context {
            context.charge_metadata(self.layout_control_bytes()?)?;
        }
        let plan = SafetensorsHeaderPlan::prepare(&self.entries)?;
        let mut header = vec![0; plan.header_bytes()];
        plan.write_header(&mut header)?;
        let file_bytes = plan.file_bytes();
        drop(plan);
        Ok(CacheShardLayout {
            inner: Arc::new(Layout {
                entries: self.entries,
                order: self.order,
                header,
                file_bytes,
            }),
            source_funding: self.funding,
            header_funding: context.and_then(WorkspaceContext::metadata_funding),
        })
    }
}
impl CacheShardLayout {
    /// Actual header plus payload file extent, never inferred from a caller quota.
    pub fn file_bytes(&self) -> usize {
        self.inner.file_bytes
    }
    /// Tests whether both declarations share this exact immutable writer layout.
    /// Equal shapes or byte lengths cannot substitute for the retained owner.
    pub fn same_layout(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
    /// The exact header produced once by the shared schema encoder.
    pub fn header(&self) -> &[u8] {
        &self.inner.header
    }
    /// Borrows this writer's actual shape, scalar encoding and payload length
    /// in portable representation order. This declares destination geometry;
    /// reading still requires the exact file-version and header validation.
    pub fn tensor_metadata(&self) -> [(&[usize], Dtype, usize); 2] {
        let mut metadata = [(&[][..], Dtype::F32, 0); 2];
        for (entry, source) in self.inner.entries.iter().zip(self.inner.order) {
            let info = &entry.1;
            metadata[source] = (
                &info.shape,
                info.dtype,
                info.data_offsets.1 - info.data_offsets.0,
            );
        }
        metadata
    }
    /// Writes an already prepared unique destination using the same header/order.
    /// The caller owns File/path/source/queue resources and incomplete-write cleanup.
    pub fn write_to(&self, file: &mut File, payloads: [&[u8]; 2]) -> Result<(), CacheShardError> {
        for (info, source) in self.inner.entries.iter().zip(self.inner.order) {
            if payloads[source].len() != info.1.data_offsets.1 - info.1.data_offsets.0 {
                return Err(CacheShardError::Payload);
            }
        }
        file.write_all(&self.inner.header)?;
        for source in self.inner.order {
            file.write_all(payloads[source])?;
        }
        file.flush()?;
        Ok(())
    }
    /// Validates this writer's complete actual header/extent and lends tensors
    /// in the original portable representation order, without parsing a new map.
    pub fn tensors<'a>(
        &'a self,
        bytes: &'a [u8],
    ) -> Result<[CacheShardTensor<'a>; 2], CacheShardError> {
        if bytes.len() != self.inner.file_bytes
            || bytes.get(..self.inner.header.len()) != Some(self.inner.header.as_slice())
        {
            return Err(CacheShardError::Header);
        }
        let payload = &bytes[self.inner.header.len()..];
        let index = |original| {
            self.inner
                .order
                .iter()
                .position(|source| *source == original)
                .expect("two source permutation")
        };
        let tensor = |index: usize| {
            let info = &self.inner.entries[index].1;
            CacheShardTensor {
                info,
                data: &payload[info.data_offsets.0..info.data_offsets.1],
            }
        };
        Ok([tensor(index(0)), tensor(index(1))])
    }
    /// Exact canonical names in the original representation order.
    pub fn names(&self) -> [&'static str; 2] {
        let mut names = ["", ""];
        for (index, source) in self.inner.order.iter().enumerate() {
            names[*source] = self.inner.entries[index].0;
        }
        names
    }
}
#[cfg(test)]
#[path = "shard/tests.rs"]
mod tests;
