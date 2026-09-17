//! Owning/supplied metadata for one actual physical group, without source I/O.
use super::*;
use eredu_gguf::{
    StorageFamily, StorageProvider, StorageRequestBound, StoredBuffer, SuppliedStorageError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Selection<'a> {
    Full,
    Range {
        axis: usize,
        start: usize,
        end: usize,
    },
    Indices {
        axis: usize,
        indices: &'a [usize],
    },
    Span {
        offset: u64,
        shape: &'a [u64],
    },
}
/// Borrowed physical identity, shared by ordinary and supplied cache keys.
/// Comparison consumes no allocation and retains no additional source owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GgufCacheIdentityView<'a> {
    source: &'a crate::store::SourceStorageIdentity,
    checkpoint: usize,
    name: &'a str,
    selection: Selection<'a>,
}
impl GgufLeaseIdentity {
    /// Borrow the exact source and normalized physical selection.
    pub fn cache_view(&self) -> GgufCacheIdentityView<'_> {
        GgufCacheIdentityView {
            source: &self.source,
            checkpoint: self.checkpoint,
            name: &self.physical_name,
            selection: match &self.selection {
                None => Selection::Full,
                Some(GgufPhysicalSelection::Axis(GgufTensorSelection::Range {
                    axis,
                    start,
                    end,
                })) => Selection::Range {
                    axis: *axis,
                    start: *start,
                    end: *end,
                },
                Some(GgufPhysicalSelection::Axis(GgufTensorSelection::Indices {
                    axis,
                    indices,
                })) => Selection::Indices {
                    axis: *axis,
                    indices,
                },
                Some(GgufPhysicalSelection::DenseSpan(span)) => Selection::Span {
                    offset: span.offset_elements(),
                    shape: span.shape(),
                },
            },
        }
    }
    /// Actual reached key-buffer requests; no output shape/payload is inferred.
    pub fn cache_storage_requests(&self) -> Option<StorageRequestBound> {
        let name = StorageRequestBound::one::<u8>(self.physical_name.len())?;
        match &self.selection {
            Some(GgufPhysicalSelection::Axis(GgufTensorSelection::Indices { indices, .. })) => {
                name.checked_add(StorageRequestBound::one::<usize>(indices.len())?)
            }
            Some(GgufPhysicalSelection::DenseSpan(span)) => {
                name.checked_add(StorageRequestBound::one::<u64>(span.shape().len())?)
            }
            _ => Some(name),
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum Kind {
    Full,
    Range {
        axis: usize,
        start: usize,
        end: usize,
    },
    Indices {
        axis: usize,
    },
    Span {
        offset: u64,
    },
}
/// Same physical identity with intact supplied buffers. The source token is
/// weak; its retained shared shell remains a separate source-lifetime term.
#[derive(Debug)]
pub struct StoredGgufCacheIdentity<F: StorageFamily> {
    source: crate::store::SourceStorageIdentity,
    checkpoint: usize,
    name: StoredBuffer<F, u8>,
    indices: StoredBuffer<F, usize>,
    shape: StoredBuffer<F, u64>,
    kind: Kind,
    complete: bool,
}
/// Failed key preparation, preserving every supplied owner and the actual cause.
#[derive(Debug)]
pub struct StoredGgufCacheIdentityFailure<F: StorageFamily> {
    cause: SuppliedStorageError<F::Error>,
    identity: StoredGgufCacheIdentity<F>,
}
impl<F: StorageFamily> StoredGgufCacheIdentityFailure<F> {
    /// Move the exact cause and partial destination without cloning a source.
    pub fn into_parts(self) -> (SuppliedStorageError<F::Error>, StoredGgufCacheIdentity<F>) {
        (self.cause, self.identity)
    }
}
impl<F: StorageFamily> StoredGgufCacheIdentity<F> {
    /// Actual named preparation/return transports; providers price their own
    /// calls and backing separately. Name then selection copies are sequential.
    pub fn preparation_control_bytes() -> Option<usize> {
        let copy = StoredBuffer::<F, u8>::copy_control_bytes()?
            .max(StoredBuffer::<F, usize>::copy_control_bytes()?)
            .max(StoredBuffer::<F, u64>::copy_control_bytes()?);
        [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, StoredGgufCacheIdentityFailure<F>>>(),
            std::mem::size_of::<StoredGgufCacheIdentityFailure<F>>(),
            std::mem::size_of::<Result<(), SuppliedStorageError<F::Error>>>(),
            std::mem::size_of::<GgufCacheIdentityView<'static>>(),
            std::mem::size_of::<(Kind, &mut Self, &GgufLeaseIdentity, &mut ())>(),
            copy,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    /// Prepare only the reached key buffers, preserving the provider's refusal.
    /// The caller must reserve the finite construction before invoking this.
    pub fn prepare<P: StorageProvider<Family = F>>(
        source: &GgufLeaseIdentity,
        provider: &mut P,
    ) -> Result<Self, StoredGgufCacheIdentityFailure<F>> {
        let mut out = Self {
            source: source.source.clone(),
            checkpoint: source.checkpoint,
            complete: false,
            name: StoredBuffer::default(),
            indices: StoredBuffer::default(),
            shape: StoredBuffer::default(),
            kind: match source.cache_view().selection {
                Selection::Full => Kind::Full,
                Selection::Range { axis, start, end } => Kind::Range { axis, start, end },
                Selection::Indices { axis, .. } => Kind::Indices { axis },
                Selection::Span { offset, .. } => Kind::Span { offset },
            },
        };
        let result = (|| {
            copy(&mut out.name, provider, source.physical_name.as_bytes(), 0)?;
            match source.cache_view().selection {
                Selection::Indices { indices, .. } => copy(&mut out.indices, provider, indices, 0)?,
                Selection::Span { shape, .. } => copy(&mut out.shape, provider, shape, 0)?,
                _ => {}
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                out.complete = true;
                Ok(out)
            }
            Err(cause) => Err(StoredGgufCacheIdentityFailure {
                cause,
                identity: out,
            }),
        }
    }
    /// Borrow this key; successful preparation preserves the original UTF-8.
    /// Failed destinations are retained for retirement and must not be indexed.
    pub fn cache_view(&self) -> GgufCacheIdentityView<'_> {
        assert!(
            self.complete,
            "failed cache-key destinations cannot be indexed"
        );
        GgufCacheIdentityView {
            source: &self.source,
            checkpoint: self.checkpoint,
            name: std::str::from_utf8(self.name.as_slice()).expect("copied source name"),
            selection: match self.kind {
                Kind::Full => Selection::Full,
                Kind::Range { axis, start, end } => Selection::Range { axis, start, end },
                Kind::Indices { axis } => Selection::Indices {
                    axis,
                    indices: self.indices.as_slice(),
                },
                Kind::Span { offset } => Selection::Span {
                    offset,
                    shape: self.shape.as_slice(),
                },
            },
        }
    }
}
fn copy<F: StorageFamily, P: StorageProvider<Family = F>, T: Copy + std::fmt::Debug + 'static>(
    out: &mut StoredBuffer<F, T>,
    provider: &mut P,
    source: &[T],
    initial: T,
) -> Result<(), SuppliedStorageError<F::Error>> {
    match StoredBuffer::try_copy(provider, source, initial) {
        Ok(values) => {
            *out = values;
            Ok(())
        }
        Err((cause, values)) => {
            *out = values;
            Err(cause)
        }
    }
}
