//! Caller-owned raw storage at the original G1 preparation boundary.
use super::*;

/// Owned initialized bytes. Capacity is an observation, never admission evidence.
/// Implementations retain any allocation receipt until their bytes are destroyed.
pub trait GgufRawStorage: AsRef<[u8]> + AsMut<[u8]> + std::fmt::Debug + 'static {
    /// Actual exposed storage capacity in bytes.
    fn capacity(&self) -> usize;
}
impl GgufRawStorage for Vec<u8> {
    fn capacity(&self) -> usize {
        Vec::capacity(self)
    }
}

/// Borrowed from the actual consumed lease, after checked G1 layout validation.
#[derive(Debug)]
pub struct GgufRawStorageRequest<'a> {
    pub(super) lease: &'a GgufLease,
    pub(super) layout: Layout,
}
impl GgufRawStorageRequest<'_> {
    /// Exact physical encoded-byte layout from this lease's retained read proof.
    pub fn layout(&self) -> Layout {
        self.layout
    }
    /// The actual source/selection; no descriptor, key or plan is cloned.
    pub fn lease(&self) -> &GgufLease {
        self.lease
    }
}

/// Storage mechanism only. Called before cache entry/I/O at the G1 reserve point.
/// It must retain all allocated prefixes in its returned storage or error.
pub trait GgufRawStorageProvider {
    /// Initialized final destination, including any required allocation custody.
    type Storage: GgufRawStorage;
    /// Owning pre-allocation refusal or partial destination failure.
    type Error: std::error::Error + 'static;
    /// Construct the one requested destination without acquiring another source.
    fn prepare(self, request: GgufRawStorageRequest<'_>) -> Result<Self::Storage, Self::Error>;
}

/// Actual failure with the same Box and source/destination owners.
#[derive(Debug)]
pub enum PreparedGgufSuppliedFailure<S: GgufRawStorage, E: std::error::Error + 'static> {
    /// G1 layout is not representable; the provider was not called.
    Layout {
        /// Same consumed lease allocation.
        lease: Box<GgufLease>,
    },
    /// The provider's owning refusal/prefix, before any read or conversion.
    Provider {
        /// Actual provider cause and retained prefix.
        cause: E,
        /// Same consumed lease allocation.
        lease: Box<GgufLease>,
    },
    /// Supplied initialized length/capacity differs from the genuine request.
    Extent {
        /// Required initialized encoded-byte length.
        expected: usize,
        /// Actual complete destination, not a successful read.
        storage: S,
        /// Same consumed lease allocation.
        lease: Box<GgufLease>,
    },
    /// Existing G2/G3/reader failure with supplied raw storage retained.
    Prepared(PreparedGgufBoxedFailure<S>),
}
impl<S: GgufRawStorage, E: std::error::Error + 'static> PreparedGgufSuppliedFailure<S, E> {
    /// Exact source lease, including each failure stage.
    pub fn lease(&self) -> &GgufLease {
        match self {
            Self::Layout { lease } | Self::Provider { lease, .. } | Self::Extent { lease, .. } => {
                lease
            }
            Self::Prepared(failure) => failure.lease(),
        }
    }
}
impl<S: GgufRawStorage, E: std::error::Error + 'static> std::fmt::Display
    for PreparedGgufSuppliedFailure<S, E>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Layout { .. } => f.write_str("GGUF raw destination layout is not representable"),
            Self::Provider { cause, .. } => std::fmt::Display::fmt(cause, f),
            Self::Extent {
                expected, storage, ..
            } => write!(
                f,
                "GGUF raw destination has {} bytes; expected {expected}",
                storage.as_ref().len()
            ),
            Self::Prepared(cause) => std::fmt::Display::fmt(cause, f),
        }
    }
}
impl<S: GgufRawStorage, E: std::error::Error + 'static> std::error::Error
    for PreparedGgufSuppliedFailure<S, E>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider { cause, .. } => Some(cause),
            Self::Prepared(cause) => Some(cause),
            _ => None,
        }
    }
}

impl GgufLease {
    /// Same G1/G2/G3 driver and same Box, with caller-owned admitted G1 storage.
    /// The provider is invoked only after the original G1 layout checks and before
    /// later descriptor/planning/conversion errors. No cache/I/O effect is added.
    pub fn materialize_prepared_boxed_with_storage<P: GgufRawStorageProvider>(
        lease: Box<Self>,
        provider: P,
    ) -> Result<
        (ConvertedCheckpointTensor, Box<Self>),
        PreparedGgufSuppliedFailure<P::Storage, P::Error>,
    > {
        let layout = usize::try_from(lease.proof.length_bytes)
            .ok()
            .and_then(|len| Layout::array::<u8>(len).ok());
        let Some(layout) = layout else {
            return Err(PreparedGgufSuppliedFailure::Layout { lease });
        };
        let raw = match provider.prepare(GgufRawStorageRequest {
            lease: &lease,
            layout,
        }) {
            Ok(storage) => storage,
            Err(cause) => return Err(PreparedGgufSuppliedFailure::Provider { cause, lease }),
        };
        if raw.as_ref().len() != layout.size() || raw.capacity() < layout.size() {
            return Err(PreparedGgufSuppliedFailure::Extent {
                expected: layout.size(),
                storage: raw,
                lease,
            });
        }
        let prepared = PreparedGgufRead {
            raw,
            layout,
            lease: LeaseOwner::Boxed(lease),
        }
        .prepare_conversion()
        .map_err(|e| {
            PreparedGgufSuppliedFailure::Prepared(PreparedGgufBoxedFailure::Conversion(e))
        })?
        .prepare_result_metadata()
        .map_err(|e| PreparedGgufSuppliedFailure::Prepared(PreparedGgufBoxedFailure::Tensor(e)))?;
        let (output, owner) = prepared.materialize_retaining().map_err(|e| {
            PreparedGgufSuppliedFailure::Prepared(PreparedGgufBoxedFailure::Tensor(e))
        })?;
        let lease = owner.into_boxed_lease();
        Ok((output, lease))
    }
}
