//! The caller's same boxed lease accompanies every preparation/fill outcome.
use super::*;

/// Owns the actual failing G1, G2 or G3 destination, including the caller's
/// unchanged lease allocation. No successful result or implicit retry is exposed.
#[derive(Debug)]
pub enum PreparedGgufBoxedFailure<S: GgufRawStorage = Vec<u8>> {
    /// Raw layout/reserve preparation failed.
    Read(PreparedGgufReadFailure<S>),
    /// Conversion preparation failed after raw preparation.
    Conversion(PreparedGgufConversionFailure<S>),
    /// Metadata preparation or the shared read/conversion worker failed.
    Tensor(PreparedGgufTensorFailure<S>),
}
impl<S: GgufRawStorage> PreparedGgufBoxedFailure<S> {
    /// Exact boxed source retained until all failed destinations retire.
    pub fn lease(&self) -> &GgufLease {
        match self {
            Self::Read(e) => e.lease(),
            Self::Conversion(e) => e.lease(),
            Self::Tensor(e) => e.lease(),
        }
    }
    /// Original checkpoint cause, when that operation failed.
    pub fn store_error(&self) -> Option<&StoreError> {
        match self {
            Self::Read(e) => e.store_error(),
            Self::Conversion(e) => e.store_error(),
            Self::Tensor(e) => e.store_error(),
        }
    }
}
impl<S: GgufRawStorage> std::fmt::Display for PreparedGgufBoxedFailure<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(e) => e.fmt(f),
            Self::Conversion(e) => e.fmt(f),
            Self::Tensor(e) => e.fmt(f),
        }
    }
}
impl<S: GgufRawStorage> std::error::Error for PreparedGgufBoxedFailure<S> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Read(e) => e,
            Self::Conversion(e) => e,
            Self::Tensor(e) => e,
        })
    }
}
impl GgufLease {
    /// Uses the existing G1/G2/G3 preparation and shared cache worker, returning
    /// the identical boxed lease on success. Every failure retains that box and
    /// its actual partial destinations. The caller funds actual miss storage;
    /// neither this operation nor its observed capacities grant a byte budget.
    pub fn materialize_prepared_boxed(
        lease: Box<Self>,
    ) -> Result<(ConvertedCheckpointTensor, Box<Self>), PreparedGgufBoxedFailure> {
        let prepared = PreparedGgufRead::prepare(LeaseOwner::Boxed(lease))
            .map_err(PreparedGgufBoxedFailure::Read)?
            .prepare_conversion()
            .map_err(PreparedGgufBoxedFailure::Conversion)?
            .prepare_result_metadata()
            .map_err(PreparedGgufBoxedFailure::Tensor)?;
        let (output, owner) = prepared
            .materialize_retaining()
            .map_err(PreparedGgufBoxedFailure::Tensor)?;
        let lease = owner.into_boxed_lease();
        // Remaining metadata/conversion/raw scratch is destroyed here; the
        // output and original box move to the caller without source cloning.
        Ok((output, lease))
    }
}
