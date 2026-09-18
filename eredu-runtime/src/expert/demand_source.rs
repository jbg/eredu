//! Completed demand metadata stays with its issuing source through bank use.
use eredu_core::ErasedSharedStorageOwner;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

#[derive(Debug, Clone)]
struct Custody {
    source: ErasedSharedStorageOwner,
    funding: HostMetadataFunding,
}

/// One actual ordered demand result. This owner grants no acquisition, native
/// execution or completion authority; a backend still validates its own source.
/// Demand storage retires before its issuing source and metadata account.
#[derive(Debug)]
pub struct IndexedDemandSource {
    demands: Vec<(usize, u64)>,
    custody: Option<Custody>,
}
impl IndexedDemandSource {
    /// Wraps the existing ordinary discovery result without allocation.
    pub fn ordinary(demands: Vec<(usize, u64)>) -> Self {
        Self { demands, custody: None }
    }
    /// Moves a completed producer's already-funded demand destination and exact
    /// source into the shared driver. The producer must pay the actual vector
    /// capacity and `control_bytes` before creating these owners. This method
    /// copies no data and does not infer or reconstruct a source from counts.
    pub fn from_prepared(demands: Vec<(usize, u64)>, source: ErasedSharedStorageOwner,
        funding: HostMetadataFunding) -> Self {
        Self { demands, custody: Some(Custody { source, funding }) }
    }
    /// Borrows actual identities and positive occurrence counts.
    pub fn demands(&self) -> &[(usize, u64)] { &self.demands }
    /// The original cumulative account, if the selected producer supplied one.
    pub fn funding(&self) -> Option<&HostMetadataFunding> {
        self.custody.as_ref().map(|value| &value.funding)
    }
    /// Exact original source, available only by a borrowed type inspection.
    pub fn source(&self) -> Option<&ErasedSharedStorageOwner> {
        self.custody.as_ref().map(|value| &value.source)
    }
    /// Fixed owner, refusal and source-retention population. Dynamic demand and
    /// consumer destinations are separate and must use the same account.
    pub fn control_bytes() -> Option<usize> {
        let frames = [size_of::<Self>(), size_of::<Custody>(), size_of::<Failure>(),
            size_of::<Cause>(), size_of::<Option<Custody>>(), size_of::<eredu_nn::Error>(),
            size_of::<HostMetadataFundingError>(), size_of::<Result<(), eredu_nn::Error>>(),
            eredu_nn::Error::retained_source_construction_bytes::<Failure>()?];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Prices derived consumer storage before construction. A refusal keeps
    /// the completed producer and its spent account without copying demands.
    pub fn reserve_metadata(&self, bytes: Option<usize>) -> Result<(), eredu_nn::Error> {
        let refuse = |cause| eredu_nn::Error::backend_retained_source(Failure {
            cause, retained: Self { demands: Vec::new(), custody: self.custody.clone() },
        });
        let bytes = bytes.ok_or_else(|| refuse(Cause::Overflow))?;
        if let Some(custody) = &self.custody {
            custody.funding.reserve_metadata(bytes)
                .map_err(|cause| refuse(Cause::Funding(cause)))?;
        }
        Ok(())
    }
    /// Keeps the actual demand destination and completed source with any later
    /// acquisition, binding, grouped execution or completion failure.
    pub fn retain_error(self, cause: eredu_nn::Error) -> eredu_nn::Error {
        if self.custody.is_none() { return cause; }
        eredu_nn::Error::backend_retained_source(Failure { cause: Cause::Execution(cause), retained: self })
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("addressable demand consumer metadata extent overflowed")]
    Overflow,
    #[error("addressable demand metadata funding: {0}")]
    Funding(#[source] HostMetadataFundingError),
    #[error("addressable demand consumer: {0}")]
    Execution(#[source] eredu_nn::Error),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source] cause: Cause,
    retained: IndexedDemandSource,
}


/// Lexical loan of the backend acquisition source. It does not own a cache
/// attempt or grant capacity; the concrete bank must authenticate both.
#[derive(Clone,Copy)]
pub struct PreparedIndexedDemandLoan<'a> {
    source:&'a dyn std::any::Any,
    funding:&'a HostMetadataFunding,
}
impl<'a> PreparedIndexedDemandLoan<'a> {
    /// Lends the explicitly selected source and its original account.
    pub fn new(source:&'a dyn std::any::Any,funding:&'a HostMetadataFunding)->Self {Self{source,funding}}
    /// Borrows the exact private source without reconstructing ownership.
    pub fn source<T:std::any::Any>(&self)->Option<&'a T>{self.source.downcast_ref()}
    /// The cumulative account used by completed discovery.
    pub fn funding(&self)->&'a HostMetadataFunding{self.funding}
}
/// Funded demands cannot silently select ordinary acquisition.
#[derive(Debug,thiserror::Error)]
pub enum IndexedDemandLoanError<E:std::error::Error+'static> {
    /// The selected mechanism supplies no exact acquisition source.
    #[error("addressable demand has no selected acquisition source")]
    MissingProducer,
    /// Concrete source, storage or completion refusal.
    #[error(transparent)]
    Backend(E),
}
