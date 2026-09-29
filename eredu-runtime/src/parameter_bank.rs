//! Generic parameter-bank acquisition, completion ownership and reporting.
use crate::{ParameterBankAccess, ParameterBankKey};
use eredu_nn::{NeuralBackend, Tensor};

/// Exact generic request for an independently addressable bank acquisition.
#[derive(Debug, Clone, Copy)]
pub struct ParameterBankAcquisition<'a> {
    entries: &'a [(ParameterBankKey, u64)],
    access: ParameterBankAccess,
}

impl<'a> ParameterBankAcquisition<'a> {
    /// Creates one deterministic acquisition request in compact-bank order.
    pub const fn new(entries: &'a [(ParameterBankKey, u64)], access: ParameterBankAccess) -> Self {
        Self { entries, access }
    }

    /// Returns generic bank keys and duplicate-preserving demand counts.
    pub const fn entries(&self) -> &'a [(ParameterBankKey, u64)] {
        self.entries
    }

    /// Returns the selected generic storage access class.
    pub const fn access(&self) -> ParameterBankAccess {
        self.access
    }
}

/// Addressable parameter storage independent of the consuming operator.
/// Architecture construction supplies exact keys and residency policy. Native
/// implementations keep source and residency leases until output completion.
pub trait ParameterBank<B: NeuralBackend> {
    /// Live native storage retained across consuming operations.
    type Acquisition;
    /// Generic bank telemetry snapshot.
    type Report;
    /// Storage, transfer, lowering, or construction failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns the selected byte geometry for one admitted bank member.
    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64>;

    /// Acquires exact generic keys in caller-supplied compact order.
    fn acquire(
        &mut self,
        request: ParameterBankAcquisition<'_>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Acquisition, Self::Error>;

    /// Retains acquired storage until the consuming output is natively complete.
    fn complete(
        &mut self,
        acquisition: Self::Acquisition,
        output: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Returns generic key, byte, tier, acquisition, and eviction telemetry.
    fn report(&self) -> Result<Self::Report, Self::Error>;
}
