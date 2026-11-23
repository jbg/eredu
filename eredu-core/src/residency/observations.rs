//! Native transfer operations and physical observations for resident weights.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    projection::ProjectionStorageFacts, ResidencyObservationBindings, ResidentParameterConversion,
};
use crate::resources::{
    detached::{DetachedResourceLimits, DetachedResourceObservation},
    ResourceDescription, ResourceDescriptionError,
};

/// Native storage facts that augment the portable logical residency ledger.
pub struct NativeResidencyObservations {
    /// Current host allocation descriptions.
    pub host_storage: ResourceDescription,
    /// Resident parameter conversion allocations and bindings.
    pub conversions: Vec<ResidentParameterConversion>,
    /// Native backing facts by logical projection identity.
    pub projection_storage: BTreeMap<String, ProjectionStorageFacts>,
    /// Normalization bindings with guaranteed F32 output.
    pub f32_normalization_gains: BTreeSet<String>,
}

/// Native transfer and physical observation operations for resident weight storage.
pub trait DenseResidencyMechanisms:
    super::ResidencyStorageMechanisms<
    Error: From<eredu_checkpoint::store::StoreError> + From<ResourceDescriptionError>,
>
{
    /// Identity of the selected native transfer stream.
    fn transfer_stream_index(storage: &Self::Storage) -> Result<i32, Self::Error>;
    /// Settles the exact native transfer.
    fn settle_transfer(transfer: &mut Self::Transfer) -> Result<(), Self::Error>;
    /// Samples physical allocator counters.
    fn allocator_memory(
        &self,
        storage: &Self::Storage,
    ) -> Result<crate::AllocatorTelemetry, Self::Error>;
    /// Reads native physical facts for the exact retained owners.
    fn observe_residency(
        &self,
        storage: &Self::Storage,
        bindings: ResidencyObservationBindings<'_>,
    ) -> Result<NativeResidencyObservations, Self::Error>;

    /// Observes detached submissions outside the installed storage lock.
    fn observe_detached(
        &self,
        detached_limits: DetachedResourceLimits,
    ) -> Result<DetachedResourceObservation, Self::Error>;
}
