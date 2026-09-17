//! Explicit cold GGUF query through the existing authorization/source route.
use super::*;
use crate::gguf_store::{GgufConversionPlan, GgufConversionPlanError};

/// Metadata from the genuine retained source root and exact logical request.
/// It does not acquire a lease or replace later acquisition/post-lease checks.
pub struct SelectedGgufConversionPlan {
    physical: GgufConversionPlan,
    request: TensorReadRequest,
    source: RetainedCheckpointSource,
    route: Option<Arc<RetainedGgufRoute>>,
}
impl SelectedGgufConversionPlan {
    /// Follow existing wrapper authorization to inspect the actual GGUF group.
    /// `None` means an unavailable or non-GGUF route, never a zero-byte fit.
    /// This can allocate descriptor/selection metadata, but no tensor payload,
    /// shard reader or native resource. The caller owns its cold metadata cost.
    pub fn query(
        source: SharedCheckpointSource,
        request: TensorReadRequest,
    ) -> Result<Option<Self>, GgufConversionPlanError> {
        Self::query_retained(source.into(), request)
    }
    /// Same selection driver retaining an opaque closed root. No compatibility
    /// Arc is constructed, and accepted acquisition uses its concrete route.
    /// Cold route/selection metadata remains a separate caller contribution.
    pub fn query_retained(
        source: RetainedCheckpointSource,
        request: TensorReadRequest,
    ) -> Result<Option<Self>, GgufConversionPlanError> {
        let Some(Leaf::Gguf(store)) = leaf(source.as_ref(), &request)? else {
            return Ok(None);
        };
        let physical = store.conversion_plan(&request)?;
        let route = RetainedGgufRoute::prepare(&source, &request)?;
        Ok(Some(Self {
            physical,
            request,
            source,
            route,
        }))
    }
    /// Exact physical selection and shared requested conversion layouts.
    pub fn physical(&self) -> &GgufConversionPlan {
        &self.physical
    }
    /// Exact logical request, including policy and ordered repeated indices.
    pub fn request(&self) -> &TensorReadRequest {
        &self.request
    }
    /// Verify original root identity and request without routing or allocation.
    pub fn matches_request(
        &self,
        source: &SharedCheckpointSource,
        request: &TensorReadRequest,
    ) -> bool {
        self.source.matches_ordinary(source) && self.request == *request
    }
    /// Compare the exact retained source and selected request without routing.
    pub fn matches_retained_request(
        &self,
        source: &RetainedCheckpointSource,
        request: &TensorReadRequest,
    ) -> bool {
        self.source.same_source(source) && self.request == *request
    }
    /// Cold shared route payload layout; metadata_bytes includes its payload
    /// and Vec backing, but the Arc header requires the caller's qualification.
    pub fn acquisition_route_shared_layout(&self) -> Option<std::alloc::Layout> {
        self.route
            .as_ref()
            .map(|_| RetainedGgufRoute::shared_payload_layout())
    }
    /// Actual metadata representations/capacities, excluding Arc headers, shared source,
    /// allocator charge and payload/peak bounds.
    pub fn metadata_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(
                self.physical
                    .metadata_bytes()?
                    .checked_sub(std::mem::size_of::<GgufConversionPlan>())?,
            )?
            .checked_add(
                self.route
                    .as_ref()
                    .map_or(Some(0), |route| route.metadata_bytes())?,
            )?
            .checked_add(self.request.key.capacity())?
            .checked_add(super::storage::selection(&self.request.selection)?)
    }
}
impl std::fmt::Debug for SelectedGgufConversionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectedGgufConversionPlan")
            .field("request", &self.request)
            .field("physical", &self.physical)
            .finish_non_exhaustive()
    }
}

/// Requested final G4 storage from a genuine selected source. Allocator/clone
/// qualification and pre-allocation admission remain with the owning caller.
#[derive(Clone, Copy, Debug)]
pub struct SelectedGgufAcquisitionStorage {
    /// Complete retained request, catalog/identity buffers and final lease Box.
    retained_bytes: usize,
    diagnostic_key_bytes: usize,
    /// Maximum actual owned wrapper-refusal key/contract strings.
    refusal_bytes: usize,
    /// Named preparation/route/clone transports, separate from heap storage.
    control_bytes: usize,
}
impl SelectedGgufAcquisitionStorage {
    /// Actual lease metadata key used by downstream selected-byte validation.
    pub const fn diagnostic_key_bytes(self) -> usize {
        self.diagnostic_key_bytes
    }
    /// Requested final owned buffers and lease Box.
    pub const fn retained_bytes(self) -> usize {
        self.retained_bytes
    }
    /// Maximum owned checkpoint-wrapper rejection storage.
    pub const fn refusal_bytes(self) -> usize {
        self.refusal_bytes
    }
    /// Named constructor and route control values.
    pub const fn control_bytes(self) -> usize {
        self.control_bytes
    }
}
impl SelectedGgufConversionPlan {
    /// No payload read or destination construction. Authorization is the same
    /// concrete route used by preparation/acquisition; arbitrary unavailable
    /// routes have no companion and never receive a zero bound.
    pub fn acquisition_storage(
        &self,
    ) -> Result<Option<SelectedGgufAcquisitionStorage>, GgufConversionPlanError> {
        let Some(route) = self.route.as_ref() else {
            return Ok(None);
        };
        if !self.physical.matches_store(route.store()) {
            return Ok(None);
        }
        let refusal = route
            .refusal_bytes(&self.request)
            .ok_or(eredu_gguf::ConversionDestinationError::Layout)?;
        let requested = || {
            let request = self.request.key.len().checked_add(
                crate::store::SelectionCloneLayout::of(&self.request.selection)?
                    .elements
                    .map_or(0, |l| l.size()),
            )?;
            let retained_bytes =
                request.checked_add(self.physical.acquisition_storage(&self.request)?)?;
            let control_bytes = route
                .controls()?
                .checked_add(crate::gguf_store::GgufConversionPlan::acquisition_controls()?)?
                .checked_add(std::mem::size_of::<PreparedCheckpointAcquisition>())?
                .checked_add(std::mem::size_of::<Fill>())?
                .checked_add(std::mem::size_of::<PreparedAcquisitionFailure>())?
                .checked_add(std::mem::size_of::<
                    Result<CheckpointLease, PreparedAcquisitionFailure>,
                >())?;
            Some(SelectedGgufAcquisitionStorage {
                retained_bytes,
                diagnostic_key_bytes: self
                    .physical
                    .acquisition_diagnostic_key_bytes(&self.request)?,
                refusal_bytes: refusal,
                control_bytes,
            })
        };
        requested()
            .map(Some)
            .ok_or(eredu_gguf::ConversionDestinationError::Layout.into())
    }
    /// Construct the final same-Box destination after the caller has funded the
    /// complete recipe. No physical selector or converter is run again.
    pub fn prepare_acquisition(
        &self,
    ) -> Result<Option<PreparedCheckpointAcquisition>, PreparedAcquisitionFailure> {
        let Some(route) = self.route.as_ref() else {
            return Ok(None);
        };
        let request = self.request.clone();
        let source = self.source.clone();
        let result = (|| {
            route.validate_before(&request)?;
            crate::gguf_store::PreparedGgufLease::prepare_plan(
                route.store(),
                &request,
                &self.physical,
            )
            .map(Destination::Gguf)
            .map(Some)
            .ok_or(Cause::SourceChanged)
        })();
        match result {
            Ok(Some(destination)) => Ok(Some(PreparedCheckpointAcquisition {
                destination,
                request,
                source,
                retained_route: Some(Arc::clone(route)),
            })),
            Ok(None) => Ok(None),
            Err(cause) => Err(PreparedAcquisitionFailure {
                cause,
                pending: None,
                rejected: None,
                request,
                source,
            }),
        }
    }
}
