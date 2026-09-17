//! Closed, source-owned routing of prepared final lease destinations.
use super::*;
mod conversion_plan;
pub(super) mod retained_route;
pub use conversion_plan::{SelectedGgufAcquisitionStorage, SelectedGgufConversionPlan};
pub use retained_route::PreparedAcquisitionOwner;
use retained_route::RetainedGgufRoute;

/// Loan of a concrete source's existing authorization/acquisition path.
/// Its variants are private: numerical metadata and arbitrary callbacks cannot
/// manufacture a provider or a successful acquisition. This is not fit authority.
pub struct PreparedAcquisitionSource<'a>(pub(super) Route<'a>);
#[derive(Clone, Copy)]
pub(super) enum Route<'a> {
    Unavailable,
    Safetensors(&'a SafetensorsWeightStore),
    Memory(&'a MemoryWeightStore),
    Gguf(&'a crate::gguf_store::GgufWeightStore),
    Prepared(&'a PreparedCheckpointSource),
    Restricted(&'a RestrictedCheckpointSource),
    Composite(&'a CompositeCheckpointSource),
    Resolved(&'a ResolvedCheckpointSource),
}
impl<'a> PreparedAcquisitionSource<'a> {
    pub(crate) fn gguf(source: &'a crate::gguf_store::GgufWeightStore) -> Self {
        Self(Route::Gguf(source))
    }
    pub(super) fn unavailable() -> Self {
        Self(Route::Unavailable)
    }
}

/// Once-only acquisition bound to the actual retained root and exact request.
/// Preparation owns final SafeTensors/Memory destinations or the actual boxed
/// lazy GGUF lease. It does not acquire a payload shard, copy selected source
/// bytes, or prepay the separate GGUF materialization/conversion storage.
/// Shared-cache allocation bounds and native producer fits remain unqualified.
///
/// ```compile_fail
/// use eredu_checkpoint::store::PreparedCheckpointAcquisition;
/// fn twice(prepared: PreparedCheckpointAcquisition) {
///     let _ = prepared.acquire();
///     let _ = prepared.acquire();
/// }
/// ```
pub struct PreparedCheckpointAcquisition {
    destination: Destination,
    request: TensorReadRequest,
    source: RetainedCheckpointSource,
    retained_route: Option<Arc<RetainedGgufRoute>>,
}
impl PreparedCheckpointAcquisition {
    /// Follow the actual authorization path and prepare its final destination.
    /// `None` means that route has no destination companion; no ordinary
    /// acquisition, alternate source, or payload fallback is attempted.
    /// All allocations here need caller funding, independently of this API.
    pub fn prepare(
        source: impl Into<RetainedCheckpointSource>,
        request: TensorReadRequest,
    ) -> Result<Option<Self>, PreparedAcquisitionFailure> {
        Self::prepare_with_payload(source.into(), request, true)
    }
    pub(super) fn prepare_deferred(
        source: impl Into<RetainedCheckpointSource>,
        request: TensorReadRequest,
    ) -> Result<Option<Self>, PreparedAcquisitionFailure> {
        Self::prepare_with_payload(source.into(), request, false)
    }
    fn prepare_with_payload(
        source: RetainedCheckpointSource,
        request: TensorReadRequest,
        allocate_payload: bool,
    ) -> Result<Option<Self>, PreparedAcquisitionFailure> {
        let result = (|| {
            let Some(store) = leaf(source.as_ref(), &request)? else {
                return Ok(None);
            };
            match store {
                Leaf::Safetensors(store) => leaf_loan(store, &request)?
                    .prepare_with_payload(
                        request.selection.clone(),
                        request.policy,
                        allocate_payload,
                    )
                    .map(Destination::Safetensors)
                    .map(Some)
                    .map_err(Cause::Destination),
                Leaf::Memory(store) => (if allocate_payload {
                    memory_destination::Prepared::prepare(store, &request)
                } else {
                    memory_destination::Prepared::prepare_with_payload(store, &request, false)
                })
                .map(Destination::Memory)
                .map(Some)
                .map_err(Cause::Memory),
                Leaf::Gguf(store) => crate::gguf_store::PreparedGgufLease::prepare(store, &request)
                    .map(Destination::Gguf)
                    .map(Some)
                    .map_err(Cause::Store),
            }
        })();
        match result {
            Ok(Some(destination)) => Ok(Some(Self {
                destination,
                request,
                source,
                retained_route: None,
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

    /// Exact retained root identity and request values, without reading payloads.
    /// Acquisition itself takes no replacement source or request.
    pub fn matches_request(
        &self,
        source: &SharedCheckpointSource,
        request: &TensorReadRequest,
    ) -> bool {
        self.source.matches_ordinary(source) && self.request == *request
    }

    /// Compare the exact opaque root retained by a selected G4 acquisition.
    pub fn matches_retained_request(
        &self,
        source: &RetainedCheckpointSource,
        request: &TensorReadRequest,
    ) -> bool {
        self.source.same_source(source) && self.request == *request
    }

    /// Compare the retained root and request through borrowed inputs. No key,
    /// selection, wrapper or source allocation is created by this check.
    pub fn matches_source_request(
        &self,
        source: &dyn CheckpointSource,
        key: &str,
        selection: &TensorSelection,
        policy: ReadPolicy,
    ) -> bool {
        std::ptr::addr_eq(self.source.as_ref(), source)
            && self.request.key == key
            && self.request.selection == *selection
            && self.request.policy == policy
    }

    /// Recheck the retained route, consume its matching final destination, and
    /// apply ordinary post-lease validation while unwinding the wrapper chain.
    /// Any failure retains the actual remaining destination or rejected lease.
    pub fn acquire(self) -> Result<CheckpointLease, PreparedAcquisitionFailure> {
        let Self {
            destination,
            request,
            source,
            retained_route,
        } = self;
        let mut state = Fill {
            pending: Some(destination),
            lease: None,
        };
        let result = match retained_route.as_ref() {
            Some(route) => route.fill(&request, &mut state),
            None => fill(source.as_ref(), &request, &mut state),
        };
        match result {
            Ok(()) => Ok(state.lease.expect("closed leaf produced a lease")),
            Err(cause) => Err(PreparedAcquisitionFailure {
                cause,
                pending: state.pending,
                rejected: state.lease,
                request,
                source,
            }),
        }
    }
}
impl std::fmt::Debug for PreparedCheckpointAcquisition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedCheckpointAcquisition")
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}

// Each branch delegates the existing concrete wrapper check. Preparation and
// fill share this selection; post-lease checks stay at their actual unwind site.
enum Leaf<'a> {
    Safetensors(&'a SafetensorsWeightStore),
    Memory(&'a MemoryWeightStore),
    Gguf(&'a crate::gguf_store::GgufWeightStore),
}
#[derive(Debug)]
enum Destination {
    Safetensors(PreparedSafetensorsLease),
    Memory(memory_destination::Prepared),
    Gguf(crate::gguf_store::PreparedGgufLease),
}
enum Step<'a> {
    Unavailable,
    Leaf(Leaf<'a>),
    Child {
        source: &'a dyn CheckpointSource,
        validate: Option<&'a PreparedCheckpointSource>,
    },
}
fn step<'a>(
    source: &'a dyn CheckpointSource,
    request: &TensorReadRequest,
) -> Result<Step<'a>, StoreError> {
    step_route(source.prepared_acquisition_source().0, request)
}
fn step_route<'a>(route: Route<'a>, request: &TensorReadRequest) -> Result<Step<'a>, StoreError> {
    Ok(match route {
        Route::Unavailable => Step::Unavailable,
        Route::Safetensors(store) => Step::Leaf(Leaf::Safetensors(store)),
        Route::Memory(store) => Step::Leaf(Leaf::Memory(store)),
        Route::Gguf(store) => Step::Leaf(Leaf::Gguf(store)),
        Route::Prepared(owner) => {
            owner.expected(&request.key)?;
            Step::Child {
                source: owner.source.as_ref(),
                validate: Some(owner),
            }
        }
        Route::Restricted(owner) => {
            owner.authorize(&request.key)?;
            Step::Child {
                source: owner.source.as_ref(),
                validate: None,
            }
        }
        Route::Composite(owner) => Step::Child {
            source: owner.source_for(&request.key)?,
            validate: None,
        },
        Route::Resolved(owner) => {
            owner.authorize_acquisition(&request.key)?;
            Step::Child {
                source: owner.source.as_ref(),
                validate: None,
            }
        }
    })
}
fn leaf<'a>(
    source: &'a dyn CheckpointSource,
    request: &TensorReadRequest,
) -> Result<Option<Leaf<'a>>, StoreError> {
    match step(source, request)? {
        Step::Unavailable => Ok(None),
        Step::Leaf(store) => Ok(Some(store)),
        Step::Child { source, .. } => leaf(source, request),
    }
}
fn leaf_loan<'a>(
    store: &'a SafetensorsWeightStore,
    request: &'a TensorReadRequest,
) -> Result<SafetensorsLeaseSource<'a>, Cause> {
    let entry = store
        .catalog
        .get(&request.key)
        .ok_or_else(|| StoreError::UnknownTensor {
            key: request.key.clone(),
        })?;
    let header = store
        .shards
        .admission(&entry.shard)
        .header
        .get()
        .ok_or(Cause::HeaderUnavailable)?;
    let header = header
        .as_ref()
        .map_err(|error| Cause::Store(error.clone()))?;
    let metadata = header
        .tensors
        .get(&request.key)
        .ok_or_else(|| StoreError::UnknownTensor {
            key: request.key.clone(),
        })?;
    Ok(SafetensorsLeaseSource::new(store, &request.key, metadata))
}
struct Fill {
    pending: Option<Destination>,
    lease: Option<CheckpointLease>,
}
fn fill(
    source: &dyn CheckpointSource,
    request: &TensorReadRequest,
    state: &mut Fill,
) -> Result<(), Cause> {
    match step(source, request)? {
        Step::Unavailable => Err(Cause::SourceChanged),
        Step::Leaf(store) => {
            let pending = state.pending.as_ref().expect("one closed leaf");
            let matches = match (pending, store) {
                (Destination::Safetensors(pending), Leaf::Safetensors(store)) => pending
                    .matches_request(
                        leaf_loan(store, request)?,
                        &request.selection,
                        request.policy,
                    ),
                (Destination::Memory(pending), Leaf::Memory(store)) => {
                    pending.matches(store, request)
                }
                (Destination::Gguf(pending), Leaf::Gguf(store)) => pending.matches(store, request),
                _ => false,
            };
            if !matches {
                return Err(Cause::SourceChanged);
            }
            state.lease = Some(match state.pending.take().unwrap() {
                Destination::Safetensors(pending) => {
                    pending.acquire_checkpoint().map_err(Cause::Destination)?
                }
                Destination::Memory(pending) => {
                    pending.acquire(&request.key).map_err(Cause::Memory)?
                }
                Destination::Gguf(pending) => pending.into_lease(),
            });
            Ok(())
        }
        Step::Child { source, validate } => {
            fill(source, request, state)?;
            if let Some(owner) = validate {
                owner.validate_acquired(request, state.lease.as_ref().unwrap())?;
            }
            Ok(())
        }
    }
}

#[derive(Debug)]
enum Cause {
    Store(StoreError),
    Destination(PreparedSafetensorsLeaseFailure),
    Memory(memory_destination::Failure),
    HeaderUnavailable,
    SourceChanged,
}
impl From<StoreError> for Cause {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}
/// Owning failure. The actual payload/destination retires before the retained
/// request and root source. No retry, clone, or successful lease escape is exposed.
pub struct PreparedAcquisitionFailure {
    cause: Cause,
    pending: Option<Destination>,
    rejected: Option<CheckpointLease>,
    request: TensorReadRequest,
    source: RetainedCheckpointSource,
}
impl PreparedAcquisitionFailure {
    /// The same closed source remains owned through acquisition refusal.
    pub fn matches_retained_request(
        &self,
        source: &RetainedCheckpointSource,
        request: &TensorReadRequest,
    ) -> bool {
        self.source.same_source(source) && self.request == *request
    }
    /// Exact ordinary wrapper/cache/selection error when that operation failed.
    pub fn store_error(&self) -> Option<&StoreError> {
        match &self.cause {
            Cause::Store(error) => Some(error),
            Cause::Destination(error) => error.store_error(),
            Cause::Memory(error) => error.store_error(),
            _ => None,
        }
    }
    /// Actual destination/read failure, including retained file/shard owners.
    pub fn destination_error(&self) -> Option<&PreparedSafetensorsLeaseFailure> {
        match &self.cause {
            Cause::Destination(error) => Some(error),
            _ => None,
        }
    }
    /// A genuine acquired lease was refused by a wrapper's post-validation.
    pub fn retains_rejected_lease(&self) -> bool {
        self.rejected.is_some()
    }
    /// Exact source/request binding remains retained until failure retirement.
    pub fn matches_request(
        &self,
        source: &SharedCheckpointSource,
        request: &TensorReadRequest,
    ) -> bool {
        self.source.matches_ordinary(source) && self.request == *request
    }
}
impl std::fmt::Debug for PreparedAcquisitionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedAcquisitionFailure")
            .field("cause", &self.cause)
            .field("pending", &self.pending.is_some())
            .field("rejected", &self.rejected.is_some())
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Display for PreparedAcquisitionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cause {
            Cause::Store(error) => error.fmt(f),
            Cause::Destination(error) => error.fmt(f),
            Cause::Memory(error) => error.fmt(f),
            Cause::HeaderUnavailable => f.write_str("source header is not prepared"),
            Cause::SourceChanged => f.write_str("prepared checkpoint acquisition source changed"),
        }
    }
}
impl std::error::Error for PreparedAcquisitionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Store(error) => Some(error),
            Cause::Destination(error) => Some(error),
            Cause::Memory(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod retained_tests;
#[cfg(test)]
mod tests;

mod bank;
pub use bank::{
    PreparedAcquisitionBank, PreparedAcquisitionBankError, PreparedAcquisitionRefusal,
    PreparedAcquisitionStorage,
};

pub(crate) mod storage;
