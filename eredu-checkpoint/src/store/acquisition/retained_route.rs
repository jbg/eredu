//! Concrete owners captured cold; no virtual source callbacks after selection.
use super::*;
use crate::store::storage::SourceHandle;
use std::alloc::Layout;

/// Opaque owning route loan. The actual root and every selected child must
/// return its own built-in owner. Forwarding another owner's loan is rejected.
pub struct PreparedAcquisitionOwner(Owner);
enum Owner {
    RetainedPrepared(SourceHandle<PreparedCheckpointSource>),
    RetainedResolved(SourceHandle<ResolvedCheckpointSource>),
    RetainedSafetensors(SourceHandle<SafetensorsWeightStore>),
    RetainedGguf(SourceHandle<crate::gguf_store::GgufWeightStore>),
    RetainedComposite(SourceHandle<CompositeCheckpointSource>),
    Memory(Arc<MemoryWeightStore>),
    Safetensors(Arc<SafetensorsWeightStore>),
    Gguf(Arc<crate::gguf_store::GgufWeightStore>),
    Prepared(Arc<PreparedCheckpointSource>),
    Restricted(Arc<RestrictedCheckpointSource>),
    Composite(Arc<CompositeCheckpointSource>),
    Resolved(Arc<ResolvedCheckpointSource>),
}
impl PreparedAcquisitionOwner {
    pub(in crate::store) fn encoded_contract(&self) -> Option<&str> {
        match self.route() {
            Route::Restricted(owner) => Some(&owner.contract),
            Route::Resolved(owner) => Some(owner.contract.identity()),
            _ => None,
        }
    }

    pub(in crate::store) fn retained_prepared(owner: SourceHandle<PreparedCheckpointSource>) -> Self {
        Self(Owner::RetainedPrepared(owner))
    }
    pub(in crate::store) fn retained_resolved(owner: SourceHandle<ResolvedCheckpointSource>) -> Self {
        Self(Owner::RetainedResolved(owner))
    }

    pub(in crate::store) fn retained_safetensors(
        owner: SourceHandle<SafetensorsWeightStore>,
    ) -> Self {
        Self(Owner::RetainedSafetensors(owner))
    }
    pub(crate) fn gguf(owner: Arc<crate::gguf_store::GgufWeightStore>) -> Self {
        Self(Owner::Gguf(owner))
    }
    pub(in crate::store) fn retained_gguf(
        owner: SourceHandle<crate::gguf_store::GgufWeightStore>,
    ) -> Self {
        Self(Owner::RetainedGguf(owner))
    }
    pub(in crate::store) fn retained_composite(
        owner: SourceHandle<CompositeCheckpointSource>,
    ) -> Self {
        Self(Owner::RetainedComposite(owner))
    }
    pub(in crate::store) fn source(&self) -> &dyn CheckpointSource {
        match &self.0 {
            Owner::RetainedSafetensors(owner) => &**owner,
            Owner::RetainedGguf(owner) => &**owner,
            Owner::RetainedComposite(owner) => &**owner,
            Owner::Memory(owner) => owner.as_ref(),
            Owner::Safetensors(owner) => owner.as_ref(),
            Owner::Gguf(owner) => owner.as_ref(),
            Owner::RetainedPrepared(owner) => &**owner,
            Owner::RetainedResolved(owner) => &**owner,
            Owner::Prepared(owner) => owner.as_ref(),
            Owner::Restricted(owner) => owner.as_ref(),
            Owner::Composite(owner) => owner.as_ref(),
            Owner::Resolved(owner) => owner.as_ref(),
        }
    }
    fn route(&self) -> Route<'_> {
        match &self.0 {
            Owner::RetainedSafetensors(owner) => Route::Safetensors(owner),
            Owner::RetainedGguf(owner) => Route::Gguf(owner),
            Owner::RetainedComposite(owner) => Route::Composite(owner),
            Owner::Memory(owner) => Route::Memory(owner),
            Owner::Safetensors(owner) => Route::Safetensors(owner),
            Owner::Gguf(owner) => Route::Gguf(owner),
            Owner::RetainedPrepared(owner) => Route::Prepared(owner),
            Owner::RetainedResolved(owner) => Route::Resolved(owner),
            Owner::Prepared(owner) => Route::Prepared(owner),
            Owner::Restricted(owner) => Route::Restricted(owner),
            Owner::Composite(owner) => Route::Composite(owner),
            Owner::Resolved(owner) => Route::Resolved(owner),
        }
    }
}
pub(in crate::store) fn owner_memory(owner: Arc<MemoryWeightStore>) -> PreparedAcquisitionOwner {
    PreparedAcquisitionOwner(Owner::Memory(owner))
}
pub(in crate::store) fn owner_safetensors(
    owner: Arc<SafetensorsWeightStore>,
) -> PreparedAcquisitionOwner {
    PreparedAcquisitionOwner(Owner::Safetensors(owner))
}
pub(in crate::store) fn owner_prepared(
    owner: Arc<PreparedCheckpointSource>,
) -> PreparedAcquisitionOwner {
    PreparedAcquisitionOwner(Owner::Prepared(owner))
}
pub(in crate::store) fn owner_restricted(
    owner: Arc<RestrictedCheckpointSource>,
) -> PreparedAcquisitionOwner {
    PreparedAcquisitionOwner(Owner::Restricted(owner))
}
pub(in crate::store) fn owner_composite(
    owner: Arc<CompositeCheckpointSource>,
) -> PreparedAcquisitionOwner {
    PreparedAcquisitionOwner(Owner::Composite(owner))
}
pub(in crate::store) fn owner_resolved(
    owner: Arc<ResolvedCheckpointSource>,
) -> PreparedAcquisitionOwner {
    PreparedAcquisitionOwner(Owner::Resolved(owner))
}
// Presence, not cache identity: Prepared guarantees Some even if its child
// changes. Composite depends on every source, including off-route children.
// The opaque concrete owner authenticates each dependency before inspecting it.
fn concrete_owner(source: &RetainedCheckpointSource) -> Option<PreparedAcquisitionOwner> {
    source.acquisition_owner()
}
fn stable_recipe_presence(source: &RetainedCheckpointSource) -> Option<bool> {
    match concrete_owner(source)?.route() {
        Route::Unavailable => None,
        Route::Gguf(_) | Route::Prepared(_) | Route::Memory(_) | Route::Safetensors(_) => {
            Some(true)
        }
        Route::Restricted(owner) => stable_recipe_presence(&owner.source),
        Route::Resolved(owner) => stable_recipe_presence(&owner.source),
        Route::Composite(owner) => {
            let mut all = true;
            for source in &owner.sources {
                all &= stable_recipe_presence(source)?;
            }
            Some(all)
        }
    }
}
fn stable_materialized_key(source: &RetainedCheckpointSource, key: &str) -> Option<bool> {
    match concrete_owner(source)?.route() {
        Route::Unavailable => None,
        Route::Gguf(_) | Route::Memory(_) | Route::Safetensors(_) => Some(false),
        Route::Prepared(owner) => stable_materialized_key(&owner.source, key),
        Route::Resolved(owner) => stable_materialized_key(&owner.source, key),
        Route::Restricted(owner) => {
            if owner.is_authorized(key) {
                stable_materialized_key(&owner.source, key)
            } else {
                Some(false)
            }
        }
        Route::Composite(owner) => {
            match owner.owners.get(key).and_then(|index| owner.sources.get(*index)) {
                Some(child) => stable_materialized_key(child, key),
                None => Some(false),
            }
        }
    }
}

struct StepOwner {
    owner: PreparedAcquisitionOwner,
    // The same pre/post predicates run while cold, before the complete chain
    // is known. They are retained only if every child is concrete and immutable.
    validate_prepared: bool,
    materialized: bool,
}

mod encoded;
pub(in crate::store) use encoded::{encoded_file_source, encoded_memory_source};

pub(super) struct RetainedGgufRoute {
    steps: Vec<StepOwner>,
}
impl RetainedGgufRoute {
    pub(super) fn prepare(
        source: &RetainedCheckpointSource,
        request: &TensorReadRequest,
    ) -> Result<Option<Arc<Self>>, StoreError> {
        let mut current = source.clone();
        let mut steps = Vec::new();
        loop {
            let Some(owner) = current.acquisition_owner() else {
                return Ok(None);
            };
            let mut validate_prepared = false;
            let mut materialized = false;
            let child = match owner.route() {
                Route::Unavailable => return Ok(None),
                Route::Gguf(_) => None,
                Route::Memory(_) | Route::Safetensors(_) => return Ok(None),
                Route::Prepared(owner) => {
                    owner.expected(&request.key)?;
                    let Some(present) = stable_recipe_presence(&owner.source) else {
                        return Ok(None);
                    };
                    validate_prepared = !present;
                    Some(owner.source.clone())
                }
                Route::Restricted(owner) => {
                    owner.authorize(&request.key)?;
                    Some(owner.source.clone())
                }
                Route::Composite(owner) => {
                    // Same concrete source_for lookup as ordinary routing.
                    Some(owner.source_owner_for(&request.key)?.clone())
                }
                Route::Resolved(owner) => {
                    let Some(known) = stable_materialized_key(&owner.source, &request.key) else {
                        return Ok(None);
                    };
                    materialized = known;
                    if !materialized {
                        owner.authorize(&request.key)?;
                    }
                    Some(owner.source.clone())
                }
            };
            steps.push(StepOwner {
                owner,
                validate_prepared,
                materialized,
            });
            match child {
                Some(child) => current = child,
                None => return Ok(Some(Arc::new(Self { steps }))),
            }
        }
    }
    pub(super) fn store(&self) -> &crate::gguf_store::GgufWeightStore {
        let Route::Gguf(store) = self
            .steps
            .last()
            .expect("closed route has leaf")
            .owner
            .route()
        else {
            unreachable!("only GGUF closes retained route")
        };
        store
    }
    pub(super) fn validate_before(&self, request: &TensorReadRequest) -> Result<(), StoreError> {
        for step in &self.steps {
            match step.owner.route() {
                Route::Prepared(owner) => {
                    owner.expected(&request.key)?;
                }
                Route::Restricted(owner) => owner.authorize(&request.key)?,
                Route::Composite(owner) => {
                    owner.source_for(&request.key)?;
                }
                Route::Resolved(owner) if !step.materialized => owner.authorize(&request.key)?,
                Route::Unavailable => unreachable!("concrete owner"),
                Route::Resolved(_) | Route::Gguf(_) | Route::Memory(_) | Route::Safetensors(_) => {}
            }
        }
        Ok(())
    }
    pub(super) fn fill(&self, request: &TensorReadRequest, state: &mut Fill) -> Result<(), Cause> {
        self.validate_before(request)?;
        let Some(Destination::Gguf(pending)) = state.pending.as_ref() else {
            return Err(Cause::SourceChanged);
        };
        if !pending.matches(self.store(), request) {
            return Err(Cause::SourceChanged);
        }
        let Some(Destination::Gguf(pending)) = state.pending.take() else {
            unreachable!()
        };
        state.lease = Some(pending.into_lease());
        for step in self.steps.iter().rev() {
            if step.validate_prepared {
                let Route::Prepared(owner) = step.owner.route() else {
                    unreachable!()
                };
                owner.validate_lease(request, state.lease.as_ref().unwrap())?;
            }
        }
        Ok(())
    }
    pub(super) fn refusal_bytes(&self, request: &TensorReadRequest) -> Option<usize> {
        self.steps
            .iter()
            .try_fold(request.key.len(), |maximum, step| {
                let contract = match step.owner.route() {
                    Route::Restricted(owner) => owner.contract.len(),
                    Route::Resolved(owner) => owner.contract.identity().len(),
                    _ => 0,
                };
                Some(maximum.max(request.key.len().checked_add(contract)?))
            })
    }
    pub(super) fn controls(&self) -> Option<usize> {
        std::mem::size_of::<&StepOwner>()
            .checked_add(std::mem::size_of::<Route<'static>>())?
            .checked_add(std::mem::size_of::<Result<(), StoreError>>())?
            .checked_add(std::mem::size_of::<std::slice::Iter<'static, StepOwner>>())
    }
    pub(super) fn metadata_bytes(&self) -> Option<usize> {
        Self::shared_payload_layout().size().checked_add(
            Layout::array::<StepOwner>(self.steps.capacity())
                .ok()?
                .size(),
        )
    }
    pub(super) fn shared_payload_layout() -> Layout {
        Layout::new::<Self>()
    }
}
