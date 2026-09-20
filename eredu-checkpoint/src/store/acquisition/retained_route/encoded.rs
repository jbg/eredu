//! Shared batch authorization for retained memory and borrowed file constructors.
use super::*;

enum Refusal {
    Unknown { index: usize },
    Unauthorized { index: usize },
}
impl From<Refusal> for MemoryEncodedReadRouteError {
    fn from(value: Refusal) -> Self {
        match value {
            Refusal::Unknown { index } => {
                MemoryEncodedReadPlanError::UnknownTensor { index }.into()
            }
            Refusal::Unauthorized { index, .. } => Self::UnauthorizedTensor { index },
        }
    }
}
// A leaf or an unqualified dependency has no child. Callers select their concrete
// leaf before this shared wrapper transition, preserving format-specific APIs.
fn child<'a>(
    route: Route<'a>,
    keys: &[String],
) -> Result<Option<&'a RetainedCheckpointSource>, Refusal> {
    Ok(match route {
        Route::Unavailable | Route::Memory(_) | Route::Safetensors(_) | Route::Gguf(_) => None,
        Route::Materialized(owner) => owner.encoded_read_owner(keys),
        Route::Prepared(owner) => {
            // The ordinary encoded path delegates immediately when its child
            // promises a fixed catalog. Authenticate all concrete dependencies
            // of that promise, including unselected composite children.
            if stable_recipe_presence(&owner.source) != Some(true) {
                return Ok(None);
            }
            Some(&owner.source)
        }
        Route::Restricted(owner) => {
            for (index, key) in keys.iter().enumerate() {
                if !owner.is_authorized(key) {
                    return Err(Refusal::Unauthorized { index });
                }
            }
            Some(&owner.source)
        }
        Route::Composite(owner) => owner
            .encoded_read_owner(keys)
            .map_err(|index| Refusal::Unknown { index })?,
        Route::Resolved(owner) => {
            for (index, key) in keys.iter().enumerate() {
                let Some(materialized) = stable_materialized_key(&owner.source, key) else {
                    return Ok(None);
                };
                if !materialized && !owner.contract.source_keys().contains(key) {
                    return Err(Refusal::Unauthorized { index });
                }
            }
            Some(&owner.source)
        }
    })
}

/// An owning memory route can release its enclosing views before construction.
pub(in crate::store) fn encoded_memory_source(
    source: &RetainedCheckpointSource,
    keys: &[String],
) -> Result<Option<Arc<MemoryWeightStore>>, MemoryEncodedReadRouteError> {
    let mut current = source.clone();
    loop {
        let Some(owner) = current.acquisition_owner() else {
            return Ok(None);
        };
        if let Owner::Memory(store) = owner.0 {
            return Ok(Some(store));
        }
        let Some(next) = child(owner.route(), keys).map_err(MemoryEncodedReadRouteError::from)?
        else {
            return Ok(None);
        };
        current = next.clone();
    }
}

// The source lends this route for its own lifetime. Its loan must name exactly
// the concrete owner already authenticated by acquisition_owner, in the same
// variant. Neither a matching catalog nor another owner's loan can satisfy it.
fn borrowed_route<'a>(
    owner: &PreparedAcquisitionOwner,
    source: &'a RetainedCheckpointSource,
) -> Option<Route<'a>> {
    let loan = source.prepared_acquisition_source().0;
    let same = match (owner.route(), loan) {
        (Route::Memory(a), Route::Memory(b)) => std::ptr::eq(a, b),
        (Route::Materialized(a), Route::Materialized(b)) => std::ptr::eq(a, b),
        (Route::Safetensors(a), Route::Safetensors(b)) => std::ptr::eq(a, b),
        (Route::Gguf(a), Route::Gguf(b)) => std::ptr::eq(a, b),
        (Route::Prepared(a), Route::Prepared(b)) => std::ptr::eq(a, b),
        (Route::Restricted(a), Route::Restricted(b)) => std::ptr::eq(a, b),
        (Route::Composite(a), Route::Composite(b)) => std::ptr::eq(a, b),
        (Route::Resolved(a), Route::Resolved(b)) => std::ptr::eq(a, b),
        _ => false,
    };
    same.then_some(loan)
}

/// File plans borrow their source during inspection/construction. Refusals retain
/// the actual authorization or header owner independently of that loan. Routing
/// creates no allocated rows and invokes no ordinary read preparation.
pub(in crate::store) fn encoded_file_source<'a>(
    source: &'a RetainedCheckpointSource,
    keys: &[String],
) -> Result<Option<&'a SafetensorsWeightStore>, SafetensorsEncodedReadPlanError> {
    let mut current = source;
    loop {
        let Some(owner) = current.acquisition_owner() else {
            return Ok(None);
        };
        let Some(route) = borrowed_route(&owner, current) else {
            return Ok(None);
        };
        if let Route::Safetensors(store) = route {
            return Ok(Some(store));
        }
        let next = match child(route, keys) {
            Ok(next) => next,
            Err(Refusal::Unknown { index }) => {
                return Err(SafetensorsEncodedReadPlanError::unknown(index));
            }
            Err(Refusal::Unauthorized { index, .. }) => {
                return Err(SafetensorsEncodedReadPlanError::unauthorized(index, owner));
            }
        };
        let Some(next) = next else {
            return Ok(None);
        };
        current = next;
    }
}
