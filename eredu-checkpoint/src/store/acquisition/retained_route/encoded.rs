//! Shared batch authorization for retained memory and borrowed file constructors.
use super::*;

enum Refusal<'a> {
    Unknown { index: usize },
    Unauthorized { index: usize, contract: &'a str },
}
impl From<Refusal<'_>> for MemoryEncodedReadRouteError {
    fn from(value: Refusal<'_>) -> Self {
        match value {
            Refusal::Unknown { index } => {
                MemoryEncodedReadPlanError::UnknownTensor { index }.into()
            }
            Refusal::Unauthorized { index, .. } => Self::UnauthorizedTensor { index },
        }
    }
}
impl<'a> From<Refusal<'a>> for SafetensorsEncodedReadPlanError<'a> {
    fn from(value: Refusal<'a>) -> Self {
        match value {
            Refusal::Unknown { index } => Self::UnknownTensor { index },
            Refusal::Unauthorized { index, contract } => {
                Self::UnauthorizedTensor { index, contract }
            }
        }
    }
}

// A leaf or an unqualified dependency has no child. Callers select their concrete
// leaf before this shared wrapper transition, preserving format-specific APIs.
fn child<'a>(
    route: Route<'a>,
    keys: &[String],
) -> Result<Option<&'a RetainedCheckpointSource>, Refusal<'a>> {
    Ok(match route {
        Route::Unavailable | Route::Memory(_) | Route::Safetensors(_) | Route::Gguf(_) => None,
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
                    return Err(Refusal::Unauthorized {
                        index,
                        contract: &owner.contract,
                    });
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
                    return Err(Refusal::Unauthorized {
                        index,
                        contract: owner.contract.identity(),
                    });
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

/// File plans borrow their source so a retained header error remains an exact
/// loan. The original root keeps every selected wrapper/leaf alive during this
/// cold selection, inspection and construction, with no allocated route rows.
pub(in crate::store) fn encoded_file_source<'a>(
    source: &'a RetainedCheckpointSource,
    keys: &[String],
) -> Result<Option<&'a SafetensorsWeightStore>, SafetensorsEncodedReadPlanError<'a>> {
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
        let Some(next) = child(route, keys).map_err(SafetensorsEncodedReadPlanError::from)? else {
            return Ok(None);
        };
        current = next;
    }
}
