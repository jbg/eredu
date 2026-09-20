//! Shared batch authorization for retained memory and borrowed file constructors.
use super::*;

enum Refusal {
    Unknown { index: usize },
    CatalogMismatch { index: usize },
    Unauthorized { index: usize },
}
impl From<Refusal> for MemoryEncodedReadRouteError {
    fn from(value: Refusal) -> Self {
        match value {
            Refusal::Unknown { index } => {
                MemoryEncodedReadPlanError::UnknownTensor { index }.into()
            }
            Refusal::Unauthorized { index, .. } => Self::UnauthorizedTensor { index },
            Refusal::CatalogMismatch { index } => Self::PreparedCatalogMismatch { index },
        }
    }
}
// A leaf or an unqualified dependency has no child. Callers select their concrete
// leaf before this shared wrapper transition, preserving format-specific APIs.
fn child<'a>(
    route: Route<'a>,
    keys: &[String],
) -> Result<
    Option<(
        &'a RetainedCheckpointSource,
        Option<&'a PreparedCheckpointSource>,
    )>,
    Refusal,
> {
    let mut prepared = None;
    let source = match route {
        Route::Unavailable | Route::Memory(_) | Route::Safetensors(_) | Route::Gguf(_) => None,
        Route::Materialized(owner) => owner.encoded_read_owner(keys),
        Route::Prepared(owner) => {
            // The ordinary encoded path delegates immediately when its child
            // promises a fixed catalog. Authenticate all concrete dependencies
            // of that promise, including unselected composite children. A known
            // cacheless child instead needs its expected entries and a comparison
            // against the completed immutable leaf metadata.
            match stable_recipe_presence(&owner.source) {
                None => return Ok(None),
                Some(true) => {}
                Some(false) => {
                    prepared = Some(owner);
                    for (index, key) in keys.iter().enumerate() {
                        if !owner.catalog.contains_key(key) {
                            return Err(Refusal::Unknown { index });
                        }
                    }
                }
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
    };
    Ok(source.map(|source| (source, prepared)))
}

// Compare after the concrete child is inspected, from inner to outer views,
// preserving ordinary encoded-read validation and error order. The leaf plan
// keeps immutable metadata; no route rows, catalog clones or cache are created.
fn validate_prepared<'a>(
    prepared: Option<&PreparedCheckpointSource>,
    count: usize,
    mut metadata: impl FnMut(usize) -> &'a TensorMetadata,
) -> Result<(), Refusal> {
    let Some(owner) = prepared else { return Ok(()) };
    for index in 0..count {
        let metadata = metadata(index);
        let expected = owner
            .catalog
            .get(&metadata.name)
            .ok_or(Refusal::Unknown { index })?;
        if !expected.matches_encoded_metadata(metadata) {
            return Err(Refusal::CatalogMismatch { index });
        }
    }
    Ok(())
}

/// An owning memory plan can release its enclosing views after inspection.
pub(in crate::store) fn encoded_memory_plan<'a>(
    source: &RetainedCheckpointSource,
    keys: &'a [String],
) -> Result<Option<MemoryEncodedReadPlan<'a>>, MemoryEncodedReadRouteError> {
    let Some(owner) = source.acquisition_owner() else {
        return Ok(None);
    };
    if let Owner::Memory(store) = owner.0 {
        return MemoryEncodedReadPlan::retained(store, keys)
            .map(Some)
            .map_err(Into::into);
    }
    let Some((next, prepared)) =
        child(owner.route(), keys).map_err(MemoryEncodedReadRouteError::from)?
    else {
        return Ok(None);
    };
    let Some(plan) = encoded_memory_plan(next, keys)? else {
        return Ok(None);
    };
    validate_prepared(prepared, keys.len(), |index| plan.metadata(index))
        .map_err(MemoryEncodedReadRouteError::from)?;
    Ok(Some(plan))
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
pub(in crate::store) fn encoded_file_plan<'a>(
    source: &'a RetainedCheckpointSource,
    keys: &'a [String],
) -> Result<Option<SafetensorsEncodedReadPlan<'a>>, SafetensorsEncodedReadPlanError> {
    let Some(owner) = source.acquisition_owner() else {
        return Ok(None);
    };
    let Some(route) = borrowed_route(&owner, source) else {
        return Ok(None);
    };
    if let Route::Safetensors(store) = route {
        return SafetensorsEncodedReadPlan::new(store, keys).map(Some);
    }
    let refusal = |error| match error {
        Refusal::Unknown { index } => SafetensorsEncodedReadPlanError::unknown(index),
        Refusal::Unauthorized { index } => {
            SafetensorsEncodedReadPlanError::unauthorized(index, owner)
        }
        Refusal::CatalogMismatch { index } => {
            SafetensorsEncodedReadPlanError::catalog_mismatch(index)
        }
    };
    let (next, prepared) = match child(route, keys) {
        Ok(Some(next)) => next,
        Ok(None) => return Ok(None),
        Err(error) => return Err(refusal(error)),
    };
    let Some(plan) = encoded_file_plan(next, keys)? else {
        return Ok(None);
    };
    validate_prepared(prepared, keys.len(), |index| plan.metadata(index)).map_err(refusal)?;
    Ok(Some(plan))
}
