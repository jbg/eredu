//! Cold validation of logical recipe owners and independent binding lifetimes.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    ParameterGroupOwner, RealtimeMaterializationComponent, RealtimeMaterializationTask,
    RealtimeModelContractError,
};
use eredu_checkpoint::store::CheckpointSource;

fn invalid(detail: impl Into<String>) -> RealtimeModelContractError {
    RealtimeModelContractError::BindingPlan {
        detail: detail.into(),
    }
}

fn transformed(task: &RealtimeMaterializationTask) -> bool {
    matches!(
        task.lowering().kind(),
        crate::WeightLoweringKind::Transform | crate::WeightLoweringKind::DerivedTransform
    )
}

/// Validates the entire original source contract before deriving local replicas.
pub(super) fn validate<'a>(
    tasks: &'a [RealtimeMaterializationTask],
    source: &dyn CheckpointSource,
) -> Result<BTreeMap<&'a str, &'a str>, RealtimeModelContractError> {
    let mut components = BTreeMap::<&str, &RealtimeMaterializationComponent>::new();
    for task in tasks {
        for component in task.components() {
            let target = component.requirement().target().as_str();
            if components.insert(target, component).is_some() {
                return Err(invalid(format!(
                    "realtime component {target:?} is duplicated"
                )));
            }
            validate_component(component, source, transformed(task))?;
        }
    }

    // Iterative resolution keeps even a long cold alias chain off the stack.
    let mut canonical = BTreeMap::<&str, &str>::new();
    for &start in components.keys() {
        let mut current = start;
        let mut visited = BTreeSet::new();
        let root = loop {
            if let Some(&root) = canonical.get(current) {
                break root;
            }
            if !visited.insert(current) {
                return Err(invalid(format!(
                    "realtime alias cycle contains {current:?}"
                )));
            }
            let component = components[current];
            let Some(owner) = component.requirement().recipe_owner() else {
                break current;
            };
            let owner = owner.as_str();
            if owner == current {
                break current;
            }
            let owner_component = components
                .get(owner)
                .ok_or_else(|| invalid(format!("realtime alias owner {owner:?} is absent")))?;
            if component.requirement().recipe_identity()
                != owner_component.requirement().recipe_identity()
                || component.recipe() != owner_component.recipe()
                || component.recipe_output() != owner_component.recipe_output()
                || component.source_provenance() != owner_component.source_provenance()
            {
                return Err(invalid(format!(
                    "realtime alias {current:?} differs from its source owner {owner:?}"
                )));
            }
            current = owner;
        };
        for target in visited {
            canonical.insert(target, root);
        }
    }
    Ok(canonical)
}

fn partition(task: &RealtimeMaterializationTask) -> Option<&ParameterGroupOwner> {
    match task.owner() {
        ParameterGroupOwner::ExecutionUnit { .. } => Some(task.owner()),
        ParameterGroupOwner::StaticRole(_)
        | ParameterGroupOwner::StaticAnyOf(_)
        | ParameterGroupOwner::StaticUnitConsumers { .. } => None,
    }
}

/// Picks an existing target, never a fabricated extra owner, in each partition.
pub(super) fn local_owners<'a>(
    tasks: &'a [RealtimeMaterializationTask],
    canonical: &BTreeMap<&'a str, &'a str>,
) -> BTreeMap<&'a str, &'a str> {
    let mut representatives = BTreeMap::new();
    for task in tasks.iter().filter(|task| !transformed(task)) {
        for component in task.components() {
            let target = component.requirement().target().as_str();
            let root = canonical[target];
            let entry = representatives
                .entry((partition(task), root))
                .or_insert(target);
            // Preserve the original physical binding when it is local, even
            // if an alias appeared earlier in the selected task order.
            if root == target {
                *entry = target;
            }
        }
    }
    let mut local = BTreeMap::new();
    for task in tasks.iter().filter(|task| !transformed(task)) {
        for component in task.components() {
            let target = component.requirement().target().as_str();
            local.insert(
                target,
                representatives[&(partition(task), canonical[target])],
            );
        }
    }
    local
}

/// Validates exact original provenance and recipe metadata for one component.
pub(super) fn validate_component(
    component: &RealtimeMaterializationComponent,
    source: &dyn CheckpointSource,
    is_transformed: bool,
) -> Result<(), RealtimeModelContractError> {
    let target = component.requirement().target().as_str();
    for admitted in component.source_provenance() {
        let actual = source
            .source_provenance(&admitted.catalog_key)
            .map_err(|error| invalid(error.to_string()))?;
        if &actual != admitted {
            return Err(invalid(format!(
                "realtime component {target:?} differs from admitted source provenance"
            )));
        }
    }
    if let Some(recipe) = component.recipe() {
        let actual = recipe
            .infer(source)
            .map_err(|error| invalid(error.to_string()))?;
        if component.recipe_output() != Some(&actual) {
            return Err(invalid(format!(
                "realtime component {target:?} recipe output drifted"
            )));
        }
    } else if !is_transformed {
        return Err(invalid(format!(
            "realtime component {target:?} has no source recipe"
        )));
    }
    Ok(())
}

/// Validate published packed values, separately from the admitted dense recipe.
pub(super) fn materialized_metadata(
    task: &RealtimeMaterializationTask,
    component: &RealtimeMaterializationComponent,
    source: &dyn CheckpointSource,
) -> Result<eredu_checkpoint::store::TensorMetadata, RealtimeModelContractError> {
    use crate::RealtimeWeightComponentRole;
    use eredu_checkpoint::{LinearFormat, StoredDtype};
    let target = component.requirement().target().as_str();
    if !source.is_authoritative_materialized_key(target) {
        return Err(invalid(format!(
            "transformed realtime output {target:?} is not authoritative"
        )));
    }
    let descriptor = task.lowering().descriptor();
    let (bits, group, affine) = match descriptor.executable() {
        LinearFormat::Affine(config) => {
            config
                .validate()
                .map_err(|error| invalid(error.to_string()))?;
            (config.bits as usize, config.group_size as usize, true)
        }
        LinearFormat::MxFp4 => (4, 32, false),
        _ => {
            return Err(invalid(format!(
                "realtime transformed output {target:?} has no supported packed layout"
            )))
        }
    };
    let axis = descriptor.packed_axis().ok_or_else(|| {
        invalid(format!(
            "realtime transformed output {target:?} has no packed axis"
        ))
    })?;
    let mut shape = descriptor.logical_shape().to_vec();
    let extent = *shape
        .get(axis)
        .ok_or_else(|| invalid("realtime packed axis is outside the logical shape"))?;
    if extent % group != 0 {
        return Err(invalid(format!(
            "realtime transformed output {target:?} has incomplete quantization groups"
        )));
    }
    let metadata = source
        .source_metadata(target)
        .map_err(|error| invalid(error.to_string()))?;
    let scalar_bytes = match component.requirement().role() {
        RealtimeWeightComponentRole::Primary => {
            let packed = extent
                .checked_mul(bits)
                .ok_or_else(|| invalid("realtime packed extent overflow"))?;
            if packed % 32 != 0 || metadata.stored_dtype != StoredDtype::U32 {
                return Err(invalid(format!(
                    "realtime transformed primary {target:?} has invalid packed encoding"
                )));
            }
            shape[axis] = packed / 32;
            4u64
        }
        RealtimeWeightComponentRole::Scale | RealtimeWeightComponentRole::AffineBias => {
            if !affine && component.requirement().role() == RealtimeWeightComponentRole::AffineBias
            {
                return Err(invalid("MXFP4 has no affine bias output"));
            }
            shape[axis] = extent / group;
            // A recipe-backed requirement describes the original source;
            // generated companions describe the selected output directly.
            if component.recipe().is_none() && component.requirement().physical_shape() != shape {
                return Err(invalid(format!(
                    "realtime transformed companion {target:?} differs from selected geometry"
                )));
            }
            match (&metadata.stored_dtype, affine) {
                (StoredDtype::U8, false) => 1,
                (StoredDtype::F16 | StoredDtype::BF16, true) => 2,
                (StoredDtype::F32, true) => 4,
                _ => {
                    return Err(invalid(format!(
                        "realtime transformed companion {target:?} has invalid encoding"
                    )))
                }
            }
        }
    };
    let bytes = shape
        .iter()
        .try_fold(scalar_bytes, |bytes, &n| {
            u64::try_from(n).ok().and_then(|n| bytes.checked_mul(n))
        })
        .ok_or_else(|| invalid("realtime transformed output byte extent overflow"))?;
    if metadata.logical_shape != shape
        || metadata.physical_shape != shape
        || metadata.encoded_byte_len != bytes
    {
        return Err(invalid(format!(
            "realtime transformed output {target:?} differs from selected packed geometry"
        )));
    }
    Ok(metadata)
}
