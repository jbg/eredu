//! Bind reusable mechanism storage to neutral resource identities.
//!
//! Logical tensor extents are never promoted to physical allocations. Explicit
//! backing identities establish sharing; shape or source names do not. This
//! adapter retains lifetime declarations but does not compose a live peak.

use eredu_core::{resources::*, ObservationKind, Observed};
use eredu_nn::mechanism_memory::*;
use std::collections::BTreeMap;

/// Physical context supplied by the invocation and parameter residency owners.
#[derive(Debug, Clone)]
pub struct MechanismResourceQuery {
    /// Unique invocation owner; repeated calls for the same invocation reuse it.
    pub invocation: ResourceIdentity,
    /// Physical execution pool, or a specific reason it is unknown.
    pub execution_pool: Observed<ResourceIdentity>,
    /// Physical host pool. Unified memory can use the same identity as execution.
    pub host_pool: Observed<ResourceIdentity>,
    /// Authoritative allocation identities for backend `Owner` keys.
    /// Missing entries remain unbound, never invocation-local copies.
    pub owner_backings: BTreeMap<String, ResourceIdentity>,
}

/// Storage facts with their original geometry and retention declarations.
#[derive(Debug, Clone)]
pub struct MechanismResourceDescription {
    /// Identified storage only, with explicit incomplete coverage when necessary.
    pub resources: ResourceDescription,
    /// Original contract, including unbound storage and retention boundaries.
    pub contract: MechanismMemoryContract,
    /// Contract storage names mapped to allocation identities. Multiple names may
    /// refer to one allocation; omitted names lack authoritative backing facts.
    pub storage_bindings: BTreeMap<String, ResourceIdentity>,
}

/// Resolves placement and allocation ownership without executing the mechanism.
///
/// Conflicting facts for one backing are rejected. Sharing an owner identity
/// requires agreement about its payload, capacity and physical pool; it does
/// not establish invocation order or justify adding simultaneous lifetimes.
pub fn describe_mechanism_resources(
    contract: MechanismMemoryContract,
    query: &MechanismResourceQuery,
) -> Result<MechanismResourceDescription, ResourceDescriptionError> {
    contract
        .validate()
        .map_err(|error| ResourceDescriptionError::Invalid(error.to_string()))?;
    let mut missing = contract.missing.clone();
    let mut allocations: BTreeMap<ResourceIdentity, ResourceAllocation> = BTreeMap::new();
    let mut bindings = BTreeMap::new();
    for storage in &contract.storage {
        let identity = match &storage.backing {
            MechanismBacking::Invocation => {
                invocation_storage_identity(&query.invocation, &storage.name)
            }
            MechanismBacking::Owner(key) => {
                let Some(identity) = query.owner_backings.get(key) else {
                    missing.push(format!(
                        "{}: backing owner {key} is not bound",
                        storage.name
                    ));
                    continue;
                };
                identity.clone()
            }
            MechanismBacking::Unknown => {
                missing.push(format!(
                    "{}: physical backing identity is unknown",
                    storage.name
                ));
                continue;
            }
        };
        let placement = match storage.placement {
            MechanismPlacement::Execution => query.execution_pool.clone(),
            MechanismPlacement::Host => query.host_pool.clone(),
            MechanismPlacement::Unknown => Observed::Unavailable {
                reason: format!("{}: mechanism physical pool is unknown", storage.name),
            },
        };
        // Validate provenance before aliases are merged: a guessed host pool
        // must not inherit authority from an equal execution-pool identity.
        if let Observed::Available { kind, source, .. } = &placement {
            if !matches!(
                kind,
                ObservationKind::Exact | ObservationKind::Observational
            ) || source.trim().is_empty()
            {
                return Err(ResourceDescriptionError::Invalid(
                    "physical pool binding needs authoritative provenance".into(),
                ));
            }
        }
        if storage.retention == StorageRetention::Unknown {
            missing.push(format!("{}: storage retention is unknown", storage.name));
        }
        let extent = ResourceExtent {
            payload: bounds(storage.payload, &storage.detail),
            capacity: bounds(storage.capacity, &storage.detail),
        };
        let usage = ResourceUse {
            owner: query.invocation.clone(),
            role: match storage.role {
                MechanismStorageRole::Output => ResourceRole::RetainedTensor,
                MechanismStorageRole::State => ResourceRole::MutableState,
                MechanismStorageRole::Scratch => ResourceRole::Workspace,
                MechanismStorageRole::ParameterConversion => ResourceRole::Parameters,
            },
        };
        if let Some(existing) = allocations.get_mut(&identity) {
            let ResourceSize::Fixed { extent: prior } = &existing.size else {
                unreachable!()
            };
            if !same_bounds(&prior.payload, &extent.payload)
                || !same_bounds(&prior.capacity, &extent.capacity)
                || !same_pool(&existing.placement, &placement)
            {
                return Err(ResourceDescriptionError::Invalid(format!(
                    "{}: conflicting descriptions of shared backing",
                    storage.name
                )));
            }
            if !existing.uses.contains(&usage) {
                existing.uses.push(usage);
            }
        } else {
            allocations.insert(
                identity.clone(),
                ResourceAllocation {
                    identity: identity.clone(),
                    uses: vec![usage],
                    placement,
                    size: ResourceSize::Fixed { extent },
                },
            );
        }
        bindings.insert(storage.name.clone(), identity);
    }
    missing.sort();
    missing.dedup();
    let resources = ResourceDescription {
        schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
        scope: query.invocation.clone(),
        context: ResourceContext::default(),
        horizon: ResourceContext::default(),
        coverage: if missing.is_empty() {
            ResourceCoverage::Complete
        } else {
            ResourceCoverage::Partial { reasons: missing }
        },
        allocations: allocations.into_values().collect(),
    };
    resources.validate()?;
    Ok(MechanismResourceDescription {
        resources,
        contract,
        storage_bindings: bindings,
    })
}

pub(crate) fn invocation_storage_identity(
    invocation: &ResourceIdentity,
    name: &str,
) -> ResourceIdentity {
    ResourceIdentity {
        scope: invocation.scope.clone(),
        // Length framing separates invocation keys from local names.
        key: format!(
            "mechanism:{}:{}{}",
            invocation.key.len(),
            invocation.key,
            name
        ),
    }
}

fn bounds(bytes: MechanismBytes, detail: &str) -> ResourceByteBounds {
    ResourceByteBounds {
        lower_bytes: bytes.lower,
        upper_bytes: bytes.upper,
        kind: if bytes.upper == Some(bytes.lower) {
            ObservationKind::Exact
        } else {
            ObservationKind::Estimated
        },
        detail: detail.into(),
    }
}

fn same_bounds(left: &ResourceByteBounds, right: &ResourceByteBounds) -> bool {
    left.lower_bytes == right.lower_bytes
        && left.upper_bytes == right.upper_bytes
        && left.kind == right.kind
}

fn same_pool(left: &Observed<ResourceIdentity>, right: &Observed<ResourceIdentity>) -> bool {
    match (left, right) {
        (Observed::Available { value: left, .. }, Observed::Available { value: right, .. }) => {
            left == right
        }
        _ => left == right,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_nn::TensorElementType;

    fn id(key: &str) -> ResourceIdentity {
        ResourceIdentity {
            scope: "model-1".into(),
            key: key.into(),
        }
    }
    fn query() -> MechanismResourceQuery {
        let pool = Observed::Available {
            value: id("unified"),
            kind: ObservationKind::Exact,
            source: "device".into(),
        };
        MechanismResourceQuery {
            invocation: id("linear-1"),
            execution_pool: pool.clone(),
            host_pool: pool,
            owner_backings: BTreeMap::from([("converted-weight".into(), id("weight-f32"))]),
        }
    }
    fn storage(name: &str, backing: MechanismBacking) -> MechanismStorage {
        MechanismStorage {
            name: name.into(),
            role: MechanismStorageRole::ParameterConversion,
            payload: MechanismBytes::exact(512),
            capacity: MechanismBytes::unknown(512),
            backing,
            placement: MechanismPlacement::Execution,
            retention: StorageRetention::ParameterOwner,
            detail: "retained conversion".into(),
        }
    }
    fn contract(storage: Vec<MechanismStorage>) -> MechanismMemoryContract {
        MechanismMemoryContract {
            values: vec![LogicalValue {
                name: "borrowed-broadcast".into(),
                shape: vec![1024, 1024],
                element: TensorElementType::F32,
                kind: LogicalValueKind::Input,
            }],
            storage,
            missing: vec![],
        }
    }

    #[test]
    fn explicit_backings_share_storage_but_logical_views_allocate_nothing() {
        let mut alias = storage("alias", MechanismBacking::Owner("converted-weight".into()));
        alias.detail = "same physical conversion through another use".into();
        alias.placement = MechanismPlacement::Host;
        let description = describe_mechanism_resources(
            contract(vec![
                storage("first", MechanismBacking::Owner("converted-weight".into())),
                alias,
            ]),
            &query(),
        )
        .unwrap();
        assert_eq!(description.resources.allocations.len(), 1);
        assert_eq!(
            description.storage_bindings["first"],
            description.storage_bindings["alias"]
        );
        assert_eq!(
            description.contract.storage[0].retention,
            StorageRetention::ParameterOwner
        );
        let ResourceSize::Fixed { extent } = &description.resources.allocations[0].size else {
            panic!()
        };
        assert_eq!(extent.capacity.upper_bytes, None);
        assert_eq!(extent.payload.lower_bytes, 512);
    }

    #[test]
    fn unknown_and_unbound_storage_cannot_become_copies() {
        let description = describe_mechanism_resources(
            contract(vec![
                storage("alias", MechanismBacking::Unknown),
                storage(
                    "conversion",
                    MechanismBacking::Owner("missing-owner".into()),
                ),
            ]),
            &query(),
        )
        .unwrap();
        assert!(description.resources.allocations.is_empty());
        assert!(
            matches!(description.resources.coverage, ResourceCoverage::Partial { reasons } if reasons.len() == 2)
        );
        assert_eq!(description.contract.storage.len(), 2);
    }

    #[test]
    fn shared_identity_does_not_launder_estimated_placement() {
        let first = storage("device", MechanismBacking::Owner("converted-weight".into()));
        let mut second = first.clone();
        second.name = "host".into();
        second.placement = MechanismPlacement::Host;
        let mut query = query();
        if let Observed::Available { kind, .. } = &mut query.host_pool {
            *kind = ObservationKind::Estimated;
        }
        assert!(describe_mechanism_resources(contract(vec![first, second]), &query).is_err());
    }

    #[test]
    fn independent_invocations_get_distinct_storage_and_shared_conflicts_fail() {
        let value = contract(vec![storage("output", MechanismBacking::Invocation)]);
        let first = describe_mechanism_resources(value.clone(), &query()).unwrap();
        let mut other = query();
        other.invocation = id("linear-2");
        let second = describe_mechanism_resources(value, &other).unwrap();
        assert_ne!(
            first.storage_bindings["output"],
            second.storage_bindings["output"]
        );
        let a = storage("a", MechanismBacking::Owner("converted-weight".into()));
        let mut b = a.clone();
        b.name = "b".into();
        b.payload = MechanismBytes::exact(1024);
        assert!(describe_mechanism_resources(contract(vec![a, b]), &query()).is_err());
    }
}
