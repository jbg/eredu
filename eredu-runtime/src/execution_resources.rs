//! Resource producers over ordinary selected execution and exact prepared slots.
//!
//! These describe hypothetical prepared geometry at an explicit prefix, not a
//! measurement of a live session. They never acquire a residency unit, inspect a
//! native tensor, or compose an execution peak. Missing physical decomposition,
//! capacity and mechanism contracts remain visible.

use std::collections::{BTreeMap, BTreeSet};

use eredu_core::{
    cache::{StateComponentRole, StateElementBounds},
    resources::*,
    ObservationKind, Observed,
};
use eredu_nn::ParameterMetadata;

use crate::{
    parameter_operations::{PreparedParameterLocation, PreparedParameterSlot},
    PreparedReplicatedTextContract, SelectedReplicatedTextRealization, SelectedStateRealization,
    StateComponentPlacement, StateSegmentLifetime,
};

/// Typed hypothetical dimensions for the selected execution's ordinary state.
///
/// A scope identifies one independently materialized execution instance, not its
/// checkpoint. Separate instances/ranks require distinct scopes. A supplied pool
/// identifies physical capacity (host/device aliases on unified memory use the
/// same identity). Omit knowledge with `Observed::Unavailable`, never a guessed
/// host pool. The query does not assert a current installed state or residency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedResourceQuery {
    /// Instance namespace and document scope.
    pub scope: ResourceIdentity,
    /// Positive batch extent for symbolic state dimensions.
    pub batch_size: u64,
    /// Hypothetical persisted prefix at the starting boundary.
    pub prefix_positions: u64,
    /// Inclusive growth interval beginning at `prefix_positions`.
    pub additional_positions: u64,
    /// Authoritative or observed physical pool for device-resident outputs.
    pub device_pool: Observed<ResourceIdentity>,
}

impl SelectedReplicatedTextRealization {
    /// Describes selected state geometry without creating modules or reading weights.
    ///
    /// Cold selection retains task geometry, but not proof of backing allocation
    /// sharing/decomposition. Parameters and mechanism scratch remain named gaps.
    /// Exact prepared slot facts can be supplied through [`describe_prepared_resources`].
    pub fn describe_prepared_resources(
        &self,
        query: &PreparedResourceQuery,
    ) -> Result<ResourceDescription, ResourceDescriptionError> {
        describe_prepared_resources(self, Some(self.state()), &[], &[], query)
    }
}

impl PreparedReplicatedTextContract {
    /// Describes the retained selection without binding, loading or executing it.
    pub fn describe_prepared_resources(
        &self,
        query: &PreparedResourceQuery,
    ) -> Result<ResourceDescription, ResourceDescriptionError> {
        self.selected().describe_prepared_resources(query)
    }
}

/// Derives storage descriptions from the exact prepared module/binding pair.
///
/// `state` is the selected **local** state realization, or `None` for a stateless
/// rank. `slots` and `declarations` are those retained by ordinary preparation,
/// including its rank-local physical geometry; they must not be reconstructed
/// from source names. Only slots with canonical materialization backing facts
/// become allocations. Bank catalogs aggregating members remain explicit gaps.
/// This function is also used by `ReplicatedTextSession::describe_prepared_resources`.
pub fn describe_prepared_resources(
    selected: &SelectedReplicatedTextRealization,
    state: Option<&SelectedStateRealization>,
    slots: &[PreparedParameterSlot],
    declarations: &[ParameterMetadata],
    query: &PreparedResourceQuery,
) -> Result<ResourceDescription, ResourceDescriptionError> {
    if query.batch_size == 0 {
        return Err(invalid("resource query batch must be positive"));
    }
    let end = query
        .prefix_positions
        .checked_add(query.additional_positions)
        .ok_or_else(|| invalid("resource prefix horizon overflowed"))?;
    let mut description = ResourceDescription {
        schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
        scope: query.scope.clone(),
        context: ResourceContext {
            dimensions: BTreeMap::from([
                ("batch_size".into(), query.batch_size),
                ("prefix_positions".into(), query.prefix_positions),
            ]),
        },
        horizon: ResourceContext {
            dimensions: BTreeMap::from([(
                "additional_positions".into(),
                query.additional_positions,
            )]),
        },
        coverage: ResourceCoverage::Unspecified,
        allocations: Vec::new(),
    };
    // Validate caller identities and pool provenance even when no device output exists.
    description.validate()?;
    ResourceAllocation {
        identity: id(query, "query-validation"),
        uses: vec![ResourceUse {
            owner: id(query, "query-validation"),
            role: ResourceRole::Workspace,
        }],
        placement: query.device_pool.clone(),
        size: ResourceSize::Fixed {
            extent: ResourceExtent {
                payload: ResourceByteBounds::exact(0),
                capacity: ResourceByteBounds::exact(0),
            },
        },
    }
    .validate()?;
    let mut missing = BTreeSet::new();
    parameter_resources(
        selected,
        slots,
        declarations,
        query,
        &mut description.allocations,
        &mut missing,
    )?;
    if let Some(state) = state {
        state_resources(
            state,
            query,
            end,
            &mut description.allocations,
            &mut missing,
        )?;
    }
    for (group_index, group) in selected
        .requirements()
        .execution_graph()
        .groups()
        .iter()
        .enumerate()
    {
        let units = selected.requirements().execution_units();
        for ordinal in 0..units.len() {
            let address = units.address(ordinal).expect("validated execution unit");
            if address.group() == group_index {
                missing.insert(format!("execution group {:?} unit {}: mechanism workspace and retained outputs are not described", group.id(), address.index()));
            }
        }
    }
    for parameter in selected.requirements().parameters() {
        if let crate::ReplicatedTextParameterOwner::StaticRole(role)
        | crate::ReplicatedTextParameterOwner::StaticUnitConsumers { role, .. } =
            parameter.owner()
        {
            missing.insert(format!("static module {role:?}: mechanism workspace and retained outputs are not described"));
        }
    }
    missing.insert(format!("selected operators {:?}: implementation scratch, cached conversions, allocator padding and evaluation retention are not described", selected.requirements().operators()));
    missing.insert("processor inputs, sampling, observation records and transaction/snapshot storage are outside the prepared module description".into());
    description.coverage = ResourceCoverage::Partial {
        reasons: missing.into_iter().collect(),
    };
    description.validate()?;
    Ok(description)
}

fn parameter_resources(
    selected: &SelectedReplicatedTextRealization,
    slots: &[PreparedParameterSlot],
    declarations: &[ParameterMetadata],
    query: &PreparedResourceQuery,
    allocations: &mut Vec<ResourceAllocation>,
    missing: &mut BTreeSet<String>,
) -> Result<(), ResourceDescriptionError> {
    let mut grouped =
        BTreeMap::<String, (eredu_checkpoint::recipe::RecipeMetadata, ResourceAllocation)>::new();
    let mut present = BTreeSet::new();
    for slot in slots {
        let name = slot.parameter.id.as_str();
        present.insert(name);
        let Some(backing) = &slot.backing else {
            missing.insert(format!(
                "parameter {name:?}: materialization backing/sharing is not described"
            ));
            continue;
        };
        if backing.trim().is_empty() {
            return Err(invalid("prepared parameter backing is empty"));
        }
        let batch = match &slot.location {
            PreparedParameterLocation::Static { .. } => "static".to_owned(),
            PreparedParameterLocation::Unit { ordinal, address } => {
                format!("unit/{ordinal}/{}/{}", address.group(), address.index())
            }
            PreparedParameterLocation::Prediction { module } => {
                missing.insert(format!("prediction module {module}: invocation and parameter resource contracts are not described"));
                continue;
            }
            PreparedParameterLocation::Bank { bank, unit } => {
                missing.insert(format!("parameter bank {bank} unit {unit}: per-member backing allocations are not described"));
                continue;
            }
        };
        let key = format!("parameter/{batch}/{backing}");
        let placement = if selected.residency().is_fully_resident()
            || matches!(slot.location, PreparedParameterLocation::Static { .. })
        {
            query.device_pool.clone()
        } else {
            missing.insert(format!("parameter {name:?}: selected bounded residency can create host/device copies whose backing is not described"));
            Observed::Unavailable {
                reason: "bounded parameter residency does not identify one physical pool".into(),
            }
        };
        let payload = ResourceByteBounds::exact(slot.materialized.byte_len);
        let usage = ResourceUse {
            owner: id(
                query,
                format!("module/{:?}/parameter/{name}", slot.location),
            ),
            role: ResourceRole::Parameters,
        };
        match grouped.entry(key.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((
                    slot.materialized.clone(),
                    ResourceAllocation {
                        identity: id(query, key),
                        uses: vec![usage],
                        placement,
                        size: ResourceSize::Fixed {
                            extent: extent(payload),
                        },
                    },
                ));
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                let (metadata, allocation) = entry.get_mut();
                if metadata != &slot.materialized || allocation.placement != placement {
                    return Err(invalid(format!("shared prepared backing {key:?} has conflicting output geometry or placement")));
                }
                if allocation.uses.contains(&usage) {
                    return Err(invalid(format!(
                        "prepared parameter use {name:?} is duplicated"
                    )));
                }
                allocation.uses.push(usage);
            }
        }
    }
    for declaration in declarations {
        if !present.contains(declaration.id.as_str()) {
            missing.insert(format!("parameter {:?}: declaration has no retained local prepared output (remote, bank-owned, or unavailable)", declaration.id.as_str()));
        }
    }
    if slots.is_empty() {
        for task in selected
            .materialization_tasks()
            .iter()
            .chain(selected.auxiliary_materialization_tasks())
        {
            missing.insert(format!(
                "parameter {:?} selected {:?}/{:?}: exact prepared output backings are unavailable",
                task.name(),
                task.executable(),
                task.lowering()
            ));
        }
    }
    allocations.extend(grouped.into_values().map(|(_, allocation)| allocation));
    Ok(())
}

fn state_resources(
    state: &SelectedStateRealization,
    query: &PreparedResourceQuery,
    end: u64,
    allocations: &mut Vec<ResourceAllocation>,
    missing: &mut BTreeSet<String>,
) -> Result<(), ResourceDescriptionError> {
    for selected in state.components() {
        let component = selected.component();
        let key = format!(
            "state/{}/{}",
            selected.layer(),
            component.role().stable_name()
        );
        if selected.placement() == StateComponentPlacement::Paged {
            missing.insert(format!(
                "{key}: paged block allocation and tier decomposition are not described"
            ));
            continue;
        }
        let segment = state
            .layout()
            .segment_for_layer(selected.layer())
            .expect("selected component belongs to state segment");
        if segment.lifetime() != StateSegmentLifetime::Persistent {
            missing.insert(format!(
                "{key}: frame-local segment {:?} needs a frame horizon",
                segment.id().as_str()
            ));
            continue;
        }
        let offset = u64::from(segment.processed_token_offset().unsigned_abs());
        let start = query.prefix_positions.saturating_sub(offset);
        let end = end.saturating_sub(offset);
        let current = component
            .element_bounds(query.batch_size, start)
            .map_err(|e| invalid(e.to_string()))?;
        let peak = component
            .element_peak_bounds(query.batch_size, start, end)
            .map_err(|e| invalid(e.to_string()))?;
        let width = u64::from(selected.storage_dtype().bytes().get());
        let mut current = byte_bounds(current, width)?;
        let mut peak = byte_bounds(peak, width)?;
        if !matches!(component.role(), StateComponentRole::Fixed(_)) {
            if let Some(window) = state
                .layout()
                .layer(selected.layer())
                .and_then(|layer| layer.attention())
                .and_then(|attention| attention.window())
            {
                // Attention visibility does not force storage truncation. The
                // selected implementation may keep the complete prefix.
                current.lower_bytes = byte_bounds(
                    component
                        .element_bounds(query.batch_size, start.min(u64::from(window.get())))
                        .map_err(|e| invalid(e.to_string()))?,
                    width,
                )?
                .lower_bytes;
                peak.lower_bytes = byte_bounds(
                    component
                        .element_peak_bounds(
                            query.batch_size,
                            start.min(u64::from(window.get())),
                            end.min(u64::from(window.get())),
                        )
                        .map_err(|e| invalid(e.to_string()))?,
                    width,
                )?
                .lower_bytes;
                for bound in [&mut current, &mut peak] {
                    if bound.upper_bytes != Some(bound.lower_bytes) {
                        bound.kind = ObservationKind::Estimated;
                    }
                    bound.detail = "selected sliding attention permits window payload through full-prefix retention; truncation mechanism is not described".into();
                }
            }
        }
        missing.insert(format!("{key}: allocation capacity, replacement copies and implementation retention are not described"));
        allocations.push(ResourceAllocation {
            identity: id(query, &key),
            uses: vec![ResourceUse {
                owner: id(
                    query,
                    format!(
                        "segment/{}/layer/{}/{}",
                        segment.id().as_str(),
                        selected.layer(),
                        component.role().stable_name()
                    ),
                ),
                role: ResourceRole::MutableState,
            }],
            placement: query.device_pool.clone(),
            size: ResourceSize::ContextDependent {
                current: extent(current),
                horizon_peak: extent(peak),
            },
        });
    }
    Ok(())
}

fn byte_bounds(
    elements: StateElementBounds,
    width: u64,
) -> Result<ResourceByteBounds, ResourceDescriptionError> {
    let lower = elements
        .minimum
        .checked_mul(width)
        .ok_or_else(|| invalid("state payload lower bound overflowed"))?;
    let upper = elements
        .maximum
        .checked_mul(width)
        .ok_or_else(|| invalid("state payload upper bound overflowed"))?;
    Ok(ResourceByteBounds {
        lower_bytes: lower,
        upper_bytes: Some(upper),
        kind: if lower == upper {
            ObservationKind::Exact
        } else {
            ObservationKind::Estimated
        },
        detail: "ordinary selected component shape, presence and storage dtype".into(),
    })
}

fn extent(payload: ResourceByteBounds) -> ResourceExtent {
    ResourceExtent {
        capacity: ResourceByteBounds::unknown(
            payload.lower_bytes,
            "backing capacity/alignment is not specified by the ordinary prepared contract",
        ),
        payload,
    }
}

fn id(query: &PreparedResourceQuery, key: impl Into<String>) -> ResourceIdentity {
    ResourceIdentity {
        scope: format!(
            "{}:{}{}",
            query.scope.scope.len(),
            query.scope.scope,
            query.scope.key
        ),
        key: key.into(),
    }
}

fn invalid(detail: impl Into<String>) -> ResourceDescriptionError {
    ResourceDescriptionError::Invalid(detail.into())
}
