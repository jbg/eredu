//! Incremental admission of independently leased execution-unit resources.

use std::collections::{BTreeMap, BTreeSet};

use eredu_checkpoint::store::{CheckpointSource, SharedCheckpointSource, StoreError};
use eredu_core::residency::{OffloadError, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy};

use crate::{ExecutionUnitAddress, LayerWeightResidency};
use eredu_core::residency::parameter_binding::{
    physical_binding_bytes, OffloadUnit, ResidencyDeclarationError, WeightBinding,
};

/// Invalid execution-unit resource count or aggregate capacity.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionUnitPreparationError {
    /// Selected logical consumers are not a subset of the retained declaration.
    #[error(transparent)]
    Selection(#[from] crate::ExecutionUnitAddressMapError),
    /// A retained parameter alias refers to an ordinary owner removed by selection.
    #[error("selected alias {alias:?} requires discarded parameter owner {owner:?}")]
    DiscardedAliasOwner {
        /// Retained alias parameter.
        alias: String,
        /// Canonical parameter whose ordinary resource was discarded.
        owner: String,
    },
    /// Prepared resources do not match the selected physical owner count.
    #[error("prepared {actual} execution units for a layout of {expected}")]
    UnitCount {
        /// Selected physical owner count.
        expected: usize,
        /// Number of supplied units.
        actual: usize,
    },
    /// A byte total cannot be represented.
    #[error("execution unit {address:?} overflows {context}")]
    ArithmeticOverflow {
        /// Canonical physical owner being admitted.
        address: ExecutionUnitAddress,
        /// Capacity whose addition failed.
        context: &'static str,
    },
}

/// Exact resource declarations and capacity facts for selected execution units.
/// Static and supplementary resources retain their own residency boundaries.
pub struct PreparedExecutionUnitResources {
    invocations: crate::ExecutionUnitAddressMap,
    residency: LayerWeightResidency,
    /// Independently materialized units in physical catalog order.
    pub definitions: Vec<OffloadUnit>,
    /// Initial residency policy for each declared unit.
    pub specs: Vec<OffloadUnitSpec>,
    /// Stable residency identities in physical catalog order.
    pub unit_ids: Vec<OffloadUnitId>,
    /// Physical device payloads, excluding aliases, in physical catalog order.
    pub unit_bytes: Vec<u64>,
    /// Backend host-transfer allocation capacity for each physical owner.
    pub host_bytes: Vec<u64>,
    /// Exact physical source inputs consumed by these units.
    pub source_keys: BTreeSet<String>,
    /// Sum of physical unit parameter payloads.
    pub parameter_bytes: u64,
    /// Sum of backend-declared host-transfer capacities.
    pub total_host_bytes: u64,
    /// Largest independently acquired host-transfer capacity.
    pub maximum_host_bytes: u64,
}

impl PreparedExecutionUnitResources {
    /// Projects already prepared physical owners through selected logical consumers.
    /// Recipes, placement and backend capacity facts move unchanged; no source is
    /// reopened and no transformation or host-capacity query runs again.
    pub fn select_invocations(
        mut self,
        selected: impl IntoIterator<Item = ExecutionUnitAddress>,
    ) -> Result<Self, ExecutionUnitPreparationError> {
        let invocations = self.invocations.select_invocations(selected)?;
        let original = self.invocations.storage_layout();
        for actual in [
            self.definitions.len(),
            self.specs.len(),
            self.unit_ids.len(),
            self.unit_bytes.len(),
            self.host_bytes.len(),
        ] {
            if actual != original.len() {
                return Err(ExecutionUnitPreparationError::UnitCount {
                    expected: original.len(),
                    actual,
                });
            }
        }
        let mut retained = vec![false; original.len()];
        for ordinal in 0..invocations.storage_layout().len() {
            let address = invocations
                .storage_layout()
                .address(ordinal)
                .expect("selected resource");
            let ordinal = original
                .ordinal(address.group(), address.index())
                .expect("retained resource");
            retained[ordinal] = true;
        }
        let mut targets = BTreeMap::new();
        for (unit, &keep) in self.definitions.iter().zip(&retained) {
            for binding in unit.bindings() {
                let target = binding.logical_target().unwrap_or(binding.name());
                targets
                    .entry(target)
                    .and_modify(|selected| *selected |= keep)
                    .or_insert(keep);
            }
        }
        for binding in self
            .definitions
            .iter()
            .zip(&retained)
            .filter(|(_, keep)| **keep)
            .flat_map(|(unit, _)| unit.bindings())
        {
            if let Some(owner) = binding.alias_of() {
                if targets.get(owner) == Some(&false) {
                    return Err(ExecutionUnitPreparationError::DiscardedAliasOwner {
                        alias: binding.name().to_owned(),
                        owner: owner.to_owned(),
                    });
                }
            }
        }
        fn retain<T>(values: &mut Vec<T>, retained: &[bool]) {
            let mut retained = retained.iter();
            values.retain(|_| *retained.next().expect("validated resource count"));
        }
        retain(&mut self.definitions, &retained);
        retain(&mut self.specs, &retained);
        retain(&mut self.unit_ids, &retained);
        retain(&mut self.unit_bytes, &retained);
        retain(&mut self.host_bytes, &retained);
        self.source_keys = self
            .definitions
            .iter()
            .flat_map(|unit| unit.bindings())
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned))
            .collect();
        let sum = |bytes: &[u64], context| {
            bytes
                .iter()
                .enumerate()
                .try_fold(0u64, |sum, (ordinal, bytes)| {
                    sum.checked_add(*bytes).ok_or(
                        ExecutionUnitPreparationError::ArithmeticOverflow {
                            address: invocations
                                .storage_layout()
                                .address(ordinal)
                                .expect("selected resource"),
                            context,
                        },
                    )
                })
        };
        self.parameter_bytes = sum(&self.unit_bytes, "selected parameter bytes")?;
        self.total_host_bytes = sum(&self.host_bytes, "selected host capacities")?;
        self.maximum_host_bytes = self.host_bytes.iter().copied().max().unwrap_or(0);
        self.invocations = invocations;
        Ok(self)
    }
}

/// Consuming admission that keeps backend host-capacity queries in unit order.
/// No checkpoint payload, backend tensor, stream or completion is required.
pub struct ExecutionUnitResourcePreparation<'a> {
    namespace: &'a str,
    resources: PreparedExecutionUnitResources,
}

impl<'a> ExecutionUnitResourcePreparation<'a> {
    /// Starts admission for exact invocation references and their physical owners.
    pub fn new(
        namespace: &'a str,
        invocations: &crate::ExecutionUnitAddressMap,
        residency: LayerWeightResidency,
    ) -> Self {
        let layout = invocations.storage_layout();
        Self {
            namespace,
            resources: PreparedExecutionUnitResources {
                invocations: invocations.clone(),
                residency,
                definitions: Vec::new(),
                specs: Vec::new(),
                unit_ids: Vec::with_capacity(layout.len()),
                unit_bytes: Vec::with_capacity(layout.len()),
                host_bytes: Vec::with_capacity(layout.len()),
                source_keys: BTreeSet::new(),
                parameter_bytes: 0,
                total_host_bytes: 0,
                maximum_host_bytes: 0,
            },
        }
    }

    /// Admits the next unit, preserving independent leasing and source authority.
    /// A failure returns no reusable partial preparation.
    pub fn push<E>(
        mut self,
        bindings: Vec<WeightBinding>,
        host_capacity: impl FnOnce(&[WeightBinding]) -> Result<u64, E>,
    ) -> Result<Self, E>
    where
        E: From<ExecutionUnitPreparationError>
            + From<ResidencyDeclarationError>
            + From<OffloadError>,
    {
        let resources = &mut self.resources;
        let ordinal = resources.unit_ids.len();
        let address = resources
            .invocations
            .storage_layout()
            .address(ordinal)
            .ok_or(ExecutionUnitPreparationError::UnitCount {
                expected: resources.invocations.storage_layout().len(),
                actual: ordinal + 1,
            })?;
        let bytes = physical_binding_bytes(&bindings)?;
        resources.parameter_bytes = resources.parameter_bytes.checked_add(bytes).ok_or(
            ExecutionUnitPreparationError::ArithmeticOverflow {
                address,
                context: "execution parameter bytes",
            },
        )?;
        let host_bytes = host_capacity(&bindings)?;
        resources.total_host_bytes = resources.total_host_bytes.checked_add(host_bytes).ok_or(
            ExecutionUnitPreparationError::ArithmeticOverflow {
                address,
                context: "host unit capacity",
            },
        )?;
        resources.maximum_host_bytes = resources.maximum_host_bytes.max(host_bytes);
        resources.source_keys.extend(
            bindings
                .iter()
                .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned)),
        );
        let group = resources
            .invocations
            .storage_layout()
            .group_id(address.group())
            .expect("validated execution group");
        let id = OffloadUnitId::new(format!(
            "{}.{}.{:05}",
            self.namespace,
            group.as_str(),
            address.index()
        ))?;
        resources
            .definitions
            .push(OffloadUnit::new(id.clone(), bindings)?);
        resources
            .specs
            .push(resources.residency.parameter_unit_spec(
                id.clone(),
                bytes,
                ResidencyPolicy::Windowed,
            )?);
        resources.unit_ids.push(id);
        resources.unit_bytes.push(bytes);
        resources.host_bytes.push(host_bytes);
        Ok(self)
    }

    /// Completes admission only when every selected physical owner has bindings.
    pub fn finish(self) -> Result<PreparedExecutionUnitResources, ExecutionUnitPreparationError> {
        let expected = self.resources.invocations.storage_layout().len();
        if self.resources.unit_ids.len() != expected {
            return Err(ExecutionUnitPreparationError::UnitCount {
                expected,
                actual: self.resources.unit_ids.len(),
            });
        }
        Ok(self.resources)
    }
}

/// Parameter capacity required by ordinary and auxiliary execution phases.
/// Pinned statics, conversions, activations and mutable state are excluded.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ExecutionUnitResourceRequirements {
    /// Ordinary execution's largest group-local device window.
    pub device_window_bytes: u64,
    /// Host capacity retained across the selected phases.
    pub required_host_bytes: u64,
    /// Device capacity required in addition to pinned static parameters.
    pub required_device_bytes: u64,
}

/// Combines exact unit payloads with backend host-capacity facts and auxiliary lifetimes.
/// Sequential phases share bounded windows; resident resources coexist.
pub fn execution_unit_resource_requirements(
    invocations: &crate::ExecutionUnitAddressMap,
    unit_bytes: &[u64],
    host_bytes: &[u64],
    residency: LayerWeightResidency,
    auxiliary: crate::AuxiliaryWeightRequirements,
) -> Result<ExecutionUnitResourceRequirements, crate::BoundedResidencySizingError> {
    use crate::BoundedResidencySizingError::ArithmeticOverflow;
    let device_window_bytes =
        crate::execution_unit_parameter_window_bytes(invocations, unit_bytes, residency)?;
    let dense = residency.dense();
    if host_bytes.len() != invocations.storage_layout().len() {
        return Err(crate::BoundedResidencySizingError::InvalidUnitCount {
            expected: invocations.storage_layout().len(),
            actual: host_bytes.len(),
        });
    }
    let host = match dense {
        Some(options) if options.host_budget_bytes() > 0 => {
            crate::automatic_support::execution_unit_window_bytes(
                invocations,
                host_bytes,
                options.host_lookahead(),
            )?
        }
        Some(_) => 0,
        None if residency.is_fully_resident() => 0,
        None => host_bytes.iter().try_fold(0u64, |total, bytes| {
            total
                .checked_add(*bytes)
                .ok_or(ArithmeticOverflow("host execution resources"))
        })?,
    };
    let auxiliary_host = auxiliary
        .host_bytes(residency)
        .ok_or(ArithmeticOverflow("auxiliary host capacity"))?;
    let required_host_bytes = if dense.is_some() {
        host.max(auxiliary_host)
    } else {
        host.checked_add(auxiliary_host)
            .ok_or(ArithmeticOverflow("combined host capacity"))?
    };
    let auxiliary_device = auxiliary
        .device_bytes(residency)
        .ok_or(ArithmeticOverflow("auxiliary device capacity"))?;
    let required_device_bytes = if residency.is_fully_resident() {
        device_window_bytes
            .checked_add(auxiliary_device)
            .ok_or(ArithmeticOverflow("combined device capacity"))?
    } else {
        device_window_bytes.max(auxiliary_device)
    };
    Ok(ExecutionUnitResourceRequirements {
        device_window_bytes,
        required_host_bytes,
        required_device_bytes,
    })
}

/// Failure finalizing the exact parameter resource plan.
#[derive(Debug, thiserror::Error)]
pub enum ParameterResourcePreparationError {
    /// A native capacity estimate failed.
    #[error(transparent)]
    Backend(#[from] eredu_core::BackendFailure),
    /// The selected residency policy is invalid.
    #[error(transparent)]
    Policy(#[from] crate::WeightResidencyPolicyError),
    /// Resource capacity cannot be represented.
    #[error(transparent)]
    Sizing(#[from] crate::BoundedResidencySizingError),
    /// Resource identities or initial tier policies are invalid.
    #[error(transparent)]
    Plan(#[from] OffloadError),
    /// An ordinary or pinned parameter binding is invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
    /// The retained execution units could not be admitted.
    #[error(transparent)]
    Units(#[from] ExecutionUnitPreparationError),
    /// A restricted source does not contain the retained recipe inputs.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// Strict loading found unrelated checkpoint tensors.
    #[error("strict layerwise loading found unexpected checkpoint parameters: {unused:?}")]
    UnexpectedCheckpointParameters {
        /// Unexpected keys in stable source order.
        unused: Vec<String>,
    },
    /// Host allocations exceed the selected budget.
    #[error(
        "host budget {budget} bytes cannot contain {required} bytes of host-transfer allocations"
    )]
    HostBudget {
        /// Required host allocation capacity.
        required: u64,
        /// Selected host budget.
        budget: u64,
    },
    /// Pinned and concurrently live device resources exceed the selected budget.
    #[error("device budget {budget} bytes cannot contain {static_bytes} static bytes plus the depth-{depth} unit window ({window_bytes} bytes, {required} total)")]
    DeviceBudget {
        /// Pinned device payload.
        static_bytes: u64,
        /// Required ordinary or auxiliary device phase capacity.
        window_bytes: u64,
        /// Selected ordinary unit window depth.
        depth: usize,
        /// Total required device capacity.
        required: u64,
        /// Selected device budget.
        budget: u64,
    },
}

/// Additional physical owner with an independently retained source and lease boundary.
pub struct SupplementaryResidencyUnit {
    /// Canonical owner and immutable bindings.
    pub definition: OffloadUnit,
    /// Prepared checkpoint source for this owner.
    pub source: SharedCheckpointSource,
    /// Whether the owner overlaps every sequential auxiliary invocation.
    pub shared: bool,
}

/// Admits pinned, ordinary and supplementary owners against one selected unit layout.
///
/// The backend supplies only capacity facts for each proposed allocation.
pub fn prepare_parameter_resources(
    source: &dyn CheckpointSource,
    invocations: &crate::ExecutionUnitAddressMap,
    residency: LayerWeightResidency,
    static_bindings: Vec<WeightBinding>,
    unit_bindings: Vec<Vec<WeightBinding>>,
    supplementary: Vec<SupplementaryResidencyUnit>,
    ignored: impl Fn(&str) -> bool,
    host_capacity: impl Fn(&[WeightBinding]) -> Result<u64, eredu_core::BackendFailure>,
) -> Result<PreparedParameterResources, ParameterResourcePreparationError> {
    let unit_count = invocations.storage_layout().len();
    if unit_count == 0 || unit_bindings.len() != unit_count {
        return Err(ExecutionUnitPreparationError::UnitCount {
            expected: unit_count,
            actual: unit_bindings.len(),
        }
        .into());
    }
    residency.offload()?;
    let pinned = crate::StaticUnitBindings::new("model.static", static_bindings)?
        .prepare::<ParameterResourcePreparationError>()?;
    let mut units = ExecutionUnitResourcePreparation::new("model", invocations, residency);
    for bindings in unit_bindings {
        units = units.push::<ParameterResourcePreparationError>(bindings, |bindings| {
            host_capacity(bindings).map_err(Into::into)
        })?;
    }
    let mut resources = ParameterResourcePreparation::new(pinned, units.finish()?);
    for unit in supplementary {
        resources = resources.push_auxiliary::<ParameterResourcePreparationError>(
            unit.definition,
            unit.source,
            unit.shared,
            |bindings| host_capacity(bindings).map_err(Into::into),
        )?;
    }
    let mut consumed = resources.source_keys().clone();
    consumed.extend(source.materialized_source_keys());
    let unused: Vec<_> = source
        .source_keys()
        .into_iter()
        .filter(|key| !consumed.contains(key) && !ignored(key))
        .collect();
    if !unused.is_empty() {
        return Err(ParameterResourcePreparationError::UnexpectedCheckpointParameters { unused });
    }
    resources.finish()
}

/// Consuming assembly of independently leased pinned, ordinary and auxiliary resources.
/// Exact auxiliary sources stay attached to their physical owners. No native resource
/// is created, and sequential owners retain their individual lease boundaries.
pub struct ParameterResourcePreparation {
    pinned: crate::PreparedStaticUnit,
    units: PreparedExecutionUnitResources,
    auxiliary: crate::AuxiliaryWeightRequirements,
    sources:
        std::collections::BTreeMap<OffloadUnitId, eredu_checkpoint::store::SharedCheckpointSource>,
}

/// Complete parameter plan consumed by native materialization and cold reporting.
pub struct PreparedParameterResources {
    primary_source_keys: BTreeSet<String>,
    /// Retained residency mechanism selection used to admit every resource.
    pub residency: LayerWeightResidency,
    /// Exact invocation references and physical layout used for admission.
    pub invocations: crate::ExecutionUnitAddressMap,
    /// Validated initial resource policies and capacity limits.
    pub plan: eredu_core::residency::OffloadPlan,
    /// Independent physical definitions, including pinned and auxiliary owners.
    pub definitions: Vec<OffloadUnit>,
    /// Restricted sources retained for auxiliary physical owners.
    pub sources:
        std::collections::BTreeMap<OffloadUnitId, eredu_checkpoint::store::SharedCheckpointSource>,
    /// Ordinary physical resource identities in physical catalog order.
    pub unit_ids: Vec<OffloadUnitId>,
    /// Pinned resource identity, absent for an empty pinned module.
    pub static_id: Option<OffloadUnitId>,
    /// Exact pinned destinations to populate.
    pub static_parameters: BTreeSet<String>,
    /// Residency report derived from the same selected resources and limits.
    pub metadata: crate::LayerwiseModelMetadata,
}

impl PreparedParameterResources {
    /// Restricts the primary source to actual pinned and ordinary recipe inputs.
    /// Auxiliary owners retain independent source views even when keys coincide.
    pub fn restrict_primary_source(
        &self,
        source: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<eredu_checkpoint::store::SharedCheckpointSource, eredu_checkpoint::store::StoreError>
    {
        Ok(std::sync::Arc::new(
            eredu_checkpoint::store::RestrictedCheckpointSource::including(
                source,
                "prepared primary parameters",
                self.primary_source_keys.clone(),
            )?,
        ))
    }
}

impl ParameterResourcePreparation {
    /// Retains admitted pinned and ordinary resources in their materialization order.
    pub fn new(
        mut pinned: crate::PreparedStaticUnit,
        mut units: PreparedExecutionUnitResources,
    ) -> Self {
        units.source_keys.append(&mut pinned.source_keys);
        Self {
            pinned,
            units,
            auxiliary: Default::default(),
            sources: Default::default(),
        }
    }

    /// Exact physical inputs consumed from the ordinary source. Auxiliary owners
    /// retain separate source authority and do not claim keys in this source.
    pub fn source_keys(&self) -> &BTreeSet<String> {
        &self.units.source_keys
    }

    /// Retains one independently leased auxiliary owner and its exact source.
    pub fn push_auxiliary<E>(
        mut self,
        definition: OffloadUnit,
        source: eredu_checkpoint::store::SharedCheckpointSource,
        shared: bool,
        host_capacity: impl FnOnce(&[WeightBinding]) -> Result<u64, E>,
    ) -> Result<Self, E>
    where
        E: From<ResidencyDeclarationError>
            + From<OffloadError>
            + From<crate::BoundedResidencySizingError>
            + From<eredu_checkpoint::store::StoreError>,
    {
        let keys = definition
            .bindings()
            .iter()
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned))
            .collect();
        let source = std::sync::Arc::new(
            eredu_checkpoint::store::RestrictedCheckpointSource::including(
                source,
                definition.id().as_str(),
                keys,
            )?,
        );
        let (requirements, spec) =
            self.auxiliary
                .admit(&definition, self.units.residency, shared, host_capacity)?;
        self.auxiliary = requirements;
        self.sources.insert(definition.id().clone(), source);
        self.units.specs.push(spec);
        self.units.definitions.push(definition);
        Ok(self)
    }

    /// Validates coexistence budgets and transfers the exact plan into materialization.
    pub fn finish(
        mut self,
    ) -> Result<PreparedParameterResources, ParameterResourcePreparationError> {
        let layout = self.units.invocations.storage_layout();
        let residency = self.units.residency;
        let config = residency.offload()?;
        let requirements = execution_unit_resource_requirements(
            &self.units.invocations,
            &self.units.unit_bytes,
            &self.units.host_bytes,
            residency,
            self.auxiliary,
        )?;
        if let Some(budget) = config.host_budget_bytes() {
            if requirements.required_host_bytes > budget {
                return Err(ParameterResourcePreparationError::HostBudget {
                    required: requirements.required_host_bytes,
                    budget,
                });
            }
        }
        let required = self
            .pinned
            .bytes
            .checked_add(requirements.required_device_bytes)
            .ok_or(crate::BoundedResidencySizingError::ArithmeticOverflow(
                "static plus device-window byte total",
            ))?;
        let depth = residency.device_depth(layout.len());
        if let Some(budget) = config.device_budget_bytes() {
            if required > budget {
                return Err(ParameterResourcePreparationError::DeviceBudget {
                    static_bytes: self.pinned.bytes,
                    window_bytes: requirements.required_device_bytes,
                    depth,
                    required,
                    budget,
                });
            }
        }
        let static_id = self.pinned.resource.map(|(definition, spec)| {
            let id = definition.id().clone();
            self.units.definitions.insert(0, definition);
            self.units.specs.insert(0, spec);
            id
        });
        let plan = eredu_core::residency::OffloadPlan::new(config, self.units.specs)?;
        let metadata = crate::LayerwiseModelMetadata::new(
            "generic",
            None,
            layout.len(),
            self.pinned.bytes,
            residency.execution_residency(),
            self.units.parameter_bytes,
            requirements.device_window_bytes,
            self.units.maximum_host_bytes,
            depth,
        );
        Ok(PreparedParameterResources {
            primary_source_keys: self.units.source_keys,
            residency,
            invocations: self.units.invocations,
            plan,
            definitions: self.units.definitions,
            sources: self.sources,
            unit_ids: self.units.unit_ids,
            static_id,
            static_parameters: self.pinned.parameters,
            metadata,
        })
    }
}
