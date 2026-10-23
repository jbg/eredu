//! Incremental admission of independently leased execution-unit resources.

use std::collections::BTreeSet;

use eredu_core::residency::{
    MemoryTier, OffloadError, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
};

use crate::{
    physical_binding_bytes, ExecutionUnitAddress, ExecutionUnitLayout, LayerWeightResidency,
    OffloadUnit, ResidencyDeclarationError, WeightBinding,
};

/// Invalid execution-unit resource count or aggregate capacity.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionUnitPreparationError {
    /// Prepared resources do not match the selected invocation count.
    #[error("prepared {actual} execution units for a layout of {expected}")]
    UnitCount {
        /// Selected logical invocation count.
        expected: usize,
        /// Number of supplied units.
        actual: usize,
    },
    /// A byte total cannot be represented.
    #[error("execution unit {address:?} overflows {context}")]
    ArithmeticOverflow {
        /// Architecture-global invocation being admitted.
        address: ExecutionUnitAddress,
        /// Capacity whose addition failed.
        context: &'static str,
    },
}

/// Exact resource declarations and capacity facts for selected execution units.
/// Static and supplementary resources retain their own residency boundaries.
pub struct PreparedExecutionUnitResources {
    /// Independently materialized units in selected invocation order.
    pub definitions: Vec<OffloadUnit>,
    /// Initial residency policy for each declared unit.
    pub specs: Vec<OffloadUnitSpec>,
    /// Stable residency identities in selected invocation order.
    pub unit_ids: Vec<OffloadUnitId>,
    /// Physical device payloads, excluding aliases, in selected invocation order.
    pub unit_bytes: Vec<u64>,
    /// Exact physical source inputs consumed by these units.
    pub source_keys: BTreeSet<String>,
    /// Sum of physical unit parameter payloads.
    pub parameter_bytes: u64,
    /// Sum of backend-declared host-transfer capacities.
    pub total_host_bytes: u64,
    /// Largest independently acquired host-transfer capacity.
    pub maximum_host_bytes: u64,
}

/// Consuming admission that keeps backend host-capacity queries in unit order.
/// No checkpoint payload, backend tensor, stream or completion is required.
pub struct ExecutionUnitResourcePreparation<'a> {
    namespace: &'a str,
    layout: &'a ExecutionUnitLayout,
    residency: LayerWeightResidency,
    resources: PreparedExecutionUnitResources,
}

impl<'a> ExecutionUnitResourcePreparation<'a> {
    /// Starts admission for one exact selected invocation layout.
    pub fn new(
        namespace: &'a str,
        layout: &'a ExecutionUnitLayout,
        residency: LayerWeightResidency,
    ) -> Self {
        Self {
            namespace,
            layout,
            residency,
            resources: PreparedExecutionUnitResources {
                definitions: Vec::new(),
                specs: Vec::new(),
                unit_ids: Vec::with_capacity(layout.len()),
                unit_bytes: Vec::with_capacity(layout.len()),
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
        let ordinal = self.resources.unit_ids.len();
        let address =
            self.layout
                .address(ordinal)
                .ok_or(ExecutionUnitPreparationError::UnitCount {
                    expected: self.layout.len(),
                    actual: ordinal + 1,
                })?;
        let bytes = physical_binding_bytes(&bindings)?;
        self.resources.parameter_bytes = self.resources.parameter_bytes.checked_add(bytes).ok_or(
            ExecutionUnitPreparationError::ArithmeticOverflow {
                address,
                context: "execution parameter bytes",
            },
        )?;
        let host_bytes = host_capacity(&bindings)?;
        self.resources.total_host_bytes =
            self.resources
                .total_host_bytes
                .checked_add(host_bytes)
                .ok_or(ExecutionUnitPreparationError::ArithmeticOverflow {
                    address,
                    context: "host unit capacity",
                })?;
        self.resources.maximum_host_bytes = self.resources.maximum_host_bytes.max(host_bytes);
        self.resources.source_keys.extend(
            bindings
                .iter()
                .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned)),
        );
        let group = self
            .layout
            .group_id(address.group())
            .expect("validated execution group");
        let id = OffloadUnitId::new(format!(
            "{}.{}.{:05}",
            self.namespace,
            group.as_str(),
            address.index()
        ))?;
        self.resources
            .definitions
            .push(OffloadUnit::new(id.clone(), bindings)?);
        let resident = self.residency.is_fully_resident();
        let dense = self.residency.dense().is_some();
        self.resources.specs.push(OffloadUnitSpec::new(
            id.clone(),
            bytes,
            if resident {
                ResidencyPolicy::Pinned
            } else if dense {
                ResidencyPolicy::Cacheable
            } else {
                ResidencyPolicy::Windowed
            },
            if resident {
                MemoryTier::Device
            } else if dense {
                MemoryTier::Disk
            } else {
                MemoryTier::Host
            },
        )?);
        self.resources.unit_ids.push(id);
        self.resources.unit_bytes.push(bytes);
        Ok(self)
    }

    /// Completes admission only when every selected invocation has resources.
    pub fn finish(self) -> Result<PreparedExecutionUnitResources, ExecutionUnitPreparationError> {
        if self.resources.unit_ids.len() != self.layout.len() {
            return Err(ExecutionUnitPreparationError::UnitCount {
                expected: self.layout.len(),
                actual: self.resources.unit_ids.len(),
            });
        }
        Ok(self.resources)
    }
}
