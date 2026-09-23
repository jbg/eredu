//! Descriptive execution resources, independent of forecasting or allocation.
//!
//! Producers describe storage selected by ordinary execution contracts. The
//! descriptions carry neither native handles nor permission to allocate. They
//! do not specify execution order, overlap, resident credit, or a fit verdict.
//! In particular, adding every entry is not a valid peak-memory calculation.
//! Producing a description must not submit, poll, synchronize, or advance
//! execution, allocate execution resources, or mutate admission budgets. Ordinary
//! host allocation to construct the description itself is allowed.
//!
//! Sizes are evaluated for explicit producer-defined dimensions and a requested
//! horizon. This permits nonlinear geometry, rounded allocation capacity, and
//! temporary interior peaks without a second language for model equations.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{ObservationKind, Observed};

#[cfg(test)]
mod tests;

/// Version of the neutral resource-description document.
pub const RESOURCE_DESCRIPTION_SCHEMA_VERSION: u32 = 1;

/// Logical identity in an explicitly declared namespace, never a native pointer.
///
/// Allocation, owner, and physical-pool identities occupy separate namespaces.
/// Within each namespace the entire `(scope, key)` pair is authoritative. Scopes
/// must remain stable when descriptions are composed. Independently loaded
/// instances of the same artifact must not automatically use the same scope.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceIdentity {
    /// Instance or registry that owns the identity namespace.
    pub scope: String,
    /// Identity within that namespace.
    pub key: String,
}

impl ResourceIdentity {
    fn validate(&self) -> Result<(), ResourceDescriptionError> {
        nonempty(&self.scope, "identity scope")?;
        nonempty(&self.key, "identity key")
    }
}

/// Logical use of storage; a single allocation can serve several uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceRole {
    /// Parameters, including a separately allocated cached conversion.
    Parameters,
    /// Mutable execution state, including cache storage.
    MutableState,
    /// Tensor storage kept across invocations or evaluation boundaries.
    RetainedTensor,
    /// Temporary storage used while an invocation executes.
    Workspace,
}

/// An owner or logical use referring to an allocation's backing storage.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceUse {
    /// Logical module, state slot, or invocation owner.
    pub owner: ResourceIdentity,
    /// Why this owner uses the storage.
    pub role: ResourceRole,
}

/// Producer-defined, named logical extents for a sizing query.
///
/// Names and units are part of the producer's ordinary execution contract. For
/// example, attention can use `query_rows` and `key_positions`; a continuation
/// horizon can use `additional_positions`. No dimension is implicitly a token
/// count, endpoint, multiplier, or monotone axis. Producers must reject a query
/// they cannot interpret or preserve the affected size as unknown.
/// These dimensions are evaluated sizing provenance derived from ordinary typed
/// invocation/state facts, not a new family-specific application configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceContext {
    /// Explicit nonnegative dimension values; omission is not a value of zero.
    pub dimensions: BTreeMap<String, u64>,
}

impl ResourceContext {
    fn validate(&self) -> Result<(), ResourceDescriptionError> {
        for name in self.dimensions.keys() {
            nonempty(name, "dimension name")?;
        }
        Ok(())
    }
}

/// Size bounds with independent provenance and an explicitly unknown upper end.
///
/// Unlike runtime forecast intervals, these describe one resource, not a phase
/// total or additional-to-resident cost. A zero lower end with no upper end is
/// unknown, not free storage. An exact interval is a byte count, not a claim
/// about whether or when the resource is live.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceByteBounds {
    /// Known minimum bytes.
    pub lower_bytes: u64,
    /// Finite maximum, or `None` when not known.
    pub upper_bytes: Option<u64>,
    /// How the bound was established.
    pub kind: ObservationKind,
    /// Source, assumption, or missing fact.
    pub detail: String,
}

impl ResourceByteBounds {
    /// Exactly known bytes, including an explicitly absent payload or capacity.
    pub fn exact(bytes: u64) -> Self {
        Self {
            lower_bytes: bytes,
            upper_bytes: Some(bytes),
            kind: ObservationKind::Exact,
            detail: "exact resource bytes".into(),
        }
    }

    /// A known contribution with an unknown remainder.
    pub fn unknown(lower_bytes: u64, reason: impl Into<String>) -> Self {
        Self {
            lower_bytes,
            upper_bytes: None,
            kind: ObservationKind::Estimated,
            detail: reason.into(),
        }
    }

    /// Validates a constructed or decoded interval without changing it.
    pub fn validate(&self) -> Result<(), ResourceDescriptionError> {
        nonempty(&self.detail, "byte-bound provenance")?;
        if self
            .upper_bytes
            .is_some_and(|upper| upper < self.lower_bytes)
        {
            return Err(invalid("byte upper bound is below its lower bound"));
        }
        if self.kind == ObservationKind::Exact && self.upper_bytes != Some(self.lower_bytes) {
            return Err(invalid("exact byte bounds must describe one finite value"));
        }
        Ok(())
    }
}

/// Logical payload and its backing allocation capacity, measured independently.
///
/// Capacity includes payload, padding, and reserved unused space; the two fields
/// must never be added. Capacity may be unknown even when payload is exact.
/// Each bound covers this allocation alone, excluding other allocations used
/// during a replacement, conversion, or operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceExtent {
    /// Bytes occupied by values in their actual stored representation, including
    /// encoding/quantization metadata, not their hypothetical uncompressed size.
    pub payload: ResourceByteBounds,
    /// Bytes reserved by the backing allocation, including unused capacity.
    pub capacity: ResourceByteBounds,
}

impl ResourceExtent {
    /// Validates bounds and rejects a capacity known to be smaller than payload.
    pub fn validate(&self) -> Result<(), ResourceDescriptionError> {
        self.payload.validate()?;
        self.capacity.validate()?;
        if self
            .capacity
            .upper_bytes
            .is_some_and(|upper| upper < self.payload.lower_bytes)
        {
            return Err(invalid("allocation capacity cannot cover the payload"));
        }
        Ok(())
    }
}

/// Evaluated size behavior for the containing description's context and horizon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResourceSize {
    /// Context-independent size for one live allocation, reusable across queries.
    Fixed {
        /// Payload and backing capacity whenever this allocation is live.
        extent: ResourceExtent,
    },
    /// Evaluated context-dependent sizes; no linear growth is implied.
    ContextDependent {
        /// Installed state at the query's starting boundary. A future allocation
        /// can explicitly have zero current payload and zero current capacity.
        current: ResourceExtent,
        /// Maximum of this resource over the entire requested horizon, including
        /// the starting boundary and intermediate allocation peaks. This is not
        /// an endpoint size or an aggregate peak across resources. Payload and
        /// capacity maxima need not occur at the same instant.
        horizon_peak: ResourceExtent,
    },
}

impl ResourceSize {
    /// Validates sizes and that the horizon bounds include the starting state.
    pub fn validate(&self) -> Result<(), ResourceDescriptionError> {
        match self {
            Self::Fixed { extent } => extent.validate(),
            Self::ContextDependent {
                current,
                horizon_peak,
            } => {
                current.validate()?;
                horizon_peak.validate()?;
                contains_start(&current.payload, &horizon_peak.payload)?;
                contains_start(&current.capacity, &horizon_peak.capacity)
            }
        }
    }
}

/// One independently accounted backing allocation, with all its logical uses.
///
/// Tied parameters and tensor views share an allocation identity. A replica,
/// copied snapshot, cached dtype conversion, or replacement allocation has a
/// distinct identity even when its logical owner or contents are the same.
/// Allocator address reuse does not establish logical sharing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Shared backing identity; listed once in a description.
    pub identity: ResourceIdentity,
    /// Logical owners/uses sharing this backing, without repeated size charges.
    pub uses: Vec<ResourceUse>,
    /// Disjoint physical capacity pool. Host/device aliases on unified memory
    /// use exactly the same identity; separate hosts or independent devices use
    /// distinct identities. Identities must be authoritative (`exact`) or a
    /// point-in-time observation (`observational`), never guessed. Missing
    /// placement stays unavailable/unsupported rather than becoming host memory.
    pub placement: Observed<ResourceIdentity>,
    /// Payload and allocation capacity for the declared query.
    pub size: ResourceSize,
}

impl ResourceAllocation {
    /// Validates identities, placement, logical uses and size facts.
    pub fn validate(&self) -> Result<(), ResourceDescriptionError> {
        self.identity.validate()?;
        if self.uses.is_empty() {
            return Err(invalid("allocation has no logical use"));
        }
        let mut uses = BTreeSet::new();
        for usage in &self.uses {
            usage.owner.validate()?;
            if !uses.insert(usage) {
                return Err(invalid("allocation repeats a logical use"));
            }
        }
        match &self.placement {
            Observed::Available {
                value,
                kind,
                source,
            } => {
                value.validate()?;
                nonempty(source, "placement provenance")?;
                if !matches!(
                    kind,
                    ObservationKind::Exact | ObservationKind::Observational
                ) {
                    return Err(invalid("physical pool identity must not be estimated"));
                }
            }
            Observed::Unavailable { reason } | Observed::Unsupported { reason } => {
                nonempty(reason, "missing placement reason")?;
            }
        }
        self.size.validate()
    }
}

/// Whether all storage relevant to this description's declared scope is listed.
/// Coverage says nothing about whether listed sizes or placement are known.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResourceCoverage {
    /// No claim of resource-set completeness, including an omitted wire field.
    #[default]
    Unspecified,
    /// Every relevant allocation is listed; an intentionally empty set is valid.
    Complete,
    /// Known allocations are listed but other storage remains undescribed.
    Partial {
        /// Missing components; at least one nonempty reason is required.
        reasons: Vec<String>,
    },
}

/// A neutral answer to a resource query, not a lifetime plan or memory forecast.
///
/// The producer declares which execution/module scope it describes. Consumers
/// must retain incomplete coverage, unknown sizes and unknown placement. They
/// cannot infer a finite total from the known entries or an empty list. Future
/// composition must separately establish liveness and sharing across documents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceDescription {
    /// Contract version; validated before using a decoded document.
    pub schema_version: u32,
    /// Execution, module or mechanism scope covered by this answer.
    pub scope: ResourceIdentity,
    /// Logical extents at the starting boundary or invocation.
    pub context: ResourceContext,
    /// Explicit requested work/extents covered by context-dependent maxima.
    pub horizon: ResourceContext,
    /// Explicit resource-set coverage; missing metadata never means complete.
    #[serde(default)]
    pub coverage: ResourceCoverage,
    /// Unique backing allocations, with their logical aliases collected together.
    pub allocations: Vec<ResourceAllocation>,
}

impl ResourceDescription {
    /// Validates a constructed or decoded description without executing work.
    /// This checks internal consistency, not the producer's equations, claims of
    /// coverage, identity registry, or interpretation of context dimensions.
    pub fn validate(&self) -> Result<(), ResourceDescriptionError> {
        if self.schema_version != RESOURCE_DESCRIPTION_SCHEMA_VERSION {
            return Err(ResourceDescriptionError::UnsupportedVersion(
                self.schema_version,
            ));
        }
        self.scope.validate()?;
        self.context.validate()?;
        self.horizon.validate()?;
        if let ResourceCoverage::Partial { reasons } = &self.coverage {
            if reasons.is_empty() {
                return Err(invalid("partial coverage needs a missing-resource reason"));
            }
            for reason in reasons {
                nonempty(reason, "missing-resource reason")?;
            }
        }
        let mut allocations = BTreeSet::new();
        for allocation in &self.allocations {
            allocation.validate()?;
            if !allocations.insert(&allocation.identity) {
                return Err(invalid(
                    "backing allocation identity is listed more than once",
                ));
            }
        }
        Ok(())
    }
}

/// Invalid or unsupported neutral resource description.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResourceDescriptionError {
    /// A different document contract requires explicit interpretation.
    #[error("unsupported resource-description version {0}")]
    UnsupportedVersion(u32),
    /// Identities, bounds, placement or completeness are internally inconsistent.
    #[error("invalid resource description: {0}")]
    Invalid(String),
}

fn nonempty(value: &str, field: &str) -> Result<(), ResourceDescriptionError> {
    if value.trim().is_empty() {
        return Err(invalid(format!("{field} must be nonempty")));
    }
    Ok(())
}

fn contains_start(
    current: &ResourceByteBounds,
    peak: &ResourceByteBounds,
) -> Result<(), ResourceDescriptionError> {
    if peak.lower_bytes < current.lower_bytes {
        return Err(invalid("horizon lower bound excludes the starting state"));
    }
    // Unknown starting capacity must not become finite merely by requesting a
    // horizon. Producers with a stronger proof should also refine `current`.
    if let Some(upper) = peak.upper_bytes {
        if current.upper_bytes.is_none_or(|start| upper < start) {
            return Err(invalid("horizon upper bound excludes the starting state"));
        }
    }
    Ok(())
}

fn invalid(reason: impl Into<String>) -> ResourceDescriptionError {
    ResourceDescriptionError::Invalid(reason.into())
}
