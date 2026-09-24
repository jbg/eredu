//! Model-scoped retention of optional parameter conversions.
//!
//! These contracts describe policy and observations, not an allocator or a
//! reservation controller. Observing them must not create conversions, settle
//! execution, trim owners, or consume execution/observation budgets.

use serde::{Deserialize, Serialize};

use crate::{resources::ResourceIdentity, Observed};

/// Optional conversion payload retained across invocations of a loaded execution.
///
/// This policy is independent of allocator caching. Its bound covers retained
/// payload plus outstanding retention reservations, not backing capacity,
/// temporary casts, graph storage, process RSS, or total execution memory.
/// Admission is first-admitted, without automatic eviction. A denied conversion
/// uses the ordinary temporary execution path without changing precision.
///
/// Select the policy before loading with
/// [`crate::ExecutionPlan::with_parameter_conversion_retention`]. An absent
/// override uses [`Self::MANAGED_DEFAULT`] for eligible executions. Ordinary
/// reset preserves admitted conversions; explicit settled trimming releases
/// optional claims while preserving weights, request state and future eligibility.
/// Requested and effective policy can differ: host-layerwise, disk-streamed,
/// explicit device ceilings and unsupported mechanisms disable retention.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParameterConversionRetentionPolicy {
    /// Do not retain optional conversions; temporary casts remain permitted.
    Disabled,
    /// Bound the group's retained and reserved payload in bytes.
    Bounded {
        /// A zero-byte allowance normalizes to disabled retention.
        max_bytes: u64,
    },
    /// Retain without a payload ceiling; requires an explicit request.
    Unlimited,
}

impl ParameterConversionRetentionPolicy {
    /// Finite managed policy for eligible executions: 256 MiB.
    ///
    /// Eligible loaded executions use this when the caller omits an override.
    /// Missing historical policy observations must remain unknown.
    pub const MANAGED_DEFAULT: Self = Self::Bounded {
        max_bytes: 256 * 1024 * 1024,
    };

    /// Canonicalizes a zero-byte bound without changing any other policy.
    pub const fn normalized(self) -> Self {
        match self {
            Self::Bounded { max_bytes: 0 } => Self::Disabled,
            policy => policy,
        }
    }
}

/// Origin of a loaded execution's requested conversion-retention policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterConversionRetentionPolicySource {
    /// The caller omitted the setting and selection supplied the finite default.
    ManagedDefault,
    /// The caller explicitly selected the policy, including zero or unlimited.
    Explicit,
}

/// Whether selected execution resources permit optional retained conversions.
///
/// Eligibility is independent of the request: an eligible execution can still
/// request disabled retention. Ineligible executions have an effective disabled
/// policy, including when the request was explicitly unlimited.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParameterConversionRetentionEligibility {
    /// Permanent parameter owners can participate in the shared budget group.
    Eligible,
    /// Host-layerwise execution must preserve its bounded residency behavior.
    HostLayerwise,
    /// Disk-streamed execution must preserve its bounded residency behavior.
    DiskStreamed,
    /// An explicit device-residency ceiling excludes optional retained copies.
    DeviceResidencyLimit,
    /// The selected mechanism cannot provide conversion retention.
    Unsupported {
        /// Concrete mechanism or resource constraint, not a missing observation.
        reason: String,
    },
}

/// Requested and normalized effective policy for one selected budget group.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParameterConversionRetentionPolicyReport {
    /// Original explicit request, or the finite policy chosen by the runtime.
    /// A zero-byte bounded request is preserved here for diagnostics.
    pub requested: ParameterConversionRetentionPolicy,
    /// Normalized request if eligible; otherwise disabled.
    pub effective: ParameterConversionRetentionPolicy,
    /// Whether the request was explicit or supplied by managed selection.
    pub source: ParameterConversionRetentionPolicySource,
    /// Eligibility or the reason the selected execution excludes retention.
    pub eligibility: ParameterConversionRetentionEligibility,
}

impl ParameterConversionRetentionPolicyReport {
    /// Resolves an optional request against already-selected eligibility facts.
    ///
    /// This is pure normalization; it neither selects a native mechanism nor
    /// changes a live budget. Do not use it to fill absent historical reports.
    pub fn resolve(
        requested: Option<ParameterConversionRetentionPolicy>,
        eligibility: ParameterConversionRetentionEligibility,
    ) -> Self {
        let (requested, source) = match requested {
            Some(policy) => (policy, ParameterConversionRetentionPolicySource::Explicit),
            None => (
                ParameterConversionRetentionPolicy::MANAGED_DEFAULT,
                ParameterConversionRetentionPolicySource::ManagedDefault,
            ),
        };
        let effective = match eligibility {
            ParameterConversionRetentionEligibility::Eligible => requested.normalized(),
            _ => ParameterConversionRetentionPolicy::Disabled,
        };
        Self {
            requested,
            effective,
            source,
            eligibility,
        }
    }
}

/// A retention-budget namespace, separate from parameter owners and allocations.
///
/// The entire scoped identity is authoritative. One selected loaded execution
/// shares a group across permanent units and embedded prediction owners. Native
/// partitions must share that execution's admission authority; implementations
/// without coordinated admission report unsupported eligibility and disable
/// retention rather than grant an allowance per rank. Separately loaded external
/// drafters use distinct groups. Equal checkpoint names never imply equal groups.
///
/// This wrapper reuses resource identity conventions without replacing
/// `ResidentParameterConversion` allocation identities or binding owners.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ParameterConversionRetentionGroup(pub ResourceIdentity);

/// A coherent observation of one group's retention claims and reservations.
///
/// Aliases within a group charge one claim per conversion allocation. Independent
/// groups sharing an allocation each admit their own claim; summing group payload
/// is therefore not a physical residency total. Physical reports continue to
/// deduplicate `ResidentParameterConversion::allocation` identities.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParameterConversionRetentionUsage {
    /// Distinct published retention claims in this group, not binding references.
    pub retained_claims: u64,
    /// Published payload charged to this group, a subset of resident parameters.
    pub retained_payload_bytes: u64,
    /// Payload reserved for conversions not yet published as retained claims.
    /// This is admission usage, not necessarily materialized resident storage.
    pub reserved_payload_bytes: u64,
    /// Native backing capacity for this group's retained allocations, where
    /// observable. This overlaps payload and may include padding or spare capacity.
    /// It is neither an extra charge nor covered by the payload policy ceiling.
    #[serde(default = "unreported_backing_capacity")]
    pub retained_backing_capacity_bytes: Observed<u64>,
}

/// Policy and current usage for one selected execution's shared budget group.
///
/// Read the effective policy here instead of assuming a requested limit applies.
/// Usage can be unavailable or unsupported rather than zero. Published payload
/// is already included in resident parameter observations; do not add it again.
/// Reservations consume allowance without proving a materialized allocation.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParameterConversionRetentionReport {
    /// Budget scope, independent of allocation and parameter identities.
    pub group: ParameterConversionRetentionGroup,
    /// Retained selection facts. Missing facts never imply today's managed default.
    #[serde(default = "unreported_policy")]
    pub policy: Observed<ParameterConversionRetentionPolicyReport>,
    /// Current group accounting. Unsupported/unavailable is distinct from an
    /// observed usage with zero claims, retained payload and reservations.
    #[serde(default = "unreported_usage")]
    pub usage: Observed<ParameterConversionRetentionUsage>,
}

/// Result of releasing optional conversion-retention claims at a settled boundary.
///
/// This describes an explicit trim operation; it does not authorize settlement.
/// Pending controlled transactions require the operation's typed rejection unless
/// its submission/completion contract explicitly permits settlement. Ordinary
/// reset retains admitted conversions. Trim preserves source parameters, request
/// state and budgets, and eligibility for future admission; it is distinct from
/// parameter invalidation and allocator-cache flushing.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParameterConversionRetentionTrimReport {
    /// Group whose claims were released; other groups retain their own claims.
    pub group: ParameterConversionRetentionGroup,
    /// Number of this group's published claims released, deduplicating aliases.
    pub released_claims: u64,
    /// Payload removed from this group's retention accounting.
    /// Dropping claims does not establish physical reclamation.
    pub released_payload_bytes: u64,
    /// Coherent remaining group claims, payload and reservations after trimming.
    pub remaining: ParameterConversionRetentionUsage,
    /// Backing bytes observed to cease being live because of this operation.
    /// Shared owners, snapshots or native graphs can keep the backing alive.
    /// Even an observed nonzero value does not mean bytes returned to the OS:
    /// native allocator caches may retain them. Unknown reclamation stays unknown.
    #[serde(default = "unreported_reclamation")]
    pub reclaimed_backing_bytes: Observed<u64>,
}

fn unreported_policy() -> Observed<ParameterConversionRetentionPolicyReport> {
    Observed::unavailable("parameter conversion retention policy was not reported")
}

fn unreported_usage() -> Observed<ParameterConversionRetentionUsage> {
    Observed::unavailable("parameter conversion retention usage was not reported")
}

fn unreported_backing_capacity() -> Observed<u64> {
    Observed::unavailable("retained parameter conversion backing capacity was not reported")
}

fn unreported_reclamation() -> Observed<u64> {
    Observed::unavailable("parameter conversion backing reclamation was not reported")
}

/// Role-labelled retention scopes for a composed execution. These are admission
/// claims, not additive physical-residency counters. Embedded prediction owners
/// are included in `target`; only a separately loaded drafter has a second entry.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionConversionRetentionReport {
    /// Target execution, including any embedded prediction owners.
    pub target: Observed<Vec<ParameterConversionRetentionReport>>,
    /// Independent external drafter. `None` means no such participant; an
    /// unsupported observation means the participant exists but cannot report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_drafter: Option<Observed<Vec<ParameterConversionRetentionReport>>>,
}

/// Read-only observation of a separately owned drafting execution. Implementations
/// must not allocate native resources, settle, evaluate, or modify execution state.
pub trait ParameterConversionRetentionObserver {
    /// Reports load-selected policy and live admission usage without mutation.
    fn parameter_conversion_retention(
        &self,
    ) -> Result<Observed<Vec<ParameterConversionRetentionReport>>, crate::BackendFailure>;
}

/// Compatibility default for documents recorded before retention was observable.
pub fn unreported_parameter_conversion_retention(
) -> Observed<Vec<ParameterConversionRetentionReport>> {
    Observed::unavailable("parameter conversion retention was not reported")
}

#[cfg(test)]
mod tests;

/// Failure to release optional conversions at a healthy settled boundary.
#[derive(Debug, thiserror::Error)]
pub enum ParameterConversionTrimError {
    /// This implementation has no conversion trimming mechanism.
    #[error("parameter conversion trimming is unsupported")]
    Unsupported,
    /// Native work or a speculative transaction remains pending.
    #[error("parameter conversion trimming requires a canonical settled boundary")]
    NotQuiescent,
    /// Native failure, retaining its original source.
    #[error(transparent)]
    Backend(#[from] crate::BackendFailure),
}

/// Release results for the independent participants of one composed execution.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionConversionRetentionTrimReport {
    /// Target, including shared embedded prediction owners.
    pub target: Vec<ParameterConversionRetentionTrimReport>,
    /// Separately loaded drafter, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_drafter: Option<Vec<ParameterConversionRetentionTrimReport>>,
}

/// Settled-boundary release for a separately loaded drafting execution.
/// Implementations preserve request state and retain unresolved native resources
/// on failure; they must never infer completion from a polling error.
pub trait ParameterConversionRetentionTrimmer {
    /// Settles ordinary submitted work when supported, then releases caller claims.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError>;
}
