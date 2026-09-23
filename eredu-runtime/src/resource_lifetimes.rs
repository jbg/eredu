//! Family-independent resource lifetime and physical-pool peak composition.
//!
//! Plans describe completion order, not native execution. An acquire before a
//! preceding completion permits overlap; completing before acquiring is sequential.
//! Allocation identities establish sharing, never tensor names or equal sizes.

use std::collections::{BTreeMap, BTreeSet};

use eredu_core::{resources::*, ObservationKind, Observed};
use eredu_nn::mechanism_memory::{MechanismBacking, MechanismPlacement, StorageRetention};

use crate::mechanism_resources::{invocation_storage_identity, MechanismResourceDescription};

/// The event that ends one reference to an allocation. Namespaces are disjoint.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResourceLifetime {
    /// A submitted native operation completes.
    NativeCompletion(ResourceIdentity),
    /// Evaluation of a lazy graph completes, including its dependencies.
    Evaluation(ResourceIdentity),
    /// The owner of a returned value, state or parameter releases it.
    Owner(ResourceIdentity),
    /// No safe release boundary is known. Never silently released.
    Unknown,
}

/// Described allocations and the references that keep them live.
#[derive(Debug, Clone)]
pub struct ResourceLifetimeDescription {
    /// Ordinary prepared or mechanism resources, preserving coverage and bounds.
    pub resources: ResourceDescription,
    /// All retention references for each backing. Missing/empty entries are unknown.
    pub lifetimes: BTreeMap<ResourceIdentity, Vec<ResourceLifetime>>,
    /// Whether the current extents coexist at acquisition. Set false for a
    /// mechanism's internal envelope whose intermediates need not coexist.
    /// Fixed extents with known retention remain guaranteed live until release;
    /// context-dependent current extents are guaranteed only at acquisition.
    pub live_at_acquire: bool,
}

/// Schedule-owned bindings for a mechanism's declared retention classes.
#[derive(Debug, Clone)]
pub struct MechanismLifetimeBindings {
    /// Native completion boundary of this invocation.
    pub completion: ResourceIdentity,
    /// Completed lazy evaluation boundary, if established by the schedule.
    pub evaluation: Option<ResourceIdentity>,
    /// Returned/state/parameter owners indexed by authoritative backing identity.
    /// Missing owners remain unknown; invocation identity does not invent them.
    pub owners: BTreeMap<ResourceIdentity, ResourceIdentity>,
}

/// Bind mechanism retention without assuming its internal temporaries coexist.
///
/// The public wrapper is revalidated, including every storage-name binding and
/// payload/capacity agreement. Missing storage identities remain coverage gaps.
pub fn describe_mechanism_lifetimes(
    description: &MechanismResourceDescription,
    bindings: &MechanismLifetimeBindings,
) -> Result<ResourceLifetimeDescription, ResourceDescriptionError> {
    description.resources.validate()?;
    description
        .contract
        .validate()
        .map_err(|e| invalid(e.to_string()))?;
    valid_id(&bindings.completion)?;
    if let Some(id) = &bindings.evaluation {
        valid_id(id)?;
    }
    for (allocation, owner) in &bindings.owners {
        valid_id(allocation)?;
        valid_id(owner)?;
    }
    let mut resources = description.resources.clone();
    let mut missing = coverage_reasons(&resources.coverage, "mechanism resources");
    missing.extend(description.contract.missing.clone());
    let mut lifetimes: BTreeMap<_, Vec<_>> = BTreeMap::new();
    let mut execution_pool = None;
    let mut host_pool = None;
    let mut owner_backings = BTreeMap::new();
    for name in description.storage_bindings.keys() {
        if !description.contract.storage.iter().any(|s| &s.name == name) {
            return Err(invalid(format!(
                "binding for unknown mechanism storage {name}"
            )));
        }
    }
    for storage in &description.contract.storage {
        let Some(identity) = description.storage_bindings.get(&storage.name) else {
            missing.push(format!(
                "{}: storage has no allocation binding",
                storage.name
            ));
            continue;
        };
        match &storage.backing {
            MechanismBacking::Invocation
                if *identity != invocation_storage_identity(&resources.scope, &storage.name) =>
            {
                return Err(invalid(
                    "invocation-local storage cannot be rebound as an alias",
                ));
            }
            MechanismBacking::Unknown => {
                return Err(invalid(
                    "unknown storage backing cannot have a resolved binding",
                ))
            }
            _ => {}
        }
        let allocation = resources
            .allocations
            .iter()
            .find(|a| &a.identity == identity)
            .ok_or_else(|| invalid(format!("{}: binding names absent allocation", storage.name)))?;
        if storage.placement == MechanismPlacement::Unknown
            && matches!(allocation.placement, Observed::Available { .. })
        {
            return Err(invalid(
                "unknown mechanism placement cannot become a known pool",
            ));
        }
        let prior_pool = match storage.placement {
            MechanismPlacement::Execution => Some(&mut execution_pool),
            MechanismPlacement::Host => Some(&mut host_pool),
            MechanismPlacement::Unknown => None,
        };
        if let Some(prior_pool) = prior_pool {
            if let Some(prior) = prior_pool.as_ref() {
                if !same_pool(prior, &allocation.placement) {
                    return Err(invalid(
                        "mechanism placement records disagree on their physical pool",
                    ));
                }
            } else {
                *prior_pool = Some(allocation.placement.clone());
            }
        }
        if let MechanismBacking::Owner(owner) = &storage.backing {
            if owner_backings
                .insert(owner, identity)
                .is_some_and(|prior| prior != identity)
            {
                return Err(invalid(
                    "mechanism owner key has conflicting allocation bindings",
                ));
            }
        }
        let ResourceSize::Fixed { extent } = &allocation.size else {
            return Err(invalid("mechanism storage needs fixed invocation extents"));
        };
        if extent.payload.lower_bytes != storage.payload.lower
            || extent.payload.upper_bytes != storage.payload.upper
            || extent.capacity.lower_bytes != storage.capacity.lower
            || extent.capacity.upper_bytes != storage.capacity.upper
        {
            return Err(invalid(format!(
                "{}: storage and allocation bounds disagree",
                storage.name
            )));
        }
        let lifetime = match storage.retention {
            StorageRetention::NativeCompletion => {
                ResourceLifetime::NativeCompletion(bindings.completion.clone())
            }
            StorageRetention::Evaluation => bindings
                .evaluation
                .clone()
                .map(ResourceLifetime::Evaluation)
                .unwrap_or(ResourceLifetime::Unknown),
            StorageRetention::Returned | StorageRetention::ParameterOwner => bindings
                .owners
                .get(identity)
                .cloned()
                .map(ResourceLifetime::Owner)
                .unwrap_or(ResourceLifetime::Unknown),
            StorageRetention::Unknown => ResourceLifetime::Unknown,
        };
        lifetimes
            .entry(identity.clone())
            .or_default()
            .push(lifetime);
    }
    for allocation in &resources.allocations {
        if !lifetimes.contains_key(&allocation.identity) {
            return Err(invalid("mechanism allocation has no storage declaration"));
        }
    }
    for lifetime in lifetimes.values_mut() {
        lifetime.sort();
        lifetime.dedup();
    }
    missing.sort();
    missing.dedup();
    resources.coverage = if missing.is_empty() {
        ResourceCoverage::Complete
    } else {
        ResourceCoverage::Partial { reasons: missing }
    };
    Ok(ResourceLifetimeDescription {
        resources,
        lifetimes,
        live_at_acquire: false,
    })
}

/// One ordered schedule event. Completion is explicit, including lazy evaluation.
#[derive(Debug, Clone)]
pub enum ResourceLifetimeEvent {
    /// Introduce resources and retention references; earlier live resources overlap.
    Acquire(ResourceLifetimeDescription),
    /// End native-completion references with this token only.
    Complete(ResourceIdentity),
    /// End lazy-evaluation references with this token only.
    Evaluate(ResourceIdentity),
    /// End owner references with this token only.
    Release(ResourceIdentity),
}

/// A finite, explicitly ordered execution envelope; no native values are required.
#[derive(Debug, Clone, Default)]
pub struct ResourceLifetimePlan {
    /// Whether the plan covers all execution resources, including empty work.
    pub coverage: ResourceCoverage,
    /// Acquisition and release order. Concurrent operations have overlapping holds.
    pub events: Vec<ResourceLifetimeEvent>,
}

/// Peak payload and capacity in one disjoint physical pool. Never add these fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePoolPeak {
    /// Authoritative physical pool; host/device aliases use one identity.
    pub pool: ResourceIdentity,
    /// Peak bounds. Payload and capacity peaks can occur at different events.
    pub peak: ResourceExtent,
}

/// Composed bounds, without budget comparison, resident credit or fit policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePeakReport {
    /// Independently composed physical pools; peaks need not occur simultaneously.
    pub pools: Vec<ResourcePoolPeak>,
    /// Resource-set/placement/retention gaps invalidate every pool upper bound.
    /// Unknown byte bounds affect only that bound in its known physical pool.
    pub missing: Vec<String>,
    /// Allocations whose physical pool could not be established.
    pub unplaced: Vec<ResourceIdentity>,
}

#[derive(Clone)]
struct AllocationFacts {
    allocation: ResourceAllocation,
    context: ResourceContext,
    horizon: ResourceContext,
}
#[derive(Default)]
struct Active {
    // A true reference establishes simultaneous fixed-size liveness.
    holds: BTreeMap<ResourceLifetime, bool>,
}
#[derive(Default)]
struct Peak {
    payload: Interval,
    capacity: Interval,
}
#[derive(Default)]
struct Interval {
    lower: u64,
    upper: Option<u64>,
}

/// Compose a checked schedule. Unknown facts never silently become finite totals.
///
/// Upper bounds sum potentially overlapping per-allocation horizon maxima. Lower
/// bounds use coexisting current minima and individual horizon minima, never the
/// sum of independent horizon maxima. Unknown-retention holds do not establish
/// continued minimum liveness. Shared allocation facts must agree numerically;
/// context-dependent aliases must additionally describe the same query.
///
/// Completion/evaluation/owner tokens cannot be retired twice or reused. Owners
/// still live at the end are valid: the plan need not destroy installed state.
pub fn compose_resource_peaks(
    plan: &ResourceLifetimePlan,
) -> Result<ResourcePeakReport, ResourceDescriptionError> {
    validate_coverage(&plan.coverage)?;
    let mut global_missing = coverage_reasons(&plan.coverage, "execution plan");
    let mut missing = Vec::new();
    let mut unplaced = BTreeSet::new();
    let mut facts: BTreeMap<ResourceIdentity, AllocationFacts> = BTreeMap::new();
    let mut active: BTreeMap<ResourceIdentity, Active> = BTreeMap::new();
    let mut retired = BTreeSet::new();
    let mut peaks: BTreeMap<ResourceIdentity, Peak> = BTreeMap::new();
    // Include every alias description: the first exact observation cannot erase
    // a later estimate of the same numerical extent.
    let mut nonexact: BTreeMap<ResourceIdentity, (bool, bool)> = BTreeMap::new();
    for event in &plan.events {
        let mut current = BTreeSet::new();
        match event {
            ResourceLifetimeEvent::Acquire(description) => {
                description.resources.validate()?;
                global_missing.extend(coverage_reasons(
                    &description.resources.coverage,
                    &format!("resource scope {:?}", description.resources.scope),
                ));
                for identity in description.lifetimes.keys() {
                    if !description
                        .resources
                        .allocations
                        .iter()
                        .any(|a| &a.identity == identity)
                    {
                        return Err(invalid("lifetime names an absent allocation"));
                    }
                }
                for allocation in &description.resources.allocations {
                    if let Some(prior) = facts.get(&allocation.identity) {
                        if !compatible(prior, allocation, &description.resources) {
                            return Err(invalid(format!(
                                "conflicting facts for shared allocation {:?}",
                                allocation.identity
                            )));
                        }
                    } else {
                        facts.insert(
                            allocation.identity.clone(),
                            AllocationFacts {
                                allocation: allocation.clone(),
                                context: description.resources.context.clone(),
                                horizon: description.resources.horizon.clone(),
                            },
                        );
                    }
                    if let Observed::Available { value: pool, .. } = &allocation.placement {
                        let quality = nonexact.entry(pool.clone()).or_default();
                        let extents = match &allocation.size {
                            ResourceSize::Fixed { extent } => [extent, extent],
                            ResourceSize::ContextDependent {
                                current,
                                horizon_peak,
                            } => [current, horizon_peak],
                        };
                        for extent in extents {
                            quality.0 |= extent.payload.kind != ObservationKind::Exact;
                            quality.1 |= extent.capacity.kind != ObservationKind::Exact
                                || (extent.payload.lower_bytes > extent.capacity.lower_bytes
                                    && extent.payload.kind != ObservationKind::Exact);
                        }
                    }
                    if !matches!(allocation.placement, Observed::Available { .. }) {
                        unplaced.insert(allocation.identity.clone());
                        global_missing.push(format!(
                            "{:?}: physical pool is unknown",
                            allocation.identity
                        ));
                    }
                    let fallback = [ResourceLifetime::Unknown];
                    let holds = description
                        .lifetimes
                        .get(&allocation.identity)
                        .filter(|h| !h.is_empty())
                        .map(Vec::as_slice)
                        .unwrap_or(&fallback);
                    for hold in holds {
                        match hold {
                            ResourceLifetime::NativeCompletion(id)
                            | ResourceLifetime::Evaluation(id)
                            | ResourceLifetime::Owner(id) => valid_id(id)?,
                            ResourceLifetime::Unknown => global_missing.push(format!(
                                "{:?}: retention boundary is unknown",
                                allocation.identity
                            )),
                        }
                        if retired.contains(hold) {
                            return Err(invalid("acquisition reuses a retired lifetime token"));
                        }
                        let guaranteed =
                            description.live_at_acquire && *hold != ResourceLifetime::Unknown;
                        let value = active
                            .entry(allocation.identity.clone())
                            .or_default()
                            .holds
                            .entry(hold.clone())
                            .or_default();
                        *value |= guaranteed;
                    }
                    if description.live_at_acquire {
                        current.insert(allocation.identity.clone());
                    }
                }
            }
            ResourceLifetimeEvent::Complete(id)
            | ResourceLifetimeEvent::Evaluate(id)
            | ResourceLifetimeEvent::Release(id) => {
                valid_id(id)?;
                let hold = match event {
                    ResourceLifetimeEvent::Complete(_) => {
                        ResourceLifetime::NativeCompletion(id.clone())
                    }
                    ResourceLifetimeEvent::Evaluate(_) => ResourceLifetime::Evaluation(id.clone()),
                    _ => ResourceLifetime::Owner(id.clone()),
                };
                if retired.contains(&hold) {
                    return Err(invalid("lifetime token is retired twice"));
                }
                let mut found = false;
                for allocation in active.values_mut() {
                    found |= allocation.holds.remove(&hold).is_some();
                }
                if !found {
                    return Err(invalid("release boundary has no acquired reference"));
                }
                active.retain(|_, allocation| !allocation.holds.is_empty());
                retired.insert(hold);
            }
        }
        observe(&facts, &active, &current, &mut peaks)?;
    }
    let incomplete = !global_missing.is_empty();
    missing.extend(global_missing);
    let pools = peaks
        .into_iter()
        .map(|(pool, peak)| {
            let quality = nonexact.get(&pool).copied().unwrap_or_default();
            let make =
                |mut interval: Interval, field: &str, nonexact: bool, missing: &mut Vec<String>| {
                    if interval.upper.is_none() {
                        missing.push(format!("{pool:?}: {field} upper bound is unknown"));
                    }
                    if incomplete {
                        interval.upper = None;
                    }
                    ResourceByteBounds {
                        lower_bytes: interval.lower,
                        upper_bytes: interval.upper,
                        kind: if !nonexact && interval.upper == Some(interval.lower) {
                            ObservationKind::Exact
                        } else {
                            ObservationKind::Estimated
                        },
                        detail: format!("composed {field} peak over declared lifetime schedule"),
                    }
                };
            ResourcePoolPeak {
                pool: pool.clone(),
                peak: ResourceExtent {
                    payload: make(peak.payload, "payload", quality.0, &mut missing),
                    capacity: make(peak.capacity, "capacity", quality.1, &mut missing),
                },
            }
        })
        .collect();
    missing.sort();
    missing.dedup();
    Ok(ResourcePeakReport {
        pools,
        missing,
        unplaced: unplaced.into_iter().collect(),
    })
}

fn observe(
    facts: &BTreeMap<ResourceIdentity, AllocationFacts>,
    active: &BTreeMap<ResourceIdentity, Active>,
    current: &BTreeSet<ResourceIdentity>,
    peaks: &mut BTreeMap<ResourceIdentity, Peak>,
) -> Result<(), ResourceDescriptionError> {
    // (sum current minima, maximum individual horizon minimum, sum horizon upper)
    type Accumulator = (u64, u64, Option<u64>);
    let mut sums: BTreeMap<ResourceIdentity, (Accumulator, Accumulator)> = BTreeMap::new();
    for (id, held) in active {
        let allocation = &facts[id].allocation;
        let Observed::Available { value: pool, .. } = &allocation.placement else {
            continue;
        };
        let (start, horizon, fixed) = match &allocation.size {
            ResourceSize::Fixed { extent } => (extent, extent, true),
            ResourceSize::ContextDependent {
                current,
                horizon_peak,
            } => (current, horizon_peak, false),
        };
        let guaranteed = current.contains(id) || (fixed && held.holds.values().any(|g| *g));
        let sum = sums
            .entry(pool.clone())
            .or_insert(((0, 0, Some(0)), (0, 0, Some(0))));
        for (sum, start_lower, horizon_lower, horizon_upper) in [
            (
                &mut sum.0,
                start.payload.lower_bytes,
                horizon.payload.lower_bytes,
                horizon.payload.upper_bytes,
            ),
            (
                &mut sum.1,
                start.capacity.lower_bytes.max(start.payload.lower_bytes),
                horizon
                    .capacity
                    .lower_bytes
                    .max(horizon.payload.lower_bytes),
                horizon.capacity.upper_bytes,
            ),
        ] {
            if guaranteed {
                sum.0 = checked_add(sum.0, start_lower)?;
            }
            sum.1 = sum.1.max(horizon_lower);
            sum.2 = match (sum.2, horizon_upper) {
                (Some(a), Some(b)) => Some(checked_add(a, b)?),
                _ => None,
            };
        }
    }
    for (pool, sum) in sums {
        let peak = peaks.entry(pool).or_insert_with(|| Peak {
            payload: Interval {
                lower: 0,
                upper: Some(0),
            },
            capacity: Interval {
                lower: 0,
                upper: Some(0),
            },
        });
        for (peak, sum) in [(&mut peak.payload, sum.0), (&mut peak.capacity, sum.1)] {
            peak.lower = peak.lower.max(sum.0.max(sum.1));
            peak.upper = match (peak.upper, sum.2) {
                (Some(a), Some(b)) => Some(a.max(b)),
                _ => None,
            };
        }
    }
    Ok(())
}

fn compatible(
    prior: &AllocationFacts,
    allocation: &ResourceAllocation,
    description: &ResourceDescription,
) -> bool {
    let same_bounds = |a: &ResourceExtent, b: &ResourceExtent| {
        a.payload.lower_bytes == b.payload.lower_bytes
            && a.payload.upper_bytes == b.payload.upper_bytes
            && a.capacity.lower_bytes == b.capacity.lower_bytes
            && a.capacity.upper_bytes == b.capacity.upper_bytes
    };
    let same_size = match (&prior.allocation.size, &allocation.size) {
        (ResourceSize::Fixed { extent: a }, ResourceSize::Fixed { extent: b }) => same_bounds(a, b),
        (
            ResourceSize::ContextDependent {
                current: a,
                horizon_peak: ap,
            },
            ResourceSize::ContextDependent {
                current: b,
                horizon_peak: bp,
            },
        ) => {
            same_bounds(a, b)
                && same_bounds(ap, bp)
                && prior.context == description.context
                && prior.horizon == description.horizon
        }
        _ => false,
    };
    same_size && same_pool(&prior.allocation.placement, &allocation.placement)
}
fn same_pool(left: &Observed<ResourceIdentity>, right: &Observed<ResourceIdentity>) -> bool {
    match (left, right) {
        (Observed::Available { value: a, .. }, Observed::Available { value: b, .. }) => a == b,
        (a, b) => a == b,
    }
}

fn coverage_reasons(coverage: &ResourceCoverage, scope: &str) -> Vec<String> {
    match coverage {
        ResourceCoverage::Complete => vec![],
        ResourceCoverage::Unspecified => vec![format!("{scope}: resource coverage is unspecified")],
        ResourceCoverage::Partial { reasons } => {
            reasons.iter().map(|r| format!("{scope}: {r}")).collect()
        }
    }
}
fn validate_coverage(coverage: &ResourceCoverage) -> Result<(), ResourceDescriptionError> {
    if let ResourceCoverage::Partial { reasons } = coverage {
        if reasons.is_empty() || reasons.iter().any(|r| r.trim().is_empty()) {
            return Err(invalid("partial coverage requires nonempty reasons"));
        }
    }
    Ok(())
}
fn valid_id(id: &ResourceIdentity) -> Result<(), ResourceDescriptionError> {
    if id.scope.trim().is_empty() || id.key.trim().is_empty() {
        return Err(invalid("lifetime identity must be nonempty"));
    }
    Ok(())
}
fn checked_add(a: u64, b: u64) -> Result<u64, ResourceDescriptionError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("resource peak byte sum overflows u64"))
}
fn invalid(message: impl Into<String>) -> ResourceDescriptionError {
    ResourceDescriptionError::Invalid(message.into())
}

#[cfg(test)]
mod tests;
