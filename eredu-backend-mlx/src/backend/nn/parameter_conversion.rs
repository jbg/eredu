//! Reusable promotions owned by permanently resident parameter materializations.
//!
//! Registry entries and backing lookups are weak. Each owner retains native
//! storage only alongside its own execution group's admitted claim. Lock order
//! is native registry, native entry, owner binding, then portable controller.
//! Native conversion/evaluation runs outside all these locks; publication is
//! serialized with invalidation by the native entry lock.

use eredu_core::{
    residency::{
        ParameterConversionRetentionEligibility, ParameterConversionRetentionGroup,
        ParameterConversionRetentionPolicy, ParameterConversionRetentionPolicyReport,
    },
    resources::ResourceIdentity,
};
use eredu_runtime::residency::conversion_retention::{
    conversion_retention_payload_bytes, ConversionRetentionAdmission, ConversionRetentionBudget,
    ConversionRetentionClaim, ConversionRetentionError, ConversionRetentionParameter,
    ConversionRetentionRegistry,
};
use safemlx::{error::Exception, Array, Dtype, Stream};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock, Weak,
    },
};

struct Conversion {
    _source: Array,
    parameter: ConversionRetentionParameter,
    state: Mutex<ConversionState>,
}
struct ConversionState {
    active: bool,
    value: Weak<NativeConversion>,
    owners: Vec<Weak<OwnerBinding>>,
}
struct NativeConversion {
    array: Array,
}
struct OwnerBinding {
    budget: ConversionRetentionBudget,
    epoch: AtomicU64,
    retained: Mutex<Option<(ConversionRetentionClaim, Arc<NativeConversion>)>>,
}
struct RegisteredConversion {
    entry: Arc<Conversion>,
    binding: Arc<OwnerBinding>,
}

type Registry = BTreeMap<usize, Weak<Conversion>>;
fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(Mutex::default)
}
fn controller() -> &'static ConversionRetentionRegistry {
    static CONTROLLER: OnceLock<ConversionRetentionRegistry> = OnceLock::new();
    CONTROLLER.get_or_init(ConversionRetentionRegistry::default)
}
fn next_identity() -> u64 {
    static NEXT_IDENTITY: AtomicU64 = AtomicU64::new(1);
    NEXT_IDENTITY.fetch_add(1, Ordering::Relaxed)
}
fn identity(scope: &str) -> ResourceIdentity {
    ResourceIdentity {
        scope: scope.into(),
        key: next_identity().to_string(),
    }
}

/// Creates a fixed execution-scoped budget without allocating native storage.
pub(crate) fn execution_budget(
    requested: Option<ParameterConversionRetentionPolicy>,
    eligibility: ParameterConversionRetentionEligibility,
) -> Result<ConversionRetentionBudget, ConversionRetentionError> {
    controller().budget(
        ParameterConversionRetentionGroup(identity("mlx.parameter_conversion_group")),
        ParameterConversionRetentionPolicyReport::resolve(requested, eligibility),
    )
}

/// Cache ownership follows one permanently resident device materialization.
pub(crate) struct ResidentParameterConversions {
    owner: u64,
    entries: BTreeMap<usize, RegisteredConversion>,
}
impl Default for ResidentParameterConversions {
    fn default() -> Self {
        Self {
            owner: next_identity(),
            entries: BTreeMap::new(),
        }
    }
}
pub(crate) struct ResidentConversionObservation {
    pub(crate) allocation: ResourceIdentity,
    pub(crate) payload_bytes: u64,
}
impl ResidentParameterConversions {
    pub(crate) fn owner_identity(&self) -> ResourceIdentity {
        ResourceIdentity {
            scope: "mlx.parameter_materialization".into(),
            key: self.owner.to_string(),
        }
    }
    pub(crate) fn resident_conversions(&self) -> BTreeMap<usize, ResidentConversionObservation> {
        self.entries
            .iter()
            .filter_map(|(&identity, registered)| {
                let retained = registered
                    .binding
                    .retained
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                retained
                    .as_ref()
                    .filter(|(claim, _)| claim.is_active())
                    .map(|(claim, _)| {
                        (
                            identity,
                            ResidentConversionObservation {
                                allocation: claim.allocation().clone(),
                                payload_bytes: claim.payload_bytes(),
                            },
                        )
                    })
            })
            .collect()
    }
    pub(crate) fn register<'a>(
        &mut self,
        arrays: impl IntoIterator<Item = &'a Array>,
        budget: &ConversionRetentionBudget,
    ) -> Result<(), ConversionRetentionError> {
        // Native descriptor access and shallow clones precede registry locking.
        let candidates = arrays
            .into_iter()
            .filter(|array| matches!(array.dtype(), Dtype::Bfloat16 | Dtype::Float16))
            .map(|array| {
                let bytes = conversion_retention_payload_bytes(array.size() as u64, 4)?;
                Ok((array.graph_identity(), array.clone(), bytes))
            })
            .collect::<Result<Vec<_>, ConversionRetentionError>>()?;
        let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
        registry.retain(|_, entry| entry.strong_count() > 0);
        for (key, source, bytes) in candidates {
            if self.entries.contains_key(&key) {
                continue;
            }
            let entry = match registry.get(&key).and_then(Weak::upgrade) {
                Some(entry) => entry,
                None => Arc::new(Conversion {
                    _source: source,
                    parameter: controller()
                        .parameter(identity("mlx.immutable_parameter"), bytes)?,
                    state: Mutex::new(ConversionState {
                        active: true,
                        value: Weak::new(),
                        owners: Vec::new(),
                    }),
                }),
            };
            let binding = Arc::new(OwnerBinding {
                budget: budget.clone(),
                epoch: AtomicU64::new(0),
                retained: Mutex::new(None),
            });
            {
                let mut state = entry
                    .state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                // Joining an existing backing still requires this group's claim.
                if let Some(value) = state.value.upgrade() {
                    if let Ok(ConversionRetentionAdmission::Retained(claim)) =
                        entry.parameter.reserve(budget)
                    {
                        *binding
                            .retained
                            .lock()
                            .unwrap_or_else(|error| error.into_inner()) = Some((claim, value));
                    }
                }
                state.owners.retain(|owner| owner.strong_count() > 0);
                state.owners.push(Arc::downgrade(&binding));
            }
            registry.insert(key, Arc::downgrade(&entry));
            self.entries
                .insert(key, RegisteredConversion { entry, binding });
        }
        Ok(())
    }
}
impl Drop for ResidentParameterConversions {
    fn drop(&mut self) {
        let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
        for (identity, registered) in &self.entries {
            if Arc::strong_count(&registered.entry) == 1
                && registry
                    .get(identity)
                    .is_some_and(|entry| entry.ptr_eq(&Arc::downgrade(&registered.entry)))
            {
                registry.remove(identity);
            }
        }
    }
}

/// Releases this execution group's owners while keeping registrations eligible.
/// The session authority must be settled before entering this operation.
pub(crate) fn trim_budget(
    budget: &ConversionRetentionBudget,
) -> Result<eredu_core::residency::ParameterConversionRetentionTrimReport, ConversionRetentionError>
{
    let registry = registry().lock().unwrap_or_else(|error| error.into_inner());
    let entries: Vec<_> = registry.values().filter_map(Weak::upgrade).collect();
    let states: Vec<_> = entries
        .iter()
        .map(|entry| {
            entry
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
        })
        .collect();
    // Holding every native entry prevents publication or new reservations until
    // detachment and the atomic ledger update have both completed.
    if budget
        .report()
        .usage
        .value()
        .expect("exact usage")
        .reserved_payload_bytes
        != 0
    {
        return Err(ConversionRetentionError::Busy);
    }
    let mut claims = Vec::new();
    let mut arrays = Vec::new();
    for state in &states {
        for owner in state.owners.iter().filter_map(Weak::upgrade) {
            if owner.budget.group() == budget.group() {
                owner.epoch.fetch_add(1, Ordering::Relaxed);
                if let Some((claim, value)) = owner
                    .retained
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                {
                    claims.push(claim);
                    arrays.push(value);
                }
            }
        }
    }
    let result = budget.release_claims(&mut claims);
    drop(states);
    drop(registry);
    // Native destruction happens outside admission and registry locks.
    drop(arrays);
    result
}

/// Publication revokes old handles; restoring a source requires fresh registration.
pub(crate) fn invalidate(weight: &Array) {
    let entry = registry()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .remove(&weight.graph_identity())
        .and_then(|entry| entry.upgrade());
    if let Some(entry) = entry {
        let mut state = entry
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.active = false;
        entry.parameter.invalidate();
        state.value = Weak::new();
        for owner in state.owners.iter().filter_map(Weak::upgrade) {
            owner
                .retained
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take();
        }
    }
}

/// Match ordinary F32 promotion; denied admissions use the unchanged matmul path.
pub(crate) fn promoted_weight(
    input: &Array,
    weight: &Array,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    promote_with(input, weight, || {
        let converted = weight.as_dtype(Dtype::Float32, stream)?;
        safemlx::transforms::eval([&converted])?;
        Ok(converted)
    })
}

fn promote_with(
    input: &Array,
    weight: &Array,
    convert: impl FnOnce() -> Result<Array, Exception>,
) -> Result<Option<Array>, Exception> {
    if input.dtype() != Dtype::Float32
        || !matches!(weight.dtype(), Dtype::Bfloat16 | Dtype::Float16)
    {
        return Ok(None);
    }
    let entry = registry()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&weight.graph_identity())
        .and_then(Weak::upgrade);
    let Some(entry) = entry else {
        return Ok(None);
    };
    let (publisher, reservation, owners) = {
        let state = entry
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !state.active {
            return Ok(None);
        }
        if let Some(value) = state.value.upgrade() {
            // Trim preserves eligibility; joining existing shared storage still
            // requires fresh admission for each previously released owner.
            for owner in state.owners.iter().filter_map(Weak::upgrade) {
                if let Ok(ConversionRetentionAdmission::Retained(claim)) =
                    entry.parameter.reserve(&owner.budget)
                {
                    *owner
                        .retained
                        .lock()
                        .unwrap_or_else(|error| error.into_inner()) =
                        Some((claim, Arc::clone(&value)));
                }
            }
            return Ok(Some(value.array.clone()));
        }
        let Some(admission) = state
            .owners
            .iter()
            .filter_map(Weak::upgrade)
            .find_map(|owner| match entry.parameter.reserve(&owner.budget) {
                Ok(ConversionRetentionAdmission::Reserved(reservation)) => {
                    Some((owner, reservation))
                }
                _ => None,
            })
        else {
            return Ok(None);
        };
        let owners = state
            .owners
            .iter()
            .filter_map(Weak::upgrade)
            .map(|owner| {
                let epoch = owner.epoch.load(Ordering::Relaxed);
                (owner, epoch)
            })
            .collect::<Vec<_>>();
        (admission.0, admission.1, owners)
    };
    // The move-only reservation restores capacity on errors or invalidation.
    let value = Arc::new(NativeConversion { array: convert()? });
    let allocation = match controller().allocation(
        identity("mlx.cached_parameter_conversion"),
        value.array.nbytes() as u64,
    ) {
        Ok(allocation) => allocation,
        Err(_) => return Ok(None),
    };
    let mut state = entry
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !state.active {
        return Ok(None);
    }
    let Ok(claim) = reservation.publish(allocation) else {
        return Ok(None);
    };
    *publisher
        .retained
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = Some((claim, Arc::clone(&value)));
    state.value = Arc::downgrade(&value);
    for (owner, epoch) in owners {
        // Work started by another group before trim must not restore this
        // owner's released claim when its delayed evaluation publishes.
        if owner.epoch.load(Ordering::Relaxed) != epoch {
            continue;
        }
        if let Ok(ConversionRetentionAdmission::Retained(claim)) =
            entry.parameter.reserve(&owner.budget)
        {
            *owner
                .retained
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some((claim, Arc::clone(&value)));
        }
    }
    Ok(Some(value.array.clone()))
}
