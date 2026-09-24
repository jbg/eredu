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
    #[cfg(test)]
    pub(crate) fn resident_bytes(&self) -> BTreeMap<usize, u64> {
        let observations = self.resident_conversions();
        self.entries
            .keys()
            .map(|identity| {
                (
                    *identity,
                    observations
                        .get(identity)
                        .map_or(0, |value| value.payload_bytes),
                )
            })
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType};

    fn fixture(dtype: Dtype, stream: &Stream) -> Array {
        Array::from_slice(&[0.1234f32, -2.375, 0.53125, 1.0625], &[2, 2])
            .as_dtype(dtype, stream)
            .unwrap()
    }

    fn usage(
        budget: &ConversionRetentionBudget,
    ) -> eredu_core::residency::ParameterConversionRetentionUsage {
        budget.report().usage.value().unwrap().clone()
    }

    #[test]
    fn trim_is_idempotent_preserves_shared_owners_and_future_admission() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let input = Array::from_slice(&[0.5f32, -0.25], &[1, 2]);
        let weight = fixture(Dtype::Bfloat16, &stream);
        let make_budget = || {
            execution_budget(
                Some(ParameterConversionRetentionPolicy::Bounded { max_bytes: 16 }),
                ParameterConversionRetentionEligibility::Eligible,
            )
            .unwrap()
        };
        let target = make_budget();
        let other = make_budget();
        let mut owner = ResidentParameterConversions::default();
        let mut shared = ResidentParameterConversions::default();
        owner.register([&weight], &target).unwrap();
        shared.register([&weight], &other).unwrap();
        assert_eq!(trim_budget(&target).unwrap().released_claims, 0);
        let retained_graph = promoted_weight(&input, &weight, &stream).unwrap().unwrap();
        let expected = retained_graph
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let report = trim_budget(&target).unwrap();
        assert_eq!(
            (report.released_claims, report.released_payload_bytes),
            (1, 16)
        );
        assert!(report.reclaimed_backing_bytes.value().is_none());
        assert!(owner.resident_conversions().is_empty());
        assert_eq!(usage(&other).retained_payload_bytes, 16);
        assert_eq!(trim_budget(&target).unwrap().released_claims, 0);
        assert_eq!(
            retained_graph.evaluated().unwrap().as_slice::<f32>(),
            expected
        );
        let reused = promoted_weight(&input, &weight, &stream).unwrap().unwrap();
        assert_eq!(reused.graph_identity(), retained_graph.graph_identity());
        assert_eq!(usage(&target).retained_payload_bytes, 16);
        trim_budget(&target).unwrap();
        trim_budget(&other).unwrap();
        let fresh = promoted_weight(&input, &weight, &stream).unwrap().unwrap();
        assert_eq!(fresh.evaluated().unwrap().as_slice::<f32>(), expected);
        assert_eq!(usage(&target).retained_payload_bytes, 16);
        invalidate(&weight);
        trim_budget(&target).unwrap();
        assert!(promoted_weight(&input, &weight, &stream).unwrap().is_none());
    }

    #[test]
    fn trimming_during_other_groups_evaluation_cannot_be_undone_by_late_publication() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let input = Array::from_slice(&[0.5f32, -0.25], &[1, 2]);
        let weight = fixture(Dtype::Float16, &stream);
        let make_budget = || {
            execution_budget(
                Some(ParameterConversionRetentionPolicy::Bounded { max_bytes: 16 }),
                ParameterConversionRetentionEligibility::Eligible,
            )
            .unwrap()
        };
        let publisher = make_budget();
        let other = make_budget();
        let mut first = ResidentParameterConversions::default();
        let mut second = ResidentParameterConversions::default();
        first.register([&weight], &publisher).unwrap();
        second.register([&weight], &other).unwrap();
        promote_with(&input, &weight, || {
            assert!(matches!(
                trim_budget(&publisher),
                Err(ConversionRetentionError::Busy)
            ));
            assert_eq!(
                trim_budget(&other)
                    .unwrap()
                    .remaining
                    .retained_payload_bytes,
                0
            );
            let converted = weight.as_dtype(Dtype::Float32, &stream)?;
            safemlx::transforms::eval([&converted])?;
            Ok(converted)
        })
        .unwrap()
        .unwrap();
        assert_eq!(usage(&publisher).retained_payload_bytes, 16);
        assert_eq!(usage(&other).retained_payload_bytes, 0);
        assert!(second.resident_conversions().is_empty());
        promoted_weight(&input, &weight, &stream).unwrap().unwrap();
        assert_eq!(usage(&other).retained_payload_bytes, 16);
    }

    #[test]
    fn conversion_policy_matrix_bounds_mixed_promotions_independently_of_allocator_cache() {
        struct RestoreCache(usize);
        impl Drop for RestoreCache {
            fn drop(&mut self) {
                safemlx::memory::set_cache_limit(self.0).unwrap();
            }
        }
        let _cache = RestoreCache(safemlx::memory::set_cache_limit(0).unwrap());
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let input = Array::from_slice(&[0.375f32, -0.625, 1.125, 0.25], &[2, 2]);
        use ParameterConversionRetentionPolicy::{Bounded, Disabled, Unlimited};
        for allocator_cap in [0, 256 << 20] {
            safemlx::memory::set_cache_limit(allocator_cap).unwrap();
            for (policy, expected_bytes) in [
                (Disabled, 0),
                (Bounded { max_bytes: 1 }, 0),
                (Bounded { max_bytes: 16 }, 16),
                (Bounded { max_bytes: 24 }, 16),
                (Bounded { max_bytes: 32 }, 32),
                (
                    Bounded {
                        max_bytes: 32 << 20,
                    },
                    32,
                ),
                (
                    Bounded {
                        max_bytes: 256 << 20,
                    },
                    32,
                ),
                (
                    Bounded {
                        max_bytes: 512 << 20,
                    },
                    32,
                ),
                (Unlimited, 32),
            ] {
                let budget = execution_budget(
                    Some(policy),
                    ParameterConversionRetentionEligibility::Eligible,
                )
                .unwrap();
                let weights = [
                    fixture(Dtype::Float16, &stream),
                    fixture(Dtype::Bfloat16, &stream),
                ];
                // Distinct permanent units share exactly one allowance.
                let mut owners = [
                    ResidentParameterConversions::default(),
                    ResidentParameterConversions::default(),
                ];
                for (owner, weight) in owners.iter_mut().zip(&weights) {
                    owner.register([weight, &weight.clone()], &budget).unwrap();
                }
                assert_eq!(usage(&budget).retained_payload_bytes, 0);
                for weight in &weights {
                    let narrow = input.as_dtype(weight.dtype(), &stream).unwrap();
                    assert!(promoted_weight(&narrow, weight, &stream).unwrap().is_none());
                    let promoted = promoted_weight(&input, weight, &stream).unwrap();
                    let expected = input
                        .matmul(weight.transpose(&stream).unwrap(), &stream)
                        .unwrap();
                    let actual = input
                        .matmul(
                            promoted
                                .as_ref()
                                .unwrap_or(weight)
                                .transpose(&stream)
                                .unwrap(),
                            &stream,
                        )
                        .unwrap();
                    assert_eq!(
                        expected.evaluated().unwrap().as_slice::<f32>(),
                        actual.evaluated().unwrap().as_slice::<f32>()
                    );
                    let expected_token =
                        safemlx::ops::indexing::argmax_axis(&expected, -1, false, &stream).unwrap();
                    let actual_token =
                        safemlx::ops::indexing::argmax_axis(&actual, -1, false, &stream).unwrap();
                    assert_eq!(
                        expected_token.evaluated().unwrap().as_slice::<u32>(),
                        actual_token.evaluated().unwrap().as_slice::<u32>()
                    );
                }
                assert_eq!(usage(&budget).retained_payload_bytes, expected_bytes);
                assert_eq!(usage(&budget).reserved_payload_bytes, 0);
                let retained = promoted_weight(&input, &weights[0], &stream).unwrap();
                if let Some(retained) = retained {
                    assert_eq!(
                        promoted_weight(&input, &weights[0], &stream)
                            .unwrap()
                            .unwrap()
                            .graph_identity(),
                        retained.graph_identity()
                    );
                }
                drop(owners);
                assert_eq!(usage(&budget).retained_payload_bytes, 0);
                assert!(promoted_weight(&input, &weights[0], &stream)
                    .unwrap()
                    .is_none());
            }
        }
    }

    #[test]
    fn conversion_groups_require_independent_claims_and_denied_owners_do_not_retain() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let input = Array::from_slice(&[0.5f32, -0.25], &[1, 2]);
        let weight = fixture(Dtype::Bfloat16, &stream);
        let full = execution_budget(
            Some(ParameterConversionRetentionPolicy::Bounded { max_bytes: 16 }),
            ParameterConversionRetentionEligibility::Eligible,
        )
        .unwrap();
        let denied = execution_budget(
            Some(ParameterConversionRetentionPolicy::Bounded { max_bytes: 15 }),
            ParameterConversionRetentionEligibility::Eligible,
        )
        .unwrap();
        let independent = execution_budget(
            Some(ParameterConversionRetentionPolicy::Unlimited),
            ParameterConversionRetentionEligibility::Eligible,
        )
        .unwrap();
        let mut target = ResidentParameterConversions::default();
        let mut blocked = ResidentParameterConversions::default();
        let mut other = ResidentParameterConversions::default();
        target.register([&weight], &full).unwrap();
        blocked.register([&weight], &denied).unwrap();
        promoted_weight(&input, &weight, &stream).unwrap().unwrap();
        other.register([&weight], &independent).unwrap();
        assert_eq!(usage(&full).retained_payload_bytes, 16);
        assert_eq!(usage(&independent).retained_payload_bytes, 16);
        assert_eq!(usage(&denied).retained_payload_bytes, 0);
        assert!(blocked.resident_conversions().is_empty());
        assert_eq!(
            target.resident_conversions()[&weight.graph_identity()].allocation,
            other.resident_conversions()[&weight.graph_identity()].allocation
        );
        drop(target);
        assert_eq!(usage(&full).retained_payload_bytes, 0);
        assert_eq!(usage(&independent).retained_payload_bytes, 16);
        drop(other);
        assert_eq!(usage(&independent).retained_payload_bytes, 0);
        assert!(promoted_weight(&input, &weight, &stream).unwrap().is_none());
    }

    #[test]
    fn conversion_failure_and_invalidation_during_evaluation_cancel_reservations() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let input = Array::from_slice(&[0.5f32, -0.25], &[1, 2]);
        let weight = fixture(Dtype::Float16, &stream);
        let budget = execution_budget(
            Some(ParameterConversionRetentionPolicy::Bounded { max_bytes: 16 }),
            ParameterConversionRetentionEligibility::Eligible,
        )
        .unwrap();
        let mut owner = ResidentParameterConversions::default();
        owner.register([&weight], &budget).unwrap();
        assert!(promote_with(&input, &weight, || {
            assert_eq!(usage(&budget).reserved_payload_bytes, 16);
            // A real MLX shape failure while the native publisher owns its ticket.
            weight.reshape(&[3], &stream)
        })
        .is_err());
        assert_eq!(usage(&budget).reserved_payload_bytes, 0);
        assert_eq!(usage(&budget).retained_payload_bytes, 0);
        assert!(promote_with(&input, &weight, || {
            assert_eq!(usage(&budget).reserved_payload_bytes, 16);
            invalidate(&weight);
            let converted = weight.as_dtype(Dtype::Float32, &stream)?;
            safemlx::transforms::eval([&converted])?;
            Ok(converted)
        })
        .unwrap()
        .is_none());
        assert_eq!(usage(&budget).reserved_payload_bytes, 0);
        assert_eq!(usage(&budget).retained_payload_bytes, 0);
        assert!(promoted_weight(&input, &weight, &stream).unwrap().is_none());
        let mut restored = ResidentParameterConversions::default();
        restored.register([&weight], &budget).unwrap();
        promoted_weight(&input, &weight, &stream).unwrap().unwrap();
        assert_eq!(usage(&budget).retained_payload_bytes, 16);
    }

    #[test]
    fn conversion_observation_preserves_shared_backing_and_owner_lifetimes() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let input = Array::from_slice(&[0.5f32, -0.25], &[1, 2]);
        let weight = Array::from_slice(&[1.0f32, -2.0, 0.25, 1.5], &[2, 2])
            .as_dtype(Dtype::Bfloat16, &stream)
            .unwrap();
        let separate = Array::from_slice(&[1.0f32, -2.0, 0.25, 1.5], &[2, 2])
            .as_dtype(Dtype::Bfloat16, &stream)
            .unwrap();
        let mut target = ResidentParameterConversions::default();
        let mut prediction = ResidentParameterConversions::default();
        target
            .register(
                [&weight, &separate],
                &execution_budget(None, ParameterConversionRetentionEligibility::Eligible).unwrap(),
            )
            .unwrap();
        prediction
            .register(
                [&weight],
                &execution_budget(None, ParameterConversionRetentionEligibility::Eligible).unwrap(),
            )
            .unwrap();
        assert_ne!(target.owner_identity(), prediction.owner_identity());
        // Observation cannot fill an eligible but unused cache.
        assert!(target.resident_conversions().is_empty());
        assert!(prediction.resident_conversions().is_empty());
        assert!(target.resident_conversions().is_empty());
        promoted_weight(&input, &weight, &stream).unwrap().unwrap();
        promoted_weight(&input, &separate, &stream)
            .unwrap()
            .unwrap();
        let target_report = target.resident_conversions();
        let prediction_report = prediction.resident_conversions();
        assert_eq!(target_report.len(), 2);
        assert_eq!(prediction_report.len(), 1);
        let shared = &target_report[&weight.graph_identity()];
        assert_eq!(shared.payload_bytes, 16);
        assert_eq!(
            shared.allocation,
            prediction_report[&weight.graph_identity()].allocation
        );
        assert_ne!(
            shared.allocation,
            target_report[&separate.graph_identity()].allocation
        );
        drop(target);
        assert_eq!(
            prediction.resident_conversions()[&weight.graph_identity()].allocation,
            shared.allocation
        );
        invalidate(&weight);
        assert!(prediction.resident_conversions().is_empty());
        let mut replacement = ResidentParameterConversions::default();
        replacement
            .register(
                [&weight],
                &execution_budget(None, ParameterConversionRetentionEligibility::Eligible).unwrap(),
            )
            .unwrap();
        promoted_weight(&input, &weight, &stream).unwrap().unwrap();
        assert_ne!(
            replacement.resident_conversions()[&weight.graph_identity()].allocation,
            shared.allocation
        );
    }

    #[test]
    fn parameter_conversion_reuses_aliases_preserves_values_and_expires_with_owner() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let input = Array::from_slice(&[0.375f32, -0.625, 1.125, 0.25], &[2, 2]);
        for dtype in [Dtype::Bfloat16, Dtype::Float16] {
            let weight = Array::from_slice(&[0.1234f32, -2.375, 0.53125, 1.0625], &[2, 2])
                .as_dtype(dtype, &stream)
                .unwrap();
            let alias = weight.clone();
            assert_eq!(weight.graph_identity(), alias.graph_identity());
            assert!(promoted_weight(&input, &weight, &stream).unwrap().is_none());
            let mut owner = ResidentParameterConversions::default();
            owner
                .register(
                    [&weight, &alias],
                    &execution_budget(None, ParameterConversionRetentionEligibility::Eligible)
                        .unwrap(),
                )
                .unwrap();
            assert_eq!(owner.resident_bytes().values().sum::<u64>(), 0);
            let promoted = promoted_weight(&input, &weight, &stream).unwrap().unwrap();
            let reused = promoted_weight(&input, &alias, &stream).unwrap().unwrap();
            assert_eq!(promoted.graph_identity(), reused.graph_identity());
            assert_eq!(owner.resident_bytes().values().sum::<u64>(), 16);
            let expected = input
                .matmul(weight.transpose(&stream).unwrap(), &stream)
                .unwrap();
            let actual = input
                .matmul(promoted.transpose(&stream).unwrap(), &stream)
                .unwrap();
            assert_eq!(
                expected.evaluated().unwrap().as_slice::<f32>(),
                actual.evaluated().unwrap().as_slice::<f32>()
            );
            let narrow_input = input.as_dtype(dtype, &stream).unwrap();
            let gain = Array::from_slice(&[1.0f32, 1.0], &[2]);
            let normalized = crate::backend::nn::normalization::input_precision_rms(
                &narrow_input,
                &gain,
                1e-5,
                &stream,
            )
            .unwrap();
            assert_eq!(normalized.dtype(), Dtype::Float32);

            assert!(promoted_weight(&narrow_input, &weight, &stream)
                .unwrap()
                .is_none());
            let replacement = weight
                .add(Array::from_f32(0.5), &stream)
                .unwrap()
                .as_dtype(dtype, &stream)
                .unwrap();
            assert_ne!(replacement.graph_identity(), weight.graph_identity());
            assert!(promoted_weight(&input, &replacement, &stream)
                .unwrap()
                .is_none());
            assert_eq!(
                promoted_weight(&input, &weight, &stream)
                    .unwrap()
                    .unwrap()
                    .graph_identity(),
                promoted.graph_identity()
            );
            // Publication revokes retired conversions so they cannot reduce
            // the future-cast allowance while an edit is active. Merely
            // borrowing a mutable slot for catalog inspection preserves reuse.
            use crate::module::{PhysicalParam, PhysicalParameter};
            use eredu_nn::Tensor;
            let mut physical = PhysicalParam::new(weight.clone());
            let _ = physical.as_nested_value_mut();
            assert_eq!(owner.resident_bytes().values().sum::<u64>(), 16);
            let mut slot = crate::MlxTensor::from_array(weight.clone());
            slot.publish_parameter(&crate::MlxTensor::from_array(replacement));
            assert_eq!(owner.resident_bytes().values().sum::<u64>(), 0);
            assert!(promoted_weight(&input, &alias, &stream).unwrap().is_none());
            slot.publish_parameter(&crate::MlxTensor::from_array(weight.clone()));
            assert!(promoted_weight(&input, slot.as_array(), &stream)
                .unwrap()
                .is_none());
            let mut replacement_owner = ResidentParameterConversions::default();
            replacement_owner
                .register(
                    [&weight],
                    &execution_budget(None, ParameterConversionRetentionEligibility::Eligible)
                        .unwrap(),
                )
                .unwrap();
            let newly_cached = promoted_weight(&input, &weight, &stream).unwrap().unwrap();
            assert_eq!(owner.resident_bytes().values().sum::<u64>(), 0);
            assert_eq!(replacement_owner.resident_bytes().values().sum::<u64>(), 16);
            drop(owner);
            assert_eq!(
                promoted_weight(&input, &weight, &stream)
                    .unwrap()
                    .unwrap()
                    .graph_identity(),
                newly_cached.graph_identity()
            );
            drop(replacement_owner);
            assert!(promoted_weight(&input, &weight, &stream).unwrap().is_none());
        }
    }
}
