use super::*;
use crate::backend::runtime::residency::manager::ResidencyManager;
use eredu_checkpoint::store::{MemoryWeightStore, RetainedCheckpointSource, TensorSelection};
use eredu_core::residency::{
    MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
};
use eredu_nn::{Parameter, ParameterSpec};
use eredu_runtime::{
    ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout, LayerwisePolicy, OffloadUnit,
};
use safemlx::{Device, DeviceType};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::mpsc,
    time::Duration,
};

type Unit = Vec<Parameter<MlxTensor>>;
type Selected = MlxSelectedLayerwisePolicy<Unit, MlxSelectiveUnitPopulator>;
const VALUES: [[i32; 2]; 2] = [[3, 7], [11, 13]];

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|calls| calls.set(calls.get() + 1));
}
struct Hook;
impl Drop for Hook {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}
fn cold<R>(operation: impl FnOnce() -> R) -> R {
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let hook = Hook;
    HOUSEKEEPING.with(|calls| calls.set(0));
    let result = operation();
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(hook);
    result
}

fn fixture(resident: bool) -> (Selected, Stream, ExecutionUnitLayout) {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source: RetainedCheckpointSource = (Arc::new(
        MemoryWeightStore::from_safetensors(VALUES.iter().enumerate().map(|(i, values)| {
            (
                format!("w{i}"),
                safetensors::Dtype::I32,
                vec![2],
                values.iter().copied().flat_map(i32::to_le_bytes).collect(),
            )
        }))
        .unwrap(),
    ))
    .into();
    let ids: Vec<_> = (0..2)
        .map(|i| OffloadUnitId::new(format!("unit{i}")).unwrap())
        .collect();
    let definitions: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            OffloadUnit::new(
                id.clone(),
                [
                    WeightBinding::new("weight", format!("w{i}"), TensorSelection::Full, 8)
                        .unwrap(),
                ],
            )
            .unwrap()
        })
        .collect();
    let plan = OffloadPlan::new(
        OffloadConfig::new(Some(65_536), Some(65_536), 1).unwrap(),
        ids.iter().map(|id| {
            OffloadUnitSpec::new(id.clone(), 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)
                .unwrap()
        }),
    )
    .unwrap();
    let manager = ResidencyManager::new_shared(
        source.clone(),
        plan,
        definitions,
        stream.clone(),
        stream.clone(),
    )
    .unwrap();
    manager.initialize().unwrap();
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("units")], "units").unwrap();
    let layout = ExecutionUnitLayout::new(&graph, [2]).unwrap();
    let mut policy = MlxLayerwisePolicy::new(
        manager,
        source,
        ids,
        layout.clone(),
        1,
        MlxSelectiveUnitPopulator::new(BTreeSet::new()),
        Vec::new(),
        None,
        false,
        false,
    )
    .unwrap();
    let addresses: Vec<_> = (0..2).map(|i| layout.address(i).unwrap()).collect();
    let selected = if resident {
        let units: Vec<Unit> = (0..2)
            .map(|_| {
                vec![Parameter::unloaded_i32(
                    ParameterSpec::trainable("weight").unwrap(),
                    &[2],
                    &stream,
                )
                .unwrap()]
            })
            .collect();
        let policy = policy.into_resident_units(units, &stream).unwrap();
        Selected::resident(policy, &layout, &addresses).unwrap()
    } else {
        // The actual bounded policy keeps override owners while unloaded units
        // stay in its residency manager. This visitor covers those overrides.
        let replacements: BTreeMap<_, _> = VALUES
            .iter()
            .enumerate()
            .map(|(i, values)| {
                (
                    format!("override{i}"),
                    MlxTensor::from_array(Array::from_slice(values, &[2])),
                )
            })
            .collect();
        assert!(policy
            .publish_parameter_replacements(&replacements, true)
            .unwrap());
        Selected::bounded(policy, &layout, &addresses).unwrap()
    };
    (selected, stream, layout)
}

// Independent inspection of the retained fields, before the tested wrapper.
// Alias creation uses the existing limited inspection clone; numerical reads
// happen later, outside the policy mutex and measured cold callback.
fn originals(policy: &Selected) -> (Vec<usize>, Vec<Array>) {
    let inner = policy.inner.lock().unwrap();
    let mut pointers = Vec::new();
    let mut aliases = Vec::new();
    let mut visit = |value: &MlxTensor| {
        pointers.push(value as *const MlxTensor as usize);
        aliases.push(value.as_array().try_clone_for_inspection().unwrap());
    };
    let complete = match &*inner {
        MlxSelectedLayerwisePolicyInner::Resident(policy) => {
            policy.visit_retained_values(&mut visit)
        }
        MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
            policy.visit_retained_values(&mut visit)
        }
    };
    assert!(complete);
    assert_eq!(pointers.len(), 2);
    (pointers, aliases)
}

fn assert_values(aliases: &[Array]) {
    assert_eq!(aliases.len(), 2);
    for (alias, expected) in aliases.iter().zip(VALUES) {
        assert_eq!(
            alias.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
            expected
        );
    }
}

// Construct, inspect and destroy native owners on one thread. The actual enum
// contains thread-affine recovery even when its active branch is Resident; it
// cannot safely be sent to a foreign lock holder. The outer timeout bounds a
// regression to same-thread blocking without inventing a Send policy wrapper.
fn on_owner_thread(operation: fn()) {
    let (done_tx, done_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        operation();
        done_tx.send(()).unwrap();
    });
    done_rx
        .recv_timeout(Duration::from_secs(20))
        .expect("selected-policy inspection must not wait on its own mutex");
    worker.join().unwrap();
}

#[test]
fn selected_retained_visits_borrow_both_real_nonzero_branches_without_housekeeping() {
    for resident in [true, false] {
        let (policy, stream, _) = fixture(resident);
        let (expected, aliases) = originals(&policy);
        let mut actual = Vec::with_capacity(expected.len());
        assert!(cold(|| policy.visit_retained_values(&mut |value| {
            actual.push(value as *const MlxTensor as usize);
        })));
        assert_eq!(actual, expected);
        assert_eq!(policy.retained_value_slot_bound(), Some(2));
        drop(policy);
        crate::backend::ordinary_retirement::reclaim_all();
        assert_values(&aliases);
        drop((aliases, stream));
    }
}

#[test]
fn selected_retained_visits_reject_held_mutex_without_callbacks_then_recover() {
    on_owner_thread(|| {
        for resident in [true, false] {
            let (policy, _stream, _) = fixture(resident);
            let held = policy.inner.lock().unwrap();
            let mut calls = 0;
            assert!(!cold(|| policy.visit_retained_values(&mut |_| calls += 1)));
            assert_eq!(calls, 0);
            assert_eq!(policy.retained_value_slot_bound(), None);
            drop(held);
            assert!(cold(|| policy.visit_retained_values(&mut |_| calls += 1)));
            assert_eq!(calls, 2);
        }
        crate::backend::ordinary_retirement::reclaim_all();
    });
}

#[test]
fn selected_retained_callback_reentry_is_incomplete_without_losing_outer_prefix() {
    on_owner_thread(|| {
        for resident in [true, false] {
            let (policy, _stream, _) = fixture(resident);
            let (expected, aliases) = originals(&policy);
            let sibling = policy.clone();
            let mut actual = Vec::with_capacity(2);
            let mut nested = 0;
            assert!(cold(|| policy.visit_retained_values(&mut |value| {
                actual.push(value as *const MlxTensor as usize);
                assert!(!sibling.visit_retained_values(&mut |_| nested += 1));
                assert_eq!(sibling.retained_value_slot_bound(), None);
            })));
            assert_eq!(nested, 0);
            assert_eq!(actual, expected);
            assert_values(&aliases);
        }
        crate::backend::ordinary_retirement::reclaim_all();
    });
}

#[test]
fn selected_poison_is_incomplete_and_keeps_actual_native_owners() {
    for resident in [true, false] {
        let (policy, _stream, _) = fixture(resident);
        let (_, aliases) = originals(&policy);
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _held = policy.inner.lock().unwrap();
            panic!("deliberate selected-policy poison");
        }));
        assert!(panic.is_err());
        let mut calls = 0;
        assert!(!cold(|| policy.visit_retained_values(&mut |_| calls += 1)));
        assert_eq!(calls, 0);
        assert_eq!(policy.retained_value_slot_bound(), None);
        assert_values(&aliases);
        drop(policy);
        crate::backend::ordinary_retirement::reclaim_all();
        assert_values(&aliases);
    }
}

#[test]
fn selected_callback_unwind_preserves_visited_alias_and_poisoned_owner() {
    for resident in [true, false] {
        let (policy, _stream, _) = fixture(resident);
        let mut prefix = None;
        let panic = catch_unwind(AssertUnwindSafe(|| {
            cold(|| {
                policy.visit_retained_values(&mut |value| {
                    prefix = Some(value.as_array().try_clone_for_inspection().unwrap());
                    panic!("retained callback failed after retaining the first owner");
                })
            });
        }));
        assert!(panic.is_err());
        assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
        assert!(!cold(|| policy.visit_retained_values(&mut |_| {
            panic!("poison must not enter another callback")
        })));
        drop(policy);
        crate::backend::ordinary_retirement::reclaim_all();
        assert_eq!(
            prefix
                .unwrap()
                .evaluated()
                .unwrap()
                .try_to_vec::<i32>()
                .unwrap(),
            VALUES[0]
        );
    }
}

#[test]
fn selected_resident_missing_unit_keeps_available_prefix_without_acquire_or_reap() {
    let (mut policy, stream, layout) = fixture(true);
    let (expected, aliases) = originals(&policy);
    let address = layout.address(0).unwrap();
    let lease = policy
        .acquire::<Error, _>(
            0,
            address,
            |_| panic!("resident acquire must not build"),
            &stream,
        )
        .unwrap();
    let mut actual = Vec::with_capacity(2);
    assert!(!cold(|| policy.visit_retained_values(&mut |value| {
        actual.push(value as *const MlxTensor as usize);
    })));
    assert_eq!(actual, [expected[1]]);
    assert_eq!(policy.retained_value_slot_bound(), None);
    policy.abort(Some((0, address, lease)), &stream);
    actual.clear();
    assert!(cold(|| policy.visit_retained_values(&mut |value| {
        actual.push(value as *const MlxTensor as usize);
    })));
    assert_eq!(actual, expected);
    assert_values(&aliases);
}
