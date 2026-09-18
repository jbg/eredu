//! Invoked by the genuine composition fixture. Additional table preparation is
//! tested as a mechanism; this does not admit this separate manager under Q.
use super::super::{
    materialization::{PreparedArrayValues, PreparedResidentArrays},
    tests as ordinary,
};
use super::*;
use eredu_checkpoint::store::TensorSelection;
use eredu_core::residency::{
    MemoryTier, OffloadConfig, OffloadPlan, ResidencyPolicy, TransferDirection,
};
use eredu_runtime::{working_memory::WorkingMemoryPool, WeightBinding};

pub(crate) struct NamedArraysFixture {
    manager: ResidencyManager,
    foreign: ResidencyManager,
    _directory: tempfile::TempDir,
    first: Option<Array>,
    second: Option<Array>,
    duplicate: Option<Array>,
    next_second: Option<Array>,
    warm_first: Option<Array>,
}
pub(crate) struct FirstNamedOwners {
    first: ResidentArraysOwner,
    second: ResidentArraysOwner,
}
impl FirstNamedOwners {
    pub(crate) fn collect_plain_before_canonical(
        &self,
    ) -> super::super::super::storage::CanonicalCollectionFixture {
        use super::super::super::storage::{CanonicalCollectionFixture, RetainedStorageRef};
        let cell = self
            .first
            .arrays
            .arrays
            .retained_values()
            .find_map(|value| match value {
                RetainedStorageRef::CanonicalArray(cell)
                    if std::ptr::eq(
                        cell.array(),
                        self.first.arrays.arrays.get("own_a").unwrap(),
                    ) =>
                {
                    Some(cell)
                }
                _ => None,
            })
            .expect("actual prepared canonical destination");
        let collected = CanonicalCollectionFixture::new(cell);
        collected.assert_values();
        collected
    }
}
impl NamedArraysFixture {
    /// Every stream, source and reference numerical value is prepared before
    /// entering either original request's Source role.
    pub(crate) fn new() -> Self {
        let (directory, source) = ordinary::fixture_store();
        let first = ordinary::unit(
            "first",
            [
                ordinary::binding("own_a", "a", TensorSelection::Full, 8)
                    .with_logical_target("first.owner")
                    .unwrap(),
                WeightBinding::alias("other_b", "second.owner", 8).unwrap(),
            ],
        );
        let second = ordinary::unit(
            "second",
            [
                ordinary::binding("own_b", "b", TensorSelection::Full, 8)
                    .with_logical_target("second.owner")
                    .unwrap(),
                WeightBinding::alias("other_a", "first.owner", 8).unwrap(),
            ],
        );
        let make = || {
            ResidencyManager::new(
                Arc::clone(&source),
                OffloadPlan::new(
                    OffloadConfig::new(Some(16), Some(0), 1).unwrap(),
                    [
                        ordinary::spec("first", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                        ordinary::spec("second", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                    ],
                )
                .unwrap(),
                [first.clone(), second.clone()],
                ordinary::cpu_stream(),
                ordinary::cpu_stream(),
            )
            .unwrap()
        };
        let manager = make();
        let foreign = make();
        manager.initialize().unwrap();
        let mut transfer = manager
            .acquire_many_with_transfer(&[(ordinary::id("first"), 1)], MemoryTier::Device)
            .unwrap();
        transfer.synchronize().unwrap();
        let first_lease = &transfer.leases()[0];
        let second_lease = manager
            .acquire(&ordinary::id("second"), MemoryTier::Device)
            .unwrap();
        assert_eq!(
            first_lease
                .device_value("own_a")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2]
        );
        assert_eq!(
            first_lease
                .device_value("other_b")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[3, 4]
        );
        let a = first_lease.device_value("own_a").unwrap().clone();
        let duplicate = a.clone();
        let warm_first = a.clone();
        let b = second_lease.device_value("own_b").unwrap().clone();
        let next_second = b.clone();
        drop((second_lease, transfer));
        Self {
            manager,
            foreign,
            _directory: directory,
            first: Some(a),
            second: Some(b),
            duplicate: Some(duplicate),
            next_second: Some(next_second),
            warm_first: Some(warm_first),
        }
    }

    pub(crate) fn first_request(
        &mut self,
        controls: &OriginalTextControlGuard,
    ) -> FirstNamedOwners {
        let roots = [ordinary::id("first")];
        let mut scratch = [ResidencyClosureSlot::default(); 2];
        let facts = self
            .manager
            .named_storage_layout(&roots, &mut scratch)
            .unwrap();
        assert_eq!((facts.units, facts.rows, facts.physical_cells), (2, 4, 2));
        let catalog = self
            .manager
            .prepare_name_catalog(&roots, &mut scratch, controls.metadata_custody().into())
            .unwrap_or_else(|error| panic!("actual catalog: {}", error.into_source()));
        let mut window = catalog
            .prepare_window(&self.manager, &roots, &mut scratch)
            .unwrap_or_else(|_| panic!("actual window"));
        let state = self.manager.inner.state.lock().unwrap();
        let first_unit = state.control.unit(&roots[0]).unwrap();
        let second_id = ordinary::id("second");
        let second_unit = state.control.unit(&second_id).unwrap();
        let mut first = window.take_unit(&self.manager.inner, first_unit).unwrap();
        assert!(matches!(
            window.take_unit(&self.manager.inner, first_unit),
            Err(NamedArrayError::UnknownUnit)
        ));
        let mut second = window.take_unit(&self.manager.inner, second_unit).unwrap();
        let a = self.first.take().unwrap();
        let a_handle = a.as_ptr().ctx;
        let (cause, a) = first.arrays.put("unknown", a).err().unwrap();
        assert_eq!(cause, NamedArrayError::UnknownName);
        assert_eq!(a.as_ptr().ctx, a_handle);
        first.arrays.put("own_a", a).unwrap();
        let duplicate = self.duplicate.take().unwrap();
        let duplicate_handle = duplicate.as_ptr().ctx;
        let (cause, duplicate) = first.arrays.put("own_a", duplicate).err().unwrap();
        assert_eq!(cause, NamedArrayError::DuplicateValue);
        assert_eq!(duplicate.as_ptr().ctx, duplicate_handle);
        drop(duplicate);
        second
            .arrays
            .put("own_b", self.second.take().unwrap())
            .unwrap();
        assert_eq!(
            first.validate_publication(&self.manager.inner, first_unit),
            Err(NamedArrayError::MissingValue)
        );
        assert_eq!(
            second.validate_publication(&self.foreign.inner, second_unit),
            Err(NamedArrayError::ForeignManager)
        );
        // Every rejection left both actual published ordinary maps untouched.
        assert!(!state.storage[&roots[0]]
            .device
            .as_ref()
            .unwrap()
            .arrays
            .arrays
            .is_prepared());
        assert!(!state.storage[&second_id]
            .device
            .as_ref()
            .unwrap()
            .arrays
            .arrays
            .is_prepared());
        let mut prepared = vec![
            (
                roots[0].clone(),
                PreparedResidentArrays {
                    arrays: PreparedArrayValues::Original(first),
                    direction: TransferDirection::DiskToDevice,
                },
            ),
            (
                second_id.clone(),
                PreparedResidentArrays {
                    arrays: PreparedArrayValues::Original(second),
                    direction: TransferDirection::DiskToDevice,
                },
            ),
        ];
        super::super::transfer::bind_device_for_test(&state, &mut prepared).unwrap();
        assert!(std::ptr::eq(
            prepared[0].1.arrays.get("own_a").unwrap(),
            prepared[1].1.arrays.get("other_a").unwrap()
        ));
        assert!(std::ptr::eq(
            prepared[1].1.arrays.get("own_b").unwrap(),
            prepared[0].1.arrays.get("other_b").unwrap()
        ));
        let pointer = prepared[0].1.arrays.get("own_a").unwrap().as_ptr().ctx;
        assert_eq!(
            prepared[0].1.arrays.reset_unpublished(),
            Err(NamedArrayError::SharedDestination)
        );
        assert_eq!(
            prepared[0].1.arrays.get("own_a").unwrap().as_ptr().ctx,
            pointer
        );
        for (id, values) in &mut prepared {
            values
                .arrays
                .validate_publication(Some(&self.manager.inner), state.control.unit(id).unwrap())
                .unwrap();
            let ordinary = &state.storage[id].device.as_ref().unwrap().arrays.arrays;
            assert_eq!(
                values.arrays.keys().collect::<Vec<_>>(),
                ordinary.keys().collect::<Vec<_>>()
            );
            for (name, value) in values.arrays.iter() {
                assert_eq!(
                    value.try_metadata_snapshot().unwrap().allocation(),
                    ordinary
                        .get(name)
                        .unwrap()
                        .try_metadata_snapshot()
                        .unwrap()
                        .allocation()
                );
            }
        }
        let (_, second) = prepared.pop().unwrap();
        let (_, first) = prepared.pop().unwrap();
        let first = first
            .arrays
            .publish(Some(&self.manager.inner), first_unit)
            .unwrap();
        let second = second
            .arrays
            .publish(Some(&self.manager.inner), second_unit)
            .unwrap();
        assert!(first.arrays.arrays.is_prepared() && second.arrays.arrays.is_prepared());
        FirstNamedOwners { first, second }
    }

    /// A new genuine request has unrelated custody. Reusing an old canonical
    /// cell must continue to retain the first request after its tables retire.
    pub(crate) fn second_request(
        &mut self,
        old: FirstNamedOwners,
        mut collected: super::super::super::storage::CanonicalCollectionFixture,
        controls: &OriginalTextControlGuard,
        old_pool: &WorkingMemoryPool,
    ) {
        let roots = [ordinary::id("second")];
        let mut scratch = [ResidencyClosureSlot::default(); 2];
        let catalog = self
            .manager
            .prepare_name_catalog(&roots, &mut scratch, controls.metadata_custody().into())
            .unwrap_or_else(|_| panic!("second request catalog"));
        let mut window = catalog
            .prepare_window(&self.manager, &roots, &mut scratch)
            .unwrap_or_else(|_| panic!("second request window"));
        let state = self.manager.inner.state.lock().unwrap();
        let unit = state.control.unit(&roots[0]).unwrap();
        let first_id = ordinary::id("first");
        let first_unit = state.control.unit(&first_id).unwrap();
        let mut warm_alias = window.take_unit(&self.manager.inner, first_unit).unwrap();
        let ordinary_owner = state.storage[&roots[0]].device.as_ref().unwrap();
        let warm = ordinary_owner
            .alias_value("own_b", &roots[0], &catalog)
            .unwrap();
        warm_alias.arrays.put_alias("other_b", warm).unwrap();
        assert!(std::ptr::eq(
            warm_alias.arrays.get("other_b").unwrap(),
            ordinary_owner.arrays.arrays.get("own_b").unwrap()
        ));
        assert_eq!(
            warm_alias.validate_publication(&self.manager.inner, first_unit),
            Err(NamedArrayError::MissingValue)
        );
        warm_alias
            .arrays
            .put("own_a", self.warm_first.take().unwrap())
            .unwrap();
        let warm_alias = warm_alias
            .publish(&self.manager.inner, first_unit)
            .unwrap_or_else(|_| panic!("published ordinary warm-owner alias"));
        assert!(std::ptr::eq(
            warm_alias.arrays.arrays.get("other_b").unwrap(),
            ordinary_owner.arrays.arrays.get("own_b").unwrap()
        ));
        let mut second = window.take_unit(&self.manager.inner, unit).unwrap();
        second
            .arrays
            .put("own_b", self.next_second.take().unwrap())
            .unwrap();
        let wrong = old
            .second
            .alias_value("own_b", &ordinary::id("second"), &catalog)
            .unwrap();
        assert_eq!(
            second.arrays.put_alias("other_a", wrong),
            Err(NamedArrayError::InvalidSource)
        );
        assert!(second.arrays.get("other_a").is_none());
        let alias = old
            .first
            .alias_value("own_a", &ordinary::id("first"), &catalog)
            .unwrap();
        second.arrays.put_alias("other_a", alias.clone()).unwrap();
        assert_eq!(
            second.arrays.put_alias("other_a", alias),
            Err(NamedArrayError::DuplicateValue)
        );
        let old_a = old.first.arrays.arrays.get("own_a").unwrap().as_ptr().ctx;
        assert_eq!(second.arrays.get("other_a").unwrap().as_ptr().ctx, old_a);
        let second = second
            .publish(&self.manager.inner, unit)
            .unwrap_or_else(|_| panic!("complete second table"));
        drop(state);
        // A different genuine prepared cell contains the same physical Array,
        // but has this later request's custody. It must not replace A's proof
        // source merely because its native allocation key is equal.
        let conflicting = warm_alias
            .arrays
            .arrays
            .retained_values()
            .find_map(|value| match value {
                super::super::super::storage::RetainedStorageRef::CanonicalArray(cell) => Some(cell),
                _ => None,
            })
            .expect("later request canonical cell");
        collected.reject_conflict(conflicting);
        drop(warm_alias);
        drop(old);
        assert!(
            old_pool.used_bytes().unwrap() > 0,
            "the old physical cell still owns old Q"
        );
        assert_eq!(
            second.arrays.arrays.get("other_a").unwrap().as_ptr().ctx,
            old_a
        );
        drop(second);
        collected.assert_values();
        let with_original_cell = old_pool.used_bytes().unwrap();
        drop(collected);
        assert!(
            old_pool.used_bytes().unwrap() < with_original_cell,
            "retiring the last upgraded row releases A while B remains live"
        );
        // The composition caller performs the final ordinary pool settlement
        // after this distinct original role also tears down. No global reap is
        // used as evidence while the old cell is live.
    }
}
