//! Actual original pin banks, source registrations, canonical contexts/tickets
//! and final owners; thread controls are test-only and are not funding facts.
use super::*;
use std::sync::Barrier;

enum FinalOwner {
    Typed(BoundedRegisteredStorage<Key>),
    Parcel(SettledCaptureSourceParcel),
}
impl FinalOwner {
    fn retire(self) {
        match self {
            Self::Typed(owner) => drop(owner),
            Self::Parcel(owner) => drop(owner),
        }
    }
}
pub(super) fn retire_final_owners(
    groups: Vec<BoundedRegisteredStorage<Key>>,
    parcels: Vec<SettledCaptureSourceParcel>,
    pool: &WorkingMemoryPool,
    original_floor: u64,
) {
    assert!(!groups.is_empty());
    for group in &groups {
        group.validate_source(pool).unwrap();
    }
    let count = groups.len() + parcels.len();
    let ready = Arc::new(Barrier::new(count + 1));
    let release = Arc::new(Barrier::new(count + 1));
    let owners = groups
        .into_iter()
        .map(FinalOwner::Typed)
        .chain(parcels.into_iter().map(FinalOwner::Parcel));
    std::thread::scope(|scope| {
        let workers = owners
            .map(|owner| {
                let ready = ready.clone();
                let release = release.clone();
                scope.spawn(move || {
                    ready.wait();
                    release.wait();
                    owner.retire();
                })
            })
            .collect::<Vec<_>>();
        ready.wait();
        let held = pool.used_bytes();
        release.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert!(held.unwrap() >= original_floor);
    });
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn canonical_typed_pin_final_aliases_retire_concurrently_after_original_run() {
    for mode in [Mode::Success, Mode::Empty] {
        exercise_pins_retirement(mode, 3, None, true);
    }
}

#[test]
fn canonical_typed_and_erased_parcel_owners_share_one_concurrent_retirement() {
    exercise_pins_retirement(Mode::Success, 3, Some(OpeningMode::Success), true);
}
