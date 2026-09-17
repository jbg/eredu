use super::*;
use std::convert::Infallible;

thread_local! {
    pub(super) static FAIL_RESERVE: Cell<bool> = const { Cell::new(false) };
}

struct Witness(Rc<Cell<usize>>);
impl Drop for Witness {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
struct CompleteSlot {
    observed_node: Box<usize>,
    payload_node: Box<usize>,
    _custody: Witness,
}

#[test]
fn zero_bank_never_calls_factory_and_exhaustion_stays_fixed() {
    let calls = Cell::new(0);
    let mut bank = PreparedOperationBank::<CompleteSlot>::try_new(0, |_| {
        calls.set(calls.get() + 1);
        Err::<CompleteSlot, ()>(())
    })
    .unwrap();
    assert_eq!((bank.len(), bank.remaining(), calls.get()), (0, 0, 0));
    for _ in 0..2 {
        assert_eq!(bank.checkout().err(), Some(BankExhausted { prepared: 0 }));
    }
    assert_eq!(calls.get(), 0);
}

#[test]
fn exact_complete_slots_keep_both_node_identities_and_custody_without_refill() {
    let drops = Rc::new(Cell::new(0));
    let mut identities = Vec::with_capacity(3);
    let mut bank = PreparedOperationBank::try_new(3, |ordinal| {
        let slot = CompleteSlot {
            observed_node: Box::new(ordinal),
            payload_node: Box::new(ordinal + 10),
            _custody: Witness(drops.clone()),
        };
        identities.push((
            std::ptr::from_ref(&*slot.observed_node),
            std::ptr::from_ref(&*slot.payload_node),
        ));
        Ok::<_, Infallible>(slot)
    })
    .unwrap();
    let capacity = bank.slot_capacity();
    let mut checked_out = Vec::with_capacity(3);
    for ordinal in 0..3 {
        let slot = bank.checkout().unwrap();
        assert_eq!(
            (
                std::ptr::from_ref(&*slot.observed_node),
                std::ptr::from_ref(&*slot.payload_node)
            ),
            identities[ordinal]
        );
        assert_eq!(
            (*slot.observed_node, *slot.payload_node),
            (ordinal, ordinal + 10)
        );
        assert_eq!(bank.remaining(), 2 - ordinal);
        assert_eq!(bank.slot_capacity(), capacity);
        checked_out.push(slot);
    }
    assert_eq!(bank.checkout().err(), Some(BankExhausted { prepared: 3 }));
    assert_eq!(bank.checkout().err(), Some(BankExhausted { prepared: 3 }));
    assert_eq!(bank.slot_capacity(), capacity);
    drop(bank);
    assert_eq!(drops.get(), 0);
    drop(checked_out.remove(0));
    assert_eq!(drops.get(), 1);
    drop(checked_out);
    assert_eq!(drops.get(), 3);
}

#[test]
fn slot_failure_owns_actual_cause_prefix_and_unissued_factory_custody() {
    let slot_drops = Rc::new(Cell::new(0));
    let factory_drops = Rc::new(Cell::new(0));
    let factory = Witness(factory_drops.clone());
    let calls = Rc::new(Cell::new(0));
    let cause = Rc::new(29usize);
    let called = calls.clone();
    let failure_cause = cause.clone();
    let prepared_drops = slot_drops.clone();
    let error = PreparedOperationBank::try_new(4, move |ordinal| {
        let _retain_factory = &factory;
        called.set(called.get() + 1);
        if ordinal == 2 {
            return Err(failure_cause.clone());
        }
        Ok(CompleteSlot {
            observed_node: Box::new(ordinal),
            payload_node: Box::new(ordinal + 10),
            _custody: Witness(prepared_drops.clone()),
        })
    })
    .unwrap_err();
    assert_eq!(error.prepared_len(), 2);
    assert_eq!(
        (calls.get(), slot_drops.get(), factory_drops.get()),
        (3, 0, 0)
    );
    let (failure, mut prefix, prepare) = error.into_parts();
    let BankPreparationCause::Slot {
        ordinal,
        cause: actual,
    } = failure
    else {
        panic!("actual slot failure expected")
    };
    assert_eq!(ordinal, 2);
    assert!(Rc::ptr_eq(&cause, &actual));
    let first = prefix.checkout().unwrap();
    assert_eq!(*first.observed_node, 0);
    drop(prefix);
    assert_eq!(slot_drops.get(), 1);
    drop(first);
    assert_eq!(slot_drops.get(), 2);
    drop(prepare);
    assert_eq!(factory_drops.get(), 1);
}

#[test]
fn overflow_and_real_vec_capacity_failure_keep_owned_factory_before_any_slot() {
    for force_reserve in [false, true] {
        let drops = Rc::new(Cell::new(0));
        let calls = Rc::new(Cell::new(0));
        let witness = Witness(drops.clone());
        let called = calls.clone();
        FAIL_RESERVE.with(|flag| flag.set(force_reserve));
        let error = PreparedOperationBank::<CompleteSlot>::try_new(
            if force_reserve { 1 } else { usize::MAX },
            move |_| {
                let _retained = &witness;
                called.set(called.get() + 1);
                Err::<CompleteSlot, ()>(())
            },
        )
        .unwrap_err();
        assert_eq!(error.prepared_len(), 0);
        assert_eq!((calls.get(), drops.get()), (0, 0));
        if force_reserve {
            // The real slot Vec receives usize::MAX: actual CapacityOverflow,
            // not a claim that allocator exhaustion was injected.
            assert!(matches!(&error.cause, BankPreparationCause::Reserve(_)));
        } else {
            assert!(matches!(&error.cause, BankPreparationCause::Overflow));
        }
        drop(error);
        assert_eq!(drops.get(), 1);
    }
}

#[test]
fn cold_bank_layout_counts_actual_option_slots_and_checked_prepared_children() {
    type Factory = fn(usize) -> Result<CompleteSlot, ()>;
    let zero = PreparedOperationBank::<CompleteSlot>::layout::<Factory, ()>(0, 73).unwrap();
    let one = PreparedOperationBank::<CompleteSlot>::layout::<Factory, ()>(1, 73).unwrap();
    let two = PreparedOperationBank::<CompleteSlot>::layout::<Factory, ()>(2, 73).unwrap();
    assert_eq!(zero.slot_array_bytes, 0);
    assert_eq!(one.slot_array_bytes, size_of::<Option<CompleteSlot>>());
    assert_eq!(two.slot_array_bytes, 2 * size_of::<Option<CompleteSlot>>());
    assert_eq!(two.prepared_slot_control_bytes, 146);
    assert_eq!(
        two.total_control_bytes - one.total_control_bytes,
        73 + size_of::<Option<CompleteSlot>>() as u64
    );
    assert_eq!(
        one.total_control_bytes - zero.total_control_bytes,
        two.total_control_bytes - one.total_control_bytes
    );
    assert!(PreparedOperationBank::<CompleteSlot>::layout::<Factory, ()>(usize::MAX, 1).is_none());
    assert!(PreparedOperationBank::<CompleteSlot>::layout::<Factory, ()>(2, u64::MAX).is_none());
}
