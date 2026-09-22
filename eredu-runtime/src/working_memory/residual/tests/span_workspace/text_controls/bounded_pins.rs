use super::*;
use crate::working_memory::PreparedPrefillStoragePinPlan;
use std::mem::size_of;

#[test]
fn typed_rows_add_s_once_before_original_exact_admission_and_extract_once() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let raw = replacement_quote(&pool, geometry(), 0).into_incremental();
    let before = raw.incremental_bytes().unwrap();
    let pins =
        PreparedPrefillStoragePinPlan::<u32>::prepare(raw.span_workspace().plan(), |_| Some(3))
            .unwrap();
    let s = pins.control_peak_bytes();
    assert!(s > 3 * size_of::<u32>() as u64);
    let c = prepared(&source, &raw)
        .with_prefill_storage_pins(pins)
        .unwrap();
    assert_eq!(c.storage_pin_control_bytes(), s);
    let q = raw.with_span_workspace_and_text_controls(c).unwrap();
    let p = q.span_workspace().retention_peak_bytes().unwrap();
    assert_eq!(q.incremental_bytes().unwrap(), before + p + 51 + s);
    let exact = exact_capacity(&pool, &q);
    assert!(matches!(
        sealed_plan(&pool, &q, exact - 1),
        Err(PrefillPlanningError::Reservation(
            capacity_error
        )) if matches!(capacity_numbers(&capacity_error), Some((_, _)))));
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    let (r, accepted) = sealed_plan(&pool, &q, exact).unwrap();
    let (r, run) = r.into_funding().unwrap();
    let (mut owner, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let held = account(&pool, &r);
    assert_eq!(held.1, p + 51 + s);
    assert!(owner.take_prefill_storage_pins::<u64>().is_err());
    let bank = owner.take_prefill_storage_pins::<u32>().unwrap();
    assert_eq!(bank.spent_rows(), 0);
    assert!(owner.take_prefill_storage_pins::<u32>().is_err());
    assert_eq!(
        account(&pool, &r),
        held,
        "extraction creates no extra hold or scope"
    );
    drop((owner, q, r, run, root));
    assert!(pool.payload_used_bytes().unwrap() >= p + 51 + s);
    drop(bank);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn slot_layout_rejects_equal_other_plan_rebind_unknown_and_overflow() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let raw = replacement_quote(&pool, geometry(), 0).into_incremental();
    let other = replacement_quote(&pool, geometry(), 0).into_incremental();
    let make = |q: &IncrementalInferenceQuote| {
        PreparedPrefillStoragePinPlan::<u32>::prepare(q.span_workspace().plan(), |_| Some(0))
            .unwrap()
    };
    assert!(
        prepared(&source, &raw)
            .with_prefill_storage_pins(make(&other))
            .is_err()
    );
    let bound = prepared(&source, &raw)
        .with_prefill_storage_pins(make(&raw))
        .unwrap();
    assert!(bound.with_prefill_storage_pins(make(&raw)).is_err());
    assert!(matches!(
        PreparedPrefillStoragePinPlan::<u32>::prepare(raw.span_workspace().plan(), |_| None),
        Err(WorkingMemoryError::UnknownBound)
    ));
    assert!(matches!(
        PreparedPrefillStoragePinPlan::<u32>::prepare(raw.span_workspace().plan(), |_| Some(
            usize::MAX
        )),
        Err(WorkingMemoryError::Overflow)
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    drop((raw, other, root));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
