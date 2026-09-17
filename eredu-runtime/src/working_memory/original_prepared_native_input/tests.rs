use super::*;
use crate::input::host::{HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan};
use eredu_core::{InputModality, InputPayloadKind};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[test]
fn original_native_account_transition_has_only_one_armed_owner() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let calls = AtomicUsize::new(0);
    let dropped = Arc::new(AtomicUsize::new(0));
    let bytes =
        WorkingMemoryPool::prepared_native_input_required_bytes(&plan(&i, &pool, &calls, &dropped))
            .unwrap();
    let allowance = pool.admit_source_compiler(bytes).unwrap();
    let dormant = Account::new_unarmed(pool.clone(), bytes);
    drop(dormant);
    assert_eq!(pool.used_bytes().unwrap(), i.original_bytes() + bytes);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    let account = allowance.into_prepared_native_account();
    assert_eq!(pool.used_bytes().unwrap(), i.original_bytes() + bytes);
    drop(account);
    assert_eq!(pool.used_bytes().unwrap(), i.original_bytes());
    drop(i);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

fn source(pool: &WorkingMemoryPool) -> OriginalPreparedHostInput {
    let values = [7u32, 11, 19];
    let part = HostInputPart {
        modality: InputModality::Text,
        kind: InputPayloadKind::TokenIds,
        payload: HostTensorView {
            shape: &[1, 3],
            values: HostTensorValues::U32(&values),
        },
        metadata: &[],
        extents: &[],
    };
    pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&[part]).unwrap())
        .unwrap()
}
#[derive(Debug, thiserror::Error)]
#[error("actual mock native destination reserve failed")]
struct Failed;
struct Output {
    bytes: Vec<u8>,
    pool: WorkingMemoryPool,
    held: u64,
    dropped: Arc<AtomicUsize>,
    owner: Option<OriginalPreparedInputCustody>,
}
impl Drop for Output {
    fn drop(&mut self) {
        if let Ok(used) = self.pool.used_bytes() {
            assert!(used >= self.held);
        }
        self.bytes.clear();
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
struct Plan<'a> {
    source: &'a OriginalPreparedHostInput,
    pool: &'a WorkingMemoryPool,
    calls: &'a AtomicUsize,
    dropped: Arc<AtomicUsize>,
    escape: Option<&'a Mutex<Option<OriginalPreparedInputCustody>>>,
    fail: bool,
    poison: bool,
    unwind: bool,
}
impl PreparedNativeInputCompiler for Plan<'_> {
    type Output = Output;
    type Error = Failed;
    fn source(&self) -> &OriginalPreparedHostInput {
        self.source
    }
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        Ok(32)
    }
    fn compile(self, owner: OriginalPreparedInputCustody) -> Result<Output, (Output, Failed)> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(matches!(
            self.pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
        let held = self.pool.used_bytes().unwrap();
        let mut output = Output {
            bytes: Vec::new(),
            pool: self.pool.clone(),
            held,
            dropped: self.dropped,
            owner: Some(owner),
        };
        output.bytes.try_reserve_exact(8).unwrap();
        output.bytes.extend_from_slice(&[7, 11, 19]);
        if let Some(escape) = self.escape {
            *escape.lock().unwrap() = output.owner.take();
        }
        assert!(!self.unwind, "after actual native prefix");
        if self.poison {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = self.pool.0.usage.lock().unwrap();
                panic!("settlement poison");
            }));
        }
        if self.fail {
            assert!(output.bytes.try_reserve_exact(usize::MAX).is_err());
            Err((output, Failed))
        } else {
            Ok(output)
        }
    }
}
fn plan<'a>(
    source: &'a OriginalPreparedHostInput,
    pool: &'a WorkingMemoryPool,
    calls: &'a AtomicUsize,
    dropped: &Arc<AtomicUsize>,
) -> Plan<'a> {
    Plan {
        source,
        pool,
        calls,
        dropped: dropped.clone(),
        escape: None,
        fail: false,
        poison: false,
        unwind: false,
    }
}
#[test]
fn original_native_compare_is_exact_and_short_or_foreign_does_no_constructor_work() {
    let seed = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&seed);
    let calls = AtomicUsize::new(0);
    let dropped = Arc::new(AtomicUsize::new(0));
    let b =
        WorkingMemoryPool::prepared_native_input_required_bytes(&plan(&i, &seed, &calls, &dropped))
            .unwrap();
    let ibytes = i.original_bytes();
    drop(i);
    drop(seed);
    for short in [true, false] {
        let pool = WorkingMemoryPool::new(ibytes + b - u64::from(short), 0).unwrap();
        let i = source(&pool);
        let result = pool.compile_prepared_native_input(plan(&i, &pool, &calls, &dropped));
        if short {
            let e = result.unwrap_err();
            assert_eq!(e.retained_bytes(), 0);
            assert!(
                matches!(e.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded {required_bytes,available_bytes}) if *required_bytes==b&&*available_bytes==b-1)
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(pool.used_bytes().unwrap(), ibytes);
        } else {
            let value = result.unwrap();
            assert_eq!(value.original_bytes(), b);
            assert!(value.source().same_source(&i));
            assert_eq!(pool.used_bytes().unwrap(), ibytes + b);
            drop(pool.acquire_unquoted().unwrap());
            drop(value);
            assert_eq!(pool.used_bytes().unwrap(), ibytes);
        }
    }
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let other = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let before = calls.load(Ordering::SeqCst);
    let e = other
        .compile_prepared_native_input(plan(&i, &pool, &calls, &dropped))
        .unwrap_err();
    assert_eq!(
        e.accounting_failure(),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(calls.load(Ordering::SeqCst), before);
    assert_eq!(other.used_bytes().unwrap(), 0);
}
#[test]
fn native_callback_and_rust_source_each_hold_the_same_original_charge_until_final_owner() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let calls = AtomicUsize::new(0);
    let dropped = Arc::new(AtomicUsize::new(0));
    let escape = Mutex::new(None);
    let mut p = plan(&i, &pool, &calls, &dropped);
    p.escape = Some(&escape);
    let value = pool.compile_prepared_native_input(p).unwrap();
    let held = pool.used_bytes().unwrap();
    let owner = escape.lock().unwrap().take().unwrap();
    drop(value);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(pool.used_bytes().unwrap(), held);
    let (release, wait) = std::sync::mpsc::channel::<()>();
    let worker = std::thread::spawn(move || {
        let _ = wait.recv();
        drop(owner);
    });
    let still_held = pool.used_bytes().unwrap();
    drop(release);
    worker.join().unwrap();
    assert_eq!(still_held, held);
    assert_eq!(pool.used_bytes().unwrap(), i.original_bytes());
    drop(i);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn failed_actual_prefix_and_unwind_retire_storage_before_original_account() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let calls = AtomicUsize::new(0);
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut p = plan(&i, &pool, &calls, &dropped);
    p.fail = true;
    let error = pool.compile_prepared_native_input(p).unwrap_err();
    assert!(error.compiler_failure().is_some());
    assert!(error.retained_bytes() > 0);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(pool.used_bytes().unwrap(), i.original_bytes());
    let mut p = plan(&i, &pool, &calls, &dropped);
    p.unwind = true;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || pool.compile_prepared_native_input(p)
    ))
    .is_err());
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
    assert_eq!(pool.used_bytes().unwrap(), i.original_bytes());
}
#[test]
fn completed_native_storage_is_retained_by_poisoned_settlement_error_without_refund() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let calls = AtomicUsize::new(0);
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut p = plan(&i, &pool, &calls, &dropped);
    p.poison = true;
    let error = pool.compile_prepared_native_input(p).unwrap_err();
    assert_eq!(
        error.accounting_failure(),
        Some(&WorkingMemoryError::Poisoned)
    );
    assert!(error.retained_bytes() > 0);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert!(usage.reserved > i.original_bytes());
    assert_eq!(usage.reservations, 1);
}

#[test]
fn retired_materialization_failure_drops_native_prefix_but_keeps_original_charge() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let input = source(&pool);
    let calls = AtomicUsize::new(0);
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut prepared = plan(&input, &pool, &calls, &dropped);
    prepared.fail = true;
    let failure = pool.compile_prepared_native_input(prepared).unwrap_err();
    let bytes = failure.retained_bytes();
    let held = pool.used_bytes().unwrap();
    assert!(bytes > 0);
    let retired = failure.retire_storage();
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(retired.retained_bytes(), bytes);
    assert!(retired.source().same_source(&input));
    assert!(retired.compiler_failure().is_some());
    assert_eq!(pool.used_bytes().unwrap(), held);
    // The public source accepts this Send + Sync accounting-only failure. Its
    // ordinary test shell does not change original I/B ownership.
    let failure = eredu_core::BackendFailure::from_error(retired);
    drop(input);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
