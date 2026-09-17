//! Actual standard-thread startup refusal, joined retirement and abandonment.
use super::*;
use crate::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
use std::time::Duration;

fn units() -> Vec<OffloadUnitId> {
    vec![OffloadUnitId::new("actual").unwrap()]
}
fn success(_: &OffloadUnitId) -> Result<(), String> {
    Ok(())
}
fn plan<F>(operation: &F) -> Option<HostThreadStartupPlan>
where
    F: Fn(&OffloadUnitId) -> Result<(), String> + Send + Sync + 'static,
{
    match BackgroundPrefetchWorker::thread_startup_plan(operation, "paid-prefetch-startup") {
        Ok(plan) => Some(plan),
        Err(WorkingMemoryError::UnknownBound) => {
            assert!(std::env::var_os("EREDU_REQUIRE_STATIC_BASELINE_QUALIFICATION").is_none());
            None
        }
        Err(cause) => panic!("unexpected startup source refusal: {cause}"),
    }
}
#[test]
fn admitted_thread_refuses_before_spawn_and_refunds_after_join() {
    let operation = success as fn(&OffloadUnitId) -> Result<(), String>;
    let Some(plan) = plan(&operation) else { return };
    let bytes = plan.required_bytes();
    let execution = InferenceExecutionIdentity::default();
    let small = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    assert!(matches!(
        plan.prepare(&small, &execution, bytes - 1),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(small.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let startup = plan.prepare(&pool, &execution, bytes).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    let ids = units();
    let storage = PreparedPrefetchStorage::ordinary(1, Some(ids.clone())).unwrap();
    let worker = BackgroundPrefetchWorker::for_admitted_prepared_units_retaining(
        storage, startup, operation,
    )
    .unwrap();
    worker.submit(&ids[0], false).unwrap();
    assert!(matches!(
        worker.wait(&ids[0]).unwrap(),
        PrefetchDemandResolution::Ready
    ));
    worker.finish().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn detached_thread_never_refunds_startup_at_callback_return() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let operation_gate = gate.clone();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (retired_tx, retired_rx) = mpsc::channel();
    struct Retired(mpsc::Sender<()>);
    impl Drop for Retired {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let retired = Retired(retired_tx);
    let operation = move |_: &OffloadUnitId| {
        let _retained = &retired;
        entered_tx.send(()).unwrap();
        let mut ready = operation_gate.0.lock().unwrap();
        while !*ready {
            ready = operation_gate.1.wait(ready).unwrap();
        }
        Ok(())
    };
    let Some(plan) = plan(&operation) else { return };
    let bytes = plan.required_bytes();
    let execution = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let startup = plan.prepare(&pool, &execution, bytes).unwrap();
    let ids = units();
    let storage = PreparedPrefetchStorage::ordinary(1, Some(ids.clone())).unwrap();
    let worker = BackgroundPrefetchWorker::for_admitted_prepared_units_retaining(
        storage, startup, operation,
    )
    .unwrap()
    .with_nonblocking_drop();
    worker.submit(&ids[0], false).unwrap();
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    drop(worker);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    retired_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), bytes);
}
