//! Actual mask refusal and copied-destination custody through scoped frames.
use super::*;
use crate::memory_fixture::{LedgerFixture as _, StorageFixture as _};
use eredu_core::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::MemoryLedger;
use std::sync::atomic::AtomicBool;

#[derive(Clone, Debug)]
struct Counts {
    calls: Arc<AtomicUsize>,
    cut: Arc<AtomicUsize>,
    spent: Arc<AtomicUsize>,
    retired: Arc<AtomicBool>,
}
#[derive(Debug)]
struct Account(Counts);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let call = self.0.calls.fetch_add(1, Ordering::SeqCst);
        let current = self.0.spent.load(Ordering::SeqCst);
        let available = (64usize << 20).saturating_sub(current);
        if call >= self.0.cut.load(Ordering::SeqCst) || bytes > available {
            return Err(HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            });
        }
        self.0.spent.fetch_add(bytes, Ordering::SeqCst);
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
fn account() -> (HostMetadataFunding, Counts) {
    let counts = Counts {
        calls: Arc::new(AtomicUsize::new(0)),
        cut: Arc::new(AtomicUsize::new(usize::MAX)),
        spent: Arc::new(AtomicUsize::new(0)),
        retired: Arc::new(AtomicBool::new(false)),
    };
    (
        HostMetadataFunding::new(Account(counts.clone())).unwrap(),
        counts,
    )
}
#[test]
fn scoped_mask_refusals_keep_exact_destination_payer_and_failed_prefix() {
    let (tokenizer, eos) = tokenizer();
    let pool = crate::memory_fixture::host_ledger(1 << 28, 0).unwrap();
    let (plan, receipt) = original_plan(
        &pool,
        &tokenizer,
        &eos,
        &ordinary_tools(),
        ToolChoice::Required,
    );
    let (historical, source_counts) = account();
    let source = plan
        .generation_constraint()
        .inner
        .original_grammar_state(&receipt, &historical)
        .unwrap();
    let historical_calls = source_counts.calls.load(Ordering::SeqCst);
    source_counts.cut.store(historical_calls, Ordering::SeqCst);

    let (destination, counts) = account();
    let copied = source.try_copy(&destination).unwrap();
    let setup = counts.calls.load(Ordering::SeqCst);
    let copied = copied.compute_mask().unwrap();
    let expected = copied.token_mask().unwrap().clone();
    let mask_calls = counts.calls.load(Ordering::SeqCst) - setup;
    eprintln!("SCOPED_FRAME_MASK reached_requests={mask_calls}");
    assert!(
        mask_calls > 2 && mask_calls < 10_000,
        "finite reached mask trace: {mask_calls}"
    );
    assert_eq!(source_counts.calls.load(Ordering::SeqCst), historical_calls);
    drop(destination);
    assert!(!counts.retired.load(Ordering::SeqCst));
    drop(copied);
    assert!(counts.retired.load(Ordering::SeqCst));

    for cut in 0..mask_calls {
        let (destination, counts) = account();
        let copied = source.try_copy(&destination).unwrap();
        let setup = counts.calls.load(Ordering::SeqCst);
        counts.cut.store(setup + cut, Ordering::SeqCst);
        let failure = copied.compute_mask().unwrap_err();
        assert_eq!(counts.calls.load(Ordering::SeqCst), setup + cut + 1);
        let mut cause: &(dyn std::error::Error + 'static) = &failure;
        loop {
            if let Some(HostMetadataFundingError::Capacity { available: 0, .. }) =
                cause.downcast_ref::<HostMetadataFundingError>()
            {
                break;
            }
            cause = cause.source().unwrap_or_else(|| {
                panic!("cut {cut}: missing typed funding leaf at {cause:?}: {failure}")
            });
        }
        assert_eq!(source_counts.calls.load(Ordering::SeqCst), historical_calls);
        drop(destination);
        assert!(!counts.retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(counts.retired.load(Ordering::SeqCst));
    }
    let (destination, counts) = account();
    let copied = source
        .try_copy(&destination)
        .unwrap()
        .compute_mask()
        .unwrap();
    assert_eq!(copied.token_mask().unwrap(), &expected);
    assert_eq!(source_counts.calls.load(Ordering::SeqCst), historical_calls);
    drop((historical, source));
    assert!(!source_counts.retired.load(Ordering::SeqCst));
    drop(destination);
    assert!(!counts.retired.load(Ordering::SeqCst));
    drop(copied);
    assert!(counts.retired.load(Ordering::SeqCst));
    assert!(source_counts.retired.load(Ordering::SeqCst));
    drop((plan, receipt, tokenizer));
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
}
