//! The actual stamped activation transaction consumes selected scalar coverage.
use super::*;
use crate::working_memory::{
    NativeStorageObservation, NativeStorageRegistration, NativeStorageSelection,
    OriginalNativeBudgetCustody, OriginalNativeStorageMechanism, PreparedNativeStoragePlan,
};
#[derive(Clone)]
struct CoverageOnly(NativeStorageSelection);
impl OriginalNativeStorageMechanism for CoverageOnly {
    type Key = u32;
    type Budget = std::sync::Arc<OriginalNativeBudgetCustody>;
    type Root = ();
    type Attachment = NativeStorageRegistration<u32>;
    type Error = WorkingMemoryError;
    type Observation<'a> = ();
    fn selection(&self) -> &NativeStorageSelection {
        &self.0
    }
    fn create_budget(&self, _: OriginalNativeBudgetCustody) -> Result<Self::Budget, Self::Error> {
        unreachable!("this test issues the accounting partition, not a native counter")
    }
    fn observe<'a>(&'a self, _: &'a Self::Budget, _: &'a ()) -> Result<(), Self::Error> {
        unreachable!()
    }
    fn describe(_: &()) -> NativeStorageObservation<u32> {
        unreachable!()
    }
    fn prepare_attachment(&self, _: Self::Attachment) -> Result<Self::Attachment, Self::Error> {
        unreachable!()
    }
    fn registration(owner: &Self::Attachment) -> &Self::Attachment {
        owner
    }
    fn attach(_: (), _: Self::Attachment) -> Result<(), (Self::Error, Self::Attachment)> {
        unreachable!()
    }
}

#[test]
fn actual_span_subtracts_only_selected_native_coverage_and_checks_exact_partition_under_usage() {
    // Missing, foreign-sized, exact remainder, and one-byte-short remainder.
    for case in 0..4 {
        let pool = WorkingMemoryPool::new(4_000_000, 0).unwrap();
        let source = source();
        let opening_root = pool.register_storage([(73u32, 8)]).unwrap();
        let original = quote(&pool, plan(&source).initialization_peak_bytes());
        let pins =
            PreparedPrefillStoragePinPlan::<u32>::prepare(original.span_workspace().plan(), |_| {
                Some(1)
            })
            .unwrap();
        let selection = NativeStorageSelection::default();
        let native = PreparedNativeStoragePlan::<CoverageOnly>::prepare(
            original.span_workspace(),
            &selection,
            Some(64),
            Some((0, 0)),
            original
                .span_workspace()
                .plan()
                .records()
                .iter()
                .map(|_| Some(8)),
            Some(0),
            Some(0), // This private fixture constructs no native counter/sidecar.
        )
        .unwrap();
        let controls = PreparedTextControlWorkspace::prepare(
            &source,
            geometry(),
            original.span_workspace().plan(),
            TextHostControlFacts::new(Some(11), Some(17), Some(23)),
        )
        .unwrap()
        .with_prefill_storage_pins(pins)
        .unwrap()
        .with_native_storage(native)
        .unwrap();
        let (r, run, q) = reserve_sealed(
            &pool,
            original
                .with_span_workspace_and_text_controls(controls)
                .unwrap(),
        );
        let (mut owner, _) = q
            .into_funded_text_span_workspace(&run, r.memory_reservation().unwrap())
            .unwrap();
        let native_bank = if case >= 2 {
            owner
                .take_native_storage_bank::<CoverageOnly>(&run, &selection)
                .unwrap()
        } else {
            None
        };
        // Existing private ledger realization simulates a contradictory cap;
        // there is no public arbitrary-byte grant and no native object here.
        let wrong_partition = if case == 1 {
            Some(
                run.take_native_partition(
                    crate::working_memory::funding::native_partition::test_receipt(&run, 63),
                )
                .unwrap(),
            )
        } else {
            None
        };
        let mut pins = owner.take_prefill_storage_pins::<u32>().unwrap();
        let bank = bank(&run, &r, &source);
        let mut native = run.scope().unwrap();
        let c = chunk();
        let cx = context(&r, &c, 83);
        let (mut segment, reg) = bank
            .prefill_source_bootstrap()
            .unwrap()
            .begin_segment(&mut native, &cx)
            .unwrap();
        let mut pin = pins.begin(&cx, &native, &segment).unwrap();
        pin.push_owned(73u32, 8).unwrap();
        let mut opening = Some(pin.pin_registered(&native, &segment).unwrap());
        segment
            .install_opening_group(&mut native, &cx, &mut opening)
            .unwrap();
        let receipt = owner.as_reserved_text_span_workspace();
        assert_eq!(
            receipt.span_bytes(&InferenceWorkspaceSpan::Prefill(c.clone())),
            Some(16)
        );
        let pressure = if case >= 2 {
            let free = {
                let u = pool.0.usage.lock().unwrap();
                u.funding[&r.memory_reservation().unwrap().0.funding.unwrap()]
                    .spendable_remaining()
                    .unwrap()
            };
            Some(
                native
                    .adopt_storage_individually([(71u32, free - 8 + u64::from(case == 3))])
                    .unwrap(),
            )
        } else {
            None
        };
        let before = ledger(&pool);
        let result = segment.activate_reserved_text_span(&mut native, &receipt, &cx);
        if case < 2 {
            assert!(matches!(result, Err(WorkingMemoryError::IdentityMismatch)));
            assert_eq!(ledger(&pool), before);
            run.scope().unwrap().certify().unwrap(); // failed activation installed no marker
        } else if case == 3 {
            assert!(matches!(
                result,
                Err(WorkingMemoryError::BudgetExceeded {
                    required_bytes: 8,
                    available_bytes: 7
                })
            ));
            assert_eq!(ledger(&pool), before);
            drop(pressure);
            segment
                .activate_reserved_text_span(&mut native, &receipt, &cx)
                .unwrap();
            fenced(run.scope());
        } else {
            result.unwrap();
            assert_eq!(ledger(&pool), before);
            fenced(run.scope());
            drop(pressure);
        }
        let settled = ticket(reg, &cx);
        let parcel = segment.take_settled_sources(&mut native, &settled).unwrap();
        drop((parcel, settled, segment, receipt));
        native.certify().unwrap();
        drop((
            native_bank,
            wrong_partition,
            bank,
            pins,
            owner,
            r,
            run,
            source,
            opening_root,
        ));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
