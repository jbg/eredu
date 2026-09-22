use super::*;
use crate::workspace::{facts::owned::WorkspaceFactPreparation, *};
use std::{
    cell::Cell,
    convert::Infallible,
    rc::Rc,
    sync::{Arc, Mutex},
};

#[derive(Debug)]
struct AccountState {
    remaining: Mutex<usize>,
    retired: std::sync::atomic::AtomicBool,
}

#[derive(Debug)]
struct TestAccount(Arc<AccountState>);

impl HostMetadataAccount for TestAccount {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let mut remaining = self.0.remaining.lock().unwrap();
        *remaining = remaining
            .checked_sub(bytes)
            .ok_or(HostMetadataFundingError::Capacity {
                required: bytes.try_into().unwrap(),
                available: (*remaining).try_into().unwrap(),
            })?;
        Ok(())
    }
}

impl Drop for TestAccount {
    fn drop(&mut self) {
        self.0
            .retired
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[derive(Clone, Debug, Default)]
struct Probe {
    reads: Rc<Cell<usize>>,
    writes: Rc<Cell<usize>>,
}

impl WorkspaceMechanisms for Probe {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("funded fact emission must not use the ordinary provider")
    }
}

impl WorkspaceFactMechanisms for Probe {
    type Error = Infallible;

    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        self.reads.set(self.reads.get() + 1);
        Ok(Some(WorkspaceOperationFacts {
            layout: WorkspaceEffectLayout {
                outputs: 1,
                aliases: 0,
                assumption_bytes: 1,
            },
            scratch_bytes: 0,
        }))
    }

    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        self.writes.set(self.writes.get() + 1);
        let facts = self.operation_facts(operation)?.unwrap();
        destination.validate(facts.layout).unwrap();
        destination.outputs[0] = WorkspaceOutputEffect::Allocate(4);
        destination.assumptions[0] = b'x';
        Ok(Some(facts))
    }

    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        Ok(None)
    }

    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        panic!("absent host facts must not emit")
    }
}

#[test]
fn funded_facts_refuse_before_control_inspection_and_buffer_emission() {
    let state = Arc::new(AccountState {
        remaining: Mutex::new(usize::MAX),
        retired: std::sync::atomic::AtomicBool::new(false),
    });
    let funding = HostMetadataFunding::new(TestAccount(state.clone())).unwrap();
    let probe = Probe::default();
    let operation = WorkspaceOperation {
        kind: WorkspaceOperationKind::Initialize,
        inputs: Vec::new(),
        outputs: vec![WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap()],
    };
    let context = WorkspaceContext::new_with_metadata_funding(probe.clone(), funding).unwrap();
    let retained = context.metadata_funding().unwrap();
    let controls = context
        .facts
        .as_ref()
        .unwrap()
        .emission_control_bytes()
        .unwrap();
    let complete = WorkspaceFactPreparation::inspect(operation.as_view(), &probe)
        .unwrap()
        .charged_bytes();
    let buffers = complete - controls;
    assert!(buffers > 0);
    probe.reads.set(0);

    for (available, required, refusal_available, expected_reads, spent) in [
        (controls - 1, controls, controls - 1, 0, 0),
        (complete - 1, buffers, buffers - 1, 1, controls),
    ] {
        *state.remaining.lock().unwrap() = available;
        let before = context.identity.fact_remaining.get();
        let error = context.emit_operation_facts(&operation).err().unwrap();
        assert!(matches!(
            error.storage,
            crate::ErrorStorage::WorkspaceMetadata(WorkspaceMetadataError::Funding(
                HostMetadataFundingError::Capacity { required: actual, available: left }
            )) if actual == required as u64 && left == refusal_available as u64
        ));
        assert_eq!(probe.reads.get(), expected_reads);
        assert_eq!(probe.writes.get(), 0);
        assert_eq!(context.identity.fact_remaining.get(), before - spent);
        assert_eq!(*state.remaining.lock().unwrap(), available - spent);
    }

    *state.remaining.lock().unwrap() = complete;
    let emitted = context.emit_operation_facts(&operation).unwrap();
    assert_eq!(
        emitted.0.as_ref().unwrap().output(0),
        Some(WorkspaceOutputStorageView::Allocate(4))
    );
    assert!(emitted.1.is_none());
    assert_eq!(probe.writes.get(), 1);
    assert_eq!(*state.remaining.lock().unwrap(), 0);
    assert_eq!(context.identity.fact_bytes.get(), Some(controls + complete));

    drop(context);
    assert!(!state.retired.load(std::sync::atomic::Ordering::SeqCst));
    drop(emitted);
    drop(retained);
    assert!(state.retired.load(std::sync::atomic::Ordering::SeqCst));
}

#[test]
fn prepaid_host_partition_spends_once_and_keeps_exact_owner_custody() {
    use eredu_core::{HostMetadataFunding, HostMetadataFundingError, HostPreparationAuthority};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    #[derive(Debug)]
    struct Retained(Arc<AtomicBool>);
    impl Drop for Retained {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let retired = Arc::new(AtomicBool::new(false));
    let host = HostPreparationAuthority::retain(Retained(retired.clone()));
    let controls = HostMetadataFunding::prepaid_control_bytes().unwrap();
    assert!(HostMetadataFunding::from_prepaid(controls - 1, host.clone()).is_err());
    assert!(!retired.load(Ordering::SeqCst));
    let funding = HostMetadataFunding::from_prepaid(controls + 9, host.clone()).unwrap();
    funding.reserve_metadata(4).unwrap();
    let escaped = funding.clone();
    escaped.reserve_metadata(5).unwrap();
    assert!(
        matches!(funding.reserve_metadata(1), Err(HostMetadataFundingError::Capacity {
        required, available,
    }) if required == (controls + 10) as u64 && available == (controls + 9) as u64)
    );
    // Refusal cannot spend the last byte twice or refund an already used span.
    funding.reserve_metadata(0).unwrap();
    drop((funding, host));
    assert!(!retired.load(Ordering::SeqCst));
    drop(escaped);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn pure_range_coordinates_refuse_each_reached_metadata_producer() {
    #[derive(Debug, Default)]
    struct RangeProbe(Probe);
    impl WorkspaceMechanisms for RangeProbe {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            panic!("funded range must use its fact writer");
        }
    }
    impl WorkspaceFactMechanisms for RangeProbe {
        type Error = Infallible;
        fn operation_facts(
            &self,
            operation: WorkspaceOperationView<'_>,
        ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
            self.0.operation_facts(operation)
        }
        fn write_operation_facts(
            &self,
            operation: WorkspaceOperationView<'_>,
            destination: WorkspaceEffectDestination<'_>,
        ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
            let facts = self.operation_facts(operation)?.unwrap();
            destination.validate(facts.layout).unwrap();
            destination.outputs[0] = WorkspaceOutputEffect::AliasInput(0);
            destination.assumptions[0] = b'x';
            Ok(Some(facts))
        }
        fn host_facts(
            &self,
            _: WorkspaceOperationView<'_>,
        ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
            Ok(None)
        }
        fn write_host_facts(
            &self,
            _: WorkspaceOperationView<'_>,
            _: WorkspaceHostDestination<'_>,
        ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
            unreachable!("no host allocation")
        }
    }
    #[derive(Debug)]
    struct CutAccount(Arc<Mutex<(usize, usize)>>);
    impl HostMetadataAccount for CutAccount {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            let mut state = self.0.lock().unwrap();
            let ordinal = state.0;
            state.0 += 1;
            if ordinal == state.1 {
                Err(HostMetadataFundingError::Capacity {
                    required: bytes as u64,
                    available: 0,
                })
            } else {
                Ok(())
            }
        }
    }
    fn run(cut: usize) -> (bool, usize) {
        use crate::{Index, Tensor};
        let state = Arc::new(Mutex::new((0, usize::MAX)));
        let funding = HostMetadataFunding::new(CutAccount(state.clone())).unwrap();
        let context =
            WorkspaceContext::new_with_metadata_funding(RangeProbe::default(), funding).unwrap();
        let input = WorkspaceTensor::existing(
            context
                .layout(&[2, 3, 7, 4], WorkspaceDtype::Float32)
                .unwrap(),
            &context,
        )
        .unwrap();
        *state.lock().unwrap() = (0, cut);
        let result = input.index(
            &[Index::Full, Index::Range(1, 3), Index::Range(-5, -1)],
            &context,
        );
        let accepted = result.is_ok();
        if let Err(error) = &result {
            assert!(
                matches!(
                    error.storage,
                    crate::ErrorStorage::WorkspaceMetadata(WorkspaceMetadataError::Funding(
                        HostMetadataFundingError::Capacity { available: 0, .. }
                    ))
                ),
                "cut={cut}: {error:?}"
            );
        }
        let reached = state.lock().unwrap().0;
        drop(result);
        drop(input);
        drop(context);
        (accepted, reached)
    }
    let (accepted, reached) = run(usize::MAX);
    assert!(accepted && reached > 4);
    for cut in 0..reached {
        assert!(
            !run(cut).0,
            "reached metadata producer {cut} ignored refusal"
        );
    }
}

#[derive(Clone, Debug)]
struct AllocatingSourceProbe {
    probe: Probe,
    cells: usize,
}
impl WorkspaceMechanisms for AllocatingSourceProbe {
    fn operation_bound(&self, _: &WorkspaceOperation) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("recording source must use fact emission")
    }
}
impl WorkspaceFactMechanisms for AllocatingSourceProbe {
    type Error = Error;
    fn with_prepared_facts_context<T>(
        &self,
        _: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Error> {
        let mut source = context.metadata_vec::<u64>(self.cells)?;
        source.resize(self.cells, 17);
        let result = visit(self);
        assert!(source.iter().all(|value| *value == 17));
        Ok(result)
    }
    fn operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Error> {
        Ok(self.probe.operation_facts(op).unwrap())
    }
    fn write_operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Error> {
        Ok(self.probe.write_operation_facts(op, out).unwrap())
    }
    fn host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Error> {
        Ok(self.probe.host_facts(op).unwrap())
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Error> {
        unreachable!("no host fact")
    }
}

#[test]
fn lexical_source_construction_uses_the_recorded_caller_budget_and_refuses_one_short() {
    let operation = WorkspaceOperation {
        kind: WorkspaceOperationKind::Initialize,
        inputs: vec![],
        outputs: vec![WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap()],
    };
    let source = AllocatingSourceProbe {
        probe: Probe::default(),
        cells: 23,
    };
    let recording = WorkspaceContext::new_recording_facts(source.clone());
    recording.emit_operation_facts(&operation).unwrap();
    let metadata = recording.identity.metadata_bytes.get().unwrap();
    assert!(metadata >= 23 * std::mem::size_of::<u64>());
    let quoted = metadata
        .checked_add(recording.identity.fact_bytes.get().unwrap())
        .unwrap();
    drop(recording);
    for available in [quoted - 1, quoted] {
        source.probe.reads.set(0);
        source.probe.writes.set(0);
        let state = Arc::new(AccountState {
            remaining: Mutex::new(usize::MAX),
            retired: std::sync::atomic::AtomicBool::new(false),
        });
        let funding = HostMetadataFunding::new(TestAccount(state.clone())).unwrap();
        let context = WorkspaceContext::new_with_metadata_funding(source.clone(), funding).unwrap();
        *state.remaining.lock().unwrap() = available;
        let result = context.emit_operation_facts(&operation);
        assert_eq!(result.is_ok(), available == quoted);
        assert_eq!(source.probe.writes.get(), usize::from(available == quoted));
        if result.is_ok() {
            assert_eq!(*state.remaining.lock().unwrap(), 0);
        }
        drop(result);
        drop(context);
        assert!(state.retired.load(std::sync::atomic::Ordering::SeqCst));
    }
}
