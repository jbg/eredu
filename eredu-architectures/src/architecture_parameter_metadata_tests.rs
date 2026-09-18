//! Behavioral witnesses for the canonical family declaration destination.
use eredu_nn::{Error, workspace::*};
use eredu_runtime::{ArchitectureParameterDescription, ArchitectureParameters, PartitionState};
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        unreachable!()
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
        unreachable!()
    }
}
#[derive(Debug)]
struct Account(Arc<AtomicUsize>, usize);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let call = self.0.fetch_add(1, Ordering::SeqCst);
        if call == self.1 {
            Err(HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            })
        } else {
            Ok(())
        }
    }
}
fn context(cut: usize) -> (WorkspaceContext, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let funding = HostMetadataFunding::new(Account(calls.clone(), cut)).unwrap();
    (
        WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap(),
        calls,
    )
}
pub(crate) fn exercise<A>(build: impl FnOnce(&WorkspaceContext) -> Result<A, Error>)
where
    A: ArchitectureParameters<WorkspaceBackend, DefinitionError = Error>
        + eredu_runtime::LayeredArchitecture<
            WorkspaceBackend,
            eredu_runtime::DeviceState<
                WorkspaceBackend, eredu_runtime::working_memory::WorkspaceResidentLayerState,
            >, Error = Error,
        >,
{
    let ordinary = WorkspaceContext::new(Facts);
    let model = build(&ordinary).unwrap();
    // Exercise the actual family producer, including its routed-bank schedule,
    // through the same optional destination used during session binding.
    for media in [false, true] {
        let rows = |metadata: Option<&WorkspaceContext>| {
            if media {
                model.media_prefill_observation_declarations(metadata)
            } else {
                model.prefill_observation_declarations(metadata)
            }
        };
        if !rows(None).unwrap().is_empty() {
            boundary(rows);
        }
    }
    let expected = model.parameter_description(&ordinary).unwrap();
    let layout = model.state_layout(None).unwrap();
    let state = PartitionState::new(layout.clone(), 0).unwrap();
    let identity = model
        .state_identity(&state, Default::default(), None)
        .unwrap();
    let (paid, calls) = context(usize::MAX);
    let checkpoint = calls.load(Ordering::SeqCst);
    assert_eq!(model.state_layout(Some(&paid)).unwrap(), layout);
    assert_eq!(
        model
            .state_identity(&state, Default::default(), Some(&paid))
            .unwrap(),
        identity
    );
    let actual = model.parameter_description(&paid).unwrap();
    assert_eq!(actual.as_ref(), expected.as_ref());
    // Owning consumers copy the already validated exact source through the same account.
    let copied = ArchitectureParameterDescription::into_owned(
        std::borrow::Cow::Borrowed(&actual),
        Some(&paid),
    )
    .unwrap();
    assert_eq!(&copied, expected.as_ref());
    assert!(calls.load(Ordering::SeqCst) > checkpoint);
    drop((copied, actual, paid));
    // The first refusal, an interior producer, and the final reached producer all
    // stop at that exact callback. Full format expansion has a separate every-cut test.
    let (probe, probe_calls) = context(usize::MAX);
    let start = probe_calls.load(Ordering::SeqCst);
    model.parameter_description(&probe).unwrap();
    let total = probe_calls.load(Ordering::SeqCst) - start;
    assert!(total > 8);
    for ordinal in [0, total / 2, total - 1] {
        let (paid, calls) = context(start + ordinal);
        let result = model.parameter_description(&paid);
        assert!(result.is_err(), "cut {ordinal}/{total}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            start + ordinal + 1,
            "no later producer after refusal"
        );
    }
}

pub(crate) fn boundary<T: std::fmt::Debug + PartialEq>(
    operation: impl Fn(Option<&WorkspaceContext>) -> Result<T, Error>,
) {
    let expected = operation(None).unwrap();
    let (paid, calls) = context(usize::MAX);
    let start = calls.load(Ordering::SeqCst);
    assert_eq!(operation(Some(&paid)).unwrap(), expected);
    let total = calls.load(Ordering::SeqCst) - start;
    assert!(total > 0);
    assert_eq!(paid.operation_count(), 0);
    for ordinal in 0..total {
        let (paid, calls) = context(start + ordinal);
        let error = operation(Some(&paid)).expect_err("every reached producer is funded");
        assert!(matches!(
            error.into_metadata_funding_error(),
            Ok(HostMetadataFundingError::Capacity { .. })
        ));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            start + ordinal + 1,
            "no producer after refusal"
        );
        assert_eq!(paid.operation_count(), 0);
    }
}
