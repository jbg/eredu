use super::*;
use eredu_nn::{Error, workspace::*};
use eredu_runtime::ArchitectureParameters;
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

#[derive(Debug, Default)]
struct Counter {
    calls: usize,
    limit: Option<usize>,
    bytes: usize,
    refused: bool,
}
#[derive(Debug)]
struct Account(Arc<Mutex<Counter>>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let mut state = self.0.lock().unwrap();
        assert!(!state.refused, "producer called payer after first refusal");
        state.calls += 1;
        if state.limit == Some(state.calls - 1) {
            state.refused = true;
            return Err(HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            });
        }
        state.bytes = state.bytes.checked_add(bytes).unwrap();
        Ok(())
    }
}
#[derive(Debug)]
struct SetupFacts;
impl WorkspaceMechanisms for SetupFacts {
    fn operation_bound(&self, _: &WorkspaceOperation) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
#[derive(Debug)]
struct NoTensors;
impl WorkspaceMechanisms for NoTensors {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("state and schema declarations must not execute tensor equations")
    }
}
impl WorkspaceFactMechanisms for NoTensors {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        panic!("no equations")
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        panic!("no equations")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        panic!("no equations")
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        panic!("no equations")
    }
}
fn funded(limit: Option<usize>) -> (WorkspaceContext, Arc<Mutex<Counter>>) {
    let counter = Arc::new(Mutex::new(Counter::default()));
    let funding = HostMetadataFunding::new(Account(counter.clone())).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(NoTensors, funding).unwrap();
    *counter.lock().unwrap() = Counter {
        limit,
        ..Counter::default()
    };
    (context, counter)
}
fn refusal(error: &Error) {
    let mut current: &(dyn std::error::Error + 'static) = error;
    loop {
        if matches!(
            current.downcast_ref::<HostMetadataFundingError>(),
            Some(HostMetadataFundingError::Capacity { available: 0, .. })
        ) {
            return;
        }
        current = current
            .source()
            .unwrap_or_else(|| panic!("lost original funding refusal: {error:?}"));
    }
}
fn every_cut<T: std::fmt::Debug + PartialEq>(
    expected: &T,
    mut produce: impl FnMut(&WorkspaceContext) -> Result<T, Error>,
) {
    let (context, counter) = funded(None);
    assert_eq!(&produce(&context).unwrap(), expected);
    let calls = counter.lock().unwrap().calls;
    assert!(calls > 0);
    assert!(counter.lock().unwrap().bytes > 0);
    assert_eq!(context.operation_count(), 0);
    for cut in 0..calls {
        let (context, counter) = funded(Some(cut));
        let error = produce(&context).expect_err("reached refusal was ignored");
        refusal(&error);
        let state = counter.lock().unwrap();
        assert_eq!(state.calls, cut + 1);
        assert!(state.refused);
        assert_eq!(context.operation_count(), 0);
    }
}

#[test]
fn moshi_paid_state_and_identity_match_both_segments_and_stop_at_every_refusal() {
    let config = tests::tiny_config();
    let ordinary = WorkspaceContext::new(SetupFacts);
    let model = LayeredModel::<WorkspaceBackend>::new(config.clone(), &ordinary).unwrap();
    let layout = model.state_layout(None).unwrap();
    assert_eq!(layout, super::super::model::state_layout(&config).unwrap());
    assert_eq!(layout.segments().len(), 2);
    assert_eq!(
        layout.segments()[0].lifetime(),
        StateSegmentLifetime::Persistent
    );
    assert_eq!(
        layout.segments()[1].lifetime(),
        StateSegmentLifetime::FrameLocal
    );
    every_cut(&layout, |context| model.state_layout(Some(context)));
    let state = eredu_runtime::PartitionState::new(layout.clone(), 0).unwrap();
    let topology = eredu_core::cache::PromptCacheTopology::new(None, None, None, false).unwrap();
    let expected = model
        .state_identity(&state, topology.clone(), None)
        .unwrap();
    every_cut(&expected, |context| {
        model.state_identity(&state, topology.clone(), Some(context))
    });
    let local = local_geometry(&config, &tests::local_layout(&config), std::iter::empty()).unwrap();
    let local_layout = local.state_layout().clone();
    let parallel =
        LayeredModel::<WorkspaceBackend>::new_parallel(config, local, &ordinary).unwrap();
    every_cut(&local_layout, |context| {
        parallel.state_layout(Some(context))
    });
    let local_state = eredu_runtime::PartitionState::new(local_layout, 0).unwrap();
    let local_identity = parallel
        .state_identity(&local_state, topology.clone(), None)
        .unwrap();
    every_cut(&local_identity, |context| {
        parallel.state_identity(&local_state, topology.clone(), Some(context))
    });
    let invalid = eredu_runtime::PartitionState::new(layout, 1).unwrap();
    let (context, counter) = funded(None);
    assert!(
        model
            .state_identity(&invalid, topology, Some(&context))
            .is_err()
    );
    assert!(counter.lock().unwrap().bytes > 0);
}

#[test]
fn moshi_paid_dense_and_affine_descriptions_match_contract_and_stop_at_every_refusal() {
    for quantization in [
        None,
        Some(eredu_checkpoint::WeightQuantization::Affine(
            eredu_checkpoint::AffineQuantization::new(16, 4).unwrap(),
        )),
    ] {
        let config = tests::tiny_config()
            .with_native_quantization(quantization)
            .unwrap();
        let expected = parameter_contract(&config).unwrap().into_description();
        assert_eq!(parameter_description(&config).unwrap(), expected);
        let ordinary = WorkspaceContext::new(SetupFacts);
        let model = LayeredModel::<WorkspaceBackend>::new(config, &ordinary).unwrap();
        assert_eq!(*model.parameter_description(&ordinary).unwrap(), expected);
        every_cut(&expected, |context| {
            model
                .parameter_description(context)
                .map(std::borrow::Cow::into_owned)
        });
    }
}
