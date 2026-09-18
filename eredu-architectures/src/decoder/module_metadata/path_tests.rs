use super::*;
use eredu_nn::workspace::*;
use std::{convert::Infallible, sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}}};

#[derive(Debug, Default)]
struct State { remaining: Mutex<usize>, calls: AtomicUsize, retired: AtomicBool, fail_at: AtomicUsize }
#[derive(Debug)]
struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let call = self.0.calls.fetch_add(1, Ordering::SeqCst);
        if call == self.0.fail_at.load(Ordering::SeqCst) {
            return Err(HostMetadataFundingError::Capacity { required: bytes as u64, available: 0 });
        }
        let mut remaining = self.0.remaining.lock().unwrap();
        *remaining = remaining.checked_sub(bytes).ok_or(HostMetadataFundingError::Capacity {
            required: bytes as u64, available: *remaining as u64,
        })?;
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) { self.0.retired.store(true, Ordering::SeqCst); }
}
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(&self, _: &WorkspaceOperation) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("funded fixture must use the finite fact interface")
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = Infallible;
    fn operation_facts(&self, _: WorkspaceOperationView<'_>) -> Result<Option<WorkspaceOperationFacts>, Infallible> { Ok(None) }
    fn write_operation_facts(&self, _: WorkspaceOperationView<'_>, _: WorkspaceEffectDestination<'_>) -> Result<Option<WorkspaceOperationFacts>, Infallible> { Ok(None) }
    fn host_facts(&self, _: WorkspaceOperationView<'_>) -> Result<Option<WorkspaceHostFacts>, Infallible> { Ok(None) }
    fn write_host_facts(&self, _: WorkspaceOperationView<'_>, _: WorkspaceHostDestination<'_>) -> Result<Option<WorkspaceHostFacts>, Infallible> { Ok(None) }
}
fn funded_context() -> (WorkspaceContext, Arc<State>) {
    let state = Arc::new(State::default());
    *state.remaining.lock().unwrap() = usize::MAX;
    state.fail_at.store(usize::MAX, Ordering::SeqCst);
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
    (context, state)
}

#[test]
fn decoder_path_destination_preserves_declarations_and_stops_at_each_reached_refusal() {
    let groups = crate::decoder::SequentialPredictionGroups::new_pattern("model.layers", 3, "mtp.layers", 2, 2).unwrap();
    for (group, unit) in [(0, 2), (1, 0), (2, 1), (0, 3), (2, 2), (3, 0)] {
        let ordinary = groups.unit_path(group, unit, None).map_err(|cause| cause.to_string());
        let (context, state) = funded_context();
        let start = state.calls.load(Ordering::SeqCst);
        let actual = groups.unit_path(group, unit, Some(&context)).map_err(|cause| cause.to_string());
        assert_eq!(actual, ordinary);
        let requests = state.calls.load(Ordering::SeqCst) - start;
        assert!(requests > 0);
        drop(context);
        assert!(state.retired.load(Ordering::SeqCst));
        for cut in 0..requests {
            let (context, state) = funded_context();
            let start = state.calls.load(Ordering::SeqCst);
            state.fail_at.store(start + cut, Ordering::SeqCst);
            let error = groups.unit_path(group, unit, Some(&context)).unwrap_err();
            assert!(matches!(error.into_metadata_funding_error(), Ok(HostMetadataFundingError::Capacity { available: 0, .. })));
            assert_eq!(state.calls.load(Ordering::SeqCst), start + cut + 1);
            drop(context);
            assert!(state.retired.load(Ordering::SeqCst));
        }
    }
    let (context, _) = funded_context();
    for group in 0..4 {
        assert_eq!(groups.unit_count(group, Some(&context)).map_err(|error| error.to_string()),
            groups.unit_count(group, None).map_err(|error| error.to_string()));
    }
    assert_eq!(ModuleMetadata::destination(Some(&context)).optional_path(Some("model.merge")).unwrap(), Some("model.merge".to_owned()));
    assert_eq!(ModuleMetadata::destination(Some(&context)).optional_path(None).unwrap(), None);
}


#[test]
fn media_contract_failure_uses_the_typed_source_and_refuses_before_new_owner() {
    use eredu_runtime::media_prefill::{MediaIngressError, PrefillIngressArchitecture};
    use std::error::Error as _;
    type Model = crate::qwen::vl::LayeredModel<WorkspaceBackend>;
    type DeviceState = eredu_runtime::DeviceState<WorkspaceBackend, eredu_runtime::working_memory::WorkspaceResidentLayerState>;
    let (context, state) = funded_context();
    let error = <Model as PrefillIngressArchitecture<WorkspaceBackend, DeviceState>>::ingress_error(
        MediaIngressError::ForeignGraph, Some(&context));
    assert!(matches!(error.source().unwrap().downcast_ref::<MediaIngressError>(), Some(MediaIngressError::ForeignGraph)));
    let before = state.calls.load(Ordering::SeqCst);
    state.fail_at.store(before, Ordering::SeqCst);
    let error = <Model as PrefillIngressArchitecture<WorkspaceBackend, DeviceState>>::ingress_error(
        MediaIngressError::ForeignGraph, Some(&context));
    assert!(matches!(error.into_metadata_funding_error(), Ok(HostMetadataFundingError::Capacity { available: 0, .. })));
    assert_eq!(state.calls.load(Ordering::SeqCst), before + 1);
    drop(context);
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
fn row_declarations_and_shared_routing_points_use_the_exact_destination() {
    use crate::decoder::prefill_observations::{ordinary_prefill_observation_declarations as rows,
        append_routed_prefill_observations};
    use eredu_runtime::{RoutedBankId, RoutedObservationPoints};
    fn produce(context: Option<&WorkspaceContext>) -> Result<(Vec<eredu_runtime::layered::PrefillObservationDeclaration>, RoutedObservationPoints), Error> {
        let metadata = ModuleMetadata::destination(context);
        let unit = metadata.text(format_args!("model.layers.0"))?;
        let mut rows = rows([Ok(unit)], true, context)?;
        let bank = RoutedObservationPoints::new(RoutedBankId::new(7), format_args!("model.layers.0.experts"), 4, context)?;
        let alias = bank.clone();
        let mut bank = bank.with_bank(RoutedBankId::new(2), format_args!("model.layers.0.secondary"), 8, context)?;
        assert_eq!(alias.iter().count(), 1);
        assert!(alias.bank(RoutedBankId::new(2)).is_none());
        append_routed_prefill_observations(&mut rows, &bank, context)?;
        // A cloned immutable source retains paths; another new bank preserves it.
        bank = bank.with_bank(RoutedBankId::new(9), format_args!("model.layers.0.tertiary"), 2, context)?;
        Ok((rows, bank))
    }
    let ordinary = produce(None).unwrap();
    let (context, state) = funded_context();
    let start = state.calls.load(Ordering::SeqCst);
    let actual = produce(Some(&context)).unwrap();
    assert_eq!(actual, ordinary);
    let requests = state.calls.load(Ordering::SeqCst) - start;
    let before_clone = state.calls.load(Ordering::SeqCst);
    let alias = actual.1.clone();
    assert_eq!(state.calls.load(Ordering::SeqCst), before_clone);
    drop(actual);
    drop(context);
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(alias);
    assert!(state.retired.load(Ordering::SeqCst));
    for cut in 0..requests {
        let (context, state) = funded_context();
        let start = state.calls.load(Ordering::SeqCst);
        state.fail_at.store(start + cut, Ordering::SeqCst);
        let error = produce(Some(&context)).unwrap_err();
        assert!(matches!(error.into_metadata_funding_error(), Ok(HostMetadataFundingError::Capacity { available: 0, .. })));
        assert_eq!(state.calls.load(Ordering::SeqCst), start + cut + 1, "cut {cut}");
        drop(context);
        assert!(state.retired.load(Ordering::SeqCst), "cut {cut}");
    }
    let (context, state) = funded_context();
    let (_, points) = produce(Some(&context)).unwrap();
    let (foreign, _) = funded_context();
    let error = points.with_bank(RoutedBankId::new(3), format_args!("foreign"), 3, Some(&foreign)).unwrap_err();
    assert!(matches!(std::error::Error::source(&error).and_then(|source| source.downcast_ref::<WorkspaceMetadataError>()), Some(WorkspaceMetadataError::Unqualified)));
    drop(context);
    assert!(state.retired.load(Ordering::SeqCst));
}
