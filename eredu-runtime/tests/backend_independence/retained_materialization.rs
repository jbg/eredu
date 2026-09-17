//! The actual neutral initial contract is the only witness producer.
use super::*;
type State = DeviceState<FakeBackend, FakeLayerState>;
#[test]
fn retained_materialization_reuses_exact_source_and_refuses_changed_exclusions_before_architecture() {
    let counters = ReplicatedSessionCounters::default();
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator, trace: Vec::new(), counters: counters.clone(),
        inconsistent_transport: false, inconsistent_identity: false,
    };
    let selected = selected_reference_text(&architecture, LayerWeightResidency::FullyResident);
    let identity = selected.requirements().architecture_identity().to_owned();
    let initial = eredu_runtime::prepare_layered_text_contract_with_metadata::<_, FakeBackend, State>(
        &architecture, None, selected.clone(), &identity,
        eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
        ["decoder.weight"], &(),
    ).unwrap();
    assert!(initial.materialization_tasks().is_empty());
    assert_eq!(initial.addressable_parameters(), ["decoder.weight"]);
    let source = initial.materialization_source().unwrap().clone();
    let escaped = eredu_runtime::prepare_layered_text_contract_with_materialization::<_, FakeBackend, State>(
        &architecture, None, selected.clone(), &identity,
        eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
        ["decoder.weight", "decoder.weight"], &source, &(),
    ).unwrap();
    assert!(std::ptr::eq(initial.addressable_parameters(), escaped.addressable_parameters()));
    assert!(escaped.materialization_tasks().is_empty());

    // Invalid architecture proves that identity/exclusion refusal precedes its validation.
    let inconsistent = OrdinaryTextFixture {
        static_modules: FakeOperator, trace: Vec::new(), counters: counters.clone(),
        inconsistent_transport: true, inconsistent_identity: true,
    };
    for (declarations, message) in [
        (Vec::<&str>::new(), "complete declared addressable set"),
        (vec!["decoder.weight", "foreign.weight"], "does not declare addressable parameter"),
    ] {
        let error = eredu_runtime::prepare_layered_text_contract_with_materialization::<_, FakeBackend, State>(
            &inconsistent, None, selected.clone(), &identity,
            eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
            declarations, &source, &(),
        ).err().expect("changed exclusion set was accepted");
        assert!(error.to_string().contains(message), "{error}");
    }
    let foreign = selected_reference_text(&architecture, LayerWeightResidency::FullyResident);
    let error = eredu_runtime::prepare_layered_text_contract_with_materialization::<_, FakeBackend, State>(
        &inconsistent, None, foreign, &identity,
        eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
        ["decoder.weight"], &source, &(),
    ).err().expect("semantically equal foreign selection reused exact source");
    assert!(error.to_string().contains("different selected realization"), "{error}");
    assert_eq!(counters.snapshot(), ReplicatedSessionCounts::default());
    drop((initial, source, selected, architecture, inconsistent));
    assert!(escaped.materialization_tasks().is_empty());
    assert_eq!(escaped.addressable_parameters(), ["decoder.weight"]);
}
