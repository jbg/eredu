use super::*;
use crate::backend::runtime::residency::storage::RetainedStorage;
use std::collections::BTreeMap;

fn scaled_artifact() -> tempfile::TempDir {
    let root = tiny_artifact("llama", false);
    let path = root.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["rope_scaling"] = serde_json::json!({
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 32
    });
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    root
}

fn load_scaled(
    root: &Path,
    residency: eredu_runtime::WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> crate::backend::MlxModel {
    let inspection = eredu_architectures::configuration::inspect_artifact(root).unwrap();
    let options = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
    );
    let plan = eredu_core::plan_model_preparation(
        inspection,
        options.normalized().preparation_policy().unwrap(),
        eredu_core::SessionCapabilities::default(),
    )
    .unwrap();
    materialize_model_plan(plan, options, stream, weights_stream).unwrap()
}

#[derive(Default)]
struct EditableValues {
    metadata: Vec<eredu_nn::ParameterMetadata>,
    values: Vec<MlxTensor>,
}

impl eredu_nn::ParameterSlotVisitor<MlxTensor> for EditableValues {
    fn visit_slot(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &mut MlxTensor) {
        self.metadata.push(metadata.to_owned());
        self.values.push(value.clone());
    }
}

fn editable_values(target: &mut dyn ErasedReplicatedTextExecutable) -> EditableValues {
    let mut values = EditableValues::default();
    assert!(target.visit_loaded_parameters(&mut values));
    values
        .metadata
        .sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
    values
}

fn parameter_storage(values: &EditableValues) -> RetainedStorage {
    let mut storage = RetainedStorage::default();
    for value in &values.values {
        storage.include_array(value.as_array()).unwrap();
    }
    storage
}

fn reads(target: &dyn ErasedReplicatedTextExecutable) -> u64 {
    target
        .residency_report()
        .unwrap()
        .unwrap()
        .weight_store()
        .physical_reads
}

fn check_nonzero(output: &Array) {
    let output = output.evaluated().unwrap();
    let values = output.as_slice::<f32>();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > 1e-12));
}

fn prefill(target: &mut dyn ErasedReplicatedTextExecutable, stream: &Stream) {
    let tokens = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
    let parts = [input::token_ids_part(&tokens).unwrap()];
    check_nonzero(
        &target
            .prefill(input::ModelInput::new(&parts), stream)
            .unwrap(),
    );
}

fn inventory_facts(
    storage: &RetainedStorage,
) -> (
    Option<u64>,
    BTreeMap<safemlx::AllocationIdentity, u64>,
    usize,
) {
    (
        storage.byte_bound().unwrap(),
        storage.array_allocation_facts(),
        storage.unknown_arrays().len(),
    )
}

fn cold_inventory(
    target: &dyn ErasedReplicatedTextExecutable,
    frequencies: &Array,
) -> RetainedStorage {
    let frontier = target.state_snapshot();
    let physical_reads = reads(target);
    let helper = frequencies.allocation_info().unwrap();
    let inventory = target.retained_target_module_storage().unwrap();
    let repeated = target.retained_target_module_storage().unwrap();
    assert_eq!(inventory_facts(&inventory), inventory_facts(&repeated));
    assert_eq!(frequencies.allocation_info().unwrap(), helper);
    assert_eq!(target.state_snapshot(), frontier);
    assert_eq!(reads(target), physical_reads);
    inventory
}

#[test]
fn resident_scaled_rotary_inventory_reaches_helpers_without_making_them_parameters() {
    let (stream, weights_stream) = execution_streams();
    let root = scaled_artifact();
    let model = load_scaled(
        root.path(),
        eredu_runtime::WeightResidency::fully_resident(),
        &stream,
        &weights_stream,
    );
    let mut executable = model.into_executable();
    let target = executable.erased_mut();
    let initial_frontier = target.state_snapshot();
    let initial_reads = reads(target);
    let editable_before = editable_values(target);
    let parameters_before = parameter_storage(&editable_before);
    let parameter_bytes = parameters_before.byte_bound().unwrap().unwrap();
    let parameter_facts = parameters_before.array_allocation_facts();
    assert!(parameter_bytes > 0);

    // Scaled RoPE constructs a lazy numerical root during resident loading.
    // An inventory must retain that root and report unknown physical backing,
    // while all checkpoint parameters already have completed backing.
    let cold = target.retained_target_module_storage().unwrap();
    assert_eq!(cold.byte_bound().unwrap(), None);
    assert_eq!(cold.array_allocation_facts(), parameter_facts);
    assert_eq!(cold.unknown_arrays().len(), 1);
    let frequencies = &cold.unknown_arrays()[0];
    assert_eq!(frequencies.shape(), &[4]);
    assert_eq!(frequencies.allocation_info().unwrap(), None);
    assert_eq!(
        target
            .retained_target_module_storage()
            .unwrap()
            .byte_bound()
            .unwrap(),
        None
    );
    assert_eq!(target.state_snapshot(), initial_frontier);
    assert_eq!(reads(target), initial_reads);
    assert_eq!(editable_values(target).metadata, editable_before.metadata);

    prefill(target, &stream);
    assert!(target
        .state_snapshot()
        .iter()
        .all(|(position, _)| *position == 3));
    // Execution may certify the helper descriptor or leave its completion
    // event attached. Inventory preserves either status without polling it.
    let after_prefill = cold_inventory(target, frequencies);
    let prefill_facts = inventory_facts(&after_prefill);

    check_nonzero(
        &target
            .decode(&Array::from_slice(&[4_u32], &[1, 1]), &stream)
            .unwrap(),
    );
    let frontier = target.state_snapshot();
    assert!(frontier.iter().all(|(position, _)| *position == 4));
    let completed_reads = reads(target);
    let editable_after = editable_values(target);
    let parameters_after = parameter_storage(&editable_after);
    let before_validation = cold_inventory(target, frequencies);
    let before_validation_facts = inventory_facts(&before_validation);
    assert_eq!(editable_after.metadata, editable_before.metadata);
    assert_eq!(parameters_after.array_allocation_facts(), parameter_facts);
    assert_eq!(target.state_snapshot(), frontier);
    assert_eq!(reads(target), completed_reads);

    // Validate the actual retained numerical root independently of inventory.
    // For this fixture, wavelengths 2*pi*[1,10,100,1000] leave the first
    // denominator unchanged and scale the other three by eight. Reading this
    // root explicitly certifies its own descriptor if execution has not already
    // done so; no inventory call changes that descriptor's completion status.
    {
        let evaluated = frequencies.evaluated().unwrap();
        for (&actual, expected) in evaluated
            .as_slice::<f32>()
            .iter()
            .zip([1.0_f32, 80.0, 800.0, 8000.0])
        {
            assert!((actual - expected).abs() <= expected * 1e-5);
        }
    }
    let helper = frequencies.allocation_info().unwrap().unwrap();
    assert_eq!(cold.byte_bound().unwrap(), None);
    assert_eq!(inventory_facts(&after_prefill), prefill_facts);
    assert_eq!(inventory_facts(&before_validation), before_validation_facts);

    let certified = target.retained_target_module_storage().unwrap();
    let certified_bytes = certified.byte_bound().unwrap().unwrap();
    let certified_facts = certified.array_allocation_facts();
    let helper_facts = certified_facts
        .iter()
        .filter(|(identity, _)| !parameter_facts.contains_key(identity))
        .map(|(identity, bytes)| (*identity, *bytes))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        helper_facts.len(),
        1,
        "one scaled frequency root in the one-layer fixture"
    );
    assert!(helper_facts.values().all(|bytes| *bytes > 0));
    assert_eq!(
        helper_facts,
        BTreeMap::from([(helper.identity(), u64::try_from(helper.bytes()).unwrap())])
    );
    assert_eq!(
        certified_bytes,
        parameter_bytes + helper_facts.values().sum::<u64>()
    );
    for (identity, bytes) in &parameter_facts {
        assert_eq!(certified_facts.get(identity), Some(bytes));
    }
    // Every earlier snapshot keeps exactly the facts it captured. A known
    // snapshot includes the same helper allocation; an unknown one retains
    // the helper root without retroactively publishing its capacity.
    for (bound, facts, unknown_count) in [&prefill_facts, &before_validation_facts] {
        match bound {
            Some(bytes) => {
                assert_eq!(*bytes, certified_bytes);
                assert_eq!(*facts, certified_facts);
                assert_eq!(*unknown_count, 0);
            }
            None => {
                assert_eq!(*facts, parameter_facts);
                assert_eq!(*unknown_count, 1);
            }
        }
    }

    let mut after_decode = target.retained_target_module_storage().unwrap();
    assert_eq!(after_decode.array_allocation_facts(), certified_facts);
    assert_eq!(after_decode.byte_bound().unwrap(), Some(certified_bytes));
    // Keep both inventories alive while comparing identities. Merging their
    // shared backing must count both parameters and helper storage exactly once.
    after_decode.merge(certified).unwrap();
    after_decode.merge(parameters_after).unwrap();
    assert_eq!(after_decode.byte_bound().unwrap(), Some(certified_bytes));
    assert_eq!(target.state_snapshot(), frontier);
    assert_eq!(reads(target), completed_reads);
    assert_eq!(
        completed_reads, initial_reads,
        "resident execution does not reread weights"
    );
    drop(executable);
    assert_eq!(after_decode.array_allocation_facts(), certified_facts);
    assert_eq!(after_decode.byte_bound().unwrap(), Some(certified_bytes));
}

#[test]
fn bounded_scaled_rotary_inventory_does_not_keep_unloaded_unit_helpers() {
    let (stream, weights_stream) = execution_streams();
    let host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
    );
    for residency in [
        eredu_runtime::WeightResidency::layerwise_host(host),
        eredu_runtime::WeightResidency::dense_disk_stream(
            eredu_runtime::DenseDiskStreamLoadOptions::default(),
        ),
    ] {
        let root = scaled_artifact();
        let model = load_scaled(root.path(), residency, &stream, &weights_stream);
        let mut executable = model.into_executable();
        let target = executable.erased_mut();
        let topology = target
            .prepared_parameter_slots()
            .iter()
            .map(|slot| slot.parameter.id.as_str().to_owned())
            .collect::<Vec<_>>();
        let before = target.retained_target_module_storage().unwrap();
        let before_bytes = before.byte_bound().unwrap().unwrap();
        let before_facts = before.array_allocation_facts();
        let initial_frontier = target.state_snapshot();
        let initial_reads = reads(target);
        assert_eq!(
            target
                .retained_target_module_storage()
                .unwrap()
                .array_allocation_facts(),
            before_facts
        );
        assert_eq!(target.state_snapshot(), initial_frontier);
        assert_eq!(reads(target), initial_reads);

        prefill(target, &stream);
        check_nonzero(
            &target
                .decode(&Array::from_slice(&[4_u32], &[1, 1]), &stream)
                .unwrap(),
        );
        let frontier = target.state_snapshot();
        assert!(frontier.iter().all(|(position, _)| *position == 4));
        let completed_reads = reads(target);
        let mut after = target.retained_target_module_storage().unwrap();
        assert_eq!(after.byte_bound().unwrap(), Some(before_bytes));
        assert_eq!(
            after.array_allocation_facts(),
            before_facts,
            "temporary unit frequency roots retire with unloaded bounded units"
        );
        after.merge(before).unwrap();
        assert_eq!(after.byte_bound().unwrap(), Some(before_bytes));
        assert_eq!(
            target
                .prepared_parameter_slots()
                .iter()
                .map(|slot| slot.parameter.id.as_str().to_owned())
                .collect::<Vec<_>>(),
            topology
        );
        assert_eq!(target.state_snapshot(), frontier);
        assert_eq!(reads(target), completed_reads);
    }
}
