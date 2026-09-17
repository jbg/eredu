//! Selected host and disk populations share the same cold architecture equations.

use super::*;

#[path = "layerwise/existing_sampling.rs"]
mod existing_sampling;
use eredu_architectures::{
    prepared_execution::{PreparedTextGenerationWorkspace, WorkspaceLayerwiseParameters},
    prepared_sources::PreparedModelSources,
};
use eredu_checkpoint::{recipe::RecipeDtype, StoredDtype};
use eredu_nn::ParameterId;
use eredu_runtime::{ExecutionUnitAddress, ExecutionUnitLayout};

pub(super) struct SelectedParameters {
    layout: ExecutionUnitLayout,
    units: Vec<Vec<(ParameterId, WorkspaceLayout, u64)>>,
    calls: RefCell<Vec<(usize, ExecutionUnitAddress)>>,
}

impl SelectedParameters {
    pub(super) fn new(sources: &PreparedModelSources) -> Self {
        let selected = sources.selected().text_realization();
        let layout = selected.requirements().execution_units().clone();
        let tasks = selected.materialization_tasks();
        let partition =
            eredu_runtime::plan_replicated_text_materialization_tasks(tasks, &layout).unwrap();
        let units = partition
            .unit_tasks(tasks)
            .unwrap()
            .into_iter()
            .map(|tasks| {
                assert!(!tasks.is_empty());
                tasks
                    .into_iter()
                    .map(|task| {
                        // These fixtures preserve dense F32 checkpoint storage.
                        // Derived expert stacks use their admitted recipe output,
                        // without executing the recipe or reading its sources.
                        assert!(task.output_companions().is_empty());
                        let (shape, bytes) = if let Some(output) = task.derived_output() {
                            assert_eq!(output.dtype, RecipeDtype::F32);
                            (&output.shape, output.byte_len)
                        } else {
                            assert_eq!(task.sources().len(), 1);
                            let source = &sources.source_metadata()[&task.sources()[0]];
                            assert_eq!(source.stored_dtype, StoredDtype::F32);
                            (&source.logical_shape, source.encoded_byte_len)
                        };
                        assert_eq!(shape, task.logical_shape());
                        let shape = shape
                            .iter()
                            .map(|size| i32::try_from(*size).unwrap())
                            .collect::<Vec<_>>();
                        let value = WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap();
                        assert_eq!(value.bytes().unwrap(), bytes);
                        (ParameterId::new(task.name()).unwrap(), value, bytes)
                    })
                    .collect()
            })
            .collect();
        Self {
            layout,
            units,
            calls: RefCell::default(),
        }
    }
}

impl WorkspaceLayerwiseParameters for SelectedParameters {
    fn layout(&self) -> &ExecutionUnitLayout {
        &self.layout
    }

    fn parameters(
        &self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        context: &WorkspaceContext,
    ) -> Result<BTreeMap<ParameterId, WorkspaceTensor>, Error> {
        assert_eq!(self.layout.address(ordinal), Some(address));
        self.calls.borrow_mut().push((ordinal, address));
        self.units[ordinal]
            .iter()
            .map(|(id, layout, bytes)| {
                // A future population can allocate independently. These roots
                // describe that selected capacity, not registered source credit.
                let storage = WorkspaceExistingStorage::new(Some(*bytes), context);
                Ok((
                    id.clone(),
                    WorkspaceTensor::existing_with_storage(layout.clone(), &storage, context)?,
                ))
            })
            .collect()
    }
}

fn residencies() -> [eredu_core::ResidencyPlan; 2] {
    [
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: None,
            host_budget_bytes: None,
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 16 << 20,
            host_budget_bytes: 32 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ]
}

type OperationSignature = (String, Vec<WorkspaceLayout>, Vec<WorkspaceLayout>);

fn selected_quote(
    sources: &PreparedModelSources,
    temperature: f32,
) -> (PreparedTextGenerationWorkspace, Vec<OperationSignature>) {
    let before = sources.target().source_diagnostics().unwrap();
    let provider = SelectedParameters::new(sources);
    let facts = Facts::default();
    let context = WorkspaceContext::new(facts.clone());
    let state = WorkspaceResidentStateFactory::new(
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(256).unwrap(),
        &context,
    )
    .unwrap()
    .realize(sources.selected().text_realization().state().layout())
    .unwrap();
    let request = eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(temperature),
                repetition_penalty: Some(1.2),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let geometry = InferenceGeometry {
        batch_size: 1,
        ..geometry(2, OutputDemand::LastPosition)
    };
    let report = sources
        .inference_blueprint()
        .quote_replicated_layerwise_text_with_sampling(
            geometry,
            &state,
            &context,
            request,
            &TokenFilter::All,
            &provider,
        )
        .unwrap();
    assert_eq!(report.equations.geometry(), geometry);
    assert_eq!(report.equations.completed_spans(), 6); // 2 + 2 + 1, then three decodes.
    assert!(report.equations.transient().bytes().unwrap() > 0);
    assert!(report.equations.retained_peak_bytes().unwrap() > 0);
    assert_eq!(report.equations.first_gap(), None);
    assert_eq!(report.sampling.steps, 3);
    assert!(report.sampling.peak.bytes().unwrap() > 0);
    assert_eq!(report.sampling.first_gap, None);
    assert!(state.as_ref().iter().all(|layer| layer.position() == 0));

    let expected = (0..6)
        .flat_map(|_| {
            (0..provider.layout.len())
                .map(|ordinal| (ordinal, provider.layout.address(ordinal).unwrap()))
        })
        .collect::<Vec<_>>();
    assert_eq!(*provider.calls.borrow(), expected);
    let operations = facts.operations.lock().unwrap();
    let slots = provider.units.iter().map(Vec::len).sum::<usize>();
    assert!(
        operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::ParameterPlaceholder))
            .count()
            >= 6 * slots,
        "every forward span must include fresh unit parameter construction"
    );
    assert!(operations
        .iter()
        .any(|op| matches!(op.kind, WorkspaceOperationKind::Rotary(..))));
    assert!(operations
        .iter()
        .any(|op| matches!(op.kind, WorkspaceOperationKind::Sampling(_))));
    let signature = operations
        .iter()
        .map(|op| {
            (
                format!("{:?}", op.kind),
                op.inputs.clone(),
                op.outputs.clone(),
            )
        })
        .collect();
    let after = sources.target().source_diagnostics().unwrap();
    assert_eq!(before, after);
    (report, signature)
}

#[test]
fn selected_host_and_disk_share_ordinary_and_routed_equations_and_sampling() {
    for family in ["llama", "qwen3_moe"] {
        let mut config = config(family, false);
        config["num_hidden_layers"] = 3.into();
        let (artifact, values) = prepared_adapter::payload_fixture_config(&config, 1.0);
        assert!(values.values().any(|(_, bits)| {
            bits.iter().any(|bits| {
                let value = f32::from_bits(*bits);
                value.is_finite() && value != 0.0 && value != 1.0
            })
        }));
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        for temperature in [0.0, 0.7] {
            let quotes = residencies().map(|residency| {
                let disk = matches!(residency, eredu_core::ResidencyPlan::DenseDiskStream { .. });
                let sources = prepared_adapter::prepare(
                    &inspection,
                    &prepared_adapter::plan(None).with_residency(residency),
                    &prepared_adapter::NumericPreparationProvider { addressable: false },
                )
                .unwrap();
                assert_eq!(
                    matches!(
                        sources.selected().text_realization().residency(),
                        eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                    ),
                    disk
                );
                assert_eq!(
                    sources
                        .selected()
                        .text_realization()
                        .requirements()
                        .execution_units()
                        .len(),
                    3
                );
                selected_quote(&sources, temperature)
            });
            let [(host, host_ops), (disk, disk_ops)] = quotes;
            assert_eq!(host_ops, disk_ops, "{family} temperature={temperature}");
            assert_eq!(host.equations.transient(), disk.equations.transient());
            assert_eq!(
                host.equations.retained_peak_bytes(),
                disk.equations.retained_peak_bytes()
            );
            assert_eq!(
                host.equations.tensor_transient_peak_bytes(),
                disk.equations.tensor_transient_peak_bytes()
            );
            assert_eq!(
                host.equations.host_peak_bytes(),
                disk.equations.host_peak_bytes()
            );
            assert_eq!(host.equations.peak_span(), disk.equations.peak_span());
            assert_eq!(host.sampling.output_width, disk.sampling.output_width);
            assert_eq!(host.sampling.peak, disk.sampling.peak);
            assert_eq!(
                host.sampling.tensor_peak_bytes,
                disk.sampling.tensor_peak_bytes
            );
            assert_eq!(host.sampling.host_peak_bytes, disk.sampling.host_peak_bytes);
            assert_eq!(
                host.sampling.final_history_bytes,
                disk.sampling.final_history_bytes
            );
        }
    }
}

#[test]
fn selected_layerwise_equations_do_not_supply_missing_materialization_coverage() {
    let config = config("llama", false);
    let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    for residency in residencies() {
        let sources = prepared_adapter::prepare(
            &inspection,
            &prepared_adapter::plan(None).with_residency(residency),
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        let (report, _) = selected_quote(&sources, 0.7);
        let geometry = report.equations.geometry();
        let layout = sources.selected().text_realization().state().layout();
        let state = eredu_core::estimate_runtime_state(
            &eredu_core::StateMemoryLayout::new(
                layout.layers().clone(),
                layout.layer_prefix_offsets(),
                config["hidden_size"].as_u64().unwrap(),
                256,
                eredu_core::EstimationCompleteness::Complete,
            )
            .unwrap(),
            eredu_core::InputTokenCount::text(geometry.input_positions),
            geometry.max_output_tokens,
            geometry.batch_size,
            std::num::NonZeroU8::new(4).unwrap(),
        )
        .unwrap();
        let absent =
            || WorkspaceBound::bounded(0, "no additional work in this isolated composition");
        let missing = WorkspaceBound::Unknown {
            reason: "selected layerwise population has no materialization/transfer bound".into(),
        };
        let full = report
            .compose(
                report.equations.refine_state_backing(state).unwrap(),
                eredu_core::ExecutionWorkspaceEstimate {
                    geometry,
                    activations: absent(),
                    attention: absent(),
                    vocabulary: absent(),
                    state_update: absent(),
                    materialization: missing.clone(),
                    retained: absent(),
                },
            )
            .unwrap();
        let workspace = full.execution_workspace.unwrap();
        assert_eq!(workspace.materialization, missing);
        assert_eq!(workspace.peak_bytes().unwrap(), None);
        assert_ne!(
            full.completeness,
            eredu_core::EstimationCompleteness::Complete
        );
    }
}
