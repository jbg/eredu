//! Existing sampler state is borrowed by the same selected equation traversal.

use super::*;
use eredu_architectures::prepared_execution::{
    BorrowedTextSamplingWorkspace, PreparedExecutionError,
};
use eredu_core::{TextFilterWorkspace, TextGenerationConfig};
use eredu_runtime::ConfiguredTextSampler;

const PREFIX: [u32; 5] = [3, 11, 7, 19, 5];
const KEY_BYTES: u64 = 1024;

fn request(adaptive: bool, temperature: f32) -> TextGenerationConfig {
    let request = TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(2),
                temperature: Some(temperature),
                top_k: Some(7),
                top_p: Some(0.83),
                min_p: Some(0.07),
                repetition_penalty: Some(1.19),
                repeat_last_n: Some(-1),
                frequency_penalty: Some(0.13),
                presence_penalty: Some(0.41),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    if adaptive {
        request.with_mirostat_v2(3.7, 0.23).unwrap()
    } else {
        request
    }
}
fn populated(request: TextGenerationConfig) -> ConfiguredTextSampler {
    let mut sampler = ConfiguredTextSampler::from_config(request).unwrap();
    for (token, probability) in PREFIX.into_iter().zip([0.2, 0.3, 0.1, 0.4, 0.05]) {
        match &mut sampler {
            ConfiguredTextSampler::Standard(s) => s.accept_token(token),
            ConfiguredTextSampler::MirostatV2(s) => s.accept_token(token, probability).unwrap(),
        }
    }
    assert_eq!(sampler.history_len(), 5);
    assert_eq!(sampler.history_capacity(), 8);
    if let ConfiguredTextSampler::MirostatV2(s) = &sampler {
        assert_ne!(s.mu(), 2.0 * s.tau());
    }
    sampler
}
fn source_snapshot(sampler: &ConfiguredTextSampler) -> (Vec<u32>, *const u32, usize, String) {
    let history = match sampler {
        ConfiguredTextSampler::Standard(s) => s.generated_tokens(),
        ConfiguredTextSampler::MirostatV2(s) => s.generated_tokens(),
    };
    (
        history.to_vec(),
        history.as_ptr(),
        sampler.history_capacity(),
        format!("{sampler:?}"),
    )
}
fn key(context: &WorkspaceContext, bytes: Option<u64>) -> WorkspaceSamplingRandomState {
    WorkspaceSamplingRandomState::from_key(
        WorkspaceTensor::existing_with_storage(
            WorkspaceLayout::new(&[2], WorkspaceDtype::Uint32).unwrap(),
            &WorkspaceExistingStorage::new(bytes, context),
            context,
        )
        .unwrap(),
    )
    .unwrap()
}
fn state(
    sources: &PreparedModelSources,
    context: &WorkspaceContext,
) -> DeviceState<WorkspaceBackend, WorkspaceResidentLayerState> {
    WorkspaceResidentStateFactory::new(
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(256).unwrap(),
        context,
    )
    .unwrap()
    .realize(sources.selected().text_realization().state().layout())
    .unwrap()
}
fn future(outputs: u64) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        max_output_tokens: outputs,
        ..geometry(2, OutputDemand::LastPosition)
    }
}
fn signatures(facts: &Facts) -> Vec<OperationSignature> {
    facts
        .operations
        .lock()
        .unwrap()
        .iter()
        .map(|op| {
            (
                format!("{:?}", op.kind),
                op.inputs.clone(),
                op.outputs.clone(),
            )
        })
        .collect()
}
fn same_equations(a: &InferenceWorkspaceReport, b: &InferenceWorkspaceReport) {
    assert_eq!(a.geometry(), b.geometry());
    assert_eq!(a.completed_spans(), b.completed_spans());
    assert_eq!(a.transient(), b.transient());
    assert_eq!(a.retained_peak_bytes(), b.retained_peak_bytes());
    assert_eq!(
        a.tensor_transient_peak_bytes(),
        b.tensor_transient_peak_bytes()
    );
    assert_eq!(a.host_peak_bytes(), b.host_peak_bytes());
    assert_eq!(a.peak_span(), b.peak_span());
    assert_eq!(a.first_gap(), b.first_gap());
}
fn same_sampling(a: &SamplingWorkspaceReport, b: &SamplingWorkspaceReport) {
    assert_eq!(a.output_width, b.output_width);
    assert_eq!(a.steps, b.steps);
    assert_eq!(a.peak.bytes(), b.peak.bytes());
    assert_eq!(a.tensor_peak_bytes, b.tensor_peak_bytes);
    assert_eq!(a.host_peak_bytes, b.host_peak_bytes);
    assert_eq!(a.first_gap, b.first_gap);
    assert_eq!(a.final_history_bytes, b.final_history_bytes);
}
fn sampling_trace(facts: &Facts) -> Vec<WorkspaceSamplingOperation> {
    facts
        .operations
        .lock()
        .unwrap()
        .iter()
        .filter_map(|op| match &op.kind {
            WorkspaceOperationKind::Sampling(sampling) => Some(sampling.clone()),
            _ => None,
        })
        .collect()
}
fn selected(family: &str) -> serde_json::Value {
    let mut config = config(family, false);
    config["vocab_size"] = 37.into();
    config["num_hidden_layers"] = 3.into();
    config
}

#[test]
fn populated_resident_sampling_preserves_equations_and_quotes_the_full_future_allowance() {
    let (artifact, values) = prepared_adapter::payload_fixture_config(&selected("llama"), 1.0);
    assert!(values.values().any(|(_, bits)| bits.iter().any(|bits| {
        let value = f32::from_bits(*bits);
        value.is_finite() && value != 0.0 && value != 1.0
    })));
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let before = sources.target().source_diagnostics().unwrap();
    let blueprint = sources.inference_blueprint();
    // Geometry is the complete future allowance; the original configured limit
    // and five already accepted tokens must not be subtracted from it.
    let geometry = future(11);
    let equation_facts = Facts::default();
    let equation_context = WorkspaceContext::new(equation_facts.clone());
    let equations = blueprint
        .quote_replicated_resident_text(
            geometry,
            &state(&sources, &equation_context),
            &equation_context,
        )
        .unwrap();
    let equation_trace = signatures(&equation_facts);
    for adaptive in [false, true] {
        let request = request(adaptive, 0.7);
        let sampler = populated(request);
        let snapshot = source_snapshot(&sampler);
        let facts = Facts::default();
        let context = WorkspaceContext::new(facts.clone());
        let state = state(&sources, &context);
        let random = key(&context, Some(KEY_BYTES));
        let filter = TokenFilter::Allowed(vec![true, false, true]);
        let report = blueprint
            .quote_replicated_resident_text_with_existing_sampling(
                geometry,
                &state,
                &context,
                BorrowedTextSamplingWorkspace::new(&sampler, 0.7, Some(&random), &filter),
            )
            .unwrap();
        same_equations(&equations, &report.equations);
        assert_eq!(
            &signatures(&facts)[..equation_trace.len()],
            equation_trace.as_slice()
        );
        assert_eq!(report.equations.completed_spans(), 14);
        assert_eq!(report.sampling.steps, 11);
        assert_eq!(report.sampling.output_width, 37);
        assert_eq!(report.sampling.final_history_bytes, 16 * 4);
        assert!(report.sampling.tensor_peak_bytes.unwrap() >= KEY_BYTES);
        assert_eq!(report.sampling.first_gap, None);
        let trace = sampling_trace(&facts);
        assert!(!trace
            .iter()
            .any(|op| matches!(op, WorkspaceSamplingOperation::CreateRandomKey)));
        assert_eq!(
            trace
                .iter()
                .filter(|op| matches!(op, WorkspaceSamplingOperation::SplitRandomKey))
                .count(),
            11
        );
        assert_eq!(
            trace
                .iter()
                .filter_map(|op| match op {
                    WorkspaceSamplingOperation::Penalties {
                        history_positions, ..
                    } => Some(*history_positions),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            (5..16).collect::<Vec<_>>()
        );
        assert_eq!(
            trace
                .iter()
                .filter(|op| matches!(op, WorkspaceSamplingOperation::MirostatCutoff))
                .count(),
            if adaptive { 11 } else { 0 }
        );
        let ops = facts.operations.lock().unwrap();
        let sampled = ops
            .iter()
            .filter(|op| {
                matches!(
                    op.kind,
                    WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Categorical)
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(sampled.len(), 11);
        assert!(sampled.iter().all(|op| op.inputs[0].shape() == [1, 1, 37]));
        drop(ops);
        let direct_context = WorkspaceContext::new(Facts::default());
        let direct_key = key(&direct_context, Some(KEY_BYTES));
        let direct = quote_sampling_workspace(
            &sampler,
            0.7,
            Some(&direct_key),
            &WorkspaceLayout::new(&[1, 1, 37], WorkspaceDtype::Float32).unwrap(),
            &filter,
            11,
            &direct_context,
        )
        .unwrap();
        same_sampling(&report.sampling, &direct);
        let legacy_facts = Facts::default();
        let legacy_context = WorkspaceContext::new(legacy_facts.clone());
        let legacy = blueprint
            .quote_replicated_resident_text_with_sampling(
                geometry,
                &self::state(&sources, &legacy_context),
                &legacy_context,
                request,
                &filter,
            )
            .unwrap();
        same_equations(&legacy.equations, &report.equations);
        assert_eq!(
            &signatures(&legacy_facts)[..equation_trace.len()],
            equation_trace.as_slice()
        );
        assert!(state.as_ref().iter().all(|layer| layer.position() == 0));
        assert_eq!(
            context
                .report(std::slice::from_ref(random.key()))
                .unwrap()
                .state
                .unwrap()
                .retained_bytes,
            Some(KEY_BYTES)
        );
        assert_eq!(source_snapshot(&sampler), snapshot);
    }
    assert_eq!(sources.target().source_diagnostics().unwrap(), before);
}

#[test]
fn populated_layerwise_sampling_uses_the_same_host_and_disk_routed_traversal() {
    for family in ["llama", "qwen3_moe"] {
        let (artifact, _) = prepared_adapter::payload_fixture_config(&selected(family), 1.0);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        for adaptive in [false, true] {
            let request = request(adaptive, 0.7);
            let sampler = populated(request);
            let snapshot = source_snapshot(&sampler);
            let reports = residencies().map(|residency| {
                let sources = prepared_adapter::prepare(
                    &inspection,
                    &prepared_adapter::plan(None).with_residency(residency),
                    &prepared_adapter::NumericPreparationProvider { addressable: false },
                )
                .unwrap();
                let before = sources.target().source_diagnostics().unwrap();
                let parameters = SelectedParameters::new(&sources);
                let facts = Facts::default();
                let context = WorkspaceContext::new(facts.clone());
                let state = state(&sources, &context);
                let random = key(&context, Some(KEY_BYTES));
                let report = sources
                    .inference_blueprint()
                    .quote_replicated_layerwise_text_with_existing_sampling(
                        future(3),
                        &state,
                        &context,
                        BorrowedTextSamplingWorkspace::new(
                            &sampler,
                            0.7,
                            Some(&random),
                            &TokenFilter::All,
                        ),
                        &parameters,
                    )
                    .unwrap();
                assert_eq!(report.equations.completed_spans(), 6);
                assert_eq!(report.sampling.steps, 3);
                assert_eq!(report.sampling.final_history_bytes, 32);
                assert!(report.sampling.peak.bytes().is_some());
                let expected = (0..6)
                    .flat_map(|_| {
                        (0..parameters.layout.len())
                            .map(|ordinal| (ordinal, parameters.layout.address(ordinal).unwrap()))
                    })
                    .collect::<Vec<_>>();
                assert_eq!(*parameters.calls.borrow(), expected);
                let trace = signatures(&facts);
                assert!(facts
                    .operations
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|op| matches!(op.kind, WorkspaceOperationKind::ParameterPlaceholder)));
                let old_context = WorkspaceContext::new(Facts::default());
                let legacy = sources
                    .inference_blueprint()
                    .quote_replicated_layerwise_text_with_sampling(
                        future(3),
                        &self::state(&sources, &old_context),
                        &old_context,
                        request,
                        &TokenFilter::All,
                        &parameters,
                    )
                    .unwrap();
                same_equations(&legacy.equations, &report.equations);
                assert!(state.as_ref().iter().all(|layer| layer.position() == 0));
                assert_eq!(source_snapshot(&sampler), snapshot);
                assert_eq!(sources.target().source_diagnostics().unwrap(), before);
                (report, trace)
            });
            let [(host, host_trace), (disk, disk_trace)] = reports;
            same_equations(&host.equations, &disk.equations);
            same_sampling(&host.sampling, &disk.sampling);
            assert_eq!(host_trace, disk_trace);
        }
    }
}

#[derive(Debug)]
struct MissingScoreFacts(Facts);
impl WorkspaceMechanisms for MissingScoreFacts {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let mut bound = self.0.operation_bound(op)?.unwrap();
        if !matches!(op.kind, WorkspaceOperationKind::Sampling(_))
            && op.outputs.iter().any(|out| out.shape() == [1, 1, 37])
        {
            return Ok(None);
        }
        if matches!(
            op.kind,
            WorkspaceOperationKind::Sampling(
                WorkspaceSamplingOperation::Greedy | WorkspaceSamplingOperation::ReadToken
            )
        ) {
            bound.outputs[0] = WorkspaceOutputStorage::AliasInput(0);
        }
        Ok(Some(bound))
    }
    fn host_workspace_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        self.0.host_workspace_bound(op)
    }
}

#[test]
fn existing_random_and_score_backing_gaps_stay_unknown_and_foreign_context_rejects() {
    let (artifact, _) = prepared_adapter::payload_fixture_config(&selected("llama"), 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let before = sources.target().source_diagnostics().unwrap();
    let sampler = populated(request(false, 0.7));
    let snapshot = source_snapshot(&sampler);
    let context = WorkspaceContext::new(Facts::default());
    let random = key(&context, None);
    let report = sources
        .inference_blueprint()
        .quote_replicated_resident_text_with_existing_sampling(
            future(3),
            &state(&sources, &context),
            &context,
            BorrowedTextSamplingWorkspace::new(&sampler, 0.7, Some(&random), &TokenFilter::All),
        )
        .unwrap();
    assert!(report.equations.transient().bytes().is_some());
    assert!(report.sampling.peak.bytes().is_none());
    assert!(report.sampling.tensor_peak_bytes.is_none());
    assert_eq!(report.sampling.first_gap, Some(0));
    assert_eq!(report.sampling.steps, 3);
    let foreign = WorkspaceContext::new(Facts::default());
    let random = key(&foreign, Some(KEY_BYTES));
    let facts = Facts::default();
    let context = WorkspaceContext::new(facts.clone());
    let error = sources
        .inference_blueprint()
        .quote_replicated_resident_text_with_existing_sampling(
            future(3),
            &state(&sources, &context),
            &context,
            BorrowedTextSamplingWorkspace::new(&sampler, 0.7, Some(&random), &TokenFilter::All),
        )
        .unwrap_err();
    assert!(matches!(error, PreparedExecutionError::Backend(_)));
    assert!(sampling_trace(&facts).is_empty());
    assert_eq!(source_snapshot(&sampler), snapshot);
    let greedy = populated(TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.0),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
                repetition_penalty: Some(1.0),
                frequency_penalty: Some(0.0),
                presence_penalty: Some(0.0),
                ..Default::default()
            },
        )
        .unwrap(),
    ));
    let greedy_snapshot = source_snapshot(&greedy);
    let context = WorkspaceContext::new(MissingScoreFacts(Facts::default()));
    let report = sources
        .inference_blueprint()
        .quote_replicated_resident_text_with_existing_sampling(
            future(3),
            &state(&sources, &context),
            &context,
            BorrowedTextSamplingWorkspace::new(&greedy, 0.0, None, &TokenFilter::All),
        )
        .unwrap();
    assert!(report.equations.first_gap().is_some());
    assert!(report.sampling.peak.bytes().is_none());
    assert!(report.sampling.first_gap.is_some());
    assert_eq!(report.sampling.steps, 3);
    assert_eq!(source_snapshot(&greedy), greedy_snapshot);
    assert_eq!(sources.target().source_diagnostics().unwrap(), before);
}

#[test]
fn zero_future_outputs_preserve_cleared_capacity_and_price_the_existing_key() {
    let (artifact, _) = prepared_adapter::payload_fixture_config(&selected("llama"), 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let before = sources.target().source_diagnostics().unwrap();
    for adaptive in [false, true] {
        let mut sampler = populated(request(adaptive, 0.7));
        match &mut sampler {
            ConfiguredTextSampler::Standard(s) => s.clear_generated_tokens(),
            ConfiguredTextSampler::MirostatV2(s) => s.reset(),
        }
        assert_eq!(sampler.history_len(), 0);
        assert_eq!(sampler.history_capacity(), 8);
        let snapshot = source_snapshot(&sampler);
        let facts = Facts::default();
        let context = WorkspaceContext::new(facts.clone());
        let random = key(&context, Some(KEY_BYTES));
        let report = sources
            .inference_blueprint()
            .quote_replicated_resident_text_with_existing_sampling(
                future(0),
                &state(&sources, &context),
                &context,
                BorrowedTextSamplingWorkspace::new(
                    &sampler,
                    0.7,
                    Some(&random),
                    TextFilterWorkspace::OptionalMask {
                        max_mask_positions: 37,
                        mask_capacity_bytes: 64,
                    },
                ),
            )
            .unwrap();
        assert_eq!(report.equations.completed_spans(), 3);
        assert_eq!(report.sampling.steps, 0);
        assert_eq!(report.sampling.output_width, 37);
        assert_eq!(report.sampling.final_history_bytes, 32);
        assert_eq!(report.sampling.tensor_peak_bytes, Some(KEY_BYTES));
        assert_eq!(
            report.sampling.host_peak_bytes,
            Some(std::mem::size_of::<ConfiguredTextSampler>() as u64 + 32)
        );
        assert!(report.sampling.peak.bytes().is_some());
        assert!(sampling_trace(&facts).is_empty());
        assert_eq!(source_snapshot(&sampler), snapshot);
    }
    assert_eq!(sources.target().source_diagnostics().unwrap(), before);
}
