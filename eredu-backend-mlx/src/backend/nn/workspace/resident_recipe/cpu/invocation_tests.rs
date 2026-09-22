//! Complete existing Gemma DraftStep under its actual CPU native source census.
//! This is a native equation test; it does not bypass public request admission.
use super::*;
use crate::{
    backend::{
        managed_memory::gpu_stream::PreparedExecutionStreams, nn::shared::MlxNeuralBackend,
        runtime::cache::kv::ConcatKeyValueCache, MlxBackend, MlxDeviceIdentity,
    },
    MlxTensor,
};
use eredu_architectures::{
    external_assistant::invocation::ExternalAssistantOperation,
    gemma4::{
        assistant::invocation::{DraftStep, DraftStepArguments},
        Assistant, AssistantConfig, AssistantState,
    },
};
use eredu_nn::{
    CpuMatmulImplementation, ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor,
};
use safemlx::{
    Array, Device, DeviceType, OperationEvent, OriginalBufferBudget, OriginalScopeObserver,
    PrefillRoots, PrefillRootsRuntime, PreparedOriginalBufferBudget, PreparedPrefillFailure,
    PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner,
    SubmissionScope,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

const CONFIG: &str = r#"{"model_type":"gemma4_assistant","backbone_hidden_size":32,
"use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":4,
"text_config":{"model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":1,
"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
"rms_norm_eps":0.00001,"vocab_size":32,"max_position_embeddings":128,
"tie_word_embeddings":false,"attention_k_eq_v":false,"layer_types":["full_attention"]}}"#;

fn tensor(shape: &[i32], phase: usize) -> MlxTensor {
    let count: usize = shape.iter().map(|n| usize::try_from(*n).unwrap()).product();
    let data = (0..count)
        .map(|i| (((i * 13 + phase * 7) % 31) as f32 - 15.0) * 0.025)
        .collect::<Vec<_>>();
    MlxTensor::from_array(Array::from_slice(&data, shape))
}
struct BindNative {
    source: Vec<(String, Array)>,
    ordered: bool,
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for BindNative {
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut MlxTensor,
    ) {
        // The shared-KV block visits its multiplicative layer scalar at phase
        // 11; the general signed fixture sequence is exactly zero there.
        // Preserve a nonzero residual instead of annihilating the entire block.
        *value = if metadata.id().as_str().ends_with(".layer_scalar") {
            assert_eq!(value.shape(), [1]);
            if !self.ordered {
                assert_eq!(self.source.len() + 1, 11);
            }
            MlxTensor::from_array(Array::from_slice(&[0.75f32], &[1]))
        } else if metadata.id().as_str() == "masked_embedding.token_ordering" {
            assert!(self.ordered);
            assert_eq!(value.shape(), [32]);
            let permutation = (0..32).map(|i| (i * 13 + 7) % 32).collect::<Vec<i32>>();
            MlxTensor::from_array(Array::from_slice(&permutation, &[32]))
        } else {
            tensor(value.shape(), self.source.len() + 1)
        };
        value.as_array().evaluated().unwrap();
        self.source
            .push((metadata.id().as_str().to_owned(), value.as_array().clone()));
    }
}
struct BindQuote<'a> {
    sources: &'a [(String, Array)],
    index: usize,
    context: &'a WorkspaceContext,
}
fn project(value: &Array, context: &WorkspaceContext) -> WorkspaceTensor {
    let descriptor = value.try_descriptor().unwrap();
    assert_eq!(descriptor.row_contiguous(), Some(true));
    assert!(descriptor.facts().allocation().is_some());
    let (dtype, representation) = match descriptor.facts().dtype() {
        safemlx::Dtype::Float32 => (
            WorkspaceDtype::Float32,
            Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            )),
        ),
        safemlx::Dtype::Int32 => (WorkspaceDtype::Int32, None),
        other => panic!("unexpected actual parameter dtype: {other:?}"),
    };
    WorkspaceTensor::existing(
        context
            .layout(descriptor.shape(), dtype)
            .unwrap()
            .with_representation(representation),
        context,
    )
    .unwrap()
}
impl<'a> ParameterVisitorMut<'a, WorkspaceTensor> for BindQuote<'_> {
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut WorkspaceTensor,
    ) {
        let (name, source) = &self.sources[self.index];
        self.index += 1;
        assert_eq!(metadata.id().as_str(), name);
        assert_eq!(value.shape(), source.shape());
        *value = project(source, self.context);
    }
}
fn state(hidden: &MlxTensor, keys: &MlxTensor, values: &MlxTensor) -> AssistantState<MlxTensor> {
    AssistantState {
        evidence: None,
        hidden: hidden.clone(),
        kv_offset: 3,
        shared_kv: std::collections::HashMap::from([(
            eredu_core::AttentionPolicy::Full,
            (keys.clone(), values.clone()),
        )])
        .into(),
    }
}
fn compare(actual: &Array, expected: &[f32]) {
    let read = actual.evaluated().unwrap();
    let actual = read.try_iter::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.zip(expected) {
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-5 + 2e-5 * b.abs(),
            "actual {a}, ordinary {b}"
        );
    }
}
#[derive(Debug)]
struct Lifetime(Arc<AtomicBool>);
impl Drop for Lifetime {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[test]
#[ignore = "requires qualified native allocator and selected CPU SIMD execution"]
fn original_cpu_gemma_complete_draft_step_matches_ordinary_and_retains_escaped_state() {
    complete_draft_step(false);
}
#[test]
#[ignore = "requires qualified native allocator and selected CPU SIMD execution"]
fn original_cpu_ordered_gemma_complete_draft_step_matches_ordinary_and_retains_escaped_state() {
    complete_draft_step(true);
}
fn complete_draft_step(ordered: bool) {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected = MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, selected)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    );
    let environment = backend.original_copy_environment().unwrap();
    let stream = environment.stream();
    let runtime = PrefillRootsRuntime::prepare_for_stream(stream, stream).unwrap();
    let allocator = environment.input_runtime().unwrap();
    let mut config = AssistantConfig::from_json(CONFIG.as_bytes()).unwrap();
    if ordered {
        config.use_ordered_embeddings = true;
        config.num_centroids = 4;
        config.centroid_intermediate_top_k = 2;
    }
    let mut module = Assistant::<MlxNeuralBackend>::new(config.clone(), stream).unwrap();
    let mut sources = BindNative {
        source: Vec::new(),
        ordered,
    };
    module.visit_parameters_mut(&mut sources);
    let embedding = tensor(&[1, 1, 32], 101);
    let hidden = tensor(&[1, 1, 32], 102);
    let keys = tensor(&[1, 2, 3, 8], 103);
    let values = tensor(&[1, 2, 3, 8], 104);
    for value in [&embedding, &hidden, &keys, &values] {
        value.as_array().evaluated().unwrap();
    }
    let mut actual_state = state(&hidden, &keys, &values);
    let mut reference_state = state(&hidden, &keys, &values);
    let reference = DraftStep::execute::<MlxNeuralBackend, ConcatKeyValueCache>(
        &mut module,
        DraftStepArguments {
            embedding: &embedding,
            state: &mut reference_state,
        },
        stream,
    )
    .unwrap();
    let expected = reference
        .as_array()
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    let expected_hidden = reference_state
        .hidden
        .as_array()
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(expected.iter().all(|v| v.is_finite()) && expected.iter().any(|v| v.abs() > 1e-5));
    assert!(expected_hidden.iter().any(|v| v.abs() > 1e-5));
    assert_eq!(reference_state.kv_offset, 4);
    drop((reference, reference_state));

    // Actual completed source descriptors are projected through the same family
    // operation. Its retained state and returned logits require five roots.
    let context = WorkspaceContext::new(cpu);
    let mut quoted = DraftStep::workspace_module(&config, &context).unwrap();
    let mut binding = BindQuote {
        sources: &sources.source,
        index: 0,
        context: &context,
    };
    quoted.visit_parameters_mut(&mut binding);
    assert_eq!(binding.index, sources.source.len());
    let arguments = DraftStepArguments {
        embedding: &embedding,
        state: &mut actual_state,
    };
    let mut projected = DraftStep::project(
        &arguments,
        |v| Ok(project(v.as_array(), &context)),
        &context,
    )
    .unwrap();
    let geometry = DraftStep::geometry(&projected, &context).unwrap();
    let mut opening = Vec::new();
    DraftStep::visit_projected(&projected, &mut |v| opening.push(v.clone()));
    context.begin_state_span(&opening).unwrap();
    let out = DraftStep::trace(&mut quoted, &mut projected, &context).unwrap();
    let mut closing = Vec::new();
    DraftStep::visit_projected(&projected, &mut |v| closing.push(v.clone()));
    assert_eq!(opening.len(), 4);
    assert_eq!(closing.len(), 4);
    closing.push(out);
    let report = context.report(&closing).unwrap();
    assert_eq!(
        report.operations.iter().any(|op| matches!(
            op.kind,
            WorkspaceOperationKind::MaskedOutputProjection { .. }
        )),
        ordered
    );
    let recorder =
        ResidentRecipeRecorder::with_cpu_context(geometry, ordinary, cpu, &context).unwrap();
    let reduced = recorder.reduce_trace(&report, None, 4, 1).unwrap();
    assert_eq!(reduced.first_missing_operation, None);
    assert_eq!(reduced.validation_roots, 0);
    assert_eq!(reduced.nested_completions, 0);
    let storage = reduced.mutable_storage.unwrap();
    let completion = ResidentCompletionRecipe {
        validation_roots: 0,
        grouped_outputs: reduced.grouped_outputs,
        traversal: reduced.traversal.unwrap(),
        graph: reduced.graph.unwrap(),
        dispatch: reduced.dispatch,
        nested_completions: 0,
        nested_root_capacity: 0,
    };
    assert_eq!(completion.traversal.roots(), 5);
    let graph_bytes = usize::try_from(
        graph_capacity::ResidentGraphStorage::for_completion(completion)
            .unwrap()
            .full_capacity
            .unwrap(),
    )
    .unwrap();
    let record_bytes = usize::try_from(
        record_capacity::ResidentRecordStorage::for_completion(completion)
            .unwrap()
            .full_capacity
            .unwrap(),
    )
    .unwrap();
    let physical = OriginalBufferBudget::population_layout(
        &allocator,
        usize::try_from(storage.mutable_bytes()).unwrap(),
        storage.maximum_births(),
    )
    .unwrap()
    .capacity();
    assert!(graph_bytes > 0 && record_bytes > 0 && physical > 0);

    // Same native domain constructors and synchronous completion worker as the
    // external phase. This fixture grants no neutral request/role admission.
    let released = Arc::new(AtomicBool::new(false));
    let owner = Arc::new(Lifetime(released.clone()));
    let graph = PreparedSubmissionGraphQuota::try_new(graph_bytes, owner.clone())
        .unwrap()
        .try_allocate()
        .unwrap();
    let records = PreparedSubmissionRecordQuota::try_new(record_bytes, owner.clone())
        .unwrap()
        .try_allocate()
        .unwrap();
    let budget = PreparedOriginalBufferBudget::try_new(&allocator, physical, owner.clone())
        .unwrap()
        .try_allocate()
        .unwrap();
    let failure = PreparedPrefillFailure::try_new(owner.clone())
        .unwrap()
        .try_allocate()
        .unwrap();
    let mut roots = PrefillRoots::new_retained(&runtime, 5, &graph, &failure).unwrap();
    let mut scope = SubmissionScope::try_begin_retaining(
        PreparedSubmissionScopeOwner::try_new(owner.clone())
            .unwrap()
            .with_graph_quota(graph.clone())
            .with_record_quota(records.clone()),
    )
    .unwrap();
    scope.enable_scoped_observation().unwrap();
    scope.require_original_native_controls().unwrap();
    roots.bind_scope(&scope).unwrap();
    scope.enable_original_native_controls().unwrap();
    scope.bind_original_buffer_budget(&budget).unwrap();
    let observer = OriginalScopeObserver::require_current().unwrap();
    for (_, leaf) in &sources.source {
        OperationEvent::validate_traversal_leaf(leaf, &observer).unwrap();
    }
    for value in [&embedding, &hidden, &keys, &values] {
        OperationEvent::validate_traversal_leaf(value.as_array(), &observer).unwrap();
    }
    let bank = OperationEvent::prepare_resident_graph(completion.graph, &observer).unwrap();
    let output = DraftStep::execute_with_metadata::<MlxNeuralBackend, ConcatKeyValueCache>(
        &mut module,
        DraftStepArguments {
            embedding: &embedding,
            state: &mut actual_state,
        },
        stream,
        &context,
    )
    .unwrap();
    drop(bank);
    for value in [&embedding, &actual_state.hidden, &keys, &values, &output] {
        roots.append(value.as_array()).unwrap();
    }
    roots
        .complete_current_scope_on_stream_prepared(stream, &completion.traversal)
        .unwrap_or_else(|cause| panic!("complete CPU assistant roots: {cause}"));
    assert!(!observer.status().failed());
    assert_eq!(actual_state.kv_offset, 4);
    assert!(budget.occupied_bytes() > 0 && budget.occupied_bytes() <= physical);
    compare(output.as_array(), &expected);
    compare(actual_state.hidden.as_array(), &expected_hidden);
    scope.seal();
    // Output/event readiness can race the CPU worker's final record-frontier
    // update. Keep the actual observer through settlement; retirement is not
    // an implicit progress operation. Share the existing ten-second cap with
    // the final independently escaped logits/state owner checks below.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        let (progress, status) = observer.progress().unwrap();
        assert_eq!(progress, safemlx::ScopedSubmissionProgress::Observed);
        assert!(
            !status.failed() && !status.blocked(),
            "CPU assistant terminal observation: {status:?}"
        );
        if status.is_settled() {
            return true;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "CPU assistant record settlement: {status:?}"
        );
        false
    });
    assert_eq!(
        observer.retire_completed_records().unwrap(),
        safemlx::SubmissionRetirement::CompleteSnapshot
    );
    safemlx::try_with_submission_retirement(|| {
        drop((roots, scope, observer, failure, records, graph, budget))
    })
    .unwrap();
    drop((owner, sources, module, hidden, embedding, keys, values));
    safemlx::reclaim_allocation_owners();
    assert!(
        !released.load(Ordering::SeqCst),
        "escaped logits/state retain original native custody"
    );
    compare(output.as_array(), &expected);
    compare(actual_state.hidden.as_array(), &expected_hidden);
    drop(output);
    safemlx::reclaim_allocation_owners();
    assert!(
        !released.load(Ordering::SeqCst),
        "the actual updated hidden state owns its backing independently"
    );
    drop(actual_state);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        // This component owns its Scope directly, without a backend Recovery
        // node. Exact record settlement/retirement was proved above; this pass
        // now drains final array/backing custody after the escaped aliases drop.
        safemlx::try_retire_completed_submissions().unwrap();
        MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        if released.load(Ordering::SeqCst) {
            return true;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "CPU assistant final escaped owner retirement"
        );
        false
    });
}
