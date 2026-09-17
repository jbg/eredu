//! The admitted route must execute, not merely accept an unadvanced prompt.
use super::*;
#[path = "execution/gemma.rs"]
mod gemma;
use crate::tests::support::media_completion;
use eredu_core::execution_control::SnapshotLimits;
use eredu_core::{
    GenerationSequenceConsumerLayout, RetainedGenerationSequence, TokenTerminalSignals,
};
use eredu_runtime::execution_control::{
    SnapshotBudget, SnapshotTokenController, TextContinuationSnapshot, TextSnapshotBackend,
    TextSnapshotError,
};
use eredu_runtime::working_memory::WorkspaceCopyLimits;

struct AllowAll;
impl TokenFilterController for AllowAll {
    type Error = std::convert::Infallible;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }
    fn inference_workspace(&self, count: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        NoCallbacks.inference_workspace(count)
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

impl SnapshotTokenController for AllowAll {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        self.original_snapshot_storage_bytes()
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(Self)
    }
    fn original_snapshot_storage_bytes(&self) -> Option<u64> {
        u64::try_from(std::mem::size_of::<Self>()).ok()
    }
    fn fork_original_snapshot(&self) -> Option<Self> {
        Some(Self)
    }
}

type Arrays = Vec<(Vec<i32>, Vec<f32>)>;
type Fixed = Vec<(
    usize,
    eredu_core::cache::StateTensorRole,
    Vec<i32>,
    Vec<f32>,
)>;
struct ResultRow {
    ids: Vec<u32>,
    arrays: Arrays,
    fixed: Fixed,
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (position, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!(
            (a - b).abs() <= 3e-4 + 3e-4 * b.abs(),
            "state value {position}: {a} != {b}"
        );
    }
}
fn same(actual: &ResultRow, expected: &ResultRow) {
    assert_eq!(actual.ids, expected.ids);
    assert_eq!(actual.arrays.len(), expected.arrays.len());
    for ((shape, a), (other, b)) in actual.arrays.iter().zip(&expected.arrays) {
        assert_eq!(shape, other);
        close(a, b);
    }
    assert_eq!(actual.fixed.len(), expected.fixed.len());
    for ((layer, role, shape, a), (other, other_role, other_shape, b)) in
        actual.fixed.iter().zip(&expected.fixed)
    {
        assert_eq!((layer, role, shape), (other, other_role, other_shape));
        close(a, b);
    }
}
fn backing(prompt: &MlxModelInput) -> Vec<safemlx::AllocationIdentity> {
    prompt.with_borrowed(|view| {
        view.parts
            .iter()
            .flat_map(|part| {
                std::iter::once(part.payload().value()).chain(part.metadata().values())
            })
            .map(|array| {
                array
                    .try_allocation_info()
                    .unwrap()
                    .expect("completed B")
                    .identity()
            })
            .collect()
    })
}

fn numeric_state(runtime: &mut ModelRuntime<MlxBackend<'_>>, ids: Vec<u32>) -> ResultRow {
    // Read the completed state through an immutable loan. The mutable fixture
    // hook permits arbitrary native work and rightly refuses while B is held.
    runtime.session().ensure_no_submission_in_flight().unwrap();
    let target = runtime.session().payload.model.erased();
    ResultRow {
        ids,
        arrays: target.retained_numeric_state_snapshot().unwrap().unwrap(),
        fixed: target.fixed_numeric_state_snapshot().unwrap(),
    }
}

// One saved pending F0/A0 or committed F9/A1 drives two fresh requests after
// the source reaches F11/A3. The actual copied sequence provider owns H.
fn snapshot_resumes(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    prompt: MlxModelInput,
    pool: &WorkingMemoryPool,
    chunk: &mut u64,
    pending_media: bool,
    input_positions: u64,
) -> Vec<u32> {
    let prefix = usize::from(!pending_media);
    let saved_frontier = if pending_media { 0 } else { input_positions };
    let source_backing = backing(&prompt);
    let cancellation = eredu_core::GenerationCancellationToken::new();
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: u64::MAX,
        cumulative_copy_bytes: u64::MAX,
    });
    let consumer = GenerationSequenceConsumerLayout::for_driver_types::<
        RetainedGenerationSequence,
        TextSnapshotError<Error>,
        BackendFailure,
    >()
    .unwrap();
    let (saved, saved_sequence, ids) = {
        let mut generation = ControlledTextGeneration::from_input_with_sequence(
            runtime,
            TextGenerationInput::OriginalPrepared(prompt),
            config(),
            AllowAll,
            None,
            GenerationSequenceRequest::new(3, &[]).with_consumer(&consumer),
        )
        .unwrap();
        *chunk = generation
            .preparation_report()
            .unwrap()
            .geometry
            .prefill_chunk_positions;
        assert!((1..=2).contains(&*chunk));
        let prepared = generation
            .take_prepared_sequence()
            .unwrap()
            .prepare_storage();
        let local = prepared
            .as_ref()
            .map(|_| Some(()))
            .map_err(|_| "sequence storage");
        generation
            .finish_text_preparation_cancellable(
                eredu_core::run_preparation::TextPreparationStage::Delivery,
                local,
                |_| "delivery agreement",
            )
            .unwrap()
            .unwrap();
        let mut sequence = prepared.unwrap();
        let mut ids = Vec::new();
        if !pending_media {
            let first = generation
                .next_cancellable(&cancellation)
                .unwrap()
                .unwrap()
                .token_id();
            sequence
                .commit(first, TokenTerminalSignals::default())
                .unwrap();
            ids.push(first);
        }
        let (saved, saved_sequence) = {
            let mut boundary = generation.snapshot_source().unwrap();
            let (runtime, state, pending) = boundary.parts();
            assert_eq!(
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .original_request_media_binding()
                    .unwrap()
                    .frontier(),
                saved_frontier
            );
            let bytes = MlxBackend::original_saved_components_preparation_bytes(
                runtime,
                MlxBackend::sampling_state(state),
                pending,
            )
            .unwrap()
            .unwrap();
            let host = pool.prepare_generation_snapshot_host_copy::<
                RetainedGenerationSequence, TextSnapshotError<Error>, MlxBackend<'_>, AllowAll,
            >(&sequence, u64::MAX, bytes).unwrap();
            TextContinuationSnapshot::capture_original_host(
                &mut boundary,
                &budget,
                host,
                WorkspaceCopyLimits::new(u64::MAX),
            )
            .unwrap()
        };
        assert_eq!(saved.next_prediction(), prefix as u64);
        assert_eq!(saved.remaining_tokens(), Some(3 - prefix));
        assert_eq!(saved_sequence.tokens(), ids);
        while let Some(next) = generation.next_cancellable(&cancellation) {
            let id = next.unwrap().token_id();
            sequence
                .commit(id, TokenTerminalSignals::default())
                .unwrap();
            ids.push(id);
        }
        assert_eq!(sequence.tokens(), ids);
        (saved, saved_sequence, ids)
    };
    assert_eq!(ids.len(), 3);
    let expected = numeric_state(runtime, ids[prefix..].to_vec());
    let mut sampling = config().sampling();
    sampling.max_new_tokens = Some(3 - prefix);
    let resumed_config =
        TextGenerationConfig::new(sampling).with_inference_policy(config().inference_policy());
    let initial_copies = budget.usage().cumulative_copy_bytes;
    for branch in [false, true] {
        let previous_copies = budget.usage().cumulative_copy_bytes;
        let bytes = saved
            .original_resume_preparation_bytes(runtime, resumed_config)
            .unwrap();
        let host = pool.prepare_generation_resume_host_copy::<
            RetainedGenerationSequence, TextSnapshotError<Error>, MlxBackend<'_>, AllowAll,
        >(&saved_sequence, u64::MAX, bytes).unwrap();
        let (mut generation, mut sequence) = if branch {
            saved.fork_original_host(runtime, resumed_config, host, &cancellation)
        } else {
            saved.restore_original_host(runtime, resumed_config, host, &cancellation)
        }
        .unwrap()
        .unwrap();
        assert_eq!(sequence.tokens(), &ids[..prefix]);
        let geometry = generation.preparation_report().unwrap().geometry;
        assert_eq!(
            (geometry.cached_positions, geometry.input_positions),
            (saved_frontier, if pending_media { input_positions } else { 1 })
        );
        assert_eq!(geometry.max_output_tokens, (3 - prefix) as u64);
        if pending_media {
            assert_eq!(geometry.prefill_chunk_positions, *chunk);
            let mut boundary = generation.snapshot_source().unwrap();
            let (_, _, pending) = boundary.parts();
            let Some(eredu_core::PendingTextInput::Prefill(prompt)) = pending else {
                panic!("restored unadvanced media must remain genuine pending prefill");
            };
            assert_eq!(
                backing(prompt),
                source_backing,
                "same completed B after restore/fork"
            );
        }
        let mut actual = Vec::new();
        while let Some(next) = generation.next_cancellable(&cancellation) {
            let id = next.unwrap().token_id();
            sequence
                .commit(id, TokenTerminalSignals::default())
                .unwrap();
            actual.push(id);
        }
        assert_eq!(sequence.tokens(), ids);
        drop((generation, sequence));
        same(&numeric_state(runtime, actual), &expected);
        assert_eq!(
            saved_sequence.tokens(),
            &ids[..prefix],
            "immutable saved provider"
        );
        assert_eq!(saved.next_prediction(), prefix as u64);
        assert!(budget.usage().cumulative_copy_bytes > previous_copies);
    }
    assert!(budget.usage().cumulative_copy_bytes > initial_copies);
    drop((saved, saved_sequence));
    // Exact final request/account retirement is checked by run after model/source drop.
    assert_eq!(budget.usage().snapshots, 0);
    ids
}

fn run(
    pool: &WorkingMemoryPool,
    path: &std::path::Path,
    conditional: bool,
    mode: usize,
    route: usize,
) -> ResultRow {
    run_original(pool, path, mode, route, 9, 64, |pool| {
        source(pool, if conditional { 16 } else { 64 })
    })
}
fn run_original(
    pool: &WorkingMemoryPool,
    path: &std::path::Path,
    mode: usize,
    route: usize,
    input_positions: u64,
    vocabulary: u32,
    source: impl FnOnce(&WorkingMemoryPool) -> OriginalPreparedHostInput,
) -> ResultRow {
    eprintln!("media case: positions={input_positions}, residency={mode}, route={route}");
    // The process registry retains admitted stream/source-worker birth accounts
    // after the backend drops. Establish those owners before the request baseline.
    let backend = original_request_backend(pool);
    let baseline = pool.used_bytes().unwrap();
    let baseline_owners = pool.unquoted_owner_count().unwrap();
    let host = source(pool);
    let selected = admitted_media_config(&backend, path, mode);
    let semantics = selected
        .prepared_sources()
        .plan_original_media_semantics(&host)
        .unwrap()
        .compile(&pool)
        .unwrap();
    let completed = MlxPreparedInputMaterializer::prepare()
        .unwrap()
        .model_input_plan(&semantics)
        .unwrap()
        .materialize(&pool)
        .unwrap();
    let model = backend.prepare_model_borrowed(&selected).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    runtime
        .session()
        .payload
        .model
        .erased()
        .prepare_completed_media_binding_fixture()
        .unwrap();
    let prompt = completed.bind(&runtime, semantics).unwrap();
    let source_hold = prompt.clone();
    let expected_backing = backing(&source_hold);
    let expected_identity = source_hold.cache_identity().unwrap().clone();
    let before = runtime
        .session()
        .payload
        .model
        .erased()
        .original_request_media_binding()
        .unwrap();
    let compiles = COMPILES.get();
    let mut chunk = input_positions;
    let (ids, roots) = media_completion::observe(None, || {
        if route == 0 {
            // Same original B and ordinary equations, one complete prefill span.
            TextGeneration::from_prompt(
                &mut runtime,
                prompt.with_prefill_chunk_positions(input_positions.try_into().unwrap()),
                TextGenerationConfig::new(config().sampling()),
            )
            .unwrap()
            .map(|next| next.unwrap().token_id().unwrap())
            .collect::<Vec<_>>()
        } else if route == 1 {
            let mut generation = TextGeneration::from_input_with_sequence(
                &mut runtime,
                TextGenerationInput::OriginalPrepared(prompt),
                config(),
                TokenFilter::All,
                None,
                GenerationSequenceRequest::new(3, &[]),
            )
            .unwrap();
            let report = generation
                .preparation_report()
                .expect("actual accepted media quote");
            assert_eq!(report.geometry.input_positions, input_positions);
            chunk = report.geometry.prefill_chunk_positions;
            assert!((1..=2).contains(&chunk));
            generation
                .by_ref()
                .map(|next| next.unwrap().token_id().unwrap())
                .collect()
        } else if route == 3 || route == 4 {
            snapshot_resumes(&mut runtime, prompt, pool, &mut chunk, route == 4, input_positions)
        } else {
            let mut generation = ControlledTextGeneration::from_input_with_sequence(
                &mut runtime,
                TextGenerationInput::OriginalPrepared(prompt),
                config(),
                AllowAll,
                None,
                GenerationSequenceRequest::new(3, &[]),
            )
            .unwrap();
            let report = generation
                .preparation_report()
                .expect("same controlled accepted media quote");
            assert_eq!(report.geometry.input_positions, input_positions);
            chunk = report.geometry.prefill_chunk_positions;
            assert!((1..=2).contains(&chunk));
            let cancellation = eredu_core::GenerationCancellationToken::new();
            let mut ids = Vec::new();
            while let Some(next) = generation.next_cancellable(&cancellation) {
                ids.push(next.unwrap().token_id());
            }
            ids
        }
    });
    assert_eq!(
        ids.len(),
        3,
        "prefill plus two genuine cached decode predictions"
    );
    assert!(ids.iter().all(|id| *id < vocabulary));
    assert_eq!(COMPILES.get(), compiles, "no repeated B compilation/upload");
    assert_eq!(
        backing(&source_hold),
        expected_backing,
        "exact completed input allocations retained"
    );
    let packet = source_hold.original_media.as_ref().unwrap();
    assert!(packet.semantics().source().same_source(&host));
    let after = runtime
        .session()
        .payload
        .model
        .erased()
        .original_request_media_binding()
        .unwrap();
    assert!(before.same_origin(&after));
    assert_eq!(
        after.frontier(),
        input_positions + 2,
        "all prompt positions and two cached decode inputs"
    );
    let request_samples = 2 * usize::try_from(input_positions.div_ceil(chunk)).unwrap();
    assert_eq!(
        roots.len(),
        request_samples * if route == 4 { 3 } else { 1 }
    );
    for request in roots.chunks_exact(request_samples) {
        let settled = &request[1];
        assert!(settled.after && !settled.backing.is_empty());
        assert!(settled.backing.iter().all(Option::is_some));
        for pair in request.chunks_exact(2) {
            assert!(!pair[0].after && pair[1].after);
            assert_eq!(
                pair[1].backing, settled.backing,
                "one encoder result retained across this request's decoder chunks"
            );
            assert!(pair[1].ready.iter().all(|ready| *ready));
        }
    }
    // Read the completed state through an immutable loan. The mutable fixture
    // hook permits arbitrary native work and rightly refuses while B is held.
    runtime.session().ensure_no_submission_in_flight().unwrap();
    let target = runtime.session().payload.model.erased();
    assert_eq!(
        target
            .resident_copy_input_identity()
            .unwrap()
            .as_ref()
            .map(AsRef::as_ref),
        Some(&expected_identity)
    );
    let arrays = target.retained_numeric_state_snapshot().unwrap().unwrap();
    let fixed = target.fixed_numeric_state_snapshot().unwrap();
    assert!(
        arrays
            .iter()
            .flat_map(|(_, values)| values)
            .any(|value| value.abs() > 1e-5),
        "nonzero actual decoder state"
    );
    let result = ResultRow { ids, arrays, fixed };
    drop((source_hold, runtime, selected, host));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.used_bytes().unwrap() == baseline
            && pool.unquoted_owner_count().unwrap() == baseline_owners
    });
    assert_eq!(pool.used_bytes().unwrap(), baseline);
    result
}

#[test]
fn admitted_original_media_runs_bounded_prefill_and_cached_decode_with_controlled_parity() {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    assert!(safemlx::metal::is_available().unwrap());
    for conditional in [false, true] {
        let root = tempfile::tempdir().unwrap();
        if conditional {
            crate::tests::distributed_pipeline_ring::write_qwen35_conditional_component_fixture(
                root.path(),
                false,
            );
        } else {
            crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
                root.path(),
                false,
                false,
            );
        }
        for mode in 0..3 {
            let expected = run(&pool, root.path(), conditional, mode, 0);
            same(&run(&pool, root.path(), conditional, mode, 1), &expected);
            same(&run(&pool, root.path(), conditional, mode, 2), &expected);
        }
    }
}

#[test]
fn admitted_original_media_postcommit_restore_and_fork_preserve_decoder_and_input() {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    assert!(safemlx::metal::is_available().unwrap());
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let expected = run(&pool, root.path(), false, 0, 0);
    same(&run(&pool, root.path(), false, 0, 3), &expected);
}

#[test]
fn admitted_original_media_pending_restore_and_fork_preserve_source_and_decode() {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    assert!(safemlx::metal::is_available().unwrap());
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let expected = run(&pool, root.path(), false, 0, 0);
    same(&run(&pool, root.path(), false, 0, 4), &expected);
}
