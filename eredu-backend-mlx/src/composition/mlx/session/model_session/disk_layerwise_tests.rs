//! Managed foreground disk execution uses retained encoded-file read plans.

use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use crate::backend::runtime::residency::storage::RetainedStorage;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    residency::{MemoryTier, ResidencyPolicy},
    ControlledTextGeneration, InferenceGeometry, OutputDemand, TextGeneration, TextGenerationInput,
    TextPreparationInput, TokenFilterController,
};
use eredu_runtime::working_memory::{MemoryLedger, WorkingMemoryError};

const LAYERS: usize = 3;
const DEVICE_BUDGET: u64 = 1 << 30;

#[derive(Clone, Default)]
pub(super) struct Controller(pub(super) Rc<Cell<(usize, usize)>>);

impl TokenFilterController for Controller {
    type Error = std::convert::Infallible;

    fn inference_workspace_is_run_owned(&self) -> bool {
        true // Shared counters are test instrumentation, without numerical payload.
    }

    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 0,
        })
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let (decisions, commits) = self.0.get();
        self.0.set((decisions + 1, commits));
        Ok(TokenFilter::All)
    }

    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        let (decisions, commits) = self.0.get();
        self.0.set((decisions, commits + 1));
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

pub(super) fn config(temperature: f32, chunk: u64, capacity: u64) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(temperature),
                max_new_tokens: Some(4),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
                repetition_penalty: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(19)
    .with_inference_policy(eredu_core::TextInferencePolicy {
        prefill_chunk_positions: std::num::NonZeroU64::new(chunk),
        memory_limits: eredu_core::MemoryLimitDeclarations::new([(
            "host".into(),
            eredu_core::MemoryLimit::Finite(capacity),
        )]),
        submission_tracking_capacity_bytes: None,
        graph_metadata_capacity_bytes: None,
    })
}

pub(super) fn tokens() -> Vec<u32> {
    vec![1, 2, 3, 4, 5]
}

pub(super) fn evidence(ids: &Vec<u32>) -> TextPreparationInput<'static, MlxModelInput> {
    TextPreparationInput::TokenIds {
        positions: ids.len() as u64,
        capacity_bytes: (ids.capacity() * std::mem::size_of::<u32>()) as u64,
    }
}

pub(super) fn artifact() -> tempfile::TempDir {
    artifact_with_layers(LAYERS, true)
}

pub(super) fn artifact_with_layers(layers: usize, tied: bool) -> tempfile::TempDir {
    use safetensors::tensor::{serialize_to_file, TensorView};

    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", tied);
    let config_path = root.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["num_hidden_layers"] = layers.into();
    std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    let path = root.path().join("model.safetensors");
    let bytes = std::fs::read(&path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut views = Vec::new();
    let mut repeated = 0;
    for (name, value) in tensors.tensors() {
        if let Some(suffix) = name.strip_prefix("model.layers.0.") {
            repeated += 1;
            for ordinal in 0..layers {
                views.push((
                    format!("model.layers.{ordinal}.{suffix}"),
                    TensorView::new(value.dtype(), value.shape().to_vec(), value.data()).unwrap(),
                ));
            }
        } else {
            views.push((name, value));
        }
    }
    assert!(repeated > 0);
    serialize_to_file(views, None, &path).unwrap();
    root
}

pub(super) fn load_runtime(
    stream: &Stream,
    pool: &MemoryLedger,
    disk: bool,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    load_runtime_from_artifact(stream, pool, disk, artifact())
}

pub(super) fn load_runtime_with_context(
    stream: &Stream,
    pool: &MemoryLedger,
    disk: bool,
    maximum_positions: usize,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let artifact = artifact();
    let path = artifact.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["max_position_embeddings"] = maximum_positions.into();
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    load_runtime_from_artifact(stream, pool, disk, artifact)
}

fn load_runtime_from_artifact(
    stream: &Stream,
    pool: &MemoryLedger,
    disk: bool,
    artifact: tempfile::TempDir,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let source = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(stream, &source).with_memory_ledger(pool.clone());
    let residency = if disk {
        eredu_runtime::WeightResidency::dense_disk_stream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(DEVICE_BUDGET, 0, 0, 0).unwrap(),
        )
    } else {
        eredu_runtime::WeightResidency::fully_resident()
    };
    let options = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
    );
    let prepared = eredu_core::load_model(&backend, artifact.path(), options).unwrap();
    let runtime = ModelRuntime::from_prepared(backend, prepared).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    if disk {
        let report = report(&runtime);
        let layers = disk_units(&report);
        assert_eq!(layers.len(), LAYERS);
        assert!(layers
            .iter()
            .all(|unit| !unit.host_resident() && !unit.device_resident()));
        assert_eq!(
            report.weight_store().backend,
            eredu_checkpoint::store::WeightStoreBackend::Safetensors
        );
        assert!(report.unit_sources().values().all(
            |source| source.backend == eredu_checkpoint::store::WeightStoreBackend::Safetensors
        ));
        assert!(
            report
                .units()
                .iter()
                .map(|unit| unit.expected_bytes())
                .sum::<u64>()
                < DEVICE_BUDGET
        );
        let workspace = runtime
            .session()
            .payload
            .model
            .layerwise_workspace()
            .unwrap()
            .unwrap();
        assert_eq!(workspace.layout().len(), LAYERS);
        assert!(workspace.materialization().bytes().unwrap() > 0);
    }
    (runtime, artifact)
}

pub(super) fn report(runtime: &ModelRuntime<MlxBackend<'_>>) -> eredu_runtime::ResidencyReport {
    runtime.session().residency_report().unwrap().unwrap()
}

fn disk_units(
    report: &eredu_runtime::ResidencyReport,
) -> Vec<&eredu_core::residency::UnitResidencyReport> {
    report
        .units()
        .iter()
        .filter(|unit| unit.planned_tier() == MemoryTier::Disk)
        .collect()
}

pub(super) fn assert_disk_window(runtime: &ModelRuntime<MlxBackend<'_>>) {
    let report = report(runtime);
    let layers = disk_units(&report);
    assert_eq!(layers.len(), LAYERS);
    let selected = runtime
        .session()
        .payload
        .model
        .inference_blueprint()
        .unwrap()
        .selected();
    let depth = selected.text_realization().residency().device_depth(LAYERS);
    assert!(LAYERS > depth);
    assert!(layers.iter().all(|unit| !unit.host_resident()));
    let live = layers.iter().filter(|unit| unit.device_resident()).count();
    assert!(live > 0 && live <= depth);
    assert!(report.offload().tier_evictions(MemoryTier::Device).count() > 0);
    let pinned = report
        .units()
        .iter()
        .filter(|unit| unit.policy() == ResidencyPolicy::Pinned)
        .count();
    assert!(
        report
            .offload()
            .peak_resident_units()
            .get(MemoryTier::Device)
            <= pinned + depth
    );
    assert!(report.weight_store().physical_reads > 0);
    assert!(report.weight_store().physical_read_bytes > 0);
}

pub(super) fn reclaim() {
    safemlx::memory::clear_cache();
    MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

pub(super) fn settle(pool: &MemoryLedger, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.fixture_host_charge().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}

pub(super) fn live_bytes(runtime: Option<&ModelRuntime<MlxBackend<'_>>>, arrays: &[&Array]) -> u64 {
    let mut storage = match runtime {
        Some(runtime) => {
            runtime.session().ensure_no_submission_in_flight().unwrap();
            let (mut nonstate, decoder) = runtime
                .session()
                .payload
                .retained_idle_storage()
                .unwrap()
                .into_parts();
            nonstate.merge(decoder).unwrap();
            nonstate
        }
        None => RetainedStorage::default(),
    };
    for array in arrays {
        storage.include_array(array).unwrap();
    }
    storage
        .byte_bound()
        .unwrap()
        .expect("complete physical inventory")
}

pub(super) fn exact_capacity(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    pool: &MemoryLedger,
    ids: &Vec<u32>,
    temperature: f32,
    chunk: u64,
) -> (u64, u64) {
    let baseline = pool.fixture_host_charge().unwrap();
    let preparation = MlxBackend::admit_text_preparation(
        runtime,
        &evidence(ids),
        config(temperature, chunk, u64::MAX),
        &Controller::default(),
    )
    .unwrap();
    let bytes = preparation
        .request
        .as_ref()
        .unwrap()
        .request()
        .memory_reservation()
        .requirements()
        .get(crate::memory_fixture::topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    assert!(bytes > 0);
    let capacity = pool.fixture_host_charge().unwrap();
    assert_eq!(capacity, baseline + bytes);
    drop(preparation);
    settle(pool, baseline);
    (capacity, bytes)
}

pub(super) fn outputs(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    ids: Vec<u32>,
    generation: TextGenerationConfig,
    controlled: bool,
) -> Vec<MlxTextToken> {
    let controller = Controller::default();
    let outputs = if controlled {
        let outputs = ControlledTextGeneration::from_token_ids_with_sequence(
            runtime,
            eredu_core::TokenIdsInputPlan::new(&ids).unwrap(),
            generation,
            controller.clone(),
            None,
            eredu_core::GenerationSequenceRequest::new(4, &[]),
        )
        .unwrap()
        .map(|token| token.unwrap().into_output())
        .collect::<Vec<_>>();
        assert_eq!(controller.0.get(), (4, 4));
        outputs
    } else {
        TextGeneration::from_token_ids_with_sequence(
            runtime,
            eredu_core::TokenIdsInputPlan::new(&ids).unwrap(),
            generation,
            TokenFilter::All,
            None,
            eredu_core::GenerationSequenceRequest::new(4, &[]),
        )
        .unwrap()
        .map(Result::unwrap)
        .collect::<Vec<_>>()
    };
    assert_eq!(outputs.len(), 4);
    for (attempt, token) in outputs.iter().enumerate() {
        assert_eq!(token.step_receipt().unwrap().attempt(), attempt as u64);
    }
    outputs
}

pub(super) fn token_ids(tokens: &[MlxTextToken]) -> Vec<u32> {
    tokens
        .iter()
        .map(|token| token.token_id().unwrap())
        .collect()
}

pub(super) fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return Some(found);
        }
        error = error.source()?;
    }
}

#[test]
fn disk_layerwise_cold_quote_and_exact_capacity_reject_before_reads_or_callbacks() {
    if !crate::tests::support::native_process::enter("main") {
        return;
    }
    let (streams, pool, native_baseline) = crate::tests::support::native_process::metal();
    let stream = streams.execution();
    let (mut runtime, artifact) = load_runtime(&stream, &pool, true);
    let controller = Controller::default();
    let before = report(&runtime);
    let paths_before = paths::snapshot();
    let inputs = paths::session_input_creation_attempts();
    let resets = paths::session_reset_attempts();
    let baseline = pool.fixture_host_charge().unwrap();
    let peak = pool.fixture_host_peak().unwrap();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 4,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let mut previous = None;
    for _ in 0..3 {
        let (quote, width) = text_quote::quote(
            runtime.session(),
            geometry,
            20,
            config(0.7, 1, u64::MAX),
            controller.inference_workspace(4).unwrap(),
        )
        .unwrap();
        assert_eq!(width, 64);
        assert!(
            quote
                .execution_workspace
                .as_ref()
                .unwrap()
                .materialization
                .bytes()
                .unwrap()
                > 0
        );
        if let Some(previous) = &previous {
            assert_eq!(&quote, previous);
        }
        previous = Some(quote);
    }
    assert_eq!(pool.fixture_host_peak().unwrap(), peak);
    assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
    assert_eq!(report(&runtime), before);
    assert_eq!(paths::snapshot(), paths_before);
    let (capacity, _) = exact_capacity(&runtime, &pool, &tokens(), 0.7, 1);
    let failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
        "short disk budget must reject before sampler".into(),
    ));
    let error = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(tokens()),
        config(0.7, 1, capacity - 1),
        controller.clone(),
    )
    .err()
    .expect("minimum chunk cannot shrink to fit one byte less");
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    assert_eq!(controller.0.get(), (0, 0));
    assert_eq!(paths::snapshot(), paths_before);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(paths::session_reset_attempts(), resets);
    assert_eq!(report(&runtime), before);
    assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
    drop((error, failure));
    let outputs = outputs(&mut runtime, tokens(), config(0.7, 1, capacity), true);
    assert_disk_window(&runtime);
    assert!(pool.fixture_host_peak().unwrap() <= capacity);
    drop((outputs, runtime, artifact));
    settle(&pool, native_baseline);
}

#[test]
fn disk_layerwise_uneven_prefill_and_cached_decodes_match_resident_in_both_drivers() {
    if !crate::tests::support::native_process::enter("main") {
        return;
    }
    let (streams, pool, native_baseline) = crate::tests::support::native_process::metal();
    let stream = streams.execution();
    for temperature in [0.0, 0.7] {
        let mut reference = None;
        for disk in [false, true] {
            for controlled in [false, true] {
                let (mut runtime, artifact) = load_runtime(&stream, &pool, disk);
                let (capacity, _) = exact_capacity(&runtime, &pool, &tokens(), temperature, 2);
                let previous_peak = pool.fixture_host_peak().unwrap();
                let acquisitions = paths::bounded_unit_acquisitions();
                let before = disk.then(|| report(&runtime));
                let output = outputs(
                    &mut runtime,
                    tokens(),
                    config(temperature, 2, capacity),
                    controlled,
                );
                let ids = token_ids(&output);
                assert!(ids.iter().all(|id| *id < 64));
                if let Some(reference) = &reference {
                    assert_eq!(&ids, reference);
                } else {
                    reference = Some(ids);
                }
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .validate_text_frontier(8)
                    .unwrap();
                if let Some(before) = before {
                    assert!(paths::bounded_unit_acquisitions() - acquisitions >= LAYERS * (3 + 3));
                    assert_disk_window(&runtime);
                    let after = report(&runtime);
                    assert!(
                        after.weight_store().physical_reads > before.weight_store().physical_reads
                    );
                    assert!(
                        after.offload().tier_evictions(MemoryTier::Device).count()
                            > before.offload().tier_evictions(MemoryTier::Device).count()
                    );
                }
                assert!(pool.fixture_host_peak().unwrap() <= previous_peak.max(capacity));
                drop((output, runtime, artifact));
                settle(&pool, native_baseline);
            }
        }
    }
}

#[test]
fn disk_layerwise_next_request_progresses_with_escaped_old_token_and_releases_physical_roots() {
    if !crate::tests::support::native_process::enter("main") {
        return;
    }
    let (streams, pool, native_baseline) = crate::tests::support::native_process::metal();
    let stream = streams.execution();
    for controlled in [false, true] {
        let (mut runtime, artifact) = load_runtime(&stream, &pool, true);
        let retained_overhead = pool
            .fixture_host_charge()
            .unwrap()
            .checked_sub(live_bytes(Some(&runtime), &[]))
            .unwrap();
        let (cold_capacity, _) = exact_capacity(&runtime, &pool, &tokens(), 0.0, 2);
        // Exercise reuse within one finite ceiling. The separate capacity
        // handoff tests cover an explicitly larger successor policy.
        let capacity = cold_capacity.checked_mul(2).unwrap();
        let output_a = outputs(&mut runtime, tokens(), config(0.0, 2, capacity), controlled);
        let ids_a = token_ids(&output_a);
        assert_disk_window(&runtime);
        let alias = output_a[0]
            .value
            .as_strided(&[1][..], &[1][..], 0, &stream)
            .unwrap();
        alias.evaluated().unwrap();
        assert_eq!(
            alias.allocation_info().unwrap(),
            output_a[0].value.allocation_info().unwrap()
        );
        let old_owner = output_a[0].owner.clone();
        assert!(old_owner.resources_releasable());
        assert!(old_owner.direct_route.borrow().is_none());
        let old_arrays = output_a
            .iter()
            .map(|token| &token.value)
            .collect::<Vec<_>>();
        settle(
            &pool,
            retained_overhead + live_bytes(Some(&runtime), &old_arrays),
        );
        drop(old_arrays);
        assert_eq!(pool.fixture_host_limit().unwrap(), capacity);

        // A's final token was emitted but not decoded. Reuse appends it once.
        let continuation = vec![ids_a[3], 6, 7];
        let before = report(&runtime);
        let baseline = pool.fixture_host_charge().unwrap();
        let preparation = MlxBackend::admit_text_preparation(
            &runtime,
            &evidence(&continuation),
            config(0.0, 2, capacity),
            &Controller::default(),
        )
        .unwrap();
        let charge_b = preparation
            .request
            .as_ref()
            .unwrap()
            .request()
            .memory_reservation()
            .requirements()
            .get(crate::memory_fixture::topology().host_domain())
            .unwrap()
            .total()
            .unwrap();
        assert_eq!(pool.fixture_host_charge().unwrap(), baseline + charge_b);
        assert!(baseline + charge_b <= capacity);
        assert_eq!(pool.fixture_host_limit().unwrap(), capacity);
        drop(preparation);
        settle(&pool, baseline);
        let output_b = outputs(
            &mut runtime,
            continuation,
            config(0.0, 2, capacity),
            controlled,
        );
        let ids_b = token_ids(&output_b);
        runtime
            .session()
            .payload
            .model
            .erased()
            .validate_text_frontier(14)
            .unwrap();
        assert_disk_window(&runtime);
        assert!(
            report(&runtime).weight_store().physical_reads > before.weight_store().physical_reads
        );
        assert_eq!(token_ids(&output_a), ids_a);
        assert!(pool.fixture_host_peak().unwrap() <= capacity);
        assert_eq!(pool.fixture_host_limit().unwrap(), capacity);

        let reference_pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let (mut resident, resident_artifact) = load_runtime(&stream, &reference_pool, false);
        let mut full = tokens();
        full.extend_from_slice(&ids_a);
        full.extend([6, 7]);
        let (reference_capacity, _) = exact_capacity(&resident, &reference_pool, &full, 0.0, 2);
        let reference = outputs(
            &mut resident,
            full,
            config(0.0, 2, reference_capacity),
            controlled,
        );
        assert_eq!(ids_b, token_ids(&reference));
        drop((reference, resident, resident_artifact));
        settle(&reference_pool, 0);

        let all = output_a
            .iter()
            .chain(output_b.iter())
            .map(|token| &token.value)
            .collect::<Vec<_>>();
        settle(&pool, retained_overhead + live_bytes(Some(&runtime), &all));
        drop(all);
        let token_bytes = live_bytes(None, &[&alias]);
        assert!(token_bytes > 0);
        drop((output_a, output_b, old_owner, runtime, artifact));
        settle(&pool, native_baseline + token_bytes);
        assert_eq!(pool.fixture_host_limit().unwrap(), capacity);
        assert_eq!(alias.evaluated().unwrap().item::<u32>(), ids_a[0]);
        drop(alias);
        settle(&pool, native_baseline);
        assert_eq!(pool.fixture_host_limit().unwrap(), u64::MAX);
    }
}

#[test]
fn disk_payload_release_waits_for_recovery_ticket_but_not_completed_owner_aliases() {
    if !crate::tests::support::native_process::enter("main") {
        return;
    }
    let (streams, pool, native_baseline) = crate::tests::support::native_process::metal();
    let stream = streams.execution();
    let (runtime, artifact) = load_runtime(&stream, &pool, true);
    let retired = runtime.session().test_payload_retirement_probe();
    let mut authority = eredu_core::SessionAuthority::new();
    let owner = SubmissionResources::with_purpose(
        authority.begin_submission().unwrap(),
        Rc::new(Cell::new(false)),
        SubmissionPurpose::OriginalModel,
    );
    owner
        .payload
        .replace(Some(runtime.session().payload.clone()));
    let bytes = pool.fixture_host_charge().unwrap();
    assert!(bytes > native_baseline);
    let ticket = owner.ticket();
    let escaped_owner = owner.clone();
    owner.request_release();
    assert!(authority.require_idle().is_err());
    assert!(authority.begin_submission().is_err());
    assert!(owner.payload.borrow().is_some());
    drop((owner, runtime, artifact));
    reclaim();
    assert!(!retired());
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    drop(ticket);
    authority.require_idle().unwrap();
    assert!(escaped_owner.payload.borrow().is_none());
    assert!(escaped_owner.resources_releasable());
    let next = authority.begin_submission().unwrap();
    drop(next);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        retired()
    });
    settle(&pool, native_baseline);
    drop(escaped_owner);
}

#[test]
fn disk_layerwise_changed_source_preserves_typed_failure_and_uncertified_funding() {
    if !crate::tests::support::native_process::enter("main") {
        return;
    }
    let (streams, pool, native_baseline) = crate::tests::support::native_process::metal();
    let stream = streams.execution();
    let (mut runtime, artifact) = load_runtime(&stream, &pool, true);
    let (capacity, charge) = exact_capacity(&runtime, &pool, &tokens(), 0.7, 2);
    let controller = Controller::default();
    let mut run = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(tokens()),
        config(0.7, 2, capacity),
        controller.clone(),
    )
    .unwrap();
    let first = run.next().unwrap().unwrap().into_output();
    assert_eq!(controller.0.get(), (1, 1));
    let first_id = first.token_id().unwrap();
    // Replace the admitted filesystem object, preserving valid tensor contents.
    // A retained read plan must reject identity drift instead of reopening it.
    let path = artifact.path().join("model.safetensors");
    let replacement = artifact.path().join("replacement.safetensors");
    std::fs::copy(&path, &replacement).unwrap();
    std::fs::rename(&replacement, &path).unwrap();
    let error = run
        .next()
        .expect("second operation reports changed source")
        .err()
        .expect("retained source identity must reject replacement");
    assert!(
        matches!(
            cause::<eredu_checkpoint::store::EncodedReadFailure>(&error),
            Some(eredu_checkpoint::store::EncodedReadFailure {
                batch: Some(0),
                shard: Some(0),
                completed_shards: 0,
                cause: eredu_checkpoint::store::EncodedReadFailureCause::Changed,
            })
        ),
        "changed source must preserve its typed cause: {error:?}",
    );
    assert_eq!(controller.0.get().1, 1, "failed work commits no token");
    assert!(run.next().is_none());
    assert!(pool.fixture_host_charge().unwrap() >= native_baseline + charge);
    assert_eq!(first.token_id().unwrap(), first_id);
    drop((run, error, first));
    let retained = pool.fixture_host_charge().unwrap();
    let peak = pool.fixture_host_peak().unwrap();
    let before = report(&runtime);
    let paths_before = paths::snapshot();
    let inputs = paths::session_input_creation_attempts();
    let calls = controller.0.get();
    let next = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(&tokens()),
        config(0.7, 2, capacity.checked_mul(2).unwrap()),
        &controller,
    )
    .unwrap_err();
    assert!(
        matches!(
            cause::<WorkingMemoryError>(&next),
            Some(WorkingMemoryError::ExecutionFenced | WorkingMemoryError::UnknownBound)
        ) || matches!(cause::<Error>(&next), Some(Error::ArchitectureModel(_))),
        "failed native work must reject the successor before any capacity handoff: {next:?}"
    );
    assert_eq!(pool.fixture_host_limit().unwrap(), capacity);
    assert_eq!(pool.fixture_host_charge().unwrap(), retained);
    assert_eq!(pool.fixture_host_peak().unwrap(), peak);
    assert_eq!(report(&runtime), before);
    assert_eq!(paths::snapshot(), paths_before);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(controller.0.get(), calls);
    drop(next);
    // Observe session-handle retirement only after its final mutable operation.
    let retired = runtime.session().test_payload_retirement_probe();
    drop((runtime, artifact));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        retired()
    });
    reclaim();
    // Source failure leaves this operation's complete physical inventory
    // uncertified. Recovery and request teardown cannot refund that envelope.
    assert!(pool.fixture_host_charge().unwrap() >= native_baseline + charge);
}
