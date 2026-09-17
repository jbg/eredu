#[cfg(all(feature = "metal", not(feature = "cuda")))]
mod execution;
mod legacy_state;
use super::super::super::tests::{cold_config, source};
use super::*;
use eredu_core::{
    BackendFailure, Completion, ControlledTextGeneration, ControlledTextGenerationError,
    GenerationSequenceRequest, PreparedRequestRejection as R, TextGeneration,
    TextGenerationBackend, TextGenerationConfig, TextGenerationInput, TextPreparationOptions,
    TokenFilter, TokenFilterController,
};

struct NoCallbacks;
impl TokenFilterController for NoCallbacks {
    type Error = std::convert::Infallible;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }
    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        // This stateless fixture emits All and retains no filter or callback payload.
        static FILTER: TokenFilter = TokenFilter::All;
        Some(eredu_core::TextControllerWorkspace {
            filter: (&FILTER).into(),
            additional_host_bytes: 0,
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        panic!("no controller work before complete original admission")
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        panic!("no token commitment")
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        panic!("no prediction during refused startup")
    }
}
fn config() -> TextGenerationConfig {
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            do_sample: Some(false),
            max_new_tokens: Some(3),
            ..Default::default()
        },
    )
    .unwrap();
    TextGenerationConfig::new(sampling).with_inference_policy(eredu_core::TextInferencePolicy {
        managed_memory_capacity_bytes: Some(u64::MAX),
        prefill_chunk_positions: Some(2.try_into().unwrap()),
        ..Default::default()
    })
}
fn failure(runtime: &mut ModelRuntime<MlxBackend<'_>>, prompt: MlxModelInput, manual: bool) -> R {
    failure_with_options(runtime, prompt, manual, None)
}
fn failure_with_options(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    prompt: MlxModelInput,
    manual: bool,
    options: Option<TextPreparationOptions>,
) -> R {
    attempt_preparation(runtime, prompt, manual, options).expect("fixed source refusal")
}
fn attempt_preparation(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    prompt: MlxModelInput,
    manual: bool,
    options: Option<TextPreparationOptions>,
) -> Option<R> {
    // Retain the supplied source through refusal observations; consuming a last
    // source alias is independent of the admission operation itself.
    let source_hold = prompt.clone();
    let _ = crate::composition::mlx::session::model_session::text_quote::take_original_cold_facts();
    let request = GenerationSequenceRequest::new(3, &[7]);
    let compiles = COMPILES.get();
    let pool = runtime.backend().memory_pool().clone();
    let owners = pool.unquoted_owner_count().unwrap();
    let used = pool.used_bytes().unwrap();
    let (error, roots) = crate::tests::support::media_completion::observe(None, || {
        if manual {
            match ControlledTextGeneration::from_input_with_sequence(
                runtime,
                TextGenerationInput::OriginalPrepared(prompt),
                config(),
                NoCallbacks,
                options,
                request,
            ) {
                Ok(generation) => {
                    assert!(
                        pool.used_bytes().unwrap() > used,
                        "accepted Q and planning metadata remain owned by the prepared generation"
                    );
                    drop(generation);
                    None
                }
                Err(ControlledTextGenerationError::Preparation(error)) => Some(error),
                Err(other) => panic!("unexpected startup error {other}"),
            }
        } else {
            match TextGeneration::from_input_with_sequence(
                runtime,
                TextGenerationInput::OriginalPrepared(prompt),
                config(),
                TokenFilter::All,
                options,
                request,
            ) {
                Ok(generation) => {
                    assert!(
                        pool.used_bytes().unwrap() > used,
                        "accepted Q and planning metadata remain owned by the prepared generation"
                    );
                    drop(generation);
                    None
                }
                Err(error) => Some(error),
            }
        }
    });
    assert!(
        roots.is_empty(),
        "no media source completion or span execution"
    );
    assert_eq!(COMPILES.get(), compiles, "no B reconstruction");
    assert_eq!(
        pool.unquoted_owner_count().unwrap(),
        owners,
        "no ordinary inspection owner"
    );
    let Some(error) = error else {
        assert_eq!(
            pool.used_bytes().unwrap(),
            used,
            "dropping the prepared generation retires Q and planning while the independent B remains live"
        );
        drop(source_hold);
        return None;
    };
    let error: BackendFailure = error;
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    let rejection = loop {
        let current = cause.unwrap_or_else(|| {
            use std::fmt::Write as _;
            let mut chain = String::new();
            let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
            while let Some(current) = cause {
                writeln!(&mut chain, "{current:?}\n  display: {current}").unwrap();
                cause = current.source();
            }
            panic!("missing typed original refusal: {error}\nretained source chain:\n{chain}")
        });
        if let Some(rejection) = current.downcast_ref::<R>() {
            break *rejection;
        }
        cause = current.source();
    };
    if matches!(
        rejection,
        R::MissingNumericalWorkspace | R::MissingExecutionControls
    ) {
        assert_eq!(error.kind(), eredu_core::BackendFailureKind::Unsupported);
        let detail = error.to_string();
        match rejection {
            R::MissingNumericalWorkspace => {
                let missing = detail.split_once("original media native recipe is missing ")
                    .map(|(_, missing)| missing).unwrap_or_else(|| panic!("missing actual recipe diagnostic: {detail}"));
                if let Some(equation) = missing.strip_prefix("span ") {
                    let (span, operation) = equation.split_once(" operation ").expect("equation ordinal");
                    span.parse::<usize>().expect("actual span index");
                    let (operation, kind) = operation.split_once(": ").expect("actual operation kind");
                    operation.parse::<usize>().expect("actual operation index");
                    assert!(kind.starts_with("Some(") && kind.len() > 6, "observed operation kind: {detail}");
                } else {
                    let sampling = missing.strip_prefix("sampling ").expect("equation or sampling gap");
                    let (phase, operation) = sampling.rsplit_once(" operation ").expect("sampling ordinal");
                    assert!(!phase.is_empty(), "actual sampling phase");
                    operation.parse::<usize>().expect("actual sampling operation index");
                }
            }
            R::MissingExecutionControls => assert!(
                detail.contains("original media native recipe requires its request funding and execution-control join"),
                "complete reduction still requires actual native request controls: {detail}",
            ),
            _ => unreachable!(),
        }
        assert!(
            pool.used_bytes().unwrap() > used,
            "the owning diagnostic retains its actual planning charge"
        );
    } else {
        assert_eq!(
            pool.used_bytes().unwrap(),
            used,
            "fixed refusal allocates no planning account: manual={manual}, rejection={rejection:?}, error={error}"
        );
        assert!(
            crate::composition::mlx::session::model_session::text_quote::take_original_cold_facts()
                .is_none(),
            "earlier source/capture failure must not derive facts: manual={manual}, rejection={rejection:?}, error={error}"
        );
    }
    // Keep the independently supplied B alive while testing the error's last
    // planning alias. Its retirement cannot be mistaken for source-B refund.
    drop(error);
    assert_eq!(
        pool.used_bytes().unwrap(),
        used,
        "last diagnostic owner retires all planning metadata"
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), owners);
    drop(source_hold);
    Some(rejection)
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[test]
fn original_prepared_core_claim_reaches_precise_native_gap_in_both_families_and_selected_residencies() {
    request_facts_matrix(false);
}
#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[test]
fn original_prepared_video_and_projected_media_keep_exact_canonical_attribution_in_selected_residencies()
{
    request_facts_matrix(true);
}
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(super) fn original_request_backend(pool: &WorkingMemoryPool) -> MlxBackend<'static> {
    use crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams;
    use crate::backend::{MlxAcceleratorFamily, MlxDeviceIdentity};
    // The preceding independent ordinary fixture can leave completed native
    // retirement nodes. Drain them before beginning another admitted factory.
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == 0
    });
    let streams = PreparedExecutionStreams::for_factory(pool)
        .unwrap_or_else(|cause| panic!("prepared stream initialization: {cause:?}; unquoted owners={:?}", pool.unquoted_owner_count()))
        .expect("original media requests require admitted native stream owners");
    let identity = MlxDeviceIdentity::from_realized_device(
        &safemlx::Device::new(safemlx::DeviceType::Gpu, 0),
        Some(MlxAcceleratorFamily::Metal),
    )
    .unwrap();
    MlxBackend::for_prepared_execution_plan(streams, identity)
}

// These admitted cases cover resident, ready-host and foreground-disk paths.
// The ordinary source fixture's disk mode enables background prefetch, whose
// worker/transfer ownership still requires a separate managed implementation.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(super) fn admitted_media_config(
    backend: &MlxBackend<'_>,
    path: &std::path::Path,
    mode: usize,
) -> crate::composition::mlx::loading::MlxModelConfig {
    if mode < 2 {
        return cold_config(backend, path, mode);
    }
    assert_eq!(mode, 2);
    use eredu_core::ModelLoadingBackend;
    use eredu_runtime::{DenseDiskStreamLoadOptions, NormalizedLoadRequest, WeightResidency};
    let inspection = eredu_core::inspect_artifact(path, backend.configuration_resolver()).unwrap();
    let weights = crate::MlxLoadRequest::from_normalized(
        NormalizedLoadRequest::default().with_weight_residency(
            WeightResidency::dense_disk_stream(
                DenseDiskStreamLoadOptions::new(1 << 26, 0, 0, 0).unwrap(),
            ),
        ),
    );
    eredu_core::prepare_inspected_model_config(backend, inspection, weights).unwrap()
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn request_facts_matrix(extended: bool) {
    // Admit the actual process/native owners before any fixture native work.
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    assert!(
        safemlx::metal::is_available().unwrap(),
        "explicit native Metal recipe prerequisite"
    );
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
            let backend = original_request_backend(&pool);
            let hidden = if conditional { 16 } else { 64 };
            let input = if extended {
                extended_source(&pool, hidden, conditional)
            } else {
                source(&pool, hidden)
            };
            let expected_media = if extended {
                if conditional { 15 } else { 12 }
            } else {
                4
            };
            let expected_decoder = 5 + expected_media;
            let stream = backend.stream().clone();
            let selected = admitted_media_config(&backend, root.path(), mode);
            let a = selected
                .prepared_sources()
                .plan_original_media_semantics(&input)
                .unwrap()
                .compile(&pool)
                .unwrap();
            let full = MlxPreparedInputMaterializer::prepare()
                .unwrap()
                .model_input_plan(&a)
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
            let prompt = full.bind(&runtime, a).unwrap();
            let compiles = COMPILES.get();
            let owners = pool.unquoted_owner_count().unwrap();
            let used = pool.used_bytes().unwrap();
            let (_, roots) = crate::tests::support::media_completion::observe(None, || {
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .inspection_adapters_for_test(eredu_core::InferenceGeometry {
                        batch_size: 1,
                        cached_positions: 0,
                        input_positions: expected_decoder,
                        max_output_tokens: 3,
                        prefill_chunk_positions: 2,
                        output: eredu_core::OutputDemand::LastPosition,
                    })
                    .unwrap();
            });
            assert!(roots.is_empty());
            assert_eq!(COMPILES.get(), compiles);
            assert_eq!(pool.used_bytes().unwrap(), used);
            assert_eq!(pool.unquoted_owner_count().unwrap(), owners);

            // Independent ordinary attribution retains its real ordinary owner.
            // The original prompt remains unchanged and gains no such authority.
            use eredu_core::{
                PreparedControlInput, PreparedControlInputBackend, PromptTokenAttribution,
            };
            let ordinary = MlxBackend::prepare_control_input(&runtime, prompt.clone()).unwrap();
            let attribution = ordinary.attribution();
            let input::OriginalMediaPacket::Original(packet) =
                prompt.original_media.as_ref().unwrap()
            else {
                unreachable!()
            };
            let semantics = packet.borrowed_semantics();
            let positions = semantics.request_position_facts().unwrap();
            assert!(positions.source().same_source(&input));
            assert_eq!(positions.decoder_positions(), attribution.decoder_positions);
            assert_eq!(
                (
                    positions.canonical_tokens(),
                    positions.non_tokenized_text_positions(),
                    positions.media_positions()
                ),
                (3, 2, expected_media)
            );
            let segments = positions.segments().collect::<Vec<_>>();
            assert_eq!(segments.len(), attribution.segments.len());
            for (a, b) in segments.iter().zip(&attribution.segments) {
                assert_eq!(a.plan, b.plan);
                assert_eq!(
                    a.canonical_range,
                    match &b.tokens {
                        PromptTokenAttribution::Canonical { range } => Some(*range),
                        PromptTokenAttribution::NotTokenized => None,
                    }
                );
            }
            assert_eq!(attribution.canonical_token_ids, [1, 2, 3]);
            let capability = runtime
                .session()
                .payload
                .model
                .erased()
                .capability_estimate();
            let input_count = positions
                .legacy_input_accounting(4.try_into().unwrap())
                .unwrap();
            let legacy_count = prompt
                .with_borrowed(|view| {
                    crate::composition::mlx::capability::count_prepared_input(
                        runtime.session(),
                        view,
                        &stream,
                    )
                })
                .unwrap();
            assert_eq!(
                legacy_count.text_tokens,
                input_count.text_tokens + positions.non_tokenized_text_positions()
            );
            assert_eq!(legacy_count.model_positions, input_count.model_positions);
            assert_eq!(legacy_count.media_positions, input_count.media_positions);
            assert_eq!(
                legacy_count.media_execution_workspace_bytes(),
                input_count.media_execution_workspace_bytes()
            );
            assert_eq!(
                legacy_count.media_execution_workspace_kind(),
                input_count.media_execution_workspace_kind()
            );
            // This independent attribution adapter owns an ordinary exclusion
            // lease. Its comparisons are finished; retire it before requesting
            // a real managed planning account for the unchanged original input.
            drop(attribution);
            drop(ordinary);
            let expected = legacy_state::estimate_runtime_state(
                capability.state_layout(),
                input_count,
                3,
                1,
                runtime.session().floating_state_dtype_bytes(),
            )
            .unwrap();
            let capability_address = capability as *const _ as usize;
            let blueprint_address = runtime
                .session()
                .payload
                .model
                .inference_blueprint()
                .unwrap() as *const _ as usize;
            let source_address = positions.source() as *const _ as usize;
            let before = runtime
                .session()
                .payload
                .model
                .erased()
                .original_request_media_binding()
                .unwrap();
            for manual in [false, true] {
                eprintln!("media admission case: conditional={conditional}, residency={mode}, controlled={manual}, extended={extended}");
                if let Some(refusal) =
                    attempt_preparation(&mut runtime, prompt.clone(), manual, None)
                {
                    assert!(matches!(
                        refusal,
                        R::MissingNumericalWorkspace | R::MissingExecutionControls
                    ));
                }
                let facts = crate::composition::mlx::session::model_session::text_quote::take_original_cold_facts().expect("actual native cold consumer executed");
                assert_eq!(
                    (
                        facts.canonical,
                        facts.projected_text,
                        facts.media,
                        facts.decoder,
                        facts.requested
                    ),
                    (3, 2, expected_media, expected_decoder, expected_decoder + 3)
                );
                assert_eq!(
                    (facts.fixed_bytes, facts.context_bytes),
                    (expected.fixed_state_bytes, expected.context_state_bytes)
                );
                assert_eq!(
                    facts.window_count,
                    expected.assumptions.sliding_window_bounds.len()
                );
                assert_eq!(
                    (facts.capability, facts.blueprint, facts.source),
                    (capability_address, blueprint_address, source_address)
                );
                assert!(facts.state_slots.is_some());
                assert!(!facts.complete);
                assert_eq!(
                    (
                        facts.output_width,
                        facts.native_backing,
                        facts.native_workspace
                    ),
                    (Some(64), None, None)
                );
                let after = runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .original_request_media_binding()
                    .unwrap();
                assert!(before.matches(&after));
                assert_eq!(after.frontier(), 0);
            }
            // An ordinary preinstalled collector is not an original capture
            // owner, even for an empty source. Explicit source options instead
            // enter the genuine original capture admission tested publicly.
            let discovery = MlxBackend::capture_discovery(&runtime).unwrap();
            let input::OriginalMediaPacket::Original(packet) =
                prompt.original_media.as_ref().unwrap()
            else {
                unreachable!()
            };
            let shape = packet.shape();
            let empty = empty_capture(&discovery, shape);
            let paths = runtime.session().payload.model.erased().shared_observation_paths().unwrap();
            let selection = paths.prepare_media_capture_selection(&empty).unwrap();
            let ordinary = eredu_runtime::capture::OrdinaryPrefillCapture::new(
                selection,
                eredu_core::InferenceGeometry {
                    batch_size: shape[0], cached_positions: 0, input_positions: shape[1],
                    max_output_tokens: 3, prefill_chunk_positions: 2.min(shape[1]),
                    output: eredu_core::OutputDemand::LastPosition,
                },
            ).unwrap();
            for manual in [false, true] {
                let mut installed = prompt.clone();
                installed.prepared_capture = Some(ordinary.clone());
                assert_eq!(
                    failure_with_options(
                        &mut runtime,
                        installed,
                        manual,
                        Some(TextPreparationOptions {
                            interventions: None, capture: Some(empty.clone())
                        })
                    ),
                    R::MissingCapture
                );
                assert!(crate::composition::mlx::session::model_session::text_quote::take_original_cold_facts().is_none(), "capture rejects before fact derivation");
            }
            if extended {
                continue;
            }
            // Existing ordinary execution still consumes the identical genuine
            // prepared packet through the shared selected media driver.
            let output = runtime
                .prefill(prompt.with_prefill_chunk_positions(2.try_into().unwrap()))
                .unwrap();
            output.completion.wait().unwrap();
            drop(output);
            assert!(
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .original_request_media_binding()
                    .unwrap()
                    .frontier()
                    > 0
            );
        }
    }
}

#[test]
fn original_prepared_source_substitution_and_stale_state_precede_missing_native_contribution() {
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let input = source(&pool, 64);
    let equal = source(&pool, 64);
    assert_eq!(input.content_digest(), equal.content_digest());
    assert!(!input.same_source(&equal));
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let selected = cold_config(&backend, root.path(), 0);
    let a = selected
        .prepared_sources()
        .plan_original_media_semantics(&input)
        .unwrap()
        .compile(&pool)
        .unwrap();
    let a_equal = selected
        .prepared_sources()
        .plan_original_media_semantics(&equal)
        .unwrap()
        .compile(&pool)
        .unwrap();
    let materializer = MlxPreparedInputMaterializer::prepare().unwrap();
    let full = materializer
        .model_input_plan(&a)
        .unwrap()
        .materialize(&pool)
        .unwrap();
    let full_equal = materializer
        .model_input_plan(&a_equal)
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
    let prompt = full.bind(&runtime, a).unwrap();
    let equal_prompt = full_equal.bind(&runtime, a_equal).unwrap();
    let mut substituted = prompt.clone();
    substituted.cache_identity = equal_prompt.cache_identity.clone();
    assert_eq!(
        failure(&mut runtime, substituted, false),
        R::IdentityMismatch
    );
    let mut substituted = prompt.clone();
    substituted.parts = equal_prompt.parts.clone();
    assert_eq!(
        failure(&mut runtime, substituted, true),
        R::IdentityMismatch
    );
    // A second real session over the same retained selected source has its own
    // execution/control/revision identities. Equal geometry never authenticates it.
    let other_backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let other_model = other_backend.prepare_model_borrowed(&selected).unwrap();
    let mut other = ModelRuntime::from_prepared(other_backend, other_model).unwrap();
    other
        .session()
        .payload
        .model
        .erased()
        .prepare_completed_media_binding_fixture()
        .unwrap();
    assert_eq!(
        failure(&mut other, prompt.clone(), false),
        R::IdentityMismatch
    );
    drop(other);
    let foreign_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let other_backend = MlxBackend::new(&stream, &stream).with_memory_pool(foreign_pool);
    let foreign_selected = cold_config(&other_backend, root.path(), 0);
    let other_model = other_backend
        .prepare_model_borrowed(&foreign_selected)
        .unwrap();
    let mut other = ModelRuntime::from_prepared(other_backend, other_model).unwrap();
    other
        .session()
        .payload
        .model
        .erased()
        .prepare_completed_media_binding_fixture()
        .unwrap();
    assert_eq!(
        failure(&mut other, prompt.clone(), true),
        R::IdentityMismatch
    );
    drop(other);
    let raw = prompt.with_borrowed(|view| MlxModelInput::from(view));
    assert_eq!(failure(&mut runtime, raw, false), R::SourceUnavailable);
    let lease = runtime
        .session()
        .authority
        .borrow_mut()
        .begin_submission()
        .unwrap();
    assert_eq!(failure(&mut runtime, prompt.clone(), true), R::Busy);
    drop(lease);
    let stale = prompt.clone();
    let output = runtime
        .prefill(prompt.with_prefill_chunk_positions(2.try_into().unwrap()))
        .unwrap();
    output.completion.wait().unwrap();
    drop(output);
    assert_eq!(failure(&mut runtime, stale, true), R::IdentityMismatch);
}

fn empty_capture(
    discovery: &eredu_core::capture::CaptureDiscovery,
    shape: [u64; 2],
) -> eredu_core::capture::SharedCapturePlan {
    use eredu_core::capture::*;
    let usage = CaptureUsage {
        captures: 16,
        retained_bytes: 1 << 20,
        host_bytes: 1 << 20,
        encoded_bytes: 1 << 20,
    };
    SharedCapturePlan::new(
        CapturePlan {
            schema_version: 1,
            selections: vec![],
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: shape[0],
                prompt_tokens: shape[1],
                max_predictions: 3,
            },
            CaptureTextOrigin {
                cached_positions: 0,
            },
        )
        .unwrap(),
    )
}

// Original host input, with separate real image/video grids and conditional
// projected media. Preparation vectors here are ordinary fixture setup; I's
// actual compile performs the original source comparison and copies its slots.
fn extended_source(
    pool: &WorkingMemoryPool,
    hidden: usize,
    conditional: bool,
) -> OriginalPreparedHostInput {
    use eredu_core::{InputMetadataKey, InputModality as M, InputPayloadKind as K};
    use eredu_runtime::input::host::{
        HostInputPart, HostTensorValues as V, HostTensorView, PreparedHostInputPlan,
    };
    let ids = [1u32, 2];
    let tail = [3u32];
    let image = (0..192)
        .map(|n| (n as f32 - 37.) / 113.)
        .collect::<Vec<_>>();
    let video = (0..384)
        .map(|n| (n as f32 - 91.) / 157.)
        .collect::<Vec<_>>();
    let projected = (0..2 * hidden)
        .map(|n| (n as f32 - 13.) / 29.)
        .collect::<Vec<_>>();
    let image_grid = [1i32, 4, 4];
    let video_grid = [2i32, 4, 4];
    let image_meta = [(
        InputMetadataKey::PatchGrid,
        HostTensorView {
            shape: &[1, 3],
            values: V::I32(&image_grid),
        },
    )];
    let video_meta = [(
        InputMetadataKey::PatchGrid,
        HostTensorView {
            shape: &[1, 3],
            values: V::I32(&video_grid),
        },
    )];
    let projected_shape = [1, 2, hidden];
    let projected_video_shape = [1, 1, hidden];
    let mut parts = vec![
        HostInputPart {
            modality: M::Text,
            kind: K::TokenIds,
            payload: HostTensorView {
                shape: &[1, 2],
                values: V::U32(&ids),
            },
            metadata: &[],
            extents: &[],
        },
        HostInputPart {
            modality: M::Image,
            kind: K::Tensor,
            payload: HostTensorView {
                shape: &[16, 12],
                values: V::F32(&image),
            },
            metadata: &image_meta,
            extents: &[],
        },
        HostInputPart {
            modality: M::Text,
            kind: K::Embeddings,
            payload: HostTensorView {
                shape: &projected_shape,
                values: V::F32(&projected),
            },
            metadata: &[],
            extents: &[],
        },
        HostInputPart {
            modality: M::Video,
            kind: K::Tensor,
            payload: HostTensorView {
                shape: &[32, 12],
                values: V::F32(&video),
            },
            metadata: &video_meta,
            extents: &[],
        },
    ];
    if conditional {
        parts.push(HostInputPart {
            modality: M::Image,
            kind: K::Embeddings,
            payload: HostTensorView {
                shape: &projected_shape,
                values: V::F32(&projected),
            },
            metadata: &[],
            extents: &[],
        });
        parts.push(HostInputPart {
            modality: M::Video,
            kind: K::Embeddings,
            payload: HostTensorView {
                shape: &projected_video_shape,
                values: V::F32(&projected[..hidden]),
            },
            metadata: &[],
            extents: &[],
        });
    }
    parts.push(HostInputPart {
        modality: M::Text,
        kind: K::TokenIds,
        payload: HostTensorView {
            shape: &[1, 1],
            values: V::U32(&tail),
        },
        metadata: &[],
        extents: &[],
    });
    pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
        .unwrap()
}
