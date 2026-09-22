//! Numerical media comparisons enter the complete source and ordinary core driver.
use super::*;
use eredu_core::capture::*;
use eredu_core::{
    ControlledTextGeneration, GenerationSequenceRequest, TextGeneration, TextGenerationBackend,
    TextGenerationConfig, TextGenerationInput, TextPreparationOptions, TokenFilter,
    TokenFilterController,
};
use eredu_runtime::input::OriginalModelInputBackend;

pub(in crate::composition::mlx::session::model_session::original_host_input) fn enter() -> bool {
    if !crate::tests::support::native_process::enter("original-host-input") {
        return false;
    }
    crate::tests::support::test_utils::initialize_original_sources();
    true
}
pub(in crate::composition::mlx::session::model_session::original_host_input) struct All;
impl TokenFilterController for All {
    type Error = std::convert::Infallible;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }
    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        static FILTER: TokenFilter = TokenFilter::All;
        Some(eredu_core::TextControllerWorkspace {
            filter: (&FILTER).into(),
            additional_host_bytes: 0,
        })
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
pub(in crate::composition::mlx::session::model_session::original_host_input) fn config(
    predictions: usize,
) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                do_sample: Some(false),
                max_new_tokens: Some(predictions),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}
pub(in crate::composition::mlx::session::model_session::original_host_input) fn backend(
    pool: &MemoryLedger,
) -> MlxBackend<'static> {
    use crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams;
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == 0
    });
    let device = if cfg!(feature = "metal") {
        DeviceType::Gpu
    } else {
        DeviceType::Cpu
    };
    let streams = PreparedExecutionStreams::for_device_factory(pool, device)
        .unwrap()
        .unwrap();
    let identity = crate::backend::MlxDeviceIdentity::from_realized_device(
        &Device::new(device, 0),
        (device == DeviceType::Gpu).then_some(crate::backend::MlxAcceleratorFamily::Metal),
    )
    .unwrap();
    MlxBackend::for_prepared_execution_plan(streams, identity)
}
pub(in crate::composition::mlx::session::model_session::original_host_input) fn selected(
    backend: &MlxBackend<'_>,
    path: &std::path::Path,
    mode: usize,
) -> crate::composition::mlx::loading::MlxModelConfig {
    if mode != 2 {
        return cold_config(backend, path, mode);
    }
    use eredu_core::ModelLoadingBackend;
    let inspection = eredu_core::inspect_artifact(path, backend.configuration_resolver()).unwrap();
    let weights = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::dense_disk_stream(
                eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 26, 0, 0, 0).unwrap(),
            ),
        ),
    );
    eredu_core::prepare_inspected_model_config(backend, inspection, weights).unwrap()
}
pub(in crate::composition::mlx::session::model_session::original_host_input) type State = (
    Vec<(Vec<i32>, Vec<f32>)>,
    Vec<(
        usize,
        eredu_core::cache::StateTensorRole,
        Vec<i32>,
        Vec<f32>,
    )>,
);
pub(in crate::composition::mlx::session::model_session::original_host_input) type Report =
    (Vec<Vec<f64>>, Vec<State>);
pub(in crate::composition::mlx::session::model_session::original_host_input) fn snapshot(
    runtime: &ModelRuntime<MlxBackend<'_>>,
) -> State {
    runtime.session().ensure_no_submission_in_flight().unwrap();
    let target = runtime.session().payload.model.erased();
    (
        target.retained_numeric_state_snapshot().unwrap().unwrap(),
        target.fixed_numeric_state_snapshot().unwrap(),
    )
}
fn capture(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    positions: u64,
    predictions: usize,
) -> SharedCapturePlan {
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let usage = CaptureUsage {
        captures: predictions as u64,
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    let mut plan = CapturePlan::none();
    plan.selections.push(CaptureSelection {
        id: "source-logits".into(),
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    plan.limits.per_step = usage;
    plan.limits.cumulative = usage;
    SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: positions,
                max_predictions: predictions as u64,
            },
            CaptureTextOrigin {
                cached_positions: runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .original_request_media_binding()
                    .unwrap()
                    .frontier(),
            },
        )
        .unwrap(),
    )
}
fn logits(delivery: SharedCapturedStep) -> Vec<f64> {
    let records = &delivery.as_step().records;
    assert_eq!(records.len(), 1);
    let tensor = records[0].payload.as_ref().unwrap().as_tensor().unwrap();
    let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
        panic!("F32 logits")
    };
    let width = *tensor.shape().last().unwrap();
    values[values.len() - width..]
        .iter()
        .map(|value| f64::from(*value))
        .collect()
}
pub(in crate::composition::mlx::session::model_session::original_host_input) fn collect(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    prompt: MlxModelInput,
    predictions: usize,
    manual: bool,
) -> Report {
    let positions = MlxBackend::original_model_input_semantics(&prompt)
        .unwrap()
        .layout()
        .positions() as u64;
    collect_input(
        runtime,
        TextGenerationInput::OriginalPrepared(prompt),
        positions,
        predictions,
        manual,
        GenerationSequenceRequest::new(predictions, &[]),
    )
}
fn collect_input(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    input: TextGenerationInput<MlxModelInput>,
    positions: u64,
    predictions: usize,
    manual: bool,
    request: GenerationSequenceRequest<'_>,
) -> Report {
    let options = Some(TextPreparationOptions {
        capture: Some(capture(runtime, positions, predictions)),
        interventions: None,
    });
    let mut outputs = Vec::new();
    let mut states = Vec::new();
    if manual {
        let mut generation = ControlledTextGeneration::from_input_with_sequence(
            runtime,
            input,
            config(predictions),
            All,
            options,
            request,
        )
        .unwrap();
        for _ in 0..predictions {
            drop(
                generation
                    .next()
                    .unwrap()
                    .unwrap_or_else(|cause| operation_failure(&cause)),
            );
            outputs.push(logits(
                generation.take_captured_delivery().unwrap().unwrap(),
            ));
            states.push(snapshot(generation.snapshot_source().unwrap().parts().0));
        }
        assert!(generation.next().is_none());
    } else {
        let mut generation = TextGeneration::from_input_with_sequence(
            runtime,
            input,
            config(predictions),
            TokenFilter::All,
            options,
            request,
        )
        .unwrap();
        for _ in 0..predictions {
            drop(
                generation
                    .next()
                    .unwrap()
                    .unwrap_or_else(|cause| operation_failure(&cause)),
            );
            outputs.push(logits(
                generation.take_captured_delivery().unwrap().unwrap(),
            ));
        }
        assert!(generation.next().is_none());
        drop(generation);
        states.push(snapshot(runtime));
    }
    assert!(outputs
        .iter()
        .all(|values| values.iter().any(|value| value.abs() > 1e-5)));
    assert!(states.iter().all(|state| !state.0.is_empty()));
    (outputs, states)
}
fn operation_failure(cause: &(dyn std::error::Error + 'static)) -> ! {
    let mut source = Some(cause);
    while let Some(error) = source {
        eprintln!("media operation: {error}");
        source = error.source();
    }
    panic!("admitted media operation failed")
}
pub(in crate::composition::mlx::session::model_session::original_host_input) fn prepare(
    path: &std::path::Path,
    mode: usize,
    hidden: usize,
) -> (ModelRuntime<MlxBackend<'static>>, MlxModelInput) {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = backend(&pool);
    let source = source(&pool, hidden);
    let selected = selected(&backend, path, mode);
    let semantics = selected
        .prepared_sources()
        .plan_original_media_semantics(&source)
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
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    runtime
        .session()
        .payload
        .model
        .erased()
        .prepare_completed_media_binding_fixture()
        .unwrap();
    let prompt = completed.bind(&runtime, semantics).unwrap();
    (runtime, prompt)
}
pub(in crate::composition::mlx::session::model_session::original_host_input) fn run(
    path: &std::path::Path,
    mode: usize,
    hidden: usize,
    whole_prompt: bool,
    manual: bool,
) -> Report {
    let (mut runtime, prompt) = prepare(path, mode, hidden);
    let chunk = if whole_prompt { 9 } else { 2 };
    collect(
        &mut runtime,
        prompt.with_prefill_chunk_positions(chunk.try_into().unwrap()),
        4,
        manual,
    )
}
pub(in crate::composition::mlx::session::model_session::original_host_input) fn continue_text(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
) -> Report {
    let plan = eredu_core::TokenIdsInputPlan::new(&[4]).unwrap();
    collect_input(
        runtime,
        TextGenerationInput::OriginalTokenIds,
        1,
        3,
        true,
        GenerationSequenceRequest::new(3, &[]).with_token_input(&plan),
    )
}
pub(in crate::composition::mlx::session::model_session::original_host_input) fn rejects_upload(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    prompt: &MlxModelInput,
) {
    let frontier = runtime.session().payload.model.erased().state_snapshot();
    for finite in [false, true] {
        let mut policy = config(4).inference_policy().clone();
        policy.memory_limits = if finite {
            crate::memory_fixture::limits(u64::MAX)
        } else {
            Default::default()
        };
        let config = config(4).with_inference_policy(policy);
        let error = MlxBackend::admit_text_preparation(
            runtime,
            &eredu_core::TextPreparationInput::Prepared(prompt),
            config.clone(),
            &All,
        )
        .err()
        .unwrap();
        assert!(caused_by::<WorkingMemoryError>(&error)
            .is_some_and(|cause| matches!(cause, WorkingMemoryError::UnknownBound)));
        let error = ControlledTextGeneration::from_input_with_sequence(
            runtime,
            TextGenerationInput::OriginalPrepared(prompt.clone()),
            config,
            All,
            None,
            GenerationSequenceRequest::new(4, &[]),
        )
        .err()
        .unwrap();
        assert!(
            caused_by::<eredu_core::PreparedRequestRejection>(&error).is_some_and(
                |cause| *cause == eredu_core::PreparedRequestRejection::SourceUnavailable
            )
        );
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            frontier
        );
    }
}
fn caused_by<'a, E: std::error::Error + 'static>(
    error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a E> {
    let mut error = Some(error);
    while let Some(cause) = error {
        if let Some(value) = cause.downcast_ref::<E>() {
            return Some(value);
        }
        error = cause.source();
    }
    None
}
