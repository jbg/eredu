//! Actual loaded declarations; original-context equations only, no gate/admission.
use super::*;
use crate::composition::mlx::session::capture_workspace::CaptureWorkspaceObserver;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_core::{capture::*, TextGenerationBackend};
use eredu_nn::{
    workspace::*, LinearOperator, NeuralBackend, ProjectionInputObserver,
    RetainedGeneratedTensorFactory, Tensor,
};
use eredu_runtime::{layered::PreparedCaptureSelection, ActivationObserver};
fn source(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    path: &str,
    preview: u64,
    count: usize,
) -> SharedCapturePlan {
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let mut raw = CapturePlan::none();
    raw.selections = (0..count)
        .map(|i| CaptureSelection {
            id: i.to_string(),
            path: path.into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::Preview {
                max_elements: preview,
            },
        })
        .collect();
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    raw.limits.per_step = unlimited;
    raw.limits.cumulative = unlimited;
    SharedCapturePlan::new(
        raw.admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
        )
        .unwrap(),
    )
}
fn geometry(selection: &PreparedCaptureSelection) -> eredu_core::InferenceGeometry {
    eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 2,
        output: selection.physical_output(eredu_core::OutputDemand::LastPosition),
    }
}
fn retire(runtime: ModelRuntime<MlxBackend<'_>>, pool: &MemoryLedger, stream: &Stream) {
    runtime.synchronize().unwrap();
    drop(runtime);
    stream.synchronize().unwrap();
    safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
        .unwrap()
        .synchronize()
        .unwrap();
    settle(pool, 0);
}
#[test]
fn actual_resident_host_disk_bound_body_and_readout_quotes_keep_original_state_and_source() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for route in 0..3 {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let (runtime, _artifact) = match route {
            0 => host::runtime(&stream, &pool, None),
            1 => host::runtime(&stream, &pool, Some(1)),
            _ => disk::load_runtime(&stream, &pool, true),
        };
        for path in [
            "readout.embedding",
            eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
        ] {
            let source = source(&runtime, path, 5, 1);
            let paths = paths(&runtime);
            let selected = paths.prepare_capture_selection(&source).unwrap();
            let g = geometry(&selected);
            assert_eq!(
                g.output,
                if path == "readout.embedding" {
                    eredu_core::OutputDemand::LastPosition
                } else {
                    eredu_core::OutputDemand::Sequence
                }
            );
            let executable = &runtime.session().payload.model;
            let blueprint = executable.inference_blueprint().unwrap();
            let context = WorkspaceContext::new(executable.workspace_mechanisms().unwrap());
            let state = executable
                .erased()
                .project_resident_workspace(std::num::NonZeroU32::new(1).unwrap(), &context)
                .unwrap();
            let parameters = executable.layerwise_workspace().unwrap();
            let before = (
                pool.fixture_host_charge().unwrap(),
                pool.fixture_host_peak().unwrap(),
            );
            let frontier = executable.erased().state_snapshot();
            let (mut observer, h) = CaptureWorkspaceObserver::with_prefill(
                selected.bind_geometry(g).unwrap(),
                &context,
            )
            .unwrap();
            assert!(std::ptr::eq(h.source(), selected.source()));
            assert!(h.source().same_storage(&source));
            let report = match parameters.as_ref() {
                Some(p) => blueprint.quote_replicated_layerwise_text_observed(
                    g,
                    &state,
                    &context,
                    &Parameters(p),
                    &paths,
                    &mut observer,
                ),
                None => blueprint.quote_replicated_resident_text_observed(
                    g,
                    &state,
                    &context,
                    &paths,
                    &mut observer,
                ),
            }
            .unwrap();
            assert!(report.transient().bytes().is_some());
            assert_eq!(report.completed_spans(), 6);
            assert_eq!(executable.erased().state_snapshot(), frontier);
            assert_eq!(
                (
                    pool.fixture_host_charge().unwrap(),
                    pool.fixture_host_peak().unwrap()
                ),
                before
            );
        }
        retire(runtime, &pool, &stream);
    }
}
#[derive(Debug)]
struct Facts {
    missing: bool,
}
impl WorkspaceMechanisms for Facts {
    fn projection_input_observation_mechanism(
        &self,
        _: &eredu_nn::LinearFormatSpec,
    ) -> Result<Option<eredu_nn::ProjectionInputObservationMechanism>, eredu_nn::Error> {
        Ok(Some(
            eredu_nn::ProjectionInputObservationMechanism::BlockFp8Gpu,
        ))
    }
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        if self.missing && matches!(op.kind, WorkspaceOperationKind::BlockFp8ActivationDecode) {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|o| o.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<Vec<_>, _>>()?,
            scratch_bytes: 0,
            assumptions: "explicit all-allocated selected operator fixture".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "no host numerical staging in fixture".into(),
        }))
    }
}
struct Projection<'a, 'p>(&'a mut CaptureWorkspaceObserver<'p>);
impl ProjectionInputObserver<WorkspaceTensor> for Projection<'_, '_> {
    fn observe(&mut self, _: &WorkspaceTensor) -> Result<(), eredu_nn::Error> {
        panic!("retained FP8 producer")
    }
    fn observe_generated(
        &mut self,
        _: &WorkspaceTensor,
        _: &eredu_nn::GeneratedTensorSource,
        _: &mut dyn FnMut() -> Result<WorkspaceTensor, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        panic!("retained factory protocol required")
    }
    fn observe_generated_retained(
        &mut self,
        p: &WorkspaceTensor,
        s: &eredu_nn::GeneratedTensorSource,
        f: &mut dyn RetainedGeneratedTensorFactory<WorkspaceTensor, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        self.0.observe_generated_retained(
            "readout.projection_input",
            p,
            &eredu_runtime::capture::generated_capture_source(s),
            f,
        )
    }
}
fn linear(context: &WorkspaceContext, width: i32) -> WorkspaceLinear {
    use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat};
    WorkspaceBackend::linear(
        eredu_nn::LinearSpec {
            input: width,
            output: 3,
            weight: eredu_nn::ParameterSpec::trainable("matrix.weight").unwrap(),
            bias: None,
            format: eredu_nn::LinearFormatSpec::scaled(
                LinearFormat::E4M3BlockFp8(
                    BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
                ),
                eredu_nn::ParameterSpec::trainable("matrix.scales").unwrap(),
            )
            .unwrap(),
        },
        context,
    )
    .unwrap()
}
#[test]
fn bound_generated_projection_quotes_global_preview_and_all_intermediates_until_each_span_end() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = host::runtime(&stream, &pool, None);
    for preview in [0, 5] {
        for missing in [false, true] {
            let source = source(&runtime, "readout.projection_input", preview, 2);
            let paths = paths(&runtime);
            let selected = paths.prepare_capture_selection(&source).unwrap();
            let g = geometry(&selected);
            let context = WorkspaceContext::new(Facts { missing });
            let (mut observer, _) = CaptureWorkspaceObserver::with_prefill(
                selected.bind_geometry(g).unwrap(),
                &context,
            )
            .unwrap();
            let assembly = CapturePrefillRowAssembly::prepare(source.admission(), 0, g).unwrap();
            let width = *assembly.logical_geometry().source_shape().last().unwrap() as i32;
            let mut total_decodes = 0;
            for k in 0..assembly.chunk_count() {
                let f = assembly.fragment(k).unwrap();
                let chunk = eredu_runtime::prefill::PrefillChunk {
                    input: f.input().clone(),
                    position: f.position(),
                    output: f.output_demand(),
                };
                let span = InferenceWorkspaceSpan::Prefill(chunk);
                assert!(observer.begin_span(g, &span, 0, &context).unwrap());
                let shape = f
                    .source_shape()
                    .iter()
                    .map(|&n| n as i32)
                    .collect::<Vec<_>>();
                let input = WorkspaceTensor::existing(
                    WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap(),
                    &context,
                )
                .unwrap();
                let mut linear = linear(&context, width);
                let output = linear
                    .forward_with_input_observer(
                        &input,
                        &context,
                        Some(&mut Projection(&mut observer)),
                    )
                    .unwrap();
                let mut roots = vec![output];
                observer.visit_retained(&mut |v| roots.push(v.clone()));
                assert!(
                    roots.len() >= 10,
                    "actual two compact plus seven outputs and selection roots"
                );
                let report = context.report(&roots).unwrap();
                let decodes = report
                    .operations
                    .iter()
                    .filter(|o| matches!(o.kind, WorkspaceOperationKind::BlockFp8ActivationDecode))
                    .count();
                assert_eq!(
                    decodes,
                    total_decodes + 1,
                    "two selections share one physical factory"
                );
                total_decodes = decodes;
                assert_eq!(report.total_bytes.is_some(), !missing);
                drop((roots, report));
                observer.end_span(&span, &context).unwrap();
                let mut retained = 0;
                observer.visit_retained(&mut |_| retained += 1);
                assert_eq!(retained, 0);
            }
            assert_eq!(total_decodes, 2);
        }
    }
    retire(runtime, &pool, &stream);
}

#[test]
fn changed_candidate_and_equal_content_foreign_paths_reject_before_equations() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let other_pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = host::runtime(&stream, &pool, None);
    let (other, _other_artifact) = host::runtime(&stream, &other_pool, None);
    {
        let source = source(&runtime, "readout.embedding", 5, 1);
        let actual_paths = paths(&runtime);
        let foreign_paths = paths(&other);
        assert!(!actual_paths.same_storage(&foreign_paths));
        let selected = actual_paths.prepare_capture_selection(&source).unwrap();
        let g = geometry(&selected);
        let executable = &runtime.session().payload.model;
        let blueprint = executable.inference_blueprint().unwrap();
        for foreign in [false, true] {
            let context = WorkspaceContext::new(executable.workspace_mechanisms().unwrap());
            let state = executable
                .erased()
                .project_resident_workspace(std::num::NonZeroU32::new(1).unwrap(), &context)
                .unwrap();
            let (mut observer, _) = CaptureWorkspaceObserver::with_prefill(
                selected.bind_geometry(g).unwrap(),
                &context,
            )
            .unwrap();
            let before = context.report(&[]).unwrap().operations.len();
            let changed = if foreign {
                g
            } else {
                eredu_core::InferenceGeometry {
                    prefill_chunk_positions: 1,
                    ..g
                }
            };
            let error = blueprint
                .quote_replicated_resident_text_observed(
                    changed,
                    &state,
                    &context,
                    if foreign {
                        &foreign_paths
                    } else {
                        &actual_paths
                    },
                    &mut observer,
                )
                .unwrap_err();
            // PreparedExecutionError retains its generic backend value directly;
            // follow the same explicit variant match as neutral quote tests.
            let eredu_architectures::prepared_execution::PreparedExecutionError::Backend(inner) =
                &error
            else {
                panic!("typed backend rejection expected: {error}");
            };
            let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(inner);
            let mut found = false;
            while let Some(error) = cause {
                found |= matches!(
                    error.downcast_ref::<eredu_runtime::layered::PreparedCaptureSelectionError>(),
                    Some(eredu_runtime::layered::PreparedCaptureSelectionError::Identity)
                );
                cause = error.source();
            }
            assert!(found, "preserved typed source/candidate rejection");
            assert_eq!(context.report(&[]).unwrap().operations.len(), before);
        }
    }
    retire(runtime, &pool, &stream);
    retire(other, &other_pool, &stream);
}

mod model_route;
