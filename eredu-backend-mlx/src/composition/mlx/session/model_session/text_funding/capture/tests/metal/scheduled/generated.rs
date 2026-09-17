use super::*;
use crate::composition::mlx::session::capture_workspace::CaptureWorkspaceObserver;
use crate::{
    backend::nn::linear::{NativeProjectionInputObserver, PhysicalLinear},
    module::Module,
};
use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat};
use eredu_nn::{
    workspace::*, LinearOperator, NeuralBackend, ParameterSpec, ProjectionInputObserver,
    RetainedGeneratedTensorFactory,
};
use eredu_runtime::working_memory::{InferenceWorkspaceObserver, InferenceWorkspaceSpan};
use safemlx::error::Exception;

fn selected_source(width: usize, preview: Option<u64>, empty: bool) -> SharedCapturePlan {
    let a = admitted(
        vec![SymbolicDimension::Known(2), SymbolicDimension::Known(width)],
        preview.map_or(CaptureTransform::FullTensor, |max_elements| {
            CaptureTransform::Preview { max_elements }
        }),
        if empty {
            vec![CaptureSlice {
                axis: "axis1".into(),
                start: 0,
                end: 0,
                stride: 1,
            }]
        } else {
            vec![]
        },
    );
    let mut raw = a.plan().clone();
    raw.selections[0].schedule.prefill = false;
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: "block.output".into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    SharedCapturePlan::new(
        raw.admit(
            &ObservationCatalog {
                schema_version: 1,
                points: a.points().to_vec(),
                completeness: DescriptionCompleteness::Complete,
            },
            &support,
            &CaptureCapabilities {
                transformations: vec![
                    CaptureTransformKind::FullTensor,
                    CaptureTransformKind::Preview,
                ],
                max_histogram_bins: 0,
                physical_native_limit: false,
                conditions: vec![],
            },
            a.request().clone(),
        )
        .unwrap(),
    )
}
fn format() -> LinearFormat {
    LinearFormat::E4M3BlockFp8(
        BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
    )
}
fn module(width: i32, stream: &Stream) -> PhysicalLinear {
    let mut m = PhysicalLinear::unloaded(width, 2, false, format(), stream).unwrap();
    m.weight.value = Array::from_slice(&vec![0x38u8; 2 * width as usize], &[2, width]);
    m.weight_scale_inv.value = Some(Array::from_slice(
        &vec![1f32; (width as usize).div_ceil(128)],
        &[1, (width as usize).div_ceil(128) as i32],
    ));
    m
}
struct QuoteProjection<'a, 'p>(&'a mut CaptureWorkspaceObserver<'p>);
impl ProjectionInputObserver<WorkspaceTensor> for QuoteProjection<'_, '_> {
    fn observe(&mut self, _: &WorkspaceTensor) -> Result<(), eredu_nn::Error> {
        panic!("FP8")
    }
    fn observe_generated(
        &mut self,
        _: &WorkspaceTensor,
        _: &eredu_nn::GeneratedTensorSource,
        _: &mut dyn FnMut() -> Result<WorkspaceTensor, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        panic!("retained FP8 protocol required")
    }
    fn observe_generated_retained(
        &mut self,
        p: &WorkspaceTensor,
        s: &eredu_nn::GeneratedTensorSource,
        f: &mut dyn RetainedGeneratedTensorFactory<WorkspaceTensor, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        self.0.observe_generated_retained(
            "block.output",
            p,
            &eredu_runtime::capture::generated_capture_source(s),
            f,
        )
    }
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 3,
        output: OutputDemand::LastPosition,
    }
}
fn native_quote(input: &Array, source: &SharedCapturePlan) -> u64 {
    let c = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let mut p = ExistingArrayProjection::new(&c);
    let input = p.project(input).unwrap();
    let mut layer = WorkspaceBackend::linear(
        eredu_nn::LinearSpec {
            input: *input.shape().last().unwrap(),
            output: 2,
            weight: ParameterSpec::trainable("matrix.weight").unwrap(),
            bias: None,
            format: eredu_nn::LinearFormatSpec::scaled(
                format(),
                ParameterSpec::trainable("matrix.scales").unwrap(),
            )
            .unwrap(),
        },
        &c,
    )
    .unwrap();
    let (mut o, _) = CaptureWorkspaceObserver::new(source, geometry(), &c).unwrap();
    let prefill = InferenceWorkspaceSpan::Prefill(eredu_runtime::prefill::PrefillChunk {
        input: 0..3,
        position: 0,
        output: OutputDemand::LastPosition,
    });
    assert!(!o.begin_span(geometry(), &prefill, 0, &c).unwrap());
    let decode = InferenceWorkspaceSpan::Decode {
        index: 0,
        position: 3,
        output: OutputDemand::LastPosition,
    };
    c.begin_state_span([&input]).unwrap();
    o.begin_span(geometry(), &decode, 1, &c).unwrap();
    let output = layer
        .forward_with_input_observer(&input, &c, Some(&mut QuoteProjection(&mut o)))
        .unwrap();
    let mut roots = vec![output];
    o.visit_retained(&mut |v| roots.push(v.clone()));
    let report = c.report(&roots).unwrap();
    assert!(report.unpriced_operations.is_empty());
    assert!(report.unpriced_host_operations.is_empty());
    let bytes = report.total_bytes.unwrap();
    o.end_span(&decode, &c).unwrap();
    bytes
}
struct Count<'a> {
    factory: &'a mut dyn RetainedGeneratedTensorFactory<Array, Error>,
    calls: &'a mut usize,
}
impl RetainedGeneratedTensorFactory<Array, Error> for Count<'_> {
    fn program(&self) -> eredu_nn::GeneratedTensorProgram<'_> {
        self.factory.program()
    }
    fn visit_sources(
        &mut self,
        r: &mut dyn FnMut(eredu_nn::GeneratedTensorSourceRole, &Array) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.factory.visit_sources(r)
    }
    fn generate(&mut self, r: &mut dyn FnMut(&Array) -> Result<(), Error>) -> Result<Array, Error> {
        *self.calls += 1;
        self.factory.generate(r)
    }
}
struct NativeProjection<'a> {
    observer: &'a mut dyn ActivationObserver<Array, Error>,
    calls: usize,
    failure: Option<Error>,
}
impl NativeProjectionInputObserver for NativeProjection<'_> {
    fn observe(&mut self, _: &Array) -> Result<(), Exception> {
        panic!("FP8")
    }
    fn observe_generated(
        &mut self,
        _: &Array,
        _: &eredu_nn::GeneratedTensorSource,
        _: &mut dyn FnMut() -> Result<Array, Exception>,
    ) -> Result<(), Exception> {
        panic!("retained FP8 protocol required")
    }
    fn observe_generated_retained(
        &mut self,
        p: &Array,
        s: &eredu_nn::GeneratedTensorSource,
        f: &mut dyn RetainedGeneratedTensorFactory<Array, Exception>,
    ) -> Result<(), Exception> {
        fn borrowed(a: &Array) -> &Array {
            a
        }
        let mut mapped = eredu_nn::MappedGeneratedTensorFactory::new(
            f,
            borrowed,
            std::convert::identity::<Array>,
            Error::from,
            |_: &Error| Exception::from_source(eredu_nn::GeneratedTensorRetentionSignal),
        );
        let mut counted = Count {
            factory: &mut mapped,
            calls: &mut self.calls,
        };
        self.observer
            .observe_generated_retained(
                "block.output",
                p,
                &eredu_runtime::capture::generated_capture_source(s),
                &mut counted,
            )
            .map_err(|e| {
                self.failure = Some(e);
                Exception::from_source(eredu_nn::GeneratedTensorRetentionSignal)
            })
    }
}
fn execute(
    f: &mut SessionFixture,
    module: &mut PhysicalLinear,
    stream: &Stream,
) -> Result<(Array, usize), Error> {
    let mut native = NativeScheduledCapture::for_test(&f.work, stream);
    let epoch = DistributedCommitEpoch::new(1).unwrap();
    f.capture
        .with_observer(&mut native, 0, &observer_error, |o| {
            o.prepare_transaction(epoch, eredu_runtime::ExpertPass::Prefill)?;
            o.complete_transaction(epoch)?;
            o.finish_transaction(epoch, true);
            Ok::<_, Error>(())
        })
        .unwrap()?;
    drop(f.capture.take_shared_step().unwrap());
    let epoch = DistributedCommitEpoch::new(2).unwrap();
    let work = f.work.clone();
    f.capture
        .with_observer(&mut native, 1, &observer_error, |o| {
            // with_observer owns the lexical funded observer; its Drop aborts
            // every unfinished transaction on return or unwind.
            o.prepare_transaction(epoch, eredu_runtime::ExpertPass::Decode)?;
            let mut bridge = NativeProjection {
                observer: &mut *o,
                calls: 0,
                failure: None,
            };
            let result = module.forward_with_input_observer(&f.input, stream, Some(&mut bridge));
            let calls = bridge.calls;
            if let Some(error) = bridge.failure.take() {
                return Err(error);
            }
            drop(bridge);
            let output = result?;
            work.retain(&output);
            o.complete_transaction(epoch)?;
            o.finish_transaction(epoch, true);
            Ok((output, calls))
        })
        .unwrap()
}
fn fixture_generated(
    width: usize,
    dtype: Dtype,
    preview: Option<u64>,
    empty: bool,
    stream: &Stream,
) -> (SessionFixture, PhysicalLinear) {
    let bootstrap = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let bootstrap_owner = NativeMemoryOwner::acquire(&bootstrap).unwrap();
    let input = Array::from_slice(
        &(0..2 * width)
            .map(|i| (i % 17) as f32 * 0.25 - 2.0)
            .collect::<Vec<_>>(),
        &[2, width as i32],
    )
    .as_dtype(dtype, stream)
    .unwrap();
    let module = module(width as i32, stream);
    input.evaluated().unwrap();
    module.weight.value.evaluated().unwrap();
    module
        .weight_scale_inv
        .value
        .as_ref()
        .unwrap()
        .evaluated()
        .unwrap();
    let plan = selected_source(width, preview, empty);
    let h = CaptureRunHostPlan::prepare(&plan)
        .unwrap()
        .initialization_peak_bytes();
    let n = native_quote(&input, &plan);
    let mut initial = RetainedStorage::default();
    initial.include_array(&input).unwrap();
    initial.include_array(&module.weight.value).unwrap();
    initial
        .include_array(module.weight_scale_inv.value.as_ref().unwrap())
        .unwrap();
    initial.include_capture_plan(plan.clone()).unwrap();
    let baseline = initial.byte_bound().unwrap().unwrap();
    let pool = WorkingMemoryPool::new(baseline + h + n, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let sources = initial.publish_unquoted(&owner).unwrap();
    drop((owner, bootstrap_owner));
    let (r, run) = fresh(&pool, h + n);
    let bank = run
        .prepare_capture_run(&r, CaptureRunHostPlan::prepare(&plan).unwrap())
        .unwrap();
    let work = FundedWork::new(run.scope().unwrap());
    (
        SessionFixture {
            pool,
            input,
            _plan: plan,
            _reservation: r,
            _run: Some(run),
            work,
            capture: bank.into_capture_session().unwrap(),
            h,
            n,
            _sources: sources,
        },
        module,
    )
}
#[test]
fn actual_fp8_generated_capture_uses_original_account_and_shared_f32_custody() {
    for width in [1, 128, 259] {
        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let stream = stream();
            let (mut f, mut module) = fixture_generated(width, dtype, None, false, &stream);
            let ceiling = f.pool.effective_capacity().unwrap();
            assert_eq!(f.pool.used_bytes().unwrap(), ceiling);
            // Reuse the actual component's admitted diagnostics at a strictly short
            // independent ceiling. This creates no native state or copy permission.
            let short = WorkingMemoryPool::new(f.h + f.n - 1, 0).unwrap();
            assert!(matches!(
                short.reserve_with_capacity(
                    &InferenceExecutionIdentity::default(),
                    f._reservation.admission(),
                    short.effective_capacity().unwrap()
                ),
                Err(WorkingMemoryError::BudgetExceeded { .. })
            ));
            let (output, calls) = execute(&mut f, &mut module, &stream).unwrap();
            assert_eq!(calls, 1);
            assert!(f.work.roots.borrow().len() >= 10);
            assert!(f.work.scope.borrow().is_some());
            assert!(!f.work.published.get());
            assert!(f.pool.used_bytes().unwrap() <= ceiling);
            retire_native(&f.work);
            let step = f.capture.take_shared_step().unwrap().unwrap();
            assert_eq!(
                step.records()[0].source_dtype,
                Some(eredu_core::checkpoint::TensorDtype::F32)
            );
            let data = step.records()[0]
                .payload
                .as_ref()
                .unwrap()
                .as_tensor()
                .unwrap();
            assert_eq!(data.shape(), &[2, width]);
            assert!(matches!(data.data(),TensorObservationData::F32(v)if v.iter().any(|x|*x!=0.0)));
            let alias = step.clone();
            let pool = f.pool.clone();
            // Frame custody retains H; generated roots and FP8 operands below
            // have already been published and their work scope certified.
            let protected = f.h;
            drop((step, output, module, f));
            retire_records();
            settled_bytes(&pool, protected);
            drop(alias);
            retire_records();
            settled_bytes(&pool, 0);
        }
    }
}
#[test]
fn native_preview_zero_runs_factory_but_empty_slice_does_not() {
    for empty in [false, true] {
        let stream = stream();
        let (mut f, mut module) = fixture_generated(259, Dtype::Bfloat16, Some(0), empty, &stream);
        let (output, calls) = execute(&mut f, &mut module, &stream).unwrap();
        assert_eq!(calls, usize::from(!empty));
        retire_native(&f.work);
        let step = f.capture.take_shared_step().unwrap().unwrap();
        assert!(step.records()[0]
            .payload
            .as_ref()
            .unwrap()
            .as_tensor()
            .unwrap()
            .data()
            .is_empty());
        let pool = f.pool.clone();
        drop((step, output, module, f));
        retire_records();
        settled_bytes(&pool, 0);
    }
}
#[test]
fn late_native_failure_keeps_generated_roots_and_original_error_without_certification() {
    for panic in [false, true] {
        let stream = stream();
        let (mut f, mut module) = fixture_generated(259, Dtype::Float16, None, false, &stream);
        if panic {
            PANIC_PUBLICATION.set(true)
        } else {
            FAIL_PUBLICATION.set(true)
        }
        let result = catch_unwind(AssertUnwindSafe(|| execute(&mut f, &mut module, &stream)));
        if panic {
            assert!(result.is_err());
        } else {
            assert!(caused_by::<InjectedIngress>(&result.unwrap().unwrap_err()));
        }
        assert!(f.work.roots.borrow().len() >= 9);
        assert!(f.work.scope.borrow().is_some());
        assert!(!f.work.published.get());
        let step = f.capture.take_shared_step().unwrap().unwrap();
        assert_eq!(step.outcome(), CaptureStepOutcome::Aborted);
        let pool = f.pool.clone();
        drop((step, module, f));
        retire_records();
        assert!(pool.used_bytes().unwrap() > 0);
    }
}
