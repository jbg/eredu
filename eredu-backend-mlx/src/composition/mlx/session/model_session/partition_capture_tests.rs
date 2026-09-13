use super::*;
use eredu_core::{capture::*, component::ComponentCoordinateMap, BackendProvider as _};
use eredu_runtime::capture::{partition::*, CaptureSession};

// This fixture verifies the actual native forward/observer lifecycle with a
// singleton transport. Multi-process subgroup participation has separate Ring
// coverage; this is not a claim of public distributed generation support.
struct SingletonLayout;
impl PartitionCaptureLayout for SingletonLayout {
    fn capture_placement(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        self.capture_placement_at(plan, index, phase, prediction, None, limits)
    }
    fn capture_placement_at(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        let point = &plan.points()[index];
        let axis = point
            .axes
            .as_ref()
            .unwrap()
            .iter()
            .position(|axis| axis.name == "component")
            .unwrap();
        let shape = plan
            .geometry_at(phase, prediction, invocation)?
            .resolve(point)?
            .unwrap();
        let slice = resolve_slice(point, &plan.plan().selections[index], &shape)?;
        let map = ComponentCoordinateMap::range(shape[axis] as usize, 0..shape[axis] as usize)
            .map_err(|error| CaptureError::Invalid(error.to_string()))?;
        Ok(PartitionCapturePlacement {
            hook_members: vec![0],
            source_shapes: vec![shape.clone()],
            producers: vec![PartitionCaptureProducer {
                rank: 0,
                projection: CaptureSlicePartition::new(
                    &shape,
                    &slice,
                    axis,
                    &map,
                    limits.max_fragments,
                )?,
            }],
        })
    }
}

#[test]
fn native_partition_observer_commits_real_component_prefill_and_cached_decode() {
    verify_native_partition_components(false, false, safemlx::DeviceType::Cpu);
}

#[test]
fn native_partition_invocation_preserves_multirow_cached_geometry() {
    verify_native_partition_components(true, false, safemlx::DeviceType::Cpu);
}

#[test]
#[ignore = "requires Metal device access"]
fn native_partition_invocation_preserves_multirow_cached_geometry_metal() {
    verify_native_partition_components(true, false, safemlx::DeviceType::Gpu);
}

#[test]
fn native_partition_speculative_provider_preserves_real_components_cpu() {
    verify_native_partition_components(true, true, safemlx::DeviceType::Cpu);
}
#[test]
#[ignore = "requires Metal device access"]
fn native_partition_speculative_provider_preserves_real_components_metal() {
    verify_native_partition_components(true, true, safemlx::DeviceType::Gpu);
}
impl eredu_runtime::intervention::PartitionActivationLayout for SingletonLayout {
    fn activation_members<'a>(
        &'a self,
        _: &'a eredu_core::intervention::AdmittedInterventionPlan,
        _: usize,
        _: CapturePhase,
        _: u64,
        _: usize,
        _: usize,
    ) -> Result<Vec<eredu_runtime::intervention::PartitionActivationMember<'a>>, CaptureError> {
        panic!("this fixture admits capture only")
    }
}

fn verify_native_partition_components(
    independent: bool,
    speculative: bool,
    device: safemlx::DeviceType,
) {
    use eredu_core::speculative::{
        SpeculativeActivationOrigin, SpeculativeActivationPhase, SpeculativeCaptureScope,
    };
    use eredu_runtime::inspection::{with_speculative_activation, SpeculativeActivationObserver};

    let context = crate::backend::ExecutionContext::new(safemlx::Device::new(device, 0));
    let source_context =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let stream = context.stream();
    let backend = MlxBackend::new(stream, source_context.stream());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let manifest = eredu_runtime::CommunicationManifest::new(1, 0, vec![], vec![])
        .unwrap()
        .with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(5),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    let transport = MlxDistributedSession::from_manifest(&manifest, &world, stream).unwrap();
    let mut results = Vec::new();
    for partitioned in [false, true] {
        let prepared =
            eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default())
                .unwrap();
        let mut session = backend.create_session(prepared).unwrap();
        let discovery = session
            .capture_discovery
            .as_ref()
            .unwrap()
            .capture()
            .unwrap();
        let budget = CaptureUsage {
            captures: 1000,
            retained_bytes: 100_000_000,
            host_bytes: 100_000_000,
            encoded_bytes: 100_000_000,
        };
        let selections = discovery
            .catalog
            .points
            .iter()
            .filter_map(|point| {
                let component = point
                    .axes
                    .as_ref()?
                    .iter()
                    .find(|axis| axis.name == "component")?;
                let eredu_core::SymbolicDimension::Known(width) = component.dimension else {
                    return None;
                };
                Some(CaptureSelection {
                    id: point.path.clone(),
                    path: point.path.clone(),
                    schedule: CaptureSchedule::default(),
                    transform: CaptureTransform::Slice,
                    slices: vec![CaptureSlice {
                        axis: "component".into(),
                        start: 0,
                        end: width as u64,
                        stride: 3,
                    }],
                })
            })
            .collect::<Vec<_>>();
        assert!(
            selections.len() >= 4,
            "fixture includes actual and effective FFN/attention components"
        );
        let plan = CapturePlan {
            schema_version: 1,
            selections,
            limits: CaptureLimits {
                per_step: budget,
                cumulative: budget,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        };
        let plan = if independent {
            plan.admit_invocations(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                CaptureInvocationBounds {
                    batch: 1,
                    max_sequence: 3,
                    max_context: if speculative { None } else { Some(6) },
                    max_predictions: 8,
                },
            )
        } else {
            plan.admit(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 2,
                    max_predictions: 3,
                },
            )
        }
        .unwrap();
        let count = plan.points().len();
        let mut capture = Some(CaptureSession::new(plan));
        let identity = PartitionCaptureIdentity::for_session(
            discovery.artifact_identity,
            session
                .capture_discovery
                .as_ref()
                .unwrap()
                .execution_identity()
                .into(),
            transport.session_identity(),
            None,
        )
        .unwrap();
        let mut collector: Option<Box<dyn SpeculativeActivationObserver<MlxTensor, Error>>> =
            if partitioned && speculative {
                let provider = eredu_runtime::capture::PartitionCaptureBackendProvider::new(
                    super::super::bounded_capture::SpeculativeCaptureProvider::new(
                        stream.clone(),
                        Some(transport.clone()),
                    ),
                    std::sync::Arc::new(transport.clone()),
                    std::sync::Arc::new(SingletonLayout),
                    identity,
                    PartitionCaptureReceiptLimits {
                        max_producers: 1,
                        max_fragments: 16,
                        // Identity-bound providers reserve the conservative encoded
                        // fragment bound before work, including strided slices.
                        max_record_bytes: 1 << 16,
                    },
                    |shape: &[u64], selection: &CaptureSelection, slice: &ResolvedCaptureSlice| {
                        Ok(PartitionCaptureNativeEstimate {
                            capture: super::super::bounded_capture::estimate_shape(
                                shape, selection, slice,
                            )?,
                            generated_creation_bytes: 0,
                        })
                    },
                    |error: PartitionCaptureObserverError<Error>| match error {
                        PartitionCaptureObserverError::Capture(error) => error,
                        error => eredu_runtime::capture::CaptureExecutionError::Backend(
                            Error::observation(error),
                        ),
                    },
                );
                Some(Box::new(
                    eredu_runtime::capture::SpeculativeCaptureObserver::new(
                        capture.take().unwrap(),
                        provider,
                        |error: &eredu_runtime::capture::CaptureExecutionError<Error>| {
                            Error::ArchitectureModel(error.to_string())
                        },
                        eredu_core::SpeculativeRequestId::new(0),
                        vec![SpeculativeCaptureScope::Target; count],
                        vec![],
                    )
                    .unwrap(),
                ))
            } else {
                if partitioned {
                    capture
                        .as_mut()
                        .unwrap()
                        .configure_partition_capture(identity)
                        .unwrap();
                }
                None
            };
        let mut steps = Vec::new();
        let mut logits = Vec::new();
        for step_index in 0..3 {
            let prediction = if independent { 4 } else { step_index };
            let phase = if step_index == 0 {
                CapturePhase::Prefill
            } else {
                CapturePhase::Decode
            };
            let sequence = if step_index == 0 {
                2
            } else if independent && step_index == 1 {
                3
            } else {
                1
            };
            let invocation = independent.then_some(CaptureInvocationShape {
                batch: 1,
                sequence,
                context: (!speculative).then_some(if step_index == 0 {
                    2
                } else if step_index == 1 {
                    5
                } else {
                    6
                }),
            });
            if let Some(invocation) = invocation.filter(|_| collector.is_none()) {
                capture
                    .as_mut()
                    .unwrap()
                    .begin_invocation(phase, prediction, invocation, Default::default())
                    .unwrap();
            }
            let mut submit = |observer: &mut dyn RuntimeActivationObserver<MlxTensor, Error>| -> Result<Vec<f32>, Error> {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            let submitted = if step_index == 0 {
                session.submit_prefill_with_observer(
                    &backend,
                    MlxBackend::prepare_text_prompt(&backend, vec![1, 2]).unwrap(),
                    &mut borrowed,
                )
            } else {
                session.submit_decode_with_observer(
                    &backend,
                    || {
                        Ok(Array::from_slice(
                            &(0..sequence)
                                .map(|row| step_index as u32 + 2 + row as u32)
                                .collect::<Vec<_>>(),
                            &[1, sequence as i32],
                        ))
                    },
                    &mut borrowed,
                )
            }
            ?;
            let output = submitted.wait()?;
            Ok(MlxTensor::from_array(output).to_f32_vec(stream)?)
            };
            let output = if let Some(collector) = collector.as_mut() {
                collector.set_activation_origin(Some(SpeculativeActivationOrigin {
                    request: eredu_core::SpeculativeRequestId::new(0),
                    committed_tokens: 0,
                    prediction: prediction as usize,
                    prefix_digest: [7; 32],
                    optimistic: false,
                }));
                with_speculative_activation(
                    Some(&mut **collector),
                    if step_index == 0 {
                        SpeculativeActivationPhase::TargetPrefill
                    } else {
                        SpeculativeActivationPhase::Verification
                    },
                    sequence as usize,
                    |observer| submit(observer.unwrap()),
                )
                .unwrap()
            } else {
                let mut observer: Box<dyn RuntimeActivationObserver<MlxTensor, Error> + '_> =
                    if partitioned {
                        Box::new(PartitionCaptureObserver::for_step(
                            capture.as_mut().unwrap(),
                            super::super::bounded_capture::NativeCapture {
                                partition: Some(&transport),
                                stream,
                                domain: None,
                            },
                            &transport,
                            &SingletonLayout,
                            prediction,
                            PartitionCaptureReceiptLimits {
                                max_producers: 1,
                                max_fragments: 16,
                                max_record_bytes: 8192,
                            },
                            |shape: &[u64],
                             selection: &CaptureSelection,
                             slice: &ResolvedCaptureSlice| {
                                Ok(PartitionCaptureNativeEstimate {
                                    capture: super::super::bounded_capture::estimate_shape(
                                        shape, selection, slice,
                                    )?,
                                    generated_creation_bytes: 0,
                                })
                            },
                            |error: PartitionCaptureObserverError<Error>| Error::observation(error),
                        ))
                    } else if independent {
                        Box::new(eredu_runtime::intervention::CaptureObserver::new(
                            capture.as_mut().unwrap(),
                            super::super::bounded_capture::NativeCapture {
                                partition: None,
                                stream,
                                domain: None,
                            },
                            super::super::bounded_capture::capture_error,
                        ))
                    } else {
                        Box::new(super::super::bounded_capture::observer(
                            capture.as_mut().unwrap(),
                            stream,
                            None,
                            prediction,
                        ))
                    };
                submit(&mut *observer).unwrap()
            };
            logits.push(output);
            let step = match collector.as_mut() {
                Some(collector) => {
                    let result = collector.take_activation_capture().unwrap();
                    assert!(result.completed);
                    assert_eq!(result.invocation, step_index);
                    result.captures
                }
                None => capture.as_mut().unwrap().take_step().unwrap(),
            };
            assert_eq!(step.prediction_index, prediction);
            assert_eq!(step.invocation, invocation);
            assert!(step
                .records
                .iter()
                .all(|record| record.outcome == CaptureOutcome::Captured));
            assert_eq!(
                step.partitions.len(),
                if partitioned { step.records.len() } else { 0 }
            );
            for evidence in &step.partitions {
                assert_eq!(evidence.context.prediction, prediction);
                assert_eq!(evidence.context.invocation, invocation);
                assert_eq!(
                    evidence.context.run_identity,
                    format!("capture:{}:1", transport.session_identity())
                );
                assert_eq!(evidence.context.phase, phase);
            }
            steps.push(step);
        }
        results.push((steps, logits));
    }
    assert_eq!(
        results[0].1, results[1].1,
        "capture must not change actual cached model execution"
    );
    let mut nonzero = false;
    for (ordinary, partitioned) in results[0].0.iter().zip(&results[1].0) {
        for (ordinary, partitioned) in ordinary.records.iter().zip(&partitioned.records) {
            assert_eq!(ordinary.payload, partitioned.payload);
            if let Some(CapturePayload::Tensor(value)) = &ordinary.payload {
                if let eredu_core::TensorObservationData::F32(values) = value.data() {
                    nonzero |= values.iter().any(|value| value.abs() > 1e-6);
                }
            }
        }
    }
    assert!(nonzero, "component fixture must exercise nonzero values");
}

#[derive(Default)]
struct InputPreparationProbe {
    prepared: usize,
    finished: Vec<(u64, bool)>,
}
impl RuntimeActivationObserver<MlxTensor, Error> for InputPreparationProbe {
    fn transactional(&self) -> bool {
        true
    }
    fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), Error> {
        Ok(())
    }
    fn prepare_transaction(
        &mut self,
        _: eredu_core::DistributedCommitEpoch,
        _: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        self.prepared += 1;
        Ok(())
    }
    fn finish_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch, committed: bool) {
        self.finished.push((epoch.value(), committed));
    }
}

#[test]
fn native_input_rejection_preserves_cause_cached_state_and_capture_epoch() {
    let context =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let stream = context.stream();
    let backend = MlxBackend::new(stream, stream);
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let mut results = Vec::new();
    for reject in [false, true] {
        let prepared =
            eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default())
                .unwrap();
        let mut session = backend.create_session(prepared).unwrap();
        session
            .submit_prefill_with_observer(
                &backend,
                MlxBackend::prepare_text_prompt(&backend, vec![1, 2]).unwrap(),
                &mut eredu_runtime::NoopObserver,
            )
            .unwrap()
            .wait()
            .unwrap();
        let mut observer = InputPreparationProbe::default();
        if reject {
            for prefill in [true, false] {
                let cause = || Error::Io(std::io::Error::other("native input preparation failed"));
                let result = if prefill {
                    session.submit_prefill_result_with_observer(
                        &backend,
                        Err(cause()),
                        None,
                        &mut observer,
                    )
                } else {
                    session.submit_decode_with_observer(&backend, || Err(cause()), &mut observer)
                };
                let error = match result {
                    Err(error) => error,
                    Ok(_) => panic!("invalid input entered model"),
                };
                assert!(error.model_state_preserved(), "{error}");
                let mut original: &(dyn std::error::Error + 'static) = &error;
                while let Some(source) = original.source() {
                    original = source;
                }
                let Some(Error::Io(cause)) = original.downcast_ref::<Error>() else {
                    panic!("native preparation cause was not retained: {original:?}");
                };
                assert_eq!(cause.to_string(), "native input preparation failed");
                super::super::recovery::wait_for_retirement(|| {
                    session.ensure_no_submission_in_flight().is_ok()
                });
            }
            assert_eq!(observer.prepared, 0);
            assert_eq!(observer.finished, [(2, false), (3, false)]);
        }
        let output = session
            .submit_decode_with_observer(
                &backend,
                || Ok(Array::from_slice(&[3u32], &[1, 1])),
                &mut observer,
            )
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(observer.prepared, 1);
        assert_eq!(
            observer.finished.last(),
            Some(&(if reject { 4 } else { 2 }, true))
        );
        results.push(MlxTensor::from_array(output).to_f32_vec(stream).unwrap());
    }
    assert!(results[0].iter().any(|value| value.abs() > 0.01));
    assert_eq!(
        results[0], results[1],
        "rejected prefill and decode preserve the existing KV state"
    );
}

#[test]
fn ordinary_generation_prefill_revalidates_exact_core_session_admission() {
    let context =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let stream = context.stream();
    let backend = MlxBackend::new(stream, stream);
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let prepared =
        eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, prepared).unwrap();
    let admitted = runtime.capabilities();
    runtime.session_mut().capabilities =
        admitted.with_activation_inspection(!admitted.activation_inspection());
    let config = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            temperature: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    let mut state =
        MlxBackend::start_text_generation(runtime.backend(), TextGenerationConfig::new(config))
            .unwrap();
    let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2]).unwrap();
    let before = crate::tests::support::path_instrumentation::snapshot().forwards;
    let error = match MlxBackend::submit_text_prefill(
        &mut runtime,
        prompt,
        &TokenFilter::All,
        &mut state,
    ) {
        Err(error) => error,
        Ok(_) => panic!("changed session admission entered native generation"),
    };
    assert!(error.to_string().contains("capabilities"), "{error}");
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot().forwards,
        before
    );
    runtime.session_mut().capabilities = admitted;
    let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2]).unwrap();
    MlxBackend::submit_text_prefill(&mut runtime, prompt, &TokenFilter::All, &mut state)
        .unwrap()
        .wait()
        .unwrap();
    runtime.synchronize().unwrap();
}
