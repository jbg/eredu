use super::*;
use crate::backend::managed_memory::NativeMemoryOwner;
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, InferenceRequest, WorkingMemoryError, WorkingMemoryPool,
};
use eredu_runtime::{RoutedUnitBatch, RoutedUnitInvocation, RoutedUnitObserver};

fn authority() -> (
    WorkingMemoryPool,
    WorkingMemoryPool,
    ArrayObserverAllocationAuthority,
) {
    use eredu_core::{
        cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
        InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
        WorkspaceBound,
    };
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    // Synthetic isolated ownership evidence, not a native allocation bound.
    let bound = |bytes| WorkspaceBound::bounded(bytes, "observer charge ownership fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: bound(96),
        attention: bound(0),
        vocabulary: bound(0),
        state_update: bound(0),
        materialization: bound(0),
        retained: bound(0),
    })
    .unwrap();
    let admitted = Admission {
        requested_positions: 1,
        state,
        incremental_required_bytes: 96,
        available_memory_bytes: None,
    };
    let requests = WorkingMemoryPool::new(96, 0).unwrap();
    let native = WorkingMemoryPool::new(0, 0).unwrap();
    let request: InferenceRequest = requests
        .reserve(&InferenceExecutionIdentity::default(), &admitted)
        .unwrap()
        .into();
    let mut inference = InferenceRetention::new();
    inference.retain(&request);
    let owner = NativeMemoryOwner::acquire(&native).unwrap();
    let authority = ArrayObserverAllocationAuthority::new(
        NativeMemoryRetention::from_owner(&owner),
        &inference,
    );
    (requests, native, authority)
}

fn settled(requests: &WorkingMemoryPool, native: &WorkingMemoryPool, retained: bool) {
    let expected_bytes = if retained { 96 } else { 0 };
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        requests.used_bytes().unwrap() == expected_bytes
            && native.unquoted_owner_count().unwrap() == usize::from(retained)
    });
    if retained {
        assert!(matches!(
            requests.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
    }
}

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

fn lazy(stream: &Stream) -> Array {
    Array::from_slice(&[1_f32, 2., 3., 4., 5., 6., 7., 8.], &[8])
        .square(stream)
        .unwrap()
}

#[derive(Clone, Copy, Default)]
enum Failure {
    #[default]
    None,
    Error,
    Panic,
}

#[derive(Debug, thiserror::Error)]
#[error("observer failed after retaining its input")]
struct CallbackFailure;

#[derive(Default)]
struct Recorder {
    arrays: Vec<Array>,
    generate: bool,
    failure: Failure,
    replacement: Option<Array>,
}

impl Recorder {
    fn receive(&mut self, value: &MlxTensor) -> Result<(), Error> {
        self.arrays.push(value.as_array().clone());
        match self.failure {
            Failure::None => Ok(()),
            Failure::Error => Err(Error::Other(Box::new(CallbackFailure))),
            Failure::Panic => panic!("observer panic after retaining its input"),
        }
    }

    fn receive_batch(
        &mut self,
        batch: &RoutedUnitBatch<'_, MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        for value in [
            batch.units.values,
            batch.units.group_indices,
            batch.units.selection_indices,
            batch.units.token_indices,
            batch.units.coefficients,
            batch.source_groups,
        ] {
            self.receive(value)
                .map_err(eredu_nn::Error::backend_retained_source)?;
        }
        Ok(())
    }
}

impl RuntimeActivationObserver<MlxTensor, Error> for Recorder {
    fn observe(&mut self, _: &str, value: &MlxTensor) -> Result<(), Error> {
        self.receive(value)
    }
    fn observe_replica(&mut self, _: &str, value: &MlxTensor) -> Result<(), Error> {
        self.receive(value)
    }
    fn observe_generated(
        &mut self,
        _: &str,
        prototype: &MlxTensor,
        _: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<MlxTensor, Error>,
    ) -> Result<(), Error> {
        self.receive(prototype)?;
        if self.generate {
            self.receive(&generate()?)?;
        }
        Ok(())
    }
    fn intervene(&mut self, _: &str, value: &MlxTensor) -> Result<Option<MlxTensor>, Error> {
        self.receive(value)?;
        let replacement = self.replacement.take();
        if let Some(value) = &replacement {
            self.arrays.push(value.clone());
        }
        Ok(replacement.map(MlxTensor::from_array))
    }
    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, MlxTensor>,
    ) -> Result<(), Error> {
        routing.for_each_tensor(|_, value| self.arrays.push(value.as_array().clone()));
        Ok(())
    }
    fn routing_applied(
        &mut self,
        _: &str,
        original: Option<eredu_runtime::RoutingDecision<'_, MlxTensor>>,
        effective: eredu_runtime::RoutingDecision<'_, MlxTensor>,
    ) -> Result<(), Error> {
        for decision in original.iter().chain(std::iter::once(&effective)) {
            self.receive(decision.ids)?;
            self.receive(decision.coefficients)?;
        }
        Ok(())
    }
    fn routed_unit_observer(
        &mut self,
        _: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<MlxTensor>>, Error> {
        Ok(Some(self))
    }
}

impl RoutedUnitObserver<MlxTensor> for Recorder {
    fn begin_invocation(
        &mut self,
        invocation: &RoutedUnitInvocation<'_, MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.receive(invocation.input)
            .map_err(eredu_nn::Error::backend_retained_source)
    }
    fn observe(&mut self, batch: &RoutedUnitBatch<'_, MlxTensor>) -> Result<(), eredu_nn::Error> {
        self.receive_batch(batch)
    }
    fn observe_effective(
        &mut self,
        batch: &RoutedUnitBatch<'_, MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.receive_batch(batch)
    }
    fn intervene(
        &mut self,
        batch: &RoutedUnitBatch<'_, MlxTensor>,
    ) -> Result<Option<MlxTensor>, eredu_nn::Error> {
        self.receive_batch(batch)?;
        let replacement = self.replacement.take();
        if let Some(value) = &replacement {
            self.arrays.push(value.clone());
        }
        Ok(replacement.map(MlxTensor::from_array))
    }
}

fn adapter(
    inner: &mut Recorder,
    authority: Option<ArrayObserverAllocationAuthority>,
) -> ArrayObserverAdapter<'_, Recorder> {
    ArrayObserverAdapter {
        inner,
        routed_path: None,
        routed_invocation_active: false,
        allocation_authority: authority,
    }
}

fn generated_source() -> eredu_core::capture::GeneratedCaptureSource {
    eredu_core::capture::GeneratedCaptureSource {
        creation_bytes: 32,
        source_dtype: None,
    }
}

#[test]
fn skipped_generated_values_keep_prototypes_lazy_and_none_keeps_attachment_optional() {
    let stream = stream();
    for attach in [false, true] {
        let (requests, native, authority) = authority();
        let prototype = lazy(&stream);
        let mut recorder = Recorder::default();
        let mut calls = 0;
        {
            let mut adapter = adapter(&mut recorder, attach.then(|| authority.clone()));
            RuntimeActivationObserver::observe_generated(
                &mut adapter,
                "generated",
                &prototype,
                &generated_source(),
                &mut || {
                    calls += 1;
                    Ok(lazy(&stream))
                },
            )
            .unwrap();
        }
        assert_eq!(calls, 0);
        assert_eq!(prototype.allocation_info().unwrap(), None);
        let escaped = recorder.arrays.pop().unwrap();
        drop((prototype, recorder, authority));
        assert_eq!(escaped.allocation_info().unwrap(), None);
        settled(&requests, &native, attach);
        drop(escaped);
        settled(&requests, &native, false);
    }
}

#[test]
fn ordinary_and_replica_callbacks_keep_raw_strided_views_owned_after_success_error_and_panic() {
    for device in [safemlx::DeviceType::Cpu, safemlx::DeviceType::Gpu] {
        if device == safemlx::DeviceType::Gpu && !cfg!(feature = "metal") {
            continue;
        }
        let stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
        for replica in [false, true] {
            for failure in [Failure::None, Failure::Error, Failure::Panic] {
                let (requests, native, authority) = authority();
                let source = lazy(&stream);
                let mut recorder = Recorder {
                    failure,
                    ..Default::default()
                };
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut adapter = adapter(&mut recorder, Some(authority.clone()));
                    if replica {
                        RuntimeActivationObserver::observe_replica(&mut adapter, "value", &source)
                    } else {
                        RuntimeActivationObserver::observe(&mut adapter, "value", &source)
                    }
                }));
                match failure {
                    Failure::None => outcome.unwrap().unwrap(),
                    Failure::Error => {
                        let error = outcome.unwrap().unwrap_err();
                        let mut cause: &(dyn std::error::Error + 'static) = &error;
                        while let Some(next) = cause.source() {
                            cause = next;
                        }
                        assert!(cause.is::<CallbackFailure>());
                    }
                    Failure::Panic => assert!(outcome.is_err()),
                }
                assert_eq!(source.allocation_info().unwrap(), None);
                let escaped = recorder.arrays.pop().unwrap();
                drop((source, recorder, authority));
                settled(&requests, &native, true);
                escaped.evaluated().unwrap();
                let allocation = escaped.allocation_info().unwrap().unwrap();
                let view = escaped.as_strided(&[4][..], &[2][..], 1, &stream).unwrap();
                view.evaluated().unwrap();
                assert_eq!(view.allocation_info().unwrap(), Some(allocation));
                drop(escaped);
                settled(&requests, &native, true);
                assert_eq!(
                    view.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
                    vec![4., 16., 36., 64.]
                );
                drop(view);
                settled(&requests, &native, false);
            }
        }
    }
}

fn routed_arrays(stream: &Stream) -> Vec<Array> {
    vec![
        lazy(stream).reshape(&[1, 8], stream).unwrap(),
        Array::from_slice(&[0_u32], &[1]),
        Array::from_slice(&[0_u32], &[1]),
        Array::from_slice(&[0_u32], &[1]),
        Array::from_slice(&[0.75_f32], &[1, 1]),
        Array::from_slice(&[0_u32], &[1, 1]),
    ]
}

fn batch(arrays: &[Array]) -> RoutedUnitBatch<'_, Array> {
    RoutedUnitBatch {
        units: eredu_nn::GroupedUnitBatch {
            values: &arrays[0],
            group_indices: &arrays[1],
            selection_indices: &arrays[2],
            token_indices: &arrays[3],
            coefficients: &arrays[4],
            token_offset: 0,
            total_token_count: 1,
            group_count: 1,
        },
        source_groups: &arrays[5],
        global_groups: None,
        provider_token_offset: 0,
        origins: None,
        unit_coordinates: None,
    }
}

#[test]
fn generated_and_returned_intervention_values_receive_their_own_deferred_attachment() {
    let stream = stream();
    for kind in 0..3 {
        let (requests, native, authority) = authority();
        let sources = routed_arrays(&stream);
        let replacement = lazy(&stream).square(&stream).unwrap();
        let mut recorder = Recorder {
            generate: true,
            replacement: (kind != 0).then(|| replacement.clone()),
            ..Default::default()
        };
        let mut calls = 0;
        {
            let mut adapter = adapter(&mut recorder, Some(authority.clone()));
            match kind {
                0 => RuntimeActivationObserver::observe_generated(
                    &mut adapter,
                    "generated",
                    &sources[0],
                    &generated_source(),
                    &mut || {
                        calls += 1;
                        Ok(replacement.clone())
                    },
                )
                .unwrap(),
                1 => {
                    RuntimeActivationObserver::intervene(&mut adapter, "value", &sources[0])
                        .unwrap();
                }
                _ => {
                    RuntimeActivationObserver::routed_unit_observer(&mut adapter, "units").unwrap();
                    RoutedUnitObserver::intervene(&mut adapter, &batch(&sources)).unwrap();
                }
            }
        }
        assert_eq!(calls, usize::from(kind == 0));
        assert_eq!(replacement.allocation_info().unwrap(), None);
        let escaped = recorder.arrays.pop().unwrap();
        drop((recorder, replacement, sources, authority));
        settled(&requests, &native, true);
        assert_eq!(
            escaped.evaluated().unwrap().as_slice::<f32>(),
            &[1., 16., 81., 256., 625., 1296., 2401., 4096.]
        );
        settled(&requests, &native, true);
        drop(escaped);
        settled(&requests, &native, false);
    }
}

#[test]
fn routed_invocation_and_every_batch_component_keep_authority_when_retained_alone() {
    let stream = stream();
    for seam in 0..4 {
        for retained in 0..if seam == 0 { 1 } else { 6 } {
            let (requests, native, authority) = authority();
            let sources = routed_arrays(&stream);
            let mut recorder = Recorder::default();
            {
                let mut adapter = adapter(&mut recorder, Some(authority.clone()));
                RuntimeActivationObserver::routed_unit_observer(&mut adapter, "units").unwrap();
                match seam {
                    0 => RoutedUnitObserver::begin_invocation(
                        &mut adapter,
                        &RoutedUnitInvocation {
                            input: &sources[0],
                            origins: None,
                            unit_coordinates: None,
                        },
                    )
                    .unwrap(),
                    1 => RoutedUnitObserver::observe(&mut adapter, &batch(&sources)).unwrap(),
                    2 => RoutedUnitObserver::observe_effective(&mut adapter, &batch(&sources))
                        .unwrap(),
                    _ => {
                        RoutedUnitObserver::intervene(&mut adapter, &batch(&sources)).unwrap();
                    }
                }
            }
            let escaped = recorder.arrays.remove(retained);
            drop((sources, recorder, authority));
            settled(&requests, &native, true);
            escaped.evaluated().unwrap();
            drop(escaped);
            settled(&requests, &native, false);
        }
    }
}

#[test]
fn routing_observation_and_original_effective_decisions_cover_every_exposed_array() {
    let stream = stream();
    for decisions in [false, true] {
        let count = if decisions { 4 } else { 8 };
        for retained in 0..count {
            let (requests, native, authority) = authority();
            let sources = (0..count)
                .map(|index| {
                    if index == 0 || (decisions && index == 2) {
                        Array::from_slice(&[0_u32, 1, 2, 3], &[4])
                            .square(&stream)
                            .unwrap()
                    } else {
                        lazy(&stream)
                    }
                })
                .collect::<Vec<_>>();
            let mut recorder = Recorder::default();
            {
                let mut adapter = adapter(&mut recorder, Some(authority.clone()));
                if decisions {
                    RuntimeActivationObserver::routing_applied(
                        &mut adapter,
                        "routing",
                        Some(eredu_runtime::RoutingDecision {
                            ids: &sources[0],
                            coefficients: &sources[1],
                        }),
                        eredu_runtime::RoutingDecision {
                            ids: &sources[2],
                            coefficients: &sources[3],
                        },
                    )
                    .unwrap();
                } else {
                    RuntimeActivationObserver::observe_routing(
                        &mut adapter,
                        eredu_runtime::RoutingObservation {
                            path: "routing",
                            selected_experts: &sources[0],
                            selected_scores: &sources[1],
                            coefficients: &sources[2],
                            routed_output: &sources[3],
                            local_routed_output: Some(&sources[4]),
                            reduced_routed_output: Some(&sources[5]),
                            shared_output: Some(&sources[6]),
                            combined_output: Some(&sources[7]),
                            expert_count: 16,
                        },
                    )
                    .unwrap();
                }
            }
            let escaped = recorder.arrays.remove(retained);
            drop((sources, recorder, authority));
            assert_eq!(escaped.allocation_info().unwrap(), None);
            settled(&requests, &native, true);
            drop(escaped);
            settled(&requests, &native, false);
        }
    }
}

#[test]
fn session_observer_intermediate_keeps_execution_domain_after_session_retirement() {
    #[derive(Default)]
    struct Intermediate {
        array: Option<Array>,
        was_lazy: bool,
    }
    impl RuntimeActivationObserver<MlxTensor, Error> for Intermediate {
        fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), Error> {
            if self.array.is_none() && path.ends_with(".attention.output") {
                // The callback only borrows and clones; it must not evaluate work.
                self.was_lazy = value.as_array().allocation_info()?.is_none();
                self.array = Some(value.as_array().clone());
            }
            Ok(())
        }
    }

    let stream = stream();
    let model_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let operation_pool = WorkingMemoryPool::new(0, 0).unwrap();
    let source = MlxBackend::new(&stream, &stream).with_memory_pool(model_pool.clone());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&source, root.path(), crate::MlxLoadRequest::default()).unwrap();
    let prompt = MlxBackend::prepare_text_prompt(&source, vec![1, 2, 3]).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(operation_pool.clone());
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(operation_pool.clone());
    let mut observer = Intermediate::default();
    let output = runtime
        .session_mut()
        .submit_prefill_with_observer(&backend, prompt, &mut observer)
        .unwrap()
        .wait()
        .unwrap();
    assert!(observer.was_lazy);
    let escaped = observer
        .array
        .take()
        .expect("capture an attention intermediate");
    // A Weak probe intentionally prevents Rc::get_mut, so install it only
    // after the submission has finished all exclusive payload access.
    let retired = runtime.session().test_payload_retirement_probe();
    drop((output, observer, runtime, backend, source, root));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        retired() && operation_pool.unquoted_owner_count().unwrap() == 1
    });

    let values = escaped.evaluated().unwrap().try_to_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > 1e-6));
    let allocation = escaped.allocation_info().unwrap().unwrap();
    let view = escaped.as_strided(&[4][..], &[2][..], 1, &stream).unwrap();
    view.evaluated().unwrap();
    assert_eq!(view.allocation_info().unwrap(), Some(allocation));
    drop(escaped);
    safemlx::reclaim_allocation_owners();
    assert_eq!(operation_pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(
        view.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
        vec![values[1], values[3], values[5], values[7]]
    );
    drop(view);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        operation_pool.unquoted_owner_count().unwrap() == 0
            && model_pool.unquoted_owner_count().unwrap() == 0
    });
}
