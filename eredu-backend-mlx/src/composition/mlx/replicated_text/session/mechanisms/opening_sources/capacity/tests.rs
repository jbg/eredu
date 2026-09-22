//! Actual native topology and original neutral seal; no model work or publication.
use super::*;
use eredu_core::{capture::*, *};
use eredu_nn::{Parameterized, workspace::*};
use eredu_runtime::working_memory::*;
use eredu_runtime::{
    ArchitectureParameters, ResidentUnitWindow, StateLayout, StaticParameterVisitorMut,
};
use std::{convert::Infallible, num::NonZeroU8};

struct Populate;
impl<'a> eredu_nn::ParameterVisitorMut<'a, MlxTensor> for Populate {
    fn visit_mut(&mut self, _: eredu_nn::ParameterMetadataView<'_>, value: &'a mut MlxTensor) {
        let shape = value.as_array().shape().to_vec();
        let n = shape.iter().map(|&n| n as usize).product::<usize>();
        *value = MlxTensor::from_array(Array::from_slice(&vec![1.25_f32; n], &shape));
    }
}
impl StaticParameterVisitorMut<MlxNeuralBackend> for Populate {
    type Error = Infallible;
    fn visit_mut<M: Parameterized<MlxTensor>>(
        &mut self,
        _: &str,
        module: &mut M,
    ) -> Result<(), Self::Error> {
        module.visit_parameters_mut(self);
        Ok(())
    }
}
pub(super) type Model = eredu_architectures::llama::LayeredModel<MlxNeuralBackend>;
type Unit = <Model as LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::Unit;
pub(super) type Execution =
    LayerwiseRuntime<Model, MlxNeuralBackend, MlxKeyValueState, ResidentUnitWindow<Unit>>;
fn model(stream: &Stream) -> Model {
    let args = eredu_architectures::llama::model_args_from_config_value(&serde_json::json!({
        "model_type":"llama", "hidden_size":8, "intermediate_size":16,
        "num_hidden_layers":1, "num_attention_heads":4, "num_key_value_heads":2,
        "head_dim":2, "vocab_size":16, "rms_norm_eps":1e-5,
        "max_position_embeddings":32, "tie_word_embeddings":true
    }))
    .unwrap();
    let mut model = Model::new(args, stream).unwrap();
    model.visit_static_parameters_mut(&mut Populate).unwrap();
    model
}
pub(super) fn execution(stream: &Stream) -> Execution {
    let model = model(stream);
    let mut unit = model.construct_unit(0, stream).unwrap();
    unit.visit_parameters_mut(&mut Populate);
    let runtime = LayerwiseRuntime::new(model, ResidentUnitWindow::new(vec![unit]));
    assert!(runtime.visit_retained_values(&mut |v| {
        let _ = v.as_array().evaluated().unwrap();
    }));
    runtime
}
// Independent physical parameter traversal: the populated dense fixture has
// embedding + final norm + two block norms + seven projection weights. Its
// unscaled rotary operator has no retained arrays. Snapshot before admission;
// these diagnostic Vecs are test setup, not part of the fixed collector.
fn execution_with_parameter_allocations(
    stream: &Stream,
) -> (Execution, Vec<safemlx::ArrayAllocationInfo>) {
    struct Snapshot(Vec<safemlx::ArrayAllocationInfo>);
    impl<'a> eredu_nn::ParameterVisitorMut<'a, MlxTensor> for Snapshot {
        fn visit_mut(&mut self, _: eredu_nn::ParameterMetadataView<'_>, value: &'a mut MlxTensor) {
            let _ = value.as_array().evaluated().unwrap();
            let fact = value.as_array().try_allocation_info().unwrap().unwrap();
            assert!(fact.bytes() > 0);
            self.0.push(fact);
        }
    }
    impl StaticParameterVisitorMut<MlxNeuralBackend> for Snapshot {
        type Error = Infallible;
        fn visit_mut<M: Parameterized<MlxTensor>>(
            &mut self,
            _: &str,
            module: &mut M,
        ) -> Result<(), Self::Error> {
            module.visit_parameters_mut(self);
            Ok(())
        }
    }
    let mut model = model(stream);
    let mut unit = model.construct_unit(0, stream).unwrap();
    unit.visit_parameters_mut(&mut Populate);
    let mut snapshot = Snapshot(Vec::new());
    // ResidentUnitWindow deliberately does not expose mutable parameter
    // inspection. Borrow the actual loaded modules before moving them into it.
    model.visit_static_parameters_mut(&mut snapshot).unwrap();
    unit.visit_parameters_mut(&mut snapshot);
    assert_eq!(snapshot.0.len(), 11);
    let runtime = LayerwiseRuntime::new(model, ResidentUnitWindow::new(vec![unit]));
    assert!(runtime.visit_retained_values(&mut |v| {
        let _ = v.as_array().evaluated().unwrap();
    }));
    (runtime, snapshot.0)
}
fn assert_array_allocations(
    owners: &FixedOpeningOwners,
    mut expected: Vec<safemlx::ArrayAllocationInfo>,
) {
    let mut actual = Vec::new();
    owners
        .visit(&mut |entry| -> Result<(), OpeningError> {
            if let OpeningEntry::Array(_, fact) = entry {
                actual.push(fact);
            }
            Ok(())
        })
        .unwrap();
    // Compare the complete multiset, retaining duplicate visits and capacities.
    expected.sort_by_key(|fact| fact.identity());
    actual.sort_by_key(|fact| fact.identity());
    assert_eq!(actual, expected);
}
pub(super) fn state() -> MlxKeyValueState {
    MlxKeyValueState::device(
        StateLayout::new(
            LayerSchedule::new(
                1,
                vec![
                    eredu_core::cache::LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 2)
                        .unwrap(),
                ],
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}
pub(super) fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 2,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}
pub(super) fn source() -> SharedCapturePlan {
    SharedCapturePlan::new(
        CapturePlan::none()
            .admit(
                &ObservationCatalog {
                    schema_version: 1,
                    points: vec![],
                    completeness: DescriptionCompleteness::Complete,
                },
                &ObservationSupportReport {
                    schema_version: 1,
                    capture: Default::default(),
                    points: vec![],
                },
                &CaptureCapabilities {
                    transformations: vec![],
                    max_histogram_bins: 0,
                    conditions: vec![],
                },
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 1,
                    max_predictions: 2,
                },
            )
            .unwrap(),
    )
}
pub(super) fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}
// This quote executes no native work. It isolates original P+Q construction
// custody; the existing native arrays remain under fixture loading ownership.
#[derive(Debug)]
struct NoOperations;
impl WorkspaceMechanisms for NoOperations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        panic!("no numerical operation in the host-custody fixture")
    }
}
pub(super) fn quote(pool: &MemoryLedger) -> IncrementalInferenceQuote {
    let g = geometry();
    let context = WorkspaceContext::new(NoOperations);
    let storage = RegisteredWorkspaceStorage::<u32>::bind(
        pool,
        &context,
        std::iter::empty::<(u32, WorkspaceExistingStorage)>(),
    )
    .unwrap();
    let report = quote_inference_workspace(g, |_| {
        context.begin_state_span(std::iter::empty::<&WorkspaceTensor>())?;
        context.report(&[])
    })
    .unwrap();
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        2,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = || {
        WorkspaceBound::bounded(
            0,
            "no native work; existing sources externally retained in this fixture",
        )
    };
    let outside = crate::memory_fixture::workspace(ExecutionWorkspaceEstimate {
        physical_domains: None,
        geometry: g,
        activations: bound(),
        attention: bound(),
        vocabulary: bound(),
        state_update: bound(),
        materialization: bound(),
        retained: bound(),
    });
    ResidualInferenceQuote::compose(&report, state, outside, &storage)
        .unwrap()
        .into_incremental()
}
pub(super) fn accept(
    pool: &MemoryLedger,
    q: IncrementalInferenceQuote,
) -> (WorkingMemoryFundingRun, OwnedTextSpanWorkspace) {
    let caps = ModelCapabilities {
        effective_model_type: "opening host fixture".into(),
        native_max_context: Observed::exact(32, "fixture"),
        effective_max_context: Observed::exact(32, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    };
    let request = AdmissionRequest {
        input: InputTokenCount::text(1),
        max_output_tokens: 2,
        batch_size: 1,
        additional_headroom: crate::memory_fixture::headroom(0),
        memory_limits: Default::default(),
    };
    let exact = q.incremental_bytes().expect("finite fixture diagnostic");
    assert!(
        plan_prefill_incremental_with_capacity(
            &InferenceExecutionIdentity::default(),
            pool,
            &caps,
            request.clone(),
            geometry(),
            crate::memory_fixture::resolved_limits(exact - 1),
            |_| Ok(q.clone())
        )
        .is_err()
    );
    let (r, q) = plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        pool,
        &caps,
        request,
        geometry(),
        crate::memory_fixture::resolved_limits(exact),
        |_| Ok(q.clone()),
    )
    .unwrap();
    let (r, run) = r.into_funding().unwrap();
    let (owned, witness) = q.into_funded_text_span_workspace(&run, &r).unwrap();
    assert!(witness.is_none());
    (run, owned)
}

#[test]
fn actual_bound_topology_counts_absent_state_slots_and_rejects_foreign_paths() {
    let stream = stream();
    let execution = execution(&stream);
    let paths = execution.prepare_observation_paths().unwrap();
    let mut state = state();
    let source = source();
    let selected = paths.source().prepare_capture_selection(&source).unwrap();
    let bound = selected.bind_geometry(geometry()).unwrap();
    let store = eredu_checkpoint::store::MemoryWeightStore::default();
    let plan = NativeOpeningCapacityPlan::prepare(&state, &execution, &store, None, bound, &paths)
        .unwrap();
    assert!(std::ptr::eq(plan.state, &state));
    assert!(std::ptr::eq(plan.execution, &execution));
    assert_eq!(
        plan.slots.arrays,
        2 + execution.retained_value_slot_bound().unwrap()
    );
    assert_eq!((plan.slots.layouts, plan.slots.tables), (1, 1));
    assert!(
        plan.control_peak_bytes().unwrap() > FixedOpeningOwners::buffer_bytes(plan.slots).unwrap()
    );
    assert_eq!(
        FixedOpeningOwners::storage_bytes(plan.slots).unwrap()
            - FixedOpeningOwners::buffer_bytes(plan.slots).unwrap(),
        (plan.slots.arrays * Array::inspection_clone_handle_bytes()) as u64
    );
    assert!(Array::inspection_clone_handle_bytes() > 0);
    let other = self::execution(&stream);
    assert!(matches!(
        NativeOpeningCapacityPlan::prepare(&state, &other, &store, None, bound, &paths),
        Err(OpeningError::Identity)
    ));
    let foreign = other.prepare_observation_paths().unwrap();
    assert!(matches!(
        NativeOpeningCapacityPlan::prepare(&state, &other, &store, None, bound, &foreign),
        Err(OpeningError::Identity)
    ));
    let resident: ResidentRuntime<Model, MlxNeuralBackend, MlxKeyValueState> =
        ResidentRuntime::new(model(&stream), &stream).unwrap();
    let resident_paths = resident.prepare_observation_paths().unwrap();
    let resident_selected = resident_paths
        .source()
        .prepare_capture_selection(&source)
        .unwrap();
    let resident_plan = NativeOpeningCapacityPlan::prepare(
        &state,
        &resident,
        &store,
        None,
        resident_selected.bind_geometry(geometry()).unwrap(),
        &resident_paths,
    )
    .unwrap();
    assert!(resident_plan.slots.arrays > 2);
    assert_eq!(
        resident_plan.slots.arrays,
        2 + resident.owner_slot_bound().unwrap()
    );
    drop((plan, resident_plan));
    let before = state.retained_owner_slot_counts().unwrap();
    crate::backend::runtime::cache::kv::KeyValueCache::update_for_attention(
        state.layer(0).unwrap(),
        Array::from_slice(&[1_f32, 3., 5., 7.], &[1, 2, 1, 2]),
        Array::from_slice(&[11_f32, 13., 17., 19.], &[1, 2, 1, 2]),
        &stream,
    )
    .unwrap();
    assert_eq!(state.retained_owner_slot_counts().unwrap(), before);
    assert!(
        matches!(
            NativeOpeningCapacityPlan::prepare(&state, &execution, &store, None, bound, &paths),
            Err(OpeningError::StateOwner(_))
        ),
        "changed opening frontier cannot reuse candidate geometry"
    );
}

#[test]
fn one_actual_original_seal_funds_fixed_capsule_and_equal_facts_cannot_substitute() {
    let stream = stream();
    let (execution, expected_arrays) = execution_with_parameter_allocations(&stream);
    let paths = execution.prepare_observation_paths().unwrap();
    let state = state();
    let source = source();
    let selected = paths.source().prepare_capture_selection(&source).unwrap();
    let bound = selected.bind_geometry(geometry()).unwrap();
    let store = eredu_checkpoint::store::MemoryWeightStore::default();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let q = quote(&pool);
    let plan = NativeOpeningCapacityPlan::prepare(&state, &execution, &store, None, bound, &paths)
        .unwrap();
    let slots = plan.slots;
    let proposal = plan
        .seal(
            q.span_workspace().plan(),
            TextHostControlFacts::new(Some(0), Some(0), Some(0)),
        )
        .unwrap();
    // Independent equal facts/plan/source still have a different original identity.
    let foreign = PreparedTextControlWorkspace::prepare(
        &source,
        geometry(),
        q.span_workspace().plan(),
        proposal.controls.facts(),
    )
    .unwrap();
    assert!(!foreign.same_binding(&proposal.controls));
    let (sealed, q) = proposal.finish(q).unwrap();
    let (run, owned) = accept(&pool, q);
    let held = owned.protected_host_bytes();
    let mut capsule = match sealed.allocate(&owned) {
        Ok(v) => v,
        Err(e) => panic!("{}", e.cause),
    };
    capsule.collect().unwrap();
    let retained = capsule.owners.retained_counts();
    // Seven dense linears reserve four absent companions each; rotary reserves
    // two absent helper arrays, and the initial KV state reserves two fields.
    assert_eq!(slots.arrays, expected_arrays.len() + 7 * 4 + 2 + 2);
    assert_eq!(
        retained,
        OpeningSlots {
            arrays: expected_arrays.len(),
            layouts: 1,
            tables: 1,
            ..OpeningSlots::default()
        }
    );
    assert_array_allocations(&capsule.owners, expected_arrays);
    assert!(matches!(
        capsule.collect(),
        Err(OpeningCollectionFailure {
            cause: OpeningError::Used,
            ..
        })
    ));
    drop(owned);
    drop(run);
    assert_eq!(pool.fixture_host_charge().unwrap(), held);
    drop(capsule);
    // Its control binding is foreign, but the earlier diagnostic still owns
    // an alias of the actual equation plan and its later original custody.
    assert_eq!(pool.fixture_host_charge().unwrap(), held);
    drop(foreign);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn separately_prepared_equal_controls_reject_before_capsule_allocation() {
    let stream = stream();
    let execution = execution(&stream);
    let paths = execution.prepare_observation_paths().unwrap();
    let state = state();
    let source = source();
    let selected = paths.source().prepare_capture_selection(&source).unwrap();
    let store = eredu_checkpoint::store::MemoryWeightStore::default();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let q = quote(&pool);
    let plan = NativeOpeningCapacityPlan::prepare(
        &state,
        &execution,
        &store,
        None,
        selected.bind_geometry(geometry()).unwrap(),
        &paths,
    )
    .unwrap();
    let proposal = plan
        .seal(
            q.span_workspace().plan(),
            TextHostControlFacts::new(Some(0), Some(0), Some(0)),
        )
        .unwrap();
    let foreign = PreparedTextControlWorkspace::prepare(
        &source,
        geometry(),
        q.span_workspace().plan(),
        proposal.controls.facts(),
    )
    .unwrap();
    let foreign_q = q
        .clone()
        .with_span_workspace_and_text_controls(foreign)
        .unwrap();
    let (sealed, legitimate_q) = proposal.finish(q).unwrap();
    let (run, owned) = accept(&pool, foreign_q);
    let before = pool.fixture_host_charge().unwrap();
    let error = match sealed.allocate(&owned) {
        Err(e) => e,
        Ok(_) => panic!("foreign original identity accepted"),
    };
    assert!(matches!(error.cause, OpeningError::Identity));
    assert_eq!(pool.fixture_host_charge().unwrap(), before);
    assert!(std::ptr::eq(error.plan.plan.state, &state));
    drop((error, legitimate_q, owned, run));
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn actual_weight_manager_host_device_and_duplicate_checkpoint_roles_fit_original_slots() {
    use eredu_checkpoint::store::{MemoryWeightStore, TensorSelection};
    use eredu_core::residency::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
    };
    use eredu_runtime::{OffloadUnit, WeightBinding};
    let stream = stream();
    let (execution, mut expected_arrays) = execution_with_parameter_allocations(&stream);
    let paths = execution.prepare_observation_paths().unwrap();
    let state = state();
    let source = source();
    let selected = paths.source().prepare_capture_selection(&source).unwrap();
    let store = Arc::new(
        MemoryWeightStore::from_safetensors([(
            "value".to_owned(),
            safetensors::tensor::Dtype::F32,
            vec![2],
            [3_f32, -7.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )])
        .unwrap(),
    );
    let id = OffloadUnitId::new("unit").unwrap();
    let host_bytes =
        safemlx::host_transfer_capacity_upper_bound(8, safemlx::HostTransferPolicy::Transfer)
            .unwrap() as u64;
    let manager = ResidencyManager::new(
        store.clone(),
        OffloadPlan::new(
            OffloadConfig::new(Some(8), Some(host_bytes), 1).unwrap(),
            [
                OffloadUnitSpec::new(id.clone(), 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)
                    .unwrap(),
            ],
        )
        .unwrap(),
        [OffloadUnit::new(
            id.clone(),
            [WeightBinding::new("weight", "value", TensorSelection::Full, 8).unwrap()],
        )
        .unwrap()],
        stream.clone(),
        stream.clone(),
    )
    .unwrap();
    let mut mechanisms: MlxReplicatedTextMechanisms<Model, MlxKeyValueState> =
        MlxReplicatedTextMechanisms::new(store.clone(), &stream, &stream).unwrap();
    assert!(matches!(
        mechanisms.prepare_fixed_opening_capacity(
            &state,
            &execution,
            selected.bind_geometry(geometry()).unwrap(),
            &paths
        ),
        Err(OpeningError::Unknown)
    ));
    mechanisms.residency_manager = Some(manager.clone());
    let plan = mechanisms
        .prepare_fixed_opening_capacity(
            &state,
            &execution,
            selected.bind_geometry(geometry()).unwrap(),
            &paths,
        )
        .unwrap();
    assert!(plan.weights.as_ref().unwrap().matches(&manager));
    assert!(std::ptr::eq(plan.store, mechanisms.store.as_ref()));
    assert_eq!((plan.slots.hosts, plan.slots.sources), (1, 2));
    let slots = plan.slots;
    manager.initialize().unwrap();
    manager.prefetch(&id, MemoryTier::Host).unwrap();
    manager.prefetch(&id, MemoryTier::Device).unwrap();
    // Retain only through the permitted inspection clone under the manager
    // loan, then inspect/drop after unlocking and before original admission.
    let mut manager_array = None;
    assert!(
        manager
            .try_visit_retained_storage(&mut |entry| -> Result<(), OpeningError> {
                if let crate::backend::runtime::residency::storage::RetainedStorageRef::Array(
                    array,
                ) = entry
                {
                    assert!(
                        manager_array.is_none(),
                        "exactly one declared manager binding"
                    );
                    manager_array = Some(array.try_clone_for_inspection()?);
                }
                Ok(())
            })
            .unwrap()
    );
    let manager_array = manager_array.expect("the populated device binding was visited");
    let manager_fact = manager_array.try_allocation_info().unwrap().unwrap();
    assert!(manager_fact.bytes() >= 8);
    expected_arrays.push(manager_fact);
    drop(manager_array);
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let q = quote(&pool);
    let proposal = plan
        .seal(
            q.span_workspace().plan(),
            TextHostControlFacts::new(Some(0), Some(0), Some(0)),
        )
        .unwrap();
    let (sealed, q) = proposal.finish(q).unwrap();
    let (run, owned) = accept(&pool, q);
    let mut capsule = match sealed.allocate(&owned) {
        Ok(v) => v,
        Err(e) => panic!("{}", e.cause),
    };
    capsule.collect().unwrap();
    let actual = capsule.owners.retained_counts();
    assert_eq!(slots.arrays, expected_arrays.len() + 7 * 4 + 2 + 2);
    assert_eq!(
        actual,
        OpeningSlots {
            arrays: expected_arrays.len(),
            hosts: 1,
            sources: 2,
            layouts: 1,
            tables: 1,
            ..OpeningSlots::default()
        }
    );
    assert_array_allocations(&capsule.owners, expected_arrays);
    assert_eq!(
        manager
            .prepare_owner_slot_bounds()
            .unwrap()
            .unwrap()
            .array_slots(),
        1
    );
    drop((capsule, owned, run));
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

struct AdversarialSource {
    bound: Option<usize>,
    mode: usize,
    root: Arc<Vec<u8>>,
    panic: Arc<()>,
    visits: std::sync::atomic::AtomicUsize,
}
impl CheckpointSource for AdversarialSource {
    fn source_storage_slot_bound(
        &self,
    ) -> Result<Option<usize>, eredu_checkpoint::store::StoreError> {
        Ok(self.bound)
    }
    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(eredu_checkpoint::store::SourceStorageRef<'_>),
    ) -> Result<bool, eredu_checkpoint::store::StoreError> {
        self.visits
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        visitor(eredu_checkpoint::store::SourceStorageRef::new(
            &self.root,
            self.root.len() as u64,
        ));
        match self.mode {
            0 => Ok(false),
            1 => Err(eredu_checkpoint::store::StoreError::Internal(
                "original source failure".into(),
            )),
            2 => {
                visitor(eredu_checkpoint::store::SourceStorageRef::new(
                    &self.root,
                    self.root.len() as u64,
                ));
                Ok(true)
            }
            _ => std::panic::panic_any(self.panic.clone()),
        }
    }
    fn source_keys(&self) -> Vec<String> {
        panic!("source catalog is not read")
    }
    fn source_metadata(
        &self,
        _: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        panic!("source metadata is not read")
    }
    fn acquire_lease(
        &self,
        _: eredu_checkpoint::store::TensorReadRequest,
    ) -> Result<eredu_checkpoint::store::CheckpointLease, eredu_checkpoint::store::StoreError> {
        panic!("source lease is not acquired")
    }
    fn source_diagnostics(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreDiagnostics, eredu_checkpoint::store::StoreError>
    {
        panic!("source diagnostics are not rebuilt")
    }
}

#[test]
fn unknown_source_rejects_cold_and_failed_visitors_keep_real_prefix_and_original_cause() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let stream = stream();
    let execution = execution(&stream);
    let paths = execution.prepare_observation_paths().unwrap();
    let state = state();
    let source = source();
    let selected = paths.source().prepare_capture_selection(&source).unwrap();
    let bound = selected.bind_geometry(geometry()).unwrap();
    for mode in 0..4 {
        let mut store = AdversarialSource {
            bound: None,
            mode,
            root: Arc::new(vec![3, 5, 7]),
            panic: Arc::new(()),
            visits: Default::default(),
        };
        assert!(matches!(
            NativeOpeningCapacityPlan::prepare(&state, &execution, &store, None, bound, &paths),
            Err(OpeningError::Unknown)
        ));
        assert_eq!(store.visits.load(std::sync::atomic::Ordering::SeqCst), 0);
        store.bound = Some(1);
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let q = quote(&pool);
        let plan =
            NativeOpeningCapacityPlan::prepare(&state, &execution, &store, None, bound, &paths)
                .unwrap();
        let proposal = plan
            .seal(
                q.span_workspace().plan(),
                TextHostControlFacts::new(Some(0), Some(0), Some(0)),
            )
            .unwrap();
        let (sealed, q) = proposal.finish(q).unwrap();
        let (run, owned) = accept(&pool, q);
        let mut capsule = match sealed.allocate(&owned) {
            Ok(v) => v,
            Err(e) => panic!("{}", e.cause),
        };
        let result = catch_unwind(AssertUnwindSafe(|| capsule.collect()));
        match (mode, &result) {
            (0, Ok(Err(error))) => assert!(matches!(error.cause, OpeningError::Incomplete)),
            (1, Ok(Err(error))) => match &error.cause {
                OpeningError::Store(eredu_checkpoint::store::StoreError::Internal(message)) => {
                    assert_eq!(message, "original source failure")
                }
                _ => panic!("wrong original source error"),
            },
            (2, Ok(Err(error))) => assert!(matches!(error.cause, OpeningError::Capacity)),
            (3, Err(payload)) => assert!(Arc::ptr_eq(
                payload.downcast_ref::<Arc<()>>().unwrap(),
                &store.panic
            )),
            _ => panic!("wrong original failure"),
        }
        assert_eq!(capsule.owners.retained_counts().sources, 1);
        assert_eq!(Arc::strong_count(&store.root), 2);
        assert!(matches!(
            capsule.collect(),
            Err(OpeningCollectionFailure {
                cause: OpeningError::Used,
                ..
            })
        ));
        let held = owned.protected_host_bytes();
        drop((owned, run));
        assert!(pool.fixture_host_charge().unwrap() >= held);
        drop(capsule);
        assert_eq!(Arc::strong_count(&store.root), 1);
        if mode != 3 {
            assert_eq!(
                pool.fixture_host_charge().unwrap(),
                held,
                "escaped error keeps exactly original P+Q after capsule retirement"
            );
        }
        drop(result);
        assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    }
}

#[test]
fn partial_buffer_allocation_failure_keeps_the_original_plan_hold_and_sources_unvisited() {
    let stream = stream();
    let execution = execution(&stream);
    let paths = execution.prepare_observation_paths().unwrap();
    let state = state();
    let source = source();
    let selected = paths.source().prepare_capture_selection(&source).unwrap();
    let store = AdversarialSource {
        bound: Some(1),
        mode: 0,
        root: Arc::new(vec![11]),
        panic: Arc::new(()),
        visits: Default::default(),
    };
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let q = quote(&pool);
    let plan = NativeOpeningCapacityPlan::prepare(
        &state,
        &execution,
        &store,
        None,
        selected.bind_geometry(geometry()).unwrap(),
        &paths,
    )
    .unwrap();
    let proposal = plan
        .seal(
            q.span_workspace().plan(),
            TextHostControlFacts::new(Some(0), Some(0), Some(0)),
        )
        .unwrap();
    let (sealed, q) = proposal.finish(q).unwrap();
    let (run, owned) = accept(&pool, q);
    let held = owned.protected_host_bytes();
    FixedOpeningOwners::fail_after_allocations(1);
    let failure = match sealed.allocate(&owned) {
        Err(e) => e,
        Ok(_) => panic!("injected allocation failure missing"),
    };
    assert!(matches!(failure.cause, OpeningError::Allocation(_)));
    assert_eq!(store.visits.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(Arc::strong_count(&store.root), 1);
    drop((owned, run));
    assert!(pool.fixture_host_charge().unwrap() >= held);
    drop(failure);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
