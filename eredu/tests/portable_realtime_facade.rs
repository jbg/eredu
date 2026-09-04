use std::{
    cell::Cell,
    collections::BTreeMap,
    convert::Infallible,
    num::NonZeroUsize,
    rc::Rc,
    time::{Duration, Instant},
};

use eredu::api::realtime::{
    inspect_moshi_realtime, prepare_realtime_model_from_catalog, select_inspected_moshi_realtime,
    MoshiRealtimeRequest, PreparedRealtimeModel, RealtimeGenerationState, RealtimeInputFrame,
    RealtimeOutputFrame, RealtimeSampling, RealtimeSessionScheduler, RequestId, SchedulerLimits,
    SessionCapabilities,
};
use eredu_architectures::moshi::{self, MoshiConfig};
use eredu_checkpoint::{
    recipe::RecipeCatalog,
    schema::StoredDtypeConstraint,
    store::{StoreError, TensorMetadata},
    validation::{CatalogTensorMetadata, SafetensorsCatalog},
    StoredDtype,
};
use eredu_core::{
    scheduler::{SemanticStateTransaction, TransitionOutput},
    Completion, CompletionCancellationMode, ParallelRankTopology, ParallelTopology,
};
use eredu_runtime::{
    CacheResidencyPolicy, CommunicationCompletionCapabilities, CommunicationCompletionPolicy,
    ExecutionResidency, LayerWeightResidency, PipelineActivationDtype, RealtimeMechanism,
    RealtimeMechanismCapabilities, RealtimeObservationRequirements, StateComponentMechanism,
    StateComponentPlacement, StateMechanismCapabilities, WeightLoweringCapability,
};

struct HeaderCatalog(BTreeMap<String, TensorMetadata>);

impl HeaderCatalog {
    fn for_config(config: &MoshiConfig) -> Self {
        let plan = moshi::safetensors_plan(config).unwrap();
        let tensors = plan
            .common_tensors
            .iter()
            .map(|tensor| {
                let stored_dtype = match &tensor.dtype {
                    StoredDtypeConstraint::Exact(dtype) => dtype.clone(),
                    StoredDtypeConstraint::Floating => StoredDtype::F32,
                    StoredDtypeConstraint::OneOf(dtypes) => dtypes[0].clone(),
                };
                let encoded_byte_len = tensor
                    .shape
                    .iter()
                    .try_fold(4_u64, |bytes, dimension| {
                        bytes.checked_mul(u64::try_from(*dimension).ok()?)
                    })
                    .unwrap();
                (
                    tensor.key.clone(),
                    TensorMetadata {
                        name: tensor.key.clone(),
                        logical_shape: tensor.shape.clone(),
                        physical_shape: tensor.shape.clone(),
                        stored_dtype,
                        encoded_byte_len,
                        backing_shard: Some("header-only.safetensors".into()),
                    },
                )
            })
            .collect();
        Self(tensors)
    }
}

impl SafetensorsCatalog for HeaderCatalog {
    fn keys(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        self.0
            .get(key)
            .map(|metadata| CatalogTensorMetadata {
                shape: metadata.logical_shape.clone(),
                stored_dtype: metadata.stored_dtype.clone(),
            })
            .ok_or_else(|| format!("unknown test tensor {key}"))
    }
}

impl RecipeCatalog for HeaderCatalog {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.0
            .get(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
}

fn config() -> MoshiConfig {
    MoshiConfig::from_json(
        r#"{
            "model_type":"moshi", "dim":4, "text_card":17,
            "n_q":2, "dep_q":1, "generated_audio_codebooks":1, "card":16,
            "num_heads":1, "num_layers":1, "dim_feedforward":6,
            "causal":true, "context":3, "max_period":10000.0,
            "positional_embedding":"rope", "depformer_dim":4,
            "depformer_dim_feedforward":6, "depformer_num_heads":1,
            "depformer_num_layers":1, "depformer_context":2,
            "depformer_max_period":10000.0, "depformer_pos_emb":"none",
            "delays":[0,0,1]
        }"#,
    )
    .unwrap()
}

fn request() -> MoshiRealtimeRequest {
    MoshiRealtimeRequest::new(
        None,
        LayerWeightResidency::FullyResident,
        CacheResidencyPolicy::Device,
        ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(4).unwrap(),
        PipelineActivationDtype::Float32,
        CommunicationCompletionPolicy::new(
            Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap(),
        RealtimeObservationRequirements::new(true, []),
    )
}

fn capabilities(
    requirements: &eredu_runtime::RealtimeArchitectureRequirements,
) -> RealtimeMechanismCapabilities {
    let lowerings = requirements.executions()[0]
        .weight_lowerings()
        .iter()
        .map(|lowering| {
            WeightLoweringCapability::new(lowering.descriptor().clone(), lowering.kind())
        });
    let layout = requirements.state_layout();
    let state = StateMechanismCapabilities::new((0..layout.len()).flat_map(|layer| {
        layout
            .components(layer)
            .unwrap()
            .iter()
            .cloned()
            .map(move |component| {
                StateComponentMechanism::new(
                    layer,
                    component,
                    Some(StateComponentPlacement::Device),
                    None,
                )
            })
            .collect::<Vec<_>>()
    }))
    .with_transactions(true, true)
    .with_reset(true)
    .with_observation_retention(true);
    RealtimeMechanismCapabilities::new(
        requirements.operators(),
        [
            RealtimeMechanism::TensorOperations,
            RealtimeMechanism::NeuralOperations,
            RealtimeMechanism::ParameterMaterialization,
            RealtimeMechanism::ParameterStorage,
            RealtimeMechanism::StateStorage,
            RealtimeMechanism::CoordinateStorage,
            RealtimeMechanism::Sampling,
            RealtimeMechanism::Randomness,
            RealtimeMechanism::HostConversion,
            RealtimeMechanism::ExactCompletion,
            RealtimeMechanism::ResourceRetention,
            RealtimeMechanism::Transfer,
            RealtimeMechanism::Observation,
            RealtimeMechanism::Collectives,
        ],
        [ExecutionResidency::FullyResident],
        lowerings.collect(),
        state,
        NonZeroUsize::new(1).unwrap(),
        CommunicationCompletionCapabilities::new([
            CompletionCancellationMode::QuarantineUntilComplete,
        ])
        .unwrap(),
        SessionCapabilities::new(true, true, true),
    )
}

#[derive(Clone)]
struct ReferenceState(usize);

impl SemanticStateTransaction for ReferenceState {
    type Branch = Self;
    type Error = Infallible;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(self.clone())
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        *self = branch;
        Ok(())
    }
}

#[derive(Clone)]
struct Ready(Rc<Cell<bool>>);

impl Completion for Ready {
    type Error = Infallible;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(self.0.get())
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

struct ReferenceOutput {
    frame: RealtimeOutputFrame,
    completion: Ready,
}

impl TransitionOutput for ReferenceOutput {
    type Error = Infallible;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.completion.is_complete()
    }

    fn retained_resources(&self) -> usize {
        1
    }
}

#[test]
fn no_default_facade_selects_runs_and_resumes_realtime_with_reference_mechanisms() {
    let config = config();
    let catalog = HeaderCatalog::for_config(&config);
    let preparation = prepare_realtime_model_from_catalog(
        "reference-artifact",
        "reference-checkpoint.safetensors",
        config,
        &catalog,
    )
    .unwrap();
    let inspected = inspect_moshi_realtime(preparation, request()).unwrap();
    let capabilities = capabilities(inspected.requirements());
    let selected = select_inspected_moshi_realtime(inspected, &capabilities);
    let prepared = selected.unwrap();
    let speech = prepared.selected().requirements().speech_schedule().clone();
    let model = PreparedRealtimeModel::new(
        "reference-mechanism",
        prepared.selected(),
        SessionCapabilities::new(true, true, true),
    );
    assert_eq!(model.mechanism(), &"reference-mechanism");
    assert!(model.session_capabilities().persistent_cache());

    let generation = RealtimeGenerationState::<ReferenceState, (), (), Ready>::new(
        ReferenceState(0),
        speech,
        RealtimeSampling::greedy(),
        vec![(), ()],
        None,
    )
    .unwrap();
    let mut scheduler = RealtimeSessionScheduler::new(
        model.session_identity().clone(),
        SchedulerLimits::new(1, 4).unwrap(),
    )
    .unwrap();
    let request = RequestId::new(41);
    scheduler.register(request, generation).unwrap();
    scheduler
        .enqueue(request, RealtimeInputFrame::new(1, vec![7, 8]))
        .unwrap();
    let progress = scheduler
        .run_local_turn(Instant::now(), |_work, input, branch| {
            branch.generation_mut().model_state_mut().0 += 1;
            let completion = Ready(Rc::new(Cell::new(true)));
            branch
                .generation_mut()
                .attach_submission_completion(completion.clone())
                .unwrap();
            Ok::<_, Infallible>(ReferenceOutput {
                frame: RealtimeOutputFrame::new(
                    input.batch(),
                    vec![11],
                    vec![12],
                    vec![12],
                    None,
                    Vec::new(),
                ),
                completion,
            })
        })
        .unwrap();
    assert_eq!(progress.committed.len(), 1);
    assert_eq!(progress.committed[0].2.frame.text_tokens(), &[11]);
    assert_eq!(scheduler.report().completed_work, 1);

    let released = scheduler.release(request).unwrap();
    assert_eq!(released.committed_batch().unwrap().get(), 1);
    let resumed = RequestId::new(42);
    scheduler.resume(resumed, released).unwrap();
    assert_eq!(
        scheduler
            .request_state(resumed)
            .unwrap()
            .generation()
            .model_state()
            .0,
        1
    );
}
