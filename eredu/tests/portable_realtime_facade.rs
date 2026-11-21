use std::{
    cell::Cell,
    collections::BTreeMap,
    convert::Infallible,
    num::NonZeroUsize,
    rc::Rc,
    time::{Duration, Instant},
};

use eredu::api::realtime::{
    prepare_realtime_selection, PreparedRealtimeArtifact, PreparedRealtimeModel,
    RealtimeInputFrame, RealtimeLoadError, RealtimeLoadOptions, RealtimeOutputFrame,
    RealtimeSampling, RealtimeSessionScheduler, RequestId, SchedulerLimits, SessionCapabilities,
};
use eredu_architectures::moshi::{self, prepare_moshi_from_catalog, MoshiConfig};
use eredu_checkpoint::{
    recipe::RecipeCatalog,
    schema::StoredDtypeConstraint,
    store::{StoreError, TensorMetadata},
    validation::{CatalogTensorMetadata, SafetensorsCatalog},
    StoredDtype,
};
use eredu_core::communication::{
    CommunicationCompletionCapabilities, CommunicationCompletionPolicy,
};
use eredu_core::{
    scheduler::{SemanticStateTransaction, TransitionOutput},
    Completion, CompletionCancellationMode,
};
use eredu_runtime::{
    BackendMechanismFacts, CacheResidencyPolicy, NormalizedLoadRequest, RealtimeLoadSupport,
    RealtimeMechanism, RealtimeMechanismFacts, RealtimeMechanismSupport,
    ReplicatedTextMechanismSupport, StateComponentPlacement, StateLifecycleCapabilities,
    StateStorageDtype, WeightLoweringDescriptor, WeightResidency,
};

struct HeaderCatalog(BTreeMap<String, TensorMetadata>);

impl HeaderCatalog {
    fn for_config(config: &MoshiConfig) -> Self {
        let contract = moshi::parameter_contract(config).unwrap();
        let plan = moshi::safetensors_plan(&contract).unwrap();
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

fn request() -> NormalizedLoadRequest {
    let completion = CommunicationCompletionPolicy::new(
        Duration::from_secs(1),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    NormalizedLoadRequest::default()
        .with_state_residency(CacheResidencyPolicy::Device)
        .with_communication_completion_policy(completion)
}

struct ReferenceSupport {
    cache: bool,
}

impl ReferenceSupport {
    const STATE: StateLifecycleCapabilities = StateLifecycleCapabilities::new()
        .with_transactions(true, true)
        .with_reset(true)
        .with_observation_retention(true);
}

impl RealtimeMechanismSupport for ReferenceSupport {
    fn facts(&self) -> RealtimeMechanismFacts {
        use RealtimeMechanism::*;
        RealtimeMechanismFacts::new(
            eredu_nn::NeuralOperatorCapabilities::ALL,
            [
                TensorOperations,
                NeuralOperations,
                ParameterMaterialization,
                ParameterStorage,
                StateStorage,
                CoordinateStorage,
                Sampling,
                Randomness,
                HostConversion,
                ExactCompletion,
                ResourceRetention,
                Transfer,
                Observation,
            ],
            [eredu_runtime::WeightResidencyMechanism::Resident],
            NonZeroUsize::new(1).unwrap(),
            CommunicationCompletionCapabilities::new([
                CompletionCancellationMode::QuarantineUntilComplete,
            ])
            .unwrap(),
            SessionCapabilities::new(true, true, true),
        )
        .with_state_lifecycle(Self::STATE)
    }

    fn supports_lowering(&self, _: &WeightLoweringDescriptor, _: bool) -> bool {
        true
    }

    fn state_component_placements(
        &self,
        _: &eredu_core::cache::StateComponentPolicy,
    ) -> (
        Option<StateComponentPlacement>,
        Option<StateComponentPlacement>,
    ) {
        (self.cache.then_some(StateComponentPlacement::Device), None)
    }
}

impl ReplicatedTextMechanismSupport for ReferenceSupport {
    fn facts(&self, _: StateComponentPlacement) -> BackendMechanismFacts {
        BackendMechanismFacts::new(
            eredu_nn::NeuralOperatorCapabilities::ALL,
            [eredu_runtime::WeightResidencyMechanism::Resident],
            Self::STATE,
        )
    }

    fn supports_direct(&self, _: &WeightLoweringDescriptor) -> bool {
        true
    }
    fn supports_transform(&self, _: &WeightLoweringDescriptor) -> bool {
        true
    }
    fn floating_state_dtype(
        &self,
        source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<StateStorageDtype> {
        (*source == eredu_core::checkpoint::TensorDtype::F32).then_some(StateStorageDtype::F32)
    }

    fn supports_state_component(
        &self,
        _: &[eredu_core::cache::StateTensorDimension],
        _: StateStorageDtype,
        placement: StateComponentPlacement,
    ) -> bool {
        self.cache && placement == StateComponentPlacement::Device
    }
}

impl RealtimeLoadSupport for ReferenceSupport {
    fn independent_state_support(&self) -> &impl ReplicatedTextMechanismSupport {
        self
    }
}

fn voicechat_geometry() -> eredu_architectures::voicechat::state::NeuralStateGeometry {
    use eredu_architectures::{
        nemotron_h,
        voicechat::{
            config::VoiceChatConfig, state::NeuralStateGeometry,
            tts_transformer::TransformerConfig, tts_transformer_specs::TransformerDeclaration,
        },
    };
    let config = VoiceChatConfig::from_json(include_str!(
        "../../eredu-architectures/src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let tts =
        TransformerDeclaration::new(TransformerConfig::new(&config.tts.backbone).unwrap()).unwrap();
    let args = nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type": "nemotron_h", "vocab_size": 16, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 2,
        "hybrid_override_pattern": "M*", "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 8, "ssm_state_size": 4,
        "mamba_num_heads": 4, "mamba_head_dim": 8, "n_groups": 1,
        "conv_kernel": 3, "expand": 2, "chunk_size": 4,
    }))
    .unwrap();
    NeuralStateGeometry::new(&nemotron_h::Declaration::new(args).unwrap(), &tts).unwrap()
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

struct ReferenceStartup;

thread_local! {
    static STATE_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

impl eredu_core::realtime_mechanisms::RealtimeStartupMechanisms for ReferenceStartup {
    type Random = u64;
    type Completion = Ready;

    fn realize_random(&self, seed: u64) -> Result<Self::Random, eredu_core::BackendFailure> {
        Ok(seed)
    }
}

impl eredu_runtime::CategoricalRealtimeState for ReferenceState {
    fn resident_categorical_state(
        _: &eredu_runtime::StateLayout,
    ) -> Result<Self, eredu_core::BackendFailure> {
        STATE_ALLOCATIONS.with(|count| count.set(count.get() + 1));
        Ok(Self(0))
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
    let preparation = prepare_moshi_from_catalog(
        "reference-artifact",
        "reference-checkpoint.safetensors",
        config,
        &catalog,
    )
    .unwrap();
    let support = ReferenceSupport { cache: true };
    let prepared =
        moshi::select_moshi_realtime_with_support(preparation.clone(), &request(), &support)
            .unwrap();
    let speech = voicechat_geometry();
    let [backbone, tts] = speech
        .select_state_realizations(
            &CacheResidencyPolicy::Device,
            support.independent_state_support(),
        )
        .unwrap();
    speech
        .validate_layouts(backbone.layout(), tts.layout())
        .unwrap();
    assert!(speech
        .select_state_realizations(
            &CacheResidencyPolicy::Device,
            &ReferenceSupport { cache: false },
        )
        .is_err());

    let artifact = PreparedRealtimeArtifact::Moshi(preparation);
    let unsupported_residency = RealtimeLoadOptions::Categorical(
        request().with_weight_residency(WeightResidency::layerwise_host(Default::default())),
    );
    assert!(matches!(
        prepare_realtime_selection(&artifact, &unsupported_residency, &support),
        Err(RealtimeLoadError::Categorical(
            moshi::MoshiRealtimePreparationError::Selection(_)
        ))
    ));
    let model = PreparedRealtimeModel::new(
        "reference-mechanism",
        prepared.selected(),
        SessionCapabilities::new(true, true, true),
    );
    assert_eq!(model.mechanism(), &"reference-mechanism");
    assert!(model.session_capabilities().persistent_cache());

    let execution = prepared.execution_descriptor().bind(());
    let startup = ReferenceStartup;
    let invalid_sampling = RealtimeSampling::greedy()
        .with_top_k(Some(i32::MAX as usize + 1), None)
        .unwrap();
    assert!(matches!(
        execution.initialize_generation::<ReferenceState, i32, _>(&startup, invalid_sampling),
        Err(moshi::MoshiRealtimeStartupError::Sampling(_))
    ));
    assert_eq!(
        STATE_ALLOCATIONS.with(Cell::get),
        0,
        "rejected startup must not allocate state"
    );
    let sampling = RealtimeSampling::new(0.7, 0.6, 42).unwrap();
    let generation = execution
        .initialize_generation::<ReferenceState, i32, _>(&startup, sampling)
        .unwrap();
    assert_eq!(generation.random_state(), Some(&42));
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
            branch
                .generation_mut()
                .model_state_mut()
                .model_state_mut()
                .0 += 1;
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
    scheduler.resume(resumed, &mut Some(released)).unwrap();
    assert_eq!(
        scheduler
            .request_state(resumed)
            .unwrap()
            .generation()
            .model_state()
            .model_state()
            .0,
        1
    );
}
