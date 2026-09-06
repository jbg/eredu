//! Scalar rank mechanisms using the complete portable preparation/construction path.

use super::*;
use eredu_architectures::{prepared_execution::*, prepared_sources::PreparedModelSources};

pub(super) fn prepare(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    topology: ParallelTopology,
    rank: usize,
    prompt_cache_persistence: bool,
) -> Result<PreparedModelSources, String> {
    let plan = prepared_adapter::plan(None)
        .with_topology(topology)
        .with_prompt_cache_persistence(prompt_cache_persistence);
    let parallel = eredu_runtime::ParallelLoadRequest::new(
        ParallelRankTopology::new(topology, rank).map_err(|error| error.to_string())?,
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        8,
        CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(2),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let request = eredu_runtime::NormalizedLoadRequest::from_execution_plan(
        &plan,
        eredu_runtime::ResidencyDiagnostics::new(false, false),
        Some(parallel),
    )
    .map_err(|error| error.to_string())?;
    let selected = eredu_architectures::select_preparation(
        inspection,
        &request,
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .map_err(|error| error.to_string())?;
    let admitted = eredu_core::ModelPreparationPlan::from_retained_admission(
        inspection.clone(),
        selected.admission(),
    )
    .map_err(|error| error.to_string())?;
    eredu_architectures::prepared_sources::prepare_model_sources(admitted, selected)
        .map_err(|error| error.to_string())
}

#[derive(Clone)]
struct NumericPreparedCommunication {
    world: Arc<NumericPartitionWorld>,
    manifest: CommunicationManifest,
}

impl NumericPreparedCommunication {
    fn realize(sources: &PreparedModelSources, context: &NumericContext) -> Result<Self, String> {
        let manifest = sources
            .selected()
            .communication_manifest()
            .ok_or("partitioned scalar selection has no communication manifest")?
            .clone();
        let partition = context
            .partition
            .as_ref()
            .ok_or("scalar context has no native rank")?;
        if partition.rank != manifest.rank() {
            return Err("scalar native rank differs from selected communication".into());
        }
        partition
            .world
            .realize_manifest(&manifest)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            world: Arc::clone(&partition.world),
            manifest,
        })
    }
}

struct PartitionAssembler<'a, E> {
    context: &'a NumericContext,
    executable: std::marker::PhantomData<E>,
}

impl<E> PreparedExecutableAssembler<NumericPreparedCommunication> for PartitionAssembler<'_, E> {
    type Executable = E;
    type Output = E;
    type Error = Error;

    fn floating_state_bytes(
        &mut self,
        _: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<std::num::NonZeroU8, Error> {
        Ok(std::num::NonZeroU8::new(4).unwrap())
    }

    fn validate_communication(
        &mut self,
        selected: &CommunicationManifest,
        communication: &NumericPreparedCommunication,
    ) -> Result<(), Error> {
        let rank = self
            .context
            .partition
            .as_ref()
            .ok_or_else(|| Error::backend("scalar context has no native rank"))?;
        if selected != &communication.manifest
            || selected.rank() != rank.rank
            || !Arc::ptr_eq(&rank.world, &communication.world)
        {
            return Err(Error::backend(
                "scalar communication manifest, world, or rank differs from selection",
            ));
        }
        Ok(())
    }

    fn finish(
        self,
        mut parts: PreparedExecutableParts<E, NumericPreparedCommunication>,
    ) -> Result<E, Error> {
        if parts.take_communication().is_none() || parts.take_processor().is_some() {
            return Err(Error::backend(
                "scalar partition lost communication or received raw-media resources",
            ));
        }
        assert_eq!(parts.floating_state_bytes().get(), 4);
        Ok(parts.into_executable())
    }
}

pub(super) fn dense(
    sources: PreparedModelSources,
    context: &NumericContext,
) -> Result<NumericPartitionExecutable, String> {
    let communication = NumericPreparedCommunication::realize(&sources, context)?;
    let route = PartitionedDenseRoute::<
        NumericBackend,
        DeviceState<NumericBackend, NumericHybridLayerState>,
        _,
    >::new(
        context,
        context,
        |resources: PreparedPartitionResources<NumericPreparedCommunication>| {
            let native = resources.into_communication();
            NumericResidentPartitionVisitor {
                world: native.world,
                context: context.clone(),
            }
        },
    );
    construct_prepared_execution(
        sources,
        Some(communication),
        PreparedExecutionRoutes::new().with_partitioned_dense(route),
        PartitionAssembler {
            context,
            executable: std::marker::PhantomData,
        },
    )
    .map_err(|error| error.to_string())
}

pub(super) fn routed(
    sources: PreparedModelSources,
    context: &NumericContext,
    provider_calls: Arc<AtomicUsize>,
    omit_inactive: Option<Arc<AtomicBool>>,
) -> Result<NumericPartitionExecutable, String> {
    let communication = NumericPreparedCommunication::realize(&sources, context)?;
    let visitor = |resources: PreparedPartitionResources<NumericPreparedCommunication>| {
        let native = resources.into_communication();
        NumericRoutedPartitionVisitor {
            world: native.world,
            context: context.clone(),
            provider_calls: Arc::clone(&provider_calls),
            omit_inactive: omit_inactive.clone(),
        }
    };
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    let route = PartitionedRoutedRoute::<NumericBackend, State, State, _, _>::new(
        context, context, visitor, visitor,
    )
    .without_relu2();
    construct_prepared_execution(
        sources,
        Some(communication),
        PreparedExecutionRoutes::new().with_partitioned_routed(route),
        PartitionAssembler {
            context,
            executable: std::marker::PhantomData,
        },
    )
    .map_err(|error| error.to_string())
}

pub(super) fn composite(
    sources: PreparedModelSources,
    context: &NumericContext,
) -> Result<NumericCompositePartitionExecutable, String> {
    let communication = NumericPreparedCommunication::realize(&sources, context)?;
    let route = PartitionedCompositeRoute::<
        NumericBackend,
        DeviceState<NumericBackend, NumericHybridLayerState>,
        _,
    >::new(
        context,
        context,
        |resources: PreparedPartitionResources<NumericPreparedCommunication>| {
            let checkpoint = Arc::clone(resources.target());
            let native = resources.into_communication();
            NumericCompositePartitionVisitor {
                world: native.world,
                context: context.clone(),
                checkpoint,
            }
        },
    );
    construct_prepared_execution(
        sources,
        Some(communication),
        PreparedExecutionRoutes::new().with_partitioned_composite(route),
        PartitionAssembler {
            context,
            executable: std::marker::PhantomData,
        },
    )
    .map_err(|error| error.to_string())
}

pub(super) fn assert_narrow_route_and_native_pairing_rejections() {
    let config = serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":16, "hidden_size":8,
        "intermediate_size":12, "num_hidden_layers":2,
        "hybrid_override_pattern":"ME", "num_attention_heads":2,
        "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":2,
        "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
        "conv_kernel":3, "chunk_size":2, "n_routed_experts":4,
        "n_shared_experts":1, "moe_intermediate_size":6,
        "moe_shared_expert_intermediate_size":6, "num_experts_per_tok":2,
        "n_group":2, "topk_group":1, "num_nextn_predict_layers":0,
        "tie_word_embeddings":false
    });
    let root = numeric_composite_artifact(&config);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
    let sources = prepare(&inspection, topology, 0, false).unwrap();
    let world = Arc::new(NumericPartitionWorld::default());
    let context = NumericContext {
        partition: Some(NumericPartitionContext::new(0, Arc::clone(&world))),
        ..NumericContext::default()
    };
    let native = NumericPreparedCommunication::realize(&sources, &context).unwrap();
    let mut assembler = PartitionAssembler::<NumericPartitionExecutable> {
        context: &context,
        executable: std::marker::PhantomData,
    };
    let wrong_world = NumericPreparedCommunication {
        world: Arc::new(NumericPartitionWorld::default()),
        manifest: native.manifest.clone(),
    };
    assert!(assembler
        .validate_communication(&native.manifest, &wrong_world)
        .is_err());
    let other_sources = prepare(&inspection, topology, 1, false).unwrap();
    assert!(assembler
        .validate_communication(
            other_sources.selected().communication_manifest().unwrap(),
            &native,
        )
        .is_err());
    assert!(NumericPreparedCommunication::realize(&other_sources, &context).is_err());

    let provider_calls = Arc::new(AtomicUsize::new(0));
    let error = routed(sources, &context, Arc::clone(&provider_calls), None)
        .err()
        .expect("a gated/pooling-only route must reject selected ReLU-squared execution");
    assert_eq!(
        error,
        "selected execution has no supplied typed construction mechanism"
    );
    assert_eq!(provider_calls.load(Ordering::Relaxed), 0);
    assert!(world.materializations().is_empty());
    assert!(world.trace().is_empty());
    assert!(context.mechanism_trace().is_empty());
}
