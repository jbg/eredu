//! Native binding of the retained VL decoder and vision unit contracts.

use std::{marker::PhantomData, num::NonZeroUsize, rc::Rc};

use crate::backend::error::Error as MlxError;
use crate::composition::mlx::replicated_text::ErasedPrefillCursor;
use eredu_architectures::lfm3::{
    vision::{
        PolicyVision, PolicyVisionSession, PreparedVisionExecution, PreparedVlDecoder,
        PreparedVlTensorDecoder, SelectedVlRankWeights, SelectedVlWeights, VisionInput,
        VisionModule, VisionPromptPreparation, VlInput, VlPrefill, VlPrefillProgress,
    },
    Block, LanguagePipelineCommunication, LayeredModel, Lfm3Input, PartitionedLayeredModel,
    PreparedLanguagePartition,
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_runtime::{
    DenseDiskStreamReport, EmptyParameterModule, LayerWeightResidency, LayerwiseRuntime,
    ResidencyReport, RuntimeLayerState,
};

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::state::MlxHybridState,
            execution::generic::{
                prepare_layerwise_policy_from_bindings, prepare_parameter_unit_policy,
                MlxSelectedLayerwisePolicy,
            },
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor, Stream,
};

/// Binds both VL components using the policies retained by portable selection.
pub fn bind_vl_weights(
    selected: SelectedVlWeights,
    weights: &Stream,
    stream: &Stream,
) -> Result<(MlxVlVision, MlxVlDecoder), BackendFailure> {
    let ((vision, vision_policy), (decoder, decoder_policy)) = selected.into_parts();
    let vision = MlxVlVision::bind(vision, vision_policy, weights, stream)?;
    let decoder = MlxVlDecoder::bind(decoder, decoder_policy, weights, stream)?;
    Ok((vision, decoder))
}

/// Binds a retained VL rank to the matching native collective group.
pub fn bind_vl_rank(
    selected: SelectedVlRankWeights,
    group: &safemlx::distributed::Group,
    weights: &Stream,
    stream: &Stream,
) -> Result<(MlxVlVision, MlxVlDecoder), BackendFailure> {
    let distributed = crate::backend::MlxDistributedSession::from_manifest(
        selected.communication_manifest(),
        group,
        stream,
    )
    .map_err(failure)?;
    bind_vl_rank_with_session(selected, distributed, weights, stream)
}

pub(super) fn bind_vl_rank_with_session(
    selected: SelectedVlRankWeights,
    distributed: crate::backend::MlxDistributedSession,
    weights: &Stream,
    stream: &Stream,
) -> Result<(MlxVlVision, MlxVlDecoder), BackendFailure> {
    let ((vision, vision_policy), (decoder, decoder_policy), communication) = selected.into_parts();
    let decoder = MlxVlDecoder::bind_partition(
        decoder,
        decoder_policy,
        communication,
        distributed,
        weights,
        stream,
    )?;
    let vision = MlxVlVision::bind(vision, vision_policy, weights, stream)?;
    Ok((vision, decoder))
}

type VlGraphRuntime = LayerwiseRuntime<
    eredu_architectures::lfm3::vision::VlLayeredModel<MlxNeuralBackend>,
    MlxNeuralBackend,
    MlxHybridState,
    MlxSelectedLayerwisePolicy<eredu_architectures::lfm3::vision::VlUnit<MlxNeuralBackend>, ()>,
>;

/// Native binding of the combined portable VL component.
pub struct MlxVlGraph {
    identity: Rc<()>,
    runtime: VlGraphRuntime,
    state_layout: eredu_runtime::StateLayout,
    selected_state: Option<eredu_runtime::SelectedStateRealization>,
    stream: Stream,
}

impl MlxVlGraph {
    /// Supplies MLX mechanism facts to the portable VL selector.
    pub fn select(
        inspection: &eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        execution: &eredu_runtime::ReplicatedTextSelectionRequest,
        processor: &eredu_runtime::ProcessorSelectionRequest,
    ) -> Result<eredu_architectures::lfm3::vision::SelectedVlGraph, BackendFailure> {
        let mechanisms = crate::composition::mlx::loading::MlxPreparationMechanisms::new(
            &crate::composition::mlx::replicated_text::GROUPED_OPERATION_CAPABILITIES,
        );
        eredu_architectures::lfm3::vision::SelectedVlGraph::select(
            inspection,
            execution,
            processor,
            &mechanisms,
        )
        .map_err(failure)
    }

    /// Materializes the exact selected graph and transfers its retained processor proof.
    pub fn bind_selected(
        prepared: eredu_architectures::lfm3::vision::PreparedVlGraph,
        weights: &Stream,
        stream: &Stream,
    ) -> Result<
        (
            Self,
            eredu_runtime::SelectedProcessorExecution,
            Option<eredu_architectures::processor_execution::PreparedProcessor>,
        ),
        BackendFailure,
    > {
        let (architecture, component, selected, processor) = detached_preparation(
            Vec::<safemlx::Array>::new(),
            || {
                prepared
                    .into_layerwise::<MlxNeuralBackend>(stream)
                    .map_err(failure)
            },
            failure,
        )?;
        let (execution, processor_selection) = selected.into_parts();
        let mut graph = Self::bind_with_retention(
            architecture,
            component,
            execution.residency(),
            execution.parameter_conversion_retention(),
            weights,
            stream,
        )?;
        graph.selected_state = Some(execution.state().clone());
        Ok((graph, processor_selection, processor))
    }

    /// Binds the supplied architecture and its exact component resources.
    pub fn bind(
        architecture: eredu_architectures::lfm3::vision::VlLayeredModel<MlxNeuralBackend>,
        prepared: eredu_runtime::PreparedComponent<
            std::sync::Arc<eredu_runtime::ComponentDefinition>,
        >,
        residency: LayerWeightResidency,
        weights: &Stream,
        stream: &Stream,
    ) -> Result<Self, BackendFailure> {
        Self::bind_with_retention(architecture, prepared, residency, None, weights, stream)
    }

    fn bind_with_retention(
        mut architecture: eredu_architectures::lfm3::vision::VlLayeredModel<MlxNeuralBackend>,
        prepared: eredu_runtime::PreparedComponent<
            std::sync::Arc<eredu_runtime::ComponentDefinition>,
        >,
        residency: LayerWeightResidency,
        retention: Option<eredu_core::residency::ParameterConversionRetentionPolicy>,
        weights: &Stream,
        stream: &Stream,
    ) -> Result<Self, BackendFailure> {
        detached_preparation(
            Vec::<safemlx::Array>::new(),
            || {
                if architecture.definition() != prepared.declaration().as_ref() {
                    return Err(failure(eredu_nn::Error::backend(
                        "VL graph bindings differ from the constructed component",
                    )));
                }
                let state_layout = prepared.declaration().state_layout().clone();
                let (_, addresses, source, units, pinned) = prepared.into_parts();
                let (policy, _) = prepare_layerwise_policy_from_bindings(
                    source,
                    &mut architecture,
                    (),
                    PhantomData::<MlxHybridState>,
                    residency,
                    stream,
                    weights,
                    |_| false,
                    addresses,
                    pinned.into_values().flatten().collect(),
                    units,
                )
                .map_err(failure)?;
                policy
                    .residency_manager()
                    .configure_parameter_conversion_retention(
                        retention,
                        eredu_core::residency::ParameterConversionRetentionEligibility::Eligible,
                    )
                    .map_err(failure)?;
                let policy = if residency.is_fully_resident() {
                    MlxSelectedLayerwisePolicy::resident(
                        policy
                            .into_resident(&architecture, stream, PhantomData::<MlxHybridState>)
                            .map_err(failure)?,
                    )
                } else {
                    MlxSelectedLayerwisePolicy::bounded(policy)
                };
                Ok(Self {
                    identity: Rc::new(()),
                    runtime: LayerwiseRuntime::new(architecture, policy),
                    state_layout,
                    selected_state: None,
                    stream: stream.clone(),
                })
            },
            failure,
        )
    }

    /// Creates an independent decoder state for the graph.
    pub fn new_state(&self) -> Result<MlxHybridState, BackendFailure> {
        if let Some(selected) = &self.selected_state {
            let manager =
                crate::composition::mlx::replicated_text::selected_state_manager(selected)
                    .map_err(failure)?;
            return MlxHybridState::from_selected(selected, manager, None).map_err(failure);
        }
        MlxHybridState::device(self.state_layout.clone(), &[]).map_err(failure)
    }

    /// Executes the portable component's admitted ingress or decoder operation.
    pub fn forward(
        &mut self,
        input: eredu_architectures::lfm3::vision::VlInput<'_, MlxTensor>,
        state: &mut MlxHybridState,
    ) -> Result<MlxTensor, BackendFailure> {
        self.runtime
            .forward(input, state, &self.stream)
            .map_err(failure)
    }

    pub(super) fn forward_observed(
        &mut self,
        input: eredu_architectures::lfm3::vision::VlInput<'_, MlxTensor>,
        state: &mut MlxHybridState,
        observer: &mut dyn eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error>,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        self.runtime
            .forward_with_observer(input, state, &self.stream, observer)
            .map_err(eredu_nn::Error::backend_source)
    }

    pub(super) fn trim_parameter_conversions(
        &self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        self.runtime.policy().trim_parameter_conversions()
    }

    /// Advances a prepared request through the same component as cached decoding.
    pub fn advance_prefill(
        &mut self,
        request: &mut VlPrefill<MlxTensor>,
        state: &mut MlxHybridState,
    ) -> Result<Option<VlPrefillProgress<MlxTensor>>, BackendFailure> {
        let stream = self.stream.clone();
        let progress = request
            .advance_input(&stream, |input| {
                self.forward(input, state)
                    .map_err(eredu_nn::Error::backend_source)
            })
            .map_err(failure)?;
        settle_vl_prefill(request).map_err(failure)?;
        Ok(progress)
    }

    /// Starts a retained request with fresh decoder state.
    pub fn start_prefill_request(
        &self,
        preparation: VisionPromptPreparation<MlxTensor>,
        chunk: NonZeroUsize,
    ) -> Result<MlxVlGraphRequest, BackendFailure> {
        Ok(MlxVlGraphRequest {
            owner: self.identity.clone(),
            cursor: Box::new(VlPrefill::new(preparation, chunk)),
            state: self.new_state()?,
        })
    }

    /// Advances a request through the same erased cursor boundary as text sessions.
    pub fn advance_prefill_request(
        &mut self,
        request: &mut MlxVlGraphRequest,
    ) -> Result<Option<VlPrefillProgress<MlxTensor>>, BackendFailure> {
        self.validate_prefill_owner(request)?;
        let cursor = request
            .cursor
            .as_any_mut()
            .downcast_mut::<VlPrefill<MlxTensor>>()
            .ok_or_else(|| failure(eredu_nn::Error::backend("VL prefill cursor type differs")))?;
        self.advance_prefill(cursor, &mut request.state)
    }

    /// Forks a settled request, sharing only immutable input and projection values.
    pub fn fork_prefill_request(
        &self,
        request: &MlxVlGraphRequest,
    ) -> Result<MlxVlGraphRequest, BackendFailure> {
        self.validate_prefill_owner(request)?;
        request.next_step()?;
        Ok(MlxVlGraphRequest {
            owner: request.owner.clone(),
            cursor: request.cursor.clone_box(),
            state: request
                .state
                .isolated_snapshot(&self.stream)
                .map_err(failure)?,
        })
    }

    fn validate_prefill_owner(&self, request: &MlxVlGraphRequest) -> Result<(), BackendFailure> {
        if Rc::ptr_eq(&self.identity, &request.owner) {
            Ok(())
        } else {
            Err(failure(eredu_nn::Error::backend(
                "VL prefill request belongs to a different graph",
            )))
        }
    }

    /// Reports the weight residency of the combined component.
    pub fn residency(
        &self,
    ) -> Result<(ResidencyReport, Option<DenseDiskStreamReport>), BackendFailure> {
        let policy = self.runtime.policy();
        Ok((
            policy.residency_report().map_err(failure)?,
            policy.dense_stream_report().map_err(failure)?,
        ))
    }
}

/// Settled VL request with an independently owned decoder state.
pub struct MlxVlGraphRequest {
    owner: Rc<()>,
    cursor: Box<dyn ErasedPrefillCursor>,
    state: MlxHybridState,
}

impl MlxVlGraphRequest {
    /// Semantic identity of the original prompt.
    pub fn cache_identity(&self) -> Option<&eredu_runtime::PreparedInputCacheIdentity> {
        self.cursor.cache_identity()
    }

    /// Next step at the current settled boundary.
    pub fn next_step(&self) -> Result<eredu_runtime::prefill::PrefillStep, BackendFailure> {
        self.cursor.next_step().map_err(failure)
    }

    /// Retained cursor footprint, excluding decoder state.
    pub fn retained_cursor_bytes(&self) -> Result<usize, BackendFailure> {
        self.cursor.retained_bytes().map_err(failure)
    }

    /// Transfers the decoder state after the final prefix.
    pub fn into_state(self) -> Result<MlxHybridState, BackendFailure> {
        if self.next_step()? == eredu_runtime::prefill::PrefillStep::Complete {
            Ok(self.state)
        } else {
            Err(failure(eredu_nn::Error::backend(
                "VL prefill is incomplete",
            )))
        }
    }
}

pub(super) fn settle_vl_prefill(
    cursor: &mut VlPrefill<MlxTensor>,
) -> Result<(), safemlx::error::Exception> {
    // A decoder slice can complete while the full prompt remains lazy.
    let result = safemlx::transforms::eval(cursor.retained_values().map(MlxTensor::as_array));
    if result.is_err() {
        cursor.invalidate();
    }
    result
}

impl ErasedPrefillCursor for VlPrefill<MlxTensor> {
    fn clone_box(&self) -> Box<dyn ErasedPrefillCursor> {
        Box::new(self.clone())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn decoder_position(&self) -> usize {
        self.decoder_position()
    }

    fn next_step(&self) -> Result<eredu_runtime::prefill::PrefillStep, MlxError> {
        self.next_step()
            .map_err(|error| MlxError::Other(Box::new(error)))
    }

    fn cache_identity(&self) -> Option<&eredu_runtime::PreparedInputCacheIdentity> {
        self.cache_identity()
    }

    fn token_count(&self) -> usize {
        self.token_count()
    }

    fn retained_bytes(&self) -> Result<usize, MlxError> {
        let unknown = || MlxError::ArchitectureModel("VL cursor storage is unavailable".into());
        let arrays = self
            .retained_values()
            .map(MlxTensor::as_array)
            .collect::<Vec<_>>();
        let survey = safemlx::inspect_storage(&arrays)?;
        if survey.values.iter().any(Option::is_none) {
            return Err(unknown());
        }
        survey.backings.into_iter().try_fold(
            self.retained_metadata_bytes().ok_or_else(unknown)?,
            |bytes, backing| {
                bytes
                    .checked_add(backing.allocator_capacity_bytes.ok_or_else(unknown)?)
                    .ok_or_else(unknown)
            },
        )
    }
}

type VisionPolicy = MlxSelectedLayerwisePolicy<VisionModule<MlxNeuralBackend>, ()>;
type DecoderPolicy = MlxSelectedLayerwisePolicy<Block<MlxNeuralBackend>, ()>;
type Decoder = LayerwiseRuntime<
    LayeredModel<MlxNeuralBackend>,
    MlxNeuralBackend,
    MlxHybridState,
    DecoderPolicy,
>;

fn failure(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
    BackendFailure::from_error(error)
}

/// MLX vision adapter for an already prepared VL component.
pub struct MlxVlVision {
    identity: Rc<()>,
    execution: PolicyVision<MlxNeuralBackend, VisionPolicy>,
    stream: Stream,
}

impl MlxVlVision {
    /// Binds the exact vision invocations with the selected weight residency.
    pub fn bind(
        prepared: PreparedVisionExecution,
        residency: LayerWeightResidency,
        weights: &Stream,
        stream: &Stream,
    ) -> Result<Self, BackendFailure> {
        detached_preparation(
            Vec::<safemlx::Array>::new(),
            || {
                let (declaration, source, addresses, bindings) = prepared.into_parts();
                let (policy, _) = prepare_parameter_unit_policy(
                    source,
                    &mut EmptyParameterModule,
                    (),
                    residency,
                    stream,
                    weights,
                    |_| false,
                    addresses.clone(),
                    Vec::new(),
                    bindings,
                    Vec::new(),
                )
                .map_err(failure)?;
                let policy = if residency.is_fully_resident() {
                    let units = declaration
                        .units()
                        .map(|unit| VisionModule::new(&declaration, unit, stream))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(failure)?;
                    MlxSelectedLayerwisePolicy::resident(
                        policy.into_resident_units(units, stream).map_err(failure)?,
                    )
                } else {
                    MlxSelectedLayerwisePolicy::bounded(policy)
                };
                Ok(Self {
                    identity: Rc::new(()),
                    execution: PolicyVision::new(declaration, addresses, policy)
                        .map_err(failure)?,
                    stream: stream.clone(),
                })
            },
            failure,
        )
    }

    /// Runs one admitted vision invocation.
    pub fn forward(
        &mut self,
        input: VisionInput<'_, MlxTensor>,
    ) -> Result<(MlxTensor, MlxTensor), BackendFailure> {
        self.execution.forward(input, &self.stream).map_err(failure)
    }

    /// Drains admitted visual ingress through the selected vision policy.
    pub fn project_prompt(
        &mut self,
        mut input: VisionPromptPreparation<MlxTensor>,
    ) -> Result<eredu_architectures::lfm3::vision::ProjectedVisionPrompt<MlxTensor>, BackendFailure>
    {
        while let Some(part) = input.next_media() {
            let (_, rows) = self.forward(part)?;
            input.accept_projection(rows).map_err(failure)?;
        }
        input.finish(&self.stream).map_err(failure)
    }

    /// Starts a unit-by-unit vision invocation.
    pub fn begin(
        &mut self,
        input: VisionInput<'_, MlxTensor>,
    ) -> Result<MlxVlVisionSession<'_>, BackendFailure> {
        self.execution
            .begin(input, &self.stream)
            .map(MlxVlVisionSession)
            .map_err(failure)
    }

    /// Reports weight residency from the bound policy.
    pub fn residency(
        &self,
    ) -> Result<(ResidencyReport, Option<DenseDiskStreamReport>), BackendFailure> {
        let policy = self.execution.policy();
        Ok((
            policy.residency_report().map_err(failure)?,
            policy.dense_stream_report().map_err(failure)?,
        ))
    }
}

/// A native realization of the architecture's controlled vision invocation.
pub struct MlxVlVisionSession<'a>(PolicyVisionSession<'a, MlxNeuralBackend, VisionPolicy>);

impl MlxVlVisionSession<'_> {
    /// Returns the next architecture unit.
    pub fn next_unit(&self) -> Option<eredu_architectures::lfm3::vision::VisionUnit> {
        self.0.next_unit()
    }

    /// Advances one unit.
    pub fn advance(
        &mut self,
    ) -> Result<eredu_architectures::lfm3::vision::PolicyVisionProgress<MlxTensor>, BackendFailure>
    {
        self.0.advance().map_err(failure)
    }
}

type Pipeline = crate::backend::runtime::execution::pipeline::MlxComponentPipeline<
    PartitionedLayeredModel<MlxNeuralBackend>,
    MlxHybridState,
>;

enum DecoderExecution {
    Layerwise {
        runtime: Decoder,
        collective: Option<crate::backend::runtime::distributed::Group>,
    },
    Pipeline {
        runtime: Pipeline,
        epoch: DistributedCommitEpoch,
        prefix: std::num::NonZeroUsize,
        publication: eredu_runtime::PartitionOutputPublication,
        output: eredu_runtime::PartitionOutputAuthority,
    },
}

/// MLX decoder adapter for an already prepared VL backbone.
pub struct MlxVlDecoder {
    identity: Rc<()>,
    execution: DecoderExecution,
    state_layout: eredu_runtime::StateLayout,
    global_layer_start: usize,
    stream: Stream,
}

impl MlxVlDecoder {
    /// Binds the retained decoder resource plan and selected residency.
    pub fn bind(
        prepared: PreparedVlDecoder,
        residency: LayerWeightResidency,
        weights: &Stream,
        stream: &Stream,
    ) -> Result<Self, BackendFailure> {
        Self::bind_selected(residency, weights, stream, None, || {
            let (architecture, prepared) = prepared.into_layerwise::<MlxNeuralBackend>(stream)?;
            let layout = prepared.declaration().state_layout().clone();
            Ok((architecture, prepared, layout))
        })
    }

    /// Binds the selected rank-local decoder to its matching tensor collective.
    pub fn bind_tensor_parallel(
        prepared: PreparedVlTensorDecoder,
        residency: LayerWeightResidency,
        group: &safemlx::distributed::Group,
        weights: &Stream,
        stream: &Stream,
    ) -> Result<Self, BackendFailure> {
        let rank = prepared.rank();
        if rank.tensor_parallel_rank() != group.rank()
            || rank.tensor_parallel_size() != group.size()
        {
            return Err(failure(eredu_nn::Error::backend(
                "LFM3 VL collective differs from retained tensor rank",
            )));
        }
        let collective = crate::backend::runtime::distributed::Group::uncontracted(group);
        Self::bind_selected(residency, weights, stream, Some(collective), || {
            prepared.into_layerwise::<MlxNeuralBackend>(stream)
        })
    }

    fn bind_selected(
        residency: LayerWeightResidency,
        weights: &Stream,
        stream: &Stream,
        collective: Option<crate::backend::runtime::distributed::Group>,
        construct: impl FnOnce() -> Result<
            (
                LayeredModel<MlxNeuralBackend>,
                eredu_runtime::PreparedComponent<
                    std::sync::Arc<eredu_runtime::ComponentDefinition>,
                >,
                eredu_runtime::StateLayout,
            ),
            eredu_nn::Error,
        >,
    ) -> Result<Self, BackendFailure> {
        detached_preparation(
            Vec::<safemlx::Array>::new(),
            || {
                let (mut architecture, prepared, state_layout) = construct().map_err(failure)?;
                let (_, addresses, source, units, pinned) = prepared.into_parts();
                let (policy, _) = prepare_layerwise_policy_from_bindings(
                    source,
                    &mut architecture,
                    (),
                    PhantomData::<MlxHybridState>,
                    residency,
                    stream,
                    weights,
                    |_| false,
                    addresses,
                    pinned.into_values().flatten().collect(),
                    units,
                )
                .map_err(failure)?;
                let policy = if residency.is_fully_resident() {
                    MlxSelectedLayerwisePolicy::resident(
                        policy
                            .into_resident(&architecture, stream, PhantomData::<MlxHybridState>)
                            .map_err(failure)?,
                    )
                } else {
                    MlxSelectedLayerwisePolicy::bounded(policy)
                };
                Ok(Self {
                    identity: Rc::new(()),
                    execution: DecoderExecution::Layerwise {
                        runtime: LayerwiseRuntime::new(architecture, policy),
                        collective,
                    },
                    state_layout,
                    global_layer_start: 0,
                    stream: stream.clone(),
                })
            },
            failure,
        )
    }

    fn bind_partition(
        prepared: PreparedLanguagePartition,
        residency: LayerWeightResidency,
        communication: LanguagePipelineCommunication,
        distributed: crate::backend::MlxDistributedSession,
        weights: &Stream,
        stream: &Stream,
    ) -> Result<Self, BackendFailure> {
        let rank = prepared.rank();
        let world = distributed.native_world();
        if rank.topology().world_size() < 2
            || rank.topology().world_size() != world.size()
            || rank.global_rank() != world.rank()
        {
            return Err(failure(eredu_nn::Error::backend(
                "LFM3 VL world differs from retained pipeline rank",
            )));
        }
        let prefix = prepared.vocabulary_prefix().ok_or_else(|| {
            failure(eredu_nn::Error::backend(
                "LFM3 VL pipeline requires a vocabulary prefix limit",
            ))
        })?;
        let partition = prepared.partition();
        let state_layout = partition
            .state()
            .ok_or_else(|| {
                failure(eredu_nn::Error::backend(
                    "LFM3 VL stage has no cache layout",
                ))
            })?
            .layout()
            .clone();
        let global_layer_start = rank
            .layer_range(prepared.complete_layer_count())
            .map_err(failure)?
            .start;
        let (architecture, source, invocations, pinned, units) =
            prepared.into_layerwise::<MlxNeuralBackend, MlxHybridState>(stream)?;
        let addressed = units
            .into_iter()
            .enumerate()
            .map(|(ordinal, unit)| {
                (
                    invocations
                        .storage_layout()
                        .address(ordinal)
                        .expect("retained language unit"),
                    unit,
                )
            })
            .collect();
        let publication = communication.plan.publication().ok_or_else(|| {
            failure(eredu_nn::Error::backend(
                "VL partition omitted output publication",
            ))
        })?;
        let output = communication
            .plan
            .publication_authority(&communication.manifest)
            .map_err(failure)?
            .expect("publication already checked");
        let (runtime, _) = Pipeline::with_pinned_session(
            architecture,
            source,
            invocations,
            pinned,
            addressed,
            communication.manifest,
            communication.plan,
            communication.tensor_group,
            eredu_runtime::PipelineActivationDtype::Float32,
            distributed,
            residency,
            weights,
            stream,
        )?;
        Ok(Self {
            identity: Rc::new(()),
            execution: DecoderExecution::Pipeline {
                runtime,
                epoch: DistributedCommitEpoch::FIRST,
                prefix,
                publication,
                output,
            },
            state_layout,
            global_layer_start,
            stream: stream.clone(),
        })
    }

    pub(super) fn partition_sampling_context(
        &self,
    ) -> Option<(
        &crate::backend::runtime::distributed::Group,
        &eredu_runtime::PartitionCommunicationAuthority,
        &Stream,
        usize,
    )> {
        let DecoderExecution::Pipeline {
            runtime,
            publication,
            output,
            ..
        } = &self.execution
        else {
            return None;
        };
        Some((
            runtime.distributed().selected_group(publication.group)?,
            runtime.communication_authority(),
            &self.stream,
            output.owner_group_rank(),
        ))
    }

    pub(super) fn partition_public_output(&self) -> bool {
        match &self.execution {
            DecoderExecution::Pipeline { output, .. } => output.local_public_output(),
            DecoderExecution::Layerwise { .. } => true,
        }
    }

    /// Starts a prepared multimodal request with fresh decoder state.
    pub fn begin_prefill<'a>(
        &'a mut self,
        vision: &'a mut MlxVlVision,
        prompt: VisionPromptPreparation<MlxTensor>,
        chunk: std::num::NonZeroUsize,
    ) -> Result<MlxVlPrefill<'a>, BackendFailure> {
        let request = self.start_prefill(vision, prompt, chunk)?;
        Ok(MlxVlPrefill {
            decoder: self,
            vision,
            request,
        })
    }

    /// Starts an owned request that can be advanced without retaining native borrows.
    pub fn start_prefill(
        &mut self,
        vision: &MlxVlVision,
        prompt: VisionPromptPreparation<MlxTensor>,
        chunk: NonZeroUsize,
    ) -> Result<MlxVlPrefillCursor, BackendFailure> {
        self.start_prefill_checked(vision, Ok((prompt, chunk)))
    }

    /// Agrees prompt preparation and fresh state allocation before exposing a request.
    pub fn start_prefill_checked(
        &mut self,
        vision: &MlxVlVision,
        prepared: Result<(VisionPromptPreparation<MlxTensor>, NonZeroUsize), BackendFailure>,
    ) -> Result<MlxVlPrefillCursor, BackendFailure> {
        let state = self.new_state();
        let prepared = prepared.and_then(|prompt| Ok((prompt, state?)));
        let (cursor, state) = self.admit_prefill(prepared)?;
        Ok(MlxVlPrefillCursor {
            decoder: self.identity.clone(),
            vision: vision.identity.clone(),
            cursor,
            state,
        })
    }

    pub(super) fn admit_prefill<S>(
        &mut self,
        prepared: Result<((VisionPromptPreparation<MlxTensor>, NonZeroUsize), S), BackendFailure>,
    ) -> Result<(VlPrefill<MlxTensor>, S), BackendFailure> {
        let prepared = prepared.map(|((prompt, chunk), state)| {
            let chunk = match &self.execution {
                DecoderExecution::Pipeline { prefix, .. } => chunk.min(*prefix),
                DecoderExecution::Layerwise { .. } => chunk,
            };
            (prompt, chunk, state)
        });
        let descriptor = prepared.as_ref().map_or([0; 8], |(prompt, chunk, _)| {
            prompt.prefill_descriptor(*chunk)
        });
        let agreed = self.agree_prefill(descriptor, prepared.is_ok());
        let (prompt, chunk, state) = prepared?;
        agreed?;
        Ok((VlPrefill::new(prompt, chunk), state))
    }

    /// Advances one request step through its original selected executors.
    pub fn advance_prefill(
        &mut self,
        vision: &mut MlxVlVision,
        request: &mut MlxVlPrefillCursor,
    ) -> Result<Option<VlPrefillProgress<MlxTensor>>, BackendFailure> {
        let owner = if Rc::ptr_eq(&self.identity, &request.decoder)
            && Rc::ptr_eq(&vision.identity, &request.vision)
        {
            Ok(())
        } else {
            Err(failure(eredu_nn::Error::backend(
                "VL prefill cursor belongs to a different executable",
            )))
        };
        self.advance_prefill_step(vision, &mut request.cursor, &mut request.state, owner)
    }

    pub(super) fn advance_prefill_step(
        &mut self,
        vision: &mut MlxVlVision,
        cursor: &mut VlPrefill<MlxTensor>,
        state: &mut MlxHybridState,
        local_admission: Result<(), BackendFailure>,
    ) -> Result<Option<VlPrefillProgress<MlxTensor>>, BackendFailure> {
        use eredu_runtime::prefill::PrefillStep;

        let local_step = local_admission.and_then(|()| cursor.next_step().map_err(failure));
        let (phase, start, end) = match &local_step {
            Ok(PrefillStep::PrepareInput) => (1, 0, 0),
            Ok(PrefillStep::Decoder(range)) => (2, range.start as u64, range.end as u64),
            Ok(PrefillStep::Complete) => (3, 0, 0),
            Err(_) => (0, 0, 0),
        };
        let descriptor = [
            0x564c505246535445,
            phase,
            start,
            end,
            cursor.token_count() as u64,
            0,
            0,
            0,
        ];
        let agreed = self.agree_prefill(descriptor, local_step.is_ok());
        if let Err(error) = local_step.and(agreed) {
            if matches!(&self.execution, DecoderExecution::Pipeline { .. }) {
                cursor.invalidate();
            }
            return Err(error);
        }

        let stream = self.stream.clone();
        let progress = cursor
            .advance_input_checked(&stream, |input| {
                let ready = self.agree_prefill(
                    [0x564c505246494e50, phase, start, end, 0, 0, 0, 0],
                    input.is_ok(),
                );
                let input = input?;
                ready.map_err(eredu_nn::Error::backend_source)?;
                match input {
                    VlInput::Media(media) => vision.forward(media).map(|(_, rows)| rows),
                    VlInput::Projected(prompt) => self.forward(Lfm3Input::Vision(prompt), state),
                    VlInput::Text(_) => unreachable!("VL prefill uses a projected prompt"),
                }
                .map_err(eredu_nn::Error::backend_source)
            })
            .map_err(failure)
            .and_then(|progress| {
                settle_vl_prefill(cursor).map_err(failure)?;
                Ok(progress)
            });
        let agreed = self.agree_prefill(
            [0x564c505246444f4e, phase, start, end, 0, 0, 0, 0],
            progress.is_ok(),
        );
        if progress.is_err() || agreed.is_err() {
            cursor.invalidate();
        }
        progress.and_then(|progress| {
            agreed?;
            Ok(progress)
        })
    }

    fn agree_prefill(&mut self, descriptor: [u64; 8], success: bool) -> Result<(), BackendFailure> {
        if let DecoderExecution::Pipeline { runtime, .. } = &mut self.execution {
            runtime.agree_descriptor(descriptor, success, &self.stream)?;
        }
        Ok(())
    }

    pub(super) fn reset_state(&mut self, state: &mut MlxHybridState) -> Result<(), BackendFailure> {
        let replacement = self.new_state();
        let agreed = self.agree_prefill(
            [0x564c524553455400, 0, 0, 0, 0, 0, 0, 0],
            replacement.is_ok(),
        );
        let replacement = replacement?;
        agreed?;
        *state = replacement;
        Ok(())
    }

    /// Creates fresh decoder cache and convolution state for this binding.
    pub fn new_state(&self) -> Result<MlxHybridState, BackendFailure> {
        MlxHybridState::device_with_global_layer_start(
            self.state_layout.clone(),
            self.global_layer_start,
            &[],
        )
        .map_err(failure)
    }

    /// Runs one decoder pass through the shared layerwise driver.
    pub fn forward(
        &mut self,
        input: Lfm3Input<'_, MlxTensor>,
        state: &mut MlxHybridState,
    ) -> Result<MlxTensor, BackendFailure> {
        self.forward_checked(Ok(input), state)
    }

    /// Agrees local input and state preparation before running the selected decoder.
    pub fn forward_checked(
        &mut self,
        input: Result<Lfm3Input<'_, MlxTensor>, BackendFailure>,
        state: &mut MlxHybridState,
    ) -> Result<MlxTensor, BackendFailure> {
        match &mut self.execution {
            DecoderExecution::Layerwise {
                runtime,
                collective,
            } => match collective {
                Some(group) => runtime.forward_parallel(input?, state, group, &self.stream),
                None => runtime.forward(input?, state, &self.stream),
            }
            .map_err(failure),
            DecoderExecution::Pipeline {
                runtime,
                epoch,
                prefix,
                ..
            } => {
                let prepared = input.and_then(|input| {
                    if input.vl_pipeline_sequence_len().map_err(failure)? > prefix.get() {
                        return Err(failure(eredu_nn::Error::backend(
                            "VL input exceeds the retained pipeline prefix limit",
                        )));
                    }
                    let descriptor = input
                        .vl_pipeline_descriptor(&self.stream)
                        .map_err(failure)?;
                    let candidate = state
                        .fork_prediction_target_state(&self.stream)
                        .map_err(failure)?;
                    let next = epoch.next().ok_or_else(|| {
                        failure(eredu_nn::Error::backend("VL pipeline epoch exhausted"))
                    })?;
                    Ok((input, descriptor, candidate, next))
                });
                let agreement = runtime.agree_descriptor(
                    [
                        0x564c505245504152,
                        epoch.value(),
                        state.offset() as u64,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    prepared.is_ok(),
                    &self.stream,
                );
                let (input, descriptor, mut candidate, next) = prepared?;
                agreement?;
                let current = *epoch;
                *epoch = next;
                let roots = input
                    .vl_pipeline_roots()
                    .map(|tensor| tensor.as_array().clone())
                    .collect();
                let (scores, outcome) = runtime.forward(
                    input,
                    &mut candidate,
                    descriptor,
                    current,
                    &self.stream,
                    roots,
                    |state| {
                        state
                            .layers()
                            .iter()
                            .flat_map(RuntimeLayerState::retained_values)
                            .map(|tensor| tensor.as_array().clone())
                            .collect()
                    },
                    |output| Ok((output.clone(), Vec::new())),
                )?;
                if outcome != DistributedCommitOutcome::Committed(current) {
                    return Err(failure(eredu_nn::Error::backend(
                        "VL pipeline did not commit its decoder pass",
                    )));
                }
                *state = candidate;
                Ok(scores)
            }
        }
    }

    /// Reports weight residency from the bound policy.
    pub fn residency(
        &self,
    ) -> Result<(ResidencyReport, Option<DenseDiskStreamReport>), BackendFailure> {
        match &self.execution {
            DecoderExecution::Layerwise { runtime, .. } => {
                let policy = runtime.policy();
                Ok((
                    policy.residency_report().map_err(failure)?,
                    policy.dense_stream_report().map_err(failure)?,
                ))
            }
            DecoderExecution::Pipeline { runtime, .. } => {
                Ok((runtime.residency_report()?, runtime.dense_stream_report()?))
            }
        }
    }
}

/// MLX realization of one admitted VL request through decoder prefill.
pub struct MlxVlPrefill<'a> {
    decoder: &'a mut MlxVlDecoder,
    vision: &'a mut MlxVlVision,
    request: MlxVlPrefillCursor,
}

impl MlxVlPrefill<'_> {
    /// Advances a media projection or a decoder prefix.
    pub fn advance(&mut self) -> Result<Option<VlPrefillProgress<MlxTensor>>, BackendFailure> {
        self.decoder.advance_prefill(self.vision, &mut self.request)
    }

    /// Transfers the committed decoder state after the last prefix.
    pub fn into_state(self) -> Result<MlxHybridState, BackendFailure> {
        self.request.into_state()
    }
}

/// Owned VL prefill progress and its request-local decoder state.
pub struct MlxVlPrefillCursor {
    decoder: Rc<()>,
    vision: Rc<()>,
    cursor: VlPrefill<MlxTensor>,
    state: MlxHybridState,
}

impl MlxVlPrefillCursor {
    /// Reports the scheduled request action before borrowing the executors.
    pub fn next_step(&self) -> Result<eredu_runtime::prefill::PrefillStep, BackendFailure> {
        self.cursor.next_step().map_err(failure)
    }

    /// Semantic identity shared with the original prepared prompt.
    pub fn cache_identity(&self) -> Option<&eredu_runtime::PreparedInputCacheIdentity> {
        self.cursor.cache_identity()
    }

    /// Transfers the committed decoder state after the last prefix.
    pub fn into_state(self) -> Result<MlxHybridState, BackendFailure> {
        if self.cursor.is_complete() {
            Ok(self.state)
        } else {
            Err(failure(eredu_nn::Error::backend(
                "LFM3 VL prefill is incomplete",
            )))
        }
    }
}

/// Retained VL media preparation over the shared MLX processor.
#[cfg(feature = "image")]
pub struct MlxVlMedia {
    processor: crate::composition::mlx::ModelProcessor,
    admission: eredu_architectures::lfm3::vision::VisionPromptAdmission,
}

#[cfg(feature = "image")]
impl MlxVlMedia {
    /// Retains processor and admission geometry independently of checkpoint weights.
    pub fn new(
        artifact: &eredu_architectures::lfm3::vision::PreparedVisionLanguage,
    ) -> Result<Self, BackendFailure> {
        use eredu_architectures::lfm3::vision::VisionPromptAdmission;
        Ok(Self {
            processor: crate::composition::mlx::ModelProcessor::from_prepared(artifact.processor()),
            admission: VisionPromptAdmission::new(artifact).map_err(failure)?,
        })
    }

    /// Retains the shared visual processor for a prepared Omni checkpoint.
    pub fn for_omni(
        artifact: &eredu_architectures::lfm3::omni::PreparedOmni,
    ) -> Result<Self, BackendFailure> {
        Ok(Self {
            processor: crate::composition::mlx::ModelProcessor::from_prepared(
                artifact.vision_processor(),
            ),
            admission: artifact.vision_admission().map_err(failure)?,
        })
    }

    /// Prepares decoded media and retains the shared request's semantic cache identity.
    pub fn prepare<E>(
        &self,
        request: &eredu_core::TokenizedMultimodalRequest,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<
        crate::backend::runtime::media::PreparedModelInput,
        eredu_core::MultimodalPreparationFailure<BackendFailure, E>,
    >
    where
        E: std::error::Error + 'static,
    {
        use crate::backend::runtime::media::ProcessorPreparationError;
        use eredu_core::MultimodalPreparationFailure as Failure;

        self.processor
            .prepare_portable_input(request, encode_text)
            .map_err(|error| match error {
                ProcessorPreparationError::Capability(error) => Failure::Capability(error),
                ProcessorPreparationError::Backend(error) => Failure::Backend(failure(error)),
                ProcessorPreparationError::Text(error) => Failure::Text(error),
            })
    }

    /// Admits shared prepared input for the VL prefill cursor without copying tensor payloads.
    pub fn admit(
        &self,
        prepared: &crate::backend::runtime::media::PreparedModelInput,
    ) -> Result<VisionPromptPreparation<MlxTensor>, BackendFailure> {
        self.admit_prompt(&crate::composition::mlx::MlxModelInput::from_prepared(
            prepared,
        ))
    }

    /// Admits the same opaque prompt used by ordinary MLX generation.
    pub fn admit_prompt(
        &self,
        input: &crate::composition::mlx::MlxModelInput,
    ) -> Result<VisionPromptPreparation<MlxTensor>, BackendFailure> {
        input.with_borrowed(|input| self.admit_borrowed(input))
    }

    fn admit_borrowed(
        &self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
    ) -> Result<VisionPromptPreparation<MlxTensor>, BackendFailure> {
        use crate::backend::runtime::media::input::{
            prepared_composite_input, MlxTensorInputInspector,
        };

        let prepared = prepared_composite_input(input).map_err(failure)?;
        self.admission
            .admit_with_identity(&prepared, &MlxTensorInputInspector, input.cache_identity())
            .map_err(failure)
    }
}
