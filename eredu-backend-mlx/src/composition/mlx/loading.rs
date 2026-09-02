//! MLX checkpoint materialization after backend-neutral planning.

use std::sync::Arc;

use eredu_checkpoint::WeightQuantization;

use eredu_architectures::processor_plan::ArtifactArchitecturePlan;
use eredu_architectures::ModelKind;
use eredu_core::{ModelArtifact, ModelPreparationPlan};
use eredu_gguf::MetadataValue as GgufMetadataValue;
use eredu_runtime::{
    select_replicated_text_realization, CacheResidencyPolicy, ReplicatedTextSelectionRequest,
};
use safemlx::Stream;

#[cfg(any(feature = "image", feature = "audio"))]
use crate::composition::mlx::ModelProcessor;

use crate::{
    backend::error::Error,
    backend::MlxModel,
    composition::mlx::{structural, Executable},
    MlxLoadRequest,
};

use super::realization::{
    requires_distributed_stage, AddressableParameterBankBinding,
    CompleteTensorParallelBinding as TensorParallelBinding, FamilyBinding, FixedGgufBinding,
    GgufBinding, QuantizedGgufBinding,
};

/// Opaque MLX model configuration selected before payloads are opened.
#[derive(Debug, Clone)]
pub struct MlxModelConfig {
    pub(crate) plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    pub(crate) selected: MlxSelectedPreparation,
}

impl MlxModelConfig {
    pub(crate) const fn new(
        plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
        selected: MlxSelectedPreparation,
    ) -> Self {
        Self { plan, selected }
    }
}

/// Opaque, authoritative MLX construction policy selected before payloads are opened.
#[derive(Debug, Clone)]
pub struct MlxSelectedPreparation {
    execution: eredu_architectures::replicated_text::SelectedReplicatedTextExecution<
        eredu_runtime::SelectedReplicatedTextRealization,
        SelectedMlxConstruction,
    >,
    session: eredu_core::SessionCapabilities,
}

impl MlxSelectedPreparation {
    const fn new(
        execution: eredu_architectures::replicated_text::SelectedReplicatedTextExecution<
            eredu_runtime::SelectedReplicatedTextRealization,
            SelectedMlxConstruction,
        >,
        session: eredu_core::SessionCapabilities,
    ) -> Self {
        Self { execution, session }
    }

    pub(crate) const fn session_capabilities(&self) -> eredu_core::SessionCapabilities {
        self.session
    }
}

/// Fully resolved construction inputs for non-replicated-text MLX composition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct SelectedMlxConstruction {
    quantization: Option<WeightQuantization>,
    pub(crate) parallel: Option<(
        super::distributed::topology::MlxParallelPlan,
        eredu_runtime::PipelineWireContract,
    )>,
    pub(crate) weight_residency: eredu_runtime::WeightResidency,
    pub(crate) state_residency: CacheResidencyPolicy,
    session: eredu_core::SessionCapabilities,
}

impl SelectedMlxConstruction {
    pub(crate) fn from_request(
        request: MlxLoadRequest,
        session: eredu_core::SessionCapabilities,
    ) -> Result<Self, Error> {
        Ok(Self {
            quantization: request.weight_quantization()?,
            parallel: request.parallel_execution(),
            weight_residency: request.weight_residency(),
            state_residency: request.state_residency().clone(),
            session,
        })
    }

    pub(crate) const fn parallel_topology(
        &self,
    ) -> Option<super::distributed::topology::MlxParallelPlan> {
        match self.parallel {
            Some((topology, _)) => Some(topology),
            None => None,
        }
    }

    pub(crate) const fn parallel_execution(
        &self,
    ) -> Option<(
        super::distributed::topology::MlxParallelPlan,
        eredu_runtime::PipelineWireContract,
    )> {
        self.parallel
    }

    pub(crate) const fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.quantization
    }
}

/// MLX arrays, modules, and media preprocessing from one GGUF artifact.
struct MaterializedGgufModel {
    model: Executable,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
}

/// Proof that a family-specific complete-model loader serves an execution
/// responsibility outside replicated text composition.
pub(crate) struct ExcludedFamilyRoute(());

pub(crate) fn validate_gguf_projector_requirement(
    architecture: eredu_architectures::GgufArchitecture,
    has_projector: bool,
) -> Result<(), Error> {
    let requirement = eredu_architectures::preparation::gguf_composite_artifact_plan(architecture)
        .media_projector_requirement();
    if requirement == eredu_architectures::preparation::GgufMediaProjectorRequirement::Required
        && !has_projector
    {
        return Err(Error::ArchitectureModel(format!(
            "{} preparation omitted its architecture-required media projector",
            architecture.model_kind().canonical_name()
        )));
    }
    Ok(())
}

fn materialize_gguf_model(
    source: &structural::AdmittedGguf,
    projector: Option<&structural::AdmittedGgufProjector>,
    family_route: &ExcludedFamilyRoute,
    options: SelectedMlxConstruction,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Executable, Error> {
    let kind = source.architecture().model_kind();
    let quantization = options.weight_quantization();
    let binding = GgufBinding::for_kind(kind).ok_or_else(|| {
        Error::ArchitectureModel(format!(
            "MLX has no GGUF realization for {}",
            kind.canonical_name()
        ))
    })?;
    validate_gguf_projector_requirement(source.architecture(), projector.is_some())?;
    structural::validate_complete_gguf_quantization(kind, quantization.is_some())?;
    let model = match binding {
        GgufBinding::Quantized(QuantizedGgufBinding::KimiLinear) => {
            let loaded = crate::composition::kimi_linear::load_kimi_linear_gguf_model(
                source,
                family_route,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            Executable::kimi_linear(kind, loaded)?
        }
        GgufBinding::Fixed(FixedGgufBinding::DeepSeekV3) => {
            let loaded = crate::composition::deepseek::load_gguf(
                source,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Executable::deepseek(kind, Box::new(loaded))?
        }
        GgufBinding::Fixed(FixedGgufBinding::DeepSeekV4) => {
            let loaded = crate::composition::deepseek::load_gguf(
                source,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Executable::deepseek(kind, Box::new(loaded))?
        }
        GgufBinding::Quantized(QuantizedGgufBinding::GptOss) => {
            let loaded = crate::composition::gpt_oss::load_gpt_oss_gguf_model(
                source,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            Executable::gpt_oss(kind, loaded)?
        }
        GgufBinding::Fixed(FixedGgufBinding::Inkling) => {
            let loaded = crate::composition::inkling::load_gguf(
                source,
                projector,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Executable::inkling(kind, loaded)?
        }
        GgufBinding::Fixed(FixedGgufBinding::Gemma4) => {
            let loaded = crate::composition::gemma4::load_gguf(
                source,
                projector,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Executable::gemma4(kind, loaded)?
        }
        GgufBinding::Quantized(QuantizedGgufBinding::Llama) => unreachable!(
            "architecture-owned eligibility rejects replicated GGUF before family binding"
        ),
        GgufBinding::Fixed(FixedGgufBinding::MuseGlimmer) => {
            let loaded = crate::composition::muse_glimmer::load_gguf(
                source,
                projector,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Executable::muse_glimmer(kind, loaded)?
        }
        GgufBinding::Quantized(QuantizedGgufBinding::Lfm2) => {
            let loaded = crate::composition::lfm2::load_lfm2_gguf_model(
                source,
                family_route,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            Executable::lfm2(kind, loaded)?
        }
        GgufBinding::Quantized(QuantizedGgufBinding::NemotronH) => {
            let loaded = crate::composition::nemotron_h::load_nemotron_h_gguf_model(
                source,
                family_route,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            Executable::nemotron_h(kind, loaded)?
        }
        GgufBinding::Quantized(QuantizedGgufBinding::Qwen) => {
            let loaded = crate::composition::qwen::load_qwen_gguf_model(
                source,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            Executable::qwen(kind, loaded)?
        }
        GgufBinding::Quantized(
            binding @ (QuantizedGgufBinding::Qwen3Vl | QuantizedGgufBinding::Qwen3VlMoe),
        ) => {
            let projector = projector.expect("required GGUF projector was validated above");
            let loaded = crate::composition::qwen::vl::load_gguf(
                source,
                projector,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            match binding {
                QuantizedGgufBinding::Qwen3Vl => Executable::qwen3_vl(kind, loaded)?,
                QuantizedGgufBinding::Qwen3VlMoe => Executable::qwen3_vl_moe(kind, loaded)?,
                _ => unreachable!(),
            }
        }
        GgufBinding::Quantized(
            binding @ (QuantizedGgufBinding::Qwen35 | QuantizedGgufBinding::Qwen3Next),
        ) => {
            let loaded = crate::composition::qwen::hybrid::load_gguf(
                source,
                projector,
                family_route,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            match binding {
                QuantizedGgufBinding::Qwen35 => Executable::qwen35(kind, loaded)?,
                QuantizedGgufBinding::Qwen3Next => Executable::qwen3_next(kind, loaded)?,
                _ => unreachable!(),
            }
        }
    };
    Ok(model)
}

pub(crate) fn select_preparation(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    policy: eredu_core::PreparationPolicy,
) -> Result<MlxSelectedPreparation, Error> {
    select_preparation_with_grouped_capabilities(
        inspection,
        options,
        policy,
        &super::replicated_text::GROUPED_OPERATION_CAPABILITIES,
    )
}

struct MlxExecutionClassSelection {
    options: MlxLoadRequest,
    policy: eredu_core::PreparationPolicy,
    admitted_session: eredu_core::SessionCapabilities,
}

impl eredu_architectures::replicated_text::ReplicatedTextExecutionClassDispatcher
    for MlxExecutionClassSelection
{
    type Replicated = eredu_runtime::SelectedReplicatedTextRealization;
    type Other = SelectedMlxConstruction;
    type Error = Error;

    fn replicated(
        self,
        requirements: eredu_runtime::ReplicatedTextRequirements,
    ) -> Result<Self::Replicated, Self::Error> {
        let mut request = ReplicatedTextSelectionRequest::new(
            self.options.weight_residency.layers(),
            self.options.state_residency().clone(),
        )
        .with_session(self.admitted_session)
        .with_prompt_cache(matches!(
            self.options.state_residency(),
            CacheResidencyPolicy::Paged(_)
        ))
        .with_exact_completion(true);
        if let Some(topology) = self.policy.topology() {
            request = request.with_topology(topology);
        }
        if let Some(quantization) = self.policy.quantization() {
            request = request.with_quantization(quantization);
        }
        select_replicated_text_realization(
            &requirements,
            &request,
            &super::replicated_text::capabilities(&requirements, &request),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn other(self) -> Result<Self::Other, Self::Error> {
        SelectedMlxConstruction::from_request(self.options, self.admitted_session)
    }
}

pub(crate) fn select_preparation_with_grouped_capabilities(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    policy: eredu_core::PreparationPolicy,
    grouped_capabilities: &[eredu_runtime::GroupedOperationRequirement],
) -> Result<MlxSelectedPreparation, Error> {
    if let Some(kind) = inspection
        .architecture_plan()
        .required_gguf_special_tokens()
    {
        return Err(Error::ArchitectureModel(format!(
            "GGUF {kind:?} media token IDs must be resolved by the facade before MLX preparation"
        )));
    }
    if policy != options.preparation_policy()? {
        return Err(Error::ArchitectureModel(
            "MLX preparation policy does not match the caller request".into(),
        ));
    }
    structural::validate_inspected_preparation(inspection, policy)?;
    let admitted_session = structural::inspected_session_capabilities(inspection, policy)?;
    let grouped_requirements = inspection
        .architecture_plan()
        .grouped_operation_requirements(policy.topology());
    let missing_grouped = grouped_requirements
        .iter()
        .filter(|required| !grouped_capabilities.contains(required))
        .collect::<Vec<_>>();
    if !missing_grouped.is_empty() {
        return Err(Error::ArchitectureModel(format!(
            "backend is missing architecture-required grouped mechanisms: {missing_grouped:?}"
        )));
    }
    let execution = eredu_architectures::replicated_text::dispatch_replicated_text_execution_class(
        inspection,
        policy.topology(),
        MlxExecutionClassSelection {
            options,
            policy,
            admitted_session,
        },
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    Ok(MlxSelectedPreparation::new(execution, admitted_session))
}

pub fn materialize_model_plan(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected: MlxSelectedPreparation,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let MlxSelectedPreparation {
        execution,
        session: _,
    } = selected;
    execution.dispatch(MlxSelectedExecutionMaterializer {
        plan,
        stream,
        weights_stream,
    })
}

struct MlxSelectedExecutionMaterializer<'a> {
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

impl
    eredu_architectures::replicated_text::SelectedReplicatedTextExecutionDispatcher<
        eredu_runtime::SelectedReplicatedTextRealization,
        SelectedMlxConstruction,
    > for MlxSelectedExecutionMaterializer<'_>
{
    type Output = MlxModel;
    type Error = Error;

    fn replicated(
        self,
        selected: eredu_runtime::SelectedReplicatedTextRealization,
    ) -> Result<Self::Output, Self::Error> {
        materialize_replicated_text_plan(self.plan, selected, self.stream, self.weights_stream)
    }

    fn other(self, selected: SelectedMlxConstruction) -> Result<Self::Output, Self::Error> {
        let family_route = ExcludedFamilyRoute(());
        materialize_excluded_model_plan(
            self.plan,
            &family_route,
            selected,
            self.stream,
            self.weights_stream,
        )
    }
}

fn materialize_replicated_text_plan(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    realization: eredu_runtime::SelectedReplicatedTextRealization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(plan.inspection())?;
    let max_cached_shards = realization.residency().max_cached_shards();
    let state_residency = realization.state().policy().clone();
    let architecture_plan = plan.inspection().architecture_plan().clone();
    let artifact = plan.into_artifact();
    let kind = prepared_model_kind(&architecture_plan);
    let executable = match artifact {
        ModelArtifact::SafeTensors {
            path: _,
            configuration,
            tensors,
            shards,
        } => {
            let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                prepared_safetensors_architecture(&architecture_plan)?.clone(),
                tensors,
                shards,
                max_cached_shards,
            )?;
            bind_replicated_text(
                &architecture_plan,
                realization,
                prepared.store(),
                stream,
                weights_stream,
            )?
        }
        ModelArtifact::Gguf { validated, .. } => {
            let architecture = prepared_gguf_plan(&architecture_plan)?.clone();
            let (source, projector) = structural::AdmittedGguf::from_admission(
                architecture,
                architecture_plan.gguf_media_projector().cloned(),
                validated,
            )?;
            if projector.is_some() {
                return Err(Error::ArchitectureModel(
                    "replicated text composition cannot bind a media projector".into(),
                ));
            }
            #[cfg(test)]
            super::path_instrumentation::payload_open();
            let store = Arc::new(
                crate::backend::runtime::checkpoint::store::open_gguf_checkpoint_source(
                    source.checkpoint().clone(),
                    source.plan().checkpoint(),
                    source.plan().tensor_mapping(),
                    max_cached_shards,
                )?,
            );
            bind_replicated_text(
                &architecture_plan,
                realization,
                store,
                stream,
                weights_stream,
            )?
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "unsupported artifact route for replicated text composition".into(),
            ));
        }
    };
    let model = MlxModel::complete(
        Executable::replicated_text(kind, executable)?,
        floating_state_dtype_bytes,
        state_residency,
    );
    attach_processor(model, &architecture_plan)
}

fn materialize_excluded_model_plan(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    family_route: &ExcludedFamilyRoute,
    options: SelectedMlxConstruction,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(plan.inspection())?;
    let state_residency = options.state_residency.clone();
    if let Some(topology) = options
        .parallel_topology()
        .filter(|topology| !topology.is_replicated())
    {
        let kind = prepared_model_kind(plan.inspection().architecture_plan());
        if requires_distributed_stage(kind, topology.topology()) {
            #[cfg(any(feature = "image", feature = "audio"))]
            let processor = ModelProcessor::from_plan(plan.inspection().architecture_plan());
            let model =
                crate::composition::mlx::distributed::pipeline::load_pipeline_model_with_options(
                    plan,
                    options,
                    stream,
                    weights_stream,
                )
                .map(|model| {
                    MlxModel::pipeline(model, floating_state_dtype_bytes, state_residency.clone())
                })?;
            #[cfg(any(feature = "image", feature = "audio"))]
            let model = model.with_processor(processor);
            return Ok(model);
        }
        let architecture_plan = plan.inspection().architecture_plan().clone();
        let artifact = plan.into_artifact();
        return match artifact {
            artifact @ ModelArtifact::Gguf { .. } => materialize_gguf_artifact(
                artifact,
                architecture_plan,
                family_route,
                options,
                stream,
                weights_stream,
            )
            .map(|model| {
                complete_gguf_model(model, floating_state_dtype_bytes, state_residency.clone())
            }),
            ModelArtifact::SafeTensors {
                path: _,
                configuration,
                tensors,
                shards,
            } => {
                let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                    configuration,
                    prepared_safetensors_architecture(&architecture_plan)?.clone(),
                    tensors,
                    shards,
                    options.weight_residency.max_cached_shards(),
                )?;
                let model = materialize_tensor_parallel(&prepared, options, stream, weights_stream)
                    .map(|model| {
                        MlxModel::complete(
                            model,
                            floating_state_dtype_bytes,
                            state_residency.clone(),
                        )
                    })?;
                attach_processor(model, &architecture_plan)
            }
            _ => Err(Error::ArchitectureModel(
                "unsupported artifact route for tensor-parallel composition".into(),
            )),
        };
    }
    let architecture_plan = plan.inspection().architecture_plan().clone();
    let artifact = plan.into_artifact();
    match artifact {
        artifact @ ModelArtifact::Gguf { .. } => materialize_gguf_artifact(
            artifact,
            architecture_plan,
            family_route,
            options,
            stream,
            weights_stream,
        )
        .map(|model| {
            complete_gguf_model(model, floating_state_dtype_bytes, state_residency.clone())
        }),
        ModelArtifact::SafeTensors {
            path: _,
            configuration,
            tensors,
            shards,
        } => {
            let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                prepared_safetensors_architecture(&architecture_plan)?.clone(),
                tensors,
                shards,
                options.weight_residency.max_cached_shards(),
            )?;
            let composite = super::model::CompositeInputProof::from_plan(&architecture_plan);
            let model = materialize_safetensors(
                &prepared,
                composite,
                family_route,
                options,
                stream,
                weights_stream,
            )
            .map(|model| {
                MlxModel::complete(model, floating_state_dtype_bytes, state_residency.clone())
            })?;
            attach_processor(model, &architecture_plan)
        }
        _ => Err(Error::ArchitectureModel(
            "unsupported artifact route for model composition".into(),
        )),
    }
}

pub(super) fn bind_replicated_text(
    architecture_plan: &ArtifactArchitecturePlan,
    selected: eredu_runtime::SelectedReplicatedTextRealization,
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn super::replicated_text::ErasedReplicatedTextExecutable>, Error> {
    let visitor = super::replicated_text::BindingVisitor {
        stream,
        weights_stream,
    };
    eredu_architectures::replicated_text::dispatch_replicated_text_architecture::<
        crate::backend::nn::shared::MlxNeuralBackend,
        _,
    >(architecture_plan, selected, store, stream, visitor)
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

fn inspected_floating_state_dtype_bytes(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<std::num::NonZeroU8, Error> {
    let source = match inspection.format() {
        eredu_core::ArtifactFormat::Gguf => {
            eredu_architectures::preparation::prepared_gguf_floating_state_dtype_source(
                prepared_gguf_plan(inspection.architecture_plan())?,
                inspection.tensors(),
            )
        }
        eredu_core::ArtifactFormat::SafeTensors => {
            eredu_architectures::preparation::prepared_safetensors_floating_state_dtype_source(
                prepared_safetensors_architecture(inspection.architecture_plan())?,
                inspection.tensors(),
            )
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "unsupported artifact format for floating-state dtype inspection".into(),
            ));
        }
    }
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    mlx_floating_state_dtype_bytes(source.dtype()).map_err(|dtype| {
        Error::ArchitectureModel(format!(
            "floating-state dtype source {:?} has unsupported MLX activation dtype {dtype:?}",
            source.checkpoint_tensor()
        ))
    })
}

fn mlx_floating_state_dtype_bytes(
    dtype: &eredu_core::checkpoint::TensorDtype,
) -> Result<std::num::NonZeroU8, eredu_core::checkpoint::TensorDtype> {
    use eredu_core::checkpoint::TensorDtype;

    let bytes = match dtype {
        TensorDtype::F16 | TensorDtype::Bf16 => 2,
        TensorDtype::F32 => 4,
        TensorDtype::F64 | TensorDtype::Complex64 => 8,
        // MLX materializes supported packed embeddings as Float32 activations.
        // These cases are reached only after the architecture schema resolved
        // the exact embedding parameter; they are not a fallback for an
        // unknown checkpoint name.
        TensorDtype::U32 | TensorDtype::Encoded(_) => 4,
        dtype => return Err(dtype.clone()),
    };
    Ok(std::num::NonZeroU8::new(bytes).expect("supported MLX activation widths are nonzero"))
}

#[cfg(test)]
#[allow(
    clippy::items_after_test_module,
    reason = "floating-state dtype tests stay adjacent to dtype resolution"
)]
mod floating_state_dtype_tests {
    use super::{
        inspected_floating_state_dtype_bytes, mlx_floating_state_dtype_bytes,
        reject_complete_tensor_parallel_quantization,
    };
    use crate::composition::mlx::realization::requires_distributed_stage;
    use crate::{backend::DeviceAssignment, native::MlxParallelPlan};
    use eredu_architectures::ModelKind;
    use eredu_checkpoint::WeightQuantization;
    use eredu_core::checkpoint::TensorDtype;
    use eredu_gguf::{GgmlType, MetadataValue, TensorInput, Writer};
    use safemlx::DeviceType;
    use std::collections::BTreeMap;

    fn write_minimal_llama_gguf(path: &std::path::Path, dtype: GgmlType) {
        let metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count".into(), MetadataValue::Uint32(1)),
            ("llama.embedding_length".into(), MetadataValue::Uint32(1)),
            (
                "llama.attention.head_count".into(),
                MetadataValue::Uint32(1),
            ),
            ("llama.feed_forward_length".into(), MetadataValue::Uint32(1)),
            (
                "llama.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
            ("llama.vocab_size".into(), MetadataValue::Uint32(1)),
            ("llama.context_length".into(), MetadataValue::Uint32(1)),
        ]);
        let data = [0_u8; 2];
        let tensor = |name, dimensions| TensorInput {
            name,
            dimensions,
            ggml_type: dtype,
            data: &data,
        };
        let tensors = [
            tensor("token_embd.weight", &[1, 1]),
            tensor("output_norm.weight", &[1]),
            tensor("blk.0.attn_norm.weight", &[1]),
            tensor("blk.0.ffn_norm.weight", &[1]),
            tensor("blk.0.attn_q.weight", &[1, 1]),
            tensor("blk.0.attn_k.weight", &[1, 1]),
            tensor("blk.0.attn_v.weight", &[1, 1]),
            tensor("blk.0.attn_output.weight", &[1, 1]),
            tensor("blk.0.ffn_gate.weight", &[1, 1]),
            tensor("blk.0.ffn_up.weight", &[1, 1]),
            tensor("blk.0.ffn_down.weight", &[1, 1]),
        ];
        Writer::default()
            .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
            .unwrap();
    }

    #[test]
    fn deepseek_pure_tp_uses_distributed_stage_loader() {
        let topology =
            MlxParallelPlan::for_rank(0, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        for kind in [ModelKind::DeepSeekV3, ModelKind::DeepSeekV4] {
            assert!(requires_distributed_stage(kind, topology.topology()));
        }
    }

    #[test]
    fn complete_tensor_parallel_loader_rejects_unbound_quantization() {
        reject_complete_tensor_parallel_quantization(None, "deepseek4").unwrap();
        let error = reject_complete_tensor_parallel_quantization(
            Some(WeightQuantization::MxFp4),
            "deepseek4",
        )
        .unwrap_err();
        assert!(matches!(
            error,
            crate::backend::error::Error::Quantization(message)
                if message.contains("deepseek4")
        ));
    }

    #[test]
    fn specialized_qwen_tp_uses_distributed_stage_loader() {
        let topology =
            MlxParallelPlan::for_rank(0, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        for kind in [
            ModelKind::Qwen3Next,
            ModelKind::Qwen35,
            ModelKind::Qwen3Vl,
            ModelKind::Qwen3VlMoe,
        ] {
            assert!(requires_distributed_stage(kind, topology.topology()));
        }
        assert!(!requires_distributed_stage(
            ModelKind::Qwen3,
            topology.topology()
        ));
    }

    #[test]
    fn expert_parallel_topology_unconditionally_uses_distributed_stage_loader() {
        let topology =
            MlxParallelPlan::for_rank(0, 1, 1, 2, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        assert!(requires_distributed_stage(
            ModelKind::Llama,
            topology.topology()
        ));
    }

    #[cfg(feature = "image")]
    #[test]
    fn mlx_processor_consumes_retained_qwen_plan_after_sidecar_removal() {
        use eredu_core::{ArtifactFormat, ModelConfigurationResolver};

        let root = tempfile::tempdir().unwrap();
        let sidecar = root
            .path()
            .join(eredu_architectures::processor_plan::PROCESSOR_CONFIG_FILENAME);
        std::fs::write(
            &sidecar,
            br#"{
                "size":{"shortest_edge":16,"longest_edge":64},
                "patch_size":2,"temporal_patch_size":2,"merge_size":2,
                "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0]
            }"#,
        )
        .unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3_vl", "image_token_id": 61, "video_token_id": 62,
            "vision_start_token_id": 44, "vision_end_token_id": 45,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_vl_text", "hidden_size": 32,
                "num_hidden_layers": 3, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "rms_norm_eps": 0.000001, "vocab_size": 64,
                "max_position_embeddings": 128, "rope_theta": 1000000.0,
                "rope_scaling": {"mrope_section": [2, 1, 1], "mrope_interleaved": true}
            },
            "vision_config": {
                "depth": 4, "hidden_size": 16, "intermediate_size": 24,
                "num_heads": 4, "num_position_embeddings": 16, "in_channels": 3,
                "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 32, "deepstack_visual_indexes": [1, 3]
            }
        });
        let (configuration, resolved_plan) =
            eredu_architectures::configuration::MODEL_CONFIGURATIONS
                .resolve_safetensors(&config)
                .unwrap()
                .into_parts();
        let architecture_plan = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .artifact_plan(
                root.path(),
                ArtifactFormat::SafeTensors,
                &configuration,
                &eredu_core::checkpoint::TensorCatalog::new([]).unwrap(),
                None,
                resolved_plan,
            )
            .unwrap();
        std::fs::remove_file(sidecar).unwrap();

        assert_eq!(architecture_plan.model_kind(), ModelKind::Qwen3Vl);
        assert!(crate::composition::mlx::ModelProcessor::from_plan(&architecture_plan).is_some());
    }

    #[test]
    fn resolved_floating_dtype_selects_runtime_state_width() {
        for (dtype, bytes) in [
            (TensorDtype::F16, 2),
            (TensorDtype::Bf16, 2),
            (TensorDtype::F32, 4),
            (TensorDtype::F64, 8),
        ] {
            assert_eq!(mlx_floating_state_dtype_bytes(&dtype).unwrap().get(), bytes);
        }
    }

    #[test]
    fn dense_half_gguf_embeddings_select_two_byte_runtime_state() {
        for dtype in [GgmlType::F16, GgmlType::Bf16] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("model.gguf");
            write_minimal_llama_gguf(&path, dtype);
            let inspection = eredu_core::inspect_artifact(
                &path,
                &eredu_architectures::configuration::MODEL_CONFIGURATIONS,
            )
            .unwrap();

            assert_eq!(
                inspected_floating_state_dtype_bytes(&inspection)
                    .unwrap()
                    .get(),
                2
            );
        }
    }

    #[test]
    fn packed_embedding_dtype_uses_known_mlx_materialization_width() {
        assert_eq!(
            mlx_floating_state_dtype_bytes(&TensorDtype::Encoded("F8_E4M3".into()))
                .unwrap()
                .get(),
            4
        );
        assert_eq!(
            mlx_floating_state_dtype_bytes(&TensorDtype::U32)
                .unwrap()
                .get(),
            4
        );
    }

    #[test]
    fn invalid_activation_dtype_does_not_silently_default() {
        assert_eq!(
            mlx_floating_state_dtype_bytes(&TensorDtype::U8),
            Err(TensorDtype::U8)
        );
    }
}

fn prepared_model_kind(plan: &ArtifactArchitecturePlan) -> ModelKind {
    plan.model_kind()
}

pub(super) fn prepared_safetensors_architecture(
    plan: &ArtifactArchitecturePlan,
) -> Result<&eredu_architectures::configuration::SafetensorsArchitecturePlan, Error> {
    plan.safetensors_architecture().ok_or_else(|| {
        Error::ArchitectureModel(
            "SafeTensors preparation omitted its validated architecture plan".into(),
        )
    })
}

fn prepared_gguf_plan(
    plan: &ArtifactArchitecturePlan,
) -> Result<&eredu_architectures::configuration::GgufArchitecturePlan, Error> {
    plan.gguf_plan().ok_or_else(|| {
        Error::ArchitectureModel("GGUF preparation omitted its validated architecture plan".into())
    })
}

fn attach_processor(
    model: MlxModel,
    architecture_plan: &ArtifactArchitecturePlan,
) -> Result<MlxModel, Error> {
    #[cfg(any(feature = "image", feature = "audio"))]
    {
        Ok(model.with_processor(ModelProcessor::from_plan(architecture_plan)))
    }
    #[cfg(not(any(feature = "image", feature = "audio")))]
    {
        let _ = architecture_plan;
        Ok(model)
    }
}

fn complete_gguf_model(
    materialized: MaterializedGgufModel,
    floating_state_dtype_bytes: std::num::NonZeroU8,
    state_residency: CacheResidencyPolicy,
) -> MlxModel {
    let model = MlxModel::complete(
        materialized.model,
        floating_state_dtype_bytes,
        state_residency,
    );
    #[cfg(any(feature = "image", feature = "audio"))]
    let model = model.with_processor(materialized.processor);
    model
}

fn materialize_tensor_parallel(
    artifact: &super::artifact::PreparedSafetensorsArtifact,
    options: SelectedMlxConstruction,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Executable, Error> {
    let kind = artifact.architecture().model_kind();
    let binding = TensorParallelBinding::for_kind(kind).ok_or_else(|| {
        if FamilyBinding::for_kind(kind).is_some() {
            Error::ArchitectureModel(format!(
                "distributed-stage-only {} reached complete tensor-parallel materialization",
                kind.canonical_name()
            ))
        } else {
            Error::ArchitectureModel(format!(
                "MLX has no tensor-parallel model realization for {}",
                kind.canonical_name()
            ))
        }
    })?;
    let topology = options.parallel_topology().ok_or_else(|| {
        Error::Parallel("tensor-parallel materialization requires a topology".into())
    })?;
    if topology.tensor_parallel_size() <= 1
        || topology.pipeline_parallel_size() != 1
        || topology.expert_parallel_size() != 1
    {
        return Err(Error::Parallel(
            "complete executable materialization supports pure tensor parallelism only".into(),
        ));
    }
    if options.weight_residency.parameter_bank_cache().is_some() {
        return Err(Error::Parallel(
            "tensor-parallel model materialization does not compose with independent expert caching"
                .into(),
        ));
    }
    reject_complete_tensor_parallel_quantization(
        options.weight_quantization(),
        kind.canonical_name(),
    )?;
    let execution = options.weight_residency.layers();
    let build = crate::composition::mlx::distributed::topology::ParallelBuildContext::new(
        topology,
        eredu_runtime::ShardingPolicy::Require,
    );
    match binding {
        TensorParallelBinding::Gemma4 => Ok(Executable::gemma4(
            kind,
            crate::composition::gemma4::load_safetensors_tensor_parallel(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )?),
        TensorParallelBinding::GptOss => Ok(Executable::gpt_oss(
            kind,
            crate::composition::gpt_oss::load_gpt_oss_tensor_parallel_model(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )?),
        TensorParallelBinding::Inkling => Ok(Executable::inkling(
            kind,
            crate::composition::inkling::load_safetensors_tensor_parallel(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )?),
        TensorParallelBinding::KimiLinear => Ok(Executable::kimi_linear(
            kind,
            crate::composition::kimi_linear::load_kimi_linear_tensor_parallel_model(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )?),
        TensorParallelBinding::Llama => Ok(Executable::partitioned_llama(
            kind,
            crate::composition::llama::load_partitioned_llama_safetensors(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )?),
        TensorParallelBinding::MuseGlimmer => Ok(Executable::muse_glimmer(
            kind,
            crate::composition::muse_glimmer::load_safetensors_tensor_parallel(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )?),
        TensorParallelBinding::Lfm2 => Ok(Executable::lfm2(
            kind,
            crate::composition::lfm2::load_lfm2_tensor_parallel_model(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )?),
        TensorParallelBinding::NemotronH => Ok(Executable::nemotron_h(
            kind,
            crate::composition::nemotron_h::load_nemotron_h_tensor_parallel_model(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )?),
        TensorParallelBinding::Qwen => Ok(Executable::qwen(
            kind,
            crate::composition::qwen::load_qwen_tensor_parallel_model(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )?),
    }
}

fn materialize_gguf_artifact(
    artifact: ModelArtifact,
    architecture_plan: ArtifactArchitecturePlan,
    family_route: &ExcludedFamilyRoute,
    options: SelectedMlxConstruction,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MaterializedGgufModel, Error> {
    let ModelArtifact::Gguf {
        path: _,
        configuration: _,
        validated,
        ..
    } = artifact
    else {
        return Err(Error::ArchitectureModel(
            "MLX GGUF materializer received a SafeTensors plan".into(),
        ));
    };
    let architecture = prepared_gguf_plan(&architecture_plan)?.clone();
    let projector_plan = architecture_plan.gguf_media_projector().cloned();
    let (source, projector) =
        structural::AdmittedGguf::from_admission(architecture, projector_plan, validated)?;
    let checkpoint = source.checkpoint();
    let metadata = source.metadata();
    validate_gguf_quantization_source(checkpoint, metadata, options.weight_quantization())?;
    #[cfg(any(feature = "image", feature = "audio"))]
    let processor = ModelProcessor::from_plan(&architecture_plan);
    if options
        .parallel_topology()
        .is_some_and(|topology| !topology.is_replicated())
    {
        let model = materialize_gguf_tensor_parallel(
            &source,
            projector.as_ref(),
            options,
            stream,
            weights_stream,
        )?;
        return Ok(MaterializedGgufModel {
            model,
            #[cfg(any(feature = "image", feature = "audio"))]
            processor,
        });
    }
    let model = materialize_gguf_model(
        &source,
        projector.as_ref(),
        family_route,
        options,
        stream,
        weights_stream,
    )?;
    Ok(MaterializedGgufModel {
        model,
        #[cfg(any(feature = "image", feature = "audio"))]
        processor,
    })
}

fn materialize_gguf_tensor_parallel(
    source: &structural::AdmittedGguf,
    projector: Option<&structural::AdmittedGgufProjector>,
    options: SelectedMlxConstruction,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Executable, Error> {
    let architecture = source.architecture();
    let kind = architecture.model_kind();
    let binding = TensorParallelBinding::for_kind(kind).ok_or_else(|| {
        if FamilyBinding::for_kind(kind).is_some() {
            Error::ArchitectureModel(format!(
                "distributed-stage-only {} reached complete GGUF tensor-parallel materialization",
                architecture.metadata_name()
            ))
        } else {
            Error::ArchitectureModel(format!(
                "MLX has no tensor-parallel GGUF realization for {}",
                architecture.metadata_name()
            ))
        }
    })?;
    let topology = options.parallel_topology().ok_or_else(|| {
        Error::Parallel("tensor-parallel GGUF materialization requires a topology".into())
    })?;
    reject_complete_tensor_parallel_quantization(
        options.weight_quantization(),
        architecture.metadata_name(),
    )?;
    let residency = options.weight_residency.layers();
    let build = crate::composition::mlx::distributed::topology::ParallelBuildContext::new(
        topology,
        eredu_runtime::ShardingPolicy::Require,
    );
    match binding {
        TensorParallelBinding::KimiLinear => {
            let model =
                crate::composition::kimi_linear::load_kimi_linear_gguf_tensor_parallel_model(
                    source,
                    residency,
                    build,
                    stream,
                    weights_stream,
                )?;
            Executable::kimi_linear(kind, model)
        }
        TensorParallelBinding::GptOss => {
            let model = crate::composition::gpt_oss::load_gpt_oss_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Executable::gpt_oss(kind, model)
        }
        TensorParallelBinding::Inkling => {
            let model = crate::composition::inkling::load_gguf_tensor_parallel(
                source,
                projector,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Executable::inkling(kind, model)
        }
        TensorParallelBinding::Gemma4 => {
            let model = crate::composition::gemma4::load_gguf_tensor_parallel(
                source,
                projector,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Executable::gemma4(kind, model)
        }
        TensorParallelBinding::Llama => {
            let model = crate::composition::llama::load_partitioned_llama_gguf(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Executable::partitioned_llama(kind, model)
        }
        TensorParallelBinding::MuseGlimmer => {
            let model = crate::composition::muse_glimmer::load_gguf_tensor_parallel(
                source,
                projector,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Executable::muse_glimmer(kind, model)
        }
        TensorParallelBinding::Lfm2 => {
            let model = crate::composition::lfm2::load_lfm2_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Executable::lfm2(kind, model)
        }
        TensorParallelBinding::NemotronH => {
            let model = crate::composition::nemotron_h::load_nemotron_h_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Executable::nemotron_h(kind, model)
        }
        TensorParallelBinding::Qwen => {
            let model = crate::composition::qwen::load_qwen_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Executable::qwen(kind, model)
        }
    }
}

fn reject_complete_tensor_parallel_quantization(
    quantization: Option<WeightQuantization>,
    architecture: &str,
) -> Result<(), Error> {
    if quantization.is_some() {
        return Err(Error::Quantization(format!(
            "load-time quantization is not implemented for complete tensor-parallel {architecture} materialization"
        )));
    }
    Ok(())
}

pub fn validate_gguf_quantization_source(
    source: &crate::backend::runtime::checkpoint::gguf::GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
) -> Result<(), Error> {
    let Some(quantization) = quantization else {
        return Ok(());
    };
    quantization.validate()?;

    let has_packed_companions = source
        .catalog()
        .tensors()
        .any(|tensor| tensor.affine().is_some());
    if has_packed_companions {
        return Err(Error::Quantization(
            "load-time quantization accepts only unquantized F32/F16/BF16 GGUF weights; packed GGUF tensors cannot be implicitly transcoded"
                .into(),
        ));
    }

    let file_type = metadata
        .get("general.file_type")
        .ok_or_else(|| {
            Error::Quantization(
                "GGUF general.file_type metadata is required to verify that load-time quantization is not transcoding packed weights"
                    .into(),
            )
        })?
        .as_i64()
        .ok_or_else(|| {
            Error::Quantization("GGUF general.file_type metadata must be an integer".into())
        })?;
    // llama.cpp's unquantized file types: ALL_F32, MOSTLY_F16, and MOSTLY_BF16.
    if !matches!(file_type, 0 | 1 | 32) {
        return Err(Error::Quantization(format!(
            "load-time quantization accepts only unquantized F32/F16/BF16 GGUF weights; general.file_type={file_type} is already quantized"
        )));
    }
    Ok(())
}

pub(super) fn materialize_safetensors(
    artifact: &super::artifact::PreparedSafetensorsArtifact,
    composite: Option<super::model::CompositeInputProof>,
    family_route: &ExcludedFamilyRoute,
    options: SelectedMlxConstruction,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Executable, Error> {
    let kind = artifact.architecture().model_kind();
    let quantization = options.weight_quantization();
    if let (Some(parameter_bank), Some(non_expert)) = (
        options.weight_residency.parameter_bank_cache(),
        options.weight_residency.ordinary_residency(),
    ) {
        let binding = AddressableParameterBankBinding::for_kind(kind)
            .ok_or_else(|| Error::ArchitectureModel(format!(
                "independent expert caching is unavailable for the normalized {} architecture on MLX",
                kind.canonical_name()
            )))?;
        return match binding {
            AddressableParameterBankBinding::KimiLinear => Ok(Executable::kimi_linear(
                kind,
                crate::composition::kimi_linear::load_kimi_linear_model(
                    artifact,
                    family_route,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::DeepSeek => Ok(Executable::deepseek(
                kind,
                Box::new(crate::composition::deepseek::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?),
            )?),
            AddressableParameterBankBinding::GptOss => Ok(Executable::gpt_oss(
                kind,
                crate::composition::gpt_oss::load_safetensors(
                    artifact,
                    options.weight_residency,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Gemma4 => Ok(Executable::gemma4(
                kind,
                crate::composition::gemma4::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Inkling => Ok(Executable::inkling(
                kind,
                crate::composition::inkling::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Lfm2 => Ok(Executable::lfm2(
                kind,
                crate::composition::lfm2::load_lfm2_model(
                    artifact,
                    family_route,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::MuseGlimmer => Ok(Executable::muse_glimmer(
                kind,
                crate::composition::muse_glimmer::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::NemotronH => Ok(Executable::nemotron_h(
                kind,
                crate::composition::nemotron_h::load_nemotron_h_model(
                    artifact,
                    family_route,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Qwen => Ok(Executable::qwen(
                kind,
                crate::composition::qwen::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Qwen3Next => Ok(Executable::qwen3_next(
                kind,
                crate::composition::qwen::hybrid::load_safetensors_with_residency(
                    artifact,
                    family_route,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Qwen3VlMoe => Ok(Executable::qwen3_vl_moe(
                kind,
                crate::composition::qwen::vl::load_safetensors_with_residency(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Qwen35 => Ok(Executable::qwen35(
                kind,
                crate::composition::qwen::hybrid::load_safetensors_with_residency(
                    artifact,
                    family_route,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
        };
    }
    let execution = options.weight_residency.layers();
    if let Some(quantization) = quantization {
        quantization.validate()?;
    }
    let binding = FamilyBinding::for_kind(kind).ok_or_else(|| {
        Error::ArchitectureModel(format!(
            "{} bounded layer residency is selected through another loading protocol",
            kind.canonical_name()
        ))
    })?;
    match binding {
        FamilyBinding::DeepSeekV3 | FamilyBinding::DeepSeekV4 => Ok(Executable::deepseek(
            kind,
            Box::new(crate::composition::deepseek::load_safetensors(
                artifact,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?),
        )?),
        FamilyBinding::Gemma4 => Ok(Executable::gemma4(
            kind,
            crate::composition::gemma4::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::Inkling => Ok(Executable::inkling(
            kind,
            crate::composition::inkling::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::KimiLinear => Ok(Executable::kimi_linear(kind, {
            crate::composition::kimi_linear::load_kimi_linear_model(
                artifact,
                family_route,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?
        })?),
        FamilyBinding::Llama => unreachable!(
            "architecture-owned eligibility rejects replicated SafeTensors before family binding"
        ),
        FamilyBinding::MuseGlimmer => Ok(Executable::muse_glimmer(
            kind,
            crate::composition::muse_glimmer::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::Qwen => Ok(Executable::qwen(
            kind,
            crate::composition::qwen::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::GptOss => Ok(Executable::gpt_oss(
            kind,
            crate::composition::gpt_oss::load_safetensors(
                artifact,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::Lfm2 => Ok(Executable::lfm2(kind, {
            crate::composition::lfm2::load_lfm2_model(
                artifact,
                family_route,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?
        })?),
        FamilyBinding::NemotronH => Ok(Executable::nemotron_h(kind, {
            crate::composition::nemotron_h::load_nemotron_h_model(
                artifact,
                family_route,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?
        })?),
        FamilyBinding::Qwen3Next => Ok(Executable::qwen3_next(kind, {
            crate::composition::qwen::hybrid::load_safetensors(
                artifact,
                family_route,
                execution,
                quantization,
                stream,
                weights_stream,
            )?
        })?),
        FamilyBinding::Qwen3Vl => Ok(Executable::qwen3_vl(
            kind,
            crate::composition::qwen::vl::load_safetensors(
                artifact,
                execution,
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::Qwen3VlMoe => Ok(Executable::qwen3_vl_moe(
            kind,
            crate::composition::qwen::vl::load_safetensors(
                artifact,
                execution,
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::Qwen35 => {
            let model = crate::composition::qwen::hybrid::load_safetensors(
                artifact,
                family_route,
                execution,
                quantization,
                stream,
                weights_stream,
            )?;
            if let Some(proof) = composite {
                Executable::qwen35_composite(kind, model, proof)
            } else {
                Executable::qwen35(kind, model)
            }
        }
    }
}
