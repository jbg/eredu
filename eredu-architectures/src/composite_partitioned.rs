//! Authoritative typed preparation for dense partitioned composite models.
//!
//! This module consumes one already selected composite admission. It does not
//! inspect artifacts independently, select communication, or reinterpret the
//! admitted rank topology. Family-owned construction supplies the exact local
//! geometry and boundary before the generic partition visitor is
//! entered.

use eredu_nn::{AttentionCache, AuxiliaryConvolutionState, Tensor};
use eredu_runtime::{
    ArchitectureParameters, ArchitecturePartition, LocalModelLayout,
    PartitionedLayeredArchitecture, ReplicatedTextMaterializationTask, RuntimeStateComponents,
};

use crate::partitioned_execution::{
    derive_partitioned_local_layout, prepare_partitioned, visit_composite_partitioned_architecture,
    CompositePartitionedArchitectureVisitor, PartitionedDispatchError,
    PreparedPartitionedAdmission, SelectedPartitionedAdmission,
};
use crate::replicated_text::{
    composite_config, qwen_hybrid_composite_with_formats, qwen_vl_with_formats, selected_formats,
    selected_linear_formats, CompositeConfig, CompositeTextRequirements,
    SelectedCompositeTextRealization,
};

/// Architecture-owned production decision for one selected composite partition.
///
/// Backends consume this decision without inspecting family configuration.
/// Unsupported selections fail before payload materialization; they are not
/// redirected into a family-owned pipeline implementation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CompositePartitionedProductionDecision {
    /// The typed resident partition contract is complete for production binding.
    Resident,
    /// The selected model is not admitted by the neutral composite runtime.
    Unsupported(&'static str),
}

/// Returns whether a normalized composite plan has an exact prediction-free
/// routed partition foundation.
///
/// This payload-free predicate exists for callers that must keep unsupported
/// expert-parallel products on their established path before partition
/// admission. It does not select topology, communication, or a realization.
pub fn routed_composite_partition_supported(
    plan: &crate::processor_plan::ArtifactArchitecturePlan,
) -> bool {
    matches!(
        composite_config(plan),
        Ok(Some(CompositeConfig::Gemma4(args)))
            if args.text.layer_schedule.iter().any(|policy| {
                policy.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe
            })
    ) || matches!(
        composite_config(plan),
        Ok(Some(CompositeConfig::QwenVl(args))) if args.text.is_moe()
    ) || matches!(
        composite_config(plan),
        Ok(Some(CompositeConfig::Muse(args))) if args.is_moe()
    ) || matches!(
        composite_config(plan),
        Ok(Some(CompositeConfig::Inkling(args)))
            if args.text_config.has_sparse_moe_layers()
                && args
                    .mtp_config
                    .as_ref()
                    .is_none_or(|prediction| prediction.num_nextn_predict_layers == 0)
    ) || matches!(
        composite_config(plan),
        Ok(Some(CompositeConfig::QwenHybrid(args)))
            if args.text.is_moe() && args.text.mtp_num_hidden_layers == 0
    )
}

/// Returns the architecture-fixed hidden width transported between pipeline
/// owners of one non-decoder composite group.
pub(crate) fn composite_group_continuation_geometry(
    requirements: &CompositeTextRequirements,
    group: usize,
    source_pipeline: usize,
    pipeline_stages: usize,
) -> Result<Option<(i32, bool, Option<i32>)>, String> {
    let config = composite_config(requirements.inspection().architecture_plan())
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "selected composite has no normalized family configuration".to_owned())?;
    let width = match config {
        CompositeConfig::Gemma4(args) => match group {
            0 => args
                .vision
                .as_ref()
                .map(|vision| (vision.hidden_size, false, None)),
            1 => args
                .audio
                .as_ref()
                .map(|audio| (audio.hidden_size, false, None)),
            _ => None,
        },
        CompositeConfig::Muse(args) => (group == 0)
            .then(|| {
                args.vision_config
                    .as_ref()
                    .map(|vision| (vision.hidden_size, false, None))
            })
            .flatten(),
        CompositeConfig::QwenVl(args) => {
            (group == 0).then_some((args.vision.hidden_size, false, None))
        }
        CompositeConfig::QwenHybrid(args) => (group == 0)
            .then(|| {
                args.vision.as_ref().map(|vision| {
                    (
                        vision.hidden_size,
                        true,
                        Some(vision.num_position_embeddings),
                    )
                })
            })
            .flatten(),
        CompositeConfig::Inkling(args) => match group {
            0 => match args.vision_config.as_ref() {
                Some(vision) => {
                    let specs = vision.layer_specs();
                    let range = eredu_core::balanced_contiguous_range(
                        specs.len(),
                        pipeline_stages,
                        source_pipeline,
                        false,
                    )
                    .map_err(|error| error.to_string())?;
                    Some((specs[range.end - 1].1, false, None))
                }
                None => None,
            },
            1 => args
                .audio_config
                .as_ref()
                .map(|audio| (audio.text_hidden_size, false, None)),
            _ => None,
        },
    };
    Ok(width)
}

/// Returns the architecture-owned learned-context schema for one composite edge.
pub(crate) fn composite_partition_boundary_schema(
    requirements: &CompositeTextRequirements,
    source_group: usize,
    destination_group: usize,
    source_pipeline: usize,
    pipeline_stages: usize,
) -> Result<Option<eredu_runtime::BoundaryWireSchema>, String> {
    let config = composite_config(requirements.inspection().architecture_plan())
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "selected composite has no normalized family configuration".to_owned())?;
    if source_group != 0 || !matches!(destination_group, 0 | 1) {
        return Ok(None);
    }
    let continuation = source_group == destination_group;
    let (vision, text_hidden, identity, unbatched) = match &config {
        CompositeConfig::QwenVl(args) => (
            &args.vision,
            args.text.hidden_size,
            if continuation {
                "qwen_vl.vision_continuation"
            } else {
                "qwen_vl.vision_to_decoder"
            },
            false,
        ),
        CompositeConfig::QwenHybrid(args) => {
            let Some(vision) = args.vision.as_ref() else {
                return Ok(None);
            };
            (
                vision,
                args.text.hidden_size,
                if continuation {
                    "qwen_conditional.vision_continuation"
                } else {
                    "qwen_conditional.vision_to_decoder"
                },
                continuation,
            )
        }
        _ => return Ok(None),
    };
    let deepstack_count = if continuation {
        let range = eredu_core::balanced_contiguous_range(
            vision.layer_count(),
            pipeline_stages,
            source_pipeline,
            false,
        )
        .map_err(|error| error.to_string())?;
        (0..range.end)
            .filter(|layer| {
                vision
                    .layer_policy(*layer)
                    .is_some_and(|policy| policy.deepstack_merger.is_some())
            })
            .count()
    } else {
        vision.deepstack_layer_count()
    };
    if matches!(config, CompositeConfig::QwenVl(_)) {
        let CompositeConfig::QwenVl(args) = config else {
            unreachable!()
        };
        return crate::qwen::vl::vision_partition_boundary_schema(
            args,
            continuation,
            deepstack_count,
        )
        .map(Some)
        .map_err(|error| error.to_string());
    }
    use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
    let primary_shape = if unbatched {
        vec![Dim::Sequence, Dim::Fixed(vision.hidden_size)]
    } else {
        vec![Dim::Batch, Dim::Sequence, Dim::Fixed(text_hidden)]
    };
    eredu_runtime::BoundaryWireSchema::new(
        identity,
        eredu_runtime::BoundaryTensorSpec::new("hidden", primary_shape, Dtype::Activation),
        (0..deepstack_count).map(|index| {
            eredu_runtime::BoundaryTensorSpec::new(
                format!("deepstack.{index}"),
                [Dim::Batch, Dim::Sequence, Dim::Fixed(text_hidden)],
                Dtype::Activation,
            )
        }),
    )
    .map(Some)
    .map_err(|error| error.to_string())
}

/// Selects the exact production binding without opening payloads or rebuilding topology.
pub fn composite_partitioned_production_decision(
    selected: &SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
) -> CompositePartitionedProductionDecision {
    let routed = matches!(
        selected.base(),
        SelectedCompositeTextRealization::Routed { .. }
    );
    if routed
        != selected
            .requirements()
            .execution()
            .routed_execution()
            .is_some()
    {
        return CompositePartitionedProductionDecision::Unsupported(
            "selected composite decoder strategy differs from its admission",
        );
    }
    let plan = selected
        .requirements()
        .execution()
        .inspection()
        .architecture_plan();
    if plan.safetensors_architecture().is_none() && plan.gguf_plan().is_none() {
        return CompositePartitionedProductionDecision::Unsupported(
            "composite partition binding requires an indexed SafeTensors or admitted GGUF plan",
        );
    }
    match composite_config(plan) {
        Ok(Some(CompositeConfig::Gemma4(_))) => CompositePartitionedProductionDecision::Resident,
        Ok(Some(CompositeConfig::QwenVl(_))) => CompositePartitionedProductionDecision::Resident,
        Ok(Some(CompositeConfig::Muse(args))) if !routed && !args.is_moe() => {
            CompositePartitionedProductionDecision::Resident
        }
        Ok(Some(CompositeConfig::Muse(args))) if routed && args.is_moe() => {
            CompositePartitionedProductionDecision::Resident
        }
        Ok(Some(CompositeConfig::Inkling(args)))
            if !routed
                && !args.text_config.has_sparse_moe_layers()
                && args
                    .mtp_config
                    .as_ref()
                    .is_none_or(|prediction| prediction.num_nextn_predict_layers == 0) =>
        {
            CompositePartitionedProductionDecision::Resident
        }
        Ok(Some(CompositeConfig::Inkling(args)))
            if routed
                && args.text_config.has_sparse_moe_layers()
                && args
                    .mtp_config
                    .as_ref()
                    .is_none_or(|prediction| prediction.num_nextn_predict_layers == 0) =>
        {
            CompositePartitionedProductionDecision::Resident
        }
        Ok(Some(CompositeConfig::QwenHybrid(args))) if args.text.mtp_num_hidden_layers == 0 => {
            CompositePartitionedProductionDecision::Resident
        }
        Ok(Some(
            CompositeConfig::Muse(_) | CompositeConfig::Inkling(_) | CompositeConfig::QwenHybrid(_),
        )) => CompositePartitionedProductionDecision::Unsupported(
            "selected composite configuration is inconsistent with prediction-free decoder admission",
        ),
        Ok(None) | Err(_) => CompositePartitionedProductionDecision::Unsupported(
            "selected artifact has no supported composite partition architecture",
        ),
    }
}

/// One compact routed bank owned at the PP x EP intersection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompositeExpertBankOwnership {
    unit: usize,
    global_expert: usize,
    owner_local_expert: usize,
}

impl CompositeExpertBankOwnership {
    /// Architecture-global decoder unit containing the bank.
    pub const fn unit(&self) -> usize {
        self.unit
    }
    /// Checkpoint-global expert identity used for addressable storage.
    pub const fn global_expert(&self) -> usize {
        self.global_expert
    }
    /// Dense ordinal within this EP owner's compact bank.
    pub const fn owner_local_expert(&self) -> usize {
        self.owner_local_expert
    }
    /// Stable addressable bank key; the compact ordinal is never used here.
    pub const fn bank_key(&self) -> eredu_runtime::ParameterBankKey {
        eredu_runtime::ParameterBankKey::new(self.unit, self.global_expert)
    }
}

/// Family geometry bound to one exact selected routed-expert realization.
#[derive(Debug, Clone)]
pub struct RoutedCompositePartitionFoundation<G, S> {
    geometry: G,
    realization: crate::ExpertRealizationPlan<S>,
    banks: Vec<CompositeExpertBankOwnership>,
}

impl<G, S> RoutedCompositePartitionFoundation<G, S> {
    /// Exact family-local TP/PP geometry.
    pub const fn geometry(&self) -> &G {
        &self.geometry
    }
    /// Immutable selected expert ownership and localized bank specifications.
    pub const fn expert_realization(&self) -> &crate::ExpertRealizationPlan<S> {
        &self.realization
    }
    /// Compact banks physically owned by this PP x EP rank.
    pub fn expert_banks(&self) -> &[CompositeExpertBankOwnership] {
        &self.banks
    }
}

fn routed_composite_foundation<G, S: Clone>(
    geometry: G,
    text_group: &str,
    text_units: std::ops::Range<usize>,
    total_units: usize,
    sparse_units: impl IntoIterator<Item = usize>,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<S>,
) -> Result<RoutedCompositePartitionFoundation<G, S>, String> {
    let expected_units = eredu_core::balanced_contiguous_range(
        total_units,
        topology.pipeline_parallel_size(),
        topology.pipeline_parallel_rank(),
        false,
    )
    .map_err(|error| error.to_string())?;
    if text_units != expected_units
        || realization.expert_parallel_size() != topology.expert_parallel_size()
        || realization.expert_parallel_rank() != topology.expert_parallel_rank()
    {
        return Err("routed composite plan differs from its Cartesian rank".into());
    }
    let expected_experts = eredu_core::balanced_contiguous_range(
        realization.global_expert_count(),
        topology.expert_parallel_size(),
        topology.expert_parallel_rank(),
        false,
    )
    .map_err(|error| error.to_string())?
    .collect::<Vec<_>>();
    if realization.local_global_group_indices() != expected_experts {
        return Err("routed composite expert ownership is not the selected balanced range".into());
    }
    let sparse_units = sparse_units
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let selected_units = realization
        .unit_specs()
        .keys()
        .map(|(group, unit)| (group.as_str(), *unit))
        .collect::<std::collections::BTreeSet<_>>();
    let expected_schedule = sparse_units
        .iter()
        .map(|unit| (text_group, *unit))
        .collect::<std::collections::BTreeSet<_>>();
    if selected_units != expected_schedule {
        return Err("routed composite expert unit schedule drifted".into());
    }
    let banks = sparse_units
        .into_iter()
        .filter(|unit| text_units.contains(unit))
        .flat_map(|unit| {
            expected_experts.iter().copied().enumerate().map(
                move |(owner_local_expert, global_expert)| CompositeExpertBankOwnership {
                    unit,
                    global_expert,
                    owner_local_expert,
                },
            )
        })
        .collect();
    Ok(RoutedCompositePartitionFoundation {
        geometry,
        realization: realization.clone(),
        banks,
    })
}

/// Binds selected sparse Gemma 4 ownership to exact optional-root geometry.
pub fn routed_gemma4_partition_foundation(
    args: &crate::gemma4::FamilyConfig,
    layout: &LocalModelLayout,
    groups: impl IntoIterator<Item = (impl AsRef<str>, std::ops::Range<usize>)>,
    ownership: &eredu_runtime::PartitionOwnership,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<
    RoutedCompositePartitionFoundation<
        crate::gemma4::PartitionLocalGeometry,
        eredu_nn::GroupedGatedProductSpec,
    >,
    String,
> {
    let geometry =
        crate::gemma4::parallel::routed_partition_local_geometry(args, layout, groups, ownership)
            .map_err(|error| error.to_string())?;
    let units = args
        .text
        .layer_schedule
        .iter()
        .enumerate()
        .filter_map(|(unit, policy)| {
            (policy.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe)
                .then_some(unit)
        });
    routed_composite_foundation(
        geometry.clone(),
        crate::gemma4::TEXT_EXECUTION_GROUP,
        geometry.text_units(),
        args.text.num_hidden_layers(),
        units,
        topology,
        realization,
    )
}

/// Binds selected sparse Muse-Glimmer ownership to exact optional-root geometry.
pub fn routed_muse_partition_foundation(
    args: &crate::muse_glimmer::DecoderConfig,
    layout: &LocalModelLayout,
    groups: impl IntoIterator<Item = (impl AsRef<str>, std::ops::Range<usize>)>,
    ownership: &eredu_runtime::PartitionOwnership,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<
    RoutedCompositePartitionFoundation<
        crate::muse_glimmer::PartitionLocalGeometry,
        eredu_nn::GroupedGatedProductSpec,
    >,
    String,
> {
    let geometry = crate::muse_glimmer::parallel::routed_partition_local_geometry(
        args, layout, groups, ownership,
    )
    .map_err(|error| error.to_string())?;
    let units =
        0..usize::try_from(args.num_hidden_layers).map_err(|_| "Muse layer count exceeds usize")?;
    routed_composite_foundation(
        geometry.clone(),
        crate::muse_glimmer::TEXT_EXECUTION_GROUP,
        geometry.text_units(),
        args.num_hidden_layers as usize,
        units,
        topology,
        realization,
    )
}

/// Binds selected sparse Inkling ownership while excluding embedded prediction.
pub fn routed_inkling_partition_foundation(
    args: &crate::inkling::ModelArgs,
    layout: &LocalModelLayout,
    groups: impl IntoIterator<Item = (impl AsRef<str>, std::ops::Range<usize>)>,
    ownership: &eredu_runtime::PartitionOwnership,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<crate::inkling::ExpertBankRealization>,
) -> Result<
    RoutedCompositePartitionFoundation<
        crate::inkling::PartitionLocalGeometry,
        crate::inkling::ExpertBankRealization,
    >,
    String,
> {
    let geometry =
        crate::inkling::parallel::routed_partition_local_geometry(args, layout, groups, ownership)
            .map_err(|error| error.to_string())?;
    let units = args
        .text_config
        .layer_schedule
        .iter()
        .enumerate()
        .filter_map(|(unit, policy)| {
            (policy.feed_forward == crate::inkling::FeedForwardPolicy::SparseMoe).then_some(unit)
        });
    routed_composite_foundation(
        geometry.clone(),
        crate::inkling::TEXT_EXECUTION_GROUP,
        geometry.text_units(),
        args.text_config.num_hidden_layers as usize,
        units,
        topology,
        realization,
    )
}

/// Binds selected sparse conditional-Qwen ownership while excluding MTP depth.
pub fn routed_conditional_qwen_partition_foundation(
    args: &crate::qwen::hybrid::ParsedHybridConfig,
    layout: &LocalModelLayout,
    groups: impl IntoIterator<Item = (impl AsRef<str>, std::ops::Range<usize>)>,
    ownership: &eredu_runtime::PartitionOwnership,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<
    RoutedCompositePartitionFoundation<
        crate::qwen::hybrid::ConditionalPartitionLocalGeometry,
        eredu_nn::GroupedGatedProductSpec,
    >,
    String,
> {
    let geometry = crate::qwen::hybrid::routed_conditional_partition_local_geometry(
        args, layout, groups, ownership,
    )
    .map_err(|error| error.to_string())?;
    let units = 0..usize::try_from(args.text.num_hidden_layers)
        .map_err(|_| "Qwen layer count exceeds usize")?;
    routed_composite_foundation(
        geometry.clone(),
        crate::decoder::TARGET_EXECUTION_GROUP,
        geometry.target_units(),
        args.text.num_hidden_layers as usize,
        units,
        topology,
        realization,
    )
}

/// Architecture-local routed execution retained with a prepared composite partition.
///
/// The grouped plan has already been localized against the family geometry and
/// Cartesian rank. A backend visitor supplies only tensor movement and provider
/// mechanisms; it must not reconstruct expert ownership or route cardinality.
#[derive(Debug, Clone)]
pub struct PreparedCompositeRoutedExecution {
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: crate::routed_text::RoutedGroupedPlan,
    routes_by_unit: std::collections::BTreeMap<usize, usize>,
    owner_units: std::collections::BTreeMap<usize, usize>,
    owner_unit_count: usize,
    tensor_reductions: std::collections::BTreeMap<usize, (usize, usize)>,
    hidden_width: usize,
    tensor_output_width: Option<usize>,
    expert_group: Option<eredu_runtime::CommunicationGroupId>,
}

/// Opaque architecture-selected construction for one composite partition executor.
///
/// This value retains group transport kinds, resolved boundary schemas, output
/// publication, collective placement, and any routed provider schedule. A
/// backend supplies only its unit policy, parallel context, tensor allocator,
/// and route-movement mechanism.
pub struct PreparedCompositeExecutorPlan {
    tensor_group: Option<eredu_runtime::CommunicationGroupId>,
    structure: crate::partitioned_execution::PreparedCompositeExecutorStructure,
    strategy: PreparedCompositeUnitStrategy,
}

enum PreparedCompositeUnitStrategy {
    Direct,
    Routed {
        provider: crate::PlannedResidentGatedProduct,
        plan: crate::routed_text::RoutedGroupedPlan,
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
    },
    RoutedCollective {
        provider: crate::PlannedResidentGatedProduct,
        plan: crate::routed_text::RoutedGroupedPlan,
        expert_group: eredu_runtime::CommunicationGroupId,
        tensor_group: Option<eredu_runtime::CommunicationGroupId>,
        waves: crate::partitioned_execution::RoutedExpertCollectiveWaveSchedule,
    },
}

impl PreparedCompositeExecutorPlan {
    fn new<A, B, S, G, W>(
        prepared: &PreparedPartitionedAdmission<
            A,
            SelectedCompositeTextRealization,
            CompositeTextRequirements,
            G,
            W,
        >,
        publication: crate::partitioned_execution::PublicationValueDescriptor,
        routed: Option<PreparedCompositeRoutedExecution>,
    ) -> Result<Self, String>
    where
        B: eredu_nn::NeuralBackend,
        S: eredu_runtime::RuntimeState<B>,
        A: crate::composite_execution::CompositeArchitecture<B, S>
            + eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>,
    {
        let selected = prepared.selected();
        let group_count = selected.partition().graph().groups().len();
        let group_kinds = (0..group_count)
            .map(|group| prepared.architecture().group_transport(group).kind)
            .collect::<Vec<_>>();
        let route_schemas = selected
            .boundary_routes()
            .iter()
            .map(|route| (route.route().route, route.schema().clone()))
            .collect::<Vec<_>>();
        validate_executor_routes(selected.boundary_routes(), &route_schemas)?;
        let topology = selected.topology();
        let tensor_group = selected.tensor_group();
        let strategy = match routed {
            Some(routed) => {
                let plan = routed.plan.gated().cloned().ok_or_else(|| {
                    "composite executor requires a gated routed realization".to_owned()
                })?;
                let provider = crate::PlannedResidentGatedProduct::new_partitioned_with_routes(
                    routed.owner_group.clone(),
                    plan.clone(),
                    routed.routes_by_unit.clone(),
                )
                .map_err(|error| error.to_string())?;
                if topology.pipeline_parallel_size() > 1 && topology.expert_parallel_size() > 1 {
                    let output_width = routed.tensor_output_width.unwrap_or(
                        usize::try_from(publication.output_width())
                            .map_err(|_| "composite output width exceeds usize")?,
                    );
                    let waves = crate::partitioned_execution::routed_expert_collective_wave_schedule_with_unit_owners_and_tensor_order(
                        &plan,
                        &routed.owner_group,
                        &routed.owner_units,
                        &routed.tensor_reductions,
                        routed.owner_unit_count,
                        topology.tensor_parallel_size(),
                        topology.tensor_parallel_rank(),
                        topology.pipeline_parallel_size(),
                        routed.hidden_width,
                        output_width,
                    )?;
                    if plan.expert_parallel_size() <= 1 {
                        return Err(
                            "routed pipeline collective waves require expert parallelism".into(),
                        );
                    }
                    if waves.stage_count() <= 1 {
                        return Err(
                            "routed expert collective waves require pipeline parallelism".into(),
                        );
                    }
                    let expert_group = routed
                        .expert_group
                        .ok_or_else(|| "routed composite has no expert group".to_owned())?;
                    PreparedCompositeUnitStrategy::RoutedCollective {
                        provider,
                        plan: plan.into(),
                        expert_group,
                        tensor_group,
                        waves,
                    }
                } else {
                    PreparedCompositeUnitStrategy::Routed {
                        provider,
                        plan: plan.into(),
                        expert_group: routed.expert_group,
                    }
                }
            }
            None => PreparedCompositeUnitStrategy::Direct,
        };
        let structure = crate::partitioned_execution::PreparedCompositeExecutorStructure::prepare::<
            A,
            B,
            S,
            G,
            W,
        >(
            prepared.architecture(),
            selected.partition(),
            group_kinds,
            selected.activation_dtype(),
            route_schemas,
            publication,
            tensor_group,
            topology,
        )?;
        Ok(Self {
            tensor_group,
            structure,
            strategy,
        })
    }

    /// Opaque tensor group needed by the backend communication realizer.
    pub const fn communication_tensor_group(&self) -> Option<eredu_runtime::CommunicationGroupId> {
        self.tensor_group
    }

    /// Binds backend mechanisms to the complete architecture-selected executor contract.
    #[allow(clippy::too_many_arguments)]
    pub fn bind<A, B, S, P, F, Movement>(
        self,
        architecture: A,
        policy: P,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        movement: Movement,
    ) -> Result<
        crate::partitioned_execution::CompositePartitionExecutor<
            A,
            B,
            S,
            P,
            F,
            crate::partitioned_execution::SelectedCompositePartitionUnitStrategy<
                crate::PlannedResidentGatedProduct,
                Movement,
            >,
        >,
        eredu_nn::Error,
    >
    where
        B: eredu_runtime::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context,
            > + eredu_runtime::CommunicationBackend
            + eredu_nn::TensorParallelGroupedNeuralBackend,
        S: eredu_runtime::RuntimeState<B>,
        A: crate::composite_execution::CompositeArchitecture<B, S>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>,
        P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
        F: crate::partitioned_execution::PartitionTensorAllocator<B>,
        B::ParallelContext: Sized,
    {
        let strategy = match self.strategy {
            PreparedCompositeUnitStrategy::Direct => {
                crate::partitioned_execution::SelectedCompositePartitionUnitStrategy::Direct
            }
            PreparedCompositeUnitStrategy::Routed {
                provider,
                plan,
                expert_group,
            } => crate::partitioned_execution::SelectedCompositePartitionUnitStrategy::routed_from_prepared_grouped_plan(
                provider, plan, expert_group, movement,
            ),
            PreparedCompositeUnitStrategy::RoutedCollective {
                provider,
                plan,
                expert_group,
                tensor_group,
                waves,
            } => crate::partitioned_execution::SelectedCompositePartitionUnitStrategy::routed_with_prepared_collective_waves(
                provider,
                plan,
                expert_group,
                movement,
                tensor_group,
                waves,
            ),
        };
        crate::partitioned_execution::CompositePartitionExecutor::from_prepared_structure(
            architecture,
            policy,
            parallel,
            allocator,
            strategy,
            self.structure,
        )
    }
}

fn validate_executor_routes(
    expected: &[crate::partitioned_execution::SelectedPartitionBoundaryRoute],
    actual: &[(
        eredu_runtime::CommunicationRouteId,
        eredu_runtime::ResolvedBoundaryWireSchema,
    )],
) -> Result<(), String> {
    let expected = expected
        .iter()
        .map(|route| (route.route().route, route.schema().clone()));
    let actual = actual.iter().cloned();
    validate_exact_executor_contract(expected, actual)
}

fn validate_exact_executor_contract<K, V>(
    expected: impl IntoIterator<Item = (K, V)>,
    actual: impl IntoIterator<Item = (K, V)>,
) -> Result<(), String>
where
    K: Ord,
    V: Eq,
{
    let expected = expected.into_iter().collect::<Vec<_>>();
    let actual = actual.into_iter().collect::<Vec<_>>();
    let exact_cardinality = expected.len() == actual.len();
    let expected = expected
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    let actual = actual
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    if exact_cardinality && expected.len() == actual.len() && expected == actual {
        Ok(())
    } else {
        Err("composite executor boundary routes differ from selected schemas".into())
    }
}

impl PreparedCompositeRoutedExecution {
    /// Canonical composite group containing the routed decoder units.
    pub const fn owner_group(&self) -> &eredu_runtime::ExecutionGroupId {
        &self.owner_group
    }

    /// Exact localized grouped-bank realization.
    pub const fn plan(&self) -> &crate::routed_text::RoutedGroupedPlan {
        &self.plan
    }

    /// Exact selected route cardinality for every provider unit.
    pub const fn routes_by_unit(&self) -> &std::collections::BTreeMap<usize, usize> {
        &self.routes_by_unit
    }

    /// Maps every provider invocation to the execution unit which owns it.
    pub const fn owner_units(&self) -> &std::collections::BTreeMap<usize, usize> {
        &self.owner_units
    }

    /// Complete unit count in the routed execution group.
    pub const fn owner_unit_count(&self) -> usize {
        self.owner_unit_count
    }

    /// Architecture-owned TP reduction order for every decoder unit.
    pub const fn tensor_reductions(&self) -> &std::collections::BTreeMap<usize, (usize, usize)> {
        &self.tensor_reductions
    }

    /// Hidden width carried by zero-row routed waves.
    pub const fn hidden_width(&self) -> usize {
        self.hidden_width
    }

    /// Physical vocabulary width gathered before any logical output trim.
    pub const fn tensor_output_width(&self) -> Option<usize> {
        self.tensor_output_width
    }

    /// Opaque expert exchange group selected during admission, when EP is active.
    pub const fn expert_group(&self) -> Option<eredu_runtime::CommunicationGroupId> {
        self.expert_group
    }
}

/// A typed composite partition together with its exact local payload projection.
pub struct PreparedCompositePartition<A, G, W> {
    prepared: PreparedPartitionedAdmission<
        A,
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
        G,
        W,
    >,
    layout: LocalModelLayout,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    publication: crate::partitioned_execution::PublicationValueDescriptor,
    routed: Option<PreparedCompositeRoutedExecution>,
}

impl<A, G, W> PreparedCompositePartition<A, G, W> {
    /// Typed architecture and validated selected partition.
    pub const fn prepared(
        &self,
    ) -> &PreparedPartitionedAdmission<
        A,
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
        G,
        W,
    > {
        &self.prepared
    }

    /// Exact tensor-parallel layout used by family-local construction.
    pub const fn layout(&self) -> &LocalModelLayout {
        &self.layout
    }

    /// Payload work projected from the preserved selected realization.
    pub fn materialization_tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        &self.tasks
    }

    /// Architecture-derived capability estimate retained across mechanism binding.
    pub const fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    /// Normalized effective model identity retained without backend inspection.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Complete architecture-declared vocabulary width for output publication.
    pub const fn output_width(&self) -> i32 {
        self.publication.output_width()
    }

    /// Exact architecture-selected model-output publication contract.
    pub const fn publication(&self) -> crate::partitioned_execution::PublicationValueDescriptor {
        self.publication
    }

    /// Exact routed provider authority, absent for dense composite execution.
    pub const fn routed_execution(&self) -> Option<&PreparedCompositeRoutedExecution> {
        self.routed.as_ref()
    }

    /// Exact multi-group runtime plan projected from this immutable selection.
    pub fn execution_plan(&self) -> Result<eredu_runtime::PartitionedExecutionPlan, String> {
        self.prepared.selected().composite_execution_plan()
    }

    /// Exact topology retained by the selected architecture admission.
    pub const fn topology(&self) -> eredu_core::ParallelRankTopology {
        self.prepared.selected().topology()
    }

    /// Opaque communication manifest selected before payload construction.
    pub const fn communication(&self) -> &eredu_runtime::CommunicationManifest {
        self.prepared.selected().communication()
    }

    /// Per-route semantic endpoints and exact selected schemas.
    pub fn boundary_routes(
        &self,
    ) -> &[crate::partitioned_execution::SelectedPartitionBoundaryRoute] {
        self.prepared.selected().boundary_routes()
    }

    /// Consumes this handoff without recomputing its layout or tasks.
    pub fn into_parts(
        self,
    ) -> (
        PreparedPartitionedAdmission<
            A,
            SelectedCompositeTextRealization,
            CompositeTextRequirements,
            G,
            W,
        >,
        LocalModelLayout,
        Vec<ReplicatedTextMaterializationTask>,
    ) {
        (self.prepared, self.layout, self.tasks)
    }

    /// Consumes this exact composite admission into the ordinary partitioned-session handoff.
    ///
    /// The prepared-input adapter, typed family partition, selected communication manifest, and
    /// payload tasks remain one authority. The backend factory therefore binds mechanisms only;
    /// it cannot substitute a different family graph or reselect composite placement.
    pub fn prepare_session_runtime<B, S, R, E, F>(
        self,
        topology: eredu_core::cache::PromptCacheTopology,
        context: &<B::Tensor as Tensor>::Context,
        factory: F,
    ) -> Result<
        eredu_runtime::PreparedPartitionedSessionRuntime<R, S>,
        eredu_runtime::PartitionedSessionPreparationError<E>,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: eredu_runtime::RuntimeState<B>,
        A: crate::composite_execution::CompositeArchitecture<B, S>
            + PartitionedLayeredArchitecture<B, S, Boundary = W>
            + 'static,
        A::InputPartPlan: 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
        F: FnOnce(
            eredu_runtime::PartitionedSessionFactoryInput<
                crate::composite_execution::PreparedCompositeArchitecture<A>,
                G,
                W,
            >,
            LocalModelLayout,
            PreparedCompositeExecutorPlan,
            &eredu_runtime::SelectedReplicatedTextRealization,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(R, S), E>,
    {
        let executor = PreparedCompositeExecutorPlan::new::<A, B, S, G, W>(
            &self.prepared,
            self.publication,
            self.routed.clone(),
        )
        .map_err(eredu_runtime::PartitionedSessionPreparationError::Contract)?;
        let Self {
            prepared,
            layout,
            tasks,
            capability_estimate: _,
            effective_model_type: _,
            publication: _,
            routed: _,
        } = self;
        let (architecture, selected) = prepared.into_parts();
        let topology_rank = selected.topology();
        let (base, partition, communication) = selected.into_parts();
        let architecture =
            crate::composite_execution::PreparedCompositeArchitecture::new(architecture);
        let parameters = architecture
            .parameter_description(context)
            .map_err(|error| {
                eredu_runtime::PartitionedSessionPreparationError::Contract(error.to_string())
            })?;
        let derived_layout = derive_partitioned_local_layout(&parameters, topology_rank)
            .map_err(eredu_runtime::PartitionedSessionPreparationError::Contract)?;
        if derived_layout != layout {
            return Err(eredu_runtime::PartitionedSessionPreparationError::Contract(
                "precomputed composite local layout differs from consumed partition authority"
                    .into(),
            ));
        }
        eredu_runtime::prepare_partitioned_session_runtime::<_, B, _, S, _, _, E, _>(
            architecture,
            base.execution().clone(),
            partition,
            communication,
            Some(&tasks),
            topology,
            eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
            context,
            move |input, selected, context| factory(input, layout, executor, selected, context),
        )
    }
}

/// Backend-generic consumer for any supported dense composite family.
///
/// The single generic method prevents a backend from acquiring family-specific
/// visitor entry points.
pub trait AuthoritativeCompositePartitionVisitor<B, S>: Sized
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism-binding failure.
    type Error;

    /// Receives one statically known architecture and exact selected partition.
    fn visit<A, G, W>(
        self,
        prepared: PreparedCompositePartition<A, G, W>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary;
}

/// Backend-generic consumer for an exact composite partition and its paired prediction extension.
pub trait AuthoritativeCompositePartitionPredictionTargetVisitor<B, S, M>: Sized
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism-binding failure.
    type Error;

    /// Receives the exact target only after architecture-owned extension pairing.
    fn visit<A, G, W>(
        self,
        prepared: PreparedCompositePartition<A, G, W>,
        extension: <crate::composite_execution::PreparedCompositeArchitecture<A> as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<M>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::Error: std::fmt::Display,
        crate::composite_execution::PreparedCompositeArchitecture<A>:
            crate::prediction_extension::MaterializedPredictionTarget<B>,
        W: eredu_runtime::ArchitectureBoundary;
}

struct EnrichedVisitor<V> {
    visitor: V,
    layout: LocalModelLayout,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    publication: crate::partitioned_execution::PublicationValueDescriptor,
    routed: Option<PreparedCompositeRoutedExecution>,
}

impl<B, S, G, W, V> CompositePartitionedArchitectureVisitor<B, S, G, W> for EnrichedVisitor<V>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    V: AuthoritativeCompositePartitionVisitor<B, S>,
    W: eredu_runtime::ArchitectureBoundary,
{
    type Output = V::Output;
    type Error = V::Error;

    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            SelectedCompositeTextRealization,
            CompositeTextRequirements,
            G,
            W,
        >,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::Error: std::fmt::Display,
    {
        self.visitor.visit(PreparedCompositePartition {
            prepared,
            layout: self.layout,
            tasks: self.tasks,
            capability_estimate: self.capability_estimate,
            effective_model_type: self.effective_model_type,
            publication: self.publication,
            routed: self.routed,
        })
    }
}

/// Failure while preparing an already selected composite partition.
#[derive(Debug, thiserror::Error)]
pub enum CompositePartitionPreparationError<E> {
    /// Architecture configuration, geometry, or selected ownership disagreed.
    #[error("invalid selected composite partition: {0}")]
    Architecture(String),
    /// The generic mechanism visitor rejected the typed handoff.
    #[error("composite partition visitor failed: {0}")]
    Visitor(E),
}

fn map_dispatch_error<E>(
    error: PartitionedDispatchError<E>,
) -> CompositePartitionPreparationError<E> {
    match error {
        PartitionedDispatchError::Architecture(error) => {
            CompositePartitionPreparationError::Architecture(error)
        }
        PartitionedDispatchError::Visitor(error) => {
            CompositePartitionPreparationError::Visitor(error)
        }
    }
}

fn prepared_gated_composite_execution<B, S, A>(
    selected: &SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    architecture: &A,
    plan: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    owner_units: std::collections::BTreeMap<usize, usize>,
    owner_unit_count: usize,
    hidden_width: usize,
) -> Result<PreparedCompositeRoutedExecution, String>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    let SelectedCompositeTextRealization::Routed { execution, .. } = selected.base() else {
        return Err("dense composite selection cannot bind a routed execution plan".into());
    };
    if execution.owner_group().as_str()
        != plan
            .unit_specs()
            .keys()
            .next()
            .map(|(group, _)| group.as_str())
            .ok_or("localized composite expert plan has no execution units")?
    {
        return Err("localized composite expert plan names a different owner group".into());
    }
    let routes_by_unit = execution.routes_by_unit().clone();
    let expected = plan
        .unit_specs()
        .keys()
        .map(|(_, unit)| *unit)
        .collect::<std::collections::BTreeSet<_>>();
    if routes_by_unit
        .keys()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        != expected
    {
        return Err("selected composite route cardinalities differ from localized units".into());
    }
    if owner_unit_count == 0
        || hidden_width == 0
        || owner_units
            .keys()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            != expected
        || owner_units.values().any(|unit| *unit >= owner_unit_count)
    {
        return Err("selected composite provider-unit ownership is invalid".into());
    }
    let routed_owner_units = owner_units
        .values()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let tensor_reductions = (0..owner_unit_count)
        .map(|unit| {
            architecture
                .routed_tensor_reductions(unit, routed_owner_units.contains(&unit))
                .map(|order| (unit, order))
                .map_err(|error| error.to_string())
        })
        .collect::<Result<std::collections::BTreeMap<_, _>, _>>()?;
    let tensor_output_width = architecture
        .routed_tensor_output_width()
        .map_err(|error| error.to_string())?;
    if tensor_output_width == Some(0) {
        return Err("selected composite routed output width is zero".into());
    }
    Ok(PreparedCompositeRoutedExecution {
        owner_group: execution.owner_group().clone(),
        plan: plan.into(),
        routes_by_unit,
        owner_units,
        owner_unit_count,
        tensor_reductions,
        hidden_width,
        tensor_output_width,
        expert_group: selected.requirements().expert_group(),
    })
}

fn inkling_gated_execution_plan(
    args: &crate::inkling::ModelArgs,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<crate::inkling::ExpertBankRealization>,
) -> Result<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>, String> {
    let layers = usize::try_from(args.text_config.num_hidden_layers)
        .map_err(|_| "Inkling text-layer count exceeds usize")?;
    let mut specs = std::collections::BTreeMap::new();
    for ((group, layer), bank) in realization.unit_specs() {
        specs.insert((group.clone(), *layer), bank.routed.clone());
        specs.insert(
            (
                group.clone(),
                layers
                    .checked_add(*layer)
                    .ok_or("Inkling shared provider unit overflowed usize")?,
            ),
            bank.shared.clone(),
        );
    }
    crate::ExpertRealizationPlan::balanced(realization.global_expert_count(), topology, specs)
        .map_err(|error| error.to_string())
}

/// Constructs and visits one selected prediction-free composite partition.
///
/// The normalized inspection retained by the selected admission is the only
/// source of family configuration. Routed families additionally retain their
/// exact localized provider plan in the typed handoff.
pub fn visit_authoritative_composite_partition<B, S, V>(
    selected: SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
) -> Result<V::Output, CompositePartitionPreparationError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + RuntimeStateComponents<B>
        + AuxiliaryConvolutionState<B::Tensor>,
    V: AuthoritativeCompositePartitionVisitor<B, S>,
{
    let routed = matches!(
        selected.base(),
        SelectedCompositeTextRealization::Routed { .. }
    );
    if routed
        != selected
            .requirements()
            .execution()
            .routed_execution()
            .is_some()
    {
        return Err(CompositePartitionPreparationError::Architecture(
            "selected composite decoder strategy differs from its requirements".into(),
        ));
    }

    let target_formats = selected_formats(selected.base().execution());
    let target_linear_formats = selected_linear_formats(
        selected.requirements().execution().execution(),
        selected.base().execution(),
    );
    let config = composite_config(
        selected
            .requirements()
            .execution()
            .inspection()
            .architecture_plan(),
    )
    .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?
    .ok_or_else(|| {
        CompositePartitionPreparationError::Architecture(
            "selected artifact is not a supported composite architecture".into(),
        )
    })?;

    macro_rules! prepare {
        ($architecture:expr, $parameters:expr, $geometry:expr, $partition_geometry:expr,
         $capability:expr, $effective:expr, $output_width:expr, $foundation:expr,
         $routed:expr) => {{
            let architecture = $architecture;
            let boundary =
                <_ as PartitionedLayeredArchitecture<B, S>>::boundary_schema(&architecture)
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
            let partition = ArchitecturePartition::from_architecture::<B, S, _, _>(
                &architecture,
                selected
                    .requirements()
                    .groups()
                    .iter()
                    .map(|group| (group.group().as_str(), group.units())),
                selected.requirements().ownership().clone(),
                $partition_geometry,
                boundary,
                &$parameters,
            )
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            if selected.requirements().state() != partition.state() {
                return Err(CompositePartitionPreparationError::Architecture(
                    "constructed composite partition state differs from cold admission".into(),
                ));
            }
            $foundation(&partition).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
                selected.materialization_tasks(),
                &$parameters,
                &partition,
            )
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let enriched = EnrichedVisitor {
                visitor,
                layout: $geometry,
                tasks,
                capability_estimate: $capability,
                effective_model_type: $effective,
                publication: crate::partitioned_execution::PublicationValueDescriptor::new(
                    $output_width,
                )
                .map_err(|error| {
                    CompositePartitionPreparationError::Architecture(error.to_string())
                })?,
                routed: $routed,
            };
            visit_composite_partitioned_architecture::<B, S, _, _, _, _>(
                architecture,
                selected,
                partition,
                enriched,
            )
            .map_err(map_dispatch_error)
        }};
    }

    match config {
        CompositeConfig::Gemma4(source) => {
            let mut exact = source.clone();
            if let Some(audio) = exact.audio.as_mut() {
                audio.output_projection_bias = selected
                    .requirements()
                    .execution()
                    .inspection()
                    .tensors()
                    .get("model.audio_tower.output_proj.bias")
                    .is_some();
            }
            let args = crate::gemma4::with_checkpoint_formats(&exact, target_formats)
                .map_err(CompositePartitionPreparationError::Architecture)?;
            let sparse = args.text.layer_schedule.iter().any(|policy| {
                policy.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe
            });
            if routed != sparse {
                return Err(CompositePartitionPreparationError::Architecture(
                    "Gemma 4 selected decoder strategy differs from its normalized configuration"
                        .into(),
                ));
            }
            let description = crate::gemma4::LayeredModel::<B>::new(args.clone(), context)
                .map_err(|error| {
                    CompositePartitionPreparationError::Architecture(error.to_string())
                })?;
            let parameters = ArchitectureParameters::parameter_description(&description, context)
                .map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let layout =
                derive_partitioned_local_layout(&parameters, selected.requirements().topology())
                    .map_err(CompositePartitionPreparationError::Architecture)?;
            let local = crate::gemma4::local_geometry(&args, &layout).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let group_ranges = selected
                .requirements()
                .groups()
                .iter()
                .map(|group| (group.group().as_str().to_owned(), group.units()))
                .collect::<Vec<_>>();
            let ownership = selected.requirements().ownership().clone();
            let rank = selected.requirements().topology();
            let geometry = if routed {
                crate::gemma4::parallel::routed_partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                )
            } else {
                crate::gemma4::partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                )
            }
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let state_offset = geometry.text_units().start;
            let architecture =
                crate::gemma4::LayeredModel::<B>::new_parallel(args.clone(), local, context)
                    .and_then(|architecture| architecture.with_partition_state_offset(state_offset))
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
            let realization = if routed {
                Some(
                    crate::gemma4::expert_realization_plan(&architecture, rank)
                        .map_err(|error| {
                            CompositePartitionPreparationError::Architecture(error.to_string())
                        })?
                        .ok_or_else(|| {
                            CompositePartitionPreparationError::Architecture(
                                "routed Gemma 4 has no expert realization".into(),
                            )
                        })?,
                )
            } else {
                None
            };
            let routed_execution = realization
                .clone()
                .map(|plan| {
                    let owner_units = plan
                        .unit_specs()
                        .keys()
                        .map(|(_, unit)| (*unit, *unit))
                        .collect();
                    prepared_gated_composite_execution::<B, S, _>(
                        &selected,
                        &architecture,
                        plan,
                        owner_units,
                        args.text.num_hidden_layers(),
                        usize::try_from(args.text.hidden_size)
                            .map_err(|_| "Gemma 4 hidden width exceeds usize".to_owned())?,
                    )
                })
                .transpose()
                .map_err(CompositePartitionPreparationError::Architecture)?;
            let capability = crate::capability::gemma4(&args).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let effective = args.effective_model_type().to_owned();
            prepare!(
                architecture,
                parameters,
                layout,
                geometry,
                capability,
                effective,
                args.text.vocab_size,
                |partition| {
                    match &realization {
                        Some(realization) => routed_gemma4_partition_foundation(
                            &args,
                            &layout,
                            group_ranges
                                .iter()
                                .map(|(group, units)| (group.as_str(), units.clone())),
                            &ownership,
                            rank,
                            realization,
                        )
                        .map(|_| ()),
                        None => crate::gemma4::PartitionLocalFoundation::from_partition(
                            &args, partition,
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string()),
                    }
                },
                routed_execution
            )
        }
        CompositeConfig::Muse(source) => {
            let args = crate::muse_glimmer::with_checkpoint_formats(source, target_formats)
                .map_err(CompositePartitionPreparationError::Architecture)?;
            if routed != args.is_moe() {
                return Err(CompositePartitionPreparationError::Architecture(
                    "Muse-Glimmer selected decoder strategy differs from its normalized configuration"
                        .into(),
                ));
            }
            let description = crate::muse_glimmer::LayeredModel::<B>::new(args.clone(), context)
                .map_err(|error| {
                    CompositePartitionPreparationError::Architecture(error.to_string())
                })?;
            let parameters = ArchitectureParameters::parameter_description(&description, context)
                .map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let layout =
                derive_partitioned_local_layout(&parameters, selected.requirements().topology())
                    .map_err(CompositePartitionPreparationError::Architecture)?;
            let local = crate::muse_glimmer::local_geometry(&args, &layout).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let group_ranges = selected
                .requirements()
                .groups()
                .iter()
                .map(|group| (group.group().as_str().to_owned(), group.units()))
                .collect::<Vec<_>>();
            let ownership = selected.requirements().ownership().clone();
            let rank = selected.requirements().topology();
            let geometry = if routed {
                crate::muse_glimmer::parallel::routed_partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                )
            } else {
                crate::muse_glimmer::partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                )
            }
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let mut architecture =
                crate::muse_glimmer::LayeredModel::<B>::new_parallel(args.clone(), local, context)
                    .and_then(|architecture| {
                        architecture.with_partition_state_offset(geometry.text_units().start)
                    })
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
            let realization = if routed {
                let realization = crate::muse_glimmer::expert_realization_plan(&architecture, rank)
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?
                    .ok_or_else(|| {
                        CompositePartitionPreparationError::Architecture(
                            "routed Muse-Glimmer has no expert realization".into(),
                        )
                    })?;
                architecture = architecture
                    .with_expert_realization(realization.clone())
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
                Some(realization)
            } else {
                None
            };
            let routed_execution = realization
                .clone()
                .map(|plan| {
                    let owner_units = plan
                        .unit_specs()
                        .keys()
                        .map(|(_, unit)| (*unit, *unit))
                        .collect();
                    prepared_gated_composite_execution::<B, S, _>(
                        &selected,
                        &architecture,
                        plan,
                        owner_units,
                        usize::try_from(args.num_hidden_layers)
                            .map_err(|_| "Muse-Glimmer layer count exceeds usize".to_owned())?,
                        usize::try_from(args.hidden_size)
                            .map_err(|_| "Muse-Glimmer hidden width exceeds usize".to_owned())?,
                    )
                })
                .transpose()
                .map_err(CompositePartitionPreparationError::Architecture)?;
            let capability = crate::capability::muse_glimmer(&args).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let effective = args.model_type.clone();
            prepare!(
                architecture,
                parameters,
                layout,
                geometry,
                capability,
                effective,
                args.vocab_size,
                |partition| {
                    match &realization {
                        Some(realization) => routed_muse_partition_foundation(
                            &args,
                            &layout,
                            group_ranges
                                .iter()
                                .map(|(group, units)| (group.as_str(), units.clone())),
                            &ownership,
                            rank,
                            realization,
                        )
                        .map(|_| ()),
                        None => crate::muse_glimmer::PartitionLocalFoundation::from_partition(
                            &args, partition,
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string()),
                    }
                },
                routed_execution
            )
        }
        CompositeConfig::QwenVl(source) => {
            let args = qwen_vl_with_formats(source, target_linear_formats)
                .map_err(CompositePartitionPreparationError::Architecture)?;
            if routed != args.text.is_moe() {
                return Err(CompositePartitionPreparationError::Architecture(
                    "Qwen VL selected decoder strategy differs from its normalized configuration"
                        .into(),
                ));
            }
            let description = crate::qwen::vl::LayeredModel::<B>::new(args.clone(), context)
                .map_err(|error| {
                    CompositePartitionPreparationError::Architecture(error.to_string())
                })?;
            let parameters = ArchitectureParameters::parameter_description(&description, context)
                .map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let layout =
                derive_partitioned_local_layout(&parameters, selected.requirements().topology())
                    .map_err(CompositePartitionPreparationError::Architecture)?;
            let local = crate::qwen::vl::local_geometry(&args, &layout).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let group_ranges = selected
                .requirements()
                .groups()
                .iter()
                .map(|group| (group.group().as_str().to_owned(), group.units()))
                .collect::<Vec<_>>();
            let ownership = selected.requirements().ownership().clone();
            let rank = selected.requirements().topology();
            let architecture =
                crate::qwen::vl::LayeredModel::<B>::new_parallel(args.clone(), local, context)
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
            let realization = if routed {
                Some(
                    crate::qwen::vl::expert_realization_plan(&architecture, rank)
                        .map_err(|error| {
                            CompositePartitionPreparationError::Architecture(error.to_string())
                        })?
                        .ok_or_else(|| {
                            CompositePartitionPreparationError::Architecture(
                                "routed Qwen VL has no expert realization".into(),
                            )
                        })?,
                )
            } else {
                None
            };
            let geometry = match &realization {
                Some(realization) => crate::qwen::vl::partition_local_routed_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                    rank,
                    realization,
                ),
                None => crate::qwen::vl::partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                ),
            }
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let architecture = architecture.with_partition_geometry(geometry.clone());
            let routed_execution = realization
                .clone()
                .map(|plan| {
                    let owner_units = plan
                        .unit_specs()
                        .keys()
                        .map(|(_, unit)| (*unit, *unit))
                        .collect();
                    prepared_gated_composite_execution::<B, S, _>(
                        &selected,
                        &architecture,
                        plan,
                        owner_units,
                        usize::try_from(args.text.num_hidden_layers)
                            .map_err(|_| "Qwen VL layer count exceeds usize".to_owned())?,
                        usize::try_from(args.text.hidden_size)
                            .map_err(|_| "Qwen VL hidden width exceeds usize".to_owned())?,
                    )
                })
                .transpose()
                .map_err(CompositePartitionPreparationError::Architecture)?;
            let capability = crate::capability::qwen_vl(&args).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let effective = args.effective_model_type().to_owned();
            prepare!(
                architecture,
                parameters,
                layout,
                geometry,
                capability,
                effective,
                args.text.vocab_size,
                |partition| {
                    crate::qwen::vl::PartitionLocalFoundation::from_partition(&args, partition)
                },
                routed_execution
            )
        }
        CompositeConfig::QwenHybrid(source) => {
            let args = qwen_hybrid_composite_with_formats(source, target_linear_formats)
                .map_err(CompositePartitionPreparationError::Architecture)?;
            if args.text.mtp_num_hidden_layers > 0 {
                return Err(CompositePartitionPreparationError::Architecture(
                    "partitioned conditional Qwen does not admit embedded prediction".into(),
                ));
            }
            if routed != args.text.is_moe() {
                return Err(CompositePartitionPreparationError::Architecture(
                    "conditional Qwen selected decoder strategy differs from its normalized configuration"
                        .into(),
                ));
            }
            let description =
                crate::qwen::hybrid::ConditionalLayeredModel::<B>::new(args.clone(), context)
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
            let parameters = ArchitectureParameters::parameter_description(&description, context)
                .map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let layout =
                derive_partitioned_local_layout(&parameters, selected.requirements().topology())
                    .map_err(CompositePartitionPreparationError::Architecture)?;
            let local = crate::qwen::hybrid::conditional_local_geometry(&args, &layout).map_err(
                |error| CompositePartitionPreparationError::Architecture(error.to_string()),
            )?;
            let group_ranges = selected
                .requirements()
                .groups()
                .iter()
                .map(|group| (group.group().as_str().to_owned(), group.units()))
                .collect::<Vec<_>>();
            let ownership = selected.requirements().ownership().clone();
            let rank = selected.requirements().topology();
            let architecture = crate::qwen::hybrid::ConditionalLayeredModel::<B>::new_parallel(
                args.clone(),
                local,
                context,
            )
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let realization = if routed {
                Some(
                    crate::qwen::hybrid::conditional_expert_realization_plan(&architecture, rank)
                        .map_err(|error| {
                            CompositePartitionPreparationError::Architecture(error.to_string())
                        })?
                        .ok_or_else(|| {
                            CompositePartitionPreparationError::Architecture(
                                "routed conditional Qwen has no expert realization".into(),
                            )
                        })?,
                )
            } else {
                None
            };
            let geometry = match &realization {
                Some(_) => crate::qwen::hybrid::routed_conditional_partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                ),
                None => crate::qwen::hybrid::conditional_partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                ),
            }
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let architecture = geometry
                .local_state_layout()
                .map_err(|error| {
                    CompositePartitionPreparationError::Architecture(error.to_string())
                })
                .and_then(|layout| {
                    architecture
                        .with_partition_state_layout(geometry.target_units().start, layout)
                        .map_err(|error| {
                            CompositePartitionPreparationError::Architecture(error.to_string())
                        })
                })?;
            let routed_execution = realization
                .clone()
                .map(|plan| {
                    let owner_units = plan
                        .unit_specs()
                        .keys()
                        .map(|(_, unit)| (*unit, *unit))
                        .collect();
                    prepared_gated_composite_execution::<B, S, _>(
                        &selected,
                        &architecture,
                        plan,
                        owner_units,
                        usize::try_from(args.text.num_hidden_layers)
                            .map_err(|_| "conditional Qwen layer count exceeds usize".to_owned())?,
                        usize::try_from(args.text.hidden_size).map_err(|_| {
                            "conditional Qwen hidden width exceeds usize".to_owned()
                        })?,
                    )
                })
                .transpose()
                .map_err(CompositePartitionPreparationError::Architecture)?;
            let capability = crate::capability::qwen_hybrid(&args).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let effective = args.text.model_type.clone();
            prepare!(
                architecture,
                parameters,
                layout,
                geometry,
                capability,
                effective,
                args.text.vocab_size,
                |partition| {
                    match &realization {
                        Some(realization) => routed_conditional_qwen_partition_foundation(
                            &args,
                            &layout,
                            group_ranges
                                .iter()
                                .map(|(group, units)| (group.as_str(), units.clone())),
                            &ownership,
                            rank,
                            realization,
                        )
                        .map(|_| ()),
                        None => crate::qwen::hybrid::ConditionalPartitionLocalFoundation::from_partition(
                            &args, partition,
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string()),
                    }
                },
                routed_execution
            )
        }
        CompositeConfig::Inkling(source) => {
            let args = crate::inkling::with_checkpoint_formats(source, target_formats)
                .map_err(CompositePartitionPreparationError::Architecture)?;
            if args
                .mtp_config
                .as_ref()
                .is_some_and(|prediction| prediction.num_nextn_predict_layers > 0)
            {
                return Err(CompositePartitionPreparationError::Architecture(
                    "partitioned Inkling does not admit embedded prediction".into(),
                ));
            }
            if routed != args.text_config.has_sparse_moe_layers() {
                return Err(CompositePartitionPreparationError::Architecture(
                    "Inkling selected decoder strategy differs from its normalized configuration"
                        .into(),
                ));
            }
            let description = crate::inkling::LayeredModel::<B>::new(args.clone(), context)
                .map_err(|error| {
                    CompositePartitionPreparationError::Architecture(error.to_string())
                })?;
            let parameters = ArchitectureParameters::parameter_description(&description, context)
                .map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let layout =
                derive_partitioned_local_layout(&parameters, selected.requirements().topology())
                    .map_err(CompositePartitionPreparationError::Architecture)?;
            let local = crate::inkling::local_geometry(&args, &layout).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let group_ranges = selected
                .requirements()
                .groups()
                .iter()
                .map(|group| (group.group().as_str().to_owned(), group.units()))
                .collect::<Vec<_>>();
            let ownership = selected.requirements().ownership().clone();
            let rank = selected.requirements().topology();
            let geometry = if routed {
                crate::inkling::parallel::routed_partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                )
            } else {
                crate::inkling::partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                )
            }
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let mut architecture = crate::inkling::LayeredModel::<B>::new_parallel(
                args.clone(),
                std::sync::Arc::new(local),
                context,
            )
            .and_then(|architecture| {
                architecture.with_partition_state_offset(geometry.text_units().start)
            })
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let realization = if routed {
                let realization = crate::inkling::expert_realization_plan(&architecture, rank)
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?
                    .ok_or_else(|| {
                        CompositePartitionPreparationError::Architecture(
                            "routed Inkling has no expert realization".into(),
                        )
                    })?;
                architecture = architecture
                    .with_expert_realization(realization.clone())
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
                Some(realization)
            } else {
                None
            };
            let routed_execution = realization
                .as_ref()
                .map(|realization| {
                    let layers = usize::try_from(args.text_config.num_hidden_layers)
                        .map_err(|_| "Inkling layer count exceeds usize".to_owned())?;
                    inkling_gated_execution_plan(&args, rank, realization).and_then(|plan| {
                        let owner_units = plan
                            .unit_specs()
                            .keys()
                            .map(|(_, unit)| (*unit, *unit % layers))
                            .collect();
                        prepared_gated_composite_execution::<B, S, _>(
                            &selected,
                            &architecture,
                            plan,
                            owner_units,
                            layers,
                            usize::try_from(args.text_config.hidden_size)
                                .map_err(|_| "Inkling hidden width exceeds usize".to_owned())?,
                        )
                    })
                })
                .transpose()
                .map_err(CompositePartitionPreparationError::Architecture)?;
            let capability = crate::capability::inkling(&args).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let effective = args.model_type.clone();
            prepare!(
                architecture,
                parameters,
                layout,
                geometry,
                capability,
                effective,
                args.text_config
                    .unpadded_vocab_size
                    .unwrap_or(args.text_config.vocab_size),
                |partition| {
                    match &realization {
                        Some(realization) => routed_inkling_partition_foundation(
                            &args,
                            &layout,
                            group_ranges
                                .iter()
                                .map(|(group, units)| (group.as_str(), units.clone())),
                            &ownership,
                            rank,
                            realization,
                        )
                        .map(|_| ()),
                        None => crate::inkling::PartitionLocalFoundation::from_partition(
                            &args, partition,
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string()),
                    }
                },
                routed_execution
            )
        }
    }
}

/// Constructs an admitted Inkling or conditional-Qwen partition and pairs its prediction
/// extension before the concrete target type can be erased by a backend adapter.
pub fn visit_authoritative_composite_prediction_target_partition<B, S, M, V>(
    selected: SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
) -> Result<V::Output, CompositePartitionPreparationError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + RuntimeStateComponents<B>
        + AuxiliaryConvolutionState<B::Tensor>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: AuthoritativeCompositePartitionPredictionTargetVisitor<B, S, M>,
{
    let routed = matches!(
        selected.base(),
        SelectedCompositeTextRealization::Routed { .. }
    );
    if routed
        != selected
            .requirements()
            .execution()
            .routed_execution()
            .is_some()
    {
        return Err(CompositePartitionPreparationError::Architecture(
            "selected composite decoder strategy differs from its requirements".into(),
        ));
    }
    let target_formats = selected_formats(selected.base().execution());
    let target_linear_formats = selected_linear_formats(
        selected.requirements().execution().execution(),
        selected.base().execution(),
    );
    let config = composite_config(
        selected
            .requirements()
            .execution()
            .inspection()
            .architecture_plan(),
    )
    .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?
    .ok_or_else(|| {
        CompositePartitionPreparationError::Architecture(
            "selected artifact is not a supported composite architecture".into(),
        )
    })?;

    macro_rules! finish_prediction_target {
        ($target:ty, $architecture:expr, $parameters:expr, $layout:expr, $geometry:expr,
         $capability:expr, $effective:expr, $output_width:expr, $foundation:expr,
         $routed:expr) => {{
            let architecture = $architecture;
            let boundary =
                <_ as PartitionedLayeredArchitecture<B, S>>::boundary_schema(&architecture)
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
            let partition = ArchitecturePartition::from_architecture::<B, S, _, _>(
                &architecture,
                selected
                    .requirements()
                    .groups()
                    .iter()
                    .map(|group| (group.group().as_str(), group.units())),
                selected.requirements().ownership().clone(),
                $geometry.clone(),
                boundary,
                &$parameters,
            )
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            if selected.requirements().state() != partition.state() {
                return Err(CompositePartitionPreparationError::Architecture(
                    "constructed composite prediction partition state differs from cold admission"
                        .into(),
                ));
            }
            $foundation(&partition).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
                selected.materialization_tasks(),
                &$parameters,
                &partition,
            )
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let publication =
                crate::partitioned_execution::PublicationValueDescriptor::new($output_width)
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
            let extension = <crate::composite_execution::PreparedCompositeArchitecture<$target> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension)
                .map_err(|error| {
                    CompositePartitionPreparationError::Architecture(error.to_string())
                })?;
            let prepared =
                prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
                    .map_err(CompositePartitionPreparationError::Architecture)?;
            visitor
                .visit(
                    PreparedCompositePartition {
                        prepared,
                        layout: $layout,
                        tasks,
                        capability_estimate: $capability,
                        effective_model_type: $effective,
                        publication,
                        routed: $routed,
                    },
                    extension,
                )
                .map_err(CompositePartitionPreparationError::Visitor)
        }};
    }

    match config {
        CompositeConfig::QwenHybrid(source) => {
            let args = qwen_hybrid_composite_with_formats(source, target_linear_formats)
                .map_err(CompositePartitionPreparationError::Architecture)?;
            if args.text.mtp_num_hidden_layers > 0 {
                return Err(CompositePartitionPreparationError::Architecture(
                    "partitioned conditional Qwen target still contains prediction units".into(),
                ));
            }
            if routed != args.text.is_moe() {
                return Err(CompositePartitionPreparationError::Architecture(
                    "conditional Qwen selected decoder strategy differs from its normalized configuration"
                        .into(),
                ));
            }
            let description =
                crate::qwen::hybrid::ConditionalLayeredModel::<B>::new(args.clone(), context)
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
            let parameters = ArchitectureParameters::parameter_description(&description, context)
                .map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let layout =
                derive_partitioned_local_layout(&parameters, selected.requirements().topology())
                    .map_err(CompositePartitionPreparationError::Architecture)?;
            let local = crate::qwen::hybrid::conditional_local_geometry(&args, &layout).map_err(
                |error| CompositePartitionPreparationError::Architecture(error.to_string()),
            )?;
            let group_ranges = selected
                .requirements()
                .groups()
                .iter()
                .map(|group| (group.group().as_str().to_owned(), group.units()))
                .collect::<Vec<_>>();
            let ownership = selected.requirements().ownership().clone();
            let rank = selected.requirements().topology();
            let architecture = crate::qwen::hybrid::ConditionalLayeredModel::<B>::new_parallel(
                args.clone(),
                local,
                context,
            )
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let realization = if routed {
                Some(
                    crate::qwen::hybrid::conditional_expert_realization_plan(&architecture, rank)
                        .map_err(|error| {
                            CompositePartitionPreparationError::Architecture(error.to_string())
                        })?
                        .ok_or_else(|| {
                            CompositePartitionPreparationError::Architecture(
                                "routed conditional Qwen has no expert realization".into(),
                            )
                        })?,
                )
            } else {
                None
            };
            let geometry = match &realization {
                Some(_) => crate::qwen::hybrid::routed_conditional_partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                ),
                None => crate::qwen::hybrid::conditional_partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                ),
            }
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let architecture = geometry
                .local_state_layout()
                .map_err(|error| {
                    CompositePartitionPreparationError::Architecture(error.to_string())
                })
                .and_then(|state| {
                    architecture
                        .with_partition_state_layout(geometry.target_units().start, state)
                        .map_err(|error| {
                            CompositePartitionPreparationError::Architecture(error.to_string())
                        })
                })?;
            let routed_execution = realization
                .clone()
                .map(|plan| {
                    let owner_units = plan
                        .unit_specs()
                        .keys()
                        .map(|(_, unit)| (*unit, *unit))
                        .collect();
                    prepared_gated_composite_execution::<B, S, _>(
                        &selected,
                        &architecture,
                        plan,
                        owner_units,
                        usize::try_from(args.text.num_hidden_layers)
                            .map_err(|_| "conditional Qwen layer count exceeds usize".to_owned())?,
                        usize::try_from(args.text.hidden_size).map_err(|_| {
                            "conditional Qwen hidden width exceeds usize".to_owned()
                        })?,
                    )
                })
                .transpose()
                .map_err(CompositePartitionPreparationError::Architecture)?;
            let capability = crate::capability::qwen_hybrid(&args).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let effective = args.text.model_type.clone();
            finish_prediction_target!(
                crate::qwen::hybrid::ConditionalLayeredModel<B>,
                architecture,
                parameters,
                layout,
                geometry,
                capability,
                effective,
                args.text.vocab_size,
                |partition| {
                    match &realization {
                        Some(realization) => routed_conditional_qwen_partition_foundation(
                            &args,
                            &layout,
                            group_ranges
                                .iter()
                                .map(|(group, units)| (group.as_str(), units.clone())),
                            &ownership,
                            rank,
                            realization,
                        )
                        .map(|_| ()),
                        None => crate::qwen::hybrid::ConditionalPartitionLocalFoundation::from_partition(
                            &args, partition,
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string()),
                    }
                },
                routed_execution
            )
        }
        CompositeConfig::Inkling(source) => {
            let args = crate::inkling::with_checkpoint_formats(source, target_formats)
                .map_err(CompositePartitionPreparationError::Architecture)?;
            if args
                .mtp_config
                .as_ref()
                .is_some_and(|prediction| prediction.num_nextn_predict_layers > 0)
            {
                return Err(CompositePartitionPreparationError::Architecture(
                    "partitioned Inkling target still contains prediction units".into(),
                ));
            }
            if routed != args.text_config.has_sparse_moe_layers() {
                return Err(CompositePartitionPreparationError::Architecture(
                    "Inkling selected decoder strategy differs from its normalized configuration"
                        .into(),
                ));
            }
            let description = crate::inkling::LayeredModel::<B>::new(args.clone(), context)
                .map_err(|error| {
                    CompositePartitionPreparationError::Architecture(error.to_string())
                })?;
            let parameters = ArchitectureParameters::parameter_description(&description, context)
                .map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let layout =
                derive_partitioned_local_layout(&parameters, selected.requirements().topology())
                    .map_err(CompositePartitionPreparationError::Architecture)?;
            let local = crate::inkling::local_geometry(&args, &layout).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let group_ranges = selected
                .requirements()
                .groups()
                .iter()
                .map(|group| (group.group().as_str().to_owned(), group.units()))
                .collect::<Vec<_>>();
            let ownership = selected.requirements().ownership().clone();
            let rank = selected.requirements().topology();
            let geometry = if routed {
                crate::inkling::parallel::routed_partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                )
            } else {
                crate::inkling::partition_local_geometry(
                    &args,
                    &layout,
                    group_ranges
                        .iter()
                        .map(|(group, units)| (group.as_str(), units.clone())),
                    &ownership,
                )
            }
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let mut architecture = crate::inkling::LayeredModel::<B>::new_parallel(
                args.clone(),
                std::sync::Arc::new(local),
                context,
            )
            .and_then(|architecture| {
                architecture.with_partition_state_offset(geometry.text_units().start)
            })
            .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            let realization = if routed {
                let realization = crate::inkling::expert_realization_plan(&architecture, rank)
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?
                    .ok_or_else(|| {
                        CompositePartitionPreparationError::Architecture(
                            "routed Inkling has no expert realization".into(),
                        )
                    })?;
                architecture = architecture
                    .with_expert_realization(realization.clone())
                    .map_err(|error| {
                        CompositePartitionPreparationError::Architecture(error.to_string())
                    })?;
                Some(realization)
            } else {
                None
            };
            let routed_execution = realization
                .as_ref()
                .map(|realization| {
                    let layers = usize::try_from(args.text_config.num_hidden_layers)
                        .map_err(|_| "Inkling layer count exceeds usize".to_owned())?;
                    inkling_gated_execution_plan(&args, rank, realization).and_then(|plan| {
                        let owner_units = plan
                            .unit_specs()
                            .keys()
                            .map(|(_, unit)| (*unit, *unit % layers))
                            .collect();
                        prepared_gated_composite_execution::<B, S, _>(
                            &selected,
                            &architecture,
                            plan,
                            owner_units,
                            layers,
                            usize::try_from(args.text_config.hidden_size)
                                .map_err(|_| "Inkling hidden width exceeds usize".to_owned())?,
                        )
                    })
                })
                .transpose()
                .map_err(CompositePartitionPreparationError::Architecture)?;
            let capability = crate::capability::inkling(&args).map_err(|error| {
                CompositePartitionPreparationError::Architecture(error.to_string())
            })?;
            let effective = args.model_type.clone();
            finish_prediction_target!(
                crate::inkling::LayeredModel<B>,
                architecture,
                parameters,
                layout,
                geometry,
                capability,
                effective,
                args.text_config
                    .unpadded_vocab_size
                    .unwrap_or(args.text_config.vocab_size),
                |partition| {
                    match &realization {
                        Some(realization) => routed_inkling_partition_foundation(
                            &args,
                            &layout,
                            group_ranges
                                .iter()
                                .map(|(group, units)| (group.as_str(), units.clone())),
                            &ownership,
                            rank,
                            realization,
                        )
                        .map(|_| ()),
                        None => crate::inkling::PartitionLocalFoundation::from_partition(
                            &args, partition,
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string()),
                    }
                },
                routed_execution
            )
        }
        CompositeConfig::Gemma4(_) | CompositeConfig::Muse(_) | CompositeConfig::QwenVl(_) => {
            Err(CompositePartitionPreparationError::Architecture(
                "composite partition does not admit an embedded prediction extension".into(),
            ))
        }
    }
}

#[cfg(test)]
mod routed_foundation_tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn routed_composite_foundation_preserves_cartesian_global_and_compact_ownership() {
        let topology = eredu_core::ParallelTopology::new(2, 2, 2, 1).unwrap();
        for rank in 0..topology.world_size() {
            let rank = eredu_core::ParallelRankTopology::new(topology, rank).unwrap();
            let group = eredu_runtime::ExecutionGroupId::new("text").unwrap();
            let specs = [1usize, 3]
                .into_iter()
                .map(|unit| ((group.clone(), unit), format!("unit-{unit}")))
                .collect::<BTreeMap<_, _>>();
            let plan = crate::ExpertRealizationPlan::balanced(4, rank, specs).unwrap();
            let units = eredu_core::balanced_contiguous_range(
                4,
                rank.pipeline_parallel_size(),
                rank.pipeline_parallel_rank(),
                false,
            )
            .unwrap();
            let foundation =
                routed_composite_foundation((), "text", units.clone(), 4, [1, 3], rank, &plan)
                    .unwrap();
            let local_experts = plan.local_global_group_indices();
            assert_eq!(
                foundation.expert_banks().len(),
                [1usize, 3]
                    .into_iter()
                    .filter(|unit| units.contains(unit))
                    .count()
                    * local_experts.len()
            );
            for bank in foundation.expert_banks() {
                assert!(units.contains(&bank.unit()));
                assert_eq!(
                    local_experts[bank.owner_local_expert()],
                    bank.global_expert()
                );
                assert_eq!(bank.bank_key().unit(), bank.unit());
                assert_eq!(bank.bank_key().member(), bank.global_expert());
            }
        }
    }

    #[test]
    fn routed_composite_foundation_rejects_pp_or_sparse_schedule_drift() {
        let topology = eredu_core::ParallelTopology::new(1, 2, 2, 1).unwrap();
        let rank = eredu_core::ParallelRankTopology::new(topology, 0).unwrap();
        let group = eredu_runtime::ExecutionGroupId::new("text").unwrap();
        let plan = crate::ExpertRealizationPlan::balanced(
            4,
            rank,
            [((group, 1), "unit-1")].into_iter().collect(),
        )
        .unwrap();
        assert!(routed_composite_foundation((), "text", 2..4, 4, [1], rank, &plan).is_err());
        assert!(routed_composite_foundation((), "text", 0..2, 4, [0], rank, &plan).is_err());
    }

    #[test]
    fn composite_executor_contract_rejects_missing_or_perturbed_route_schema() {
        let expected = [(1u8, "vision"), (2, "decoder")];
        assert!(validate_exact_executor_contract(expected, expected).is_ok());
        assert!(validate_exact_executor_contract(expected, [(1, "vision")]).is_err());
        assert!(validate_exact_executor_contract(
            expected,
            [(1, "vision"), (2, "decoder"), (2, "decoder")],
        )
        .is_err());
        assert!(
            validate_exact_executor_contract(expected, [(1, "vision"), (2, "wrong-decoder")],)
                .is_err()
        );
    }
}
