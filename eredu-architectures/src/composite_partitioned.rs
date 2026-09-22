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
    derive_partitioned_local_layout, prepare_partitioned, PreparedPartitionedAdmission,
    SelectedPartitionedAdmission,
};
use crate::replicated_text::{
    composite_config, qwen_hybrid_composite_with_formats, qwen_vl_with_formats, selected_formats,
    selected_matrix_formats, CompositeConfig, CompositeTextRequirements,
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
    source_unit_end: usize,
    maximum_decoder_sequence: i32,
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
        CompositeConfig::Muse(args) => {
            if group == 0 {
                args.vision_config
                    .as_ref()
                    .map(|vision| {
                        let maximum_patches = maximum_decoder_sequence
                            .checked_mul(vision.merge_size)
                            .and_then(|value| value.checked_mul(vision.merge_size))
                            .ok_or_else(|| {
                                "Muse selected patch continuation extent exceeds i32".to_owned()
                            })?;
                        Ok::<_, String>((vision.hidden_size, true, Some(maximum_patches)))
                    })
                    .transpose()?
            } else {
                None
            }
        }
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
                    let shape = vision.folded_shape(maximum_decoder_sequence, source_unit_end)?;
                    let maximum_rows = shape[..4]
                        .iter()
                        .try_fold(1i32, |n, d| n.checked_mul(*d))
                        .ok_or_else(|| {
                        "Inkling continuation maximum rows overflowed".to_owned()
                    })?;
                    Some((shape[4], false, Some(maximum_rows)))
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
    if let CompositeConfig::Muse(args) = &config {
        return crate::muse_glimmer::model::vision_partition_boundary_schema(
            args,
            continuation,
            None,
        )
        .map(Some)
        .map_err(|error| error.to_string());
    }
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
        eredu_runtime::ParameterBankKey::new(0, self.unit, self.global_expert)
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
    expert_group: Option<eredu_core::CollectiveGroupId>,
    partition_unit_coordinates:
        std::collections::BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>,
}

/// Opaque architecture-selected construction for one composite partition executor.
///
/// This value retains group transport kinds, resolved boundary schemas, output
/// publication, collective placement, and any routed provider schedule. A
/// backend supplies only its unit policy, parallel context, tensor allocator,
/// and route-movement mechanism.
#[derive(Clone)]
pub struct PreparedCompositeExecutorPlan {
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    requires_independent_banks: bool,
    resident_provider_plan: Option<std::sync::Arc<PreparedCompositeRoutedExecution>>,
    structure: crate::partitioned_execution::PreparedCompositeExecutorStructure,
    strategy: PreparedCompositeUnitStrategy,
}

#[derive(Clone)]
enum PreparedCompositeUnitStrategy {
    Direct,
    Routed {
        plans: std::sync::Arc<
            std::collections::BTreeMap<
                eredu_runtime::RoutedBankId,
                crate::routed_text::RoutedGroupedPlan,
            >,
        >,
        expert_group: Option<eredu_core::CollectiveGroupId>,
    },
    RoutedCollective {
        plans: std::sync::Arc<
            std::collections::BTreeMap<
                eredu_runtime::RoutedBankId,
                crate::routed_text::RoutedGroupedPlan,
            >,
        >,
        expert_group: eredu_core::CollectiveGroupId,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        wave_group: eredu_core::CollectiveGroupId,
        waves: std::sync::Arc<crate::partitioned_execution::RoutedExpertCollectiveWaveSchedule>,
    },
}

impl PreparedCompositeExecutorPlan {
    fn bank_plans(
        &self,
    ) -> Option<
        &std::collections::BTreeMap<
            eredu_runtime::RoutedBankId,
            crate::routed_text::RoutedGroupedPlan,
        >,
    > {
        match &self.strategy {
            PreparedCompositeUnitStrategy::Direct => None,
            PreparedCompositeUnitStrategy::Routed { plans, .. }
            | PreparedCompositeUnitStrategy::RoutedCollective { plans, .. } => Some(plans),
        }
    }
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
        let requires_independent_banks = matches!(
            selected.base(),
            SelectedCompositeTextRealization::Routed { execution, .. }
                if matches!(execution.bank_residency(), eredu_runtime::ParameterBankResidency::IndependentCache(_))
        );
        let resident_provider_plan = routed.map(std::sync::Arc::new);
        let strategy = match resident_provider_plan.as_deref() {
            Some(routed) => {
                let plan = routed.plan.gated().cloned().ok_or_else(|| {
                    "composite executor requires a gated routed realization".to_owned()
                })?;
                if topology.pipeline_parallel_size() > 1 && topology.expert_parallel_size() > 1 {
                    let output_width = routed.tensor_output_width.unwrap_or(
                        usize::try_from(publication.output_width())
                            .map_err(|_| "composite output width exceeds usize")?,
                    );
                    let mut waves = crate::partitioned_execution::routed_expert_collective_wave_schedule_with_unit_owners_and_tensor_order(
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
                    waves.bind_route_cardinality(
                        eredu_runtime::RoutedBankId::new(0),
                        &routed.routes_by_unit,
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
                        plans: std::sync::Arc::new(std::collections::BTreeMap::from([(
                            eredu_runtime::RoutedBankId::new(0),
                            plan.into(),
                        )])),
                        expert_group,
                        tensor_group,
                        wave_group: selected
                            .composite_execution_plan()?
                            .commit_barrier()
                            .ok_or("routed composite wave has no selected agreement group")?,
                        waves: std::sync::Arc::new(waves),
                    }
                } else {
                    PreparedCompositeUnitStrategy::Routed {
                        plans: std::sync::Arc::new(std::collections::BTreeMap::from([(
                            eredu_runtime::RoutedBankId::new(0),
                            plan.into(),
                        )])),
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
            requires_independent_banks,
            resident_provider_plan,
            structure,
            strategy,
        })
    }

    pub(crate) fn bind_borrowed_direct<A, B, S, P, F>(
        &self,
        architecture: A,
        policy: P,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<
        crate::partitioned_execution::CompositePartitionExecutor<A, B, S, P, F>,
        eredu_nn::Error,
    >
    where
        B: eredu_runtime::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context,
            > + eredu_runtime::CommunicationBackend
            + eredu_nn::TensorParallelGroupedNeuralBackend,
        S: eredu_runtime::RuntimeState<B>,
        A: crate::composite_execution::CompositeArchitecture<B, S>
            + eredu_runtime::ParallelLayeredArchitecture<B, S>,
        P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
        F: crate::partitioned_execution::PartitionTensorAllocator<B>,
        B::ParallelContext: Sized,
    {
        if !matches!(self.strategy, PreparedCompositeUnitStrategy::Direct)
            || self.requires_independent_banks
        {
            return Err(context.metadata_error(format_args!(
                "composite quote requires its exact retained provider and movement source"
            )));
        }
        crate::partitioned_execution::CompositePartitionExecutor::from_prepared_structure(
            architecture,
            policy,
            parallel,
            allocator,
            crate::partitioned_execution::DirectCompositePartitionUnitStrategy,
            self.structure.clone(),
        )
    }

    /// Opaque tensor group needed by the backend communication realizer.
    pub const fn communication_tensor_group(&self) -> Option<eredu_core::CollectiveGroupId> {
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
        if self.requires_independent_banks {
            return Err(eredu_nn::Error::backend(
                "selected independent composite banks require constructed cache providers",
            ));
        }
        let provider = self
            .resident_provider_plan
            .as_ref()
            .map(|routed| {
                let plan = routed.plan.gated().expect("prepared gated composite plan");
                let routes = routed
                    .routes_by_unit
                    .iter()
                    .map(|(unit, routes)| {
                        (
                            *unit,
                            if routed.expert_group.is_some() && !plan.unit_is_replicated(*unit) {
                                1
                            } else {
                                *routes
                            },
                        )
                    })
                    .collect();
                crate::PlannedResidentGatedProduct::new_partitioned_with_routes(
                    routed.owner_group.clone(),
                    plan.clone(),
                    routes,
                )
                .map(|provider| {
                    provider
                        .with_partition_unit_coordinates(routed.partition_unit_coordinates.clone())
                })
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
            })
            .transpose()?;
        self.bind_with_provider(
            architecture,
            policy,
            parallel,
            allocator,
            movement,
            provider,
        )
    }

    /// Binds an architecture-constructed provider under the retained composite schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_with_provider<A, B, S, P, F, Movement, Provider>(
        self,
        architecture: A,
        policy: P,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        movement: Movement,
        provider: Option<Provider>,
    ) -> Result<
        crate::partitioned_execution::CompositePartitionExecutor<
            A,
            B,
            S,
            P,
            F,
            crate::partitioned_execution::SelectedCompositePartitionUnitStrategy<
                Provider,
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
        self.bind_borrowed_with_provider(
            architecture,
            policy,
            parallel,
            allocator,
            movement,
            provider,
        )
    }

    /// Lends the exact prepared routes, addresses and wave tables to a checked quote.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_borrowed_with_provider<A, B, S, P, F, Movement, Provider>(
        &self,
        architecture: A,
        policy: P,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        movement: Movement,
        provider: Option<Provider>,
    ) -> Result<
        crate::partitioned_execution::CompositePartitionExecutor<
            A,
            B,
            S,
            P,
            F,
            crate::partitioned_execution::SelectedCompositePartitionUnitStrategy<
                Provider,
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
        let strategy = match &self.strategy {
            PreparedCompositeUnitStrategy::Direct => {
                if provider.is_some() {
                    return Err(eredu_nn::Error::backend(
                        "dense composite execution received an unselected bank provider",
                    ));
                }
                crate::partitioned_execution::SelectedCompositePartitionUnitStrategy::Direct
            }
            PreparedCompositeUnitStrategy::Routed {
                plans,
                expert_group,
            } => crate::partitioned_execution::SelectedCompositePartitionUnitStrategy::routed_from_prepared_grouped_plan(
                provider.ok_or_else(|| eredu_nn::Error::backend("routed composite provider was not constructed"))?, plans.clone(), *expert_group, self.tensor_group, movement,
            ),
            PreparedCompositeUnitStrategy::RoutedCollective {
                plans,
                expert_group,
                tensor_group,
                wave_group,
                waves,
            } => crate::partitioned_execution::SelectedCompositePartitionUnitStrategy::routed_with_prepared_collective_waves(
                provider.ok_or_else(|| eredu_nn::Error::backend("routed composite provider was not constructed"))?,
                plans.clone(),
                *expert_group,
                movement,
                *tensor_group,
                *wave_group,
                waves.clone(),
            ),
        };
        crate::partitioned_execution::CompositePartitionExecutor::from_prepared_structure(
            architecture,
            policy,
            parallel,
            allocator,
            strategy,
            self.structure.clone(),
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
    pub const fn expert_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.expert_group
    }
}

/// A typed composite partition together with its exact local payload projection.
pub struct PreparedCompositePartition<A, G, W> {
    executor: PreparedCompositeExecutorPlan,
    prepared: PreparedPartitionedAdmission<
        A,
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
        G,
        W,
    >,
    source_architecture: Option<(Box<A>, LocalModelLayout)>,
    layout: LocalModelLayout,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    publication: crate::partitioned_execution::PublicationValueDescriptor,
    routed: Option<PreparedCompositeRoutedExecution>,
    banks: Option<crate::prepared_execution::PreparedPartitionBanks>,
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

    /// Borrows the immutable plan emitted by this completed constructor.
    pub fn retained_executor_plan(&self) -> &PreparedCompositeExecutorPlan {
        &self.executor
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

    /// Exact local bank sources and residency selected for this composite partition.
    pub fn partition_banks(&self) -> Option<&crate::prepared_execution::PreparedPartitionBanks> {
        self.banks.as_ref()
    }

    /// Exact admitted recipe sources assigned to other expert owners. These are
    /// distinct from unexpected checkpoint keys and from this rank's bindings.
    pub fn unowned_expert_checkpoint_sources(&self) -> std::collections::BTreeSet<String> {
        let Some(routed) = &self.routed else {
            return std::collections::BTreeSet::new();
        };
        let SelectedCompositeTextRealization::Routed { execution, .. } =
            self.prepared.selected().base()
        else {
            return std::collections::BTreeSet::new();
        };
        let bank = execution
            .bank(eredu_runtime::RoutedBankId::new(0))
            .expect("validated composite bank");
        crate::routed_text::unowned_expert_sources(
            bank.catalog(),
            routed.plan().local_global_group_indices(),
        )
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
        Option<(Box<A>, LocalModelLayout)>,
        LocalModelLayout,
        Vec<ReplicatedTextMaterializationTask>,
    ) {
        (
            self.prepared,
            self.source_architecture,
            self.layout,
            self.tasks,
        )
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
            Option<(
                Box<crate::composite_execution::PreparedCompositeArchitecture<A>>,
                LocalModelLayout,
            )>,
            LocalModelLayout,
            PreparedCompositeExecutorPlan,
            &eredu_runtime::SelectedReplicatedTextRealization,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(R, S), E>,
    {
        let Self {
            executor,
            prepared,
            source_architecture,
            layout,
            tasks,
            capability_estimate: _,
            effective_model_type: _,
            publication: _,
            routed: _,
            banks,
        } = self;
        let excluded = banks
            .as_ref()
            .map(crate::prepared_execution::PreparedPartitionBanks::addressable_logical_targets)
            .unwrap_or_default();
        let excluded = excluded.iter().map(String::as_str).collect();
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
        let source_architecture = source_architecture
            .map(|(source, source_layout)| {
                let source = Box::new(
                    crate::composite_execution::PreparedCompositeArchitecture::new(*source),
                );
                let source_parameters = source.parameter_description(context).map_err(|error| {
                    eredu_runtime::PartitionedSessionPreparationError::Contract(error.to_string())
                })?;
                if source_parameters.graph() != parameters.graph()
                    || source_parameters.unit_layout() != parameters.unit_layout()
                    || crate::partitioned_execution::derive_partitioned_transform_source_layout(
                        &source_parameters,
                        &parameters,
                        topology_rank,
                    )
                    .map_err(eredu_runtime::PartitionedSessionPreparationError::Contract)?
                        != source_layout
                {
                    return Err(eredu_runtime::PartitionedSessionPreparationError::Contract(
                        "composite transform source changed its layout or execution-unit addresses"
                            .into(),
                    ));
                }
                Ok((source, source_layout))
            })
            .transpose()?;
        eredu_runtime::prepare_partitioned_session_runtime_with_exclusions::<_, B, _, S, _, _, E, _>(
            architecture,
            base.execution().clone(),
            partition,
            communication,
            Some(&tasks),
            &excluded,
            topology,
            eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
            context,
            move |input, selected, context| {
                factory(
                    input,
                    source_architecture,
                    layout,
                    executor,
                    selected,
                    context,
                )
            },
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

    /// Optional retained media capability, defaulting to the ordinary selected route.
    fn visit_media<A, G, W>(
        self,
        prepared: PreparedCompositePartition<A, G, W>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + crate::composite_execution::CompositeMediaIngressArchitecture<B, S>
            + PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
    {
        self.visit(prepared)
    }
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

struct CompositePartitionDetails {
    layout: LocalModelLayout,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    publication: crate::partitioned_execution::PublicationValueDescriptor,
    routed: Option<PreparedCompositeRoutedExecution>,
}

fn prepare_composite_partition_banks<G, W: eredu_runtime::ArchitectureBoundary>(
    selected: &SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    partition: &ArchitecturePartition<G, W>,
    layout: &LocalModelLayout,
    tasks: &mut Vec<ReplicatedTextMaterializationTask>,
    routed: Option<&PreparedCompositeRoutedExecution>,
) -> Result<Option<crate::prepared_execution::PreparedPartitionBanks>, String> {
    let banks = if let Some(routed) = routed {
        let SelectedCompositeTextRealization::Routed { execution, .. } = selected.base() else {
            return Err("routed composite lost its bank selection".into());
        };
        let id = eredu_runtime::RoutedBankId::new(0);
        if execution.banks().len() != 1 {
            return Err("composite bank set differs from its prepared routed plan".into());
        }
        let bank = execution
            .bank(id)
            .ok_or_else(|| "composite routed bank is absent".to_owned())?;
        let mut members = crate::routed_text::project_composite_bank_members_with_tasks(
            bank.catalog(),
            execution.text(),
            tasks,
        )
        .map_err(|error| error.to_string())?;
        let local_units = partition
            .units()
            .map(|unit| (partition.graph().groups()[unit.group()].id(), unit.index()))
            .collect::<std::collections::BTreeSet<_>>();
        members.retain(|member| {
            bank.catalog().unit(member.key()).is_some_and(|unit| {
                local_units.contains(&(unit.owner_group().as_str(), unit.owner_unit()))
                    && (unit.distribution() == crate::ExpertResidencyDistribution::Replicated
                        || routed
                            .plan
                            .local_global_group_indices()
                            .contains(&member.key().member()))
            })
        });
        let members = members
            .into_iter()
            .map(|member| member.with_owner_rank(selected.requirements().topology().global_rank()))
            .collect();
        let local_bank = bank
            .with_partition_geometry(
                routed.plan.clone(),
                bank.catalog().clone(),
                members,
                layout,
                execution.bank_residency(),
            )
            .map_err(|error| error.to_string())?;
        let banks = crate::prepared_execution::PreparedPartitionBanks::prepare(
            execution.bank_residency(),
            std::collections::BTreeMap::from([(id, local_bank)]),
            routed.expert_group.is_some(),
        )
        .map_err(|cause| cause.to_string())?;
        let excluded = banks.addressable_logical_targets();
        tasks.retain(|task| !excluded.contains(task.name()));
        Some(banks)
    } else {
        None
    };
    Ok(banks)
}

// Keep the typed handoff outside the multi-family constructor's stack frame.
// Debug builds otherwise reserve a completed admission for every match arm.
#[inline(never)]
fn finish_composite_partition<B, S, A, G, W, V, F>(
    source: Option<&crate::prepared_sources::PreparedModelSources>,
    workspace: Option<crate::prepared_execution::PreparedCompositeModelSource>,
    architecture: A,
    source_architecture: Option<(Box<A>, LocalModelLayout)>,
    selected: SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    partition: ArchitecturePartition<G, W>,
    mut details: CompositePartitionDetails,
    visitor: V,
    visit: F,
) -> Result<V::Output, CompositePartitionPreparationError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + PartitionedLayeredArchitecture<B, S, Boundary = W>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
    V: AuthoritativeCompositePartitionVisitor<B, S>,
    F: FnOnce(V, PreparedCompositePartition<A, G, W>) -> Result<V::Output, V::Error>,
{
    let mut banks = prepare_composite_partition_banks(
        &selected,
        &partition,
        &details.layout,
        &mut details.tasks,
        details.routed.as_ref(),
    )
    .map_err(CompositePartitionPreparationError::Architecture)?;
    if let (Some(routed), Some(banks)) = (&mut details.routed, &banks) {
        let bank = banks
            .banks()
            .get(&eredu_runtime::RoutedBankId::new(0))
            .ok_or_else(|| {
                CompositePartitionPreparationError::Architecture(
                    "composite bank source is absent".into(),
                )
            })?;
        // The bank localizer has already checked catalog distribution and local
        // coordinates. Its plan owns the sole physical-completion slot.
        routed.plan = bank.plan().clone();
    }
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(CompositePartitionPreparationError::Architecture)?;
    let mut executor = PreparedCompositeExecutorPlan::new::<A, B, S, G, W>(
        &prepared,
        details.publication,
        details.routed.clone(),
    )
    .map_err(CompositePartitionPreparationError::Architecture)?;
    if let (Some(source), Some(workspace), Some(state)) =
        (source, workspace, prepared.selected().partition().state())
    {
        if !crate::replicated_text::config_source::exact_selection(
            source,
            prepared.selected().base().execution(),
        ) {
            return Err(CompositePartitionPreparationError::Architecture(
                "composite workspace source differs from the retained selection".into(),
            ));
        }
        let rank = prepared.selected().topology();
        let local_state = prepared
            .selected()
            .base()
            .execution()
            .state()
            .for_partitioned_geometry(state)
            .map_err(|cause| CompositePartitionPreparationError::Architecture(cause.to_string()))?;
        let record = crate::prepared_execution::PreparedDirectPartitionSource::composite(
            workspace,
            source.selected().clone(),
            rank,
            local_state,
            state.global_layer_offset(),
            executor.clone(),
            std::sync::Arc::new(
                prepared
                    .selected()
                    .composite_execution_plan()
                    .map_err(CompositePartitionPreparationError::Architecture)?,
            ),
            prepared.selected().partition().units().collect(),
        );
        let _ = source.construction_semantics().direct_partition.set(record);
        executor = source
            .construction_semantics()
            .direct_partition
            .get()
            .ok_or_else(|| {
                CompositePartitionPreparationError::Architecture(
                    "composite source publication failed".into(),
                )
            })?
            .composite_executor(source.selected(), rank)
            .map_err(CompositePartitionPreparationError::Architecture)?;
        match (&mut banks, executor.bank_plans(), &mut details.routed) {
            (Some(banks), Some(plans), Some(routed)) => {
                banks
                    .adopt_execution_sources(plans)
                    .map_err(CompositePartitionPreparationError::Architecture)?;
                routed.plan = plans
                    .get(&eredu_runtime::RoutedBankId::new(0))
                    .ok_or_else(|| {
                        CompositePartitionPreparationError::Architecture(
                            "retained composite bank is absent".into(),
                        )
                    })?
                    .clone();
            }
            (None, None, None) => {}
            _ => {
                return Err(CompositePartitionPreparationError::Architecture(
                    "retained composite bank and execution source presence differ".into(),
                ))
            }
        }
    }
    visit(
        visitor,
        PreparedCompositePartition {
            executor,
            prepared,
            source_architecture,
            layout: details.layout,
            tasks: details.tasks,
            capability_estimate: details.capability_estimate,
            effective_model_type: details.effective_model_type,
            publication: details.publication,
            routed: details.routed,
            banks,
        },
    )
    .map_err(CompositePartitionPreparationError::Visitor)
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

fn prepared_gated_composite_execution<B, S, A>(
    selected: &SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    architecture: &A,
    layout: &LocalModelLayout,
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
    if execution
        .bank(eredu_runtime::RoutedBankId::new(0))
        .expect("validated composite bank")
        .owner_group()
        .as_str()
        != plan
            .unit_specs()
            .keys()
            .next()
            .map(|(group, _)| group.as_str())
            .ok_or("localized composite expert plan has no execution units")?
    {
        return Err("localized composite expert plan names a different owner group".into());
    }
    let routes_by_unit = execution
        .bank(eredu_runtime::RoutedBankId::new(0))
        .expect("validated composite bank")
        .routes_by_unit()
        .clone();
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
    let local_plan = crate::routed_text::RoutedGroupedPlan::from(plan).with_catalog_distribution(
        execution
            .bank(eredu_runtime::RoutedBankId::new(0))
            .expect("validated composite bank")
            .catalog(),
    )?;
    let partition_unit_coordinates = crate::component_partition::derive_bank_unit_coordinates(
        execution
            .bank(eredu_runtime::RoutedBankId::new(0))
            .expect("validated composite bank")
            .plan(),
        &local_plan,
        execution
            .bank(eredu_runtime::RoutedBankId::new(0))
            .expect("validated composite bank")
            .catalog(),
        layout,
    )
    .map_err(|error| error.to_string())?;
    Ok(PreparedCompositeRoutedExecution {
        owner_group: execution
            .bank(eredu_runtime::RoutedBankId::new(0))
            .expect("validated composite bank")
            .owner_group()
            .clone(),
        plan: local_plan,
        partition_unit_coordinates,
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
    visit_authoritative_composite_partition_impl::<B, S, V>(None, selected, context, visitor)
}

pub(crate) fn visit_authoritative_composite_partition_with_source<B, S, V>(
    source: &crate::prepared_sources::PreparedModelSources,
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
    visit_authoritative_composite_partition_impl::<B, S, V>(
        Some(source),
        selected,
        context,
        visitor,
    )
}

fn visit_authoritative_composite_partition_impl<B, S, V>(
    source: Option<&crate::prepared_sources::PreparedModelSources>,
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
    let target_linear_formats = selected_matrix_formats(
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
        ($visit:ident, $workspace:expr, $architecture:expr, $parameters:expr, $geometry:expr, $partition_geometry:expr,
         $capability:expr, $effective:expr, $output_width:expr, $foundation:expr,
         $routed:expr $(, $source:expr)?) => {{
            let source_architecture = None$(.or($source))?;
            let architecture = $architecture;
            let boundary =
                <_ as PartitionedLayeredArchitecture<B, S>>::boundary_schema(&architecture, B::construction_metadata(context))
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
            let publication = crate::partitioned_execution::PublicationValueDescriptor::new($output_width)
                .map_err(|error| CompositePartitionPreparationError::Architecture(error.to_string()))?;
            finish_composite_partition::<B, S, _, _, _, _, _>(
                source, $workspace, architecture, source_architecture, selected, partition,
                CompositePartitionDetails {
                    layout: $geometry,
                    tasks,
                    capability_estimate: $capability,
                    effective_model_type: $effective,
                    publication,
                    routed: $routed,
                }, visitor, |visitor: V, prepared| visitor.$visit(prepared),
            )
        }};
    }

    match config {
        CompositeConfig::Gemma4(family) => {
            let mut exact = family.clone();
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
            let mut architecture =
                crate::gemma4::LayeredModel::<B>::new_parallel(args.clone(), local, context)
                    .and_then(|architecture| architecture.with_partition_state(&geometry))
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
            if let Some(plan) = &realization {
                architecture =
                    architecture
                        .with_expert_realization(plan.clone())
                        .map_err(|error| {
                            CompositePartitionPreparationError::Architecture(error.to_string())
                        })?;
            }
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
                        &layout,
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
            let mut source_architecture =
                gemma4_transform_source::<B, S>(&exact, &selected, &parameters, &geometry, context)
                    .map_err(CompositePartitionPreparationError::Architecture)?;
            let workspace = if source.is_some() {
                // The completed model graph is also required by the routed
                // boundary and unit constructors. Keep its preparation separate
                // from publication of the direct composite equation source.
                let target = architecture.prepare_source(context).map_err(|cause| {
                    CompositePartitionPreparationError::Architecture(cause.to_string())
                })?;
                let transformed = source_architecture
                    .as_mut()
                    .map(|(value, _)| value.prepare_source(context))
                    .transpose()
                    .map_err(|cause| {
                        CompositePartitionPreparationError::Architecture(cause.to_string())
                    })?;
                if routed {
                    None
                } else {
                    Some(
                        crate::prepared_execution::PreparedCompositeModelSource::Gemma4(
                            target,
                            transformed,
                        ),
                    )
                }
            } else {
                None
            };
            prepare!(
                visit_media,
                workspace,
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
                routed_execution,
                source_architecture
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
                        &layout,
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
            let source_architecture = muse_glimmer_transform_source::<B, S>(
                source,
                &selected,
                &parameters,
                &geometry,
                context,
            )
            .map_err(CompositePartitionPreparationError::Architecture)?;
            let effective = args.model_type.clone();
            prepare!(
                visit_media,
                None,
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
                routed_execution,
                source_architecture
            )
        }
        CompositeConfig::QwenVl(source_args) => {
            let args = qwen_vl_with_formats(source_args, target_linear_formats)
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
            let mut architecture = architecture.with_partition_geometry(geometry.clone());
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
                        &layout,
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
            let mut source_architecture = qwen_vl_transform_source::<B, S>(
                source_args,
                &selected,
                &parameters,
                &geometry,
                context,
            )
            .map_err(CompositePartitionPreparationError::Architecture)?;
            let effective = args.effective_model_type().to_owned();
            let workspace = if source.is_some() && !routed {
                Some(
                    crate::prepared_execution::PreparedCompositeModelSource::QwenVl(
                        architecture.prepare_source(context).map_err(|cause| {
                            CompositePartitionPreparationError::Architecture(cause.to_string())
                        })?,
                        source_architecture
                            .as_mut()
                            .map(|(value, _)| value.prepare_source(context))
                            .transpose()
                            .map_err(|cause| {
                                CompositePartitionPreparationError::Architecture(cause.to_string())
                            })?,
                    ),
                )
            } else {
                None
            };
            prepare!(
                visit_media,
                workspace,
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
                routed_execution,
                source_architecture
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
            let mut architecture = geometry
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
            if let Some(plan) = &realization {
                architecture.install_expert_realization(plan.clone());
            }
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
                        &layout,
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
                visit_media,
                None,
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
        CompositeConfig::Inkling(source_config) => {
            let args = crate::inkling::with_checkpoint_formats(source_config, target_formats)
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
            let local = std::sync::Arc::new(local);
            let mut architecture = crate::inkling::LayeredModel::<B>::new_parallel(
                args.clone(),
                local.clone(),
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
                            &layout,
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
            let mut source_architecture = inkling_transform_source::<B, S>(
                source_config,
                &selected,
                &parameters,
                &geometry,
                context,
            )
            .map_err(CompositePartitionPreparationError::Architecture)?;
            let workspace = if source.is_some() && !routed {
                architecture
                    .prepare_graph_units(None, context)
                    .map_err(|cause| {
                        CompositePartitionPreparationError::Architecture(cause.to_string())
                    })?;
                if let Some((value, _)) = source_architecture.as_mut() {
                    value.prepare_graph_units(None, context).map_err(|cause| {
                        CompositePartitionPreparationError::Architecture(cause.to_string())
                    })?;
                }
                Some(
                    crate::prepared_execution::PreparedCompositeModelSource::Inkling(
                        architecture.construction_source().clone(),
                        source_architecture
                            .as_ref()
                            .map(|(value, _)| value.construction_source().clone()),
                        local,
                    ),
                )
            } else {
                None
            };
            prepare!(
                visit_media,
                workspace,
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
                routed_execution,
                source_architecture
            )
        }
    }
}

// The source keeps its original physical format through the shared materializer.
fn qwen_vl_transform_source<B, S>(
    source: &crate::qwen::vl::ModelArgs,
    selected: &SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    target_parameters: &eredu_runtime::ArchitectureParameterDescription,
    target_geometry: &crate::qwen::vl::PartitionLocalGeometry,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Option<(Box<crate::qwen::vl::LayeredModel<B>>, LocalModelLayout)>, String>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + RuntimeStateComponents<B>
        + AuxiliaryConvolutionState<B::Tensor>,
{
    if !crate::replicated_text::selected_uses_transform(selected.base().execution()) {
        return Ok(None);
    }
    let source = qwen_vl_with_formats(
        source,
        crate::replicated_text::requirement_formats(
            selected.requirements().execution().execution(),
        )
        .into_iter()
        .map(|(name, format)| (name, format.into()))
        .collect(),
    )?;
    let description = crate::qwen::vl::LayeredModel::<B>::new(source.clone(), context)
        .map_err(|e| e.to_string())?;
    let parameters = description
        .parameter_description(context)
        .map_err(|e| e.to_string())?;
    if parameters.graph() != target_parameters.graph()
        || parameters.unit_layout() != target_parameters.unit_layout()
    {
        return Err("Qwen3-VL transform source changed execution-unit addresses".into());
    }
    let rank = selected.requirements().topology();
    let layout = crate::partitioned_execution::derive_partitioned_transform_source_layout(
        &parameters,
        target_parameters,
        rank,
    )?;
    let local = crate::qwen::vl::local_geometry(&source, &layout).map_err(|e| e.to_string())?;
    let mut architecture =
        crate::qwen::vl::LayeredModel::<B>::new_parallel(source.clone(), local, context)
            .map_err(|e| e.to_string())?;
    let ranges = || {
        selected
            .requirements()
            .groups()
            .iter()
            .map(|g| (g.group().as_str(), g.units()))
    };
    let ownership = selected.requirements().ownership();
    let realization =
        crate::qwen::vl::expert_realization_plan(&architecture, rank).map_err(|e| e.to_string())?;
    let geometry = match &realization {
        Some(realization) => crate::qwen::vl::partition_local_routed_geometry(
            &source,
            &layout,
            ranges(),
            ownership,
            rank,
            realization,
        ),
        None => crate::qwen::vl::partition_local_geometry(&source, &layout, ranges(), ownership),
    }
    .map_err(|e| e.to_string())?;
    if geometry.text_units() != target_geometry.text_units()
        || geometry.vision_units() != target_geometry.vision_units()
        || geometry.complete_state_layout() != target_geometry.complete_state_layout()
        || geometry.static_roles() != target_geometry.static_roles()
        || geometry.deepstack_layers() != target_geometry.deepstack_layers()
    {
        return Err("Qwen3-VL transform source changed partition state or ownership".into());
    }
    architecture = architecture.with_partition_geometry(geometry.clone());
    let boundary = <_ as PartitionedLayeredArchitecture<B, S>>::boundary_schema(
        &architecture,
        B::construction_metadata(context),
    )
    .map_err(|e| e.to_string())?;
    let partition = ArchitecturePartition::from_architecture::<B, S, _, _>(
        &architecture,
        ranges(),
        ownership.clone(),
        geometry,
        boundary,
        &parameters,
    )
    .map_err(|e| e.to_string())?;
    if selected.requirements().state() != partition.state() {
        return Err("Qwen3-VL transform source changed selected state".into());
    }
    crate::qwen::vl::PartitionLocalFoundation::from_partition(&source, &partition)
        .map_err(|e| e.to_string())?;
    Ok(Some((Box::new(architecture), layout)))
}

fn gemma4_transform_source<B, S>(
    source: &crate::gemma4::FamilyConfig,
    selected: &SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    target_parameters: &eredu_runtime::ArchitectureParameterDescription,
    target_geometry: &crate::gemma4::PartitionLocalGeometry,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Option<(Box<crate::gemma4::LayeredModel<B>>, LocalModelLayout)>, String>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + RuntimeStateComponents<B>
        + AuxiliaryConvolutionState<B::Tensor>,
{
    if !crate::replicated_text::selected_uses_transform(selected.base().execution()) {
        return Ok(None);
    }
    let source_args = crate::gemma4::with_checkpoint_formats(
        source,
        crate::replicated_text::requirement_formats(
            selected.requirements().execution().execution(),
        ),
    )?;
    let source = &source_args;
    let sparse =
        source.text.layer_schedule.iter().any(|policy| {
            policy.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe
        });
    let description = crate::gemma4::LayeredModel::<B>::new(source.clone(), context)
        .map_err(|error| error.to_string())?;
    let parameters = description
        .parameter_description(context)
        .map_err(|error| error.to_string())?;
    if parameters.graph() != target_parameters.graph()
        || parameters.unit_layout() != target_parameters.unit_layout()
    {
        return Err("Gemma 4 transform source changed execution-unit addresses".into());
    }
    let rank = selected.requirements().topology();
    let layout = crate::partitioned_execution::derive_partitioned_transform_source_layout(
        &parameters,
        target_parameters,
        rank,
    )?;
    let local =
        crate::gemma4::local_geometry(source, &layout).map_err(|error| error.to_string())?;
    let ranges = || {
        selected
            .requirements()
            .groups()
            .iter()
            .map(|group| (group.group().as_str(), group.units()))
    };
    let ownership = selected.requirements().ownership();
    let geometry = if sparse {
        crate::gemma4::parallel::routed_partition_local_geometry(
            source,
            &layout,
            ranges(),
            ownership,
        )
    } else {
        crate::gemma4::partition_local_geometry(source, &layout, ranges(), ownership)
    }
    .map_err(|error| error.to_string())?;
    if geometry.text_units() != target_geometry.text_units()
        || geometry.vision_units() != target_geometry.vision_units()
        || geometry.audio_units() != target_geometry.audio_units()
        || geometry.per_layer_range() != target_geometry.per_layer_range()
        || geometry.complete_state_layout() != target_geometry.complete_state_layout()
        || geometry.static_roles() != target_geometry.static_roles()
        || geometry.embedding_range() != target_geometry.embedding_range()
        || geometry.output_range() != target_geometry.output_range()
    {
        return Err("Gemma 4 transform source changed partition state or ownership".into());
    }
    let mut architecture =
        crate::gemma4::LayeredModel::<B>::new_parallel(source.clone(), local, context)
            .and_then(|architecture| architecture.with_partition_state(&geometry))
            .map_err(|error| error.to_string())?;
    if sparse {
        let realization = crate::gemma4::expert_realization_plan(&architecture, rank)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "Gemma 4 transform source has no expert realization".to_owned())?;
        routed_gemma4_partition_foundation(
            source,
            &layout,
            ranges(),
            ownership,
            rank,
            &realization,
        )?;
        architecture = architecture
            .with_expert_realization(realization)
            .map_err(|error| error.to_string())?;
    }
    let boundary = <_ as PartitionedLayeredArchitecture<B, S>>::boundary_schema(
        &architecture,
        B::construction_metadata(context),
    )
    .map_err(|error| error.to_string())?;
    let partition = ArchitecturePartition::from_architecture::<B, S, _, _>(
        &architecture,
        ranges(),
        ownership.clone(),
        geometry,
        boundary,
        &parameters,
    )
    .map_err(|error| error.to_string())?;
    if selected.requirements().state() != partition.state() {
        return Err("Gemma 4 transform source changed selected state".into());
    }
    if !sparse {
        crate::gemma4::PartitionLocalFoundation::from_partition(source, &partition)
            .map_err(|error| error.to_string())?;
    }
    Ok(Some((Box::new(architecture), layout)))
}

fn muse_glimmer_transform_source<B, S>(
    source: &crate::muse_glimmer::DecoderConfig,
    selected: &SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    target_parameters: &eredu_runtime::ArchitectureParameterDescription,
    target_geometry: &crate::muse_glimmer::PartitionLocalGeometry,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Option<(Box<crate::muse_glimmer::LayeredModel<B>>, LocalModelLayout)>, String>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + RuntimeStateComponents<B>
        + AuxiliaryConvolutionState<B::Tensor>,
{
    if !crate::replicated_text::selected_uses_transform(selected.base().execution()) {
        return Ok(None);
    }
    let source_args = crate::muse_glimmer::with_checkpoint_formats(
        source,
        crate::replicated_text::requirement_formats(
            selected.requirements().execution().execution(),
        ),
    )?;
    let source = &source_args;
    let description = crate::muse_glimmer::LayeredModel::<B>::new(source.clone(), context)
        .map_err(|error| error.to_string())?;
    let parameters = description
        .parameter_description(context)
        .map_err(|error| error.to_string())?;
    if parameters.graph() != target_parameters.graph()
        || parameters.unit_layout() != target_parameters.unit_layout()
    {
        return Err("Muse-Glimmer transform source changed execution-unit addresses".into());
    }
    let rank = selected.requirements().topology();
    let layout = crate::partitioned_execution::derive_partitioned_transform_source_layout(
        &parameters,
        target_parameters,
        rank,
    )?;
    let local =
        crate::muse_glimmer::local_geometry(source, &layout).map_err(|error| error.to_string())?;
    let ranges = || {
        selected
            .requirements()
            .groups()
            .iter()
            .map(|group| (group.group().as_str(), group.units()))
    };
    let ownership = selected.requirements().ownership();
    let geometry = if source.is_moe() {
        crate::muse_glimmer::parallel::routed_partition_local_geometry(
            source,
            &layout,
            ranges(),
            ownership,
        )
    } else {
        crate::muse_glimmer::partition_local_geometry(source, &layout, ranges(), ownership)
    }
    .map_err(|error| error.to_string())?;
    if geometry.text_units() != target_geometry.text_units()
        || geometry.vision_units() != target_geometry.vision_units()
        || geometry.complete_state_layout() != target_geometry.complete_state_layout()
        || geometry.static_roles() != target_geometry.static_roles()
        || geometry.embedding_range() != target_geometry.embedding_range()
        || geometry.output_range() != target_geometry.output_range()
    {
        return Err("Muse-Glimmer transform source changed partition state or ownership".into());
    }
    let mut architecture =
        crate::muse_glimmer::LayeredModel::<B>::new_parallel(source.clone(), local, context)
            .and_then(|architecture| {
                architecture.with_partition_state_offset(geometry.text_units().start)
            })
            .map_err(|error| error.to_string())?;
    if source.is_moe() {
        let realization = crate::muse_glimmer::expert_realization_plan(&architecture, rank)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "Muse-Glimmer transform source has no expert realization".to_owned())?;
        routed_muse_partition_foundation(source, &layout, ranges(), ownership, rank, &realization)?;
        architecture = architecture
            .with_expert_realization(realization)
            .map_err(|error| error.to_string())?;
    }
    let boundary = <_ as PartitionedLayeredArchitecture<B, S>>::boundary_schema(
        &architecture,
        B::construction_metadata(context),
    )
    .map_err(|error| error.to_string())?;
    let partition = ArchitecturePartition::from_architecture::<B, S, _, _>(
        &architecture,
        ranges(),
        ownership.clone(),
        geometry,
        boundary,
        &parameters,
    )
    .map_err(|error| error.to_string())?;
    if selected.requirements().state() != partition.state() {
        return Err("Muse-Glimmer transform source changed selected state".into());
    }
    if !source.is_moe() {
        crate::muse_glimmer::PartitionLocalFoundation::from_partition(source, &partition)
            .map_err(|error| error.to_string())?;
    }
    Ok(Some((Box::new(architecture), layout)))
}

// Retain original checkpoint geometry for selected load-time transforms. This
// source is typed and partition-local; native mechanisms must not reconstruct it
// from transformed target slots or reopen the artifact.
fn inkling_transform_source<B, S>(
    source: &crate::inkling::ModelArgs,
    selected: &SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    target_parameters: &eredu_runtime::ArchitectureParameterDescription,
    target_geometry: &crate::inkling::PartitionLocalGeometry,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Option<(Box<crate::inkling::LayeredModel<B>>, LocalModelLayout)>, String>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + RuntimeStateComponents<B>
        + AuxiliaryConvolutionState<B::Tensor>,
{
    if !crate::replicated_text::selected_uses_transform(selected.base().execution()) {
        return Ok(None);
    }
    let source_args = crate::inkling::with_checkpoint_formats(
        source,
        crate::replicated_text::requirement_formats(
            selected.requirements().execution().execution(),
        ),
    )?;
    let source = &source_args;
    let description = crate::inkling::LayeredModel::<B>::new(source.clone(), context)
        .map_err(|error| error.to_string())?;
    let parameters = description
        .parameter_description(context)
        .map_err(|error| error.to_string())?;
    if parameters.graph() != target_parameters.graph()
        || parameters.unit_layout() != target_parameters.unit_layout()
    {
        return Err("Inkling transform source changed execution-unit addresses".into());
    }
    let rank = selected.requirements().topology();
    let layout = crate::partitioned_execution::derive_partitioned_transform_source_layout(
        &parameters,
        target_parameters,
        rank,
    )?;
    let local =
        crate::inkling::local_geometry(source, &layout).map_err(|error| error.to_string())?;
    let ranges = || {
        selected
            .requirements()
            .groups()
            .iter()
            .map(|group| (group.group().as_str(), group.units()))
    };
    let ownership = selected.requirements().ownership();
    let geometry = if source.text_config.has_sparse_moe_layers() {
        crate::inkling::parallel::routed_partition_local_geometry(
            source,
            &layout,
            ranges(),
            ownership,
        )
    } else {
        crate::inkling::partition_local_geometry(source, &layout, ranges(), ownership)
    }
    .map_err(|error| error.to_string())?;
    if geometry.text_units() != target_geometry.text_units()
        || geometry.vision_units() != target_geometry.vision_units()
        || geometry.complete_state_layout() != target_geometry.complete_state_layout()
        || geometry.static_roles() != target_geometry.static_roles()
        || geometry.embedding_range() != target_geometry.embedding_range()
        || geometry.output_range() != target_geometry.output_range()
    {
        return Err("Inkling transform source changed partition state or ownership".into());
    }
    let mut architecture = crate::inkling::LayeredModel::<B>::new_parallel(
        source.clone(),
        std::sync::Arc::new(local),
        context,
    )
    .and_then(|architecture| architecture.with_partition_state_offset(geometry.text_units().start))
    .map_err(|error| error.to_string())?;
    if source.text_config.has_sparse_moe_layers() {
        let realization = crate::inkling::expert_realization_plan(&architecture, rank)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "Inkling transform source has no expert realization".to_owned())?;
        routed_inkling_partition_foundation(
            source,
            &layout,
            ranges(),
            ownership,
            rank,
            &realization,
        )?;
        architecture = architecture
            .with_expert_realization(realization)
            .map_err(|error| error.to_string())?;
    }
    let boundary = <_ as PartitionedLayeredArchitecture<B, S>>::boundary_schema(
        &architecture,
        B::construction_metadata(context),
    )
    .map_err(|error| error.to_string())?;
    let partition = ArchitecturePartition::from_architecture::<B, S, _, _>(
        &architecture,
        ranges(),
        ownership.clone(),
        geometry,
        boundary,
        &parameters,
    )
    .map_err(|error| error.to_string())?;
    if selected.requirements().state() != partition.state() {
        return Err("Inkling transform source changed selected state".into());
    }
    if !source.text_config.has_sparse_moe_layers() {
        crate::inkling::PartitionLocalFoundation::from_partition(source, &partition)
            .map_err(|error| error.to_string())?;
    }
    Ok(Some((Box::new(architecture), layout)))
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
    let target_linear_formats = selected_matrix_formats(
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
         $routed:expr $(, $source:expr)?) => {{
            let source_architecture = None$(.or($source))?;
            let architecture = $architecture;
            let boundary =
                <_ as PartitionedLayeredArchitecture<B, S>>::boundary_schema(&architecture, B::construction_metadata(context))
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
            let mut tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
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
            let routed = $routed;
            let banks = prepare_composite_partition_banks(
                &selected, &partition, &$layout, &mut tasks, routed.as_ref(),
            ).map_err(CompositePartitionPreparationError::Architecture)?;
            let prepared =
                prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
                    .map_err(CompositePartitionPreparationError::Architecture)?;
            let executor = PreparedCompositeExecutorPlan::new::<_, B, S, _, _>(
                &prepared, publication, routed.clone(),
            ).map_err(CompositePartitionPreparationError::Architecture)?;
            visitor
                .visit(
                    PreparedCompositePartition {
                        executor,
                        prepared,
                        source_architecture,
                        layout: $layout,
                        tasks,
                        capability_estimate: $capability,
                        effective_model_type: $effective,
                        publication,
                        routed,
                        banks,
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
            let mut architecture = geometry
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
            if let Some(plan) = &realization {
                architecture.install_expert_realization(plan.clone());
            }
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
                        &layout,
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
                            &layout,
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
            let source_architecture = inkling_transform_source::<B, S>(
                source,
                &selected,
                &parameters,
                &geometry,
                context,
            )
            .map_err(CompositePartitionPreparationError::Architecture)?;
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
                routed_execution,
                source_architecture
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
