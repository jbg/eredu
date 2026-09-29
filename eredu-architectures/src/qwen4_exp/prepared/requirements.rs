//! Cold projection of retained target sources into shared execution contracts.
use super::*;
use crate::{
    routed_text::{RoutedBankRequirements, RoutedTextRequirements},
    ExpertParameterRecipe, ExpertParameterRole, ExpertRealizationPlan, ExpertResidencyCatalog,
    ExpertResidencyDistribution, ExpertResidencyUnit,
};
use eredu_checkpoint::store::TensorSelection;
use eredu_runtime::{
    AppendStreamBinding, ExecutionGraph, ExecutionGroupId, ExecutionUnitLayout, ParameterBankKey,
    ParameterTransformConstraint, ReplicatedTextParameterOwner as Owner,
    ReplicatedTextParameterPresence as Presence, ReplicatedTextParameterRequirement as Parameter,
    ReplicatedTextParameterRole as Role, ReplicatedTextRequirements, ReplicatedTextStateAccess,
    RoutedBankId, SelectedRowLookupPlans,
};

fn invalid(error: impl std::fmt::Display) -> PreparationError {
    PreparationError::Contract(error.to_string())
}

impl PreparedTarget {
    /// Binds a previously selected source-free row admission to this target's
    /// exact sources without repeating mechanism queries. The admission may be
    /// selected from headers before integer controls or scales are read.
    /// Dense matrix input axes declare optional aligned load-time transforms; selection
    /// reconstructs the corresponding target specification before native binding.
    /// Vision and prediction remain separate prepared roles.
    pub fn execution_plan(
        &self,
        streams: Vec<AppendStreamBinding>,
        row_admission: SelectedRowLookupPlans,
    ) -> Result<TargetExecutionPlan, PreparationError> {
        let physical = physical_sources(self.artifact.as_ref())?;
        let unit_recipes = self
            .units
            .iter()
            .map(|unit| unit.recipes().clone())
            .collect::<Vec<_>>();
        let parameters = target_parameters(
            &self.spec.config,
            &self.formats,
            self.artifact.as_ref(),
            &physical,
            self.static_parameters.recipes(),
            &unit_recipes,
            &self.expert_banks,
        )?;
        let requirements = target_requirements(
            &self.spec,
            &self.formats,
            self.artifact.as_ref(),
            &parameters,
            self.static_parameters.recipes(),
            &self.expert_banks,
            streams,
            row_admission.clone(),
        )?;
        let row_sources = self.bind_rows(&row_admission)?;
        Ok(TargetExecutionPlan {
            target: self.clone(),
            prediction: None,
            load_selection: None,
            requirements,
            row_sources,
            capability: crate::capability::qwen4_exp_target(&self.spec)?,
        })
    }
}

/// Reads exact metadata/provenance only. Payload acquisition is never needed to
/// declare ordinary or expert requirements.
pub(super) fn physical_sources(
    source: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource>, PreparationError> {
    source
        .source_keys()
        .into_iter()
        .map(|key| {
            let physical =
                crate::replicated_text::exact_physical_source(source, &key).map_err(invalid)?;
            Ok((key, physical))
        })
        .collect()
}

impl PreparedTarget {
    pub(super) fn bind_rows(
        &self,
        admission: &SelectedRowLookupPlans,
    ) -> Result<eredu_runtime::PreparedRowLookups, PreparationError> {
        let entries = self
            .tables
            .iter()
            .map(|(&layer, table)| {
                let descriptor = admission
                    .descriptors()
                    .entries()
                    .get(&table.lookup.parameter)
                    .ok_or_else(|| {
                        eredu_runtime::RowLookupError::Specification(table.lookup.parameter.clone())
                    })?;
                self.row_lookup(layer, descriptor.limits(), descriptor.range().policy())
            })
            .collect::<Result<Vec<_>, PreparationError>>()?;
        let rows = eredu_runtime::PreparedRowLookups::new(entries, self.spec.units.len())?;
        rows.validate_other_banks([0])?;
        // Prove exact descriptor equality here. Retain the unbound source set for
        // the common prepared handoff, where completion-owned row providers bind.
        if admission.descriptors() != rows.descriptors() {
            return Err(eredu_runtime::RowLookupSelectionError::DescriptorMismatch.into());
        }
        Ok(rows)
    }
}

pub(super) fn target_parameters<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
    config: &Config,
    formats: &ParameterFormats,
    source: &C,
    physical: &BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource>,
    static_recipes: &BTreeMap<String, DerivedWeightRecipe>,
    unit_recipes: &[BTreeMap<String, DerivedWeightRecipe>],
    banks: &BTreeMap<String, Arc<PreparedExpertBank>>,
) -> Result<
    (
        Vec<Parameter>,
        BTreeMap<String, DerivedWeightRecipe>,
        BTreeMap<String, eredu_checkpoint::recipe::RecipeMetadata>,
    ),
    PreparationError,
> {
    let mut declared = ParameterRequirements::new(source, physical, formats);
    for (name, recipe) in static_recipes {
        let role = if name.starts_with("model.embed_tokens.") {
            "embedding"
        } else if name.starts_with("lm_head.") {
            "output"
        } else {
            "norm"
        };
        declared.add(name, recipe, Owner::StaticRole(role.into()), false)?;
    }
    let mut ordinal = 0;
    for layer in 0..config.layers.len() {
        if config.ngram.layers.contains(&layer) {
            for (name, recipe) in &unit_recipes[ordinal] {
                declared.add(
                    name,
                    recipe,
                    Owner::ExecutionUnit {
                        group: crate::decoder::TARGET_EXECUTION_GROUP.into(),
                        unit: ordinal,
                    },
                    false,
                )?;
            }
            ordinal += 1;
        }
        let owner = Owner::ExecutionUnit {
            group: crate::decoder::TARGET_EXECUTION_GROUP.into(),
            unit: ordinal,
        };
        for (name, recipe) in &unit_recipes[ordinal] {
            declared.add(name, recipe, owner.clone(), false)?;
        }
        let root = format!("model.layers.{layer}.mlp.experts");
        for (name, recipe) in &banks
            .get(&root)
            .ok_or_else(|| invalid("missing prepared expert bank"))?
            .recipes
        {
            declared.add(&format!("{root}.{name}"), recipe, owner.clone(), true)?;
        }
        ordinal += 1;
    }
    Ok((declared.parameters, declared.derived, declared.outputs))
}

/// One family authoring algorithm shared by header admission and bound artifacts.
pub(super) fn target_requirements<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
    spec: &TargetSpec,
    formats: &ParameterFormats,
    source: &C,
    parameters: &(
        Vec<Parameter>,
        BTreeMap<String, DerivedWeightRecipe>,
        BTreeMap<String, eredu_checkpoint::recipe::RecipeMetadata>,
    ),
    static_recipes: &BTreeMap<String, DerivedWeightRecipe>,
    expert_banks: &BTreeMap<String, Arc<PreparedExpertBank>>,
    streams: Vec<AppendStreamBinding>,
    row_admission: SelectedRowLookupPlans,
) -> Result<RoutedTextRequirements, PreparationError> {
    let state_layout = spec.state_layout()?;
    AppendStreamBinding::validate_layout(&state_layout, &streams)?;
    for binding in &streams {
        if binding.lanes != spec.limits.qsa.batch as u32 {
            return Err(eredu_runtime::AppendStreamError::Geometry.into());
        }
        if binding.limits.read_entries < spec.limits.tile_blocks as usize {
            return Err(eredu_runtime::AppendStreamError::Budget {
                resource: "summary tile entries",
                required: spec.limits.tile_blocks as u64,
                limit: binding.limits.read_entries as u64,
            }
            .into());
        }
    }
    let graph = ExecutionGraph::chain([crate::decoder::TARGET_EXECUTION_GROUP]).map_err(invalid)?;
    let units = ExecutionUnitLayout::new(&graph, [spec.units.len()]).map_err(invalid)?;
    let owner = ExecutionGroupId::new(crate::decoder::TARGET_EXECUTION_GROUP).map_err(invalid)?;
    let mut specs = BTreeMap::new();
    let mut routes = BTreeMap::new();
    let mut members = Vec::new();
    for (ordinal, unit) in spec.units.iter().enumerate() {
        let UnitSpec::Decoder {
            layer,
            feed_forward,
            ..
        } = unit
        else {
            continue;
        };
        let root = format!("model.layers.{layer}.mlp.experts");
        specs.insert(
            (owner.clone(), ordinal),
            feed_forward.feed_forward.experts.clone(),
        );
        routes.insert(ordinal, spec.config.experts.selected as usize);
        for expert in 0..spec.config.experts.count as usize {
            let recipes = select_expert(expert_banks, &root, expert)?;
            let parameters = recipes
                .into_iter()
                .map(|(local, recipe)| {
                    ExpertParameterRecipe::new(
                        &local,
                        format!("{root}.{local}"),
                        recipe,
                        if matches!(local.as_str(), "gate_up_proj" | "down_proj")
                            && formats.get(&format!("{root}.{local}"))? == LinearFormat::Dense
                        {
                            ExpertParameterRole::quantizable_projection(
                                format!("{local}_scales"),
                                format!("{local}_biases"),
                            )
                        } else {
                            ExpertParameterRole::Preserved
                        },
                    )
                    .map_err(invalid)
                })
                .collect::<Result<Vec<_>, _>>()?;
            members.push(
                ExpertResidencyUnit::new(
                    ParameterBankKey::new(0, ordinal, expert),
                    owner.clone(),
                    ordinal,
                    unit.path(),
                    ExpertResidencyDistribution::ExpertParallel,
                    parameters,
                )
                .map_err(invalid)?,
            );
        }
    }
    let embedding = static_recipes["model.embed_tokens.weight"]
        .infer(source)
        .map_err(invalid)?;
    let dtype = match (formats.get("model.embed_tokens.weight")?, embedding.dtype) {
        (LinearFormat::Dense, eredu_checkpoint::recipe::RecipeDtype::F32) => {
            eredu_core::checkpoint::TensorDtype::F32
        }
        (LinearFormat::Dense, eredu_checkpoint::recipe::RecipeDtype::F16) => {
            eredu_core::checkpoint::TensorDtype::F16
        }
        (LinearFormat::Dense, eredu_checkpoint::recipe::RecipeDtype::BF16) => {
            eredu_core::checkpoint::TensorDtype::Bf16
        }
        (LinearFormat::Dense, _) => {
            return Err(invalid("target embedding must supply floating values"))
        }
        _ => match spec.limits.element {
            eredu_nn::TensorElementType::F32 => eredu_core::checkpoint::TensorDtype::F32,
            eredu_nn::TensorElementType::F16 => eredu_core::checkpoint::TensorDtype::F16,
            eredu_nn::TensorElementType::Bf16 => eredu_core::checkpoint::TensorDtype::Bf16,
            _ => return Err(invalid("target state requires F16, BF16 or F32")),
        },
    };
    let text = ReplicatedTextRequirements::new(
        spec.geometry_fingerprint(),
        spec.required_operators(),
        graph,
        units,
        vec![crate::transport::decoder()],
        state_layout,
        ReplicatedTextStateAccess::AttentionWithStreams,
        parameters.0.clone(),
    )
    .map_err(invalid)?
    .with_floating_state_source(dtype)
    .with_chunked_prefill(true)
    .with_derived_recipes(parameters.1.clone(), parameters.2.clone())
    .map_err(invalid)?
    .with_append_streams(streams)
    .map_err(invalid)?;
    let catalog = ExpertResidencyCatalog::new(members)
        .map_err(invalid)?
        .with_inferred_byte_geometry(source)
        .map_err(invalid)?;
    let plan = ExpertRealizationPlan::balanced(
        spec.config.experts.count as usize,
        eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(invalid)?,
            0,
        )
        .map_err(invalid)?,
        specs,
    )
    .map_err(invalid)?;
    let bank = RoutedBankRequirements::new(owner, plan.into(), catalog, routes).map_err(invalid)?;
    let requirements = RoutedTextRequirements::new(text, [(RoutedBankId::new(0), bank)], source)
        .map_err(invalid)?
        .with_row_lookups(row_admission)
        .map_err(invalid)?;
    Ok(requirements)
}

fn companion(name: &str) -> Option<(eredu_nn::LinearCompanionRole, String)> {
    use eredu_nn::LinearCompanionRole::{AffineBias, Scale};
    for (suffix, role, ordinary) in [
        ("_scale_inv", Scale, false),
        ("_scales", Scale, false),
        ("_biases", AffineBias, false),
        (".scales", Scale, true),
        (".biases", AffineBias, true),
    ] {
        if let Some(root) = name.strip_suffix(suffix) {
            return Some((
                role,
                if ordinary {
                    format!("{root}.weight")
                } else {
                    root.into()
                },
            ));
        }
    }
    None
}

// Family checkpoint semantics: convolution kernels are rank three, norms are
// rank one, and ordinary affine projections are rank-two `.weight` tensors.
// Expert banks have explicit grouped geometry and separate scale identities.
fn parameter_role(name: &str, shape: &[usize], grouped: bool) -> Role {
    if companion(name).is_some() {
        Role::FormatCompanion
    } else if name == "model.embed_tokens.weight" {
        Role::Embedding
    } else if grouped || (name.ends_with(".weight") && shape.len() == 2) {
        Role::LinearWeight
    } else if name.ends_with("norm.weight")
        || name.contains(".norm_")
        || name.contains(".pre_fc_norm_")
    {
        Role::Normalization
    } else if name.ends_with(".bias") {
        Role::LinearBias
    } else {
        Role::Other
    }
}

/// Shared architecture-authored projection of exact recipes, used by target and MTP.
pub(super) struct ParameterRequirements<'a, C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized> {
    source: &'a C,
    physical: &'a BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource>,
    formats: &'a ParameterFormats,
    pub(super) parameters: Vec<Parameter>,
    pub(super) derived: BTreeMap<String, DerivedWeightRecipe>,
    pub(super) outputs: BTreeMap<String, eredu_checkpoint::recipe::RecipeMetadata>,
}
impl<'a, C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized> ParameterRequirements<'a, C> {
    pub(super) fn new(
        source: &'a C,
        physical: &'a BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource>,
        formats: &'a ParameterFormats,
    ) -> Self {
        Self {
            source,
            physical,
            formats,
            parameters: Vec::new(),
            derived: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }
    }
    pub(super) fn add(
        &mut self,
        name: &str,
        recipe: &DerivedWeightRecipe,
        owner: Owner,
        grouped: bool,
    ) -> Result<(), PreparationError> {
        let output = recipe.infer(self.source).map_err(invalid)?;
        let role = parameter_role(name, &output.shape, grouped);
        let format = if matches!(role, Role::LinearWeight | Role::Embedding) {
            self.formats.get(name)?
        } else {
            LinearFormat::Dense
        };
        let mut logical_shape = output.shape.clone();
        if matches!(role, Role::LinearWeight | Role::Embedding) {
            let (units, values) = match format {
                LinearFormat::Affine(q) => (u64::try_from(q.bits).map_err(invalid)?, 32),
                LinearFormat::MxFp4 => (4, 32),
                LinearFormat::GgufIQuant { ggml_type, .. } => {
                    let (values, bytes) = ggml_type.block_and_bytes().map_err(invalid)?;
                    (bytes, values)
                }
                _ => (1, 1),
            };
            if let Some(last) = logical_shape.last_mut() {
                let n = (*last as u64)
                    .checked_mul(values)
                    .filter(|n| n % units == 0)
                    .and_then(|n| usize::try_from(n / units).ok())
                    .ok_or_else(|| invalid("invalid packed projection extent"))?;
                *last = n;
            }
        }
        let physical = recipe
            .source_keys()
            .iter()
            .map(|key| {
                self.physical
                    .get(*key)
                    .cloned()
                    .ok_or_else(|| invalid(format!("missing physical source {key}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (encoding, shape, presence) = match recipe {
            DerivedWeightRecipe::Source {
                key,
                selection: TensorSelection::Full,
            } => {
                let metadata = self.source.tensor_metadata(key)?;
                (
                    self.physical
                        .get(key)
                        .ok_or_else(|| invalid(format!("missing physical source {key}")))?
                        .source_encoding()
                        .clone(),
                    metadata.physical_shape,
                    Presence::Required,
                )
            }
            _ => {
                self.derived.insert(name.to_owned(), recipe.clone());
                self.outputs.insert(name.to_owned(), output.clone());
                (
                    eredu_checkpoint::SourceTensorEncoding::RecipeOutput(
                        crate::replicated_text::recipe_stored_dtype(&output.dtype)
                            .map_err(invalid)?,
                    ),
                    output.shape.clone(),
                    Presence::Derived {
                        recipe: format!("qwen4_exp.parameter:{name}"),
                    },
                )
            }
        };
        let transform = if format == LinearFormat::Dense
            && matches!(role, Role::LinearWeight | Role::Embedding)
        {
            ParameterTransformConstraint::LinearIfAligned {
                packed_axis: output.shape.len() - 1,
                // Shared packed conversion requires 32-column input alignment.
                alignment: 32,
            }
        } else {
            ParameterTransformConstraint::None
        };
        let mut parameter = Parameter::new(
            name,
            recipe
                .source_keys()
                .into_iter()
                .map(str::to_owned)
                .collect(),
            physical,
            vec![],
            Some(encoding),
            Some(shape),
            logical_shape,
            format,
            role,
            owner,
            presence,
            transform,
        )
        .map_err(invalid)?;
        if transform != ParameterTransformConstraint::None {
            let (scale, bias) = if grouped {
                (format!("{name}_scales"), format!("{name}_biases"))
            } else {
                let prefix = name
                    .strip_suffix(".weight")
                    .ok_or_else(|| invalid("ordinary transform target is not a weight"))?;
                (format!("{prefix}.scales"), format!("{prefix}.biases"))
            };
            parameter = parameter
                .with_transform_companions(scale, bias)
                .map_err(invalid)?;
        }
        if let Some((kind, primary)) = companion(name) {
            parameter = parameter
                .with_linear_companion(kind, primary)
                .map_err(invalid)?;
        }
        self.parameters.push(parameter);
        Ok(())
    }
}
