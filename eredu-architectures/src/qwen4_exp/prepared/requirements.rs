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
            &self.parameters,
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
    declaration: &TargetParameters,
    source: &C,
    physical: &BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource>,
    static_recipes: &BTreeMap<String, DerivedWeightRecipe>,
    unit_recipes: &[BTreeMap<String, DerivedWeightRecipe>],
    banks: &BTreeMap<String, Arc<ParameterBankRecipes>>,
) -> Result<
    (
        Vec<Parameter>,
        BTreeMap<String, DerivedWeightRecipe>,
        BTreeMap<String, eredu_checkpoint::recipe::RecipeMetadata>,
    ),
    PreparationError,
> {
    let mut declared = ParameterRequirements::new(source, physical);
    let static_parameters = &declaration.static_parameters;
    declared.declare(&static_parameters.embedding, None, |_| {
        Ok(Owner::StaticRole("embedding".into()))
    })?;
    declared.declare(&static_parameters.readout, None, |_| {
        Ok(Owner::StaticRole("norm".into()))
    })?;
    if let Some(head) = &static_parameters.head {
        declared.declare(head, None, |_| Ok(Owner::StaticRole("output".into())))?;
    }
    if unit_recipes.len() != declaration.units.len() {
        return Err(invalid(
            "target recipes do not match declared execution owners",
        ));
    }
    let mut bank_roots = Vec::with_capacity(declaration.units.len());
    for (ordinal, unit) in declaration.units.iter().enumerate() {
        let bank = unit.expert_bank();
        declared.declare(unit, bank, |_| {
            Ok(Owner::ExecutionUnit {
                group: crate::decoder::TARGET_EXECUTION_GROUP.into(),
                unit: ordinal,
            })
        })?;
        bank_roots.push(
            bank.map(|bank| {
                let packed = bank.packed_parameters()?;
                checkpoint::experts::source_root(&packed)
                    .map(str::to_owned)
                    .map_err(invalid)
            })
            .transpose()?,
        );
    }
    for (name, recipe) in static_recipes {
        declared.add(name, recipe)?;
    }
    for (unit, root) in unit_recipes.iter().zip(bank_roots) {
        for (name, recipe) in unit {
            declared.add(name, recipe)?;
        }
        if let Some(root) = root {
            let bank = banks
                .get(&root)
                .ok_or_else(|| invalid("missing prepared expert bank"))?;
            for (name, recipe) in bank.recipes() {
                declared.add(&format!("{root}.{name}"), recipe)?;
            }
        }
    }
    declared.validate_complete()?;
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
    expert_banks: &BTreeMap<String, Arc<ParameterBankRecipes>>,
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
    let mut global_experts = None;
    for (ordinal, unit) in spec.units.iter().enumerate() {
        let UnitSpec::Decoder { feed_forward, .. } = unit else {
            continue;
        };
        let packed = feed_forward
            .feed_forward
            .parameters
            .routed
            .experts
            .packed_parameters()?;
        let root = checkpoint::experts::source_root(&packed).map_err(invalid)?;
        specs.insert(
            (owner.clone(), ordinal),
            feed_forward.feed_forward.parameters.routed.experts.clone(),
        );
        let parameters = &feed_forward.feed_forward.parameters;
        let count = packed.group_count();
        if global_experts
            .replace(count)
            .is_some_and(|old| old != count)
        {
            return Err(invalid("target declarations disagree on bank cardinality"));
        }
        routes.insert(
            ordinal,
            parameters.routed.selector.selection().top_k() as usize,
        );
        for expert in 0..count {
            let recipes = select_expert(expert_banks, root, expert)?;
            let parameters = recipes
                .into_iter()
                .map(|(local, recipe)| {
                    let target =
                        checkpoint::experts::binding_target(&packed, &local).map_err(invalid)?;
                    let primary = [packed.gate_up(), packed.down()]
                        .into_iter()
                        .any(|parameter| parameter.parameter.id.as_str() == target.as_str());
                    let role = if primary && formats.get(&target)? == LinearFormat::Dense {
                        ExpertParameterRole::quantizable_projection(
                            format!("{local}_scales"),
                            format!("{local}_biases"),
                        )
                    } else {
                        ExpertParameterRole::Preserved
                    };
                    ExpertParameterRecipe::new(&local, target, recipe, role).map_err(invalid)
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
            return Err(invalid("target embedding must supply floating values"));
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
        global_experts.ok_or_else(|| invalid("target declares no routed bank"))?,
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

/// One projection of a construction slot and its explicit storage extension.
struct DeclaredSlot {
    logical_shape: Vec<usize>,
    stored_shape: Vec<usize>,
    role: Role,
    format: LinearFormat,
    owner: Owner,
    companion: Option<(eredu_nn::LinearCompanionRole, String)>,
    transform_companions: Option<(String, String)>,
    transformable: bool,
}

/// Shared architecture-authored projection of exact recipes, used by target and MTP.
pub(super) struct ParameterRequirements<'a, C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized> {
    source: &'a C,
    physical: &'a BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource>,
    slots: BTreeMap<String, DeclaredSlot>,
    pub(super) parameters: Vec<Parameter>,
    pub(super) derived: BTreeMap<String, DerivedWeightRecipe>,
    pub(super) outputs: BTreeMap<String, eredu_checkpoint::recipe::RecipeMetadata>,
}
impl<'a, C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized> ParameterRequirements<'a, C> {
    pub(super) fn new(
        source: &'a C,
        physical: &'a BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource>,
    ) -> Self {
        Self {
            source,
            physical,
            slots: BTreeMap::new(),
            parameters: Vec::new(),
            derived: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }
    }
    /// Registers primary and companion facts from typed construction, before recipes.
    pub(super) fn declare(
        &mut self,
        spec: &(impl eredu_nn::DeclareLogicalParameters + ?Sized),
        bank: Option<&eredu_nn::GroupedGatedProductSpec>,
        mut owner: impl FnMut(&str) -> Result<Owner, PreparationError>,
    ) -> Result<(), PreparationError> {
        use eredu_nn::LogicalParameterRole as Logical;
        let logical = spec.logical_parameters()?;
        let bank_weights = bank
            .map(|bank| {
                use eredu_nn::DeclareLogicalParameters;
                bank.logical_parameters().map(|parameters| {
                    parameters
                        .into_iter()
                        .filter(|parameter| parameter.role == Logical::LinearWeight)
                        .map(|parameter| parameter.parameter.id.clone())
                        .collect::<std::collections::BTreeSet<_>>()
                })
            })
            .transpose()?
            .unwrap_or_default();
        for target in &bank_weights {
            if !logical
                .iter()
                .any(|parameter| parameter.parameter.id == *target)
            {
                return Err(invalid(format!(
                    "expert transform target {target:?} is outside its declared scope"
                )));
            }
        }

        let group = eredu_runtime::logical_parameter_group(
            "prepared",
            eredu_runtime::ParameterRole::Replicated,
            None,
            spec,
            |_| eredu_runtime::MemberSharding::Replicated,
        )
        .map_err(invalid)?;
        let primary: BTreeMap<_, _> = logical
            .iter()
            .map(|d| (d.parameter.id.as_str(), d))
            .collect();
        for member in group.members() {
            let name = member.target();
            let (logical_shape, role, format, transform_companions) = if let Some(d) =
                primary.get(name)
            {
                let format = d
                    .linear_format
                    .map_or(LinearFormat::Dense, |f| f.encoding());
                let role = match (d.role, d.linear_format.is_some()) {
                    (Logical::Embedding, _) => Role::Embedding,
                    (Logical::PositionEmbedding, _) => Role::LookupTable,
                    (Logical::LinearWeight, _) => Role::LinearWeight,
                    (Logical::Normalization | Logical::NormalizationBias, _) => Role::Normalization,
                    (Logical::LinearBias, _) => Role::LinearBias,
                    (_, true) => Role::LinearWeight,
                    _ => Role::Other,
                };
                let companions = if format == LinearFormat::Dense && d.linear_format.is_some() {
                    // Standard ordinary and bank load-time conversion names are
                    // storage policy; construction supplies the logical axes.
                    Some(if bank_weights.contains(&d.parameter.id) {
                        (format!("{name}_scales"), format!("{name}_biases"))
                    } else {
                        let root = name
                            .strip_suffix(".weight")
                            .ok_or_else(|| invalid("ordinary transform target is not a weight"))?;
                        (format!("{root}.scales"), format!("{root}.biases"))
                    })
                } else {
                    None
                };
                (d.shape.clone(), role, format, companions)
            } else {
                (
                    member.global_shape().to_vec(),
                    Role::FormatCompanion,
                    LinearFormat::Dense,
                    None,
                )
            };
            let slot = DeclaredSlot {
                logical_shape,
                stored_shape: member.global_shape().to_vec(),
                role,
                format,
                owner: owner(name)?,
                companion: member.linear_companion().zip(
                    member
                        .linear_companion_of()
                        .map(|primary| primary.as_str().to_owned()),
                ),
                transform_companions,
                transformable: primary
                    .get(name)
                    .is_some_and(|parameter| parameter.linear_format.is_some()),
            };
            if self.slots.insert(name.to_owned(), slot).is_some() {
                return Err(invalid(format!("conflicting prepared parameter {name}")));
            }
        }
        Ok(())
    }
    pub(super) fn validate_complete(&self) -> Result<(), PreparationError> {
        if let Some(name) = self.slots.keys().next() {
            return Err(invalid(format!("recipes omit declared parameter {name}")));
        }
        Ok(())
    }
    pub(super) fn add(
        &mut self,
        name: &str,
        recipe: &DerivedWeightRecipe,
    ) -> Result<(), PreparationError> {
        let output = recipe.infer(self.source).map_err(invalid)?;
        let slot = self.slots.remove(name).ok_or_else(|| {
            invalid(format!(
                "recipe repeats or references undeclared parameter {name}"
            ))
        })?;
        if output.shape != slot.stored_shape {
            return Err(invalid(format!(
                "recipe geometry disagrees with declared parameter {name}"
            )));
        }
        let DeclaredSlot {
            logical_shape,
            role,
            format,
            owner,
            companion,
            transform_companions,
            transformable,
            ..
        } = slot;
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
        let transform = if format == LinearFormat::Dense && transformable {
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
        if let Some((scale, bias)) = transform_companions {
            parameter = parameter
                .with_transform_companions(scale, bias)
                .map_err(invalid)?;
        }
        if let Some((kind, primary)) = companion {
            parameter = parameter
                .with_linear_companion(kind, primary)
                .map_err(invalid)?;
        }
        self.parameters.push(parameter);
        Ok(())
    }
}
