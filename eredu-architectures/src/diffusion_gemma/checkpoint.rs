use eredu_checkpoint::schema::{
    CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
};

use super::FamilyConfig;

/// Resolves admitted physical or canonical routed tensors into the shared bank's
/// logical projections. No parameter replication is introduced by invocation.
pub fn expert_recipes<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &FamilyConfig,
    layer: usize,
) -> Result<eredu_checkpoint::expert::GatedProductExpertRecipes, String> {
    let declaration = config
        .routed(layer)
        .map_err(|e| e.to_string())?
        .ok_or("layer has no shared routed source")?;
    expert_recipes_for_declaration(catalog, config, declaration)
}

fn expert_recipes_for_declaration<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &FamilyConfig,
    declaration: &crate::gemma4::text::routed::Declaration,
) -> Result<eredu_checkpoint::expert::GatedProductExpertRecipes, String> {
    let source = declaration
        .routed
        .experts
        .packed_parameters()
        .map_err(|e| e.to_string())?;
    let gate_up = source.gate_up();
    let down = source.down();
    let logical = gate_up
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or("unresolved diffusion expert root")?
        .to_owned();
    let physical = logical
        .strip_suffix(".switch_glu")
        .ok_or("unresolved diffusion physical expert root")?
        .to_owned();
    let mut alias_plan = None;
    let mut source = |field: &str| {
        let logical = format!("{logical}.{field}");
        let physical = format!("{physical}.{field}");
        for name in [
            &logical,
            &format!("{logical}.weight"),
            &physical,
            &format!("{physical}.weight"),
        ] {
            if catalog.tensor_metadata(name).is_ok() {
                return name.clone();
            }
        }
        // Format aliases come from the same admitted schema as their geometry.
        // Native GGUF catalogs already use the canonical paths above.
        let plan = alias_plan.get_or_insert_with(|| safetensors_plan(config).ok());
        if let Some(plan) = plan {
            if let Some(constraint) = plan
                .common_tensors
                .iter()
                .chain(
                    plan.layout_groups
                        .iter()
                        .flat_map(|group| &group.variants)
                        .flat_map(|variant| &variant.tensors),
                )
                .find(|constraint| {
                    constraint.key == physical || constraint.key == format!("{physical}.weight")
                })
            {
                if let Some(name) = std::iter::once(&constraint.key)
                    .chain(&constraint.aliases)
                    .find(|name| catalog.tensor_metadata(name).is_ok())
                {
                    return name.clone();
                }
            }
        }
        physical
    };
    let names = eredu_checkpoint::expert::GatedProductExpertLayoutNames {
        target_gate_up: gate_up.parameter.id.as_str().to_owned(),
        target_down: down.parameter.id.as_str().to_owned(),
        packed_gate_up: source("gate_up_proj"),
        packed_down: source("down_proj"),
        separate_gate: source("gate_proj"),
        separate_up: source("up_proj"),
        separate_down: source("down_proj"),
        independent: Vec::new(),
    };
    eredu_checkpoint::expert::resolve_gated_product_expert_recipes(catalog, &names)
        .map_err(|error| error.to_string())
}

/// Derived checkpoint projections for one shared layer residency unit.
pub fn unit_recipes<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &FamilyConfig,
    ordinal: usize,
) -> Result<std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, String>
{
    let policy = config
        .text
        .layer_policy(ordinal)
        .ok_or("diffusion unit lies outside the stack")?;
    if policy.feed_forward == crate::gemma4::FeedForwardPolicy::Dense {
        return Ok(Default::default());
    }
    let expert = expert_recipes(catalog, config, ordinal)?;
    eredu_checkpoint::expert::gated_product_materialization_recipes(catalog, &expert)
}

/// Exact independently addressable expert storage for the shared stack.
pub fn expert_residency_catalog<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &FamilyConfig,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let experts = config
        .routed_layers()
        .next()
        .map(|(_, scope)| scope)
        .ok_or("DiffusionGemma expert residency requires a sparse invocation")?
        .routed
        .experts
        .group_count() as usize;
    let group = eredu_runtime::ExecutionGroupId::new(super::model::TEXT_EXECUTION_GROUP)
        .map_err(|error| error.to_string())?;
    let banks = config.routed_layers().map(|(index, declaration)| {
        expert_recipes_for_declaration(catalog, config, declaration)
            .map(|recipes| (index, format!("model.decoder.layers.{index}"), recipes))
    });
    crate::expert_residency::gated_product_residency_catalog(catalog, group, 0, experts, banks)
}

fn physical_name(name: &str) -> String {
    let mut name = name.to_owned();
    for (prefix, target) in [
        ("model.language_model.", "model.decoder."),
        ("model.vision_tower.", "model.encoder.vision_tower."),
        ("model.embed_vision.", "model.encoder.embed_vision."),
        ("model.layers.", "model.decoder.layers."),
        ("model.embed_tokens.", "model.decoder.embed_tokens."),
        ("model.norm.", "model.decoder.norm."),
    ] {
        if let Some(suffix) = name.strip_prefix(prefix) {
            name = format!("{target}{suffix}");
            break;
        }
    }
    if name.contains(".experts.switch_glu.") {
        name = name.replace(".experts.switch_glu.", ".experts.");
        if let Some(key) = name.strip_suffix(".weight") {
            name = key.to_owned();
        }
    }
    name
}

fn map_constraint(tensor: &mut SafetensorsTensorConstraint) {
    tensor.key = physical_name(&tensor.key);
    tensor.aliases = tensor
        .aliases
        .iter()
        .map(|name| physical_name(name))
        .filter(|name| name != &tensor.key)
        .collect();
    if let Some(companion) = &mut tensor.linear_companion {
        companion.primary = physical_name(&companion.primary);
    }
}

/// Exact shared text, independent encoder scalars, vision and conditioning schema.
pub fn safetensors_plan(config: &FamilyConfig) -> Result<SafetensorsCheckpointPlan, String> {
    let mut plan = crate::gemma4::checkpoint::shared_safetensors_plan(
        &config.text,
        config.vision.as_ref(),
        "model.encoder.vision_tower",
        None,
        config.pinned(),
        config.projectors(),
        config.blocks(),
        config.dense_scopes(),
        config.attention_scopes(),
        config.routed_scopes(),
    )?;
    for tensor in &mut plan.common_tensors {
        map_constraint(tensor);
    }
    for group in &mut plan.layout_groups {
        group.id = physical_name(&group.id);
        for variant in &mut group.variants {
            variant.discriminator_keys = variant
                .discriminator_keys
                .iter()
                .map(|key| physical_name(key))
                .collect();
            for tensor in &mut variant.tensors {
                map_constraint(tensor);
            }
        }
    }
    let declaration = config.conditioning();
    let scalar = |key: String, shape: Vec<usize>| {
        SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
    };
    for layer in 0..config.text.num_hidden_layers() {
        use eredu_nn::DeclareLogicalParameters;
        let spec = config.encoder_scalar(layer).map_err(|e| e.to_string())?;
        for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
            plan.common_tensors.push(scalar(
                parameter.parameter.id.as_str().to_owned(),
                parameter.shape,
            ));
        }
    }
    use eredu_nn::DeclareLogicalParameters;
    for parameter in declaration
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        let id = parameter.parameter.id.as_str();
        if let Some(format) = parameter.linear_format {
            plan.common_tensors.extend(
                eredu_checkpoint::schema::matrix_for_linear_format(
                    id,
                    std::iter::empty::<String>(),
                    parameter.shape,
                    format.encoding(),
                    None,
                )
                .map_err(|e| e.to_string())?,
            );
        } else {
            plan.common_tensors
                .push(scalar(id.to_owned(), parameter.shape));
        }
    }
    SafetensorsCheckpointPlan::new(
        "DiffusionGemma SafeTensors",
        plan.common_tensors,
        plan.layout_groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Exact published GGUF text geometry, including independent invocation scales.
pub fn gguf_plan(
    config: &FamilyConfig,
) -> Result<eredu_checkpoint::schema::GgufCheckpointPlan, String> {
    use eredu_checkpoint::schema::{
        GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint, TensorOperation,
        TensorRequirement,
    };
    let mut shared = crate::gemma4::checkpoint::shared_gguf_plan(
        &config.text,
        config.pinned(),
        config.blocks(),
        config.dense_scopes(),
        config.attention_scopes(),
        config.routed_scopes(),
    )?;
    let declaration = config.conditioning();
    let vector = |key: String, shape| {
        GgufTensorConstraint::required(
            key,
            shape,
            GgufTypeConstraint::OperationClass(TensorOperation::Vector),
        )
    };
    for layer in 0..config.text.num_hidden_layers() {
        use eredu_nn::DeclareLogicalParameters;
        let spec = config.encoder_scalar(layer).map_err(|e| e.to_string())?;
        for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
            shared
                .common_tensors
                .push(vector(encoder_gguf_key(layer), parameter.shape));
        }
    }
    for (parameter, source) in conditioning_gguf_slots(declaration)? {
        shared.common_tensors.push(GgufTensorConstraint::required(
            source,
            parameter.shape,
            GgufTypeConstraint::OperationClass(
                if parameter.role == eredu_nn::LogicalParameterRole::LinearWeight {
                    TensorOperation::Matrix
                } else {
                    TensorOperation::Vector
                },
            ),
        ));
    }
    let full_head = config
        .text
        .layer_schedule
        .iter()
        .map(|policy| policy.head_dim.get() as usize)
        .max()
        .ok_or("empty diffusion rotary schedule")?;
    let mut frequencies = vector("rope_freqs.weight".into(), vec![full_head / 2]);
    frequencies.requirement = TensorRequirement::Optional;
    frequencies.role = eredu_checkpoint::schema::TensorRole::ArtifactMetadata;
    shared.common_tensors.push(frequencies);
    GgufCheckpointPlan::new(
        "DiffusionGemma GGUF",
        shared.common_tensors,
        shared.layout_groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn encoder_gguf_key(layer: usize) -> String {
    format!("blk.{layer}.enc_layer_output_scale.weight")
}

fn conditioning_gguf_slots(
    spec: &super::conditioning::Declaration,
) -> Result<Vec<(eredu_nn::LogicalParameterDeclaration<'_>, &'static str)>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let mut slots = spec
        .pre_norm
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|p| (p, "self_cond_pre_norm.weight"))
        .collect::<Vec<_>>();
    for (projection, source) in spec.product.projections().into_iter().zip([
        "self_cond_gate.weight",
        "self_cond_up.weight",
        "self_cond_down.weight",
    ]) {
        for parameter in projection.logical_parameters().map_err(|e| e.to_string())? {
            if parameter.role != eredu_nn::LogicalParameterRole::LinearWeight {
                return Err(
                    "diffusion feedback GGUF extension does not cover projection bias slots".into(),
                );
            }
            slots.push((parameter, source));
        }
    }
    Ok(slots)
}

/// Prepared feedback, encoding-scale and shared-stack source identities, before payload reads.
#[derive(Debug, Clone)]
pub struct GgufNames {
    shared: crate::gemma4::TextGgufNames,
    bindings: std::collections::BTreeMap<String, String>,
}
impl GgufNames {
    /// Resolves format extensions against typed declarations before execution formats are selected.
    pub fn new(config: &FamilyConfig) -> Result<Self, String> {
        let spec = config.conditioning();
        let mut bindings = std::collections::BTreeMap::new();
        let mut insert = |source: String, target: String| -> Result<(), String> {
            if bindings.insert(source.clone(), target).is_some() {
                return Err(format!("conflicting diffusion GGUF alias {source}"));
            }
            Ok(())
        };
        for (parameter, source) in conditioning_gguf_slots(spec)? {
            let target = parameter.parameter.id.as_str();
            insert(source.into(), target.into())?;
            if parameter.role == eredu_nn::LogicalParameterRole::LinearWeight {
                let source = source
                    .strip_suffix(".weight")
                    .ok_or("feedback matrix has no source weight suffix")?;
                let target = target
                    .strip_suffix(".weight")
                    .ok_or("feedback matrix has no target weight suffix")?;
                for suffix in ["scales", "biases"] {
                    insert(format!("{source}.{suffix}"), format!("{target}.{suffix}"))?;
                }
            }
        }
        for layer in 0..config.text.num_hidden_layers() {
            let spec = config.encoder_scalar(layer).map_err(|e| e.to_string())?;
            insert(encoder_gguf_key(layer), spec.weight.id.as_str().to_owned())?;
        }
        Ok(Self {
            shared: crate::gemma4::TextGgufNames::new(
                &config.text,
                config.pinned(),
                config.blocks(),
                config.dense_scopes(),
                config.attention_scopes(),
                config.routed_scopes(),
            )?,
            bindings,
        })
    }
    /// Translates physical outputs into the shared decoder tree and independent encoding scales.
    pub fn translate(&self, name: &str) -> String {
        if let Some(target) = self.bindings.get(name) {
            return target.clone();
        }
        let shared = self.shared.translate_family(name);
        let mut target = shared.strip_prefix("model.language_model.").map_or_else(
            || shared.clone(),
            |suffix| format!("model.decoder.{suffix}"),
        );
        if target.contains(".experts.switch_glu.") {
            if let Some(key) = target.strip_suffix(".weight") {
                target = key.to_owned();
            }
        }
        canonical_safetensors_target(&target)
    }
}

/// Exact canonical targets of released shared routed projections and companions.
pub fn canonical_safetensors_target(name: &str) -> String {
    let mut name = if name.contains(".experts.") && !name.contains(".experts.switch_glu.") {
        name.replace(".experts.", ".experts.switch_glu.")
    } else {
        name.to_owned()
    };
    if name.contains(".experts.switch_glu.") {
        if let Some(prefix) = name.strip_suffix(".weight") {
            name = prefix.to_owned();
        }
        for (source, target) in [(".scales", "_scales"), (".biases", "_biases")] {
            if let Some(prefix) = name.strip_suffix(source) {
                return format!("{prefix}{target}");
            }
        }
    }
    name
}
