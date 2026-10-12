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
    if layer >= config.text.num_hidden_layers() {
        return Err("diffusion expert layer lies outside the stack".into());
    }
    let physical = format!("model.decoder.layers.{layer}.experts");
    let logical = format!("{physical}.switch_glu");
    let root = if ["gate_up_proj", "gate_proj", "down_proj"]
        .iter()
        .any(|field| {
            catalog
                .tensor_metadata(&format!("{logical}.{field}"))
                .is_ok()
                || catalog
                    .tensor_metadata(&format!("{logical}.{field}.weight"))
                    .is_ok()
        }) {
        &logical
    } else {
        &physical
    };
    let source = |field: &str| {
        let name = format!("{root}.{field}");
        if catalog.tensor_metadata(&name).is_ok() {
            name
        } else {
            format!("{name}.weight")
        }
    };
    let names = eredu_checkpoint::expert::GatedProductExpertLayoutNames {
        target_gate_up: format!("{logical}.gate_up_proj"),
        target_down: format!("{logical}.down_proj"),
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
    let experts = usize::try_from(
        config
            .text
            .num_experts
            .ok_or("diffusion expert count is missing")?,
    )
    .map_err(|error| error.to_string())?;
    let group = eredu_runtime::ExecutionGroupId::new(super::model::TEXT_EXECUTION_GROUP)
        .map_err(|error| error.to_string())?;
    let banks = (0..config.text.num_hidden_layers()).map(|index| {
        expert_recipes(catalog, config, index)
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
        None,
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
    let hidden = config.text.hidden_size as usize;
    let intermediate = config
        .text
        .layer_policy(0)
        .ok_or("empty DiffusionGemma layer schedule")?
        .intermediate_size
        .get() as usize;
    let scalar = |key: String, shape: Vec<usize>| {
        SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
    };
    for layer in 0..config.text.num_hidden_layers() {
        plan.common_tensors.push(scalar(
            format!("model.encoder.language_model.layers.{layer}.layer_scalar"),
            vec![1],
        ));
    }
    plan.common_tensors.push(scalar(
        "model.decoder.self_conditioning.pre_norm.weight".into(),
        vec![hidden],
    ));
    for (field, shape) in [
        ("gate_proj", vec![intermediate, hidden]),
        ("up_proj", vec![intermediate, hidden]),
        ("down_proj", vec![hidden, intermediate]),
    ] {
        plan.common_tensors.extend(
            eredu_checkpoint::schema::matrix_for_linear_format(
                format!("model.decoder.self_conditioning.{field}.weight"),
                std::iter::empty::<String>(),
                shape,
                config
                    .text
                    .linear_format_for(&format!("model.decoder.self_conditioning.{field}.weight")),
                None,
            )
            .map_err(|error| error.to_string())?,
        );
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
    let mut shared = crate::gemma4::checkpoint::gguf_plan(&config.text)?;
    let hidden = config.text.hidden_size as usize;
    let width = config
        .text
        .layer_policy(0)
        .ok_or("empty diffusion stack")?
        .intermediate_size
        .get() as usize;
    let vector = |key: String, shape| {
        GgufTensorConstraint::required(
            key,
            shape,
            GgufTypeConstraint::OperationClass(TensorOperation::Vector),
        )
    };
    for layer in 0..config.text.num_hidden_layers() {
        shared.common_tensors.push(vector(
            format!("blk.{layer}.enc_layer_output_scale.weight"),
            vec![1],
        ));
    }
    shared
        .common_tensors
        .push(vector("self_cond_pre_norm.weight".into(), vec![hidden]));
    for (key, shape) in [
        ("self_cond_gate.weight", vec![width, hidden]),
        ("self_cond_up.weight", vec![width, hidden]),
        ("self_cond_down.weight", vec![hidden, width]),
    ] {
        shared.common_tensors.push(GgufTensorConstraint::required(
            key,
            shape,
            GgufTypeConstraint::OperationClass(TensorOperation::Matrix),
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

/// Translates the published GGUF's physical identities to the shared modules.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("self_cond_pre_norm.weight", "pre_norm.weight"),
        ("self_cond_gate.weight", "gate_proj.weight"),
        ("self_cond_up.weight", "up_proj.weight"),
        ("self_cond_down.weight", "down_proj.weight"),
    ] {
        let source = source.strip_suffix(".weight").unwrap();
        let target = target.strip_suffix(".weight").unwrap();
        if name == source || name.starts_with(&format!("{source}.")) {
            return format!(
                "model.decoder.self_conditioning.{target}{}",
                &name[source.len()..]
            );
        }
    }
    if let Some((layer, "enc_layer_output_scale.weight")) = name
        .strip_prefix("blk.")
        .and_then(|name| name.split_once('.'))
    {
        return format!("model.encoder.language_model.layers.{layer}.layer_scalar");
    }
    let shared = crate::gemma4::checkpoint::translate_family_gguf_weight_name(name);
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
