//! Backend-neutral Inkling checkpoint name normalization and derived recipes.

use std::collections::{BTreeMap, HashMap};

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    schema::{
        AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
        GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint, TensorOperation,
    },
    store::TensorSelection,
    WeightQuantization,
};

use super::{FeedForwardPolicy, ModelArgs};

/// Derives an Inkling configuration whose text matrix formats reflect
/// load-time quantization instead of checkpoint-specific selections.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    args.with_text_formats(Some(quantization), None)
        .map_err(|e| e.to_string())
}

/// Applies canonical checkpoint format metadata to a complete Inkling
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    mut formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    normalize_gguf_weight_formats(args, &mut formats)?;
    let mut audio = HashMap::new();
    let mut vision = HashMap::new();
    formats.retain(|name, format| {
        if name.starts_with("audio.") {
            audio.insert(name.clone(), *format);
            false
        } else if name.starts_with("visual.") {
            vision.insert(name.clone(), *format);
            false
        } else {
            true
        }
    });
    args.with_text_formats(None, (!formats.is_empty()).then_some(formats))
        .and_then(|target| target.with_media_formats(audio, vision))
        .map_err(|e| e.to_string())
}

/// Canonical target and released source identity.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterAlias {
    /// Released checkpoint tensor name.
    pub source: String,
    /// Stable neutral parameter identity.
    pub target: String,
}

/// Rehomes split GGUF expert formats onto canonical fused routed and shared banks.
pub fn normalize_gguf_weight_formats(
    args: &ModelArgs,
    formats: &mut HashMap<String, WeightQuantization>,
) -> Result<(), String> {
    for (layer, parameters) in args.sparse_layers() {
        for (bank, declaration) in [
            ("experts", &parameters.routed),
            ("shared_experts", &parameters.shared),
        ] {
            let eredu_nn::GatedProductGroupLayout::Packed { gate_up, .. } = declaration.layout()
            else {
                return Err("Inkling format extension requires a declared packed bank".into());
            };
            let root = format!("model.layers.{layer}.moe.{bank}");
            let gate_name = format!("{root}.gate_proj");
            let up_name = format!("{root}.up_proj");
            let gate = formats.get(&gate_name).copied();
            let up = formats.get(&up_name).copied();
            match (gate, up) {
                (Some(gate), Some(up)) if gate == up => {
                    let target = gate_up.weight().id.as_str();
                    if formats.get(target).is_some_and(|format| *format != gate) {
                        return Err(format!("Inkling fused and split formats disagree for {target}"));
                    }
                    formats.insert(target.to_owned(), gate);
                    formats.remove(&gate_name);
                    formats.remove(&up_name);
                }
                (None, None) => {}
                (gate, up) => return Err(format!("Inkling GGUF fused expert bank {root:?} requires matching gate/up formats, got {gate:?} and {up:?}")),
            }
        }
    }
    Ok(())
}

fn visit_text_pinned_parameters(
    args: &ModelArgs,
    mut visit: impl FnMut(
        &str,
        &str,
        &eredu_nn::LogicalParameterDeclaration<'_>,
        TensorOperation,
    ) -> Result<(), String>,
) -> Result<(), String> {
    let ordinary = &args.pinned().ordinary;
    let embedding_norm = &args.pinned().embedding_norm;
    let output = ordinary.output_projection_spec();
    let slots: [(
        &str,
        &str,
        &dyn eredu_nn::DeclareLogicalParameters,
        TensorOperation,
    ); 4] = [
        (
            "model.llm.embed.weight",
            "token_embd.weight",
            ordinary.embedding_spec(),
            TensorOperation::Matrix,
        ),
        (
            "model.llm.embed_norm.weight",
            "token_embd_norm.weight",
            embedding_norm,
            TensorOperation::Vector,
        ),
        (
            "model.llm.norm.weight",
            "output_norm.weight",
            ordinary.normalization_spec(),
            TensorOperation::Vector,
        ),
        (
            "model.llm.unembed.weight",
            "output.weight",
            output.as_ref(),
            TensorOperation::Matrix,
        ),
    ];
    for (safe, gguf, spec, operation) in slots {
        visit(safe, gguf, &declared_parameter(spec)?, operation)?;
    }
    Ok(())
}

/// Exact affine and media-table geometry from construction slots and source
/// extensions. A raw tensor operation has no affine encoding declaration.
pub(crate) fn source_linear_shapes(
    args: &ModelArgs,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    args.validate().map_err(|error| error.to_string())?;
    let text = &args.text_config;
    let mut shapes = BTreeMap::new();
    visit_text_pinned_parameters(args, |_, _, parameter, _| {
        if parameter.role == eredu_nn::LogicalParameterRole::LinearWeight {
            insert_source_matrix(&mut shapes, parameter)?;
        }
        Ok(())
    })?;
    for (layer, policy) in text.layer_schedule.iter().enumerate() {
        let common = args.common(layer).map_err(|error| error.to_string())?;
        visit_block_parameters(&common, |_, _, parameter, _| {
            insert_source_matrix(&mut shapes, parameter)
        })?;
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let parameters = args.dense(layer).map_err(|error| error.to_string())?;
                project_dense_source_matrices(
                    &mut shapes,
                    &parameters,
                    &format!("model.llm.layers.{layer}"),
                )?;
            }
            FeedForwardPolicy::SparseMoe => {
                let parameters = args.sparse(layer).map_err(|error| error.to_string())?;
                project_declared_source_matrices(&mut shapes, parameters)?;
            }
        }
    }
    visit_media_parameters(args, |_, _, parameter, _| {
        insert_source_matrix(&mut shapes, parameter)
    })?;
    let prediction = args.prediction();
    for (depth, parameters) in prediction.depths.iter().enumerate() {
        project_declared_source_matrices(&mut shapes, &parameters.fusion)?;
        visit_block_parameters(&parameters.common, |_, _, parameter, _| {
            insert_source_matrix(&mut shapes, parameter)
        })?;
        project_dense_source_matrices(
            &mut shapes,
            &parameters.dense,
            &format!("model.mtp.layers.{depth}.transformer_block"),
        )?;
    }
    Ok(shapes)
}

fn insert_source_geometry(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    name: &str,
    shape: &[usize],
) -> Result<(), String> {
    if shapes.get(name).is_some_and(|previous| previous != shape) {
        return Err(format!(
            "Inkling source matrix {name:?} has conflicting logical geometry"
        ));
    }
    shapes.insert(name.to_owned(), shape.to_vec());
    Ok(())
}

fn insert_source_matrix(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    parameter: &eredu_nn::LogicalParameterDeclaration<'_>,
) -> Result<(), String> {
    if parameter.linear_format.is_some() && parameter.shape.len() == 2 {
        insert_source_geometry(shapes, parameter.parameter.id.as_str(), &parameter.shape)?;
    }
    Ok(())
}

fn project_declared_source_matrices(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    spec: &impl eredu_nn::DeclareLogicalParameters,
) -> Result<(), String> {
    for parameter in spec
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        insert_source_matrix(shapes, &parameter)?;
    }
    Ok(())
}

fn project_dense_source_matrices(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    parameters: &super::text::DenseParameters,
    released: &str,
) -> Result<(), String> {
    project_declared_source_matrices(shapes, parameters)?;
    let source = dense_input_sources(parameters, released)?;
    insert_source_geometry(shapes, &source.interleaved, &source.shape)
}

/// Builds the complete released SafeTensors catalog, including native media
/// towers and embedded prediction depths.
pub fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let text = &args.text_config;
    if text.weight_quantization.is_some()
        || text
            .quantized_weight_configs
            .as_ref()
            .is_some_and(|formats| !formats.is_empty())
        || args
            .audio_config
            .as_ref()
            .and_then(|audio| audio.quantized_weight_configs.as_ref())
            .is_some_and(|formats| !formats.is_empty())
        || args
            .vision_config
            .as_ref()
            .and_then(|vision| vision.quantized_weight_configs.as_ref())
            .is_some_and(|formats| !formats.is_empty())
    {
        return Err("Inkling SafeTensors requires floating source weights; select executable quantization through the load request".into());
    }
    let layers = dimension(text.num_hidden_layers, "layer count")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    visit_text_pinned_parameters(args, |source, _, parameter, _| {
        common.push(safe_alias(
            source,
            parameter.parameter.id.as_str(),
            parameter.shape.clone(),
        ));
        Ok(())
    })?;
    for layer in 0..layers {
        let released = format!("model.llm.layers.{layer}");
        let canonical = format!("model.layers.{layer}");
        let parameters = args.common(layer).map_err(|error| error.to_string())?;
        visit_block_parameters(&parameters, |source, _, parameter, _| {
            common.push(safe_alias(
                format!("{released}.{source}"),
                parameter.parameter.id.as_str(),
                parameter.shape.clone(),
            ));
            Ok(())
        })?;
        add_feed_forward_layout(
            args.feed_forward(layer).map_err(|e| e.to_string())?,
            &released,
            &canonical,
            &mut common,
            &mut groups,
        )?;
    }
    add_safetensors_media(args, &mut common)?;
    add_safetensors_mtp(args, &mut common, &mut groups)?;
    SafetensorsCheckpointPlan::new(
        "Inkling SafeTensors",
        common,
        groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Builds the complete text GGUF catalog admitted for Inkling.
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let text = &args.text_config;
    let layers = dimension(text.num_hidden_layers, "layer count")?;
    let mut tensors = Vec::new();
    visit_text_pinned_parameters(args, |_, source, parameter, operation| {
        tensors.push(gguf(source, parameter.shape.clone(), operation));
        Ok(())
    })?;
    for layer in 0..layers {
        let root = format!("blk.{layer}");
        let policy = text
            .layer_policy(layer)
            .ok_or_else(|| format!("Inkling layer {layer} has no normalized policy"))?;
        let parameters = args.common(layer).map_err(|error| error.to_string())?;
        visit_block_parameters(&parameters, |_, source, parameter, layout| {
            let key = format!("{root}.{source}");
            let tensor = match layout {
                BlockTensorLayout::Convolution => gguf(
                    key,
                    vec![parameter.shape[0], parameter.shape[2]],
                    layout.operation(),
                )
                .with_alternate_shapes([parameter.shape.clone()]),
                BlockTensorLayout::Profile => {
                    gguf(&key, parameter.shape.clone(), layout.operation())
                        .with_aliases([key.trim_end_matches(".weight").to_owned()])
                }
                _ => gguf(key, parameter.shape.clone(), layout.operation()),
            };
            tensors.push(tensor);
            Ok(())
        })?;
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let parameters = args.dense(layer).map_err(|error| error.to_string())?;
                for (field, spec) in [
                    ("ffn_gate", &parameters.product.gate),
                    ("ffn_up", &parameters.product.up),
                    ("ffn_down", &parameters.product.down),
                ] {
                    tensors.push(gguf(
                        format!("{root}.{field}.weight"),
                        declared_parameter(spec)?.shape,
                        TensorOperation::Matrix,
                    ));
                }
                tensors.push(
                    gguf(
                        format!("{root}.ffn_gscale"),
                        parameters
                            .scale_declaration()
                            .map_err(|error| error.to_string())?
                            .shape,
                        TensorOperation::Vector,
                    )
                    .with_aliases([format!("{root}.ffn_gscale.weight")]),
                );
            }
            FeedForwardPolicy::SparseMoe => {
                use eredu_nn::DeclareLogicalParameters;
                let parameters = args.sparse(layer).map_err(|error| error.to_string())?;
                let router = parameters
                    .router
                    .logical_parameters()
                    .map_err(|error| error.to_string())?;
                let shape = |slot: &eredu_nn::ParameterSpec| {
                    router
                        .iter()
                        .find(|parameter| parameter.parameter.id == slot.id)
                        .map(|parameter| parameter.shape.clone())
                        .ok_or_else(|| {
                            format!(
                                "Inkling source extension references undeclared slot {}",
                                slot.id
                            )
                        })
                };
                tensors.extend([
                    gguf(
                        format!("{root}.ffn_gate_inp.weight"),
                        shape(parameters.router.weight())?,
                        TensorOperation::Dense,
                    ),
                    gguf(
                        format!("{root}.exp_probs_b.bias"),
                        shape(parameters.router.correction_bias())?,
                        TensorOperation::Vector,
                    )
                    .with_aliases([
                        format!("{root}.ffn_exp_probs_b.bias"),
                        format!("{root}.ffn_exp_probs_b"),
                    ]),
                    gguf(
                        format!("{root}.ffn_gscale"),
                        shape(parameters.router.coefficient_scale())?,
                        TensorOperation::Vector,
                    )
                    .with_aliases([format!("{root}.ffn_gscale.weight")]),
                ]);
                for source in sparse_banks(&parameters) {
                    let bank = source
                        .bank
                        .packed_parameters()
                        .map_err(|error| error.to_string())?;
                    let read_shape = bank.unfused_shape();
                    for field in ["gate", "up"] {
                        tensors.push(gguf(
                            format!("{root}.ffn_{field}_{}.weight", source.gguf),
                            read_shape.clone(),
                            TensorOperation::Matrix,
                        ));
                    }
                    tensors.push(gguf(
                        format!("{root}.ffn_down_{}.weight", source.gguf),
                        bank.down().shape.clone(),
                        TensorOperation::Matrix,
                    ));
                }
            }
        }
    }
    GgufCheckpointPlan::new("Inkling GGUF", tensors, Vec::new(), CatalogPolicy::strict())
        .map_err(|error| error.to_string())
}

/// Builds the sibling audio/vision projector GGUF catalog.
pub fn mmproj_gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    if args.audio_config.is_none() {
        return Err("Inkling mmproj has no audio geometry".into());
    }
    if args.vision_config.is_none() {
        return Err("Inkling mmproj has no vision geometry".into());
    }
    let mut tensors = Vec::new();
    visit_media_parameters(args, |_, source, parameter, operation| {
        tensors.push(gguf(source, parameter.shape.clone(), operation));
        Ok(())
    })?;
    GgufCheckpointPlan::new(
        "Inkling mmproj GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

#[derive(Clone, Copy)]
enum BlockTensorLayout {
    Matrix,
    Vector,
    Profile,
    Convolution,
}
impl BlockTensorLayout {
    fn operation(self) -> TensorOperation {
        match self {
            Self::Matrix => TensorOperation::Matrix,
            Self::Vector => TensorOperation::Vector,
            Self::Profile | Self::Convolution => TensorOperation::Dense,
        }
    }
}

/// Source-layout extensions borrow exact slots from the common construction scope.
fn visit_block_parameters(
    parameters: &super::text::CommonParameters,
    mut visit: impl FnMut(
        &str,
        &str,
        &eredu_nn::LogicalParameterDeclaration<'_>,
        BlockTensorLayout,
    ) -> Result<(), String>,
) -> Result<(), String> {
    use BlockTensorLayout::*;
    let attention = &parameters.attention;
    let slots: [(
        &str,
        &str,
        &dyn eredu_nn::DeclareLogicalParameters,
        BlockTensorLayout,
    ); 14] = [
        (
            "attn_norm.weight",
            "attn_norm.weight",
            &parameters.input_norm,
            Vector,
        ),
        (
            "mlp_norm.weight",
            "ffn_norm.weight",
            &parameters.post_attention_norm,
            Vector,
        ),
        (
            "attn.wq_du.weight",
            "attn_q.weight",
            &attention.projection.query,
            Matrix,
        ),
        (
            "attn.wk_dv.weight",
            "attn_k.weight",
            &attention.projection.key,
            Matrix,
        ),
        (
            "attn.wv_dv.weight",
            "attn_v.weight",
            &attention.projection.value,
            Matrix,
        ),
        (
            "attn.wr_du.weight",
            "attn_r.weight",
            &attention.relative,
            Matrix,
        ),
        (
            "attn.wo_ud.weight",
            "attn_output.weight",
            &attention.projection.output,
            Matrix,
        ),
        (
            "attn.q_norm.weight",
            "attn_q_norm.weight",
            &attention.query_norm,
            Vector,
        ),
        (
            "attn.k_norm.weight",
            "attn_k_norm.weight",
            &attention.key_norm,
            Vector,
        ),
        (
            "attn.rel_logits_proj.proj",
            "attn_rel_proj.weight",
            &attention.relative_projection,
            Profile,
        ),
        (
            "attn.k_sconv.weight",
            "shortconv_k.weight",
            &attention.key_convolution,
            Convolution,
        ),
        (
            "attn.v_sconv.weight",
            "shortconv_v.weight",
            &attention.value_convolution,
            Convolution,
        ),
        (
            "attn_sconv.weight",
            "shortconv_attn.weight",
            &parameters.attention_convolution,
            Convolution,
        ),
        (
            "mlp_sconv.weight",
            "shortconv_mlp.weight",
            &parameters.feed_forward_convolution,
            Convolution,
        ),
    ];
    for (safe, gguf, spec, layout) in slots {
        visit(safe, gguf, &declared_parameter(spec)?, layout)?;
    }
    Ok(())
}

struct SparseBankSource<'a> {
    gate_up: &'static str,
    down: &'static str,
    gguf: &'static str,
    bank: &'a eredu_nn::GroupedGatedProductSpec,
}
fn sparse_banks(parameters: &super::text::SparseParameters) -> [SparseBankSource<'_>; 2] {
    [
        SparseBankSource {
            gate_up: "mlp.experts.w13_weight",
            down: "mlp.experts.w2_weight",
            gguf: "exps",
            bank: &parameters.routed,
        },
        SparseBankSource {
            gate_up: "mlp.shared_experts.shared_w13_weight",
            down: "mlp.shared_experts.shared_w2_weight",
            gguf: "shexp",
            bank: &parameters.shared,
        },
    ]
}
fn router_aliases(
    parameters: &eredu_nn::JointGroupSelectorSpec,
    released: &str,
) -> [ParameterAlias; 3] {
    [
        alias(
            format!("{released}.mlp.gate.weight"),
            parameters.weight().id.as_str(),
        ),
        alias(
            format!("{released}.mlp.gate.bias"),
            parameters.correction_bias().id.as_str(),
        ),
        alias(
            format!("{released}.mlp.gate.global_scale"),
            parameters.coefficient_scale().id.as_str(),
        ),
    ]
}
fn sparse_down_aliases(
    parameters: &super::text::SparseParameters,
    released: &str,
) -> Result<Vec<ParameterAlias>, String> {
    sparse_banks(parameters)
        .into_iter()
        .map(|source| {
            let declared = source
                .bank
                .packed_parameters()
                .map_err(|error| error.to_string())?;
            Ok(alias(
                format!("{released}.{}", source.down),
                declared.down().parameter.id.as_str(),
            ))
        })
        .collect()
}

fn dense_aliases(parameters: &super::text::DenseParameters, released: &str) -> [ParameterAlias; 2] {
    [
        alias(
            format!("{released}.mlp.w2_md.weight"),
            parameters.product.down.weight.id.as_str(),
        ),
        alias(
            format!("{released}.mlp.global_scale"),
            parameters.global_scale.id.as_str(),
        ),
    ]
}

struct DenseInputSources<'a> {
    gate: eredu_nn::LogicalParameterDeclaration<'a>,
    up: eredu_nn::LogicalParameterDeclaration<'a>,
    interleaved: String,
    shape: Vec<usize>,
}

fn dense_input_sources<'a>(
    parameters: &'a super::text::DenseParameters,
    released: &str,
) -> Result<DenseInputSources<'a>, String> {
    let gate = declared_parameter(&parameters.product.gate)?;
    let up = declared_parameter(&parameters.product.up)?;
    let shape = dense_interleaved_shape(&gate, &up)?;
    Ok(DenseInputSources {
        gate,
        up,
        interleaved: format!("{released}.mlp.w13_dn.weight"),
        shape,
    })
}

fn add_dense_layout(
    parameters: &super::text::DenseParameters,
    released: &str,
    canonical: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let DenseInputSources {
        gate,
        up,
        interleaved: packed,
        shape,
    } = dense_input_sources(parameters, released)?;
    let down = declared_parameter(&parameters.product.down)?;
    let scale = parameters
        .scale_declaration()
        .map_err(|error| error.to_string())?;
    groups.push(AlternativeLayoutGroup {
        id: format!("{canonical} dense gate/up storage"),
        required: true,
        variants: vec![
            LayoutVariant {
                id: "canonical split".into(),
                tensors: vec![
                    safe(gate.parameter.id.as_str(), gate.shape.clone()),
                    safe(up.parameter.id.as_str(), up.shape.clone()),
                ],
                discriminator_keys: vec![
                    gate.parameter.id.to_string(),
                    up.parameter.id.to_string(),
                ],
            },
            LayoutVariant {
                id: "released interleaved".into(),
                tensors: vec![
                    safe(&packed, shape).with_linear_format(eredu_checkpoint::LinearFormat::Dense)
                ],
                discriminator_keys: vec![packed],
            },
        ],
    });
    for (alias, parameter) in dense_aliases(parameters, released)
        .into_iter()
        .zip([down, scale])
    {
        common.push(safe_alias(alias.source, alias.target, parameter.shape));
    }
    Ok(())
}

fn add_feed_forward_layout(
    parameters: &super::text::FeedForwardParameters,
    released: &str,
    canonical: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    match parameters {
        super::text::FeedForwardParameters::Dense(parameters) => {
            add_dense_layout(&parameters, released, canonical, common, groups)?;
        }
        super::text::FeedForwardParameters::Sparse(parameters) => {
            use eredu_nn::DeclareLogicalParameters;
            let router = parameters
                .router
                .logical_parameters()
                .map_err(|error| error.to_string())?;
            for alias in router_aliases(&parameters.router, released) {
                let parameter = router
                    .iter()
                    .find(|parameter| parameter.parameter.id.as_str() == alias.target)
                    .ok_or_else(|| {
                        format!(
                            "Inkling source alias references undeclared slot {}",
                            alias.target
                        )
                    })?;
                common.push(safe_alias(
                    alias.source,
                    alias.target,
                    parameter.shape.clone(),
                ));
            }
            for source in sparse_banks(&parameters) {
                let declared = source
                    .bank
                    .packed_parameters()
                    .map_err(|error| error.to_string())?;
                for (suffix, parameter) in [
                    (source.gate_up, declared.gate_up()),
                    (source.down, declared.down()),
                ] {
                    common.push(safe_alias(
                        format!("{released}.{suffix}"),
                        parameter.parameter.id.as_str(),
                        parameter.shape.clone(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn declared_parameter(
    spec: &dyn eredu_nn::DeclareLogicalParameters,
) -> Result<eredu_nn::LogicalParameterDeclaration<'_>, String> {
    let mut parameters = spec
        .logical_parameters()
        .map_err(|error| error.to_string())?;
    if parameters.len() != 1 {
        return Err(
            "Inkling source tensor extension requires exactly one declared parameter".into(),
        );
    }
    Ok(parameters.remove(0))
}

/// Format extensions reference the same slots used by construction and placement.
fn visit_media_parameters(
    args: &ModelArgs,
    mut visit: impl FnMut(
        &str,
        &str,
        &eredu_nn::LogicalParameterDeclaration<'_>,
        TensorOperation,
    ) -> Result<(), String>,
) -> Result<(), String> {
    let mut add =
        |safe: &str, gguf: &str, spec: &dyn eredu_nn::DeclareLogicalParameters, operation| {
            visit(safe, gguf, &declared_parameter(spec)?, operation)
        };
    if let Some(config) = &args.audio_config {
        let spec = config.parameters();
        add(
            "model.audio.encoder.weight",
            "a.dmel.embedding.weight",
            &spec.embedding,
            TensorOperation::Matrix,
        )?;
        add(
            "model.audio.final_norm.weight",
            "a.dmel.final_norm.weight",
            &spec.final_norm,
            TensorOperation::Vector,
        )?;
    }
    if let Some(config) = &args.vision_config {
        for layer in 0..config.num_hidden_layers as usize {
            let spec = config
                .parameters()
                .layer(layer)
                .map_err(|error| error.to_string())?;
            add(
                &format!("model.visual.layers.linear_{layer}.weight"),
                &format!("v.hmlp.{layer}.linear.weight"),
                &spec.projection,
                TensorOperation::Matrix,
            )?;
            if let Some(norm) = &spec.norm {
                add(
                    &format!("model.visual.layers.norm_{layer}.weight"),
                    &format!("v.hmlp.{layer}.norm.weight"),
                    norm,
                    TensorOperation::Vector,
                )?;
            }
        }
        add(
            "model.visual.final_norm.weight",
            "v.hmlp.final_norm.weight",
            &config.parameters().final_norm,
            TensorOperation::Vector,
        )?;
    }
    Ok(())
}

fn add_safetensors_media(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    visit_media_parameters(args, |source, _, parameter, _| {
        output.push(safe_alias(
            source,
            parameter.parameter.id.as_str(),
            parameter.shape.clone(),
        ));
        Ok(())
    })
}

fn add_safetensors_mtp(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    use eredu_nn::DeclareLogicalParameters;
    let parameters = args.prediction();
    for (index, depth) in parameters.depths.iter().enumerate() {
        for parameter in depth
            .fusion
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            output.push(safe(parameter.parameter.id.as_str(), parameter.shape));
        }
        let root = format!("model.mtp.layers.{index}.transformer_block");
        visit_block_parameters(&depth.common, |source, _, parameter, _| {
            output.push(safe_alias(
                format!("{root}.{source}"),
                parameter.parameter.id.as_str(),
                parameter.shape.clone(),
            ));
            Ok(())
        })?;
        add_dense_layout(&depth.dense, &root, &root, output, groups)?;
    }
    if let Some(norm) = &parameters.chain_norm {
        let parameter = declared_parameter(norm)?;
        output.push(safe(parameter.parameter.id.as_str(), parameter.shape));
    }
    Ok(())
}

/// Returns the complete direct-alias catalog for released SafeTensors names.
/// Interleaved dense/expert `w13` sources are intentionally represented by
/// recipe helpers instead of aliases.
pub fn safetensors_aliases(args: &ModelArgs) -> Result<Vec<ParameterAlias>, String> {
    let mut aliases = Vec::new();
    visit_text_pinned_parameters(args, |source, _, parameter, _| {
        aliases.push(alias(source, parameter.parameter.id.as_str()));
        Ok(())
    })?;
    for layer in 0..args.text_config.num_hidden_layers as usize {
        let released = format!("model.llm.layers.{layer}");
        let policy = args
            .text_config
            .layer_policy(layer)
            .ok_or_else(|| format!("Inkling checkpoint layer {layer} is out of range"))?;
        let parameters = args.common(layer).map_err(|error| error.to_string())?;
        aliases.extend(block_aliases(&released, &parameters)?);
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let dense = args.dense(layer).map_err(|error| error.to_string())?;
                aliases.extend(dense_aliases(&dense, &released));
            }
            FeedForwardPolicy::SparseMoe => {
                let parameters = args.sparse(layer).map_err(|error| error.to_string())?;
                aliases.extend(router_aliases(&parameters.router, &released));
                aliases.extend(sparse_down_aliases(&parameters, &released)?);
            }
        }
    }
    visit_media_parameters(args, |source, _, parameter, _| {
        aliases.push(alias(source, parameter.parameter.id.as_str()));
        Ok(())
    })?;
    let prediction = args.prediction();
    for (depth, parameters) in prediction.depths.iter().enumerate() {
        use eredu_nn::DeclareLogicalParameters;
        for parameter in parameters
            .fusion
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            let id = parameter.parameter.id.as_str();
            aliases.push(alias(id, id));
        }
        let root = format!("model.mtp.layers.{depth}.transformer_block");
        aliases.extend(block_aliases(&root, &parameters.common)?);
        aliases.extend(dense_aliases(&parameters.dense, &root));
    }
    if let Some(norm) = &prediction.chain_norm {
        let parameter = declared_parameter(norm)?;
        let id = parameter.parameter.id.as_str();
        aliases.push(alias(id, id));
    }
    Ok(aliases)
}

/// Resolves the complete released SafeTensors layout into canonical
/// architecture parameter recipes.
///
/// The catalog covers direct aliases, two-dimensional released convolution
/// kernels, interleaved dense projections, routed/shared fused experts, and
/// embedded MTP blocks. Architecture selectors partition these outputs into
/// complete static and execution-unit groups before backend binding.
pub fn safetensors_recipes<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let mut recipes = BTreeMap::new();
    for alias in safetensors_aliases(args)? {
        if has_tensor(catalog, &alias.source) {
            recipes.insert(
                alias.target,
                DerivedWeightRecipe::source(alias.source, TensorSelection::Full),
            );
        }
    }

    for layer in 0..nonnegative_count(args.text_config.num_hidden_layers, "layer count")? {
        let released = format!("model.llm.layers.{layer}");
        let parameters = args.common(layer).map_err(|error| error.to_string())?;
        add_convolution_recipes(catalog, &released, &parameters, &mut recipes)?;
        match args
            .text_config
            .layer_policy(layer)
            .ok_or_else(|| format!("Inkling checkpoint layer {layer} is out of range"))?
            .feed_forward
        {
            FeedForwardPolicy::Dense => {
                let dense = args.dense(layer).map_err(|error| error.to_string())?;
                add_dense_recipes(catalog, &released, &dense, &mut recipes)?;
            }
            FeedForwardPolicy::SparseMoe => {
                let parameters = args.sparse(layer).map_err(|error| error.to_string())?;
                add_expert_recipes(catalog, &released, &parameters, &mut recipes)?;
            }
        }
    }

    let prediction = args.prediction();
    for (depth, parameters) in prediction.depths.iter().enumerate() {
        let root = format!("model.mtp.layers.{depth}.transformer_block");
        add_convolution_recipes(catalog, &root, &parameters.common, &mut recipes)?;
        add_dense_recipes(catalog, &root, &parameters.dense, &mut recipes)?;
    }
    Ok(recipes)
}

/// Selects the complete architecture-owned SafeTensors recipe group for pinned modules.
///
/// The returned recipes are not intersected with a backend module. Consumers must bind the
/// complete group and reject any recipe that remains unconsumed.
pub fn static_safetensors_recipes<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    select_safetensors_recipe_group(args, catalog, None)
}

/// Selects exactly the architecture-owned recipes for the embedded MTP extension.
///
/// Prediction adapters materialize this module separately from the ordinary
/// target architecture. Returning the exact recipe subset keeps released
/// convolution and fused-linear layouts available without requiring the
/// backend to infer checkpoint prefixes or silently ignore unrelated static
/// recipes.
pub fn mtp_safetensors_recipes<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    Ok(safetensors_recipes(args, catalog)?
        .into_iter()
        .filter(|(target, _)| target.starts_with("model.mtp."))
        .collect())
}

/// Selects the complete architecture-owned SafeTensors recipe group for one execution unit.
///
/// `group` and `index` use the canonical Inkling execution layout: vision is group zero, audio is
/// group one, and the text decoder is group two. The returned group is deliberately complete so a
/// backend binding mismatch fails on leftover recipes instead of being silently filtered.
pub fn unit_safetensors_recipes<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
    group: usize,
    index: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let available = match group {
        0 => args
            .vision_config
            .as_ref()
            .map_or(0, |vision| vision.num_hidden_layers as usize),
        1 => 0,
        2 => nonnegative_count(args.text_config.num_hidden_layers, "layer count")?,
        _ => {
            return Err(format!(
                "Inkling recipe group {group} is outside three groups"
            ))
        }
    };
    if index >= available {
        return Err(format!(
            "Inkling recipe unit {index} is outside group {group} with {available} units"
        ));
    }
    select_safetensors_recipe_group(args, catalog, Some((group, index)))
}

fn select_safetensors_recipe_group<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
    selected: Option<(usize, usize)>,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let vision_layers = args
        .vision_config
        .as_ref()
        .map_or(0, |vision| vision.num_hidden_layers as usize);
    let text_layers = nonnegative_count(args.text_config.num_hidden_layers, "layer count")?;
    let mut selected_recipes = BTreeMap::new();
    for (target, recipe) in safetensors_recipes(args, catalog)? {
        let owner = if let Some(index) = indexed_recipe_target(&target, "visual.layers.")? {
            if index >= vision_layers {
                return Err(format!(
                    "Inkling recipe {target:?} names vision unit {index}, but only {vision_layers} exist"
                ));
            }
            Some((0, index))
        } else if let Some(index) = indexed_recipe_target(&target, "model.layers.")? {
            if index >= text_layers {
                return Err(format!(
                    "Inkling recipe {target:?} names text unit {index}, but only {text_layers} exist"
                ));
            }
            Some((2, index))
        } else {
            None
        };
        if owner == selected {
            selected_recipes.insert(target, recipe);
        }
    }
    Ok(selected_recipes)
}

fn indexed_recipe_target(target: &str, root: &str) -> Result<Option<usize>, String> {
    let Some(rest) = target.strip_prefix(root) else {
        return Ok(None);
    };
    let Some((index, parameter)) = rest.split_once('.') else {
        return Err(format!(
            "architecture recipe target {target:?} has no parameter below its execution unit"
        ));
    };
    if parameter.is_empty() {
        return Err(format!(
            "architecture recipe target {target:?} has an empty unit parameter"
        ));
    }
    index
        .parse::<usize>()
        .map(Some)
        .map_err(|_| format!("architecture recipe target {target:?} has a non-numeric unit index"))
}

/// Builds the complete architecture-owned schedule for independent expert residency.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let recipes = safetensors_recipes(args, catalog)?;
    let owner_group = eredu_runtime::ExecutionGroupId::new(super::TEXT_EXECUTION_GROUP)
        .map_err(|error| error.to_string())?;
    let mut units = Vec::new();
    for (layer, parameters) in args.sparse_layers() {
        let unit_path = format!("model.layers.{layer}");
        let groups =
            super::layer_parameter_groups(args, layer).map_err(|error| error.to_string())?;
        for (bank, spec, cache_layer, distribution) in [
            (
                "experts",
                &parameters.routed,
                layer,
                crate::ExpertResidencyDistribution::ExpertParallel,
            ),
            (
                "shared_experts",
                &parameters.shared,
                super::text::shared_expert_cache_layer(&args.text_config, layer)
                    .map_err(|error| error.to_string())?,
                crate::ExpertResidencyDistribution::Replicated,
            ),
        ] {
            let declared = spec
                .packed_parameters()
                .map_err(|error| error.to_string())?;
            let count = declared.group_count();
            let gate_up = declared.gate_up();
            let down = declared.down();
            let gate_up_format = gate_up
                .linear_format
                .ok_or("undeclared Inkling gate/up encoding")?;
            let down_format = down
                .linear_format
                .ok_or("undeclared Inkling down encoding")?;
            let bank_root = format!("{unit_path}.moe.{bank}");
            let logical_group = format!("{bank_root}.intermediate");
            let group = groups
                .iter()
                .find(|group| {
                    matches!(
                        group.role(),
                        eredu_runtime::ParameterRole::ExpertIntermediate
                            | eredu_runtime::ParameterRole::SharedExpertIntermediate
                    ) && group.logical_name() == logical_group
                })
                .ok_or_else(|| {
                    format!(
                        "Inkling sparse layer {layer} is missing expert group {logical_group:?}"
                    )
                })?;
            if count == 0 {
                continue;
            }
            let bank = eredu_runtime::ParameterBankRecipes::new(
                catalog,
                group
                    .members()
                    .iter()
                    .map(|member| {
                        let target = member.target();
                        (
                            target.to_owned(),
                            recipes.get(target).cloned().unwrap_or_else(|| {
                                DerivedWeightRecipe::source(target, TensorSelection::Full)
                            }),
                        )
                    })
                    .collect(),
                count,
            )
            .map_err(|error| error.to_string())?;
            let gate_up_quantizable =
                gate_up_format.scale().is_none() && gate_up_format.affine_bias().is_none();
            let down_quantizable =
                down_format.scale().is_none() && down_format.affine_bias().is_none();
            for expert in 0..count {
                let selected = bank.member(expert).expect("validated expert bank member");
                let parameters = group
                    .members()
                    .iter()
                    .map(|member| {
                        let target = member.target();
                        let binding = target
                            .strip_prefix(&bank_root)
                            .and_then(|suffix| suffix.strip_prefix('.'))
                            .ok_or_else(|| {
                                format!(
                                    "Inkling expert target {target:?} is outside bank {bank_root:?}"
                                )
                            })?;
                        let recipe = selected
                            .get(target)
                            .expect("declared bank parameter")
                            .clone();
                        let role = if (target == gate_up.parameter.id.as_str()
                            && gate_up_quantizable)
                            || (target == down.parameter.id.as_str() && down_quantizable)
                        {
                            crate::ExpertParameterRole::quantizable_projection(
                                format!("{binding}_scales"),
                                format!("{binding}_biases"),
                            )
                        } else {
                            crate::ExpertParameterRole::Preserved
                        };
                        crate::ExpertParameterRecipe::new(binding, target, recipe, role)
                            .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                units.push(
                    crate::ExpertResidencyUnit::new(
                        eredu_runtime::ParameterBankKey::new(0, cache_layer, expert),
                        owner_group.clone(),
                        layer,
                        &unit_path,
                        distribution,
                        parameters,
                    )
                    .map_err(|error| error.to_string())?,
                );
            }
        }
    }
    crate::ExpertResidencyCatalog::new(units)
        .and_then(|residency| residency.with_inferred_byte_geometry(catalog))
        .map_err(|error| error.to_string())
}

fn add_convolution_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    released: &str,
    parameters: &super::text::CommonParameters,
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
) -> Result<(), String> {
    visit_block_parameters(parameters, |source_suffix, _, parameter, layout| {
        if !matches!(layout, BlockTensorLayout::Convolution) {
            return Ok(());
        }
        let released_source = format!("{released}.{source_suffix}");
        let target = parameter.parameter.id.to_string();
        let source = if has_tensor(catalog, &released_source) {
            released_source
        } else if has_tensor(catalog, &target) {
            target.clone()
        } else {
            return Ok(());
        };
        let metadata = catalog
            .tensor_metadata(&source)
            .map_err(|error| error.to_string())?;
        let input = DerivedWeightRecipe::source(&source, TensorSelection::Full);
        let recipe = match metadata.logical_shape.as_slice() {
            [channels, kernel] if parameter.shape == [*channels, 1, *kernel] => {
                DerivedWeightRecipe::Reshape {
                    input: Box::new(input),
                    shape: parameter.shape.clone(),
                }
            }
            shape if shape == parameter.shape => input,
            shape => {
                return Err(format!(
                    "Inkling convolution {source} has shape {shape:?}, expected {:?} or its released two-dimensional layout",
                    parameter.shape
                ));
            }
        };
        recipes.insert(target, recipe);
        Ok(())
    })
}

fn add_dense_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    released: &str,
    parameters: &super::text::DenseParameters,
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
) -> Result<(), String> {
    let DenseInputSources {
        gate,
        up,
        interleaved: source,
        shape,
    } = dense_input_sources(parameters, released)?;
    if !has_tensor(catalog, &source) {
        return Ok(());
    }
    let [gate_recipe, up_recipe] = interleaved_recipes(catalog, &source, &shape, 0)?;
    recipes.insert(gate.parameter.id.to_string(), gate_recipe);
    recipes.insert(up.parameter.id.to_string(), up_recipe);
    Ok(())
}

fn add_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    released: &str,
    parameters: &super::text::SparseParameters,
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
) -> Result<(), String> {
    for source in sparse_banks(parameters) {
        let declared = source
            .bank
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        let gate_up = declared.gate_up();
        let bank_root = gate_up
            .parameter
            .id
            .as_str()
            .strip_suffix(".gate_up_proj")
            .ok_or_else(|| "Inkling source bank has an invalid primary identity".to_string())?;
        let source = format!("{released}.{}", source.gate_up);
        let target = gate_up.parameter.id.to_string();
        if has_tensor(catalog, &source) {
            let inputs = interleaved_recipes(catalog, &source, &gate_up.shape, 1)?;
            recipes.insert(
                target,
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: inputs.into(),
                },
            );
            continue;
        }
        let gate = format!("{bank_root}.gate_proj");
        let up = format!("{bank_root}.up_proj");
        if has_tensor(catalog, &gate) && has_tensor(catalog, &up) {
            recipes.insert(
                target,
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(gate, TensorSelection::Full),
                        DerivedWeightRecipe::source(up, TensorSelection::Full),
                    ],
                },
            );
        }
    }
    Ok(())
}

fn has_tensor<C: RecipeCatalog + ?Sized>(catalog: &C, name: &str) -> bool {
    catalog.tensor_metadata(name).is_ok()
}

fn block_aliases(
    released: &str,
    parameters: &super::text::CommonParameters,
) -> Result<Vec<ParameterAlias>, String> {
    let mut aliases = Vec::new();
    visit_block_parameters(parameters, |source, _, parameter, _| {
        aliases.push(alias(
            format!("{released}.{source}"),
            parameter.parameter.id.as_str(),
        ));
        Ok(())
    })?;
    Ok(aliases)
}

fn alias(source: impl Into<String>, target: impl Into<String>) -> ParameterAlias {
    ParameterAlias {
        source: source.into(),
        target: target.into(),
    }
}

/// Projects the released interleaved matrix from both declared dense projections.
fn dense_interleaved_shape(
    gate: &eredu_nn::LogicalParameterDeclaration<'_>,
    up: &eredu_nn::LogicalParameterDeclaration<'_>,
) -> Result<Vec<usize>, String> {
    if gate.shape.len() != 2 || gate.shape != up.shape {
        return Err("Inkling dense gate/up declarations have incompatible geometry".into());
    }
    Ok(vec![
        checked_add(gate.shape[0], up.shape[0], "dense interleaved width")?,
        gate.shape[1],
    ])
}

/// Selects parity rows from an exact declared source layout without reading payloads.
fn interleaved_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    source: &str,
    shape: &[usize],
    axis: usize,
) -> Result<[DerivedWeightRecipe; 2], String> {
    let metadata = catalog
        .tensor_metadata(source)
        .map_err(|error| error.to_string())?;
    if metadata.logical_shape != shape {
        return Err(format!(
            "Inkling interleaved tensor {source} has shape {:?}, expected {shape:?}",
            metadata.logical_shape
        ));
    }
    let rows = shape
        .get(axis)
        .copied()
        .filter(|rows| *rows > 0 && rows % 2 == 0)
        .ok_or_else(|| {
            format!("Inkling interleaved tensor {source} has invalid declared row axis {axis}")
        })?;
    Ok([0, 1].map(|parity| {
        DerivedWeightRecipe::source(
            source,
            TensorSelection::Indices {
                axis,
                indices: (parity..rows).step_by(2).collect(),
            },
        )
    }))
}

/// Translates one Inkling text GGUF tensor into its neutral parameter identity.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("token_embd_norm", "model.embed_norm"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_owned();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_owned();
    };
    for (source, target) in [
        ("ffn_gate_exps", "moe.experts.gate_proj"),
        ("ffn_up_exps", "moe.experts.up_proj"),
        ("ffn_down_exps", "moe.experts.down_proj"),
        ("ffn_gate_shexp", "moe.shared_experts.gate_proj"),
        ("ffn_up_shexp", "moe.shared_experts.up_proj"),
        ("ffn_down_shexp", "moe.shared_experts.down_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let suffix = match parameter.strip_prefix(source).unwrap_or_default() {
                ".weight" => "",
                ".scales" => "_scales",
                ".biases" => "_biases",
                other => other,
            };
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    for (source, target) in [
        ("attn_norm", "input_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_r", "self_attn.r_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("shortconv_k", "self_attn.k_sconv"),
        ("shortconv_v", "self_attn.v_sconv"),
        ("shortconv_attn", "attn_sconv"),
        ("shortconv_mlp", "mlp_sconv"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "dense.gate_proj"),
        ("ffn_up", "dense.up_proj"),
        ("ffn_down", "dense.down_proj"),
        ("ffn_gate_inp", "moe.router.weight"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut translated = parameter.replacen(source, target, 1);
            if source == "ffn_gate_inp" && translated.ends_with(".weight") {
                translated.truncate(translated.len() - ".weight".len());
            }
            return format!("model.layers.{layer}.{translated}");
        }
    }
    if parameter == "attn_rel_proj.weight" || parameter == "attn_rel_proj" {
        return format!("model.layers.{layer}.self_attn.rel_proj");
    }
    if parameter == "ffn_gscale" || parameter == "ffn_gscale.weight" {
        return format!("model.layers.{layer}.dense_global_scale");
    }
    if matches!(
        parameter,
        "exp_probs_b.bias" | "ffn_exp_probs_b.bias" | "ffn_exp_probs_b"
    ) {
        return format!("model.layers.{layer}.moe.router.bias");
    }
    name.to_owned()
}

/// Translates an Inkling GGUF tensor using the layer schedule to disambiguate
/// the shared `ffn_gscale` spelling used by dense and sparse blocks.
pub fn translate_gguf_weight_name_for_model(name: &str, args: &ModelArgs) -> String {
    let translated = translate_gguf_weight_name(name);
    let Some(rest) = translated.strip_prefix("model.layers.") else {
        return translated;
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return translated;
    };
    if parameter != "dense_global_scale" {
        return translated;
    }
    let Ok(layer) = layer.parse::<usize>() else {
        return translated;
    };
    if args
        .text_config
        .layer_policy(layer)
        .is_some_and(|policy| policy.feed_forward == FeedForwardPolicy::SparseMoe)
    {
        format!("model.layers.{layer}.moe.router.global_scale")
    } else {
        translated
    }
}

/// Translates one sibling projector GGUF tensor into its neutral identity.
pub fn translate_mmproj_weight_name(name: &str) -> String {
    for (source, target) in [
        ("a.dmel.embedding", "audio.encoder"),
        ("a.dmel.final_norm", "audio.final_norm"),
        ("v.hmlp.final_norm", "visual.final_norm"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    if let Some(rest) = name.strip_prefix("v.hmlp.") {
        if let Some((layer, parameter)) = rest.split_once('.') {
            if layer.parse::<usize>().is_ok() {
                let parameter =
                    parameter
                        .replacen("linear", "projection", 1)
                        .replacen("norm", "layer_norm", 1);
                return format!("visual.layers.{layer}.{parameter}");
            }
        }
    }
    name.to_owned()
}

/// Partitions translated sibling-projector weight formats by the component
/// that owns each canonical parameter.
pub(super) fn partition_mmproj_weight_formats<T>(
    formats: HashMap<String, T>,
) -> Result<(HashMap<String, T>, HashMap<String, T>), String> {
    let mut audio = HashMap::new();
    let mut vision = HashMap::new();
    for (name, format) in formats {
        if matches!(
            name.as_str(),
            "audio.encoder.weight" | "audio.final_norm.weight"
        ) {
            audio.insert(name, format);
        } else if name == "visual.final_norm.weight" || is_visual_layer_parameter(&name) {
            vision.insert(name, format);
        } else {
            return Err(format!(
                "Inkling mmproj format map contains unknown canonical parameter {name:?}"
            ));
        }
    }
    Ok((audio, vision))
}

fn is_visual_layer_parameter(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("visual.layers.") else {
        return false;
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return false;
    };
    layer.parse::<usize>().is_ok() && matches!(parameter, "projection.weight" | "layer_norm.weight")
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn safe(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

fn safe_alias(
    released: impl Into<String>,
    canonical: impl Into<String>,
    shape: Vec<usize>,
) -> SafetensorsTensorConstraint {
    let released = released.into();
    let canonical = canonical.into();
    let tensor = safe(&released, shape);
    if released == canonical {
        tensor
    } else {
        tensor.with_aliases([canonical])
    }
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Inkling {name} must be positive, got {value}"))
}

fn nonnegative_count(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("Inkling {name} cannot be negative, got {value}"))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("Inkling {name} geometry overflows"))
}
