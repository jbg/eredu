//! Pure SafeTensors and GGUF checkpoint plans for LFM3.

use std::collections::{BTreeMap, HashMap};

use eredu_checkpoint::schema::{
    CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
    SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::WeightQuantization;
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};

use super::config::ModelArgs;
use super::{
    declarations::{FeedForwardParameters, MixerParameters},
    Declaration,
};

/// Validates the bounded positional companion before retaining GGUF execution.
/// The published global table keeps the high-frequency pairs and suppresses
/// the remaining pairs with a large divisor.
pub(crate) fn validate_gguf_rope(
    checkpoint: &eredu_gguf::Checkpoint,
    args: &ModelArgs,
) -> Result<(), String> {
    let width = usize::try_from(args.hidden_size / args.num_attention_heads / 2)
        .map_err(|e| e.to_string())?;
    let tensor = checkpoint
        .tensors()
        .find(|t| t.descriptor().name == "rope_freqs.weight")
        .ok_or("LFM3 GGUF requires rope_freqs.weight")?;
    if tensor.descriptor().row_major_shape() != [width as u64]
        || tensor.descriptor().ggml_type != eredu_gguf::GgmlType::F32
        || width > 4096
    {
        return Err("LFM3 GGUF rotary companion has invalid geometry or encoding".into());
    }
    if !checkpoint.has_payload_access() {
        return Ok(());
    }
    let payload = checkpoint
        .materializer()
        .raw_tensor("rope_freqs.weight")
        .map_err(|e| e.to_string())?;
    if payload.data().len() != width * 4 {
        return Err("LFM3 GGUF rotary companion has an invalid byte length".into());
    }
    let active = (width as f32 * args.rope.partial_rotary_factor) as usize;
    for (index, bytes) in payload.data().chunks_exact(4).enumerate() {
        let bytes = <[u8; 4]>::try_from(bytes).map_err(|e| e.to_string())?;
        let factor = match payload.endian() {
            eredu_gguf::Endian::Little => f32::from_le_bytes(bytes),
            eredu_gguf::Endian::Big => f32::from_be_bytes(bytes),
        };
        let expected = if index < active { 1.0 } else { 1e30 };
        if factor != expected {
            return Err(format!(
                "LFM3 GGUF rotary companion pair {index} has unsupported factor {factor}"
            ));
        }
    }
    Ok(())
}

/// Derives an LFM3 configuration whose physical matrix formats reflect
/// load-time quantization instead of checkpoint-specific format selections.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.weight_quantization = Some(quantization);
    target.quantized_weights = None;
    target.quantized_weight_configs = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint format metadata to a complete LFM3
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let mut target = args.clone();
    target.quantized_weights = Some(formats.keys().cloned().collect());
    target.quantized_weight_configs = Some(formats);
    target.weight_quantization = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Returns the complete derived-weight catalog for one LFM3 execution unit.
pub fn unit_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &Declaration,
    layer: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let scope = declaration
        .blocks
        .get(layer)
        .ok_or_else(|| format!("LFM3 has no declared layer {layer}"))?;
    let mut recipes = BTreeMap::new();
    if let MixerParameters::Convolution(spec) = &scope.mixer {
        let conv = spec.convolution.weight.id.as_str();
        if let Ok(metadata) = store.source_metadata(conv) {
            let expected = spec
                .convolution
                .logical_parameters()
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|d| d.parameter.id == spec.convolution.weight.id)
                .expect("declared convolution weight")
                .shape;
            let recipe = if metadata.logical_shape.len() == 2 {
                Some(DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::source(conv, TensorSelection::Full)),
                    shape: expected.clone(),
                })
            } else if metadata.logical_shape == [expected[0], expected[2], expected[1]]
                && metadata.logical_shape != expected
            {
                Some(DerivedWeightRecipe::Transpose {
                    input: Box::new(DerivedWeightRecipe::source(conv, TensorSelection::Full)),
                    axes: vec![0, 2, 1],
                })
            } else {
                None
            };
            if let Some(recipe) = recipe {
                recipes.insert(conv.to_owned(), recipe);
            }
        }
    }
    Ok(recipes)
}

/// Keeps Q4_0 blocks in their original byte layout instead of expanding
/// them into three affine outputs on every GGUF page-in.
pub(crate) fn packed_gguf_catalog(
    checkpoint: eredu_gguf::Checkpoint,
) -> Result<eredu_gguf::Checkpoint, eredu_gguf::Error> {
    let q4 = checkpoint
        .tensors()
        .filter(|tensor| tensor.descriptor().ggml_type == eredu_gguf::GgmlType::Q4_0)
        .map(|tensor| tensor.descriptor().name.clone())
        .collect::<Vec<_>>();
    checkpoint.into_tensor_representation(q4, eredu_gguf::QuantizedTensorRepresentation::Encoded)
}

/// Selects the packed formats of an admitted GGUF language backbone and binds
/// its physical outputs to the resulting parameter declaration.
pub(super) fn gguf_language_bindings(
    checkpoint: eredu_gguf::Checkpoint,
    plan: &GgufCheckpointPlan,
    declaration: &Declaration,
    max_cached_readers: usize,
) -> Result<
    (
        std::sync::Arc<Declaration>,
        eredu_checkpoint::store::SharedCheckpointSource,
        BTreeMap<String, DerivedWeightRecipe>,
    ),
    eredu_nn::Error,
> {
    use eredu_nn::Error;

    let checkpoint = packed_gguf_catalog(checkpoint).map_err(Error::backend_source)?;
    let declaration = gguf_language_declaration(&checkpoint, declaration)?;
    let mapping = checkpoint
        .translated_outputs(translate_gguf_weight_name)
        .map_err(Error::backend_source)?;
    let source: eredu_checkpoint::store::SharedCheckpointSource = std::sync::Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(max_cached_readers)
            .and_then(|builder| builder.add_checkpoint(checkpoint, plan, &mapping))
            .and_then(|builder| builder.build())
            .map_err(Error::backend_source)?,
    );
    let recipes = gguf_language_recipes(source.as_ref(), &declaration)?;
    Ok((declaration, source, recipes))
}

pub(crate) fn gguf_language_declaration(
    checkpoint: &eredu_gguf::Checkpoint,
    declaration: &Declaration,
) -> Result<std::sync::Arc<Declaration>, eredu_nn::Error> {
    use eredu_nn::Error;

    let formats = checkpoint
        .shards()
        .iter()
        .flat_map(|shard| {
            shard.tensors().iter().map(move |tensor| {
                Ok((
                    translate_gguf_weight_name(&tensor.descriptor().name),
                    crate::linear_format::gguf_tensor_format(tensor, shard.endian())?
                        .weight_quantization(),
                ))
            })
        })
        .collect::<Result<HashMap<_, _>, String>>()
        .map_err(Error::backend)?;
    let formats = declaration
        .logical_parameters()?
        .into_iter()
        .filter(|parameter| parameter.linear_format.is_some())
        .filter_map(|parameter| {
            let name = parameter.parameter.id.as_str();
            formats
                .get(name)
                .copied()
                .flatten()
                .map(|format| (name.to_owned(), format))
        })
        .collect();
    let args = with_checkpoint_formats(declaration.args(), formats).map_err(Error::backend)?;
    Ok(std::sync::Arc::new(declaration.with_formats(args)?))
}

pub(crate) fn gguf_language_recipes(
    source: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &Declaration,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, eredu_nn::Error> {
    use eredu_nn::Error;

    let mut recipes = BTreeMap::new();
    for layer in 0..declaration.blocks.len() {
        recipes.extend(unit_recipes(source, declaration, layer).map_err(Error::backend)?);
    }
    for group in declaration.definition().parameters().groups() {
        for member in group.members() {
            let target = member.target();
            let recipe = recipes
                .entry(target.to_owned())
                .or_insert_with(|| DerivedWeightRecipe::source(target, TensorSelection::Full));
            if member.linear_companion() == Some(eredu_nn::LinearCompanionRole::GlobalScale) {
                *recipe = DerivedWeightRecipe::Reshape {
                    input: Box::new(recipe.clone()),
                    shape: Vec::new(),
                };
            }
            let output = recipe.infer(source).map_err(Error::backend_source)?;
            if output.shape() != member.global_shape() {
                return Err(Error::backend(format!(
                    "LFM3 GGUF binding {target} has shape {:?}, expected {:?}",
                    output.shape(),
                    member.global_shape(),
                )));
            }
        }
    }
    Ok(recipes)
}

/// Applies a floating storage policy only to dense neural parameters.
pub(super) fn language_storage_recipe(
    member: &eredu_runtime::ParameterMemberSpec,
    recipe: DerivedWeightRecipe,
    dtype: eredu_checkpoint::recipe::RecipeDtype,
) -> DerivedWeightRecipe {
    if member.linear_companion().is_some()
        || member
            .logical_linear_format()
            .is_some_and(|format| format.encoding() != eredu_checkpoint::LinearFormat::Dense)
    {
        recipe
    } else {
        DerivedWeightRecipe::Cast {
            input: Box::new(recipe),
            dtype,
        }
    }
}

/// Builds the complete alternative-layout SafeTensors contract.
pub fn safetensors_plan(
    declaration: &Declaration,
    _allow_derived_packed: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    let statics = &declaration.statics;
    let mut common = Vec::new();
    let groups = Vec::new();
    add_safe_spec(&mut common, statics.embedding_spec())?;
    add_safe_spec(&mut common, statics.normalization_spec())?;
    if let Some(head) = statics.head_spec() {
        add_safe_spec(&mut common, head)?;
    }
    for scope in &declaration.blocks {
        for spec in [&scope.operator_norm, &scope.feed_forward_norm]
            .into_iter()
            .chain(scope.post_operator_norm.iter())
            .chain(scope.post_ffn_norm.iter())
        {
            add_safe_spec(&mut common, spec)?;
        }
        match &scope.mixer {
            MixerParameters::Convolution(spec) => {
                let mut declarations = spec.logical_parameters().map_err(|e| e.to_string())?;
                declarations.sort_by_key(|d| convolution_storage_order(d.parameter.id.as_str()));
                for declaration in declarations {
                    let name = declaration.parameter.id.as_str();
                    if declaration.parameter.id == spec.convolution.weight.id {
                        let mut alternative = declaration.shape.clone();
                        alternative.swap(1, 2);
                        common.push(
                            safe(name, declaration.shape).with_alternate_shapes([alternative]),
                        );
                    } else if declaration.role == LogicalParameterRole::LinearWeight {
                        common.extend(source_matrix_constraints(
                            name,
                            declaration.shape,
                            declaration
                                .linear_format
                                .and_then(|format| format.encoding().weight_quantization()),
                        )?);
                    } else {
                        common.push(safe(name, declaration.shape));
                    }
                }
            }
            MixerParameters::Attention(spec, _) => {
                add_safe_spec(&mut common, spec)?;
            }
        }
        match &scope.feed_forward {
            FeedForwardParameters::Dense(spec) => {
                for spec in [&spec.gate, &spec.down, &spec.up] {
                    add_safe_spec(&mut common, spec)?;
                }
            }
        }
    }
    let mut policy = CatalogPolicy::non_strict();
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("LFM3 SafeTensors", common, groups, policy)
        .map_err(|e| e.to_string())
}

fn add_safe_spec(
    output: &mut Vec<SafetensorsTensorConstraint>,
    spec: &impl DeclareLogicalParameters,
) -> Result<(), String> {
    for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
        let name = declaration.parameter.id.as_str();
        if matches!(
            declaration.role,
            LogicalParameterRole::LinearWeight | LogicalParameterRole::Embedding
        ) {
            output.extend(source_matrix_constraints(
                name,
                declaration.shape,
                declaration
                    .linear_format
                    .and_then(|format| format.encoding().weight_quantization()),
            )?);
        } else {
            output.push(safe(name, declaration.shape));
        }
    }
    Ok(())
}

fn convolution_storage_order(name: &str) -> usize {
    [
        "conv.weight",
        "in_proj.weight",
        "out_proj.weight",
        "conv.bias",
        "in_proj.bias",
        "out_proj.bias",
    ]
    .iter()
    .position(|field| name.ends_with(&format!(".{field}")))
    .expect("declared short-convolution slot")
}

/// Projects admitted ordinary matrix geometry from neural declarations and exact
/// source-layout extensions, before physical packing.
pub(crate) fn source_linear_shapes(
    declaration: &Declaration,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let shapes = declaration
        .definition()
        .parameters()
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .filter(|member| member.logical_role() == Some(LogicalParameterRole::LinearWeight))
        .filter_map(|member| {
            member
                .logical_shape()
                .filter(|shape| shape.len() == 2)
                .map(|shape| (member.target().to_owned(), shape.to_vec()))
        })
        .collect::<BTreeMap<_, _>>();
    Ok(shapes)
}

fn source_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let Some(quantization) = quantization else {
        return Ok(vec![safe(name, shape)]);
    };
    eredu_checkpoint::schema::matrix_for_packed_words(
        name,
        Vec::<String>::new(),
        shape,
        quantization,
        None,
    )
    .map_err(|error| error.to_string())
}

fn safe(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

/// Builds the complete physical GGUF tensor contract.
pub fn gguf_plan(declaration: &Declaration) -> Result<GgufCheckpointPlan, String> {
    let statics = &declaration.statics;
    let mut tensors = Vec::new();
    let mut positional = GgufTensorConstraint::required(
        "rope_freqs.weight",
        vec![
            (declaration.args().hidden_size / declaration.args().num_attention_heads / 2) as usize,
        ],
        GgufTypeConstraint::Exact(eredu_gguf::GgmlType::F32),
    );
    positional.role = eredu_checkpoint::schema::TensorRole::ArtifactMetadata;
    tensors.push(positional);
    add_gguf_spec(&mut tensors, statics.embedding_spec())?;
    add_gguf_spec(&mut tensors, statics.normalization_spec())?;
    if let Some(head) = statics.head_spec() {
        add_gguf_spec(&mut tensors, head)?;
    }
    for scope in &declaration.blocks {
        for spec in [&scope.operator_norm, &scope.feed_forward_norm]
            .into_iter()
            .chain(scope.post_operator_norm.iter())
            .chain(scope.post_ffn_norm.iter())
        {
            add_gguf_spec(&mut tensors, spec)?;
        }
        match &scope.mixer {
            MixerParameters::Convolution(spec) => {
                let mut declarations = spec.logical_parameters().map_err(|e| e.to_string())?;
                declarations.sort_by_key(|d| convolution_storage_order(d.parameter.id.as_str()));
                for declaration in declarations {
                    let mut shape = declaration.shape;
                    let operation = if declaration.parameter.id == spec.convolution.weight.id {
                        shape.remove(1); // GGUF omits the singleton input-channel axis.
                        TensorOperation::Dense
                    } else if declaration.role == LogicalParameterRole::LinearWeight {
                        TensorOperation::Matrix
                    } else {
                        TensorOperation::Vector
                    };
                    tensors.push(gguf(
                        gguf_parameter_name(declaration.parameter.id.as_str())?,
                        shape,
                        operation,
                    ));
                }
            }
            MixerParameters::Attention(spec, _) => {
                add_gguf_spec(&mut tensors, spec)?;
            }
        }
        match &scope.feed_forward {
            FeedForwardParameters::Dense(spec) => {
                for spec in [&spec.gate, &spec.down, &spec.up] {
                    add_gguf_spec(&mut tensors, spec)?;
                }
            }
        }
    }
    GgufCheckpointPlan::new(
        "LFM3 GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|e| e.to_string())
}

fn add_gguf_spec(
    output: &mut Vec<GgufTensorConstraint>,
    spec: &impl DeclareLogicalParameters,
) -> Result<(), String> {
    for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
        let operation = if matches!(
            declaration.role,
            LogicalParameterRole::LinearWeight | LogicalParameterRole::Embedding
        ) {
            TensorOperation::Matrix
        } else {
            TensorOperation::Vector
        };
        output.push(gguf(
            gguf_parameter_name(declaration.parameter.id.as_str())?,
            declaration.shape,
            operation,
        ));
    }
    Ok(())
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

const GGUF_ROOTS: [(&str, &str); 3] = [
    ("token_embd", "model.embed_tokens"),
    ("output_norm", "model.embedding_norm"),
    ("output", "lm_head"),
];

const GGUF_PARAMETERS: [(&str, &str); 17] = [
    ("post_attention_norm", "post_operator_norm"),
    ("post_ffw_norm", "post_ffn_norm"),
    ("attn_gate", "self_attn.gate_proj"),
    ("shortconv.conv", "conv.conv"),
    ("shortconv.in_proj", "conv.in_proj"),
    ("shortconv.out_proj", "conv.out_proj"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_norm", "operator_norm"),
    ("ffn_norm", "ffn_norm"),
    ("ffn_gate", "feed_forward.w1"),
    ("ffn_down", "feed_forward.w2"),
    ("ffn_up", "feed_forward.w3"),
];

fn gguf_parameter_name(canonical: &str) -> Result<String, String> {
    for (source, target) in GGUF_ROOTS {
        if canonical == target || canonical.starts_with(&format!("{target}.")) {
            return Ok(canonical.replacen(target, source, 1));
        }
    }
    if let Some((layer, parameter)) = canonical
        .strip_prefix("model.layers.")
        .and_then(|s| s.split_once('.'))
    {
        for (source, target) in GGUF_PARAMETERS {
            if parameter == target || parameter.starts_with(&format!("{target}.")) {
                return Ok(format!(
                    "blk.{layer}.{}",
                    parameter.replacen(target, source, 1)
                ));
            }
        }
    }
    Err(format!("unresolved LFM3 GGUF parameter {canonical}"))
}

/// Translates one physical GGUF tensor name to its canonical parameter identity.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in GGUF_ROOTS {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_string();
    };

    for (source, target) in GGUF_PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}
