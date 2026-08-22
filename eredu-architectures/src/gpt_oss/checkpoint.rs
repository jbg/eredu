//! Pure checkpoint schemas, name translation, and expert normalization for GPT-OSS.

use eredu_checkpoint::{
    expert::{
        canonical_gated_expert_projection_family_recipes, ExpertProjectionFamilyNames,
        GatedExpertProjectionFamilyNames, GatedExpertProjectionFamilyRecipes,
    },
    recipe::RecipeCatalog,
    schema::{
        matrix_for_linear_format, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
        GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint, TensorOperation,
    },
    validation::{validate_gguf_plan, validate_matching_gguf_encodings, CheckpointValidation},
    LinearFormat, StoredDtype,
};
use eredu_gguf::Checkpoint as GgufCheckpoint;

use super::config::ModelArgs;

const MXFP4_GROUP_SIZE: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 16;

/// Builds the exact published GPT-OSS SafeTensors catalog.
pub fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let geometry = Geometry::new(args)?;
    let root = &args.parameter_root;
    let mut tensors = Vec::new();
    add_matrix(
        args,
        &mut tensors,
        format!("{root}.embed_tokens.weight"),
        vec![geometry.vocabulary, geometry.hidden],
        true,
    )?;
    tensors.push(floating(
        format!("{root}.norm.weight"),
        vec![geometry.hidden],
    ));
    add_matrix(
        args,
        &mut tensors,
        "lm_head.weight",
        vec![geometry.vocabulary, geometry.hidden],
        true,
    )?;

    for layer in 0..geometry.layers {
        let block = format!("{root}.layers.{layer}");
        tensors.extend([
            floating(
                format!("{block}.input_layernorm.weight"),
                vec![geometry.hidden],
            ),
            floating(
                format!("{block}.post_attention_layernorm.weight"),
                vec![geometry.hidden],
            ),
            floating(
                format!("{block}.self_attn.sinks"),
                vec![geometry.query_heads],
            ),
        ]);
        for (projection, shape) in [
            ("q_proj", vec![geometry.query, geometry.hidden]),
            ("k_proj", vec![geometry.key_value, geometry.hidden]),
            ("v_proj", vec![geometry.key_value, geometry.hidden]),
            ("o_proj", vec![geometry.hidden, geometry.query]),
        ] {
            add_matrix(
                args,
                &mut tensors,
                format!("{block}.self_attn.{projection}.weight"),
                shape,
                true,
            )?;
        }
        for (projection, width) in [
            ("q_proj", geometry.query),
            ("k_proj", geometry.key_value),
            ("v_proj", geometry.key_value),
            ("o_proj", geometry.hidden),
        ] {
            tensors.push(floating(
                format!("{block}.self_attn.{projection}.bias"),
                vec![width],
            ));
        }
        // Published routers are ordinary floating-point projections. They are
        // deliberately excluded from model-wide load-time matrix formats.
        add_matrix(
            args,
            &mut tensors,
            format!("{block}.mlp.router.weight"),
            vec![geometry.experts, geometry.hidden],
            false,
        )?;
        tensors.push(floating(
            format!("{block}.mlp.router.bias"),
            vec![geometry.experts],
        ));
        tensors.extend(safetensors_expert_constraints(
            &format!("{block}.mlp.experts"),
            geometry,
        )?);
    }

    SafetensorsCheckpointPlan::new(
        "GPT-OSS SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Returns the exact published SafeTensors storage owned by routed experts.
///
/// This keeps physical native-MXFP4 names and companions in the architecture
/// checkpoint contract instead of requiring backends to classify path text.
pub fn safetensors_expert_tensors(
    args: &ModelArgs,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let geometry = Geometry::new(args)?;
    let mut tensors = Vec::new();
    for layer in 0..geometry.layers {
        tensors.extend(safetensors_expert_constraints(
            &format!("{}.layers.{layer}.mlp.experts", args.parameter_root),
            geometry,
        )?);
    }
    Ok(tensors)
}

/// Builds the exact canonical llama.cpp GPT-OSS GGUF catalog.
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let geometry = Geometry::new(args)?;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![geometry.vocabulary, geometry.hidden],
            TensorOperation::Matrix,
        ),
        gguf(
            "output_norm.weight",
            vec![geometry.hidden],
            TensorOperation::Vector,
        ),
        gguf(
            "output.weight",
            vec![geometry.vocabulary, geometry.hidden],
            TensorOperation::Matrix,
        ),
    ];
    for layer in 0..geometry.layers {
        let block = format!("blk.{layer}");
        tensors.extend([
            gguf(
                format!("{block}.attn_norm.weight"),
                vec![geometry.hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.attn_post_norm.weight"),
                vec![geometry.hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.attn_sinks.weight"),
                vec![geometry.query_heads],
                TensorOperation::Vector,
            )
            .with_aliases([format!("{block}.attn_sinks")]),
            gguf(
                format!("{block}.attn_q.weight"),
                vec![geometry.query, geometry.hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{block}.attn_q.bias"),
                vec![geometry.query],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.attn_k.weight"),
                vec![geometry.key_value, geometry.hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{block}.attn_k.bias"),
                vec![geometry.key_value],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.attn_v.weight"),
                vec![geometry.key_value, geometry.hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{block}.attn_v.bias"),
                vec![geometry.key_value],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.attn_output.weight"),
                vec![geometry.hidden, geometry.query],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{block}.attn_output.bias"),
                vec![geometry.hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.ffn_gate_inp.weight"),
                vec![geometry.experts, geometry.hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{block}.ffn_gate_inp.bias"),
                vec![geometry.experts],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.ffn_gate_exps.weight"),
                vec![geometry.experts, geometry.intermediate, geometry.hidden],
                TensorOperation::MxFp4Matrix,
            ),
            gguf(
                format!("{block}.ffn_gate_exps.bias"),
                vec![geometry.experts, geometry.intermediate],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.ffn_up_exps.weight"),
                vec![geometry.experts, geometry.intermediate, geometry.hidden],
                TensorOperation::MxFp4Matrix,
            ),
            gguf(
                format!("{block}.ffn_up_exps.bias"),
                vec![geometry.experts, geometry.intermediate],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.ffn_down_exps.weight"),
                vec![geometry.experts, geometry.hidden, geometry.intermediate],
                TensorOperation::MxFp4Matrix,
            ),
            gguf(
                format!("{block}.ffn_down_exps.bias"),
                vec![geometry.experts, geometry.hidden],
                TensorOperation::Vector,
            ),
        ]);
    }
    GgufCheckpointPlan::new("GPT-OSS GGUF", tensors, Vec::new(), CatalogPolicy::strict())
        .map_err(|error| error.to_string())
}

/// Returns translated GGUF weight targets whose storage belongs to routed experts.
///
/// These are source-layout targets, before the architecture's expert recipes
/// normalize separate gate/up/down matrices into the runtime expert bank.
pub fn gguf_expert_quantization_targets(args: &ModelArgs) -> Result<Vec<String>, String> {
    let geometry = Geometry::new(args)?;
    Ok((0..geometry.layers)
        .flat_map(|layer| {
            ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].map(move |projection| {
                translate_gguf_weight_name(&format!("blk.{layer}.{projection}.weight"))
            })
        })
        .collect())
}

/// Validates the portable GGUF catalog and matching gate/up physical encodings.
pub fn validate_gguf(checkpoint: &GgufCheckpoint, args: &ModelArgs) -> CheckpointValidation {
    let plan = match gguf_plan(args) {
        Ok(plan) => plan,
        Err(error) => {
            return CheckpointValidation::Invalid(vec![
                eredu_checkpoint::validation::CheckpointIssue {
                    kind: eredu_checkpoint::validation::CheckpointIssueKind::InvalidGeometry,
                    detail: error,
                    tensor_name: None,
                    tensor_type_code: None,
                    metadata_key: None,
                },
            ])
        }
    };
    let mut issues = match validate_gguf_plan(checkpoint, &plan) {
        CheckpointValidation::Exact => Vec::new(),
        CheckpointValidation::Invalid(issues) => issues,
        CheckpointValidation::Unverified(issue) => vec![issue],
    };
    issues.extend(validate_matching_gguf_encodings(
        checkpoint,
        (0..args.num_hidden_layers.max(0) as usize).map(|layer| {
            (
                format!("blk.{layer}.ffn_gate_exps.weight"),
                format!("blk.{layer}.ffn_up_exps.weight"),
            )
        }),
        "GPT-OSS",
    ));
    CheckpointValidation::from_issues(issues)
}

/// Translates a physical llama.cpp tensor name to its neutral parameter identity.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ] {
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
    if matches!(parameter, "attn_sinks" | "attn_sinks.weight") {
        return format!("model.layers.{layer}.self_attn.sinks");
    }
    for (source, target) in [
        ("attn_norm", "input_layernorm"),
        ("attn_post_norm", "post_attention_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("ffn_gate_inp", "mlp.router"),
        ("ffn_gate_exps", "mlp.experts.gate_proj"),
        ("ffn_up_exps", "mlp.experts.up_proj"),
        ("ffn_down_exps", "mlp.experts.down_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}

/// Resolves SafeTensors alternating rows or translated GGUF separate matrices
/// into one atomic component-major native-MXFP4 expert family.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    layer: usize,
) -> Result<GatedExpertProjectionFamilyRecipes, String> {
    let layers = dimension(args.num_hidden_layers, "layer count")?;
    if layer >= layers {
        return Err(format!(
            "GPT-OSS expert recipe layer {layer} is outside {layers} layers"
        ));
    }
    let root = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
    canonical_gated_expert_projection_family_recipes(catalog, &expert_names(&root))
        .map_err(|error| error.to_string())
}

fn expert_names(root: &str) -> GatedExpertProjectionFamilyNames {
    let projection = |weight: String, scales: String, bias: String| ExpertProjectionFamilyNames {
        weight,
        scales,
        bias,
    };
    GatedExpertProjectionFamilyNames {
        target_gate_up: projection(
            format!("{root}.gate_up_proj"),
            format!("{root}.gate_up_proj_scales"),
            format!("{root}.gate_up_proj_bias"),
        ),
        target_down: projection(
            format!("{root}.down_proj"),
            format!("{root}.down_proj_scales"),
            format!("{root}.down_proj_bias"),
        ),
        alternating_gate_up: projection(
            format!("{root}.gate_up_proj_blocks"),
            format!("{root}.gate_up_proj_scales"),
            format!("{root}.gate_up_proj_bias"),
        ),
        alternating_down: projection(
            format!("{root}.down_proj_blocks"),
            format!("{root}.down_proj_scales"),
            format!("{root}.down_proj_bias"),
        ),
        separate_gate: projection(
            format!("{root}.gate_proj.weight"),
            format!("{root}.gate_proj.scales"),
            format!("{root}.gate_proj.bias"),
        ),
        separate_up: projection(
            format!("{root}.up_proj.weight"),
            format!("{root}.up_proj.scales"),
            format!("{root}.up_proj.bias"),
        ),
        separate_down: projection(
            format!("{root}.down_proj.weight"),
            format!("{root}.down_proj.scales"),
            format!("{root}.down_proj.bias"),
        ),
    }
}

fn safetensors_expert_constraints(
    root: &str,
    geometry: Geometry,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let fused = checked_mul(2, geometry.intermediate, "fused expert width")?;
    Ok(vec![
        SafetensorsTensorConstraint::required(
            format!("{root}.gate_up_proj_blocks"),
            vec![
                geometry.experts,
                fused,
                geometry.hidden / MXFP4_GROUP_SIZE,
                MXFP4_BLOCK_BYTES,
            ],
            StoredDtypeConstraint::Exact(StoredDtype::U8),
        ),
        SafetensorsTensorConstraint::required(
            format!("{root}.gate_up_proj_scales"),
            vec![geometry.experts, fused, geometry.hidden / MXFP4_GROUP_SIZE],
            StoredDtypeConstraint::Exact(StoredDtype::U8),
        )
        .companion(),
        floating(
            format!("{root}.gate_up_proj_bias"),
            vec![geometry.experts, fused],
        )
        .companion(),
        SafetensorsTensorConstraint::required(
            format!("{root}.down_proj_blocks"),
            vec![
                geometry.experts,
                geometry.hidden,
                geometry.intermediate / MXFP4_GROUP_SIZE,
                MXFP4_BLOCK_BYTES,
            ],
            StoredDtypeConstraint::Exact(StoredDtype::U8),
        ),
        SafetensorsTensorConstraint::required(
            format!("{root}.down_proj_scales"),
            vec![
                geometry.experts,
                geometry.hidden,
                geometry.intermediate / MXFP4_GROUP_SIZE,
            ],
            StoredDtypeConstraint::Exact(StoredDtype::U8),
        )
        .companion(),
        floating(
            format!("{root}.down_proj_bias"),
            vec![geometry.experts, geometry.hidden],
        )
        .companion(),
    ])
}

fn add_matrix(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: impl Into<String>,
    shape: Vec<usize>,
    permit_configured_format: bool,
) -> Result<(), String> {
    let name = name.into();
    let format = permit_configured_format
        .then(|| args.weight_quantization_for(&name))
        .flatten();
    output.extend(
        matrix_for_linear_format(
            &name,
            Vec::<String>::new(),
            shape,
            LinearFormat::from(format),
            None,
        )
        .map_err(|error| error.to_string())?,
    );
    Ok(())
}

fn floating(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

#[derive(Clone, Copy)]
struct Geometry {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    query_heads: usize,
    query: usize,
    key_value: usize,
    vocabulary: usize,
    experts: usize,
}

impl Geometry {
    fn new(args: &ModelArgs) -> Result<Self, String> {
        let hidden = dimension(args.hidden_size, "hidden size")?;
        let intermediate = dimension(args.intermediate_size, "expert intermediate size")?;
        if !hidden.is_multiple_of(MXFP4_GROUP_SIZE)
            || !intermediate.is_multiple_of(MXFP4_GROUP_SIZE)
        {
            return Err(format!(
                "GPT-OSS MXFP4 dimensions must be divisible by {MXFP4_GROUP_SIZE}, got hidden size {hidden} and intermediate size {intermediate}"
            ));
        }
        let query_heads = dimension(args.num_attention_heads, "attention head count")?;
        let key_value_heads = dimension(args.num_key_value_heads, "key/value head count")?;
        if !query_heads.is_multiple_of(key_value_heads) {
            return Err(format!(
                "GPT-OSS query head count {query_heads} is not divisible by key/value head count {key_value_heads}"
            ));
        }
        let head = dimension(args.head_dim, "attention head width")?;
        Ok(Self {
            hidden,
            intermediate,
            layers: dimension(args.num_hidden_layers, "layer count")?,
            query_heads,
            query: checked_mul(query_heads, head, "query projection width")?,
            key_value: checked_mul(key_value_heads, head, "key/value projection width")?,
            vocabulary: dimension(args.vocab_size, "vocabulary size")?,
            experts: dimension(args.num_local_experts, "expert count")?,
        })
    }
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("GPT-OSS {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("GPT-OSS {name} geometry overflows"))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use eredu_checkpoint::{
        expert::GatedProductExpertStorageLayout,
        recipe::RecipeCatalog,
        schema::TensorRole,
        store::{StoreError, TensorMetadata},
        AffineQuantization,
    };

    use super::*;

    struct Catalog(BTreeMap<String, TensorMetadata>);

    impl RecipeCatalog for Catalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }

    fn metadata(name: &str, shape: Vec<usize>, dtype: StoredDtype) -> TensorMetadata {
        let bytes = shape.iter().product::<usize>()
            * match dtype {
                StoredDtype::U8 => 1,
                StoredDtype::U32 | StoredDtype::F32 => 4,
                _ => 2,
            };
        TensorMetadata {
            name: name.into(),
            encoded_byte_len: bytes as u64,
            logical_shape: shape.clone(),
            physical_shape: shape,
            stored_dtype: dtype,
            backing_shard: None,
        }
    }

    fn args() -> ModelArgs {
        super::super::config::model_args_from_config_value(&serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 32,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 1e-5,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            "rope_theta": 150000.0,
            "rope_scaling": null,
            "layer_types": ["sliding_attention"],
            "quantization_config": {"quant_method": "mxfp4"},
            "swiglu_limit": 7.0
        }))
        .unwrap()
    }

    #[test]
    fn freezes_strict_safetensors_native_companions_and_parameter_ids() {
        let plan = safetensors_plan(&args()).unwrap();
        assert!(plan.catalog_policy.strict);
        assert!(plan.catalog_policy.explicitly_allowed_keys.is_empty());
        assert!(plan.catalog_policy.allowed_prefixes.is_empty());
        let root = "model.layers.0.mlp.experts";
        let expected = [
            (format!("{root}.gate_up_proj_blocks"), vec![2, 64, 1, 16]),
            (format!("{root}.gate_up_proj_scales"), vec![2, 64, 1]),
            (format!("{root}.gate_up_proj_bias"), vec![2, 64]),
            (format!("{root}.down_proj_blocks"), vec![2, 32, 1, 16]),
            (format!("{root}.down_proj_scales"), vec![2, 32, 1]),
            (format!("{root}.down_proj_bias"), vec![2, 32]),
        ];
        for (key, shape) in expected {
            let tensor = plan
                .common_tensors
                .iter()
                .find(|tensor| tensor.key == key)
                .unwrap();
            assert_eq!(tensor.shape, shape);
        }
        assert_eq!(
            plan.common_tensors
                .iter()
                .find(|tensor| tensor.key.ends_with("gate_up_proj_scales"))
                .unwrap()
                .role,
            TensorRole::Companion
        );
    }

    #[test]
    fn freezes_gguf_native_types_sink_alias_and_translation() {
        let plan = gguf_plan(&args()).unwrap();
        assert!(plan.catalog_policy.strict);
        let sink = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "blk.0.attn_sinks.weight")
            .unwrap();
        assert_eq!(sink.aliases, ["blk.0.attn_sinks"]);
        assert!(plan.common_tensors.iter().any(|tensor| {
            tensor.key == "blk.0.ffn_gate_exps.weight"
                && tensor.encoding
                    == GgufTypeConstraint::OperationClass(TensorOperation::MxFp4Matrix)
        }));
        assert_eq!(
            translate_gguf_weight_name("blk.0.ffn_gate_exps.scales"),
            "model.layers.0.mlp.experts.gate_proj.scales"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.0.attn_sinks"),
            "model.layers.0.self_attn.sinks"
        );
    }

    #[test]
    fn mixed_dense_formats_are_per_parameter_and_never_reencode_native_experts_or_router() {
        let mut model_args = args();
        model_args.quantization = Some(eredu_checkpoint::WeightQuantization::MxFp4);
        model_args.quantized_weight_configs = Some(HashMap::from([(
            "model.layers.0.self_attn.k_proj.weight".into(),
            eredu_checkpoint::WeightQuantization::Affine(AffineQuantization::new(32, 8).unwrap()),
        )]));
        let plan = safetensors_plan(&model_args).unwrap();
        let tensor = |key: &str| {
            plan.common_tensors
                .iter()
                .find(|tensor| tensor.key == key)
                .unwrap()
        };
        assert_eq!(
            tensor("model.layers.0.self_attn.q_proj.weight").shape,
            [32, 4]
        );
        assert_eq!(
            tensor("model.layers.0.self_attn.k_proj.weight").shape,
            [32, 8]
        );
        assert_eq!(
            tensor("model.layers.0.mlp.router.weight").dtype,
            StoredDtypeConstraint::Floating
        );
        assert_eq!(
            tensor("model.layers.0.mlp.experts.gate_up_proj_blocks").dtype,
            StoredDtypeConstraint::Exact(StoredDtype::U8)
        );
    }

    #[test]
    fn alternating_safetensors_family_normalizes_atomically() {
        let root = "model.layers.0.mlp.experts";
        let values = [
            (
                format!("{root}.gate_up_proj_blocks"),
                vec![2, 64, 1, 16],
                StoredDtype::U8,
            ),
            (
                format!("{root}.gate_up_proj_scales"),
                vec![2, 64, 1],
                StoredDtype::U8,
            ),
            (
                format!("{root}.gate_up_proj_bias"),
                vec![2, 64],
                StoredDtype::F32,
            ),
            (
                format!("{root}.down_proj_blocks"),
                vec![2, 32, 1, 16],
                StoredDtype::U8,
            ),
            (
                format!("{root}.down_proj_scales"),
                vec![2, 32, 1],
                StoredDtype::U8,
            ),
            (
                format!("{root}.down_proj_bias"),
                vec![2, 32],
                StoredDtype::F32,
            ),
        ];
        let catalog = Catalog(
            values
                .into_iter()
                .map(|(name, shape, dtype)| {
                    let value = metadata(&name, shape, dtype);
                    (name, value)
                })
                .collect(),
        );
        let recipes = expert_recipes(&catalog, &args(), 0).unwrap();
        assert_eq!(recipes.layout, GatedProductExpertStorageLayout::Packed);
        assert_eq!(recipes.outputs.iter().count(), 6);
        assert_eq!(
            recipes
                .get(&format!("{root}.gate_up_proj"))
                .unwrap()
                .infer(&catalog)
                .unwrap()
                .shape,
            [2, 64, 4]
        );
    }

    #[test]
    fn translated_gguf_separate_family_normalizes_to_same_targets() {
        let root = "model.layers.0.mlp.experts";
        let values = [
            (
                format!("{root}.gate_proj.weight"),
                vec![2, 32, 4],
                StoredDtype::U32,
            ),
            (
                format!("{root}.gate_proj.scales"),
                vec![2, 32, 1],
                StoredDtype::U8,
            ),
            (
                format!("{root}.gate_proj.bias"),
                vec![2, 32],
                StoredDtype::F32,
            ),
            (
                format!("{root}.up_proj.weight"),
                vec![2, 32, 4],
                StoredDtype::U32,
            ),
            (
                format!("{root}.up_proj.scales"),
                vec![2, 32, 1],
                StoredDtype::U8,
            ),
            (
                format!("{root}.up_proj.bias"),
                vec![2, 32],
                StoredDtype::F32,
            ),
            (
                format!("{root}.down_proj.weight"),
                vec![2, 32, 4],
                StoredDtype::U32,
            ),
            (
                format!("{root}.down_proj.scales"),
                vec![2, 32, 1],
                StoredDtype::U8,
            ),
            (
                format!("{root}.down_proj.bias"),
                vec![2, 32],
                StoredDtype::F32,
            ),
        ];
        let catalog = Catalog(
            values
                .into_iter()
                .map(|(name, shape, dtype)| {
                    let value = metadata(&name, shape, dtype);
                    (name, value)
                })
                .collect(),
        );
        let recipes = expert_recipes(&catalog, &args(), 0).unwrap();
        assert_eq!(
            recipes.layout,
            GatedProductExpertStorageLayout::SeparatePacked
        );
        assert_eq!(recipes.outputs.iter().count(), 6);
        assert_eq!(
            recipes
                .get(&format!("{root}.gate_up_proj"))
                .unwrap()
                .infer(&catalog)
                .unwrap()
                .shape,
            [2, 64, 4]
        );
        assert_eq!(
            recipes
                .get(&format!("{root}.down_proj"))
                .unwrap()
                .infer(&catalog)
                .unwrap()
                .shape,
            [2, 32, 4]
        );
    }

    #[test]
    fn malformed_companion_and_target_collision_fail_before_outputs_escape() {
        let root = "model.layers.0.mlp.experts";
        let mut values = BTreeMap::from([
            (
                format!("{root}.gate_up_proj_blocks"),
                metadata("gate_up", vec![2, 64, 1, 16], StoredDtype::U8),
            ),
            (
                format!("{root}.gate_up_proj_scales"),
                metadata("gate_up_scales", vec![2, 63, 1], StoredDtype::U8),
            ),
            (
                format!("{root}.gate_up_proj_bias"),
                metadata("gate_up_bias", vec![2, 64], StoredDtype::F32),
            ),
            (
                format!("{root}.down_proj_blocks"),
                metadata("down", vec![2, 32, 1, 16], StoredDtype::U8),
            ),
            (
                format!("{root}.down_proj_scales"),
                metadata("down_scales", vec![2, 32, 1], StoredDtype::U8),
            ),
            (
                format!("{root}.down_proj_bias"),
                metadata("down_bias", vec![2, 32], StoredDtype::F32),
            ),
        ]);
        assert!(expert_recipes(&Catalog(values.clone()), &args(), 0).is_err());
        values
            .get_mut(&format!("{root}.gate_up_proj_scales"))
            .unwrap()
            .logical_shape = vec![2, 64, 1];
        values
            .get_mut(&format!("{root}.gate_up_proj_scales"))
            .unwrap()
            .physical_shape = vec![2, 64, 1];
        let catalog = Catalog(values);
        let mut names = expert_names(root);
        names.target_down.weight = names.target_gate_up.weight.clone();
        assert!(matches!(
            canonical_gated_expert_projection_family_recipes(&catalog, &names),
            Err(eredu_checkpoint::expert::ExpertLayoutError::InvalidRecipe(
                eredu_checkpoint::recipe::RecipeError::DuplicateOutput { .. }
            ))
        ));
    }

    #[test]
    fn malformed_native_geometry_and_layer_are_rejected() {
        let mut malformed = args();
        malformed.hidden_size = 33;
        assert!(safetensors_plan(&malformed)
            .unwrap_err()
            .contains("divisible by 32"));
        assert!(gguf_plan(&malformed)
            .unwrap_err()
            .contains("divisible by 32"));
        assert!(expert_recipes(&Catalog(BTreeMap::new()), &args(), 1)
            .unwrap_err()
            .contains("outside"));
    }
}
