//! Reusable independent miniature Qwen4Exp artifacts for evaluation.
//!
//! Moved from the portable numerical conformance suite. These are synthetic
//! nonzero weights, not a released-checkpoint parity substitute. Export ordering,
//! normalization offsets, and recurrent negative-exponential values are authored
//! independently of architecture checkpoint recipes.
use eredu_architectures::qwen4_exp::{
    checkpoint::schema::SafetensorsEncoding,
    config::NGramSourceLayout,
    prepared::{GgufTargetPlan, PreparedTarget, SafetensorsTargetPlan},
    target::TargetLimits,
};
use eredu_gguf::{GgmlType, MetadataArray, MetadataValue as V, TensorInput, Writer};
use std::{
    collections::BTreeMap,
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};

/// One physical tensor in the independent miniature export.
pub struct TensorFixture {
    /// Published GGUF tensor name.
    pub name: String,
    /// Row-major logical shape (reversed when written to GGUF).
    pub shape: Vec<usize>,
    /// Encoded tensor bytes.
    pub data: Vec<u8>,
    /// GGUF scalar or block encoding.
    pub ty: GgmlType,
}
/// Independently exported nonzero miniature Qwen4Exp checkpoint.
///
/// This test fixture encodes published layout transformations directly, without
/// consulting production recipes. It contains recurrent and grouped-query indexed
/// attention, routed/shared experts, two residual streams, and exact integer n-grams.
pub struct Fixture {
    /// Mutable physical descriptors for malformed-artifact tests.
    pub tensors: Vec<TensorFixture>,
    /// Canonical dense values before published layout transformations.
    pub expected: BTreeMap<String, Vec<f32>>,
    /// Canonical logical shapes corresponding to `expected`.
    pub expected_shapes: BTreeMap<String, Vec<usize>>,
    recurrent_heads: (usize, usize),
}
impl Fixture {
    fn add(&mut self, physical: &str, canonical: &str, shape: &[usize], transform: u8) {
        let mut original: Vec<f32> = (0..shape.iter().product())
            .map(|i| ((i * 7 + 3) % 101) as f32 / 64. - 0.75)
            .collect();
        // Values remain small so exp/log parity tests have a fixed absolute tolerance.
        let mut exported = original.clone();
        if transform == 1 {
            for v in &mut exported {
                *v += 1.;
            }
        }
        if transform == 2 {
            for v in &mut exported {
                *v = -v.exp();
            }
        }
        let (axis, prefix, heads) =
            if physical.contains("attn_qkv") || physical.contains("ssm_conv1d") {
                (0, 2 * self.recurrent_heads.0 * 8, true)
            } else if physical.contains("ssm_out") {
                (1, 0, true)
            } else {
                (
                    0,
                    0,
                    ["attn_gate", "ssm_alpha", "ssm_beta", "ssm_dt", "ssm_a"]
                        .iter()
                        .any(|s| physical.contains(s)),
                )
            };
        if heads {
            let (key_heads, value_heads) = self.recurrent_heads;
            let repeats = value_heads / key_heads;
            // Forward exporter loop: each repeat emits one head from every key group.
            let stride: usize = shape[axis + 1..].iter().product();
            let outer: usize = shape[..axis].iter().product();
            let dim = (shape[axis] - prefix) / value_heads;
            let mut tiled = exported.clone();
            for o in 0..outer {
                for repeat in 0..repeats {
                    for key in 0..key_heads {
                        for d in 0..dim {
                            for x in 0..stride {
                                let source =
                                    (o * shape[axis] + prefix + (key * repeats + repeat) * dim + d)
                                        * stride
                                        + x;
                                let dest = (o * shape[axis]
                                    + prefix
                                    + (repeat * key_heads + key) * dim
                                    + d)
                                    * stride
                                    + x;
                                tiled[dest] = exported[source];
                            }
                        }
                    }
                }
            }
            exported = tiled;
        }
        if canonical.is_empty() {
            original.clear();
        } else {
            self.expected.insert(canonical.into(), original);
            let mut shape = shape.to_vec();
            if physical.contains("conv1d") {
                shape.insert(1, 1);
            }
            self.expected_shapes.insert(canonical.into(), shape);
        }
        self.tensors.push(TensorFixture {
            name: physical.into(),
            shape: shape.into(),
            ty: GgmlType::F32,
            data: exported.into_iter().flat_map(f32::to_le_bytes).collect(),
        });
    }
    /// Construct the deterministic, nonzero miniature export.
    pub fn new() -> Self {
        Self::with_recurrent_heads(2, 6)
    }

    /// Four-way tensor-parallel fixture retaining two attention K/V heads.
    /// Only the recurrent head geometry changes; all family mechanisms remain.
    pub fn tensor_parallel_four() -> Self {
        Self::with_recurrent_heads(4, 8)
    }

    fn with_recurrent_heads(key_heads: usize, value_heads: usize) -> Self {
        let mut f = Self {
            tensors: vec![],
            expected: BTreeMap::new(),
            expected_shapes: BTreeMap::new(),
            recurrent_heads: (key_heads, value_heads),
        };
        for (p, n, s) in [
            (
                "token_embd.weight",
                "model.embed_tokens.weight",
                vec![16, 32],
            ),
            ("output.weight", "lm_head.weight", vec![16, 32]),
            (
                "output_hc_norm.weight",
                "model.hyper_connection_mixer.hc_norm.weight",
                vec![64],
            ),
            (
                "output_hc_down.weight",
                "model.hyper_connection_mixer.input_mix_weight_down.weight",
                vec![4, 64],
            ),
            (
                "output_hc_up.weight",
                "model.hyper_connection_mixer.input_mix_weight_up.weight",
                vec![64, 4],
            ),
        ] {
            f.add(p, n, &s, if p.contains("norm") { 1 } else { 0 });
        }
        f.add("per_layer_token_embd.weight", "", &[12, 16], 0);
        for l in 0..2 {
            for (p, n) in [
                ("hc_attn", "attn_hyper_connection"),
                ("hc_ffn", "mlp_hyper_connection"),
            ] {
                for (s, t, shape) in [
                    ("norm", "hc_norm", vec![64]),
                    ("down", "input_mix_weight_down", vec![4, 64]),
                    ("up", "input_mix_weight_up", vec![64, 4]),
                    ("inject", "block_inject_weight", vec![2, 64]),
                ] {
                    f.add(
                        &format!("blk.{l}.{p}_{s}.weight"),
                        &format!("model.layers.{l}.{n}.{t}.weight"),
                        &shape,
                        if s == "norm" { 1 } else { 0 },
                    );
                }
            }
            for (p, n, s) in [
                ("ffn_gate_inp", "gate", vec![3, 32]),
                ("ffn_gate_inp_shexp", "shared_expert_gate", vec![1, 32]),
                ("ffn_gate_shexp", "shared_expert.gate_proj", vec![32, 32]),
                ("ffn_up_shexp", "shared_expert.up_proj", vec![32, 32]),
                ("ffn_down_shexp", "shared_expert.down_proj", vec![32, 32]),
            ] {
                f.add(
                    &format!("blk.{l}.{p}.weight"),
                    &format!("model.layers.{l}.mlp.{n}.weight"),
                    &s,
                    0,
                );
            }
            for (p, n) in [
                ("gate", "gate_proj"),
                ("up", "up_proj"),
                ("down", "down_proj"),
            ] {
                f.add(
                    &format!("blk.{l}.ffn_{p}_exps.weight"),
                    &format!("model.layers.{l}.mlp.experts.{n}"),
                    &[3, 32, 32],
                    0,
                );
            }
        }
        let value_width = value_heads * 128;
        let qkv_width = 2 * key_heads * 8 + value_width;
        for (p, n, s, t) in [
            (
                "attn_qkv.weight",
                "in_proj_qkv.weight",
                vec![qkv_width, 32],
                0,
            ),
            (
                "attn_gate.weight",
                "in_proj_z.weight",
                vec![value_width, 32],
                0,
            ),
            (
                "ssm_alpha.weight",
                "in_proj_a.weight",
                vec![value_heads, 32],
                0,
            ),
            (
                "ssm_beta.weight",
                "in_proj_b.weight",
                vec![value_heads, 32],
                0,
            ),
            (
                "ssm_out.weight",
                "out_proj.weight",
                vec![32, value_width],
                0,
            ),
            ("ssm_conv1d.weight", "conv1d.weight", vec![qkv_width, 3], 0),
            ("ssm_dt.bias", "dt_bias", vec![value_heads], 0),
            ("ssm_a", "A_log", vec![value_heads], 2),
            ("ssm_norm.weight", "norm.weight", vec![128], 0),
        ] {
            f.add(
                &format!("blk.0.{p}"),
                &format!("model.layers.0.linear_attn.{n}"),
                &s,
                t,
            );
        }
        for (p, n, s, t) in [
            ("attn_q", "q_proj", vec![64, 32], 0),
            ("attn_k", "k_proj", vec![16, 32], 0),
            ("attn_v", "v_proj", vec![16, 32], 0),
            ("attn_output", "o_proj", vec![32, 32], 0),
            ("attn_q_norm", "q_norm", vec![8], 1),
            ("attn_k_norm", "k_norm", vec![8], 1),
            ("indexer.q_proj", "indexer.index_q_proj", vec![16, 32], 0),
            ("indexer.k_proj", "indexer.index_k_proj", vec![8, 32], 0),
            ("indexer.q_norm", "indexer.q_layernorm", vec![8], 1),
            ("indexer.k_norm", "indexer.k_layernorm", vec![8], 1),
        ] {
            f.add(
                &format!("blk.1.{p}.weight"),
                &format!("model.layers.1.self_attn.{n}.weight"),
                &s,
                t,
            );
        }
        for (p, n, s, t) in [
            ("ple_key", "key_proj", vec![64, 32], 0),
            ("ple_value", "value_proj", vec![32, 32], 0),
            ("ple_norm_key", "norm_key", vec![64], 1),
            ("ple_norm_query", "norm_query", vec![64], 1),
            ("ple_norm_conv", "norm_conv", vec![64], 1),
            ("ple_conv1d", "conv1d", vec![64, 3], 0),
        ] {
            f.add(
                &format!("blk.0.{p}.weight"),
                &format!("model.layers.0.ple.{n}.weight"),
                &s,
                t,
            );
        }
        let q = f
            .expected
            .remove("model.layers.1.self_attn.indexer.index_q_proj.weight")
            .unwrap();
        let k = f
            .expected
            .remove("model.layers.1.self_attn.indexer.index_k_proj.weight")
            .unwrap();
        f.expected.insert(
            "model.layers.1.self_attn.indexer.index_qk_proj.weight".into(),
            [q, k].concat(),
        );
        f.expected_shapes
            .remove("model.layers.1.self_attn.indexer.index_q_proj.weight");
        f.expected_shapes
            .remove("model.layers.1.self_attn.indexer.index_k_proj.weight");
        f.expected_shapes.insert(
            "model.layers.1.self_attn.indexer.index_qk_proj.weight".into(),
            vec![24, 32],
        );
        f
    }
    /// Replace one physical tensor encoding for quantization conformance fixtures.
    ///
    /// Panics if the tensor is absent.
    pub fn replace_encoding(&mut self, name: &str, ty: GgmlType, data: Vec<u8>) {
        let tensor = self.tensors.iter_mut().find(|t| t.name == name).unwrap();
        tensor.ty = ty;
        tensor.data = data;
    }
    /// Write the fixture to GGUF, panicking on invalid fixture data or I/O failure.
    pub fn write(&self, path: &std::path::Path) {
        let mut metadata = metadata();
        for (key, value) in [
            ("ssm.inner_size", self.recurrent_heads.1 * 128),
            ("ssm.time_step_rank", self.recurrent_heads.1),
            ("ssm.group_count", self.recurrent_heads.0),
        ] {
            metadata.insert(format!("qwen4exp.{key}"), V::Uint32(value as u32));
        }
        self.write_metadata(path, &metadata);
    }
    /// Write with overridden metadata for admission tests.
    ///
    /// Panics on invalid fixture data or I/O failure.
    pub fn write_metadata(&self, path: &std::path::Path, metadata: &BTreeMap<String, V>) {
        let dimensions: Vec<Vec<u64>> = self
            .tensors
            .iter()
            .map(|t| t.shape.iter().rev().map(|&d| d as u64).collect())
            .collect();
        let inputs: Vec<_> = self
            .tensors
            .iter()
            .zip(&dimensions)
            .map(|(t, d)| TensorInput {
                name: &t.name,
                dimensions: d,
                ggml_type: t.ty,
                data: &t.data,
            })
            .collect();
        Writer::default()
            .write(std::fs::File::create(path).unwrap(), metadata, &inputs)
            .unwrap();
    }
}
/// Published metadata for the miniature independent export.
pub fn metadata() -> BTreeMap<String, V> {
    let mut m = BTreeMap::from([("general.architecture".into(), V::String("qwen4exp".into()))]);
    for (k, v) in [
        ("block_count", 2),
        ("context_length", 128),
        ("embedding_length", 32),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 8),
        ("rope.dimension_count", 6),
        ("ssm.conv_kernel", 3),
        ("ssm.inner_size", 768),
        ("ssm.state_size", 8),
        ("ssm.time_step_rank", 6),
        ("ssm.group_count", 2),
        ("hyper_connection.count", 2),
        ("hyper_connection.low_rank", 4),
        ("attention.indexer.head_count", 2),
        ("attention.indexer.key_length", 8),
        ("attention.indexer.top_k", 8),
        ("expert_count", 3),
        ("expert_used_count", 2),
        ("expert_feed_forward_length", 32),
        ("expert_shared_feed_forward_length", 32),
        ("ple.ngram_size", 3),
        ("ple.heads_per_ngram", 1),
        ("ple.conv_kernel", 3),
        ("ple.eos_token_id", 0),
    ] {
        m.insert(format!("qwen4exp.{k}"), V::Uint32(v));
    }
    for (k, v) in [
        ("rope.freq_base", 10000.),
        ("attention.layer_norm_rms_epsilon", 1e-6),
    ] {
        m.insert(format!("qwen4exp.{k}"), V::Float32(v));
    }
    m.insert(
        "qwen4exp.attention.recurrent_layers".into(),
        V::Array(MetadataArray::Bool(vec![true, false])),
    );
    for (k, v) in [
        ("attention.compress_ratios", vec![0, 2]),
        ("rope.dimension_sections", vec![1, 1, 1, 0]),
        ("ple.layers", vec![0]),
        ("ple.layer_multipliers", vec![1, 9007199254740993, 37]),
        ("ple.head_offsets", vec![0, 5]),
        ("ple.head_vocab_sizes", vec![5, 7]),
    ] {
        m.insert(format!("qwen4exp.{k}"), V::Array(MetadataArray::Int64(v)));
    }
    m
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}
/// Independent SafeTensors configuration matching the miniature export.
pub fn configuration() -> serde_json::Value {
    serde_json::from_str(
        r#"{
        "model_type": "qwen4_exp_text", "vocab_size": 16, "hidden_size": 32,
        "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "eos_token_id": 0,
        "hidden_act": "silu", "output_gate_type": "sigmoid", "tie_word_embeddings": false,
        "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
        "hc_count": 2, "hc_lowrank": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "attention_bias": false,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
            "partial_rotary_factor": 0.75, "mrope_section": [1, 1, 1], "mrope_interleaved": true},
        "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 8,
        "indexer_budget": 8, "indexer_compress_ratio": 2,
        "linear_num_key_heads": 2, "linear_num_value_heads": 6,
        "linear_key_head_dim": 8, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 3,
        "num_experts": 3, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32, "norm_topk_prob": true,
        "ple_layer_ids": [1], "ngram_size": 3, "heads_per_ngram": 1,
        "ngram_vocab_size_base": 5, "make_ngram_vocab_size_divisible_by": 1,
        "split_ngram_parts": 1, "seed": 1, "ple_embed_dim": 32, "ple_conv_kernel_size": 3
    }"#,
    )
    .expect("independent miniature configuration is valid JSON")
}

/// Changes only the miniature SafeTensors artifact's declared context length.
/// Call before inspection/preparation so retained source provenance stays valid.
/// Weight tensors and rotary equations are unchanged.
pub fn set_context_length(directory: &Path, tokens: usize) -> Result<(), Box<dyn Error>> {
    let path = directory.join("config.json");
    let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    config["max_position_embeddings"] = serde_json::json!(tokens);
    std::fs::write(path, serde_json::to_vec_pretty(&config)?)?;
    Ok(())
}

/// Adds one deterministic indexed prediction depth to a miniature SafeTensors artifact.
/// The prediction block copies the fixture's indexed target block; fusion weights
/// are authored independently and shared vocabulary parameters remain target-owned.
pub fn add_prediction_weights(directory: &Path) -> Result<(), Box<dyn Error>> {
    let path = directory.join("model.safetensors");
    let bytes = std::fs::read(&path)?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut values: Vec<_> = tensors
        .tensors()
        .into_iter()
        .map(|(name, tensor)| {
            (
                name,
                tensor.dtype(),
                tensor.shape().to_vec(),
                tensor.data().to_vec(),
            )
        })
        .collect();
    let prediction: Vec<_> = values
        .iter()
        .filter_map(|(name, dtype, shape, bytes)| {
            name.strip_prefix("model.layers.1.")
                .map(|suffix| format!("mtp.layers.0.{suffix}"))
                .or_else(|| {
                    name.strip_prefix("model.hyper_connection_mixer.")
                        .map(|suffix| format!("mtp.hyper_connection_mixer.{suffix}"))
                })
                .map(|name| (name, *dtype, shape.clone(), bytes.clone()))
        })
        .collect();
    values.extend(prediction);
    for (name, shape) in [
        ("mtp.pre_fc_norm_embedding.weight", vec![32]),
        ("mtp.pre_fc_norm_hidden.weight", vec![64]),
        ("mtp.fc_embedding.weight", vec![32, 32]),
        ("mtp.fc_hidden.weight", vec![32, 32]),
    ] {
        values.push((
            name.into(),
            safetensors::Dtype::F32,
            shape.clone(),
            (0..shape.iter().product::<usize>())
                .flat_map(|i| (((i * 7 + 3) % 101) as f32 / 256. - 0.125).to_le_bytes())
                .collect(),
        ));
    }
    safetensors::tensor::serialize_to_file(
        values.iter().map(|(name, dtype, shape, bytes)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &path,
    )?;
    let config_path = directory.join("config.json");
    let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    config["mtp_num_hidden_layers"] = serde_json::json!(1);
    config["mtp"] = serde_json::json!({"num_hidden_layers":1,"layer_types":["full_attention"],"rope_theta":7777.0});
    std::fs::write(config_path, serde_json::to_vec_pretty(&config)?)?;
    Ok(())
}

/// Explicit generic execution policy used by the miniature target fixtures.
pub fn bounded_policy() -> eredu_runtime::BoundedExecutionPolicy {
    use eredu_runtime::*;
    BoundedExecutionPolicy::new(
        InvocationLimits::new(2, 32, 128).unwrap(),
        TiledSelectionLimits::new(2, 16384, 1 << 20).unwrap(),
        RowLookupLoadPolicy::new(
            RowLookupLimits {
                requests: 128,
                rows_per_acquisition: 2,
                acquisition_bytes: 128,
                host_bytes: 1 << 16,
                output_bytes: 32768,
            },
            ParameterBankLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(4096), Some(0), 1).unwrap(),
                1 << 20,
                1 << 20,
            )
            .unwrap(),
            0,
        )
        .unwrap(),
        AppendStreamLoadPolicy::new(
            AppendStreamLimits {
                entries: 64,
                page_entries: 2,
                read_entries: 2,
            },
            65536,
            65536,
            65536,
        )
        .unwrap(),
    )
    .unwrap()
}

/// Architecture bounds derived from the fixture's normalized load request.
pub fn limits() -> TargetLimits {
    TargetLimits::from_load_request(
        &eredu_runtime::NormalizedLoadRequest::default().with_bounded_execution(bounded_policy()),
        eredu_nn::TensorElementType::F32,
    )
    .unwrap()
}

/// Matching source artifacts retained by an evaluation caller.
///
/// Keep `directory` alive for as long as either prepared source is used. The
/// sources retain their exact prepared provenance; evaluation construction does
/// not reopen these paths. The fixture has no tokenizer or prediction weights.
pub struct PreparedFixtures {
    /// GGUF prepared target, with published reorder/normalization recipes.
    pub gguf: PreparedTarget,
    /// Matching dense canonical SafeTensors target.
    pub safetensors: PreparedTarget,
    /// Source-free GGUF construction retained before ordinary source opening.
    pub gguf_header: GgufTargetPlan,
    /// Source-free SafeTensors construction retained before literal binding.
    pub safetensors_header: SafetensorsTargetPlan,
    /// Exact metadata-only physical declarations admitted with the SafeTensors headers.
    pub safetensors_physical: BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource>,
    /// GGUF file retained for provenance and artifact tests.
    pub gguf_path: PathBuf,
    /// Directory holding `model.safetensors`.
    pub safetensors_path: PathBuf,
}
impl PreparedFixtures {
    /// Write deterministic matching dense artifacts under an existing directory.
    ///
    /// The `safetensors` subdirectory must not already exist. The GGUF fixture
    /// writer panics on I/O failure, matching [`Fixture::write`].
    pub fn write(directory: &Path) -> Result<Self, Box<dyn Error>> {
        Fixture::new().prepare(directory)
    }
}
impl Fixture {
    /// Write this independent export and its matching canonical SafeTensors source.
    ///
    /// Encoding changes must also update `expected` to the independently decoded
    /// values. The `safetensors` subdirectory must not already exist. GGUF writing
    /// panics on invalid fixture bytes or I/O failure, as in [`Fixture::write`].
    pub fn prepare(&self, directory: &Path) -> Result<PreparedFixtures, Box<dyn Error>> {
        let f = self;
        let path = directory.join("target.gguf");
        f.write(&path);
        let header = GgufTargetPlan::prepare(&eredu_gguf::Checkpoint::open(&path)?)?;
        let plan = header.text_plan();
        let source = Arc::new(
            eredu_checkpoint::gguf_store::GgufWeightStore::builder()
                .add_resolved_checkpoint(
                    plan.checkpoint().clone(),
                    plan.resolution(),
                    plan.mapping(),
                )?
                .build()?,
        );
        let mut config = header.text_plan().config().clone();
        let gguf = header.clone().bind(source, limits())?;
        let mut values: Vec<_> = f
            .expected
            .iter()
            .map(|(n, v)| {
                (
                    n.clone(),
                    safetensors::Dtype::F32,
                    f.expected_shapes[n].clone(),
                    v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                )
            })
            .collect();
        // SafeTensors stores fused gate/up per expert; export fixture stores split banks.
        for layer in 0..2 {
            let root = format!("model.layers.{layer}.mlp.experts");
            let gate = &f.expected[&format!("{root}.gate_proj")];
            let up = &f.expected[&format!("{root}.up_proj")];
            values.retain(|(n, _, _, _)| {
                n != &format!("{root}.gate_proj") && n != &format!("{root}.up_proj")
            });
            let packed = gate
                .chunks_exact(1024)
                .zip(up.chunks_exact(1024))
                .flat_map(|(g, u)| g.iter().chain(u))
                .flat_map(|v| v.to_le_bytes())
                .collect();
            values.push((
                format!("{root}.gate_up_proj"),
                safetensors::Dtype::F32,
                vec![3, 64, 32],
                packed,
            ));
        }
        let root = "model.layers.0.ple.ple_embedding";
        values.push((
            format!("{root}.ngram_embedding.shard_0.weight"),
            safetensors::Dtype::F32,
            vec![12, 16],
            (0..192)
                .flat_map(|i| (((i * 7 + 3) % 101) as f32 / 64. - 0.75).to_le_bytes())
                .collect(),
        ));
        for (name, values_i64) in [
            ("layer_multipliers", vec![1i64, 9007199254740993, 37]),
            ("ngram_heads_vocab_sizes", vec![5, 7]),
            ("ngram_heads_offsets", vec![0, 5]),
        ] {
            values.push((
                format!("{root}.{name}"),
                safetensors::Dtype::I64,
                vec![values_i64.len()],
                values_i64.into_iter().flat_map(i64::to_le_bytes).collect(),
            ));
        }
        config.ngram.source = NGramSourceLayout::Safetensors {
            vocabulary_base: 5,
            vocabulary_alignment: 1,
            shards: 1,
            seed: 1,
        };
        let st_dir = directory.join("safetensors");
        std::fs::create_dir(&st_dir)?;
        let mut configuration = configuration();
        configuration["linear_num_key_heads"] = serde_json::json!(f.recurrent_heads.0);
        configuration["linear_num_value_heads"] = serde_json::json!(f.recurrent_heads.1);
        std::fs::write(
            st_dir.join("config.json"),
            serde_json::to_vec_pretty(&configuration)?,
        )?;
        safetensors::tensor::serialize_to_file(
            values.iter().map(|(n, d, s, b)| {
                (
                    n.as_str(),
                    safetensors::tensor::TensorView::new(*d, s.clone(), b).unwrap(),
                )
            }),
            None,
            &st_dir.join("model.safetensors"),
        )?;
        let source = Arc::new(eredu_checkpoint::store::SafetensorsWeightStore::open(
            &st_dir,
        )?);
        let safetensors_header = SafetensorsTargetPlan::prepare(
            source.as_ref(),
            config,
            SafetensorsEncoding::from_json(&serde_json::json!({}))?,
        )?;
        let safetensors_physical = safetensors_header
            .resolution()
            .source_keys()
            .iter()
            .map(|key| {
                use eredu_checkpoint::store::CheckpointSource;
                let metadata = source.source_metadata(key)?;
                let provenance = source.source_provenance(key)?;
                let physical = eredu_runtime::ReplicatedTextPhysicalSource::new(
                    key,
                    provenance.physical_tensor,
                    provenance
                        .backing_shard
                        .ok_or("fixture source lacks a shard")?,
                    provenance.output,
                    provenance.source_encoding,
                    metadata.encoded_byte_len,
                )?;
                Ok((key.clone(), physical))
            })
            .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
        let st = safetensors_header.clone().bind(source, limits())?;
        Ok(PreparedFixtures {
            gguf,
            safetensors: st,
            gguf_header: header,
            safetensors_header,
            safetensors_physical,
            gguf_path: path,
            safetensors_path: st_dir,
        })
    }
}

/// Frozen scalar-backend trajectory for the independently exported dense fixture.
/// Contains prompt/decode IDs, final-row logits and tolerances; this is native
/// mechanism conformance data, not released-model or independent-equation parity.
pub const DENSE_TRAJECTORY_JSON: &str = include_str!("qwen4_exp/dense-trajectory.json");

mod media;
pub use media::{add_vision_weights, write_vision_projector};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrent_fixture_variants_preserve_published_order_and_container_geometry() {
        // Explicit published head order, independent of the writer's nested
        // key/repeat loop. The default remains the frozen trajectory fixture.
        for (fixture, key_heads, order) in [
            (Fixture::new(), 2, vec![0, 3, 1, 4, 2, 5]),
            (
                Fixture::tensor_parallel_four(),
                4,
                vec![0, 2, 4, 6, 1, 3, 5, 7],
            ),
        ] {
            let value_heads = order.len();
            for (physical, canonical, axis, prefix, channels, exponential) in [
                (
                    "attn_qkv.weight",
                    "in_proj_qkv.weight",
                    0,
                    key_heads * 16,
                    128,
                    false,
                ),
                (
                    "ssm_conv1d.weight",
                    "conv1d.weight",
                    0,
                    key_heads * 16,
                    128,
                    false,
                ),
                ("attn_gate.weight", "in_proj_z.weight", 0, 0, 128, false),
                ("ssm_out.weight", "out_proj.weight", 1, 0, 128, false),
                ("ssm_alpha.weight", "in_proj_a.weight", 0, 0, 1, false),
                ("ssm_dt.bias", "dt_bias", 0, 0, 1, false),
                ("ssm_a", "A_log", 0, 0, 1, true),
            ] {
                let tensor = fixture
                    .tensors
                    .iter()
                    .find(|tensor| tensor.name == format!("blk.0.{physical}"))
                    .unwrap();
                assert_eq!(tensor.shape[axis], prefix + value_heads * channels);
                let actual: Vec<_> = tensor
                    .data
                    .chunks_exact(4)
                    .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                    .collect();
                let canonical =
                    &fixture.expected[&format!("model.layers.0.linear_attn.{canonical}")];
                let stride = tensor.shape[axis + 1..].iter().product::<usize>();
                let outer = tensor.shape[..axis].iter().product::<usize>();
                for row in 0..outer {
                    for position in 0..tensor.shape[axis] {
                        let canonical_position = if position < prefix {
                            position
                        } else {
                            let value = position - prefix;
                            prefix + order[value / channels] * channels + value % channels
                        };
                        for column in 0..stride {
                            let index = (row * tensor.shape[axis] + position) * stride + column;
                            let original = canonical
                                [(row * tensor.shape[axis] + canonical_position) * stride + column];
                            let expected = if exponential {
                                -original.exp()
                            } else {
                                original
                            };
                            assert_eq!(actual[index], expected, "{physical} scalar {index}");
                        }
                    }
                }
            }
            let directory = tempfile::tempdir().unwrap();
            let prepared = fixture.prepare(directory.path()).unwrap();
            let json: serde_json::Value = serde_json::from_slice(
                &std::fs::read(prepared.safetensors_path.join("config.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(json["linear_num_key_heads"], key_heads);
            assert_eq!(json["linear_num_value_heads"], value_heads);
            for target in [&prepared.safetensors, &prepared.gguf] {
                let config = target.spec().configuration();
                assert_eq!(config.recurrent.key_heads as usize, key_heads);
                assert_eq!(config.recurrent.value_heads as usize, value_heads);
                assert_eq!((config.attention.heads, config.attention.kv_heads), (4, 2));
            }
        }
    }
}
