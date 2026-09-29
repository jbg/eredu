//! Text-weight contracts for llama.cpp 2145525a4081d66ff1a87cf43ef809f95a85ac0c.
//! Preparation is header-only; materialization is restricted to one owner/expert.
use super::recipes::ParameterScope;
use crate::qwen4_exp::{
    config::{Config, LayerKind},
    prepared::{PreparationError, PreparedParameters},
};
use eredu_checkpoint::{
    gguf_store::GgufCatalog,
    recipe::{DerivedWeightRecipe as Recipe, RecipeDtype},
    schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
        TensorOperation,
    },
    store::{
        PreparedCheckpointSource, PreparedTensorSource, SharedCheckpointSource, TensorSelection,
    },
    validation::{resolve_gguf_plan, ResolvedCheckpointPlan},
    LinearFormat,
};
use eredu_gguf::{Checkpoint, QuantizedTensorRepresentation};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Default)]
struct Transform {
    offset: bool,
    neg_log: bool,
    heads: Option<(usize, usize)>, // axis and unchanged prefix
    reshape: Option<Vec<usize>>,
}
#[derive(Clone)]
struct Entry {
    physical: String,
    canonical: String,
    shape: Vec<usize>,
    scope: ParameterScope,
    expert: bool,
    transform: Transform,
}
impl Entry {
    fn new(
        physical: impl Into<String>,
        canonical: impl Into<String>,
        shape: Vec<usize>,
        scope: ParameterScope,
    ) -> Self {
        Self {
            physical: physical.into(),
            canonical: canonical.into(),
            shape,
            scope,
            expert: false,
            transform: Transform::default(),
        }
    }
    fn offset(mut self) -> Self {
        self.transform.offset = true;
        self
    }
    fn heads(mut self, axis: usize, prefix: usize) -> Self {
        self.transform.heads = Some((axis, prefix));
        self
    }
    fn reshape(mut self, shape: Vec<usize>) -> Self {
        self.transform.reshape = Some(shape);
        self
    }
    fn neg_log(mut self) -> Self {
        self.transform.neg_log = true;
        self
    }
    fn expert(mut self) -> Self {
        self.expert = true;
        self
    }
    fn output(&self, original: &str) -> String {
        if original == self.physical {
            return self.canonical.clone();
        }
        let suffix = original
            .strip_prefix(
                self.physical
                    .strip_suffix(".weight")
                    .unwrap_or(&self.physical),
            )
            .expect("container companion prefix");
        if self.expert {
            format!("{}{}", self.canonical, suffix.replace('.', "_"))
        } else {
            format!(
                "{}{}",
                self.canonical
                    .strip_suffix(".weight")
                    .unwrap_or(&self.canonical),
                suffix
            )
        }
    }
}
fn entries(c: &Config) -> Vec<Entry> {
    let h = c.hidden_size as usize;
    let width = h * c.residual.streams as usize;
    let mut e = vec![Entry::new(
        "token_embd.weight",
        "model.embed_tokens.weight",
        vec![c.vocabulary as usize, h],
        ParameterScope::Static,
    )];
    if !c.tied_embeddings {
        e.push(Entry::new(
            "output.weight",
            "lm_head.weight",
            vec![c.vocabulary as usize, h],
            ParameterScope::Static,
        ));
    }
    let residual = |e: &mut Vec<Entry>, physical: &str, canonical: &str, scope, inject| {
        e.push(
            Entry::new(
                format!("{physical}_norm.weight"),
                format!("{canonical}.hc_norm.weight"),
                vec![width],
                scope,
            )
            .offset(),
        );
        for (p, n, s) in [
            (
                "down",
                "input_mix_weight_down",
                vec![c.residual.rank as usize, width],
            ),
            (
                "up",
                "input_mix_weight_up",
                vec![width, c.residual.rank as usize],
            ),
        ] {
            e.push(Entry::new(
                format!("{physical}_{p}.weight"),
                format!("{canonical}.{n}.weight"),
                s,
                scope,
            ));
        }
        if inject {
            e.push(Entry::new(
                format!("{physical}_inject.weight"),
                format!("{canonical}.block_inject_weight.weight"),
                vec![c.residual.streams as usize, width],
                scope,
            ));
        }
    };
    residual(
        &mut e,
        "output_hc",
        "model.hyper_connection_mixer",
        ParameterScope::Static,
        false,
    );
    for (l, kind) in c.layers.iter().enumerate() {
        let scope = ParameterScope::Target(l);
        let p = format!("blk.{l}");
        let n = format!("model.layers.{l}");
        residual(
            &mut e,
            &format!("{p}.hc_attn"),
            &format!("{n}.attn_hyper_connection"),
            scope,
            true,
        );
        residual(
            &mut e,
            &format!("{p}.hc_ffn"),
            &format!("{n}.mlp_hyper_connection"),
            scope,
            true,
        );
        match kind {
            LayerKind::Recurrent => {
                let r = &c.recurrent;
                let k = r.key_heads as usize * r.key_dim as usize;
                let v = r.value_heads as usize * r.value_dim as usize;
                let n = format!("{n}.linear_attn");
                for (physical, name, shape, axis, prefix) in [
                    ("attn_qkv", "in_proj_qkv", vec![2 * k + v, h], 0, 2 * k),
                    ("attn_gate", "in_proj_z", vec![v, h], 0, 0),
                    (
                        "ssm_alpha",
                        "in_proj_a",
                        vec![r.value_heads as usize, h],
                        0,
                        0,
                    ),
                    (
                        "ssm_beta",
                        "in_proj_b",
                        vec![r.value_heads as usize, h],
                        0,
                        0,
                    ),
                    ("ssm_out", "out_proj", vec![h, v], 1, 0),
                ] {
                    e.push(
                        Entry::new(
                            format!("{p}.{physical}.weight"),
                            format!("{n}.{name}.weight"),
                            shape,
                            scope,
                        )
                        .heads(axis, prefix),
                    );
                }
                e.push(
                    Entry::new(
                        format!("{p}.ssm_conv1d.weight"),
                        format!("{n}.conv1d.weight"),
                        vec![2 * k + v, r.kernel as usize],
                        scope,
                    )
                    .heads(0, 2 * k)
                    .reshape(vec![2 * k + v, 1, r.kernel as usize]),
                );
                e.push(
                    Entry::new(
                        format!("{p}.ssm_dt.bias"),
                        format!("{n}.dt_bias"),
                        vec![r.value_heads as usize],
                        scope,
                    )
                    .heads(0, 0),
                );
                e.push(
                    Entry::new(
                        format!("{p}.ssm_a"),
                        format!("{n}.A_log"),
                        vec![r.value_heads as usize],
                        scope,
                    )
                    .heads(0, 0)
                    .neg_log(),
                );
                e.push(Entry::new(
                    format!("{p}.ssm_norm.weight"),
                    format!("{n}.norm.weight"),
                    vec![r.value_dim as usize],
                    scope,
                ));
            }
            LayerKind::Indexed => {
                let a = &c.attention;
                let n = format!("{n}.self_attn");
                for (pname, name, rows, cols) in [
                    ("attn_q", "q_proj", 2 * a.heads * a.head_dim, c.hidden_size),
                    ("attn_k", "k_proj", a.kv_heads * a.head_dim, c.hidden_size),
                    ("attn_v", "v_proj", a.kv_heads * a.head_dim, c.hidden_size),
                    ("attn_output", "o_proj", c.hidden_size, a.heads * a.head_dim),
                    (
                        "indexer.q_proj",
                        "indexer.index_q_proj",
                        a.index_heads * a.index_head_dim,
                        c.hidden_size,
                    ),
                    (
                        "indexer.k_proj",
                        "indexer.index_k_proj",
                        a.index_kv_heads * a.index_head_dim,
                        c.hidden_size,
                    ),
                ] {
                    e.push(Entry::new(
                        format!("{p}.{pname}.weight"),
                        format!("{n}.{name}.weight"),
                        vec![rows as usize, cols as usize],
                        scope,
                    ));
                }
                for (pname, name, d) in [
                    ("attn_q_norm", "q_norm", a.head_dim),
                    ("attn_k_norm", "k_norm", a.head_dim),
                    ("indexer.q_norm", "indexer.q_layernorm", a.index_head_dim),
                    ("indexer.k_norm", "indexer.k_layernorm", a.index_head_dim),
                ] {
                    e.push(
                        Entry::new(
                            format!("{p}.{pname}.weight"),
                            format!("{n}.{name}.weight"),
                            vec![d as usize],
                            scope,
                        )
                        .offset(),
                    );
                }
                if a.bias {
                    for (physical, name, rows) in [
                        ("attn_q", "q_proj", 2 * a.heads * a.head_dim),
                        ("attn_k", "k_proj", a.kv_heads * a.head_dim),
                        ("attn_v", "v_proj", a.kv_heads * a.head_dim),
                    ] {
                        e.push(Entry::new(
                            format!("{p}.{physical}.bias"),
                            format!("{n}.{name}.bias"),
                            vec![rows as usize],
                            scope,
                        ));
                    }
                }
            }
        }
        if c.ngram.layers.contains(&l) {
            let scope = ParameterScope::Lexical(l);
            let n = format!("{n}.ple");
            for (physical, name, rows) in [
                ("ple_key", "key_proj", width),
                ("ple_value", "value_proj", h),
            ] {
                e.push(Entry::new(
                    format!("{p}.{physical}.weight"),
                    format!("{n}.{name}.weight"),
                    vec![rows, c.ngram.embedding_dim as usize],
                    scope,
                ));
            }
            for name in ["norm_key", "norm_query", "norm_conv"] {
                e.push(
                    Entry::new(
                        format!("{p}.ple_{name}.weight"),
                        format!("{n}.{name}.weight"),
                        vec![width],
                        scope,
                    )
                    .offset(),
                );
            }
            e.push(
                Entry::new(
                    format!("{p}.ple_conv1d.weight"),
                    format!("{n}.conv1d.weight"),
                    vec![width, c.ngram.kernel as usize],
                    scope,
                )
                .reshape(vec![width, 1, c.ngram.kernel as usize]),
            );
        }
        let ex = &c.experts;
        e.push(Entry::new(
            format!("{p}.ffn_gate_inp.weight"),
            format!("{n}.mlp.gate.weight"),
            vec![ex.count as usize, h],
            scope,
        ));
        e.push(Entry::new(
            format!("{p}.ffn_gate_inp_shexp.weight"),
            format!("{n}.mlp.shared_expert_gate.weight"),
            vec![1, h],
            scope,
        ));
        for (pname, name, shape) in [
            (
                "gate",
                "gate_proj",
                vec![ex.shared_intermediate as usize, h],
            ),
            ("up", "up_proj", vec![ex.shared_intermediate as usize, h]),
            (
                "down",
                "down_proj",
                vec![h, ex.shared_intermediate as usize],
            ),
        ] {
            e.push(Entry::new(
                format!("{p}.ffn_{pname}_shexp.weight"),
                format!("{n}.mlp.shared_expert.{name}.weight"),
                shape,
                scope,
            ));
        }
        for (pname, name, shape) in [
            (
                "gate",
                "gate_proj",
                vec![ex.count as usize, ex.intermediate as usize, h],
            ),
            (
                "up",
                "up_proj",
                vec![ex.count as usize, ex.intermediate as usize, h],
            ),
            (
                "down",
                "down_proj",
                vec![ex.count as usize, h, ex.intermediate as usize],
            ),
        ] {
            e.push(
                Entry::new(
                    format!("{p}.ffn_{pname}_exps.weight"),
                    format!("{n}.mlp.experts.{name}"),
                    shape,
                    scope,
                )
                .expert(),
            );
        }
    }
    e
}

/// Metadata-only text layout, representation and recipe authority. Binding consumes
/// an existing exact source; this plan never opens a store or reads tensor payloads.
#[derive(Clone)]
pub struct GgufTextPlan {
    config: Config,
    pub(in crate::qwen4_exp) table: super::GgufTableSourcePlan,
    checkpoint: Checkpoint,
    checkpoint_plan: GgufCheckpointPlan,
    mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
    resolution: ResolvedCheckpointPlan,
    catalog: GgufCatalog,
    entries: Vec<Entry>,
    pub(in crate::qwen4_exp) formats: BTreeMap<String, LinearFormat>,
}
impl GgufTextPlan {
    /// Parses released text metadata and validates every ordinary/expert descriptor.
    /// The table shares this source and is exposed only through restricted row owners.
    pub fn prepare(checkpoint: &Checkpoint) -> Result<Self, PreparationError> {
        let config =
            Config::from_gguf(checkpoint).map_err(|e| PreparationError::Contract(e.to_string()))?;
        Self::with_config(checkpoint, config)
    }
    fn with_config(checkpoint: &Checkpoint, config: Config) -> Result<Self, PreparationError> {
        let fail = |e: String| PreparationError::Contract(e);
        let table = super::GgufTableSourcePlan::prepare(checkpoint, &config)?;
        let checkpoint = table.checkpoint();
        // Reject oversized geometry before forming projection dimensions. Header
        // integers may each fit i32 while their architectural products do not.
        let a = &config.attention;
        let r = &config.recurrent;
        let products = [
            a.heads
                .checked_mul(a.head_dim)
                .and_then(|v| v.checked_mul(2)),
            a.kv_heads.checked_mul(a.head_dim),
            a.index_heads.checked_mul(a.index_head_dim),
            a.index_kv_heads.checked_mul(a.index_head_dim),
            r.key_heads
                .checked_mul(r.key_dim)
                .and_then(|k| k.checked_mul(2))
                .and_then(|k| {
                    r.value_heads
                        .checked_mul(r.value_dim)
                        .and_then(|v| k.checked_add(v))
                }),
            config.experts.intermediate.checked_mul(2),
        ];
        if products.iter().any(Option::is_none) {
            return Err(fail("GGUF projection geometry exceeds i32".into()));
        }
        let mut entries = entries(&config);
        // The published gate is [1,h]; accept its equivalent squeezed vector too.
        for entry in &mut entries {
            if entry.canonical.ends_with(".shared_expert_gate.weight")
                && checkpoint.tensors().any(|t| {
                    t.descriptor().name == entry.physical
                        && t.descriptor().row_major_shape() == [config.hidden_size as u64]
                })
            {
                entry.transform.reshape = Some(entry.shape.clone());
                entry.shape.remove(0);
            }
        }
        let mut decoded = Vec::new();
        // A head permutation across a GGML block requires scalar decoding before
        // reordering. Whole-block permutations keep encoded weights/companions.
        for entry in &entries {
            if entry.transform.heads == Some((1, 0))
                && config.recurrent.key_heads != config.recurrent.value_heads
            {
                if let Some(t) = checkpoint
                    .tensors()
                    .find(|t| t.descriptor().name == entry.physical)
                {
                    let (block, _) = t
                        .descriptor()
                        .ggml_type
                        .block_and_bytes()
                        .map_err(|e| fail(e.to_string()))?;
                    if config.recurrent.value_dim as u64 % block != 0 {
                        decoded.push(entry.physical.clone());
                    }
                }
            }
        }
        // Concatenated Q/K and gate/up matrices must have one exact encoding.
        // Differently encoded inputs get bounded F32 views, never a full bank read.
        for layer in 0..config.layers.len() {
            for pair in [
                [
                    format!("blk.{layer}.indexer.q_proj.weight"),
                    format!("blk.{layer}.indexer.k_proj.weight"),
                ],
                [
                    format!("blk.{layer}.ffn_gate_exps.weight"),
                    format!("blk.{layer}.ffn_up_exps.weight"),
                ],
            ] {
                let ts: Vec<_> = pair
                    .iter()
                    .filter_map(|name| {
                        checkpoint.shards().iter().find_map(|s| {
                            s.tensors()
                                .iter()
                                .find(|t| t.descriptor().name == *name)
                                .map(|t| (s, t))
                        })
                    })
                    .collect();
                if ts.len() == 2
                    && (crate::linear_format::gguf_tensor_format(ts[0].1, ts[0].0.endian())
                        != crate::linear_format::gguf_tensor_format(ts[1].1, ts[1].0.endian())
                        || ts[0].1.outputs()[0].dtype != ts[1].1.outputs()[0].dtype)
                {
                    for (_, t) in ts {
                        if t.descriptor()
                            .ggml_type
                            .block_and_bytes()
                            .map_err(|e| fail(e.to_string()))?
                            .0
                            > 1
                        {
                            decoded.push(t.descriptor().name.clone());
                        }
                    }
                }
            }
        }
        let checkpoint = checkpoint
            .clone()
            .into_tensor_representation(decoded, QuantizedTensorRepresentation::DecodedF32)
            .map_err(|e| fail(e.to_string()))?;
        let mut constraints: Vec<_> = entries
            .iter()
            .map(|e| {
                GgufTensorConstraint::required(
                    &e.physical,
                    e.shape.clone(),
                    GgufTypeConstraint::OperationClass(
                        if e.shape.len() == 1 || e.transform.reshape.is_some() {
                            TensorOperation::Dense
                        } else {
                            TensorOperation::Matrix
                        },
                    ),
                )
            })
            .collect();
        constraints.push(table.constraint().clone());
        // All owners share one physical source; each binding projects its exact keys.
        let plan = GgufCheckpointPlan::new(
            "qwen4_exp target weights",
            constraints,
            vec![],
            CatalogPolicy::non_strict(),
        )
        .map_err(|e| fail(e.to_string()))?;
        let mut mapped = BTreeMap::new();
        let mut formats = BTreeMap::new();
        for entry in &entries {
            if let Some((shard, tensor)) = checkpoint.shards().iter().find_map(|s| {
                s.tensors()
                    .iter()
                    .find(|t| t.descriptor().name == entry.physical)
                    .map(|t| (s, t))
            }) {
                formats.insert(
                    entry.canonical.clone(),
                    crate::linear_format::gguf_tensor_format(tensor, shard.endian())
                        .map_err(fail)?,
                );
                for output in tensor.outputs() {
                    mapped.insert(output.name.clone(), entry.output(&output.name));
                }
            }
        }
        let mapping = checkpoint
            .translated_outputs(|name| mapped.get(name).cloned().unwrap_or_else(|| name.into()))
            .map_err(|e| fail(e.to_string()))?;
        let resolution = resolve_gguf_plan(&checkpoint, &plan)
            .map_err(|e| fail(format!("GGUF text contract did not resolve: {e:?}")))?;
        let catalog = GgufCatalog::from_resolved_checkpoint(&checkpoint, &resolution, &mapping)?;
        for layer in 0..config.layers.len() {
            for (target, member) in [
                (
                    format!("model.layers.{layer}.self_attn.indexer.index_qk_proj.weight"),
                    format!("model.layers.{layer}.self_attn.indexer.index_q_proj.weight"),
                ),
                (
                    format!("model.layers.{layer}.mlp.experts.gate_up_proj"),
                    format!("model.layers.{layer}.mlp.experts.gate_proj"),
                ),
            ] {
                if let Some(&format) = formats.get(&member) {
                    formats.insert(target, format);
                }
            }
        }
        if config.tied_embeddings {
            formats.insert(
                "lm_head.weight".into(),
                formats["model.embed_tokens.weight"],
            );
        }
        let this = Self {
            config,
            table,
            checkpoint,
            checkpoint_plan: plan,
            mapping,
            resolution,
            catalog,
            entries,
            formats,
        };
        // Validate transformed shape/dtype contracts now without reading payloads.
        this.parameter_recipes(ParameterScope::Static)?;
        for l in 0..this.config.layers.len() {
            this.parameter_recipes(ParameterScope::Target(l))?;
            if this.config.ngram.layers.contains(&l) {
                this.parameter_recipes(ParameterScope::Lexical(l))?;
            }
            this.expert_recipes(l, 0)?;
        }
        Ok(this)
    }
    /// Retained checkpoint after table and projection representation selection.
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }
    /// Architecture constraints resolved against the selected representations.
    pub fn checkpoint_plan(&self) -> &GgufCheckpointPlan {
        &self.checkpoint_plan
    }
    /// Exact canonical mapping for the ordinary source factory.
    pub fn mapping(&self) -> &[eredu_gguf::TranslatedTensorLayout] {
        &self.mapping
    }
    /// Retained selected physical source set.
    pub fn resolution(&self) -> &ResolvedCheckpointPlan {
        &self.resolution
    }
    /// Shared canonical metadata projection, without readable source methods.
    pub fn catalog(&self) -> &GgufCatalog {
        &self.catalog
    }
    /// Checks every retained output before binding; never reopens the checkpoint.
    pub fn bind(self, source: SharedCheckpointSource) -> Result<GgufTextWeights, PreparationError> {
        let keys: BTreeSet<_> = self.catalog.keys().into_iter().collect();
        if source.source_keys().into_iter().collect::<BTreeSet<_>>() != keys {
            return Err(PreparationError::Contract(
                "GGUF source set differs from admitted text outputs".into(),
            ));
        }
        let expected = keys
            .into_iter()
            .map(|key| {
                let entry = PreparedTensorSource {
                    metadata: self.catalog.metadata(&key)?,
                    provenance: self.catalog.source_provenance(&key)?,
                };
                Ok((key, entry))
            })
            .collect::<Result<BTreeMap<_, _>, eredu_checkpoint::store::StoreError>>()?;
        let source: SharedCheckpointSource =
            Arc::new(PreparedCheckpointSource::new(source, expected)?);
        Ok(GgufTextWeights { plan: self, source })
    }
    fn validate_recipes(
        &self,
        recipes: BTreeMap<String, Recipe>,
    ) -> Result<BTreeMap<String, Recipe>, PreparationError> {
        for recipe in recipes.values() {
            recipe
                .infer(&self.catalog)
                .map_err(|e| PreparationError::Contract(e.to_string()))?;
        }
        Ok(recipes)
    }
    /// Text configuration normalized from the same retained artifact.
    pub fn config(&self) -> &Config {
        &self.config
    }
    /// Actual selected format, including scalar fallback for unaligned head blocks.
    pub fn format(&self, name: &str) -> Option<LinearFormat> {
        self.formats.get(name).copied()
    }
    /// One ordinary layer, lexical injection or static source owner.
    pub fn parameter_recipes(
        &self,
        scope: ParameterScope,
    ) -> Result<BTreeMap<String, Recipe>, PreparationError> {
        let fail = |e: String| PreparationError::Contract(e);
        if matches!(scope, ParameterScope::Prediction(_)) {
            return Err(PreparationError::MissingPrediction);
        }
        if !self.entries.iter().any(|e| e.scope == scope) {
            return Err(fail(
                "GGUF text artifact does not contain this parameter scope".into(),
            ));
        }
        let mut result = BTreeMap::new();
        for e in self
            .entries
            .iter()
            .filter(|e| e.scope == scope && !e.expert)
        {
            let keys = self.catalog.keys();
            for key in keys.into_iter().filter(|k| {
                *k == e.canonical
                    || e.canonical
                        .strip_suffix(".weight")
                        .is_some_and(|p| *k == format!("{p}.scales") || *k == format!("{p}.biases"))
            }) {
                let shape = self.catalog.metadata(&key)?.logical_shape;
                let mut recipe = Recipe::source(&key, TensorSelection::Full);
                if e.transform.neg_log {
                    recipe = Recipe::NegLog {
                        input: Box::new(recipe),
                    };
                }
                if e.transform.offset {
                    recipe = Recipe::SubtractOne {
                        input: Box::new(recipe),
                    };
                }
                if let Some((axis, prefix)) = e.transform.heads {
                    recipe = crate::gated_delta::checkpoint::grouped_value_heads(
                        recipe,
                        &shape,
                        axis,
                        prefix,
                        self.config.recurrent.key_heads as usize,
                        self.config.recurrent.value_heads as usize,
                    )
                    .map_err(fail)?;
                }
                if let Some(shape) = &e.transform.reshape {
                    recipe = Recipe::Reshape {
                        input: Box::new(recipe),
                        shape: shape.clone(),
                    };
                }
                result.insert(key, recipe);
            }
        }
        if let ParameterScope::Target(l) = scope {
            let root = format!("model.layers.{l}.self_attn.indexer");
            for suffix in ["weight", "scales", "biases"] {
                let q = format!("{root}.index_q_proj.{suffix}");
                let k = format!("{root}.index_k_proj.{suffix}");
                if let Some(q) = result.remove(&q) {
                    let k = result
                        .remove(&k)
                        .ok_or_else(|| fail("incomplete indexer companion pair".into()))?;
                    result.insert(
                        format!("{root}.index_qk_proj.{suffix}"),
                        Recipe::Concatenate {
                            axis: 0,
                            inputs: self.uniform_inputs(vec![q, k])?,
                        },
                    );
                }
            }
        }
        if scope == ParameterScope::Static && self.config.tied_embeddings {
            for suffix in ["weight", "scales", "biases"] {
                if let Some(r) = result.get(&format!("model.embed_tokens.{suffix}")).cloned() {
                    result.insert(format!("lm_head.{suffix}"), r);
                }
            }
        }
        self.validate_recipes(result)
    }
    fn uniform_inputs(&self, inputs: Vec<Recipe>) -> Result<Vec<Recipe>, PreparationError> {
        let dtypes = inputs
            .iter()
            .map(|r| {
                r.infer(&self.catalog)
                    .map(|m| m.dtype)
                    .map_err(|e| PreparationError::Contract(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if dtypes.windows(2).all(|d| d[0] == d[1]) {
            return Ok(inputs);
        }
        if !dtypes
            .iter()
            .all(|d| matches!(d, RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32))
        {
            return Err(PreparationError::Contract(
                "incompatible concatenated GGUF storage types".into(),
            ));
        }
        Ok(inputs
            .into_iter()
            .map(|input| Recipe::Cast {
                input: Box::new(input),
                dtype: RecipeDtype::F32,
            })
            .collect())
    }
    /// One expert selection, including every scale/bias companion; no full bank read.
    pub fn expert_recipes(
        &self,
        layer: usize,
        expert: usize,
    ) -> Result<BTreeMap<String, Recipe>, PreparationError> {
        if expert >= self.config.experts.count as usize {
            return Err(PreparationError::Contract(
                "expert is outside the declared routed unit".into(),
            ));
        }
        let bank = self.expert_bank_recipes(layer)?;
        let recipes = bank
            .iter()
            .map(|(name, recipe)| {
                Ok((
                    name.clone(),
                    recipe
                        .select_bounded(
                            &self.catalog,
                            eredu_checkpoint::store::TensorSelection::Range {
                                axis: 0,
                                start: expert,
                                end: expert + 1,
                            },
                        )
                        .map_err(|e| PreparationError::Contract(e.to_string()))?,
                ))
            })
            .collect::<Result<_, PreparationError>>()?;
        self.validate_recipes(recipes)
    }
    pub(in crate::qwen4_exp) fn expert_bank_recipes(
        &self,
        layer: usize,
    ) -> Result<BTreeMap<String, Recipe>, PreparationError> {
        if layer >= self.config.layers.len() {
            return Err(PreparationError::Contract(
                "expert layer is outside target".into(),
            ));
        }
        let root = format!("model.layers.{layer}.mlp.experts");
        let mut recipes = crate::shared_routed::checkpoint::gated_expert_recipes(
            &self.catalog,
            &root,
            0..self.config.experts.count as usize,
            TensorSelection::Full,
            super::schema::aliases,
        )
        .map_err(PreparationError::Contract)?;
        for recipe in recipes.values_mut() {
            if let Recipe::Concatenate { inputs, .. } = recipe {
                *inputs = self.uniform_inputs(std::mem::take(inputs))?;
            }
        }
        self.validate_recipes(
            recipes
                .into_iter()
                .map(|(n, r)| (format!("{root}.{n}"), r))
                .collect(),
        )
    }
}

/// Exact text recipes bound to the source checked by its header plan.
#[derive(Clone)]
pub struct GgufTextWeights {
    pub(in crate::qwen4_exp) plan: GgufTextPlan,
    pub(in crate::qwen4_exp) source: SharedCheckpointSource,
}
impl GgufTextWeights {
    /// Retained source-free representation and recipe authority.
    pub fn plan(&self) -> &GgufTextPlan {
        &self.plan
    }
    /// Exact bound source; no subsequent artifact discovery is needed.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }
    /// Normalized text geometry.
    pub fn config(&self) -> &Config {
        self.plan.config()
    }
    /// Selected executable encoding for a canonical parameter.
    pub fn format(&self, name: &str) -> Option<LinearFormat> {
        self.plan.format(name)
    }
    /// Restricted static, decoder or lexical parameter owner.
    pub fn parameters(
        &self,
        scope: ParameterScope,
    ) -> Result<PreparedParameters, PreparationError> {
        PreparedParameters::new(self.source.clone(), self.plan.parameter_recipes(scope)?)
    }
    /// One bounded expert selection with all exact companions.
    pub fn expert(
        &self,
        layer: usize,
        expert: usize,
    ) -> Result<PreparedParameters, PreparationError> {
        PreparedParameters::new(
            self.source.clone(),
            self.plan.expert_recipes(layer, expert)?,
        )
    }
}
