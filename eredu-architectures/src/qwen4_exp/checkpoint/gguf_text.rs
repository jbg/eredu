//! Text-weight contracts for llama.cpp 2145525a4081d66ff1a87cf43ef809f95a85ac0c.
//! Preparation is header-only; materialization is restricted to one owner/expert.
use super::recipes::ParameterScope;
use crate::qwen4_exp::{
    config::Config,
    prepared::PreparationError,
    target::declarations::{MixerParameters, TargetParameters, UnitParameters},
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
use eredu_runtime::PreparedParameterRecipes;
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
/// One published split projection, referring to its fused logical slot.
struct IndexerSourceProjection {
    canonical: String,
    logical: eredu_nn::ParameterId,
    shape: Vec<usize>,
}
fn indexer_sources(
    parameters: &crate::qwen4_exp::indexed::IndexedParameters,
) -> Result<[IndexerSourceProjection; 2], String> {
    use eredu_nn::DeclareLogicalParameters;
    parameters.validate().map_err(|e| e.to_string())?;
    let declarations = parameters
        .index_projection
        .logical_parameters()
        .map_err(|e| e.to_string())?;
    let projection = &declarations[0];
    let root = projection
        .parameter
        .id
        .as_str()
        .strip_suffix("index_qk_proj.weight")
        .ok_or("GGUF indexer source does not reference the released fused identity")?;
    let geometry = parameters.index_geometry;
    Ok([
        ("index_q_proj", geometry.query_width() as usize),
        ("index_k_proj", geometry.key_width() as usize),
    ]
    .map(|(suffix, rows)| IndexerSourceProjection {
        canonical: format!("{root}{suffix}.weight"),
        logical: projection.parameter.id.clone(),
        shape: vec![rows, projection.shape[1]],
    }))
}
fn entries(parameters: &TargetParameters) -> Result<Vec<Entry>, String> {
    use eredu_nn::DeclareLogicalParameters;
    parameters.logical_parameters().map_err(|e| e.to_string())?;
    let statics = &parameters.static_parameters;
    let mut e = Vec::new();
    for parameter in statics
        .embedding
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        e.push(Entry::new(
            "token_embd.weight",
            parameter.parameter.id.to_string(),
            parameter.shape,
            ParameterScope::Static,
        ));
    }
    if let Some(head) = &statics.head {
        for parameter in head.logical_parameters().map_err(|e| e.to_string())? {
            e.push(Entry::new(
                "output.weight",
                parameter.parameter.id.to_string(),
                parameter.shape,
                ParameterScope::Static,
            ));
        }
    }
    let residual = |e: &mut Vec<Entry>,
                    physical: &str,
                    spec: &eredu_nn::residual_streams::GatedResidualSpec,
                    scope|
     -> Result<(), String> {
        for parameter in spec
            .normalization
            .logical_parameters()
            .map_err(|e| e.to_string())?
        {
            e.push(
                Entry::new(
                    format!("{physical}_norm.weight"),
                    parameter.parameter.id.to_string(),
                    parameter.shape,
                    scope,
                )
                .offset(),
            );
        }
        for (suffix, projection) in [("down", &spec.down), ("up", &spec.up)]
            .into_iter()
            .chain(spec.injection.as_ref().map(|p| ("inject", p)))
        {
            for parameter in projection.logical_parameters().map_err(|e| e.to_string())? {
                e.push(Entry::new(
                    format!("{physical}_{suffix}.weight"),
                    parameter.parameter.id.to_string(),
                    parameter.shape,
                    scope,
                ));
            }
        }
        Ok(())
    };
    residual(
        &mut e,
        "output_hc",
        &statics.readout,
        ParameterScope::Static,
    )?;
    for unit in &parameters.units {
        let UnitParameters::Decoder {
            layer: l,
            mixer,
            feed_forward,
        } = unit
        else {
            continue;
        };
        let l = *l;
        let scope = ParameterScope::Target(l);
        let p = format!("blk.{l}");
        let mixer_residual = match mixer {
            MixerParameters::Recurrent(spec) => &spec.residual,
            MixerParameters::Indexed { residual, .. } => residual,
        };
        residual(&mut e, &format!("{p}.hc_attn"), mixer_residual, scope)?;
        residual(
            &mut e,
            &format!("{p}.hc_ffn"),
            &feed_forward.residual,
            scope,
        )?;
        match mixer {
            MixerParameters::Recurrent(sublayer) => {
                let spec = &sublayer.mixer;
                let key_prefix = spec.geometry.qkv_ranges().map_err(|e| e.to_string())?[2].start;
                for (physical, projection, axis, prefix) in [
                    ("attn_qkv", &spec.input_qkv, 0, key_prefix),
                    ("attn_gate", &spec.input_gate, 0, 0),
                    ("ssm_alpha", &spec.input_decay, 0, 0),
                    ("ssm_beta", &spec.input_beta, 0, 0),
                    ("ssm_out", &spec.output, 1, 0),
                ] {
                    for parameter in projection.logical_parameters().map_err(|e| e.to_string())? {
                        e.push(
                            Entry::new(
                                format!("{p}.{physical}.weight"),
                                parameter.parameter.id.to_string(),
                                parameter.shape,
                                scope,
                            )
                            .heads(axis, prefix),
                        );
                    }
                }
                for parameter in spec
                    .convolution
                    .logical_parameters()
                    .map_err(|e| e.to_string())?
                {
                    e.push(
                        Entry::new(
                            format!("{p}.ssm_conv1d.weight"),
                            parameter.parameter.id.to_string(),
                            vec![parameter.shape[0], parameter.shape[2]],
                            scope,
                        )
                        .heads(0, key_prefix)
                        .reshape(parameter.shape),
                    );
                }
                let declarations = spec.logical_parameters().map_err(|e| e.to_string())?;
                for (physical, parameter, heads, neg_log) in [
                    ("ssm_dt.bias", &spec.decay_bias, true, false),
                    ("ssm_a", &spec.transition_log, true, true),
                    ("ssm_norm.weight", &spec.normalization_weight, false, false),
                ] {
                    let declaration = declarations
                        .iter()
                        .find(|d| d.parameter.id == parameter.id)
                        .ok_or_else(|| {
                            format!("unresolved recurrent source slot {}", parameter.id)
                        })?;
                    let mut entry = Entry::new(
                        format!("{p}.{physical}"),
                        parameter.id.to_string(),
                        declaration.shape.clone(),
                        scope,
                    );
                    if heads {
                        entry = entry.heads(0, 0);
                    }
                    if neg_log {
                        entry = entry.neg_log();
                    }
                    e.push(entry);
                }
            }
            MixerParameters::Indexed { parameters, .. } => {
                use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};
                for (physical, projection) in ["attn_q", "attn_k", "attn_v", "attn_output"]
                    .into_iter()
                    .zip(parameters.projection.projections())
                {
                    for parameter in projection.logical_parameters().map_err(|e| e.to_string())? {
                        if parameter.role == LogicalParameterRole::LinearWeight {
                            e.push(Entry::new(
                                format!("{p}.{physical}.weight"),
                                parameter.parameter.id.to_string(),
                                parameter.shape,
                                scope,
                            ));
                        }
                    }
                }
                for (physical, source) in ["indexer.q_proj", "indexer.k_proj"]
                    .into_iter()
                    .zip(indexer_sources(&parameters)?)
                {
                    e.push(Entry::new(
                        format!("{p}.{physical}.weight"),
                        source.canonical,
                        source.shape,
                        scope,
                    ));
                }
                for (physical, norm) in [
                    ("attn_q_norm", &parameters.query_norm),
                    ("attn_k_norm", &parameters.key_norm),
                    ("indexer.q_norm", &parameters.index_query_norm),
                    ("indexer.k_norm", &parameters.index_key_norm),
                ] {
                    for parameter in norm.logical_parameters().map_err(|e| e.to_string())? {
                        e.push(
                            Entry::new(
                                format!("{p}.{physical}.weight"),
                                parameter.parameter.id.to_string(),
                                parameter.shape,
                                scope,
                            )
                            .offset(),
                        );
                    }
                }
                for (physical, projection) in ["attn_q", "attn_k", "attn_v"]
                    .into_iter()
                    .zip(parameters.projection.projections())
                {
                    for parameter in projection.logical_parameters().map_err(|e| e.to_string())? {
                        if parameter.role == LogicalParameterRole::LinearBias {
                            e.push(Entry::new(
                                format!("{p}.{physical}.bias"),
                                parameter.parameter.id.to_string(),
                                parameter.shape,
                                scope,
                            ));
                        }
                    }
                }
            }
        }
        if let Some(parameters) = parameters.units.iter().find_map(|unit| match unit {
            UnitParameters::Lexical { layer, parameters } if *layer == l => Some(parameters),
            _ => None,
        }) {
            let scope = ParameterScope::Lexical(l);
            for (physical, spec) in [
                ("ple_key", &parameters.key),
                ("ple_value", &parameters.value),
            ] {
                for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
                    e.push(Entry::new(
                        format!("{p}.{physical}.weight"),
                        parameter.parameter.id.as_str(),
                        parameter.shape,
                        scope,
                    ));
                }
            }
            for (physical, spec) in [
                ("norm_key", &parameters.key_norm),
                ("norm_query", &parameters.query_norm),
                ("norm_conv", &parameters.convolution_norm),
            ] {
                for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
                    e.push(
                        Entry::new(
                            format!("{p}.ple_{physical}.weight"),
                            parameter.parameter.id.as_str(),
                            parameter.shape,
                            scope,
                        )
                        .offset(),
                    );
                }
            }
            for parameter in parameters
                .convolution
                .logical_parameters()
                .map_err(|e| e.to_string())?
            {
                e.push(
                    Entry::new(
                        format!("{p}.ple_conv1d.weight"),
                        parameter.parameter.id.as_str(),
                        vec![parameter.shape[0], parameter.shape[2]],
                        scope,
                    )
                    .reshape(parameter.shape),
                );
            }
        }
        use eredu_nn::DeclareLogicalParameters;
        let parameters = &feed_forward.feed_forward.parameters;
        for parameter in parameters
            .routed
            .selector
            .logical_parameters()
            .map_err(|e| e.to_string())?
        {
            e.push(Entry::new(
                format!("{p}.ffn_gate_inp.weight"),
                parameter.parameter.id.to_string(),
                parameter.shape,
                scope,
            ));
        }
        for (physical, projection) in [
            ("gate_inp_shexp", &parameters.shared_gate),
            ("gate_shexp", &parameters.shared.gate),
            ("up_shexp", &parameters.shared.up),
            ("down_shexp", &parameters.shared.down),
        ] {
            for parameter in projection.logical_parameters().map_err(|e| e.to_string())? {
                e.push(Entry::new(
                    format!("{p}.ffn_{physical}.weight"),
                    parameter.parameter.id.to_string(),
                    parameter.shape,
                    scope,
                ));
            }
        }
        for (physical, source) in ["gate", "up", "down"]
            .into_iter()
            .zip(super::experts::sources(
                &feed_forward.feed_forward.parameters.routed.experts,
                super::experts::SourceLayout::SplitBanks,
            )?)
        {
            e.push(
                Entry::new(
                    format!("{p}.ffn_{physical}_exps.weight"),
                    source.name,
                    source.shape,
                    scope,
                )
                .expert(),
            );
        }
    }
    Ok(e)
}
/// Metadata-only text layout, representation and recipe authority. Binding consumes
/// an existing exact source; this plan never opens a store or reads tensor payloads.
#[derive(Clone)]
pub struct GgufTextPlan {
    pub(in crate::qwen4_exp) parameters: Arc<TargetParameters>,
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
            eredu_nn::GatedDeltaHeadGeometry::new(
                r.key_heads,
                r.value_heads,
                r.key_dim,
                r.value_dim,
            )
            .and_then(|geometry| geometry.qkv_width())
            .ok(),
            config.experts.intermediate.checked_mul(2),
        ];
        if products.iter().any(Option::is_none) {
            return Err(fail("GGUF projection geometry exceeds i32".into()));
        }
        let parameters = Arc::new(super::schema::target_parameters(&config).map_err(fail)?);
        let mut entries = entries(&parameters).map_err(fail)?;
        // The published gate is [1,h]; accept its equivalent squeezed vector too.
        for entry in &mut entries {
            if entry.canonical.ends_with(".shared_expert_gate.weight")
                && checkpoint.tensors().any(|t| {
                    t.descriptor().name == entry.physical
                        && t.descriptor().row_major_shape() == [entry.shape[1] as u64]
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
                    if !(config.recurrent.value_dim as u64).is_multiple_of(block) {
                        decoded.push(entry.physical.clone());
                    }
                }
            }
        }
        // Concatenated Q/K and gate/up matrices must have one exact encoding.
        // Differently encoded inputs get bounded F32 views, never a full bank read.
        for layer in parameters.units.iter().filter_map(|unit| match unit {
            UnitParameters::Decoder { layer, .. } => Some(*layer),
            _ => None,
        }) {
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
        for unit in &parameters.units {
            let UnitParameters::Decoder {
                mixer,
                feed_forward,
                ..
            } = unit
            else {
                continue;
            };
            let mut sources = super::experts::sources(
                &feed_forward.feed_forward.parameters.routed.experts,
                super::experts::SourceLayout::SplitBanks,
            )
            .map_err(fail)?;
            let gate = sources.remove(0);
            let mut links = vec![(gate.logical.to_string(), gate.name)];
            if let MixerParameters::Indexed { parameters, .. } = mixer {
                let [query, _] = indexer_sources(parameters).map_err(fail)?;
                links.push((query.logical.to_string(), query.canonical));
            }
            for (target, member) in links {
                if let Some(&format) = formats.get(&member) {
                    formats.insert(target, format);
                }
            }
        }
        if parameters.static_parameters.head.is_none() {
            let statics = &parameters.static_parameters;
            let output = statics.output_projection();
            formats.insert("lm_head.weight".into(), formats[output.weight.id.as_str()]);
        }
        let this = Self {
            parameters,
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
        for unit in &this.parameters.units {
            match unit {
                UnitParameters::Decoder { layer, .. } => {
                    this.parameter_recipes(ParameterScope::Target(*layer))?;
                    this.expert_recipes(*layer, 0)?;
                }
                UnitParameters::Lexical { layer, .. } => {
                    this.parameter_recipes(ParameterScope::Lexical(*layer))?;
                }
            }
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
    pub(crate) fn parameter_declaration(&self) -> &TargetParameters {
        &self.parameters
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
                        self.recurrent_geometry(scope)?.key_heads() as usize,
                        self.recurrent_geometry(scope)?.value_heads() as usize,
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
            if let Some((MixerParameters::Indexed { parameters, .. }, _)) =
                self.parameters.decoder(l)
            {
                let [query, key] = indexer_sources(parameters).map_err(fail)?;
                let companion = |weight: &str, suffix: &str| {
                    format!(
                        "{}.{}",
                        weight
                            .strip_suffix(".weight")
                            .expect("declared indexer weight"),
                        suffix
                    )
                };
                for suffix in ["weight", "scales", "biases"] {
                    let q = companion(&query.canonical, suffix);
                    let k = companion(&key.canonical, suffix);
                    if let Some(q) = result.remove(&q) {
                        let k = result
                            .remove(&k)
                            .ok_or_else(|| fail("incomplete indexer companion pair".into()))?;
                        result.insert(
                            companion(query.logical.as_str(), suffix),
                            Recipe::Concatenate {
                                axis: 0,
                                inputs: self.uniform_inputs(vec![q, k])?,
                            },
                        );
                    }
                }
            }
        }
        if scope == ParameterScope::Static && self.parameters.static_parameters.head.is_none() {
            let statics = &self.parameters.static_parameters;
            let output = statics.output_projection();
            let prefix = output
                .weight
                .id
                .as_str()
                .strip_suffix(".weight")
                .expect("declared vocabulary weight");
            for suffix in ["weight", "scales", "biases"] {
                if let Some(r) = result.get(&format!("{prefix}.{suffix}")).cloned() {
                    result.insert(format!("lm_head.{suffix}"), r);
                }
            }
        }
        self.validate_recipes(result)
    }
    fn recurrent_geometry(
        &self,
        scope: ParameterScope,
    ) -> Result<eredu_nn::GatedDeltaHeadGeometry, PreparationError> {
        if let ParameterScope::Target(layer) = scope {
            if let Some((MixerParameters::Recurrent(spec), _)) = self.parameters.decoder(layer) {
                return Ok(spec.mixer.geometry);
            }
        }
        Err(PreparationError::Contract(
            "recurrent transform has no declared recurrent owner".into(),
        ))
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
        let (_, feed_forward) = self
            .parameters
            .decoder(layer)
            .ok_or_else(|| PreparationError::Contract("expert layer is outside target".into()))?;
        if expert
            >= feed_forward
                .feed_forward
                .parameters
                .routed
                .experts
                .group_count() as usize
        {
            return Err(PreparationError::Contract(
                "expert is outside the declared routed unit".into(),
            ));
        }
        let bank = self.expert_bank_recipes(layer)?;
        let selection = TensorSelection::Range {
            axis: 0,
            start: expert,
            end: expert + 1,
        };
        let recipes = bank
            .iter()
            .map(|(name, recipe)| {
                eredu_runtime::select_parameter_recipe(
                    &self.catalog,
                    name,
                    recipe,
                    Some(std::slice::from_ref(&selection)),
                )
                .map(|recipe| (name.clone(), recipe))
                .map_err(PreparationError::from)
            })
            .collect::<Result<_, _>>()?;
        self.validate_recipes(recipes)
    }
    pub(in crate::qwen4_exp) fn expert_bank_recipes(
        &self,
        layer: usize,
    ) -> Result<BTreeMap<String, Recipe>, PreparationError> {
        let (_, feed_forward) = self
            .parameters
            .decoder(layer)
            .ok_or_else(|| PreparationError::Contract("expert layer is outside target".into()))?;
        let source = feed_forward
            .feed_forward
            .parameters
            .routed
            .experts
            .packed_parameters()?;
        let root = super::experts::source_root(&source).map_err(PreparationError::Contract)?;
        let mut recipes = crate::shared_routed::checkpoint::gated_expert_recipes(
            &self.catalog,
            root,
            0..source.group_count(),
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
                .map(|(name, recipe)| {
                    super::experts::binding_target(&source, &name).map(|target| (target, recipe))
                })
                .collect::<Result<_, _>>()
                .map_err(PreparationError::Contract)?,
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
    ) -> Result<PreparedParameterRecipes, PreparationError> {
        PreparedParameterRecipes::new(
            "qwen4_exp prepared owner",
            self.source.clone(),
            self.plan.parameter_recipes(scope)?,
        )
        .map_err(PreparationError::from)
    }
    /// One bounded expert selection with all exact companions.
    pub fn expert(
        &self,
        layer: usize,
        expert: usize,
    ) -> Result<PreparedParameterRecipes, PreparationError> {
        PreparedParameterRecipes::new(
            "qwen4_exp prepared owner",
            self.source.clone(),
            self.plan.expert_recipes(layer, expert)?,
        )
        .map_err(PreparationError::from)
    }
}
