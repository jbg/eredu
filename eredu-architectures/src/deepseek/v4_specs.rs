//! Typed attention, residual-stream and prediction declarations for V4 invocations.

use super::{V4Args, V4AttentionPolicy};
use eredu_nn::{
    AttentionHeadGeometry, DeclareLogicalParameters, EmbeddingSpec, Error, HyperConnectionSpec,
    HyperHeadSpec, LinearSpec, LogicalParameterDeclaration, LogicalParameterRole,
    LowRankProjectionSpec, NormalizationConstructionSpec, ParameterSpec,
};

fn hyper_connection_spec(args: &V4Args, prefix: &str) -> Result<HyperConnectionSpec, Error> {
    let spec = HyperConnectionSpec {
        streams: args.hc_mult,
        hidden_size: args.hidden_size,
        sinkhorn_iterations: usize::try_from(args.hc_sinkhorn_iters).map_err(Error::backend)?,
        epsilon: args.hc_eps,
        function: parameter(format!("{prefix}_fn"))?,
        base: parameter(format!("{prefix}_base"))?,
        scale: parameter(format!("{prefix}_scale"))?,
    };
    spec.validate()?;
    Ok(spec)
}

#[derive(Debug, Clone)]
pub(crate) struct HyperBlockSpec {
    pub attention: HyperConnectionSpec,
    pub feed_forward: HyperConnectionSpec,
}

impl DeclareLogicalParameters for HyperBlockSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.attention.logical_parameters()?;
        parameters.extend(self.feed_forward.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

pub(super) fn declare_hyper_block_spec(args: &V4Args, root: &str) -> Result<HyperBlockSpec, Error> {
    let spec = HyperBlockSpec {
        attention: hyper_connection_spec(args, &format!("{root}.hc_attn"))?,
        feed_forward: hyper_connection_spec(args, &format!("{root}.hc_ffn"))?,
    };
    spec.logical_parameters()?;
    Ok(spec)
}

pub(super) fn declare_hyper_head_spec(args: &V4Args, prefix: &str) -> Result<HyperHeadSpec, Error> {
    let spec = HyperHeadSpec {
        streams: args.hc_mult,
        hidden_size: args.hidden_size,
        norm_epsilon: args.rms_norm_eps,
        epsilon: args.hc_eps,
        function: parameter(format!("{prefix}_fn"))?,
        base: parameter(format!("{prefix}_base"))?,
        scale: parameter(format!("{prefix}_scale"))?,
    };
    spec.validate()?;
    Ok(spec)
}

fn parameter(name: String) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}

fn linear(args: &V4Args, name: String, input: i32, output: i32) -> Result<LinearSpec, Error> {
    Ok(LinearSpec {
        input,
        output,
        format: crate::linear_format::standard_linear_format(&name, args.linear_format_for(&name))?,
        weight: parameter(name)?,
        bias: None,
    })
}

fn normalization(
    args: &V4Args,
    name: String,
    dimensions: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    Ok(NormalizationConstructionSpec::learned(
        dimensions,
        args.rms_norm_eps,
        parameter(name)?,
    ))
}

#[derive(Debug, Clone)]
pub(crate) struct PredictionSpec {
    pub embedding_projection: LinearSpec,
    pub hidden_projection: LinearSpec,
    pub embedding_norm: NormalizationConstructionSpec,
    pub hidden_norm: NormalizationConstructionSpec,
    pub output_norm: NormalizationConstructionSpec,
    pub hyper_head: HyperHeadSpec,
}

impl DeclareLogicalParameters for PredictionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.embedding_projection.logical_parameters()?;
        parameters.extend(self.hidden_projection.logical_parameters()?);
        parameters.extend(self.embedding_norm.logical_parameters()?);
        parameters.extend(self.hidden_norm.logical_parameters()?);
        parameters.extend(self.output_norm.logical_parameters()?);
        parameters.extend(self.hyper_head.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

pub(super) fn declare_prediction_spec(args: &V4Args, root: &str) -> Result<PredictionSpec, Error> {
    let spec = PredictionSpec {
        embedding_projection: linear(
            args,
            format!("{root}.e_proj.weight"),
            args.hidden_size,
            args.hidden_size,
        )?,
        hidden_projection: linear(
            args,
            format!("{root}.h_proj.weight"),
            args.hidden_size,
            args.hidden_size,
        )?,
        embedding_norm: normalization(args, format!("{root}.enorm.weight"), args.hidden_size)?,
        hidden_norm: normalization(args, format!("{root}.hnorm.weight"), args.hidden_size)?,
        output_norm: normalization(args, format!("{root}.norm.weight"), args.hidden_size)?,
        hyper_head: declare_hyper_head_spec(args, &format!("{root}.hc_head"))?,
    };
    spec.logical_parameters()?;
    Ok(spec)
}

#[derive(Debug, Clone)]
pub(crate) struct DsparkSpec {
    pub main_projection: LinearSpec,
    pub main_norm: NormalizationConstructionSpec,
    pub output_norm: NormalizationConstructionSpec,
    pub hyper_head: HyperHeadSpec,
    pub markov_embedding: EmbeddingSpec,
    pub markov_output: LinearSpec,
    pub confidence_head: LinearSpec,
}

impl DeclareLogicalParameters for DsparkSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.main_projection.logical_parameters()?;
        parameters.extend(self.main_norm.logical_parameters()?);
        parameters.extend(self.output_norm.logical_parameters()?);
        parameters.extend(self.hyper_head.logical_parameters()?);
        parameters.extend(self.markov_embedding.logical_parameters()?);
        parameters.extend(self.markov_output.logical_parameters()?);
        parameters.extend(self.confidence_head.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

pub(super) fn declare_dspark_spec(args: &V4Args) -> Result<DsparkSpec, Error> {
    let config = args
        .dspark
        .as_ref()
        .ok_or_else(|| Error::backend("V4 DSpark configuration is missing"))?;
    let last = args
        .num_nextn_predict_layers
        .checked_sub(1)
        .filter(|&n| n >= 0)
        .ok_or_else(|| Error::backend("V4 DSpark requires at least one prediction layer"))?;
    let captures = args
        .target_capture_policy
        .as_ref()
        .ok_or_else(|| Error::backend("V4 DSpark target capture policy is missing"))?
        .len();
    let input = i32::try_from(captures)
        .ok()
        .and_then(|n| n.checked_mul(args.hidden_size))
        .ok_or_else(|| Error::backend("V4 DSpark capture width overflowed"))?;
    let confidence_width = args
        .hidden_size
        .checked_add(config.markov_rank)
        .ok_or_else(|| Error::backend("V4 DSpark confidence width overflowed"))?;
    let markov_weight = format!("mtp.{last}.markov_head.markov_w1.weight");
    let spec = DsparkSpec {
        main_projection: linear(
            args,
            "mtp.0.main_proj.weight".into(),
            input,
            args.hidden_size,
        )?,
        main_norm: normalization(args, "mtp.0.main_norm.weight".into(), args.hidden_size)?,
        output_norm: normalization(args, format!("mtp.{last}.norm.weight"), args.hidden_size)?,
        hyper_head: declare_hyper_head_spec(args, &format!("mtp.{last}.hc_head"))?,
        markov_embedding: EmbeddingSpec {
            vocabulary: args.vocab_size,
            dimensions: config.markov_rank,
            format: crate::linear_format::standard_linear_format(
                &markov_weight,
                args.linear_format_for(&markov_weight),
            )?,
            weight: parameter(markov_weight)?,
        },
        markov_output: linear(
            args,
            format!("mtp.{last}.markov_head.markov_w2.weight"),
            config.markov_rank,
            args.vocab_size,
        )?,
        confidence_head: linear(
            args,
            format!("mtp.{last}.confidence_head.proj.weight"),
            confidence_width,
            1,
        )?,
    };
    spec.logical_parameters()?;
    Ok(spec)
}

#[derive(Debug, Clone)]
pub(crate) struct CompressorSpec {
    pub ratio: i32,
    pub wkv: LinearSpec,
    pub wgate: LinearSpec,
    pub position: ParameterSpec,
    pub normalization: NormalizationConstructionSpec,
}
impl CompressorSpec {
    pub fn head_dimensions(&self) -> i32 {
        self.normalization.dimensions
    }
    pub fn overlapping(&self) -> bool {
        self.ratio == 4
    }
    pub fn position_shape(&self) -> [i32; 2] {
        [self.ratio, self.wkv.output]
    }
}
impl DeclareLogicalParameters for CompressorSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let dimensions = self.head_dimensions();
        let output = dimensions
            .checked_mul(if self.overlapping() { 2 } else { 1 })
            .ok_or_else(|| Error::backend("V4 compressor projection width overflowed"))?;
        if !matches!(self.ratio, 4 | 128)
            || self.wkv.input != self.wgate.input
            || self.wkv.output != output
            || self.wgate.output != output
        {
            return Err(Error::backend("incompatible V4 compressor geometry"));
        }
        let mut parameters = self.wkv.logical_parameters()?;
        parameters.extend(self.wgate.logical_parameters()?);
        parameters.push(LogicalParameterDeclaration::new(
            &self.position,
            &self.position_shape(),
            LogicalParameterRole::PositionBias,
        )?);
        parameters.extend(self.normalization.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}
fn compressor_spec(
    args: &V4Args,
    root: &str,
    ratio: i32,
    head_dimensions: i32,
) -> Result<CompressorSpec, Error> {
    let output = head_dimensions
        .checked_mul(if ratio == 4 { 2 } else { 1 })
        .ok_or_else(|| Error::backend("V4 compressor output width overflowed"))?;
    Ok(CompressorSpec {
        ratio,
        wkv: linear(args, format!("{root}.wkv.weight"), args.hidden_size, output)?,
        wgate: linear(
            args,
            format!("{root}.wgate.weight"),
            args.hidden_size,
            output,
        )?,
        position: parameter(format!("{root}.ape"))?,
        normalization: normalization(args, format!("{root}.norm.weight"), head_dimensions)?,
    })
}

#[derive(Debug, Clone)]
pub(crate) struct IndexerSpec {
    pub geometry: AttentionHeadGeometry,
    pub top_k: i32,
    pub query: LinearSpec,
    pub weights: LinearSpec,
    pub compressor: CompressorSpec,
}
impl DeclareLogicalParameters for IndexerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let geometry = self.geometry;
        if geometry.key_value_heads() != 1
            || geometry.key_dimensions() != geometry.value_dimensions()
            || self.top_k <= 0
            || self.compressor.ratio != 4
            || self.compressor.head_dimensions() != geometry.key_dimensions()
            || self.query.output != geometry.query_width()
            || self.weights.output != geometry.query_heads()
            || self.weights.input != self.compressor.wkv.input
        {
            return Err(Error::backend("incompatible V4 indexer geometry"));
        }
        let mut parameters = self.query.logical_parameters()?;
        parameters.extend(self.weights.logical_parameters()?);
        parameters.extend(self.compressor.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AttentionSpec {
    pub query: LowRankProjectionSpec,
    pub kv: LinearSpec,
    pub kv_norm: NormalizationConstructionSpec,
    pub output_a: LinearSpec,
    pub output_b: LinearSpec,
    pub sinks: ParameterSpec,
    pub compressor: Option<CompressorSpec>,
    pub indexer: Option<IndexerSpec>,
    pub geometry: AttentionHeadGeometry,
    pub groups: i32,
    pub policy: V4AttentionPolicy,
    pub normalization_epsilon: f32,
}
impl AttentionSpec {
    /// Rank of each declared grouped output block, after scope validation.
    pub fn output_rank(&self) -> i32 {
        self.output_b.input / self.groups
    }
    pub fn sink_shape(&self) -> [i32; 1] {
        [self.geometry.query_heads()]
    }
    pub fn ratio(&self) -> i32 {
        match self.policy {
            V4AttentionPolicy::Local => 0,
            V4AttentionPolicy::Compressed { ratio } => ratio,
        }
    }
}
impl DeclareLogicalParameters for AttentionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let geometry = self.geometry;
        let hidden = self.kv.input;
        if geometry.key_value_heads() != 1
            || geometry.key_dimensions() != geometry.value_dimensions()
            || self.groups <= 0
            || geometry.query_heads() % self.groups != 0
            || self
                .query
                .first
                .as_ref()
                .is_none_or(|first| first.input != hidden)
            || self.query.second.output != geometry.query_width()
            || self.kv.output != geometry.key_width()
            || self.kv_norm.dimensions != geometry.key_dimensions()
            || self.output_a.input != geometry.attended_width() / self.groups
            || self.output_a.output != self.output_b.input
            || self.output_b.input % self.groups != 0
            || self.output_b.output != hidden
            || !self.normalization_epsilon.is_finite()
            || self.normalization_epsilon <= 0.0
        {
            return Err(Error::backend(
                "incompatible V4 attention projection geometry",
            ));
        }
        let ratio = self.ratio();
        if !matches!(
            self.policy,
            V4AttentionPolicy::Local | V4AttentionPolicy::Compressed { ratio: 4 | 128 }
        ) || self.compressor.is_some() != (ratio != 0)
            || self.indexer.is_some() != (ratio == 4)
            || self.compressor.as_ref().is_some_and(|spec| {
                spec.ratio != ratio
                    || spec.wkv.input != hidden
                    || spec.head_dimensions() != geometry.key_dimensions()
            })
            || self.indexer.as_ref().is_some_and(|spec| {
                spec.weights.input != hidden
                    || spec.query.input != self.query.second.input
                    || spec.geometry.query_heads() % self.groups != 0
            })
        {
            return Err(Error::backend("incompatible V4 compression/indexer policy"));
        }
        let mut parameters = self.query.logical_parameters()?;
        parameters.extend(self.kv.logical_parameters()?);
        parameters.extend(self.kv_norm.logical_parameters()?);
        parameters.extend(self.output_a.logical_parameters()?);
        parameters.extend(self.output_b.logical_parameters()?);
        parameters.push(LogicalParameterDeclaration::new(
            &self.sinks,
            &self.sink_shape(),
            LogicalParameterRole::AttentionSink,
        )?);
        if let Some(compressor) = &self.compressor {
            parameters.extend(compressor.logical_parameters()?);
        }
        if let Some(indexer) = &self.indexer {
            parameters.extend(indexer.logical_parameters()?);
        }
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

pub(super) fn declare_attention_spec(
    args: &V4Args,
    layer: usize,
    root: &str,
) -> Result<AttentionSpec, Error> {
    args.validate().map_err(Error::backend)?;
    let policy = args
        .attention_policy(layer)
        .ok_or_else(|| Error::backend(format!("missing V4 attention policy {layer}")))?;
    let geometry = AttentionHeadGeometry::new(
        args.num_attention_heads,
        args.num_key_value_heads,
        args.head_dim,
        args.head_dim,
    )?;
    let width = geometry.query_width();
    let output_width = args
        .o_groups
        .checked_mul(args.o_lora_rank)
        .ok_or_else(|| Error::backend("V4 grouped output width overflowed"))?;
    let ratio = match policy {
        V4AttentionPolicy::Local => 0,
        V4AttentionPolicy::Compressed { ratio } => ratio,
    };
    let spec = AttentionSpec {
        geometry,
        query: LowRankProjectionSpec {
            first: Some(linear(
                args,
                format!("{root}.wq_a.weight"),
                args.hidden_size,
                args.q_lora_rank,
            )?),
            normalization: normalization(args, format!("{root}.q_norm.weight"), args.q_lora_rank)?,
            second: linear(args, format!("{root}.wq_b.weight"), args.q_lora_rank, width)?,
        },
        kv: linear(
            args,
            format!("{root}.wkv.weight"),
            args.hidden_size,
            geometry.key_dimensions(),
        )?,
        kv_norm: normalization(
            args,
            format!("{root}.kv_norm.weight"),
            geometry.key_dimensions(),
        )?,
        output_a: linear(
            args,
            format!("{root}.wo_a.weight"),
            width / args.o_groups,
            output_width,
        )?,
        output_b: linear(
            args,
            format!("{root}.wo_b.weight"),
            output_width,
            args.hidden_size,
        )?,
        sinks: parameter(format!("{root}.attn_sink"))?,
        compressor: (ratio != 0)
            .then(|| {
                compressor_spec(
                    args,
                    &format!("{root}.compressor"),
                    ratio,
                    geometry.key_dimensions(),
                )
            })
            .transpose()?,
        indexer: if ratio == 4 {
            let geometry = AttentionHeadGeometry::new(
                args.index_n_heads,
                1,
                args.index_head_dim,
                args.index_head_dim,
            )?;
            Some(IndexerSpec {
                geometry,
                top_k: args.index_topk,
                query: linear(
                    args,
                    format!("{root}.indexer.wq_b.weight"),
                    args.q_lora_rank,
                    geometry.query_width(),
                )?,
                weights: linear(
                    args,
                    format!("{root}.indexer.weights_proj.weight"),
                    args.hidden_size,
                    geometry.query_heads(),
                )?,
                compressor: compressor_spec(
                    args,
                    &format!("{root}.indexer.compressor"),
                    ratio,
                    geometry.key_dimensions(),
                )?,
            })
        } else {
            None
        },
        groups: args.o_groups,
        policy,
        normalization_epsilon: args.rms_norm_eps,
    };
    spec.logical_parameters()?;
    Ok(spec)
}

pub(super) fn declare_block_normalizations(
    args: &V4Args,
    root: &str,
) -> Result<[NormalizationConstructionSpec; 2], Error> {
    let specs = [
        normalization(args, format!("{root}.attn_norm.weight"), args.hidden_size)?,
        normalization(args, format!("{root}.ffn_norm.weight"), args.hidden_size)?,
    ];
    let mut parameters = specs[0].logical_parameters()?;
    parameters.extend(specs[1].logical_parameters()?);
    eredu_nn::validate_logical_parameter_topology(&parameters).map_err(Error::backend_source)?;
    Ok(specs)
}

#[derive(Debug, Clone)]
pub(crate) struct TokenRoutingTableSpec {
    pub parameter: ParameterSpec,
    pub vocabulary: i32,
    pub routes: i32,
}
impl TokenRoutingTableSpec {
    pub fn shape(&self) -> [i32; 2] {
        [self.vocabulary, self.routes]
    }
}
impl DeclareLogicalParameters for TokenRoutingTableSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.parameter,
            &self.shape(),
            LogicalParameterRole::RoutingTable,
        )?])
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MoeSpec {
    pub routed: super::moe::RoutedSpec,
    pub token_table: Option<TokenRoutingTableSpec>,
}
impl DeclareLogicalParameters for MoeSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.routed.logical_parameters()?;
        if let Some(table) = &self.token_table {
            parameters.extend(table.logical_parameters()?);
        }
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}
pub(super) fn declare_moe_spec(args: &V4Args, layer: usize, root: &str) -> Result<MoeSpec, Error> {
    args.validate().map_err(Error::backend)?;
    let routed = super::v4::declare_routed_spec_at(args, layer, root)?;
    let token_table = (layer < args.num_hash_layers as usize)
        .then(|| -> Result<_, Error> {
            Ok(TokenRoutingTableSpec {
                parameter: parameter(format!("{root}.gate.tid2eid"))?,
                vocabulary: args.vocab_size,
                routes: routed.parameters.selector.selection().top_k(),
            })
        })
        .transpose()?;
    let spec = MoeSpec {
        routed,
        token_table,
    };
    spec.logical_parameters()?;
    Ok(spec)
}

/// Exact retained scope projections used by cold and materializing consumers.
pub(crate) fn attention_spec(
    args: &V4Args,
    layer: usize,
    root: &str,
) -> Result<AttentionSpec, Error> {
    Ok(args
        .parameters()
        .block_at(
            layer,
            root.strip_suffix(".attn")
                .ok_or_else(|| Error::backend("unknown V4 attention root"))?,
        )?
        .attention
        .clone())
}
pub(crate) fn block_normalizations(
    args: &V4Args,
    root: &str,
) -> Result<[NormalizationConstructionSpec; 2], Error> {
    Ok(args.parameters().block(root)?.norms.clone())
}
pub(crate) fn hyper_block_spec(args: &V4Args, root: &str) -> Result<HyperBlockSpec, Error> {
    Ok(args.parameters().block(root)?.hyper.clone())
}
pub(crate) fn hyper_head_spec(args: &V4Args, prefix: &str) -> Result<HyperHeadSpec, Error> {
    args.parameters().hyper_head(prefix).cloned()
}
pub(crate) fn prediction_spec(args: &V4Args, root: &str) -> Result<PredictionSpec, Error> {
    args.parameters().prediction(root).cloned()
}
pub(crate) fn dspark_spec(args: &V4Args) -> Result<DsparkSpec, Error> {
    args.parameters()
        .dspark
        .as_ref()
        .cloned()
        .ok_or_else(|| Error::backend("V4 source has no DSpark owner"))
}
pub(crate) fn moe_spec(
    args: &V4Args,
    layer: usize,
    root: &str,
    experts: Option<eredu_nn::GroupedGatedProductSpec>,
) -> Result<MoeSpec, Error> {
    let mut spec = args
        .parameters()
        .block_at(
            layer,
            root.strip_suffix(".ffn")
                .ok_or_else(|| Error::backend("unknown V4 expert root"))?,
        )?
        .moe
        .clone();
    if let Some(experts) = experts {
        spec.routed.parameters.experts = experts;
    }
    spec.logical_parameters()?;
    Ok(spec)
}
