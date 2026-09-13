//! Embedded Inkling multi-token predictor built from ordinary decoder layers.

use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, AuxiliaryConvolutionState, Error, GroupedNeuralBackend, LinearSpec,
    NormalizationConstructionSpec, NormalizationOperator, ParameterSpec, Parameterized, Tensor,
};

use super::{DecoderLayer, FeedForwardPolicy, LayerPolicy, ModelArgs, MtpConfig, TextArgs};

/// One prediction depth's hidden/token fusion and ordinary decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MtpDepth<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Target hidden-state normalization.
    pub hidden_norm: B::Normalization,
    /// Next-token embedding normalization.
    pub embedding_norm: B::Normalization,
    /// Concatenated hidden/embedding projection.
    pub input_projection: B::Linear,
    /// The same neutral decoder layer used by the ordinary model.
    pub transformer_block: DecoderLayer<B>,
}

/// Shared learned normalization used by every embedded prediction depth.
/// The prepared extension omits this module when chain normalization is disabled.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MtpShared<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Canonical shared chain normalization.
    pub chain_norm: B::Normalization,
}

/// Hidden-state continuation returned by one embedded prediction depth.
#[derive(Debug, Clone)]
pub struct MtpOutput<T> {
    /// Prediction-space hidden state; the caller applies the ordinary LM head.
    pub hidden: T,
    /// Token identity associated with this continuation.
    pub tokens: T,
}

/// Complete embedded multi-token prediction chain.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MtpModel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Ordered prediction depths.
    pub layers: Vec<MtpDepth<B>>,
    /// Optional normalization between prediction depths.
    pub chain_norm: Option<B::Normalization>,
    #[parameter(skip)]
    policies: Vec<AttentionPolicy>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> MtpModel<B> {
    /// Builds no predictor when the checkpoint declares zero depths.
    pub fn new(
        args: &ModelArgs,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self>, Error> {
        let Some(config) = args.mtp_config.as_ref() else {
            return Ok(None);
        };
        let count = usize::try_from(config.num_nextn_predict_layers)
            .map_err(|_| Error::backend("Inkling MTP layer count is negative"))?;
        if count == 0 {
            return Ok(None);
        }
        let sliding = args
            .text_config
            .layer_schedule
            .iter()
            .find_map(|policy| policy.attention.window())
            .map(|window| AttentionPolicy::Sliding { window });
        if !config.local_layer_ids.is_empty() && sliding.is_none() {
            return Err(Error::backend(
                "Inkling MTP local layers require a backbone sliding window",
            ));
        }
        let policies = (0..count)
            .map(|depth| {
                if config.local_layer_ids.contains(&depth) {
                    sliding.expect("validated local MTP policy")
                } else {
                    AttentionPolicy::Full
                }
            })
            .collect::<Vec<_>>();
        let mut layers = Vec::with_capacity(count);
        for (depth, attention) in policies.iter().copied().enumerate() {
            let text = mtp_text_args(&args.text_config, config, attention)?;
            let root = format!("model.mtp.layers.{depth}");
            let norm = |field: &str| {
                B::normalization(
                    NormalizationConstructionSpec::learned(
                        text.hidden_size,
                        text.rms_norm_eps,
                        ParameterSpec::trainable(format!("{root}.{field}.weight"))
                            .map_err(Error::backend)?,
                    ),
                    context,
                )
            };
            let input_weight = format!("{root}.input_proj.weight");
            layers.push(MtpDepth {
                hidden_norm: norm("hidden_norm")?,
                embedding_norm: norm("embed_norm")?,
                input_projection: B::linear(
                    LinearSpec {
                        input: text.hidden_size * 2,
                        output: text.hidden_size,
                        weight: ParameterSpec::trainable(&input_weight).map_err(Error::backend)?,
                        bias: None,
                        format: crate::linear_format::standard_linear_format(
                            &input_weight,
                            text.linear_format_for(&input_weight),
                        )?,
                    },
                    context,
                )?,
                transformer_block: DecoderLayer::new_at(
                    &text,
                    LayerPolicy {
                        attention,
                        feed_forward: FeedForwardPolicy::Dense,
                    },
                    &format!("{root}.transformer_block"),
                    context,
                )?,
            });
        }
        let chain_norm = config
            .chain_hidden_post_norm
            .then(|| {
                B::normalization(
                    NormalizationConstructionSpec::learned(
                        args.text_config.hidden_size,
                        args.text_config.rms_norm_eps,
                        ParameterSpec::trainable("model.mtp.chain_norm.weight")
                            .map_err(Error::backend)?,
                    ),
                    context,
                )
            })
            .transpose()?;
        Ok(Some(Self {
            layers,
            chain_norm,
            policies,
        }))
    }

    /// Returns the exact number of prediction depths.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns whether the predictor has no depths.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns the normalized attention policy for one depth.
    pub fn policy(&self, depth: usize) -> Option<AttentionPolicy> {
        self.policies.get(depth).copied()
    }

    /// Runs one cyclic prediction depth and advances only that depth's state.
    ///
    /// The hidden input is normalized twice to preserve the released Inkling
    /// execution graph exactly.
    pub fn forward_step<S>(
        &mut self,
        hidden: &B::Tensor,
        embeddings: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut [S],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MtpOutput<B::Tensor>, Error>
    where
        S: AttentionCache<B::Tensor> + AuxiliaryConvolutionState<B::Tensor>,
    {
        self.forward_step_instrumented(
            hidden,
            embeddings,
            tokens,
            depth,
            state,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Observes the released double-normalization fusion and ordinary decoder.
    pub fn forward_step_instrumented<S>(
        &mut self,
        hidden: &B::Tensor,
        embeddings: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut [S],
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<MtpOutput<B::Tensor>, Error>
    where
        S: AttentionCache<B::Tensor> + AuxiliaryConvolutionState<B::Tensor>,
    {
        if self.layers.is_empty() || state.len() != self.layers.len() {
            return Err(Error::backend(
                "Inkling MTP state does not match prediction depths",
            ));
        }
        let depth = depth % self.layers.len();
        self.layers[depth].forward_step_instrumented(
            hidden,
            embeddings,
            tokens,
            &mut state[depth],
            self.chain_norm.as_mut(),
            context,
            instrumentation,
        )
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> MtpDepth<B> {
    /// Executes one physical depth using its own causal state and the optional
    /// shared chain norm. The whole-chain and prepared drivers use this equation.
    pub fn forward_step_instrumented<S>(
        &mut self,
        hidden: &B::Tensor,
        embeddings: &B::Tensor,
        tokens: &B::Tensor,
        state: &mut S,
        chain_norm: Option<&mut B::Normalization>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<MtpOutput<B::Tensor>, Error>
    where
        S: AttentionCache<B::Tensor> + AuxiliaryConvolutionState<B::Tensor>,
    {
        instrumentation.observe("prediction.hidden", hidden)?;
        instrumentation.observe("prediction.embedding", embeddings)?;
        let hidden = self.hidden_norm.forward(hidden, context)?;
        let hidden = instrumentation.apply("prediction.hidden.first_normalized", hidden)?;
        let hidden = self.hidden_norm.forward(&hidden, context)?;
        let hidden = instrumentation.apply("prediction.hidden.normalized", hidden)?;
        let embeddings = self.embedding_norm.forward(embeddings, context)?;
        let embeddings = instrumentation.apply("prediction.embedding.normalized", embeddings)?;
        let combined = B::Tensor::concatenate(&[hidden, embeddings], -1, context)?;
        let fused = instrumentation.project::<B>(
            "prediction.fusion.input",
            &mut self.input_projection,
            &combined,
            None,
            context,
        )?;
        let fused = instrumentation.apply("prediction.fusion.output", fused)?;
        let hidden = instrumentation.with_scope("transformer_block", |instrumentation| {
            self.transformer_block.forward_with_provider_instrumented(
                &fused,
                Some(state),
                eredu_runtime::ExpertPass::Prefill,
                &mut eredu_runtime::ResidentExpertProvider,
                context,
                instrumentation,
            )
        })?;
        let hidden = instrumentation.apply("prediction.readout.residual", hidden)?;
        let hidden = match chain_norm {
            Some(norm) => norm.forward(&hidden, context)?,
            None => hidden,
        };
        let hidden = instrumentation.apply("prediction.readout.normalized", hidden)?;
        Ok(MtpOutput {
            hidden,
            tokens: tokens.clone(),
        })
    }
}

pub(crate) fn mtp_text_args(
    backbone: &TextArgs,
    config: &MtpConfig,
    attention: AttentionPolicy,
) -> Result<TextArgs, Error> {
    let mut text = backbone.clone();
    text.num_attention_heads = config
        .num_attention_heads
        .unwrap_or(text.num_attention_heads);
    text.num_key_value_heads = config
        .num_key_value_heads
        .unwrap_or(text.num_key_value_heads);
    text.head_dim = config.head_dim.unwrap_or(text.head_dim);
    text.swa_num_attention_heads = config
        .swa_num_attention_heads
        .or(text.swa_num_attention_heads);
    text.swa_num_key_value_heads = config
        .swa_num_key_value_heads
        .or(text.swa_num_key_value_heads);
    text.swa_head_dim = config.swa_head_dim.or(text.swa_head_dim);
    text.dense_intermediate_size = config
        .dense_intermediate_size
        .or(text.dense_intermediate_size);
    text.intermediate_size = config.intermediate_size.unwrap_or(text.intermediate_size);
    text.sconv_kernel_size = config.sconv_kernel_size.unwrap_or(text.sconv_kernel_size);
    text.rel_extent = config.rel_extent.unwrap_or(text.rel_extent);
    text.d_rel = config.d_rel.unwrap_or(text.d_rel);
    text.layer_schedule = LayerSchedule::new(
        1,
        vec![LayerPolicy {
            attention,
            feed_forward: FeedForwardPolicy::Dense,
        }],
    )
    .map_err(Error::backend)?;
    text.num_hidden_layers = 1;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use eredu_core::AttentionPolicy;

    #[test]
    fn cyclic_depth_selection_is_stable() {
        let policies = [AttentionPolicy::Full, AttentionPolicy::Full];
        assert_eq!(policies[5 % policies.len()], AttentionPolicy::Full);
    }
}
