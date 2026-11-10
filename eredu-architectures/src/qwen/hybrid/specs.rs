//! Complete typed target and prediction block parameter authority.
use super::{HybridConfig, HybridLayerPolicy};
use eredu_nn::{DeclareLogicalParameters, Error, NormalizationConstructionSpec};

#[derive(Debug, Clone)]
pub(crate) enum MixerParameters {
    /// Gated-delta recurrence with independent state at each invocation.
    Recurrent(crate::gated_delta::GatedDeltaMixerSpec),
    /// Interleaved gated self-attention.
    Attention(super::block::AttentionParameters),
}
#[derive(Debug, Clone)]
pub(crate) enum FeedForwardParameters {
    Dense(eredu_nn::SplitGatedProductSpec),
    Routed(crate::shared_routed::SharedRoutedGatedProductParameters),
}
/// Exact residual, mixer and feed-forward slots for one canonical block owner.
#[derive(Debug, Clone)]
pub struct BlockParameters {
    pub(crate) norms: [NormalizationConstructionSpec; 2],
    pub(crate) mixer: MixerParameters,
    pub(crate) feed_forward: FeedForwardParameters,
}
impl BlockParameters {
    /// Projects retained target slots through the selected local policy.
    pub fn target(config: &HybridConfig, layer: usize) -> Result<Self, Error> {
        config.project_block(
            config
                .scopes()
                .target
                .get(layer)
                .ok_or_else(|| Error::backend("undeclared hybrid target layer"))?,
        )
    }
    /// Projects one retained full-attention prediction owner.
    pub fn prediction(config: &HybridConfig, depth: usize) -> Result<Self, Error> {
        config.project_block(
            config
                .scopes()
                .prediction
                .get(depth)
                .ok_or_else(|| Error::backend("undeclared hybrid prediction depth"))?,
        )
    }
    pub(crate) fn at(
        config: &HybridConfig,
        root: &str,
        policy: HybridLayerPolicy,
    ) -> Result<Self, Error> {
        let (scope, declared) = if let Some(index) = root
            .strip_prefix("model.layers.")
            .and_then(|i| i.parse::<usize>().ok())
        {
            (
                Self::target(config, index)?,
                *config
                    .layer_schedule
                    .get(index)
                    .ok_or_else(|| Error::backend("undeclared hybrid layer"))?,
            )
        } else if let Some(depth) = root
            .strip_prefix("mtp.layers.")
            .and_then(|i| i.parse::<usize>().ok())
        {
            (
                Self::prediction(config, depth)?,
                HybridLayerPolicy::SelfAttention(eredu_core::AttentionPolicy::Full),
            )
        } else {
            return Err(Error::backend("unresolved hybrid block identity"));
        };
        if policy != declared {
            return Err(Error::backend(
                "hybrid invocation policy differs from its declaration",
            ));
        }
        Ok(scope)
    }
    pub(crate) fn with_experts(
        mut self,
        bank: eredu_nn::GroupedGatedProductSpec,
    ) -> Result<Self, Error> {
        let FeedForwardParameters::Routed(parameters) = &mut self.feed_forward else {
            return Err(Error::backend("dense block cannot bind an expert bank"));
        };
        let identities=|spec:&eredu_nn::GroupedGatedProductSpec|->Result<Vec<eredu_nn::ParameterSpec>,Error>{Ok(spec.logical_parameters()?.iter().map(|p|p.parameter.clone()).collect())};
        if identities(&parameters.routed.experts)? != identities(&bank)? {
            return Err(Error::backend(
                "selected expert bank changes declared identities",
            ));
        }
        parameters.routed.experts = bank;
        parameters.validate()?;
        Ok(self)
    }
    pub(crate) fn experts(&self) -> i32 {
        match &self.feed_forward {
            FeedForwardParameters::Dense(_) => 0,
            FeedForwardParameters::Routed(p) => p.routed.selector.selection().group_count(),
        }
    }
    pub(super) fn declare_at(
        config: &super::HybridPolicy,
        root: &str,
        policy: HybridLayerPolicy,
    ) -> Result<Self, Error> {
        let norms = [
            super::block::normalization_spec(
                config,
                format!("{root}.input_layernorm.weight"),
                config.hidden_size,
            )?,
            super::block::normalization_spec(
                config,
                format!("{root}.post_attention_layernorm.weight"),
                config.hidden_size,
            )?,
        ];
        let mixer = match policy {
            HybridLayerPolicy::LinearAttention => {
                MixerParameters::Recurrent(super::linear_attention::recurrent_spec(config, root)?)
            }
            HybridLayerPolicy::SelfAttention(_) => {
                MixerParameters::Attention(super::block::AttentionParameters::new(config, root)?)
            }
        };
        let feed_forward = if config.is_moe() {
            FeedForwardParameters::Routed(super::block::shared_routed_parameters(
                config, root, None,
            )?)
        } else {
            FeedForwardParameters::Dense(super::block::mlp_parameters(
                config,
                &format!("{root}.mlp"),
                config.intermediate_size,
            )?)
        };
        let parameters = Self {
            norms,
            mixer,
            feed_forward,
        };
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for BlockParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.norms.as_slice().logical_parameters()?;
        p.extend(match &self.mixer {
            MixerParameters::Recurrent(s) => s.logical_parameters()?,
            MixerParameters::Attention(s) => s.logical_parameters()?,
        });
        p.extend(match &self.feed_forward {
            FeedForwardParameters::Dense(s) => s.logical_parameters()?,
            FeedForwardParameters::Routed(s) => s.logical_parameters()?,
        });
        Ok(p)
    }
}
