//! Immutable hybrid policy and retained construction scopes.
use super::{
    specs::{FeedForwardParameters, MixerParameters},
    BlockParameters, HybridPolicy,
};
use eredu_nn::{DeclareLogicalParameters, Error, LinearSpec};
use std::sync::Arc;

/// Validated hybrid text policy with immutable target, static and prediction scopes.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    policy: HybridPolicy,
    scopes: Arc<Parameters>,
    local: bool,
}
#[derive(Debug, Clone)]
pub(super) struct Parameters {
    pub statics: crate::decoder::StaticModuleConstruction,
    pub target: Arc<Vec<BlockParameters>>,
    pub prediction: Arc<Vec<BlockParameters>>,
    pub shared: Option<super::PredictionSharedSpec>,
}
impl DeclareLogicalParameters for Parameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.statics.logical_parameters()?;
        p.extend(self.target.as_slice().logical_parameters()?);
        p.extend(self.prediction.as_slice().logical_parameters()?);
        if let Some(s) = &self.shared {
            p.extend(s.logical_parameters()?);
        }
        Ok(p)
    }
}
impl HybridConfig {
    /// Declares each source owner once after normalizing external configuration.
    pub fn new(policy: HybridPolicy) -> Result<Self, Error> {
        policy.validate().map_err(Error::backend_source)?;
        let statics = super::static_module_spec(&policy).construction_specs()?;
        let target: Vec<_> = policy
            .layer_schedule
            .iter()
            .enumerate()
            .map(|(index, &kind)| {
                BlockParameters::declare_at(&policy, &format!("model.layers.{index}"), kind)
            })
            .collect::<Result<_, _>>()?;
        let prediction: Vec<_> = (0..policy.mtp_num_hidden_layers as usize)
            .map(|depth| {
                BlockParameters::declare_at(
                    &policy,
                    &format!("mtp.layers.{depth}"),
                    super::HybridLayerPolicy::SelfAttention(eredu_core::AttentionPolicy::Full),
                )
            })
            .collect::<Result<_, _>>()?;
        let shared = (policy.mtp_num_hidden_layers > 0)
            .then(|| super::PredictionSharedSpec::new(&policy))
            .transpose()?;
        let scopes = Parameters {
            statics,
            target: Arc::new(target),
            prediction: Arc::new(prediction),
            shared,
        };
        eredu_nn::validate_logical_parameter_topology(&scopes.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(Self {
            policy,
            scopes: Arc::new(scopes),
            local: false,
        })
    }
    /// Normalized invocation and format policy, independent of parameter ownership.
    pub fn policy(&self) -> &HybridPolicy {
        &self.policy
    }
    pub(super) fn scopes(&self) -> &Parameters {
        &self.scopes
    }
    pub(crate) fn statics(&self) -> &crate::decoder::StaticModuleConstruction {
        &self.scopes.statics
    }
    /// Shared prediction fusion and normalization slots, when prediction is declared.
    pub fn prediction_shared(&self) -> Option<&super::PredictionSharedSpec> {
        self.scopes.shared.as_ref()
    }
    /// Removes prediction invocations and their independently owned slots from a target view.
    pub fn without_prediction(&self) -> Self {
        if self.mtp_num_hidden_layers == 0 {
            return self.clone();
        }
        let mut result = self.clone();
        result.policy.mtp_num_hidden_layers = 0;
        let scopes = Arc::make_mut(&mut result.scopes);
        scopes.prediction = Arc::new(Vec::new());
        scopes.shared = None;
        result
    }
    /// Projects encodings from retained slots, rejecting non-format policy changes.
    pub fn with_formats(&self, policy: HybridPolicy) -> Result<Self, Error> {
        if policy == self.policy {
            return Ok(self.clone());
        }
        let mut expected = policy.clone();
        expected.fp8 = self.policy.fp8.clone();
        expected.quantization = self.policy.quantization;
        expected.linear_formats = self.policy.linear_formats.clone();
        if expected != self.policy {
            return Err(Error::backend(
                "hybrid format projection changes invocation policy",
            ));
        }
        policy.validate().map_err(Error::backend_source)?;
        let mut scopes = self.scopes.as_ref().clone();
        let mut embedding = scopes.statics.embedding_spec().clone();
        embedding.format = crate::linear_format::project_matrix_format(
            &embedding.weight,
            &embedding.format,
            policy
                .linear_format(embedding.weight.id.as_str())
                .weight_quantization()
                .into(),
        )?;
        let mut head = scopes.statics.head_spec().cloned();
        if let Some(h) = &mut head {
            format_linear(h, &policy)?;
        }
        scopes.statics = crate::decoder::StaticModuleConstruction::new(
            embedding,
            head,
            scopes.statics.normalization_spec().clone(),
        )?;
        for scope in Arc::make_mut(&mut scopes.target)
            .iter_mut()
            .chain(Arc::make_mut(&mut scopes.prediction).iter_mut())
        {
            scope.project_formats(&policy)?;
        }
        if let Some(shared) = &mut scopes.shared {
            shared.fusion.format = crate::linear_format::project_matrix_format(
                &shared.fusion.weight,
                &shared.fusion.format,
                policy
                    .linear_formats
                    .get(shared.fusion.weight.id.as_str())
                    .copied()
                    .unwrap_or(eredu_checkpoint::LinearFormat::Dense),
            )?;
        }
        eredu_nn::validate_logical_parameter_topology(&scopes.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(Self {
            policy,
            scopes: Arc::new(scopes),
            local: self.local,
        })
    }
    pub(crate) fn with_local_policy(&self, policy: HybridPolicy) -> Result<Self, Error> {
        let mut expected = policy.clone();
        expected.num_attention_heads = self.num_attention_heads;
        expected.num_key_value_heads = self.num_key_value_heads;
        expected.linear_num_key_heads = self.linear_num_key_heads;
        expected.linear_num_value_heads = self.linear_num_value_heads;
        expected.intermediate_size = self.intermediate_size;
        expected.moe_intermediate_size = self.moe_intermediate_size;
        expected.shared_expert_intermediate_size = self.shared_expert_intermediate_size;
        if expected != self.policy {
            return Err(Error::backend(
                "hybrid local projection changes non-placement policy",
            ));
        }
        policy.validate().map_err(Error::backend_source)?;
        for (selected, declared) in [
            (policy.num_attention_heads, self.num_attention_heads),
            (policy.num_key_value_heads, self.num_key_value_heads),
            (policy.linear_num_key_heads, self.linear_num_key_heads),
            (policy.linear_num_value_heads, self.linear_num_value_heads),
            (policy.intermediate_size, self.intermediate_size),
            (policy.moe_intermediate_size, self.moe_intermediate_size),
            (
                policy.shared_expert_intermediate_size,
                self.shared_expert_intermediate_size,
            ),
        ] {
            if selected > declared {
                return Err(Error::backend(
                    "local hybrid geometry exceeds its source declaration",
                ));
            }
        }
        Ok(Self {
            policy,
            scopes: self.scopes.clone(),
            local: true,
        })
    }
    pub(super) fn project_block(&self, scope: &BlockParameters) -> Result<BlockParameters, Error> {
        let mut scope = scope.clone();
        if self.local {
            scope.project_geometry(&self.policy)?;
        }
        Ok(scope)
    }
}
impl std::ops::Deref for HybridConfig {
    type Target = HybridPolicy;
    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}
fn format_linear(spec: &mut LinearSpec, policy: &HybridPolicy) -> Result<(), Error> {
    spec.format = crate::linear_format::project_matrix_format(
        &spec.weight,
        &spec.format,
        policy.linear_format(spec.weight.id.as_str()),
    )?;
    Ok(())
}
fn resize_mlp(spec: &mut eredu_nn::SplitGatedProductSpec, width: i32) -> Result<(), Error> {
    if width <= 0 || width > spec.gate.output {
        return Err(Error::backend(
            "local hybrid feed-forward exceeds declared geometry",
        ));
    }
    spec.gate.output = width;
    spec.up.output = width;
    spec.down.input = width;
    spec.validate()
}
impl BlockParameters {
    fn project_geometry(&mut self, policy: &HybridPolicy) -> Result<(), Error> {
        match &mut self.mixer {
            MixerParameters::Attention(s) => {
                let old = s.projection.geometry;
                if policy.num_attention_heads > old.query_heads()
                    || policy.num_key_value_heads > old.key_value_heads()
                {
                    return Err(Error::backend("local hybrid attention exceeds declaration"));
                }
                let g = eredu_nn::AttentionHeadGeometry::new(
                    policy.num_attention_heads,
                    policy.num_key_value_heads,
                    old.key_dimensions(),
                    old.value_dimensions(),
                )?;
                s.projection.geometry = g;
                s.projection.query.output = s.projection.query_layout.projected_width(g)?;
                s.projection.key.output = g.key_width();
                s.projection.value.output = g.value_width();
                s.projection.output.input = g.attended_width();
            }
            MixerParameters::Recurrent(s) => {
                let old = s.geometry;
                if policy.linear_num_key_heads > old.key_heads()
                    || policy.linear_num_value_heads > old.value_heads()
                {
                    return Err(Error::backend(
                        "local hybrid recurrence exceeds declaration",
                    ));
                }
                let g = eredu_nn::GatedDeltaHeadGeometry::new(
                    policy.linear_num_key_heads,
                    policy.linear_num_value_heads,
                    old.key_dimensions(),
                    old.value_dimensions(),
                )?;
                s.geometry = g;
                s.input_qkv.output = g.qkv_width()?;
                s.input_gate.output = g.value_width();
                s.input_beta.output = g.value_heads();
                s.input_decay.output = g.value_heads();
                s.convolution.channels = g.qkv_width()?;
                s.output.input = g.value_width();
            }
        }
        match &mut self.feed_forward {
            FeedForwardParameters::Dense(s) => resize_mlp(s, policy.intermediate_size)?,
            FeedForwardParameters::Routed(s) => {
                s.routed = s.routed.clone().with_group_geometry(
                    s.routed.experts.group_count(),
                    policy.moe_intermediate_size,
                )?;
                resize_mlp(&mut s.shared, policy.shared_expert_intermediate_size)?;
            }
        }
        self.logical_parameters()?;
        Ok(())
    }
    fn project_formats(&mut self, policy: &HybridPolicy) -> Result<(), Error> {
        match &mut self.mixer {
            MixerParameters::Attention(s) => {
                for p in [
                    &mut s.projection.query,
                    &mut s.projection.key,
                    &mut s.projection.value,
                    &mut s.projection.output,
                ] {
                    format_linear(p, policy)?;
                }
            }
            MixerParameters::Recurrent(s) => {
                for p in [
                    &mut s.input_qkv,
                    &mut s.input_gate,
                    &mut s.input_beta,
                    &mut s.input_decay,
                    &mut s.output,
                ] {
                    format_linear(p, policy)?;
                }
            }
        }
        let project_mlp = |s: &mut eredu_nn::SplitGatedProductSpec| -> Result<(), Error> {
            for p in [&mut s.gate, &mut s.up, &mut s.down] {
                format_linear(p, policy)?;
            }
            Ok(())
        };
        match &mut self.feed_forward {
            FeedForwardParameters::Dense(s) => project_mlp(s)?,
            FeedForwardParameters::Routed(s) => {
                let selector = &s.routed.selector;
                s.routed.selector =
                    selector
                        .clone()
                        .with_format(crate::linear_format::project_matrix_format(
                            selector.weight(),
                            selector.format(),
                            policy.linear_format(selector.weight().id.as_str()),
                        )?)?;
                s.routed.experts = crate::linear_format::project_packed_gated_formats(
                    &s.routed.experts,
                    |name| policy.linear_format(name),
                )?;
                project_mlp(&mut s.shared)?;
                format_linear(&mut s.shared_gate, policy)?;
            }
        }
        self.logical_parameters()?;
        Ok(())
    }
}
