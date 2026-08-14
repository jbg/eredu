//! Shared DeepSeek-V4 decoder layers.

use safemlx::{
    error::Exception, macros::ModuleParameters, module::Param, nn, ops::indexing::take, Array,
    Dtype, Stream,
};

use crate::{
    api::qwen3_5::{QwenLinear as Linear, QwenWeightFormat as WeightFormat},
    nn::{
        layers::silu,
        moe::{PackedSwiGluExperts, TopKRouter, TopKRouterConfig, TopKRouterScoreFunction},
    },
    runtime::checkpoint::quantization::WeightQuantization,
};

use super::model::ModelArgs;

pub(crate) fn projection_format(args: &ModelArgs) -> WeightFormat {
    if args.quantization_config.is_some() {
        WeightFormat::Fp8E8M0
    } else {
        WeightFormat::Dense
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct Mlp {
    #[param]
    pub(crate) gate_proj: Linear,
    #[param]
    pub(crate) up_proj: Linear,
    #[param]
    pub(crate) down_proj: Linear,
    pub(crate) swiglu_limit: f32,
}

impl Mlp {
    pub(crate) fn new(
        args: &ModelArgs,
        intermediate_size: i32,
        limited: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let format = projection_format(args);
        Ok(Self {
            gate_proj: Linear::new(args.hidden_size, intermediate_size, false, format, stream)?,
            up_proj: Linear::new(args.hidden_size, intermediate_size, false, format, stream)?,
            down_proj: Linear::new(intermediate_size, args.hidden_size, false, format, stream)?,
            swiglu_limit: if limited { args.swiglu_limit } else { 0.0 },
        })
    }

    pub(crate) fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mut gate = self.gate_proj.forward(input, stream)?;
        let mut up = self.up_proj.forward(input, stream)?;
        if self.swiglu_limit > 0.0 {
            gate = safemlx::ops::minimum(gate, Array::from_f32(self.swiglu_limit), stream)?;
            up = safemlx::ops::clip(up, (-self.swiglu_limit, self.swiglu_limit), stream)?;
        }
        let activated = silu(gate, stream)?.multiply(up, stream)?;
        self.down_proj.forward(&activated, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct MoeGate {
    #[param]
    pub(crate) router: TopKRouter,
    #[param]
    pub(crate) tid2eid: Param<Option<Array>>,
}

impl MoeGate {
    fn new(args: &ModelArgs, layer_index: i32, stream: &Stream) -> Result<Self, Exception> {
        let hash = layer_index < args.num_hash_layers;
        Ok(Self {
            router: TopKRouter::new(
                TopKRouterConfig {
                    top_k: args.num_experts_per_tok,
                    num_experts: args.n_routed_experts,
                    hidden_size: args.hidden_size,
                    score_function: TopKRouterScoreFunction::SqrtSoftplus,
                    norm_topk_prob: args.norm_topk_prob,
                    normalization_epsilon: 1e-20,
                    routed_scaling_factor: args.routed_scaling_factor,
                    n_group: 1,
                    topk_group: 1,
                    score_correction_bias: !hash,
                },
                stream,
            )?,
            tid2eid: if hash {
                Param::<Option<Array>>::unloaded_some(
                    &[args.vocab_size, args.num_experts_per_tok],
                    Dtype::Int32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
        })
    }

    fn forward(
        &mut self,
        hidden: &Array,
        input_ids: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        if let Some(table) = self.tid2eid.as_ref() {
            let ids = input_ids
                .reshape(&[-1], stream)?
                .as_dtype(Dtype::Uint32, stream)?;
            let experts = take(table, &ids, stream)?;
            self.router
                .forward_with_routing_indices(hidden, &experts, stream)
        } else {
            self.router.forward(hidden, stream)
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct Moe {
    #[param]
    pub(crate) gate: MoeGate,
    #[param]
    pub(crate) switch_mlp: PackedSwiGluExperts,
    #[param]
    pub(crate) shared_experts: Mlp,
}

impl Moe {
    pub(crate) fn new(
        args: &ModelArgs,
        layer_index: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let expert_quantization = match args.expert_dtype.as_deref() {
            Some("fp4") => Some(WeightQuantization::MxFp4),
            // Native FP8 expert banks are bound through the block-FP8 packed
            // expert recipes; dense here is the unloaded parameter fallback.
            Some("fp8") | None => None,
            Some(_) => unreachable!("validated expert dtype"),
        };
        Ok(Self {
            gate: MoeGate::new(args, layer_index, stream)?,
            switch_mlp: PackedSwiGluExperts::new(
                args.n_routed_experts,
                args.hidden_size,
                args.moe_intermediate_size,
                expert_quantization,
                expert_quantization,
                stream,
            )?
            .with_swiglu_limit(args.swiglu_limit)?,
            shared_experts: Mlp::new(
                args,
                args.moe_intermediate_size * args.n_shared_experts,
                false,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        input: &Array,
        input_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = input.shape();
        let flat = input.reshape(&[-1, input.dim(-1)], stream)?;
        let (indices, weights) = self.gate.forward(&flat, input_ids, stream)?;
        let routed = self.switch_mlp.forward(&flat, &indices, &weights, stream)?;
        let shared = self.shared_experts.forward(&flat, stream)?;
        routed.add(shared, stream)?.reshape(shape, stream)
    }
}

pub(crate) fn rms_norm(size: i32, epsilon: f32, stream: &Stream) -> Result<nn::RmsNorm, Exception> {
    nn::RmsNorm::unloaded(size, epsilon, Dtype::Float32, stream)
}
