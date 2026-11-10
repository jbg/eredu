//! Graph-visible embedded multi-token prediction units.

use eredu_nn::{
    AttentionCache, Error, GroupedNeuralBackend, LinearSpec, NormalizationOperator, Parameterized,
    Tensor,
};
use eredu_runtime::{ParameterProvider, RuntimeStateComponents};

use crate::decoder::ComponentInstrumentation;

use super::{Block, LayerGeometry, LayerPolicy, ModelArgs};

/// Boundary slots and invocation identity of one physical prediction unit.
/// Intermediate units have no boundary parameters; their block remains an
/// independent invocation with its own mutable state and placement.
#[derive(Debug, Clone)]
pub(crate) struct PredictionUnitParameters {
    pub physical: usize,
    pub policy: LayerPolicy,
    pub block: super::declarations::TargetBlockParameters,
    pub embedding_norm: Option<eredu_nn::NormalizationConstructionSpec>,
    pub hidden_norm: Option<eredu_nn::NormalizationConstructionSpec>,
    pub fusion: Option<LinearSpec>,
    pub final_norm: Option<eredu_nn::NormalizationConstructionSpec>,
}

impl PredictionUnitParameters {
    pub(crate) fn all(args: &ModelArgs) -> Result<Vec<Self>, Error> {
        let steps = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        if steps == 0 {
            return Ok(Vec::new());
        }
        let count = args.mtp_policies().map_err(Error::backend)?.len() / steps;
        (0..steps)
            .flat_map(|depth| (0..count).map(move |relative| (depth, relative)))
            .map(|(depth, relative)| Self::new(args, depth, relative))
            .collect()
    }
    pub(crate) fn with_formats(mut self, args: &ModelArgs) -> Result<Self, Error> {
        self.block = self.block.with_formats(args)?;
        if let Some(fusion) = &mut self.fusion {
            fusion.format = crate::linear_format::project_matrix_format(
                &fusion.weight,
                &fusion.format,
                args.weight_quantization_for(fusion.weight.id.as_str())
                    .into(),
            )?;
        }
        Ok(self)
    }
    pub fn new(args: &ModelArgs, depth: usize, relative: usize) -> Result<Self, Error> {
        let steps = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        let policies = args.mtp_policies().map_err(Error::backend)?;
        let count = policies
            .len()
            .checked_div(steps)
            .filter(|count| *count > 0)
            .ok_or_else(|| Error::backend("Nemotron-H MTP pattern is empty"))?;
        if depth >= steps || relative >= count {
            return Err(Error::backend(
                "Nemotron-H MTP unit is outside its schedule",
            ));
        }
        let physical = depth
            .checked_mul(count)
            .and_then(|start| start.checked_add(relative))
            .ok_or_else(|| Error::backend("Nemotron-H MTP physical index overflowed"))?;
        let root = format!("model.mtp.layers.{physical}");
        let norm =
            |field: &str| super::block::normalization_spec(args, &format!("{root}.{field}.weight"));
        let first = relative == 0;
        let parameters = Self {
            physical,
            policy: policies[physical],
            block: super::declarations::TargetBlockParameters::at(
                args,
                &root,
                policies[physical],
                (args.num_hidden_layers as usize)
                    .checked_add(physical)
                    .ok_or_else(|| Error::backend("prediction identity overflowed"))?,
                true,
            )?,
            embedding_norm: first.then(|| norm("enorm")).transpose()?,
            hidden_norm: first.then(|| norm("hnorm")).transpose()?,
            fusion: first
                .then(|| {
                    let input = args.hidden_size.checked_mul(2).ok_or_else(|| {
                        Error::backend("Nemotron prediction fusion width overflowed")
                    })?;
                    super::attention::linear_spec(
                        args,
                        &format!("{root}.eh_proj"),
                        input,
                        args.hidden_size,
                        false,
                    )
                })
                .transpose()?,
            final_norm: (relative + 1 == count)
                .then(|| norm("final_layernorm"))
                .transpose()?,
        };
        use eredu_nn::DeclareLogicalParameters;
        parameters.logical_parameters()?;
        Ok(parameters)
    }
}

impl PredictionUnitParameters {
    pub(crate) fn boundary_parameters(
        &self,
    ) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        use eredu_nn::DeclareLogicalParameters;
        let mut parameters = Vec::new();
        for norm in self.embedding_norm.iter().chain(self.hidden_norm.iter()) {
            parameters.extend(norm.logical_parameters()?);
        }
        if let Some(fusion) = &self.fusion {
            parameters.extend(fusion.logical_parameters()?);
        }
        if let Some(norm) = &self.final_norm {
            parameters.extend(norm.logical_parameters()?);
        }
        eredu_nn::validate_logical_parameter_scope(&parameters).map_err(Error::backend)?;
        Ok(parameters)
    }
}

impl eredu_nn::DeclareLogicalParameters for PredictionUnitParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.boundary_parameters()?;
        parameters.extend(self.block.logical_parameters()?);
        eredu_nn::validate_logical_parameter_scope(&parameters).map_err(Error::backend)?;
        Ok(parameters)
    }
}

/// Output selected for a target invocation; prediction passes retain vocabulary readout.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum TargetReadout {
    /// Apply final normalization and the existing vocabulary projection.
    #[default]
    Logits,
    /// Return final normalized hidden states without evaluating the vocabulary head.
    NormalizedHidden,
}

impl TargetReadout {
    /// Applies the shared target equation through borrowed canonical decoder roles.
    pub(crate) fn apply<B: eredu_nn::DistributedNeuralBackend>(
        self,
        mut resources: crate::decoder::DecoderResourcesMut<'_, B>,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = resources.normalize(hidden, context, instrumentation)?;
        match self {
            Self::NormalizedHidden => Ok(hidden),
            Self::Logits => resources.project(&hidden, parallel, context, instrumentation),
        }
    }

    /// Encodes invocation policy as an exact pipeline control tensor.
    pub fn encode<T: Tensor>(self, context: &T::Context) -> Result<T, Error> {
        T::from_i32_slice(
            &[match self {
                Self::Logits => 0,
                Self::NormalizedHidden => 1,
            }],
            &[1],
            context,
        )
    }

    /// Validates and decodes one exact invocation control received at a partition.
    pub fn decode<T: Tensor>(value: &T, context: &T::Context) -> Result<Self, Error> {
        if value.shape() != [1] || value.element_type() != Some(eredu_nn::TensorElementType::I32) {
            return Err(Error::backend(
                "Nemotron-H readout control requires exact I32[1]",
            ));
        }
        match value.to_i32_vec(context)?.as_slice() {
            [0] => Ok(Self::Logits),
            [1] => Ok(Self::NormalizedHidden),
            _ => Err(Error::backend("invalid Nemotron-H target readout control")),
        }
    }
}

pub(crate) fn validate_target_embeddings<T: Tensor>(
    tokens: &T,
    embedded: &T,
    width: i32,
) -> Result<(), Error> {
    let ids = tokens.shape();
    let shape = embedded.shape();
    if ids.len() != 2 || ids[0] <= 0 || ids[1] <= 0 || shape != [ids[0], ids[1], width] {
        return Err(Error::backend("Nemotron-H supplied target embeddings must match [batch, sequence, hidden] and token geometry"));
    }
    Ok(())
}

/// Borrowed input selecting target execution or one MTP prediction depth.
pub enum EmbeddedInput<'a, T> {
    /// Execute the complete target schedule.
    Target {
        /// Token ids shaped `[batch, sequence]`.
        tokens: &'a T,
        /// Optional caller-provided attention mask.
        mask: Option<&'a T>,
    },
    /// Execute target layers over caller-supplied embeddings without vocabulary lookup.
    TargetEmbedded {
        /// Logical token identities retained for target/MTP and pipeline context.
        tokens: &'a T,
        /// Input hidden states shaped `[batch, sequence, hidden]`.
        embedded: &'a T,
        /// Optional caller-provided attention mask.
        mask: Option<&'a T>,
        /// Final target readout policy, propagated across pipeline boundaries.
        readout: TargetReadout,
    },
    /// Execute exactly one appended MTP prediction group.
    Draft {
        /// Token ids embedded for the proposed position.
        tokens: &'a T,
        /// Target or prior prediction hidden state.
        hidden: &'a T,
        /// Zero-based MTP prediction depth.
        depth: usize,
    },
}

impl<'a, T> EmbeddedInput<'a, T> {
    /// Creates target-model input.
    pub const fn target(tokens: &'a T, mask: Option<&'a T>) -> Self {
        Self::Target { tokens, mask }
    }

    /// Creates supplied-embedding target input with an explicit readout policy.
    pub const fn target_embedded(
        tokens: &'a T,
        embedded: &'a T,
        mask: Option<&'a T>,
        readout: TargetReadout,
    ) -> Self {
        Self::TargetEmbedded {
            tokens,
            embedded,
            mask,
            readout,
        }
    }

    /// Creates one MTP draft input.
    pub const fn draft(tokens: &'a T, hidden: &'a T, depth: usize) -> Self {
        Self::Draft {
            tokens,
            hidden,
            depth,
        }
    }
}

/// Execution selection retained for one model invocation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ForwardMode {
    /// Target physical units execute.
    Target,
    /// Exactly one MTP depth executes.
    Draft(usize),
}

/// One physical operator inside an appended MTP prediction group.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PredictionUnit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// First-unit normalization of the current token embedding.
    pub embedding_norm: Option<B::Normalization>,
    /// First-unit normalization of the prior hidden state.
    pub hidden_norm: Option<B::Normalization>,
    /// First-unit projection of concatenated embedding and hidden state.
    pub fusion: Option<B::Linear>,
    /// Scheduled attention or sparse-MoE physical unit.
    pub block: Block<B>,
    /// Last-unit normalization before the shared vocabulary projection.
    pub final_norm: Option<B::Normalization>,
    #[parameter(skip)]
    experts: i32,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> PredictionUnit<B> {
    pub(crate) fn from_parameters(
        args: &ModelArgs,
        parameters: PredictionUnitParameters,
        geometry: Option<LayerGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let identity = (args.num_hidden_layers as usize)
            .checked_add(parameters.physical)
            .ok_or_else(|| Error::backend("prediction identity overflowed"))?;
        let block = match geometry {
            Some(geometry) => parameters.block.local(geometry, None)?,
            None => parameters.block,
        };
        Ok(Self {
            experts: args.n_routed_experts,
            embedding_norm: parameters
                .embedding_norm
                .map(|spec| B::normalization(spec, context))
                .transpose()?,
            hidden_norm: parameters
                .hidden_norm
                .map(|spec| B::normalization(spec, context))
                .transpose()?,
            fusion: parameters
                .fusion
                .map(|spec| B::linear(spec, context))
                .transpose()?,
            block: Block::from_scope(args, identity, parameters.policy, block, context)?,
            final_norm: parameters
                .final_norm
                .map(|spec| B::normalization(spec, context))
                .transpose()?,
        })
    }

    /// Admitted global expert count for routed observation geometry.
    pub(crate) fn expert_count(&self) -> i32 {
        self.experts
    }

    /// Executes one physical prediction unit with resident experts.
    pub fn forward<S>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.forward_with_provider(
            hidden,
            embedded,
            mask,
            state,
            context,
            &mut eredu_runtime::ResidentExpertProvider,
        )
    }

    /// Executes one physical prediction unit through a runtime expert provider.
    pub fn forward_with_provider<S, P>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_with_decoder(
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::disabled(),
            |block, fused, context, _| {
                block.forward_with_provider(fused, mask, state, context, provider)
            },
        )
    }

    /// Observes the actual fusion, scheduled operator and final normalization.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<S, P, O>(
        &mut self,
        path: &str,
        expert_count: i32,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_decoder(
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |block, fused, context, instrumentation| {
                block.forward_observed_with_provider(
                    path,
                    expert_count,
                    fused,
                    mask,
                    state,
                    context,
                    instrumentation.observer().expect("observed prediction"),
                    provider,
                )
            },
        )
    }

    /// Executes one physical prediction unit with tensor collectives.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider<S, P>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_with_decoder(
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::disabled(),
            |block, fused, context, _| {
                block.forward_parallel(fused, mask, state, parallel, context, provider)
            },
        )
    }

    /// Observes the same prediction graph with the ordinary TP reduction order.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed_with_provider<S, P, O>(
        &mut self,
        path: &str,
        expert_count: i32,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_decoder(
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |block, fused, context, instrumentation| {
                block.forward_parallel_observed_with_provider(
                    path,
                    expert_count,
                    fused,
                    mask,
                    state,
                    parallel,
                    context,
                    instrumentation.observer().expect("observed prediction"),
                    provider,
                )
            },
        )
    }

    fn forward_with_decoder<F>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        decoder: F,
    ) -> Result<B::Tensor, Error>
    where
        F: FnOnce(
            &mut Block<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let hidden = match (
            &mut self.embedding_norm,
            &mut self.hidden_norm,
            &mut self.fusion,
        ) {
            (Some(embedding_norm), Some(hidden_norm), Some(fusion)) => {
                instrumentation.observe("prediction.hidden", hidden)?;
                instrumentation.observe("prediction.embedding", embedded)?;
                let embedded = instrumentation.apply(
                    "prediction.embedding.normalized",
                    embedding_norm.forward(embedded, context)?,
                )?;
                let hidden = instrumentation.apply(
                    "prediction.hidden.normalized",
                    hidden_norm.forward(hidden, context)?,
                )?;
                let joined = B::Tensor::concatenate(&[embedded, hidden], -1, context)?;
                let fused = instrumentation.project::<B>(
                    "prediction.fusion.input",
                    fusion,
                    &joined,
                    None,
                    context,
                )?;
                instrumentation.apply("prediction.fusion.output", fused)?
            }
            (None, None, None) => hidden.clone(),
            _ => return Err(Error::backend("incomplete Nemotron-H MTP fusion unit")),
        };
        let hidden = decoder(&mut self.block, &hidden, context, instrumentation)?;
        match &mut self.final_norm {
            Some(norm) => {
                let hidden = instrumentation.apply("prediction.readout.residual", hidden)?;
                instrumentation.apply(
                    "prediction.readout.normalized",
                    norm.forward(&hidden, context)?,
                )
            }
            None => Ok(hidden),
        }
    }
}

/// Allocation-free retained tensor iterator for target and draft forwards.
pub struct RetainedValues<'a, T> {
    values: [Option<&'a T>; 5],
    next: usize,
}

impl<'a, T> RetainedValues<'a, T> {
    /// Creates the iterator from fixed request-local slots.
    pub const fn new(values: [Option<&'a T>; 5]) -> Self {
        Self { values, next: 0 }
    }
}

impl<'a, T> Iterator for RetainedValues<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.values.len() {
            let value = self.values[self.next];
            self.next += 1;
            if value.is_some() {
                return value;
            }
        }
        None
    }
}
