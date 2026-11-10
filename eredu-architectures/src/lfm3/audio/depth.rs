//! Bidirectional codebook scoring, masked refinement and speech feedback.

use super::super::{
    block::{normalization_spec, ReplicatedBlock},
    config::{model_args_for_attention, OperatorPolicy},
    declarations::BlockParameters,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    schema::{
        CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint,
    },
    store::TensorSelection,
    validation::ResolvedCheckpointPlan,
    LinearFormat,
};
use eredu_core::AttentionPolicy;
use eredu_nn::{
    DeclareLogicalParameters, Error, Index, LinearFormatSpec, LinearOperator, LinearSpec,
    LogicalParameterDeclaration, LogicalParameterRole, NeuralBackend, NeuralOperatorCapabilities,
    NormalizationConstructionSpec, NormalizationOperator, Parameter, ParameterSpec, Parameterized,
    Tensor, TensorElementType,
};
use serde_json::Value;
use std::collections::BTreeMap;

const ROOT: &str = "model.depth_decoder";
const FEEDBACK: &str = "model.codebook_embedding.weight";
const PLAN: &str = "LFM3 Audio speech output";

fn invalid(message: impl Into<String>) -> Error {
    Error::backend(message.into())
}

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name.into()).map_err(Error::backend)
}

/// Independent logical ownership of the speech-output parameters.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DepthUnit {
    /// Projects the language hidden state and supplies the learned mask vector.
    Conditioning,
    /// One tied codebook embedding, score head and head normalization.
    Codebook(usize),
    /// One bidirectional LFM3 attention block.
    Block(usize),
    /// Depth trunk final normalization.
    Normalization,
    /// Separate codebook table used as feedback by the language model.
    Feedback,
}

#[derive(Debug, Clone)]
struct CodebookSpec {
    table: ParameterSpec,
    scale: ParameterSpec,
}

/// Retained configuration and parameter scopes for speech output.
#[derive(Debug, Clone)]
pub struct DepthDeclaration {
    projection: LinearSpec,
    mask: ParameterSpec,
    blocks: Vec<BlockParameters>,
    final_norm: NormalizationConstructionSpec,
    codebooks: Vec<CodebookSpec>,
    feedback: ParameterSpec,
    vocabulary: i32,
    width: i32,
    eos: i32,
    schedule: Vec<usize>,
}

impl DepthDeclaration {
    /// Normalizes the released training-export or typed Audio configuration.
    pub fn from_config(config: &Value) -> Result<Self, Error> {
        if config["model_type"] != "lfm3_audio" {
            return Err(invalid("speech depth requires an LFM3 Audio configuration"));
        }
        let legacy = config.get("lfm").is_some();
        let (text, depth, books_key, vocab_key) = if legacy {
            (
                &config["lfm"],
                &config["depthformer"],
                "codebooks",
                "audio_vocab_size",
            )
        } else {
            (
                &config["text_config"],
                &config["depth_config"],
                "num_codebooks",
                "codebook_size",
            )
        };
        let integer = |value: &Value, key: &str, default: Option<i32>| {
            match value.get(key) {
                Some(v) => v.as_i64().and_then(|n| i32::try_from(n).ok()),
                None => default,
            }
            .ok_or_else(|| invalid(format!("invalid LFM3 speech setting {key}")))
        };
        let books = integer(config, books_key, Some(12))?;
        let vocabulary = integer(config, vocab_key, Some(2017))?;
        let eos = if legacy && config.get(vocab_key).is_some() {
            vocabulary
                .checked_sub(1)
                .ok_or_else(|| invalid("invalid speech vocabulary"))?
        } else {
            integer(config, "audio_eos_code", Some(2016))?
        };
        if (books, vocabulary, eos) != (12, 2017, 2016) {
            return Err(invalid("LFM3 Audio speech output requires twelve LiquidCodec codebooks and the released EOS code"));
        }
        let input = integer(text, "hidden_size", None)?;
        let args = model_args_for_attention(depth, false).map_err(Error::backend_source)?;
        if input <= 0
            || args.max_position_embeddings < books
            || args.weight_quantization.is_some()
            || args
                .layer_schedule
                .iter()
                .any(|p| p.operator != OperatorPolicy::SelfAttention(AttentionPolicy::Full))
        {
            return Err(invalid(
                "LFM3 Audio depth requires dense bidirectional full attention over the codebooks",
            ));
        }
        let schedule = match config.get("maskgit_schedule").filter(|v| !v.is_null()) {
            Some(value) => serde_json::from_value::<Vec<usize>>(value.clone())
                .map_err(Error::backend_source)?,
            None => {
                let passes = integer(config, "maskgit_steps", Some(3))?;
                if !(1..=books).contains(&passes) {
                    return Err(invalid("LFM3 Audio refinement pass count is out of range"));
                }
                let mut schedule = vec![(books / passes) as usize; passes as usize];
                schedule[passes as usize - 1] += (books % passes) as usize;
                schedule
            }
        };
        if schedule.contains(&0)
            || schedule
                .iter()
                .try_fold(0usize, |sum, count| sum.checked_add(*count))
                != Some(books as usize)
        {
            return Err(invalid(
                "LFM3 Audio refinement must reveal every codebook exactly once",
            ));
        }

        let width = args.hidden_size;
        let projection = LinearSpec {
            input,
            output: books
                .checked_mul(width)
                .ok_or_else(|| invalid("depth projection width overflowed"))?,
            weight: parameter(format!("{ROOT}.depth_linear.weight"))?,
            bias: Some(parameter(format!("{ROOT}.depth_linear.bias"))?),
            format: LinearFormatSpec::unscaled(LinearFormat::Dense)?,
        };
        let blocks = (0..args.layer_schedule.len())
            .map(|i| BlockParameters::at(&args, i, &format!("{ROOT}.model.layers.{i}")))
            .collect::<Result<Vec<_>, _>>()?;
        let codebooks = (0..books)
            .map(|i| {
                let root = format!("{ROOT}.depth_embeddings.{i}");
                Ok(CodebookSpec {
                    table: parameter(format!("{root}.embedding.weight"))?,
                    scale: parameter(format!("{root}.embedding_norm.weight"))?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let declaration = Self {
            projection,
            mask: parameter(format!("{ROOT}.mask_embedding"))?,
            final_norm: normalization_spec(
                &args,
                &format!("{ROOT}.model.embedding_norm.weight"),
                width,
            )?,
            blocks,
            codebooks,
            feedback: parameter(FEEDBACK)?,
            vocabulary,
            width,
            eos,
            schedule,
        };
        eredu_nn::validate_logical_parameter_topology(&declaration.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(declaration)
    }

    /// Number of new codebooks to reveal per complete depth pass.
    pub fn schedule(&self) -> &[usize] {
        &self.schedule
    }

    /// Vocabulary size including the speech end code.
    pub fn vocabulary(&self) -> i32 {
        self.vocabulary
    }

    /// Speech end code, outside the LiquidCodec synthesis vocabulary.
    pub fn eos_code(&self) -> i32 {
        self.eos
    }

    /// Parameter ownership scopes, with tied tables listed once.
    pub fn units(&self) -> impl Iterator<Item = DepthUnit> {
        std::iter::once(DepthUnit::Conditioning)
            .chain((0..self.blocks.len()).map(DepthUnit::Block))
            .chain(std::iter::once(DepthUnit::Normalization))
            .chain((0..self.codebooks.len()).map(DepthUnit::Codebook))
            .chain(std::iter::once(DepthUnit::Feedback))
    }

    /// Declared parameters for one scope.
    pub fn unit_parameters(
        &self,
        unit: DepthUnit,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match unit {
            DepthUnit::Conditioning => {
                let mut parameters = self.projection.logical_parameters()?;
                parameters.push(LogicalParameterDeclaration::new(
                    &self.mask,
                    &[self.width],
                    LogicalParameterRole::Embedding,
                )?);
                Ok(parameters)
            }
            DepthUnit::Codebook(i) => {
                let spec = self
                    .codebooks
                    .get(i)
                    .ok_or_else(|| invalid("unknown speech codebook"))?;
                Ok(vec![
                    LogicalParameterDeclaration::new(
                        &spec.table,
                        &[self.vocabulary, self.width],
                        LogicalParameterRole::Codebook,
                    )?,
                    LogicalParameterDeclaration::new(
                        &spec.scale,
                        &[self.width],
                        LogicalParameterRole::Normalization,
                    )?,
                ])
            }
            DepthUnit::Block(i) => self
                .blocks
                .get(i)
                .ok_or_else(|| invalid("unknown speech depth block"))?
                .logical_parameters(),
            DepthUnit::Normalization => self.final_norm.logical_parameters(),
            DepthUnit::Feedback => Ok(vec![LogicalParameterDeclaration::new(
                &self.feedback,
                &[
                    self.codebooks.len() as i32 * self.vocabulary,
                    self.projection.input,
                ],
                LogicalParameterRole::Codebook,
            )?]),
        }
    }

    /// Admits canonical and released SafeTensors names for the speech output path.
    pub fn safetensors_plan(&self) -> Result<SafetensorsCheckpointPlan, Error> {
        let tensors = self
            .logical_parameters()?
            .into_iter()
            .map(|p| {
                let name = p.parameter.id.to_string();
                let alias = released_alias(&name);
                SafetensorsTensorConstraint::required(
                    name,
                    p.shape,
                    StoredDtypeConstraint::Floating,
                )
                .with_aliases(alias)
            })
            .collect();
        let mut catalog = CatalogPolicy::non_strict();
        // The publisher ignores these redundant tied-head tensors when loading.
        catalog.explicitly_allowed_keys.extend(
            (0..self.codebooks.len()).map(|i| format!("depth_embeddings.{i}.to_logits.weight")),
        );
        SafetensorsCheckpointPlan::new(PLAN, tensors, vec![], catalog)
            .map_err(Error::backend_source)
    }

    /// Maps admitted aliases to canonical destinations without opening payloads.
    pub fn safetensors_recipes(
        &self,
        resolution: &ResolvedCheckpointPlan,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        if resolution.identity() != PLAN {
            return Err(invalid(
                "checkpoint resolution does not belong to LFM3 speech output",
            ));
        }
        let mut recipes = BTreeMap::new();
        for parameter in self.logical_parameters()? {
            let name = parameter.parameter.id.to_string();
            if !resolution.source_keys().contains(&name) {
                let source = released_alias(&name)
                    .filter(|alias| resolution.source_keys().contains(alias))
                    .ok_or_else(|| invalid(format!("no admitted LFM3 speech source for {name}")))?;
                recipes.insert(
                    name,
                    DerivedWeightRecipe::source(source, TensorSelection::Full),
                );
            }
        }
        Ok(recipes)
    }

    /// Starts one frame using a permutation drawn by the request's sampling owner.
    pub fn begin_frame(&self, order: &[usize]) -> Result<DepthFrame, Error> {
        let count = self.codebooks.len();
        let mut seen = vec![false; count];
        if order.len() != count
            || order
                .iter()
                .any(|&index| index >= count || std::mem::replace(&mut seen[index], true))
        {
            return Err(invalid(
                "depth order must be a permutation of all codebooks",
            ));
        }
        Ok(DepthFrame {
            order: order.to_vec(),
            codes: vec![0; count],
            masked: vec![1; count],
            schedule: self.schedule.clone(),
            vocabulary: self.vocabulary,
            eos: self.eos,
            pass: 0,
            revealed: 0,
        })
    }
}

fn released_alias(name: &str) -> Option<String> {
    if name == FEEDBACK {
        return Some("codebook_embedding.weight".into());
    }
    let suffix = name.strip_prefix("model.depth_decoder.")?;
    if let Some(trunk) = suffix.strip_prefix("model.") {
        return Some(format!("depthformer.model.{trunk}"));
    }
    Some(suffix.into())
}

impl DeclareLogicalParameters for DepthDeclaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = Vec::new();
        for unit in self.units() {
            parameters.extend(self.unit_parameters(unit)?);
        }
        Ok(parameters)
    }
}

/// Mutable state of an unfinished speech frame. Each pass publishes all its slots together.
#[derive(Debug, Clone)]
pub struct DepthFrame {
    order: Vec<usize>,
    codes: Vec<i32>,
    masked: Vec<i32>,
    schedule: Vec<usize>,
    vocabulary: i32,
    eos: i32,
    pass: usize,
    revealed: usize,
}

impl eredu_runtime::execution_control::SnapshotValue for DepthFrame {
    fn estimate_snapshot_value(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, eredu_core::BackendFailure> {
        eredu_runtime::execution_control::snapshot_value_estimate::<Self>([
            self.order.estimate_snapshot_value(),
            self.codes.estimate_snapshot_value(),
            self.masked.estimate_snapshot_value(),
            self.schedule.estimate_snapshot_value(),
        ])
    }

    fn copy_snapshot_value(
        &self,
        retention: &eredu_runtime::execution_control::SnapshotReservation,
    ) -> Result<Self, eredu_core::BackendFailure> {
        Ok(Self {
            order: self.order.copy_snapshot_value(retention)?,
            codes: self.codes.copy_snapshot_value(retention)?,
            masked: self.masked.copy_snapshot_value(retention)?,
            schedule: self.schedule.copy_snapshot_value(retention)?,
            ..*self
        })
    }
}

/// Complete speech frame, with the EOS decision separated from codec input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletedSpeechFrame<'a> {
    /// Codec codes in codebook order.
    Audio(&'a [i32]),
    /// The publisher ends the segment when any slot predicts EOS.
    End,
}

impl DepthFrame {
    /// Returns the pass index and slots eligible to reveal on this pass.
    pub fn next_pass(&self) -> Option<(usize, &[usize])> {
        let count = *self.schedule.get(self.pass)?;
        Some((self.pass, &self.order[self.revealed..self.revealed + count]))
    }

    /// Commits a complete sample group in the slot order returned by next_pass.
    /// A malformed or stale pass leaves every slot unchanged.
    pub fn reveal(&mut self, pass: usize, sampled: &[i32]) -> Result<(), Error> {
        let Some((current, slots)) = self.next_pass() else {
            return Err(invalid("depth frame is already complete"));
        };
        if pass != current
            || sampled.len() != slots.len()
            || sampled
                .iter()
                .any(|&code| !(0..self.vocabulary).contains(&code))
        {
            return Err(invalid("invalid depth pass or sampled code group"));
        }
        for (&slot, &code) in self.order[self.revealed..self.revealed + sampled.len()]
            .iter()
            .zip(sampled)
        {
            self.codes[slot] = code;
            self.masked[slot] = 0;
        }
        self.revealed += sampled.len();
        self.pass += 1;
        Ok(())
    }

    /// Publishes a frame only after every codebook has been revealed.
    pub fn completed(&self) -> Option<CompletedSpeechFrame<'_>> {
        if self.next_pass().is_some() {
            return None;
        }
        Some(if self.codes.contains(&self.eos) {
            CompletedSpeechFrame::End
        } else {
            CompletedSpeechFrame::Audio(&self.codes)
        })
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct Codebook<T: Tensor> {
    table: Parameter<T>,
    scale: Parameter<T>,
}

/// Resident speech depth decoder. Each invocation attends across all slots with no KV history.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DepthDecoder<B: NeuralBackend> {
    projection: B::Linear,
    mask: Parameter<B::Tensor>,
    blocks: Vec<ReplicatedBlock<B>>,
    final_norm: B::Normalization,
    codebooks: Vec<Codebook<B::Tensor>>,
    #[parameter(skip)]
    declaration: DepthDeclaration,
}

fn lookup_capabilities() -> NeuralOperatorCapabilities {
    NeuralOperatorCapabilities::FULL_I32
        .union(NeuralOperatorCapabilities::EQUAL_I32)
        .union(NeuralOperatorCapabilities::WHERE_CONDITION)
}

impl<B: NeuralBackend> DepthDecoder<B> {
    /// Constructs unloaded resident execution from retained parameter declarations.
    pub fn new(
        declaration: &DepthDeclaration,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        B::require_operator_capabilities(
            "LFM3 speech depth",
            lookup_capabilities()
                .union(NeuralOperatorCapabilities::FROM_I32_SLICE)
                .union(NeuralOperatorCapabilities::CAST_FLOAT)
                .union(NeuralOperatorCapabilities::RMS_NORM_WITHOUT_WEIGHT)
                .union(NeuralOperatorCapabilities::SIGMOID),
        )?;
        let d = declaration;
        Ok(Self {
            projection: B::linear(d.projection.clone(), context)?,
            mask: Parameter::unloaded(d.mask.clone(), &[d.width], context)?,
            blocks: d
                .blocks
                .iter()
                .cloned()
                .map(|parameters| ReplicatedBlock::from_parameters(parameters, context))
                .collect::<Result<_, _>>()?,
            final_norm: B::normalization(d.final_norm.clone(), context)?,
            codebooks: d
                .codebooks
                .iter()
                .map(|spec| {
                    Ok(Codebook {
                        table: Parameter::unloaded(
                            spec.table.clone(),
                            &[d.vocabulary, d.width],
                            context,
                        )?,
                        scale: Parameter::unloaded(spec.scale.clone(), &[d.width], context)?,
                    })
                })
                .collect::<Result<_, Error>>()?,
            declaration: d.clone(),
        })
    }

    /// Scores every codebook over complete bidirectional slot context.
    /// Codes use I32 and the same shape as the boolean mask `[frames, codebooks]`.
    pub fn scores(
        &mut self,
        hidden: &B::Tensor,
        codes: &B::Tensor,
        masked: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let d = &self.declaration;
        let books = d.codebooks.len() as i32;
        if hidden.shape().len() != 2
            || hidden.dim(0) <= 0
            || hidden.dim(1) != d.projection.input
            || codes.shape() != [hidden.dim(0), books]
            || masked.shape() != codes.shape()
            || codes.element_type() != Some(TensorElementType::I32)
            || masked.element_type() != Some(TensorElementType::Bool)
        {
            return Err(invalid("invalid LFM3 speech depth inputs"));
        }

        let codes = nonnegative_gather_indices(codes, d.vocabulary, context)?;
        let embeddings = self
            .codebooks
            .iter()
            .enumerate()
            .map(|(i, book)| {
                let ids = codes.index(&[Index::Full, Index::At(i as i32)], context)?;
                book.table.as_ref().take_axis(&ids, 0, context)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let embeddings = B::Tensor::stack(&embeddings, 1, context)?;
        let dtype = embeddings
            .element_type()
            .ok_or_else(|| invalid("unknown speech codebook dtype"))?;
        let mask = self.mask.as_ref().cast_float(dtype, context)?;
        let conditioning = B::Tensor::where_condition(
            &masked.expand_dims(-1, context)?,
            &mask,
            &embeddings,
            context,
        )?;
        let mut depth = self
            .projection
            .forward(hidden, context)?
            .reshape(&[hidden.dim(0), books, d.width], context)?
            .add(&conditioning, context)?;
        for block in &mut self.blocks {
            depth = block.forward_bidirectional(&depth, context)?;
        }
        let depth = self.final_norm.forward(&depth, context)?;
        let dtype = depth
            .element_type()
            .ok_or_else(|| invalid("unknown speech depth activation dtype"))?;
        let logits = self
            .codebooks
            .iter()
            .enumerate()
            .map(|(i, book)| {
                let slot = depth
                    .index(&[Index::Full, Index::At(i as i32)], context)?
                    .cast_float(TensorElementType::F32, context)?;
                let scale = book
                    .scale
                    .as_ref()
                    .cast_float(TensorElementType::F32, context)?;
                // The trained heads multiply the learned scale before the single output cast.
                let slot = B::rms_norm_with_weight(&slot, &scale, d.final_norm.epsilon, context)?
                    .cast_float(dtype, context)?;
                B::Tensor::linear(&slot, book.table.as_ref(), None, context)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        B::Tensor::stack(&logits, 1, context)
    }

    /// Scores the next pass of one unfinished frame without committing any sampled codes.
    pub fn score_frame(
        &mut self,
        hidden: &B::Tensor,
        frame: &DepthFrame,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let d = &self.declaration;
        if frame.next_pass().is_none()
            || frame.schedule != d.schedule
            || frame.eos != d.eos
            || frame.vocabulary != d.vocabulary
            || frame.codes.len() != d.codebooks.len()
        {
            return Err(invalid(
                "speech refinement frame is complete or incompatible",
            ));
        }
        let shape = [1, frame.codes.len() as i32];
        let codes = B::Tensor::from_i32_slice(&frame.codes, &shape, context)?;
        let mask =
            B::Tensor::from_i32_slice(&frame.masked, &shape, context)?.equal_i32(1, context)?;
        self.scores(hidden, &codes, &mask, context)
    }

    /// Scores and commits one pass using the caller's sampling policy.
    /// Sampling errors leave the frame untouched. The sampler owns its own rollback and RNG.
    pub fn advance(
        &mut self,
        hidden: &B::Tensor,
        frame: &mut DepthFrame,
        mut sample: impl FnMut(usize, &B::Tensor) -> Result<i32, Error>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        let logits = self.score_frame(hidden, frame, context)?;
        let (pass, slots) = frame
            .next_pass()
            .ok_or_else(|| invalid("speech depth frame is complete"))?;
        let codes = slots
            .iter()
            .map(|&slot| {
                let row = logits.index(&[Index::At(0), Index::At(slot as i32)], context)?;
                sample(slot, &row)
            })
            .collect::<Result<Vec<_>, _>>()?;
        frame.reveal(pass, &codes)
    }
}

/// Learned codebook feedback in the language model's embedding space.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Feedback<B: NeuralBackend> {
    weight: Parameter<B::Tensor>,
    #[parameter(skip)]
    codebooks: i32,
    #[parameter(skip)]
    vocabulary: i32,
    #[parameter(skip)]
    width: i32,
}

impl<B: NeuralBackend> Feedback<B> {
    /// Builds the independently bound speech feedback table.
    pub fn new(
        declaration: &DepthDeclaration,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        B::require_operator_capabilities("LFM3 speech feedback", lookup_capabilities())?;
        let codebooks = declaration.codebooks.len() as i32;
        let vocabulary = declaration.vocabulary;
        let width = declaration.projection.input;
        Ok(Self {
            weight: Parameter::unloaded(
                declaration.feedback.clone(),
                &[codebooks * vocabulary, width],
                context,
            )?,
            codebooks,
            vocabulary,
            width,
        })
    }

    /// Sums language-width codebook embeddings in each complete I32 frame.
    pub fn embed(
        &self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if codes.shape().len() != 2
            || codes.dim(0) <= 0
            || codes.dim(1) != self.codebooks
            || codes.element_type() != Some(TensorElementType::I32)
        {
            return Err(invalid("invalid LFM3 speech feedback codes"));
        }
        // Index each table independently so an invalid code cannot select a neighbor table.
        let codes = nonnegative_gather_indices(codes, self.vocabulary, context)?;
        let table = self
            .weight
            .as_ref()
            .reshape(&[self.codebooks, self.vocabulary, self.width], context)?;
        let embeddings = (0..self.codebooks)
            .map(|slot| {
                let ids = codes.index(&[Index::Full, Index::At(slot)], context)?;
                table
                    .index(&[Index::At(slot)], context)?
                    .take_axis(&ids, 0, context)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        B::Tensor::sum_axis(
            &B::Tensor::stack(&embeddings, 1, context)?,
            1,
            false,
            context,
        )
    }
}

// Ordinary gathers allow negative indices to wrap. Route them to the strict
// positive out-of-range sentinel before gathering, including under deferred validation.
fn nonnegative_gather_indices<T: Tensor>(
    codes: &T,
    vocabulary: i32,
    context: &T::Context,
) -> Result<T, Error> {
    let valid = codes
        .subtract(&codes.maximum_i32(0, context)?, context)?
        .equal_i32(0, context)?;
    T::where_condition(
        &valid,
        codes,
        &T::full_i32(vocabulary, &[], context)?,
        context,
    )
}
