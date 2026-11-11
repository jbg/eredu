//! Bounded speech depth passes with tied codebook storage.

use super::*;
use crate::lfm3::audio::{placement::prepare_audio_units, ArtifactError, BindingSource};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_runtime::{
    ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddressMap, ExecutionUnitLayout,
    LayerwiseAcquireError, LayerwisePolicy, LayerwisePolicyForward, WeightBinding,
};
use std::{iter, marker::PhantomData};

/// One materializable owner in the depth and speech feedback graph.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DepthModule<B: NeuralBackend>(Module<B>);

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum Module<B: NeuralBackend> {
    Conditioning(Conditioning<B>),
    Codebook(Codebook<B::Tensor>),
    Block(ReplicatedBlock<B>),
    Normalization(B::Normalization),
    Feedback(Feedback<B>),
}

pub(super) enum Operation<'a, T> {
    Embed(&'a T, usize),
    Condition(&'a T, &'a T, &'a T),
    Block(&'a T),
    Normalize(&'a T),
    Score(&'a T, usize),
    Feedback(&'a T),
}

impl<'a, T> Operation<'a, T> {
    fn roots(&self) -> impl Iterator<Item = &'a T> {
        match *self {
            Self::Condition(hidden, masked, embeddings) => {
                [Some(hidden), Some(masked), Some(embeddings)]
            }
            Self::Embed(value, _)
            | Self::Block(value)
            | Self::Normalize(value)
            | Self::Score(value, _)
            | Self::Feedback(value) => [Some(value), None, None],
        }
        .into_iter()
        .flatten()
    }
}

impl<B: NeuralBackend> DepthModule<B> {
    fn new(
        d: &DepthDeclaration,
        unit: DepthUnit,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self(match unit {
            DepthUnit::Conditioning => Module::Conditioning(Conditioning {
                projection: B::linear(d.projection.clone(), context)?,
                mask: Parameter::unloaded(d.mask.clone(), &[d.width], context)?,
            }),
            DepthUnit::Codebook(index) => {
                let spec = &d.codebooks[index];
                Module::Codebook(Codebook {
                    table: Parameter::unloaded(
                        spec.table.clone(),
                        &[d.vocabulary, d.width],
                        context,
                    )?,
                    scale: Parameter::unloaded(spec.scale.clone(), &[d.width], context)?,
                })
            }
            DepthUnit::Block(index) => Module::Block(ReplicatedBlock::from_parameters(
                d.blocks[index].clone(),
                context,
            )?),
            DepthUnit::Normalization => {
                Module::Normalization(B::normalization(d.final_norm.clone(), context)?)
            }
            DepthUnit::Feedback => Module::Feedback(Feedback::new(d, context)?),
        }))
    }

    pub(super) fn apply(
        &mut self,
        d: &DepthDeclaration,
        operation: &Operation<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match (&mut self.0, operation) {
            (Module::Codebook(book), Operation::Embed(codes, i)) => {
                let ids = codes.index(&[Index::Full, Index::At(*i as i32)], context)?;
                book.table.as_ref().take_axis(&ids, 0, context)
            }
            (Module::Conditioning(module), Operation::Condition(hidden, masked, embeddings)) => {
                let dtype = embeddings
                    .element_type()
                    .ok_or_else(|| invalid("unknown speech codebook dtype"))?;
                let mask = module.mask.as_ref().cast_float(dtype, context)?;
                let conditioning = B::Tensor::where_condition(
                    &masked.expand_dims(-1, context)?,
                    &mask,
                    embeddings,
                    context,
                )?;
                module
                    .projection
                    .forward(hidden, context)?
                    .reshape(&[hidden.dim(0), d.codebooks.len() as i32, d.width], context)?
                    .add(&conditioning, context)
            }
            (Module::Block(block), Operation::Block(hidden)) => {
                block.forward_bidirectional(hidden, context)
            }
            (Module::Normalization(norm), Operation::Normalize(hidden)) => {
                norm.forward(hidden, context)
            }
            (Module::Codebook(book), Operation::Score(depth, i)) => {
                let dtype = depth
                    .element_type()
                    .ok_or_else(|| invalid("unknown speech depth activation dtype"))?;
                let slot = depth
                    .index(&[Index::Full, Index::At(*i as i32)], context)?
                    .cast_float(TensorElementType::F32, context)?;
                let scale = book
                    .scale
                    .as_ref()
                    .cast_float(TensorElementType::F32, context)?;
                // The trained heads apply the learned scale before the single output cast.
                let slot = B::rms_norm_with_weight(&slot, &scale, d.final_norm.epsilon, context)?
                    .cast_float(dtype, context)?;
                B::Tensor::linear(&slot, book.table.as_ref(), None, context)
            }
            (Module::Feedback(module), Operation::Feedback(codes)) => module.embed(codes, context),
            _ => Err(invalid("speech output module differs from its invocation")),
        }
    }
}

fn require_capabilities<B: NeuralBackend>() -> Result<(), Error> {
    B::require_operator_capabilities(
        "LFM3 speech depth",
        lookup_capabilities()
            .union(NeuralOperatorCapabilities::FROM_I32_SLICE)
            .union(NeuralOperatorCapabilities::CAST_FLOAT)
            .union(NeuralOperatorCapabilities::RMS_NORM_WITHOUT_WEIGHT)
            .union(NeuralOperatorCapabilities::SIGMOID),
    )
}

pub(super) fn resident_units<B: NeuralBackend>(
    d: &DepthDeclaration,
    feedback: bool,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Vec<DepthModule<B>>, Error> {
    require_capabilities::<B>()?;
    d.units()
        .filter(|unit| feedback || *unit != DepthUnit::Feedback)
        .map(|unit| DepthModule::new(d, unit, context))
        .collect()
}

pub(super) fn scores<B: NeuralBackend>(
    d: &DepthDeclaration,
    hidden: &B::Tensor,
    codes: &B::Tensor,
    masked: &B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
    mut apply: impl FnMut(usize, Operation<'_, B::Tensor>) -> Result<B::Tensor, Error>,
) -> Result<B::Tensor, Error> {
    let books = d.codebooks.len();
    let norm = d.blocks.len() + 1;
    if hidden.shape().len() != 2
        || hidden.dim(0) <= 0
        || hidden.dim(1) != d.projection.input
        || codes.shape() != [hidden.dim(0), books as i32]
        || masked.shape() != codes.shape()
        || codes.element_type() != Some(TensorElementType::I32)
        || masked.element_type() != Some(TensorElementType::Bool)
    {
        return Err(invalid("invalid LFM3 speech depth inputs"));
    }
    let codes = nonnegative_gather_indices(codes, d.vocabulary, context)?;
    let embeddings = (0..books)
        .map(|i| apply(norm + 1 + i, Operation::Embed(&codes, i)))
        .collect::<Result<Vec<_>, _>>()?;
    let embeddings = B::Tensor::stack(&embeddings, 1, context)?;
    let mut depth = apply(0, Operation::Condition(hidden, masked, &embeddings))?;
    for i in 1..norm {
        depth = apply(i, Operation::Block(&depth))?;
    }
    let depth = apply(norm, Operation::Normalize(&depth))?;
    let logits = (0..books)
        .map(|i| apply(norm + 1 + i, Operation::Score(&depth, i)))
        .collect::<Result<Vec<_>, _>>()?;
    B::Tensor::stack(&logits, 1, context)
}

fn invocations(d: &DepthDeclaration) -> Result<ExecutionUnitAddressMap, Error> {
    let depth = "lfm3.audio.depth";
    let feedback = "lfm3.audio.feedback";
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(depth),
            ExecutionGroupSpec::with_dependencies(feedback, [depth]),
        ],
        feedback,
    )
    .map_err(Error::backend_source)?;
    let trunk = d.blocks.len() + 2;
    let books = d.codebooks.len();
    let owners =
        ExecutionUnitLayout::new(&graph, [trunk + books, 1]).map_err(Error::backend_source)?;
    let calls =
        ExecutionUnitLayout::new(&graph, [trunk + 2 * books, 1]).map_err(Error::backend_source)?;
    let order = (trunk..trunk + books)
        .chain(0..trunk)
        .chain(trunk..trunk + books + 1);
    ExecutionUnitAddressMap::new(
        &owners,
        order.enumerate().map(|(i, owner)| {
            (
                calls.address(i).expect("declared invocation"),
                owners.address(owner).expect("declared owner"),
            )
        }),
    )
    .map_err(Error::backend_source)
}

/// Exact sources and shared codebook ownership for the speech output policy.
pub struct PreparedAudioDepth {
    declaration: DepthDeclaration,
    source: SharedCheckpointSource,
    invocations: ExecutionUnitAddressMap,
    units: Vec<Vec<WeightBinding>>,
}

impl PreparedAudioDepth {
    pub(in crate::lfm3::audio) fn new(
        declaration: DepthDeclaration,
        weights: &BindingSource,
    ) -> Result<Self, ArtifactError> {
        let (source, units) = prepare_audio_units(
            weights,
            "lfm3/audio/depth",
            declaration.units().map(|unit| {
                declaration
                    .unit_parameters(unit)
                    .map_err(ArtifactError::from)
            }),
        )?;
        let invocations = invocations(&declaration)?;
        Ok(Self {
            declaration,
            source,
            invocations,
            units,
        })
    }

    /// Transfers retained placement inputs and semantic declaration to the adapter.
    pub fn into_parts(
        self,
    ) -> (
        DepthDeclaration,
        SharedCheckpointSource,
        ExecutionUnitAddressMap,
        Vec<Vec<WeightBinding>>,
    ) {
        (self.declaration, self.source, self.invocations, self.units)
    }
}

/// Speech output operations used by the controlled Audio driver.
pub trait AudioDepth<B: NeuralBackend> {
    /// Scores the next complete bidirectional refinement pass.
    fn score_frame(
        &mut self,
        hidden: &B::Tensor,
        frame: &DepthFrame,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
    /// Embeds one completed or replayed frame for feedback into the language backbone.
    fn embed(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
}

/// Architecture-owned speech depth traversal over a shared residency policy.
pub struct PolicyDepth<B: NeuralBackend, P> {
    declaration: DepthDeclaration,
    invocations: ExecutionUnitAddressMap,
    policy: P,
    backend: PhantomData<fn() -> B>,
}

impl<B: NeuralBackend, P> PolicyDepth<B, P> {
    /// Constructs one unloaded module per physical owner.
    pub fn resident_units(
        d: &DepthDeclaration,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<DepthModule<B>>, Error> {
        resident_units(d, true, context)
    }

    /// Attaches the shared policy to the retained semantic invocation map.
    pub fn new(
        declaration: DepthDeclaration,
        map: ExecutionUnitAddressMap,
        policy: P,
    ) -> Result<Self, Error> {
        require_capabilities::<B>()?;
        if map != invocations(&declaration)? {
            return Err(invalid(
                "speech output invocations differ from the retained declaration",
            ));
        }
        Ok(Self {
            declaration,
            invocations: map,
            policy,
            backend: PhantomData,
        })
    }
}

fn invoke<B, P>(
    d: &DepthDeclaration,
    execution: &mut LayerwisePolicyForward<'_, B, DepthModule<B>, P>,
    ordinal: usize,
    address: eredu_runtime::ExecutionUnitAddress,
    unit: DepthUnit,
    operation: Operation<'_, B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, DepthModule<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    let module = execution
        .acquire(ordinal, address, |ctx| DepthModule::new(d, unit, ctx))
        .map_err(|error| match error {
            LayerwiseAcquireError::Architecture(error) => error,
            LayerwiseAcquireError::Policy(error) => Error::backend_source(error),
        })?;
    let output = module.apply(d, &operation, context)?;
    execution
        .complete(&output, iter::empty(), operation.roots())
        .map_err(Error::backend_source)?;
    Ok(output)
}

impl<B, P> AudioDepth<B> for PolicyDepth<B, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, DepthModule<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn score_frame(
        &mut self,
        hidden: &B::Tensor,
        frame: &DepthFrame,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let d = &self.declaration;
        let (codes, masked) = d.frame_inputs(frame, context)?;
        let units = d.units().collect::<Vec<_>>();
        let mut addresses = self.invocations.logical_addresses().enumerate();
        let mut execution = LayerwisePolicyForward::begin(&mut self.policy, hidden, context)
            .map_err(Error::backend_source)?;
        let output = scores::<B>(d, hidden, &codes, &masked, context, |owner, operation| {
            let (ordinal, address) = addresses.next().expect("declared depth invocation");
            invoke(
                d,
                &mut execution,
                ordinal,
                address,
                units[owner],
                operation,
                context,
            )
        })?;
        execution.finish(&output).map_err(Error::backend_source)?;
        Ok(output)
    }

    fn embed(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let (ordinal, address) = self
            .invocations
            .logical_addresses()
            .enumerate()
            .last()
            .expect("declared feedback invocation");
        let mut execution = LayerwisePolicyForward::begin(&mut self.policy, codes, context)
            .map_err(Error::backend_source)?;
        let output = invoke(
            &self.declaration,
            &mut execution,
            ordinal,
            address,
            DepthUnit::Feedback,
            Operation::Feedback(codes),
            context,
        )?;
        execution.finish(&output).map_err(Error::backend_source)?;
        Ok(output)
    }
}
