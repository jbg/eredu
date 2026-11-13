//! Independently leased speech encoder and projector over the shared runtime policy.

use super::*;
use crate::lfm3::audio::{placement::prepare_audio_units, ArtifactError, BindingSource};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_runtime::{
    ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddressMap, ExecutionUnitLayout,
    LayerwiseAcquireError, LayerwisePolicy, LayerwisePolicyForward, RuntimeLayerState,
    WeightBinding,
};
use std::{iter, marker::PhantomData, ops::Range};

/// One independently materialized speech input module.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct EncoderModule<B: NeuralBackend>(EncoderOperation<B>);

#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum EncoderOperation<B: NeuralBackend> {
    Downsampler(Downsampler<B>),
    Block(ReplicatedBlock<B>),
    Normalization(B::Normalization),
    Projector(Projector<B>),
}

fn build<B: NeuralBackend>(
    declaration: &Declaration,
    unit: Unit,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<EncoderModule<B>, Error> {
    Ok(EncoderModule(match unit {
        Unit::Downsampler => EncoderOperation::Downsampler(Downsampler::new(declaration, context)?),
        Unit::Block(index) => EncoderOperation::Block(ReplicatedBlock::from_parameters(
            declaration.blocks[index].clone(),
            context,
        )?),
        Unit::Normalization => EncoderOperation::Normalization(B::normalization(
            declaration.final_norm.clone(),
            context,
        )?),
        Unit::Projector => EncoderOperation::Projector(Projector::new(declaration, context)?),
    }))
}

/// Exact prepared speech input resources and their invocation order.
pub struct PreparedAudioEncoder {
    declaration: Declaration,
    source: SharedCheckpointSource,
    invocations: ExecutionUnitAddressMap,
    units: Vec<Vec<WeightBinding>>,
}

impl PreparedAudioEncoder {
    pub(in crate::lfm3::audio) fn new(
        declaration: Declaration,
        weights: &BindingSource,
    ) -> Result<Self, ArtifactError> {
        let (source, units) = prepare_audio_units(
            weights,
            "lfm3/audio/encoder",
            declaration.units().map(|unit| {
                declaration
                    .unit_parameters(unit)
                    .map_err(ArtifactError::from)
            }),
        )?;
        let invocations = encoder_invocations(&declaration)?;
        Ok(Self {
            declaration,
            source,
            invocations,
            units,
        })
    }

    pub(in crate::lfm3) fn from_safetensors(
        declaration: Declaration,
        resolution: &eredu_checkpoint::validation::ResolvedCheckpointPlan,
        source: SharedCheckpointSource,
        dtype: eredu_checkpoint::recipe::RecipeDtype,
    ) -> Result<Self, Error> {
        use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};

        let mut aliases = declaration.safetensors_recipes(resolution)?;
        let recipes = declaration
            .logical_parameters()?
            .into_iter()
            .map(|parameter| {
                let name = parameter.parameter.id.to_string();
                let input = aliases
                    .remove(&name)
                    .unwrap_or_else(|| DerivedWeightRecipe::source(&name, TensorSelection::Full));
                (name, input)
            })
            .collect();
        Self::from_recipes(declaration, source, resolution.identity(), recipes, dtype)
    }

    pub(in crate::lfm3) fn from_recipes(
        declaration: Declaration,
        source: SharedCheckpointSource,
        scope: &str,
        recipes: std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        dtype: eredu_checkpoint::recipe::RecipeDtype,
    ) -> Result<Self, Error> {
        use eredu_checkpoint::recipe::DerivedWeightRecipe;

        let recipes = recipes
            .into_iter()
            .map(|(name, input)| {
                (
                    name,
                    DerivedWeightRecipe::Cast {
                        input: Box::new(input),
                        dtype: dtype.clone(),
                    },
                )
            })
            .collect();
        let weights =
            BindingSource::from_recipes(source, scope, recipes).map_err(Error::backend_source)?;
        Self::new(declaration, &weights).map_err(Error::backend_source)
    }

    /// Cache layout for one independent clip.
    pub fn state_layout(&self) -> &eredu_runtime::StateLayout {
        self.declaration.state_layout()
    }

    /// Returns the neutral unit constructor and exact retained policy inputs.
    pub fn into_parts(
        self,
    ) -> (
        Declaration,
        SharedCheckpointSource,
        ExecutionUnitAddressMap,
        Vec<Vec<WeightBinding>>,
    ) {
        (self.declaration, self.source, self.invocations, self.units)
    }
}

fn encoder_invocations(declaration: &Declaration) -> Result<ExecutionUnitAddressMap, Error> {
    let id = "lfm3.audio.encoder";
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root(id)], id)
        .map_err(Error::backend_source)?;
    let layout = ExecutionUnitLayout::new(&graph, [declaration.units().count()])
        .map_err(Error::backend_source)?;
    Ok(ExecutionUnitAddressMap::identity(&layout))
}

/// Speech input operations shared by finite prompts, streamed ingress and LiveASR.
pub trait AudioEncoder<B: NeuralBackend, S> {
    /// Constructs the finite stream state without acquiring any learned weights.
    fn stream(&self, state: S, batch: i32) -> Result<EncoderState<S, B::Tensor>, Error>;
    /// Encodes and projects the complete prompt.
    fn project(
        &mut self,
        features: &B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
    /// Encodes and projects newly available positions in one stream.
    fn advance(
        &mut self,
        features: &B::Tensor,
        state: &mut EncoderState<S, B::Tensor>,
        finalize: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
}

/// The shared speech input traversal over the selected residency policy.
pub struct PolicyEncoder<B: NeuralBackend, P> {
    declaration: Declaration,
    invocations: ExecutionUnitAddressMap,
    policy: P,
    backend: PhantomData<fn() -> B>,
}

impl<B: NeuralBackend, P> PolicyEncoder<B, P> {
    /// Builds unloaded resident modules in canonical order.
    pub fn resident_units(
        declaration: &Declaration,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<EncoderModule<B>>, Error> {
        declaration
            .units()
            .map(|unit| build(declaration, unit, context))
            .collect()
    }

    /// Connects the retained architecture and shared parameter policy.
    pub fn new(
        declaration: Declaration,
        invocations: ExecutionUnitAddressMap,
        policy: P,
    ) -> Result<Self, Error> {
        if invocations != encoder_invocations(&declaration)? {
            return Err(invalid(
                "speech input invocations differ from the retained declaration",
            ));
        }
        Ok(Self {
            declaration,
            invocations,
            policy,
            backend: PhantomData,
        })
    }
}

impl<B, P> PolicyEncoder<B, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, EncoderModule<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn run<S>(
        declaration: &Declaration,
        invocations: &ExecutionUnitAddressMap,
        policy: &mut P,
        features: &B::Tensor,
        state: &mut S,
        slice: Option<Range<i32>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let mut execution = LayerwisePolicyForward::begin(policy, features, context)
            .map_err(Error::backend_source)?;
        let mut hidden = features.clone();
        for (ordinal, (unit, address)) in declaration
            .units()
            .zip(invocations.logical_addresses())
            .enumerate()
        {
            let module = execution
                .acquire(ordinal, address, |context| {
                    build(declaration, unit, context)
                })
                .map_err(|error| match error {
                    LayerwiseAcquireError::Architecture(error) => error,
                    LayerwiseAcquireError::Policy(error) => Error::backend_source(error),
                })?;
            let layer = if let Unit::Block(index) = unit {
                Some(state.layer(index).map_err(Error::backend_source)?)
            } else {
                None
            };
            let output = match (&mut module.0, layer) {
                (EncoderOperation::Downsampler(module), None) => {
                    let output = module.forward(&hidden, context)?;
                    if let Some(range) = &slice {
                        output.index(
                            &[
                                Index::Full,
                                Index::Range(range.start, range.end),
                                Index::Full,
                            ],
                            context,
                        )?
                    } else {
                        output
                    }
                }
                (EncoderOperation::Block(module), Some(layer)) => {
                    let output = module.forward(&hidden, None, layer, context)?;
                    execution
                        .complete(&output, layer.retained_values(), iter::once(&hidden))
                        .map_err(Error::backend_source)?;
                    hidden = output;
                    continue;
                }
                (EncoderOperation::Normalization(module), None) => {
                    module.forward(&hidden, context)?
                }
                (EncoderOperation::Projector(module), None) => module.forward(&hidden, context)?,
                _ => return Err(invalid("speech input module differs from its invocation")),
            };
            execution
                .complete(&output, iter::empty(), iter::once(&hidden))
                .map_err(Error::backend_source)?;
            hidden = output;
        }
        execution.finish(&hidden).map_err(Error::backend_source)?;
        Ok(hidden)
    }
}

impl<B, S, P> AudioEncoder<B, S> for PolicyEncoder<B, P>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    P: LayerwisePolicy<B, EncoderModule<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn stream(&self, state: S, batch: i32) -> Result<EncoderState<S, B::Tensor>, Error> {
        self.declaration.stream::<B, S>(state, batch)
    }

    fn project(
        &mut self,
        features: &B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.declaration
            .forward_with::<B, S>(features, state, |features, state| {
                Self::run(
                    &self.declaration,
                    &self.invocations,
                    &mut self.policy,
                    features,
                    state,
                    None,
                    context,
                )
            })
    }

    fn advance(
        &mut self,
        features: &B::Tensor,
        state: &mut EncoderState<S, B::Tensor>,
        finalize: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.declaration.advance_with::<B, S>(
            features,
            state,
            finalize,
            context,
            |batch| {
                B::Tensor::full_f32(
                    0.0,
                    &[batch, 0, self.declaration.projector_out.output],
                    context,
                )
            },
            |features, state, range| {
                Self::run(
                    &self.declaration,
                    &self.invocations,
                    &mut self.policy,
                    features,
                    state,
                    Some(range),
                    context,
                )
                .map_err(AdvanceError::State)
            },
        )
    }
}
