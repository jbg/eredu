//! Independently leased speech encoder and projector over the shared runtime policy.

use super::*;
use crate::lfm3::audio::{placement::prepare_audio_units, ArtifactError, BindingSource};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::BackendFailure;
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
    pub(in crate::lfm3::audio) fn parameter_bytes(
        &self,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Result<u64, crate::lfm3::audio::ResidentAudioError> {
        crate::lfm3::audio::placement::admit_audio_weights(
            "encoder",
            &self.invocations,
            &self.units,
            0,
            residency,
        )
    }

    pub(in crate::lfm3::audio) fn new(
        declaration: Declaration,
        weights: &BindingSource,
        dtype: &eredu_checkpoint::recipe::RecipeDtype,
    ) -> Result<Self, ArtifactError> {
        let (source, units) = prepare_audio_units(
            weights,
            "lfm3/audio/encoder",
            dtype,
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
        let weights =
            BindingSource::from_recipes(source, scope, recipes).map_err(Error::backend_source)?;
        Self::new(declaration, &weights, &dtype).map_err(Error::backend_source)
    }

    /// Connects the retained encoder units to a generic parameter materializer.
    pub fn bind<B, P>(
        self,
        context: &<B::Tensor as Tensor>::Context,
        bind_units: impl FnOnce(
            SharedCheckpointSource,
            ExecutionUnitAddressMap,
            Vec<Vec<WeightBinding>>,
            &dyn Fn() -> Result<Vec<EncoderModule<B>>, BackendFailure>,
        ) -> Result<P, BackendFailure>,
    ) -> Result<PolicyEncoder<B, P>, BackendFailure>
    where
        B: NeuralBackend,
        P: LayerwisePolicy<B, EncoderModule<B>>,
    {
        let Self {
            declaration,
            source,
            invocations,
            units,
        } = self;
        let policy = bind_units(source, invocations.clone(), units, &|| {
            declaration
                .units()
                .map(|unit| build(&declaration, unit, context))
                .collect::<Result<Vec<_>, _>>()
                .map_err(BackendFailure::from_error)
        })?;
        Ok(PolicyEncoder {
            declaration,
            invocations,
            policy,
            backend: PhantomData,
        })
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
    /// Cache geometry for one independently projected clip.
    fn state_layout(&self) -> &eredu_runtime::StateLayout;

    /// Projects one complete clip into decoder-width rows with fresh encoder state.
    fn project_clip(
        &mut self,
        features: &eredu_media::audio::LogMelFeatures,
        context: &<B::Tensor as Tensor>::Context,
        allocate: impl FnOnce(&eredu_runtime::StateLayout) -> Result<S, BackendFailure>,
    ) -> Result<B::Tensor, BackendFailure> {
        let frames = i32::try_from(features.frames).map_err(BackendFailure::from_error)?;
        let bins = i32::try_from(features.mel_bins).map_err(BackendFailure::from_error)?;
        let input = B::Tensor::from_f32_slice(&features.values, &[1, frames, bins], context)
            .and_then(|features| features.transpose_axes(&[0, 2, 1], context))
            .map_err(BackendFailure::from_error)?;
        let mut state = allocate(self.state_layout())?;
        self.project(&input, &mut state, context)
            .and_then(|rows| rows.squeeze_axes(&[0], context))
            .map_err(BackendFailure::from_error)
    }

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
    fn state_layout(&self) -> &eredu_runtime::StateLayout {
        self.declaration.state_layout()
    }

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
