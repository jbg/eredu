//! LiquidCodec neural equations, retained source slots and caller-owned state.
use super::{Analysis, Config, Quantizer, QuantizerSpec, Synthesis};
use crate::{spectral::OverlapAddStream, streaming::StreamPhase, Error};
use eredu_core::{
    cache::{LayerCachePolicy, StateTensorRole},
    AttentionPolicy, LayerSchedule,
};
use eredu_nn::{
    AttentionCache, AttentionRequest, DeclareLogicalParameters, GatedShortConvolution,
    GatedShortConvolutionSpec, Index, LinearOperator, LinearSpec, LogicalParameterDeclaration,
    LogicalParameterRole, NeuralBackend, NormalizationConstructionSpec, NormalizationOperator,
    Parameter, ParameterSpec, Parameterized, RotaryOperator, Tensor, TensorElementType,
};
use eredu_runtime::{LayerRuntimeState, RuntimeStateComponents, StateLayout};

const LAYERS: usize = 8;
const WIDTH: i32 = 512;
const HEADS: i32 = 8;
const KV: i32 = 2;
const HEAD: i32 = 64;
const INNER: i32 = 2304;
const CONTEXT: i32 = 131_072;

fn invalid(s: impl Into<String>) -> Error {
    Error::InvalidShape(s.into())
}
fn parameter(s: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(s.into()).map_err(|e| invalid(e.to_string()))
}
fn linear(root: &str, input: i32, output: i32, bias: bool) -> Result<LinearSpec, Error> {
    Ok(LinearSpec {
        input,
        output,
        weight: parameter(format!("{root}.weight"))?,
        bias: bias
            .then(|| parameter(format!("{root}.bias")))
            .transpose()?,
        format: eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
    })
}
fn norm(root: &str, dimensions: i32) -> Result<NormalizationConstructionSpec, Error> {
    Ok(NormalizationConstructionSpec::learned(
        dimensions,
        1e-5,
        parameter(format!("{root}.weight"))?,
    ))
}
fn attention_layer(i: usize) -> bool {
    matches!(i, 2 | 4 | 6)
}

/// Semantic ownership identity of one neural or projection invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// Convolutional spectral front end.
    Frontend,
    /// One causal encoder backbone block.
    EncoderBlock(usize),
    /// Encoder final normalization and quantizer-width projection.
    EncoderOutput,
    /// Learned finite-scalar projections.
    Quantizer,
    /// Latent-to-decoder-width projection.
    DecoderInput,
    /// One causal decoder backbone block.
    DecoderBlock(usize),
    /// Decoder normalization, polar projection and retained synthesis window.
    DecoderOutput,
}

#[derive(Debug, Clone)]
struct AttentionSpec {
    projections: eredu_nn::SelfAttentionProjectionSpec,
    query_norm: NormalizationConstructionSpec,
    key_norm: NormalizationConstructionSpec,
    gate: LinearSpec,
}
#[derive(Debug, Clone)]
enum MixerSpec {
    Attention(AttentionSpec),
    Convolution(GatedShortConvolutionSpec),
}
#[derive(Debug, Clone)]
struct BlockSpec {
    mixer: MixerSpec,
    norms: [NormalizationConstructionSpec; 4],
    feed_forward: eredu_nn::SplitGatedProductSpec,
}
impl BlockSpec {
    fn new(root: &str, layer: usize) -> Result<Self, Error> {
        let p = format!("{root}.layers.{layer}");
        let mixer = if attention_layer(layer) {
            let p = format!("{p}.self_attn");
            MixerSpec::Attention(AttentionSpec {
                projections: eredu_nn::SelfAttentionProjectionSpec::new(
                    eredu_nn::AttentionHeadGeometry::new(HEADS, KV, HEAD, HEAD)?,
                    eredu_nn::AttentionQueryLayout::QueryOnly,
                    linear(&format!("{p}.q_proj"), WIDTH, HEADS * HEAD, false)?,
                    linear(&format!("{p}.k_proj"), WIDTH, KV * HEAD, false)?,
                    linear(&format!("{p}.v_proj"), WIDTH, KV * HEAD, false)?,
                    linear(&format!("{p}.o_proj"), HEADS * HEAD, WIDTH, false)?,
                )?,
                query_norm: norm(&format!("{p}.q_norm"), HEAD)?,
                key_norm: norm(&format!("{p}.k_norm"), HEAD)?,
                gate: linear(&format!("{p}.gate_proj"), WIDTH, HEADS, false)?,
            })
        } else {
            MixerSpec::Convolution(GatedShortConvolutionSpec {
                input_dimensions: WIDTH,
                channels: WIDTH,
                output_dimensions: WIDTH,
                input_projection: linear(&format!("{p}.conv.in_proj"), WIDTH, 3 * WIDTH, false)?,
                output_projection: linear(&format!("{p}.conv.out_proj"), WIDTH, WIDTH, false)?,
                convolution: eredu_nn::CausalDepthwiseConvolutionSpec {
                    arithmetic: eredu_nn::ConvolutionArithmetic::Native,
                    dilation: 1,
                    channels: WIDTH,
                    kernel_size: 7,
                    weight: parameter(format!("{p}.conv.conv.weight"))?,
                    bias: None,
                    activation: eredu_nn::ConvolutionActivation::Identity,
                },
            })
        };
        let f = format!("{p}.feed_forward");
        Ok(Self {
            mixer,
            norms: [
                norm(&format!("{p}.operator_norm"), WIDTH)?,
                norm(&format!("{p}.post_operator_norm"), WIDTH)?,
                norm(&format!("{p}.ffn_norm"), WIDTH)?,
                norm(&format!("{p}.post_ffn_norm"), WIDTH)?,
            ],
            feed_forward: eredu_nn::SplitGatedProductSpec::new(
                linear(&format!("{f}.w1"), WIDTH, INNER, false)?,
                linear(&format!("{f}.w3"), WIDTH, INNER, false)?,
                linear(&format!("{f}.w2"), INNER, WIDTH, false)?,
                eredu_nn::GatedProductPolicy::default(),
            )?,
        })
    }
}
impl DeclareLogicalParameters for BlockSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        let mut p = self.norms.as_slice().logical_parameters()?;
        p.extend(self.feed_forward.logical_parameters()?);
        match &self.mixer {
            MixerSpec::Attention(s) => {
                p.extend(s.projections.logical_parameters()?);
                p.extend(s.query_norm.logical_parameters()?);
                p.extend(s.key_norm.logical_parameters()?);
                p.extend(s.gate.logical_parameters()?);
            }
            MixerSpec::Convolution(s) => p.extend(s.logical_parameters()?),
        }
        Ok(p)
    }
}

#[derive(Debug, Clone)]
struct ConvSpec {
    weight: ParameterSpec,
    bias: ParameterSpec,
    input: i32,
    output: i32,
    kernel: i32,
    stride: i32,
    padding: i32,
}
impl ConvSpec {
    fn new(
        root: &str,
        input: i32,
        output: i32,
        kernel: i32,
        stride: i32,
        padding: i32,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: parameter(format!("{root}.weight"))?,
            bias: parameter(format!("{root}.bias"))?,
            input,
            output,
            kernel,
            stride,
            padding,
        })
    }
}
impl DeclareLogicalParameters for ConvSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        Ok(vec![
            LogicalParameterDeclaration::new(
                &self.weight,
                &[self.output, self.input, self.kernel],
                LogicalParameterRole::LinearWeight,
            )?,
            LogicalParameterDeclaration::new(
                &self.bias,
                &[self.output],
                LogicalParameterRole::LinearBias,
            )?,
        ])
    }
}

/// Complete SafeTensors parameter topology and independent encoder/decoder geometry.
#[derive(Debug, Clone)]
pub struct Declaration {
    config: Config,
    frontend: [ConvSpec; 3],
    encoder: Vec<BlockSpec>,
    decoder: Vec<BlockSpec>,
    encoder_norm: NormalizationConstructionSpec,
    decoder_norm: NormalizationConstructionSpec,
    to_quantizer: LinearSpec,
    from_quantizer: LinearSpec,
    polar: LinearSpec,
    window: ParameterSpec,
    quantizer: QuantizerSpec,
    encoder_state: StateLayout,
    decoder_state: StateLayout,
}
impl Declaration {
    /// Declares the neural codec without selecting a device or reopening an artifact.
    pub fn new(config: Config) -> Result<Self, Error> {
        config.validate()?;
        if config.encoder_dim != WIDTH {
            return Err(invalid(
                "LiquidCodec's released eight-head, 64-wide backbone requires width 512",
            ));
        }
        let mut window = parameter("decoder.istft.window")?;
        window.trainable = false;
        let s = Self {
            frontend: [
                ConvSpec::new(
                    "frontend.in_proj",
                    3 * (config.stft_n_fft / 2 + 1),
                    config.encoder_conv_dim,
                    1,
                    1,
                    0,
                )?,
                ConvSpec::new(
                    "frontend.conv",
                    config.encoder_conv_dim,
                    config.encoder_conv_dim,
                    3,
                    1,
                    2,
                )?,
                ConvSpec::new(
                    "frontend.out",
                    config.encoder_conv_dim,
                    WIDTH,
                    config.stft_frame_stack,
                    config.stft_frame_stack,
                    0,
                )?,
            ],
            encoder: (0..LAYERS)
                .map(|i| BlockSpec::new("encoder.lfm", i))
                .collect::<Result<_, _>>()?,
            decoder: (0..LAYERS)
                .map(|i| BlockSpec::new("decoder.lfm", i))
                .collect::<Result<_, _>>()?,
            encoder_norm: norm("encoder.lfm.embedding_norm", WIDTH)?,
            decoder_norm: norm("decoder.lfm.embedding_norm", WIDTH)?,
            to_quantizer: linear("to_quantizer", WIDTH, config.quantizer_dim, true)?,
            from_quantizer: linear("from_quantizer", config.quantizer_dim, WIDTH, true)?,
            polar: linear("decoder.lin", WIDTH, 2 * (config.stft_n_fft / 2 + 1), true)?,
            window,
            quantizer: QuantizerSpec::new(&config, "quantizer")?,
            encoder_state: state_layout(None)?,
            decoder_state: state_layout(Some(config.decoder_window))?,
            config,
        };
        eredu_nn::validate_logical_parameter_topology(&s.logical_parameters()?)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(s)
    }
    /// Retained codec configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }
    /// Backend-neutral causal encoder state geometry.
    pub fn encoder_state_layout(&self) -> &StateLayout {
        &self.encoder_state
    }
    /// Backend-neutral sliding decoder state geometry.
    pub fn decoder_state_layout(&self) -> &StateLayout {
        &self.decoder_state
    }
    /// Ordered, independent learned unit identities.
    pub fn units(&self) -> impl Iterator<Item = Unit> {
        std::iter::once(Unit::Frontend)
            .chain((0..LAYERS).map(Unit::EncoderBlock))
            .chain([Unit::EncoderOutput, Unit::Quantizer, Unit::DecoderInput])
            .chain((0..LAYERS).map(Unit::DecoderBlock))
            .chain([Unit::DecoderOutput])
    }
    /// Source identities and shapes owned by one independent learned unit.
    pub fn unit_parameters(
        &self,
        unit: Unit,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = Vec::new();
        match unit {
            Unit::Frontend => p.extend(self.frontend.as_slice().logical_parameters()?),
            Unit::EncoderBlock(i) => p.extend(
                self.encoder
                    .get(i)
                    .ok_or_else(|| invalid("codec encoder layer out of range"))?
                    .logical_parameters()?,
            ),
            Unit::DecoderBlock(i) => p.extend(
                self.decoder
                    .get(i)
                    .ok_or_else(|| invalid("codec decoder layer out of range"))?
                    .logical_parameters()?,
            ),
            Unit::EncoderOutput => {
                p.extend(self.encoder_norm.logical_parameters()?);
                p.extend(self.to_quantizer.logical_parameters()?);
            }
            Unit::Quantizer => p.extend(self.quantizer.logical_parameters()?),
            Unit::DecoderInput => p.extend(self.from_quantizer.logical_parameters()?),
            Unit::DecoderOutput => {
                p.extend(self.decoder_norm.logical_parameters()?);
                p.extend(self.polar.logical_parameters()?);
                p.push(LogicalParameterDeclaration::new(
                    &self.window,
                    &[self.config.stft_n_fft],
                    LogicalParameterRole::Normalization,
                )?);
            }
        }
        Ok(p)
    }
    /// Exact official floating SafeTensors schema, including the F32 synthesis buffer.
    pub fn safetensors_plan(
        &self,
    ) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, Error> {
        use eredu_checkpoint::schema::*;
        let constraints = self
            .logical_parameters()?
            .into_iter()
            .map(|p| {
                let name = p.parameter.id.to_string();
                let mut aliases = Vec::new();
                if let Some(s) = name.strip_prefix("frontend.") {
                    aliases.push(format!("_frontend.{s}"));
                }
                if let Some(s) = name.strip_prefix("quantizer.") {
                    aliases.push(format!("quantizer.fsq.{s}"));
                }
                let dtype = if name == "decoder.istft.window" {
                    StoredDtypeConstraint::Exact(eredu_checkpoint::StoredDtype::F32)
                } else {
                    StoredDtypeConstraint::Floating
                };
                SafetensorsTensorConstraint::required(name, p.shape, dtype).with_aliases(aliases)
            })
            .collect();
        SafetensorsCheckpointPlan::new(
            "LiquidCodec neural codec",
            constraints,
            vec![],
            CatalogPolicy::strict(),
        )
        .map_err(|e| invalid(e.to_string()))
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        let mut p = Vec::new();
        for unit in self.units() {
            p.extend(
                self.unit_parameters(unit)
                    .map_err(eredu_nn::Error::backend_source)?,
            );
        }
        Ok(p)
    }
}
fn state_layout(window: Option<i32>) -> Result<StateLayout, Error> {
    use eredu_core::cache::{
        MutableStateResidency, StateTensorDimension as D, StateTensorDtype, StateTensorPolicy,
    };
    let mut policies = Vec::new();
    for i in 0..LAYERS {
        policies.push(if attention_layer(i) {
            LayerCachePolicy::key_value(
                match window {
                    Some(w) => {
                        AttentionPolicy::sliding(w as u32).map_err(|e| invalid(e.to_string()))?
                    }
                    None => AttentionPolicy::Full,
                },
                KV,
                HEAD,
            )
            .map_err(|e| invalid(e.to_string()))?
        } else {
            LayerCachePolicy::fixed_only(vec![StateTensorPolicy::new(
                StateTensorRole::Convolution { slot: 0 },
                vec![
                    D::Batch,
                    D::fixed(6).map_err(|e| invalid(e.to_string()))?,
                    D::fixed(WIDTH).map_err(|e| invalid(e.to_string()))?,
                ],
                StateTensorDtype::Floating,
                MutableStateResidency::AlwaysDeviceMutable,
            )
            .map_err(|e| invalid(e.to_string()))?])
            .map_err(|e| invalid(e.to_string()))?
        });
    }
    Ok(StateLayout::new(LayerSchedule::new(LAYERS, policies)?)?)
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct Conv<T: Tensor> {
    weight: Parameter<T>,
    bias: Parameter<T>,
    #[parameter(skip)]
    spec: ConvSpec,
}
impl<T: Tensor> Conv<T> {
    fn new(spec: ConvSpec, ctx: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            weight: Parameter::unloaded(
                spec.weight.clone(),
                &[spec.output, spec.input, spec.kernel],
                ctx,
            )?,
            bias: Parameter::unloaded(spec.bias.clone(), &[spec.output], ctx)?,
            spec,
        })
    }
    fn forward(&self, x: &T, ctx: &T::Context) -> Result<T, Error> {
        let weight = self.weight.as_ref().transpose_axes(&[0, 2, 1], ctx)?;
        let y = T::conv1d(
            &x.transpose_axes(&[0, 2, 1], ctx)?,
            &weight,
            self.spec.stride,
            self.spec.padding,
            1,
            1,
            ctx,
        )?;
        Ok(y.add(self.bias.as_ref(), ctx)?
            .transpose_axes(&[0, 2, 1], ctx)?)
    }
}
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct Attention<B: NeuralBackend> {
    q: B::Linear,
    k: B::Linear,
    v: B::Linear,
    o: B::Linear,
    gate: B::Linear,
    qn: B::Normalization,
    kn: B::Normalization,
    rope: B::Rotary,
    #[parameter(skip)]
    window: Option<i32>,
}
impl<B: NeuralBackend> Attention<B> {
    fn new(
        s: AttentionSpec,
        window: Option<i32>,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            q: B::linear(s.projections.query, ctx)?,
            k: B::linear(s.projections.key, ctx)?,
            v: B::linear(s.projections.value, ctx)?,
            o: B::linear(s.projections.output, ctx)?,
            gate: B::linear(s.gate, ctx)?,
            qn: B::normalization(s.query_norm, ctx)?,
            kn: B::normalization(s.key_norm, ctx)?,
            rope: B::rotary(
                eredu_nn::RotarySpec {
                    arithmetic: eredu_nn::RotaryArithmetic::Float32Products,
                    dimensions: HEAD,
                    base: 1_000_000.0,
                    traditional: false,
                    algorithm: eredu_nn::RotaryAlgorithm::ZeroTail {
                        rotary_fraction: 1.0,
                    },
                },
                ctx,
            )?,
            window,
        })
    }
    fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        x: &B::Tensor,
        state: &mut C,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let (batch, time) = (x.dim(0), x.dim(1));
        let offset = state.offset();
        let q = self
            .q
            .forward(x, ctx)?
            .reshape(&[batch, time, HEADS, HEAD], ctx)?
            .transpose_axes(&[0, 2, 1, 3], ctx)?;
        let k = self
            .k
            .forward(x, ctx)?
            .reshape(&[batch, time, KV, HEAD], ctx)?
            .transpose_axes(&[0, 2, 1, 3], ctx)?;
        let v = self
            .v
            .forward(x, ctx)?
            .reshape(&[batch, time, KV, HEAD], ctx)?
            .transpose_axes(&[0, 2, 1, 3], ctx)?;
        let q = self.qn.forward(&q, ctx)?;
        let k = self.kn.forward(&k, ctx)?;
        let q = self
            .rope
            .forward(&q, eredu_nn::RotaryPosition::Offset(offset), ctx)?;
        let k = self
            .rope
            .forward(&k, eredu_nn::RotaryPosition::Offset(offset), ctx)?;
        let (k, v) = state.update_for_attention(k, v, ctx)?;
        let mask = if self.window.is_none() && time > 1 {
            Some(B::causal_mask(time, offset, None, ctx)?)
        } else {
            None
        };
        let request = AttentionRequest {
            arithmetic: eredu_nn::AttentionArithmetic::InputScores,
            queries: q,
            keys: k,
            values: v,
            scale: (HEAD as f32).powf(-0.5),
            softcap: None,
            mask: mask.as_ref(),
            sinks: None,
        };
        let attended = if let Some(w) = self.window.filter(|_| time > 1) {
            B::sliding_window_attention_with_sinks(request, w, offset, ctx)?
                .reshape(&[batch, time, HEADS, HEAD], ctx)?
        } else {
            state
                .attention(request, ctx)?
                .transpose_axes(&[0, 2, 1, 3], ctx)?
        };
        let gate = B::sigmoid(self.gate.forward(x, ctx)?, ctx)?.expand_dims(-1, ctx)?;
        let attended = attended
            .multiply(&gate, ctx)?
            .reshape(&[batch, time, WIDTH], ctx)?;
        Ok(self.o.forward(&attended, ctx)?)
    }
}
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum Mixer<B: NeuralBackend> {
    Attention(Attention<B>),
    Convolution(GatedShortConvolution<B>),
}
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
/// Independently constructible codec backbone block with caller-owned layer state.
pub struct BackboneBlock<B: NeuralBackend> {
    mixer: Mixer<B>,
    norms: Vec<B::Normalization>,
    gate: B::Linear,
    up: B::Linear,
    down: B::Linear,
    #[parameter(skip)]
    policy: eredu_nn::GatedProductPolicy,
}
impl<B: NeuralBackend> BackboneBlock<B> {
    /// Constructs exactly one retained encoder or decoder block.
    pub fn new(
        declaration: &Declaration,
        unit: Unit,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let (spec, window) = match unit {
            Unit::EncoderBlock(i) => (declaration.encoder.get(i), None),
            Unit::DecoderBlock(i) => (
                declaration.decoder.get(i),
                Some(declaration.config.decoder_window),
            ),
            _ => return Err(invalid("codec backbone construction requires a block unit")),
        };
        Self::from_spec(
            spec.ok_or_else(|| invalid("codec backbone block out of range"))?
                .clone(),
            window,
            ctx,
        )
    }

    fn from_spec(
        s: BlockSpec,
        window: Option<i32>,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            mixer: match s.mixer {
                MixerSpec::Attention(s) => Mixer::Attention(Attention::new(s, window, ctx)?),
                MixerSpec::Convolution(s) => {
                    Mixer::Convolution(GatedShortConvolution::new(s, ctx)?)
                }
            },
            norms: s
                .norms
                .into_iter()
                .map(|n| B::normalization(n, ctx))
                .collect::<Result<_, _>>()?,
            gate: B::linear(s.feed_forward.gate, ctx)?,
            up: B::linear(s.feed_forward.up, ctx)?,
            down: B::linear(s.feed_forward.down, ctx)?,
            policy: s.feed_forward.policy,
        })
    }
    /// Advances one layer using the same equations as resident codec composition.
    /// A control owner supplies transaction state and establishes native completion.
    pub fn forward<C>(
        &mut self,
        x: &B::Tensor,
        state: &mut C,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let n = self.norms[0].forward(x, ctx)?;
        let mixed = match &mut self.mixer {
            Mixer::Attention(a) => a.forward(&n, state, ctx)?,
            Mixer::Convolution(c) => {
                let role = StateTensorRole::Convolution { slot: 0 };
                let y = c.forward(&n, state.fixed_component(role)?.as_ref(), ctx)?;
                *state.fixed_component(role)? = y.history;
                state.advance_fixed(x.dim(1))?;
                y.output
            }
        };
        let hidden = x.add(&self.norms[1].forward(&mixed, ctx)?, ctx)?;
        let n = self.norms[2].forward(&hidden, ctx)?;
        let product = B::gated_product(
            self.gate.forward(&n, ctx)?,
            self.up.forward(&n, ctx)?,
            self.policy,
            ctx,
        )?;
        let branch = self.down.forward(&product, ctx)?;
        Ok(hidden.add(&self.norms[3].forward(&branch, ctx)?, ctx)?)
    }
}

/// Caller-owned neural decoder state plus independent overlap-add continuation.
/// Native completion, transactional rollback and snapshot isolation remain the
/// responsibility of the shared execution/control owner that supplies `S`.
#[derive(Debug)]
pub struct DecoderState<S, T: Tensor> {
    neural: S,
    overlap: OverlapAddStream<T>,
    config: Config,
    batch: i32,
}
impl<S, T: Tensor> DecoderState<S, T> {
    /// Current spectral finite-stream lifecycle.
    pub fn phase(&self) -> StreamPhase {
        self.overlap.phase()
    }
    /// Borrows the concrete neutral-contract neural state.
    pub fn neural(&self) -> &S {
        &self.neural
    }
    /// Parameter-free sample/envelope roots retained in addition to neural state.
    pub fn overlap_values(&self) -> impl Iterator<Item = &T> {
        self.overlap.retained_values()
    }
}

/// Resident composition of declared LiquidCodec units with caller-owned state.
/// This does not select application tasks, distributed placement or residency.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct NeuralCodec<B: NeuralBackend> {
    frontend: Vec<Conv<B::Tensor>>,
    encoder: Vec<BackboneBlock<B>>,
    decoder: Vec<BackboneBlock<B>>,
    encoder_norm: B::Normalization,
    decoder_norm: B::Normalization,
    to_quantizer: B::Linear,
    from_quantizer: B::Linear,
    quantizer: Quantizer<B>,
    polar: B::Linear,
    window: Parameter<B::Tensor>,
    #[parameter(skip)]
    declaration: Declaration,
    #[parameter(skip)]
    analysis: Analysis<B::Tensor>,
    #[parameter(skip)]
    synthesis: Synthesis<B>,
}

/// Finite audio-tokenizer adapter and streaming convenience calls over the same
/// neural codec. The caller supplies a neutral state factory; no backend or
/// artifact discovery is performed by this adapter.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Tokenizer<B: NeuralBackend, F: eredu_runtime::ArchitectureStateFactory<B>> {
    model: NeuralCodec<B>,
    #[parameter(skip)]
    factory: F,
    #[parameter(skip)]
    stream: Option<DecoderState<F::State, B::Tensor>>,
}
impl<B, F> Tokenizer<B, F>
where
    B: NeuralBackend,
    F: eredu_runtime::ArchitectureStateFactory<B>,
    F::State: LayerRuntimeState<B>,
    <F::State as LayerRuntimeState<B>>::LayerState:
        AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    F::Error: std::error::Error + Send + Sync + 'static,
{
    /// Composes an already bound neural codec with its concrete state realization.
    pub fn new(model: NeuralCodec<B>, factory: F) -> Self {
        Self {
            model,
            factory,
            stream: None,
        }
    }
    /// Borrows neural composition for explicit caller-owned state operations.
    pub fn model_mut(&mut self) -> &mut NeuralCodec<B> {
        &mut self.model
    }
    fn realize(&mut self, decoder: bool) -> Result<F::State, Error> {
        let layout = if decoder {
            self.model.declaration.decoder_state_layout()
        } else {
            self.model.declaration.encoder_state_layout()
        };
        self.factory
            .realize(layout)
            .map_err(|e| Error::Compute(eredu_nn::Error::backend_source(e)))
    }
    /// Starts a new independent output stream, discarding any previous stream.
    pub fn start_decode_stream(&mut self, batch: i32) -> Result<(), Error> {
        let neural = self.realize(true)?;
        let state = self.model.decoder_state(neural, batch)?;
        self.stream = Some(state);
        Ok(())
    }
    /// Decodes a caller-advanced stream. `None` can flush once after preparation.
    /// Finite convenience decode uses the same neural and spectral equations.
    pub fn decode_chunk(
        &mut self,
        codes: Option<&B::Tensor>,
        finalize: bool,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let state = self
            .stream
            .as_mut()
            .ok_or_else(|| invalid("prepare a LiquidCodec decode stream before advancement"))?;
        let result = self.model.decode_chunk(codes, state, finalize, ctx);
        if result.is_err() {
            self.stream = None;
        }
        result
    }
}
impl<B, F> crate::AudioTokenizer for Tokenizer<B, F>
where
    B: NeuralBackend,
    F: eredu_runtime::ArchitectureStateFactory<B>,
    F::State: LayerRuntimeState<B>,
    <F::State as LayerRuntimeState<B>>::LayerState:
        AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    F::Error: std::error::Error + Send + Sync + 'static,
{
    type Tensor = B::Tensor;
    fn config(&self) -> crate::AudioTokenizerConfig {
        let c = &self.model.declaration.config;
        crate::AudioTokenizerConfig {
            sample_rate: f64::from(c.sampling_rate),
            frame_rate: f64::from(c.sampling_rate)
                / (f64::from(c.stft_hop_length) * f64::from(c.stft_frame_stack)),
            channels: 1,
            codebooks: c.n_q,
            cardinality: c.codebook_size().expect("validated codec cardinality"),
        }
    }
    fn encode(
        &mut self,
        pcm: &Self::Tensor,
        ctx: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let mut state = self.realize(false)?;
        self.model.encode(pcm, &mut state, ctx)
    }
    fn decode(
        &mut self,
        codes: &Self::Tensor,
        ctx: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let mut state = self.realize(true)?;
        self.model.decode(codes, &mut state, ctx)
    }
}
impl<B: NeuralBackend> NeuralCodec<B> {
    /// Retained source and state declaration, without reopening artifacts.
    pub fn declaration(&self) -> &Declaration {
        &self.declaration
    }

    /// Builds unloaded units from retained source geometry; no artifact is opened.
    pub fn new(
        declaration: &Declaration,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        B::require_operator_capabilities(
            "LiquidCodec neural codec",
            Analysis::<B::Tensor>::required_capabilities()
                .union(eredu_nn::NeuralOperatorCapabilities::SIGMOID),
        )?;
        let d = declaration;
        Ok(Self {
            frontend: d
                .frontend
                .iter()
                .cloned()
                .map(|s| Conv::new(s, ctx))
                .collect::<Result<_, _>>()?,
            encoder: d
                .encoder
                .iter()
                .cloned()
                .map(|s| BackboneBlock::from_spec(s, None, ctx))
                .collect::<Result<_, _>>()?,
            decoder: d
                .decoder
                .iter()
                .cloned()
                .map(|s| BackboneBlock::from_spec(s, Some(d.config.decoder_window), ctx))
                .collect::<Result<_, _>>()?,
            encoder_norm: B::normalization(d.encoder_norm.clone(), ctx)?,
            decoder_norm: B::normalization(d.decoder_norm.clone(), ctx)?,
            to_quantizer: B::linear(d.to_quantizer.clone(), ctx)?,
            from_quantizer: B::linear(d.from_quantizer.clone(), ctx)?,
            quantizer: Quantizer::new(&d.quantizer, ctx)?,
            polar: B::linear(d.polar.clone(), ctx)?,
            window: Parameter::unloaded(d.window.clone(), &[d.config.stft_n_fft], ctx)?,
            analysis: Analysis::new(&d.config, ctx)?,
            synthesis: Synthesis::new(&d.config, ctx)?,
            declaration: d.clone(),
        })
    }
    fn validate_state<S>(
        &self,
        state: &mut S,
        decoder: bool,
        tokens: i32,
        fresh: bool,
    ) -> Result<(), Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        let expected = if decoder {
            &self.declaration.decoder_state
        } else {
            &self.declaration.encoder_state
        };
        if state.optional_layout() != Some(expected) {
            return Err(invalid(
                "LiquidCodec state belongs to a different backbone layout",
            ));
        }
        let mut frontier = None;
        for i in 0..LAYERS {
            let position = state.layer(i)?.position();
            if position < 0
                || frontier.is_some_and(|n| n != position)
                || (fresh && position != 0)
                || position.checked_add(tokens).is_none_or(|n| n > CONTEXT)
            {
                return Err(invalid(
                    "LiquidCodec state frontier is inconsistent, nonempty or exceeds context",
                ));
            }
            frontier = Some(position);
        }
        Ok(())
    }
    /// Computes trained encoder latents from finite PCM using an empty encoder state.
    pub fn encode_latents<S>(
        &mut self,
        pcm: &B::Tensor,
        state: &mut S,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        if pcm.shape().len() != 3 || pcm.dim(0) <= 0 || pcm.dim(1) != 1 || pcm.dim(2) <= 0 {
            return Err(invalid(
                "LiquidCodec analysis requires nonempty mono [batch,1,samples]",
            ));
        }
        let frame = self.declaration.config.frame_samples()?;
        let frames = pcm
            .dim(2)
            .checked_add(frame - 1)
            .ok_or_else(|| invalid("LiquidCodec padded waveform exceeds I32"))?
            / frame;
        self.validate_state(state, false, frames, true)?;
        let f = self.analysis.features(pcm, ctx)?;
        let h = B::silu(self.frontend[0].forward(&f, ctx)?, ctx)?;
        let n = h.dim(2);
        let h = self.frontend[1]
            .forward(&h, ctx)?
            .index(&[Index::Full, Index::Full, Index::Range(0, n)], ctx)?;
        let h = B::silu(h, ctx)?;
        let mut h = self.frontend[2]
            .forward(&h, ctx)?
            .transpose_axes(&[0, 2, 1], ctx)?;
        for (i, b) in self.encoder.iter_mut().enumerate() {
            h = b.forward(&h, state.layer(i)?, ctx)?;
        }
        let h = self.encoder_norm.forward(&h, ctx)?;
        Ok(self
            .to_quantizer
            .forward(&h, ctx)?
            .transpose_axes(&[0, 2, 1], ctx)?)
    }
    /// Encodes finite mono PCM to exact I32 codebooks using caller-owned fresh state.
    pub fn encode<S>(
        &mut self,
        pcm: &B::Tensor,
        state: &mut S,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let latents = self.encode_latents(pcm, state, ctx)?;
        self.quantizer.encode(&latents, ctx)
    }
    fn decode_spectrum<S>(
        &mut self,
        codes: &B::Tensor,
        state: &mut S,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        if codes.shape().len() != 3
            || codes.dim(0) <= 0
            || codes.dim(1) != self.declaration.config.n_q
            || codes.dim(2) <= 0
        {
            return Err(invalid(
                "LiquidCodec decoding requires complete nonempty codebooks",
            ));
        }
        let time = codes
            .dim(2)
            .checked_mul(self.declaration.config.stft_frame_stack)
            .ok_or_else(|| invalid("codec spectral frame count overflow"))?;
        self.validate_state(state, true, time, false)?;
        let latents = self
            .quantizer
            .decode(codes, ctx)?
            .transpose_axes(&[0, 2, 1], ctx)?;
        let h = self.from_quantizer.forward(&latents, ctx)?;
        let stack = self.declaration.config.stft_frame_stack;
        let mut h = h
            .expand_dims(2, ctx)?
            .broadcast_to(&[h.dim(0), h.dim(1), stack, WIDTH], ctx)?
            .reshape(&[h.dim(0), time, WIDTH], ctx)?;
        for (i, b) in self.decoder.iter_mut().enumerate() {
            h = b.forward(&h, state.layer(i)?, ctx)?;
        }
        let h = self.decoder_norm.forward(&h, ctx)?;
        let polar = self
            .polar
            .forward(&h, ctx)?
            .transpose_axes(&[0, 2, 1], ctx)?;
        let bins = self.declaration.config.stft_n_fft / 2 + 1;
        Ok((
            polar.index(&[Index::Full, Index::Range(0, bins), Index::Full], ctx)?,
            polar.index(
                &[Index::Full, Index::Range(bins, 2 * bins), Index::Full],
                ctx,
            )?,
        ))
    }
    /// Decodes complete codebooks with a fresh decoder state and symmetric endpoints.
    pub fn decode<S>(
        &mut self,
        codes: &B::Tensor,
        state: &mut S,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.validate_state(state, true, 0, true)?;
        if codes.shape().len() == 3
            && codes.dim(0) > 0
            && codes.dim(1) == self.declaration.config.n_q
            && codes.dim(2) == 0
            && codes.element_type() == Some(TensorElementType::I32)
        {
            return Ok(B::Tensor::full_f32(0.0, &[codes.dim(0), 1, 0], ctx)?);
        }
        let (magnitude, angle) = self.decode_spectrum(codes, state, ctx)?;
        self.synthesis.set_window(self.window.as_ref(), ctx)?;
        let pcm = self.synthesis.reconstruct(&magnitude, &angle, ctx)?;
        pcm.require_finite(ctx)?;
        Ok(pcm)
    }
    /// Pairs a fresh caller-realized neural state with bounded spectral carry.
    pub fn decoder_state<S>(
        &self,
        mut neural: S,
        batch: i32,
    ) -> Result<DecoderState<S, B::Tensor>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        self.validate_state(&mut neural, true, 0, true)?;
        Ok(DecoderState {
            neural,
            overlap: OverlapAddStream::new(
                batch,
                self.declaration.config.stft_n_fft,
                self.declaration.config.stft_hop_length,
            )?,
            config: self.declaration.config.clone(),
            batch,
        })
    }
    /// Advances the shared neural decoder and overlap-add state. Empty input can
    /// flush once; it never advances convolution or KV frontiers.
    pub fn decode_chunk<S>(
        &mut self,
        codes: Option<&B::Tensor>,
        state: &mut DecoderState<S, B::Tensor>,
        finalize: bool,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        if state.phase() == StreamPhase::Finished {
            return Err(Error::StreamFinished);
        }
        if state.config != self.declaration.config {
            return Err(invalid(
                "LiquidCodec stream belongs to a different configuration",
            ));
        }
        let raw = if let Some(codes) = codes {
            if codes.shape().len() != 3
                || codes.dim(0) != state.batch
                || codes.dim(1) != state.config.n_q
                || codes.element_type() != Some(TensorElementType::I32)
            {
                return Err(invalid("LiquidCodec stream code geometry changed"));
            }
            if codes.dim(2) == 0 {
                None
            } else {
                let (magnitude, angle) = self.decode_spectrum(codes, &mut state.neural, ctx)?;
                self.synthesis.set_window(self.window.as_ref(), ctx)?;
                Some(self.synthesis.overlap_add(&magnitude, &angle, ctx)?)
            }
        } else {
            None
        };
        let pcm = state.overlap.push(raw, finalize, ctx)?;
        pcm.require_finite(ctx)?;
        Ok(pcm)
    }
}
