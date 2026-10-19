//! Fixed-mean probabilistic residual vector quantization.
//!
//! Latents use `[batch, frames, channels]` and codes use `[batch, frames, depth]`.
//! Each depth owns an independently visitable mean table and variance scalar;
//! no projection, sampling policy, or special token interpretation is implicit.

use std::ops::Range;

use eredu_nn::{NeuralBackend, Parameter, ParameterSpec, Parameterized, Tensor, TensorElementType};

use crate::Error;

/// Geometry shared by a sequence of residual codebooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// Number of ordered codebooks.
    pub depth: usize,
    /// Number of means in each codebook.
    pub cardinality: i32,
    /// Width of each mean and latent vector.
    pub channels: i32,
}

impl Geometry {
    /// Checks all codebook dimensions without allocating learned tensors.
    pub fn validate(self) -> Result<(), Error> {
        if self.depth == 0
            || i32::try_from(self.depth).is_err()
            || self.cardinality <= 0
            || self.channels <= 0
            || self.cardinality.checked_mul(self.channels).is_none()
        {
            return Err(invalid("invalid residual codebook geometry"));
        }
        Ok(())
    }
}

/// Numerically distinct nearest-mean scoring equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// `sum(residual²) + sum(mean²) - 2 residual·mean`.
    SquaredEuclidean,
    /// `sum(mean²) - 2 residual·mean`, omitting the constant residual norm.
    CodebookNormMinusTwiceDot,
}

/// One non-mutating residual quantization transition.
#[derive(Debug, Clone)]
pub struct QuantizedStage<T> {
    /// First minimum's integer indices, shaped `[batch, frames]`.
    pub codes: T,
    /// Selected means, shaped `[batch, frames, channels]`.
    pub selected: T,
    /// Input residual minus the selected means.
    pub residual: T,
}

/// Result of an ordered range of residual codebooks.
#[derive(Debug, Clone)]
pub struct Quantized<T> {
    /// Indices shaped `[batch, frames, selected_depth]`.
    pub codes: T,
    /// Residual remaining after the selected depths.
    pub residual: T,
    /// Input minus final residual, preserving the encoder's subtraction order.
    pub quantized: T,
}

/// Quantizes a fresh residual against one mean table.
///
/// Both inputs must be F32. Equal scores select the first mean. The caller
/// chooses the scoring equation explicitly because dropping the constant norm
/// can affect rounding and therefore a near-tied categorical decision.
pub fn quantize_residual<T: Tensor>(
    residual: &T,
    means: &T,
    metric: DistanceMetric,
    context: &T::Context,
) -> Result<QuantizedStage<T>, Error> {
    quantize_residual_with_score_reduction(residual, means, metric, context, |scores| scores)
}

/// Quantizes channel-sharded residuals after combining each rank's distance scores.
///
/// The reduction preserves the complete vocabulary axis and returns F32 scores
/// of the same shape. Argmin and residual subtraction are shared with ordinary
/// quantization, so equal global scores retain the first mean. The callback receives
/// local preparation failures so a collective owner can agree failure before reduction.
pub fn quantize_residual_with_score_reduction<T: Tensor>(
    residual: &T,
    means: &T,
    metric: DistanceMetric,
    context: &T::Context,
    reduce: impl FnOnce(Result<T, Error>) -> Result<T, Error>,
) -> Result<QuantizedStage<T>, Error> {
    let means_norm = (|| {
        validate_means(means)?;
        Ok::<_, Error>(T::sum_axis(&means.square(context)?, -1, false, context)?)
    })();
    quantize_with_norms(residual, means, means_norm, metric, context, reduce)
}

/// Quantizes with caller-retained squared codebook norms. The norms belong to
/// the exact immutable means supplied here; replacing means invalidates them.
/// Local tensor shards supply local norms before the ordinary score reduction.
pub fn quantize_residual_with_norms_and_score_reduction<T: Tensor>(
    residual: &T,
    means: &T,
    means_norm: &T,
    metric: DistanceMetric,
    context: &T::Context,
    reduce: impl FnOnce(Result<T, Error>) -> Result<T, Error>,
) -> Result<QuantizedStage<T>, Error> {
    quantize_with_norms(
        residual,
        means,
        Ok(means_norm.clone()),
        metric,
        context,
        reduce,
    )
}

fn quantize_with_norms<T: Tensor>(
    residual: &T,
    means: &T,
    means_norm: Result<T, Error>,
    metric: DistanceMetric,
    context: &T::Context,
    reduce: impl FnOnce(Result<T, Error>) -> Result<T, Error>,
) -> Result<QuantizedStage<T>, Error> {
    let prepared = (|| {
        validate_means(means)?;
        require_f32(residual)?;
        if residual.shape().len() != 3 || residual.dim(2) != means.dim(1) {
            return Err(invalid(
                "residual must have shape [batch, frames, channels]",
            ));
        }
        let rows = checked_rows(residual.dim(0), residual.dim(1))?;
        let flat = residual.reshape(&[rows, means.dim(1)], context)?;
        let means_norm = means_norm?;
        require_f32(&means_norm)?;
        if means_norm.shape() != [means.dim(0)] {
            return Err(invalid(
                "precomputed codebook norms differ from vocabulary geometry",
            ));
        }
        let norms = match metric {
            DistanceMetric::SquaredEuclidean => {
                T::sum_axis(&flat.square(context)?, -1, true, context)?.add(&means_norm, context)?
            }
            DistanceMetric::CodebookNormMinusTwiceDot => means_norm.clone(),
        };
        let dot = T::matmul(&flat, &means.transpose(context)?, context)?;
        let scores = norms.subtract(&dot.multiply_scalar(2.0, context)?, context)?;
        Ok(scores)
    })();
    let expected_shape = prepared.as_ref().ok().map(|scores| scores.shape().to_vec());
    let scores = reduce(prepared)?;
    require_f32(&scores)?;
    if Some(scores.shape()) != expected_shape.as_deref() {
        return Err(invalid("reduced codebook scores changed geometry"));
    }
    let codes = T::argmin_axis(&scores, -1, false, context)?
        .reshape(&[residual.dim(0), residual.dim(1)], context)?;
    let selected = lookup_codes(means, &codes, context)?;
    let residual = residual.subtract(&selected, context)?;
    Ok(QuantizedStage {
        codes,
        selected,
        residual,
    })
}

/// Looks up integer codes `[batch, frames]` in a plain F32 `[vocabulary, channels]` table.
///
/// The tensor gather mechanism validates index bounds. This function assigns no
/// special meaning to any table row and performs no host copy of the codes.
pub fn lookup_codes<T: Tensor>(means: &T, codes: &T, context: &T::Context) -> Result<T, Error> {
    validate_means(means)?;
    if codes.shape().len() != 2
        || !matches!(
            codes.element_type(),
            Some(
                TensorElementType::I32
                    | TensorElementType::U32
                    | TensorElementType::I64
                    | TensorElementType::U64
            )
        )
    {
        return Err(invalid(
            "codebook indices must be an integer [batch, frames] tensor",
        ));
    }
    let rows = checked_rows(codes.dim(0), codes.dim(1))?;
    Ok(means
        .take_axis(&codes.reshape(&[rows], context)?, 0, context)?
        .reshape(&[codes.dim(0), codes.dim(1), means.dim(1)], context)?)
}

/// One independently materializable fixed-mean codebook.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct Codebook<T: Tensor> {
    /// Mean vectors `[cardinality, channels]`.
    pub means: Parameter<T>,
    /// Retained scalar variance; nearest-mean encode/decode does not use it.
    pub variance: Parameter<T>,
    #[parameter(skip)]
    geometry: Geometry,
    #[parameter(skip)]
    depth: usize,
    #[parameter(skip)]
    prefix: String,
}

impl<T: Tensor> Codebook<T> {
    /// Declares the released `mus_list.N` and `_variance_list.N.variance` slots.
    pub fn unloaded(
        geometry: Geometry,
        prefix: &str,
        depth: usize,
        context: &T::Context,
    ) -> Result<Self, Error> {
        geometry.validate()?;
        if prefix.is_empty() || depth >= geometry.depth {
            return Err(invalid("invalid codebook parameter prefix or depth"));
        }
        let group = format!("{prefix}.codebook.{depth}");
        let parameter = |name: String, shape: &[i32]| {
            let mut spec =
                ParameterSpec::trainable(name).map_err(|error| invalid(error.to_string()))?;
            spec.trainable = false;
            spec.group = Some(group.clone());
            Ok::<_, Error>(Parameter::unloaded(spec, shape, context)?)
        };
        Ok(Self {
            means: parameter(
                format!("{prefix}.mus_list.{depth}"),
                &[geometry.cardinality, geometry.channels],
            )?,
            variance: parameter(format!("{prefix}._variance_list.{depth}.variance"), &[])?,
            geometry,
            depth,
            prefix: prefix.into(),
        })
    }
    /// Exact selected geometry.
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }
    /// Global residual depth of this independently leased unit.
    pub fn depth(&self) -> usize {
        self.depth
    }
    /// Canonical source namespace.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
    /// Validates the loaded table and scalar before any numerical operation.
    pub fn validate_bound(&self) -> Result<(), Error> {
        if self.means.as_ref().shape() != [self.geometry.cardinality, self.geometry.channels]
            || !self.variance.as_ref().shape().is_empty()
        {
            return Err(invalid(
                "bound residual codebook geometry does not match declaration",
            ));
        }
        require_f32(self.variance.as_ref())?;
        validate_means(self.means.as_ref())
    }
}

/// Resident fixed codebooks using the same selected-depth layered traversal.
pub struct ResidualQuantizer<B: NeuralBackend> {
    runtime: eredu_runtime::ResidentRuntime<
        eredu_runtime::ComponentAdapter<
            eredu_runtime::ComponentDefinition,
            crate::prvq_layered::QuantizerKernel<B>,
        >,
        B,
        eredu_runtime::StatelessComponentState,
    >,
}
impl<B: NeuralBackend> std::fmt::Debug for ResidualQuantizer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualQuantizer")
            .field("geometry", &self.geometry())
            .finish()
    }
}
impl<B: NeuralBackend> Parameterized<B::Tensor> for ResidualQuantizer<B> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitor<'a, B::Tensor>,
    {
        for book in self.codebooks() {
            book.visit_parameters(visitor);
        }
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>,
    {
        for book in self.codebooks_mut() {
            book.visit_parameters_mut(visitor);
        }
    }
    fn set_trainable(&mut self, value: bool) {
        for book in self.codebooks_mut() {
            book.set_trainable(value);
        }
    }
}
impl<B: NeuralBackend> ResidualQuantizer<B> {
    /// Declares one unloaded unit per residual depth.
    pub fn new(
        geometry: Geometry,
        prefix: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        geometry.validate()?;
        let books = (0..geometry.depth)
            .map(|depth| Codebook::unloaded(geometry, prefix, depth, context))
            .collect::<Result<_, _>>()?;
        Self::from_codebooks(geometry, books)
    }
    /// Adopts complete canonical codebooks without copying learned tensors.
    pub fn from_codebooks(
        geometry: Geometry,
        codebooks: Vec<Codebook<B::Tensor>>,
    ) -> Result<Self, Error> {
        geometry.validate()?;
        let prefix = codebooks
            .first()
            .ok_or_else(|| invalid("residual quantizer has no codebooks"))?
            .prefix()
            .to_owned();
        if codebooks.len() != geometry.depth
            || codebooks.iter().enumerate().any(|(depth, book)| {
                book.geometry() != geometry
                    || book.depth() != depth
                    || book.prefix() != prefix
                    || book.means.as_ref().shape() != [geometry.cardinality, geometry.channels]
                    || !book.variance.as_ref().shape().is_empty()
            })
        {
            return Err(invalid(
                "residual codebooks do not match configured order, namespace and geometry",
            ));
        }
        let architecture =
            crate::prvq_layered::QuantizerKernel::new(geometry, &prefix)?.into_component()?;
        let runtime = eredu_runtime::ResidentRuntime::from_units(
            architecture,
            codebooks.into_iter().map(|book| vec![book]).collect(),
        )
        .map_err(|error| match error {
            eredu_runtime::ResidentUnitsError::Architecture(error) => error,
            _ => invalid("residual quantizer unit count mismatch"),
        })?;
        Ok(Self { runtime })
    }
    /// Exact selected codebook geometry.
    pub fn geometry(&self) -> Geometry {
        self.runtime.architecture().kernel().geometry()
    }
    /// Independently leased depth modules in residual order.
    pub fn codebooks(&self) -> impl ExactSizeIterator<Item = &Codebook<B::Tensor>> {
        self.runtime.units().iter().map(|group| &group[0])
    }
    /// Independently bindable depth modules in residual order.
    pub fn codebooks_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Codebook<B::Tensor>> {
        self.runtime
            .units_mut()
            .iter_mut()
            .map(|group| &mut group[0])
    }
    /// Encodes all depths with complete squared-distance scoring.
    pub fn quantize(
        &mut self,
        latent: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Quantized<B::Tensor>, Error> {
        crate::prvq_layered::QuantizerExecution::quantize(self, latent, context)
    }
    /// Encodes only the selected range, starting from the supplied residual.
    pub fn quantize_range(
        &mut self,
        latent: &B::Tensor,
        range: Range<usize>,
        metric: DistanceMetric,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Quantized<B::Tensor>, Error> {
        crate::prvq_layered::QuantizerExecution::quantize_range(
            self, latent, range, metric, context,
        )
    }
    /// Decodes a nonempty prefix, preserving ordered additions.
    pub fn decode(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        crate::prvq_layered::QuantizerExecution::decode(self, codes, context)
    }
    /// Decodes codes corresponding exactly to the selected range.
    pub fn decode_range(
        &mut self,
        codes: &B::Tensor,
        range: Range<usize>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        crate::prvq_layered::QuantizerExecution::decode_range(self, codes, range, context)
    }
}
impl<B: NeuralBackend> crate::prvq_layered::QuantizerExecution<B> for ResidualQuantizer<B> {
    fn validate_coordinated_execution(&self) -> Result<(), Error> {
        Ok(())
    }
    fn geometry(&self) -> Geometry {
        self.geometry()
    }
    fn decode_range(
        &mut self,
        codes: &B::Tensor,
        range: Range<usize>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        use eredu_runtime::LayeredInvocation;
        let mut state = eredu_runtime::StatelessComponentState::new()?;
        self.runtime.invoke(
            crate::prvq_layered::QuantizerInput::Decode { codes, range },
            &mut state,
            context,
        )
    }
    fn quantize_range(
        &mut self,
        latent: &B::Tensor,
        range: Range<usize>,
        metric: DistanceMetric,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Quantized<B::Tensor>, Error> {
        use eredu_runtime::LayeredInvocation;
        let mut state = eredu_runtime::StatelessComponentState::new()?;
        self.runtime
            .invoke_with_context(
                crate::prvq_layered::QuantizerInput::Quantize {
                    latent,
                    range,
                    metric,
                },
                &mut state,
                context,
            )?
            .1
            .into_quantized()
    }
}

pub(crate) fn checked_rows(batch: i32, frames: i32) -> Result<i32, Error> {
    if batch <= 0 || frames <= 0 {
        return Err(invalid("codebook batch and frames must be positive"));
    }
    batch
        .checked_mul(frames)
        .ok_or_else(|| invalid("codebook frame count overflows"))
}

fn validate_means<T: Tensor>(means: &T) -> Result<(), Error> {
    require_f32(means)?;
    if means.shape().len() != 2 || means.dim(0) <= 0 || means.dim(1) <= 0 {
        return Err(invalid(
            "codebook means must have shape [cardinality, channels]",
        ));
    }
    Ok(())
}

pub(crate) fn require_f32<T: Tensor>(value: &T) -> Result<(), Error> {
    if value.element_type() != Some(TensorElementType::F32) {
        return Err(invalid(
            "probabilistic residual quantization requires F32 execution",
        ));
    }
    Ok(())
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidShape(message.into())
}
