//! Residual stream geometry and gated mixing composed from portable operators.
use crate::{
    Error, LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, Parameterized, Tensor,
};

/// Shared logical layout for residual values and pipeline transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualStreamGeometry {
    streams: i32,
    hidden: i32,
    width: i32,
}

impl ResidualStreamGeometry {
    /// Checks both per-stream and flattened tensor geometry.
    pub fn new(streams: i32, hidden: i32) -> Result<Self, Error> {
        let width = streams
            .checked_mul(hidden)
            .filter(|_| streams > 0 && hidden > 0)
            .ok_or_else(|| Error::backend("invalid residual stream geometry"))?;
        Ok(Self {
            streams,
            hidden,
            width,
        })
    }
    /// Number of complete residual streams.
    pub const fn streams(self) -> i32 {
        self.streams
    }
    /// Feature count of each stream.
    pub const fn hidden_size(self) -> i32 {
        self.hidden
    }
    /// Feature count after flattening the stream axis.
    pub const fn flattened_width(self) -> i32 {
        self.width
    }

    /// Broadcasts ordinary `[batch, sequence, hidden]` values to complete streams.
    pub fn expand<T: Tensor>(self, input: &T, context: &T::Context) -> Result<T, Error> {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != self.hidden {
            return Err(Error::backend(
                "residual expansion expects [batch, sequence, hidden]",
            ));
        }
        input
            .expand_dims(2, context)?
            .broadcast_to(&[shape[0], shape[1], self.streams, self.hidden], context)
    }

    /// Validates the complete value carried across a residual/pipeline boundary.
    pub fn validate_streams(self, shape: &[i32]) -> Result<(), Error> {
        if shape.len() != 4
            || shape[0] <= 0
            || shape[1] <= 0
            || shape[2] != self.streams
            || shape[3] != self.hidden
        {
            return Err(Error::backend(
                "expected complete [batch, sequence, streams, hidden] residual",
            ));
        }
        Ok(())
    }

    /// Flattens complete streams for learned mixing projections.
    pub fn flatten<T: Tensor>(self, input: &T, context: &T::Context) -> Result<T, Error> {
        self.validate_streams(input.shape())?;
        input.reshape(&[input.dim(0), input.dim(1), self.width], context)
    }

    /// Restores complete streams from a flattened projection.
    pub fn unflatten<T: Tensor>(self, input: &T, context: &T::Context) -> Result<T, Error> {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != self.width {
            return Err(Error::backend(
                "residual projection has incorrect flattened width",
            ));
        }
        input.reshape(&[shape[0], shape[1], self.streams, self.hidden], context)
    }
}

/// Construction policy for low-rank, gated residual mixing.
///
/// This is independent of Sinkhorn residual mixing. Normalization scales and
/// checkpoint identities are supplied by the architecture.
#[derive(Debug, Clone)]
pub struct GatedResidualSpec {
    /// Logical stream and hidden geometry.
    pub geometry: ResidualStreamGeometry,
    /// Independent per-stream RMS reductions with per-feature scales.
    pub normalization: NormalizationConstructionSpec,
    /// Projection from flattened residual width to the mixing rank.
    pub down: LinearSpec,
    /// Projection from mixing rank to flattened residual width.
    pub up: LinearSpec,
    /// Optional projection to per-stream injection coefficients.
    /// A final collapse has no injection projection.
    pub injection: Option<LinearSpec>,
}

impl GatedResidualSpec {
    /// Validates all geometry and formats before backend construction.
    pub fn validate(&self) -> Result<(), Error> {
        self.normalization.validate()?;
        let width = self.geometry.width;
        if self.normalization.dimensions != width
            || self.normalization.groups != Some(self.geometry.streams)
            || self.down.input != width
            || self.down.output <= 0
            || self.up.input != self.down.output
            || self.up.output != width
        {
            return Err(Error::backend(
                "gated residual projection/normalization geometry mismatch",
            ));
        }
        for projection in [&self.down, &self.up]
            .into_iter()
            .chain(self.injection.iter())
        {
            projection.format.validate_for_weight(&projection.weight)?;
            if projection.bias.is_some() {
                return Err(Error::backend(
                    "gated residual projections must be bias-free",
                ));
            }
        }
        if self
            .injection
            .as_ref()
            .is_some_and(|p| p.input != width || p.output != self.geometry.streams)
        {
            return Err(Error::backend("gated residual injection geometry mismatch"));
        }
        Ok(())
    }
}

/// Values retained between residual collapse and sublayer injection.
#[derive(Debug, Clone)]
pub struct GatedResidualInput<T> {
    /// Normalized, gated stream mean supplied to the sublayer.
    pub mixed: T,
    residual: T,
    injection: Option<T>,
    geometry: ResidualStreamGeometry,
}

impl<T: Tensor> GatedResidualInput<T> {
    /// Adds the sublayer output to each original stream with its learned gate.
    pub fn inject(self, output: &T, context: &T::Context) -> Result<T, Error> {
        let injection = self
            .injection
            .ok_or_else(|| Error::backend("final residual collapse has no injection projection"))?;
        if output.shape() != self.mixed.shape() {
            return Err(Error::backend(
                "gated residual sublayer output geometry mismatch",
            ));
        }
        let shape = self.residual.shape();
        let output = self.geometry.expand(output, context)?;
        let gates = injection
            .expand_dims(-1, context)?
            .broadcast_to(shape, context)?;
        self.residual
            .add(&output.multiply(&gates, context)?, context)
    }
}

/// Portable gated residual algorithm, using ordinary backend mechanisms only.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct GatedResidual<B: NeuralBackend> {
    #[parameter(skip)]
    geometry: ResidualStreamGeometry,
    normalization: B::Normalization,
    down: B::Linear,
    up: B::Linear,
    injection: Option<B::Linear>,
}

impl<B: NeuralBackend> Clone for GatedResidual<B> {
    fn clone(&self) -> Self {
        Self {
            geometry: self.geometry,
            normalization: self.normalization.clone(),
            down: self.down.clone(),
            up: self.up.clone(),
            injection: self.injection.clone(),
        }
    }
}

impl<B: NeuralBackend> GatedResidual<B> {
    /// Builds the selected projections and normalization from an exact spec.
    pub fn new(
        spec: GatedResidualSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            geometry: spec.geometry,
            normalization: B::normalization(spec.normalization, context)?,
            down: B::linear(spec.down, context)?,
            up: B::linear(spec.up, context)?,
            injection: spec.injection.map(|p| B::linear(p, context)).transpose()?,
        })
    }

    /// Normalizes each stream, mixes it through SiLU and sigmoid projections,
    /// and averages the gated normalized streams. Injection retains the original
    /// residual values and uses `2 * sigmoid(projection / stream_count)`.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<GatedResidualInput<B::Tensor>, Error> {
        let input_flat = self.geometry.flatten(input, context)?;
        let normalized = self.normalization.forward(&input_flat, context)?;
        let low_rank = self
            .down
            .forward(&normalized, context)?
            .multiply_scalar(1.0 / self.geometry.streams as f32, context)?;
        let low_rank = B::silu(low_rank, context)?;
        let weights = B::sigmoid(self.up.forward(&low_rank, context)?, context)?;
        let weighted = self
            .geometry
            .unflatten(&weights.multiply(&normalized, context)?, context)?;
        let mixed = B::Tensor::mean_axis(&weighted, 2, false, context)?;
        let injection = self
            .injection
            .as_mut()
            .map(|projection| {
                let logits = projection
                    .forward(&normalized, context)?
                    .multiply_scalar(1.0 / self.geometry.streams as f32, context)?;
                B::sigmoid(logits, context)?.multiply_scalar(2.0, context)
            })
            .transpose()?;
        Ok(GatedResidualInput {
            mixed,
            residual: input.clone(),
            injection,
            geometry: self.geometry,
        })
    }
}
