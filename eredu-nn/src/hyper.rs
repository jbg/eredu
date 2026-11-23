//! Portable FP32 residual stream mixing on native tensor values.

use crate::{
    DeclareLogicalParameters, Error, HyperConnectionOperator, HyperConnectionSpec,
    HyperConnectionState, HyperHeadOperator, HyperHeadSpec, Index, LogicalParameterRole,
    NeuralBackend, Parameter, ParameterSpec, Parameterized, Tensor, TensorElementType,
    TensorValueObserver,
};

#[derive(Clone, Debug, Parameterized)]
#[parameterized(tensor = "T")]
struct Parameters<T: Tensor> {
    function: Parameter<T>,
    base: Parameter<T>,
    scale: Parameter<T>,
}

impl<T: Tensor> Parameters<T> {
    fn new(
        spec: &impl DeclareLogicalParameters,
        slots: [&ParameterSpec; 3],
        context: &T::Context,
    ) -> Result<Self, Error> {
        let declarations = spec.logical_parameters()?;
        let parameter = |role, slot: &ParameterSpec| {
            let dimensions = &declarations
                .iter()
                .find(|d| d.role == role)
                .ok_or_else(|| Error::backend("missing residual parameter declaration"))?
                .shape;
            let shape = dimensions
                .iter()
                .copied()
                .map(i32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(Error::backend_source)?;
            Ok::<_, Error>(Parameter::new(
                slot.clone(),
                T::unloaded_f32(&shape, context)?,
            ))
        };
        Ok(Self {
            function: parameter(LogicalParameterRole::ResidualMixWeight, slots[0])?,
            base: parameter(LogicalParameterRole::ResidualMixBias, slots[1])?,
            scale: parameter(LogicalParameterRole::ResidualMixScale, slots[2])?,
        })
    }

    fn project(
        &self,
        value: &T,
        streams: i32,
        hidden: i32,
        epsilon: f32,
        context: &T::Context,
    ) -> Result<(T, T), Error> {
        let [batch, sequence, actual_streams, actual_hidden] = value.shape() else {
            return Err(Error::backend("residual mixing requires four axes"));
        };
        if *actual_streams != streams
            || *actual_hidden != hidden
            || !epsilon.is_finite()
            || epsilon <= 0.
        {
            return Err(Error::backend(
                "invalid residual mixing geometry or epsilon",
            ));
        }
        let width = streams
            .checked_mul(hidden)
            .ok_or_else(|| Error::backend("residual width exceeds i32"))?;
        let fp32 = value.cast_float(TensorElementType::F32, context)?;
        let flat = fp32.reshape(&[*batch, *sequence, width], context)?;
        let offset = T::from_f32_slice(&[epsilon], &[], context)?;
        let variance =
            T::mean_axis(&flat.square(context)?, -1, true, context)?.add(&offset, context)?;
        let normalized = flat.multiply(&variance.reciprocal_sqrt(context)?, context)?;
        let logits = T::matmul(
            &normalized,
            &self.function.as_ref().transpose(context)?,
            context,
        )?;
        Ok((fp32, logits))
    }
}

fn precision<T: Tensor>(value: &T) -> Result<TensorElementType, Error> {
    value
        .element_type()
        .ok_or_else(|| Error::backend("missing residual precision"))
}

fn sum_streams<T: Tensor>(residual: &T, weights: &T, context: &T::Context) -> Result<T, Error> {
    T::sum_axis(
        &weights
            .expand_dims(-1, context)?
            .multiply(residual, context)?,
        2,
        false,
        context,
    )
}

/// Portable residual stream connection with native parameter handles.
#[derive(Clone, Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PortableHyperConnection<B: NeuralBackend> {
    parameters: Parameters<B::Tensor>,
    #[parameter(skip)]
    spec: HyperConnectionSpec,
}

impl<B: NeuralBackend> PortableHyperConnection<B> {
    /// Allocates the declared FP32 parameter slots.
    pub fn new(
        spec: HyperConnectionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let parameters =
            Parameters::new(&spec, [&spec.function, &spec.base, &spec.scale], context)?;
        Ok(Self { parameters, spec })
    }
}

impl<B: NeuralBackend + Clone + std::fmt::Debug> HyperConnectionOperator<B::Tensor>
    for PortableHyperConnection<B>
{
    fn collapse(
        &mut self,
        residual: &B::Tensor,
        norm_epsilon: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<HyperConnectionState<B::Tensor>, Error> {
        let spec = &self.spec;
        let (fp32, logits) = self.parameters.project(
            residual,
            spec.streams,
            spec.hidden_size,
            norm_epsilon,
            context,
        )?;
        let logits_for = |start, end, scale| {
            let axis = Index::Range(start, end);
            let scale = self
                .parameters
                .scale
                .as_ref()
                .index(&[Index::At(scale)], context)?;
            let bias = self.parameters.base.as_ref().index(&[axis], context)?;
            logits
                .index(&[Index::Full, Index::Full, axis], context)?
                .multiply(&scale, context)?
                .add(&bias, context)
        };
        let streams = spec.streams;
        let epsilon = B::Tensor::from_f32_slice(&[spec.epsilon], &[], context)?;
        let pre = B::sigmoid(logits_for(0, streams, 0)?, context)?.add(&epsilon, context)?;
        let post = B::sigmoid(logits_for(streams, 2 * streams, 1)?, context)?
            .multiply_scalar(2., context)?;
        let shape = [logits.dim(0), logits.dim(1), streams, streams];
        let raw = logits_for(2 * streams, logits.dim(2), 2)?.reshape(&shape, context)?;
        let mut combination = raw
            .softmax_axis(-1, true, context)?
            .add(&epsilon, context)?;
        let normalize = |value: &B::Tensor, axis| {
            value.divide(
                &B::Tensor::sum_axis(value, axis, true, context)?.add(&epsilon, context)?,
                context,
            )
        };
        combination = normalize(&combination, -2)?;
        for _ in 1..spec.sinkhorn_iterations {
            combination = normalize(&combination, -1)?;
            combination = normalize(&combination, -2)?;
        }
        let collapsed =
            sum_streams(&fp32, &pre, context)?.cast_float(precision(residual)?, context)?;
        Ok(HyperConnectionState {
            collapsed,
            pre,
            post,
            combination,
        })
    }

    fn expand(
        &mut self,
        sublayer: &B::Tensor,
        residual: &B::Tensor,
        state: &HyperConnectionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let old = residual.cast_float(TensorElementType::F32, context)?;
        let update = sublayer
            .cast_float(TensorElementType::F32, context)?
            .expand_dims(-2, context)?;
        let injected = state
            .post
            .expand_dims(-1, context)?
            .multiply(&update, context)?;
        let combination = state.combination.transpose_axes(&[0, 1, 3, 2], context)?;
        let mixed = B::Tensor::matmul(&combination, &old, context)?;
        injected
            .add(&mixed, context)?
            .cast_float(precision(sublayer)?, context)
    }
}

/// Portable final residual stream collapse with native parameter handles.
#[derive(Clone, Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PortableHyperHead<B: NeuralBackend> {
    parameters: Parameters<B::Tensor>,
    #[parameter(skip)]
    spec: HyperHeadSpec,
}

impl<B: NeuralBackend> PortableHyperHead<B> {
    /// Allocates the declared FP32 parameter slots.
    pub fn new(
        spec: HyperHeadSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let parameters =
            Parameters::new(&spec, [&spec.function, &spec.base, &spec.scale], context)?;
        Ok(Self { parameters, spec })
    }
}

impl<B: NeuralBackend + Clone + std::fmt::Debug> HyperHeadOperator<B::Tensor>
    for PortableHyperHead<B>
{
    fn forward(
        &mut self,
        residual: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_coefficients_observer(residual, context, None)
    }

    fn forward_with_coefficients_observer(
        &mut self,
        residual: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        observer: Option<&mut dyn TensorValueObserver<B::Tensor>>,
    ) -> Result<B::Tensor, Error> {
        let spec = &self.spec;
        let (fp32, logits) = self.parameters.project(
            residual,
            spec.streams,
            spec.hidden_size,
            spec.norm_epsilon,
            context,
        )?;
        let logits = logits
            .multiply(self.parameters.scale.as_ref(), context)?
            .add(self.parameters.base.as_ref(), context)?;
        let epsilon = B::Tensor::from_f32_slice(&[spec.epsilon], &[], context)?;
        let weights = B::sigmoid(logits, context)?.add(&epsilon, context)?;
        if let Some(observer) = observer {
            observer.observe(&weights)?;
        }
        sum_streams(&fp32, &weights, context)?.cast_float(precision(residual)?, context)
    }
}
