//! Recurrent inference composed from ordinary affine and tensor mechanisms.

use crate::{Error, Index, LinearOperator, LinearSpec, NeuralBackend, Parameterized, Tensor};

/// Independent input and recurrent projections for an LSTM layer.
///
/// Both projections concatenate gates in input, forget, candidate, output order.
/// Their bias slots are independent and may have unrelated checkpoint identities.
#[derive(Debug, Clone)]
pub struct LstmLayerSpec {
    /// Maps input features to four hidden-width gates.
    pub input: LinearSpec,
    /// Maps previous hidden features to the same four gates.
    pub recurrent: LinearSpec,
}

impl LstmLayerSpec {
    /// Validates dimensions and physical projection formats before allocation.
    pub fn validate(&self) -> Result<(), Error> {
        if self.input.input <= 0
            || self.recurrent.input <= 0
            || self.recurrent.input.checked_mul(4) != Some(self.input.output)
            || self.recurrent.output != self.input.output
        {
            return Err(Error::backend("invalid LSTM input/recurrent gate geometry"));
        }
        self.input.format.validate_for_weight(&self.input.weight)?;
        self.recurrent
            .format
            .validate_for_weight(&self.recurrent.weight)?;
        Ok(())
    }
}

/// One layer's retained recurrent values, each shaped `[batch, hidden]`.
///
/// Cloning retains tensor handles. Steps build new values and never mutate these
/// tensors, so an unpublished step can be discarded without changing its parent.
#[derive(Debug, Clone)]
pub struct LstmLayerState<T> {
    /// Hidden output from the previous step.
    pub hidden: T,
    /// Cell accumulator from the previous step.
    pub cell: T,
}

/// Complete retained state of a unidirectional LSTM stack.
#[derive(Debug, Clone)]
pub struct LstmState<T> {
    /// Recurrent values ordered from the input layer to the output layer.
    pub layers: Vec<LstmLayerState<T>>,
}

impl<T> LstmState<T> {
    /// Returns the output of the final layer, when this state has a layer.
    pub fn output(&self) -> Option<&T> {
        self.layers.last().map(|layer| &layer.hidden)
    }
}

impl<T> crate::RetainedTensorRoots<T> for LstmState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        for layer in &self.layers {
            visitor(&layer.hidden);
            visitor(&layer.cell);
        }
    }
}

#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct LstmLayer<B: NeuralBackend> {
    #[parameter(skip)]
    input_width: i32,
    #[parameter(skip)]
    hidden_width: i32,
    input: B::Linear,
    recurrent: B::Linear,
}

impl<B: NeuralBackend> Clone for LstmLayer<B> {
    fn clone(&self) -> Self {
        Self {
            input_width: self.input_width,
            hidden_width: self.hidden_width,
            input: self.input.clone(),
            recurrent: self.recurrent.clone(),
        }
    }
}

/// Stacked unidirectional LSTM inference without dropout or projection.
///
/// Parameter identities and storage formats belong to the caller. Backend
/// linear operators preserve ordinary packed-weight and parameter-binding paths.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Lstm<B: NeuralBackend> {
    layers: Vec<LstmLayer<B>>,
}

impl<B: NeuralBackend> Clone for Lstm<B> {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
        }
    }
}

impl<B: NeuralBackend> Lstm<B> {
    /// Validates the entire stack before constructing any backend parameters.
    pub fn new(
        specs: Vec<LstmLayerSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if specs.is_empty() {
            return Err(Error::backend("an LSTM stack requires at least one layer"));
        }
        for (index, spec) in specs.iter().enumerate() {
            spec.validate()?;
            if index > 0 && spec.input.input != specs[index - 1].recurrent.input {
                return Err(Error::backend("adjacent LSTM layer widths do not agree"));
            }
        }
        let layers = specs
            .into_iter()
            .map(|spec| {
                Ok(LstmLayer {
                    input_width: spec.input.input,
                    hidden_width: spec.recurrent.input,
                    input: B::linear(spec.input, context)?,
                    recurrent: B::linear(spec.recurrent, context)?,
                })
            })
            .collect::<Result<_, Error>>()?;
        Ok(Self { layers })
    }

    /// Width accepted by the first layer.
    pub fn input_width(&self) -> i32 {
        self.layers[0].input_width
    }

    /// Allocates zero F32 hidden and cell values for each layer.
    ///
    /// Callers using another recurrent precision can supply matching state
    /// tensors directly, including dtype-preserving zeros made by their backend.
    pub fn zero_state(
        &self,
        batch: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LstmState<B::Tensor>, Error> {
        if batch <= 0 {
            return Err(Error::backend("LSTM batch size must be positive"));
        }
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let shape = [batch, layer.hidden_width];
                Ok(LstmLayerState {
                    hidden: B::Tensor::full_f32(0.0, &shape, context)?,
                    cell: B::Tensor::full_f32(0.0, &shape, context)?,
                })
            })
            .collect::<Result<_, Error>>()?;
        Ok(LstmState { layers })
    }

    /// Allocates zero state in the prototype's floating representation.
    ///
    /// The prototype is shaped `[batch, input_width]`. A bounded zero column is
    /// broadcast to each recurrent width without reading values on the host.
    pub fn zero_state_like(
        &self,
        prototype: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LstmState<B::Tensor>, Error> {
        let shape = prototype.shape();
        if shape.len() != 2 || shape[0] <= 0 || shape[1] != self.input_width() {
            return Err(Error::backend(
                "LSTM state prototype expects [batch, input_width]",
            ));
        }
        let zero = prototype
            .index(&[Index::Full, Index::Range(0, 1)], context)?
            .zeros_like(context)?;
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let hidden = zero
                    .broadcast_to(&[shape[0], layer.hidden_width], context)?
                    .compact(context)?;
                // The initial hidden/cell tensors may share immutable zero
                // storage; recurrent steps always construct replacement values.
                let cell = hidden.clone();
                Ok(LstmLayerState { hidden, cell })
            })
            .collect::<Result<_, Error>>()?;
        Ok(LstmState { layers })
    }

    /// Computes one `[batch, input]` step and returns a new, unpublished state.
    ///
    /// Gate order and both affine biases match the conventional PyTorch LSTM:
    /// `c = sigmoid(f) * c_previous + sigmoid(i) * tanh(g)`,
    /// `h = sigmoid(o) * tanh(c)`. No part of `previous` is modified, including
    /// when a later layer fails. Completion and publication belong to the caller.
    pub fn step(
        &mut self,
        input: &B::Tensor,
        previous: &LstmState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LstmState<B::Tensor>, Error> {
        let shape = input.shape();
        if shape.len() != 2 || shape[0] <= 0 || shape[1] != self.input_width() {
            return Err(Error::backend("LSTM step expects [batch, input_width]"));
        }
        if previous.layers.len() != self.layers.len() {
            return Err(Error::backend("LSTM state has the wrong layer count"));
        }
        // Validate every layer before submitting any projection.
        for (layer, state) in self.layers.iter().zip(&previous.layers) {
            let expected = [shape[0], layer.hidden_width];
            if state.hidden.shape() != expected || state.cell.shape() != expected {
                return Err(Error::backend("LSTM hidden/cell state geometry mismatch"));
            }
        }
        let mut value = input.clone();
        let mut layers = Vec::with_capacity(self.layers.len());
        for (layer, state) in self.layers.iter_mut().zip(&previous.layers) {
            let gates = layer
                .input
                .forward(&value, context)?
                .add(&layer.recurrent.forward(&state.hidden, context)?, context)?;
            let width = layer.hidden_width;
            let gate = |index: i32| {
                gates.index(
                    &[
                        Index::Full,
                        Index::Range(index * width, (index + 1) * width),
                    ],
                    context,
                )
            };
            let input_gate = B::sigmoid(gate(0)?, context)?;
            let forget_gate = B::sigmoid(gate(1)?, context)?;
            let candidate = gate(2)?.tanh(context)?;
            let output_gate = B::sigmoid(gate(3)?, context)?;
            let cell = forget_gate
                .multiply(&state.cell, context)?
                .add(&input_gate.multiply(&candidate, context)?, context)?
                .compact(context)?;
            let hidden = output_gate
                .multiply(&cell.tanh(context)?, context)?
                .compact(context)?;
            value = hidden.clone();
            layers.push(LstmLayerState { hidden, cell });
        }
        Ok(LstmState { layers })
    }
}
