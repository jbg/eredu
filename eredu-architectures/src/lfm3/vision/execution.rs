//! Ordered execution of independently materialized vision units.

use eredu_nn::{Error, NeuralBackend, Parameterized, Tensor};

use super::{
    Declaration, VisionBlock, VisionInput, VisionPatchifier, VisionProjector, VisionState,
    VisionUnit,
};

/// Typed module for one retained vision weight scope.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum VisionModule<B: NeuralBackend> {
    /// Converts packed pixels to the first vision activation.
    Patchifier(VisionPatchifier<B>),
    /// Runs one encoder block at its declared ordinal.
    Block(VisionBlock<B>),
    /// Merges final encoder activations into language-width rows.
    Projector(VisionProjector<B>),
}

impl<B: NeuralBackend> VisionModule<B> {
    /// Constructs one module from its semantic address.
    pub fn new(
        declaration: &Declaration,
        unit: VisionUnit,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(match unit {
            VisionUnit::Patchifier => {
                Self::Patchifier(VisionPatchifier::new(declaration, context)?)
            }
            VisionUnit::Block(layer) => Self::Block(VisionBlock::new(declaration, layer, context)?),
            VisionUnit::Projector => Self::Projector(VisionProjector::new(declaration, context)?),
        })
    }

    /// Semantic address for binding and traversal.
    pub fn unit(&self) -> VisionUnit {
        match self {
            Self::Patchifier(_) => VisionUnit::Patchifier,
            Self::Block(block) => VisionUnit::Block(block.layer()),
            Self::Projector(_) => VisionUnit::Projector,
        }
    }
}

/// Parameter-free input or intermediate state of one media invocation.
pub struct VisionExecution<T> {
    layers: usize,
    step: VisionStep<T>,
}

enum VisionStep<T> {
    Input {
        pixels: T,
        grid: Vec<(i32, i32, i32)>,
    },
    Hidden {
        hidden: T,
        state: VisionState<T>,
        completed_blocks: usize,
    },
}

/// Submitted result of one unit. The caller establishes native completion
/// before adopting this result or releasing the unit's weight lease.
pub enum VisionAdvance<T> {
    /// Another declared unit remains.
    Continue(VisionExecution<T>),
    /// The projector has produced language-width rows.
    Projected(T),
}

impl<T: Tensor> VisionExecution<T> {
    /// Retains the admitted invocation at its patchifier boundary.
    pub fn new(declaration: &Declaration, input: VisionInput<'_, T>) -> Self {
        Self {
            layers: declaration.config().num_hidden_layers,
            step: VisionStep::Input {
                pixels: input.pixels.clone(),
                grid: input.grid.to_vec(),
            },
        }
    }

    /// Address of the next required unit.
    pub fn next_unit(&self) -> VisionUnit {
        match &self.step {
            VisionStep::Input { .. } => VisionUnit::Patchifier,
            VisionStep::Hidden {
                completed_blocks, ..
            } if *completed_blocks < self.layers => VisionUnit::Block(*completed_blocks),
            VisionStep::Hidden { .. } => VisionUnit::Projector,
        }
    }

    pub(super) fn input_or_activation(&self) -> &T {
        match &self.step {
            VisionStep::Input { pixels, .. } => pixels,
            VisionStep::Hidden { hidden, .. } => hidden,
        }
    }

    pub(super) fn retained_positions(&self) -> Option<[&T; 2]> {
        match &self.step {
            VisionStep::Input { .. } => None,
            VisionStep::Hidden { state, .. } => Some(state.retained_values()),
        }
    }

    /// Runs exactly one unit. The current continuation stays available until
    /// the caller commits the result.
    pub fn advance<B: NeuralBackend<Tensor = T>>(
        &self,
        module: &mut VisionModule<B>,
        context: &T::Context,
    ) -> Result<VisionAdvance<T>, Error> {
        self.advance_with_activation(module, self.input_or_activation(), context)
    }

    pub(super) fn advance_with_activation<B: NeuralBackend<Tensor = T>>(
        &self,
        module: &mut VisionModule<B>,
        activation: &T,
        context: &T::Context,
    ) -> Result<VisionAdvance<T>, Error> {
        if module.unit() != self.next_unit() {
            return Err(Error::backend(
                "LFM3 vision unit does not match the continuation",
            ));
        }
        let step = match (&self.step, module) {
            (VisionStep::Input { grid, .. }, VisionModule::Patchifier(module)) => {
                let (hidden, state) = module.forward(
                    VisionInput {
                        pixels: activation,
                        grid,
                    },
                    context,
                )?;
                VisionStep::Hidden {
                    hidden,
                    state,
                    completed_blocks: 0,
                }
            }
            (
                VisionStep::Hidden {
                    state,
                    completed_blocks,
                    ..
                },
                VisionModule::Block(module),
            ) => VisionStep::Hidden {
                hidden: module.forward(activation, state, context)?,
                state: state.clone(),
                completed_blocks: completed_blocks + 1,
            },
            (VisionStep::Hidden { state, .. }, VisionModule::Projector(module)) => {
                return module
                    .forward(activation, state, context)
                    .map(VisionAdvance::Projected);
            }
            _ => unreachable!("unit identity is checked before dispatch"),
        };
        Ok(VisionAdvance::Continue(Self {
            layers: self.layers,
            step,
        }))
    }
}
