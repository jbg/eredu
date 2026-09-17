//! Shared hidden-position selection before architecture vocabulary equations.

use eredu_core::OutputDemand;
use eredu_nn::{Error, Tensor};

/// Executes readout only for demanded hidden positions. `sequence_axis` is an
/// architecture declaration, never inferred from a family name or tensor rank.
/// The selected view retains that axis so tied, packed and parallel operators
/// receive their ordinary input geometry. Prediction captures are retained
/// separately by the caller before this selection.
pub fn execute_readout<T: Tensor>(
    hidden: &T,
    demand: OutputDemand,
    sequence_axis: usize,
    context: &T::Context,
    project: impl FnOnce(&T) -> Result<T, Error>,
) -> Result<Option<T>, Error> {
    select_readout_positions(hidden, demand, sequence_axis, context)?
        .as_ref()
        .map(project)
        .transpose()
}

/// Selects the input to an existing readout equation without modifying its
/// architecture context or captures. State-only execution skips that equation.
pub fn select_readout_positions<T: Tensor>(
    hidden: &T,
    demand: OutputDemand,
    sequence_axis: usize,
    context: &T::Context,
) -> Result<Option<T>, Error> {
    let &positions = hidden
        .shape()
        .get(sequence_axis)
        .ok_or_else(|| Error::backend("readout sequence axis is outside hidden geometry"))?;
    if positions <= 0 {
        return Err(Error::backend("readout requires nonempty hidden positions"));
    }
    match demand {
        OutputDemand::StateOnly => Ok(None),
        OutputDemand::Sequence => Ok(Some(hidden.clone())),
        // Cached decode and a one-position final prefill chunk already have the
        // required geometry. Preserve their source directly without indexing.
        OutputDemand::LastPosition if positions == 1 => Ok(Some(hidden.clone())),
        OutputDemand::LastPosition => {
            hidden
                .narrow_axis(sequence_axis, positions - 1, positions, context)
                .map(Some)
        }
    }
}
