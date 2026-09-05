//! Operations over the single architecture-erased executable.

use safemlx::{Array, Stream};

use super::Executable;
use crate::backend::error::Error;
use crate::backend::runtime::media::input;

pub(super) fn prefill_model(
    executable: &mut Executable,
    input: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    executable.prefill(input, stream)
}

pub(super) fn decode_model(
    executable: &mut Executable,
    input: &Array,
    stream: &Stream,
) -> Result<Array, Error> {
    executable.decode(input, stream)
}
