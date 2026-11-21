//! Device-side compaction used by MLX distributed selection.

use safemlx::{error::Result, Array, Dtype, Stream};

fn bool_condition(mask: &Array, stream: &Stream) -> Result<Array> {
    if mask.dtype() == Dtype::Bool {
        Ok(mask.clone())
    } else {
        mask.ne(Array::from_int(0), stream)
    }
}

/// Counts nonzero elements without materializing the count on the host.
pub(crate) fn count_nonzero(value: &Array, stream: &Stream) -> Result<Array> {
    bool_condition(value, stream)?
        .as_type::<i32>(stream)?
        .sum(None, stream)
}
