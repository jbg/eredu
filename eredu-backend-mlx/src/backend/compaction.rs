//! Device-side compaction used by MLX distributed selection.

use safemlx::{
    error::{Exception, Result},
    ops::{indexing::scatter_max_single, r#where},
    Array, Dtype, Stream,
};

/// Padded compact indices plus the device-side number of valid entries.
#[derive(Clone, Debug)]
pub(crate) struct CompactIndices {
    pub(crate) indices: Array,
    pub(crate) count: Array,
}

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

/// Compacts flattened nonzero indices into a fixed-capacity device buffer.
pub(crate) fn compact_indices(mask: &Array, stream: &Stream) -> Result<CompactIndices> {
    let flat = bool_condition(mask, stream)?.reshape(&[-1], stream)?;
    let n = i32::try_from(flat.size())
        .map_err(|_| Exception::from("array is too large to compact as i32"))?;
    let flags = flat.as_type::<i32>(stream)?;
    let count = flags.sum(None, stream)?;
    let padded = Array::full::<i32>(&[n], Array::from_int(-1), stream)?;
    if n == 0 {
        return Ok(CompactIndices {
            indices: padded,
            count,
        });
    }
    let positions = flags
        .cumsum(0, None, None, stream)?
        .subtract(Array::from_int(1), stream)?;
    let scatter_positions = r#where(&flat, positions, Array::zeros::<i32>(&[n], stream)?, stream)?;
    let flat_indices = Array::arange::<_, i32>(None, n, None, stream)?;
    let masked = r#where(&flat, flat_indices, padded.clone(), stream)?.reshape(&[n, 1], stream)?;
    let indices = scatter_max_single(padded, scatter_positions, masked, 0, stream)?;
    Ok(CompactIndices { indices, count })
}
