//! Cold preparation of bounded and resident execution policies.

use super::*;

/// Supplies the native weight and transfer streams for the selected placement.
pub(crate) fn parameter_residency_streams(
    distinct_transfer_stream: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Stream, Stream), Error> {
    let residency_stream = if distinct_transfer_stream {
        Stream::new_with_device(&stream.get_device()?)
    } else {
        stream.clone()
    };
    Ok((weights_stream.clone(), residency_stream))
}
