//! Shared decoded-video validation, sampling, and timing operations.

use crate::{backend::error::Error, backend::runtime::media::RgbImageView};

/// Validates that a decoded frame sequence is non-empty and has stable dimensions.
pub fn validate_rgb_frames(frames: &[RgbImageView<'_>]) -> Result<(u32, u32), Error> {
    let first = frames
        .first()
        .ok_or_else(|| Error::Processor("video must contain at least one frame".to_string()))?;
    let dimensions = (first.width(), first.height());
    if let Some((index, frame)) = frames
        .iter()
        .enumerate()
        .find(|(_, frame)| (frame.width(), frame.height()) != dimensions)
    {
        return Err(Error::Processor(format!(
            "video frame {index} is {}x{}, expected {}x{}",
            frame.width(),
            frame.height(),
            dimensions.0,
            dimensions.1
        )));
    }
    Ok(dimensions)
}
