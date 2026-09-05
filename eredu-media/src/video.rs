//! Shared decoded-video validation, sampling, and timing operations.

use crate::{image::RgbImageView, MediaError};

/// Validates that a decoded frame sequence is non-empty and has stable dimensions.
pub fn validate_rgb_frames(frames: &[RgbImageView<'_>]) -> Result<(u32, u32), MediaError> {
    let first = frames
        .first()
        .ok_or_else(|| MediaError::invalid("video must contain at least one frame"))?;
    let dimensions = (first.width(), first.height());
    if let Some((index, frame)) = frames
        .iter()
        .enumerate()
        .find(|(_, frame)| (frame.width(), frame.height()) != dimensions)
    {
        return Err(MediaError::invalid(format!(
            "video frame {index} is {}x{}, expected {}x{}",
            frame.width(),
            frame.height(),
            dimensions.0,
            dimensions.1
        )));
    }
    Ok(dimensions)
}

/// Validates that a portable decoded video is non-empty with stable RGB geometry.
pub fn validate_video(video: &eredu_core::Video) -> Result<(u32, u32), MediaError> {
    let first = video
        .frames()
        .first()
        .ok_or_else(|| MediaError::invalid("video must contain at least one frame"))?;
    let dimensions = (first.width(), first.height());
    if let Some((index, frame)) = video
        .frames()
        .iter()
        .enumerate()
        .find(|(_, frame)| (frame.width(), frame.height()) != dimensions)
    {
        return Err(MediaError::invalid(format!(
            "video frame {index} is {}x{}, expected {}x{}",
            frame.width(),
            frame.height(),
            dimensions.0,
            dimensions.1
        )));
    }
    Ok(dimensions)
}

#[cfg(test)]
mod tests {
    use super::{validate_rgb_frames, validate_video};
    use crate::image::RgbImageView;
    use eredu_core::{RgbImage, Video, VideoSampling};

    #[test]
    fn validates_stable_geometry_and_rejects_mismatch() {
        let one = [0; 3];
        let two = [0; 6];
        let first = RgbImageView::packed(&one, 1, 1).unwrap();
        let second = RgbImageView::packed(&two, 2, 1).unwrap();
        assert!(validate_rgb_frames(&[]).is_err());
        assert!(validate_rgb_frames(&[first, second]).is_err());

        let video = Video::new(
            vec![
                RgbImage::new(one.to_vec(), 1, 1).unwrap(),
                RgbImage::new(two.to_vec(), 2, 1).unwrap(),
            ],
            Some(1.0),
            VideoSampling::All,
        )
        .unwrap();
        assert!(validate_video(&video).is_err());
    }
}
