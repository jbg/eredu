//! Backend-neutral deterministic host media transformations.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod audio;
mod error;
pub mod image;
pub mod video;

pub use error::MediaError;

/// Scalar dtype carried by a portable processed media buffer.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ProcessedMediaDtype {
    /// IEEE-754 single-precision floating point.
    F32,
}

#[cfg(test)]
mod feature_tests {
    #[cfg(not(feature = "audio"))]
    #[test]
    fn disabled_audio_executor_fails_without_processing() {
        let waveform = crate::audio::AudioWaveform::new(&[0.0], 16_000).unwrap();
        let error = crate::audio::extract_log_mel(
            waveform,
            &crate::audio::LogMelConfig {
                sample_rate: 16_000,
                frame_length: 1,
                hop_length: 1,
                fft_length: 2,
                mel_bins: 1,
                min_frequency: 0.0,
                max_frequency: 8_000.0,
                mel_floor: 1e-3,
                max_samples: 1,
                pad_to_multiple: 1,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("`audio` feature"));
    }

    #[cfg(not(feature = "image"))]
    #[test]
    fn disabled_image_executor_fails_without_processing() {
        let view = crate::image::RgbImageView::packed(&[0, 0, 0], 1, 1).unwrap();
        let error = crate::image::resize_rgb8_bicubic(view, 1, 1).unwrap_err();
        assert!(error.to_string().contains("`image` feature"));
    }
}
