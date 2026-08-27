//! Thinking Machines Lab Inkling image and dMel host preprocessing.

#[cfg(feature = "image")]
use eredu_architectures::processor_plan::InklingImagePlan;
use eredu_architectures::processor_plan::InklingProcessorPlan;
#[cfg(feature = "image")]
use eredu_architectures::processor_plan::ProcessorPlanError;
#[cfg(feature = "audio")]
use eredu_architectures::processor_plan::{
    AudioFrameCount, AudioWindow, InklingAudioPlan, Logarithm, MelNormalization, MelScale,
    SpectrumValue,
};
#[cfg(feature = "audio")]
use eredu_core::{InputExtent, InputMetadataKey};
#[cfg(any(feature = "image", feature = "audio"))]
use safemlx::Array;

use crate::backend::error::Error;
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_core::InputModality;

#[cfg(any(feature = "image", feature = "audio"))]
use crate::backend::runtime::media::MediaPayload;
use crate::backend::runtime::media::{
    media_input_part, prepared_model_input, push_text_token_ids, InputPart, MediaInput,
    PreparedModelInput, ProcessorInput, ProcessorPreparationError,
};

#[derive(Debug, Clone)]
pub struct InklingProcessor {
    plan: InklingProcessorPlan,
}

impl InklingProcessor {
    pub fn from_plan(plan: InklingProcessorPlan) -> Self {
        Self { plan }
    }

    pub fn prepare_input<E>(
        &self,
        input: &[ProcessorInput<'_>],
        _: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>> {
        let mut parts = Vec::new();
        for item in input {
            match *item {
                ProcessorInput::TokenIds(ids) => push_text_token_ids(&mut parts, ids)?,
                ProcessorInput::Media(media) => self.push_media(&mut parts, media)?,
            }
        }
        Ok(prepared_model_input(parts)?)
    }

    fn push_media(&self, _parts: &mut Vec<InputPart>, media: MediaInput<'_>) -> Result<(), Error> {
        match (media.modality, media.payload) {
            #[cfg(feature = "image")]
            (InputModality::Image, MediaPayload::Rgb8(image)) => {
                let plan = self
                    .plan
                    .image(image.height() as usize, image.width() as usize)
                    .map_err(processor_error)?;
                push_text_token_ids(_parts, &[plan.start_token_id])?;
                _parts.push(process_image(image, plan)?);
                Ok(())
            }
            #[cfg(feature = "audio")]
            (InputModality::Audio, MediaPayload::AudioF32(waveform)) => {
                let plan = self.plan.audio().map_err(processor_error)?;
                push_text_token_ids(_parts, &[plan.start_token_id])?;
                _parts.push(self.process_audio(waveform, plan)?);
                Ok(())
            }
            _ => Err(Error::Processor(format!(
                "Inkling processor does not support {} media with the enabled features",
                media.modality.as_str()
            ))),
        }
    }

    #[cfg(feature = "audio")]
    fn process_audio(
        &self,
        waveform: crate::backend::runtime::media::audio::AudioWaveform<'_>,
        plan: InklingAudioPlan,
    ) -> Result<InputPart, Error> {
        let features = inkling_log_mel(waveform, plan)?;
        let span = (plan.dmel_max - plan.dmel_min) as f64;
        let centers = (0..plan.dmel_bins)
            .map(|index| plan.dmel_min as f64 + span * index as f64 / (plan.dmel_bins - 1) as f64)
            .collect::<Vec<_>>();
        let ids = features
            .iter()
            .map(|value| {
                let value = (*value as f64).clamp(plan.dmel_min as f64, plan.dmel_max as f64);
                centers
                    .iter()
                    .enumerate()
                    .min_by(|(_, left), (_, right)| {
                        (value - **left).abs().total_cmp(&(value - **right).abs())
                    })
                    .map(|(index, _)| index as i32)
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        let frames = ids.len() / plan.mel_bins;
        let tensor = Array::from_slice(&ids, &[1, frames as i32, plan.mel_bins as i32]);
        let mask = Array::from_slice(&vec![true; frames], &[1, frames as i32]);
        media_input_part(
            InputModality::Audio,
            tensor,
            [(InputMetadataKey::AudioMask, mask)],
            [InputExtent::AudioValidFrames(frames)],
        )
    }
}

#[cfg(feature = "image")]
fn processor_error(error: ProcessorPlanError) -> Error {
    Error::Processor(error.to_string())
}

#[cfg(feature = "image")]
fn process_image(
    image: crate::backend::runtime::media::image::RgbImageView<'_>,
    plan: InklingImagePlan,
) -> Result<InputPart, Error> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let pixels = image.packed_pixels();
    let patch = plan.patch_size;
    let mut output = Vec::with_capacity(
        plan.patch_rows * plan.patch_columns * plan.temporal_patch_size * patch * patch * 3,
    );
    for row in 0..plan.patch_rows {
        for col in 0..plan.patch_columns {
            let mut values = vec![0.0f32; plan.temporal_patch_size * patch * patch * 3];
            for time in 0..plan.temporal_patch_size {
                for y in 0..patch {
                    for x in 0..patch {
                        let source_y = row * patch + y;
                        let source_x = col * patch + x;
                        for channel in 0..3 {
                            let raw = if source_y < height && source_x < width {
                                pixels[(source_y * width + source_x) * 3 + channel] as f32
                            } else {
                                plan.padding_value
                            };
                            let normalized = (raw * plan.rescale_factor - plan.mean[channel])
                                / plan.std[channel];
                            values[((time * patch + y) * patch + x) * 3 + channel] = normalized;
                        }
                    }
                }
            }
            output.extend(values);
        }
    }
    media_input_part(
        InputModality::Image,
        Array::from_slice(
            &output,
            &[
                (plan.patch_rows * plan.patch_columns) as i32,
                plan.temporal_patch_size as i32,
                patch as i32,
                patch as i32,
                3,
            ],
        ),
        [],
        [],
    )
}

#[cfg(feature = "audio")]
fn inkling_log_mel(
    waveform: crate::backend::runtime::media::audio::AudioWaveform<'_>,
    plan: InklingAudioPlan,
) -> Result<Vec<f32>, Error> {
    use rustfft::{num_complex::Complex32, FftPlanner};

    if waveform.sample_rate() != plan.sample_rate {
        return Err(Error::Processor(format!(
            "Inkling audio requires {} Hz PCM, got {} Hz",
            plan.sample_rate,
            waveform.sample_rate()
        )));
    }
    let samples = waveform.samples();
    let frames = match plan.framing.frame_count {
        AudioFrameCount::InputDivHopCeil => samples.len().div_ceil(plan.hop_length),
    };
    let padded_len = frames
        .checked_sub(1)
        .map_or(plan.framing.leading_zeros, |last_frame| {
            plan.framing.leading_zeros.max(
                last_frame
                    .saturating_mul(plan.hop_length)
                    .saturating_add(plan.fft_length),
            )
        });
    let mut padded = vec![plan.framing.trailing_padding_value; padded_len];
    let waveform_end = plan.framing.leading_zeros + samples.len();
    if waveform_end > padded.len() {
        return Err(Error::Processor(
            "Inkling audio framing does not contain the waveform".into(),
        ));
    }
    padded[plan.framing.leading_zeros..waveform_end].copy_from_slice(samples);
    let filters = mel_filters(plan);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(plan.fft_length);
    let mut spectrum = vec![Complex32::default(); plan.fft_length];
    let mut output = vec![0.0f32; frames * plan.mel_bins];
    for frame in 0..frames {
        spectrum.fill(Complex32::default());
        let start = frame * plan.hop_length;
        for index in 0..plan.fft_length {
            let window = match plan.window {
                AudioWindow::PeriodicHann => {
                    0.5 - 0.5
                        * (2.0 * std::f32::consts::PI * index as f32 / plan.fft_length as f32).cos()
                }
            };
            spectrum[index].re = padded[start + index] * window;
        }
        fft.process(&mut spectrum);
        for mel in 0..plan.mel_bins {
            let mut energy = 0.0f32;
            for frequency in 0..=plan.fft_length / 2 {
                let spectrum_value = match plan.spectrum {
                    SpectrumValue::Magnitude => spectrum[frequency].norm(),
                };
                energy += spectrum_value * filters[mel * (plan.fft_length / 2 + 1) + frequency];
            }
            output[frame * plan.mel_bins + mel] = match plan.logarithm {
                Logarithm::Base10 => energy.max(plan.energy_floor).log10(),
            };
        }
    }
    Ok(output)
}

#[cfg(feature = "audio")]
fn mel_filters(plan: InklingAudioPlan) -> Vec<f32> {
    let hz_to_mel = |hz: f64| match plan.mel_scale {
        MelScale::Slaney if hz < 1_000.0 => hz / (200.0 / 3.0),
        MelScale::Slaney => 15.0 + (hz / 1_000.0).ln() / (6.4f64.ln() / 27.0),
    };
    let mel_to_hz = |mel: f64| match plan.mel_scale {
        MelScale::Slaney if mel < 15.0 => mel * (200.0 / 3.0),
        MelScale::Slaney => 1_000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp(),
    };
    let mel_min = hz_to_mel(plan.min_frequency as f64);
    let mel_max = hz_to_mel(plan.max_frequency as f64);
    let edges = (0..plan.mel_bins + 2)
        .map(|index| {
            mel_to_hz(mel_min + (mel_max - mel_min) * index as f64 / (plan.mel_bins + 1) as f64)
        })
        .collect::<Vec<_>>();
    let frequency_bins = plan.fft_length / 2 + 1;
    let mut filters = vec![0.0f32; plan.mel_bins * frequency_bins];
    for mel in 0..plan.mel_bins {
        let normalization = match plan.mel_normalization {
            MelNormalization::SlaneyArea => 2.0 / (edges[mel + 2] - edges[mel]),
        };
        for frequency in 0..frequency_bins {
            let hz = plan.sample_rate as f64 * frequency as f64 / plan.fft_length as f64;
            let lower = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
            let upper = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
            filters[mel * frequency_bins + frequency] =
                (lower.min(upper).max(0.0) * normalization) as f32;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "image", feature = "audio"))]
    #[test]
    fn gguf_processor_uses_facade_resolved_media_markers() {
        use eredu_architectures::processor_plan::InklingProcessorPlan;

        let plan = InklingProcessorPlan::from_gguf_token_ids(1, 0).unwrap();
        let processor = super::InklingProcessor::from_plan(plan);
        #[cfg(feature = "image")]
        assert_eq!(processor.plan.image(40, 40).unwrap().start_token_id, 1);
        #[cfg(feature = "audio")]
        assert_eq!(processor.plan.audio().unwrap().start_token_id, 0);
    }

    #[cfg(feature = "image")]
    #[test]
    fn exact_patch_width_keeps_reference_extra_column() {
        let plan = eredu_architectures::processor_plan::InklingProcessorPlan::from_hf_json(
            br#"{"model_type":"inkling_mm_model"}"#,
        )
        .unwrap()
        .unwrap();
        let exact = plan.image(40, 40).unwrap();
        assert_eq!((exact.patch_rows, exact.patch_columns), (1, 2));
        let partial = plan.image(41, 39).unwrap();
        assert_eq!((partial.patch_rows, partial.patch_columns), (2, 1));
    }

    #[cfg(feature = "audio")]
    #[test]
    fn dmel_frontend_uses_fifty_millisecond_frames() {
        let samples = vec![0.0f32; 801];
        let waveform =
            crate::backend::runtime::media::AudioWaveform::new(&samples, 16_000).unwrap();
        let plan = eredu_architectures::processor_plan::InklingProcessorPlan::from_hf_json(
            br#"{"model_type":"inkling_mm_model"}"#,
        )
        .unwrap()
        .unwrap()
        .audio()
        .unwrap();
        let features = super::inkling_log_mel(waveform, plan).unwrap();
        assert_eq!(features.len(), 2 * 80);
        assert!(features.iter().all(|value| (*value + 10.0).abs() < 1e-6));
    }
}
