//! Architecture selection for MLX media preprocessing.

#[cfg(feature = "image")]
use eredu_core::VideoSampling;
use eredu_core::{Media as PortableMedia, TokenizedMultimodalRequest, TokenizedMultimodalSegment};

#[cfg(feature = "image")]
use crate::{
    backend::runtime::media::RgbImageView,
    composition::{muse_glimmer_processor as muse_glimmer, qwen::processor as qwen},
};
use crate::{
    backend::{
        error::Error,
        runtime::media::{
            MediaInput, PreparedModelInput, ProcessorInput, ProcessorPreparationError,
        },
    },
    composition::{gemma4_processor as gemma4, inkling_processor as inkling},
};

/// Architecture-erased media processor selected during model composition.
#[derive(Debug, Clone)]
pub(crate) struct ModelProcessor {
    kind: ProcessorKind,
}

#[derive(Debug, Clone)]
enum ProcessorKind {
    Gemma4(gemma4::Gemma4Processor),
    Inkling(inkling::InklingProcessor),
    #[cfg(feature = "image")]
    MuseGlimmer(muse_glimmer::MuseGlimmerProcessor),
    #[cfg(feature = "image")]
    Qwen(qwen::QwenProcessor),
}

#[cfg(any(not(feature = "image"), not(feature = "audio")))]
fn missing_media_feature(modality: &str, feature: &str) -> Error {
    Error::Processor(format!(
        "MLX {modality} preparation requires feature `{feature}` on `eredu-backend-mlx` \
         (or feature `{feature}` on the `eredu` facade)"
    ))
}

enum PortableMediaView<'a> {
    #[cfg(feature = "image")]
    Image(RgbImageView<'a>),
    #[cfg(feature = "image")]
    Video {
        frames: Vec<RgbImageView<'a>>,
        source_fps: Option<f64>,
        sampling: VideoSampling,
    },
    #[cfg(feature = "audio")]
    Audio {
        samples: &'a [f32],
        sample_rate: u32,
    },
    #[allow(dead_code)]
    Unavailable(std::marker::PhantomData<&'a ()>),
}

impl<'a> PortableMediaView<'a> {
    fn new(media: &'a PortableMedia) -> Result<Self, Error> {
        match media {
            PortableMedia::Image(image) => {
                #[cfg(feature = "image")]
                {
                    Ok(Self::Image(RgbImageView::packed(
                        image.pixels(),
                        image.width(),
                        image.height(),
                    )?))
                }
                #[cfg(not(feature = "image"))]
                {
                    let _ = image;
                    Err(missing_media_feature("image", "image"))
                }
            }
            PortableMedia::Video(video) => {
                #[cfg(feature = "image")]
                {
                    let frames = video
                        .frames()
                        .iter()
                        .map(|frame| {
                            RgbImageView::packed(frame.pixels(), frame.width(), frame.height())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Self::Video {
                        frames,
                        source_fps: video.source_fps(),
                        sampling: video.sampling(),
                    })
                }
                #[cfg(not(feature = "image"))]
                {
                    let _ = video;
                    Err(missing_media_feature("video", "image"))
                }
            }
            PortableMedia::Audio(audio) => {
                #[cfg(feature = "audio")]
                {
                    Ok(Self::Audio {
                        samples: audio.samples(),
                        sample_rate: audio.sample_rate(),
                    })
                }
                #[cfg(not(feature = "audio"))]
                {
                    let _ = audio;
                    Err(missing_media_feature("audio", "audio"))
                }
            }
        }
    }

    fn as_input(&self) -> Result<MediaInput<'_>, Error> {
        match self {
            #[cfg(feature = "image")]
            Self::Image(image) => Ok(MediaInput::image_rgb8(*image)),
            #[cfg(feature = "image")]
            Self::Video {
                frames,
                source_fps,
                sampling,
            } => Ok(MediaInput::video_rgb8_with_sampling(
                frames,
                *source_fps,
                *sampling,
            )),
            #[cfg(feature = "audio")]
            Self::Audio {
                samples,
                sample_rate,
            } => MediaInput::audio_f32(samples, *sample_rate),
            Self::Unavailable(_) => unreachable!("unsupported media is rejected during mapping"),
        }
    }
}

impl ModelProcessor {
    /// Lowers the authoritative architecture-owned processor plan to MLX execution.
    pub fn from_plan(
        plan: &eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    ) -> Option<Self> {
        if let Some(processor) = plan
            .gemma4()
            .cloned()
            .and_then(gemma4::Gemma4Processor::from_plan)
        {
            return Some(Self {
                kind: ProcessorKind::Gemma4(processor),
            });
        }
        if let Some(processor) = plan.inkling().cloned() {
            return Some(Self {
                kind: ProcessorKind::Inkling(inkling::InklingProcessor::from_plan(processor)),
            });
        }
        #[cfg(feature = "image")]
        if let Some(processor) = plan.muse().cloned() {
            return Some(Self {
                kind: ProcessorKind::MuseGlimmer(muse_glimmer::MuseGlimmerProcessor::from_plan(
                    processor,
                )),
            });
        }
        #[cfg(feature = "image")]
        if let Some(processor) = plan.qwen().cloned() {
            return Some(Self {
                kind: ProcessorKind::Qwen(qwen::QwenProcessor::from_plan(processor)),
            });
        }
        None
    }

    fn prepare_input<E>(
        &self,
        input: &[ProcessorInput<'_>],
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>> {
        #[cfg(not(feature = "image"))]
        let _ = &encode_text;
        match &self.kind {
            ProcessorKind::Gemma4(processor) => processor.prepare_input(input, encode_text),
            ProcessorKind::Inkling(processor) => processor.prepare_input(input, encode_text),
            #[cfg(feature = "image")]
            ProcessorKind::MuseGlimmer(processor) => processor.prepare_input(input, encode_text),
            #[cfg(feature = "image")]
            ProcessorKind::Qwen(processor) => processor.prepare_input(input, encode_text),
        }
    }

    /// Converts a portable ordered request into owned MLX model input.
    pub fn prepare_portable_input<E>(
        &self,
        request: &TokenizedMultimodalRequest,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>> {
        let media = request
            .segments()
            .iter()
            .filter_map(|segment| match segment {
                TokenizedMultimodalSegment::Media(media) => Some(PortableMediaView::new(media)),
                TokenizedMultimodalSegment::TokenIds(_) => None,
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut media = media.iter();
        let input = request
            .segments()
            .iter()
            .map(|segment| match segment {
                TokenizedMultimodalSegment::TokenIds(ids) => Ok(ProcessorInput::TokenIds(ids)),
                TokenizedMultimodalSegment::Media(_) => Ok(ProcessorInput::Media(
                    media
                        .next()
                        .expect("one prepared view exists for every media segment")
                        .as_input()?,
                )),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        debug_assert!(media.next().is_none());
        self.prepare_input(&input, encode_text)
    }
}

#[cfg(all(test, any(not(feature = "image"), not(feature = "audio"))))]
mod feature_diagnostic_tests {
    #[cfg(not(feature = "audio"))]
    use eredu_core::Audio;
    use eredu_core::Media;
    #[cfg(not(feature = "image"))]
    use eredu_core::RgbImage;

    use super::PortableMediaView;

    #[cfg(not(feature = "image"))]
    #[test]
    fn missing_image_diagnostic_names_backend_and_facade_features() {
        let media = Media::Image(RgbImage::new(vec![0, 0, 0], 1, 1).unwrap());
        let image = PortableMediaView::new(&media)
            .err()
            .expect("image feature is disabled")
            .to_string();
        assert!(image.contains("feature `image` on `eredu-backend-mlx`"));
        assert!(image.contains("feature `image` on the `eredu` facade"));
    }

    #[cfg(not(feature = "audio"))]
    #[test]
    fn missing_audio_diagnostic_names_backend_and_facade_features() {
        let media = Media::Audio(Audio::new(vec![0.0], 16_000).unwrap());
        let audio = PortableMediaView::new(&media)
            .err()
            .expect("audio feature is disabled")
            .to_string();
        assert!(audio.contains("feature `audio` on `eredu-backend-mlx`"));
        assert!(audio.contains("feature `audio` on the `eredu` facade"));
    }
}

#[cfg(all(test, feature = "image"))]
mod tests {
    use std::{collections::HashMap, convert::Infallible};

    use eredu_architectures::processor_plan::Gemma4ProcessorPlan;
    use eredu_core::{Media, MultimodalRequest, MultimodalSegment, RgbImage};
    use eredu_gguf::MetadataValue;

    use super::{ModelProcessor, ProcessorKind};
    use eredu_core::InputModality;

    #[test]
    fn mlx_processor_materializes_the_portable_ordered_request() {
        let model = HashMap::from([
            ("gemma4.boi_token_id".into(), MetadataValue::Uint32(43)),
            ("gemma4.eoi_token_id".into(), MetadataValue::Uint32(44)),
        ]);
        let projector = HashMap::from([
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(2)),
            (
                "clip.vision.pooling_kernel_size".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "clip.vision.max_soft_tokens".into(),
                MetadataValue::Uint32(70),
            ),
        ]);
        let plan = Gemma4ProcessorPlan::from_gguf_metadata(&model, &projector).unwrap();
        let processor = ModelProcessor {
            kind: ProcessorKind::Gemma4(
                crate::composition::gemma4_processor::Gemma4Processor::from_plan(plan).unwrap(),
            ),
        };
        let request = MultimodalRequest::new(vec![
            MultimodalSegment::TokenIds(vec![7]),
            MultimodalSegment::Media(Media::Image(
                RgbImage::new(vec![128; 4 * 4 * 3], 4, 4).unwrap(),
            )),
            MultimodalSegment::TokenIds(vec![8]),
        ])
        .unwrap()
        .tokenize::<Infallible>(|_| unreachable!("request contains token ids"))
        .unwrap();

        let prepared = processor
            .prepare_portable_input(&request, &mut |_| Ok::<_, Infallible>(Vec::new()))
            .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[2].modality(), InputModality::Image);
    }
}
