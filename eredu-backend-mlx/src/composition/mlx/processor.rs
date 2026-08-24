//! Architecture selection for MLX media preprocessing.

use std::path::Path;

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
pub struct ModelProcessor {
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
                    Err(Error::Processor(
                        "MLX image preparation requires the mlx-image feature".into(),
                    ))
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
                    Err(Error::Processor(
                        "MLX video preparation requires the mlx-image feature".into(),
                    ))
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
                    Err(Error::Processor(
                        "MLX audio preparation requires the mlx-audio feature".into(),
                    ))
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
    pub fn load_gemma4(model_dir: &Path, model: &[u8]) -> Result<Option<Self>, Error> {
        gemma4::Gemma4Processor::load(model_dir, model).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Gemma4(processor),
            })
        })
    }

    #[cfg(any(feature = "image", feature = "audio"))]
    pub fn load_gemma4_gguf(
        model_metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
        projector_metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
    ) -> Result<Self, Error> {
        Ok(Self {
            kind: ProcessorKind::Gemma4(gemma4::Gemma4Processor::from_gguf(
                model_metadata,
                projector_metadata,
            )?),
        })
    }

    pub fn load_inkling(model: &[u8]) -> Result<Option<Self>, Error> {
        inkling::InklingProcessor::load(model).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Inkling(processor),
            })
        })
    }

    pub fn load_inkling_gguf(
        metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
    ) -> Result<Self, Error> {
        Ok(Self {
            kind: ProcessorKind::Inkling(inkling::InklingProcessor::from_gguf(metadata)?),
        })
    }

    #[cfg(feature = "image")]
    pub fn load_muse_glimmer(model_dir: &Path) -> Result<Option<Self>, Error> {
        muse_glimmer::MuseGlimmerProcessor::load(model_dir).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::MuseGlimmer(processor),
            })
        })
    }

    #[cfg(feature = "image")]
    pub fn load_muse_glimmer_gguf(
        projector_metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
    ) -> Result<Self, Error> {
        Ok(Self {
            kind: ProcessorKind::MuseGlimmer(muse_glimmer::MuseGlimmerProcessor::from_gguf(
                projector_metadata,
            )?),
        })
    }

    #[cfg(feature = "image")]
    pub fn load_qwen(model_dir: &Path, model: &[u8]) -> Result<Option<Self>, Error> {
        qwen::QwenProcessor::load(model_dir, model).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Qwen(processor),
            })
        })
    }

    #[cfg(feature = "image")]
    pub fn load_qwen_directory(model_dir: &Path) -> Result<Option<Self>, Error> {
        qwen::QwenProcessor::load_directory(model_dir).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Qwen(processor),
            })
        })
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

/// Loads a supported media processor without loading model weights.
pub fn load_processor(
    kind: eredu_architectures::ModelKind,
    model_dir: impl AsRef<Path>,
    config: &serde_json::Value,
) -> Result<Option<ModelProcessor>, Error> {
    let model_dir = model_dir.as_ref();
    let model = serde_json::to_vec(config)?;
    match kind {
        eredu_architectures::ModelKind::Inkling => ModelProcessor::load_inkling(&model),
        eredu_architectures::ModelKind::Gemma4 => ModelProcessor::load_gemma4(model_dir, &model),
        eredu_architectures::ModelKind::MuseGlimmer => {
            #[cfg(feature = "image")]
            {
                ModelProcessor::load_muse_glimmer(model_dir)
            }
            #[cfg(not(feature = "image"))]
            {
                Ok(None)
            }
        }
        eredu_architectures::ModelKind::Qwen3Vl
        | eredu_architectures::ModelKind::Qwen3VlMoe
        | eredu_architectures::ModelKind::Qwen35 => {
            #[cfg(feature = "image")]
            {
                ModelProcessor::load_qwen(model_dir, &model)
            }
            #[cfg(not(feature = "image"))]
            {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

#[cfg(all(test, feature = "image"))]
mod tests {
    use std::{collections::HashMap, convert::Infallible};

    use eredu_core::{Media, MultimodalRequest, MultimodalSegment, RgbImage};
    use safemlx::ops::GgufMetadataValue;

    use super::ModelProcessor;
    use crate::backend::runtime::media::input::Modality;

    #[test]
    fn mlx_processor_materializes_the_portable_ordered_request() {
        let model = HashMap::from([
            ("gemma4.boi_token_id".into(), GgufMetadataValue::Uint32(43)),
            ("gemma4.eoi_token_id".into(), GgufMetadataValue::Uint32(44)),
        ]);
        let projector = HashMap::from([
            (
                "clip.vision.patch_size".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "clip.vision.pooling_kernel_size".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "clip.vision.max_soft_tokens".into(),
                GgufMetadataValue::Uint32(70),
            ),
        ]);
        let processor = ModelProcessor::load_gemma4_gguf(&model, &projector).unwrap();
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
        assert_eq!(parts[2].modality, Modality::Image);
    }
}
