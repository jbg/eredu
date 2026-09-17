//! Allocation-free text readout geometry from normalized family configuration.

use super::{GgufModelConfig, SafetensorsModelConfig};
use crate::decoder::Config;

fn positive_width(value: i32) -> Option<usize> {
    usize::try_from(value).ok().filter(|width| *width != 0)
}

impl SafetensorsModelConfig {
    pub(crate) fn text_output_width(&self) -> Option<usize> {
        positive_width(match self {
            Self::DeepSeekV3(args) => args.vocab_size,
            Self::DeepSeekV4(args) => args.vocab_size,
            Self::Gemma4(args) => args.text.vocab_size,
            Self::GptOss(args) => args.vocab_size,
            Self::Inkling(args) => {
                let padded = positive_width(args.text_config.vocab_size)?;
                return positive_width(
                    args.text_config
                        .unpadded_vocab_size
                        .unwrap_or(args.text_config.vocab_size),
                )
                .filter(|width| *width <= padded);
            }
            Self::K2Horizon(args) => args.vocabulary_size(),
            Self::KimiLinear(args) => args.vocab_size,
            Self::Llama(args) => args.vocab_size,
            Self::Gemma2(args) => args.vocabulary_size(),
            Self::Nanbeige(args) => args.vocabulary_size(),
            Self::MuseGlimmer(args) => args.vocab_size,
            Self::Lfm2(args) => args.vocab_size,
            Self::NemotronH(args) => args.vocab_size,
            Self::Qwen(args) => args.vocab_size,
            Self::QwenHybrid(args) => args.text.vocab_size,
            Self::Moshi(args) => args.text_vocabulary_size(),
            Self::QwenVl(args) => args.text.vocab_size,
        })
    }
}

impl GgufModelConfig {
    pub(crate) fn text_output_width(&self) -> Option<usize> {
        positive_width(match self {
            Self::DeepSeekV3(args) => args.vocab_size,
            Self::DeepSeekV4(args) => args.vocab_size,
            Self::Gemma4(args) => args.text.vocab_size,
            Self::GptOss(args) => args.vocab_size,
            Self::Inkling(args) => {
                let padded = positive_width(args.text_config.vocab_size)?;
                return positive_width(
                    args.text_config
                        .unpadded_vocab_size
                        .unwrap_or(args.text_config.vocab_size),
                )
                .filter(|width| *width <= padded);
            }
            Self::K2Horizon(args) => args.vocabulary_size(),
            Self::KimiLinear(args) => args.vocab_size,
            Self::Llama(args) => args.vocab_size,
            Self::Gemma2(args) => args.vocabulary_size(),
            Self::Nanbeige(args) => args.vocabulary_size(),
            Self::MuseGlimmer(args) => args.vocab_size,
            Self::Lfm2(args) => args.vocab_size,
            Self::NemotronH(args) => args.vocab_size,
            Self::Qwen(args) => args.vocab_size,
            Self::QwenHybrid(args) => args.text.vocab_size,
        })
    }
}
