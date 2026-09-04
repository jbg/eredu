//! Operations over the single architecture-erased executable.

use eredu_runtime::CausalModel;
use safemlx::{error::Exception, Array, Stream};

use super::Executable;
use crate::backend::error::Error;
use crate::backend::runtime::media::input;
use crate::MlxTensor;

fn prefill_pair<M, C>(
    model: &mut M,
    cache: &mut C,
    input: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<Array, Error>
where
    M: CausalModel<C, Tensor = MlxTensor, Error = Exception>,
    for<'input> M: CausalModel<C, Input<'input> = input::ModelInput<'input>>,
{
    let logits = model.prefill_input_logits(input, cache, stream)?;
    model
        .adjust_prefill_logits(logits, cache, stream)
        .map(MlxTensor::into_array)
        .map_err(Into::into)
}

fn decode_pair<M, C>(
    model: &mut M,
    cache: &mut C,
    input: &Array,
    stream: &Stream,
) -> Result<Array, Error>
where
    M: CausalModel<C, Tensor = MlxTensor, Error = Exception>,
    for<'input> M: CausalModel<C, Input<'input> = input::ModelInput<'input>>,
{
    model
        .decode_logits(&MlxTensor::from_array(input.clone()), cache, stream)
        .map(MlxTensor::into_array)
        .map_err(Into::into)
}

pub(super) fn prefill_model(
    executable: &mut Executable,
    input: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    match executable {
        Executable::DeepSeek(_, model, cache) => prefill_pair(model.as_mut(), cache, input, stream),
        Executable::Gemma4(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::GptOss(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::Inkling(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::KimiLinear(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::Lfm2(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::ReplicatedText(_, model) => model.prefill(input, stream),
        Executable::MuseGlimmer(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::NemotronH(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::Qwen(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::Qwen3Next(_, model, cache) | Executable::Qwen35(_, model, cache) => {
            prefill_pair(model, cache, input, stream)
        }
    }
}

pub(super) fn decode_model(
    executable: &mut Executable,
    input: &Array,
    stream: &Stream,
) -> Result<Array, Error> {
    match executable {
        Executable::DeepSeek(_, model, cache) => decode_pair(model.as_mut(), cache, input, stream),
        Executable::Gemma4(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::GptOss(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::Inkling(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::KimiLinear(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::Lfm2(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::ReplicatedText(_, model) => model.decode(input, stream),
        Executable::MuseGlimmer(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::NemotronH(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::Qwen(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::Qwen3Next(_, model, cache) | Executable::Qwen35(_, model, cache) => {
            decode_pair(model, cache, input, stream)
        }
    }
}
