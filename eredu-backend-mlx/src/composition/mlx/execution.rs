//! Operations over the single architecture-erased executable.

use eredu_runtime::{ActivationObserver as RuntimeActivationObserver, CausalModel};
use ref_cast::RefCast;
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use super::session::ArrayObserverAdapter;
use super::{Executable, MlxCompletion, MlxDistributedSession, MlxModelInput};
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
        Executable::Llama(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::MuseGlimmer(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::NemotronH(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::Qwen(_, model, cache) => prefill_pair(model, cache, input, stream),
        Executable::Qwen3Next(_, model, cache) | Executable::Qwen35(_, model, cache) => {
            prefill_pair(model, cache, input, stream)
        }
        Executable::Qwen3Vl(_, model, cache) | Executable::Qwen3VlMoe(_, model, cache) => {
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
        Executable::Llama(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::MuseGlimmer(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::NemotronH(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::Qwen(_, model, cache) => decode_pair(model, cache, input, stream),
        Executable::Qwen3Next(_, model, cache) | Executable::Qwen35(_, model, cache) => {
            decode_pair(model, cache, input, stream)
        }
        Executable::Qwen3Vl(_, model, cache) | Executable::Qwen3VlMoe(_, model, cache) => {
            decode_pair(model, cache, input, stream)
        }
    }
}

pub fn submit_prefill(
    executable: &mut Executable,
    input: MlxModelInput,
    stream: &Stream,
) -> Result<eredu_core::Submission<Array, MlxCompletion>, Error> {
    let output = input.with_borrowed(|input| prefill_model(executable, input, stream))?;
    MlxCompletion::submission(output)
}

pub fn submit_decode(
    executable: &mut Executable,
    input: Array,
    stream: &Stream,
) -> Result<eredu_core::Submission<Array, MlxCompletion>, Error> {
    MlxCompletion::submission(decode_model(executable, &input, stream)?)
}

fn last_token_logits(logits: Array, stream: &Stream) -> Result<Array, Error> {
    logits
        .try_index_device((.., -1, ..), stream)
        .map_err(Into::into)
}

pub(super) fn prefill_model_tensor_parallel(
    executable: &mut Executable,
    input: input::ModelInput<'_>,
    distributed: &MlxDistributedSession<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    let group = distributed.tensor_group().ok_or_else(|| {
        Error::Parallel("tensor-parallel model session has no tensor communicator".into())
    })?;
    let logits = match executable {
        Executable::Gemma4(_, model, cache) => model
            .prefill_tensor_parallel(input, cache, group, stream)?
            .into_array(),
        Executable::Inkling(_, model, cache) => model
            .prefill_tensor_parallel(input, cache, group, stream)?
            .into_array(),
        Executable::MuseGlimmer(_, model, cache) => model
            .prefill_tensor_parallel(input, cache, group, stream)?
            .into_array(),
        executable @ (Executable::DeepSeek(_, _, _)
        | Executable::GptOss(_, _, _)
        | Executable::KimiLinear(_, _, _)
        | Executable::Lfm2(_, _, _)
        | Executable::Llama(_, _, _)
        | Executable::NemotronH(_, _, _)
        | Executable::Qwen(_, _, _)
        | Executable::Qwen3Next(_, _, _)
        | Executable::Qwen3Vl(_, _, _)
        | Executable::Qwen3VlMoe(_, _, _)
        | Executable::Qwen35(_, _, _)) => {
            let tokens = input::text_token_ids(input, stream)?;
            forward_model_tensor_parallel(executable, &tokens, group, stream)?
        }
    };
    last_token_logits(logits, stream)
}

pub(super) fn decode_model_tensor_parallel(
    executable: &mut Executable,
    input: &Array,
    distributed: &MlxDistributedSession<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    let group = distributed.tensor_group().ok_or_else(|| {
        Error::Parallel("tensor-parallel model session has no tensor communicator".into())
    })?;
    last_token_logits(
        forward_model_tensor_parallel(executable, input, group, stream)?,
        stream,
    )
}

fn forward_model_tensor_parallel(
    executable: &mut Executable,
    input: &Array,
    group: &safemlx::distributed::Group,
    stream: &Stream,
) -> Result<Array, Error> {
    let tensor_input = MlxTensor::from_array(input.clone());
    match executable {
        Executable::GptOss(_, model, cache) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        Executable::Inkling(_, model, cache) => model
            .forward_tensor_parallel(&tensor_input, cache, group, stream)
            .map(MlxTensor::into_array),
        Executable::KimiLinear(_, model, cache) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        Executable::Lfm2(_, model, cache) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        Executable::Llama(_, model, cache) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        Executable::NemotronH(_, model, cache) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        Executable::Gemma4(_, model, cache) => model
            .forward_tensor_parallel(&tensor_input, cache, group, stream)
            .map(MlxTensor::into_array),
        Executable::Qwen(_, model, cache) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        Executable::MuseGlimmer(_, model, cache) => model
            .forward_tensor_parallel(&tensor_input, cache, group, stream)
            .map(MlxTensor::into_array),
        Executable::DeepSeek(_, _, _)
        | Executable::Qwen3Next(_, _, _)
        | Executable::Qwen3Vl(_, _, _)
        | Executable::Qwen3VlMoe(_, _, _)
        | Executable::Qwen35(_, _, _) => Err(Error::Parallel(
            "this architecture is materialized as a pipeline model for distributed execution"
                .into(),
        )),
    }
}

pub(super) fn prefill_model_tensor_parallel_with_observer(
    executable: &mut Executable,
    input: input::ModelInput<'_>,
    distributed: &MlxDistributedSession<'_>,
    stream: &Stream,
    observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
) -> Result<Array, Error> {
    let group = distributed.tensor_group().ok_or_else(|| {
        Error::Parallel("tensor-parallel model session has no tensor communicator".into())
    })?;
    let logits = match executable {
        Executable::Gemma4(_, model, cache) => model
            .prefill_tensor_parallel_with_observer(
                input,
                cache,
                group,
                stream,
                &mut ArrayObserverAdapter { inner: observer },
            )?
            .into_array(),
        Executable::Inkling(_, model, cache) => model
            .prefill_tensor_parallel_with_observer(
                input,
                cache,
                group,
                stream,
                &mut ArrayObserverAdapter { inner: observer },
            )?
            .into_array(),
        Executable::MuseGlimmer(_, model, cache) => model
            .prefill_tensor_parallel_with_observer(
                input,
                cache,
                group,
                stream,
                &mut ArrayObserverAdapter { inner: observer },
            )?
            .into_array(),
        executable @ (Executable::DeepSeek(_, _, _)
        | Executable::GptOss(_, _, _)
        | Executable::KimiLinear(_, _, _)
        | Executable::Lfm2(_, _, _)
        | Executable::Llama(_, _, _)
        | Executable::NemotronH(_, _, _)
        | Executable::Qwen(_, _, _)
        | Executable::Qwen3Next(_, _, _)
        | Executable::Qwen3Vl(_, _, _)
        | Executable::Qwen3VlMoe(_, _, _)
        | Executable::Qwen35(_, _, _)) => {
            let tokens = input::text_token_ids(input, stream)?;
            forward_model_tensor_parallel_with_observer(
                executable, &tokens, group, stream, observer,
            )?
        }
    };
    last_token_logits(logits, stream)
}

pub(super) fn forward_model_tensor_parallel_with_observer(
    executable: &mut Executable,
    input: &Array,
    group: &safemlx::distributed::Group,
    stream: &Stream,
    observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
) -> Result<Array, Error> {
    let mut observer = ArrayObserverAdapter { inner: observer };
    match executable {
        Executable::GptOss(_, model, cache) => {
            model.forward_tensor_parallel_with_observer(input, cache, group, stream, &mut observer)
        }
        Executable::Inkling(_, model, cache) => model
            .forward_tensor_parallel_with_observer(
                MlxTensor::ref_cast(input),
                cache,
                group,
                stream,
                &mut observer,
            )
            .map(MlxTensor::into_array),
        Executable::KimiLinear(_, model, cache) => {
            model.forward_tensor_parallel_with_observer(input, cache, group, stream, &mut observer)
        }
        Executable::Lfm2(_, model, cache) => {
            model.forward_tensor_parallel_with_observer(input, cache, group, stream, &mut observer)
        }
        Executable::Llama(_, model, cache) => {
            model.forward_tensor_parallel_with_observer(input, cache, group, stream, &mut observer)
        }
        Executable::NemotronH(_, model, cache) => {
            model.forward_tensor_parallel_with_observer(input, cache, group, stream, &mut observer)
        }
        Executable::Gemma4(_, model, cache) => model
            .forward_tensor_parallel_with_observer(
                MlxTensor::ref_cast(input),
                cache,
                group,
                stream,
                &mut observer,
            )
            .map(MlxTensor::into_array),
        Executable::Qwen(_, model, cache) => {
            model.forward_tensor_parallel_with_observer(input, cache, group, stream, &mut observer)
        }
        Executable::MuseGlimmer(_, model, cache) => model
            .forward_tensor_parallel_with_observer(
                MlxTensor::ref_cast(input),
                cache,
                group,
                stream,
                &mut observer,
            )
            .map(MlxTensor::into_array),
        Executable::DeepSeek(_, _, _)
        | Executable::Qwen3Next(_, _, _)
        | Executable::Qwen3Vl(_, _, _)
        | Executable::Qwen3VlMoe(_, _, _)
        | Executable::Qwen35(_, _, _) => Err(Error::Parallel(
            "this architecture is materialized as a pipeline model for distributed execution"
                .into(),
        )),
    }
}
