//! MLX checkpoint adapter for the backend-neutral Mimi architecture.

use std::{fs::File, path::Path};

use eredu_codec::{
    mimi::{checkpoint_tensor_plan, CheckpointTensorLayout, Config, Mimi},
    Error,
};
use memmap2::MmapOptions;
use safemlx::{Array, Stream};
use safetensors::SafeTensors;

use crate::MlxTensor;

/// Loads a released Mimi SafeTensors checkpoint into the MLX backend.
///
/// Checkpoint naming and layout decisions come from `eredu-codec`; this
/// adapter owns file mapping, MLX array construction, device-side layout
/// conversion, and stream-local copies.
pub fn load(
    path: impl AsRef<Path>,
    num_codebooks: Option<i32>,
    stream: &Stream,
) -> Result<Mimi<MlxTensor>, Error> {
    let mut model = Mimi::new(Config::v0_1(num_codebooks), stream)?;
    model.load_parameters(load_checkpoint_parameters(path, stream)?)?;
    Ok(model)
}

fn load_checkpoint_parameters(
    path: impl AsRef<Path>,
    stream: &Stream,
) -> Result<Vec<(String, MlxTensor)>, Error> {
    let file = File::open(path)?;
    // SAFETY: `mmap` is kept alive while every SafeTensors view is converted
    // into an owned, stream-local MLX array below. No mapped view escapes this
    // function.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(boxed)?;
    let mut loaded = Vec::new();
    for (checkpoint_name, view) in tensors.iter() {
        let Some(plan) = checkpoint_tensor_plan(checkpoint_name) else {
            continue;
        };
        let mut value = Array::try_from(view).map_err(boxed)?;
        if let CheckpointTensorLayout::Transpose3d(axes) = plan.layout {
            if value.shape().len() == 3 {
                value = value
                    .transpose_axes(&axes, stream)
                    .map_err(eredu_nn::Error::backend)?;
            }
        }
        let value = value.copy(stream).map_err(eredu_nn::Error::backend)?;
        loaded.push((plan.parameter, MlxTensor::from_array(value)));
    }
    Ok(loaded)
}

fn boxed(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Other(Box::new(error))
}
