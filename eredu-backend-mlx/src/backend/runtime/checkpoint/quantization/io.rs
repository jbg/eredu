use std::path::{Path, PathBuf};

use safemlx::{Array, Stream};
use safetensors::SafeTensors;

use crate::backend::error::Error;

/// Visits every tensor in one safetensors file as an MLX-owned array.
pub(in crate::backend::runtime::checkpoint) fn for_each_safetensor_array<F>(
    path: impl AsRef<Path>,
    stream: &Stream,
    mut f: F,
) -> Result<(), Error>
where
    F: FnMut(String, Array) -> Result<(), Error>,
{
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|err| Error::Other(Box::new(err)))?;

    for (key, view) in tensors.iter() {
        let value = Array::try_from(view).map_err(|err| Error::Other(Box::new(err)))?;
        let value = value.copy(stream)?;
        f(key.to_string(), value)?;
    }

    Ok(())
}

/// Returns the validated safetensors payloads referenced by a model directory.
pub(in crate::backend::runtime::checkpoint) fn safetensors_files(
    model_dir: impl AsRef<Path>,
) -> Result<Vec<PathBuf>, Error> {
    Ok(eredu_checkpoint::safetensors::SafetensorsShards::discover(model_dir)?.into_payload_paths())
}
