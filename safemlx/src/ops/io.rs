use crate::error::IoError;
use crate::utils::guard::Guarded;
use crate::utils::io::SafeTensors;
use crate::utils::SUCCESS;
use crate::{Array, Stream};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

fn check_file_extension(path: &Path, expected: &str) -> Result<(), IoError> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext == expected => Ok(()),
        _ => Err(IoError::UnsupportedFormat),
    }
}

impl Array {
    /// Load an array through the native MLX `.npy` reader.
    pub fn load_numpy(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, IoError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(IoError::NotFile);
        }
        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;
        check_file_extension(path, "npy")?;
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_load(res, c_path.as_ptr(), stream.as_ref().as_ptr())
        })
        .map_err(Into::into)
    }

    /// Load arrays through the native MLX `safetensors` reader.
    pub fn load_safetensors(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<HashMap<String, Array>, IoError> {
        Ok(SafeTensors::load_device(path.as_ref(), stream)?.data()?)
    }

    /// Load arrays and metadata through the native MLX `safetensors` reader.
    #[allow(clippy::type_complexity)]
    pub fn load_safetensors_with_metadata(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<(HashMap<String, Array>, HashMap<String, String>), IoError> {
        let safetensors = SafeTensors::load_device(path.as_ref(), stream)?;
        Ok((safetensors.data()?, safetensors.metadata()?))
    }

    /// Save an array through the native MLX `.npy` writer.
    pub fn save_numpy(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let path = path.as_ref();
        check_file_extension(path, "npy")?;
        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;
        unsafe { safemlx_sys::mlx_save(c_path.as_ptr(), self.as_ptr()) };
        Ok(())
    }

    /// Save arrays through the native MLX `safetensors` writer.
    pub fn save_safetensors<'a, I, S, V>(
        arrays: I,
        metadata: impl Into<Option<&'a HashMap<String, String>>>,
        path: impl AsRef<Path>,
    ) -> Result<(), IoError>
    where
        I: IntoIterator<Item = (S, V)>,
        S: AsRef<str>,
        V: AsRef<Array>,
    {
        crate::error::ensure_mlx_error_handler();
        let path = path.as_ref();
        check_file_extension(path, "safetensors")?;
        let entries = arrays.into_iter().collect::<Vec<_>>();
        crate::transforms::eval(entries.iter().map(|(_, array)| array.as_ref()))?;

        let arrays = unsafe {
            let data = safemlx_sys::mlx_map_string_to_array_new();
            for (key, array) in &entries {
                let key = CString::new(key.as_ref())?;
                if safemlx_sys::mlx_map_string_to_array_insert(
                    data,
                    key.as_ptr(),
                    array.as_ref().as_ptr(),
                ) != SUCCESS
                {
                    safemlx_sys::mlx_map_string_to_array_free(data);
                    return Err(crate::error::get_and_clear_last_mlx_error()
                        .expect("MLX returned an error status without an error")
                        .into());
                }
            }
            data
        };

        let default_metadata = HashMap::new();
        let metadata = unsafe {
            let data = safemlx_sys::mlx_map_string_to_string_new();
            for (key, value) in metadata.into().unwrap_or(&default_metadata) {
                let key = CString::new(key.as_str())?;
                let value = CString::new(value.as_str())?;
                if safemlx_sys::mlx_map_string_to_string_insert(data, key.as_ptr(), value.as_ptr())
                    != SUCCESS
                {
                    safemlx_sys::mlx_map_string_to_string_free(data);
                    safemlx_sys::mlx_map_string_to_array_free(arrays);
                    return Err(crate::error::get_and_clear_last_mlx_error()
                        .expect("MLX returned an error status without an error")
                        .into());
                }
            }
            data
        };

        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;
        unsafe {
            let status = safemlx_sys::mlx_save_safetensors(c_path.as_ptr(), arrays, metadata);
            let error = (status != SUCCESS).then(crate::error::get_and_clear_last_mlx_error);
            safemlx_sys::mlx_map_string_to_array_free(arrays);
            safemlx_sys::mlx_map_string_to_string_free(metadata);
            if let Some(error) = error.flatten() {
                return Err(error.into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Array;
    use std::path::PathBuf;

    fn io_test_dir() -> (tempfile::TempDir, PathBuf) {
        let temp_dir = tempfile::tempdir().unwrap();
        let test_dir = temp_dir.path().join("formats with spaces");
        std::fs::create_dir(&test_dir).unwrap();
        (temp_dir, test_dir)
    }

    #[test]
    fn test_save_arrays() {
        let stream = crate::test_stream();
        let (_tmp_dir, test_dir) = io_test_dir();
        let path = test_dir.join("test tensors.safetensors");
        let arrays = std::collections::HashMap::from([
            (
                "foo".to_owned(),
                Array::ones::<i32>(&[1, 2], stream).unwrap(),
            ),
            (
                "bar".to_owned(),
                Array::zeros::<i32>(&[2, 1], stream).unwrap(),
            ),
        ]);
        Array::save_safetensors(&arrays, None, &path).unwrap();
        let loaded = Array::load_safetensors(&path, stream).unwrap();
        assert_eq!(loaded.len(), arrays.len());
        for (name, original) in arrays {
            assert!(loaded[&name]
                .all_close(&original, None, None, None, stream)
                .unwrap()
                .item::<bool>(&stream));
        }
    }

    #[test]
    fn test_save_array() {
        let stream = crate::test_stream();
        let (_tmp_dir, test_dir) = io_test_dir();
        let path = test_dir.join("test array.npy");
        let original = Array::ones::<i32>(&[2, 4], stream).unwrap();
        original.save_numpy(&path).unwrap();
        let loaded = Array::load_numpy(&path, stream).unwrap();
        assert!(original
            .all_close(&loaded, None, None, None, stream)
            .unwrap()
            .item::<bool>(&stream));
    }
}
