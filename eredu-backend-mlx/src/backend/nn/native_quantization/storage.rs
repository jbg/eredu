use super::decode::*;
use super::dispatch::*;
use super::*;

/// Physical quantization encoding retained from a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeQuantizationFormat {
    /// GGUF/GGML Q4_K blocks: 256 weights in 144 bytes.
    GgufQ4K,
    /// GGUF/GGML Q5_K blocks: 256 weights in 176 bytes.
    GgufQ5K,
    /// GGUF/GGML Q6_K blocks: 256 weights in 210 bytes.
    GgufQ6K,
    /// GGUF/GGML Q5_1 blocks: 32 weights in 24 bytes.
    GgufQ5_1,
    /// GGUF/GGML Q8_0 blocks: 32 signed weights and one FP16 scale in 34 bytes.
    GgufQ8_0,
    /// GGUF/GGML IQ2_XXS codebook blocks.
    GgufIQ2XXS,
    /// GGUF/GGML IQ2_XS codebook blocks.
    GgufIQ2XS,
    /// GGUF/GGML IQ3_XXS codebook blocks.
    GgufIQ3XXS,
    /// GGUF/GGML IQ1_S codebook blocks.
    GgufIQ1S,
    /// GGUF/GGML IQ4_NL nonlinear blocks.
    GgufIQ4NL,
    /// GGUF/GGML IQ3_S codebook blocks.
    GgufIQ3S,
    /// GGUF/GGML IQ2_S codebook blocks.
    GgufIQ2S,
    /// GGUF/GGML IQ4_XS nonlinear blocks.
    GgufIQ4XS,
    /// GGUF/GGML IQ1_M codebook blocks.
    GgufIQ1M,
}

impl NativeQuantizationFormat {
    /// Maps a native-executable GGML type to its execution metadata.
    pub fn from_ggml_type(ty: GgmlType) -> Option<Self> {
        Some(match ty {
            GgmlType::Q4K => Self::GgufQ4K,
            GgmlType::Q5K => Self::GgufQ5K,
            GgmlType::Q6K => Self::GgufQ6K,
            GgmlType::Q5_1 => Self::GgufQ5_1,
            GgmlType::Q8_0 => Self::GgufQ8_0,
            GgmlType::IQ2XXS => Self::GgufIQ2XXS,
            GgmlType::IQ2XS => Self::GgufIQ2XS,
            GgmlType::IQ3XXS => Self::GgufIQ3XXS,
            GgmlType::IQ1S => Self::GgufIQ1S,
            GgmlType::IQ4NL => Self::GgufIQ4NL,
            GgmlType::IQ3S => Self::GgufIQ3S,
            GgmlType::IQ2S => Self::GgufIQ2S,
            GgmlType::IQ4XS => Self::GgufIQ4XS,
            GgmlType::IQ1M => Self::GgufIQ1M,
            _ => return None,
        })
    }

    /// Returns the GGML type for this native format.
    pub fn ggml_type(self) -> Option<GgmlType> {
        Some(match self {
            Self::GgufQ4K => GgmlType::Q4K,
            Self::GgufQ5K => GgmlType::Q5K,
            Self::GgufQ6K => GgmlType::Q6K,
            Self::GgufQ5_1 => GgmlType::Q5_1,
            Self::GgufQ8_0 => GgmlType::Q8_0,
            Self::GgufIQ2XXS => GgmlType::IQ2XXS,
            Self::GgufIQ2XS => GgmlType::IQ2XS,
            Self::GgufIQ3XXS => GgmlType::IQ3XXS,
            Self::GgufIQ1S => GgmlType::IQ1S,
            Self::GgufIQ4NL => GgmlType::IQ4NL,
            Self::GgufIQ3S => GgmlType::IQ3S,
            Self::GgufIQ2S => GgmlType::IQ2S,
            Self::GgufIQ4XS => GgmlType::IQ4XS,
            Self::GgufIQ1M => GgmlType::IQ1M,
        })
    }

    /// Returns `(values_per_block, bytes_per_block)`.
    pub fn block_geometry(self) -> (i32, i32) {
        match self {
            Self::GgufQ4K => (Q4_K_BLOCK_VALUES, Q4_K_BLOCK_BYTES),
            Self::GgufQ5K => (Q5_K_BLOCK_VALUES, Q5_K_BLOCK_BYTES),
            Self::GgufQ6K => (Q6_K_BLOCK_VALUES, Q6_K_BLOCK_BYTES),
            Self::GgufQ5_1 => (Q5_1_BLOCK_VALUES, Q5_1_BLOCK_BYTES),
            Self::GgufQ8_0 => (Q8_0_BLOCK_VALUES, Q8_0_BLOCK_BYTES),
            iq => {
                let (values, bytes) = iq
                    .ggml_type()
                    .expect("IQ format")
                    .block_and_bytes()
                    .expect("canonical IQ geometry");
                (values as i32, bytes as i32)
            }
        }
    }
}

/// Device execution backend selected independently of model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeExecutionBackend {
    /// Custom kernels evaluated through [`MetalKernel`].
    Metal,
    /// Transient dequantization followed by portable MLX operations.
    GenericFallback,
}

/// Persistent physical byte storage shared by one or more logical views.
#[derive(Debug)]
pub(super) struct NativeStorage {
    pub(super) format: NativeQuantizationFormat,
    pub(super) endian: GgufEndian,
    pub(super) bytes: Array,
}

impl NativeStorage {
    /// Raw MLX byte array consumed by device kernels.
    pub(super) fn bytes(&self) -> &Array {
        &self.bytes
    }
}

/// A zero-copy logical matrix-bank view over checkpoint-native physical rows.
///
/// Physical rows are `[matrix, physical_row, input]`. `row_start..row_start +
/// rows` selects the logical rows inside every matrix, which represents fused
/// gate/up group banks without splitting or repacking their bytes.
#[derive(Debug, Clone)]
pub struct NativeQuantizedTensor {
    pub(super) storage: Arc<NativeStorage>,
    pub(super) matrix_count: i32,
    pub(super) physical_rows: i32,
    pub(super) row_start: i32,
    pub(super) rows: i32,
    pub(super) columns: i32,
}

impl NativeQuantizedTensor {
    /// Copies the packed storage to another execution stream while preserving
    /// this tensor's logical matrix and row view.
    #[cfg(test)]
    pub(super) fn copy_to_stream(&self, stream: &Stream) -> Result<Self, Exception> {
        let bytes = self.storage.bytes.copy(stream)?;
        eval([&bytes])?;
        Ok(Self {
            storage: Arc::new(NativeStorage {
                format: self.storage.format,
                endian: self.storage.endian,
                bytes,
            }),
            matrix_count: self.matrix_count,
            physical_rows: self.physical_rows,
            row_start: self.row_start,
            rows: self.rows,
            columns: self.columns,
        })
    }

    /// Copies one little-endian Q4_K physical matrix bank into raw MLX storage.
    ///
    /// `shape` may be `[rows, columns]` or `[matrices, rows, columns]`.
    #[cfg(test)]
    pub(super) fn from_q4k_bytes(
        data: &[u8],
        shape: &[i32],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::from_native_bytes(
            data,
            shape,
            NativeQuantizationFormat::GgufQ4K,
            Q4_K_BLOCK_VALUES,
            Q4_K_BLOCK_BYTES,
            GgufEndian::Little,
            stream,
        )
    }

    /// Copies one little-endian Q5_K physical matrix bank into raw MLX storage.
    ///
    /// `shape` may be `[rows, columns]` or `[matrices, rows, columns]`.
    #[cfg(test)]
    pub(super) fn from_q5k_bytes(
        data: &[u8],
        shape: &[i32],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::from_native_bytes(
            data,
            shape,
            NativeQuantizationFormat::GgufQ5K,
            Q5_K_BLOCK_VALUES,
            Q5_K_BLOCK_BYTES,
            GgufEndian::Little,
            stream,
        )
    }

    /// Copies one little-endian Q6_K physical matrix bank into raw MLX storage.
    ///
    /// `shape` may be `[rows, columns]` or `[matrices, rows, columns]`.
    #[cfg(test)]
    pub(super) fn from_q6k_bytes(
        data: &[u8],
        shape: &[i32],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::from_native_bytes(
            data,
            shape,
            NativeQuantizationFormat::GgufQ6K,
            Q6_K_BLOCK_VALUES,
            Q6_K_BLOCK_BYTES,
            GgufEndian::Little,
            stream,
        )
    }

    /// Copies one little-endian Q5_1 physical matrix bank into raw MLX storage.
    ///
    /// `shape` may be `[rows, columns]` or `[matrices, rows, columns]`.
    #[cfg(test)]
    pub(super) fn from_q5_1_bytes(
        data: &[u8],
        shape: &[i32],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::from_native_bytes(
            data,
            shape,
            NativeQuantizationFormat::GgufQ5_1,
            Q5_1_BLOCK_VALUES,
            Q5_1_BLOCK_BYTES,
            GgufEndian::Little,
            stream,
        )
    }

    /// Copies one little-endian Q8_0 physical matrix bank into raw MLX storage.
    ///
    /// `shape` may be `[rows, columns]` or `[matrices, rows, columns]`.
    #[cfg(test)]
    pub(super) fn from_q8_0_bytes(
        data: &[u8],
        shape: &[i32],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::from_native_bytes(
            data,
            shape,
            NativeQuantizationFormat::GgufQ8_0,
            Q8_0_BLOCK_VALUES,
            Q8_0_BLOCK_BYTES,
            GgufEndian::Little,
            stream,
        )
    }

    /// Retains an already materialized MLX byte array in native GGML blocks.
    pub fn from_iq_array(
        bytes: Array,
        shape: &[i32],
        ty: GgmlType,
        endian: GgufEndian,
    ) -> Result<Self, Exception> {
        let format = NativeQuantizationFormat::from_ggml_type(ty)
            .ok_or_else(|| Exception::custom(format!("{ty:?} has no native execution support")))?;
        let (block_values, block_bytes) = format.block_geometry();
        Self::from_native_array(bytes, shape, format, block_values, block_bytes, endian)
    }

    #[cfg(test)]
    pub(super) fn from_native_bytes(
        data: &[u8],
        shape: &[i32],
        format: NativeQuantizationFormat,
        block_values: i32,
        block_bytes: i32,
        endian: GgufEndian,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let (matrix_count, physical_rows, columns) = match shape {
            [rows, columns] => (1, *rows, *columns),
            [matrices, rows, columns] => (*matrices, *rows, *columns),
            _ => {
                return Err(Exception::custom(format!(
                    "native {format:?} expects rank-2 or rank-3 shape, got {shape:?}"
                )))
            }
        };
        if matrix_count <= 0 || physical_rows <= 0 || columns <= 0 || columns % block_values != 0 {
            return Err(Exception::custom(format!(
                "invalid native {format:?} shape {shape:?}; input columns must be a positive multiple of {block_values}"
            )));
        }
        let expected = i64::from(matrix_count)
            * i64::from(physical_rows)
            * i64::from(columns / block_values)
            * i64::from(block_bytes);
        if expected != data.len() as i64 {
            return Err(Exception::custom(format!(
                "native {format:?} payload has {} bytes, expected {expected} for shape {shape:?}",
                data.len()
            )));
        }
        let len = i32::try_from(data.len())
            .map_err(|_| Exception::custom("native tensor exceeds MLX i32 array limits"))?;
        let source = Array::from_slice(data, &[len]);
        let bytes = source.copy(stream)?;
        eval([&bytes])?;
        let storage = Arc::new(NativeStorage {
            format,
            endian,
            bytes,
        });
        Ok(Self {
            storage,
            matrix_count,
            physical_rows,
            row_start: 0,
            rows: physical_rows,
            columns,
        })
    }

    pub(super) fn from_native_array(
        bytes: Array,
        shape: &[i32],
        format: NativeQuantizationFormat,
        block_values: i32,
        block_bytes: i32,
        endian: GgufEndian,
    ) -> Result<Self, Exception> {
        let (matrix_count, physical_rows, columns) = match shape {
            [rows, columns] => (1, *rows, *columns),
            [matrices, rows, columns] => (*matrices, *rows, *columns),
            _ => {
                return Err(Exception::custom(format!(
                    "native {format:?} expects rank-2 or rank-3 shape, got {shape:?}"
                )))
            }
        };
        if bytes.dtype() != Dtype::Uint8 {
            return Err(Exception::custom(format!(
                "native {format:?} storage must be uint8, got {:?}",
                bytes.dtype()
            )));
        }
        let expected = i64::from(matrix_count)
            * i64::from(physical_rows)
            * i64::from(columns / block_values)
            * i64::from(block_bytes);
        if matrix_count <= 0
            || physical_rows <= 0
            || columns <= 0
            || columns % block_values != 0
            || expected != bytes.size() as i64
        {
            return Err(Exception::custom(format!(
                "native {format:?} storage shape mismatch: logical {shape:?}, {} bytes",
                bytes.size()
            )));
        }
        Ok(Self {
            storage: Arc::new(NativeStorage {
                format,
                endian,
                bytes,
            }),
            matrix_count,
            physical_rows,
            row_start: 0,
            rows: physical_rows,
            columns,
        })
    }

    /// Physical storage shared by this logical view.
    #[cfg(test)]
    pub(super) fn storage(&self) -> &Arc<NativeStorage> {
        &self.storage
    }

    /// Native encoding.
    pub fn format(&self) -> NativeQuantizationFormat {
        self.storage.format
    }

    /// Logical shape, including an group/matrix dimension when present.
    pub fn shape(&self) -> Vec<i32> {
        if self.matrix_count == 1 {
            vec![self.rows, self.columns]
        } else {
            vec![self.matrix_count, self.rows, self.columns]
        }
    }

    /// Logical starting row within every physical matrix.
    #[cfg(test)]
    pub(super) fn row_start(&self) -> i32 {
        self.row_start
    }

    /// Number of physical rows per matrix.
    #[cfg(test)]
    pub(super) fn physical_rows(&self) -> i32 {
        self.physical_rows
    }

    /// Creates a zero-copy logical row segment inside every physical matrix.
    #[cfg(test)]
    pub(super) fn row_view(&self, row_start: i32, rows: i32) -> Result<Self, Exception> {
        if row_start < 0 || rows <= 0 || row_start + rows > self.rows {
            return Err(Exception::custom(format!(
                "native row view {row_start}..{} exceeds {} logical rows",
                row_start + rows,
                self.rows
            )));
        }
        Ok(Self {
            storage: Arc::clone(&self.storage),
            matrix_count: self.matrix_count,
            physical_rows: self.physical_rows,
            row_start: self.row_start + row_start,
            rows,
            columns: self.columns,
        })
    }

    /// Applies this native matrix to `input`.
    ///
    /// Metal uses direct block dequantization. CPU and unsupported devices use
    /// a transient float matrix without retaining affine companions.
    pub fn linear(
        &self,
        input: &Array,
        transpose: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if self.matrix_count != 1 {
            return Err(Exception::custom(
                "native linear expects a single logical matrix",
            ));
        }
        if native_execution_backend(stream)? == NativeExecutionBackend::Metal && transpose {
            return match self.format() {
                NativeQuantizationFormat::GgufQ4K => q4k_linear_metal(input, self, stream),
                NativeQuantizationFormat::GgufQ5K => q5k_linear_metal(input, self, stream),
                NativeQuantizationFormat::GgufQ6K => q6k_linear_metal(input, self, stream),
                NativeQuantizationFormat::GgufQ8_0 => q8_0_linear_metal(input, self, stream),
                NativeQuantizationFormat::GgufQ5_1 => q5_1_linear_metal(input, self, stream),
                _ => iq_linear_metal(input, self, stream),
            };
        }
        self.linear_fallback(input, transpose, stream)
    }

    /// Looks up and dequantizes selected rows from a native matrix.
    pub fn embedding(&self, indices: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.matrix_count != 1 {
            return Err(Exception::custom(
                "native embedding expects a single logical matrix",
            ));
        }
        if native_execution_backend(stream)? == NativeExecutionBackend::Metal {
            return match self.format() {
                NativeQuantizationFormat::GgufQ4K => q4k_embedding_metal(indices, self, stream),
                NativeQuantizationFormat::GgufQ8_0 => q8_0_embedding_metal(indices, self, stream),
                NativeQuantizationFormat::GgufQ5_1 => q5_1_embedding_metal(indices, self, stream),
                _ => iq_embedding_metal(indices, self, stream),
            };
        }
        self.embedding_cpu_streaming(indices, stream)
    }

    /// Transiently dequantizes this logical view to float32.
    pub fn dequantize(&self, stream: &Stream) -> Result<Array, Exception> {
        let evaluated = self.storage.bytes.evaluated()?;
        let raw = evaluated.as_slice::<u8>();
        let values = match self.format() {
            NativeQuantizationFormat::GgufQ4K => decode_q4k_view(raw, self)?,
            NativeQuantizationFormat::GgufQ5_1 => decode_q5_1_view(raw, self)?,
            NativeQuantizationFormat::GgufQ8_0 => decode_q8_0_view(raw, self)?,
            format => {
                let ty = format.ggml_type().expect("IQ format");
                let physical_shape = if self.matrix_count == 1 {
                    vec![self.physical_rows as u64, self.columns as u64]
                } else {
                    vec![
                        self.matrix_count as u64,
                        self.physical_rows as u64,
                        self.columns as u64,
                    ]
                };
                let tensor = eredu_gguf::IQuantTensor {
                    shape: physical_shape,
                    ggml_type: ty,
                    endian: self.storage.endian,
                    data: raw.to_vec(),
                };
                let all = tensor
                    .dequantize_f32()
                    .map_err(|error| Exception::custom(error.to_string()))?;
                if self.row_start == 0 && self.rows == self.physical_rows {
                    all
                } else {
                    let mut selected = Vec::with_capacity(
                        self.matrix_count as usize * self.rows as usize * self.columns as usize,
                    );
                    let matrix_stride = self.physical_rows as usize * self.columns as usize;
                    for matrix in 0..self.matrix_count as usize {
                        let start = matrix * matrix_stride
                            + self.row_start as usize * self.columns as usize;
                        let end = start + self.rows as usize * self.columns as usize;
                        selected.extend_from_slice(&all[start..end]);
                    }
                    selected
                }
            }
        };
        let shape = if self.matrix_count == 1 {
            vec![self.rows, self.columns]
        } else {
            vec![self.matrix_count, self.rows, self.columns]
        };
        let dense = Array::from_slice(&values, &shape).copy(stream)?;
        eval([&dense])?;
        Ok(dense)
    }

    pub(super) fn linear_fallback(
        &self,
        input: &Array,
        transpose: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if native_execution_backend(stream)? == NativeExecutionBackend::GenericFallback {
            return self.linear_cpu_streaming(input, transpose, stream);
        }
        let dense = self.dequantize(stream)?;
        if transpose {
            matmul(input, dense.transpose(stream)?, stream)
        } else {
            matmul(input, dense, stream)
        }
    }

    pub(super) fn linear_cpu_streaming(
        &self,
        input: &Array,
        transpose: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let expected = if transpose { self.columns } else { self.rows };
        if input.dim(-1) != expected {
            return Err(Exception::custom(format!(
                "native CPU linear expected trailing dimension {expected}, got {:?}",
                input.shape()
            )));
        }
        let outer = input.size() as i32 / expected;
        let input = input.as_dtype(Dtype::Float32, stream)?;
        eval([&input, self.storage.bytes()])?;
        let evaluated_input = input.evaluated()?;
        let input_values = evaluated_input.as_slice::<f32>();
        let evaluated_storage = self.storage.bytes.evaluated()?;
        let raw = evaluated_storage.as_slice::<u8>();

        let output_width = if transpose { self.rows } else { self.columns };
        let mut output = vec![0.0f32; outer as usize * output_width as usize];
        if transpose {
            for output_row in 0..self.rows {
                let weights = decode_native_row(raw, self, 0, output_row)?;
                for input_row in 0..outer as usize {
                    let input_start = input_row * self.columns as usize;
                    output[input_row * self.rows as usize + output_row as usize] = dot_f32(
                        &input_values[input_start..input_start + self.columns as usize],
                        &weights,
                    );
                }
            }
        } else {
            for weight_row in 0..self.rows {
                let weights = decode_native_row(raw, self, 0, weight_row)?;
                for input_row in 0..outer as usize {
                    let scale = input_values[input_row * self.rows as usize + weight_row as usize];
                    let output_row = &mut output[input_row * self.columns as usize
                        ..(input_row + 1) * self.columns as usize];
                    for (value, weight) in output_row.iter_mut().zip(&weights) {
                        *value += scale * weight;
                    }
                }
            }
        }
        let mut shape = input.shape()[..input.ndim() - 1].to_vec();
        shape.push(output_width);
        Array::from_slice(&output, &shape).copy(stream)
    }

    pub(super) fn embedding_cpu_streaming(
        &self,
        indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let indices = indices.as_dtype(Dtype::Int32, stream)?;
        eval([&indices, self.storage.bytes()])?;
        let evaluated_indices = indices.evaluated()?;
        let index_values = evaluated_indices.as_slice::<i32>();
        let evaluated_storage = self.storage.bytes.evaluated()?;
        let raw = evaluated_storage.as_slice::<u8>();
        let mut output = Vec::with_capacity(index_values.len() * self.columns as usize);
        for &index in index_values {
            if index < 0 || index >= self.rows {
                return Err(Exception::custom(format!(
                    "native embedding index {index} is outside 0..{}",
                    self.rows
                )));
            }
            output.extend(decode_native_row(raw, self, 0, index)?);
        }
        let mut shape = indices.shape().to_vec();
        shape.push(self.columns);
        Array::from_slice(&output, &shape).copy(stream)
    }
}
