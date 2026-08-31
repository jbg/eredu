use eredu_gguf::{DenseTensorSpan, Endian, GgmlType, MetadataValue, TensorSelection};
use safemlx::{error::IoError, Array};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn gguf_error(error: eredu_gguf::Error) -> IoError {
    IoError::InvalidFormat(error.to_string())
}

/// A validated GGUF checkpoint that materializes one physical tensor at a time.
#[derive(Debug, Clone)]
pub struct GgufCheckpoint {
    inner: eredu_gguf::Checkpoint,
}

impl std::ops::Deref for GgufCheckpoint {
    type Target = eredu_gguf::Checkpoint;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// One named MLX array produced from a GGUF tensor.
#[derive(Debug)]
pub struct GgufArray {
    name: String,
    array: Array,
}

impl GgufArray {
    /// Logical checkpoint name of the array.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Materialized MLX array.
    pub fn array(&self) -> &Array {
        &self.array
    }

    /// Consume the value into its logical name and MLX array.
    pub fn into_parts(self) -> (String, Array) {
        (self.name, self.array)
    }
}

/// Atomic MLX representation of one packed affine GGUF tensor.
#[derive(Debug)]
pub struct GgufAffineTensor {
    physical_name: String,
    bits: u8,
    group_size: u32,
    weight: GgufArray,
    scales: GgufArray,
    biases: GgufArray,
}

/// Atomic MLX representation of one GGML type-39 MXFP4 tensor.
#[derive(Debug)]
pub struct GgufMxFp4Tensor {
    physical_name: String,
    weight: GgufArray,
    scales: GgufArray,
}

impl GgufMxFp4Tensor {
    /// Name of the physical tensor in the GGUF file.
    pub fn physical_name(&self) -> &str {
        &self.physical_name
    }

    /// Packed E2M1 values in the layout consumed by MLX MXFP4 kernels.
    pub fn weight(&self) -> &GgufArray {
        &self.weight
    }

    /// One E8M0 scale byte per 32 logical values.
    pub fn scales(&self) -> &GgufArray {
        &self.scales
    }

    /// Consume the group into its packed weights and scale arrays.
    pub fn into_arrays(self) -> [GgufArray; 2] {
        [self.weight, self.scales]
    }
}

impl GgufAffineTensor {
    /// Name of the physical tensor in the GGUF file.
    pub fn physical_name(&self) -> &str {
        &self.physical_name
    }

    /// Number of quantized bits per weight.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Quantization group size.
    pub fn group_size(&self) -> u32 {
        self.group_size
    }

    /// Packed weight array.
    pub fn weight(&self) -> &GgufArray {
        &self.weight
    }

    /// Per-group scale array.
    pub fn scales(&self) -> &GgufArray {
        &self.scales
    }

    /// Per-group bias array.
    pub fn biases(&self) -> &GgufArray {
        &self.biases
    }

    /// Consume the group into weight, scales, and biases arrays.
    pub fn into_arrays(self) -> [GgufArray; 3] {
        [self.weight, self.scales, self.biases]
    }
}

/// One checkpoint-native GGML block tensor.
#[derive(Debug)]
pub struct GgufIQuantTensor {
    physical_name: String,
    ggml_type: GgmlType,
    endian: Endian,
    logical_shape: Vec<i32>,
    packed: GgufArray,
}

impl GgufIQuantTensor {
    /// Returns the physical checkpoint tensor name.
    pub fn physical_name(&self) -> &str {
        &self.physical_name
    }

    /// Returns the GGML block encoding.
    pub fn ggml_type(&self) -> GgmlType {
        self.ggml_type
    }

    /// Returns the stored byte order.
    pub fn endian(&self) -> Endian {
        self.endian
    }

    /// Returns the logical dense shape.
    pub fn logical_shape(&self) -> &[i32] {
        &self.logical_shape
    }

    /// Returns the packed MLX byte array.
    pub fn packed(&self) -> &GgufArray {
        &self.packed
    }

    /// Consumes this value and returns its packed MLX byte array.
    pub fn into_packed(self) -> GgufArray {
        self.packed
    }
}

/// One converted physical GGUF tensor.
#[derive(Debug)]
pub enum GgufTensor {
    /// A physical tensor represented by one dense MLX array.
    Dense(GgufArray),
    /// A nonlinear tensor retained in its GGML block encoding.
    IQuant(GgufIQuantTensor),
    /// A packed tensor represented by one atomic affine triple.
    Affine(GgufAffineTensor),
    /// A GGML type-39 tensor represented as MLX MXFP4 weights and scales.
    MxFp4(GgufMxFp4Tensor),
}

impl GgufTensor {
    /// Converts a backend-neutral portable GGUF group into owned host-backed MLX arrays.
    pub fn from_portable_host(
        tensor: eredu_gguf::ConvertedCheckpointTensor,
    ) -> Result<Self, IoError> {
        convert_tensor(tensor, true)
    }

    /// Name of the physical tensor in the GGUF file.
    pub fn physical_name(&self) -> &str {
        match self {
            Self::Dense(tensor) => tensor.name(),
            Self::IQuant(tensor) => tensor.physical_name(),
            Self::Affine(tensor) => tensor.physical_name(),
            Self::MxFp4(tensor) => tensor.physical_name(),
        }
    }

    /// Consume the tensor group into its logical named arrays.
    pub fn into_arrays(self) -> Vec<(String, Array)> {
        match self {
            Self::Dense(tensor) => vec![tensor.into_parts()],
            Self::IQuant(tensor) => vec![tensor.into_packed().into_parts()],
            Self::Affine(tensor) => tensor
                .into_arrays()
                .into_iter()
                .map(GgufArray::into_parts)
                .collect(),
            Self::MxFp4(tensor) => tensor
                .into_arrays()
                .into_iter()
                .map(GgufArray::into_parts)
                .collect(),
        }
    }
}

/// Fallible iterator over materialized MLX tensor groups.
pub struct GgufTensorIter<'a> {
    inner: eredu_gguf::ConvertedTensorIter<'a>,
}

/// Indexed named-tensor materializer that reuses the current shard reader.
pub struct GgufMaterializer {
    inner: eredu_gguf::TensorMaterializer,
}

/// One physical GGUF tensor retained in its checkpoint-native byte encoding.
#[derive(Debug)]
pub struct GgufRawTensor {
    inner: eredu_gguf::RawCheckpointTensor,
}

impl GgufRawTensor {
    /// Endianness declared by the containing GGUF shard.
    pub fn endian(&self) -> eredu_gguf::Endian {
        self.inner.endian()
    }

    /// Physical tensor descriptor.
    pub fn descriptor(&self) -> &eredu_gguf::TensorDescriptor {
        self.inner.descriptor()
    }

    /// Checkpoint-native payload bytes.
    pub fn data(&self) -> &[u8] {
        self.inner.data()
    }

    /// Consume this tensor and return its checkpoint-native payload.
    pub fn into_data(self) -> Vec<u8> {
        self.inner.into_data()
    }
}

impl std::fmt::Debug for GgufTensorIter<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgufTensorIter")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for GgufMaterializer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgufMaterializer")
            .finish_non_exhaustive()
    }
}

impl Iterator for GgufTensorIter<'_> {
    type Item = Result<GgufTensor, IoError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|result| {
            result
                .map_err(gguf_error)
                .and_then(|tensor| convert_tensor(tensor, false))
        })
    }
}

impl GgufCheckpoint {
    /// Wrap a checkpoint already inspected by a backend-neutral planner.
    ///
    /// No header is reopened and no tensor payload is read.
    pub fn from_portable(inner: eredu_gguf::Checkpoint) -> Self {
        Self { inner }
    }

    /// Open and validate a single-file or canonically sharded GGUF checkpoint.
    ///
    /// This parses all shard headers and descriptors, but reads no tensor payloads.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(IoError::NotFile);
        }
        if !path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            return Err(IoError::UnsupportedFormat);
        }
        Ok(Self {
            inner: eredu_gguf::Checkpoint::open(path).map_err(gguf_error)?,
        })
    }

    /// Typed metadata from the first checkpoint shard.
    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        self.inner.metadata()
    }

    /// Validated header-only checkpoint description.
    pub fn catalog(&self) -> &eredu_gguf::Checkpoint {
        &self.inner
    }

    /// Iterate over converted tensors without retaining earlier payloads.
    pub fn converted_tensors(&self) -> GgufTensorIter<'_> {
        GgufTensorIter {
            inner: self.inner.converted_tensors(),
        }
    }

    /// Create an indexed named-tensor materializer with bounded reader reuse.
    pub fn materializer(&self) -> GgufMaterializer {
        GgufMaterializer {
            inner: self.inner.materializer(),
        }
    }

    /// Materialize and visit one physical tensor at a time.
    pub fn for_each_converted_tensor<F>(&self, mut visitor: F) -> Result<(), IoError>
    where
        F: FnMut(GgufTensor) -> Result<(), IoError>,
    {
        for tensor in self.converted_tensors() {
            visitor(tensor?)?;
        }
        Ok(())
    }
}

impl GgufMaterializer {
    /// Path of the shard containing `name`, without opening its payload reader.
    pub fn shard_path_for_tensor(&self, name: &str) -> Result<&Path, IoError> {
        self.inner.shard_path_for_tensor(name).map_err(gguf_error)
    }

    /// Path of the currently cached shard reader, if any.
    pub fn open_shard_path(&self) -> Option<&Path> {
        self.inner.open_shard_path()
    }

    /// Close the currently cached shard reader.
    pub fn close_reader(&mut self) -> Option<PathBuf> {
        self.inner.close_reader()
    }

    /// Materialize one physical tensor by its GGUF name.
    pub fn converted_tensor(&mut self, name: &str) -> Result<GgufTensor, IoError> {
        convert_tensor(
            self.inner.converted_tensor(name).map_err(gguf_error)?,
            false,
        )
    }

    /// Materialize one physical tensor as owned host-backed arrays.
    ///
    /// Converted buffers transfer directly into MLX instead of being copied
    /// through the process-default device.
    pub fn converted_tensor_host(&mut self, name: &str) -> Result<GgufTensor, IoError> {
        convert_tensor(self.inner.converted_tensor(name).map_err(gguf_error)?, true)
    }

    /// Materialize a bounded selection along one MLX tensor axis.
    pub fn converted_tensor_selected(
        &mut self,
        name: &str,
        selection: &TensorSelection,
    ) -> Result<GgufTensor, IoError> {
        convert_tensor(
            self.inner
                .converted_tensor_selected(name, selection)
                .map_err(gguf_error)?,
            false,
        )
    }

    /// Materialize a bounded tensor selection as owned host-backed arrays.
    ///
    /// Converted buffers transfer directly into MLX instead of being copied
    /// through the process-default device.
    pub fn converted_tensor_selected_host(
        &mut self,
        name: &str,
        selection: &TensorSelection,
    ) -> Result<GgufTensor, IoError> {
        convert_tensor(
            self.inner
                .converted_tensor_selected(name, selection)
                .map_err(gguf_error)?,
            true,
        )
    }

    /// Materialize a bounded contiguous span from an unquantized dense tensor.
    pub fn converted_dense_tensor_span(
        &mut self,
        name: &str,
        selection: &DenseTensorSpan,
    ) -> Result<GgufTensor, IoError> {
        convert_tensor(
            self.inner
                .converted_dense_tensor_span(name, selection)
                .map_err(gguf_error)?,
            false,
        )
    }

    /// Materialize a bounded dense contiguous span as owned host-backed data.
    pub fn converted_dense_tensor_span_host(
        &mut self,
        name: &str,
        selection: &DenseTensorSpan,
    ) -> Result<GgufTensor, IoError> {
        convert_tensor(
            self.inner
                .converted_dense_tensor_span(name, selection)
                .map_err(gguf_error)?,
            true,
        )
    }

    /// Materialize one physical tensor without converting its GGUF blocks.
    pub fn raw_tensor(&mut self, name: &str) -> Result<GgufRawTensor, IoError> {
        Ok(GgufRawTensor {
            inner: self.inner.raw_tensor(name).map_err(gguf_error)?,
        })
    }
}

fn convert_tensor(
    tensor: eredu_gguf::ConvertedCheckpointTensor,
    host_owned: bool,
) -> Result<GgufTensor, IoError> {
    let (descriptor, output_names, converted) = tensor.into_parts();
    match converted {
        eredu_gguf::ConvertedTensor::Dense(dense) => {
            let [name] = converted_output_names(&descriptor.name, output_names)?;
            let shape = mlx_shape_i32(&descriptor.name, &dense.shape)?;
            let array = match dense.dtype {
                eredu_gguf::DenseDtype::F32 => array_from_owned_data(
                    decode_native(dense.data, f32::from_ne_bytes)?,
                    &shape,
                    host_owned,
                )?,
                eredu_gguf::DenseDtype::F16 => array_from_owned_data(
                    decode_native(dense.data, |bytes| {
                        half::f16::from_bits(u16::from_ne_bytes(bytes))
                    })?,
                    &shape,
                    host_owned,
                )?,
                eredu_gguf::DenseDtype::Bf16 => array_from_owned_data(
                    decode_native(dense.data, |bytes| {
                        half::bf16::from_bits(u16::from_ne_bytes(bytes))
                    })?,
                    &shape,
                    host_owned,
                )?,
                eredu_gguf::DenseDtype::I8 => array_from_owned_data(
                    dense.data.into_iter().map(|value| value as i8).collect(),
                    &shape,
                    host_owned,
                )?,
                eredu_gguf::DenseDtype::I16 => array_from_owned_data(
                    decode_native(dense.data, i16::from_ne_bytes)?,
                    &shape,
                    host_owned,
                )?,
                eredu_gguf::DenseDtype::I32 => array_from_owned_data(
                    decode_native(dense.data, i32::from_ne_bytes)?,
                    &shape,
                    host_owned,
                )?,
                eredu_gguf::DenseDtype::I64 => array_from_owned_data(
                    decode_native(dense.data, i64::from_ne_bytes)?,
                    &shape,
                    host_owned,
                )?,
                eredu_gguf::DenseDtype::F64 => array_from_owned_data(
                    decode_native(dense.data, f64::from_ne_bytes)?,
                    &shape,
                    host_owned,
                )?,
            };
            Ok(GgufTensor::Dense(GgufArray { name, array }))
        }
        eredu_gguf::ConvertedTensor::IQuant(iquant) => {
            let [name] = converted_output_names(&descriptor.name, output_names)?;
            let packed_shape = mlx_shape_i32(
                &descriptor.name,
                &iquant.packed_shape().map_err(gguf_error)?,
            )?;
            let logical_shape = mlx_shape_i32(&descriptor.name, &iquant.shape)?;
            let array = array_from_owned_data(iquant.data, &packed_shape, host_owned)?;
            Ok(GgufTensor::IQuant(GgufIQuantTensor {
                physical_name: descriptor.name.clone(),
                ggml_type: iquant.ggml_type,
                endian: iquant.endian,
                logical_shape,
                packed: GgufArray { name, array },
            }))
        }
        eredu_gguf::ConvertedTensor::Affine(affine) => {
            let [weight_name, scales_name, biases_name] =
                converted_output_names(&descriptor.name, output_names)?;
            let weight_shape = mlx_shape_i32(&descriptor.name, &affine.weight_shape)?;
            let scale_shape = mlx_shape_i32(&descriptor.name, &affine.scale_shape)?;
            let weight = array_from_owned_data(affine.weights, &weight_shape, host_owned)?;
            let scales = array_from_owned_data(affine.scales, &scale_shape, host_owned)?;
            let biases = array_from_owned_data(affine.biases, &scale_shape, host_owned)?;
            Ok(GgufTensor::Affine(GgufAffineTensor {
                physical_name: descriptor.name,
                bits: affine.bits,
                group_size: affine.group_size,
                weight: GgufArray {
                    name: weight_name,
                    array: weight,
                },
                scales: GgufArray {
                    name: scales_name,
                    array: scales,
                },
                biases: GgufArray {
                    name: biases_name,
                    array: biases,
                },
            }))
        }
        eredu_gguf::ConvertedTensor::MxFp4(mxfp4) => {
            let [weight_name, scales_name] =
                converted_output_names(&descriptor.name, output_names)?;
            let weight_shape = mlx_shape_i32(&descriptor.name, &mxfp4.weight_shape)?;
            let scale_shape = mlx_shape_i32(&descriptor.name, &mxfp4.scale_shape)?;
            let weight = array_from_owned_data(mxfp4.weights, &weight_shape, host_owned)?;
            let scales = array_from_owned_data(mxfp4.scales, &scale_shape, host_owned)?;
            Ok(GgufTensor::MxFp4(GgufMxFp4Tensor {
                physical_name: descriptor.name,
                weight: GgufArray {
                    name: weight_name,
                    array: weight,
                },
                scales: GgufArray {
                    name: scales_name,
                    array: scales,
                },
            }))
        }
    }
}

fn converted_output_names<const N: usize>(
    physical_name: &str,
    names: Vec<String>,
) -> Result<[String; N], IoError> {
    names.try_into().map_err(|names: Vec<String>| {
        IoError::InvalidFormat(format!(
            "GGUF tensor {physical_name:?} cataloged {} logical outputs, expected {N}",
            names.len()
        ))
    })
}

fn array_from_owned_data<T: safemlx::ArrayElement + 'static>(
    values: Vec<T>,
    shape: &[i32],
    host_owned: bool,
) -> Result<Array, IoError> {
    Ok(if host_owned {
        Array::try_from_owned_data(values, shape)?
    } else {
        Array::try_from_data(&values, shape)?
    })
}

fn decode_native<T, const N: usize>(
    bytes: Vec<u8>,
    decode: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>, IoError> {
    if !bytes.len().is_multiple_of(N) {
        return Err(IoError::InvalidFormat(format!(
            "dense payload length {} is not divisible by element width {N}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(N)
        .map(|chunk| decode(chunk.try_into().expect("chunk length is exact")))
        .collect())
}

fn mlx_shape_i32(name: &str, shape: &[u64]) -> Result<Vec<i32>, IoError> {
    shape
        .iter()
        .map(|&value| {
            i32::try_from(value).map_err(|_| {
                IoError::InvalidFormat(format!(
                    "tensor {name:?} dimension {value} exceeds MLX i32 shape limits"
                ))
            })
        })
        .collect()
}
