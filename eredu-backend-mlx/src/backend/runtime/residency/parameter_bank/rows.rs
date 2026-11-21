//! Row operator binding over the ordinary shared residency/completion manager.
use super::*;
#[path = "rows_memory.rs"]
mod memory;
use crate::backend::nn::native_quantization::{NativeQuantizationFormat, NativeQuantizedTensor};
use crate::backend::runtime::residency::manager::MlxResidencyResources;
use eredu_core::preparation::rows::RowDecodeBinding;
use eredu_core::{
    preparation::rows::{
        NativeRowLookupSupport, RowDecode, RowDecodeGeometry, RowDecodeRequest, RowEncoding,
        RowLookupWorkspace,
    },
    residency::AddressableStorageCapabilities,
    BackendFailure,
};
use eredu_nn::TensorElementType;

/// Decodes acquired row bytes and owns the exact native scalar, when selected.
pub struct MlxRowDecoder {
    scale: Option<MlxTensor>,
}

impl MlxRowDecoder {
    fn new(
        scale: Option<(eredu_checkpoint::recipe::RecipeDtype, MlxTensor)>,
    ) -> Result<Self, Error> {
        if let Some((expected, value)) = &scale {
            let array = value.as_array();
            if array.size() != 1
                || !matches!(
                    array.dtype(),
                    Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
                )
                || *expected
                    != crate::backend::runtime::checkpoint::recipe::recipe_dtype_from_mlx(
                        array.dtype(),
                    )
            {
                return Err(Error::ArchitectureModel(
                    "row decoder requires its exact prepared scalar companion".into(),
                ));
            }
        }
        Ok(Self {
            scale: scale.map(|(_, tensor)| tensor),
        })
    }
}

impl RowDecode<MlxNeuralBackend, MlxResidencyResources> for MlxRowDecoder {
    type Error = Error;

    type Loan = super::movement::MlxParameterLoan;

    fn begin_loan(transfer: ResidentTransfer) -> Result<Self::Loan, Error> {
        Self::Loan::begin(transfer)
    }

    fn workspace(
        &self,
        rows: u64,
        row_bytes: u64,
        geometry: RowDecodeGeometry<'_>,
    ) -> Result<u64, Error> {
        memory::workspace(rows, row_bytes, geometry.dimensions, geometry.encoding)
            .map_err(Into::into)
    }

    fn decode(
        &mut self,
        loan: &mut Self::Loan,
        binding: &str,
        geometry: RowDecodeGeometry<'_>,
        stream: &Stream,
    ) -> Result<MlxTensor, Error> {
        let acquired = loan.transfer();
        let encoded = super::movement::compact_binding(acquired, binding, stream)?;
        let rows = i32::try_from(acquired.leases().len()).map_err(|_| {
            Error::ArchitectureModel("row acquisition exceeds native indexing".into())
        })?;
        let indices = Array::arange::<_, i32>(Some(0), rows, None::<i32>, stream)?;
        let decoded = match *geometry.encoding {
            RowEncoding::Dense => encoded,
            RowEncoding::ScalarE4M3 { .. } => encoded.from_fp8(Dtype::Float32, stream)?.multiply(
                self.scale
                    .as_ref()
                    .expect("validated scale")
                    .as_array()
                    .as_dtype(Dtype::Float32, stream)?,
                stream,
            )?,
            RowEncoding::Gguf { encoding, endian } => {
                if let Some(decoder) = scalar_decoder(geometry.encoding) {
                    // Only this already admitted compact acquisition is made
                    // host-visible. No encoded copy or conversion companion is
                    // allocated; output storage was reserved before acquisition.
                    eval([&encoded])?;
                    let evaluated = encoded.evaluated()?;
                    let bytes = evaluated.as_slice::<u8>();
                    let count = decoder
                        .output_len(bytes.len())
                        .map_err(|e| Error::ArchitectureModel(e.to_string()))?;
                    if count != rows as usize * geometry.dimensions as usize {
                        return Err(Error::ArchitectureModel(
                            "encoded row result geometry differs from admission".into(),
                        ));
                    }
                    let mut decoded = vec![0f32; count];
                    decoder
                        .decode_into(bytes, &mut decoded)
                        .map_err(|e| Error::ArchitectureModel(e.to_string()))?;
                    Array::from_slice(&decoded, &[rows, geometry.dimensions])
                        .ensure_device_storage(stream)?
                } else {
                    let matrix = NativeQuantizedTensor::from_iq_array(
                        encoded,
                        &[rows, geometry.dimensions],
                        encoding,
                        endian,
                    )?;
                    matrix.embedding(&indices, stream)?
                }
            }
        };
        let dtype = match geometry.output_type {
            TensorElementType::F32 => Dtype::Float32,
            TensorElementType::F16 => Dtype::Float16,
            TensorElementType::Bf16 => Dtype::Bfloat16,
            _ => unreachable!("validated arithmetic"),
        };
        // MLX copy/contiguous may alias. Gather establishes independent
        // compact output even for one dense row, before source lease release.
        Ok(MlxTensor::from_array(
            decoded
                .as_dtype(dtype, stream)?
                .take_axis(&indices, 0, stream)?,
        ))
    }
}

/// Exact, device-independent decoder and storage facts for retained row recipes.
pub struct MlxRowLookupSupport;

fn scalar_decoder(encoding: &RowEncoding) -> Option<eredu_gguf::BlockDecoder> {
    let RowEncoding::Gguf { encoding, endian } = *encoding else {
        return None;
    };
    if NativeQuantizationFormat::from_ggml_type(encoding).is_some() {
        return None;
    }
    eredu_gguf::BlockDecoder::new(encoding, endian).ok()
}

impl NativeRowLookupSupport for MlxRowLookupSupport {
    fn row_storage(&self) -> Option<AddressableStorageCapabilities> {
        Some(AddressableStorageCapabilities::new(
            true,
            true,
            true,
            u64::MAX,
        ))
    }

    fn row_decode_memory(
        &self,
        request: RowDecodeRequest<'_>,
        logical_memory: eredu_nn::mechanism_memory::MechanismMemoryContract,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, BackendFailure> {
        memory::describe(request, logical_memory)
    }

    fn row_workspace(
        &self,
        request: RowDecodeRequest<'_>,
    ) -> Result<Option<RowLookupWorkspace>, BackendFailure> {
        if let RowEncoding::Gguf { encoding, .. } = *request.geometry.encoding {
            if NativeQuantizationFormat::from_ggml_type(encoding).is_none()
                && scalar_decoder(request.geometry.encoding).is_none()
            {
                return Ok(None);
            }
        }
        let decode_bytes = memory::workspace(
            request.rows,
            request.encoded_row_bytes,
            request.geometry.dimensions,
            request.geometry.encoding,
        )?;
        let scalar_bytes = if let Some(scale) = request.scalar {
            let Ok(workspace) =
                crate::backend::runtime::checkpoint::recipe::native_recipe_workspace(
                    scale.recipe,
                    &scale,
                )
            else {
                return Ok(None);
            };
            if crate::backend::runtime::checkpoint::recipe::preflight_mlx_recipe_metadata(
                scale.recipe,
                &scale,
                &|key| {
                    scale
                        .sources
                        .get(key)
                        .map(|entry| entry.encoding.clone())
                        .ok_or_else(|| eredu_checkpoint::store::StoreError::UnknownTensor {
                            key: key.into(),
                        })
                },
            )
            .is_err()
            {
                return Ok(None);
            }
            workspace
        } else {
            0
        };
        Ok(Some(RowLookupWorkspace {
            decode_bytes,
            scalar_bytes,
        }))
    }
}

/// Native materialization contexts for exact retained row scales.
pub struct MlxRowDecodeBinding {
    source_stream: Stream,
    stream: Stream,
}

impl MlxRowDecodeBinding {
    pub(crate) fn new(source_stream: &Stream, stream: &Stream) -> Self {
        Self {
            source_stream: source_stream.clone(),
            stream: stream.clone(),
        }
    }
}

impl RowDecodeBinding<MlxNeuralBackend, MlxResidencyResources, MlxRowDecoder>
    for MlxRowDecodeBinding
{
    fn support(&self) -> impl eredu_core::preparation::rows::NativeRowLookupSupport {
        MlxRowLookupSupport
    }

    type ParameterReads =
        crate::backend::runtime::checkpoint::store::MlxParameterMaterializationContext;

    fn parameter_reads(&self) -> Self::ParameterReads {
        Self::ParameterReads::new(&self.source_stream, &self.stream)
    }

    fn decoder(
        &self,
        scale: Option<(eredu_checkpoint::recipe::RecipeDtype, MlxTensor)>,
    ) -> Result<MlxRowDecoder, Error> {
        MlxRowDecoder::new(scale)
    }
}
