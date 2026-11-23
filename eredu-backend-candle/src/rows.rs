//! Native decoding of leased row acquisitions.

use candle_core::{DType, Tensor as NativeTensor};
use eredu_checkpoint::recipe::RecipeDtype;
use eredu_core::{
    preparation::rows::{
        NativeRowLookupSupport, RowDecode, RowDecodeGeometry, RowDecodeRequest, RowEncoding,
        RowLookupWorkspace,
    },
    residency::{AddressableStorageCapabilities, NativeBankMechanisms, ParameterLoan},
    BackendFailure,
};
use eredu_nn::{Error, TensorElementType};

use crate::{
    residency::{CandleParameterLoan, CandleWeightResidency, CandleWeightTransfer},
    CandleContext, CandleNeuralBackend, CandlePlacement, CandleTensor,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct CandleRowLookupSupport(CandlePlacement);

impl CandleRowLookupSupport {
    pub(crate) fn new(placement: CandlePlacement) -> Self {
        Self(placement)
    }
}

fn dimensions(geometry: RowDecodeGeometry<'_>) -> Result<(usize, DType), BackendFailure> {
    let width = usize::try_from(geometry.dimensions).map_err(BackendFailure::from_error)?;
    let dtype = match geometry.output_type {
        TensorElementType::F32 => DType::F32,
        TensorElementType::F16 => DType::F16,
        TensorElementType::Bf16 => DType::BF16,
        _ => {
            return Err(crate::text::invalid(
                "row decoding requires a floating output dtype",
            ))
        }
    };
    if width == 0 {
        return Err(crate::text::invalid(
            "row decoding requires a positive width",
        ));
    }
    Ok((width, dtype))
}

fn workspace(
    rows: u64,
    bytes: u64,
    geometry: RowDecodeGeometry<'_>,
) -> Result<u64, BackendFailure> {
    let (width, _) = dimensions(geometry)?;
    if rows == 0 || rows > i32::MAX as u64 {
        return Err(crate::text::invalid(
            "row decode count exceeds native geometry",
        ));
    }
    // Compaction and host transfer may each copy encoded bytes. Include
    // decoded host, native staging and the final cast/independent copy.
    bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(width as u64 * 16))
        .and_then(|bytes| bytes.checked_mul(rows))
        .ok_or_else(|| crate::text::invalid("row decode workspace overflows"))
}

impl NativeRowLookupSupport for CandleRowLookupSupport {
    fn row_storage(&self) -> Option<AddressableStorageCapabilities> {
        self.0
            .supports_local_text_extensions()
            .then_some(AddressableStorageCapabilities::new(
                true,
                true,
                true,
                usize::MAX as u64,
            ))
    }

    fn row_workspace(
        &self,
        request: RowDecodeRequest<'_>,
    ) -> Result<Option<RowLookupWorkspace>, BackendFailure> {
        if self.row_storage().is_none() {
            return Ok(None);
        }
        let width = dimensions(request.geometry)?.0;
        let bytes = request.encoded_row_bytes;
        let compatible = match *request.geometry.encoding {
            RowEncoding::Dense => bytes == width as u64 * 2 || bytes == width as u64 * 4,
            RowEncoding::ScalarE4M3 { .. } => request.scalar.is_some() && bytes == width as u64,
            RowEncoding::Gguf { encoding, endian } => {
                eredu_gguf::BlockDecoder::new(encoding, endian)
                    .ok()
                    .and_then(|decoder| decoder.output_len(usize::try_from(bytes).ok()?).ok())
                    == Some(width)
            }
        };
        if !compatible {
            return Ok(None);
        }
        let scalar_bytes = if let Some(scalar) = request.scalar {
            let Ok(bytes) =
                eredu_core::checkpoint_mechanisms::recipe_execution::conservative_workspace_bytes(
                    scalar.recipe,
                    &scalar,
                )
            else {
                return Ok(None);
            };
            if crate::checkpoint::preflight_recipe_catalog(scalar.recipe, &scalar, &|key| {
                scalar
                    .sources
                    .get(key)
                    .map(|entry| entry.encoding.clone())
                    .ok_or_else(|| eredu_checkpoint::store::StoreError::UnknownTensor {
                        key: key.into(),
                    })
            })
            .is_err()
            {
                return Ok(None);
            }
            bytes
        } else {
            0
        };
        Ok(Some(RowLookupWorkspace {
            decode_bytes: workspace(request.rows, bytes, request.geometry)?,
            scalar_bytes,
        }))
    }
}

/// Native decoder for independently leased parameter rows.
pub struct CandleRowDecoder {
    scale: Option<CandleTensor>,
}

impl CandleRowDecoder {
    pub(crate) fn new(scale: Option<(RecipeDtype, CandleTensor)>) -> Result<Self, BackendFailure> {
        if let Some((expected, value)) = &scale {
            if !matches!(
                expected,
                RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
            ) || crate::checkpoint::native_dtype(expected).map_err(BackendFailure::from_error)?
                != value.value.dtype()
                || value.value.elem_count() != 1
            {
                return Err(crate::text::invalid(
                    "row decoder requires its exact prepared scalar",
                ));
            }
        }
        Ok(Self {
            scale: scale.map(|(_, value)| value),
        })
    }
}

impl RowDecode<CandleNeuralBackend, CandleWeightResidency> for CandleRowDecoder {
    type Error = BackendFailure;
    type Loan = CandleParameterLoan;

    fn begin_loan(
        transfer: CandleWeightTransfer,
        context: &CandleContext,
    ) -> Result<Self::Loan, BackendFailure> {
        CandleWeightResidency::begin_loan(transfer, context)
    }

    fn workspace(
        &self,
        rows: u64,
        bytes: u64,
        geometry: RowDecodeGeometry<'_>,
    ) -> Result<u64, BackendFailure> {
        workspace(rows, bytes, geometry)
    }

    fn decode(
        &mut self,
        loan: &mut Self::Loan,
        binding: &str,
        geometry: RowDecodeGeometry<'_>,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        loan.check_queue(context)?;
        let rows = loan.transfer().len();
        if let Some(scale) = &self.scale {
            loan.retain(scale);
        }
        let (width, dtype) = dimensions(geometry)?;
        let values = (0..rows)
            .map(|index| CandleWeightResidency::member(loan, index, binding))
            .collect::<Result<Vec<_>, _>>()?;
        let result = (|| -> Result<_, Error> {
            let encoded = NativeTensor::cat(
                &values.iter().map(|value| &value.value).collect::<Vec<_>>(),
                0,
            )
            .map_err(Error::backend_source)?;
            loan.retain(&CandleTensor::wrap(Ok(encoded.clone()))?);
            let decoded = match *geometry.encoding {
                RowEncoding::Dense => encoded,
                RowEncoding::ScalarE4M3 { .. } => crate::checkpoint::decode_scaled_e4m3(
                    &encoded,
                    &self
                        .scale
                        .as_ref()
                        .ok_or_else(|| Error::backend("missing row scale"))?
                        .value,
                )?,
                RowEncoding::Gguf { encoding, endian } => {
                    crate::quantized::decode_block_rows(&encoded, rows, width, encoding, endian)
                        .map_err(Error::backend_source)?
                }
            };
            if decoded.dims() != [rows, width] {
                return Err(Error::backend("decoded rows differ from admitted geometry"));
            }
            CandleTensor::wrap(
                decoded
                    .to_dtype(dtype)
                    .and_then(|value| value.force_contiguous()),
            )
        })();
        result.map_err(BackendFailure::from_error)
    }
}
