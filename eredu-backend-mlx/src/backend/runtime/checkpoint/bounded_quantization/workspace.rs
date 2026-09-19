//! Logical temporary payloads of the selected native quantizer.
//!
//! Allocator rounding, graph/worker controls and source preparation are separate
//! contributions. These payload facts grant no native construction authority.
use super::{preflight::quantization_error, *};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum QuantizerWorkspace {
    Direct,
    CpuMxFp4,
}

impl QuantizerWorkspace {
    pub(super) fn selected(quantization: WeightQuantization, device: safemlx::DeviceType) -> Self {
        if quantization == WeightQuantization::MxFp4 && device == safemlx::DeviceType::Cpu {
            Self::CpuMxFp4
        } else {
            Self::Direct
        }
    }

    pub(super) fn payload(
        self,
        dtype: &RecipeDtype,
        columns: usize,
    ) -> Result<QuantizerPayload, Error> {
        if self == Self::Direct {
            return Ok(QuantizerPayload {
                row_bytes: 0,
                fixed_bytes: 0,
            });
        }
        let dtype = match dtype {
            RecipeDtype::F16 => Dtype::Float16,
            RecipeDtype::BF16 => Dtype::Bfloat16,
            RecipeDtype::F32 => Dtype::Float32,
            _ => return Err(quantization_error("CPU MXFP4 requires F16, BF16 or F32 input")),
        };
        let layout = safemlx::OperationEvent::cpu_mxfp4_quantize_payload_layout(dtype, 1, columns)
            .ok_or_else(|| quantization_error("CPU MXFP4 payload geometry is not representable"))?;
        Ok(QuantizerPayload {
            row_bytes: u64::try_from(layout.temporary_row_bytes())
                .map_err(|_| quantization_error("CPU MXFP4 temporary row payload overflow"))?,
            fixed_bytes: u64::try_from(layout.temporary_fixed_bytes())
                .map_err(|_| quantization_error("CPU MXFP4 fixed payload overflow"))?,
        })
    }

    pub(super) fn maximum_submission_elements(self) -> usize {
        match self {
            Self::Direct => MAX_QUANTIZATION_SUBMISSION_ELEMENTS,
            Self::CpuMxFp4 => MAX_QUANTIZATION_SUBMISSION_ELEMENTS / 16,
        }
    }
}

pub(super) struct QuantizerPayload {
    pub(super) row_bytes: u64,
    pub(super) fixed_bytes: u64,
}

/// CPU fallback contribution for one row when choosing a device-neutral load
/// allowance. Execution still selects its actual stream's payload profile.
pub(crate) fn cpu_quantization_temporary_row_bytes(
    quantization: WeightQuantization,
    dtype: &RecipeDtype,
    columns: usize,
) -> Result<u64, Error> {
    let payload = QuantizerWorkspace::selected(quantization, safemlx::DeviceType::Cpu)
        .payload(dtype, columns)?;
    payload
        .row_bytes
        .checked_add(payload.fixed_bytes)
        .ok_or_else(|| quantization_error("CPU quantization row payload overflow"))
}
