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
        let scalar = match dtype {
            RecipeDtype::F16 | RecipeDtype::BF16 => 2u64,
            RecipeDtype::F32 => 4,
            _ => {
                return Err(quantization_error(
                    "CPU MXFP4 requires F16, BF16 or F32 input",
                ))
            }
        };
        let columns = u64::try_from(columns)
            .map_err(|_| quantization_error("CPU MXFP4 column count overflow"))?;
        if columns == 0 || columns % 32 != 0 {
            return Err(quantization_error(
                "CPU MXFP4 columns must be a positive multiple of 32",
            ));
        }
        // The CPU fp_quantize fallback retains the input precision throughout
        // its floating arithmetic, including Select/Power after I32 casts.
        // Reserve all potential numerical destinations without relying on
        // donation or early retirement:
        // - one input compaction, Abs and normalized values: 3N floating;
        // - Subtract and Abs distances to 16 codebook entries: 32N floating;
        // - ArgReduce indices and shifted packed codes: 2N U32;
        // - max, scale division, log, round, cast back, Select, Power and Add:
        //   eight floating values per group, plus I32 and Boolean values.
        let row_bytes = columns
            .checked_mul(35 * scalar + 8)
            .and_then(|bytes| bytes.checked_add((columns / 32).checked_mul(8 * scalar + 5)?))
            .ok_or_else(|| quantization_error("CPU MXFP4 temporary payload overflow"))?;
        // Four floating scalars, two 32-bit scalars, a 16-entry F32 codebook,
        // eight U32 shifts plus their eight Arange inputs, and the codebook
        // conversion when the source precision is narrower than F32.
        let fixed_bytes = 136 + 4 * scalar + if scalar == 2 { 16 * scalar } else { 0 };
        Ok(QuantizerPayload {
            row_bytes,
            fixed_bytes,
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
