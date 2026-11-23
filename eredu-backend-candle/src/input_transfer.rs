//! Exact wire tensor copies across local Candle devices.

use candle_core::Device;
use eredu_core::{
    input::{InputTensorTransfer, PreparedInputInspector},
    BackendFailure, CapabilityError, InputTensorIdentity, PreparedInputError,
};

use crate::{text::invalid, text_native::CandleInputInspector, CandleContext, CandleTensor};

/// Bounded copy mechanism for one prepared prompt on its source context.
pub struct CandlePromptTransfer {
    source: CandleContext,
    existing_shared_capacity: u64,
}

impl CandlePromptTransfer {
    pub(crate) fn new(source: &CandleContext) -> Self {
        // Candle's shared Metal pool can reuse a larger buffer. Existing total
        // native allocation is an upper bound for any one recycled buffer.
        let existing_shared_capacity = crate::memory::context_facts(source)
            .device_allocated_bytes
            .value()
            .copied()
            .unwrap_or(0);
        Self {
            source: source.clone(),
            existing_shared_capacity,
        }
    }
}

impl PreparedInputInspector<CandleTensor> for CandlePromptTransfer {
    fn identity(&self, value: &CandleTensor) -> Result<InputTensorIdentity, PreparedInputError> {
        CandleInputInspector.identity(value)
    }

    fn i32_values(&self, value: &CandleTensor) -> Result<Vec<i32>, CapabilityError> {
        CandleInputInspector.i32_values(value)
    }

    fn bool_values(&self, value: &CandleTensor) -> Result<Vec<bool>, CapabilityError> {
        CandleInputInspector.bool_values(value)
    }
}

impl InputTensorTransfer for CandlePromptTransfer {
    type Tensor = CandleTensor;
    type Context = CandleContext;
    type Error = BackendFailure;

    fn host_staging_capacity(&self, payload_bytes: u64) -> Result<u64, BackendFailure> {
        // A newly allocated Metal download buffer rounds up to a power of two.
        // An exact-length host tensor can coexist with it during readback.
        payload_bytes
            .max(1)
            .checked_next_power_of_two()
            .map(|capacity| capacity.max(self.existing_shared_capacity))
            .and_then(|capacity| capacity.checked_add(payload_bytes))
            .ok_or_else(|| invalid("prepared input staging extent overflow"))
    }

    fn copy_input_tensors(
        &self,
        values: Vec<&CandleTensor>,
        context: &CandleContext,
        staging_capacity_bytes: u64,
    ) -> Result<Vec<CandleTensor>, BackendFailure> {
        self.source.check_healthy()?;
        context.check_healthy()?;
        for value in &values {
            self.source.validate_tensor_device(value)?;
            let bytes = (value.value.elem_count() as u64)
                .checked_mul(value.value.dtype().size_in_bytes() as u64)
                .ok_or_else(|| invalid("prepared input tensor byte extent overflow"))?;
            if value.packed.is_some()
                || !self.source.placement().supports_local_text_extensions()
                || !context.placement().supports_local_text_extensions()
                || self.host_staging_capacity(bytes)? > staging_capacity_bytes
            {
                return Err(invalid(
                    "prepared input tensor exceeds its admitted copy mechanism",
                ));
            }
        }
        values
            .into_iter()
            .map(|value| copy_tensor(value, context))
            .collect()
    }
}

pub(crate) fn copy_tensor(
    value: &CandleTensor,
    context: &CandleContext,
) -> Result<CandleTensor, BackendFailure> {
    context.check_healthy()?;
    let compacted = value.independent_copy().map_err(BackendFailure::from_error);
    CandleContext::settle_roots(std::iter::once(value).chain(compacted.as_ref().ok()))?;
    let compacted = compacted?;
    if compacted.value.device().same_device(&context.device) {
        return Ok(compacted);
    }
    // The compact temporary guarantees that a view copies only its own extent.
    let staged = compacted
        .value
        .to_device(&Device::Cpu)
        .map_err(BackendFailure::from_error);
    CandleContext::settle_roots([&compacted])?;
    let staged = staged?;
    let mut settlement = crate::device::CandleSettlement::new(context, (compacted, staged, None));
    let (_, staged, _) = settlement.resources();
    let mut copied = CandleTensor::wrap(
        staged
            .to_device(&context.device)
            .map(|value| value.detach()),
    )
    .map_err(BackendFailure::from_error)?;
    copied.boolean = value.boolean;
    settlement.resources_mut().2 = Some(copied);
    let (_, _, copied) = settlement.finish()?;
    Ok(copied.expect("settled prepared input copy"))
}
