//! Logical row-decoder geometry from exact retained recipes, without payload reads.
use super::*;
use eredu_checkpoint::recipe::RecipeDtype;
use eredu_nn::mechanism_memory::*;

impl RowLookupDescriptor {
    /// Logical input/result for one maximum-size acquisition. Encoded source rows
    /// and the shared scale remain borrowed values, not invocation allocations.
    /// Smaller calls are bounded by this geometry. Scalar preparation, portable
    /// lookup planning/reordering and shared residency are separate mechanisms.
    pub fn decode_memory(&self) -> Result<MechanismMemoryContract, RowLookupError> {
        let rows = self.maximum_acquisition_rows();
        let (width, element) = match self.spec().encoding {
            RowEncoding::Dense => (
                self.spec().dimensions as u64,
                floating(&self.metadata().dtype)?,
            ),
            _ => (self.range().bytes(), TensorElementType::U8),
        };
        let mut values = vec![LogicalValue {
            name: "encoded".into(),
            shape: vec![rows, width],
            element,
            kind: LogicalValueKind::Input,
        }];
        if let Some(scale) = self.scale() {
            let metadata = scale.metadata();
            values.push(LogicalValue {
                name: "scale".into(),
                shape: vec![1],
                element: floating(&metadata.dtype)?,
                kind: LogicalValueKind::Input,
            });
        }
        values.push(LogicalValue {
            name: "output".into(),
            shape: vec![rows, self.spec().dimensions as u64],
            element: self.spec().output_type,
            kind: LogicalValueKind::Output,
        });
        let contract = MechanismMemoryContract {
            values, storage: vec![],
            missing: vec!["row decoder backing, capacity, intermediate buffers and completion retention are undescribed".into()],
        };
        contract.validate()?;
        Ok(contract)
    }
}
fn floating(dtype: &RecipeDtype) -> Result<TensorElementType, RowLookupError> {
    match dtype {
        RecipeDtype::F32 => Ok(TensorElementType::F32),
        RecipeDtype::F16 => Ok(TensorElementType::F16),
        RecipeDtype::BF16 => Ok(TensorElementType::Bf16),
        _ => Err(RowLookupError::Geometry),
    }
}
