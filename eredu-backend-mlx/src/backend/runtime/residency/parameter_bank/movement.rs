//! Device-side index discovery, gather, and scatter movement.

use super::*;

/// MLX integer discovery and device-side indexed movement mechanism.
#[derive(Debug, Default, Clone, Copy)]
pub struct MlxIndexedMovement;

impl IndexedMovement<MlxNeuralBackend> for MlxIndexedMovement {
    type Error = Error;

    fn index_demands(
        &mut self,
        indices: &MlxTensor,
        upper_bound: usize,
        stream: &Stream,
    ) -> Result<Vec<(usize, u64)>, Self::Error> {
        if !matches!(
            indices.as_array().dtype(),
            Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
        ) {
            return Err(AddressableParameterBankError::InvalidSelectionDtype {
                actual: indices.as_array().dtype(),
            }
            .into());
        }
        let upper = i32::try_from(upper_bound).map_err(|_| {
            Error::ArchitectureModel("indexed movement upper bound exceeds MLX i32 indexing".into())
        })?;
        if upper == 0 {
            return Err(Error::ArchitectureModel(
                "indexed movement upper bound must be nonzero".into(),
            ));
        }
        let flat = indices.as_array().reshape(&[-1], stream)?;
        let below = flat.lt(Array::from_int(upper), stream)?;
        let valid = if matches!(flat.dtype(), Dtype::Uint32 | Dtype::Uint64) {
            below
        } else {
            flat.ge(Array::from_int(0), stream)?
                .logical_and(below, stream)?
        };
        let invalid =
            crate::backend::compaction::count_nonzero(&valid.logical_not(stream)?, stream)?;
        let flat_i32 = if flat.dtype() == Dtype::Int32 {
            flat
        } else {
            flat.as_dtype(Dtype::Int32, stream)?
        };
        let safe = r#where(
            &valid,
            flat_i32,
            Array::zeros::<i32>(&[indices.as_array().size() as i32], stream)?,
            stream,
        )?;
        let ones = Array::ones::<i32>(&[safe.size() as i32], stream)?;
        let histogram = segment_sum(&ones, &safe, upper, 0, stream)?;
        eval([&histogram, &invalid])?;
        let invalid_count = invalid.evaluated()?.as_slice::<i32>()[0];
        if invalid_count != 0 {
            return Err(AddressableParameterBankError::InvalidSelectionSet {
                namespace: 0,
                invalid_count: invalid_count as usize,
                global_span: upper_bound,
            }
            .into());
        }
        Ok(histogram
            .evaluated()?
            .as_slice::<i32>()
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, count)| *count != 0)
            .map(|(index, count)| (index, count as u64))
            .collect())
    }

    fn remap_indices(
        &mut self,
        indices: &MlxTensor,
        mapping: &[(usize, usize)],
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        let span = mapping
            .iter()
            .map(|(source, _)| source.saturating_add(1))
            .max()
            .ok_or_else(|| Error::ArchitectureModel("indexed remapping is empty".into()))?;
        let mut lookup = vec![-1i32; span];
        for &(source, destination) in mapping {
            let destination = i32::try_from(destination).map_err(|_| {
                Error::ArchitectureModel("compact index exceeds MLX i32 indexing".into())
            })?;
            if source >= span || lookup[source] != -1 {
                return Err(Error::ArchitectureModel(
                    "indexed remapping contains a duplicate source".into(),
                ));
            }
            lookup[source] = destination;
        }
        let lookup = Array::from_slice(&lookup, &[span as i32]).ensure_device_storage(stream)?;
        let normalized = if indices.as_array().dtype() == Dtype::Int32 {
            indices.as_array().clone()
        } else {
            indices.as_array().as_dtype(Dtype::Int32, stream)?
        };
        Ok(MlxTensor::from_array(lookup.take(&normalized, stream)?))
    }

    fn select_rows(
        &mut self,
        value: &MlxTensor,
        start: usize,
        end: usize,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        let start = i32::try_from(start)
            .map_err(|_| Error::ArchitectureModel("row start exceeds MLX indexing".into()))?;
        let end = i32::try_from(end)
            .map_err(|_| Error::ArchitectureModel("row end exceeds MLX indexing".into()))?;
        Ok(MlxTensor::from_array(
            value.as_array().try_index_device(start..end, stream)?,
        ))
    }

    fn concatenate_rows(
        &mut self,
        values: &[MlxTensor],
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        if values.is_empty() {
            return Err(Error::ArchitectureModel(
                "row concatenation requires at least one partition".into(),
            ));
        }
        let values = values.iter().map(MlxTensor::as_array).collect::<Vec<_>>();
        Ok(MlxTensor::from_array(concatenate_axis(&values, 0, stream)?))
    }
}

/// Concatenates the exact named tensors protected by the selected transfer.
pub(super) fn compact_binding(
    transfer: &ResidentTransfer,
    name: &str,
    stream: &Stream,
) -> Result<Array, AddressableParameterBankError> {
    let values = transfer
        .leases()
        .iter()
        .map(|lease| lease.device_value(name).cloned())
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(AddressableParameterBankError::EmptyCompactBinding {
            name: name.to_owned(),
        });
    }
    Ok(concatenate_axis(&values, 0, stream)?)
}

struct ParameterLoanResources {
    transfer: ResidentTransfer,
    output: Option<MlxTensor>,
}

impl crate::backend::submission_recovery::Retention for ParameterLoanResources {
    fn observe(&self, _: crate::backend::submission_recovery::Status) {}
}

/// Native recovery and completion for an exact borrowed parameter transfer.
#[doc(hidden)]
pub struct MlxParameterLoan {
    recovery: crate::backend::submission_recovery::Recovery<ParameterLoanResources>,
}

impl MlxParameterLoan {
    pub(super) fn begin(transfer: ResidentTransfer) -> Result<Self, Error> {
        let recovery =
            crate::backend::submission_recovery::Recovery::begin(ParameterLoanResources {
                transfer,
                output: None,
            })?;
        Ok(Self { recovery })
    }

    pub(super) fn transfer(&self) -> &ResidentTransfer {
        &self.recovery.retention().transfer
    }
}

impl eredu_core::residency::ParameterLoan<MlxTensor, Stream> for MlxParameterLoan {
    type Error = Error;

    fn copy_binding(&mut self, binding: &str, stream: &Stream) -> Result<MlxTensor, Error> {
        let lease = self.transfer().leases().first().ok_or_else(|| {
            Error::ArchitectureModel("parameter loan has no acquired storage".into())
        })?;
        let value =
            MlxTensor::from_array(lease.device_value(binding)?.ensure_device_storage(stream)?);
        self.recovery.retention_mut().output = Some(value.clone());
        Ok(value)
    }

    fn finish(mut self, output: &MlxTensor) -> Result<(), Error> {
        self.recovery.retention_mut().output = Some(output.clone());
        let submitted = eval([output.as_array()]);
        self.recovery.seal();
        submitted?;
        self.recovery.retention_mut().transfer.synchronize()?;
        let status = self.recovery.finish();
        if status.failed || status.blocked {
            return Err(Error::ArchitectureModel(
                "parameter-bank consumer failed; unresolved leases remain retained".into(),
            ));
        }
        Ok(())
    }
}
