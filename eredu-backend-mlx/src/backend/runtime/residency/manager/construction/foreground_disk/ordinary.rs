//! Ordinary execution consumes the same retained reads as managed execution.
use super::*;
use safemlx::{HostTransferBuffer, HostTransferPolicy};

impl ForegroundDiskDescriptors {
    /// Read final host buffers through the retained, validated source descriptors.
    /// This ordinary allocation path grants no managed-request capacity or origin.
    pub(in crate::backend::runtime::residency::manager) fn read_ordinary_host(
        &self,
        id: &OffloadUnitId,
    ) -> Result<ResidentHostBuffers, ResidencyError> {
        let unit = self
            .value
            .units
            .iter()
            .find(|unit| unit.definition.id() == id)
            .ok_or(ResidencyError::StatePoisoned)?;
        let native_error = |source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "detached disk host destination",
            source,
        };
        let mut filling = Vec::with_capacity(unit.own_reads.len());
        for index in unit.own_reads.clone() {
            let native = &self.value.native_reads[index];
            filling.push(
                HostTransferBuffer::new(&native.shape, native.dtype, HostTransferPolicy::Transfer)
                    .map_err(native_error)?,
            );
        }
        let mut destinations = filling
            .iter_mut()
            .map(HostTransferBuffer::as_bytes_mut)
            .collect::<Result<Vec<_>, _>>()
            .map_err(native_error)?;
        self.value
            .source
            .read_slice(unit.own_reads.clone())
            .ok_or(ResidencyError::StatePoisoned)?
            .read_many_into(&mut destinations)
            .map_err(|source| {
                ResidencyError::DetachedDiskRead(eredu_core::BackendFailure::from_error(source))
            })?;
        drop(destinations);
        let ready: Vec<RetainedHostBuffer> = filling
            .into_iter()
            .map(|buffer| Arc::new(buffer.freeze()).into())
            .collect();
        let mut buffers = Vec::with_capacity(unit.definition.bindings().len());
        for (binding, index) in unit.definition.bindings().iter().zip(&unit.canonical_reads) {
            // External owners are joined by the existing atomic closure worker.
            if unit.own_reads.contains(index) {
                buffers.push((
                    binding.name().to_owned(),
                    ready[index - unit.own_reads.start].clone(),
                ));
            }
        }
        Ok(ResidentHostBuffers {
            buffers: rows::Rows::from_sorted(buffers),
        })
    }
}
