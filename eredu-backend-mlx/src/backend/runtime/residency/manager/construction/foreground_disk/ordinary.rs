//! Ordinary execution consumes the same retained reads as managed execution.
use super::*;
use crate::backend::runtime::checkpoint::store::MaterializationView;
use eredu_nn::workspace::HostMetadataFunding;
use safemlx::{HostTransferBuffer, HostTransferPolicy};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

impl ForegroundDiskDescriptors {
    /// Only new recipe/leaf/detachment allocations; the final immutable Host
    /// buffer and ordinary reader metadata have independent source queries.
    pub(in crate::backend::runtime::residency::manager) fn ordinary_materialized_quote(
        &self,
        id: &OffloadUnitId,
        pool: &MemoryLedger,
        runtime: &safemlx::PreparedInputRuntime,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        stream: &Stream,
        funding: &HostMetadataFunding,
    ) -> Result<Option<host_birth::materialized::quotation::Quotation>, ResidencyError> {
        let unit = self
            .value
            .units
            .iter()
            .find(|unit| unit.definition.id() == id)
            .ok_or(ResidencyError::StatePoisoned)?;
        let reports = eredu_runtime::working_memory::WorkspaceReportMetadata::with_funding(funding);
        let mut total: Option<host_birth::materialized::quotation::Quotation> = None;
        for index in unit.own_reads.clone() {
            let Some(plan) = &self.value.native_reads[index].materialized else {
                continue;
            };
            let quote = host_birth::materialized::quotation::quote(
                plan, pool, runtime, allocation, stream, funding,
            )
            .map_err(ResidencyError::OriginalCache)?;
            match &mut total {
                None => total = Some(quote),
                Some(total) => {
                    total.requirements = reports
                        .combine_domain_requirements(&total.requirements, &quote.requirements, true)
                        .map_err(|cause| {
                            ResidencyError::OriginalSourceRecipe(WeightRecipeError::Workspace(
                                reports.error(cause),
                            ))
                        })?;
                    total.metadata = total.metadata.checked_add(quote.metadata).ok_or(
                        ResidencyError::HostMetadataFunding(
                            eredu_nn::workspace::HostMetadataFundingError::Overflow,
                        ),
                    )?;
                }
            }
        }
        Ok(total)
    }

    pub(in crate::backend::runtime::residency::manager) fn ordinary_backing_capacity(
        &self,
        id: &OffloadUnitId,
        runtime: &safemlx::PreparedInputRuntime,
    ) -> Result<u64, ResidencyError> {
        let unit = self
            .value
            .units
            .iter()
            .find(|unit| unit.definition.id() == id)
            .ok_or(ResidencyError::StatePoisoned)?;
        unit.own_reads.clone().try_fold(0u64, |total, index| {
            let metadata = self
                .read_output(index)
                .ok_or(ResidencyError::StatePoisoned)?;
            let bytes = usize::try_from(metadata.byte_len()).map_err(|_| {
                ResidencyError::ArithmeticOverflow {
                    context: "ordinary foreground byte extent",
                }
            })?;
            let (capacity, placement) = HostTransferBuffer::ordinary_capacity(runtime, bytes)
                .map_err(|cause| match cause {
                    safemlx::PreparedInputCause::Invalid => ResidencyError::ArithmeticOverflow {
                        context: "ordinary host capacity",
                    },
                    _ => ResidencyError::OrdinaryHostControlSource,
                })?;
            if placement != runtime.allocation_placement() {
                return Err(ResidencyError::OrdinaryHostControlSource);
            }
            total
                .checked_add(u64::try_from(capacity).map_err(|_| {
                    ResidencyError::ArithmeticOverflow {
                        context: "ordinary host capacity conversion",
                    }
                })?)
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "ordinary host backing sum",
                })
        })
    }

    /// Native shared-control allocations reported by the existing ordinary
    /// physical observer, once for each owned read destination. These must not
    /// also be debited through the reader's host metadata payer.
    pub(in crate::backend::runtime::residency::manager) fn ordinary_observed_control_bytes(
        &self,
        id: &OffloadUnitId,
        runtime: &safemlx::PreparedInputRuntime,
    ) -> Result<usize, ResidencyError> {
        let unit = self
            .value
            .units
            .iter()
            .find(|unit| unit.definition.id() == id)
            .ok_or(ResidencyError::StatePoisoned)?;
        unit.own_reads.clone().try_fold(0usize, |total, index| {
            let rank = self.value.native_reads[index].shape.len();
            let controls = HostTransferBuffer::ordinary_observed_control_bytes(runtime, rank)
                .ok_or(ResidencyError::OrdinaryHostControlSource)?;
            total
                .checked_add(controls)
                .ok_or(ResidencyError::HostMetadataFunding(
                    eredu_nn::workspace::HostMetadataFundingError::Overflow,
                ))
        })
    }

    /// Read final host buffers through the retained, validated source descriptors.
    /// This ordinary allocation path grants no managed-request capacity or origin.
    pub(in crate::backend::runtime::residency::manager) fn read_ordinary_host(
        &self,
        id: &OffloadUnitId,
    ) -> Result<ResidentHostBuffers, ResidencyError> {
        self.read_ordinary_host_with_metadata(id, None, None)
    }

    /// Exact Rust destinations and the actual retained-reader scratch. Native
    /// host-buffer construction and physical backing are separate requirements.
    pub(in crate::backend::runtime::residency::manager) fn ordinary_read_control_bytes(
        &self,
        id: &OffloadUnitId,
    ) -> Result<usize, ResidencyError> {
        let unit = self
            .value
            .units
            .iter()
            .find(|unit| unit.definition.id() == id)
            .ok_or(ResidencyError::StatePoisoned)?;
        let overflow = || {
            ResidencyError::HostMetadataFunding(
                eredu_nn::workspace::HostMetadataFundingError::Overflow,
            )
        };
        let count = unit.own_reads.len();
        let native_wrappers = unit.own_reads.clone().try_fold(0usize, |total, index| {
            let rank = self.value.native_reads[index].shape.len();
            let bytes = HostTransferBuffer::ordinary_constructor_control_bytes(rank)
                .ok_or(ResidencyError::OrdinaryHostControlSource)?;
            total.checked_add(bytes).ok_or_else(overflow)
        })?;
        let read_bytes = if unit
            .own_reads
            .clone()
            .any(|index| self.value.native_reads[index].materialized.is_some())
        {
            // The prepared recipe callback reads each genuine leaf exactly
            // once. Direct siblings use the same single-leaf reader.
            unit.own_leaves.clone().try_fold(0usize, |total, index| {
                let read = self
                    .value
                    .source
                    .read_slice(index..index + 1)
                    .and_then(|read| read.read_layout())
                    .ok_or(ResidencyError::StatePoisoned)?;
                total
                    .checked_add(read.required_bytes())
                    .ok_or_else(overflow)
            })?
        } else {
            self.value
                .source
                .read_slice(unit.own_leaves.clone())
                .and_then(|read| read.read_layout())
                .ok_or(ResidencyError::StatePoisoned)?
                .required_bytes()
        };
        let name_bytes = unit
            .definition
            .bindings()
            .iter()
            .try_fold(0usize, |n, binding| n.checked_add(binding.name().len()))
            .ok_or_else(overflow)?;
        let shell = usize::try_from(
            RetainedHostBuffer::ordinary_storage_bytes()
                .map_err(ResidencyError::OriginalInventory)?,
        )
        .map_err(|_| overflow())?;
        let fields = [
            Layout::array::<HostTransferBuffer>(count)
                .map_err(|_| overflow())?
                .size(),
            Layout::array::<&mut [u8]>(count)
                .map_err(|_| overflow())?
                .size(),
            Layout::array::<RetainedHostBuffer>(count)
                .map_err(|_| overflow())?
                .size(),
            Layout::array::<(String, RetainedHostBuffer)>(unit.definition.bindings().len())
                .map_err(|_| overflow())?
                .size(),
            name_bytes,
            native_wrappers,
            id.as_str().len(),
            shell.checked_mul(count).ok_or_else(overflow)?,
            read_bytes,
            size_of::<(
                &Self,
                &OffloadUnitId,
                Option<&HostMetadataFunding>,
                Option<MaterializationView<'_>>,
            )>(),
            size_of::<Option<crate::backend::nn::shared::OrdinaryExecutionOwner>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Vec<HostTransferBuffer>>(),
            size_of::<Vec<&mut [u8]>>(),
            size_of::<Vec<RetainedHostBuffer>>(),
            size_of::<Vec<(String, RetainedHostBuffer)>>(),
            size_of::<ResidentHostBuffers>(),
            size_of::<Result<ResidentHostBuffers, ResidencyError>>(),
            size_of::<ResidentHostOwner>(),
            size_of::<String>(),
            size_of::<(&OffloadUnitId, &'static str, safemlx::error::Exception)>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<
                eredu_checkpoint::store::DetachedReadFailure<ManagerCustody>,
            >()
            .ok_or_else(overflow)?,
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
            .ok_or_else(overflow)
    }

    pub(in crate::backend::runtime::residency::manager) fn read_ordinary_host_with_metadata(
        &self,
        id: &OffloadUnitId,
        funding: Option<&HostMetadataFunding>,
        context: Option<MaterializationView<'_>>,
    ) -> Result<ResidentHostBuffers, ResidencyError> {
        if let Some(funding) = funding {
            funding
                .reserve_metadata(self.ordinary_read_control_bytes(id)?)
                .map_err(ResidencyError::HostMetadataFunding)?;
        }
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
        let execution = if unit
            .own_reads
            .clone()
            .any(|index| self.value.native_reads[index].materialized.is_some())
        {
            context.ok_or(ResidencyError::OrdinaryMaterializationSource)?;
            if funding.is_none() {
                return Err(ResidencyError::OrdinaryMaterializationSource);
            }
            Some(
                crate::backend::nn::shared::current_ordinary_execution_owner()
                    .map_err(native_error)?
                    .ok_or(ResidencyError::OrdinaryMaterializationSource)?,
            )
        } else {
            None
        };
        let mut filling = Vec::with_capacity(unit.own_reads.len());
        for index in unit.own_reads.clone() {
            let native = &self.value.native_reads[index];
            filling.push(
                HostTransferBuffer::new(&native.shape, native.dtype, HostTransferPolicy::Transfer)
                    .map_err(native_error)?,
            );
        }
        if let Some(execution) = execution {
            let context = context.expect("authenticated materialization context");
            for (index, destination) in unit.own_reads.clone().zip(&mut filling) {
                let native = &self.value.native_reads[index];
                if let Some(plan) = &native.materialized {
                    host_birth::materialized::execute(
                        plan,
                        &self.value.source,
                        native.leaves.clone(),
                        destination,
                        context,
                        execution.observer(),
                        execution.host().clone(),
                    )
                    .map_err(|cause| {
                        ResidencyError::MaterializedDiskRead(
                            eredu_core::BackendFailure::from_error(cause),
                        )
                    })?;
                } else {
                    self.value
                        .source
                        .read_slice(native.leaves.clone())
                        .ok_or(ResidencyError::StatePoisoned)?
                        .read_many_into(&mut [destination.as_bytes_mut().map_err(native_error)?])
                        .map_err(|cause| {
                            ResidencyError::DetachedDiskRead(
                                eredu_core::BackendFailure::from_error(cause),
                            )
                        })?;
                }
            }
        } else {
            let mut destinations = filling
                .iter_mut()
                .map(HostTransferBuffer::as_bytes_mut)
                .collect::<Result<Vec<_>, _>>()
                .map_err(native_error)?;
            self.value
                .source
                .read_slice(unit.own_leaves.clone())
                .ok_or(ResidencyError::StatePoisoned)?
                .read_many_into(&mut destinations)
                .map_err(|cause| {
                    ResidencyError::DetachedDiskRead(eredu_core::BackendFailure::from_error(cause))
                })?;
        }
        let ready: Vec<RetainedHostBuffer> = filling
            .into_iter()
            .map(|buffer| match funding {
                Some(funding) => {
                    RetainedHostBuffer::ordinary_with_metadata(buffer.freeze(), funding.clone())
                }
                None => Arc::new(buffer.freeze()).into(),
            })
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
