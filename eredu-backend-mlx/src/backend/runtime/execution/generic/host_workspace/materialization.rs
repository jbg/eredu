//! Source requirements for the same recorded completed layerwise windows.
use super::*;
use eredu_core::DomainMemoryRequirements;
use eredu_runtime::{residency::ResidencyClosureSlot, working_memory::WorkspaceReportMetadata};

pub(crate) struct OrdinaryWindowMaterialization {
    pub(crate) requirements: Option<DomainMemoryRequirements>,
    pub(crate) metadata: u64,
}

impl LayerwiseWorkspace {
    pub(crate) fn ordinary_materialization_requirements(
        &self,
        ordinals: &[usize],
        pool: &eredu_runtime::working_memory::MemoryLedger,
        runtime: &safemlx::PreparedInputRuntime,
        allocation: NativeAllocationFacts,
        context: &WorkspaceContext,
    ) -> Result<OrdinaryWindowMaterialization, Error> {
        let mut result = OrdinaryWindowMaterialization {
            requirements: None,
            metadata: 0,
        };
        context
            .charge_metadata(std::mem::size_of::<(
                OrdinaryWindowMaterialization,
                Result<OrdinaryWindowMaterialization, Error>,
                Vec<ResidencyClosureSlot>,
                Vec<OffloadUnitId>,
                &Self,
                &[usize],
                &eredu_runtime::working_memory::MemoryLedger,
                &safemlx::PreparedInputRuntime,
                NativeAllocationFacts,
                &WorkspaceContext,
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        let funding = context.metadata_funding().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))?;
        let Some(units) = self
            .manager
            .ordinary_materialization_scratch_len(&funding)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
        else {
            return Ok(result);
        };
        let mut scratch = context
            .metadata_vec(units)
            .map_err(|cause| Error::Neural(cause.into()))?;
        scratch.resize(units, ResidencyClosureSlot::default());
        let mut roots = context
            .metadata_vec(self.identity.depth())
            .map_err(|cause| Error::Neural(cause.into()))?;
        let depth =
            std::num::NonZeroUsize::new(self.identity.depth()).ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))?;
        let reports = WorkspaceReportMetadata::new(context);
        for &ordinal in ordinals {
            let range = self
                .layout()
                .window_range(ordinal, depth)
                .ok_or(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ))?;
            roots.clear();
            for index in range {
                let unit = self
                    .requested_unit(index)
                    .and_then(|index| self.unit(index))
                    .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
                roots.push(
                    OffloadUnitId::new(
                        context
                            .metadata_string(format_args!("{}", unit.id.as_str()))
                            .map_err(|cause| Error::Neural(cause.into()))?,
                    )
                    .map_err(|cause| Error::Neural(context.metadata_source(cause)))?,
                );
            }
            let Some(quote) = self
                .manager
                .ordinary_foreground_materialization_requirements(
                    &roots,
                    &mut scratch,
                    pool,
                    runtime,
                    allocation,
                    &funding,
                )
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
            else {
                continue;
            };
            if let Some(requirements) = quote.native {
                result.requirements = Some(match result.requirements.take() {
                    None => requirements,
                    Some(previous) => reports
                        .combine_domain_requirements(&previous, &requirements, true)
                        .map_err(|cause| Error::Neural(reports.error(cause)))?,
                });
            }
            result.metadata = u64::try_from(quote.metadata)
                .ok()
                .and_then(|bytes| result.metadata.checked_add(bytes))
                .ok_or(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                ))?;
        }
        Ok(result)
    }
}
