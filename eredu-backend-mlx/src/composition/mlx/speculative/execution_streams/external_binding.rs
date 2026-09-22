//! Bind retained placement using paid scalar snapshots of its actual streams.
use super::*;
use eredu_nn::workspace::HostMetadataFundingError;
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::{size_of, size_of_val};

impl<'a> SpeculativeExecutionStreams<'a> {
    pub(crate) fn bind_original_external(
        source: &'a dyn ExternalInvocationSource,
        target: &'a OriginalCopyEnvironment<'a>,
        draft: &'a OriginalCopyEnvironment<'a>,
        topology: SpeculativeExecutionTopology,
    ) -> Result<Self, Error> {
        let numerical = source.numerical_sources();
        let funding = numerical.metadata_funding();
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<(
                &dyn ExternalInvocationSource,
                &OriginalCopyEnvironment<'_>,
                &OriginalCopyEnvironment<'_>,
                SpeculativeExecutionTopology,
            )>(),
            size_of::<[safemlx::StreamCopyPlan<()>; 2]>(),
            size_of::<Result<safemlx::StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
            size_of::<(bool, bool)>(),
            size_of::<(
                safemlx::DeviceType,
                safemlx::DeviceType,
                safemlx::CpuMatmulKernel,
                safemlx::CpuMatmulKernel,
            )>(),
            size_of::<[usize; 2]>(),
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        numerical.validate_environment(target)?;
        numerical.validate_environment(draft)?;
        let target_value = safemlx::StreamCopyPlan::<()>::capture(target.stream())
            .map_err(|cause| numerical.retain_startup_error(cause))?;
        let draft_value = safemlx::StreamCopyPlan::<()>::capture(draft.stream())
            .map_err(|cause| numerical.retain_startup_error(cause))?;
        funding
            .reserve_metadata(
                target_value
                    .source_comparison_control_bytes()
                    .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let same_stream = target_value.matches_source(draft.stream());
        let same_device = target_value.device_type() == draft_value.device_type()
            && target_value.device_index() == draft_value.device_index();
        let valid = match topology {
            SpeculativeExecutionTopology::Single => same_stream,
            SpeculativeExecutionTopology::SameDeviceSplit => !same_stream && same_device,
            // Both sides keep their retained creator/source ownership. The CPU
            // model worker is selected at factory construction, never here.
            // Exact model/numerical/copy recipes remain mandatory per operation.
            SpeculativeExecutionTopology::CrossDeviceSplit => {
                !same_device
                    && match (target_value.device_type(), draft_value.device_type()) {
                        (safemlx::DeviceType::Gpu, safemlx::DeviceType::Cpu) => {
                            draft_value.cpu_matmul() == safemlx::CpuMatmulKernel::Float32Tiles
                        }
                        (safemlx::DeviceType::Cpu, safemlx::DeviceType::Gpu) => {
                            target_value.cpu_matmul() == safemlx::CpuMatmulKernel::Float32Tiles
                        }
                        _ => false,
                    }
            }
            _ => false,
        };
        if !valid || !target.pool().same_ledger(draft.pool()) {
            return Err(numerical.retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        let mut context = Self::single(target.stream());
        context.draft = draft.stream();
        context.topology = topology;
        context.with_original_external_environments(source, target, draft)
    }
}
