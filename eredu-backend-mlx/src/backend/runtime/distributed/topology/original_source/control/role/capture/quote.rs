//! The shared protocol gather's native source and prospective caller population.
use super::*;
use crate::backend::runtime::distributed::completion::prepared::{
    CompletionResourceLayout, PreparedCompletionResources, ReadyCompletionResources,
};
use crate::backend::runtime::distributed::topology::original_source::{
    OriginalCommunicationCompletedOperation, OriginalCommunicationOperation,
    OriginalCommunicatorPersistent,
};
use crate::backend::submission_recovery::native_role::{self, NativeRoleCapacity};

#[derive(Clone, Copy, Debug)]
pub(crate) struct GatherRequirements {
    pub(crate) capacity: AgreementCapacity,
    pub(crate) metadata: usize,
}
impl<O: CaptureSourceOwner> OriginalCaptureTransport<O> {
    /// The same actual world worker used by `gather`, before protocol words exist.
    /// No input constructor, source identity or native authority is produced.
    pub(crate) fn gather_requirements(
        source: &OriginalCommunicationSource<'_>,
        runtime: &safemlx::PreparedInputRuntime,
        words: usize,
        source_loan_controls: usize,
    ) -> Result<GatherRequirements, Error> {
        let unknown = |stage| {
            Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
                .at_speculative_stage(stage)
        };
        let group = source.world();
        let native = group.native_group();
        let shape = [i32::try_from(words).map_err(|_| overflow())?];
        let input_shape = [words];
        let frames = [
            size_of::<GatherRequirements>(),
            size_of::<Result<GatherRequirements, Error>>(),
            size_of::<safemlx::PreparedInputLayout>(),
            size_of::<safemlx::distributed::GroupCpuLayoutStorage<'_>>(),
            size_of::<safemlx::distributed::GroupCpuCompletionLayoutStorage<'_>>(),
            size_of::<safemlx::distributed::GroupPersistentStorage<'_>>(),
            size_of::<(
                &OriginalCommunicationSource<'_>,
                &safemlx::PreparedInputRuntime,
                usize,
                usize,
            )>(),
            size_of::<[i32; 1]>(),
            size_of::<[usize; 1]>(),
            safemlx::PreparedInputRuntime::u32_layout_control_bytes(),
            native
                .cpu_layout_storage_control_bytes()
                .ok_or_else(|| unknown("partition gather layout query controls"))?,
            native
                .persistent_storage_control_bytes()
                .ok_or_else(|| unknown("partition gather persistent query controls"))?,
        ];
        source.funding().reserve_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or_else(overflow)?,
        )?;
        source.validate()?;
        let input = runtime
            .u32_layout(&input_shape)
            .map_err(|_| unknown("partition gather copied U32 layout"))?;
        let operation = native
            .cpu_layout_storage(&shape, safemlx::Dtype::Uint32, GroupWorkerOperation::Gather)
            .map_err(|_| unknown("partition gather native CPU layout"))?;
        let expected = words
            .checked_mul(source.source().manifest().world_size())
            .ok_or_else(overflow)?;
        if words == 0 || operation.output_geometry() != (1, expected) {
            return Err(unknown("partition gather output geometry"));
        }
        source.funding().reserve_metadata(
            operation
                .backing_control_bytes()
                .and_then(|n| n.checked_add(operation.completion_layout_control_bytes()?))
                .ok_or_else(|| unknown("partition gather backing and completion controls"))?,
        )?;
        let backing = operation
            .backing_capacity(runtime)
            .map_err(|_| unknown("partition gather native backing"))?;
        let backing_controls = OriginalCommunicationOperation::backing_control_bytes(
            operation
                .input_backing_control_bytes()
                .ok_or_else(|| unknown("partition gather input backing controls"))?,
        )
        .ok_or_else(overflow)?;
        let completion_controls = OriginalCommunicationOperation::completion_control_bytes(
            operation
                .input_completion_control_bytes()
                .ok_or_else(|| unknown("partition gather input completion controls"))?,
        )
        .ok_or_else(overflow)?;
        let completed = operation
            .with_completion_layout()
            .map_err(|_| unknown("partition gather native completion layout"))?;
        let persistent = native
            .persistent_storage()
            .map_err(|_| unknown("partition gather persistent source"))?;
        if persistent.has_unqualified_storage() {
            return Err(unknown("partition gather persistent qualification"));
        }
        let validation =
            OriginalCommunicationSource::validation_control_bytes().ok_or_else(overflow)?;
        let persistent_controls = OriginalCommunicationSource::persistent_query_control_bytes(
            native
                .persistent_storage_control_bytes()
                .ok_or_else(|| unknown("partition gather persistent query population"))?,
        )
        .ok_or_else(overflow)?;
        let persistent_owner = OriginalCommunicatorPersistent::retained_owner_control_bytes(
            &persistent,
            group,
            source.registered_buffers.is_some(),
        )
        .map_err(|_| unknown("partition gather retained persistent owner"))?;
        let operation_controls = OriginalCommunicationSource::operation_storage_control_bytes(
            native
                .cpu_operation_storage_control_bytes()
                .ok_or_else(|| unknown("partition gather operation query controls"))?,
        )
        .ok_or_else(overflow)?;
        // One actual world operation lookup performs one persistent loan and one
        // operation query, then the same input-bound completion query.
        let operation_parts = [
            OriginalCommunicationSource::operation_lookup_control_bytes().ok_or_else(overflow)?,
            persistent_controls,
            validation,
            persistent_owner,
            operation_controls,
            validation,
            completion_controls,
        ];
        let operation_controls = operation_parts
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(overflow)?;
        let capacity_parts = [
            control_bytes::<O::Custody>().ok_or_else(overflow)?,
            capacity_control_bytes::<O>().ok_or_else(overflow)?,
            operation_controls,
            backing_controls,
        ];
        let capacity_controls = capacity_parts
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(overflow)?;
        let capacity = AgreementCapacity {
            graph: completed.graph_capacity(),
            records: completed.record_capacity(),
            backing,
        };
        let native_capacity = NativeRoleCapacity {
            graph: capacity.graph,
            records: capacity.records,
            backing: capacity.backing,
        };
        let traversal = completed.traversal();
        let layout = CompletionResourceLayout {
            arrays: 0,
            counts: &[],
            groups: 1,
            routes: 0,
            streams: 0,
        };
        let controls = control_bytes::<O::Custody>().ok_or_else(overflow)?;
        let parts = [
            // gather and its actual copied U32 leaf before the native child.
            controls,
            Self::gather_control_bytes().ok_or_else(overflow)?,
            source_loan_controls,
            super::super::super::super::inputs::source_layout_storage_bytes(&[input]).map_err(
                |cause| {
                    Error::PrefillControl(cause)
                        .at_speculative_stage("partition gather input source storage")
                },
            )?,
            validation,
            capacity_controls,
            controls,
            PartitionCaptureBuffer::<u32>::funded_control_bytes(expected).ok_or_else(overflow)?,
            // The child has a fixed zero-capture callback. Its graph/record
            // arenas are charged by this same metadata worker, once.
            native_role::control_bytes::<Invocation<O>, O::Custody>(native_capacity, None)
                .map_err(|_| unknown("partition gather child role controls"))?,
            native_role::callback_control_bytes::<PartitionCaptureBuffer<u32>, Error>(0)
                .ok_or_else(overflow)?,
            // Invocation::run rechecks the same source/capacity, then constructs
            // its separate accepted operation and exact completed word reader.
            controls,
            Invocation::<O>::invocation_control_bytes().ok_or_else(overflow)?,
            source_loan_controls,
            capacity_controls,
            operation_controls,
            PreparedCommunicationU32Words::original_operation_control_bytes(expected)
                .ok_or_else(overflow)?,
            validation,
            OriginalCommunicationCompletedOperation::prepare_resources_control_bytes()
                .ok_or_else(overflow)?,
            validation,
            PreparedCompletionResources::original_control_bytes(layout, &traversal)
                .ok_or_else(overflow)?,
            validation,
            PreparedCompletionResources::selection_control_bytes().ok_or_else(overflow)?,
            group
                .retention_copy_bytes()
                .ok_or_else(|| unknown("partition gather retained group controls"))?,
            validation,
            validation, // prepare, retain_control_world, finish
            OriginalCommunicationCompletedOperation::construction_control_bytes(
                completed
                    .execution_control_bytes()
                    .ok_or_else(|| unknown("partition gather completed execution controls"))?,
            )
            .ok_or_else(overflow)?,
            validation,
            OriginalCommunicationCompletedOperation::accepted_control_bytes()
                .ok_or_else(overflow)?,
            ReadyCompletionResources::original_one_root_submit_control_bytes()
                .ok_or_else(overflow)?,
        ];
        let metadata = parts
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(overflow)?;
        Ok(GatherRequirements { capacity, metadata })
    }
}
