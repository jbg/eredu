//! Exact I32 status leaf and completion worker census from its retained Group.
use super::super::operation_storage::OriginalCommunicationOperation;
use super::*;
use crate::backend::runtime::distributed::completion::prepared::{
    CompletionResourceLayout, PreparedCompletionResources,
};
/// Native requirements and the actual individual metadata-debit branches.
/// These values contain no execution permission or producing input.
pub(super) struct LeafQuote {
    pub(super) capacity: AgreementCapacity,
    pub(super) operation: usize,
    pub(super) backing: usize,
    pub(super) resources: usize,
    pub(super) submit: usize,
}
impl LeafQuote {
    pub(super) fn prepare(
        source: &OriginalCommunicationSource<'_>,
        runtime: &PreparedInputRuntime,
        order: usize,
        operation: GroupWorkerOperation,
        status: bool,
    ) -> Result<Self, Error> {
        let unknown = || failure(Cause::Resource, source.source(), source.funding());
        let group = source.group(order).ok_or_else(unknown)?.0;
        let native = group.native_group();
        reserve(
            source.funding(),
            &[
                size_of::<Self>(),
                size_of::<Result<Self, Error>>(),
                size_of::<(
                    &OriginalCommunicationSource<'_>,
                    &PreparedInputRuntime,
                    usize,
                    GroupWorkerOperation,
                    bool,
                )>(),
                size_of::<[i32; 1]>(),
                native
                    .cpu_layout_storage_control_bytes()
                    .ok_or_else(unknown)?,
                native
                    .persistent_storage_control_bytes()
                    .ok_or_else(unknown)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        source.validate()?;
        let shape = [1_i32];
        let layout = native
            .cpu_layout_storage(&shape, safemlx::Dtype::Int32, operation)
            .map_err(|_| unknown())?;
        if layout.output_geometry() != (1, 1) {
            return Err(unknown());
        }
        reserve(
            source.funding(),
            &[
                layout.backing_control_bytes().ok_or_else(unknown)?,
                layout
                    .completion_layout_control_bytes()
                    .ok_or_else(unknown)?,
            ],
        )?;
        let backing = layout.backing_capacity(runtime).map_err(|_| unknown())?;
        let backing_controls = OriginalCommunicationOperation::backing_control_bytes(
            layout.input_backing_control_bytes().ok_or_else(unknown)?,
        )
        .ok_or_else(overflow)?;
        let completion_controls = OriginalCommunicationOperation::completion_control_bytes(
            layout
                .input_completion_control_bytes()
                .ok_or_else(unknown)?,
        )
        .ok_or_else(overflow)?;
        let complete = layout.with_completion_layout().map_err(|_| unknown())?;
        let persistent = native.persistent_storage().map_err(|_| unknown())?;
        if persistent.has_unqualified_storage() {
            return Err(unknown());
        }
        let validation =
            OriginalCommunicationSource::validation_control_bytes().ok_or_else(overflow)?;
        let persistent_controls = OriginalCommunicationSource::persistent_query_control_bytes(
            native
                .persistent_storage_control_bytes()
                .ok_or_else(unknown)?,
        )
        .ok_or_else(overflow)?;
        let owner = OriginalCommunicatorPersistent::retained_owner_control_bytes(
            &persistent,
            group,
            source.registered_buffers.is_some(),
        )
        .map_err(|_| unknown())?;
        let native_query = native
            .cpu_operation_storage_control_bytes()
            .ok_or_else(unknown)?;
        let selected = if status {
            OriginalCommunicationSource::status_operation_control_bytes()
                .ok_or_else(overflow)?
                .checked_add(native_query)
                .ok_or_else(overflow)?
        } else {
            OriginalCommunicationSource::operation_storage_control_bytes(native_query)
                .ok_or_else(overflow)?
        };
        let operation_parts = [
            OriginalCommunicationSource::operation_lookup_control_bytes().ok_or_else(overflow)?,
            selected,
            validation,
            persistent_controls,
            validation,
            owner,
            completion_controls,
        ];
        let operation = operation_parts
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(overflow)?;
        let traversal = complete.traversal();
        let layout = CompletionResourceLayout {
            arrays: 0,
            counts: &[],
            groups: 1,
            routes: 0,
            streams: 0,
        };
        let resources_parts = [
            OriginalCommunicationCompletedOperation::prepare_resources_control_bytes()
                .ok_or_else(overflow)?,
            validation,
            PreparedCompletionResources::original_control_bytes(layout, &traversal)
                .ok_or_else(overflow)?,
            validation,
            PreparedCompletionResources::selection_control_bytes().ok_or_else(overflow)?,
            group.retention_copy_bytes().ok_or_else(unknown)?,
            validation,
            validation,
        ];
        let resources = resources_parts
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(overflow)?;
        let submit_parts = [
            OriginalCommunicationCompletedOperation::construction_control_bytes(
                complete.execution_control_bytes().ok_or_else(unknown)?,
            )
            .ok_or_else(overflow)?,
            validation,
            OriginalCommunicationCompletedOperation::accepted_control_bytes()
                .ok_or_else(overflow)?,
            ReadyCompletionResources::original_one_root_submit_control_bytes()
                .ok_or_else(overflow)?,
        ];
        let submit = submit_parts
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(overflow)?;
        Ok(Self {
            capacity: AgreementCapacity {
                graph: complete.graph_capacity(),
                records: complete.record_capacity(),
                backing,
            },
            operation,
            backing: backing_controls,
            resources,
            submit,
        })
    }
}

/// Completion destination construction shared by each status arithmetic stage.
pub(in super::super) fn arithmetic_resources(
    group: &Group,
    traversal: &safemlx::OperationEvalTraversalLayout,
) -> Option<usize> {
    let validation = OriginalCommunicationSource::validation_control_bytes()?;
    let parts = [
        validation,
        PreparedCompletionResources::original_control_bytes(
            CompletionResourceLayout {
                arrays: 0,
                counts: &[],
                groups: 1,
                routes: 0,
                streams: 0,
            },
            traversal,
        )?,
        validation,
        PreparedCompletionResources::selection_control_bytes()?,
        group.retention_copy_bytes()?,
        validation,
        validation,
    ];
    // These are separate existing debits, not one new aggregate reserve.
    parts.into_iter().try_fold(0usize, usize::checked_add)
}
