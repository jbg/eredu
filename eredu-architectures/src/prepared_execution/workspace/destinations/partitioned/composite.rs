//! Composite partitions reuse their exact typed adapter and destination worker.
use super::*;
use crate::composite_execution::{CompositeArchitecture, PreparedCompositeArchitecture};
use crate::composite_partitioned::{
    AuthoritativeCompositePartitionVisitor, PreparedCompositePartition,
};

impl AuthoritativeCompositePartitionVisitor<WorkspaceBackend, ResidentState>
    for PartitionDestinationVisitor<'_>
{
    type Output = PartitionedTextBindingDestinations;
    type Error = Error;
    fn visit<A, G, W>(
        self,
        prepared: PreparedCompositePartition<A, G, W>,
    ) -> Result<Self::Output, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::PartitionedLayeredArchitecture<
                WorkspaceBackend,
                ResidentState,
                Boundary = W,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
    {
        let excluded = prepared
            .partition_banks()
            .map(PreparedPartitionBanks::addressable_logical_targets)
            .unwrap_or_default();
        let (prepared, source, physical_layout, tasks) = prepared.into_parts();
        if source.is_some() {
            return Err(Error::backend(
                "partition destination source requires a numerical transform",
            ));
        }
        let (architecture, bound) = prepared.into_parts();
        let (selected, partition, _communication) = bound.into_parts();
        self.collect_architecture(
            PreparedCompositeArchitecture::new(architecture),
            selected.execution().clone(),
            partition,
            physical_layout,
            tasks,
            PartitionedUnitScope::Owned,
            excluded,
        )
    }
}
