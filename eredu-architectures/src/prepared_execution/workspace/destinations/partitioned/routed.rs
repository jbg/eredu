//! Routed destinations consume the same retained selected bank ownership.
use super::*;
use crate::partitioned_execution::{
    PreparedRoutedPartitionedArchitecture, RoutedPartitionedProductionVisitor,
};

impl RoutedPartitionedProductionVisitor<WorkspaceBackend, ResidentState>
    for PartitionDestinationVisitor<'_>
{
    type Output = PartitionedTextBindingDestinations;
    type Error = Error;
    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<WorkspaceBackend, A, G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend, ResidentState>>::Boundary>,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: TextPartitionArchitecture<WorkspaceBackend, ResidentState>
            + ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        prepared.dispatch_execution(
            self,
            |prepared, visitor| visitor.collect_routed(prepared, PartitionedUnitScope::All),
            |prepared, visitor| visitor.collect_routed(prepared, PartitionedUnitScope::Owned),
        )
    }
}
impl PartitionDestinationVisitor<'_> {
    fn collect_routed<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<WorkspaceBackend, A, G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend, ResidentState>>::Boundary>,
        scope: PartitionedUnitScope,
    ) -> Result<PartitionedTextBindingDestinations, Error>
    where
        A: TextPartitionArchitecture<WorkspaceBackend, ResidentState>
            + ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        // Whole selected catalogs include idle expert owners and companions;
        // local member presence is not the exclusion authority.
        let excluded = prepared.addressable_logical_targets();
        let (prepared, source, physical_layout, tasks) = prepared.into_parts();
        if source.is_some() {
            return Err(Error::backend(
                "partition destination source requires a numerical transform",
            ));
        }
        let (architecture, bound) = prepared.into_parts();
        let (selected, partition, _communication) = bound.into_parts();
        self.collect_architecture(
            architecture,
            selected.text().clone(),
            partition,
            physical_layout,
            tasks,
            scope,
            excluded,
        )
    }
}
