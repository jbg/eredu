//! Raw output backing rows enter the same identity and lifetime graph as leaves.
use super::*;
impl WorkspaceContext {
    pub(in crate::workspace) fn output_population_storage(
        &self,
        source: WorkspaceAllocationPopulation,
        expected: u64,
        mut aliases: Vec<Rc<Storage>>,
        trace: &mut Trace,
    ) -> Result<Rc<Storage>, Error> {
        // Resolved alternative maxima are requirement groups, not allocation
        // identities. Only actual simultaneous raw rows may become output roots.
        if source.domain_population().is_some() || source.backing_bytes() != Some(expected) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        source.validate_domains(self.memory_topology().ok_or_else(|| {
            Error::backend_retained_source(WorkspacePlacementError::MissingTopology)
        })?)?;
        self.charge_metadata(std::mem::size_of::<(
            WorkspaceAllocationPopulation,
            Result<Rc<Storage>, Error>,
            Vec<Rc<Storage>>,
        )>())?;
        let mut created = self.metadata_vec(source.allocations().len())?;
        self.reserve_metadata_vec(&mut aliases, source.allocations().len())?;
        self.reserve_metadata_vec(&mut trace.allocations, source.allocations().len())?;
        for row in source.allocations() {
            let births = row
                .maximum_allocations
                .ok_or(WorkspaceMetadataError::Unqualified)?;
            if row.bytes != 0 && births == 0 {
                return Err(WorkspaceMetadataError::Unqualified.into());
            }
            let mut root = self.try_new_storage(Some(row.bytes), Vec::new())?;
            let allocation = Rc::get_mut(&mut root).expect("unpublished output backing");
            allocation.maximum_allocations = births;
            allocation.placement = self.copy_placement(row.placement.as_ref())?;
            allocation.host_control_bytes = row.host_control_bytes;
            allocation.population = Some(source.clone());
            aliases.push(root.clone());
            created.push(root);
        }
        let mut output = self.try_new_storage(Some(0), aliases)?;
        Rc::get_mut(&mut output)
            .expect("unpublished output union")
            .population = Some(source);
        trace.allocations.extend(created);
        Ok(output)
    }
}
