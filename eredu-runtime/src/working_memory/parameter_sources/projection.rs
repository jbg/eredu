//! Paid source table and prospective roots reused by the shared unit binder.
use super::*;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceExistingStorage, WorkspaceMetadataError, WorkspaceTensor,
};
use eredu_nn::{Error, Parameterized};
use std::mem::{size_of, size_of_val};

/// One immutable parameter-source loan and its counted workspace destinations.
/// Prospective roots describe future materialization. Retained rows must instead
/// resolve a backing already installed by the same source. Neither route grants
/// native allocation permission. Invocation roots retire before the next acquire;
/// trace roots preserve only the source's explicit alias lifetime.
pub struct WorkspaceParameterProjection<'source> {
    source: &'source dyn WorkspaceParameterRows,
    counts: WorkspaceParameterCounts,
    units: Vec<WorkspaceParameterUnitRecord>,
    requested: Vec<WorkspaceParameterRequestRecord>,
    rows: Vec<WorkspaceParameterRecord>,
    names: Vec<u8>,
    shapes: Vec<i32>,
    roots: Vec<WorkspaceParameterRootRecord>,
    window_members: Vec<usize>,
    storage: Vec<Option<WorkspaceExistingStorage>>,
    // Every funded destination retires before this cumulative context loan.
    context: WorkspaceContext,
}
impl std::fmt::Debug for WorkspaceParameterProjection<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceParameterProjection")
            .field("counts", &self.counts)
            .finish_non_exhaustive()
    }
}
impl<'source> WorkspaceParameterSourceLoan<'source> {
    /// Counts and fills the same source through its existing shared workers.
    /// No ordinary map adapter is invoked on refusal or on the successful path.
    pub fn prepare_projection(
        self,
        context: &WorkspaceContext,
    ) -> std::result::Result<WorkspaceParameterProjection<'source>, Error> {
        let layouts = workspace_parameter_control_layouts();
        let controls = layouts
            .iter()
            .try_fold(size_of_val(&layouts), |bytes, (_, layout)| {
                bytes.checked_add(layout.bytes)
            })
            .ok_or(WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(
            controls
                .checked_add(size_of::<(
                    WorkspaceParameterProjection<'_>,
                    std::result::Result<WorkspaceParameterProjection<'_>, Error>,
                    &WorkspaceContext,
                )>())
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let source = self.source;
        let counted = self
            .count()
            .map_err(|cause| context.metadata_source(cause))?;
        let counts = counted.counts();
        let mut projection = WorkspaceParameterProjection {
            source,
            counts,
            units: destination(counts.units, context)?,
            requested: destination(counts.requested, context)?,
            rows: destination(counts.rows, context)?,
            names: destination(counts.name_bytes, context)?,
            shapes: destination(counts.shape_elements, context)?,
            roots: destination(counts.roots, context)?,
            window_members: destination(counts.window_members, context)?,
            storage: destination(counts.roots, context)?,
            context: context.clone(),
        };
        drop(
            counted
                .fill(projection.destinations())
                .map_err(|cause| context.metadata_source(cause))?,
        );
        Ok(projection)
    }
}
fn destination<T: Default>(
    count: usize,
    context: &WorkspaceContext,
) -> std::result::Result<Vec<T>, Error> {
    context.charge_metadata(size_of::<(usize, Vec<T>, std::result::Result<Vec<T>, Error>)>())?;
    let mut result = context.metadata_vec(count)?;
    result.resize_with(count, T::default);
    Ok(result)
}
impl WorkspaceParameterProjection<'_> {
    fn destinations(&mut self) -> WorkspaceParameterDestinations<'_> {
        WorkspaceParameterDestinations {
            units: &mut self.units,
            requested: &mut self.requested,
            rows: &mut self.rows,
            names: &mut self.names,
            shapes: &mut self.shapes,
            roots: &mut self.roots,
            window_members: &mut self.window_members,
        }
    }
    /// Complete prepublication binding from the source's exact requested unit.
    /// The unit remains unchanged if shared immutable/mutable traversal refuses.
    pub fn bind<M: Parameterized<WorkspaceTensor>>(
        &mut self,
        module: &mut M,
        ordinal: usize,
        address: ExecutionUnitAddress,
        context: &WorkspaceContext,
    ) -> std::result::Result<(), Error> {
        let frames = [
            size_of::<(
                &mut Self,
                &mut M,
                usize,
                ExecutionUnitAddress,
                &WorkspaceContext,
            )>(),
            size_of::<Vec<crate::PreparedParameterBinding<'_, WorkspaceTensor>>>(),
            size_of::<crate::PreparedParameterBinding<'_, WorkspaceTensor>>(),
            size_of::<WorkspaceParameterRecord>(),
            size_of::<WorkspaceParameterRootRecord>(),
            size_of::<WorkspaceLayoutView<'_>>(),
            size_of::<WorkspaceExistingStorage>(),
            size_of::<WorkspaceTensor>(),
            size_of::<Range<usize>>(),
            size_of::<std::result::Result<WorkspaceTensor, Error>>(),
            size_of::<std::result::Result<(), Error>>(),
            size_of::<(
                &dyn WorkspaceParameterRows,
                &eredu_nn::ParameterId,
                &str,
                bool,
            )>(),
        ];
        context.charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let fail = || context.metadata_source(WorkspaceParameterSourceError::SourceMismatch);
        if !self.context.shares_trace(context)
            || self.source.execution_address(ordinal) != Some(address)
        {
            return Err(fail());
        }
        let unit = self.requested.get(ordinal).ok_or_else(fail)?.unit;
        let range = self.units.get(unit).ok_or_else(fail)?.rows.clone();
        // Repeated visits get fresh invocation roots, including when the next
        // unit names the same physical owner. Trace aliases retain their roots.
        for (root, storage) in self.roots.iter().zip(&mut self.storage) {
            if root.owner.lifetime == WorkspaceParameterLifetime::Invocation {
                *storage = None;
            }
        }
        let mut bindings = context.metadata_vec(range.len())?;
        for row in &self.rows[range] {
            let prototype = &self.roots[row.root];
            let name = std::str::from_utf8(&self.names[row.name.clone()]).map_err(|_| fail())?;
            let layout = context
                .layout(&self.shapes[row.shape.clone()], row.dtype.ok_or_else(fail)?)?
                .with_representation(row.representation);
            if row.retained {
                let root = context
                    .retained_parameter_backing(name, layout.as_view())?
                    .ok_or(WorkspaceMetadataError::Unqualified)?;
                if root.capacity_bytes() != Some(prototype.capacity_bytes)
                    || (context.memory_topology().is_some()
                        && root.placement()
                            != self
                                .source
                                .placement(prototype.owner.unit, prototype.owner.row))
                {
                    return Err(fail());
                }
                let value = WorkspaceTensor::existing_with_storage(layout, &root, context)?;
                bindings.push(crate::PreparedParameterBinding::new(name, value));
                continue;
            }
            let root = &mut self.storage[row.root];
            if root.is_none() {
                *root = Some(if context.memory_topology().is_some() {
                    let placement = self
                        .source
                        .placement(prototype.owner.unit, prototype.owner.row)
                        .ok_or(WorkspaceMetadataError::Unqualified)?;
                    WorkspaceExistingStorage::try_new_placed(
                        Some(prototype.capacity_bytes),
                        placement,
                        context,
                    )?
                } else {
                    WorkspaceExistingStorage::try_new(Some(prototype.capacity_bytes), context)?
                });
            }
            let value = WorkspaceTensor::existing_with_storage(
                layout,
                root.as_ref().expect("created root"),
                context,
            )?;
            bindings.push(crate::PreparedParameterBinding::new(name, value));
        }
        crate::working_memory::bind_prepared_workspace_parameters(
            module,
            &mut bindings,
            context,
            |id| self.source.excludes_parameter(id.as_str()),
        )
    }
}
