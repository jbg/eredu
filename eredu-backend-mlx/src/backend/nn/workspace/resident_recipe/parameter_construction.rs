//! Unloaded unit slots from the same authoritative inventory as strict binding.
use super::*;
use crate::backend::runtime::execution::generic::LayerwiseWorkspace;
use eredu_runtime::working_memory::{
    WorkingMemoryError, WorkspaceParameterRow, WorkspaceParameterRows,
    WorkspaceParameterSourceError, WorkspaceParameterUnit,
};
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Constructors {
    pub(super) slots: usize,
    pub(super) scalar_bytes: u64,
    rank: usize,
    dtypes: [usize; 5],
}

fn source_error(cause: WorkspaceParameterSourceError) -> crate::backend::error::Error {
    crate::backend::error::Error::PrefillControl(match cause {
        WorkspaceParameterSourceError::Overflow => WorkingMemoryError::Overflow,
        _ => WorkingMemoryError::UnknownBound,
    })
}

impl Constructors {
    pub(super) fn native_source(self,query:usize)
        ->Result<(safemlx::ResidentGraphLayout,usize),crate::backend::error::Error> {
        use crate::backend::error::Error as NativeError;
        let unknown=||NativeError::PrefillControl(WorkingMemoryError::UnknownBound);
        let overflow=||NativeError::PrefillControl(WorkingMemoryError::Overflow);
        // PhysicalParam::unloaded -> zeros_dtype -> mlx::zeros: one eager
        // array(0,dtype), Broadcast, same-dtype AsType candidate, and Full.
        // The strict binder destroys Full before any evaluation. Only the
        // scalar is a physical allocation; its lazy logical weight extent is
        // a Graph descriptor shape, never another tensor allocation here.
        let primitives = self.slots.checked_mul(3).ok_or_else(overflow)?;
        let additional = safemlx::OperationEvent::resident_graph_layout_with_shells(
            primitives,
            self.slots,
            self.rank,
            4,
            0,
        )
        .ok_or_else(unknown)?;
        // The actual unit U/Box and authoritative binding rows are already paid
        // by original operation storage. These are the reused safe constructor
        // call/return frames; Graph owns C shells and native descriptor payloads.
        let controls = [
            additional.control_bytes().ok_or_else(unknown)?,
            size_of::<crate::backend::nn::module::PhysicalParam<safemlx::Array>>(),
            size_of::<crate::backend::nn::module::PhysicalParam<Option<safemlx::Array>>>(),
            size_of::<
                Result<
                    crate::backend::nn::module::PhysicalParam<safemlx::Array>,
                    safemlx::error::Exception,
                >,
            >(),
            size_of::<
                Result<
                    crate::backend::nn::module::PhysicalParam<Option<safemlx::Array>>,
                    safemlx::error::Exception,
                >,
            >(),
            size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
            size_of::<(&[i32], safemlx::Dtype, &safemlx::Stream)>(),
        ];
        let controls = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| {
                n.checked_add(query)
            })
            .ok_or_else(overflow)?;
        Ok((additional,controls))
    }
}

impl ResidentNativeRecipe {
    /// The selected native acquisition worker constructs each requested unit
    /// once per forward, then its strict binder replaces every parameter slot.
    /// Count every named slot, including aliases: shared eventual weight storage
    /// does not share the independently constructed scalar seeds.
    ///
    /// Source identity/layout checks and the native original binding worker stay
    /// mandatory. This method cannot qualify an unbound placeholder operation,
    /// a legacy disk source, or an arbitrary constructor callback.
    pub(crate) fn bind_layerwise_parameter_constructors(
        &mut self,
        source: &LayerwiseWorkspace,
    ) -> Result<(), crate::backend::error::Error> {
        use crate::backend::error::Error as NativeError;
        let unknown = || NativeError::PrefillControl(WorkingMemoryError::UnknownBound);
        let overflow = || NativeError::PrefillControl(WorkingMemoryError::Overflow);
        let identity = || NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch);
        if self.host_copies.ok_or_else(unknown)?.parameters.is_some() {
            return Err(identity());
        }

        let (constructors, query) = constructor_source(source, self.planning_metadata.as_ref())?;
        if self.layerwise_constructors.is_some_and(|expected| expected != constructors) {
            return Err(identity());
        }

        let (additional,controls)=constructors.native_source(
            if self.planning_metadata.is_none(){query}else{0})?;
        let primitives=additional.primitives();
        for row in &mut self.records {
            let graph = row.graph.ok_or_else(unknown)?;
            let storage = row.mutable_storage.as_mut().ok_or_else(unknown)?;
            row.graph = Some(
                safemlx::OperationEvent::resident_graph_layout_with_shells(
                    graph
                        .primitives()
                        .checked_add(primitives)
                        .ok_or_else(overflow)?,
                    graph
                        .seeds()
                        .checked_add(constructors.slots)
                        .ok_or_else(overflow)?,
                    graph.maximum_rank().max(constructors.rank),
                    graph.maximum_operands().max(4),
                    graph.additional_shells(),
                )
                .ok_or_else(unknown)?,
            );
            storage.mutable_bytes = storage
                .mutable_bytes
                .checked_add(constructors.scalar_bytes)
                .ok_or_else(overflow)?;
            storage.maximum_births = storage
                .maximum_births
                .checked_add(constructors.slots)
                .ok_or_else(overflow)?;
            row.maximum_rank = row.maximum_rank.max(constructors.rank);
            row.host_primitive_nodes = row
                .host_primitive_nodes
                .checked_add(primitives)
                .ok_or_else(overflow)?;
            row.query_controls = Some(
                row.query_controls
                    .ok_or_else(unknown)?
                    .checked_add(controls)
                    .ok_or_else(overflow)?,
            );
            // No Record, shader, nested Eval or worker population is added:
            // these replaced Full/Broadcast nodes never enter an Eval DAG.
        }
        self.host_copies
            .as_mut()
            .expect("validated source-copy recipe")
            .parameters = Some(super::host_copies::HostParameterConstruction::FreshUnits);
        Ok(())
    }
}

impl ResidentNativeRecipe {
    /// The caller lends actual retained modules and their immutable source plan.
    /// Lease binding is already in the source-copy recipe. Replacement/restore
    /// maps add these actual C-handle copies, without fresh scalar or Full nodes.
    pub(crate) fn bind_retained_parameter_slots(
        &mut self,
        source: &LayerwiseWorkspace,
        retained_handle_copies: usize,
    ) -> Result<(), crate::backend::error::Error> {
        use crate::backend::error::Error as NativeError;
        let unknown = || NativeError::PrefillControl(WorkingMemoryError::UnknownBound);
        let overflow = || NativeError::PrefillControl(WorkingMemoryError::Overflow);
        let identity = || NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch);
        if self.host_copies.ok_or_else(unknown)?.parameters.is_some() {
            return Err(identity());
        }
        let frames = [
            size_of::<(&mut Self, &LayerwiseWorkspace, usize)>(),
            size_of::<eredu_runtime::working_memory::WorkspaceParameterSourceLoan<'_>>(),
            size_of::<eredu_runtime::working_memory::CountedWorkspaceParameterSource<'_>>(),
            size_of::<Result<eredu_runtime::working_memory::CountedWorkspaceParameterSource<'_>, WorkspaceParameterSourceError>>(),
            size_of::<Result<(), NativeError>>(),
        ];
        let query = frames.into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add).ok_or_else(overflow)?;
        if let Some(funding) = &self.planning_metadata {
            funding.reserve_metadata(query).map_err(NativeError::WorkspacePlanning)?;
        }
        let counted = source.parameter_source().count().map_err(source_error)?;
        if counted.counts().requested != source.layout().len() {
            return Err(identity());
        }
        for row in &mut self.records {
            let graph = row.graph.ok_or_else(unknown)?;
            let additional = safemlx::OperationEvent::resident_graph_layout_with_shells(
                graph.primitives(), graph.seeds(), graph.maximum_rank(),
                graph.maximum_operands(),
                graph.additional_shells().checked_add(retained_handle_copies).ok_or_else(overflow)?,
            ).ok_or_else(unknown)?;
            let controls = additional.control_bytes().ok_or_else(unknown)?
                .checked_add(if self.planning_metadata.is_none() { query } else { 0 })
                .ok_or_else(overflow)?;
            row.query_controls = Some(row.query_controls.ok_or_else(unknown)?
                .checked_add(controls).ok_or_else(overflow)?);
            row.graph = Some(additional);
        }
        self.host_copies.as_mut().expect("validated source-copy recipe").parameters =
            Some(super::host_copies::HostParameterConstruction::RetainedModules);
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("layerwise constructor trace differs from retained parameter source: actual {actual:?}, expected {expected:?}")]
struct TraceMismatch { actual: Constructors, expected: Constructors }

fn dtype_index(dtype: WorkspaceDtype) -> usize {
    match dtype {
        WorkspaceDtype::Float32 => 0, WorkspaceDtype::Int32 => 1,
        WorkspaceDtype::Bool => 2, WorkspaceDtype::Uint8 => 3,
        WorkspaceDtype::Uint32 => 4,
    }
}

// One borrowed census for cold trace qualification and the later native graph
// producer. It observes the selected logical unit list, preserving repeat slots.
pub(super) fn constructor_source(source: &LayerwiseWorkspace,
    funding: Option<&eredu_nn::workspace::WorkspaceMetadataFunding>)
    -> Result<(Constructors, usize), crate::backend::error::Error>
{
    use crate::backend::error::Error as NativeError;
    let overflow = || NativeError::PrefillControl(WorkingMemoryError::Overflow);
    let identity = || NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch);
        // These are cold borrowed reads and fixed frames, not a second source
        // snapshot. Pay the actual participating planning account before query.
        let frames = [
            size_of::<Constructors>(),
            size_of::<&LayerwiseWorkspace>(),
            size_of::<WorkspaceParameterRow<'_>>(),
            size_of::<WorkspaceParameterUnit<'_>>(),
            size_of::<Result<WorkspaceParameterRow<'_>, WorkspaceParameterSourceError>>(),
            size_of::<Result<WorkspaceParameterUnit<'_>, WorkspaceParameterSourceError>>(),
            size_of::<eredu_runtime::working_memory::WorkspaceParameterSourceLoan<'_>>(),
            size_of::<eredu_runtime::working_memory::CountedWorkspaceParameterSource<'_>>(),
            size_of::<
                Result<
                    eredu_runtime::working_memory::CountedWorkspaceParameterSource<'_>,
                    WorkspaceParameterSourceError,
                >,
            >(),
            size_of::<Result<(Constructors, usize), NativeError>>(),
            size_of::<Option<&eredu_nn::workspace::WorkspaceMetadataFunding>>(),
        ];
        let query = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or_else(overflow)?;
        if let Some(funding) = funding {
            funding
                .reserve_metadata(query)
                .map_err(NativeError::WorkspacePlanning)?;
        }
        // Validate the complete retained closure without allocating maps, names,
        // source values or a temporary provider. Then visit selected ordinals,
        // not the larger lookahead/canonical closure or copy retry population.
        // Repeated logical ordinals remain repeated even when requested_unit
        // resolves them to one shared physical owner. The source constructor
        // already refuses excluded/overridden parameter sets.
        let counted = source.parameter_source().count().map_err(source_error)?;
        if counted.counts().requested != source.layout().len() {
            return Err(identity());
        }
        let mut constructors = Constructors::default();
        source.with_execution_ordinals(|selected| {
        let count=selected.map_or(source.layout().len(),<[usize]>::len);
        for index in 0..count {
            let ordinal=selected.map_or(index,|ordinals|ordinals[index]);
            let unit = source.requested_unit(ordinal).map_err(source_error)?;
            let rows = source.unit(unit).map_err(source_error)?.rows;
            for index in 0..rows {
                let row = source.row(unit, index).map_err(source_error)?;
                constructors.slots = constructors.slots.checked_add(1).ok_or_else(overflow)?;
                // The source projector uses the constructor's neutral dtype:
                // floating slots start F32 even when later bound to FP16/BF16;
                // packed/integer/bool slots retain their exact declared dtype.
                constructors.scalar_bytes = constructors
                    .scalar_bytes
                    .checked_add(row.dtype.bytes())
                    .ok_or_else(overflow)?;
                constructors.rank = constructors.rank.max(row.shape.len());
                let dtype = dtype_index(row.dtype);
                constructors.dtypes[dtype] = constructors.dtypes[dtype].checked_add(1).ok_or_else(overflow)?;
            }
        }

        Ok::<_,NativeError>(())
        })?;

    Ok((constructors, query))
}

impl ResidentRecipeRecorder {
    pub(crate) fn bind_layerwise_constructor_source(&mut self, source: &LayerwiseWorkspace)
        -> Result<(), Error>
    {
        if self.layerwise_constructors.is_some() || !self.records.is_empty() || self.cpu.is_some() {
            return Err(self.metadata_error("layerwise constructor source already bound or differs"));
        }
        let funding = self.context.as_ref().and_then(|context| context.metadata_funding());
        let (constructors, _) = constructor_source(source, funding.as_ref())
            .map_err(|cause| self.metadata_source(cause))?;
        self.layerwise_constructors = Some(constructors);
        Ok(())
    }

    pub(super) fn validate_layerwise_constructor_trace(&self, report: &WorkspaceTraceReport)
        -> Result<(), Error>
    {
        let Some(expected) = self.layerwise_constructors else { return Ok(()); };
        if let Some(context) = &self.context {
            context.charge_metadata(size_of::<(Constructors, &WorkspaceTraceReport,
                &Self, Result<(), Error>, &WorkspaceOperation)>())?;
        }
        let mut actual = Constructors::default();
        for operation in &report.operations {
            if !matches!(operation.kind, WorkspaceOperationKind::ParameterPlaceholder) { continue; }
            if !operation.inputs.is_empty() || operation.outputs.len() != 1
                || !operation.outputs[0].shape().is_empty()
            { return Err(self.metadata_error("layerwise constructor trace changed its scalar seed")); }
            let dtype = operation.outputs[0].dtype();
            let index = dtype_index(dtype);
            actual.slots = self.add(actual.slots, 1)?;
            actual.dtypes[index] = self.add(actual.dtypes[index], 1)?;
            actual.scalar_bytes = actual.scalar_bytes.checked_add(dtype.bytes())
                .ok_or_else(|| self.metadata_error("layerwise constructor seed bytes overflow"))?;
        }
        if actual.slots != expected.slots || actual.scalar_bytes != expected.scalar_bytes
            || actual.dtypes != expected.dtypes
        { return Err(self.metadata_source(TraceMismatch { actual, expected })); }
        Ok(())
    }
}
