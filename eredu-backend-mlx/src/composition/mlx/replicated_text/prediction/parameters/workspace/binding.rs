use super::*;
use eredu_nn::{ParameterMetadata, ParameterMetadataView, ParameterVisitor};

struct SourceContext<'a> {
    context: &'a WorkspaceContext,
    error: Option<Error>,
}
impl<'v> ParameterVisitor<'v, WorkspaceTensor> for SourceContext<'_> {
    fn requires_borrowed_metadata(&self) -> bool {
        true
    }
    fn borrowed_metadata_unavailable(&mut self) {
        if self.error.is_none() {
            self.error = Some(failure(self.context, SourceError::Metadata));
        }
    }
    fn visit(&mut self, _: ParameterMetadata, _: &'v WorkspaceTensor) {
        self.borrowed_metadata_unavailable();
    }
    fn visit_borrowed(&mut self, _: ParameterMetadataView<'_>, value: &'v WorkspaceTensor) {
        if self.error.is_none() {
            self.error = self.context.validate_values([value]).err();
        }
    }
}

pub(super) struct Bind<'a, U> {
    selected: usize,
    total: usize,
    visited: usize,
    bound: bool,
    expected: &'a Module,
    source: &'a U,
    local: &'a mut U,
    tasks: &'a [ReplicatedTextMaterializationTask],
    layout: Option<&'a LocalModelLayout>,
    layerwise: Option<&'a LayerwiseWorkspace>,
    context: &'a WorkspaceContext,
}
impl<'a, U> Bind<'a, U> {
    pub(super) fn new(
        selected: usize,
        total: usize,
        expected: &'a Module,
        source: &'a U,
        local: &'a mut U,
        tasks: &'a [ReplicatedTextMaterializationTask],
        layout: Option<&'a LocalModelLayout>,
        layerwise: Option<&'a LayerwiseWorkspace>,
        context: &'a WorkspaceContext,
    ) -> Self {
        Self {
            selected,
            total,
            visited: 0,
            bound: false,
            expected,
            source,
            local,
            tasks,
            layout,
            layerwise,
            context,
        }
    }
    pub(super) fn finish(self) -> Result<(), Error> {
        if !self.bound || self.visited != self.total {
            return Err(failure(self.context, SourceError::Traversal));
        }
        Ok(())
    }
}
impl<U: Parameterized<WorkspaceTensor>>
    PredictionResourceVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer> for Bind<'_, U>
{
    type Error = Error;
    fn module<M: Parameterized<MlxTensor>>(
        &mut self,
        physical: usize,
        module: &MlxPredictionModule<M>,
    ) -> Result<(), Error> {
        let index = self.visited;
        self.visited = self
            .visited
            .checked_add(1)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        if index != self.selected {
            return Ok(());
        }
        let manager = module
            .manager
            .get()
            .ok_or_else(|| failure(self.context, SourceError::Module))?;
        let id = module
            .id
            .as_ref()
            .ok_or_else(|| failure(self.context, SourceError::Module))?;
        if self.bound
            || physical != self.expected.physical
            || id.as_str() != self.expected.id
            || !manager.same_source_manager(&self.expected.manager)
            || module.tasks.as_slice() != self.tasks
            || module.layout.as_deref() != self.layout
            || module.bindings.len() != self.expected.rows.len()
        {
            return Err(failure(self.context, SourceError::Module));
        }
        if !module.residency.is_fully_resident() {
            controls(self.context, &[
                size_of::<Option<&LayerwiseWorkspace>>(),
                size_of::<Option<&crate::backend::runtime::residency::manager::SupplementaryResidencySource>>(),
                size_of::<Result<(), crate::backend::error::Error>>(),
                size_of::<Option<usize>>(),
            ])?;
            let declared = manager
                .supplementary_residency_source()
                .ok_or_else(|| failure(self.context, SourceError::Module))?;
            self.context.charge_metadata(
                declared
                    .projection_control_bytes()
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )?;
            let source = self
                .layerwise
                .ok_or_else(|| failure(self.context, SourceError::Module))?;
            source
                .validate_supplementary_policy(manager, declared)
                .map_err(|cause| self.context.metadata_source(cause))?;
            if declared.ordinal(id).is_none() {
                return Err(failure(self.context, SourceError::Module));
            }
        }
        controls(
            self.context,
            &[
                size_of::<SourceContext<'_>>(),
                size_of::<Result<(), Error>>(),
                ResidentParameterSource::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
                size_of::<Option<ResidentParameterSource>>(),
                Array::descriptor_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
            ],
        )?;
        let mut source = SourceContext {
            context: self.context,
            error: None,
        };
        self.source.visit_parameters(&mut source);
        if let Some(cause) = source.error {
            return Err(cause);
        }
        let resident = if module.residency.is_fully_resident() {
            Some(
                manager
                    .resident_parameter_source(id, &module.bindings)
                    .map_err(|cause| self.context.metadata_source(cause))?,
            )
        } else {
            None
        };
        let mut rows = self.context.metadata_vec(self.expected.rows.len())?;
        for (binding, row) in module.bindings.iter().zip(&self.expected.rows) {
            if binding.name() != row.name {
                return Err(failure(self.context, SourceError::Module));
            }
            if let Some((identity, bytes)) = row.native {
                let actual = match module.replacements.get(binding.name()) {
                    Some(value) => value.as_array(),
                    None => resident
                        .as_ref()
                        .and_then(|owner| owner.value(binding.name()))
                        .ok_or_else(|| failure(self.context, SourceError::Module))?,
                };
                // Source backing is authenticated by native identity as well as
                // full bytes. Shape equality alone cannot certify this alias.
                controls(
                    self.context,
                    &[
                        Array::descriptor_control_bytes()
                            .ok_or(WorkspaceMetadataError::Overflow)?,
                        size_of::<(safemlx::AllocationIdentity, u64)>(),
                    ],
                )?;
                let descriptor = actual
                    .try_descriptor()
                    .map_err(|cause| self.context.metadata_source(cause))?;
                let allocation = descriptor
                    .facts()
                    .allocation()
                    .ok_or_else(|| failure(self.context, SourceError::Module))?;
                if allocation.identity() != identity
                    || u64::try_from(allocation.bytes()).ok() != Some(bytes)
                {
                    return Err(failure(self.context, SourceError::Module));
                }
            } else if module.replacements.contains_key(binding.name()) || resident.is_some() {
                return Err(failure(self.context, SourceError::Module));
            }
            rows.push(eredu_runtime::PreparedParameterBinding::new(
                &row.name,
                row.value.clone(),
            ));
        }
        eredu_runtime::working_memory::bind_prepared_workspace_parameters(
            self.local,
            &mut rows,
            self.context,
        )?;
        self.bound = true;
        Ok(())
    }
    fn pooling_state(
        &mut self,
        _: &OwnedPredictionCache<MlxPoolingAttentionCache>,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn model_state(&mut self, _: &super::super::super::MlxHybridState) -> Result<(), Error> {
        Ok(())
    }
}
