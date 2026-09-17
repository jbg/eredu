use super::*;

#[derive(Clone, Copy, Default)]
pub(super) struct Counts {
    pub(super) modules: usize,
    pub(super) rows: usize,
    pub(super) native: usize,
}
impl PredictionResourceVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer> for Counts {
    type Error = WorkspaceMetadataError;
    fn module<U: Parameterized<MlxTensor>>(
        &mut self,
        _: usize,
        module: &MlxPredictionModule<U>,
    ) -> Result<(), Self::Error> {
        self.modules = self
            .modules
            .checked_add(1)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.rows = self
            .rows
            .checked_add(module.bindings.len())
            .ok_or(WorkspaceMetadataError::Overflow)?;
        for binding in &module.bindings {
            if module.residency.is_fully_resident()
                || module.replacements.contains_key(binding.name())
            {
                self.native = self
                    .native
                    .checked_add(1)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
            }
        }
        Ok(())
    }
    fn pooling_state(
        &mut self,
        _: &OwnedPredictionCache<MlxPoolingAttentionCache>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn model_state(&mut self, _: &super::super::super::MlxHybridState) -> Result<(), Self::Error> {
        Ok(())
    }
}

pub(super) enum Value {
    Native(Array),
    Prospective(WorkspaceTensor),
}
pub(super) struct SnapshotRow {
    pub(super) name: String,
    pub(super) value: Value,
}
pub(super) struct Snapshot {
    pub(super) physical: usize,
    pub(super) id: String,
    pub(super) manager: ResidencyManager,
    pub(super) rows: Vec<SnapshotRow>,
}
struct Root {
    owner: WorkspaceParameterOwner,
    invocation: Option<usize>,
    value: WorkspaceExistingStorage,
}
pub(super) struct Collect<'a> {
    counts: Counts,
    target: Option<&'a LayerwiseWorkspace>,
    source: Option<LayerwiseWorkspace>,
    snapshots: Vec<Snapshot>,
    roots: Vec<Root>,
    native: usize,
    context: &'a WorkspaceContext,
}
impl<'a> Collect<'a> {
    pub(super) fn new(
        counts: Counts,
        source: Option<&'a LayerwiseWorkspace>,
        context: &'a WorkspaceContext,
    ) -> Result<Self, Error> {
        controls(
            context,
            &[
                size_of::<Self>(),
                size_of::<Snapshot>(),
                size_of::<SnapshotRow>(),
                size_of::<Root>(),
                size_of::<Result<Self, Error>>(),
                size_of::<(Vec<Snapshot>, Option<LayerwiseWorkspace>)>(),
                size_of::<Result<(Vec<Snapshot>, Option<LayerwiseWorkspace>), Error>>(),
                size_of::<Value>(),
            ],
        )?;
        Ok(Self {
            counts,
            target: source,
            source: None,
            snapshots: context.metadata_vec(counts.modules)?,
            roots: context.metadata_vec(
                counts
                    .rows
                    .checked_sub(counts.native)
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )?,
            native: 0,
            context,
        })
    }
    pub(super) fn finish(self) -> Result<(Vec<Snapshot>, Option<LayerwiseWorkspace>), Error> {
        let rows = self
            .snapshots
            .iter()
            .try_fold(0usize, |count, module| count.checked_add(module.rows.len()))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        if self.snapshots.len() != self.counts.modules
            || self.native != self.counts.native
            || rows != self.counts.rows
        {
            return Err(failure(self.context, SourceError::Traversal));
        }
        Ok((self.snapshots, self.source))
    }
    fn native(&mut self, array: &Array) -> Result<Value, Error> {
        if self.native == self.counts.native {
            return Err(failure(self.context, SourceError::Traversal));
        }
        controls(
            self.context,
            &[
                Array::inspection_clone_handle_bytes(),
                safemlx::PreparedArrayClone::control_bytes()
                    .ok_or(WorkspaceMetadataError::Overflow)?,
                size_of::<Result<Array, safemlx::PreparedArrayCloneCause>>(),
            ],
        )?;
        let mut clone = safemlx::PreparedArrayClone::try_prepare_for_inspection()
            .map_err(|cause| self.context.metadata_source(cause))?;
        let value = clone
            .fill_for_inspection(array)
            .map_err(|cause| self.context.metadata_source(cause))?;
        self.native += 1;
        Ok(Value::Native(value))
    }
    fn prospective(&mut self, unit: usize, index: usize, dense: usize) -> Result<Value, Error> {
        let source = self
            .source
            .as_ref()
            .ok_or_else(|| failure(self.context, SourceError::Module))?;
        let row = source
            .row(unit, index)
            .map_err(|cause| self.context.metadata_source(cause))?;
        let invocation =
            (row.owner.lifetime == WorkspaceParameterLifetime::Invocation).then_some(dense);
        let root = if let Some(root) = self
            .roots
            .iter()
            .find(|root| root.owner == row.owner && root.invocation == invocation)
        {
            if root.value.capacity_bytes() != Some(row.capacity_bytes) {
                return Err(failure(self.context, SourceError::Module));
            }
            root.value.clone()
        } else {
            if self.roots.len() == self.counts.rows - self.counts.native {
                return Err(failure(self.context, SourceError::Traversal));
            }
            let root = WorkspaceExistingStorage::try_new(Some(row.capacity_bytes), self.context)?;
            self.roots.push(Root {
                owner: row.owner,
                invocation,
                value: root.clone(),
            });
            root
        };
        let layout = self.context.layout(row.shape, row.dtype)?;
        Ok(Value::Prospective(WorkspaceTensor::existing_with_storage(
            layout,
            &root,
            self.context,
        )?))
    }
}
impl PredictionResourceVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
    for Collect<'_>
{
    type Error = Error;
    fn module<U: Parameterized<MlxTensor>>(
        &mut self,
        physical: usize,
        module: &MlxPredictionModule<U>,
    ) -> Result<(), Error> {
        let dense = self.snapshots.len();
        if dense == self.counts.modules || self.snapshots.iter().any(|row| row.physical == physical)
        {
            return Err(failure(self.context, SourceError::Traversal));
        }
        let manager = module
            .manager
            .get()
            .ok_or_else(|| failure(self.context, SourceError::Module))?;
        let id = module
            .id
            .as_ref()
            .ok_or_else(|| failure(self.context, SourceError::Module))?;
        if module
            .replacements
            .keys()
            .any(|name| !module.bindings.iter().any(|binding| binding.name() == name))
        {
            return Err(failure(self.context, SourceError::Module));
        }
        controls(
            self.context,
            &[
                ResidentParameterSource::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
                size_of::<Option<ResidentParameterSource>>(),
                size_of::<Option<usize>>(),
                size_of::<Option<&LayerwiseWorkspace>>(),
                size_of::<
                    Option<
                        &crate::backend::runtime::residency::manager::SupplementaryResidencySource,
                    >,
                >(),
                size_of::<Result<(), crate::backend::error::Error>>(),
                size_of::<eredu_runtime::working_memory::WorkspaceParameterRow<'_>>(),
                size_of::<eredu_runtime::working_memory::WorkspaceParameterUnit<'_>>(),
            ],
        )?;
        let resident = if module.residency.is_fully_resident() {
            Some(
                manager
                    .resident_parameter_source(id, &module.bindings)
                    .map_err(|cause| self.context.metadata_source(cause))?,
            )
        } else {
            None
        };
        let unit = if resident.is_none() {
            if self
                .target
                .is_some_and(|source| !source.parameter_source_matches_manager(manager))
            {
                return Err(failure(self.context, SourceError::Module));
            }
            let declared = manager
                .supplementary_residency_source()
                .ok_or_else(|| failure(self.context, SourceError::Module))?;
            self.context.charge_metadata(
                declared
                    .projection_control_bytes()
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )?;
            if self.source.is_none() {
                let source =
                    LayerwiseWorkspace::from_supplementary_source(manager, declared, self.context)
                        .map_err(|cause| self.context.metadata_source(cause))?;
                source
                    .parameter_source()
                    .count()
                    .map_err(|cause| self.context.metadata_source(cause))?;
                self.source = Some(source);
            }
            let source = self
                .source
                .as_ref()
                .ok_or_else(|| failure(self.context, SourceError::Module))?;
            source
                .validate_supplementary_policy(manager, declared)
                .map_err(|cause| self.context.metadata_source(cause))?;
            let mut selected = None;
            for candidate in 0..source.unit_count() {
                let unit = source
                    .unit(candidate)
                    .map_err(|cause| self.context.metadata_source(cause))?;
                if unit.id == id {
                    if selected.is_some() || unit.rows != module.bindings.len() {
                        return Err(failure(self.context, SourceError::Module));
                    }
                    selected = Some(candidate);
                }
            }
            Some(selected.ok_or_else(|| failure(self.context, SourceError::Module))?)
        } else {
            None
        };
        let mut rows = self.context.metadata_vec(module.bindings.len())?;
        for (index, binding) in module.bindings.iter().enumerate() {
            if let Some(unit) = unit {
                let row = self
                    .source
                    .as_ref()
                    .expect("selected source")
                    .row(unit, index)
                    .map_err(|cause| self.context.metadata_source(cause))?;
                if row.binding != binding {
                    return Err(failure(self.context, SourceError::Module));
                }
            }
            let value = if let Some(replacement) = module.replacements.get(binding.name()) {
                self.native(replacement.as_array())?
            } else if let Some(resident) = &resident {
                self.native(
                    resident
                        .value(binding.name())
                        .ok_or_else(|| failure(self.context, SourceError::Module))?,
                )?
            } else {
                self.prospective(unit.expect("streamed source"), index, dense)?
            };
            rows.push(SnapshotRow {
                name: self
                    .context
                    .metadata_string(format_args!("{}", binding.name()))?,
                value,
            });
        }
        self.snapshots.push(Snapshot {
            physical,
            id: self
                .context
                .metadata_string(format_args!("{}", id.as_str()))?,
            manager: manager.clone(),
            rows,
        });
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
