//! Once-resolved selected catalog and its exact original read manager.
use super::*;

pub(crate) struct PreparedAddressableSource {
    origin: eredu_checkpoint::store::RetainedCheckpointSource,
    selected: SelectedAddressableEntries,
    options: ParameterBankOptions,
    resolved: MaterializedSelected,
    manager: Option<ResidencyManager>,
}

pub(super) struct MaterializedSelected {
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    entries: Vec<ParameterBankEntry>,
    placements: BTreeMap<ParameterBankKey, eredu_runtime::AddressableBankMemberPlacement>,
    parameter_members: Vec<eredu_runtime::parameter_operations::PreparedBankParameterMember>,
    formats: Vec<WeightQuantization>,
    report: Option<WeightMaterializationReport>,
}

impl PreparedAddressableSource {
    pub(super) fn prepare(
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        selected: SelectedAddressableEntries,
        options: ParameterBankOptions,
        source_stream: &Stream,
        device_stream: &Stream,
        pool: &eredu_runtime::working_memory::MemoryLedger,
    ) -> Result<Option<Self>, AddressableParameterBankError> {
        options.validate()?;
        let origin = store.clone();
        // This ordinary load descriptor authenticates the later selected binder.
        // It is consumed with the once-prepared store, before request execution.
        let declaration = selected.clone();
        let resolved = resolve_selected(store, selected, options, source_stream)?;
        let specs = resolved
            .entries
            .iter()
            .map(|entry| {
                OffloadUnitSpec::new(
                    entry.identity.unit_id(),
                    entry.bytes,
                    ResidencyPolicy::Cacheable,
                    MemoryTier::Disk,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let definitions = resolved
            .entries
            .iter()
            .map(|entry| entry.unit.clone())
            .collect::<Vec<_>>();
        let plan = OffloadPlan::new(options.storage, specs)?;
        let manager = ResidencyManager::prepare_original_addressable(
            resolved.store.clone(),
            &plan,
            &definitions,
            source_stream,
            device_stream,
            pool,
        )
        .map_err(|cause| AddressableParameterBankError::Transformation {
            source: Box::new(Error::Other(Box::new(cause))),
        })?;
        if resolved.entries.is_empty() {
            return Ok(None);
        }
        // None remains a typed absence of original construction qualification
        // (for example ordinary CPU loading). Retain the once-resolved catalog;
        // the existing ordinary initializer handles it, and managed guards still
        // require the real original manager source before admitting execution.
        Ok(Some(Self {
            origin,
            selected: declaration,
            options,
            resolved,
            manager,
        }))
    }

    pub(super) fn into_selected(
        self,
        source: &eredu_checkpoint::store::RetainedCheckpointSource,
        selected: &SelectedAddressableEntries,
        options: ParameterBankOptions,
    ) -> Result<(MaterializedSelected, Option<ResidencyManager>), AddressableParameterBankError>
    {
        if !self.origin.same_source(source) || self.selected != *selected || self.options != options
        {
            return Err(mismatch());
        }
        // The actual store and rewritten recipes remain paired with the manager.
        // into_bank checks that manager against the complete plan and streams.
        Ok((self.resolved, self.manager))
    }
}

impl MaterializedSelected {
    pub(super) fn into_bank(
        self,
        options: ParameterBankOptions,
        source_stream: Stream,
        device_stream: Stream,
        manager: Option<ResidencyManager>,
    ) -> Result<AddressableParameterBank, AddressableParameterBankError> {
        let mut bank = AddressableParameterBank::new_shared_with_prepared_policy(
            self.store,
            self.entries,
            options,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
            source_stream,
            device_stream,
            self.formats,
            self.placements,
            self.report,
            manager,
        )?;
        bank.parameter_members = self.parameter_members;
        Ok(bank)
    }
}

fn mismatch() -> AddressableParameterBankError {
    AddressableParameterBankError::Transformation {
        source: Box::new(Error::ArchitectureModel(
            "prepared addressable catalog differs from the selected source, transformations or policy".into())),
    }
}

pub(super) fn resolve_selected(
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    selected: SelectedAddressableEntries,
    options: ParameterBankOptions,
    source_stream: &Stream,
) -> Result<MaterializedSelected, AddressableParameterBankError> {
    options.validate()?;
    preflight_selected_entry_bindings(store.as_ref(), &selected.entries)?;
    let selected_keys = selected
        .entries
        .iter()
        .map(|entry| entry.identity)
        .collect::<std::collections::BTreeSet<_>>();
    if selected_keys
        != selected
            .placements
            .keys()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
    {
        return Err(AddressableParameterBankError::Transformation {
            source: Box::new(Error::ArchitectureModel(
                "selected addressable placements do not cover the exact entry keys".into(),
            )),
        });
    }
    for entry in &selected.entries {
        let expected = selected
            .expected_bytes
            .get(&entry.identity)
            .ok_or_else(|| AddressableParameterBankError::Transformation {
                source: Box::new(Error::ArchitectureModel(format!(
                    "selected addressable entry {:?} has no neutral byte total",
                    entry.identity
                ))),
            })?;
        let projected = entry
            .unit
            .bindings()
            .iter()
            .try_fold(0u64, |total, binding| {
                let bytes = if let Some(transform) = selected
                    .transformations
                    .get(&(entry.identity, binding.name().to_owned()))
                {
                    let metadata = binding.source_recipe().infer(store.as_ref())?;
                    packed_projection_bytes(
                        metadata.shape(),
                        transform.quantization,
                        &transform.companion_dtype,
                    )?
                } else {
                    binding.expected_bytes()
                };
                total.checked_add(bytes).ok_or_else(|| {
                    Error::ArchitectureModel("selected addressable entry bytes overflowed".into())
                })
            })
            .map_err(|source| AddressableParameterBankError::Transformation {
                source: Box::new(source),
            })?;
        if projected != *expected {
            return Err(AddressableParameterBankError::Transformation {
                source: Box::new(Error::ArchitectureModel(format!(
                    "selected addressable entry {:?} bytes differ: expected {}, projected {}",
                    entry.identity, expected, projected
                ))),
            });
        }
    }
    if selected.transformations.is_empty() {
        let parameter_members = prepared_parameter_members(
            &selected.entries,
            &selected.parameter_targets,
            store.as_ref(),
        )?;
        return Ok(MaterializedSelected {
            store,
            entries: selected.entries,
            placements: selected.placements,
            parameter_members,
            formats: Vec::new(),
            report: None,
        });
    }
    let telemetry_formats = selected_transformation_formats(&selected.transformations);
    let transformed = quantize_selected_entry_catalog(
        store,
        selected.entries,
        selected.transformations,
        options.compact_bank_scratch_bytes,
        source_stream,
    )
    .map_err(|source| AddressableParameterBankError::Transformation {
        source: Box::new(source),
    })?;
    for entry in &transformed.entries {
        if selected.expected_bytes.get(&entry.identity) != Some(&entry.bytes) {
            return Err(AddressableParameterBankError::Transformation {
                source: Box::new(Error::ArchitectureModel(format!(
                    "materialized addressable entry {:?} differs from its neutral selected bytes",
                    entry.identity
                ))),
            });
        }
    }
    let parameter_members = prepared_parameter_members(
        &transformed.entries,
        &selected.parameter_targets,
        transformed.store.as_ref(),
    )?;
    Ok(MaterializedSelected {
        store: transformed.store,
        entries: transformed.entries,
        placements: selected.placements,
        parameter_members,
        formats: telemetry_formats,
        report: Some(transformed.report),
    })
}
