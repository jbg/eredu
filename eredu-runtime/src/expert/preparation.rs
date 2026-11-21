//! Consuming selected bank admission and post-materialization preparation.
use super::{
    plan_addressable_bank_residency, validate_addressable_bank_selection, AddressableBankCatalog,
    AddressableBankEntry, AddressableBankMemberPlacement, AddressableBankResidencyError,
    AddressableBankSelectionError, AddressableBankTransformPlan, PreparedAddressableBankResidency,
};
use crate::parameter_operations::{
    prepare_bank_parameter_members, BankParameterIndexError, BankParameterPreparationError,
    PreparedBankParameterIndex,
};
use crate::residency::{PreparedResidency, ResidencyInstallation};
use crate::{ParameterBankKey, RowResidencyRange};
use eredu_checkpoint::recipe::RecipeCatalog;
use eredu_core::checkpoint_mechanisms::{
    CheckpointEncodingMaterializer, CheckpointParameterBackend,
};
use eredu_core::residency::parameter_binding::preflight_bindings;
use eredu_core::residency::{MemoryTier, OffloadConfig, ResidencyPolicy, ResidencyStorage};
use std::collections::{BTreeMap, BTreeSet};

/// Invalid selected or returned independent bank resources.
#[derive(Debug, thiserror::Error)]
pub enum AddressableBankPreparationError {
    /// Cold selected ownership or geometry is inconsistent.
    #[error(transparent)]
    Selection(#[from] AddressableBankSelectionError),
    /// Independent storage or row residency admission failed.
    #[error(transparent)]
    Residency(#[from] AddressableBankResidencyError),
    /// Final bindings cannot be joined to their canonical parameter identities.
    #[error(transparent)]
    Parameter(#[from] BankParameterPreparationError),
    /// Final physical bindings have ambiguous canonical ownership.
    #[error(transparent)]
    Index(#[from] BankParameterIndexError),
    /// The materializer returned a different set of physical members.
    #[error("materialized bank omits {missing:?} and unexpectedly returns {unexpected:?}")]
    MemberCoverage {
        /// Selected members absent from the result.
        missing: BTreeSet<ParameterBankKey>,
        /// Returned members outside the retained selection.
        unexpected: BTreeSet<ParameterBankKey>,
    },
    /// A returned member differs from its selected executable extent.
    #[error("materialized bank member {key:?} bytes differ: expected {expected}, got {actual}")]
    ByteMismatch {
        /// Original physical member identity.
        key: ParameterBankKey,
        /// Selected executable extent.
        expected: u64,
        /// Returned admitted extent.
        actual: u64,
    },
}

/// Retained selection authority across backend materialization. Consuming binding
/// finalizes one retained selection exactly once.
/// Native operator and checkpoint-format support are admitted by the backend first.
pub struct AddressableBankPreparation {
    targets: BTreeMap<(ParameterBankKey, String), String>,
    placements: BTreeMap<ParameterBankKey, AddressableBankMemberPlacement>,
    expected: BTreeMap<ParameterBankKey, u64>,
    rows: Vec<RowResidencyRange>,
    config: OffloadConfig,
}

impl AddressableBankPreparation {
    /// Admits selected executable geometry and budgets without reading payloads,
    /// returning exact source entries and the retained ordinary conversion plan.
    pub fn admit<C: RecipeCatalog + ?Sized>(
        selected: AddressableBankCatalog,
        rows: Vec<RowResidencyRange>,
        config: OffloadConfig,
        source: &C,
    ) -> Result<
        (
            Vec<AddressableBankEntry>,
            AddressableBankTransformPlan,
            Self,
        ),
        AddressableBankPreparationError,
    > {
        validate_addressable_bank_selection(
            selected
                .entries
                .iter()
                .map(|entry| (entry.key(), entry.unit().bindings())),
            &selected.placements,
            &selected.expected_bytes,
            &selected.transformations,
            source,
        )?;
        let mut expected = selected.expected_bytes;
        expected.retain(|key, _| selected.placements.contains_key(key));
        plan_addressable_bank_residency(
            selected
                .entries
                .iter()
                .map(|entry| (entry.key(), expected[&entry.key()])),
            &rows,
            config,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
        )?;
        Ok((
            selected.entries,
            AddressableBankTransformPlan::new(selected.transformations),
            Self {
                targets: selected.parameter_targets,
                placements: selected.placements,
                expected,
                rows,
                config,
            },
        ))
    }

    /// Consumes final materializer bindings against the retained selection. Direct
    /// and converted resources use the same identity, byte and metadata admission.
    pub fn bind<C: RecipeCatalog + ?Sized>(
        self,
        entries: Vec<AddressableBankEntry>,
        source: &C,
    ) -> Result<PreparedSelectedAddressableBank, AddressableBankPreparationError> {
        let actual_keys: BTreeSet<_> = entries.iter().map(AddressableBankEntry::key).collect();
        if !self.expected.keys().eq(actual_keys.iter()) {
            return Err(AddressableBankPreparationError::MemberCoverage {
                missing: self
                    .expected
                    .keys()
                    .filter(|key| !actual_keys.contains(*key))
                    .copied()
                    .collect(),
                unexpected: actual_keys
                    .into_iter()
                    .filter(|key| !self.expected.contains_key(key))
                    .collect(),
            });
        }
        for entry in &entries {
            let expected = self.expected[&entry.key()];
            if entry.bytes() != expected {
                return Err(AddressableBankPreparationError::ByteMismatch {
                    key: entry.key(),
                    expected,
                    actual: entry.bytes(),
                });
            }
        }
        let members = prepare_bank_parameter_members(
            entries
                .iter()
                .map(|entry| (entry.key(), entry.unit().bindings())),
            &self.targets,
            source,
        )?;
        let members = PreparedBankParameterIndex::new(members)?;
        let residency = PreparedAddressableBankResidency::new(
            entries,
            self.rows,
            self.config,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
        )?;
        Ok(PreparedSelectedAddressableBank {
            residency,
            members,
            placements: self.placements,
        })
    }
}

/// Final neutral resources shared by direct and converted selected bank execution.
pub struct PreparedSelectedAddressableBank {
    residency: PreparedAddressableBankResidency,
    members: PreparedBankParameterIndex,
    placements: BTreeMap<ParameterBankKey, AddressableBankMemberPlacement>,
}
impl PreparedSelectedAddressableBank {
    /// Moves exact cache inputs, canonical parameter metadata and ownership to the backend.
    pub fn into_parts(
        self,
    ) -> (
        PreparedAddressableBankResidency,
        PreparedBankParameterIndex,
        BTreeMap<ParameterBankKey, AddressableBankMemberPlacement>,
    ) {
        (self.residency, self.members, self.placements)
    }
}

/// Admits native formats and prepares exact resident resources before opening a native pool.
pub fn prepare_selected_addressable_bank<M: CheckpointEncodingMaterializer>(
    source: eredu_checkpoint::store::SharedCheckpointSource,
    catalog: AddressableBankCatalog,
    rows: &crate::SelectedRowLookups,
    support: &impl crate::RowLookupMechanismSupport,
    native: &M,
) -> Result<PreparedAddressableBankStorage<M::Parameters>, eredu_core::BackendFailure> {
    use eredu_core::BackendFailure;

    let options = rows.options();
    rows.validate_binding(options.offload(), support)
        .map_err(BackendFailure::from_error)?;
    let rows = rows.prepared();
    rows.validate_other_banks(catalog.entries.iter().map(|entry| entry.key().bank()))
        .map_err(BackendFailure::from_error)?;
    preflight_addressable_bank(&*source, &catalog.entries, native)?;

    let (mut entries, transforms, preparation) = AddressableBankPreparation::admit(
        catalog,
        rows.ranges().cloned().collect(),
        options.offload(),
        &*source,
    )
    .map_err(BackendFailure::from_error)?;
    let formats = transforms.formats().to_vec();
    let mut source = source;
    let mut materialization: Option<eredu_checkpoint::quantization::WeightMaterializationReport> =
        None;
    for group in transforms.into_groups() {
        let (transform, selected) = group.into_parts();
        let (converted, rewritten, report) = materialize_addressable_bank_conversion(
            source,
            entries,
            transform,
            &selected,
            options.compact_bank_scratch_bytes(),
            native,
        )?;
        source = converted;
        entries = rewritten;
        match &mut materialization {
            Some(combined) => combined.merge(report),
            None => materialization = Some(report),
        }
    }
    let (residency, members, placements) = preparation
        .bind(entries, &*source)
        .map_err(BackendFailure::from_error)?
        .into_parts();
    let (catalog, definitions, row_ranges, plan) = residency.into_parts();
    let residency = PreparedResidency::new(source, BTreeMap::new(), plan, definitions, row_ranges)
        .map_err(BackendFailure::from_error)?;
    Ok(AddressableBankStorage {
        residency,
        catalog,
        members,
        placements,
        options,
        formats,
        materialization,
    })
}

/// Checks admitted bindings against native format support without reading payloads.
pub fn preflight_addressable_bank<M: CheckpointEncodingMaterializer>(
    source: &dyn eredu_checkpoint::store::CheckpointSource,
    entries: &[AddressableBankEntry],
    _native: &M,
) -> Result<(), eredu_core::BackendFailure> {
    for entry in entries {
        preflight_bindings::<M::Parameters>(source, entry.unit().bindings())
            .map_err(eredu_core::BackendFailure::from_error)?;
    }
    Ok(())
}

/// Transforms one selected bank encoding group and binds its rewritten checkpoint keys.
pub fn materialize_addressable_bank_conversion(
    source: eredu_checkpoint::store::SharedCheckpointSource,
    entries: Vec<AddressableBankEntry>,
    transform: crate::AddressableBindingTransform,
    selected: &BTreeSet<(ParameterBankKey, String)>,
    max_working_set_bytes: u64,
    native: &impl CheckpointEncodingMaterializer,
) -> Result<
    (
        eredu_checkpoint::store::SharedCheckpointSource,
        Vec<AddressableBankEntry>,
        eredu_checkpoint::quantization::WeightMaterializationReport,
    ),
    eredu_core::BackendFailure,
> {
    use eredu_checkpoint::quantization::PreparedQuantizedSource;
    use eredu_core::BackendFailure;
    use std::sync::Arc;

    let (plan, bindings) = super::prepare_addressable_bank_quantization(
        &*source,
        entries,
        transform,
        selected,
        max_working_set_bytes,
    )
    .map_err(BackendFailure::from_error)?;
    let source: eredu_checkpoint::store::SharedCheckpointSource =
        Arc::new(PreparedQuantizedSource::new(source, plan).map_err(BackendFailure::from_error)?);
    let report = crate::checkpoint_quantization::realize_prepared_quantization(&source, || {
        native.weights_quantizer()
    })?
    .expect("new admitted bank conversion requires realization");
    let entries = bindings
        .bind(&*source)
        .map_err(BackendFailure::from_error)?;
    Ok((source, entries, report))
}

/// Exact prepared source and residency plan awaiting portable installation.
pub type PreparedAddressableBankStorage<B> = AddressableBankStorage<PreparedResidency<B>>;

/// Addressable storage, its exact member catalog, and conversion evidence.
pub struct AddressableBankStorage<R> {
    residency: R,
    catalog: BTreeMap<ParameterBankKey, u64>,
    members: PreparedBankParameterIndex,
    placements: BTreeMap<ParameterBankKey, AddressableBankMemberPlacement>,
    options: crate::ParameterBankLoadOptions,
    formats: Vec<eredu_checkpoint::WeightQuantization>,
    materialization: Option<eredu_checkpoint::quantization::WeightMaterializationReport>,
}

impl<B: CheckpointParameterBackend> PreparedAddressableBankStorage<B> {
    /// Installs the admitted ledger and initializes it before native bank attachment.
    pub fn install<R>(self, context: R::Context) -> Result<AddressableBankStorage<R>, R::Error>
    where
        R: ResidencyInstallation,
        R::Storage: ResidencyStorage<Backend = B>,
    {
        let residency = self.residency.install::<R>(context)?;
        residency.initialize()?;
        Ok(AddressableBankStorage {
            residency,
            catalog: self.catalog,
            members: self.members,
            placements: self.placements,
            options: self.options,
            formats: self.formats,
            materialization: self.materialization,
        })
    }
}

impl<R> AddressableBankStorage<R> {
    /// Consumes the admitted ledger, parameter catalog, and conversion evidence together.
    pub fn into_parts(
        self,
    ) -> (
        R,
        BTreeMap<ParameterBankKey, u64>,
        PreparedBankParameterIndex,
        BTreeMap<ParameterBankKey, AddressableBankMemberPlacement>,
        crate::ParameterBankLoadOptions,
        Vec<eredu_checkpoint::WeightQuantization>,
        Option<eredu_checkpoint::quantization::WeightMaterializationReport>,
    ) {
        (
            self.residency,
            self.catalog,
            self.members,
            self.placements,
            self.options,
            self.formats,
            self.materialization,
        )
    }
}

/// Resolves the selected local members before preparing an addressable storage pool.
pub fn prepare_addressable_bank_members<M: CheckpointEncodingMaterializer>(
    members: &[crate::AddressableBankMember],
    source: eredu_checkpoint::store::SharedCheckpointSource,
    rows: &crate::SelectedRowLookups,
    support: &impl crate::RowLookupMechanismSupport,
    native: &M,
) -> Result<PreparedAddressableBankStorage<M::Parameters>, eredu_core::BackendFailure> {
    use eredu_core::BackendFailure;
    let plans = super::plan_addressable_bank_bindings(members, &*source, |_, recipe, store| {
        native.lower_mxfp4(recipe, store)
    })
    .map_err(BackendFailure::from_error)?;
    let catalog =
        AddressableBankCatalog::from_binding_plans(plans).map_err(BackendFailure::from_error)?;
    prepare_selected_addressable_bank(source, catalog, rows, support, native)
}
