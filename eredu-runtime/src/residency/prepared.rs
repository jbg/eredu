//! Cold residency preparation shared by host and device storage realizations.

use super::*;
use eredu_core::checkpoint_mechanisms::CheckpointParameterBackend;
use eredu_core::residency::parameter_binding::preflight_bindings;
use eredu_core::residency::{
    conversion_claim::ConversionRetentionError,
    ParameterConversionRetentionEligibility as Eligibility, ParameterConversionRetentionPolicy,
    ParameterConversionRetentionPolicyReport, ResidencyStorage,
};

/// Immutable source identity is retained per physical unit. Transformed modules
/// may use the same source key with different geometry or values.
pub struct ResidencySources {
    primary: eredu_checkpoint::store::SharedCheckpointSource,
    units: BTreeMap<OffloadUnitId, eredu_checkpoint::store::SharedCheckpointSource>,
    row_ranges: Vec<RowResidencyRange>,
}
impl ResidencySources {
    /// Returns the primary source retained for diagnostics.
    pub fn primary(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.primary.as_ref()
    }

    /// Looks up a compact declared catalog by prefix without materializing it.
    pub fn row_range(&self, prefix: &OffloadUnitId) -> Option<&RowResidencyRange> {
        self.row_ranges
            .iter()
            .find(|range| range.range().prefix() == prefix)
    }

    /// Enumerates explicitly selected source overrides.
    pub fn unit_sources(
        &self,
    ) -> impl Iterator<
        Item = (
            &OffloadUnitId,
            &eredu_checkpoint::store::SharedCheckpointSource,
        ),
    > {
        self.units.iter()
    }

    /// Returns the exact source for this explicitly or compactly declared unit.
    pub fn source(&self, unit: &OffloadUnitId) -> &dyn eredu_checkpoint::store::CheckpointSource {
        if let Some(source) = self.units.get(unit) {
            return source.as_ref();
        }
        if let Some(range) = self
            .row_ranges
            .iter()
            .find(|range| range.range().member(unit).is_some())
        {
            return range.source();
        }
        self.primary.as_ref()
    }
}

/// Failure validating the selected residency declarations before native binding.
#[derive(Debug, thiserror::Error)]
pub enum PreparedResidencyError {
    /// Residency geometry or ownership disagrees with the selected declaration.
    #[error(transparent)]
    Controller(#[from] ResidencyControllerError),
    /// A backend cannot materialize a selected owner from its exact source.
    #[error("residency binding preflight failed: {0}")]
    BindingPreflight(String),
}

/// Exact sources, unit declarations and reservation ledger admitted for native storage.
pub struct PreparedResidency<B: CheckpointParameterBackend> {
    sources: ResidencySources,
    control: ResidencyController,
    conversion_policy: ParameterConversionRetentionPolicyReport,
    backend: std::marker::PhantomData<fn() -> B>,
}

impl<B: CheckpointParameterBackend> PreparedResidency<B> {
    /// Validates source roles, alias ownership and backend materialization
    /// support without acquiring a checkpoint lease or allocating native storage.
    pub fn new(
        primary: eredu_checkpoint::store::SharedCheckpointSource,
        units: BTreeMap<OffloadUnitId, eredu_checkpoint::store::SharedCheckpointSource>,
        plan: OffloadPlan,
        definitions: impl IntoIterator<Item = OffloadUnit>,
        row_ranges: Vec<RowResidencyRange>,
    ) -> Result<Self, PreparedResidencyError> {
        let eligibility = if plan.config().device_budget_bytes().is_some() {
            Eligibility::DeviceResidencyLimit
        } else if !row_ranges.is_empty()
            || plan
                .units()
                .iter()
                .any(|unit| unit.tier() == MemoryTier::Disk)
        {
            Eligibility::DiskStreamed
        } else if plan
            .units()
            .iter()
            .any(|unit| unit.tier() == MemoryTier::Host)
        {
            Eligibility::HostLayerwise
        } else {
            Eligibility::Eligible
        };
        let sources = ResidencySources {
            primary,
            units,
            row_ranges: row_ranges.clone(),
        };
        let control = ResidencyController::new_with_row_ranges(
            |id| sources.source(id),
            plan,
            definitions,
            row_ranges,
        )?;
        for id in sources.units.keys() {
            if control.unit(id).is_none() {
                return Err(
                    ResidencyControllerError::UnexpectedUnitDefinition { id: id.clone() }.into(),
                );
            }
        }
        for unit in control.units() {
            // The controller establishes global alias identity. Per-unit
            // capability checks inspect only physical owners.
            let owners = unit
                .bindings()
                .iter()
                .filter(|binding| !binding.is_alias())
                .cloned()
                .collect::<Vec<_>>();
            preflight_bindings::<B>(sources.source(unit.id()), &owners)
                .map_err(|error| PreparedResidencyError::BindingPreflight(error.to_string()))?;
        }
        Ok(Self {
            sources,
            control,
            conversion_policy: ParameterConversionRetentionPolicyReport::resolve(None, eligibility),
            backend: std::marker::PhantomData,
        })
    }

    /// Selects conversion retention before installing a native storage owner.
    pub fn select_conversion_retention(
        mut self,
        requested: Option<ParameterConversionRetentionPolicy>,
        mechanism: Eligibility,
    ) -> Self {
        let eligibility = conversion_retention::apply_mechanism_eligibility(
            self.conversion_policy.eligibility,
            mechanism,
        );
        self.conversion_policy =
            ParameterConversionRetentionPolicyReport::resolve(requested, eligibility);
        self
    }

    /// Builds the portable controller and attaches exact native storage.
    pub fn install<R>(self, context: R::Context) -> Result<R, R::Error>
    where
        R: ResidencyInstallation,
        R::Storage: ResidencyStorage<Backend = B>,
    {
        let budget = conversion_retention::execution_budget(self.conversion_policy)?;
        let (storage, resources) = R::open_storage(context)?;
        let owner = ResidencyOwner::new(
            self.sources,
            self.control.with_conversion_retention(budget),
            storage,
        );
        Ok(R::retain(owner, resources))
    }
}

/// Native storage creation and lifecycle for an installed portable residency ledger.
pub trait ResidencyInstallation: ResidencyAccess<Error: From<ConversionRetentionError>> {
    /// Native context used for checkpoint realization and transfer.
    type Context;
    /// Native resources retained alongside the shared ledger, including health evidence.
    type Resources;

    /// Allocates empty native storage at the already selected placement.
    fn open_storage(
        context: Self::Context,
    ) -> Result<(Self::Storage, Self::Resources), Self::Error>;

    /// Retains native resources and the installed ledger handle.
    fn retain(
        owner: std::sync::Arc<ResidencyOwner<Self::Storage>>,
        resources: Self::Resources,
    ) -> Self;
}
