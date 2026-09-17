//! Shared controller payload custody independent of a generation run.

use super::{
    WorkingMemoryError, WorkingMemoryFundingScope, WorkingMemoryPool, WorkingMemoryStorage,
};
use eredu_core::{
    ControllerDeclarationData, SharedControllerBytes, SharedControllerDeclaration,
    SharedControllerSource, SharedStorageAttachmentError, SharedStorageIdentity, SharedTokenFilter,
    TextControllerStorage, TextControllerWorkspace, TokenFilter, TokenFilterController,
    TokenSamplingDecision,
};
use std::collections::BTreeMap;

mod contribution;
mod original;
use super::residual::OriginalTokenDomainBinding;
pub use contribution::{
    ControllerWorkspaceContribution, ControllerWorkspaceEstimate, ControllerWorkspaceMetadataError,
};

/// A controller cannot prove the complete lifetime of its numerical payload.
#[derive(Debug, thiserror::Error)]
pub enum ControllerStorageError {
    /// An immutable source factory failed while retaining its actual partial
    /// destination. The construction exclusion retires after that typed cause.
    #[error("controller declaration construction failed: {cause}")]
    Construction {
        /// Original source/constructor cause and any owned partial destination.
        #[source]
        cause: eredu_core::BackendFailure,
        /// Payload-free construction custody, released after the cause.
        authority: eredu_core::HostPreparationAuthority,
    },
    /// Invalid controller filtering mechanism, logical extent or payload bound.
    #[error("controller workspace contract rejected: {0}")]
    Contract(
        #[from]
        #[source]
        eredu_core::TextControllerContractError,
    ),
    /// Invalid request geometry or workspace composition.
    #[error("controller workspace estimate rejected: {0}")]
    Estimate(
        #[from]
        #[source]
        eredu_core::CapabilityError,
    ),
    /// Unknown ownership, mismatched identities or failed physical registration.
    #[error("controller storage rejected: {0}")]
    Storage(
        #[from]
        #[source]
        WorkingMemoryError,
    ),
    /// Per-owner attachment failed before publishing complete accounting custody.
    #[error("controller storage attachment failed: {0}")]
    Attachment(
        #[from]
        #[source]
        SharedStorageAttachmentError<WorkingMemoryError>,
    ),
    /// The ordinary size declaration omits separately retained shared sources.
    #[error(
        "shared controller storage requires {required_bytes} bytes but its additional host allowance is {available_bytes}"
    )]
    UnpricedSharedStorage {
        /// Unique shared numerical storage capacity.
        required_bytes: u64,
        /// Declared allowance beyond the separately emitted final filter.
        available_bytes: u64,
    },
}

/// Sealed controller source evidence. Legacy inventories retain only registered
/// identity metadata. Original-domain mode instead retains the actual closed C
/// source and original decoder binding, with no registry/adoption/credit path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControllerStorageContract {
    shared: BTreeMap<SharedStorageIdentity, SourceEvidence>,
    shared_bytes: u64,
    original: Option<OriginalTokenDomainBinding>,
}

/// Existing-only accounting custody for one exact shared-controller inventory.
///
/// This proof retains identity metadata and registered charges, never the payloads
/// themselves. It grants no allocation permission. Actual shared-owner
/// attachment remains necessary before admitted preparation can escape.
#[derive(Debug, Clone)]
pub struct RegisteredControllerStorage {
    contract: ControllerStorageContract,
    pool: WorkingMemoryPool,
    registration: WorkingMemoryStorage<SharedStorageIdentity>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    Filter,
    Bytes,
    Declaration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceEvidence {
    kind: SourceKind,
    bytes: u64,
}

type Inventory<'a> = BTreeMap<SharedStorageIdentity, (SourceEvidence, SharedControllerSource<'a>)>;

impl RegisteredControllerStorage {
    /// Exact managed domain whose existing source charges are pinned.
    pub fn pool(&self) -> &WorkingMemoryPool {
        &self.pool
    }

    /// Unique source allocation capacity, including spare mask or byte storage.
    pub fn source_bytes(&self) -> u64 {
        self.contract.shared_bytes
    }
}

impl WorkingMemoryPool {
    /// Constructs a shared host filter under unquoted domain ownership and
    /// publishes its exact allocation before releasing that ownership. This
    /// covers loading-time masks even if their model is never used for inference.
    ///
    /// A live reservation rejects before invoking `factory`. Unquoted ownership
    /// excludes concurrent admitted work; it is not an allocation-size bound.
    /// On rejection or unwind, the new payload retires before that ownership.
    pub fn prepare_shared_token_filter(
        &self,
        factory: impl FnOnce() -> TokenFilter,
    ) -> Result<SharedTokenFilter, ControllerStorageError> {
        let ownership = self.acquire_unquoted()?;
        let filter = SharedTokenFilter::new(factory());
        self.attach_prepared_source(SharedControllerSource::Filter(&filter))?;
        drop(ownership);
        Ok(filter)
    }

    /// Constructs immutable controller bytes under unquoted domain ownership,
    /// then attaches their exact allocation capacity before publishing the owner.
    /// A live reservation rejects before the factory runs, including a zero-byte
    /// reservation. Existing aliases share subsequent accounting custody.
    ///
    /// The closed factory must transfer all its numerical payload into the
    /// returned Vec and must not escape other unaccounted owners. On rejection
    /// or unwind, newly constructed payload retires before unquoted ownership.
    pub fn prepare_shared_controller_bytes(
        &self,
        factory: impl FnOnce() -> Vec<u8>,
    ) -> Result<SharedControllerBytes, ControllerStorageError> {
        let ownership = self.acquire_unquoted()?;
        let bytes = SharedControllerBytes::new(factory());
        self.attach_prepared_source(SharedControllerSource::Bytes(&bytes))?;
        drop(ownership);
        Ok(bytes)
    }

    /// Constructs an independently owned immutable declaration under the same
    /// cold exclusion/registration worker as bytes. A live reservation rejects
    /// before the factory; failed partial sources retain exclusion until dropped.
    pub fn prepare_shared_controller_declaration<T: ControllerDeclarationData>(
        &self,
        factory: impl FnOnce() -> Result<T, eredu_core::BackendFailure>,
    ) -> Result<SharedControllerDeclaration, ControllerStorageError> {
        let ownership = self.acquire_unquoted()?;
        let value = match factory() {
            Ok(value) => value,
            Err(cause) => {
                return Err(ControllerStorageError::Construction {
                    cause,
                    authority: eredu_core::HostPreparationAuthority::retain(ownership),
                });
            }
        };
        let declaration = SharedControllerDeclaration::new(value);
        self.attach_prepared_source(SharedControllerSource::Declaration(&declaration))?;
        drop(ownership);
        Ok(declaration)
    }

    fn attach_prepared_source(
        &self,
        source: SharedControllerSource<'_>,
    ) -> Result<(), ControllerStorageError> {
        let bytes = source
            .capacity_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if bytes != 0 {
            let identity = source.identity().clone();
            source.try_attach(&self.0.storage_domain, || {
                let charge = self.register_storage([(identity, bytes)])?;
                Ok::<Box<dyn Send + Sync>, WorkingMemoryError>(Box::new(charge))
            })?;
        }
        Ok(())
    }
}

impl ControllerStorageContract {
    /// Inspects a complete cold lifetime declaration without registering bytes,
    /// advancing a controller or cloning its numerical storage.
    pub fn inspect<C: TokenFilterController>(
        controller: &C,
    ) -> Result<Self, ControllerStorageError> {
        let shared = inventory(controller)?
            .into_iter()
            .map(|(identity, (evidence, _))| (identity, evidence))
            .collect::<BTreeMap<_, _>>();
        let shared_bytes = shared.values().try_fold(0u64, |total, evidence| {
            total
                .checked_add(evidence.bytes)
                .ok_or(WorkingMemoryError::Overflow)
        })?;
        Ok(Self {
            shared,
            shared_bytes,
            original: None,
        })
    }

    /// Ensures shared sources are priced beyond the final emitted filter, whose
    /// independent decision allocation is covered by the sampling quote.
    pub fn validate_workspace(
        &self,
        workspace: TextControllerWorkspace<'_>,
    ) -> Result<(), ControllerStorageError> {
        if let Some(binding) = &self.original {
            return Self::validate_original_workspace(binding, workspace);
        }
        if self.shared_bytes > workspace.additional_host_bytes {
            return Err(ControllerStorageError::UnpricedSharedStorage {
                required_bytes: self.shared_bytes,
                available_bytes: workspace.additional_host_bytes,
            });
        }
        Ok(())
    }

    /// Rechecks completeness and exact identity/capacity before preparation or
    /// a controller callback. Equal values and totals do not identify storage.
    pub fn validate<C: TokenFilterController>(
        &self,
        controller: &C,
    ) -> Result<(), ControllerStorageError> {
        if let Some(binding) = &self.original {
            Self::validate_original_declaration(binding, controller.inference_storage())?;
            let maximum =
                u64::try_from(binding.maximum()).map_err(|_| WorkingMemoryError::Overflow)?;
            let workspace = controller
                .inference_workspace(maximum)
                .ok_or(WorkingMemoryError::UnknownBound)?;
            return Self::validate_original_workspace(binding, workspace);
        }
        if self.shared != Self::inspect(controller)?.shared {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(())
    }

    /// Pins this exact inventory only when every nonzero source is already
    /// registered in the supplied pool at its full capacity. Rejection creates
    /// no partial pin and changes neither known usage nor historical peak.
    ///
    /// Zero-byte source identities still participate in contract validation,
    /// but need no registry entry. A successful pin does not attach custody to
    /// surviving source aliases; [`Self::adopt`] remains required before native
    /// preparation uses the reduced allowance.
    pub fn pin_registered<C: TokenFilterController>(
        &self,
        controller: &C,
        pool: &WorkingMemoryPool,
    ) -> Result<RegisteredControllerStorage, ControllerStorageError> {
        let sources = self.validated_inventory(controller)?;
        let registration = pool.pin_registered_storage(sources.into_iter().filter_map(
            |(identity, (evidence, _))| (evidence.bytes != 0).then_some((identity, evidence.bytes)),
        ))?;
        Ok(RegisteredControllerStorage {
            contract: self.clone(),
            pool: pool.clone(),
            registration,
        })
    }

    /// Transfers already existing shared payloads from the admitted envelope to
    /// their actual owners. Every successful attachment remains until its final
    /// alias retires, including after a later preparation rejection.
    ///
    /// The caller must have validated the full ordinary workspace and acquired
    /// this scope before calling. Certify it only after complete success. A
    /// partial failure must retain conservative funding until independently
    /// accounted storage or safe recovery proves every live payload covered.
    pub fn adopt<C: TokenFilterController>(
        &self,
        controller: &C,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<(), ControllerStorageError> {
        let sources = self.validated_inventory(controller)?;
        let pool = scope.pool();
        for (identity, (evidence, source)) in sources {
            let bytes = evidence.bytes;
            // All and zero-capacity buffers have no numerical allocation. Their
            // identities still participate in later declaration validation.
            if bytes == 0 {
                continue;
            }
            source.try_attach(&pool.0.storage_domain, || {
                let mut registered =
                    scope.adopt_storage_individually([(identity.clone(), bytes)])?;
                let charge = registered
                    .remove(&identity)
                    .expect("single registered controller source");
                // The key contains only identity metadata, never the payload.
                // This closed provider cannot invoke controller callbacks or
                // acquire another shared-owner lock while accounting is held.
                Ok::<Box<dyn Send + Sync>, WorkingMemoryError>(Box::new(charge))
            })?;
        }
        Ok(())
    }

    /// Checks the shared tokenizer owner again after the callback, before native
    /// work. A mixed shared declaration must preserve borrowed tokenizer-owner
    /// provenance; an unlabelled borrow cannot substitute a same-sized mask.
    /// Declared byte/declaration sources neither satisfy that provenance nor require a
    /// tokenizer label when the controller has no shared filter sources. They
    /// require an exact borrowed post-callback source witness, including zero-
    /// capacity owners. Any supplied witness must match the complete inventory.
    pub fn validate_decision(
        &self,
        decision: &TokenSamplingDecision<'_>,
    ) -> Result<(), ControllerStorageError> {
        // Legacy entry cannot authenticate an original domain or silently omit it.
        if self.original.is_some() || decision.original_tokenizer_validity().is_some() {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        match decision.controller_storage() {
            Some(storage) => {
                self.validate_inventory(inventory_from_declaration(storage)?)?;
            }
            None if self
                .shared
                .values()
                .any(|source| source.kind != SourceKind::Filter) =>
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            None => {}
        }
        if let Some(filter) = decision.shared_tokenizer_validity() {
            let bytes = filter
                .capacity_bytes()
                .ok_or(WorkingMemoryError::UnknownBound)?;
            if self.shared.get(filter.identity())
                != Some(&SourceEvidence {
                    kind: SourceKind::Filter,
                    bytes,
                })
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
        } else if self
            .shared
            .values()
            .any(|source| source.kind == SourceKind::Filter)
            && decision.tokenizer_validity().is_some()
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(())
    }

    fn validated_inventory<'a, C: TokenFilterController>(
        &self,
        controller: &'a C,
    ) -> Result<Inventory<'a>, ControllerStorageError> {
        self.validate_inventory(inventory(controller)?)
    }

    fn validate_inventory<'a>(
        &self,
        sources: Inventory<'a>,
    ) -> Result<Inventory<'a>, ControllerStorageError> {
        if self.original.is_some() {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        if sources.len() != self.shared.len()
            || sources
                .iter()
                .any(|(identity, (evidence, _))| self.shared.get(identity) != Some(evidence))
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(sources)
    }
}

fn inventory<C: TokenFilterController>(
    controller: &C,
) -> Result<Inventory<'_>, ControllerStorageError> {
    inventory_from_declaration(controller.inference_storage())
}

fn inventory_from_declaration(
    storage: TextControllerStorage<'_>,
) -> Result<Inventory<'_>, ControllerStorageError> {
    let sources = storage
        .shared_sources()
        .ok_or(WorkingMemoryError::UnknownBound)?;
    let mut inventory = BTreeMap::new();
    for source in sources {
        let evidence = SourceEvidence {
            kind: match source {
                SharedControllerSource::Filter(_) => SourceKind::Filter,
                SharedControllerSource::Bytes(_) => SourceKind::Bytes,
                SharedControllerSource::Declaration(_) => SourceKind::Declaration,
            },
            bytes: source
                .capacity_bytes()
                .ok_or(WorkingMemoryError::UnknownBound)?,
        };
        if let Some((previous, _)) = inventory.insert(source.identity().clone(), (evidence, source))
        {
            if previous != evidence {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
        }
    }
    Ok(inventory)
}

#[cfg(test)]
mod tests;
