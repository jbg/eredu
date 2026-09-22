//! Authentication for one originally constructed immutable C token domain.
use super::*;
use crate::working_memory::{
    InferenceExecutionIdentity, OriginalTokenizer, WorkingMemoryFundingRun,
    WorkingMemoryReservation,
};
use eredu_core::{
    BackendFailure, GenerationSequenceBankRejection, GenerationSequencePreparation,
    OriginalSourceWitness, TextControllerContract, TextFilterWorkspace,
};

impl ControllerStorageContract {
    /// Authenticates the actual C domain and aggregate plain header before the
    /// one header take and adaptive admission candidates. Failure uses core's
    /// fixed rejection owner; this creates no registry entry, allocation or grant.
    pub fn inspect_original_sequence<C: TokenFilterController>(
        controller: &C,
        workspace: TextControllerWorkspace<'_>,
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self, BackendFailure> {
        let reject = || GenerationSequenceBankRejection::IdentityMismatch.into_backend_failure();
        let declaration = controller.inference_storage();
        let semantic =
            Self::semantic_binding(declaration, pool, execution).map_err(|_| reject())?;
        let source = match &semantic {
            Some(binding) => binding.tokenizer(),
            None => declaration
                .original_token_domain()
                .and_then(|witness| witness.downcast_ref::<OriginalTokenizer>())
                .ok_or_else(reject)?,
        };
        let original = if let Some(binding) = &semantic {
            let state = claim
                .request()
                .semantic_state()
                .and_then(|owner| owner.prepared_source())
                .and_then(|source| {
                    source.downcast_ref::<crate::working_memory::PreparedSemanticState>()
                })
                .ok_or_else(reject)?;
            let TextControllerStorage::RunOwnedWithPreparedSemantic { source: actual, .. } =
                declaration
            else {
                return Err(reject());
            };
            binding
                .validate_semantic_state(state, actual, claim.request().max_new_tokens())
                .map_err(|_| reject())?;
            OriginalTokenDomainBinding::prepare_semantic(source, claim, pool)
                .map_err(|_| reject())?
        } else {
            OriginalTokenDomainBinding::prepare(source, claim, pool).map_err(|_| reject())?
        };
        Self::validate_original_workspace(&original, workspace).map_err(|_| reject())?;
        Ok(Self {
            shared: BTreeMap::new(),
            shared_bytes: 0,
            original: Some(original),
            semantic,
        })
    }

    /// Authenticates the immutable tokenizer/filter retained by a saved cursor,
    /// or its complete empty shared-source inventory, for a fresh request. The caller retains the saved controller and validates
    /// its source pair separately. This constructor issues no sequence, account,
    /// native role, or input header and never adopts ordinary controller storage.
    pub fn inspect_original_retained<C: TokenFilterController>(
        controller: &C,
        workspace: TextControllerWorkspace<'_>,
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        maximum: u64,
    ) -> Result<Self, BackendFailure> {
        let reject = || GenerationSequenceBankRejection::IdentityMismatch.into_backend_failure();
        let declaration = controller.inference_storage();
        let semantic =
            Self::semantic_binding(declaration, pool, execution).map_err(|_| reject())?;
        let witness = semantic
            .as_ref()
            .map(|binding| OriginalSourceWitness::new(binding.tokenizer()))
            .or_else(|| declaration.original_token_domain());
        let Some(witness) = witness else {
            // An already copied run-owned controller can have no shared source.
            // Match initial original-input admission: unknown/nonempty ordinary
            // inventories cannot enter a retained original request.
            if !declaration
                .shared_sources()
                .is_some_and(|mut sources| sources.next().is_none())
            {
                return Err(reject());
            }
            let contract = Self {
                shared: BTreeMap::new(),
                shared_bytes: 0,
                original: None,
                semantic: None,
            };
            contract
                .validate_workspace(workspace)
                .map_err(|_| reject())?;
            return Ok(contract);
        };
        let source = witness
            .downcast_ref::<OriginalTokenizer>()
            .ok_or_else(reject)?;
        let maximum = usize::try_from(maximum).map_err(|_| reject())?;
        let original = OriginalTokenDomainBinding::prepare_retained(source, pool, maximum)
            .map_err(|_| reject())?;
        Self::validate_original_workspace(&original, workspace).map_err(|_| reject())?;
        Ok(Self {
            shared: BTreeMap::new(),
            shared_bytes: 0,
            original: Some(original),
            semantic,
        })
    }

    fn semantic_binding(
        declaration: TextControllerStorage<'_>,
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
    ) -> Result<Option<PreparedControllerBinding>, WorkingMemoryError> {
        let TextControllerStorage::RunOwnedWithPreparedSemantic { binding, source } = declaration
        else {
            return Ok(None);
        };
        let binding = binding
            .downcast_ref::<PreparedControllerBinding>()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        binding.validate_execution(pool, execution)?;
        binding.validate_source(source)?;
        Ok(Some(binding.clone()))
    }

    /// Whether this sealed contract uses the authenticated original-source path.
    /// The value is diagnostic; only the constructor can install that evidence.
    pub fn has_original_domain(&self) -> bool {
        self.original.is_some()
    }

    fn original_source<'a>(
        binding: &'a OriginalTokenDomainBinding,
        witness: OriginalSourceWitness<'_>,
    ) -> Result<&'a OriginalTokenizer, ControllerStorageError> {
        let supplied = witness
            .downcast_ref::<OriginalTokenizer>()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let source = binding.source();
        if !source.same_source(supplied) || source.generation_domain().is_none() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(source)
    }

    pub(super) fn validate_original_declaration(
        &self,
        binding: &OriginalTokenDomainBinding,
        declaration: TextControllerStorage<'_>,
    ) -> Result<(), ControllerStorageError> {
        if let Some(expected) = &self.semantic {
            let TextControllerStorage::RunOwnedWithPreparedSemantic { binding, source } =
                declaration
            else {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            };
            let actual = binding
                .downcast_ref::<PreparedControllerBinding>()
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            if actual != expected {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            expected.validate_source(source)?;
            return Ok(());
        }
        let witness = declaration
            .original_token_domain()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        Self::original_source(binding, witness)?;
        Ok(())
    }

    pub(super) fn validate_original_workspace(
        binding: &OriginalTokenDomainBinding,
        workspace: TextControllerWorkspace<'_>,
    ) -> Result<(), ControllerStorageError> {
        let TextFilterWorkspace::Exact(actual) = workspace.filter else {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        };
        let expected = binding
            .source()
            .generation_domain()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        // Compare objects, not Vec pointers: empty buffers can share an address.
        if !std::ptr::eq(actual, expected) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(())
    }

    pub(super) fn validate_original_pool(
        &self,
        pool: &MemoryLedger,
    ) -> Result<(), ControllerStorageError> {
        if let Some(binding) = &self.original {
            binding.source().validate_pool(pool)?;
        }
        Ok(())
    }

    /// Rechecks only this original source and the actual accepted account before
    /// core binds the run. No attachment, scope, registration or certification
    /// occurs; the existing sequence bank later checks its genuine run context.
    pub fn prepare_original_source<C: TokenFilterController>(
        &self,
        controller: &C,
        funding: &WorkingMemoryFundingRun,
        reservation: &WorkingMemoryReservation,
    ) -> Result<(), ControllerStorageError> {
        if self.original.is_none() && (!self.shared.is_empty() || self.shared_bytes != 0) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        // The empty inventory route is validation only: there is no shared
        // payload to adopt, pin, or discount. Its copied controller stays owned
        // by the ordinary shared snapshot driver and its actual host account.
        self.validate(controller)?;
        self.validate_original_pool(funding.pool())?;
        funding.validate_reservation(reservation)?;
        Ok(())
    }

    /// Composite post-callback check. Legacy decisions retain the conservative
    /// full validation. Original decisions first prove their actual closed C,
    /// pool, sealed header and exact borrowed mask before owned-only validation.
    pub fn validate_sampling_decision(
        &self,
        contract: &TextControllerContract,
        decision: &TokenSamplingDecision<'_>,
        pool: &MemoryLedger,
    ) -> Result<(), ControllerStorageError> {
        let Some(binding) = &self.original else {
            contract.validate_decision(decision)?;
            return self.validate_decision(decision);
        };
        self.validate_original_pool(pool)?;
        self.validate_original_declaration(
            binding,
            decision
                .controller_storage()
                .ok_or(WorkingMemoryError::IdentityMismatch)?,
        )?;
        let source = Self::original_source(
            binding,
            decision
                .original_tokenizer_validity()
                .ok_or(WorkingMemoryError::IdentityMismatch)?,
        )?;
        let actual = decision
            .tokenizer_validity()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let expected = source
            .generation_domain()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !std::ptr::eq(actual, expected) || decision.shared_tokenizer_validity().is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        contract.validate_owned_decision_payload(decision)?;
        Ok(())
    }
}
