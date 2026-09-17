//! Exact original forbidden inputs through the existing immutable source compiler.
mod tokenizer;
use super::{
    WorkingMemoryError, WorkingMemoryPool, loaded_decode_source::Allowance,
    original_declaration_source::Account,
};
use eredu_core::{
    HostPreparationAuthority,
    speculative::{
        ForbiddenControllerError, ForbiddenControllerInputs, ForbiddenControllerSource,
        PreparedForbiddenInputCopy,
    },
};
use std::mem::{size_of, size_of_val};

/// Only this pool's original compiler can construct the source proof. The real
/// immutable byte owner and its account survive all aliases; no raw adoption.
#[derive(Debug, Clone)]
pub struct OriginalForbiddenSource {
    inputs: ForbiddenControllerInputs,
    account: Account,
}
impl OriginalForbiddenSource {
    /// Actual fresh copied vocabulary and trigger, with original custody inside.
    pub fn inputs(&self) -> &ForbiddenControllerInputs {
        &self.inputs
    }
    /// Exact original compiler domain; no source bytes are credited to a request.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        self.account.validate(pool)
    }
    /// Checks the actual input owner, not equal byte contents or a caller token.
    pub fn validate_controller(
        &self,
        source: ForbiddenControllerSource<'_>,
        pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_pool(pool)?;
        if !self.inputs.same_source(source.inputs()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    /// Fixed compiler-account/identity validation frames, before source use.
    pub fn validation_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<&Self>(),
            size_of::<&WorkingMemoryPool>(),
            size_of::<ForbiddenControllerSource<'_>>(),
            size_of::<std::sync::MutexGuard<'_, Allowance>>(),
            size_of::<WorkingMemoryError>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Result<(), eredu_core::SharedStorageAttachmentError<std::convert::Infallible>>>(
            ),
            size_of::<Option<eredu_core::OriginalTokenDomainWitness<'_>>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Copy(#[from] ForbiddenControllerError),
    #[error(transparent)]
    TokenBytes(#[from] eredu_text::token_bytes::TokenByteError),
    #[error(transparent)]
    Buffer(#[from] eredu_core::SpeculativeBufferAllocationError),
}
/// Fixed refusal or failed-copy prefix. Actual input buffers retire before the
/// source compiler's account; failures expose neither partial adoption nor retry.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalForbiddenSourceError {
    #[source]
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    completed: Option<ForbiddenControllerInputs>,
    packed: Option<eredu_core::SpeculativeBuffer<u8>>,
    account: Option<Account>,
}
impl OriginalForbiddenSourceError {
    /// Fixed unavailable-backend or changed source refusal before construction.
    pub fn rejected(cause: WorkingMemoryError) -> Self {
        Self::refused(cause)
    }
    fn refused(cause: impl Into<Cause>) -> Self {
        Self {
            cause: cause.into(),
            settlement: None,
            completed: None,
            packed: None,
            account: None,
        }
    }
}
impl WorkingMemoryPool {
    /// Exact copy destinations and compiler/header controls from the actual plan.
    pub fn forbidden_source_required_bytes(
        plan: PreparedForbiddenInputCopy<'_>,
    ) -> Result<u64, WorkingMemoryError> {
        let parts = [
            plan.required_bytes().ok_or(WorkingMemoryError::Overflow)?,
            Account::control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<OriginalForbiddenSource>(),
            size_of::<OriginalForbiddenSourceError>(),
            size_of::<Result<OriginalForbiddenSource, OriginalForbiddenSourceError>>(),
            size_of::<Result<ForbiddenControllerInputs, ForbiddenControllerError>>(),
            size_of::<(&WorkingMemoryPool, PreparedForbiddenInputCopy<'_>)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Pays the existing original source compiler before any destination birth,
    /// then executes the same packed-input copy worker. No native resource exists.
    pub fn compile_forbidden_source(
        &self,
        plan: PreparedForbiddenInputCopy<'_>,
    ) -> Result<OriginalForbiddenSource, OriginalForbiddenSourceError> {
        let bytes = Self::forbidden_source_required_bytes(plan)
            .map_err(OriginalForbiddenSourceError::refused)?;
        let account = Account::admit(self, bytes).map_err(OriginalForbiddenSourceError::refused)?;
        let host = HostPreparationAuthority::retain(account.clone());
        match plan.copy(host) {
            Err(cause) => {
                let settlement = account.finish().err();
                Err(OriginalForbiddenSourceError {
                    cause: cause.into(),
                    settlement,
                    completed: None,
                    packed: None,
                    account: Some(account),
                })
            }
            Ok(inputs) => match account.finish() {
                Ok(()) => Ok(OriginalForbiddenSource { inputs, account }),
                Err(cause) => Err(OriginalForbiddenSourceError {
                    cause: cause.into(),
                    settlement: None,
                    completed: Some(inputs),
                    packed: None,
                    account: Some(account),
                }),
            },
        }
    }
}
