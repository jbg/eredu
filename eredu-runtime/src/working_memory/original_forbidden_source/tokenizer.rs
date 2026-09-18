//! Original tokenizer loan -> shared packed worker -> same immutable source copy.
use super::*;
use crate::working_memory::OriginalTokenizer;
use eredu_core::SpeculativeBuffer;
use eredu_text::token_bytes::PackedTokenBytePlan;

impl WorkingMemoryPool {
    /// Exact packed temporary plus independent immutable input destinations.
    /// The plan is a borrow; this query grants neither a source nor a work lease.
    pub fn forbidden_tokenizer_source_required_bytes(
        plan: &PackedTokenBytePlan<'_>,
        trigger_bytes: usize,
    ) -> Result<u64, WorkingMemoryError> {
        let parts = [
            plan.control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            SpeculativeBuffer::<u8>::retained_control_bytes(plan.packed_bytes())
                .ok_or(WorkingMemoryError::Overflow)?,
            ForbiddenControllerInputs::copy_metadata_bytes(plan.packed_bytes(), trigger_bytes)
                .ok_or(WorkingMemoryError::Overflow)?,
            Account::control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<OriginalForbiddenSource>(),
            size_of::<OriginalForbiddenSourceError>(),
            size_of::<Result<OriginalForbiddenSource, OriginalForbiddenSourceError>>(),
            size_of::<Result<ForbiddenControllerInputs, Cause>>(),
            size_of::<Result<ForbiddenControllerInputs, ForbiddenControllerError>>(),
            size_of::<Result<PreparedForbiddenInputCopy<'_>, ForbiddenControllerError>>(),
            size_of::<Result<PackedTokenBytePlan<'_>, eredu_text::token_bytes::TokenByteError>>(),
            size_of::<Result<(), eredu_text::token_bytes::TokenByteError>>(),
            size_of::<Result<SpeculativeBuffer<u8>, eredu_core::SpeculativeBufferAllocationError>>(
            ),
            size_of::<Result<(), eredu_core::generation::GenerationError>>(),
            size_of::<(&WorkingMemoryPool, &OriginalTokenizer, &[u8])>(),
            size_of::<(
                &PackedTokenBytePlan<'_>,
                &[u8],
                &mut Option<SpeculativeBuffer<u8>>,
            )>(),
            size_of::<(PreparedForbiddenInputCopy<'_>, HostPreparationAuthority)>(),
            size_of::<Option<SpeculativeBuffer<u8>>>(),
            size_of::<std::iter::Take<std::iter::Repeat<u8>>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Uses the real tokenizer's lexical byte worker and the same closed source
    /// copy compiler. No TokEnv, matcher, JSON parse or ordinary Vec factory runs.
    /// Temporary packing and copied immutable bytes are both paid while live.
    pub fn compile_forbidden_tokenizer_source(
        &self,
        tokenizer: &OriginalTokenizer,
        trigger: &[u8],
    ) -> Result<OriginalForbiddenSource, OriginalForbiddenSourceError> {
        tokenizer
            .validate_pool(self)
            .map_err(OriginalForbiddenSourceError::refused)?;
        if trigger.is_empty() {
            return Err(OriginalForbiddenSourceError::refused(
                ForbiddenControllerError::Source,
            ));
        }
        let plan = tokenizer
            .token_byte_vocabulary()
            .map_err(OriginalForbiddenSourceError::refused)?;
        let bytes = Self::forbidden_tokenizer_source_required_bytes(&plan, trigger.len())
            .map_err(OriginalForbiddenSourceError::refused)?;
        let account = Account::admit(self, bytes).map_err(OriginalForbiddenSourceError::refused)?;
        let host = HostPreparationAuthority::retain(account.clone());
        let mut packed = None;
        let result = (|| -> Result<ForbiddenControllerInputs, Cause> {
            packed = Some(SpeculativeBuffer::try_new_retained(
                plan.packed_bytes(),
                host.clone(),
            )?);
            let packed = packed.as_mut().expect("constructed packed destination");
            packed
                .try_extend(std::iter::repeat(0).take(plan.packed_bytes()))
                .map_err(|_| ForbiddenControllerError::Source)?;
            plan.write(packed)?;
            let copy = PreparedForbiddenInputCopy::new(
                packed,
                plan.token_count(),
                plan.maximum_token_bytes(),
                trigger,
            )?;
            Ok(copy.copy(host.clone())?)
        })();
        match result {
            Err(cause) => {
                let settlement = account.finish().err();
                Err(OriginalForbiddenSourceError {
                    cause,
                    settlement,
                    completed: None,
                    packed,
                    account: Some(account),
                })
            }
            Ok(inputs) => {
                // The independent copied bytes own their real account. Packing
                // retires before settlement and does not create a scalar refund.
                drop(packed);
                match account.finish() {
                    Ok(()) => Ok(OriginalForbiddenSource { inputs, tokenizer: Some(tokenizer.clone()), account }),
                    Err(cause) => Err(OriginalForbiddenSourceError {
                        cause: cause.into(),
                        settlement: None,
                        completed: Some(inputs),
                        packed: None,
                        account: Some(account),
                    }),
                }
            }
        }
    }
}
