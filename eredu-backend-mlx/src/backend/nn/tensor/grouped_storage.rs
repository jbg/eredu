//! Fixed output destinations of the actual packed-bank chunk loop. Original
//! ingress and ordinary workers prepare their exact tables before filling them.
use super::*;
use crate::backend::error::Error;
use eredu_runtime::working_memory::WorkingMemoryError;
use std::{alloc::Layout, mem::size_of};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GroupedOutputStorage {
    pub(crate) calls: usize,
    pub(crate) chunks: usize,
    pub(crate) unit_observers: usize,
    pub(crate) observer_shape_rank: usize,
}
impl GroupedOutputStorage {
    pub(crate) fn merge(self, other: Self) -> Option<Self> {
        Some(Self {
            calls: self.calls.checked_add(other.calls)?,
            chunks: self.chunks.max(other.chunks),
            unit_observers: self.unit_observers.checked_add(other.unit_observers)?,
            observer_shape_rank: self.observer_shape_rank.max(other.observer_shape_rank),
        })
    }
    pub(crate) fn control_bytes(self) -> Option<usize> {
        if self.unit_observers != 0 && self.observer_shape_rank < 2 {
            return None;
        }
        let observers = Layout::array::<Option<PreparedGroupedUnitError>>(self.unit_observers)
            .ok()?
            .size()
            .checked_add(self.unit_observers.checked_mul(
                PreparedGroupedUnitError::control_bytes(self.observer_shape_rank)?,
            )?)?;
        if self.calls == 0 && self.unit_observers == 0 {
            return observers.checked_add(size_of::<PreparedGroupedOutputs>());
        }
        crate::backend::runtime::residency::storage::native_storage::Bank::shared_borrowed_owner_bytes()?;
        let payload = Layout::array::<Array>(self.chunks).ok()?.size();
        let slots = Layout::array::<Option<GroupedChunkOutputs>>(self.calls)
            .ok()?
            .size();
        let per_call = if self.calls == 0 {
            0
        } else {
            payload.checked_add(safemlx::ops::concatenate_axis_control_bytes()?)?
        };
        [
            size_of::<PreparedGroupedOutputs>(),
            size_of::<GroupedChunkOutputs>(),
            size_of::<Result<GroupedChunkOutputs, Exception>>(),
            size_of::<Option<GroupedChunkOutputs>>(),
            size_of::<Result<PreparedGroupedOutputs, Error>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<TokenValidationCustody>(),
            size_of::<std::cell::RefMut<'static, Option<ActiveTokenValidations>>>(),
        ]
        .into_iter()
        .try_fold(
            observers
                .checked_add(slots)?
                .checked_add(self.calls.checked_mul(per_call)?)?,
            usize::checked_add,
        )
    }
}

#[derive(Default)]
pub(super) struct PreparedGroupedOutputs {
    slots: Vec<Option<GroupedChunkOutputs>>,
    next: usize,
    unit_errors: Vec<Option<PreparedGroupedUnitError>>,
    next_unit_error: usize,
    // The directory and all rejected/unused prefixes retire before this custody.
    _custody: Option<TokenValidationCustody>,
}
impl PreparedGroupedOutputs {
    pub(super) fn prepare(
        limits: GroupedOutputStorage,
        custody: TokenValidationCustody,
    ) -> Result<Self, Error> {
        if limits.control_bytes().is_none() {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
        }
        let mut bank = Self {
            _custody: Some(custody),
            ..Self::default()
        };
        bank.unit_errors
            .try_reserve_exact(limits.unit_observers)
            .map_err(reserve_error)?;
        if bank.unit_errors.capacity() != limits.unit_observers {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
                .at_speculative_stage("grouped observer destination capacity"));
        }
        for _ in 0..limits.unit_observers {
            bank.unit_errors.push(Some(PreparedGroupedUnitError::new(
                limits.observer_shape_rank,
                bank._custody.as_ref().expect("prepared custody").clone(),
            )?));
        }
        bank.slots
            .try_reserve_exact(limits.calls)
            .map_err(reserve_error)?;
        if bank.slots.capacity() != limits.calls {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
                .at_speculative_stage("grouped output call capacity"));
        }
        for _ in 0..limits.calls {
            let mut output = GroupedChunkOutputs {
                values: Vec::new(),
                limit: limits.chunks,
                _custody: bank._custody.clone(),
            };
            output
                .values
                .try_reserve_exact(limits.chunks)
                .map_err(reserve_error)?;
            if output.values.capacity() != limits.chunks {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
                    .at_speculative_stage("grouped output value capacity"));
            }
            bank.slots.push(Some(output));
        }
        Ok(bank)
    }
    pub(super) fn take_unit_error(&mut self) -> Option<PreparedGroupedUnitError> {
        let value = self.unit_errors.get_mut(self.next_unit_error)?.take()?;
        self.next_unit_error += 1;
        Some(value)
    }
    pub(super) fn take(&mut self, chunks: usize) -> Option<GroupedChunkOutputs> {
        let slot = self.slots.get_mut(self.next)?;
        if chunks > slot.as_ref()?.limit {
            return None;
        }
        let mut output = slot.take()?;
        self.next += 1;
        output.limit = chunks;
        Some(output)
    }
}
fn reserve_error(error: std::collections::TryReserveError) -> Error {
    Error::PrefillControl(WorkingMemoryError::ControlStorageReserve(error))
}

pub(crate) struct GroupedChunkOutputs {
    pub(super) values: Vec<Array>,
    limit: usize,
    // Remains attached when the destination leaves TLS, including partial fill
    // failure and the temporary C concatenation header's entire lifetime.
    _custody: Option<TokenValidationCustody>,
}

#[derive(Debug, thiserror::Error)]
enum OrdinaryGroupedOutputCause {
    #[error("grouped output table allocation failed")]
    Reserve(#[source] std::collections::TryReserveError),
    #[error("grouped output table differs from its quoted capacity")]
    Capacity,
    #[error("grouped output table exhausted")]
    Exhausted,
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct OrdinaryGroupedOutputFailure {
    #[source]
    cause: OrdinaryGroupedOutputCause,
    _host: Option<eredu_core::HostPreparationAuthority>,
}

fn ordinary_failure(
    cause: OrdinaryGroupedOutputCause,
    host: Option<eredu_core::HostPreparationAuthority>,
) -> Exception {
    Exception::from_retained_source(OrdinaryGroupedOutputFailure { cause, _host: host })
}

impl GroupedChunkOutputs {
    /// The ordinary chunk worker reserves this exact table before its first
    /// output. Its metadata and any returned refusal keep the admitted host
    /// owner; the enclosing submission keeps native completion custody.
    pub(crate) fn ordinary_control_bytes(chunks: usize) -> Option<usize> {
        let parts = [
            Layout::array::<Array>(chunks).ok()?.size(),
            Exception::retained_source_control_bytes::<OrdinaryGroupedOutputFailure>()?,
            size_of::<Self>(),
            size_of::<Vec<Array>>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<Option<eredu_core::HostPreparationAuthority>>(),
            size_of::<Option<crate::backend::nn::shared::OrdinaryExecutionOwner>>(),
            size_of::<Result<Self, Exception>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    pub(crate) fn prepare(chunks: usize) -> Result<Self, Exception> {
        let Some(observer) = safemlx::OriginalScopeObserver::try_current()? else {
            let host = crate::backend::nn::shared::current_ordinary_execution_owner()?
                .map(|owner| owner.host().clone());
            let mut values = Vec::new();
            values.try_reserve_exact(chunks).map_err(|cause| {
                ordinary_failure(OrdinaryGroupedOutputCause::Reserve(cause), host.clone())
            })?;
            if values.capacity() != chunks {
                return Err(ordinary_failure(OrdinaryGroupedOutputCause::Capacity, host));
            }
            return Ok(Self {
                values,
                limit: chunks,
                _custody: host.map(TokenValidationCustody::Ordinary),
            });
        };
        TOKEN_VALIDATION_SCOPE.with(|slot| {
            let mut slot = slot
                .try_borrow_mut()
                .map_err(|_| observer.capacity_error())?;
            slot.as_mut()
                .filter(|active| active.remaining.is_some())
                .and_then(|active| active.grouped_outputs.take(chunks))
                .ok_or_else(|| observer.capacity_error())
        })
    }
    pub(crate) fn push(&mut self, value: Array) -> Result<(), Exception> {
        if self.values.len() >= self.limit || self.values.len() == self.values.capacity() {
            return Err(match &self._custody {
                Some(TokenValidationCustody::Ordinary(host)) => {
                    ordinary_failure(OrdinaryGroupedOutputCause::Exhausted, Some(host.clone()))
                }
                Some(_) => safemlx::OriginalScopeObserver::require_current()?.capacity_error(),
                None => ordinary_failure(OrdinaryGroupedOutputCause::Exhausted, None),
            });
        }
        self.values.push(value);
        Ok(())
    }
    pub(crate) fn as_slice(&self) -> &[Array] {
        &self.values
    }
}

#[cfg(all(test, target_vendor = "apple", not(feature = "cuda")))]
#[path = "sliding_attention_tests.rs"]
mod sliding_attention_tests;
