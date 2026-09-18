//! Shared tensor helpers used by model implementations.

use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    Array, Dtype, Stream,
};
use std::cell::RefCell;
mod grouped_storage;
mod grouped_unit_error;
pub(crate) use grouped_unit_error::PreparedGroupedUnitError;
mod validation_storage;
mod original_child;
mod original_role;
pub(crate) use original_role::{TokenValidationParent, TokenValidationRole};
pub(crate) use original_child::PreparedTokenChild;
use grouped_storage::PreparedGroupedOutputs;
pub(crate) use grouped_storage::{GroupedChunkOutputs, GroupedOutputStorage};
pub(crate) use validation_storage::{
    request_control_bytes as token_validation_control_bytes, TokenValidationIngress,
};
use validation_storage::{PreparedTokenValidations, TokenValidationCustody};

struct ActiveTokenValidations {
    observer: Option<safemlx::OriginalScopeObserver>,
    // Supplied only by accepted numerical child entry, never current-scope lookup.
    capture_parent: Option<safemlx::OriginalScopeObserver>,
    remaining: Option<usize>,
    batch: TokenValidationBatch,
    grouped_outputs: PreparedGroupedOutputs,
}

thread_local! {
    static TOKEN_VALIDATION_SCOPE: RefCell<Option<ActiveTokenValidations>> = const {
        RefCell::new(None)
    };
}

/// One lazy device-side index-domain assertion.
pub struct TokenValidation {
    invalid: Array,
    failure: TokenValidationFailure,
}

// Store the semantic bounds alongside the lazy reduction. Valid requests never
// construct diagnostic strings; formatting occurs only after a failed reduction.
#[derive(Clone, Copy, Debug)]
enum TokenValidationFailure {
    Domain {
        cardinality: i32,
        sentinel: Option<i32>,
    },
    Gather {
        minimum: i32,
        extent: i32,
    },
}

impl std::fmt::Display for TokenValidationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain {
                cardinality,
                sentinel,
            } => {
                write!(f, "token ID is outside 0..{cardinality}")?;
                if let Some(sentinel) = sentinel {
                    write!(f, " and sentinel {sentinel}")?;
                }
                Ok(())
            }
            Self::Gather { minimum, extent } => {
                write!(f, "gather index is outside {minimum}..{extent}")
            }
        }
    }
}

impl std::error::Error for TokenValidationFailure {}
#[derive(Debug)]
struct OriginalValidationFailure {
    cause: TokenValidationFailure,
    _custody: TokenValidationCustody,
}
impl std::fmt::Display for OriginalValidationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for OriginalValidationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// Assertions collected while constructing one asynchronous submission.
#[derive(Default)]
pub struct TokenValidationBatch {
    validations: Vec<TokenValidation>,
    // Last: final batch storage retires before its original host custody.
    _original: Option<TokenValidationCustody>,
}

impl TokenValidationBatch {
    /// Whether the submission registered no token-domain assertions.
    pub fn is_empty(&self) -> bool {
        self.validations.is_empty()
    }

    /// Device reductions that must be included in the submission event.
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.validations
            .iter()
            .map(|validation| &validation.invalid)
    }

    /// Checks completed one-bit reductions without submitting or waiting for work.
    pub fn validate_completed(&self) -> Result<(), Exception> {
        for validation in &self.validations {
            let invalid = validation.invalid.evaluated()?;
            if invalid.as_slice::<bool>().first().copied() == Some(true) {
                return Err(Exception::custom(validation.failure.to_string()));
            }
        }
        Ok(())
    }
}

/// RAII collector for token assertions belonging to one semantic submission.
pub struct TokenValidationScope {
    active: bool,
}

impl TokenValidationScope {
    /// Starts one non-nestable ordinary submission scope.
    /// Original submissions require their prepared validation storage.
    pub fn begin() -> Result<Self, Exception> {
        if let Some(original) = safemlx::OriginalScopeObserver::try_current()? {
            return Err(original.capacity_error());
        }
        Self::install(ActiveTokenValidations {
            batch: TokenValidationBatch::default(),
            grouped_outputs: PreparedGroupedOutputs::default(),
            remaining: None,
            observer: None,
            capture_parent: None,
        })
    }

    fn begin_prepared(prepared: PreparedTokenValidations) -> Result<Self, Exception> {
        let observer = safemlx::OriginalScopeObserver::require_current()?;
        let remaining = prepared.0.validations.capacity();
        Self::install(ActiveTokenValidations {
            batch: prepared.0,
            grouped_outputs: prepared.1,
            remaining: Some(remaining),
            observer: Some(observer),
            capture_parent: None,
        })
    }

    fn install(active: ActiveTokenValidations) -> Result<Self, Exception> {
        // Rejected prepared storage and its native aliases leave the TLS loan
        // before destruction. The already active scope is never replaced.
        let rejected = TOKEN_VALIDATION_SCOPE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if let Some(current) = slot.as_ref() {
                let cause = current
                    .observer
                    .as_ref()
                    .map(safemlx::OriginalScopeObserver::capacity_error);
                Some((active, cause))
            } else {
                *slot = Some(active);
                None
            }
        });
        if let Some((active, cause)) = rejected {
            return Err(cause.unwrap_or_else(|| {
                active.observer.as_ref().map_or_else(
                    || Exception::custom("token validation submission scopes cannot be nested"),
                    safemlx::OriginalScopeObserver::capacity_error,
                )
            }));
        }
        Ok(Self { active: true })
    }

    /// Seals the collected lazy assertions for completion ownership.
    pub fn finish(mut self) -> TokenValidationBatch {
        let active = TOKEN_VALIDATION_SCOPE
            .with(|slot| slot.borrow_mut().take())
            .expect("active token validation scope must own a collector");
        self.active = false;
        active.batch
    }
}

impl Drop for TokenValidationScope {
    fn drop(&mut self) {
        if self.active {
            let active = TOKEN_VALIDATION_SCOPE.with(|slot| slot.borrow_mut().take());
            drop(active);
        }
    }
}

// Claim before constructing even the first validation-mask native node. A
// failed graph prefix spends the attempt; it cannot replenish this finite scope.
fn reserve_token_validation() -> Result<(), Exception> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(active) = slot.as_mut() {
            if let Some(remaining) = &mut active.remaining {
                if *remaining == 0 {
                    return Err(active
                        .observer
                        .as_ref()
                        .expect("original validation scope")
                        .capacity_error());
                }
                *remaining -= 1;
            }
        }
        Ok(())
    })
}

fn register_token_validation(validation: TokenValidation) -> Result<(), Exception> {
    // Return a refused or unscoped owner out of the TLS loan before native
    // evaluation or destruction; successful insertion cannot grow original Vecs.
    let pending = TOKEN_VALIDATION_SCOPE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(active) = slot.as_mut() {
            if active.remaining.is_some()
                && active.batch.validations.len() == active.batch.validations.capacity()
            {
                return Err((
                    validation,
                    active
                        .observer
                        .as_ref()
                        .expect("original validation scope")
                        .capacity_error(),
                ));
            }
            active.batch.validations.push(validation);
            Ok(None)
        } else {
            Ok(Some(validation))
        }
    });
    let validation = match pending {
        Ok(None) => return Ok(()),
        Ok(Some(validation)) => validation,
        Err((_validation, cause)) => return Err(cause),
    };
    // Internal synchronous callers preserve immediate assertion semantics.
    let invalid = validation.invalid.evaluated()?;
    if invalid.as_slice::<bool>().first().copied() == Some(true) {
        return Err(Exception::custom(validation.failure.to_string()));
    }
    Ok(())
}

/// Clones the lazy reductions owned by the active semantic submission.
pub(crate) fn active_token_validation_arrays() -> Vec<Array> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|validations| {
                validations
                    .batch
                    .validations
                    .iter()
                    .map(|validation| validation.invalid.clone())
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Ordinary population observation. This does not predict a future original
/// count; original providers derive that from their complete invocation plan.
pub(crate) fn active_token_validation_count() -> Result<usize, safemlx::PrefillRootsCause> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        let loan = slot
            .try_borrow()
            .map_err(|_| safemlx::PrefillRootsCause::RuntimeBusy)?;
        Ok(loan
            .as_ref()
            .map_or(0, |active| active.batch.validations.len()))
    })
}
/// Closed no-hooks fill: only bounded native-array copies while the TLS loan is
/// live. No arbitrary visitor, C-wrapper clone, error String or runtime entry.
pub(crate) fn append_active_token_validation_roots(
    roots: &mut safemlx::PrefillRoots,
) -> Result<(), safemlx::PrefillRootsCause> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        let loan = slot
            .try_borrow()
            .map_err(|_| safemlx::PrefillRootsCause::RuntimeBusy)?;
        if let Some(values) = loan.as_ref() {
            for value in &values.batch.validations {
                roots.append_validation(&value.invalid)?;
            }
        }
        Ok(())
    })
}

/// Fills already paid handle slots for a concrete module completion. No caller
/// callback or borrowed TLS value escapes this helper. The owned aliases remain
/// in the module lease until its existing observed retirement completes.
pub(crate) fn append_active_token_validation_copies(
    clones: &mut [safemlx::PreparedArrayClone],
    roots: &mut Vec<crate::MlxTensor>,
    limit: usize,
    observer: &safemlx::OriginalScopeObserver,
) -> Result<(), Exception> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        let loan = slot.try_borrow().map_err(|_| observer.capacity_error())?;
        let active = loan
            .as_ref()
            .filter(|active| {
                active
                    .observer
                    .as_ref()
                    .is_some_and(|value| value.same_scope(observer))
            })
            .ok_or_else(|| observer.domain_error())?;
        let count = active.batch.validations.len();
        if count > clones.len()
            || roots
                .len()
                .checked_add(count)
                .is_none_or(|total| total > limit || total > roots.capacity())
        {
            return Err(observer.capacity_error());
        }
        for (validation, clone) in active.batch.validations.iter().zip(clones) {
            roots.push(crate::MlxTensor::from(
                clone.fill_in_original_scope(&validation.invalid, observer)?,
            ));
        }
        Ok(())
    })
}

/// Validates reductions after the exact model/state completion boundary.
pub(crate) fn validate_active_token_validations() -> Result<(), Exception> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        let slot = slot.borrow();
        let Some(validations) = slot.as_ref() else {
            return Ok(());
        };
        for validation in &validations.batch.validations {
            let invalid = validation.invalid.evaluated()?;
            if invalid.as_slice::<bool>().first().copied() == Some(true) {
                return Err(Exception::custom(validation.failure.to_string()));
            }
        }
        Ok(())
    })
}

/// Consumes completed assertion receipts from the exact installed role. It
/// never evaluates, waits, formats diagnostics or allocates a numerical buffer.
pub(crate) fn validate_active_original_token_validations(
    observer: &safemlx::OriginalScopeObserver,
) -> Result<(), Exception> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        let slot = slot.try_borrow().map_err(|_| observer.capacity_error())?;
        let active = slot
            .as_ref()
            .filter(|active| {
                active
                    .observer
                    .as_ref()
                    .is_some_and(|value| value.same_scope(observer))
            })
            .ok_or_else(|| observer.domain_error())?;
        let custody = active
            .batch
            ._original
            .as_ref()
            .ok_or_else(|| observer.domain_error())?;
        for validation in &active.batch.validations {
            let completed = validation.invalid.completed_in_original_scope(observer)?;
            let values = completed
                .try_as_slice::<bool>()
                .map_err(|_| observer.invalid_input_error())?;
            let [invalid] = values else {
                return Err(observer.invalid_input_error());
            };
            if *invalid {
                return Err(Exception::from_retained_source(OriginalValidationFailure {
                    cause: validation.failure,
                    _custody: custody.clone(),
                }));
            }
        }
        Ok(())
    })
}

/// Registers a lazy device-side token-domain assertion and normalizes IDs to
/// `int32` for embedding and sequential-decision handoff.
pub fn validate_token_domain(
    tokens: &Array,
    cardinality: i32,
    sentinel: Option<i32>,
    stream: &Stream,
) -> Result<Array, Exception> {
    if cardinality <= 0 {
        return Err(Exception::custom("token domain must be non-empty"));
    }
    if !matches!(tokens.dtype(), Dtype::Int32 | Dtype::Uint32 | Dtype::Int64) {
        return Err(Exception::custom(format!(
            "token IDs must use int32, uint32, or architecture-internal int64 storage, got {:?}",
            tokens.dtype()
        )));
    }
    if tokens.size() == 0 {
        return tokens.as_type::<i32>(stream);
    }
    reserve_token_validation()?;
    let range_tokens = if tokens.dtype() == Dtype::Int64 {
        tokens.clone()
    } else {
        tokens.as_type::<i32>(stream)?
    };
    let ordinary = range_tokens
        .ge(Array::try_from_int(0)?, stream)?
        .logical_and(
            &range_tokens.lt(Array::try_from_int(cardinality)?, stream)?,
            stream,
        )?;
    let valid = match sentinel {
        Some(sentinel) => ordinary.logical_or(
            &range_tokens.eq(Array::try_from_int(sentinel)?, stream)?,
            stream,
        )?,
        None => ordinary,
    };
    let invalid = valid.logical_not(stream)?.any(false, stream)?;
    register_token_validation(TokenValidation {
        invalid,
        failure: TokenValidationFailure::Domain {
            cardinality,
            sentinel,
        },
    })?;
    range_tokens.as_type::<i32>(stream)
}

/// Original grouped kernels cannot synchronously evaluate an index assertion
/// during host construction. Claim the existing prepared batch before making
/// its lazy predicate, and mask invalid indices before a native kernel can
/// read them. Ordinary callers retain their immediate assertion path.
pub(crate) fn original_group_indices(
    indices: &Array,
    cardinality: i32,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    let Some(observer) = safemlx::OriginalScopeObserver::try_current()? else {
        return Ok(None);
    };
    let prepared = TOKEN_VALIDATION_SCOPE.with(|slot| {
        slot.try_borrow()
            .map(|slot| {
                slot.as_ref()
                    .is_some_and(|active| active.remaining.is_some())
            })
            .unwrap_or(false)
    });
    if !prepared
        || cardinality <= 0
        || !matches!(
            indices.dtype(),
            Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
        )
    {
        return Err(observer.capacity_error());
    }
    if indices.size() == 0 {
        return Ok(Some(indices.clone()));
    }
    reserve_token_validation()?;
    // Match the shared token normalization for 32-bit IDs. Preserve a wide
    // comparison for 64-bit IDs; unsigned values above I64::MAX become negative
    // and are rejected, while every valid bank ID fits the positive I32 range.
    let range_indices = if matches!(indices.dtype(), Dtype::Int64 | Dtype::Uint64) {
        indices.as_type::<i64>(stream)?
    } else {
        indices.as_type::<i32>(stream)?
    };
    let valid = range_indices
        .ge(Array::try_from_int(0)?, stream)?
        .logical_and(
            &range_indices.lt(Array::try_from_int(cardinality)?, stream)?,
            stream,
        )?;
    let invalid = valid.logical_not(stream)?.any(false, stream)?;
    // Preserve the pending assertion and its exact native owner even if the
    // later safe-index construction refuses after accepting a graph prefix.
    register_token_validation(TokenValidation {
        invalid,
        failure: TokenValidationFailure::Domain {
            cardinality,
            sentinel: None,
        },
    })?;
    safemlx::ops::r#where(
        &valid,
        &range_indices,
        &safemlx::ops::zeros_like(&range_indices, stream)?,
        stream,
    )
    .map(Some)
}

/// Retains a bounds assertion and returns safe native gather indices while an
/// asynchronous submission has not yet completed its validation reductions.
/// Negative signed indices retain ordinary wrap-from-end semantics.
pub(crate) fn validate_take_indices(
    indexes: &Array,
    extent: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let signed = match indexes.dtype() {
        Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64 => true,
        Dtype::Uint8 | Dtype::Uint16 | Dtype::Uint32 | Dtype::Uint64 => false,
        dtype => {
            return Err(Exception::custom(format!(
                "gather indices must use integer storage, got {dtype:?}"
            )));
        }
    };
    if indexes.size() == 0 {
        return Ok(indexes.clone());
    }
    if extent <= 0 {
        return Err(Exception::custom("cannot gather from an empty axis"));
    }
    reserve_token_validation()?;
    let minimum = if signed { -extent } else { 0 };
    let valid = indexes
        .ge(Array::try_from_int(minimum)?, stream)?
        .logical_and(&indexes.lt(Array::try_from_int(extent)?, stream)?, stream)?;
    register_token_validation(TokenValidation {
        invalid: valid.logical_not(stream)?.any(false, stream)?,
        failure: TokenValidationFailure::Gather { minimum, extent },
    })?;
    safemlx::ops::r#where(
        &valid,
        indexes,
        &safemlx::ops::zeros_like(indexes, stream)?,
        stream,
    )
}

#[allow(non_snake_case)]
/// Builds a causal attention mask with optional window and sequence lengths.
pub fn create_causal_mask(
    N: i32,
    offset: Option<i32>,
    window_size: Option<i32>,
    lengths: Option<Array>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let geometry =
        eredu_nn::operation_geometry::CausalMaskGeometry::new(N, offset.unwrap_or(0), window_size)
            .map_err(|error| Exception::custom(error.to_string()))?;
    let offset = geometry.offset();

    // The macro defaults to F32 even for integer endpoints. Coordinates must
    // remain exact above 2^24, including zero-distance sliding masks.
    let rinds = Array::arange::<_, i32>(None, geometry.keys(), None, stream)?;
    let linds = Array::arange::<_, i32>(offset, geometry.keys(), None, stream)?;
    // Both aranges are contiguous rank-one coordinates. Fixed reshapes add
    // the broadcast axes without constructing a general indexing plan.
    let linds = linds.reshape(&[geometry.sequence(), 1], stream)?;
    let rinds = rinds.reshape(&[1, geometry.keys()], stream)?;

    let mut mask = linds.ge(&rinds, stream)?;
    if let Some(window_size) = window_size {
        // Subtract from nonnegative query positions instead of adding to keys:
        // a valid maximum I32 lookback must not overflow its comparison operand.
        let earliest_key = linds.subtract(Array::try_from_int(window_size)?, stream)?;
        mask = mask.logical_and(&rinds.ge(&earliest_key, stream)?, stream)?;
    }

    if let Some(lengths) = lengths {
        let lengths = lengths.try_index_device((.., NewAxis, NewAxis, NewAxis), stream)?;
        mask = mask.logical_and(&linds.lt(&lengths, stream)?, stream)?;
    }

    Ok(mask)
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod grouped_original_tests;
#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod row_movement_tests;
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
pub(crate) use grouped_original_tests::GroupedOriginalTestPlan;
