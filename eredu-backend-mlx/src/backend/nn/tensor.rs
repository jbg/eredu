//! Shared tensor helpers used by model implementations.

use safemlx::{
    arange,
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    Array, Dtype, Stream,
};
use std::cell::RefCell;

thread_local! {
    static TOKEN_VALIDATION_SCOPE: RefCell<Option<Vec<TokenValidation>>> = const {
        RefCell::new(None)
    };
}

/// One lazy device-side token-domain assertion.
pub struct TokenValidation {
    invalid: Array,
    message: String,
}

/// Assertions collected while constructing one asynchronous submission.
#[derive(Default)]
pub struct TokenValidationBatch {
    validations: Vec<TokenValidation>,
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
                return Err(Exception::custom(validation.message.clone()));
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
    /// Starts one non-nestable submission scope.
    pub fn begin() -> Result<Self, Exception> {
        TOKEN_VALIDATION_SCOPE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err(Exception::custom(
                    "token validation submission scopes cannot be nested",
                ));
            }
            *slot = Some(Vec::new());
            Ok(Self { active: true })
        })
    }

    /// Seals the collected lazy assertions for completion ownership.
    pub fn finish(mut self) -> TokenValidationBatch {
        let validations = TOKEN_VALIDATION_SCOPE.with(|slot| {
            slot.borrow_mut()
                .take()
                .expect("active token validation scope must own a collector")
        });
        self.active = false;
        TokenValidationBatch { validations }
    }
}

impl Drop for TokenValidationScope {
    fn drop(&mut self) {
        if self.active {
            TOKEN_VALIDATION_SCOPE.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
}

fn register_token_validation(validation: TokenValidation) -> Result<(), Exception> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(validations) = slot.as_mut() {
            validations.push(validation);
            return Ok(());
        }

        // Internal synchronous composition and conformance entry points do not
        // create a public asynchronous submission. Complete their assertion at
        // the point of use; public sessions retain the lazy reduction in the
        // explicit scope above and transfer it to their completion object.
        let invalid = validation.invalid.evaluated()?;
        if invalid.as_slice::<bool>().first().copied() == Some(true) {
            return Err(Exception::custom(validation.message));
        }
        Ok(())
    })
}

/// Clones the lazy reductions owned by the active semantic submission.
pub(crate) fn active_token_validation_arrays() -> Vec<Array> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|validations| {
                validations
                    .iter()
                    .map(|validation| validation.invalid.clone())
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Validates reductions after the exact model/state completion boundary.
pub(crate) fn validate_active_token_validations() -> Result<(), Exception> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        let slot = slot.borrow();
        let Some(validations) = slot.as_ref() else {
            return Ok(());
        };
        for validation in validations {
            let invalid = validation.invalid.evaluated()?;
            if invalid.as_slice::<bool>().first().copied() == Some(true) {
                return Err(Exception::custom(validation.message.clone()));
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
    let range_tokens = if tokens.dtype() == Dtype::Int64 {
        tokens.clone()
    } else {
        tokens.as_type::<i32>(stream)?
    };
    let ordinary = range_tokens.ge(Array::from_int(0), stream)?.logical_and(
        &range_tokens.lt(Array::from_int(cardinality), stream)?,
        stream,
    )?;
    let valid = match sentinel {
        Some(sentinel) => {
            ordinary.logical_or(&range_tokens.eq(Array::from_int(sentinel), stream)?, stream)?
        }
        None => ordinary,
    };
    let invalid = valid.logical_not(stream)?.any(false, stream)?;
    register_token_validation(TokenValidation {
        invalid,
        message: format!(
            "token ID is outside 0..{cardinality}{}",
            sentinel.map_or_else(String::new, |sentinel| format!(" and sentinel {sentinel}"))
        ),
    })?;
    range_tokens.as_type::<i32>(stream)
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
    let offset = offset.unwrap_or(0);

    let rinds = arange!(stop = offset + N, stream = stream)?;
    let linds = arange!(start = offset, stop = offset + N, stream = stream)?;
    let linds = linds.try_index_device((.., NewAxis), stream)?;
    let rinds = rinds.try_index_device(NewAxis, stream)?;

    let mut mask = linds.ge(&rinds, stream)?;
    if let Some(window_size) = window_size {
        let rinds_window = rinds.add(Array::from_int(window_size), stream)?;
        mask = mask.logical_and(&linds.le(&rinds_window, stream)?, stream)?;
    }

    if let Some(lengths) = lengths {
        let lengths = lengths.try_index_device((.., NewAxis, NewAxis, NewAxis), stream)?;
        mask = mask.logical_and(&linds.lt(&lengths, stream)?, stream)?;
    }

    Ok(mask)
}
