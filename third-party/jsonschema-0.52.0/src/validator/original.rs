//! Original entry into the same compiled validation graph and body dispatch.
use super::{
    workspace::{Error, Funding},
    Validate, ValidationContext, Validator,
};
use crate::Json;
use std::mem::{size_of, size_of_val};

/// Explicit borrowed-input contract for original validation.
///
/// Primitive access, object lookup/iteration, array iteration, identities and
/// integer tests must borrow existing storage without allocating. The declared
/// controls include their concrete nested call frames. Cold conversions such as
/// `to_value`, numeric decimal formatting and default uniqueness/equality are
/// not thereby qualified: their validator bodies must separately declare and
/// fund those operations. This contract grants no source-construction authority.
pub trait OriginalJson: Json {
    /// Exact fixed input-access control population for one validator dispatch.
    fn original_input_controls() -> Option<usize>;
    /// Exact controls for default string-buffer construction and each string-node
    /// invocation. Returning None leaves these operations unqualified before
    /// either is called. The canonical representation hook separately reserves
    /// any string backing before the body invokes its child validator.
    fn original_string_controls() -> Option<(usize, usize)> {
        None
    }
    /// Actual retained allocation of one cold prepared property key.
    fn original_key_bytes(key: &Self::PreparedKey) -> Option<usize>;
    /// Actual additional backing of a cold schema number. Return None when
    /// the selected serde number representation has no source inspector.
    fn original_number_bytes(number: &serde_json::Number) -> Option<usize>;
}

/// A cold immutable compiled source plus its initialized hash seed.
///
/// Construction belongs to source preparation before admission. The caller must
/// retain and account for this compiled source separately from invocation scratch.
/// Cloning shares the same compiled graph and copies only the fixed hash seed.
/// Each invocation still acquires a fresh explicit funding loan.
pub struct OriginalValidationSource<F: OriginalJson> {
    validator: Validator<F>,
    seed: ahash::RandomState,
}
impl<F: OriginalJson> Clone for OriginalValidationSource<F> {
    fn clone(&self) -> Self {
        Self {
            validator: self.validator.clone(),
            seed: self.seed.clone(),
        }
    }
}
impl<F: OriginalJson> std::fmt::Debug for OriginalValidationSource<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalValidationSource")
            .finish_non_exhaustive()
    }
}
impl<F: OriginalJson> OriginalValidationSource<F> {
    /// Retain an already compiled graph and initialize its cold hash source.
    /// This uses the compiled graph's retained source policy, not an invocation loan.
    ///
    /// # Errors
    /// Returns the original storage or host-entropy failure.
    pub fn new(validator: Validator<F>) -> Result<Self, crate::CompilationError> {
        let funding = &validator.funding;
        funding.reserve(size_of::<(Self, Result<Self, crate::CompilationError>)>())?;
        let seed = funding.random_state()?;
        Ok(Self { validator, seed })
    }
    /// Inspect only the fixed entry frames. Source backing and the borrowed
    /// input owner remain separate obligations of the caller.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<ValidationContext<'_>>(),
            size_of::<F::Node<'_>>(),
            size_of::<Option<Error>>(),
            size_of::<Result<bool, Error>>(),
            size_of::<bool>(),
            size_of::<(&Self, F::Node<'_>, &mut dyn Funding)>(),
            F::original_input_controls()?,
            size_of::<Option<(usize, usize)>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Inspect actual immutable backing requests of this compiled source.
    /// This is a cold, allocating graph traversal, never an invocation worker.
    /// The graph's retained source funds actual inspection scratch. The caller
    /// must keep the graph immutable after storing this requirement.
    ///
    /// # Errors
    /// Returns the original scratch refusal or an unqualified retained source.
    pub fn retained_bytes(&self) -> Result<usize, crate::CompilationError> {
        let funding = &self.validator.funding;
        funding.reserve(size_of::<(
            super::source::Inspector<F>,
            Result<usize, crate::CompilationError>,
            &Self,
        )>())?;
        let mut source = super::source::Inspector::<F>::new(
            F::original_key_bytes,
            F::original_number_bytes,
            funding.clone(),
        );
        let result = source.node(&self.validator.root);
        if let Some(error) = source.take_failure() {
            return Err(error);
        }
        result.map_err(|error| funding.workspace_error(error))?;
        Ok(source.bytes())
    }
    /// Use the ordinary body dispatch with fresh, fallible scratch funding.
    /// Its first reached refusal wins over any Boolean branch result. Scratch
    /// is destroyed before return; caller-owned source/input/payer stay borrowed.
    pub fn is_valid(&self, input: F::Node<'_>, funding: &mut dyn Funding) -> Result<bool, Error> {
        let controls = Self::control_bytes().ok_or(Error::Overflow)?;
        if !funding.reserve(controls) {
            return Err(Error::Funding);
        }
        let mut context = ValidationContext::with_workspace(funding, &self.seed);
        context
            .workspace
            .set_input_controls(F::original_input_controls().ok_or(Error::Overflow)?);
        context
            .workspace
            .set_string_controls(F::original_string_controls());
        let valid = self.validator.root.is_valid(&input, &mut context);
        match context.workspace_failure() {
            Some(cause) => Err(cause),
            None => Ok(valid),
        }
    }
}

impl OriginalJson for crate::SerdeJson {
    fn original_input_controls() -> Option<usize> {
        use crate::{Json, Object};
        use serde_json::{Map, Number, Value};
        type Members<'a> = <&'a Map<String, Value> as Object<'a, crate::SerdeJson>>::MembersIter;
        let parts = [
            size_of::<<Self as Json>::Node<'_>>(),
            size_of::<Members<'_>>(),
            size_of::<std::slice::Iter<'_, Value>>(),
            size_of::<&Map<String, Value>>(),
            size_of::<&[Value]>(),
            size_of::<&Number>(),
            size_of::<Option<&Number>>(),
            size_of::<Option<(&str, &Value)>>(),
            size_of::<Option<&Value>>(),
            size_of::<Option<std::borrow::Cow<'_, str>>>(),
            size_of::<Option<crate::NodeIdentity>>(),
            size_of::<(&Value, &str)>(),
            size_of::<(&Map<String, Value>, &String)>(),
            size_of::<(usize, usize, bool)>(),
            size_of::<std::cmp::Ordering>(),
            size_of::<Option<i64>>(),
            size_of::<Option<u64>>(),
            size_of::<Option<f64>>(),
            size_of::<std::str::Chars<'_>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn original_string_controls() -> Option<(usize, usize)> {
        Some((
            size_of::<serde_json::Value>(),
            size_of::<(&mut serde_json::Value, &str, usize)>(),
        ))
    }
    fn original_key_bytes(key: &String) -> Option<usize> {
        Some(key.capacity())
    }
    fn original_number_bytes(number: &serde_json::Number) -> Option<usize> {
        Some(number.allocation_size())
    }
}
