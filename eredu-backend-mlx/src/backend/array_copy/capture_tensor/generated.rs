//! Original-scope retention for the existing fixed reconstruction factory.
//! The caller executes the neutral seven-output driver; this adapter only
//! authenticates its actual inputs and preserves every emitted root in Q.
use super::*;
use eredu_core::capture::GeneratedCaptureSource;
use eredu_nn::{BlockFp8InputReconstructionPlan, GeneratedTensorProgram, GeneratedTensorSourceRole};
use eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody;
use safemlx::{OriginalScopeObserver, PreparedArrayClone};
use std::mem::{size_of, size_of_val};

pub(crate) struct GeneratedCaptureRetention<'a> {
    stream: &'a Stream,
    roots: &'a RefCell<Vec<Array>>,
    custody: &'a OriginalSpeculativeBudgetCustody,
    observer: &'a OriginalScopeObserver,
}
impl<'a> GeneratedCaptureRetention<'a> {
    /// Two required compact operands and the shared driver's seven outputs.
    pub(crate) const ROOTS: usize = 9;

    pub(crate) fn new(
        stream: &'a Stream,
        roots: &'a RefCell<Vec<Array>>,
        custody: &'a OriginalSpeculativeBudgetCustody,
        observer: &'a OriginalScopeObserver,
    ) -> Self {
        Self { stream, roots, custody, observer }
    }
    fn validate(&self, claim: &CaptureTensorClaim<'_, '_>) -> Result<(), CaptureTensorNativeError> {
        claim.validate_model_custody(self.custody)?;
        CaptureCompletion::Original(self.observer).validate_identity()
    }
    pub(crate) fn prepare(
        &self,
        prototype: &Array,
        source: &GeneratedCaptureSource,
        program: GeneratedTensorProgram<'_>,
        construct: bool,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), CaptureTensorNativeError> {
        self.validate(claim)?;
        PreparedCaptureTensor::validate_stream(self.stream)?;
        PreparedCaptureTensor::validate_borrowed_source(prototype, claim.geometry())?;
        // The shared funded policy already checked the exact invocation geometry
        // and charged this claim. Revalidate the borrowed actual native program
        // without regenerating an ordinary full-prompt geometry or any DTO.
        let GeneratedTensorProgram::BlockFp8Input(plan) = program;
        let logical = plan.logical_capture_source()
            .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
        if plan.shape() != prototype.shape()
            || source.source_dtype != Some(eredu_core::checkpoint::TensorDtype::F32)
            || source.creation_bytes != logical.creation_bytes
        {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        if construct {
            CaptureCompletion::Original(self.observer).reserve_roots(self.roots, Self::ROOTS)?;
        }
        Ok(())
    }
    pub(crate) fn source(
        &self,
        prototype: &Array,
        role: GeneratedTensorSourceRole,
        value: &Array,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), CaptureTensorNativeError> {
        let plan = BlockFp8InputReconstructionPlan::new(prototype.shape())
            .map_err(|_| CaptureTensorNativeError::ShapeMismatch)?;
        let (shape, dtype) = match role {
            GeneratedTensorSourceRole::CompactValues => (plan.values_shape(), Dtype::Uint8),
            GeneratedTensorSourceRole::BlockScales => (plan.scales_shape(), Dtype::Float32),
        };
        if value.shape() != shape || value.dtype() != dtype {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        self.retain(value, claim)
    }
    pub(crate) fn retain(
        &self,
        value: &Array,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), CaptureTensorNativeError> {
        // Preflight already authenticated the spent claim before generation.
        // Preserve the newly emitted root before a later account health check
        // can refuse; the prepared clone itself checks the exact active scope.
        let completion = CaptureCompletion::Original(self.observer);
        completion.reserve_roots(self.roots, 1)?;
        let retained = completion.clone_array(value)?;
        {
            let mut roots = self.roots.try_borrow_mut()
                .map_err(|_| CaptureTensorNativeError::CollectorBusy)?;
            if roots.len() == roots.capacity() {
                return Err(WorkingMemoryError::UnknownBound.into());
            }
            roots.push(retained);
        }
        // Prior outputs remain in the original recovery owner if this check or
        // the producer's next operation refuses. No guard escapes to the factory.
        self.validate(claim)
    }
    /// Per accepted selection, including an empty slice or a reused producer.
    pub(crate) fn preparation_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<GeneratedTensorProgram<'_>>(),
            size_of::<BlockFp8InputReconstructionPlan<'_>>(),
            size_of::<eredu_nn::GeneratedTensorSource>(),
            size_of::<CaptureCompletion<'_>>(),
            size_of::<CaptureTensorNativeError>(),
            size_of::<Result<(), CaptureTensorNativeError>>(),
            size_of::<(&Array, &GeneratedCaptureSource, bool, &CaptureTensorClaim<'_, '_>)>(),
            OriginalScopeObserver::control_bytes()?,
            Stream::device_type_control_bytes()?,
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// One actual producer. Its numerical constructors are already in the same
    /// equation trace; these are only the nine retained aliases and validations.
    pub(crate) fn retention_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<BlockFp8InputReconstructionPlan<'_>>(),
            size_of::<GeneratedTensorSourceRole>(),
            size_of::<([i32; 2], Dtype)>(),
            size_of::<(&Array, &CaptureTensorClaim<'_, '_>)>(),
            size_of::<std::cell::RefMut<'_, Vec<Array>>>(),
            size_of::<CaptureCompletion<'_>>(),
            size_of::<CaptureTensorNativeError>(),
            size_of::<Result<(), CaptureTensorNativeError>>(),
            OriginalScopeObserver::control_bytes()?.checked_mul(3)?,
            PreparedArrayClone::control_bytes()?,
            Array::inspection_clone_handle_bytes(),
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)?
            .checked_mul(Self::ROOTS)
    }
}
