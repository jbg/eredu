//! Actual physical chunk source for the existing selected-operation runner.
use super::*;
use eredu_runtime::working_memory::{
    CapturePrefillFragmentClaim, CapturePrefillFragmentTransfer, CaptureSourceSegment,
};

/// Borrows the actual source and exact immutable fragment used by its short
/// target claim. This leaf grants no native work, quota or completion authority.
/// The caller retains the original work, source and intermediates through errors.
pub(crate) struct PreparedCaptureFragment<'s, 'f, 'p, 'a> {
    source: &'s Array,
    observed: ArrayMetadataSnapshot,
    fragment: &'f CapturePrefillFragment<'p, 'a>,
    program: Selection,
}

impl<'s, 'f, 'p, 'a> PreparedCaptureFragment<'s, 'f, 'p, 'a> {
    pub(crate) fn new(
        source: &'s Array,
        fragment: &'f CapturePrefillFragment<'p, 'a>,
    ) -> Result<Self, CaptureTensorNativeError> {
        let observed = Self::validate_source_geometry(source, fragment)?;
        if observed.allocation().is_none() {
            return Err(CaptureTensorNativeError::UnsettledSource);
        }
        let mut program = Selection::from_fragment(fragment)?;
        // Only this actual-source constructor can specialize erased precision.
        // Empty output retains the same selection/Preview operations and has no
        // F32 read or conversion, as in the existing whole-value leaf.
        program.cast_f32 = fragment.output_elements() != 0 && observed.dtype() != Dtype::Float32;
        Ok(Self {
            source,
            observed,
            fragment,
            program,
        })
    }

    /// Borrowed pre-quota descriptor check; does not require, create or settle
    /// an allocation. The enclosing work later evaluates and publishes the real
    /// source before constructing this prepared leaf.
    pub(crate) fn validate_borrowed_source(
        source: &Array,
        fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<Dtype, CaptureTensorNativeError> {
        PreparedCaptureTensor::validate_mechanism()?;
        let dtype = source.dtype();
        PreparedCaptureTensor::validate_source_axes(
            source.shape(),
            dtype,
            fragment.source_shape(),
        )?;
        Ok(dtype)
    }

    pub(crate) fn validate_source_geometry(
        source: &Array,
        fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<ArrayMetadataSnapshot, CaptureTensorNativeError> {
        PreparedCaptureTensor::validate_mechanism()?;
        if source.shape().len() > 32 {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        let observed = source.try_metadata_snapshot()?;
        PreparedCaptureTensor::validate_source_axes(
            observed.shape(),
            observed.dtype(),
            fragment.source_shape(),
        )?;
        Ok(observed)
    }

    #[cfg(test)]
    pub(super) fn program(&self) -> &Selection {
        &self.program
    }

    pub(crate) fn recovery_descriptors(&self) -> usize {
        2 + usize::from(self.program.preview.is_some()) * 2 + usize::from(self.program.cast_f32)
    }

    fn validate(&self) -> Result<(), CaptureTensorNativeError> {
        if self.source.try_metadata_snapshot()? != self.observed {
            return Err(CaptureTensorNativeError::SourceChanged);
        }
        Ok(())
    }

    /// Actual source metadata on the existing span; never resets its ledger.
    pub(crate) fn trace(
        &self,
        projection: &mut ExistingArrayProjection<'s>,
    ) -> Result<WorkspaceTensor, CaptureTensorNativeError> {
        self.validate()?;
        let source = projection.project(self.source)?;
        self.program.trace_within(&source, projection.context())
    }

    /// Bind this exact claim and canonical chunk segment before selecting or
    /// reading native values. Source/intermediate retention remains with the
    /// original work; finishing this borrow completes one host fragment only.
    pub(crate) fn transfer<'t>(
        self,
        claim: CapturePrefillFragmentClaim<'t, 'f, 'p, 'a>,
        native: &mut WorkingMemoryFundingScope,
        segment: &mut CaptureSourceSegment,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<(), CaptureTensorNativeError> {
        self.transfer_with_completion(
            claim,
            native,
            segment,
            stream,
            roots,
            CaptureCompletion::Ordinary,
        )
    }
    pub(crate) fn transfer_with_completion<'t>(
        self,
        claim: CapturePrefillFragmentClaim<'t, 'f, 'p, 'a>,
        native: &mut WorkingMemoryFundingScope,
        segment: &mut CaptureSourceSegment,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        completion: CaptureCompletion<'_>,
    ) -> Result<(), CaptureTensorNativeError> {
        if !std::ptr::eq(self.fragment, claim.fragment()) {
            return Err(CaptureTensorNativeError::ClaimMismatch);
        }
        segment.validate_prefill_fragment(self.fragment)?;
        segment.validate_native_scope(native)?;
        PreparedCaptureTensor::validate_stream(stream)?;
        self.validate()?;
        let source_pin = PreparedCaptureTensor::prepare_registered_source_with_completion(
            &self.observed,
            native.pool(),
            roots,
            self.recovery_descriptors(),
            completion,
        )?;
        let mut destination = claim.prepare_with_segment_source(native, segment, source_pin)?;
        execute_selected(
            self.source,
            &self.program,
            &mut destination,
            stream,
            roots,
            completion,
        )?;
        destination
            .finish()
            .map_err(CaptureTensorNativeError::Claim)
    }
}

impl TransferDestination for CapturePrefillFragmentTransfer<'_, '_, '_, '_, '_, StorageIdentity> {
    type Error = CaptureRunHostError;
    fn validate(&self) -> Result<(), Self::Error> {
        CapturePrefillFragmentTransfer::validate(self)
    }
    fn push_f32(&mut self, value: f32) -> Result<(), Self::Error> {
        CapturePrefillFragmentTransfer::push_f32(self, value)
    }
}
