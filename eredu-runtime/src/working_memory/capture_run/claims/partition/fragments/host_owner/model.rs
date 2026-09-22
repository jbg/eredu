//! Move-only fragment source consumed by an admitted model capture frame.
use super::*;
use crate::working_memory::OriginalSpeculativeBudgetCustody;

/// Owned cold fragment quotation. Its receipt retains exact source geometry;
/// it grants no Host allocation, native work, or communication permission.
#[derive(Debug)]
pub struct OwnedPartitionFragmentHostPlan {
    prototype: PartitionCaptureReceiptPlan,
    prefill: Option<eredu_core::InferenceGeometry>,
    bytes: u64,
}
impl OwnedPartitionFragmentHostPlan {
    /// Retain the immutable receipt and the same table/fragment quotation used
    /// by ordinary capture. Moving this token does not allocate destinations.
    pub fn prepare(
        prototype: PartitionCaptureReceiptPlan,
        prefill: Option<eredu_core::InferenceGeometry>,
    ) -> Result<Self, PartitionFragmentHostPreparationError> {
        let result = match prefill {
            Some(geometry) => PartitionFragmentHostPlan::prepare_prefill(&prototype, geometry),
            None => PartitionFragmentHostPlan::prepare(&prototype),
        }
        .map(|plan| plan.initialization_peak_bytes());
        match result {
            Ok(bytes) => Ok(Self {
                prototype,
                prefill,
                bytes,
            }),
            Err(cause) => Err(PartitionFragmentHostPreparationError {
                cause: cause.into(),
                _prototype: prototype,
                _model_custody: None,
            }),
        }
    }
    /// Actual table, fragment, scratch and construction capacities retained by
    /// this quotation. The enclosing frame admits them once before consumption.
    pub const fn initialization_peak_bytes(&self) -> u64 {
        self.bytes
    }
    /// Borrow the exact immutable receipt retained by this consumed host plan.
    /// Descriptive inspection grants no allocation or execution authority.
    pub fn receipt(&self) -> &PartitionCaptureReceiptPlan {
        &self.prototype
    }
    /// Checked receipt payload population used by the same transport worker.
    /// The owning frame must separately quote reached protocol occurrences.
    pub fn maximum_payload_words(&self) -> Result<usize, CaptureError> {
        self.prototype.maximum_payload_words()
    }
    /// Source-qualified callback population of this same receipt exchange.
    pub fn transport_demands(
        &self,
    ) -> Result<
        impl Iterator<Item = crate::capture::partition::PartitionCaptureTransportDemand> + '_,
        CaptureError,
    > {
        self.prototype.transport_demands()
    }
    /// Exact receipt stages from the same original exchange source.
    pub fn protocol_frames(
        &self,
    ) -> Result<
        std::array::IntoIter<(crate::capture::partition::PartitionCaptureFrameKind, usize), 3>,
        CaptureError,
    > {
        self.prototype.protocol_frames()
    }
    /// Authenticate the immutable source and logical receipt coordinates against
    /// the frame's actual physical invocation and optional logical row window.
    pub fn matches_invocation(
        &self,
        source: &SharedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        physical: CaptureInvocationShape,
        window: Option<CaptureInvocationWindow>,
    ) -> bool {
        let logical = match window {
            Some(window) => match window.validate(physical) {
                Ok(shape) => shape,
                Err(_) => return false,
            },
            None => physical,
        };
        let context = self.prototype.context();
        self.prefill.is_none()
            && self.prototype.shared_plan_source().same_storage(source)
            && context.phase == phase
            && context.prediction == prediction
            && context.matches_invocation(Some(physical), window)
            && source
                .admission()
                .geometry_at(phase, prediction, Some(physical))
                .is_ok()
            && source
                .admission()
                .geometry_at(phase, prediction, Some(logical))
                .is_ok()
    }
    /// Only the neutral consumed frame-plan constructor can lend this account.
    /// The shared allocation worker creates the same finite destination slots.
    pub(in crate::working_memory) fn construct_model(
        self,
        custody: OriginalSpeculativeBudgetCustody,
    ) -> Result<PreparedPartitionFragmentHostFunding, PartitionFragmentHostPreparationError> {
        let result = (|| -> Result<FragmentHostStorage, PreparationCause> {
            let plan = match self.prefill {
                Some(geometry) => {
                    PartitionFragmentHostPlan::prepare_prefill(&self.prototype, geometry)
                }
                None => PartitionFragmentHostPlan::prepare(&self.prototype),
            }?;
            if plan.initialization_peak_bytes() != self.bytes {
                return Err(CaptureRunHostError::ReceiptMismatch.into());
            }
            Ok(construct_fragment_host_storage(
                &plan,
                FragmentCustodySource::Model(&custody),
            )?)
        })();
        match result {
            Ok(storage) => Ok(PreparedPartitionFragmentHostFunding {
                prototype: self.prototype,
                storage,
                bytes: self.bytes,
            }),
            Err(cause) => Err(PartitionFragmentHostPreparationError {
                cause,
                _prototype: self.prototype,
                _model_custody: Some(custody),
            }),
        }
    }
}
