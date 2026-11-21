//! Admission and retained observer setup for portable text capture.

use eredu_core::communication::CommunicationSessionIdentity;
use std::{cell::OnceCell, sync::Arc};

use eredu_core::{
    capture::*,
    intervention::{
        AdmittedInterventionPlan, InterventionBackend, InterventionDiscovery, InterventionEstimator,
    },
    BackendFailure, Completion,
};

use crate::{
    capture::{
        partition::{
            LoadedPartitionCapture, PartitionCaptureCollectorFacts, PartitionCaptureHookTransport,
            PartitionCaptureIdentity, PartitionCaptureObserver, PartitionCaptureObserverError,
            PartitionCaptureReceiptLimits,
        },
        CaptureExecutionError, CaptureSession,
    },
    executable_binding::{
        discovery::LoadedModelDiscovery,
        retained_session::{capture_media_shape, PreparedInputPartPlan},
    },
};

use super::{TextSessionDriver, TextSessionResources};
use crate::prefill::{ErasedPrefillCursor, PrefillCursorError};

pub(super) struct TextCaptureCatalog {
    pub(super) discovery: Option<Arc<dyn LoadedModelDiscovery>>,
    partition: OnceCell<Result<Arc<LoadedPartitionCapture>, CaptureError>>,
}

impl TextCaptureCatalog {
    pub(super) fn new(discovery: Option<Arc<dyn LoadedModelDiscovery>>) -> Self {
        Self {
            discovery,
            partition: OnceCell::new(),
        }
    }

    fn require_discovery(&self) -> Result<&dyn LoadedModelDiscovery, CaptureError> {
        self.discovery.as_deref().ok_or_else(|| {
            CaptureError::Unsupported("session has no retained capture catalog".into())
        })
    }

    fn bind(
        &self,
        facts: &dyn PartitionCaptureCollectorFacts,
        identity: CommunicationSessionIdentity,
    ) -> Result<Arc<LoadedPartitionCapture>, CaptureError> {
        self.partition
            .get_or_init(|| {
                self.require_discovery()?
                    .bind_partition_capture(facts, identity)
                    .map(Arc::new)
            })
            .clone()
    }
}

/// Portable projection of the installed executable's capture geometry.
pub trait TextCaptureGeometry {
    /// Current decoder frontier, when exact.
    fn capture_frontier(&self) -> Result<Option<i32>, CaptureError>;
    /// Selected refinement canvas length, when applicable.
    fn capture_canvas_length(&self) -> Option<u32>;
}

impl<M: eredu_core::backend::TextExecutableTypes> TextCaptureGeometry
    for crate::executable_binding::InstalledTextExecutable<M>
{
    fn capture_frontier(&self) -> Result<Option<i32>, CaptureError> {
        self.forecast_state_offset()
            .map_err(|error| CaptureError::Invalid(error.to_string()))
    }

    fn capture_canvas_length(&self) -> Option<u32> {
        self.block_refinement_geometry()
            .map(|geometry| geometry.canvas_length.get())
    }
}

/// Portable capture uses native facts and derives partition collector support.
pub trait TextCaptureMechanisms<Retained>:
    TextCaptureNative<Retained, Transport: PartitionCaptureCollectorFacts>
{
}

impl<N, Retained> TextCaptureMechanisms<Retained> for N where
    N: TextCaptureNative<Retained, Transport: PartitionCaptureCollectorFacts>
{
}

/// Selected inputs retained through one observer invocation.
pub enum PreparedTextCapture<T> {
    /// Ordinary capture is collected on one device.
    Local,
    /// Partition capture has an exact transport and provenance identity.
    Partitioned {
        /// Native collector for the admitted rank layout.
        transport: T,
        /// Selected layouts and retained artifact projections.
        loaded: Arc<LoadedPartitionCapture>,
        /// Provenance for this capture and active overlay.
        identity: PartitionCaptureIdentity,
    },
    /// Rejection which must enter distributed observer preparation.
    Rejected(CaptureError),
}

/// Opens the collector for an unpartitioned observation.
pub fn local_text_capture<'a, B>(
    capture: &'a mut CaptureSession,
    native: B,
    prediction: u64,
) -> impl crate::ActivationObserver<B::Tensor, B::Error> + 'a
where
    B: InterventionBackend + 'a,
    B::Error: From<BackendFailure> + Send + Sync,
{
    let map_error = CaptureExecutionError::into_operation_error;
    if capture.plan().invocation_bounds().is_some() {
        crate::intervention::CaptureObserver::new(capture, native, map_error)
    } else {
        crate::intervention::CaptureObserver::for_step(capture, native, prediction, map_error)
    }
}

struct RejectedCapture {
    error: CaptureError,
}

impl<T, E: From<BackendFailure>> crate::ActivationObserver<T, E> for RejectedCapture {
    fn transactional(&self) -> bool {
        true
    }

    fn prepare_transaction(
        &mut self,
        _: eredu_core::DistributedCommitEpoch,
        _: crate::ExpertPass,
    ) -> Result<(), E> {
        Err(BackendFailure::from_error(self.error.clone()).into())
    }

    fn observe(&mut self, _: &str, _: &T) -> Result<(), E> {
        Err(BackendFailure::from_error(self.error.clone()).into())
    }
}

impl<T> PreparedTextCapture<T>
where
    T: PartitionCaptureHookTransport + 'static,
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    /// Opens the selected observer over exact native operations.
    pub fn open<'a, N: CaptureTensorOperations<T> + 'a>(
        &'a self,
        capture: &'a mut CaptureSession,
        context: &'a N::Context,
        domain: Option<CaptureTokenDomain<'a>>,
        prediction: u64,
    ) -> Box<dyn crate::ActivationObserver<N::Tensor, N::Error> + 'a> {
        match self {
            Self::Local => Box::new(local_text_capture(
                capture,
                N::native(context, domain, None),
                prediction,
            )),
            Self::Partitioned {
                transport,
                loaded,
                identity,
            } => {
                let max_record_bytes = capture.plan().plan().limits.per_step.encoded_bytes;
                Box::new(
                    PartitionCaptureObserver::for_step(
                        capture,
                        N::native(context, domain, Some(transport)),
                        transport,
                        loaded.layouts(),
                        prediction,
                        PartitionCaptureReceiptLimits {
                            max_producers: loaded.participant_count(),
                            max_fragments: 65_536,
                            max_record_bytes,
                        },
                        estimate_partition_workspace::<T, N>,
                        |error| match error {
                            PartitionCaptureObserverError::Capture(error) => {
                                error.into_operation_error()
                            }
                            PartitionCaptureObserverError::Exchange(error) => {
                                BackendFailure::from_error(error).into()
                            }
                            error @ PartitionCaptureObserverError::HookRejected => {
                                BackendFailure::from_error(error).into()
                            }
                        },
                    )
                    .with_session_identity(identity.clone())
                    .with_interventions(),
                )
            }
            Self::Rejected(error) => Box::new(RejectedCapture {
                error: error.clone(),
            }),
        }
    }
}

/// Computes capture extent from the retained prompt-part plans.
pub fn prepared_prompt_capture_shape(
    plans: &[PreparedInputPartPlan],
) -> Result<(u64, Option<CaptureMediaShape>), CaptureError> {
    let sequence = plans.iter().try_fold(0u64, |total, part| {
        total
            .checked_add(part.decoder_positions())
            .ok_or(CaptureError::Overflow)
    })?;
    let media =
        capture_media_shape(plans).map_err(|error| CaptureError::Invalid(error.to_string()))?;
    Ok((sequence, media))
}

/// Exact decoder span of a retained prompt, when it differs from full prefill.
pub fn retained_prefill_span(
    cursor: &ErasedPrefillCursor,
) -> Result<Option<CapturePrefillSpan>, PrefillCursorError> {
    let range = cursor.final_decoder_range()?;
    let total = cursor.token_count();
    Ok(
        (range.start != 0 || range.end != total).then_some(CapturePrefillSpan {
            start: range.start as u64,
            end: range.end as u64,
            total: total as u64,
        }),
    )
}

impl<R, N, Model, Retained, Value, Processor> TextSessionDriver<R, N>
where
    R: std::ops::Deref<Target = TextSessionResources<Model, Retained, Value, Processor>> + 'static,
    N: TextCaptureMechanisms<Retained>,
    Model: TextCaptureGeometry,
{
    /// Binds the architecture's retained layout to exact native collector facts.
    pub fn loaded_partition_capture(
        &self,
    ) -> Result<Option<Arc<LoadedPartitionCapture>>, CaptureError> {
        N::capture_transport(&self.resources().retained)
            .map(|(transport, id)| self.capture.bind(transport, id))
            .transpose()
    }

    /// Reads declarations without resolving content identity.
    #[cfg(feature = "tensor-capture")]
    pub fn text_capture_declarations(&self) -> Result<CaptureDeclarations, CaptureError> {
        if let Some(partition) = self.loaded_partition_capture()? {
            return Ok(partition.declarations().clone());
        }
        Ok(self.capture.require_discovery()?.capture_declarations())
    }

    /// Reads the capture catalog with exact artifact content identity.
    #[cfg(feature = "tensor-capture")]
    pub fn text_capture_discovery(&self) -> Result<CaptureDiscovery, CaptureError> {
        if let Some(partition) = self.loaded_partition_capture()? {
            return partition.content_discovery();
        }
        self.capture.require_discovery()?.capture()
    }

    /// Validates the complete capture plan without opening any native scope.
    #[cfg(feature = "tensor-capture")]
    pub fn validate_text_capture(&self, plan: &AdmittedCapturePlan) -> Result<(), CaptureError> {
        if plan.is_empty() {
            return Ok(());
        }
        if plan.request().batch != 1 {
            return Err(CaptureError::Unsupported(
                "bounded capture requires single-sequence text generation".into(),
            ));
        }
        crate::capture::validate_session(
            plan,
            &self.text_capture_declarations()?,
            |shape, selection, slice| N::capture_workspace().estimate(shape, selection, slice),
        )
    }

    /// Projects intervention support for the exact installed model and collector.
    #[cfg(feature = "tensor-capture")]
    pub fn text_intervention_discovery(&self) -> Result<InterventionDiscovery, CaptureError> {
        let prepared = self.model_discovery().ok_or_else(|| {
            CaptureError::Unsupported("session has no retained intervention catalog".into())
        })?;
        let mechanisms = N::intervention_mechanisms();
        let mut discovery = prepared.intervention(&mechanisms)?;
        if let Some(partition) = self.loaded_partition_capture()? {
            discovery = crate::inspection::intervention_support(
                discovery.points,
                &partition.content_discovery()?,
                &mechanisms,
            );
        }
        discovery.session_identity = Some(self.intervention_identity().to_owned());
        Ok(discovery)
    }

    /// Checks capture and intervention against the same retained invocation.
    #[cfg(feature = "tensor-capture")]
    pub fn validate_text_interventions(
        &self,
        capture: &AdmittedCapturePlan,
        plan: &AdmittedInterventionPlan,
        estimator: &dyn InterventionEstimator,
    ) -> Result<(), CaptureError> {
        self.validate_text_capture(capture)?;
        if plan.is_empty() {
            return Ok(());
        }
        crate::intervention::validate_session(
            capture,
            plan,
            &self.text_intervention_discovery()?,
            estimator,
        )
    }

    /// Installs capture and any already admitted intervention after checking exact provenance.
    #[cfg(feature = "tensor-capture")]
    pub fn configure_text_capture(
        &self,
        slot: &mut Option<CaptureSession>,
        plan: AdmittedCapturePlan,
        intervention: Option<(AdmittedInterventionPlan, Arc<dyn InterventionEstimator>)>,
    ) -> Result<(), CaptureError> {
        match &intervention {
            Some((intervention, estimator)) => {
                self.validate_text_interventions(&plan, intervention, estimator.as_ref())?;
            }
            None => self.validate_text_capture(&plan)?,
        }
        if plan.plan().provenance == CaptureProvenance::ContentExact {
            self.text_capture_discovery()?;
        }
        // Publish only after identity binding. A failed binding leaves the existing
        // slot intact and allows the caller to retry.
        if slot.is_some() {
            return Err(CaptureError::Invalid(
                "capture/intervention plan already installed".into(),
            ));
        }
        let mut admitted = None;
        crate::intervention::install_session(&mut admitted, plan, intervention)?;
        if let (Some(capture), Some(partition)) = (&mut admitted, self.loaded_partition_capture()?)
        {
            capture.ensure_partition_capture(partition.identity(
                capture.requires_content_identity(),
                self.parameter_overlay_identity(),
            )?)?;
        }
        *slot = admitted;
        Ok(())
    }

    fn capture_frontier(&self) -> Result<u64, CaptureError> {
        let frontier = self.resources().model.capture_frontier()?.ok_or_else(|| {
            CaptureError::Unsupported("execution has no exact context frontier".into())
        })?;
        u64::try_from(frontier)
            .map_err(|_| CaptureError::Invalid("negative context frontier".into()))
    }

    /// Invocation geometry for the next retained decoder or refinement operation.
    pub fn text_capture_shape(
        &self,
        sequence: u64,
    ) -> Result<CaptureInvocationShape, CaptureError> {
        Ok(CaptureInvocationShape {
            media: None,
            batch: 1,
            sequence,
            context: Some(
                self.capture_frontier()?
                    .checked_add(sequence)
                    .ok_or(CaptureError::Overflow)?,
            ),
        })
    }

    /// Maximum capture extent needed for the selected refinement run.
    pub fn text_capture_invocation_bounds(
        &self,
        request: CaptureRequestShape,
    ) -> Result<Option<CaptureInvocationBounds>, CaptureError> {
        let Some(canvas) = self
            .resources()
            .model
            .capture_canvas_length()
            .map(u64::from)
        else {
            return Ok(None);
        };
        Ok(Some(CaptureInvocationBounds {
            batch: request.batch,
            max_sequence: request.prompt_tokens.max(canvas),
            max_context: Some(
                self.capture_frontier()?
                    .checked_add(request.prompt_tokens)
                    .and_then(|n| n.checked_add(request.max_predictions))
                    .and_then(|n| n.checked_add(canvas))
                    .ok_or(CaptureError::Overflow)?,
            ),
            max_predictions: request.max_predictions,
        }))
    }

    /// Prepares the local or partitioned observer against the retained provenance.
    pub fn prepare_capture_observer(
        &self,
        capture: &CaptureSession,
    ) -> PreparedTextCapture<N::Transport> {
        let Some((transport, id)) = N::capture_transport(&self.resources().retained) else {
            return PreparedTextCapture::Local;
        };
        let result = self.capture.bind(transport, id).and_then(|loaded| {
            let identity = loaded.identity(
                capture.requires_content_identity(),
                self.parameter_overlay_identity(),
            )?;
            Ok(PreparedTextCapture::Partitioned {
                transport: transport.clone(),
                loaded,
                identity,
            })
        });
        result.unwrap_or_else(PreparedTextCapture::Rejected)
    }

    /// Prepares a bounded invocation or publishes its rejection to observer agreement.
    pub fn prepare_capture_invocation(
        &self,
        capture: &mut CaptureSession,
        prediction: u64,
        invocation: Result<(CapturePhase, CaptureInvocationShape), CaptureError>,
    ) -> PreparedTextCapture<N::Transport> {
        let admitted = invocation.and_then(|(phase, shape)| {
            capture.set_commitment_boundary(prediction)?;
            capture.begin_invocation(phase, prediction, shape, Default::default())
        });
        match admitted {
            Ok(()) => self.prepare_capture_observer(capture),
            Err(error) => PreparedTextCapture::Rejected(error),
        }
    }

    /// Applies the plan's invocation bounds before preparing the exact collector.
    pub fn prepare_text_capture_observer(
        &self,
        capture: &mut CaptureSession,
        prediction: u64,
        phase: CapturePhase,
        geometry: Result<(u64, Option<CaptureMediaShape>), CaptureError>,
    ) -> PreparedTextCapture<N::Transport> {
        if capture.plan().invocation_bounds().is_none() {
            return self.prepare_capture_observer(capture);
        }
        let invocation = geometry.and_then(|(sequence, media)| {
            let mut shape = self.text_capture_shape(sequence)?;
            shape.media = media;
            Ok((phase, shape))
        });
        self.prepare_capture_invocation(capture, prediction, invocation)
    }
}

fn estimate_partition_workspace<T: 'static, N: CaptureTensorOperations<T>>(
    shape: &[u64],
    selection: &CaptureSelection,
    slice: &ResolvedCaptureSlice,
) -> Result<PartitionCaptureNativeEstimate, CaptureError> {
    Ok(PartitionCaptureNativeEstimate {
        capture: N::capture_workspace().estimate(shape, selection, slice)?,
        generated_creation_bytes: N::generated_creation_bytes(shape)?,
    })
}
