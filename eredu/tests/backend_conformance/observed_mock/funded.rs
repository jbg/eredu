//! Actual original host capture over the same fixed scalar model equations.
use super::*;
use eredu_core::{HostMetadataFunding, HostMetadataFundingError, HostPreparationAuthority};
use eredu_runtime::capture::{FundedCaptureError, ScheduledCaptureBackend};
use eredu_runtime::working_memory::*;
use std::mem::size_of;

pub(in super::super) fn plan<'a>(
    runtime: &ModelRuntime<MockBackend>,
    options: Option<&'a eredu_core::TextPreparationOptions>,
    config: TextGenerationConfig,
) -> Result<Option<CaptureRunHostPlan<'a>>, eredu_core::BackendFailure> {
    let Some(options) = options else {
        return Ok(None);
    };
    let Some(capture) = &options.capture else {
        return if options.interventions.is_some() {
            Err(eredu_core::TokenInputRejection::IdentityMismatch.into_backend_failure())
        } else {
            Ok(None)
        };
    };
    let env = MockBackend::source_environment(runtime);
    let capacity = config
        .inference_policy()
        .memory_limits
        .resolve(env.pool.topology())
        .map_err(|e| {
            eredu_core::BackendFailure::from_error(
                eredu_runtime::working_memory::WorkingMemoryError::from(e),
            )
        })?;
    let funding = env
        .pool
        .prepare_workspace_metadata(&env.execution, capacity)?;
    let required = eredu_runtime::intervention::StaticInterventionPreflight::required_bytes()
        .and_then(|n| {
            n.checked_add(HostPreparationAuthority::retention_bytes::<
                HostMetadataFunding,
            >()?)
        })
        .and_then(|n| {
            n.checked_add(
                size_of::<CaptureRunHostPlan<'_>>()
                    + size_of::<Option<OriginalInterventionSource>>(),
            )
        })
        .ok_or(HostMetadataFundingError::Overflow)?;
    funding.reserve_metadata(required)?;
    let error = |cause| admitted_text::funded_error(cause, &funding);
    // Borrow the loaded declarations. Raw public admission already validates
    // schedules/actions; source identity and exact selected capabilities are
    // checked again here without cloning discovery or changing the source.
    if capture.admission().points().iter().any(|point| {
        runtime
            .session()
            .capture_discovery
            .catalog
            .points
            .iter()
            .find(|p| p.path == point.path)
            != Some(point)
    }) {
        return Err(eredu_core::TokenInputRejection::IdentityMismatch.into_backend_failure());
    }
    let mut scratch = eredu_runtime::intervention::StaticInterventionPreflight::prepare(
        HostPreparationAuthority::retain(funding.clone()),
    )
    .map_err(|e| admitted_text::funded_error(e, &funding))?;
    let mut host = CaptureRunHostPlan::prepare(capture).map_err(error)?;
    if let Some(intervention) = &options.interventions {
        let original = original_sources::compile_intervention::<MockBackend>(
            runtime,
            intervention.admission().plan(),
            capture,
            intervention.admission().session_id(),
            &funding,
        )?;
        if original.plan().admission().identity() != intervention.admission().identity() {
            return Err(eredu_core::TokenInputRejection::IdentityMismatch.into_backend_failure());
        }
        scratch
            .run_with_source(capture.admission(), original.plan(), &Estimates)
            .map_err(|e| admitted_text::funded_error(e, &funding))?;
        host = host.with_interventions(&original).map_err(error)?;
    } else {
        scratch
            .run_capture(capture.admission(), &Estimates)
            .map_err(|e| admitted_text::funded_error(e, &funding))?;
    }
    Ok(Some(host))
}

pub(crate) struct HostMechanism {
    slice: std::cell::RefCell<ResolvedCaptureSlice>,
    source: Option<OriginalInterventionSource>,
    funding: HostMetadataFunding,
}
impl HostMechanism {
    pub(super) fn new(
        source: Option<OriginalInterventionSource>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, MockError> {
        funding.reserve_metadata(
            size_of::<Self>()
                + 4 * 3 * size_of::<u64>()
                + size_of::<[Vec<u64>; 4]>()
                + size_of::<Result<Self, MockError>>(),
        )?;
        let mut buffers: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::new());
        for buffer in &mut buffers {
            buffer
                .try_reserve_exact(3)
                .map_err(|_| WorkingMemoryError::UnknownBound)?;
            buffer.resize(3, 0);
        }
        let [starts, ends, strides, shape] = buffers;
        Ok(Self {
            slice: std::cell::RefCell::new(ResolvedCaptureSlice {
                starts,
                ends,
                strides,
                shape,
            }),
            source,
            funding: funding.clone(),
        })
    }
    fn resolve(
        &self,
        value: &Value,
        claim: &CaptureInterventionClaim<'_>,
    ) -> Result<(), FundedCaptureError<MockError>> {
        claim
            .validate_source(
                self.source
                    .as_ref()
                    .ok_or(WorkingMemoryError::IdentityMismatch)
                    .map_err(CaptureRunHostError::from)?,
            )
            .map_err(CaptureRunHostError::from)?;
        let (phase, prediction) = claim.coordinate();
        claim
            .admission()
            .resolve_prepared_at(
                claim.index(),
                phase,
                prediction,
                &value.shape,
                InterventionDtype::Float32,
                &mut self.slice.borrow_mut(),
            )
            .map_err(|_| eredu_runtime::capture::CaptureProtocolError::Geometry)?;
        Ok(())
    }
}
impl ScheduledCaptureBackend for HostMechanism {
    type Tensor = Value;
    type Error = MockError;
    fn validate_source(
        &self,
        value: &Value,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, MockError> {
        if value
            .shape
            .iter()
            .copied()
            .eq(geometry.source_shape().iter().map(|&n| n as u64))
        {
            Ok(TensorDtype::F32)
        } else {
            Err(MockError::CaptureGeometry)
        }
    }
    fn estimate(
        &self,
        value: &Value,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        cost(&value.shape)
    }
    fn transform(
        &mut self,
        _: &Value,
        _: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, MockError> {
        Err(MockError::CaptureGeometry)
    }
    fn validate_summary_source(
        &self,
        value: &Value,
        geometry: &CaptureSummaryGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<MockError>> {
        if value
            .shape
            .iter()
            .copied()
            .eq(geometry.source_shape().iter().map(|&n| n as u64))
        {
            Ok(TensorDtype::F32)
        } else {
            Err(FundedCaptureError::Backend(MockError::CaptureGeometry))
        }
    }
    fn estimate_summary(
        &self,
        geometry: &CaptureSummaryGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        let shape: [u64; 3] = geometry
            .source_shape()
            .try_into()
            .map(|s: [usize; 3]| s.map(|n| n as u64))
            .map_err(|_| CaptureError::Overflow)?;
        cost(&shape)
    }
    fn transform_summary(
        &mut self,
        value: &Value,
        claim: CaptureSummaryClaim<'_, '_>,
    ) -> Result<ClaimedCaptureSummary, FundedCaptureError<MockError>> {
        if claim.geometry().admission().plan().selections[claim.geometry().selection_index()].id
            == "injected-native-failure"
        {
            return Err(FundedCaptureError::Backend(MockError::InjectedCapture));
        }
        Ok(claim.summarize_host_f32(
            &value.shape.map(|n| n as usize),
            CaptureHostF32::Uniform(value.scale),
        )?)
    }
    fn intervention_usage(
        &self,
        value: &Value,
        claim: &CaptureInterventionClaim<'_>,
    ) -> Result<CaptureUsage, FundedCaptureError<MockError>> {
        claim
            .validate_source(
                self.source
                    .as_ref()
                    .ok_or(WorkingMemoryError::IdentityMismatch)
                    .map_err(CaptureRunHostError::from)?,
            )
            .map_err(CaptureRunHostError::from)?;
        self.resolve(value, claim)?;
        Ok(Estimates.activation_usage(
            &value.shape,
            &self.slice.borrow(),
            &claim.admission().plan().operations[claim.index()].action,
        )?)
    }
    fn apply_intervention(
        &mut self,
        value: &Value,
        claim: CaptureInterventionClaim<'_>,
        charged: CaptureUsage,
    ) -> Result<(Value, ClaimedIntervention), FundedCaptureError<MockError>> {
        self.resolve(value, &claim)?;
        let action = &claim.admission().plan().operations[claim.index()].action;
        let output = eredu_runtime::intervention::apply_activation_with_source_shape(
            &mut Mechanism,
            value,
            action,
            &self.slice.borrow(),
            &value.shape,
        )
        .map_err(|e| match e {
            eredu_runtime::capture::CaptureExecutionError::Backend(e) => {
                FundedCaptureError::Backend(e)
            }
            eredu_runtime::capture::CaptureExecutionError::Admission(e) => {
                FundedCaptureError::Admission(e)
            }
        })?;
        Ok((output, claim.finish(charged)?))
    }
    fn intervention_evidence_usage(
        &self,
        value: &Value,
        claim: &CaptureInterventionEvidenceClaim<'_, '_>,
    ) -> Result<(TensorDtype, CaptureUsage), FundedCaptureError<MockError>> {
        claim
            .validate_source(
                self.source
                    .as_ref()
                    .ok_or(WorkingMemoryError::IdentityMismatch)
                    .map_err(CaptureRunHostError::from)?,
            )
            .map_err(CaptureRunHostError::from)?;
        match claim.kind() {
            CaptureInterventionEvidenceKind::Summary(summary) => {
                self.validate_summary_source(value, summary.geometry())?;
                Ok((TensorDtype::F32, cost(&value.shape)?))
            }
            _ => Err(eredu_runtime::capture::CaptureProtocolError::Geometry.into()),
        }
    }
    fn capture_intervention_evidence<'a>(
        &mut self,
        value: &Value,
        claim: CaptureInterventionEvidenceClaim<'a, '_>,
    ) -> Result<ClaimedInterventionEvidence<'a>, FundedCaptureError<MockError>> {
        claim
            .validate_source(
                self.source
                    .as_ref()
                    .ok_or(WorkingMemoryError::IdentityMismatch)
                    .map_err(CaptureRunHostError::from)?,
            )
            .map_err(CaptureRunHostError::from)?;
        let (receipt, kind) = claim.into_parts();
        match kind {
            CaptureInterventionEvidenceKind::Summary(summary) => {
                let value = self.transform_summary(value, summary)?;
                Ok(receipt.finish_summary(value)?)
            }
            _ => Err(eredu_runtime::capture::CaptureProtocolError::Geometry.into()),
        }
    }
}

impl State {
    pub(in super::super) fn install_funded(
        &mut self,
        bank: eredu_runtime::capture::FundedCaptureSession,
        funding: &HostMetadataFunding,
    ) -> Result<(), MockError> {
        let mechanism = HostMechanism::new(bank.intervention_source().cloned(), funding)?;
        self.host_mechanism = Some(mechanism);
        self.funded = Some(bank);
        Ok(())
    }
    pub(super) fn observe_funded(
        &mut self,
        value: Value,
        phase: CapturePhase,
    ) -> Result<Value, MockError> {
        let mechanism = self
            .host_mechanism
            .as_mut()
            .ok_or(MockError::CaptureGeometry)?;
        let funding = mechanism.funding.clone();
        funding.reserve_metadata(
            size_of::<Value>() * 4
                + size_of::<Result<Value, MockError>>()
                + size_of::<FundedCaptureError<MockError>>()
                + size_of::<&dyn eredu_runtime::ActivationObserver<Value, MockError>>()
                + size_of::<eredu_core::DistributedCommitEpoch>()
                + size_of::<eredu_runtime::ExpertPass>()
                + size_of::<Result<(), MockError>>(),
        )?;
        let map = |cause| MockError::ProviderRetained(admitted_text::funded_error(cause, &funding));
        let epoch = eredu_core::DistributedCommitEpoch::new(
            self.sampling
                .prediction
                .checked_add(1)
                .ok_or(WorkingMemoryError::Overflow)?,
        )
        .ok_or(WorkingMemoryError::Overflow)?;
        let pass = match phase {
            CapturePhase::Prefill => eredu_runtime::ExpertPass::Prefill,
            CapturePhase::Decode => eredu_runtime::ExpertPass::Decode,
            _ => return Err(MockError::CaptureGeometry),
        };
        self.funded.as_mut().unwrap().with_observer(
            mechanism,
            self.sampling.prediction,
            &map,
            |observer| {
                let result = (|| {
                    observer.prepare_transaction(epoch, pass)?;
                    observer.coordinate_transaction(epoch)?;
                    let result = eredu_runtime::observe_and_intervene(
                        observer,
                        eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                        &value,
                    )?;
                    observer.complete_transaction(epoch)?;
                    Ok(result)
                })();
                observer.finish_transaction(epoch, result.is_ok());
                result
            },
        )?
    }
}
