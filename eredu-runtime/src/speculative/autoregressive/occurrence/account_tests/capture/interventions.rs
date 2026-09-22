//! Paid immutable edit source and aborted attributed outcomes on a real phase.
use super::*;
use crate::working_memory::{
    CaptureInterventionClaim, ClaimedIntervention, OriginalInterventionSource,
};
use eredu_core::intervention::*;
fn edits(capture: &SharedCapturePlan) -> (InterventionDiscovery, AdmittedInterventionPlan) {
    let observation = &capture.admission().points()[0];
    let discovery = InterventionDiscovery {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        artifact_identity: "actual-source".into(),
        session_identity: Some("actual-session".into()),
        points: vec![InterventionPoint {
            path: observation.path.clone(),
            node_id: observation.node_id.clone(),
            stage: InterventionStage::LogitsBeforeSampling,
            axes: observation.axes.clone().unwrap(),
            dtypes: vec![InterventionDtype::Float32],
            operations: vec![InterventionKind::Scale],
            score_stages: vec![],
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            conditions: vec![],
            routing: None,
            routed_units: None,
        }],
    };
    let plan = InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: vec![InterventionOperation {
            id: "scale-é".into(),
            target: MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            action: InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 2.0,
            },
            evidence: InterventionEvidence::None,
        }],
    }
    .admit(&discovery, capture.admission().request(), "facade-session")
    .unwrap();
    (discovery, plan)
}
#[test]
fn numerical_intervention_source_and_aborted_outcomes_preserve_identity_spending_and_custody() {
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let schedule = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(16).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let pass = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let report = report(
        schedule
            .workspace_geometry(2, pass, NonZeroU64::new(1).unwrap())
            .unwrap(),
    );
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 24, 0).unwrap();
    let request = OriginalSpeculativeRequest::prepare(
        &pool,
        &InferenceExecutionIdentity::default(),
        &schedule,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 24),
    )
    .unwrap();
    let mut cursor = schedule.into_cursor();
    let role = request
        .reserve_role(
            cursor.claim(2, pass).unwrap(),
            requirements(report.span_workspace_plan()),
        )
        .unwrap();
    let model = role.budget_custody();
    let source = capture_source();
    let (mut discovery, caller) = edits(&source);
    caller.validate_discovery(&discovery).unwrap();
    let invoked = caller
        .plan()
        .clone()
        .admit_invocations(
            &discovery,
            CaptureInvocationBounds {
                batch: 1,
                max_sequence: 1,
                max_context: Some(4),
                max_predictions: 4,
            },
            "facade-session",
        )
        .unwrap();
    invoked.validate_discovery(&discovery).unwrap();
    discovery.session_identity = Some("another-session".into());
    assert_eq!(
        caller.validate_discovery(&discovery),
        Err(InterventionSourceError::Identity)
    );
    let copy = PreparedInterventionPlanCopy::inspect(&caller).unwrap();
    let required = MemoryLedger::intervention_source_required_bytes(&copy).unwrap();
    let short = crate::working_memory::memory_fixture::host_ledger(required - 1, 0).unwrap();
    assert!(short.compile_intervention_source(copy).is_err());
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    let edit = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&caller).unwrap())
        .unwrap();
    assert_eq!(edit.plan().admission().identity(), caller.identity());
    assert_eq!(edit.plan().admission().plan(), caller.plan());
    assert_ne!(
        edit.plan().admission().plan().operations.as_ptr(),
        caller.plan().operations.as_ptr()
    );
    let foreign_pool = crate::working_memory::memory_fixture::host_ledger(1 << 24, 0).unwrap();
    let foreign = foreign_pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&caller).unwrap())
        .unwrap();
    let foreign_host = host(&source, 1).with_interventions(&foreign).unwrap();
    let requirements = SpeculativeNumericalRequirements::new(
        program(1),
        Some(0),
        Some(0),
        Some(0),
        Some(0),
        std::sync::Arc::new(crate::working_memory::memory_fixture::host_placement().clone()),
    )
    .unwrap()
    .with_capture_destination(&foreign_host)
    .unwrap();
    let failed = request
        .reserve_numerical(requirements, &[SpeculativeNumericalSource::Model(&model)])
        .unwrap_err();
    assert!(matches!(
        failed.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    drop(failed);
    drop(foreign_host);
    drop(foreign);
    assert_eq!(foreign_pool.payload_used_bytes().unwrap(), 0);
    let prepared = host(&source, 1).with_interventions(&edit).unwrap();
    let requirements = SpeculativeNumericalRequirements::new(
        program(1),
        Some(0),
        Some(0),
        Some(0),
        Some(0),
        std::sync::Arc::new(crate::working_memory::memory_fixture::host_placement().clone()),
    )
    .unwrap()
    .with_capture_destination(&prepared)
    .unwrap();
    let phase = request
        .reserve_numerical(requirements, &[SpeculativeNumericalSource::Model(&model)])
        .unwrap();
    assert_eq!(
        phase.ordinal(),
        0,
        "foreign source refused before physical claim"
    );
    let (budget, mut invocation) = phase.begin_with_capture(prepared).unwrap();
    let mut backend = HostBackend::default();
    // No native edit producer is installed yet. A mere observation must never
    // certify the pending edit, but the real aborted evidence remains deliverable.
    assert!(invocation
        .observe(&mut backend, &[1.0, 2.0, 3.0, 4.0])
        .is_err());
    let escaped = invocation.take_failed_evidence().unwrap().unwrap();
    assert_eq!(escaped.as_ref().outcome, CaptureStepOutcome::Aborted);
    let record = &escaped.as_ref().interventions[0];
    assert_eq!(record.plan_id, caller.identity());
    assert_eq!(record.operation_id, "scale-é");
    assert_eq!(record.prediction_index, 1);
    assert_eq!(record.phase, CapturePhase::Decode);
    assert_eq!(record.outcome, InterventionOutcome::Missing);
    assert!(record.evidence.is_empty());
    assert!(escaped.as_ref().step_usage.host_bytes >= record.charged.host_bytes);
    assert_eq!(escaped.as_ref().cumulative_usage.captures, 1);
    assert!(invocation.observe(&mut backend, &[1.0; 4]).is_err());
    assert_eq!(backend.calls, 1);
    request.close().unwrap();
    drop((invocation, budget, model, role, request, edit, source));
    assert!(pool.payload_used_bytes().unwrap() > 0);
    drop(escaped);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

struct EditBackend<'a> {
    capture: HostBackend,
    custody: &'a OriginalSpeculativeNumericalBudgetCustody,
    source: &'a OriginalInterventionSource,
    refuse: bool,
    calls: usize,
}
impl ScheduledCaptureBackend for EditBackend<'_> {
    type Tensor = [f32; 4];
    type Error = CaptureRunHostError;
    fn validate_source(
        &self,
        value: &Self::Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Self::Error> {
        self.capture.validate_source(value, geometry)
    }
    fn estimate(
        &self,
        value: &Self::Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        self.capture.estimate(value, geometry)
    }
    fn transform(
        &mut self,
        value: &Self::Tensor,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Self::Error> {
        self.capture.transform(value, claim)
    }
    fn intervention_usage(
        &self,
        _: &Self::Tensor,
        claim: &CaptureInterventionClaim<'_>,
    ) -> Result<CaptureUsage, crate::capture::FundedCaptureError<Self::Error>> {
        claim
            .validate_numerical_custody(self.custody)
            .map_err(CaptureRunHostError::from)?;
        claim
            .validate_source(self.source)
            .map_err(CaptureRunHostError::from)?;
        Ok(CaptureUsage {
            retained_bytes: 64,
            host_bytes: 32,
            ..Default::default()
        })
    }
    fn apply_intervention(
        &mut self,
        value: &Self::Tensor,
        claim: CaptureInterventionClaim<'_>,
        charged: CaptureUsage,
    ) -> Result<(Self::Tensor, ClaimedIntervention), crate::capture::FundedCaptureError<Self::Error>>
    {
        self.calls += 1;
        if self.refuse {
            return Err(crate::capture::FundedCaptureError::Backend(
                WorkingMemoryError::UnknownBound.into(),
            ));
        }
        let InterventionAction::Scale { factor, .. } =
            claim.admission().plan().operations[claim.index()].action
        else {
            panic!("scale fixture")
        };
        Ok((value.map(|n| n * factor), claim.finish(charged)?))
    }
}
#[test]
fn numerical_intervention_callback_preserves_original_capture_and_failed_spending() {
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let schedule = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(16).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let pass = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let report = report(
        schedule
            .workspace_geometry(2, pass, NonZeroU64::new(1).unwrap())
            .unwrap(),
    );
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 24, 0).unwrap();
    let request = OriginalSpeculativeRequest::prepare(
        &pool,
        &InferenceExecutionIdentity::default(),
        &schedule,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 24),
    )
    .unwrap();
    let mut cursor = schedule.into_cursor();
    let role = request
        .reserve_role(
            cursor.claim(2, pass).unwrap(),
            requirements(report.span_workspace_plan()),
        )
        .unwrap();
    let model = role.budget_custody();
    let source = capture_source();
    let (_, caller) = edits(&source);
    let edit = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&caller).unwrap())
        .unwrap();
    let invoke = || {
        let host = host(&source, 1).with_interventions(&edit).unwrap();
        let quote = SpeculativeNumericalRequirements::new(
            program(1),
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            std::sync::Arc::new(crate::working_memory::memory_fixture::host_placement().clone()),
        )
        .unwrap()
        .with_capture_destination(&host)
        .unwrap();
        request
            .reserve_numerical(quote, &[SpeculativeNumericalSource::Model(&model)])
            .unwrap()
            .begin_with_capture(host)
            .unwrap()
    };
    let (budget, mut invocation) = invoke();
    let mut backend = EditBackend {
        capture: HostBackend::default(),
        custody: &budget,
        source: &edit,
        refuse: false,
        calls: 0,
    };
    let original = [1.0, -2.0, 3.0, 4.0];
    let metadata = crate::intervention::intervention_metadata(
        &caller.plan().operations[0],
        &caller.points()[0],
        caller.identity(),
    )
    .unwrap();
    let effective = invocation
        .observe_and_intervene(&mut backend, &original)
        .unwrap()
        .unwrap();
    assert_eq!(effective, [2.0, -4.0, 6.0, 8.0]);
    assert_eq!(backend.calls, 1);
    let escaped = invocation.take_shared_step().unwrap().unwrap();
    assert_eq!(
        escaped.as_ref().interventions[0].outcome,
        InterventionOutcome::Applied
    );
    assert_eq!(escaped.as_ref().cumulative_usage.captures, 1);
    assert_eq!(
        escaped.as_ref().interventions[0].charged.host_bytes,
        metadata.host_bytes + 32
    );
    let Some(CapturePayload::SharedTensor(raw)) = &escaped.as_ref().records[0].payload else {
        panic!("paid raw tensor")
    };
    assert_eq!(
        raw.as_ref().data(),
        &TensorObservationData::F32(original.to_vec())
    );
    // The same coordinate uses a new physical phase and the same live ledger.
    let (next_budget, mut next) = invoke();
    let mut refused = EditBackend {
        capture: HostBackend::default(),
        custody: &next_budget,
        source: &edit,
        refuse: true,
        calls: 0,
    };
    assert!(next.observe_and_intervene(&mut refused, &original).is_err());
    assert_eq!(refused.calls, 1);
    let failed = next.take_failed_evidence().unwrap().unwrap();
    assert_eq!(failed.as_ref().outcome, CaptureStepOutcome::Aborted);
    assert!(matches!(
        failed.as_ref().interventions[0].outcome,
        InterventionOutcome::Failed { .. }
    ));
    assert_eq!(
        failed.as_ref().interventions[0].charged.host_bytes,
        metadata.host_bytes + 32
    );
    assert_eq!(failed.as_ref().cumulative_usage.captures, 2);
    assert!(next.observe_and_intervene(&mut refused, &original).is_err());
    assert_eq!(refused.calls, 1);
    drop((backend, refused));
    request.close().unwrap();
    drop((
        invocation,
        next,
        budget,
        next_budget,
        model,
        role,
        request,
        edit,
        source,
    ));
    assert!(pool.payload_used_bytes().unwrap() > 0);
    drop((escaped, failed));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

mod evidence;
