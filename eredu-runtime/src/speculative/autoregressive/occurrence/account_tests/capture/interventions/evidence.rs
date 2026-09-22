//! Actual numerical destination: attribution, logical limits and aborted custody.
use super::*;
use crate::capture::FundedCaptureError;
use crate::working_memory::{
    CaptureInterventionEvidenceClaim, CaptureInterventionEvidenceKind, ClaimedInterventionEvidence,
};

fn capture_with_evidence_budget(captures: u64) -> SharedCapturePlan {
    let source = capture_source();
    let mut plan = source.admission().plan().clone();
    plan.limits.per_step.captures = captures;
    plan.limits.cumulative.captures = captures * 2;
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: source.admission().points().to_vec(),
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: MODEL_LOGITS_OBSERVATION_PATH.into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::FullTensor],
        max_histogram_bins: 0,
        conditions: vec![],
    };
    SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &catalog,
            &support,
            &capabilities,
            source.admission().request(),
            CaptureTextOrigin {
                cached_positions: 0,
            },
        )
        .unwrap(),
    )
}
struct EvidenceBackend<'a> {
    edit: EditBackend<'a>,
    refuse_after: bool,
    evidence_calls: usize,
}
impl ScheduledCaptureBackend for EvidenceBackend<'_> {
    type Tensor = [f32; 4];
    type Error = CaptureRunHostError;
    fn validate_source(
        &self,
        value: &Self::Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Self::Error> {
        self.edit.validate_source(value, geometry)
    }
    fn estimate(
        &self,
        value: &Self::Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        self.edit.estimate(value, geometry)
    }
    fn transform(
        &mut self,
        value: &Self::Tensor,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Self::Error> {
        self.edit.transform(value, claim)
    }
    fn intervention_usage(
        &self,
        value: &Self::Tensor,
        claim: &CaptureInterventionClaim<'_>,
    ) -> Result<CaptureUsage, FundedCaptureError<Self::Error>> {
        self.edit.intervention_usage(value, claim)
    }
    fn apply_intervention(
        &mut self,
        value: &Self::Tensor,
        claim: CaptureInterventionClaim<'_>,
        usage: CaptureUsage,
    ) -> Result<(Self::Tensor, ClaimedIntervention), FundedCaptureError<Self::Error>> {
        self.edit.apply_intervention(value, claim, usage)
    }
    fn intervention_evidence_usage(
        &self,
        _: &Self::Tensor,
        claim: &CaptureInterventionEvidenceClaim<'_, '_>,
    ) -> Result<(TensorDtype, CaptureUsage), FundedCaptureError<Self::Error>> {
        claim
            .validate_source(self.edit.source)
            .map_err(CaptureRunHostError::from)?;
        claim
            .validate_numerical_custody(self.edit.custody)
            .map_err(CaptureRunHostError::from)?;
        let elements = match claim.kind() {
            CaptureInterventionEvidenceKind::Preview(claim) => claim.geometry().elements(),
            CaptureInterventionEvidenceKind::Summary(claim) => claim.geometry().elements(),
        };
        Ok((
            TensorDtype::F32,
            CaptureUsage {
                captures: 1,
                retained_bytes: elements as u64 * 4,
                host_bytes: 128,
                encoded_bytes: 8192,
            },
        ))
    }
    fn capture_intervention_evidence<'a>(
        &mut self,
        value: &Self::Tensor,
        claim: CaptureInterventionEvidenceClaim<'a, '_>,
    ) -> Result<ClaimedInterventionEvidence<'a>, FundedCaptureError<Self::Error>> {
        self.evidence_calls += 1;
        if self.refuse_after && claim.coordinate().1 == InterventionEvidenceSide::After {
            return Err(FundedCaptureError::Backend(
                WorkingMemoryError::UnknownBound.into(),
            ));
        }
        let (receipt, kind) = claim.into_parts();
        Ok(match kind {
            CaptureInterventionEvidenceKind::Preview(claim) => {
                let elements = claim.geometry().elements();
                let mut writer = claim.prepare()?;
                for &value in value.iter().take(elements) {
                    writer.push_f32(value).unwrap();
                }
                receipt.finish_preview(writer.finish().unwrap())?
            }
            CaptureInterventionEvidenceKind::Summary(claim) => {
                let sum: f64 = value.iter().map(|n| f64::from(*n)).sum();
                let squares: f64 = value.iter().map(|n| f64::from(*n).powi(2)).sum();
                let summary = CaptureSummary {
                    elements: 4,
                    finite: 4,
                    non_finite: 0,
                    nan: 0,
                    positive_infinity: 0,
                    negative_infinity: 0,
                    min: Some(value.iter().copied().reduce(f32::min).unwrap() as f64),
                    max: Some(value.iter().copied().reduce(f32::max).unwrap() as f64),
                    mean: Some(sum / 4.0),
                    rms: Some((squares / 4.0).sqrt()),
                };
                receipt
                    .finish_summary(claim.finish_numerical(self.edit.custody, summary).unwrap())?
            }
        })
    }
}
#[test]
fn numerical_edit_evidence_preserves_sides_limits_and_aborted_original_custody() {
    for evidence in [
        InterventionEvidence::Preview { max_elements: 3 },
        InterventionEvidence::Summary,
    ] {
        for limited in [false, true] {
            exercise(evidence.clone(), limited);
        }
    }
}
fn exercise(evidence: InterventionEvidence, limited: bool) {
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
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 26, 0).unwrap();
    let request = OriginalSpeculativeRequest::prepare(
        &pool,
        &InferenceExecutionIdentity::default(),
        &schedule,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 26),
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
    let source = capture_with_evidence_budget(if limited { 1 } else { 3 });
    let (discovery, plain) = edits(&source);
    let mut plan = plain.plan().clone();
    plan.operations[0].evidence = evidence.clone();
    let caller = plan
        .admit(&discovery, source.admission().request(), "facade-session")
        .unwrap();
    let copy = PreparedInterventionPlanCopy::inspect(&caller).unwrap();
    let required = MemoryLedger::intervention_source_required_bytes(&copy).unwrap();
    let short = crate::working_memory::memory_fixture::host_ledger(required - 1, 0).unwrap();
    assert!(short.compile_intervention_source(copy).is_err());
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    let edit = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&caller).unwrap())
        .unwrap();
    let companion = edit.plan().evidence(0).unwrap();
    assert_eq!(companion.operation(), 0);
    assert_eq!(
        companion.geometry_source().plan().limits,
        CapturePlan::none().limits
    );
    assert_eq!(
        companion.geometry_source().plan().selections[0].id,
        "scale-é:before:None"
    );
    assert_eq!(
        companion.geometry_source().plan().selections[1].id,
        "scale-é:after:None"
    );
    let declared =
        crate::intervention::evidence_selections(&caller.plan().operations[0], &caller.points()[0]);
    for ((selection, point), (actual_selection, actual_point)) in declared.iter().zip(
        companion
            .geometry_source()
            .plan()
            .selections
            .iter()
            .zip(companion.geometry_source().points()),
    ) {
        assert_eq!(selection, actual_selection);
        assert_eq!(point, actual_point);
    }
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
    let original = [1.0, -2.0, 3.0, 4.0];
    let (budget, mut invocation) = invoke();
    let mut backend = EvidenceBackend {
        edit: EditBackend {
            capture: HostBackend::default(),
            custody: &budget,
            source: &edit,
            refuse: false,
            calls: 0,
        },
        refuse_after: false,
        evidence_calls: 0,
    };
    assert_eq!(
        invocation
            .observe_and_intervene(&mut backend, &original)
            .unwrap()
            .unwrap(),
        [2.0, -4.0, 6.0, 8.0]
    );
    let escaped = invocation.take_shared_step().unwrap().unwrap();
    let record = &escaped.as_ref().interventions[0];
    assert_eq!(record.outcome, InterventionOutcome::Applied);
    assert_eq!(record.evidence.len(), 2);
    assert_eq!(backend.evidence_calls, if limited { 0 } else { 2 });
    assert_eq!(
        escaped.as_ref().step_usage.captures,
        if limited { 1 } else { 3 }
    );
    for (side, row) in record.evidence.iter().enumerate() {
        assert_eq!(record.prediction_index, 1);
        assert_eq!(record.phase, CapturePhase::Decode);
        assert_eq!(row.selected_shape.as_deref(), Some([1, 1, 4].as_slice()));
        if limited {
            assert!(matches!(row.outcome, CaptureOutcome::Skipped { .. }));
            assert!(row.payload.is_none());
            continue;
        }
        let scale = if side == 0 { 1.0 } else { 2.0 };
        match (&evidence, &row.payload) {
            (InterventionEvidence::Preview { .. }, Some(CapturePayload::SharedTensor(value))) => {
                assert_eq!(
                    value.as_ref().data(),
                    &TensorObservationData::F32(original[..3].iter().map(|n| n * scale).collect())
                )
            }
            (InterventionEvidence::Summary, Some(CapturePayload::Summary(value))) => {
                assert_eq!(value.elements, 4);
                assert_eq!(value.min, Some(-2.0 * scale as f64));
                assert_eq!(value.max, Some(4.0 * scale as f64));
                assert_eq!(value.mean, Some(1.5 * scale as f64));
            }
            _ => panic!("actual evidence payload"),
        }
    }
    let (next_budget, mut next) = invoke();
    let mut refused = EvidenceBackend {
        edit: EditBackend {
            capture: HostBackend::default(),
            custody: &next_budget,
            source: &edit,
            refuse: false,
            calls: 0,
        },
        refuse_after: true,
        evidence_calls: 0,
    };
    let result = next.observe_and_intervene(&mut refused, &original);
    let failed = if limited {
        assert!(result.is_ok());
        next.take_shared_step().unwrap().unwrap()
    } else {
        assert!(result.is_err());
        let frame = next.take_failed_evidence().unwrap().unwrap();
        assert_eq!(frame.as_ref().outcome, CaptureStepOutcome::Aborted);
        assert!(matches!(
            frame.as_ref().interventions[0].outcome,
            InterventionOutcome::Failed { .. }
        ));
        assert!(frame.as_ref().interventions[0].evidence[0]
            .payload
            .is_some());
        assert!(matches!(
            frame.as_ref().interventions[0].evidence[1].outcome,
            CaptureOutcome::Failed { .. }
        ));
        frame
    };
    assert_eq!(
        failed.as_ref().cumulative_usage.captures,
        if limited { 2 } else { 6 }
    );
    assert!(next.observe_and_intervene(&mut refused, &original).is_err());
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
