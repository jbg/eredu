//! Exact numerical claim custody, real fixed destinations and escaped records.
use super::*;
use crate::capture::FundedCaptureError;
struct ReadoutBackend {
    expected: OriginalSpeculativeNumericalBudgetCustody,
    foreign: OriginalSpeculativeNumericalBudgetCustody,
}
fn domain() -> CandidateDomain {
    CandidateDomain {
        allowed_tokens: 3,
        vocabulary: 4,
        constrained: false,
    }
}
impl ScheduledCaptureBackend for ReadoutBackend {
    type Tensor = [f32; 4];
    type Error = CaptureRunHostError;
    fn validate_source(
        &self,
        _: &Self::Tensor,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Self::Error> {
        unreachable!()
    }
    fn estimate(
        &self,
        _: &Self::Tensor,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        unreachable!()
    }
    fn transform(
        &mut self,
        _: &Self::Tensor,
        _: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Self::Error> {
        unreachable!()
    }
    fn validate_histogram_source(
        &self,
        _: &Self::Tensor,
        geometry: &CaptureHistogramGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        assert_eq!(geometry.source_shape(), &[1, 1, 4]);
        Ok(TensorDtype::F32)
    }
    fn estimate_histogram(
        &self,
        _: &CaptureHistogramGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 128,
            host_bytes: 128,
            encoded_bytes: 8192,
        })
    }
    fn transform_histogram(
        &mut self,
        _: &Self::Tensor,
        claim: CaptureHistogramClaim<'_, '_>,
    ) -> Result<ClaimedCaptureHistogram, FundedCaptureError<Self::Error>> {
        assert!(matches!(
            claim.validate_numerical_custody(&self.foreign),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        claim.validate_numerical_custody(&self.expected).unwrap();
        let mut output = claim.prepare().unwrap();
        output.add_bin(0, 1).unwrap();
        output.add_bin(1, 2).unwrap();
        Ok(output.finish_numerical(&self.expected, 1, 0, 0).unwrap())
    }
    fn validate_summary_source(
        &self,
        _: &Self::Tensor,
        geometry: &CaptureSummaryGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        assert_eq!(geometry.source_shape(), &[1, 1, 4]);
        Ok(TensorDtype::F32)
    }
    fn estimate_summary(
        &self,
        _: &CaptureSummaryGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 128,
            host_bytes: 128,
            encoded_bytes: 8192,
        })
    }
    fn transform_summary(
        &mut self,
        _: &Self::Tensor,
        claim: CaptureSummaryClaim<'_, '_>,
    ) -> Result<ClaimedCaptureSummary, FundedCaptureError<Self::Error>> {
        assert!(matches!(
            claim.validate_numerical_custody(&self.foreign),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        claim.validate_numerical_custody(&self.expected).unwrap();
        Ok(claim
            .finish_numerical(
                &self.expected,
                CaptureSummary {
                    elements: 4,
                    finite: 4,
                    non_finite: 0,
                    nan: 0,
                    positive_infinity: 0,
                    negative_infinity: 0,
                    min: Some(0.0),
                    max: Some(3.0),
                    mean: Some(1.5),
                    rms: Some(3.5f64.sqrt()),
                },
            )
            .unwrap())
    }
    fn validate_candidate_source(
        &self,
        _: &Self::Tensor,
        geometry: &CaptureCandidateGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        assert_eq!(geometry.source_shape(), &[1, 1, 4]);
        Ok(TensorDtype::F32)
    }
    fn validate_token_score_source(
        &self,
        _: &Self::Tensor,
        geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        assert_eq!(geometry.source_shape(), &[1, 1, 4]);
        Ok(TensorDtype::F32)
    }
    fn estimate_candidates(
        &self,
        _: &CaptureCandidateGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 16,
            host_bytes: 64,
            encoded_bytes: 8192,
        })
    }
    fn estimate_token_scores(
        &self,
        _: &CaptureTokenScoreGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 16,
            host_bytes: 128,
            encoded_bytes: 8192,
        })
    }
    fn transform_candidates(
        &mut self,
        values: &Self::Tensor,
        claim: CaptureCandidateClaim<'_, '_>,
    ) -> Result<ClaimedCaptureCandidates, FundedCaptureError<Self::Error>> {
        assert!(matches!(
            claim.validate_numerical_custody(&self.foreign),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        claim.validate_numerical_custody(&self.expected).unwrap();
        let mut output = claim.prepare(Some(domain())).unwrap();
        output.push(3, values[3], false).unwrap();
        output.push(2, values[2], true).unwrap();
        Ok(output.finish().unwrap())
    }
    fn transform_token_scores(
        &mut self,
        values: &Self::Tensor,
        claim: CaptureTokenScoreClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTokenScores, FundedCaptureError<Self::Error>> {
        assert!(matches!(
            claim.validate_numerical_custody(&self.foreign),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        claim.validate_numerical_custody(&self.expected).unwrap();
        let mut output = claim.prepare(Some(domain())).unwrap();
        let partition = values.iter().map(|&x| f64::from(x).exp()).sum::<f64>().ln();
        output
            .push(CaptureTokenScore {
                target: CaptureCandidate {
                    token_id: 0,
                    score: values[0],
                    allowed: true,
                },
                log_probability: f64::from(values[0]) - partition,
                rank: 4,
                strongest_alternative: Some(CaptureCandidate {
                    token_id: 3,
                    score: values[3],
                    allowed: false,
                }),
            })
            .unwrap();
        Ok(output.finish(partition).unwrap())
    }
}
#[test]
fn numerical_readout_claims_require_exact_phase_and_keep_escaped_payload_custody() {
    for transform in [
        CaptureTransform::TopCandidates { count: 2 },
        CaptureTransform::TokenScores { token_ids: vec![0] },
    ] {
        check(transform);
    }
}

#[test]
fn numerical_histogram_claim_requires_exact_phase_and_keeps_escaped_bin_custody() {
    check(CaptureTransform::Histogram {
        edges: vec![0.5, 1.5, 3.0],
    });
}

#[test]
fn numerical_summary_claim_requires_exact_phase_and_keeps_escaped_payload_custody() {
    check(CaptureTransform::Summary);
}

fn check(transform: CaptureTransform) {
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let plan = AutoregressiveSchedulePlan::new(
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
        plan.workspace_geometry(2, pass, NonZeroU64::new(1).unwrap())
            .unwrap(),
    );
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 24, 0).unwrap();
    let request = OriginalSpeculativeRequest::prepare(
        &pool,
        &InferenceExecutionIdentity::default(),
        &plan,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 24),
    )
    .unwrap();
    let mut cursor = plan.into_cursor();
    let role = request
        .reserve_role(
            cursor.claim(2, pass).unwrap(),
            requirements(report.span_workspace_plan()),
        )
        .unwrap();
    let model = role.budget_custody();
    let source = capture_source_with_transform(transform);
    let (budget, mut invocation) = invoke(&request, &source, &model, 1);
    let (foreign, unused) = invoke(&request, &source, &model, 1);
    let mut backend = ReadoutBackend {
        expected: budget.clone(),
        foreign: foreign.clone(),
    };
    invocation
        .observe(&mut backend, &[0.0, 1.0, 2.0, 3.0])
        .unwrap();
    let frame = invocation.take_shared_step().unwrap().unwrap();
    assert!(invocation.take_shared_step().unwrap().is_none());
    assert_eq!(frame.as_ref().cumulative_usage.captures, 1);
    assert_eq!(frame.as_ref().outcome, CaptureStepOutcome::Untracked);
    assert_eq!(frame.as_ref().records[0].outcome, CaptureOutcome::Captured);
    match frame.as_ref().records[0].payload.as_ref().unwrap() {
        CapturePayload::Candidates(values) => {
            assert_eq!(values.domain, Some(domain()));
            assert_eq!(
                values
                    .candidates
                    .iter()
                    .map(|v| (v.token_id, v.allowed))
                    .collect::<Vec<_>>(),
                [(3, false), (2, true)]
            );
        }
        CapturePayload::TokenScores(values) => {
            assert_eq!(values.domain, Some(domain()));
            assert_eq!(values.scores[0].target.token_id, 0);
            assert!(
                !values.scores[0]
                    .strongest_alternative
                    .as_ref()
                    .unwrap()
                    .allowed
            );
        }
        CapturePayload::Histogram(histogram) => {
            assert_eq!(histogram.edges, [0.5, 1.5, 3.0]);
            assert_eq!(histogram.counts, [1, 2]);
            assert_eq!(
                (histogram.below, histogram.above, histogram.non_finite),
                (1, 0, 0)
            );
        }
        CapturePayload::Summary(summary) => {
            assert_eq!(summary.elements, 4);
            assert_eq!(summary.finite, 4);
            assert_eq!(summary.mean, Some(1.5));
            assert_eq!(summary.rms, Some(3.5f64.sqrt()));
        }
        _ => panic!("typed readout payload"),
    }
    request.close().unwrap();
    drop((
        backend, unused, foreign, invocation, budget, model, role, request, source,
    ));
    assert!(pool.payload_used_bytes().unwrap() > 0);
    drop(frame);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
