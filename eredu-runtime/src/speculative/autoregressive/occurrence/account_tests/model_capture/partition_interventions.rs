//! The same funded observer receives a genuine AR role and original member source.
use super::*;
use crate::capture::partition::*;
use crate::intervention::PreparedPartitionInterventionProjection;
use eredu_core::consensus::{BoundedConsensusTransport, ConsensusTransport};
use eredu_core::intervention::*;
use eredu_core::{
    BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    CompletionCancellationMode, DistributedCommitEpoch, Submission,
};
use eredu_nn::workspace::HostMetadataFunding;
struct Done;
impl Completion for Done {
    type Error = std::io::Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl BoundedCompletion for Done {
    fn wait_bounded(
        self,
        _: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        Ok(BoundedCompletionOutcome::Completed)
    }
}
struct Local;
impl ConsensusTransport for Local {
    type Error = std::io::Error;
    fn participant_count(&self) -> usize {
        1
    }
    fn all_gather_words(&self, words: &[u32]) -> Result<Vec<u32>, Self::Error> {
        Ok(words.to_vec())
    }
}
impl BoundedConsensusTransport for Local {
    type Completion = Done;
    type GatherOutput = Vec<u32>;
    fn submit_all_gather_words(
        &self,
        words: &[u32],
    ) -> Result<Submission<Vec<u32>, Done>, Self::Error> {
        Ok(Submission {
            output: words.to_vec(),
            completion: Done,
        })
    }
    fn resolve_all_gather_words(&self, output: Vec<u32>) -> Result<Vec<u32>, Self::Error> {
        Ok(output)
    }
}
impl PartitionCaptureTransport for Local {
    fn capture_rank(&self) -> usize {
        0
    }
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError> {
        Ok(BoundedCompletionWait::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap())
    }
    fn ensure_capture_active(&self) -> Result<(), eredu_core::BackendFailure> {
        Ok(())
    }
    fn estimate_capture_gather(&self, words: usize) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            retained_bytes: (words * 4) as u64,
            host_bytes: (words * 4) as u64,
            ..Default::default()
        })
    }
    fn fail_capture_exchange(&self, _: &PartitionCaptureExchangeError) {}
}
impl PartitionCaptureHookTransport for Local {
    type HookOutput = ();
    fn estimate_capture_hook(&self, members: &[usize]) -> Result<CaptureUsage, CaptureError> {
        assert_eq!(members, &[0]);
        Ok(CaptureUsage::default())
    }
    fn submit_capture_hook(
        &self,
        _: &[usize],
        _: bool,
    ) -> Result<Submission<(), Done>, Self::Error> {
        panic!("singleton group has no member collective")
    }
    fn resolve_capture_hook(&self, _: ()) -> Result<bool, Self::Error> {
        unreachable!()
    }
}
struct Model<'a> {
    program: PreparedPartitionCaptureProgram<'a, Local>,
    custody: OriginalSpeculativeBudgetCustody,
    projection: PreparedPartitionInterventionProjection,
    shape: [u64; 2],
    physical: CaptureInvocationShape,
    window: CaptureInvocationWindow,
    usage: CaptureUsage,
    calls: usize,
}
impl ScheduledCaptureBackend for Model<'_> {
    type Tensor = Values;
    type Error = Failure;
    fn validate_source(
        &self,
        _: &Values,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<eredu_core::checkpoint::TensorDtype, Failure> {
        panic!("no capture selections")
    }
    fn estimate(
        &self,
        _: &Values,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        panic!("no capture selections")
    }
    fn transform(
        &mut self,
        _: &Values,
        _: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Failure> {
        panic!("no capture selections")
    }
    fn partition_capture(&mut self) -> Option<&mut (dyn ScheduledPartitionCapture + '_)> {
        Some(&mut self.program)
    }
    fn validate_partition_intervention_source(
        &self,
        value: &Values,
        claim: &CaptureInterventionClaim<'_>,
        window: Option<crate::intervention::InterventionPrefillWindow>,
    ) -> Result<(), FundedCaptureError<Failure>> {
        claim
            .validate_model_custody(&self.custody)
            .map_err(|e| FundedCaptureError::Backend(e.into()))?;
        claim
            .validate_source(self.projection.source())
            .map_err(|e| FundedCaptureError::Backend(e.into()))?;
        assert!(window.is_none());
        assert_eq!(claim.invocation(), Some(self.physical));
        assert_eq!(claim.invocation_window(), Some(self.window));
        assert_eq!(value.rows, self.physical.sequence);
        Ok(())
    }
    fn apply_partition_intervention(
        &mut self,
        value: &Values,
        claim: &CaptureInterventionClaim<'_>,
        allowance: &mut PartitionInterventionLocalAllowance,
    ) -> Result<Option<Values>, FundedCaptureError<Failure>> {
        self.validate_partition_intervention_source(value, claim, None)?;
        allowance
            .validate(
                claim,
                &self.projection,
                None,
                [3; 32],
                &self.shape,
                self.usage,
                [CaptureUsage::default(); 2],
                CaptureUsage::default(),
            )
            .map_err(|_| FundedCaptureError::Backend(Failure::Injected))?;
        allowance
            .charge_projection([CaptureUsage::default(); 2])
            .unwrap();
        allowance.charge_source(CaptureUsage::default()).unwrap();
        allowance.charge_execution(self.usage).unwrap();
        assert!(allowance.charge_execution(self.usage).is_err());
        self.calls += 1;
        Ok(Some(Values {
            rows: value.rows,
            data: value.data.iter().map(|x| x * 2.0).collect(),
        }))
    }
}
fn sources(pool: &MemoryLedger) -> (OriginalCaptureSource, OriginalInterventionSource) {
    sources_with_evidence(pool, InterventionEvidence::None)
}
fn sources_with_evidence(
    pool: &MemoryLedger,
    evidence: InterventionEvidence,
) -> (OriginalCaptureSource, OriginalInterventionSource) {
    let full = capture_source(pool, 3, CaptureTransform::FullTensor, 100);
    let original = full.plan().admission();
    let mut raw = original.plan().clone();
    raw.selections.clear();
    let bounds = original.invocation_bounds().unwrap();
    let admitted = raw
        .admit_invocations(
            &ObservationCatalog {
                schema_version: 1,
                points: vec![],
                completeness: DescriptionCompleteness::Complete,
            },
            &ObservationSupportReport {
                schema_version: 1,
                capture: Default::default(),
                points: vec![],
            },
            &CaptureCapabilities::default(),
            bounds,
        )
        .unwrap();
    let capture = pool
        .compile_capture_source(PreparedCapturePlanCopy::inspect(&admitted).unwrap())
        .unwrap();
    let edits = InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: vec![InterventionOperation {
            id: "scale".into(),
            target: "block.output".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            action: InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 2.0,
            },
            evidence,
        }],
    }
    .admit_invocations(
        &InterventionDiscovery {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            artifact_identity: "fixture".into(),
            session_identity: Some("session".into()),
            points: vec![InterventionPoint {
                path: "block.output".into(),
                node_id: "block".into(),
                stage: InterventionStage::Activation,
                axes: original.points()[0].axes.clone().unwrap(),
                dtypes: vec![InterventionDtype::Float32],
                operations: vec![InterventionKind::Scale],
                score_stages: vec![],
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                conditions: vec![],
                routing: None,
                routed_units: None,
            }],
        },
        bounds,
        "session",
    )
    .unwrap();
    let edits = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&edits).unwrap())
        .unwrap();
    (capture, edits)
}
#[test]
fn autoregressive_partition_intervention_uses_actual_role_and_shared_observer() {
    for (width, enabled, provided) in [
        (2, true, true),
        (3, true, true),
        (1, true, true),
        (1, false, false),
        (1, true, false),
        (1, false, true),
    ] {
        let mask = [enabled];
        let selected = selected();
        let config = SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: 1,
            ..Default::default()
        };
        let schedule = AutoregressiveSchedulePlan::new(
            &selected,
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(width).unwrap(),
            NonZeroU64::new(16).unwrap(),
            &config,
            SpeculativeSchedulerOptions::default(),
        )
        .unwrap();
        let invocation = AutoregressiveInvocation::prefill(
            AutoregressivePass::TargetPrefill,
            usize::try_from(width).unwrap(),
        )
        .unwrap();
        let mut geometry = schedule
            .workspace_geometry(0, invocation, NonZeroU64::new(width).unwrap())
            .unwrap();
        if enabled {
            geometry.output = eredu_core::OutputDemand::Sequence;
        }
        let report = report(geometry);
        let capacity = 1 << 26;
        let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
        let (source, edits) = sources(&pool);
        let baseline = pool.payload_used_bytes().unwrap();
        let execution = InferenceExecutionIdentity::default();
        let request = OriginalSpeculativeRequest::prepare(
            &pool,
            &execution,
            &schedule,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
        let lineage = request.prepare_model_capture_lineage(&source).unwrap();
        let plans = frames(
            &source,
            &lineage,
            invocation,
            report.span_workspace_plan(),
            &[],
        )
        .into_iter()
        .map(|plan| {
            plan.with_interventions(&edits, &mask)
                .unwrap()
                .with_transport_metadata(1 << 20)
                .unwrap()
        })
        .collect::<Vec<_>>();
        let mut cursor = schedule.into_cursor();
        let (role, prepared) = request
            .reserve_role_with_capture(
                cursor.claim(0, invocation).unwrap(),
                requirements(report.span_workspace_plan()),
                AutoregressiveCaptureHostPlan::prepare(&plans).unwrap(),
            )
            .unwrap();
        let mut bank = prepared.construct().unwrap();
        drop(plans);
        role.begin_prefill(&execution, geometry).unwrap();
        let InferenceWorkspaceSpan::Prefill(chunk) =
            report.span_workspace_plan().records()[0].span()
        else {
            unreachable!()
        };
        let span = role.claim_prefill_span(chunk).unwrap();
        let mut owner = bank.begin_prefill(&span).unwrap();
        let metadata = owner.take_partition_metadata().unwrap();
        let transport = Local;
        let setup = crate::establish_communication_session(
            &transport,
            &crate::CommunicationManifest::new(1, 0, vec![], vec![]).unwrap(),
            Some([7; 32]),
        )
        .unwrap()
        .identity();
        let identity = owner
            .prepare_partition_run_identity("fixture", "execution", setup, None, &metadata)
            .unwrap();
        let physical = CaptureInvocationShape {
            batch: 1,
            sequence: width,
            context: None,
        };
        let window = CaptureInvocationWindow {
            logical_sequence: width,
            start: 0,
        };
        let shape = [width, 2];
        let projection = PreparedPartitionInterventionProjection::prepare(
            &edits,
            0,
            CapturePhase::Prefill,
            0,
            Some(physical),
            &shape,
            1,
            &eredu_core::component::ComponentCoordinateMap::range(2, 0..2).unwrap(),
            None,
            2,
            metadata.clone(),
        )
        .unwrap();
        let usage = CaptureUsage {
            retained_bytes: width * 8,
            host_bytes: width * 8,
            ..Default::default()
        };
        let declaration = PreparedPartitionInterventionSource::new_invocation(
            &edits,
            0,
            CapturePhase::Prefill,
            0,
            1,
            physical,
            Some(window),
            &[PartitionInterventionMemberSource {
                rank: 0,
                projection: &projection,
                shape: &shape,
                execution_identity: [3; 32],
                usage,
                projection_usage: [CaptureUsage::default(); 2],
                source_usage: CaptureUsage::default(),
            }],
            &metadata,
        )
        .unwrap();
        let mut program = PreparedPartitionCaptureProgram::new_selected_for_run_at(
            &transport,
            source.plan(),
            &identity,
            CapturePhase::Prefill,
            0,
            Some(physical),
            Some(eredu_core::capture::PartitionCaptureInvocationWindow { physical, window }),
            &[],
            PartitionCaptureReceiptLimits {
                max_producers: 1,
                max_fragments: 1,
                max_record_bytes: 1 << 20,
            },
            &metadata,
        )
        .unwrap();
        program
            .set_intervention_sources(&edits, std::iter::once(provided.then_some(declaration)))
            .unwrap();
        let mut backend = Model {
            program,
            custody: role.budget_custody(),
            projection,
            shape,
            physical,
            window,
            usage,
            calls: 0,
        };
        let input = Values {
            rows: width,
            data: (0..width * 2).map(|i| i as f32 * 0.375 - 1.0).collect(),
        };
        let result = owner
            .with_observer(&mut backend, &|e| e, |observer| {
                let guard = crate::inspection::ObservationTransactionGuard::new(
                    observer,
                    DistributedCommitEpoch::FIRST,
                );
                guard.observer.prepare_transaction(
                    DistributedCommitEpoch::FIRST,
                    crate::ExpertPass::Prefill,
                )?;
                guard
                    .observer
                    .coordinate_transaction(DistributedCommitEpoch::FIRST)?;
                let output = guard.observer.intervene("block.output", &input)?;
                if enabled {
                    assert_eq!(
                        output.unwrap().data,
                        input.data.iter().map(|x| x * 2.0).collect::<Vec<_>>()
                    );
                } else {
                    assert!(output.is_none());
                }
                guard
                    .observer
                    .complete_transaction(DistributedCommitEpoch::FIRST)?;
                guard.finish(true);
                Ok::<_, FundedCaptureError<Failure>>(())
            })
            .unwrap();
        if enabled != provided {
            assert!(
                result.is_err(),
                "source population must match the actual admitted mask"
            );
            assert_eq!(backend.calls, 0, "refuse before the numerical edit");
            drop(result);
            drop((backend, identity, metadata, owner, span, bank));
            request.close().unwrap();
            drop((role, request, lineage));
            assert_eq!(pool.payload_used_bytes().unwrap(), baseline);
            drop((source, edits));
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
            continue;
        }
        result.unwrap();
        assert_eq!(backend.calls, usize::from(enabled));
        let frame = owner.take_shared_step().unwrap().unwrap();
        assert_eq!(
            frame.interventions()[0].outcome,
            if enabled {
                InterventionOutcome::Applied
            } else {
                InterventionOutcome::Inactive
            }
        );
        let spent = owner.usage();
        assert!(spent.host_bytes > 0);
        drop((frame, backend, identity, metadata, owner, span, bank));
        request.close().unwrap();
        drop((role, request, lineage));
        assert_eq!(pool.payload_used_bytes().unwrap(), baseline);
        drop((source, edits));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

fn evidence_hosts(
    parent: &OriginalCaptureSource,
    source: &OriginalInterventionSource,
    metadata: &HostMetadataFunding,
) -> [OwnedPartitionFragmentHostPlan; 2] {
    let companion = source.plan().evidence(0).unwrap().shared_geometry_source();
    std::array::from_fn(|side| {
        let context = PartitionCaptureContext {
            artifact_identity: "fixture".into(),
            execution_identity: "execution".into(),
            run_identity: "actual-evidence-run".into(),
            overlay_identity: None,
            capture_plan_identity: companion.admission().identity().into(),
            selection_index: side,
            phase: CapturePhase::Decode,
            prediction: 0,
            forward_epoch: 1,
            invocation: Some(CaptureInvocationShape {
                batch: 1,
                sequence: 1,
                context: None,
            }),
            invocation_window: None,
        };
        let mut ledger = CaptureLedger::new(parent.plan().admission());
        ledger.begin_step();
        let receipt = PartitionCaptureReceiptPlan::new_complete_shared_funded(
            companion,
            &context,
            0,
            1,
            PartitionCaptureReceiptLimits {
                max_producers: 1,
                max_fragments: 1,
                max_record_bytes: 4096,
            },
            metadata,
            &mut ledger,
        )
        .unwrap();
        OwnedPartitionFragmentHostPlan::prepare(receipt, None).unwrap()
    })
}
#[test]
fn autoregressive_evidence_hosts_authenticate_companions_and_retain_the_same_role() {
    use eredu_nn::workspace::WorkspaceMetadataAllocation;
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
    let invocation = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let report = report(
        schedule
            .workspace_geometry(2, invocation, NonZeroU64::new(1).unwrap())
            .unwrap(),
    );
    let capacity = 1 << 26;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let limits = crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity);
    let metadata = pool
        .prepare_workspace_metadata(&execution, limits.clone())
        .unwrap();
    let (source, edits) = sources_with_evidence(&pool, InterventionEvidence::Summary);
    let (foreign_capture, foreign) = sources_with_evidence(&pool, InterventionEvidence::Summary);
    let request =
        OriginalSpeculativeRequest::prepare(&pool, &execution, &schedule, limits).unwrap();
    let lineage = request.prepare_model_capture_lineage(&source).unwrap();
    let make_frame = || {
        frames(
            &source,
            &lineage,
            invocation,
            report.span_workspace_plan(),
            &[],
        )
        .pop()
        .unwrap()
        .with_intervention_evidence(&edits, &[true], Some(&[[None, None]]))
        .unwrap()
    };
    let foreign_plans = evidence_hosts(&foreign_capture, &foreign, &metadata);
    let before = pool.payload_used_bytes().unwrap();
    let wrong = make_frame().with_partition_evidence_fragments(&edits, vec![(0, foreign_plans)]);
    assert!(matches!(
        wrong,
        Err(CaptureRunHostError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        before,
        "source refusal does not admit a role"
    );
    let [before, after] = evidence_hosts(&source, &edits, &metadata);
    let swapped =
        make_frame().with_partition_evidence_fragments(&edits, vec![(0, [after, before])]);
    assert!(matches!(
        swapped,
        Err(CaptureRunHostError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let frame = make_frame();
    let base = frame.initialization_peak_bytes();
    let sides = evidence_hosts(&source, &edits, &metadata);
    let peak = sides
        .iter()
        .map(|plan| plan.initialization_peak_bytes())
        .sum::<u64>();
    assert!(peak > 0);
    let mut rows = metadata.metadata_vec(1).unwrap();
    rows.push((0, sides));
    let frame = frame
        .with_partition_evidence_fragments(&edits, rows)
        .unwrap();
    assert!(frame.initialization_peak_bytes() > base + peak);
    let plans = [frame];
    let mut cursor = schedule.into_cursor();
    let (role, prepared) = request
        .reserve_role_with_capture(
            cursor.claim(2, invocation).unwrap(),
            requirements(report.span_workspace_plan()),
            AutoregressiveCaptureHostPlan::prepare(&plans).unwrap(),
        )
        .unwrap();
    let admitted = pool.payload_used_bytes().unwrap();
    let mut bank = prepared.construct().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), admitted);
    let mut owner = bank.begin_decode().unwrap();
    let hosts = owner.take_partition_evidence_fragments().unwrap();
    assert_eq!(hosts.len(), 1);
    assert_eq!(hosts[0].0, 0);
    assert!(owner.take_partition_evidence_fragments().is_none());
    assert_eq!(pool.payload_used_bytes().unwrap(), admitted);
    drop((owner, bank, plans, role));
    request.close().unwrap();
    drop((
        request,
        lineage,
        source,
        edits,
        foreign_capture,
        foreign,
        metadata,
    ));
    assert!(
        pool.payload_used_bytes().unwrap() > 0,
        "both real evidence Host tokens retain their original role account"
    );
    drop(hosts);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
