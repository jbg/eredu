#[path="interventions/evidence.rs"]
mod evidence;
use super::*;
use eredu_core::intervention::*;

fn edit_source(
    pool: &WorkingMemoryPool,
    capture: &OriginalCaptureSource,
) -> OriginalInterventionSource {
    edit_source_window(pool,capture,false)
}
fn edit_source_window(pool:&WorkingMemoryPool,capture:&OriginalCaptureSource,no_overlap:bool)->OriginalInterventionSource {
    edit_source_evidence(pool,capture,no_overlap,false)
}
fn edit_source_evidence(pool:&WorkingMemoryPool,capture:&OriginalCaptureSource,no_overlap:bool,evidence:bool)->OriginalInterventionSource {
    let axes = capture.plan().admission().points()[0].axes.clone().unwrap();
    let points = ["late", "early", "disabled"].map(|path| InterventionPoint {
        path: path.into(),
        node_id: path.into(),
        stage: InterventionStage::Activation,
        axes: axes.clone(),
        dtypes: vec![InterventionDtype::Float32],
        operations: vec![InterventionKind::Scale],
        score_stages: vec![],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
        routing: None,
        routed_units: None,
    });
    let discovery = InterventionDiscovery {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        artifact_identity: "actual neutral declaration".into(),
        session_identity: Some("loaded session".into()),
        points: points.into(),
    };
    let operations = [
        ("late", 2.0),
        ("early", 3.0),
        ("early", -1.0),
        ("disabled", 100.0),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (target, factor))| InterventionOperation {
        id: index.to_string(),
        target: target.into(),
        schedule: Default::default(),
        slices: if no_overlap && index==0 { vec![CaptureSlice { axis:"sequence".into(), start:3, end:5, stride:1 }] } else {vec![]},
        action: InterventionAction::Scale {
            dtype: InterventionDtype::Float32,
            factor,
        },
        evidence: if !evidence {InterventionEvidence::None} else if index==1 {InterventionEvidence::Summary} else {InterventionEvidence::Preview {max_elements:3}},
    })
    .collect();
    let plan = InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations,
    }
    .admit_invocations(
        &discovery,
        capture.plan().admission().invocation_bounds().unwrap(),
        "loaded session",
    )
    .unwrap();
    pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&plan).unwrap())
        .unwrap()
}
struct EditBackend {
    capture: Backend,
    source: OriginalInterventionSource,
    foreign_source: OriginalInterventionSource,
    foreign_role: OriginalSpeculativeBudgetCustody,
    order: Vec<usize>,
    window: Option<CaptureInvocationWindow>,
    acknowledged: usize,
}
impl ScheduledCaptureBackend for EditBackend {
    type Tensor = [f32; 6];
    type Error = Failure;
    fn validate_source(
        &self,
        value: &Self::Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<eredu_core::checkpoint::TensorDtype, Failure> {
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
    ) -> Result<ClaimedCaptureTensor, Failure> {
        self.capture.transform(value, claim)
    }
    fn intervention_usage(
        &self,
        _: &Self::Tensor,
        claim: &CaptureInterventionClaim<'_>,
    ) -> Result<CaptureUsage, FundedCaptureError<Failure>> {
        claim
            .validate_model_custody(&self.capture.custody)
            .map_err(|cause| FundedCaptureError::Backend(Failure::from(cause)))?;
        claim
            .validate_source(&self.source)
            .map_err(|cause| FundedCaptureError::Backend(Failure::from(cause)))?;
        assert!(claim.validate_model_custody(&self.foreign_role).is_err());
        assert!(claim.validate_source(&self.foreign_source).is_err());
        assert_eq!(
            claim.invocation(),
            Some(CaptureInvocationShape {
                batch: 1,
                sequence: 3,
                context: None
            })
        );
        assert_eq!(claim.invocation_window(),self.window);
        if self.window.is_some() && claim.index()==0 {return Ok(CaptureUsage::default());}
        Ok(CaptureUsage {
            retained_bytes: 24,
            host_bytes: 24,
            ..Default::default()
        })
    }
    fn intervention_projection_usage(&self,value:&Self::Tensor,claim:&CaptureInterventionClaim<'_>)->Result<[CaptureUsage;2],FundedCaptureError<Failure>> {
        self.intervention_usage(value,claim)?;
        let Some(window)=self.window else {return Ok([CaptureUsage::default();2]);};
        let index=claim.index(); let plan=claim.admission();
        let operation=&plan.plan().operations[index]; let point=&plan.points()[index];
        let physical=claim.invocation().unwrap();
        let mut global=[0;2];
        let (logical,axis)=window.source_axes_into(physical,Some(&point.axes),&[3,2],&mut global).unwrap();
        let buffer=||ResolvedCaptureSlice {starts:vec![0;2],ends:vec![0;2],strides:vec![0;2],shape:vec![0;2]};
        let mut selected=buffer();let mut local=buffer();let mut destination=buffer();
        plan.resolve_prepared_invocation_at(index,CapturePhase::Prefill,0,logical,&global,InterventionDtype::Float32,&mut selected).unwrap();
        let overlap=CaptureSlicePartition::contiguous_fragment_into(&global,&selected,axis,window.start..window.start+physical.sequence,&mut local,&mut destination).unwrap();
        assert_eq!(overlap,index!=0);
        let mut cost=crate::intervention::PartitionInterventionProjectionCost::new(plan,2)?;
        if overlap {cost.include(&operation.action,&local.shape)?;}
        Ok([crate::intervention::intervention_window_metadata(operation,point)?,cost.usage()])
    }
    fn apply_intervention_projected(&mut self,value:&Self::Tensor,claim:CaptureInterventionClaim<'_>,charged:CaptureUsage,projection:[CaptureUsage;2])->Result<(Option<Self::Tensor>,ClaimedIntervention),FundedCaptureError<Failure>> {
        if self.window.is_none() {return self.apply_intervention(value,claim,charged).map(|(value,receipt)|(Some(value),receipt));}
        assert_eq!(self.intervention_projection_usage(value,&claim)?,projection);
        let total=charged.checked_add(projection[0])?.checked_add(projection[1])?;
        if claim.index()==0 {
            assert_eq!(charged,CaptureUsage::default());self.acknowledged+=1;
            return Ok((None,claim.finish(total)?));
        }
        let InterventionAction::Scale {factor,..}=&claim.admission().plan().operations[claim.index()].action else {panic!("scale")};
        self.order.push(claim.index());
        Ok((Some(value.map(|value|value* *factor)),claim.finish(total)?))
    }
    fn intervention_evidence_usage(&self,value:&Self::Tensor,claim:&CaptureInterventionEvidenceClaim<'_, '_>)->Result<(eredu_core::checkpoint::TensorDtype,CaptureUsage),FundedCaptureError<Failure>> {
        evidence::usage(self,value,claim)
    }
    fn capture_intervention_evidence<'a>(&mut self,value:&Self::Tensor,claim:CaptureInterventionEvidenceClaim<'a,'_>)->Result<ClaimedInterventionEvidence<'a>,FundedCaptureError<Failure>> {
        evidence::capture(self,value,claim)
    }
    fn apply_intervention(
        &mut self,
        value: &Self::Tensor,
        claim: CaptureInterventionClaim<'_>,
        charged: CaptureUsage,
    ) -> Result<(Self::Tensor, ClaimedIntervention), FundedCaptureError<Failure>> {
        self.intervention_usage(value, &claim)?;
        let InterventionAction::Scale { factor, .. } =
            &claim.admission().plan().operations[claim.index()].action
        else {
            panic!("scale")
        };
        self.order.push(claim.index());
        let value = value.map(|v| v * *factor);
        Ok((value, claim.finish(charged)?))
    }
}

#[test]
fn model_intervention_claims_keep_hook_order_scope_exact_custody_and_escaped_outcomes() {
    run(false);
}
#[test]
fn model_window_interventions_charge_projection_and_acknowledge_no_overlap_without_tensor() {
    run(true);
}
fn run(window:bool) {run_mode(window,false);}
fn run_mode(window:bool,evidence:bool) {
    let window=window.then_some(CaptureInvocationWindow {logical_sequence:5,start:0});
    let selected = selected(SpeculativeStrategyClass::EmbeddedSequential);
    let schedule = plan(&selected, 3, true);
    let (invocation, _) = schedule.prefill_invocations(0).unwrap();
    let workspace = EmbeddedInvocationWorkspace::target(invocation).unwrap();
    let report = report(workspace.geometry());
    let capacity = 1 << 26;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let source = capture_source_with_transform(&pool,if window.is_some() {5} else {3},CaptureTransform::FullTensor,if evidence {5} else {1});
    let edits = edit_source_evidence(&pool, &source,window.is_some(),evidence);
    let foreign_source = edit_source_evidence(&pool, &source,window.is_some(),evidence);
    assert!(!edits.same_source(&foreign_source));
    let request = OriginalSpeculativeRequest::prepare_embedded(
        &pool,
        &InferenceExecutionIdentity::default(),
        &schedule,
        capacity,
    )
    .unwrap();
    let mut cursor = schedule.into_cursor();
    let foreign_role = request
        .reserve_embedded_role(
            cursor.claim(invocation).unwrap(),
            workspace,
            requirements(report.span_workspace_plan()),
        )
        .unwrap();
    let shape = CaptureInvocationShape {
        batch: 1,
        sequence: 3,
        context: None,
    };
    let origin = eredu_core::speculative::SpeculativeActivationOrigin {
        request: SpeculativeRequestId::new(81),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [0; 32],
        optimistic: false,
    };
    let selected_captures = [true];
    let selected_edits = [true, true, true, false];
    let host = EmbeddedCaptureHostPlan::prepare(
        &source,
        CaptureRunHostPlan::prepare_invocation_window(
            source.plan(),
            CapturePhase::Prefill,
            0,
            shape,
            &selected_captures,
            window,
        )
        .unwrap(),
        workspace,
        origin,
    )
    .unwrap();
    let without_edits = host.initialization_peak_bytes();
    let mut evidence_skips: [[Option<CaptureSkipReason>;2];4]=std::array::from_fn(|_|[None,None]);
    evidence_skips[0][0]=Some(CaptureSkipReason::Limit {budget:CaptureBudget::Captures,cumulative:true});
    let host = if evidence {
        assert!(EmbeddedCaptureHostPlan::prepare(&source,CaptureRunHostPlan::prepare_invocation_window(source.plan(),CapturePhase::Prefill,0,shape,&selected_captures,window).unwrap(),workspace,origin).unwrap().with_intervention_evidence(&edits,&selected_edits,Some(&evidence_skips[..3])).is_err());
        host.with_intervention_evidence(&edits,&selected_edits,Some(&evidence_skips)).unwrap()
    } else {host.with_interventions(&edits,&selected_edits).unwrap()};
    assert!(host.initialization_peak_bytes() > without_edits);
    let (role, pending) = request
        .reserve_embedded_role_with_capture(
            cursor.claim(invocation).unwrap(),
            workspace,
            requirements(report.span_workspace_plan()),
            host,
        )
        .unwrap();
    let mut owner = pending.begin().unwrap();
    let mut backend = EditBackend {
        capture: Backend {
            custody: role.budget_custody(),
            calls: 0,
        },
        source: edits.clone(),
        foreign_source,
        foreign_role: foreign_role.budget_custody(),
        order: vec![],
        window,
        acknowledged:0,
    };
    let input = [0.5, -1.0, 2.25, 3.5, -4.0, 5.75];
    let epoch = eredu_core::DistributedCommitEpoch::new(33).unwrap();
    owner
        .with_observer(&mut backend, &|cause| cause, |observer| {
            let guard = crate::inspection::ObservationTransactionGuard::new(observer, epoch);
            guard
                .observer
                .prepare_transaction(epoch, crate::ExpertPass::Prefill)?;
            // Graph order differs from declaration order across distinct hooks.
            let early = guard.observer.intervene("early", &input)?.unwrap();
            let late = guard.observer.intervene("late", &early)?;
            assert_eq!(late.is_none(),window.is_some());
            let output=late.as_ref().unwrap_or(&early);
            assert!(guard.observer.intervene("disabled", output)?.is_none());
            assert_eq!(*output, input.map(|v| v * if window.is_some() {-3.0} else {-6.0}));
            guard.observer.observe("block.output", output)?;
            guard.observer.complete_transaction(epoch)?;
            guard.finish(true);
            Ok::<_, FundedCaptureError<Failure>>(())
        })
        .unwrap()
        .unwrap();
    assert_eq!(backend.order,if window.is_some() {vec![1,2]} else {vec![1,2,0]});
    assert_eq!(backend.acknowledged,usize::from(window.is_some()));
    let frame = owner.take_shared_step().unwrap().unwrap();
    assert_eq!(
        frame
            .as_ref()
            .interventions
            .iter()
            .map(|r| &r.outcome)
            .collect::<Vec<_>>(),
        vec![
            &InterventionOutcome::Applied,
            &InterventionOutcome::Applied,
            &InterventionOutcome::Applied,
            &InterventionOutcome::Inactive
        ]
    );
    if window.is_some() {
        let plan=edits.plan().admission();
        let operation=&plan.plan().operations[0];let point=&plan.points()[0];
        let initial=crate::intervention::intervention_metadata(operation,point,plan.identity()).unwrap();
        let metadata=crate::intervention::intervention_window_metadata(operation,point).unwrap();
        let projection=crate::intervention::PartitionInterventionProjectionCost::new(plan,2).unwrap().usage();
        assert!(metadata.encoded_bytes>0);
        assert_eq!(frame.as_ref().interventions[0].charged,initial.checked_add(metadata).unwrap().checked_add(projection).unwrap());
    }
    assert_eq!(owner.usage().captures, if evidence {5} else {1});
    if evidence {evidence::check(&frame,&input);}
    let used = owner.usage();
    assert!(
        owner
            .with_observer(&mut backend, &|cause| cause, |observer| observer
                .prepare_transaction(
                    eredu_core::DistributedCommitEpoch::new(34).unwrap(),
                    crate::ExpertPass::Prefill
                ))
            .unwrap()
            .is_err()
    );
    assert_eq!(owner.usage(), used);
    request.close().unwrap();
    drop((owner, backend, role, foreign_role, request));
    let retained = pool.used_bytes().unwrap();
    let tensor = frame.records()[0]
        .payload
        .as_ref()
        .unwrap()
        .as_tensor()
        .unwrap();
    assert!(
        matches!(tensor.data(),TensorObservationData::F32(values) if values.as_slice()==input.map(|v|v * if window.is_some() {-3.0} else {-6.0}).as_slice())
    );
    drop(frame);
    assert!(pool.used_bytes().unwrap() < retained);
    drop((edits, source));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
