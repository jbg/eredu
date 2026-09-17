use super::*;
pub(super) mod fragments;
pub(super) mod replica;
pub(super) mod generated;
pub(super) mod interventions;
pub(super) mod routed;
mod prefill;
use prefill::PrefillAggregation;

enum Frame<'a> {
    Before(Option<&'a mut PreparedCaptureRun>),
    Active(ScheduledCaptureStep<'a>),
    Sealed(PreparedCaptureDelivery),
    Empty,
    SealedEmpty,
    Finished,
}
// Only borrowed backend/error mapper handles are stored. All payload-bearing
// frame states precede borrowed session/bank lifetime release on unwind.
pub(super) struct FundedCaptureObserver<'a, T, E: std::error::Error + Send + Sync + 'static, N> {
    frame: Frame<'a>,
    session: &'a mut CaptureSession,
    delivery: &'a mut Option<FundedDelivery>,
    backend: &'a mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
    map_error: &'a dyn Fn(FundedCaptureError<E>) -> N,
    prediction: u64,
    invocation: Option<(CapturePhase, CaptureInvocationShape)>,
    window: Option<CaptureInvocationWindow>,
    // First logical epoch; annotated chunks retain independent native epochs.
    epoch: Option<DistributedCommitEpoch>,
    prefill: Option<PrefillAggregation>,
    bound: Option<crate::layered::BoundCaptureSelection<'a>>,
    admitted: Option<&'a crate::working_memory::AdmittedPrefillCapture<'a>>,
    continuation: Option<&'a crate::working_memory::AdmittedCaptureContinuation<'a>>,
    continuation_announced: bool,
    continuation_epoch: Option<DistributedCommitEpoch>,
    untracked: bool,
    envelope_usage: Option<CaptureUsage>,
    prepaid_metadata: bool,
    invocation_readout: Option<bool>,
    partition_delivered: bool,
    routed_selection: Option<usize>,
    routed_intervention_selection:Option<usize>,
    routed_active: bool,
    routed_partition_hooks:Option<crate::capture::partition::PartitionCaptureRoutedHooks>,
}
impl<'a, T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'a, T, E, N> {
    fn stage_error(&self, error: FundedCaptureError<E>, stage: &'static str) -> N {
        let error = match error {
            FundedCaptureError::Protocol(CaptureProtocolError::Transaction) =>
                FundedCaptureError::Protocol(CaptureProtocolError::TransactionPhase(stage)),
            other => other,
        };
        (self.map_error)(error)
    }

    pub(super) fn new(
        session: &'a mut CaptureSession,
        delivery: &'a mut Option<FundedDelivery>,
        run: &'a mut PreparedCaptureRun,
        backend: &'a mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
        prediction: u64,
        map_error: &'a dyn Fn(FundedCaptureError<E>) -> N,
        bound: Option<crate::layered::BoundCaptureSelection<'a>>,
        admitted: Option<&'a crate::working_memory::AdmittedPrefillCapture<'a>>,
        continuation: Option<&'a crate::working_memory::AdmittedCaptureContinuation<'a>>,
        untracked: bool,
        envelope_usage: Option<CaptureUsage>,
        prepaid_metadata: bool,
        invocation_readout: Option<bool>,
    ) -> Self {
        let invocation = run.invocation();
        let window = run.invocation_window();
        Self {
            frame: Frame::Before(Some(run)),
            invocation,
            window,
            session,
            delivery,
            backend,
            prediction,
            map_error,
            epoch: None,
            prefill: None,
            bound,
            admitted,
            continuation,
            continuation_announced: false,
            continuation_epoch: None,
            untracked,
            envelope_usage,
            prepaid_metadata,
            invocation_readout,
            partition_delivered: false,
            routed_selection: None,
            routed_intervention_selection:None,
            routed_active: false,
            routed_partition_hooks:None,
        }
    }
    fn prepare(
        &mut self,
        epoch: DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), FundedCaptureError<E>> {
        if self.prefill.is_some() {
            return self.prepare_chunk(epoch, pass);
        }
        if self.bound.is_some() {
            return Err(CaptureProtocolError::Transaction.into());
        }
        self.prepare_first(epoch, pass)
    }
    fn prepare_first(
        &mut self,
        epoch: DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), FundedCaptureError<E>> {
        if !matches!(self.frame, Frame::Before(Some(_))) || self.epoch.is_some() {
            return Err(CaptureProtocolError::Transaction.into());
        }

        self.session.claim_prepared_step_epoch(
            epoch,
            self.delivery.is_some(),
            self.invocation.map(|v| v.1),
        )?;
        self.epoch = Some(epoch);
        self.session.invocation_window = self.window;

        self.session.validate_step_start(self.prediction)?;
        let phase = if let Some(continuation) = self.continuation {
            if pass != crate::ExpertPass::Prefill
                || !self.continuation_announced
                || self.continuation_epoch != Some(epoch)
                || self.prediction != continuation.first_prediction()
            {
                return Err(CaptureProtocolError::Transaction.into());
            }
            CapturePhase::Decode
        } else {
            match pass {
                crate::ExpertPass::Prefill => CapturePhase::Prefill,
                crate::ExpertPass::Decode => CapturePhase::Decode,
            }
        };
        let Frame::Before(bank) = &mut self.frame else {
            unreachable!()
        };
        let bank = bank.take().expect("one prepared bank borrow");

        bank.validate_session_account()
            .map_err(CaptureRunHostError::from)?;
        // Only a schedule with no paid frame constructors may skip the claim.
        // Intervention-only plans still own outcome rows for every prediction,
        // including inactive operations, and use the ordinary frame delivery.
        let claim = if !bank.has_frame_claims() {
            let expected = if self.session.has_step {
                self.session
                    .prediction
                    .checked_add(1)
                    .ok_or(CaptureRunHostError::Coordinate)?
            } else {
                0
            };
            let expected_phase = if expected == 0 {
                CapturePhase::Prefill
            } else {
                CapturePhase::Decode
            };
            if self.prediction != expected || phase != expected_phase {
                return Err(CaptureRunHostError::Coordinate.into());
            }
            None
        } else {
            Some(bank.begin_step(phase, self.prediction)?)
        };

        if self.prepaid_metadata {
            // The exact closed prefix already installed these step reservations.
            // The claim bank above still spends this physical frame once.
            self.session.checkpoint_ready = false;
            self.session.has_step = true;
        } else {
            self.session.reset_step_ledger()?;
            CaptureObservationStep::with_invocation(
                &self.session.plan,
                phase,
                self.prediction,
                self.invocation.map(|value| value.1),
            )?
            .with_window(self.window)?
            .reserve_metadata(&mut self.session.ledger)?;
            if let Some(claim) = &claim {
                claim.reserve_intervention_metadata(&mut self.session.ledger)?;
            }
        }
        self.session.phase = phase;
        self.session.prediction = self.prediction;
        self.session.capture_seconds = 0.0;

        self.frame = match claim {
            Some(claim) => Frame::Active(match self.bound {
                Some(bound) => claim.prepare_prefill_with_progression(bound.geometry())?,
                None => claim.prepare()?,
            }),
            None => Frame::Empty,
        };
        // A failed reservation still owns the already prepared frame. The same
        // transaction guard publishes its Aborted evidence; paid metadata and
        // cumulative usage survive. No native callback can precede this charge.
        if let Some(usage) = self.envelope_usage.filter(|_| !self.prepaid_metadata) {
            if let Some(CaptureSkipReason::Limit { budget, cumulative }) =
                self.session.ledger.reserve(usage)?
            {
                return Err(CaptureError::Limit { budget, cumulative }.into());
            }
        }

        if let Some(partition) = self.backend.partition_capture() {
            let source = self.session.plan.shared().ok_or(CaptureProtocolError::Transaction)?;
            let Frame::Active(frame) = &mut self.frame else {
                return Err(CaptureProtocolError::Transaction.into());
            };
            partition.prepare(source, phase, self.prediction, epoch, frame, &mut self.session.ledger)?;
        }
        Ok(())
    }
    fn coordinate_partition(&mut self, epoch: DistributedCommitEpoch) -> Result<(), FundedCaptureError<E>> {
        if self.backend.partition_capture().is_none() { return Ok(()); }
        let chunk = self.bound.map(|bound| self.current_fragment_chunk().map(|chunk| (bound, chunk))).transpose()?;
        if let Some(partition) = self.backend.partition_capture() {
            partition.coordinate(epoch, &self.session.ledger)?;
            if let Some((bound, chunk)) = chunk {
                let Frame::Active(frame) = &mut self.frame else { return Err(CaptureProtocolError::Transaction.into()); };
                partition.remote_prefill(frame, bound, &chunk)?;
            }
        }
        Ok(())
    }
    fn complete_partition_delivery(&mut self) -> Result<(), FundedCaptureError<E>> {
        if self.partition_delivered { return Ok(()); }
        if let Some(partition) = self.backend.partition_capture() {
            let Frame::Active(frame) = &mut self.frame else { return Err(CaptureProtocolError::Transaction.into()); };
            self.partition_delivered = true;
            partition.deliver(frame)?;
        }
        Ok(())
    }
    fn prepare_delivery(
        &mut self,
        epoch: DistributedCommitEpoch,
    ) -> Result<(), FundedCaptureError<E>> {
        if self.prefill.is_some() {
            return self.complete_chunk(epoch);
        }
        self.seal_delivery(epoch)
    }
    fn seal_delivery(
        &mut self,
        epoch: DistributedCommitEpoch,
    ) -> Result<(), FundedCaptureError<E>> {
        if self.epoch != Some(epoch)
            || self.session.transaction != Some((epoch, CaptureTransactionStatus::Pending))
        {
            return Err(CaptureProtocolError::Transaction.into());
        }
        self.complete_partition_delivery()?;
        if let Frame::Active(frame) = &self.frame {
            frame.validate_interventions_complete()?;
        }
        let frame = std::mem::replace(&mut self.frame, Frame::Finished);
        match frame {
            Frame::Active(frame) => match frame.prepare_delivery(
                self.session.ledger.step(),
                self.session.ledger.total(),
                self.session.capture_seconds,
            ) {
                Ok(prepared) => self.frame = Frame::Sealed(prepared),
                Err(error) => {
                    let (frame, cause) = error.into_parts();
                    self.frame = Frame::Active(frame);
                    return Err(CaptureRunHostError::Step(cause).into());
                }
            },
            Frame::Empty => self.frame = Frame::SealedEmpty,
            other => {
                self.frame = other;
                return Err(CaptureProtocolError::Transaction.into());
            }
        }
        Ok(())
    }
    fn terminal(&mut self, epoch: DistributedCommitEpoch, committed: bool) {
        if self.prefill.is_some() {
            self.finish_chunk(epoch, committed);
        } else {
            self.publish_terminal(epoch, committed);
        }
    }
    fn publish_terminal(&mut self, epoch: DistributedCommitEpoch, committed: bool) {
        if self.epoch != Some(epoch) {
            return;
        }
        self.epoch = None;
        let frame = std::mem::replace(&mut self.frame, Frame::Finished);
        // A commit notification alone cannot promote an unsealed partial frame.
        let accepted = committed && matches!(frame, Frame::Sealed(_) | Frame::SealedEmpty);
        let delivery = match frame {
            Frame::Sealed(prepared) => Some(FundedDelivery::Ready(prepared.finish(if !accepted {
                CaptureStepOutcome::Aborted
            } else if self.untracked {
                CaptureStepOutcome::Untracked
            } else {
                CaptureStepOutcome::Committed
            }))),
            Frame::Active(frame) => Some(FundedDelivery::Aborted(frame.into_aborted_pending(
                self.session.ledger.step(),
                self.session.ledger.total(),
                self.session.capture_seconds,
            ))),
            _ => None,
        };
        // No previous slot can exist: entry checked it before borrowing the bank,
        // and only this lexical observer can write it until return.
        debug_assert!(self.delivery.is_none());
        *self.delivery = delivery;
        self.session.finish_transaction(epoch, accepted);
    }
    fn observe_value(&mut self, path: &str, value: &T) -> Result<(), FundedCaptureError<E>> {
        self.complete_receiver_sources(path, value)?;
        if self.bound.is_some() {
            return self.observe_fragment_value(path, value);
        }
        if !self.active_transaction() {
            return Err(CaptureProtocolError::Transaction.into());
        }
        if matches!(self.frame, Frame::Empty) {
            return Ok(());
        }
        for index in 0..self.session.plan.plan().selections.len() {
            let Frame::Active(frame) = &self.frame else {
                return Err(CaptureProtocolError::Transaction.into());
            };
            if !policy::selected_record(
                &self.session.plan.plan().selections[index],
                &frame.records()[index],
                path,
            )? {
                continue;
            }
            if let Some(partition) = self.backend.partition_capture() {
                if !partition.produces(index)? { continue; }
            }
            let started = std::time::Instant::now();
            let mut source_dtype = None;
            let mut charged = CaptureUsage::default();
            let result = self.observe_one(index, value, &mut source_dtype, &mut charged);
            self.session.capture_seconds += started.elapsed().as_secs_f64();
            if let Err(error) = result {
                let reason = match &error {
                    FundedCaptureError::Admission(error) => policy::failure_reason(error),
                    FundedCaptureError::Backend(_) => CaptureFailureReason::Native,
                    _ => CaptureFailureReason::Invalid,
                };
                if let Frame::Active(frame) = &mut self.frame {
                    // Encoding failure already atomically converted success to
                    // Failed. Never replace it or spend a second constructor.
                    if matches!(frame.records()[index].outcome, CaptureOutcome::Missing) {
                        let _ = frame.record_failure(
                            index,
                            reason,
                            "capture operation failed",
                            source_dtype,
                            charged,
                        );
                    }
                }
                // Preserve the original error even if a fenced parent prevents
                // recording the diagnostic; partial custody still remains here.
                return Err(error);
            }
        }
        Ok(())
    }
    fn observe_one(
        &mut self,
        index: usize,
        value: &T,
        dtype: &mut Option<TensorDtype>,
        charged: &mut CaptureUsage,
    ) -> Result<(), FundedCaptureError<E>> {
        let projected=match self.backend.partition_capture(){
            Some(program)=>program.take_invocation_projection(index)?,None=>None,
        };
        if let Some(hook)=projected {
            let hook=hook.observe_invocation(self.backend,value)?;
            self.backend.partition_capture().ok_or(CaptureProtocolError::Transaction)?
                .return_invocation_projection(index,hook)?;
            return Ok(());
        }
        let policy = CaptureObservationStep::with_invocation(
            &self.session.plan,
            self.session.phase,
            self.session.prediction,
            self.invocation.map(|value| value.1),
        )?
        .with_window(self.window)?;
        let fragment = if let Some(usage) = policy.window_metadata_usage(index)? {
            if let Some(reason) = policy.reserve_value(&mut self.session.ledger, usage)? {
                let Frame::Active(frame) = &mut self.frame else {
                    return Err(CaptureProtocolError::Transaction.into());
                };
                frame.record_skip(index, reason, None, CaptureUsage::default())?;
                return Ok(());
            }
            *charged = usage;
            usage
        } else {
            CaptureUsage::default()
        };
        if matches!(
            self.session.plan.plan().selections[index].transform,
            CaptureTransform::TopCandidates { .. }
        ) {
            let geometry = CaptureCandidateGeometry::prepare(
                &self.session.plan,
                index,
                self.session.phase,
                self.session.prediction,
                self.invocation.map(|value| value.1),
            )
            .map_err(|e| CaptureRunHostError::Step(e.into()))?;
            let actual = self.backend.validate_candidate_source(value, &geometry)?;
            *dtype = Some(actual.clone());
            let usage = self.backend.estimate_candidates(&geometry)?;
            let Frame::Active(frame) = &mut self.frame else {
                return Err(CaptureProtocolError::Transaction.into());
            };
            let skipped=match self.backend.partition_capture() {
                Some(partition)=>policy.reserve_value(partition.reservation(index,&actual,usage)?,usage)?,
                None=>policy.reserve_value(&mut self.session.ledger,usage)?,
            };
            if let Some(reason) = skipped {
                frame.record_skip(index, reason, Some(actual), fragment)?;
                return Ok(());
            }
            *charged = fragment.checked_add(usage)?;
            let claim = frame.take_candidates(index)?;
            let receipt = self.backend.transform_candidates(value, claim)?;
            frame.record_candidates(receipt, actual, *charged)?;
            frame.validate_record_encoding(index)?;
            return Ok(());
        }

        if matches!(
            self.session.plan.plan().selections[index].transform,
            CaptureTransform::TokenScores { .. }
        ) {
            let geometry = CaptureTokenScoreGeometry::prepare(
                &self.session.plan,
                index,
                self.session.phase,
                self.session.prediction,
                self.invocation.map(|value| value.1),
            )
            .map_err(|e| CaptureRunHostError::Step(e.into()))?;

            let actual = self.backend.validate_token_score_source(value, &geometry)?;
            *dtype = Some(actual.clone());

            let usage = self.backend.estimate_token_scores(&geometry)?;
            let Frame::Active(frame) = &mut self.frame else {
                return Err(CaptureProtocolError::Transaction.into());
            };
            let skipped=match self.backend.partition_capture() {
                Some(partition)=>policy.reserve_value(partition.reservation(index,&actual,usage)?,usage)?,
                None=>policy.reserve_value(&mut self.session.ledger,usage)?,
            };
            if let Some(reason) = skipped {
                frame.record_skip(index, reason, Some(actual), fragment)?;
                return Ok(());
            }
            *charged = fragment.checked_add(usage)?;

            let claim = frame.take_token_scores(index)?;

            let receipt = self.backend.transform_token_scores(value, claim)?;

            frame.record_token_scores(receipt, actual, *charged)?;
            frame.validate_record_encoding(index)?;
            return Ok(());
        }

        if matches!(
            self.session.plan.plan().selections[index].transform,
            CaptureTransform::Summary
        ) {
            let geometry = policy
                .summary_geometry(index)
                .map_err(|e| CaptureRunHostError::Step(e.into()))?;
            let actual = self.backend.validate_summary_source(value, &geometry)?;
            *dtype = Some(actual.clone());
            if policy.window().is_some() && geometry.elements() == 0 {
                let Frame::Active(frame) = &mut self.frame else {
                    return Err(CaptureProtocolError::Transaction.into());
                };
                frame.record_window_empty(index, actual, fragment)?;
                frame.validate_record_encoding(index)?;
                return Ok(());
            }
            let usage = self.backend.estimate_summary(&geometry)?;
            let Frame::Active(frame) = &mut self.frame else {
                return Err(CaptureProtocolError::Transaction.into());
            };
            let skipped = match self.backend.partition_capture() {
                Some(partition) => policy.reserve_value(partition.reservation(index, &actual, usage)?, usage)?,
                None => policy.reserve_value(&mut self.session.ledger, usage)?,
            };
            if let Some(reason) = skipped {
                frame.record_skip(index, reason, Some(actual), fragment)?;
                return Ok(());
            }
            *charged = fragment.checked_add(usage)?;
            let claim = frame.take_summary(index)?;
            let receipt = self.backend.transform_summary(value, claim)?;
            frame.record_summary(receipt, actual, *charged)?;
            frame.validate_record_encoding(index)?;
            return Ok(());
        }
        if matches!(
            self.session.plan.plan().selections[index].transform,
            CaptureTransform::Histogram { .. }
        ) {
            let geometry = policy
                .histogram_geometry(index)
                .map_err(|e| CaptureRunHostError::Step(e.into()))?;
            let actual = self.backend.validate_histogram_source(value, &geometry)?;
            *dtype = Some(actual.clone());
            if policy.window().is_some() && geometry.elements() == 0 {
                let Frame::Active(frame) = &mut self.frame else {
                    return Err(CaptureProtocolError::Transaction.into());
                };
                frame.record_window_empty(index, actual, fragment)?;
                frame.validate_record_encoding(index)?;
                return Ok(());
            }
            let usage = self.backend.estimate_histogram(&geometry)?;
            let Frame::Active(frame) = &mut self.frame else {
                return Err(CaptureProtocolError::Transaction.into());
            };
            let skipped = match self.backend.partition_capture() {
                Some(partition) => policy.reserve_value(partition.reservation(index, &actual, usage)?, usage)?,
                None => policy.reserve_value(&mut self.session.ledger, usage)?,
            };
            if let Some(reason) = skipped {
                frame.record_skip(index, reason, Some(actual), fragment)?;
                return Ok(());
            }
            *charged = fragment.checked_add(usage)?;
            let claim = frame.take_histogram(index)?;
            let receipt = self.backend.transform_histogram(value, claim)?;
            frame.record_histogram(receipt, actual, *charged)?;
            frame.validate_record_encoding(index)?;
            return Ok(());
        }
        let geometry = policy
            .tensor_geometry(index)
            .map_err(|e| CaptureRunHostError::Step(e.into()))?;
        let actual = self
            .backend
            .validate_source(value, &geometry)
            .map_err(FundedCaptureError::Backend)?;
        if !matches!(
            actual,
            TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16
        ) {
            return Err(
                CaptureRunHostError::Step(CaptureStepError::TensorMismatch { index }).into(),
            );
        }
        *dtype = Some(actual.clone()); // only these three allocation-free variants
        if policy.window().is_some()
            && geometry
                .starts()
                .iter()
                .zip(geometry.ends())
                .any(|(a, b)| a == b)
        {
            let Frame::Active(frame) = &mut self.frame else {
                return Err(CaptureProtocolError::Transaction.into());
            };
            frame.record_window_empty(index, actual, fragment)?;
            frame.validate_record_encoding(index)?;
            return Ok(());
        }
        let usage = self.backend.estimate(value, &geometry)?;
        let Frame::Active(frame) = &mut self.frame else {
            return Err(CaptureProtocolError::Transaction.into());
        };
        let skipped = match self.backend.partition_capture() {
            Some(partition) => policy.reserve_value(partition.reservation(index, &actual, usage)?, usage)?,
            None => policy.reserve_value(&mut self.session.ledger, usage)?,
        };
        if let Some(reason) = skipped {
            frame.record_skip(index, reason, Some(actual), fragment)?;
            return Ok(());
        }
        *charged = fragment.checked_add(usage)?; // Successful logical reservation is never refunded.
        let claim = frame.take_tensor(index)?;
        let receipt = self
            .backend
            .transform(value, claim)
            .map_err(FundedCaptureError::Backend)?;
        frame.record_tensor(receipt, actual, *charged)?;
        frame.validate_record_encoding(index)?;
        Ok(())
    }
    fn generated(&mut self, path: &str) -> Result<(), FundedCaptureError<E>> {
        if !self.active_transaction() {
            return Err(CaptureProtocolError::Transaction.into());
        }
        if matches!(self.frame, Frame::Empty) {
            return Ok(());
        }
        let Frame::Active(frame) = &mut self.frame else {
            return Err(CaptureProtocolError::Transaction.into());
        };
        for (index, selection) in self.session.plan.plan().selections.iter().enumerate() {
            if policy::selected_record(selection, &frame.records()[index], path)? {
                let _ = frame.record_failure(
                    index,
                    CaptureFailureReason::Unsupported,
                    "generated source is not admitted by this collector",
                    None,
                    CaptureUsage::default(),
                );
                return Err(CaptureProtocolError::GeneratedSource.into());
            }
        }
        Ok(())
    }
}
impl<T, E: std::error::Error + Send + Sync + 'static, N> crate::ActivationObserver<T, N>
    for FundedCaptureObserver<'_, T, E, N>
{
    fn routed_unit_observer(
        &mut self, path: &str,
    ) -> Result<Option<&mut dyn crate::RoutedUnitObserver<T>>, N> {
        let index = self.session.plan.points().iter().position(|point|
            matches!(&point.value_type, eredu_core::ObservationValueType::RoutedUnits { routing, .. }
                if routing == path));
        let edit=match &self.frame {
            Frame::Active(frame)=>frame.intervention_admission().and_then(|plan|plan.points().iter()
                .position(|point|point.routed_units.as_ref().is_some_and(|routed|routed.routing==path))),
            _=>None,
        };
        if index.is_none() && edit.is_none(){return Ok(None);}
        if self.routed_active && (self.routed_selection!=index || self.routed_intervention_selection!=edit) {
            return Err((self.map_error)(CaptureProtocolError::Transaction.into()));
        }
        // Prepared distributed/window-aggregation edits retain their own pending
        // source joins; the completed direct invocation path is distinct.
        if edit.is_some() && (self.prefill.is_some() || self.invocation.is_none()) {
            return Err((self.map_error)(CaptureProtocolError::Transaction.into()));
        }
        if let Some(program)=self.backend.partition_capture() {
            if edit.is_some(){return Err((self.map_error)(CaptureProtocolError::Transaction.into()));}
            if let Some(index)=index {
                if !program.routed_source(index).map_err(|cause|(self.map_error)(cause.into()))? {return Ok(None);}
            }
        }
        self.routed_selection=index;
        self.routed_intervention_selection=edit;
        Ok(Some(self))
    }
    fn admitted_prefill_capture(
        &self,
    ) -> Option<&crate::working_memory::AdmittedPrefillCapture<'_>> {
        self.admitted
    }
    fn admitted_capture_continuation(
        &self,
    ) -> Option<&crate::working_memory::AdmittedCaptureContinuation<'_>> {
        self.continuation
    }
    fn requires_sequence_readout(&self) -> bool {
        if let Some(required) = self.invocation_readout {
            return required;
        }
        if let Some(bound) = self.bound {
            return bound.geometry().output == eredu_core::OutputDemand::Sequence;
        }
        let phase = self.invocation.map_or_else(
            || {
                if self.prediction == 0 {
                    CapturePhase::Prefill
                } else {
                    CapturePhase::Decode
                }
            },
            |value| value.0,
        );
        CaptureObservationStep::with_invocation(
            &self.session.plan,
            phase,
            self.prediction,
            self.invocation.map(|value| value.1),
        )
        .map_or(true, |policy| policy.requires_sequence_readout())
    }
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn transactional(&self) -> bool {
        true
    }
    fn begin_prefill_chunk(&mut self, chunk: &crate::prefill::PrefillChunk) -> Result<(), N> {
        if let Some(continuation) = self.continuation {
            let geometry = continuation.geometry();
            if self.continuation_announced
                || self.epoch.is_some()
                || chunk.input != (0..1)
                || chunk.position != geometry.cached_positions
                || chunk.output != geometry.output
            {
                return Err((self.map_error)(CaptureProtocolError::Geometry.into()));
            }
            self.continuation_announced = true;
            Ok(())
        } else {
            self.begin_chunk(chunk).map_err(|error| self.stage_error(error, "prefill chunk announcement"))
        }
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.finish_prompt(committed);
    }
    fn prepare_prefill_chunk_retention(
        &mut self,
        context: &crate::inspection::PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<crate::inspection::PreparedPrefillChunkRetention>, N> {
        if let Some(continuation) = self.continuation {
            if !self.continuation_announced
                || self.continuation_epoch.is_some()
                || context.geometry() != continuation.geometry()
                || context.chunk().input != (0..1)
                || context.chunk().position != continuation.geometry().cached_positions
                || context.chunk().output != continuation.geometry().output
            {
                return Err((self.map_error)(CaptureProtocolError::Geometry.into()));
            }
            // One whole token is captured directly under its active span; no
            // cross-chunk source retention or fragment destination is constructed.
            self.continuation_epoch = Some(context.epoch());
            Ok(None)
        } else {
            self.prepare_retention(context).map_err(|error| self.stage_error(error, "prefill retention preparation"))
        }
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        settled: crate::inspection::SettledPrefillChunkRetention,
    ) -> Result<(), N> {
        // The final frame is already Sealed here; only backend custody changes.
        self.backend
            .retire_prefill_chunk_retention(settled)
            .map_err(self.map_error)
    }
    fn prepare_transaction(
        &mut self,
        epoch: DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), N> {
        self.prepare(epoch, pass).map_err(|error| self.stage_error(error, "transaction preparation"))
    }
    fn coordinate_transaction(&mut self, epoch: DistributedCommitEpoch) -> Result<(), N> {
        self.coordinate_partition(epoch).map_err(|error| self.stage_error(error, "partition coordination"))
    }
    fn complete_transaction(&mut self, epoch: DistributedCommitEpoch) -> Result<(), N> {
        self.prepare_delivery(epoch).map_err(|error| self.stage_error(error, "transaction completion"))
    }
    fn finish_transaction(&mut self, epoch: DistributedCommitEpoch, committed: bool) {
        self.terminal(epoch, committed);
    }
    fn intervene(&mut self, path: &str, value: &T) -> Result<Option<T>, N> {
        self.intervene_value(path, value).map_err(|error| self.stage_error(error, "activation intervention"))
    }
    fn observe(&mut self, path: &str, value: &T) -> Result<(), N> {
        self.observe_value(path, value).map_err(|error| self.stage_error(error, "activation observation"))
    }
    fn observe_replica(&mut self, path: &str, value: &T) -> Result<(), N> {
        if self.backend.partition_capture().is_none() { return Ok(()); }
        self.observe_value(path, value).map_err(|error| self.stage_error(error, "activation observation"))
    }
    fn observe_generated(
        &mut self,
        path: &str,
        _prototype: &T,
        _source: &GeneratedCaptureSource,
        _generate: &mut dyn FnMut() -> Result<T, N>,
    ) -> Result<(), N> {
        self.generated(path).map_err(|error| self.stage_error(error, "generated observation"))
    }
    fn observe_generated_retained(
        &mut self,
        path: &str,
        prototype: &T,
        source: &GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<T, N>,
    ) -> Result<(), N> {
        self.observe_retained_generated(path, prototype, source, factory)
    }
}
impl<T, E: std::error::Error + Send + Sync + 'static, N> Drop
    for FundedCaptureObserver<'_, T, E, N>
{
    fn drop(&mut self) {
        if let Some(epoch) = self.epoch {
            self.publish_terminal(epoch, false);
        }
    }
}
