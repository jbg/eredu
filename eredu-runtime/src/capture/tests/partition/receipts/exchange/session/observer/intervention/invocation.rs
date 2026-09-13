use super::*;
mod provider;

#[test]
fn partition_observer_uses_invocation_scopes_for_real_edits_and_evidence() {
    verify_invocation_scopes(false);
}

#[test]
fn speculative_borrow_retains_partition_work_through_the_complete_forward() {
    verify_invocation_scopes(true);
}

fn verify_invocation_scopes(scoped: bool) {
    let all = world(5);
    let hooks = world(4);
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 5,
        max_context: None,
        max_predictions: 8,
    };
    let results = std::thread::scope(|scope| {
        (0..5)
            .map(|rank| {
                let all = Arc::clone(&all);
                let hooks = Arc::clone(&hooks);
                scope.spawn(move || {
                    let members = vec![0, 2, 3, 4];
                    let transport = HookTransport {
                        transport: transport(all, rank, Fault::None),
                        hook: members
                            .iter()
                            .position(|r| *r == rank)
                            .map(|r| transport(hooks, r, Fault::None)),
                        members,
                    };
                    let layout = layout();
                    let (capture, plan) = plans_at(
                        rank,
                        InterventionEvidence::Preview { max_elements: 100 },
                        false,
                        Some(bounds),
                    );
                    let catalog = discovery(&capture);
                    let mut session = configured(capture, 5);
                    session
                        .enable_interventions(plan, Arc::new(Estimates))
                        .unwrap();
                    let saved = session.checkpoint(&catalog).unwrap();
                    let mut epoch = DistributedCommitEpoch::FIRST;
                    let mut results = Vec::new();
                    let mut spent = CaptureUsage::default();
                    for (index, (rows, capture, keep, scale)) in [
                        (5, true, true, true),
                        (2, false, false, true),
                        (1, true, true, false),
                        (5, true, true, true),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if index == 3 {
                            session.restore(&saved).unwrap();
                        }
                        let invocation = CaptureInvocationShape {
                            batch: 1,
                            sequence: rows,
                            context: None,
                        };
                        let mut execute =
                            |observer: &mut dyn crate::ActivationObserver<Value, Error>| {
                                observer.prepare_transaction(epoch, crate::ExpertPass::Decode)?;
                                observer.coordinate_transaction(epoch)?;
                                if let Some((_, map)) = layout.0.iter().find(|(r, _)| *r == rank) {
                                    let global = value(
                                        vec![rows, 20],
                                        (0..rows * 20).map(|i| i as f32 * 0.25 - 3.).collect(),
                                    );
                                    let input = local_value(&global, map);
                                    observer.observe("block.output", &input)?;
                                    let output = observer.intervene("block.output", &input)?;
                                    let mut expected = data(&global).to_vec();
                                    for (i, value) in expected.iter_mut().enumerate() {
                                        if keep && i < 20 && ![2, 9, 17].contains(&(i % 20)) {
                                            *value = 0.;
                                        }
                                        if scale {
                                            *value *= 2.;
                                        }
                                    }
                                    let expected =
                                        local_value(&value(vec![rows, 20], expected), map);
                                    assert_eq!(
                                        data(output.as_ref().unwrap_or(&input)),
                                        data(&expected)
                                    );
                                    assert_eq!(data(&input), data(&local_value(&global, map)));
                                }
                                observer.complete_transaction(epoch)?;
                                Ok(())
                            };
                        let mut owner = ScopedPartition {
                            session: &mut session,
                            transport: &transport,
                            layout: &layout,
                            invocation,
                            captures: [capture],
                            interventions: [keep, scale],
                            epoch,
                        };
                        if scoped {
                            crate::inspection::with_speculative_activation(
                                Some(&mut owner),
                                eredu_core::speculative::SpeculativeActivationPhase::Verification,
                                rows as usize,
                                |observer| execute(observer.unwrap()),
                            )
                            .unwrap();
                        } else {
                            use crate::inspection::SpeculativeActivationObserver;
                            owner.begin_activation_invocation(
                                eredu_core::speculative::SpeculativeActivationPhase::Verification,
                                rows as usize,
                            ).unwrap();
                            owner.with_activation_observer(&mut execute).unwrap();
                        }
                        assert!(session.take_step().is_none());
                        session.finish_transaction(epoch, true);
                        let result = session.take_step().unwrap();
                        assert_eq!(result.records[0].payload.is_some(), capture);
                        for (operation, active) in result.interventions.iter().zip([keep, scale]) {
                            assert_eq!(
                                operation.outcome,
                                if active {
                                    InterventionOutcome::Applied
                                } else {
                                    InterventionOutcome::Inactive
                                }
                            );
                            assert!(operation
                                .evidence
                                .iter()
                                .all(|record| record.payload.is_some() == active));
                        }
                        assert!(result
                            .partitions
                            .iter()
                            .all(|record| record.context.invocation == Some(invocation)));
                        assert!(result.cumulative_usage.host_bytes > spent.host_bytes);
                        spent = result.cumulative_usage;
                        results.push(result);
                        epoch = epoch.next().unwrap();
                    }
                    assert_eq!(results[0].records[0].payload, results[3].records[0].payload);
                    for (a, b) in results[0]
                        .interventions
                        .iter()
                        .zip(&results[3].interventions)
                    {
                        for (a, b) in a.evidence.iter().zip(&b.evidence) {
                            assert_eq!(a.payload, b.payload);
                        }
                    }
                    results
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    for result in &results[1..] {
        for (actual, expected) in result.iter().zip(&results[0]) {
            assert_eq!(actual.cumulative_usage, expected.cumulative_usage);
            assert_eq!(actual.records, expected.records);
            assert_eq!(actual.partitions, expected.partitions);
        }
    }
}

type Error = PartitionCaptureObserverError<std::io::Error>;

// The outer owner retains admission; one stack-owned partition observer must
// keep move-only work alive across preparation, hooks and shared completion.
struct ScopedPartition<'a> {
    session: &'a mut CaptureSession,
    transport: &'a HookTransport,
    layout: &'a EditLayout,
    invocation: CaptureInvocationShape,
    captures: [bool; 1],
    interventions: [bool; 2],
    epoch: DistributedCommitEpoch,
}
impl crate::ActivationObserver<Value, Error> for ScopedPartition<'_> {
    fn observe(&mut self, _: &str, _: &Value) -> Result<(), Error> {
        panic!("the forward must use the borrowed partition observer")
    }
}
impl crate::inspection::SpeculativeActivationObserver<Value, Error> for ScopedPartition<'_> {
    fn begin_activation_invocation(
        &mut self,
        phase: eredu_core::speculative::SpeculativeActivationPhase,
        sequence: usize,
    ) -> Result<(), Error> {
        assert_eq!(
            phase,
            eredu_core::speculative::SpeculativeActivationPhase::Verification
        );
        assert_eq!(sequence as u64, self.invocation.sequence);
        self.session
            .begin_invocation(
                CapturePhase::Decode,
                4,
                self.invocation,
                CaptureInvocationSelection {
                    captures: Some(&self.captures),
                    interventions: Some(&self.interventions),
                },
            )
            .map_err(Into::into)
    }
    fn with_activation_observer(
        &mut self,
        operation: &mut dyn FnMut(
            &mut dyn crate::ActivationObserver<Value, Error>,
        ) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let mut observer = PartitionCaptureObserver::for_step(
            self.session,
            Backend::default(),
            self.transport,
            self.layout,
            4,
            PartitionCaptureReceiptLimits {
                max_producers: 5,
                max_fragments: 64,
                max_record_bytes: 1 << 16,
            },
            estimate,
            |error| error,
        )
        .with_interventions();
        operation(&mut observer)
    }
    fn complete_activation_invocation(&mut self) -> Result<(), Error> {
        // Borrowed work is gone, but final commit still belongs to the executor.
        assert!(self.session.take_step().is_none());
        Ok(())
    }
    fn finish_activation_invocation(&mut self, success: bool) {
        if !success {
            self.session.finish_transaction(self.epoch, false);
        }
    }
}
