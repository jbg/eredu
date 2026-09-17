use super::*;

#[derive(Debug)]
struct Scheduled(bool);
impl WorkspaceMechanisms for Scheduled {
    fn grouped_observation_schedule(
        &self,
        _: &WorkspaceGroupedBank,
        _: u32,
    ) -> Result<Option<WorkspaceGroupedObservationSchedule>, Error> {
        Ok(self
            .0
            .then_some(WorkspaceGroupedObservationSchedule::TokenChunks(
                std::num::NonZeroU32::new(3).unwrap(),
            )))
    }
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let mut bound = AllocatingMechanism.operation_bound(op)?.unwrap();
        if matches!(
            op.kind,
            WorkspaceOperationKind::Grouped {
                phase: WorkspaceGroupedPhase::Finish,
                ..
            }
        ) {
            // This test mechanism returns a view of the effective unit storage.
            // It exposes the joined chunk owners to ordinary report traversal.
            bound
                .outputs
                .fill(WorkspaceOutputStorage::AliasInput(op.inputs.len() - 4));
        }
        Ok(Some(bound))
    }
    fn host_workspace_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        AllocatingMechanism.host_workspace_bound(op)
    }
}

struct ScheduledObserver<'a> {
    context: &'a WorkspaceContext,
    events: Vec<(u8, usize, i32)>,
    total: usize,
    failure: Option<(u8, usize)>,
    retained: Vec<WorkspaceTensor>,
    alias_first: bool,
}
impl ScheduledObserver<'_> {
    fn event(&mut self, phase: u8, b: &GroupedUnitBatch<'_, WorkspaceTensor>) -> Result<(), Error> {
        let rows = b.coefficients.shape()[0];
        self.events.push((phase, b.token_offset, rows));
        assert_eq!(b.total_token_count, self.total);
        assert_eq!(b.values.shape(), [rows * 2, 32]);
        assert_eq!(b.coefficients.shape(), [rows, 2]);
        for ids in [b.group_indices, b.selection_indices, b.token_indices] {
            assert_eq!(ids.shape(), [rows * 2]);
        }
        if self.failure == Some((phase, b.token_offset)) {
            return Err(Error::backend("observer budget exhausted"));
        }
        Ok(())
    }
}
impl GroupedUnitObserver<WorkspaceTensor> for ScheduledObserver<'_> {
    fn observe(&mut self, b: &GroupedUnitBatch<'_, WorkspaceTensor>) -> Result<(), Error> {
        self.event(0, b)
    }
    fn intervene(
        &mut self,
        b: &GroupedUnitBatch<'_, WorkspaceTensor>,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        self.event(1, b)?;
        if self.alias_first && b.token_offset == 0 {
            return Ok(None);
        }
        Ok(Some(b.values.square(self.context)?))
    }
    fn observe_effective(
        &mut self,
        b: &GroupedUnitBatch<'_, WorkspaceTensor>,
    ) -> Result<(), Error> {
        self.event(2, b)?;
        self.retained.push(b.values.clone());
        Ok(())
    }
}

#[test]
fn grouped_observer_schedule_preserves_coordinates_order_and_all_chunk_owners() {
    for count in [0, 1, 3, 4, 7] {
        for alias_first in [false, true] {
            let c = WorkspaceContext::new(Scheduled(true));
            let mut m =
                WorkspaceBackend::grouped_gated_product(bank(false, false, false), &c).unwrap();
            let input = existing_f32(&[1, count, 64], &c).unwrap();
            let selection = routes(&c, count);
            let mut observer = ScheduledObserver {
                context: &c,
                events: vec![],
                total: count as usize,
                failure: None,
                retained: vec![],
                alias_first,
            };
            c.begin_span();
            let before = c.report_workspace_layout().unwrap();
            let output = m
                .forward_grouped_with_unit_observer(&input, &selection, &c, Some(&mut observer))
                .unwrap();
            let after = c.report_workspace_layout().unwrap();
            let trace = c.trace.borrow();
            let joined = usize::from(count > 3);
            assert_eq!(
                after.nodes(),
                before.nodes() + trace.allocations.len() + joined
            );
            assert_eq!(
                after.edges(),
                before.edges()
                    + trace
                        .allocations
                        .iter()
                        .map(|n| n.possible_aliases.len())
                        .sum::<usize>()
                    + if joined != 0 {
                        (count as usize).div_ceil(3)
                    } else {
                        0
                    }
            );
            drop(trace);
            let mut expected = Vec::new();
            for start in (0..count.max(1)).step_by(3) {
                for phase in 0..3 {
                    expected.push((phase, start as usize, (count - start).min(3)));
                }
            }
            assert_eq!(observer.events, expected);
            let report = c.report(&[output]).unwrap();
            let bytes = if alias_first {
                // The first view retains the complete original unit backing.
                (count + (count - 3).max(0)) as u64 * 2 * 32 * 4
            } else {
                count as u64 * 2 * 32 * 4
            };
            assert_eq!(report.tensor_buffers.retained_bytes, Some(bytes));
            assert_eq!(
                c.report(&observer.retained)
                    .unwrap()
                    .tensor_buffers
                    .retained_bytes,
                Some(bytes)
            );
        }
    }
}

#[test]
fn grouped_observer_schedule_stops_later_callbacks_and_preserves_spent_workspace() {
    for phase in 0..3 {
        for failed_offset in [0, 3, 6] {
            let c = WorkspaceContext::new(Scheduled(true));
            let mut m =
                WorkspaceBackend::grouped_gated_product(bank(false, false, false), &c).unwrap();
            let input = existing_f32(&[7, 64], &c).unwrap();
            let selection = routes(&c, 7);
            let mut observer = ScheduledObserver {
                context: &c,
                events: vec![],
                total: 7,
                failure: Some((phase, failed_offset)),
                retained: vec![],
                alias_first: false,
            };
            c.begin_span();
            assert!(m
                .forward_grouped_with_unit_observer(&input, &selection, &c, Some(&mut observer))
                .is_err());
            assert_eq!(observer.events.len(), failed_offset + phase as usize + 1);
            assert_eq!(observer.events.last().unwrap().0, phase);
            assert_eq!(observer.events.last().unwrap().1, failed_offset);
            let report = c.report(&observer.retained).unwrap();
            assert!(report.total_bytes.unwrap() > 0);
            assert!(!report.operations.iter().any(|op| matches!(
                op.kind,
                WorkspaceOperationKind::Grouped {
                    phase: WorkspaceGroupedPhase::Finish,
                    ..
                }
            )));
            assert!(
                report.tensor_buffers.retained_bytes.unwrap() >= failed_offset as u64 * 2 * 32 * 4
            );
        }
    }
}

#[test]
fn missing_grouped_observer_schedule_rejects_before_callbacks_or_unit_work() {
    let c = WorkspaceContext::new(Scheduled(false));
    let mut m = WorkspaceBackend::grouped_gated_product(bank(false, false, false), &c).unwrap();
    let input = existing_f32(&[7, 64], &c).unwrap();
    let selection = routes(&c, 7);
    let mut observer = ScheduledObserver {
        context: &c,
        events: vec![],
        total: 7,
        failure: None,
        retained: vec![],
        alias_first: false,
    };
    c.begin_span();
    let error = m
        .forward_grouped_with_unit_observer(&input, &selection, &c, Some(&mut observer))
        .unwrap_err();
    assert!(std::error::Error::source(&error)
        .unwrap()
        .is::<WorkspaceGroupedObservationScheduleUnavailable>());
    assert!(observer.events.is_empty());
    assert!(c.report(&[]).unwrap().operations.is_empty());
}
