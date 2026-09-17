use super::*;

#[test]
fn fixed_history_hold_and_late_outputs_fit_sixteen_and_thirty_two_steps() {
    for steps in [16_u64, 32] {
        let sampler = plain_sampler();
        let (report, emissions) = retained_output_quote(
            &sampler,
            0.0,
            &TokenFilter::All,
            steps,
            TokenBacking::Allocated(64),
            None,
        )
        .unwrap();
        let inline = std::mem::size_of::<ConfiguredTextSampler>() as u64;
        let mut capacity = 0;
        let mut maximum_overlap = 0;
        let mut previous_simultaneous_peak = 0;
        for accepted in 0..steps as usize {
            let old = capacity;
            if accepted == capacity {
                capacity = crate::generation::next_history_capacity(capacity).unwrap();
            }
            let overlap = if old == capacity {
                capacity
            } else {
                old + capacity
            };
            maximum_overlap = maximum_overlap.max(overlap);
            // The fixture retains one 64-byte output per step. Greedy sampling
            // and its scalar observation each use seven scratch and three host
            // bytes; observations alias the output's backing.
            let tensor = (accepted as u64 + 1) * 64 + 14;
            let host = inline + overlap as u64 * 4 + 6;
            previous_simultaneous_peak = previous_simultaneous_peak.max(tensor + host);
        }
        let canonical_sampler_hold = inline + maximum_overlap as u64 * 4;
        assert_eq!(maximum_overlap as u64, steps + steps / 2);
        assert_eq!(emissions as u64, steps);
        assert_eq!(report.tensor_peak_bytes, Some(steps * 64 + 14));
        assert_eq!(report.host_peak_bytes, Some(canonical_sampler_hold + 6));
        assert_eq!(report.final_history_bytes, steps * 4);
        assert_eq!(
            report.peak.bytes(),
            Some(steps * 64 + 14 + canonical_sampler_hold + 6)
        );
        assert!(
            report.peak.bytes().unwrap() > previous_simultaneous_peak,
            "the earlier replacement overlap remains protected at the final output"
        );
        assert_eq!(sampler.history_len(), 0);
        assert_eq!(sampler.history_capacity(), 0);
    }
}

#[derive(Debug)]
struct SeparatedPeaks {
    emissions: Cell<usize>,
}

impl WorkspaceMechanisms for SeparatedPeaks {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let outputs = match operation.kind {
            WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Greedy) => {
                let emitted = self.emissions.get();
                self.emissions.set(emitted + 1);
                vec![WorkspaceOutputStorage::Allocate(if emitted == 0 {
                    4
                } else {
                    u64::MAX / 2 + 64
                })]
            }
            WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::ReadToken) => {
                vec![WorkspaceOutputStorage::AliasInput(0)]
            }
            _ => panic!("unexpected operation in a plain greedy sampler"),
        };
        Ok(Some(WorkspaceOperationBound {
            outputs,
            scratch_bytes: 0,
            assumptions: "small first output and large retained second output".into(),
        }))
    }

    fn host_workspace_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        let first = self.emissions.get() == 1
            && matches!(
                operation.kind,
                WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Greedy)
            );
        Ok(Some(WorkspaceHostBound {
            bytes: if first { u64::MAX / 2 } else { 0 },
            assumptions: "only the first step needs the large host workspace".into(),
        }))
    }
}

#[test]
fn separately_representable_peaks_reject_unrepresentable_held_envelope() {
    let quote = |steps| {
        let context = WorkspaceContext::new(SeparatedPeaks {
            emissions: Cell::new(0),
        });
        quote_sampling_workspace(
            &plain_sampler(),
            0.0,
            None,
            &layout(),
            &TokenFilter::All,
            steps,
            &context,
        )
    };
    assert!(quote(1).unwrap().peak.bytes().is_some());
    let error = quote(2).unwrap_err();
    assert!(matches!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<crate::working_memory::WorkingMemoryError>()),
        Some(crate::working_memory::WorkingMemoryError::Overflow)
    ));
}
