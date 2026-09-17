//! Test-only bridge preserves the actual ordinary typed session before erasure.
use super::*;
use eredu_core::{capture::*, *};

impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    pub(in crate::composition::mlx::replicated_text) fn check_fixed_opening_pair_for_test(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self>,
        expected: Option<&eredu_runtime::SharedLayeredObservationPaths>,
    ) -> Result<(usize, usize, u64), Error> {
        let source = SharedCapturePlan::new(
            CapturePlan::none()
                .admit(
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
                    &CaptureCapabilities {
                        transformations: vec![],
                        max_histogram_bins: 0,
                        physical_native_limit: false,
                        conditions: vec![],
                    },
                    CaptureRequestShape {
                        batch: 1,
                        prompt_tokens: 1,
                        max_predictions: 2,
                    },
                )
                .unwrap(),
        );
        let selected = expected
            .unwrap_or_else(|| session.shared_observation_paths().unwrap())
            .prepare_capture_selection(&source)
            .unwrap();
        let bound = selected
            .bind_geometry(InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: 1,
                max_output_tokens: 2,
                prefill_chunk_positions: 1,
                output: OutputDemand::LastPosition,
            })
            .unwrap();
        let plan = Self::prepare_session_fixed_opening_capacity(session, bound)
            .map_err(|e| Error::Other(Box::new(e)))?;
        let before = session.report().map_err(|e| Error::Other(Box::new(e)))?;
        let mut direct = Vec::new();
        let complete = session
            .visit_retained_values(&mut |v| direct.push(v as *const MlxTensor))
            .map_err(|e| Error::Other(Box::new(e)))?;
        let mut paired = Vec::new();
        assert_eq!(
            plan.execution
                .visit_owners(&mut |v| paired.push(v as *const MlxTensor)),
            complete
        );
        assert!(complete);
        assert_eq!(direct, paired);
        assert!(
            !paired.is_empty(),
            "nonzero loaded model owns actual native tensor handles"
        );
        session
            .inspect_runtime_execution(|m, state, execution| {
                assert!(std::ptr::eq(plan.state, state));
                assert!(std::ptr::eq(plan.execution, execution));
                assert!(std::ptr::eq(plan.store, m.store.as_ref()));
                assert!(std::ptr::eq(
                    plan.weights.as_ref().unwrap().source(),
                    m.residency_manager.as_ref().unwrap()
                ));
                assert!(std::ptr::eq(
                    plan.paths,
                    session.prepared_observation_paths().unwrap()
                ));
                assert!(plan.paths.source().same_storage(selected.paths()));
                assert!(plan.selection.selection().source().same_storage(&source));
                Ok(())
            })
            .map_err(|e| Error::Other(Box::new(e)))?;
        let actual = plan.execution.owner_slot_bound().unwrap();
        assert!(actual >= paired.len());
        assert!(plan.slots.arrays >= actual);
        let bytes = plan
            .control_peak_bytes()
            .map_err(|e| Error::Other(Box::new(e)))?;
        let after = session.report().map_err(|e| Error::Other(Box::new(e)))?;
        assert_eq!(
            before.state_report().presence,
            after.state_report().presence
        );
        Ok((paired.len(), plan.slots.arrays, bytes))
    }
}
