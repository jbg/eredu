//! Depends on the observer-aware blueprint companion, not a native capture gate.
use super::*;
use crate::composition::mlx::session::model_session::{
    disk_layerwise_tests as disk, host_layerwise_tests as host,
};
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceTensor};
use eredu_runtime::working_memory::{InferenceWorkspaceObserver, InferenceWorkspaceSpan};

struct Parameters<'a>(&'a crate::backend::runtime::execution::generic::LayerwiseWorkspace);
impl eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters for Parameters<'_> {
    fn layout(&self) -> &eredu_runtime::ExecutionUnitLayout {
        self.0.layout()
    }
    fn parameters(
        &self,
        ordinal: usize,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &WorkspaceContext,
    ) -> Result<std::collections::BTreeMap<eredu_nn::ParameterId, WorkspaceTensor>, eredu_nn::Error>
    {
        self.0.parameters(ordinal, address, context)
    }
}
struct BorrowedPaths<'a> {
    source: &'a SharedLayeredObservationPaths,
    callbacks: usize,
    predictions: Vec<u64>,
}
impl eredu_runtime::ActivationObserver<WorkspaceTensor, eredu_nn::Error> for BorrowedPaths<'_> {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn observe(&mut self, path: &str, _: &WorkspaceTensor) -> Result<(), eredu_nn::Error> {
        // This ordinary Llama fixture leaves unit boundaries to the traversal.
        // Architecture-internal hook paths are intentionally not compared.
        for group in 0..self.source.group_count() {
            for index in 0..self.source.unit_count(group).unwrap() {
                let (input, output) = self.source.unit_paths(group, index).unwrap();
                for expected in [input, output] {
                    if path == expected {
                        assert_eq!(path.as_ptr(), expected.as_ptr());
                        self.callbacks += 1;
                    }
                }
            }
            for expected in [
                self.source.group_input(group),
                self.source.group_output(group),
            ]
            .into_iter()
            .flatten()
            {
                if path == expected {
                    assert_eq!(path.as_ptr(), expected.as_ptr());
                    self.callbacks += 1;
                }
            }
        }
        Ok(())
    }
}
impl InferenceWorkspaceObserver for BorrowedPaths<'_> {
    fn begin_span(
        &mut self,
        _: eredu_core::InferenceGeometry,
        span: &InferenceWorkspaceSpan,
        prediction: u64,
        _: &WorkspaceContext,
    ) -> Result<bool, eredu_nn::Error> {
        let active = matches!(span, InferenceWorkspaceSpan::Decode { .. });
        if active {
            self.predictions.push(prediction);
        }
        Ok(active)
    }
    fn visit_retained(&self, _: &mut dyn FnMut(&WorkspaceTensor)) {}
}

#[test]
fn actual_native_source_is_rebound_by_metadata_quotes_without_copying_path_payloads() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for route in 0..3 {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let (runtime, _artifact) = match route {
            0 => host::runtime(&stream, &pool, None),
            1 => host::runtime(&stream, &pool, Some(1)),
            _ => disk::load_runtime(&stream, &pool, true),
        };
        let source = paths(&runtime);
        let executable = &runtime.session().payload.model;
        let blueprint = executable.inference_blueprint().unwrap();
        let context = WorkspaceContext::new(executable.workspace_mechanisms().unwrap());
        let state = executable
            .erased()
            .project_resident_workspace(std::num::NonZeroU32::new(1).unwrap(), &context)
            .unwrap();
        let layerwise = executable.layerwise_workspace().unwrap();
        let geometry = eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 3,
            max_output_tokens: 3,
            prefill_chunk_positions: 3,
            output: eredu_core::OutputDemand::LastPosition,
        };
        let frontier = executable.erased().state_snapshot();
        let usage = (
            pool.fixture_host_charge().unwrap(),
            pool.fixture_host_peak().unwrap(),
        );
        let mut observer = BorrowedPaths {
            source: &source,
            callbacks: 0,
            predictions: Vec::new(),
        };
        let report = match layerwise.as_ref() {
            Some(parameters) => blueprint.quote_replicated_layerwise_text_observed(
                geometry,
                &state,
                &context,
                &Parameters(parameters),
                &source,
                &mut observer,
            ),
            None => blueprint.quote_replicated_resident_text_observed(
                geometry,
                &state,
                &context,
                &source,
                &mut observer,
            ),
        }
        .unwrap();
        assert!(report.completed_spans() >= 3);
        assert_eq!(observer.predictions, [1, 2]);
        assert!(observer.callbacks >= 4);
        assert_eq!(executable.erased().state_snapshot(), frontier);
        assert_eq!(
            (
                pool.fixture_host_charge().unwrap(),
                pool.fixture_host_peak().unwrap()
            ),
            usage
        );
        assert!(source.same_storage(executable.erased().shared_observation_paths().unwrap()));
        // Metadata tracing retained neither model payload nor a new path owner.
        let retained = source.capacity_bytes().unwrap();
        // Loading uses the actual CPU source stream as well as the GPU stream.
        // The quote's cold assertions precede this ordinary cleanup boundary.
        runtime.synchronize().unwrap();
        drop((observer, report, state, context, layerwise, runtime));
        stream.synchronize().unwrap();
        // Stream completion does not retire native submission primitive owners.
        // An empty completion wait performs ordinary terminal-record cleanup,
        // without evaluating arrays or submitting numerical work.
        safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
            .unwrap()
            .synchronize()
            .unwrap();
        settle(&pool, retained);
        drop(source);
        settle(&pool, 0);
    }
}

mod fragments;
