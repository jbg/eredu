//! Adds only the exact cut notification to the existing observation callbacks.
use super::*;
use eredu_runtime::inspection::*;
pub(super) struct CutObserver<'a> {
    pub(super) observer: &'a mut dyn eredu_runtime::ActivationObserver<WorkspaceTensor, Error>,
    pub(super) cut:
        &'a mut dyn FnMut(&mut dyn FnMut(&mut dyn FnMut(&WorkspaceTensor))) -> Result<(), Error>,
}
impl eredu_runtime::ActivationObserver<WorkspaceTensor, Error> for CutObserver<'_> {
    fn observes_activations(&self)->bool { self.observer.observes_activations() }

    fn retained_media_cut(
        &mut self,
        visit: &mut dyn FnMut(&mut dyn FnMut(&WorkspaceTensor)),
    ) -> Result<(), Error> {
        (self.cut)(visit)
    }
    fn requires_prepared_traversal(&self) -> bool {
        self.observer.requires_prepared_traversal()
    }
    fn supports_prefill_spans(&self) -> bool {
        self.observer.supports_prefill_spans()
    }
    fn supports_prefill_context(&self) -> bool {
        self.observer.supports_prefill_context()
    }
    fn begin_prefill_context(&mut self, frontier: u64) -> Result<(), Error> {
        self.observer.begin_prefill_context(frontier)
    }
    fn requires_sequence_readout(&self) -> bool {
        self.observer.requires_sequence_readout()
    }
    fn admitted_prefill_capture(
        &self,
    ) -> Option<&eredu_runtime::working_memory::AdmittedPrefillCapture<'_>> {
        self.observer.admitted_prefill_capture()
    }
    fn ordinary_prefill_capture(&self) -> Option<&eredu_runtime::capture::OrdinaryPrefillCapture> {
        self.observer.ordinary_prefill_capture()
    }
    fn admitted_capture_continuation(
        &self,
    ) -> Option<&eredu_runtime::working_memory::AdmittedCaptureContinuation<'_>> {
        self.observer.admitted_capture_continuation()
    }

    fn original_speculative_capture(
        &self,
    ) -> Option<eredu_runtime::capture::OriginalSpeculativeCaptureInvocation<'_>> {
        self.observer.original_speculative_capture()
    }
    fn retain_original_speculative_capture(
        &mut self,
        capture: eredu_core::speculative::SpeculativeActivationCapture,
    ) -> Result<(), eredu_runtime::capture::CaptureProtocolError> {
        self.observer.retain_original_speculative_capture(capture)
    }

    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<WorkspaceTensor>>, Error> {
        self.observer.routed_unit_observer(path)
    }
    fn transactional(&self) -> bool {
        self.observer.transactional()
    }
    fn begin_prefill_chunk(
        &mut self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
    ) -> Result<(), Error> {
        self.observer.begin_prefill_chunk(chunk)
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.observer.finish_prefill(committed);
    }
    fn requires_prefill_opening_state(&self) -> bool {
        self.observer.requires_prefill_opening_state()
    }
    fn prepare_prefill_chunk_with_opening(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
        opening: &PrefillOpeningState<'_, WorkspaceTensor>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, Error> {
        self.observer
            .prepare_prefill_chunk_with_opening(context, opening)
    }
    fn prepare_prefill_chunk_retention(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, Error> {
        self.observer.prepare_prefill_chunk_retention(context)
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        settled: SettledPrefillChunkRetention,
    ) -> Result<(), Error> {
        self.observer.retire_prefill_chunk_retention(settled)
    }
    fn prepare_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        self.observer.prepare_transaction(epoch, pass)
    }
    fn coordinate_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), Error> {
        self.observer.coordinate_transaction(epoch)
    }
    fn complete_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), Error> {
        self.observer.complete_transaction(epoch)
    }
    fn finish_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch, committed: bool) {
        self.observer.finish_transaction(epoch, committed)
    }
    fn observe(&mut self, path: &str, value: &WorkspaceTensor) -> Result<(), Error> {
        self.observer.observe(path, value)
    }
    fn observe_replica(&mut self, path: &str, value: &WorkspaceTensor) -> Result<(), Error> {
        self.observer.observe_replica(path, value)
    }
    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &WorkspaceTensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<WorkspaceTensor, Error>,
    ) -> Result<(), Error> {
        self.observer
            .observe_generated(path, prototype, source, generate)
    }
    /// Forward the actual generated program and its caller-owned root retention.
    fn observe_generated_retained(
        &mut self,
        path: &str,
        prototype: &WorkspaceTensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<WorkspaceTensor, Error>,
    ) -> Result<(), Error> {
        self.observer
            .observe_generated_retained(path, prototype, source, factory)
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &WorkspaceTensor,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        self.observer.intervene(path, value)
    }
    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Error> {
        self.observer.routing_control(path, rows)
    }
    /// Declares ordinary-decision interest without allocation or native work.
    fn routing_unmodified_interest(&self, path: &str) -> RoutingUnmodifiedInterest {
        self.observer.routing_unmodified_interest(path)
    }

    fn routing_unmodified(
        &mut self,
        path: &str,
        effective: RoutingDecision<'_, WorkspaceTensor>,
    ) -> Result<(), Error> {
        self.observer.routing_unmodified(path, effective)
    }

    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<RoutingDecision<'_, WorkspaceTensor>>,
        effective: RoutingDecision<'_, WorkspaceTensor>,
    ) -> Result<(), Error> {
        self.observer.routing_applied(path, original, effective)
    }
    fn routing_failed(&mut self, path: &str, message: &str) {
        self.observer.routing_failed(path, message)
    }
    fn observe_routing(
        &mut self,
        routing: RoutingObservation<'_, WorkspaceTensor>,
    ) -> Result<(), Error> {
        self.observer.observe_routing(routing)
    }
    fn finish(&mut self) -> Result<(), Error> {
        self.observer.finish()
    }
}
