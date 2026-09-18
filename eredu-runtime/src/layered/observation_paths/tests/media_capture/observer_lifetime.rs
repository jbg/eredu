use super::*;
use crate::{intervention::CaptureObserver, ActivationObserver};
use eredu_core::intervention::{InterventionBackend, InterventionDtype, InterventionTensor};

// This fixture has capture admission only. No intervention primitive may run.
impl InterventionBackend for Backend {
    fn intervention_dtype(&self, _: &Self::Tensor) -> Result<InterventionDtype, Failure> {
        Err(Failure)
    }
    fn validate_intervention_geometry(
        &self,
        _: &[u64],
        _: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        Err(CaptureError::Unsupported("capture-only fixture".into()))
    }
    fn select_region(
        &mut self,
        _: &Self::Tensor,
        _: &ResolvedCaptureSlice,
    ) -> Result<Self::Tensor, Failure> {
        Err(Failure)
    }
    fn update_region(
        &mut self,
        _: &Self::Tensor,
        _: &ResolvedCaptureSlice,
        _: &Self::Tensor,
    ) -> Result<Self::Tensor, Failure> {
        Err(Failure)
    }
    fn zeros(&mut self, _: &[u64], _: InterventionDtype) -> Result<Self::Tensor, Failure> {
        Err(Failure)
    }
    fn scale(&mut self, _: &Self::Tensor, _: f32) -> Result<Self::Tensor, Failure> {
        Err(Failure)
    }
    fn fill_masked(
        &mut self,
        _: &Self::Tensor,
        _: &[bool],
        _: f32,
    ) -> Result<Self::Tensor, Failure> {
        Err(Failure)
    }
    fn mask_components(
        &mut self,
        _: &Self::Tensor,
        _: &[u32],
        _: bool,
    ) -> Result<Self::Tensor, Failure> {
        Err(Failure)
    }
    fn realize_tensor(&mut self, _: &InterventionTensor) -> Result<Self::Tensor, Failure> {
        Err(Failure)
    }
    fn add(&mut self, _: &Self::Tensor, _: &Self::Tensor) -> Result<Self::Tensor, Failure> {
        Err(Failure)
    }
    fn fill_columns(
        &mut self,
        _: &Self::Tensor,
        _: &[u32],
        _: f32,
    ) -> Result<Self::Tensor, Failure> {
        Err(Failure)
    }
}
fn assert_mode(run: &mut CaptureSession, prepared: bool, p0: bool) {
    let observer = CaptureObserver::new(
        run,
        Backend::default(),
        |error: crate::capture::CaptureExecutionError<Failure>| error,
    );
    assert_eq!(observer.requires_prepared_traversal(), prepared);
    assert_eq!(observer.supports_prefill_spans(), p0);
    assert_eq!(observer.ordinary_prefill_capture().is_some(), p0);
}

#[test]
fn ordinary_prepared_traversal_survives_prefill_drain_decode_and_restore() {
    // An installed empty selection still authenticates a real prepared source.
    let selected = binding(false, true, limits());
    let d = discovery(&selected);
    let mut legacy = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(selected.source().admission().clone()));
    assert_mode(&mut legacy, false, false);
    let mut run =
        CaptureSession::with_ordinary_prefill(selected, &HostPreparationAuthority::default())
            .unwrap();
    let saved = run.checkpoint(&d).unwrap();
    assert_mode(&mut run, true, true);

    assert!(advance(&mut run, &mut Backend::default(), None, true));
    let prefill = run.take_shared_step().unwrap();
    assert_eq!(prefill.phase(), CapturePhase::Prefill);
    assert_eq!(prefill.outcome(), CaptureStepOutcome::Committed);
    assert_mode(&mut run, true, false);

    let epoch = DistributedCommitEpoch::FIRST
        .next()
        .unwrap()
        .next()
        .unwrap()
        .next()
        .unwrap();
    {
        let mut observer = CaptureObserver::for_step(
            &mut run,
            Backend::default(),
            1,
            |error: crate::capture::CaptureExecutionError<Failure>| error,
        );
        observer
            .prepare_transaction(epoch, crate::ExpertPass::Decode)
            .unwrap();
        assert!(observer.requires_prepared_traversal());
        assert!(!observer.supports_prefill_spans());
        observer.complete_transaction(epoch).unwrap();
        observer.finish_transaction(epoch, true);
    }
    let decoded = run.take_shared_step().unwrap();
    assert_eq!(decoded.phase(), CapturePhase::Decode);
    assert_eq!(decoded.prediction_index(), 1);
    assert_eq!(decoded.outcome(), CaptureStepOutcome::Committed);
    assert!(decoded.cumulative_usage().host_bytes > prefill.cumulative_usage().host_bytes);
    assert_mode(&mut run, true, false);

    let spent = run.cumulative_usage();
    run.restore(&saved).unwrap();
    assert_eq!(run.cumulative_usage(), spent);
    assert_mode(&mut run, true, true);
    // Restore rewinds scheduling but never reuses an already consumed epoch.
    assert!(advance_from_epoch(
        &mut run,
        &mut Backend::default(),
        None,
        true,
        epoch.next().unwrap(),
    ));
    assert!(run.take_shared_step().is_some());
    assert_mode(&mut run, true, false);
}
