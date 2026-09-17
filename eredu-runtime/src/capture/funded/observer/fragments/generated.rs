//! Per-hook physical factory, per-selection full logical creation quota.
use super::super::generated::{generate_once, Failure, GeneratedState, Retention};
use super::*;
use eredu_nn::RetainedGeneratedTensorFactory;

impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    pub(in crate::capture::funded::observer) fn observe_fragment_generated(
        &mut self,
        path: &str,
        prototype: &T,
        source: &GeneratedCaptureSource,
        factory: &mut dyn RetainedGeneratedTensorFactory<T, N>,
    ) -> Result<(), N> {
        let chunk = self
            .current_fragment_chunk()
            .map_err(|e| (self.map_error)(e.into()))?;
        if matches!(self.frame, Frame::Empty) {
            return Ok(());
        }
        let bound = self.bound.expect("checked fragment route");
        let policy = CapturePrefillObservationPolicy::from_bound(bound)
            .map_err(|e| (self.map_error)(progress_error(e)))?;
        let mut generated = GeneratedState {
            value: None,
            sources: 0,
            outputs: 0,
        };
        for index in 0..bound
            .selection()
            .source()
            .admission()
            .plan()
            .selections
            .len()
        {
            let row = policy
                .row(index)
                .map_err(|e| (self.map_error)(progress_error(e)))?;
            let Frame::Active(frame) = &mut self.frame else {
                return Err((self.map_error)(CaptureProtocolError::Transaction.into()));
            };
            let decision = frame
                .begin_prefill_hook(index, &chunk, path)
                .map_err(|e| (self.map_error)(e.into()))?;
            if decision == CapturePrefillHookDecision::Ignore {
                continue;
            }
            let started = std::time::Instant::now();
            let result: Result<(), Failure<E, N>> = (|| {
                let assembly = row
                    .assembly()
                    .ok_or_else(|| FundedCaptureError::from(CaptureProtocolError::Geometry))?;
                let fragment = assembly
                    .fragment(chunk.input.start / bound.geometry().prefill_chunk_positions)
                    .map_err(|e| progress_error(CapturePrefillProgressError::from(e)))?;
                self.backend.validate_prefill_source(prototype, &fragment)?;
                row.validate_generated_fragment(&fragment, source, factory.program())
                    .map_err(progress_error)?;
                if decision == CapturePrefillHookDecision::First {
                    let usage = self
                        .backend
                        .estimate_prefill(assembly.logical_geometry())
                        .map_err(FundedCaptureError::from)?;
                    let usage = row.full_generated_usage(usage).map_err(progress_error)?;
                    if frame
                        .reserve_prefill_hook(
                            index,
                            &mut self.session.ledger,
                            TensorDtype::F32,
                            usage,
                        )
                        .map_err(FundedCaptureError::from)?
                        .is_some()
                    {
                        return Ok(());
                    }
                }
                let construct = row.factory_required(&fragment).map_err(progress_error)?;
                if construct {
                    // Claim first even for Preview(0). The actual physical
                    // reconstruction still runs once for a nonempty slice.
                    let claim = frame
                        .take_prefill_fragment(index, &fragment)
                        .map_err(FundedCaptureError::from)?;
                    self.backend.preflight_generated_fragment(
                        prototype,
                        source,
                        factory.program(),
                        generated.value.is_none(),
                        &claim,
                    )?;
                    generate_once(&mut generated, factory, self.map_error, |event, value| {
                        match event {
                            Retention::Source(role) => self
                                .backend
                                .retain_generated_fragment_source(prototype, role, value, &claim),
                            Retention::Output => {
                                self.backend.retain_generated_fragment_output(value, &claim)
                            }
                        }
                    })?;
                    let value = generated.value.as_ref().expect("one factory per hook");
                    if self.backend.validate_prefill_source(value, &fragment)? != TensorDtype::F32 {
                        return Err(FundedCaptureError::from(
                            CaptureProtocolError::GeneratedSource,
                        )
                        .into());
                    }
                    self.backend.transform_prefill_fragment(value, claim)?;
                } else if decision == CapturePrefillHookDecision::First
                    && assembly.logical_geometry().elements() == 0
                {
                    // Truly empty pretransform slice: no factory or native work.
                    frame
                        .take_prefill_fragment(index, &fragment)
                        .map_err(FundedCaptureError::from)?
                        .prepare()
                        .map_err(FundedCaptureError::from)?
                        .finish()
                        .map_err(FundedCaptureError::from)?;
                }
                frame
                    .finish_prefill_hook(index, &fragment)
                    .map_err(FundedCaptureError::from)?;
                Ok(())
            })();
            self.session.capture_seconds += started.elapsed().as_secs_f64();
            if let Err(error) = result {
                match &error {
                    Failure::Capture(error) => {
                        self.fail_fragment(index, Some(TensorDtype::F32), error)
                    }
                    Failure::Factory(_) => {
                        // Preserve the original N untouched; the same frame keeps
                        // every partial target and its already charged full usage.
                        self.fail_fragment(
                            index,
                            Some(TensorDtype::F32),
                            &CaptureProtocolError::GeneratedSource.into(),
                        );
                    }
                }
                return Err(match error {
                    Failure::Factory(e) => e,
                    Failure::Capture(e) => (self.map_error)(e),
                });
            }
        }
        Ok(())
    }
}
