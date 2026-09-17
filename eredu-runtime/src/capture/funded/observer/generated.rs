//! Generated sources share the original ledger, frame and native work.
use super::*;
use eredu_nn::{GeneratedTensorSourceRole, RetainedGeneratedTensorFactory};

// The scalar controls are part of fixed session H. T is a lexical caller-owned
// tensor handle, never a copied host value buffer or a new numerical grant.
pub(in crate::capture::funded) struct GeneratedState<T> {
    pub(super) value: Option<T>,
    pub(super) sources: usize,
    pub(super) outputs: usize,
}
pub(super) enum Failure<E: std::error::Error + 'static, N> {
    Capture(FundedCaptureError<E>),
    Factory(N),
}
impl<E: std::error::Error + 'static, N> From<FundedCaptureError<E>> for Failure<E, N> {
    fn from(error: FundedCaptureError<E>) -> Self {
        Self::Capture(error)
    }
}

/// One physical producer protocol shared by whole-value and fragment targets.
pub(super) enum Retention {
    Source(GeneratedTensorSourceRole),
    Output,
}
pub(super) fn generate_once<T, E: std::error::Error + 'static, N>(
    state: &mut GeneratedState<T>,
    factory: &mut dyn RetainedGeneratedTensorFactory<T, N>,
    map_error: &dyn Fn(FundedCaptureError<E>) -> N,
    mut retain: impl FnMut(Retention, &T) -> Result<(), FundedCaptureError<E>>,
) -> Result<(), Failure<E, N>> {
    if state.value.is_some() {
        return Ok(());
    }
    factory
        .visit_sources(&mut |role, value| {
            let expected = match state.sources {
                0 => GeneratedTensorSourceRole::CompactValues,
                1 => GeneratedTensorSourceRole::BlockScales,
                _ => return Err(map_error(CaptureProtocolError::GeneratedSource.into())),
            };
            if role != expected {
                return Err(map_error(CaptureProtocolError::GeneratedSource.into()));
            }
            retain(Retention::Source(role), value).map_err(map_error)?;
            state.sources += 1;
            Ok(())
        })
        .map_err(Failure::Factory)?;
    if state.sources != 2 {
        return Err(FundedCaptureError::from(CaptureProtocolError::GeneratedSource).into());
    }
    let value = factory
        .generate(&mut |value| {
            if state.outputs == 7 {
                return Err(map_error(CaptureProtocolError::GeneratedSource.into()));
            }
            retain(Retention::Output, value).map_err(map_error)?;
            state.outputs += 1;
            Ok(())
        })
        .map_err(Failure::Factory)?;
    if state.outputs != 7 {
        return Err(FundedCaptureError::from(CaptureProtocolError::GeneratedSource).into());
    }
    state.value = Some(value);
    Ok(())
}

impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    pub(super) fn observe_retained_generated(
        &mut self,
        path: &str,
        prototype: &T,
        source: &GeneratedCaptureSource,
        factory: &mut dyn RetainedGeneratedTensorFactory<T, N>,
    ) -> Result<(), N> {
        if self.bound.is_some() {
            return self.observe_fragment_generated(path, prototype, source, factory);
        }
        if !self.active_transaction() {
            return Err((self.map_error)(CaptureProtocolError::Transaction.into()));
        }
        if matches!(self.frame, Frame::Empty) {
            return Ok(());
        }
        let mut generated = GeneratedState {
            value: None,
            sources: 0,
            outputs: 0,
        };
        for index in 0..self.session.plan.plan().selections.len() {
            let Frame::Active(frame) = &self.frame else {
                return Err((self.map_error)(CaptureProtocolError::Transaction.into()));
            };
            if !policy::selected_record(
                &self.session.plan.plan().selections[index],
                &frame.records()[index],
                path,
            )
            .map_err(|error| (self.map_error)(error.into()))?
            {
                continue;
            }
            let started = std::time::Instant::now();
            let mut dtype = None;
            let mut charged = CaptureUsage::default();
            let result = self.generated_one(
                index,
                prototype,
                source,
                factory,
                &mut generated,
                &mut dtype,
                &mut charged,
            );
            self.session.capture_seconds += started.elapsed().as_secs_f64();
            if let Err(error) = result {
                let reason = match &error {
                    Failure::Capture(FundedCaptureError::Admission(error)) => {
                        policy::failure_reason(error)
                    }
                    Failure::Factory(_) | Failure::Capture(FundedCaptureError::Backend(_)) => {
                        CaptureFailureReason::Native
                    }
                    _ => CaptureFailureReason::Invalid,
                };
                if let Frame::Active(frame) = &mut self.frame {
                    if matches!(frame.records()[index].outcome, CaptureOutcome::Missing) {
                        let _ = frame.record_failure(
                            index,
                            reason,
                            "generated capture operation failed",
                            dtype,
                            charged,
                        );
                    }
                }
                return Err(match error {
                    Failure::Factory(original) => original,
                    Failure::Capture(error) => (self.map_error)(error),
                });
            }
        }
        Ok(())
    }

    fn generated_one(
        &mut self,
        index: usize,
        prototype: &T,
        source: &GeneratedCaptureSource,
        factory: &mut dyn RetainedGeneratedTensorFactory<T, N>,
        generated: &mut GeneratedState<T>,
        dtype: &mut Option<TensorDtype>,
        charged: &mut CaptureUsage,
    ) -> Result<(), Failure<E, N>> {
        let policy = CaptureObservationStep::with_invocation(
            &self.session.plan,
            self.session.phase,
            self.session.prediction,
            self.invocation.map(|value| value.1),
        )
        .and_then(|policy| policy.with_window(self.window))
        .map_err(FundedCaptureError::from)?;
        let fragment = if let Some(usage) = policy
            .window_metadata_usage(index)
            .map_err(FundedCaptureError::from)?
        {
            if let Some(reason) = policy
                .reserve_value(&mut self.session.ledger, usage)
                .map_err(FundedCaptureError::from)?
            {
                let Frame::Active(frame) = &mut self.frame else {
                    return Err(FundedCaptureError::from(CaptureProtocolError::Transaction).into());
                };
                frame
                    .record_skip(index, reason, None, CaptureUsage::default())
                    .map_err(FundedCaptureError::from)?;
                return Ok(());
            }
            *charged = usage;
            usage
        } else {
            CaptureUsage::default()
        };
        let geometry = policy
            .tensor_geometry(index)
            .map_err(|e| FundedCaptureError::from(CaptureRunHostError::Step(e.into())))?;
        // The actual prototype supplies shape; the program supplies generated
        // precision. Native FP8 may legitimately receive a half prototype.
        self.backend
            .validate_source(prototype, &geometry)
            .map_err(FundedCaptureError::Backend)?;
        policy
            .validate_generated(index, source, factory.program())
            .map_err(FundedCaptureError::from)?;
        *dtype = Some(TensorDtype::F32);
        if policy.window().is_some()
            && geometry
                .starts()
                .iter()
                .zip(geometry.ends())
                .any(|(a, b)| a == b)
        {
            let Frame::Active(frame) = &mut self.frame else {
                return Err(FundedCaptureError::from(CaptureProtocolError::Transaction).into());
            };
            frame
                .record_window_empty(index, TensorDtype::F32, fragment)
                .map_err(FundedCaptureError::from)?;
            frame
                .validate_record_encoding(index)
                .map_err(FundedCaptureError::from)?;
            return Ok(());
        }
        let usage = self
            .backend
            .estimate_generated(prototype, source, &geometry)
            .map_err(FundedCaptureError::from)?;
        let usage = policy
            .generated_usage(usage, source)
            .map_err(FundedCaptureError::from)?;
        let empty = policy
            .generated_slice_empty(index)
            .map_err(FundedCaptureError::from)?;
        let Frame::Active(frame) = &mut self.frame else {
            return Err(FundedCaptureError::from(CaptureProtocolError::Transaction).into());
        };
        if let Some(reason) = policy
            .reserve_value(&mut self.session.ledger, usage)
            .map_err(FundedCaptureError::from)?
        {
            frame
                .record_skip(index, reason, Some(TensorDtype::F32), fragment)
                .map_err(FundedCaptureError::from)?;
            return Ok(());
        }
        *charged = fragment
            .checked_add(usage)
            .map_err(FundedCaptureError::from)?;
        let claim = frame.take_tensor(index).map_err(FundedCaptureError::from)?;
        // The claim is irreversibly spent before any producer/retention call.
        // No scope borrow returned by preflight survives into those calls.
        self.backend.preflight_generated(
            prototype,
            source,
            factory.program(),
            !empty && generated.value.is_none(),
            &claim,
        )?;
        let receipt = if empty {
            let builder = claim.prepare().map_err(FundedCaptureError::from)?;
            builder
                .finish()
                .map_err(|error| FundedCaptureError::HostFinish(error.into_owned_error()))?
        } else {
            generate_once(
                generated,
                factory,
                self.map_error,
                |event, value| match event {
                    Retention::Source(role) => self
                        .backend
                        .retain_generated_source(prototype, role, value, &claim),
                    Retention::Output => self.backend.retain_generated_output(value, &claim),
                },
            )?;
            let value = generated
                .value
                .as_ref()
                .expect("one generated value for this hook");
            let actual = self
                .backend
                .validate_source(value, &geometry)
                .map_err(FundedCaptureError::Backend)?;
            if actual != TensorDtype::F32 {
                return Err(FundedCaptureError::from(CaptureError::Invalid(
                    "generated observation changed its declared element type".into(),
                ))
                .into());
            }
            self.backend
                .transform(value, claim)
                .map_err(FundedCaptureError::Backend)?
        };
        frame
            .record_tensor(receipt, TensorDtype::F32, *charged)
            .map_err(FundedCaptureError::from)?;
        frame
            .validate_record_encoding(index)
            .map_err(FundedCaptureError::from)?;
        Ok(())
    }
}
