//! Existing routed provider callbacks with the original partition hook owners.
use super::*;
impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    pub(super) fn begin_partition_routed(
        &mut self,
        invocation: &crate::RoutedUnitInvocation<'_, T>,
    ) -> Result<(), FundedCaptureError<E>> {
        if self.routed_partition_hooks.is_some() {
            return Err(CaptureProtocolError::Transaction.into());
        }
        let first = self
            .routed_selection
            .ok_or(CaptureProtocolError::Transaction)?;
        let chunk = if self.prefill.is_some() {
            Some(self.current_fragment_chunk()?)
        } else {
            None
        };
        let inference = if chunk.is_some() {
            Some(
                self.bound
                    .ok_or(CaptureProtocolError::Transaction)?
                    .geometry(),
            )
        } else {
            None
        };
        let Frame::Active(frame) = &mut self.frame else {
            return Err(CaptureProtocolError::Transaction.into());
        };
        let program = self
            .backend
            .partition_capture()
            .ok_or(CaptureProtocolError::Transaction)?;
        let scope = program
            .take_routed_hooks(first, frame, chunk.as_ref().zip(inference))?
            .ok_or(CaptureProtocolError::Transaction)?;
        self.routed_partition_hooks = Some(scope.begin(self.backend, invocation)?);
        Ok(())
    }
    pub(super) fn partition_routed_batch(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, T>,
        effective: bool,
    ) -> Result<(), FundedCaptureError<E>> {
        if !self.routed_active {
            return Err(CaptureProtocolError::Transaction.into());
        }
        let scope = self
            .routed_partition_hooks
            .take()
            .ok_or(CaptureProtocolError::Transaction)?;
        let started = std::time::Instant::now();
        let result = scope.observe(self.backend, batch, effective);
        self.session.capture_seconds += started.elapsed().as_secs_f64();
        self.routed_partition_hooks = Some(result?);
        Ok(())
    }
    pub(super) fn finish_partition_routed(
        &mut self,
        success: bool,
    ) -> Result<(), FundedCaptureError<E>> {
        let scope = self
            .routed_partition_hooks
            .take()
            .ok_or(CaptureProtocolError::Transaction)?;
        self.routed_active = false;
        let scope = scope.finish::<T, E>(success)?;
        let chunk = if self.prefill.is_some() {
            Some(self.current_fragment_chunk()?)
        } else {
            None
        };
        let inference = if chunk.is_some() {
            Some(
                self.bound
                    .ok_or(CaptureProtocolError::Transaction)?
                    .geometry(),
            )
        } else {
            None
        };
        let Frame::Active(frame) = &mut self.frame else {
            return Err(scope
                .reject("sparse callback frame is no longer active")
                .into());
        };
        let Some(program) = self.backend.partition_capture() else {
            return Err(scope
                .reject("sparse callback lost its original program")
                .into());
        };
        program.return_routed_hooks(scope, frame, chunk.as_ref().zip(inference))?;
        Ok(())
    }
}
