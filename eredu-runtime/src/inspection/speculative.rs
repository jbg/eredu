//! Explicit phases for internal speculative observations, separate from sampling.
use super::ActivationObserver;
use eredu_core::speculative::{SpeculativeActivationOrigin, SpeculativeActivationPhase};

/// Internal observation authority installed separately from raw sampling logits.
/// Implementations share cumulative allowances across snapshots and speculative
/// forks; ending a failed invocation never refunds native work or host transport.
pub trait SpeculativeActivationObserver<T, E>: ActivationObserver<T, E> {
    /// Complete host bound for saved internal authority at a drained boundary.
    fn activation_checkpoint_bytes(&self) -> Option<u64> {
        None
    }
    /// Saves authority without refunding its cumulative invocation allowance.
    fn activation_checkpoint(
        &self,
    ) -> Result<
        crate::capture::SpeculativeActivationCheckpoint,
        eredu_core::speculative::SpeculativeControlError,
    > {
        Err(
            eredu_core::speculative::SpeculativeControlError::Unsupported(
                "internal observer has no authority checkpoint",
            ),
        )
    }
    /// Prepares an infallible authority commit before native state is replaced.
    fn prepare_activation_restore<'a>(
        &'a mut self,
        _saved: &crate::capture::SpeculativeActivationCheckpoint,
    ) -> Result<
        Box<dyn crate::capture::PreparedSpeculativeActivationRestore + 'a>,
        eredu_core::speculative::SpeculativeControlError,
    > {
        Err(
            eredu_core::speculative::SpeculativeControlError::Unsupported(
                "internal observer has no authority restore",
            ),
        )
    }
    /// Mandatory admission validation over loaded discovery. An original observer
    /// replaces only this producer with its exact retained source validator.
    fn validate_activation_readmission(&self,
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        discovery: Option<&eredu_core::speculative::SpeculativeActivationDiscovery>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        plan.validate(discovery.ok_or(eredu_core::speculative::SpeculativeControlError::Unsupported(
            "loaded execution has no internal activation discovery",
        ))?).map_err(Into::into)
    }
    /// Replaces prospective internal edits while keeping capture authority fixed.
    /// The enclosing loaded controller validates exact discovery before this call.
    fn readmit_activation_interventions(
        &mut self,
        _plan: eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        Err(
            eredu_core::speculative::SpeculativeControlError::Unsupported(
                "internal observer cannot replace interventions",
            ),
        )
    }
    /// Lends exact span coordinates until this invocation is closed.
    fn set_prefill_span(&mut self, _span: Option<eredu_core::speculative::SpeculativePrefillSpan>) {
    }

    /// Declares the actual selected target/seed extents before any split hook.
    /// This scalar annotation grants no allocation or completion authority.
    fn set_prefill_reduction_geometry(
        &mut self,
        _geometry: eredu_core::speculative::SpeculativePrefillReductionGeometry,
    ) {
    }

    /// Validates complete logical coverage while the last span is still under
    /// the existing fallible completion/agreement boundary. Does not publish.
    fn complete_prefill_reductions(&mut self) -> Result<(), E> {
        Ok(())
    }

    /// Ends logical prefill after final score selection, reservation settlement
    /// and target-state exchange. Must not allocate, submit native work, or
    /// communicate; may run on unwind. False never refunds prior observations.
    fn finish_prefill_reductions(&mut self, _success: bool) {}

    /// Receives the shared scheduler's exact request and prefix coordinates.
    /// Clearing this scope performs no native work and does not discard already
    /// staged records. Direct low-level execution may have no scheduler origin.
    fn set_activation_origin(&mut self, _origin: Option<SpeculativeActivationOrigin>) {}

    /// Moves one already charged record to the consumer. Returning a record
    /// neither commits tentative output nor refunds any cumulative allowance.
    fn take_activation_capture(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeActivationCapture> {
        None
    }

    /// Drains the original typed admission failure across a native error domain.
    fn take_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        None
    }

    /// Admits this actual forward's phase and physical sequence width before
    /// internal observation work. This is not a generated-token prediction index.
    fn begin_activation_invocation(
        &mut self,
        phase: SpeculativeActivationPhase,
        sequence: usize,
    ) -> Result<(), E>;

    /// Lends one observer for the complete admitted forward. Implementations may
    /// retain producer work and coordination in a stack-owned observer throughout
    /// this call. Successful admission must invoke `operation` exactly once,
    /// preserving its error, and drop the borrowed observer before returning.
    /// The default uses this owner without
    /// allocation; completion and state recovery remain with the executor.
    fn with_activation_observer(
        &mut self,
        operation: &mut dyn FnMut(&mut dyn ActivationObserver<T, E>) -> Result<(), E>,
    ) -> Result<(), E> {
        operation(&mut super::BorrowedActivationObserver(self))
    }

    /// Stages already collected records after a successful call. Verification
    /// and proposal records remain tentative; this does not commit output tokens
    /// or replace the executor's native completion and state ownership.
    fn complete_activation_invocation(&mut self) -> Result<(), E>;

    /// Closes every attempted invocation, including admission or execution failure.
    /// Must perform no native work or communication; retained failure/completion
    /// resources stay with their existing owner. This may run during unwinding.
    fn finish_activation_invocation(&mut self, success: bool);
}

/// Runs one actual phase without installing a no-op internal observer when
/// instrumentation is absent. The enclosing driver retains state/completion.
pub fn with_speculative_activation<T, E, R>(
    observer: Option<&mut dyn SpeculativeActivationObserver<T, E>>,
    phase: SpeculativeActivationPhase,
    sequence: usize,
    operation: impl FnOnce(Option<&mut dyn ActivationObserver<T, E>>) -> Result<R, E>,
) -> Result<R, E> {
    let Some(observer) = observer else {
        return operation(None);
    };
    struct Guard<'a, T, E> {
        observer: &'a mut dyn SpeculativeActivationObserver<T, E>,
        success: bool,
    }
    impl<T, E> Drop for Guard<'_, T, E> {
        fn drop(&mut self) {
            self.observer.finish_activation_invocation(self.success);
        }
    }
    let mut guard = Guard {
        observer,
        success: false,
    };
    guard
        .observer
        .begin_activation_invocation(phase, sequence)?;
    let mut operation = Some(operation);
    let mut output = None;
    guard.observer.with_activation_observer(&mut |observer| {
        output = Some(operation.take().expect("invocation operation runs once")(
            Some(observer),
        )?);
        Ok(())
    })?;
    let output = output.expect("invocation observer must execute its operation");
    guard.observer.complete_activation_invocation()?;
    guard.success = true;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Observer {
        phases: Vec<(SpeculativeActivationPhase, usize)>,
        dispositions: Vec<bool>,
        completed: usize,
        remaining: usize,
        fail_completion: bool,
        borrowed: bool,
        dropped: usize,
    }
    impl ActivationObserver<i32, &'static str> for Observer {
        fn observe(&mut self, _: &str, _: &i32) -> Result<(), &'static str> {
            Ok(())
        }
    }
    impl SpeculativeActivationObserver<i32, &'static str> for Observer {
        fn with_activation_observer(
            &mut self,
            operation: &mut dyn FnMut(
                &mut dyn ActivationObserver<i32, &'static str>,
            ) -> Result<(), &'static str>,
        ) -> Result<(), &'static str> {
            struct Borrow<'a>(&'a mut Observer);
            impl ActivationObserver<i32, &'static str> for Borrow<'_> {
                fn observe(&mut self, _: &str, _: &i32) -> Result<(), &'static str> {
                    assert!(self.0.borrowed);
                    Ok(())
                }
            }
            impl Drop for Borrow<'_> {
                fn drop(&mut self) {
                    self.0.borrowed = false;
                    self.0.dropped += 1;
                }
            }
            assert!(!self.borrowed);
            self.borrowed = true;
            operation(&mut Borrow(self))
        }
        fn begin_activation_invocation(
            &mut self,
            phase: SpeculativeActivationPhase,
            sequence: usize,
        ) -> Result<(), &'static str> {
            self.phases.push((phase, sequence));
            self.remaining = self.remaining.checked_sub(sequence).ok_or("budget")?;
            Ok(())
        }
        fn complete_activation_invocation(&mut self) -> Result<(), &'static str> {
            assert!(!self.borrowed, "borrowed work must end before completion");
            if self.fail_completion {
                return Err("delivery");
            }
            self.completed += 1;
            Ok(())
        }
        fn finish_activation_invocation(&mut self, success: bool) {
            assert!(!self.borrowed, "borrowed work must end before finalization");
            self.dispositions.push(success);
        }
    }
    #[test]
    fn physical_phase_geometry_and_failure_never_refund_consumed_allowance() {
        let mut observer = Observer {
            remaining: 6,
            ..Default::default()
        };
        with_speculative_activation(
            Some(&mut observer),
            SpeculativeActivationPhase::PredictionPrefill,
            3,
            |observer| observer.unwrap().observe("input", &7),
        )
        .unwrap();
        assert_eq!(
            with_speculative_activation(
                Some(&mut observer),
                SpeculativeActivationPhase::Proposal { depth: 2 },
                1,
                |_| Err::<(), _>("execution")
            ),
            Err("execution")
        );
        observer.fail_completion = true;
        assert_eq!(
            with_speculative_activation(
                Some(&mut observer),
                SpeculativeActivationPhase::PredictionReplay,
                2,
                |_| Ok(())
            ),
            Err("delivery")
        );
        assert_eq!(
            with_speculative_activation(
                Some(&mut observer),
                SpeculativeActivationPhase::TargetReplay,
                1,
                |_| panic!("denied before work")
            ),
            Err::<(), _>("budget")
        );
        assert_eq!(observer.remaining, 0);
        assert_eq!(observer.dispositions, [true, false, false, false]);
        assert_eq!(observer.completed, 1);
        assert_eq!(observer.dropped, 3);
        assert_eq!(
            observer.phases[1],
            (SpeculativeActivationPhase::Proposal { depth: 2 }, 1)
        );
        with_speculative_activation::<i32, &'static str, _>(
            None,
            SpeculativeActivationPhase::Verification,
            4,
            |observer| {
                assert!(observer.is_none());
                Ok(())
            },
        )
        .unwrap();
    }
    #[test]
    fn unwinding_aborts_invocation_without_refunding_admission() {
        let mut observer = Observer {
            remaining: 1,
            ..Default::default()
        };
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_speculative_activation(
                Some(&mut observer),
                SpeculativeActivationPhase::Verification,
                1,
                |_| -> Result<(), &'static str> { panic!("injected panic") },
            )
        }));
        assert!(failed.is_err());
        assert_eq!(observer.remaining, 0);
        assert_eq!(observer.dispositions, [false]);
        assert_eq!(observer.dropped, 1);
    }
}
