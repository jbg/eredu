//! Lexical mechanism loans around the existing session driver.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// Refusal before a backend mechanism binding or its operation is entered.
#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum ExecutionMechanismBindingCause {
    /// The session is not at the existing quiescent entry boundary.
    #[error(transparent)]
    Boundary(#[from] RuntimeInspectionBoundary),
    /// The actual caller's fixed host controls could not be retained.
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
}
struct Restore<'a, S, M, C> {
    owner: &'a mut S,
    mechanisms: fn(&mut S) -> &mut M,
    token: Option<C>,
    restore: fn(&mut M, C),
}
impl<S, M, C> Drop for Restore<'_, S, M, C> {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            (self.restore)((self.mechanisms)(self.owner), token);
        }
    }
}
fn binding_control_bytes<S, M, C, T, E, I, F>() -> Option<usize> {
    let controls = [
        size_of::<Restore<'_, S, M, C>>(),
        size_of::<C>(),
        size_of::<Option<C>>(),
        size_of::<Result<C, E>>(),
        size_of::<Result<Result<T, E>, HostMetadataFundingError>>(),
        size_of::<I>(),
        size_of::<F>(),
        size_of::<T>(),
        size_of::<(
            &mut S,
            &HostMetadataFunding,
            fn(&mut S) -> &mut M,
            fn(&mut M, C),
        )>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
fn with_binding<S, M, C, T, E, I, F>(
    owner: &mut S,
    funding: &HostMetadataFunding,
    mechanisms: fn(&mut S) -> &mut M,
    install: I,
    restore: fn(&mut M, C),
    run: F,
) -> Result<Result<T, E>, HostMetadataFundingError>
where
    I: FnOnce(&mut M) -> Result<C, E>,
    F: FnOnce(&mut S) -> T,
{
    funding.reserve_metadata(
        binding_control_bytes::<S, M, C, T, E, I, F>().ok_or(HostMetadataFundingError::Overflow)?,
    )?;
    let token = match install(mechanisms(owner)) {
        Ok(token) => token,
        Err(cause) => return Ok(Err(cause)),
    };
    let guard = Restore {
        owner,
        mechanisms,
        token: Some(token),
        restore,
    };
    Ok(Ok(run(&mut *guard.owner)))
}
impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Pure quote for both fixed frame reservations in the existing lexical
    /// binding worker, using the caller's actual installation and run types.
    pub fn execution_mechanism_binding_control_bytes<C, T, E, I, F>(&self) -> Option<usize>
    where
        I: FnOnce(&mut M) -> Result<C, E>,
        F: FnOnce(&mut Self) -> T,
    {
        Self::execution_mechanism_binding_entry_control_bytes::<C, T, E, I, F>()?
            .checked_add(binding_control_bytes::<Self, M, C, T, E, I, F>()?)
    }
    fn execution_mechanism_binding_entry_control_bytes<C, T, E, I, F>() -> Option<usize> {
        let controls = [
            size_of::<(&mut Self, &HostMetadataFunding)>(),
            size_of::<I>(),
            size_of::<F>(),
            size_of::<fn(&mut M, C)>(),
            size_of::<Result<Result<T, E>, ExecutionMechanismBindingCause>>(),
            size_of::<Result<Result<(), std::convert::Infallible>, RuntimeInspectionBoundary>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    /// Installs one caller-authenticated backend loan around the shared driver.
    /// Successful installation returns its exact restoration token. `restore`
    /// must be infallible and allocation-free; it runs on return, error or unwind,
    /// including after an operation fences the session. A failed installation
    /// must leave the mechanism unchanged and retains its original error.
    ///
    /// This neither creates a native source nor grants submission or completion
    /// authority. Source admission and safe completion remain backend obligations.
    pub fn with_execution_mechanism_binding<C, T, E, I, F>(
        &mut self,
        funding: &HostMetadataFunding,
        install: I,
        restore: fn(&mut M, C),
        run: F,
    ) -> Result<Result<T, E>, ExecutionMechanismBindingCause>
    where
        I: FnOnce(&mut M) -> Result<C, E>,
        F: FnOnce(&mut Self) -> T,
    {
        funding.reserve_metadata(
            Self::execution_mechanism_binding_entry_control_bytes::<C, T, E, I, F>()
                .ok_or(HostMetadataFundingError::Overflow)?,
        )?;
        self.inspect_runtime_execution_fixed(|_, _, _| Ok::<_, std::convert::Infallible>(()))?;
        with_binding(
            self,
            funding,
            |session| &mut session.mechanisms,
            install,
            restore,
            run,
        )
        .map_err(ExecutionMechanismBindingCause::Funding)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    #[derive(Debug)]
    struct Account(Arc<AtomicBool>);
    impl eredu_nn::workspace::HostMetadataAccount for Account {
        fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
            if self.0.load(Ordering::SeqCst) {
                Ok(())
            } else {
                Err(HostMetadataFundingError::Unavailable)
            }
        }
    }
    #[derive(Default)]
    struct Session {
        slot: Option<u32>,
        fenced: bool,
        retired: usize,
    }
    fn restore(session: &mut Session, prior: Option<u32>) {
        session.slot = prior;
        session.retired += 1;
    }
    #[test]
    fn mechanism_binding_restores_after_fenced_error_and_unwind_and_refuses_before_installation() {
        let allowed = Arc::new(AtomicBool::new(true));
        let funding = HostMetadataFunding::new(Account(allowed.clone())).unwrap();
        let mut session = Session {
            slot: Some(11),
            ..Default::default()
        };
        let result = with_binding(
            &mut session,
            &funding,
            |s| s,
            |s| Ok::<_, &'static str>(s.slot.replace(23)),
            restore,
            |s| {
                assert_eq!(s.slot, Some(23));
                s.fenced = true;
                Err::<(), _>("actual operation")
            },
        );
        assert_eq!(result.unwrap().unwrap(), Err("actual operation"));
        assert_eq!(session.slot, Some(11));
        assert!(session.fenced);
        assert_eq!(session.retired, 1);
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_binding(
                &mut session,
                &funding,
                |s| s,
                |s| Ok::<_, &'static str>(s.slot.replace(31)),
                restore,
                |_| -> () { panic!("callback unwind") },
            );
        }));
        assert!(unwind.is_err());
        assert_eq!(session.slot, Some(11));
        assert_eq!(session.retired, 2);
        let rejected = with_binding(
            &mut session,
            &funding,
            |s| s,
            |_| Err::<Option<u32>, _>("actual source refusal"),
            restore,
            |_| -> () { panic!("unentered") },
        );
        assert_eq!(rejected.unwrap(), Err("actual source refusal"));
        assert_eq!(session.retired, 2);
        allowed.store(false, Ordering::SeqCst);
        let rejected = with_binding(
            &mut session,
            &funding,
            |s| s,
            |_| -> Result<Option<u32>, &str> { panic!("unfunded installation") },
            restore,
            |_| (),
        );
        assert!(matches!(
            rejected,
            Err(HostMetadataFundingError::Unavailable)
        ));
        assert_eq!(session.slot, Some(11));
        assert_eq!(session.retired, 2);
    }
}
