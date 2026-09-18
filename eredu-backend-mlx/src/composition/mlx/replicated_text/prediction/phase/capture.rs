//! Local funded observation with roots retained in the enclosing native Q.
use crate::{
    MlxTensor,
    backend::{array_copy::CaptureTensorNativeError, error::Error},
    composition::mlx::speculative::embedded_native::ActiveEmbeddedNativeInvocation,
};
use eredu_runtime::{
    ActivationObserver,
    capture::{
        CaptureProtocolError, FundedCaptureDrainError, FundedCaptureError,
        FundedEmbeddedCaptureInvocation, OriginalSpeculativeCaptureInvocation,
    },
    working_memory::{
        OriginalEmbeddedSpeculativeRole, OriginalSpeculativeBudgetCustody, WorkingMemoryError,
    },
};
use safemlx::{Array, Stream};
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    mem::{size_of, size_of_val},
    rc::Rc,
};
mod backend;
use crate::composition::mlx::session::intervention::PreparedModelInterventions;

type ObservationError = FundedCaptureError<CaptureTensorNativeError>;
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Observation(#[from] ObservationError),
    #[error(transparent)]
    Drain(#[from] FundedCaptureDrainError),
    #[error(transparent)]
    Protocol(#[from] CaptureProtocolError),
    #[error(transparent)]
    Native(#[from] CaptureTensorNativeError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    custody: OriginalSpeculativeBudgetCustody,
}
fn neural(cause: impl Into<Cause>, custody: &OriginalSpeculativeBudgetCustody) -> eredu_nn::Error {
    eredu_nn::Error::backend_retained_source(Failure {
        cause: cause.into(),
        custody: custody.clone(),
    })
}

/// The closed host collector survives Q retirement, solely for sealed delivery.
/// Each concrete alias retains its actual paying account after Rc deallocation.
pub(super) struct Owner {
    value: Option<Rc<RefCell<FundedEmbeddedCaptureInvocation>>>,
    custody: OriginalSpeculativeBudgetCustody,
}
impl Clone for Owner {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            custody: self.custody.clone(),
        }
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl Owner {
    pub(super) fn prepare(value: FundedEmbeddedCaptureInvocation) -> Self {
        let custody = value.role().budget_custody();
        Self {
            value: Some(Rc::new(RefCell::new(value))),
            custody,
        }
    }
    fn value(&self) -> &RefCell<FundedEmbeddedCaptureInvocation> {
        self.value.as_ref().expect("live capture owner")
    }
    /// Identity is paid by this exact model role; the enclosing source receiver
    /// keeps its own queue custody. A failure drains only settled host evidence.
    pub(super) fn finish(
        &self,
        invocation: OriginalSpeculativeCaptureInvocation<'_>,
        identity: String,
        success: bool,
    ) -> Result<Option<eredu_core::speculative::SpeculativeActivationCapture>, Error> {
        let mut value = self.value().try_borrow_mut().map_err(|_| {
            Error::Neural(neural(CaptureProtocolError::CumulativeInUse, &self.custody))
        })?;
        let frame = if success {
            value.take_shared_step()
        } else {
            value.take_failed_evidence()
        };
        let frame = match frame {
            Ok(Some(frame)) => frame,
            Ok(None) if !success => return Ok(None),
            Err(_) if !success => return Ok(None),
            Ok(None) => {
                return Err(Error::Neural(neural(
                    CaptureProtocolError::Transaction,
                    &self.custody,
                )));
            }
            Err(cause) => return Err(Error::Neural(neural(cause, &self.custody))),
        };
        drop(value);
        let envelope = invocation
            .envelope(identity, frame, success)
            .map_err(|cause| Error::Neural(neural(cause, &self.custody)))?;
        Ok(Some(envelope))
    }
    pub(super) fn receive(
        &self,
        receiver: &mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>,
        envelope: eredu_core::speculative::SpeculativeActivationCapture,
    ) -> Result<(), Error> {
        receiver
            .retain_original_speculative_capture(envelope)
            .map_err(|cause| Error::Neural(neural(cause, &self.custody)))
    }
}

/// Arrays retire with Q on successful completion or unresolved Recovery. The
/// shared host owner never owns or borrows this native root destination.
pub(super) struct Capture {
    roots: RefCell<Vec<Array>>,
    edits: Option<PreparedModelInterventions>,
    owner: Owner,
    role: OriginalEmbeddedSpeculativeRole,
}
impl Capture {
    pub(super) fn control_bytes(roots: usize) -> Option<usize> {
        let shared = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<RefCell<FundedEmbeddedCaptureInvocation>>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let parts = [
            size_of::<Self>(),
            size_of::<Owner>(),
            size_of::<Option<Self>>(),
            size_of::<Option<Owner>>(),
            size_of::<backend::Backend<'_>>(),
            // The runtime's generic fixed controls contain GeneratedState<()>;
            // its actual model-tensor value and factory error stay lexical here.
            size_of::<Option<MlxTensor>>(),
            size_of::<eredu_nn::Error>(),
            Layout::array::<Array>(roots).ok()?.size(),
            shared,
            size_of::<Vec<Array>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<std::cell::Ref<'_, Vec<Array>>>(),
            size_of::<std::cell::RefMut<'_, FundedEmbeddedCaptureInvocation>>(),
            size_of::<ObservationError>(),
            size_of::<Cause>(),
            size_of::<Failure>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<Option<eredu_core::capture::SharedCapturedStep>, FundedCaptureDrainError>>(),
            size_of::<Result<Option<eredu_core::speculative::SpeculativeActivationCapture>, Error>>(),
            size_of::<OriginalSpeculativeCaptureInvocation<'_>>(),
            size_of::<Option<String>>(),
            size_of::<std::cell::BorrowError>(),
            size_of::<std::cell::BorrowMutError>(),
            // One callback failure and one terminal host/protocol failure are
            // distinct possible producers; both keep the same exact role.
            eredu_nn::Error::retained_source_construction_bytes::<Failure>()?,
            eredu_nn::Error::retained_source_construction_bytes::<Failure>()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(super) fn prepare(
        owner: Owner,
        roots: usize,
        role: &OriginalEmbeddedSpeculativeRole,
        edits: Option<PreparedModelInterventions>,
    ) -> Result<Self, Error> {
        let mut values = Vec::new();
        values.try_reserve_exact(roots).map_err(|cause| {
            Error::Neural(neural(
                CaptureTensorNativeError::from(cause),
                &owner.custody,
            ))
        })?;
        if values.capacity() != roots {
            return Err(Error::Neural(neural(
                WorkingMemoryError::UnknownBound,
                &owner.custody,
            )));
        }
        Ok(Self {
            roots: RefCell::new(values),
            edits,
            owner,
            role: role.clone(),
        })
    }
    pub(super) fn run<R>(
        &self,
        active: &ActiveEmbeddedNativeInvocation<'_>,
        stream: &Stream,
        execute: impl FnOnce(
            &mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        active.validate_equation_scope(stream)?;
        if !self.role.same_role(active.role()) {
            return Err(Error::Neural(neural(
                WorkingMemoryError::IdentityMismatch,
                &self.owner.custody,
            )));
        }
        let mut owner = self.owner.value().try_borrow_mut().map_err(|_| {
            Error::Neural(neural(
                CaptureProtocolError::CumulativeInUse,
                &self.owner.custody,
            ))
        })?;
        let mut backend = backend::Backend {
            stream,
            roots: &self.roots,
            edits: self.edits.as_ref(),
            custody: &self.owner.custody,
            observer: active.observer(),
        };
        owner
            .with_observer(
                &mut backend,
                &|cause| neural(cause, &self.owner.custody),
                execute,
            )
            .map_err(|cause| Error::Neural(neural(cause, &self.owner.custody)))?
    }
    pub(super) fn visit_retained(&self, visit: &mut dyn FnMut(&Array)) -> Result<(), Error> {
        let roots = self.roots.try_borrow().map_err(|_| {
            Error::Neural(neural(
                CaptureProtocolError::CumulativeInUse,
                &self.owner.custody,
            ))
        })?;
        for value in roots.iter() {
            visit(value);
        }
        Ok(())
    }
}
