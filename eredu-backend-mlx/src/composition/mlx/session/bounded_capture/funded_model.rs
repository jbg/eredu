//! Local funded observation with roots retained in the enclosing native Q.
use super::partition::PartitionCaptureFrame;
use crate::backend::runtime::distributed::topology::original_source::control::{
    OriginalCaptureTransport,
    speculative::{CaptureActivation, SpeculativeCaptureOwner},
};
use crate::composition::mlx::session::bounded_capture::ScheduledNativeCapture;
use crate::composition::mlx::session::intervention::PreparedModelInterventions;
use crate::{
    MlxTensor,
    backend::{array_copy::CaptureTensorNativeError, error::Error},
};
use eredu_runtime::{
    ActivationObserver,
    capture::{
        CaptureProtocolError, FundedCaptureDrainError, FundedCaptureError,
        FundedModelCaptureInvocation, OriginalSpeculativeCaptureInvocation,
    },
    working_memory::{
        OriginalEmbeddedSpeculativeRole, OriginalSpeculativeBudgetCustody, OriginalSpeculativeRole,
        WorkingMemoryError,
    },
};
use safemlx::{Array, Stream};
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    mem::{size_of, size_of_val},
    rc::Rc,
};

pub(in crate::composition::mlx) trait CaptureRole: Clone {
    fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody;
    fn same_role(&self, other: &Self) -> bool;
}
impl CaptureRole for OriginalEmbeddedSpeculativeRole {
    fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody {
        self.budget_custody()
    }
    fn same_role(&self, other: &Self) -> bool {
        self.same_role(other)
    }
}
impl CaptureRole for OriginalSpeculativeRole {
    fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody {
        self.budget_custody()
    }
    fn same_role(&self, other: &Self) -> bool {
        self.same_role(other)
    }
}
pub(in crate::composition::mlx) trait CaptureExecution<R: CaptureRole> {
    fn validate_equation_scope(&self, stream: &Stream) -> Result<(), Error>;
    fn role(&self) -> &R;
    fn observer(&self) -> &safemlx::OriginalScopeObserver;
    fn activate_partition(
        &self,
        _owner: &SpeculativeCaptureOwner,
    ) -> Result<CaptureActivation, Error> {
        Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
    }
}

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
pub(in crate::composition::mlx) struct Owner<R> {
    value: Option<Rc<RefCell<FundedModelCaptureInvocation<R>>>>,
    custody: OriginalSpeculativeBudgetCustody,
}
impl<R> Clone for Owner<R> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            custody: self.custody.clone(),
        }
    }
}
impl<R> Drop for Owner<R> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl<R: CaptureRole> Owner<R> {
    pub(in crate::composition::mlx) fn prepare(value: FundedModelCaptureInvocation<R>) -> Self {
        let custody = value.role().budget_custody();
        Self {
            value: Some(Rc::new(RefCell::new(value))),
            custody,
        }
    }
    fn value(&self) -> &RefCell<FundedModelCaptureInvocation<R>> {
        self.value.as_ref().expect("live capture owner")
    }
    /// Identity is paid by this exact model role; the enclosing source receiver
    /// keeps its own queue custody. A failure drains only settled host evidence.
    pub(in crate::composition::mlx) fn finish(
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
    pub(in crate::composition::mlx) fn receive<E>(
        &self,
        receiver: &mut dyn ActivationObserver<MlxTensor, E>,
        envelope: eredu_core::speculative::SpeculativeActivationCapture,
    ) -> Result<(), Error> {
        receiver
            .retain_original_speculative_capture(envelope)
            .map_err(|cause| Error::Neural(neural(cause, &self.custody)))
    }
}

/// The exact consumed native transport and its lexical producing loan.
pub(in crate::composition::mlx) struct Partition {
    frame: RefCell<PartitionCaptureFrame<OriginalCaptureTransport<SpeculativeCaptureOwner>>>,
    owner: SpeculativeCaptureOwner,
    prediction: u64,
}
impl Partition {
    pub(in crate::composition::mlx) fn new(
        frame: PartitionCaptureFrame<OriginalCaptureTransport<SpeculativeCaptureOwner>>,
        owner: SpeculativeCaptureOwner,
        prediction: u64,
    ) -> Self {
        Self {
            frame: RefCell::new(frame),
            owner,
            prediction,
        }
    }
}

/// Arrays retire with Q on successful completion or unresolved Recovery. The
/// shared host owner never owns or borrows this native root destination.
pub(in crate::composition::mlx) struct Capture<R> {
    roots: RefCell<Vec<Array>>,
    edits: Option<PreparedModelInterventions>,
    partition: Option<Partition>,
    owner: Owner<R>,
    role: R,
}
impl<R: CaptureRole> Capture<R> {
    pub(in crate::composition::mlx) fn control_bytes(roots: usize) -> Option<usize> {
        let shared = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<RefCell<FundedModelCaptureInvocation<R>>>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let parts = [
            size_of::<Self>(),
            size_of::<Owner<R>>(),
            size_of::<Option<Self>>(),
            size_of::<Option<Owner<R>>>(),
            size_of::<ScheduledNativeCapture<'_>>(),
            size_of::<Partition>(),
            size_of::<Option<Partition>>(),
            size_of::<
                std::cell::RefMut<
                    '_,
                    PartitionCaptureFrame<OriginalCaptureTransport<SpeculativeCaptureOwner>>,
                >,
            >(),
            size_of::<Option<CaptureActivation>>(),
            size_of::<Result<CaptureActivation, Error>>(),
            // The runtime's generic fixed controls contain GeneratedState<()>;
            // its actual model-tensor value and factory error stay lexical here.
            size_of::<Option<MlxTensor>>(),
            size_of::<eredu_nn::Error>(),
            Layout::array::<Array>(roots).ok()?.size(),
            shared,
            size_of::<Vec<Array>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<std::cell::Ref<'_, Vec<Array>>>(),
            size_of::<std::cell::RefMut<'_, FundedModelCaptureInvocation<R>>>(),
            size_of::<ObservationError>(),
            size_of::<Cause>(),
            size_of::<Failure>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<
                Result<Option<eredu_core::capture::SharedCapturedStep>, FundedCaptureDrainError>,
            >(),
            size_of::<Result<Option<eredu_core::speculative::SpeculativeActivationCapture>, Error>>(
            ),
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
    pub(in crate::composition::mlx) fn prepare(
        owner: Owner<R>,
        roots: usize,
        role: &R,
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
            partition: None,
            owner,
            role: role.clone(),
        })
    }
    pub(in crate::composition::mlx) fn run<T, A: CaptureExecution<R>>(
        &self,
        active: &A,
        stream: &Stream,
        execute: impl FnOnce(
            &mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>,
        ) -> Result<T, Error>,
    ) -> Result<T, Error> {
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
        let _activation = self
            .partition
            .as_ref()
            .map(|partition| active.activate_partition(&partition.owner))
            .transpose()?;
        let mut partition = self
            .partition
            .as_ref()
            .map(|partition| {
                partition
                    .frame
                    .try_borrow_mut()
                    .map_err(|_| Error::PrefillScopeReentrant)
            })
            .transpose()?;
        let mut program = match (partition.as_mut(), self.partition.as_ref()) {
            (Some(frame), Some(source)) => {
                frame.program_with_model(&mut *owner, source.prediction, self.edits.as_ref())?
            }
            (None, None) => None,
            _ => return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)),
        };
        let mut backend = ScheduledNativeCapture {
            partition: program.as_mut().map(|value| {
                value as &mut dyn eredu_runtime::capture::partition::ScheduledPartitionCapture
            }),
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
    pub(in crate::composition::mlx) fn visit_retained(
        &self,
        visit: &mut dyn FnMut(&Array),
    ) -> Result<(), Error> {
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

impl Capture<OriginalSpeculativeRole> {
    pub(in crate::composition::mlx) fn with_partition(
        mut self,
        partition: Option<Partition>,
    ) -> Self {
        self.partition = partition;
        self
    }
}
