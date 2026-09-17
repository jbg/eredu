//! Borrow actual detached leaves until the prepared pointwise frontier submits.
use super::*;
use crate::backend::nn::workspace::{MlxMetalWorkspaceMechanisms, PointwiseTraversalFactError};
use eredu_nn::workspace::WorkspaceOperationView;

#[derive(Debug, thiserror::Error)]
pub(crate) enum PointwisePreparationCause {
    #[error("pointwise plan does not match the prepared shape or input count")]
    ShapeMismatch,
    #[error(transparent)]
    Native(#[from] Exception),
}
#[derive(Debug)]
pub(crate) struct PointwisePreparationFailure {
    pub(crate) cause: PointwisePreparationCause,
    pub(crate) plan: PointwiseSubmissionPlan,
    // Cause before the original prepared bank/guard, including refusal before
    // any leaf loan or native descriptor allocation has been admitted.
    pub(crate) prepared: PreparedNeuralSubmission,
}

/// Cold, move-only local producer recipe. Construction precedes slot allocation;
/// the exact control requirement participates in the caller's original admission.
/// This is neither submission authority nor a proof of total neural fit.
#[derive(Debug)]
pub(crate) struct PointwiseSubmissionPlan {
    shape: NeuralSubmissionShape,
    inputs: usize,
    layout: safemlx::OperationEvalTraversalLayout,
    graph: safemlx::PointwiseGraphLayout,
    eval: safemlx::OperationEvalRecordLayout,
    gpu_prologue: safemlx::GpuEvalProloguePopulation,
    control_bytes: u64,
}
impl PointwiseSubmissionPlan {
    pub(crate) fn prepare<'op>(
        shape: NeuralSubmissionShape,
        inputs: usize,
        operations: impl IntoIterator<Item = WorkspaceOperationView<'op>>,
        mechanisms: &MlxMetalWorkspaceMechanisms,
    ) -> Result<Self, PointwiseTraversalFactError> {
        use PointwiseTraversalFactError::Overflow;
        let (layout, graph, eval, gpu_prologue, query_controls) =
            mechanisms.pointwise_traversal_layout(operations, inputs, shape.arrays())?;
        let controls = PreparedNeuralSubmission::pointwise_control_bytes()
            .and_then(|bytes| bytes.checked_add(query_controls))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(Overflow)?;
        let control_bytes = PreparedNeuralSubmission::<
            OriginalTextControlGuard,
            OriginalScopeObserver,
        >::control_bytes(shape)
        .and_then(|bytes| bytes.checked_add(controls))
        .ok_or(Overflow)?;
        Ok(Self {
            shape,
            inputs,
            layout,
            graph,
            eval,
            gpu_prologue,
            control_bytes,
        })
    }
    /// Complete local slot + pointwise/query controls. Graph and Record
    /// requests use their existing separately admitted arenas, without a
    /// second charge or a promise that arbitrary fragmentation will fit.
    pub(crate) fn control_bytes(&self) -> u64 {
        self.control_bytes
    }
    pub(crate) fn layout(&self) -> safemlx::OperationEvalTraversalLayout {
        self.layout
    }
    /// Fixed Eval host Graph recipe, separate from pointwise host constructors.
    pub(crate) fn eval_layout(&self) -> safemlx::OperationEvalRecordLayout {
        self.eval
    }
    pub(crate) fn gpu_prologue(&self) -> safemlx::GpuEvalProloguePopulation {
        self.gpu_prologue
    }
    pub(crate) fn graph_layout(&self) -> safemlx::PointwiseGraphLayout {
        self.graph
    }
}

pub(crate) struct PreparedPointwiseSubmission<'a> {
    // Unused prefix, slots and header retire before the prepared source bank.
    construction: safemlx::PreparedPointwiseGraph<'a>,
    prepared: PreparedNeuralSubmission,
    inputs: &'a [&'a MlxTensor],
    observer: &'a OriginalScopeObserver,
    stream: &'a Stream,
}
impl<'a> PreparedPointwiseSubmission<'a> {
    /// The retained local request recipe; no arena charge or total-fit claim.
    pub(crate) fn layout(&self) -> safemlx::OperationEvalTraversalLayout {
        self.prepared
            .shared
            .payload
            .borrow()
            .as_ref()
            .expect("prepared payload")
            .traversal
            .expect("prepared pointwise recipe")
    }

    /// Borrow the exact validated inputs for the existing tensor-contract
    /// caller. This wrapper implements no operators or alternate interpreter.
    pub(crate) fn inputs(&self) -> &[&'a MlxTensor] {
        self.inputs
    }
    pub(crate) fn stream(&self) -> &Stream {
        self.stream
    }
    pub(crate) fn submit(
        self,
        roots: &[&MlxTensor],
    ) -> Result<OriginalNeuralSubmissionCompletion, OriginalSubmissionFailure> {
        // The source loan survives the native call. Existing root cloning and
        // Record capture retain every reachable owner after that call returns.
        let Self {
            construction,
            prepared,
            inputs: _,
            observer,
            stream,
        } = self;
        // The physical host-construction phase ends before root cloning,
        // Synchronizer, Eval Record or any worker allocation can begin.
        drop(construction);
        prepared.submit(roots, observer.clone(), stream)
    }
}

impl PreparedNeuralSubmission {
    /// Consume the cold plan and validate shape and incoming leaf provenance
    /// before the caller constructs new graph work through its tensor contract.
    /// Reserves the local host Graph blocks and Record traversal recipe.
    /// Worker/native/outside-arena owners still prevent total-fit activation.
    pub(crate) fn prepare_pointwise<'a>(
        self,
        plan: PointwiseSubmissionPlan,
        inputs: &'a [&'a MlxTensor],
        observer: &'a OriginalScopeObserver,
        stream: &'a Stream,
    ) -> Result<PreparedPointwiseSubmission<'a>, PointwisePreparationFailure> {
        let validation = (|| -> Result<(), PointwisePreparationCause> {
            if self.shape != plan.shape || inputs.len() != plan.inputs {
                return Err(PointwisePreparationCause::ShapeMismatch);
            }
            let current = OriginalScopeObserver::require_current()?;
            if !current.same_scope(observer) {
                return Err(refusal(observer, ScopedSubmissionProgress::Unobservable).into());
            }
            OperationEvent::validate_traversal_context(observer)?;
            for input in inputs {
                OperationEvent::validate_traversal_leaf(input.as_array(), observer)?;
                if input.as_array().ndim() > plan.graph.maximum_rank()
                    && plan.graph.operations() != 0
                {
                    return Err(PointwisePreparationCause::ShapeMismatch);
                }
            }
            Ok(())
        })();
        if let Err(cause) = validation {
            return Err(PointwisePreparationFailure {
                cause,
                plan,
                prepared: self,
            });
        }
        let construction = match OperationEvent::prepare_pointwise_graph(plan.graph, observer) {
            Ok(owner) => owner,
            Err(error) => {
                return Err(PointwisePreparationFailure {
                    cause: error.into(),
                    plan,
                    prepared: self,
                })
            }
        };
        self.shared
            .payload
            .borrow_mut()
            .as_mut()
            .expect("prepared payload")
            .traversal = Some(plan.layout);
        Ok(PreparedPointwiseSubmission {
            construction,
            prepared: self,
            inputs,
            observer,
            stream,
        })
    }

    /// Additional named source/transport frames; no heap allocation or credit.
    pub(crate) fn pointwise_control_bytes() -> Option<usize> {
        [
            size_of::<PreparedPointwiseSubmission<'static>>(),
            size_of::<PointwisePreparationCause>(),
            size_of::<PointwisePreparationFailure>(),
            size_of::<PointwiseSubmissionPlan>(),
            size_of::<Result<PointwiseSubmissionPlan, PointwiseTraversalFactError>>(),
            size_of::<Result<PreparedPointwiseSubmission<'static>, PointwisePreparationFailure>>(),
            size_of::<Option<safemlx::OperationEvalTraversalLayout>>(),
            size_of::<Result<(), PointwisePreparationCause>>(),
            size_of::<[usize; 3]>(),
            size_of::<&MlxTensor>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?
        .checked_add(MlxMetalWorkspaceMechanisms::pointwise_control_bytes()?)
    }
}
