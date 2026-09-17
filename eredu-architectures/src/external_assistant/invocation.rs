//! One architecture operation, used by ordinary execution and exact cold tracing.
use super::ExternalAssistantArchitecture;
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, Error, GroupedNeuralBackend, Tensor,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceTensor},
};

/// A typed family equation and its finite current-input projection. Implementors
/// describe existing equations; the backend supplies placement, source binding,
/// native admission and completion. A trace is never an execution grant.
pub trait ExternalAssistantOperation<A: ExternalAssistantArchitecture>: 'static {
    /// Actual borrowed arguments of the existing architecture worker.
    type Arguments<'a, T: Tensor + 'a>;
    /// Values returned by that worker.
    type Output<T: Tensor>;
    /// Owned metadata projection, retaining only this invocation's actual inputs.
    type Projected;

    /// Actual operation in the retained external schedule; equal geometry does
    /// not substitute another family equation or occurrence class.
    fn invocation_kind() -> eredu_runtime::speculative::external_occurrence::ExternalInvocationKind;

    /// Constructs the same unloaded module from the retained selected config.
    /// Implementations fund each metadata producer through `context` and must
    /// not clone configuration maps merely to borrow their declarations.
    fn workspace_module(
        config: &A::Config,
        context: &WorkspaceContext,
    ) -> Result<A::Module<WorkspaceBackend>, Error>;

    /// Executes the existing equation with the selected neural mechanism.
    fn execute<B, C>(
        module: &mut A::Module<B>,
        arguments: Self::Arguments<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output<B::Tensor>, Error>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend + Clone,
        C: AttentionCache<B::Tensor>;

    /// Executes with the quote's metadata destination for ordinary host vectors
    /// built inside the same model invocation. This grants no tensor authority.
    fn execute_with_metadata<B,C>(module:&mut A::Module<B>,arguments:Self::Arguments<'_,B::Tensor>,
        context:&<B::Tensor as Tensor>::Context,_metadata:&WorkspaceContext)->Result<Self::Output<B::Tensor>,Error>
    where B:GroupedNeuralBackend+DistributedNeuralBackend+Clone,C:AttentionCache<B::Tensor>{
        Self::execute::<B,C>(module,arguments,context)
    }

    /// Reborrows actual arguments so the original producer can publish their
    /// completion witness after the same mutable equation returns.
    fn reborrow<'a, 'state, T: Tensor>(arguments: &'a mut Self::Arguments<'state, T>)
        -> Self::Arguments<'a, T> where 'state: 'a;
    /// Existing exact per-state source witnesses, without a mutable latest slot.
    fn visit_evidence<T: Tensor>(arguments: &Self::Arguments<'_, T>,
        visit: &mut dyn FnMut(&crate::speculative_execution::PreparedEmbeddedEvidence));
    /// Publish only after native completion and retirement. The same witness
    /// may also be retained by the returned output; it grants no new operation.
    fn retain_evidence<T: Tensor>(arguments: &mut Self::Arguments<'_, T>,
        evidence: crate::speculative_execution::PreparedEmbeddedEvidence);

    /// Projects exact current arguments through the caller's alias-preserving
    /// native-source adapter. Geometry and iteration order remain family-owned.
    fn project<'a, T: Tensor>(
        arguments: &'a Self::Arguments<'_, T>,
        project: impl FnMut(&'a T) -> Result<WorkspaceTensor, Error>,
        context: &WorkspaceContext,
    ) -> Result<Self::Projected, Error>;

    /// Exact single-equation geometry from the current projected source.
    fn geometry(
        projected: &Self::Projected,
        context: &WorkspaceContext,
    ) -> Result<eredu_core::InferenceGeometry, Error>;

    /// Runs the same equation against its projected arguments.
    fn trace(
        module: &mut A::Module<WorkspaceBackend>,
        projected: &mut Self::Projected,
        context: &WorkspaceContext,
    ) -> Result<Self::Output<WorkspaceTensor>, Error>;

    /// Visits exact projected roots retained across the equation boundary.
    fn visit_projected(projected: &Self::Projected, visit: &mut dyn FnMut(&WorkspaceTensor));
    /// Visits actual current native inputs/state, preserving aliases and repeats.
    fn visit_arguments<T: Tensor>(arguments: &Self::Arguments<'_, T>, visit: &mut dyn FnMut(&T));
    /// Visits actual returned values for completion and source publication.
    fn visit_output<T: Tensor>(output: &Self::Output<T>, visit: &mut dyn FnMut(&T));
}
