//! Typed executable contract for architecture-owned retained prefill sessions.

use crate::{
    execution_control::SessionState,
    prefill::{
        drain_retained_prefill, ErasedPrefillCursor, PrefillCursorError, PrefillStep,
        RetainedPrefillAdvance, RetainedPrefillCursor,
    },
    PartitionOutputAuthority, PartitionOutputPublication, PreparedInputCacheIdentity,
    ResidencyReport,
};
use eredu_core::input::PreparedInputInspector;
use eredu_core::PreparedModelInput;
use eredu_core::{
    capture::CaptureRequestShape, BackendFailure, CapabilityError, SessionCapabilities,
};
use eredu_nn::Tensor;

use crate::CapabilityEstimate;
use eredu_core::InputModality;

/// Rejection before an already bound retained session can be installed.
#[derive(Debug, thiserror::Error)]
pub enum RetainedTextBindingError {
    /// Publication requires the matching native communication resource.
    #[error("retained session publication and native communication disagree")]
    Publication,
    /// The bound execution cannot realize its selected session capabilities.
    #[error(transparent)]
    Admission(#[from] eredu_core::SessionAdmissionError),
}

/// Architecture-owned admission and accounting plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedInputPartPlan {
    /// Text decoder positions supplied as token IDs or accepted embeddings.
    Text {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width embeddings supplied directly by the caller.
    Projected {
        /// Semantic modality retained by decoder input assembly.
        modality: InputModality,
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native media requiring tower execution.
    Media {
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

impl PreparedInputPartPlan {
    /// Exact decoder positions after architecture-owned media preparation.
    pub const fn decoder_positions(&self) -> u64 {
        match self {
            Self::Text { positions } | Self::Projected { positions, .. } => *positions,
            Self::Media { shape } => shape.decoder_positions,
        }
    }
}

/// Architecture-derived geometry and conservative execution workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaShapePlan {
    /// Exact packed capture axes for towers with declared internal hooks.
    pub capture_media: Option<eredu_core::capture::CaptureMediaShape>,
    /// Decoder positions occupied by the projected media.
    pub decoder_positions: u64,
    /// Conservative count of temporary execution scalars.
    pub execution_workspace_scalars: u64,
}

/// Resolves independently packed tower axes before observation transaction admission.
/// Projected caller inputs contribute no tower invocation.
pub fn capture_media_shape(
    plans: &[PreparedInputPartPlan],
) -> Result<Option<eredu_core::capture::CaptureMediaShape>, CapabilityError> {
    let mut result: Option<eredu_core::capture::CaptureMediaShape> = None;
    for plan in plans {
        if let PreparedInputPartPlan::Media { shape } = plan {
            if let Some(media) = shape.capture_media {
                let current = result.get_or_insert(eredu_core::capture::CaptureMediaShape {
                    patches: 0,
                    projected: 0,
                });
                current.patches = current.patches.checked_add(media.patches).ok_or(
                    CapabilityError::ArithmeticOverflow {
                        operation: "capture media patches",
                    },
                )?;
                current.projected = current.projected.checked_add(media.projected).ok_or(
                    CapabilityError::ArithmeticOverflow {
                        operation: "capture projected media",
                    },
                )?;
            }
        }
    }
    Ok(result)
}

/// Final executable adaptation after portable composite construction.
pub trait RetainedCompositeVisitor<B, S, C: 'static, F>: Sized
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + 'static,
    S: crate::LayerRuntimeState<B>,
    <B::Tensor as Tensor>::Context: Sized + 'static,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + crate::RuntimeStateComponents<B>,
{
    /// Shared component realization for local and partitioned execution.
    type Components: crate::ComponentRankMechanisms<B, S, C> + crate::LocalComponentMechanisms<B, S>;

    /// Creates the binder for the retained native streams.
    fn components(&self) -> Self::Components;

    /// Representations used by the installed retained executable.
    type RetainedTypes: super::retained_executable::RetainedTextMechanisms<
        Tensor = B::Tensor,
        State = S,
        Communication = C,
        Error = F,
        Context = <B::Tensor as Tensor>::Context,
    >;

    /// Retains the exact execution context for the installed session.
    fn retained_context(&self) -> <Self::RetainedTypes as super::TextExecutableTypes>::Context;
}

impl<B, S, C, F, N, Q> RetainedCompositeVisitor<B, S, C, F> for N
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + 'static,
    B::Tensor: Tensor<Context = Q>,
    S: crate::LayerRuntimeState<B>,
    C: 'static,
    Q: Clone + 'static,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + crate::RuntimeStateComponents<B>,
    N: eredu_core::checkpoint_mechanisms::TextComponentResources<
        Values: eredu_core::backend::NativeTextValues<Context = Q>,
        Components: crate::ComponentRankMechanisms<B, S, C> + crate::LocalComponentMechanisms<B, S>,
    >,
    super::PortableTextTypes<N::Values>: super::retained_executable::RetainedTextMechanisms<
        Tensor = B::Tensor,
        State = S,
        Communication = C,
        Error = F,
        Context = Q,
    >,
{
    type Components = N::Components;
    type RetainedTypes = super::PortableTextTypes<N::Values>;

    fn components(&self) -> Self::Components {
        self.text_components()
    }

    fn retained_context(&self) -> Q {
        self.component_execution().clone()
    }
}

/// Backend-neutral invocation and state contract for a retained prefill session.
pub trait PreparedRetainedTextSession<T: Tensor, S> {
    /// Typed prefill continuation held by the portable erased cursor.
    type Cursor: RetainedPrefillCursor<Tensor = T> + 'static;

    /// Retained capability and memory facts.
    fn capabilities(&self) -> &CapabilityEstimate;
    /// Realized optional session capabilities.
    fn session_capabilities(&self) -> SessionCapabilities;
    /// Installed decoder state.
    fn state(&self) -> &SessionState<S>;
    /// Installed decoder state at a mutable session boundary.
    fn state_mut(&mut self) -> &mut SessionState<S>;
    /// Selected score publication and rank-local authority.
    fn publication(&self) -> Option<(PartitionOutputPublication, PartitionOutputAuthority)> {
        None
    }
    /// Weight residency of each bound component.
    fn component_residency(&self) -> Result<Vec<ResidencyReport>, BackendFailure> {
        Ok(Vec::new())
    }
    /// Overall residency when the components share a single policy.
    fn residency(&self) -> Result<Option<ResidencyReport>, BackendFailure> {
        Ok(None)
    }
    /// Dense-stream telemetry for a shared policy, when selected.
    fn dense_stream(&self) -> Result<Option<crate::DenseDiskStreamReport>, BackendFailure> {
        Ok(None)
    }
    /// Releases optional parameter conversion retention at an idle boundary.
    fn trim_parameter_conversions(
        &self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        Err(eredu_core::residency::ParameterConversionTrimError::Unsupported)
    }
    /// Selected decoder parameter residency.
    fn decoder_residency(&self) -> crate::LayerWeightResidency;
    /// Reports the admitted input parts without starting execution.
    fn input_plans(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
    ) -> Result<Vec<PreparedInputPartPlan>, CapabilityError>;
    /// Reports the exact expanded decoder token IDs.
    fn input_tokens(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
        identity: Option<&PreparedInputCacheIdentity>,
    ) -> Result<Vec<u32>, BackendFailure>;
    /// Admits prefill on all ranks, including rank-local conversion failures.
    fn start_prefill(
        &mut self,
        input: Result<PreparedModelInput<T>, BackendFailure>,
        inspector: &impl PreparedInputInspector<T>,
        identity: Option<&PreparedInputCacheIdentity>,
        max_tokens: usize,
        capture: Option<CaptureRequestShape>,
        context: &T::Context,
    ) -> Result<Self::Cursor, BackendFailure>;
    /// Advances one typed prefill step.
    fn advance_prefill(
        &mut self,
        cursor: &mut Self::Cursor,
        context: &T::Context,
        observer: &mut dyn crate::ActivationObserver<T, eredu_nn::Error>,
    ) -> Result<Option<RetainedPrefillAdvance<T>>, BackendFailure>;
    /// Advances erased ingress without committing decoder positions.
    fn advance_retained_ingress(
        &mut self,
        cursor: &mut ErasedPrefillCursor,
        context: &T::Context,
        observer: &mut dyn crate::ActivationObserver<T, eredu_nn::Error>,
    ) -> Result<(), BackendFailure> {
        let (step, cursor) = erased_retained_step::<Self::Cursor>(cursor)?;
        if step != PrefillStep::PrepareInput {
            return Err(BackendFailure::from_error(RetainedStepError::Ingress));
        }
        match self.advance_prefill(cursor, context, observer)? {
            Some(RetainedPrefillAdvance::Ingress) => Ok(()),
            _ => Err(BackendFailure::from_error(RetainedStepError::Result)),
        }
    }

    /// Advances erased decoding after verifying the next committed range.
    fn advance_retained_decoder(
        &mut self,
        cursor: &mut ErasedPrefillCursor,
        context: &T::Context,
        observer: &mut dyn crate::ActivationObserver<T, eredu_nn::Error>,
    ) -> Result<T, BackendFailure> {
        let (step, cursor) = erased_retained_step::<Self::Cursor>(cursor)?;
        if !matches!(step, PrefillStep::Decoder(_)) {
            return Err(BackendFailure::from_error(RetainedStepError::Decoder));
        }
        match self.advance_prefill(cursor, context, observer)? {
            Some(RetainedPrefillAdvance::Decoder(scores)) => Ok(scores),
            _ => Err(BackendFailure::from_error(RetainedStepError::Result)),
        }
    }

    /// Runs a complete retained prefill using the same typed steps as controlled execution.
    fn complete_prefill(
        &mut self,
        cursor: &mut Self::Cursor,
        context: &T::Context,
        observer: &mut dyn crate::ActivationObserver<T, eredu_nn::Error>,
    ) -> Result<T, BackendFailure> {
        drain_retained_prefill(|| {
            let step = cursor.next_step().map_err(BackendFailure::from_error)?;
            if step == PrefillStep::Complete {
                return Ok(None);
            }
            match (step, self.advance_prefill(cursor, context, observer)?) {
                (PrefillStep::PrepareInput, progress @ Some(RetainedPrefillAdvance::Ingress))
                | (PrefillStep::Decoder(_), progress @ Some(RetainedPrefillAdvance::Decoder(_))) => {
                    Ok(progress)
                }
                _ => Err(BackendFailure::from_error(RetainedStepError::Result)),
            }
        })?
        .ok_or_else(|| BackendFailure::from_error(std::io::Error::other("empty retained prefill")))
    }
    /// Executes one cached decoder input, agreeing local input failures.
    fn decode(
        &mut self,
        tokens: Result<T, BackendFailure>,
        context: &T::Context,
        observer: &mut dyn crate::ActivationObserver<T, eredu_nn::Error>,
    ) -> Result<T, BackendFailure>;
    /// Allocates the rank-local state for a local reset.
    fn fresh_state(&self) -> Result<S, BackendFailure>;
    /// Resets the installed state at a coordinated rank boundary.
    fn reset(&mut self, context: &T::Context) -> Result<(), BackendFailure>;
}

#[derive(Debug, thiserror::Error)]
enum RetainedStepError {
    #[error("retained prefill is not at an ingress step")]
    Ingress,
    #[error("retained prefill is not at a decoder step")]
    Decoder,
    #[error("retained prefill produced an invalid step result")]
    Result,
}

fn erased_retained_step<C: RetainedPrefillCursor + 'static>(
    cursor: &mut ErasedPrefillCursor,
) -> Result<(PrefillStep, &mut C), BackendFailure> {
    let cursor = cursor
        .downcast_mut::<C>()
        .ok_or_else(|| BackendFailure::from_error(PrefillCursorError::Foreign))?;
    let step = cursor.next_step().map_err(BackendFailure::from_error)?;
    Ok((step, cursor))
}
