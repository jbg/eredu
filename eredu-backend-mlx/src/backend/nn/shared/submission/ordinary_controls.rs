//! The same ordinary submission and consumer workers, with prepaid custody.
use super::*;
use crate::backend::{
    nn::workspace::{OrdinaryCallControls, OrdinaryNativeControls},
    submission_recovery::PreparedRecovery,
};
use eredu_core::HostPreparationAuthority;
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

fn shared<T>() -> Option<usize> {
    let (layout, _) = Layout::new::<[usize; 2]>()
        .extend(Layout::new::<T>())
        .ok()?;
    Some(layout.pad_to_align().size())
}
fn metadata(parts: &[usize]) -> Option<u64> {
    u64::try_from(
        parts
            .iter()
            .copied()
            .try_fold(size_of_val(parts), usize::checked_add)?,
    )
    .ok()
}
fn recovery<T: Retention>() -> Option<u64> {
    PreparedRecovery::<T, HostPreparationAuthority>::control_bytes()?.checked_add(
        u64::try_from(safemlx::error::Exception::retained_source_control_bytes::<
            safemlx::SubmissionScopeOwnerCause,
        >()?)
        .ok()?,
    )
}
impl MlxNeuralBackend {
    /// The ordinary async call used by bounded `complete`: its actual C vector
    /// and completion shells, argument-vector growth, and Rust transports. The
    /// caller supplies the real output + state + context root count. Native Eval
    /// must be priced with the enclosing reachable DAG, not these roots alone.
    /// Existing lease/controller/queue ownership remains a separate producer.
    pub(crate) fn ordinary_completion_call_controls(roots: usize) -> Option<OrdinaryCallControls> {
        let mut observed = OrdinaryNativeControls::default();
        observed.include(OperationEvent::ordinary_array_vector_control_layout(roots)?)?;
        Some(OrdinaryCallControls {
            metadata_bytes: metadata(&[
                OperationEvent::ordinary_submission_wrapper_control_bytes()?,
                size_of::<std::slice::Iter<'static, Array>>(),
                size_of::<(&Stream, usize)>(),
                size_of::<Result<OperationEvent, safemlx::error::Exception>>(),
            ])?,
            observed,
        })
    }

    /// Actual fixed-root SubmissionBackend::submit worker, including each C
    /// Array clone, the exactly sized retained Vec, both Rc allocations, the
    /// empty HostResourceNode Box, and the prepared recovery/Scope owner. Group
    /// and final submissions currently pass the exact single-element array.
    /// Extra retain_until_complete payloads are not part of this worker source.
    pub(crate) fn ordinary_submission_call_controls(roots: usize) -> Option<OrdinaryCallControls> {
        let mut controls = Self::ordinary_completion_call_controls(roots)?;
        let fields = [
            Layout::array::<Array>(roots).ok()?.size(),
            Array::ordinary_clone_control_bytes()?.checked_mul(roots)?,
            shared::<SubmissionResources>()?,
            shared::<OperationEvent>()?,
            size_of::<SubmissionResources>(),
            size_of::<OperationEvent>(),
            size_of::<HostResourceNode>(),
            size_of::<HostResourceNode>(),
            size_of::<HostResources>(),
            size_of::<Vec<Array>>(),
            size_of::<Rc<SubmissionResources>>(),
            size_of::<Rc<OperationEvent>>(),
            size_of::<MlxSubmissionCompletion>(),
            size_of::<Option<OrdinaryExecutionOwner>>(),
            size_of::<OrdinaryExecutionOwner>(),
            size_of::<Result<MlxSubmissionCompletion, safemlx::error::Exception>>(),
            size_of::<(&Stream, &[&crate::MlxTensor])>(),
            size_of::<std::slice::Iter<'static, crate::MlxTensor>>(),
        ];
        controls.metadata_bytes = controls
            .metadata_bytes
            .checked_add(metadata(&fields)?)?
            .checked_add(recovery::<Rc<SubmissionResources>>()?)?;
        Some(controls)
    }

    /// Caller envelope of the shared driver's ordinary local-dependency worker.
    /// The driver supplies an exact-size copied slice, boundary-value slice map,
    /// or singleton iterator. Their fixed iterator frames are conservatively
    /// combined here; this is not a quote for an arbitrary external iterator.
    /// The outer Array destination and its clones coexist with the common
    /// communication worker's owned output and retention destinations. Native
    /// Eval, neutral communication authority and driver controls stay separate.
    pub(crate) fn ordinary_local_dependencies_call_controls(
        roots: usize,
    ) -> Option<OrdinaryCallControls> {
        if roots == 0 {
            return None;
        }
        use crate::backend::runtime::distributed::completion::prepared::CompletionResourceLayout;
        type Boundary = eredu_runtime::ArchitectureBoundaryValue<MlxTensor>;
        type Copied<'a> = std::iter::Copied<std::slice::Iter<'a, &'a MlxTensor>>;
        type BoundaryMap<'a> =
            std::iter::Map<std::slice::Iter<'a, Boundary>, fn(&'a Boundary) -> &'a MlxTensor>;
        type ClonedArrays<I> = std::iter::Map<I, fn(&MlxTensor) -> Array>;
        let mut controls = MlxCommunicationCompletion::ordinary_submit_call_controls(
            roots,
            CompletionResourceLayout {
                arrays: roots,
                counts: &[],
                groups: 0,
                routes: 0,
                streams: 1,
            },
        )?;
        let parts = [
            // These exact-size built-in iterators collect with no growth after
            // initialization. Include Vec's small non-ZST minimum capacity.
            Layout::array::<Array>(roots.max(4)).ok()?.size(),
            size_of::<Vec<Array>>(),
            Array::ordinary_clone_control_bytes()?.checked_mul(roots)?,
            size_of::<Copied<'_>>(),
            size_of::<BoundaryMap<'_>>(),
            size_of::<std::iter::Once<&MlxTensor>>(),
            size_of::<ClonedArrays<Copied<'_>>>(),
            size_of::<ClonedArrays<BoundaryMap<'_>>>(),
            size_of::<ClonedArrays<std::iter::Once<&MlxTensor>>>(),
            size_of::<(&Stream, usize)>(),
            size_of::<MlxCommunicationCompletion>(),
            size_of::<Submission<(), MlxNeuralCommunicationCompletion>>(),
            size_of::<
                Result<
                    Submission<(), MlxNeuralCommunicationCompletion>,
                    crate::backend::error::Error,
                >,
            >(),
        ];
        controls.metadata_bytes = controls.metadata_bytes.checked_add(metadata(&parts)?)?;
        Some(controls)
    }

    /// The actual retained completion's consumer worker: Stream clone,
    /// ConsumerResources, prepared recovery/Scope and fixed event-wait wrapper.
    /// The native consumer WaitRecord/task source is separate and depends on
    /// the selected device; synchronous polling creates no such consumer.
    pub(crate) fn ordinary_consumer_call_controls() -> Option<OrdinaryCallControls> {
        let fields = [
            Stream::ordinary_clone_control_bytes()?,
            OperationEvent::ordinary_wait_wrapper_control_bytes()?,
            size_of::<ConsumerResources>(),
            size_of::<ConsumerUnwind<'static>>(),
            size_of::<(&MlxSubmissionCompletion, &Stream)>(),
            size_of::<Rc<SubmissionResources>>(),
            size_of::<Status>(),
            size_of::<Result<(), safemlx::error::Exception>>(),
        ];
        Some(OrdinaryCallControls {
            metadata_bytes: metadata(&fields)?.checked_add(recovery::<ConsumerResources>()?)?,
            observed: OrdinaryNativeControls::default(),
        })
    }
}
