//! Shared invocation and state lifecycle for installed causal executables.

use std::rc::Rc;

use eredu_core::{
    backend::{NativeTextValues, TextExecutableTypes},
    execution_control::SnapshotEstimate,
    BackendFailure, BackendFailureKind,
};

use crate::{
    executable_binding::{ErasedTextExecutable, PortableTextTypes},
    text_session_driver::TextPromptView,
    ActivationObserver,
};

pub(super) type PredictionState<V> = <PortableTextTypes<V> as TextExecutableTypes>::PredictionState;

/// Independent mutable causal state on its installed context.
pub struct InstalledCausalState<V: NativeTextValues> {
    pub(super) cache: PredictionState<V>,
    pub(super) queue: V::Context,
}

impl<V: NativeTextValues> InstalledCausalState<V> {
    pub(super) fn allocate(
        model: &mut dyn ErasedTextExecutable<PortableTextTypes<V>>,
        queue: &V::Context,
    ) -> Result<Self, V::Error> {
        Ok(Self {
            cache: model.prepare_autoregressive_cache()?,
            queue: queue.clone(),
        })
    }

    pub(super) fn checkpoint(&self) -> Result<Rc<Self>, BackendFailure> {
        self.copy_boundary().map(Rc::new)
    }

    pub(super) fn copy_boundary(&self) -> Result<Self, BackendFailure> {
        Ok(Self {
            cache: self.cache.checkpoint_view()?,
            queue: self.queue.clone(),
        })
    }

    pub(super) fn estimate(&self) -> Option<SnapshotEstimate> {
        self.cache.control_estimate()
    }

    pub(super) fn retained_bytes(&self) -> Option<u64> {
        self.estimate()?
            .retained_bytes
            .checked_add(self.cache.control_growth(0)?)
    }

    pub(super) fn retain_verified_prefix(
        &mut self,
        saved: &Self,
        accepted: usize,
        queue: &V::Context,
    ) -> Result<bool, BackendFailure> {
        let length = saved
            .cache
            .generation()?
            .checked_add(accepted as u64)
            .and_then(|length| i32::try_from(length).ok())
            .ok_or_else(|| {
                BackendFailure::new(
                    BackendFailureKind::InvalidInput,
                    std::io::Error::other("accepted prefix exceeds native geometry"),
                )
            })?;
        self.cache.retain_causal_prefix(length, queue)
    }

    pub(super) fn prefill(
        &mut self,
        model: &mut dyn ErasedTextExecutable<PortableTextTypes<V>>,
        input: TextPromptView<'_, V::InputTensor>,
        observer: &mut dyn ActivationObserver<V::InvocationTensor, V::Error>,
    ) -> Result<(V::Scores, usize), V::Error> {
        model.prepared_context_forward(input, &mut self.cache, &self.queue, observer)
    }

    pub(super) fn append(
        &mut self,
        model: &mut dyn ErasedTextExecutable<PortableTextTypes<V>>,
        tokens: &V::InvocationTensor,
        observer: &mut dyn ActivationObserver<V::InvocationTensor, V::Error>,
    ) -> Result<V::Scores, V::Error> {
        model.attention_forward_with_observer(
            tokens,
            None,
            eredu_core::decoding::AttentionInvocation::CAUSAL,
            Some(&mut self.cache),
            false,
            &self.queue,
            observer,
        )
    }
}
