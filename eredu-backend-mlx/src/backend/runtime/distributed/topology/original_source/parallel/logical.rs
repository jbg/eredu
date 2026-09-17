//! The model occurrence retains only the exact array-free logical recipe.
use super::*;
use crate::backend::{
    nn::workspace::{LogicalCollectiveQuote, ResidentExecutionMechanisms},
    submission_recovery::prefill::TransientRootsProjection,
};
#[derive(Clone)]
pub(crate) struct RetainedLogicalCollective {
    value: Option<Rc<LogicalCollectiveQuote>>,
    funding: WorkspaceMetadataFunding,
}
impl RetainedLogicalCollective {
    pub(crate) fn value(&self) -> &LogicalCollectiveQuote {
        self.value.as_deref().expect("live logical source")
    }
    /// Retains a completed cold child recipe under its same original source.
    pub(crate) fn retain_prepared(source: &OriginalParallelSource, value: LogicalCollectiveQuote)
        -> Result<Self, Error> {
        reserve(source.funding(), &[size_of::<Self>(), size_of::<Result<Self, Error>>(),
            Layout::new::<[usize; 2]>().extend(Layout::new::<LogicalCollectiveQuote>())
                .map_err(|_| overflow())?.0.pad_to_align().size(), failure_control_bytes().ok_or_else(overflow)?])?;
        if !value.matches_source(source) {
            return Err(failure(Cause::Identity, source.declaration_source(), source.funding()));
        }
        Ok(Self { value: Some(Rc::new(value)), funding: source.funding().clone() })
    }
    /// Exact completed count-source projection into the same logical quote.
    pub(in crate::backend::runtime::distributed::topology::original_source) fn prepare_count_group(
        source: &OriginalParallelSource, order: usize,
        input: &super::super::inputs::CompletedCommunicationInput, stream: &Stream,
    ) -> Result<Self, Error> {
        Self::prepare_count(source, order, None, input, stream)
    }
    pub(in crate::backend::runtime::distributed::topology::original_source) fn prepare_count_routed_step(
        source: &OriginalParallelSource, order: usize, value: usize, step: usize,
        input: &super::super::inputs::CompletedCommunicationInput, stream: &Stream,
    ) -> Result<Self, Error> {
        Self::prepare_count(source, order, Some((value, step)), input, stream)
    }
    fn prepare_count(source: &OriginalParallelSource, order: usize, routed: Option<(usize, usize)>,
        input: &super::super::inputs::CompletedCommunicationInput, stream: &Stream) -> Result<Self, Error> {
        use crate::backend::nn::workspace::{ExistingArrayProjection, LogicalCollectiveKind,
            MlxMetalWorkspaceMechanisms};
        use eredu_nn::workspace::WorkspaceContext;
        reserve(source.funding(), &[
            size_of::<Self>(), size_of::<Result<Self, Error>>(), size_of::<Option<(usize, usize)>>(),
            size_of::<LogicalCollectiveQuote>(), size_of::<WorkspaceContext>(),
            size_of::<ExistingArrayProjection<'_>>(), size_of::<eredu_nn::workspace::WorkspaceTensor>(),
            Layout::new::<[usize; 2]>().extend(Layout::new::<LogicalCollectiveQuote>())
                .map_err(|_| overflow())?.0.pad_to_align().size(),
            Stream::device_type_control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        let actual = source.communication_source()?;
        if !input.belongs_to(&actual) || input.value().dtype() != safemlx::Dtype::Int32 {
            return Err(failure(Cause::Identity, source.declaration_source(), source.funding()));
        }
        #[cfg(not(all(target_vendor = "apple", feature = "metal", not(feature = "cuda"))))]
        return Err(Error::PrefillScopeUnavailable);
        #[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
        {
            let mechanism = MlxMetalWorkspaceMechanisms::current_host().map_err(Error::Neural)?;
            let mechanism = ResidentExecutionMechanisms::from_stream(mechanism, stream, source.funding())
                .map_err(Error::Neural)?;
            let context = WorkspaceContext::new_with_metadata_funding(mechanism, source.funding().clone())
                .map_err(|cause| Error::Neural(cause.into()))?;
            let mut projection = ExistingArrayProjection::with_source_count(&context, 1)
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
            let projected = projection.project(input.value()).map_err(Error::Neural)?;
            let value = match routed {
                None => LogicalCollectiveQuote::prepare_group(source, order, LogicalCollectiveKind::Gather,
                    projected.layout().as_view(), safemlx::Dtype::Int32, mechanism),
                Some((value, step)) => LogicalCollectiveQuote::prepare_routed_peer(source, order, value, step,
                    projected.layout().as_view(), safemlx::Dtype::Int32, mechanism),
            }.map_err(Error::Neural)?;
            Ok(Self { value: Some(Rc::new(value)), funding: source.funding().clone() })
        }
    }
    pub(super) fn prepare(
        source: &OriginalParallelSource,
        operation: eredu_nn::workspace::WorkspaceOperationView<'_>,
        mechanism: ResidentExecutionMechanisms,
    ) -> Result<Option<Self>, Error> {
        reserve(
            source.funding(),
            &[
                size_of::<Self>(),
                size_of::<Result<Option<Self>, Error>>(),
                Layout::new::<[usize; 2]>()
                    .extend(Layout::new::<LogicalCollectiveQuote>())
                    .map_err(|_| overflow())?
                    .0
                    .pad_to_align()
                    .size(),
                size_of::<Option<Rc<LogicalCollectiveQuote>>>(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        Ok(
            LogicalCollectiveQuote::prepare(source, operation, mechanism)
                .map_err(Error::Neural)?
                .map(|value| Self {
                    value: Some(Rc::new(value)),
                    funding: source.funding().clone(),
                }),
        )
    }
}
impl Drop for RetainedLogicalCollective {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl State {
    pub(super) fn execute_logical(
        &self,
        context: &Group,
        input: &Array,
        stream: &Stream,
        quote: &RetainedLogicalCollective,
    ) -> Result<Array, Error> {
        reserve(
            &self.funding,
            &[
                size_of::<RetainedLogicalCollective>(),
                size_of::<Option<TransientRootsProjection>>(),
                size_of::<std::cell::Ref<'_, Option<TransientRootsProjection>>>(),
                size_of::<Result<Array, Error>>(),
                TransientRootsProjection::control_bytes().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let fail = || failure(Cause::Identity, &self.source, &self.funding);
        let actual = self.quote_source.communication_source()?;
        if !quote.value().matches_source(&self.quote_source)
            || !actual.matches_group(quote.value().order, context)
            || !context.is_logical()
        {
            return Err(fail());
        }
        let roots = self
            .model_roots
            .try_borrow()
            .map_err(|_| fail())?
            .as_ref()
            .cloned()
            .ok_or_else(fail)?;
        let projection = context.original_control_request().ok_or_else(fail)?;
        projection.execute_logical_collective(context, input, stream, &roots, quote.clone())
    }
}
struct ModelRootLoan<'a>(&'a RefCell<Option<TransientRootsProjection>>);
impl Drop for ModelRootLoan<'_> {
    fn drop(&mut self) {
        self.0.borrow_mut().take();
    }
}
impl OriginalParallelInvocation {
    pub(crate) fn with_model_context<T, E, F>(
        &self,
        observer: &OriginalScopeObserver,
        compute: &Stream,
        control: Option<&super::super::control::OriginalParallelControlProjection>,
        roots: TransientRootsProjection,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(&mut Group, &WorkspaceMetadataFunding) -> Result<T, E>,
    {
        reserve(
            &self.funding,
            &[
                size_of::<F>(),
                size_of::<T>(),
                size_of::<E>(),
                size_of::<ModelRootLoan<'_>>(),
                size_of::<Option<TransientRootsProjection>>(),
                size_of::<Result<Result<T, E>, Error>>(),
                TransientRootsProjection::control_bytes().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let mut slot = self
            .state()
            .model_roots
            .try_borrow_mut()
            .map_err(|_| failure(Cause::Identity, &self.state().source, &self.funding))?;
        if slot.is_some() {
            return Err(failure(
                Cause::Identity,
                &self.state().source,
                &self.funding,
            ));
        }
        *slot = Some(roots);
        drop(slot);
        let _loan = ModelRootLoan(&self.state().model_roots);
        self.with_context_control(observer, compute, control, run)
    }
}
