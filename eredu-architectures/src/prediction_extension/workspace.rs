//! Selected prediction modules and actual current-state metadata.
use super::*;
use eredu_nn::{
    CompressedAttentionBlock, CompressedAttentionCache, CompressedAttentionScan,
    CompressedAttentionState, CompressedAttentionView, Error,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceMetadataError, WorkspaceTensor},
};
use eredu_runtime::{
    DeviceState, RuntimeLayerState, RuntimeState,
    working_memory::{
        WorkspaceCompressedCache, WorkspacePoolingLayerState, WorkspaceResidentLayerState,
    },
};
use std::{
    marker::PhantomData,
    mem::{size_of, size_of_val},
};

mod invocations;
pub use invocations::{
    WorkspacePredictionInvocation, WorkspacePredictionInvocations, WorkspacePredictionModuleCall,
};

type ModelState = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;

/// A source-owned native adapter binds actual parameter representations through
/// the shared parameter worker. Parameter/module membership remains with the
/// architecture; the adapter may not infer a complete source from placeholders.
pub trait WorkspacePredictionParameterSource {
    /// Actual borrowed selected native/source inventory and its paid destinations.
    type Context<'a>;

    /// Binds one physical module by dense ordinary materialization index. This
    /// index is visit order, not a physical ordinal (optional shared owners may
    /// leave gaps in the architecture's physical ordinal space). `source`
    /// and `local` are the actual architecture-constructed source/executable pair;
    /// tasks and layout retain their selected lowering and parameter identities.
    /// The implementation prices its containers and preserves backing aliases.
    fn bind<U: Parameterized<WorkspaceTensor>>(
        context: &mut Self::Context<'_>,
        index: usize,
        source: &U,
        local: &mut U,
        tasks: &[ReplicatedTextMaterializationTask],
        layout: Option<&LocalModelLayout>,
        workspace: &WorkspaceContext,
    ) -> Result<(), Error>;

    /// Optional source-bound recorder for the actual shared module worker.
    /// The index has already passed `bind`; no source or native authority is minted.
    fn invocation(
        _context: &mut Self::Context<'_>,
        _index: usize,
        _workspace: &WorkspaceContext,
    ) -> Result<Option<WorkspacePredictionInvocation>, Error> {
        Ok(None)
    }

    /// Completes this materialization pass before returning any modules. A
    /// retained source may reject an unconsumed tail or missing physical member.
    fn finish(_context: &mut Self::Context<'_>) -> Result<(), Error> {
        Ok(())
    }
}

/// Actual projected current prediction state. This is metadata, not source,
/// native-copy, completion, or occurrence authority. Projection must preserve
/// every backing alias and the selected mechanism's capacity/frontier.
pub enum WorkspacePredictionState {
    /// Ordered compressed members, including actual empty geometry.
    Sequential(Vec<WorkspaceCompressedCache>),
    /// Ordered local/pooling members, including pending/overlap buffers.
    Pooling(Vec<WorkspacePoolingLayerState>),
    /// Complete architecture-declared prediction model state.
    Model(ModelState),
}

/// Shared typed module adapter retaining both prepared representations.
pub struct WorkspacePredictionModule<U> {
    local: U,
    _source: U,
    invocation: Option<WorkspacePredictionInvocation>,
    // Prepared containers and parameter aliases retire before their trace owner.
    _context: WorkspaceContext,
}
impl<U> AsMut<U> for WorkspacePredictionModule<U> {
    fn as_mut(&mut self) -> &mut U {
        &mut self.local
    }
}

/// Metadata realization through the existing architecture materializer.
/// It does not select a family, execute native work or install a request scope.
pub struct WorkspacePredictionMaterializer<P>(PhantomData<fn() -> P>);

/// Borrowed state/source controls for one materialization pass.
pub struct WorkspacePredictionMaterialization<'a, P: WorkspacePredictionParameterSource> {
    parameters: P::Context<'a>,
    current: CurrentState<'a>,
    context: &'a WorkspaceContext,
    module: usize,
    pooling: usize,
}

enum CurrentState<'a> {
    Borrowed(&'a WorkspacePredictionState),
    Owned(WorkspacePredictionState),
}
impl CurrentState<'_> {
    fn get(&self) -> &WorkspacePredictionState {
        match self {
            Self::Borrowed(state) => state,
            Self::Owned(state) => state,
        }
    }
}

fn controls<T>(context: &WorkspaceContext) -> Result<(), Error> {
    let parts = [
        size_of::<T>(),
        size_of::<Result<T, Error>>(),
        size_of::<&WorkspaceContext>(),
    ];
    context.charge_metadata(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    Ok(())
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct WorkspacePredictionFailure {
    #[source]
    cause: Error,
    // Erased source shell and cause retire before the paying account.
    _funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}
fn with_failure<T, F: FnOnce() -> Result<T, Error>>(
    context: &WorkspaceContext,
    work: F,
) -> Result<T, Error> {
    controls::<(F, WorkspacePredictionFailure, Result<T, Error>)>(context)?;
    context.charge_metadata(
        WorkspaceContext::metadata_source_bytes::<WorkspacePredictionFailure>()
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    let funding = context.metadata_funding();
    work().map_err(|cause| {
        Error::backend_retained_source(WorkspacePredictionFailure {
            cause,
            _funding: funding,
        })
    })
}

fn mismatch(context: &WorkspaceContext) -> Error {
    context.metadata_error(format_args!(
        "prediction workspace state differs from its selected source layout"
    ))
}

impl WorkspacePredictionState {
    fn validate(
        &self,
        layout: PredictionStateSourceLayout<'_>,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        controls::<(PredictionStateSourceLayout<'_>, &Self)>(context)?;
        match (self, layout) {
            (Self::Sequential(states), PredictionStateSourceLayout::Sequential(policies))
                if states.len() == policies.len() =>
            {
                for (state, (_, policy)) in states.iter().zip(policies) {
                    state.validate_projected_policy(policy, context)?;
                }
            }
            (Self::Pooling(states), PredictionStateSourceLayout::Pooling(policies))
                if states.len() == policies.len() =>
            {
                for (state, (_, policy)) in states.iter().zip(policies) {
                    state.validate_projected_policy(policy, context)?;
                }
            }
            (Self::Model(state), PredictionStateSourceLayout::Model(layout))
                if state.optional_layout() == Some(layout) =>
            {
                state.validate_workspace_context(context)?;
            }
            _ => return Err(mismatch(context)),
        }
        Ok(())
    }
}

impl PreparedPredictionExtension<WorkspaceBackend> {
    /// Materializes already prepared selected modules against actual projected
    /// current state. Uses the ordinary architecture materialization order and
    /// parameter/source pair; it neither reselects the model nor binds placement.
    /// The caller retains the original native source through trace consumption.
    pub fn materialize_workspace<'a, P: WorkspacePredictionParameterSource>(
        self,
        parameters: P::Context<'a>,
        current: &'a WorkspacePredictionState,
        context: &'a WorkspaceContext,
    ) -> Result<
        MaterializedPredictionExtension<WorkspaceBackend, WorkspacePredictionMaterializer<P>>,
        Error,
    > {
        self.materialize_workspace_source::<P>(parameters, CurrentState::Borrowed(current), context)
            .map(|(extension, _)| extension)
    }

    /// Same shared materializer with an owned actual current-state projection.
    /// Its state may be constructed locally after this preparation lends its
    /// exact policy layout; parameter sources can retain a longer borrowed life.
    /// Returns the source projection beside the extension for paid lane aliasing.
    pub fn materialize_workspace_owned<'a, P: WorkspacePredictionParameterSource>(
        self,
        parameters: P::Context<'a>,
        current: WorkspacePredictionState,
        context: &'a WorkspaceContext,
    ) -> Result<
        (
            MaterializedPredictionExtension<WorkspaceBackend, WorkspacePredictionMaterializer<P>>,
            WorkspacePredictionState,
        ),
        Error,
    > {
        self.materialize_workspace_source::<P>(parameters, CurrentState::Owned(current), context)
            .map(|(extension, current)| {
                let CurrentState::Owned(current) = current else {
                    unreachable!("owned source materialization")
                };
                (extension, current)
            })
    }

    fn materialize_workspace_source<'a, P: WorkspacePredictionParameterSource>(
        self,
        parameters: P::Context<'a>,
        current: CurrentState<'a>,
        context: &'a WorkspaceContext,
    ) -> Result<
        (
            MaterializedPredictionExtension<WorkspaceBackend, WorkspacePredictionMaterializer<P>>,
            CurrentState<'a>,
        ),
        Error,
    > {
        controls::<(Self, WorkspacePredictionMaterialization<'a, P>)>(context)?;
        with_failure(context, || {
            current
                .get()
                .validate(self.state_source_layout(), context)?;
            let mut materialization = WorkspacePredictionMaterialization {
                parameters,
                current,
                context,
                module: 0,
                pooling: 0,
            };
            let extension =
                self.materialize::<WorkspacePredictionMaterializer<P>>(&mut materialization)?;
            P::finish(&mut materialization.parameters)?;
            Ok((extension, materialization.current))
        })
    }
}

impl<P: WorkspacePredictionParameterSource> PredictionExtensionMaterializer<WorkspaceBackend>
    for WorkspacePredictionMaterializer<P>
{
    type Error = Error;
    type Module<U> = WorkspacePredictionModule<U>;
    type SequentialState = WorkspacePredictionSequentialState;
    type PoolingState = WorkspacePoolingLayerState;
    type ModelState = ModelState;
    type Context<'a> = WorkspacePredictionMaterialization<'a, P>;
    type SnapshotContext<'a> = &'a WorkspaceContext;

    fn materialization_controls<T>(context: &mut Self::Context<'_>) -> Result<(), Error> {
        controls::<T>(context.context)
    }
    fn materialization_vector<T>(
        context: &mut Self::Context<'_>,
        count: usize,
    ) -> Result<Vec<T>, Error> {
        context.context.metadata_vec(count)
    }
    fn materialize_module<U: Parameterized<WorkspaceTensor>>(
        context: &mut Self::Context<'_>,
        prepared: PreparedPredictionUnit<U>,
        layout: Option<&LocalModelLayout>,
    ) -> Result<Self::Module<U>, Error> {
        controls::<(
            PreparedPredictionUnit<U>,
            Self::Module<U>,
            (U, U, std::sync::Arc<Vec<ReplicatedTextMaterializationTask>>),
            Option<&LocalModelLayout>,
        )>(context.context)?;
        let next = context
            .module
            .checked_add(1)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let (source, mut local, tasks) = prepared.into_shared_parts();
        P::bind(
            &mut context.parameters,
            context.module,
            &source,
            &mut local,
            &tasks,
            layout,
            context.context,
        )?;
        let invocation = P::invocation(&mut context.parameters, context.module, context.context)?;
        context.module = next;
        Ok(WorkspacePredictionModule {
            local,
            _source: source,
            invocation,
            _context: context.context.clone(),
        })
    }
    fn pooling_state(
        context: &mut Self::Context<'_>,
        _ordinal: usize,
        policy: LayerCachePolicy,
    ) -> Result<Self::PoolingState, Error> {
        controls::<(LayerCachePolicy, Self::PoolingState)>(context.context)?;
        let WorkspacePredictionState::Pooling(states) = context.current.get() else {
            return Err(mismatch(context.context));
        };
        let state = states
            .get(context.pooling)
            .ok_or_else(|| mismatch(context.context))?;
        state.validate_projected_policy(&policy, context.context)?;
        context.pooling = context
            .pooling
            .checked_add(1)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        Ok(state.clone())
    }
    fn model_state(
        context: &mut Self::Context<'_>,
        layout: StateLayout,
    ) -> Result<ModelState, Error> {
        controls::<(StateLayout, ModelState)>(context.context)?;
        let WorkspacePredictionState::Model(state) = context.current.get() else {
            return Err(mismatch(context.context));
        };
        if state.optional_layout() != Some(&layout) {
            return Err(mismatch(context.context));
        }
        state.try_clone_workspace(context.context)
    }
    fn sequential_state() -> Self::SequentialState {
        // The legacy infallible startup API has no source or context. An absent
        // projection never fabricates geometry; using it fails before equations.
        WorkspacePredictionSequentialState(None)
    }
    fn retain_prediction_invocation_optional<'a, O, const N: usize>(
        outcome: Result<O, Error>,
        state: impl IntoIterator<Item = &'a WorkspaceTensor>,
        outputs: impl FnOnce(&O) -> Option<[&WorkspaceTensor; N]>,
        context: &WorkspaceContext,
    ) -> PredictionInvocation<WorkspaceTensor, O> {
        let mut retained = Vec::new();
        let result = (|| {
            controls::<(
                PredictionInvocation<WorkspaceTensor, O>,
                Vec<WorkspaceTensor>,
                Option<[&WorkspaceTensor; N]>,
            )>(context)?;
            for value in state {
                context.reserve_metadata_vec(&mut retained, 1)?;
                retained.push(value.clone());
            }
            if let Ok(output) = &outcome {
                for value in outputs(output).into_iter().flatten() {
                    context.reserve_metadata_vec(&mut retained, 1)?;
                    retained.push(value.clone());
                }
            }
            Ok::<_, Error>(())
        })();
        // Only metadata is realized here. Native materializers retain their
        // unchanged native root collector and failure/recovery semantics.
        let outcome = match result {
            Ok(()) => outcome,
            Err(cause) => Err(cause),
        };
        PredictionInvocation::from_prepared_parts(outcome, retained)
    }
    fn invoke_module<U, O>(
        module: &mut Self::Module<U>,
        context: &WorkspaceContext,
        operation: impl FnOnce(&mut U) -> PredictionInvocation<WorkspaceTensor, O>,
    ) -> Result<O, Error>
    where
        U: Parameterized<WorkspaceTensor>,
    {
        controls::<(PredictionInvocation<WorkspaceTensor, O>, Option<usize>)>(context)?;
        with_failure(context, || {
            let entry = module
                .invocation
                .as_ref()
                .map(|source| source.begin(context))
                .transpose()?;
            let invocation = operation(module.as_mut());
            context.validate_values(invocation.retained_values())?;
            if let Some(entry) = entry {
                module
                    .invocation
                    .as_ref()
                    .expect("recorded module")
                    .complete(entry, invocation.retained_values().count(), context)?;
            }
            invocation.into_outcome()
        })
    }
    fn invoke_module_with_shared<U, V, O>(
        module: &mut Self::Module<U>,
        shared: Option<&mut Self::Module<V>>,
        context: &WorkspaceContext,
        operation: impl FnOnce(&mut U, Option<&mut V>) -> PredictionInvocation<WorkspaceTensor, O>,
    ) -> Result<O, Error>
    where
        U: Parameterized<WorkspaceTensor>,
        V: Parameterized<WorkspaceTensor>,
    {
        controls::<(
            PredictionInvocation<WorkspaceTensor, O>,
            Option<&mut Self::Module<V>>,
            (Option<usize>, Option<usize>),
        )>(context)?;
        with_failure(context, || {
            let mut shared = shared;
            let entry = module
                .invocation
                .as_ref()
                .map(|source| source.begin(context))
                .transpose()?;
            let shared_entry = shared
                .as_ref()
                .and_then(|module| module.invocation.as_ref())
                .map(|source| source.begin(context))
                .transpose()?;
            let invocation = operation(module.as_mut(), shared.as_deref_mut().map(AsMut::as_mut));
            context.validate_values(invocation.retained_values())?;
            let roots = invocation.retained_values().count();
            if let Some(entry) = shared_entry {
                shared
                    .as_ref()
                    .and_then(|module| module.invocation.as_ref())
                    .expect("recorded shared module")
                    .complete(entry, roots, context)?;
            }
            if let Some(entry) = entry {
                module
                    .invocation
                    .as_ref()
                    .expect("recorded module")
                    .complete(entry, roots, context)?;
            }
            invocation.into_outcome()
        })
    }
    fn complete_prediction_values<'a>(
        values: impl IntoIterator<Item = &'a WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<(), eredu_core::BackendFailure> {
        // This trait boundary is used by shared completion hooks. All equation
        // invocations above preserve their native-neutral Error directly.
        let bytes =
            eredu_core::BackendFailure::source_retention_peak_bytes::<WorkspacePredictionFailure>()
                .ok_or(eredu_core::HostMetadataFundingError::Overflow)?;
        context.charge_metadata(bytes).map_err(metadata_failure)?;
        context.validate_values(values).map_err(|cause| {
            eredu_core::BackendFailure::from_error(WorkspacePredictionFailure {
                cause,
                _funding: context.metadata_funding(),
            })
        })
    }
}

fn metadata_failure(cause: WorkspaceMetadataError) -> eredu_core::BackendFailure {
    use eredu_core::HostMetadataFundingError as E;
    match cause {
        WorkspaceMetadataError::Capacity {
            required,
            available,
        } => match (u64::try_from(required), u64::try_from(available)) {
            (Ok(required), Ok(available)) => E::Capacity {
                required,
                available,
            },
            _ => E::Overflow,
        },
        WorkspaceMetadataError::Funding(cause) => cause,
        WorkspaceMetadataError::Overflow => E::Overflow,
        _ => E::Unavailable,
    }
    .into_backend_failure()
}

impl PredictionModelState<WorkspaceBackend> for ModelState {
    type LayerState = WorkspaceResidentLayerState;
    fn prediction_layers_mut(&mut self) -> &mut [Self::LayerState] {
        self.as_mut()
    }
}

/// Explicit compressed metadata source. An absent legacy startup slot has no
/// inferred geometry and refuses append/restore before any equation is emitted.
#[derive(Debug, Clone)]
pub struct WorkspacePredictionSequentialState(Option<WorkspaceCompressedCache>);
impl WorkspacePredictionSequentialState {
    /// No cache geometry or state authority is fabricated by cold construction.
    pub(crate) fn unprojected() -> Self { Self(None) }
}
impl From<WorkspaceCompressedCache> for WorkspacePredictionSequentialState {
    fn from(source: WorkspaceCompressedCache) -> Self {
        Self(Some(source))
    }
}
impl CompressedAttentionCache<WorkspaceTensor> for WorkspacePredictionSequentialState {
    type Checkpoint = Self;
    fn offset(&self) -> i32 {
        self.0.as_ref().map_or(0, CompressedAttentionCache::offset)
    }
    fn is_paged(&self) -> bool {
        false
    }
    fn append(
        &mut self,
        state: CompressedAttentionState<WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<CompressedAttentionView<WorkspaceTensor>, Error> {
        self.0
            .as_mut()
            .ok_or_else(|| mismatch(context))?
            .append(state, context)
    }
    fn visit_blocks<F>(
        &mut self,
        query_tokens: i32,
        context: &WorkspaceContext,
        visitor: F,
    ) -> Result<CompressedAttentionScan, Error>
    where
        F: FnMut(CompressedAttentionBlock<WorkspaceTensor>) -> Result<u64, Error>,
    {
        self.0
            .as_mut()
            .ok_or_else(|| mismatch(context))?
            .visit_blocks(query_tokens, context, visitor)
    }
    fn checkpoint(&self) -> Self {
        self.clone()
    }
    fn restore(&mut self, checkpoint: &Self, context: &WorkspaceContext) -> Result<(), Error> {
        let target = self.0.as_mut().ok_or_else(|| mismatch(context))?;
        let source = checkpoint.0.as_ref().ok_or_else(|| mismatch(context))?;
        target.restore(source, context)
    }
    fn finalize(&mut self) -> Result<(), Error> {
        match &mut self.0 {
            Some(state) => state.finalize(),
            None => Err(WorkspaceMetadataError::Unqualified.into()),
        }
    }
    fn clear(&mut self) -> Result<(), Error> {
        match &mut self.0 {
            Some(state) => state.clear(),
            None => Err(WorkspaceMetadataError::Unqualified.into()),
        }
    }
}
impl RuntimeLayerState<WorkspaceBackend> for WorkspacePredictionSequentialState {
    type RetainedValues<'a> = std::iter::Flatten<
        std::option::IntoIter<
            <WorkspaceCompressedCache as RuntimeLayerState<WorkspaceBackend>>::RetainedValues<'a>,
        >,
    >;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.0
            .as_ref()
            .map(RuntimeLayerState::<WorkspaceBackend>::retained_values)
            .into_iter()
            .flatten()
    }
}

impl WorkspacePredictionState {
    /// Copies current metadata into the exact typed lane selected by the shared
    /// executor. The executor supplies only its existing profile dispatch; this
    /// source-backed factory preserves the supplied frontier/backing aliases.
    /// It never invokes the legacy infallible `new_state` or native copy worker.
    pub fn prepare_current_lane<A, E, P>(
        &self,
        executor: &E,
        context: &WorkspaceContext,
    ) -> Result<E::LaneState, Error>
    where
        P: WorkspacePredictionParameterSource,
        E: MaterializedPredictionExecutor<A, WorkspaceBackend, WorkspacePredictionMaterializer<P>>,
    {
        controls::<(CurrentStateFactory<'_>, E::LaneState)>(context)?;
        with_failure(context, || {
            executor.prepare_new_state(&mut CurrentStateFactory {
                current: self,
                context,
            })
        })
    }
}

struct CurrentStateFactory<'a> {
    current: &'a WorkspacePredictionState,
    context: &'a WorkspaceContext,
}
impl<P: WorkspacePredictionParameterSource>
    PredictionStateStartupFactory<WorkspaceBackend, WorkspacePredictionMaterializer<P>>
    for CurrentStateFactory<'_>
{
    type Error = Error;
    type Prepared<T> = T;
    fn sequential(
        &mut self,
        count: usize,
    ) -> Result<Vec<WorkspacePredictionSequentialState>, Error> {
        let WorkspacePredictionState::Sequential(states) = self.current else {
            return Err(mismatch(self.context));
        };
        if count != states.len() {
            return Err(mismatch(self.context));
        }
        let mut output = self.context.metadata_vec(count)?;
        for state in states {
            state.validate_projection_context(self.context)?;
            // Fixed metadata alias clone; the source already carries exact
            // mechanism geometry even when its physical arrays are absent.
            output.push(WorkspacePredictionSequentialState::from(state.clone()));
        }
        Ok(output)
    }
    fn pooling(
        &mut self,
        prototype: &[WorkspacePoolingLayerState],
    ) -> Result<Vec<WorkspacePoolingLayerState>, Error> {
        let WorkspacePredictionState::Pooling(states) = self.current else {
            return Err(mismatch(self.context));
        };
        if prototype.len() != states.len() {
            return Err(mismatch(self.context));
        }
        let mut output = self.context.metadata_vec(states.len())?;
        for (state, prototype) in states.iter().zip(prototype) {
            state.validate_projected_policy(prototype.projected_policy(), self.context)?;
            output.push(state.clone());
        }
        Ok(output)
    }
    fn model(&mut self, prototype: &ModelState) -> Result<ModelState, Error> {
        let WorkspacePredictionState::Model(state) = self.current else {
            return Err(mismatch(self.context));
        };
        if prototype.optional_layout() != state.optional_layout() {
            return Err(mismatch(self.context));
        }
        state.try_clone_workspace(self.context)
    }
}

#[cfg(test)]
pub(crate) mod tests;
