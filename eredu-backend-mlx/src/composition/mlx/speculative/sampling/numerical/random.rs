//! Closed completed PRNG keys, admitted by the actual request numerical engine.
use super::*;
use crate::backend::runtime::generation::MlxSamplingBackend;
use eredu_runtime::generation::{
    PreparedCategoricalPolicy, PreparedGreedyError, SpeculativeCategoricalProgram,
};

/// Immutable completed key. Cloning shares its prepaid immutable backing;
/// advancing a caller's state replaces this owner only after exact completion.
/// No native Array, naked account, or raw-key constructor is exposed.
#[derive(Clone, Debug)]
pub(crate) struct OriginalNumericalKey(OriginalNumericalValue);
impl OriginalNumericalKey {
    pub(super) fn copied(value: OriginalNumericalValue) -> Self { Self(value) }
    pub(super) fn completed(
        array: Array,
        stream: ValueStream,
        custody: OriginalSpeculativeNumericalBudgetCustody,
        funding: WorkspaceMetadataFunding,
        budget: safemlx::OriginalBufferBudget,
    ) -> Self {
        Self(OriginalNumericalValue::completed(
            array,
            stream,
            Meaning::RandomKey,
            custody,
            funding,
        ).with_original_budget(budget))
    }
    pub(in crate::composition::mlx::speculative::sampling) fn value(&self) -> &OriginalNumericalValue {
        &self.0
    }
}

#[derive(Debug, thiserror::Error)]
enum RandomCause {
    #[error("explicit-key numerical producer returned a different output kind")]
    Output,
}

/// Actual configured scalar seed; no imported native key is relabeled as paid.
pub(crate) fn create_key(
    seed: u64,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<OriginalNumericalKey, Error> {
    execute(
        context, SamplingPlacement::Target,
        program::SpeculativeNumericalKind::CreateKey { seed },
        None,
        None,
    )
    .and_then(|output| match output {
        NumericalOutput::Key(key) => Ok(key),
        _ => Err(output_error(context)),
    })
}
/// Same split-two transaction as RandomState::next_key: retain row zero and
/// return row one. Failure leaves the caller's source state intact while the
/// attempted phase remains cumulatively charged and independently recovered.
pub(crate) fn next_key(
    state: &mut OriginalNumericalKey,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<OriginalNumericalKey, Error> {
    match execute(
        context, SamplingPlacement::Target,
        program::SpeculativeNumericalKind::NextKey,
        Some(state.value()),
        None,
    )? {
        NumericalOutput::KeyPair { current, next } => {
            *state = current;
            Ok(next)
        }
        _ => Err(output_error(context)),
    }
}
/// Sequential acceptance draw. The caller's key is replaced only after both
/// the F32 scalar and the advanced key complete and their Record retires.
pub(crate) fn sample_unit_interval(
    state: &mut OriginalNumericalKey,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<f32, Error> {
    match execute(
        context, SamplingPlacement::Target,
        program::SpeculativeNumericalKind::UniformUnitInterval,
        Some(state.value()),
        None,
    )? {
        NumericalOutput::RandomUnitInterval { value, advanced } => {
            *state = advanced;
            Ok(value)
        }
        _ => Err(output_error(context)),
    }
}
/// Exact existing split(position+1)-then-select worker. Retained backing is the
/// whole selected split table, rather than a guessed two-word allocation.
pub(crate) fn key_at(
    root: &OriginalNumericalKey,
    position: eredu_core::SpeculativeDraftRandomPosition,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<OriginalNumericalKey, Error> {
    let position = u32::try_from(position.get())
        .map_err(|_| fixed(context, program::SpeculativeNumericalError::Key))?;
    match execute(
        context, SamplingPlacement::Draft,
        program::SpeculativeNumericalKind::KeyAt { position },
        Some(root.value()),
        None,
    )? {
        NumericalOutput::Key(key) => Ok(key),
        _ => Err(output_error(context)),
    }
}
/// Known ordinary processed-logit categorical worker, with the actual completed
/// state key. The scalar token and advanced key complete under the same phase.
pub(crate) fn sample_stochastic<S: SpeculativeSampler<MlxSamplingBackend>>(
    policy: &S,
    logits: &OriginalNumericalValue,
    temperature: f32,
    state: &mut OriginalNumericalKey,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<u32, Error> {
    sample_stochastic_at(policy,logits,temperature,state,SamplingPlacement::Target,context)
}
pub(crate) fn sample_stochastic_at<S:SpeculativeSampler<MlxSamplingBackend>>(
    policy:&S,logits:&OriginalNumericalValue,temperature:f32,state:&mut OriginalNumericalKey,
    placement:SamplingPlacement,context:SpeculativeExecutionStreams<'_>,
)->Result<u32,Error>{
    let (sources, _) = context.original_numerical_for(placement).ok_or_else(no_context)?;
    reserve_controls(sources.metadata_funding())?;
    let policy=match super::policy::bound_controller_choice(policy,logits,sources)? {
        Some(choice)=>choice.categorical(temperature),
        None=>policy.prepared_categorical_policy().ok_or_else(||sources.retain_startup_error(PreparedGreedyError::Unknown))?.bind(temperature),
    }.map_err(|cause|sources.retain_startup_error(cause))?;
    match execute(
        context, placement,
        program::SpeculativeNumericalKind::Categorical(policy),
        Some(logits),
        Some(state.value()),
    )? {
        NumericalOutput::RandomToken { token, advanced } => {
            *state = advanced;
            Ok(token)
        }
        _ => Err(output_error(context)),
    }
}
fn execute(
    context: SpeculativeExecutionStreams<'_>,placement:SamplingPlacement,
    kind: program::SpeculativeNumericalKind,
    left: Option<&OriginalNumericalValue>,
    right: Option<&OriginalNumericalValue>,
) -> Result<NumericalOutput, Error> {
    let (sources, _environment) = context.original_numerical_for(placement).ok_or_else(no_context)?;
    let result = (|| {
        reserve_controls(sources.metadata_funding())?;
        NumericalProducer::execute_inputs_at(context,placement,kind,left,right)
    })();
    result.map_err(|cause| sources.retain_error(cause))
}
fn no_context() -> Error {
    Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)
}
fn output_error(context: SpeculativeExecutionStreams<'_>) -> Error {
    fixed(context, RandomCause::Output)
}
fn fixed<E: std::error::Error + Send + Sync + 'static>(
    context: SpeculativeExecutionStreams<'_>,
    cause: E,
) -> Error {
    match context.original_numerical() {
        Some((sources, _)) => sources.retain_startup_error(cause),
        None => no_context(),
    }
}
fn reserve_controls(funding: &WorkspaceMetadataFunding) -> Result<(), Error> {
    let parts = [
        size_of::<SamplingPlacement>(),size_of::<Result<u32,Error>>(),
        size_of::<(&OriginalNumericalValue,f32,&mut OriginalNumericalKey,SamplingPlacement,SpeculativeExecutionStreams<'_>)>(),
        size_of::<OriginalNumericalKey>(),
        size_of::<Option<OriginalNumericalKey>>(),
        size_of::<Result<OriginalNumericalKey, Error>>(),
        size_of::<Result<u32, Error>>(),
        size_of::<Result<f32, Error>>(),
        size_of::<Option<PreparedCategoricalPolicy<'_>>>(),
        size_of::<Result<SpeculativeCategoricalProgram, PreparedGreedyError>>(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
        size_of::<program::SpeculativeNumericalKind>(),
        size_of::<NumericalOutput>(),
        size_of::<Result<NumericalOutput, Error>>(),
        size_of::<RandomCause>(),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Overflow,
        ))?;
    funding
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)
}
