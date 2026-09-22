//! Metadata realization of the ordinary runtime sampling primitives.

use crate::{PenaltyConfig, SamplingBackend, TokenDomain};
use eredu_core::{TextFilterWorkspace, TokenFilter};
use eredu_nn::{Error, Tensor, workspace::*};

/// Metadata random state with the same two-word key geometry as its selected
/// native mechanism. Existing keys must be projected with their real capacity.
#[derive(Debug, Clone)]
pub struct WorkspaceSamplingRandomState {
    key: WorkspaceTensor,
    fresh: bool,
}
impl WorkspaceSamplingRandomState {
    /// Quotes creation of a fresh explicit key without allocating native state.
    pub fn from_seed(context: &WorkspaceContext) -> Result<Self, Error> {
        let key = operation(
            WorkspaceSamplingOperation::CreateRandomKey,
            &[],
            context.layout(&[2], WorkspaceDtype::Uint32)?,
            context,
        )?;
        Ok(Self { key, fresh: true })
    }
    /// Imports a projected native random key without reading its values.
    pub fn from_key(key: WorkspaceTensor) -> Result<Self, Error> {
        if key.shape() != [2] || key.layout().dtype() != WorkspaceDtype::Uint32 {
            return Err(Error::backend(
                "sampling random key must contain two unsigned words",
            ));
        }
        Ok(Self { key, fresh: false })
    }
    /// Move the current key without creating another metadata/native handle.
    pub fn into_key(self) -> WorkspaceTensor {
        self.key
    }
    /// The same split-two and static row selection as the native state worker.
    pub fn next_key(&mut self, context: &WorkspaceContext) -> Result<WorkspaceTensor, Error> {
        let split = split_keys(&self.key, 2, context)?;
        self.key = select_key(&split, 0, context)?;
        self.fresh = false;
        select_key(&split, 1, context)
    }
    /// Same sequential split and fixed [0, 1) draw as the native state worker.
    /// Both this result and the advanced key remain part of the completion roots.
    pub fn uniform_unit_interval(
        &mut self,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        let key = self.next_key(context)?;
        operation(
            WorkspaceSamplingOperation::UniformUnitInterval,
            &[&key],
            context.layout(&[1], WorkspaceDtype::Float32)?,
            context,
        )
    }
    /// Exact position-key constructor, including the complete split table.
    pub fn key_at(
        key: &WorkspaceTensor,
        position: u32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        let count = position
            .checked_add(1)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| {
                context.metadata_error(format_args!("random subkey index exceeds i32"))
            })?;
        select_key(&split_keys(key, count, context)?, position, context)
    }
    /// Retained key after the latest metadata sampling step.
    pub fn key(&self) -> &WorkspaceTensor {
        &self.key
    }
}

mod request;
pub use request::{
    SamplingWorkspaceInputPlan, SamplingWorkspaceObserver, SamplingWorkspacePhase,
    SamplingWorkspaceReport, WorkspaceSamplingInput, WorkspaceSamplingSource,
    quote_sampling_workspace, quote_sampling_workspace_with_observer,
};

#[cfg(test)]
mod tests;

/// Runs shared standard, constrained and adaptive sampling policy without
/// native values. Scalar results are witnesses only: native bounds must cover
/// every token, probability and route consistent with each descriptor.
#[derive(Debug)]
pub struct WorkspaceSamplingBackend;

fn logits_width(logits: &WorkspaceTensor) -> Result<usize, Error> {
    logits_layout_width(logits.layout())
}
fn logits_layout_width(layout: &WorkspaceLayout) -> Result<usize, Error> {
    let width = layout.shape().last().copied().unwrap_or(0);
    if width <= 0 || layout.dtype() != WorkspaceDtype::Float32 {
        return Err(Error::backend(
            "sampling workspace requires nonempty floating logits",
        ));
    }
    Ok(width as usize)
}
fn operation(
    kind: WorkspaceSamplingOperation,
    inputs: &[&WorkspaceTensor],
    output: WorkspaceLayout,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let mut outputs = context.metadata_vec(1)?;
    outputs.push(output);
    Ok(context
        .execute(WorkspaceOperationKind::Sampling(kind), inputs, outputs)?
        .remove(0))
}
fn same_shape(
    kind: WorkspaceSamplingOperation,
    logits: &WorkspaceTensor,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    logits_width(logits)?;
    operation(kind, &[logits], logits.layout().clone(), context)
}

fn apply_workspace_token_filter(
    logits: &WorkspaceTensor,
    filter: TextFilterWorkspace<'_>,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    filter
        .validate_output_width(logits_width(logits)?)
        .map_err(Error::backend)?;
    match filter {
        TextFilterWorkspace::Exact(TokenFilter::All) => Ok(logits.clone()),
        TextFilterWorkspace::Exact(TokenFilter::Allowed(_)) => {
            same_shape(WorkspaceSamplingOperation::TokenFilter, logits, context)
        }
        TextFilterWorkspace::OptionalMask { .. } => {
            // The provider's storage effect carries both the possible masked
            // allocation and input alias through later sampling/output roots.
            same_shape(
                WorkspaceSamplingOperation::OptionalTokenFilter,
                logits,
                context,
            )
        }
    }
}

impl WorkspaceSamplingBackend {
    /// Traces the actual fixed mask selected by the shared ordinary constructor.
    pub fn apply_prepared_token_mask(
        logits: &WorkspaceTensor,
        plan: crate::generation::TokenMaskPlan<'_>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        if !plan.matches_shape(logits.shape()) {
            return Err(context.metadata_error(format_args!("token mask source geometry changed")));
        }
        if plan.is_identity() {
            Ok(logits.clone())
        } else {
            same_shape(WorkspaceSamplingOperation::TokenFilter, logits, context)
        }
    }
}

impl SamplingBackend for WorkspaceSamplingBackend {
    type Logits = WorkspaceTensor;
    type Token = WorkspaceTensor;
    type RandomState = WorkspaceSamplingRandomState;
    type Context = WorkspaceContext;
    type Error = Error;

    fn clone_token_with_host_source(
        value: &Self::Token,
        funding: &eredu_core::HostMetadataFunding,
        context: &Self::Context,
    ) -> Result<Self::Token, eredu_core::BackendFailure> {
        funding.reserve_metadata(std::mem::size_of::<(
            &WorkspaceTensor,
            &eredu_core::HostMetadataFunding,
            &WorkspaceContext,
            Result<WorkspaceTensor, eredu_core::BackendFailure>,
        )>())?;
        context
            .validate_values([value])
            .map_err(eredu_core::BackendFailure::from_error)?;
        Ok(value.clone())
    }
    fn clone_logits_with_host_source(
        value: &Self::Logits,
        funding: &eredu_core::HostMetadataFunding,
        context: &Self::Context,
    ) -> Result<Self::Logits, eredu_core::BackendFailure> {
        Self::clone_token_with_host_source(value, funding, context)
    }

    fn clone_random_with_host_source(
        value: &Self::RandomState,
        funding: &eredu_core::HostMetadataFunding,
        context: &Self::Context,
    ) -> Result<Self::RandomState, eredu_core::BackendFailure> {
        // WorkspaceTensor::clone records the actual native descriptor clone in
        // this trace and pays its own metadata layout through its source. The
        // extra wrapper is a host constructor, never a random numerical op.
        funding.reserve_metadata(std::mem::size_of::<(
            &Self::RandomState,
            &eredu_core::HostMetadataFunding,
            &Self::Context,
            Self::RandomState,
            Result<Self::RandomState, eredu_core::BackendFailure>,
        )>())?;
        context
            .validate_values([&value.key])
            .map_err(eredu_core::BackendFailure::from_error)?;
        Ok(value.clone())
    }

    fn error(message: String) -> Error {
        Error::backend_message(message)
    }
    fn validate_token(
        token: &WorkspaceTensor,
        domain: TokenDomain,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        let cardinality = u32::try_from(domain.cardinality())
            .ok()
            .filter(|n| *n > 0 && *n <= i32::MAX as u32)
            .ok_or_else(|| Error::backend("sampling token domain exceeds tensor extent"))?;
        if !matches!(
            token.layout().dtype(),
            WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
        ) {
            return Err(Error::backend("sampling token requires integer storage"));
        }
        operation(
            WorkspaceSamplingOperation::ValidateToken { cardinality },
            &[token],
            context.layout(token.shape(), WorkspaceDtype::Int32)?,
            context,
        )
    }
    fn scale_temperature(
        logits: &WorkspaceTensor,
        temperature: f32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        logits_width(logits)?;
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(Error::backend(
                "sampling temperature must be positive and finite",
            ));
        }
        logits.multiply_scalar(1.0 / temperature, context)
    }
    fn apply_penalties(
        logits: &WorkspaceTensor,
        history: &[u32],
        penalties: PenaltyConfig,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        apply_penalties_extent(logits, history.len(), penalties, context)
    }
    fn apply_top_k(
        logits: WorkspaceTensor,
        top_k: i32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        let vocabulary = logits_width(&logits)?;
        if top_k <= 0 || top_k as usize >= vocabulary {
            return Ok(logits);
        }
        same_shape(
            WorkspaceSamplingOperation::TopK { keep: top_k as u32 },
            &logits,
            context,
        )
    }
    fn apply_top_p(
        logits: WorkspaceTensor,
        top_p: f32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        logits_width(&logits)?;
        if top_p >= 1.0 {
            return Ok(logits);
        }
        same_shape(WorkspaceSamplingOperation::TopP, &logits, context)
    }
    fn apply_min_p(
        logits: WorkspaceTensor,
        min_p: f32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        logits_width(&logits)?;
        if min_p <= 0.0 {
            return Ok(logits);
        }
        same_shape(WorkspaceSamplingOperation::MinP, &logits, context)
    }
    fn apply_token_filter(
        logits: &WorkspaceTensor,
        filter: &TokenFilter,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        apply_workspace_token_filter(logits, filter.into(), context)
    }
    fn apply_mirostat(
        logits: &WorkspaceTensor,
        history: &[u32],
        penalties: PenaltyConfig,
        temperature: f32,
        _mu: f32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        apply_mirostat_extent(logits, history.len(), penalties, temperature, _mu, context)
    }
    fn sample_raw(
        logits: &WorkspaceTensor,
        temperature: f32,
        random: Option<&mut Self::RandomState>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        if temperature == 0.0 {
            Self::sample_processed(logits, temperature, random, context)
        } else {
            Self::sample_processed(
                &Self::scale_temperature(logits, temperature, context)?,
                temperature,
                random,
                context,
            )
        }
    }
    fn sample_processed(
        logits: &WorkspaceTensor,
        temperature: f32,
        random: Option<&mut Self::RandomState>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        logits_width(logits)?;
        let output = context.layout(
            &logits.shape()[..logits.shape().len() - 1],
            WorkspaceDtype::Uint32,
        )?;
        if temperature == 0.0 {
            return operation(
                WorkspaceSamplingOperation::Greedy,
                &[logits],
                output,
                context,
            );
        }
        let random =
            random.ok_or_else(|| Error::backend("sampling requires an explicit random key"))?;
        let key = random.next_key(context)?;
        operation(
            WorkspaceSamplingOperation::Categorical,
            &[logits, &key],
            output,
            context,
        )
    }
    fn token_id(token: &WorkspaceTensor, context: &WorkspaceContext) -> Result<u32, Error> {
        if token.layout().elements()? != 1 {
            return Err(Error::backend("sampling policy requires a scalar token"));
        }
        let _ = operation(
            WorkspaceSamplingOperation::ReadToken,
            &[token],
            token.layout().clone(),
            context,
        )?;
        Ok(0)
    }
    fn token_probability(
        logits: &WorkspaceTensor,
        token: u32,
        context: &WorkspaceContext,
    ) -> Result<f32, Error> {
        let vocabulary = logits_width(logits)?;
        if token as usize >= vocabulary || logits.shape().len() > 3 {
            return Err(Error::backend(
                "sampling probability selection is outside the logits",
            ));
        }
        let _ = operation(
            WorkspaceSamplingOperation::TokenProbability,
            &[logits],
            context.layout(&[], WorkspaceDtype::Float32)?,
            context,
        )?;
        Ok(1.0)
    }
}

// Native penalty bounds depend on the full applicable history extent, not on
// token values or distinct-ID observations. Both numerical metadata sampling and
// scalar-only projection use this one primitive descriptor construction.
fn apply_penalties_extent(
    logits: &WorkspaceTensor,
    history_len: usize,
    penalties: PenaltyConfig,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    logits_width(logits)?;
    if history_len == 0 || penalties.is_identity() {
        return Ok(logits.clone());
    }
    let history_positions = if penalties.repeat_last_n < 0 {
        history_len
    } else {
        history_len.min(penalties.repeat_last_n as usize)
    };
    same_shape(
        WorkspaceSamplingOperation::Penalties {
            history_positions,
            repetition: penalties.repeat_penalty != 1.0,
            additive: penalties.frequency_penalty != 0.0 || penalties.presence_penalty != 0.0,
        },
        logits,
        context,
    )
}

fn apply_mirostat_extent(
    logits: &WorkspaceTensor,
    history_len: usize,
    penalties: PenaltyConfig,
    temperature: f32,
    _mu: f32,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let vocabulary = logits_width(logits)?;
    if logits.layout().elements()? != vocabulary as u64 {
        return Err(Error::backend("Mirostat workspace requires one sequence"));
    }
    let logits = apply_penalties_extent(logits, history_len, penalties, context)?;
    let logits = WorkspaceSamplingBackend::scale_temperature(&logits, temperature, context)?;
    same_shape(WorkspaceSamplingOperation::MirostatCutoff, &logits, context)
}

fn split_keys(
    key: &WorkspaceTensor,
    count: i32,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    if key.shape() != [2] || key.layout().dtype() != WorkspaceDtype::Uint32 || count <= 0 {
        return Err(context.metadata_error(format_args!("invalid explicit key split geometry")));
    }
    operation(
        WorkspaceSamplingOperation::SplitRandomKey,
        &[key],
        context.layout(&[count, 2], WorkspaceDtype::Uint32)?,
        context,
    )
}

// Same first-axis static Slice/Reshape as the native random-state worker. The
// closed descriptor retains the row instead of losing it in a generic Index.
fn select_key(
    keys: &WorkspaceTensor,
    index: u32,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let shape = keys.shape();
    if shape.len() != 2
        || shape[0] <= 0
        || shape[1] != 2
        || keys.layout().dtype() != WorkspaceDtype::Uint32
        || u64::from(index) >= shape[0] as u64
    {
        return Err(context.metadata_error(format_args!("random key row exceeds split table")));
    }
    operation(
        WorkspaceSamplingOperation::SelectRandomKey { index },
        &[keys],
        context.layout(&[2], WorkspaceDtype::Uint32)?,
        context,
    )
}
