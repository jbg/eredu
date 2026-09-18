//! Borrowed sampling state for the shared prepared equation traversal.

use eredu_core::{TextFilterWorkspace, TextGenerationConfig};
use eredu_nn::{Error, workspace::WorkspaceContext};
use eredu_runtime::{
    ConfiguredTextSampler,
    working_memory::{
        SamplingWorkspaceObserver, SamplingWorkspaceReport, WorkspaceSamplingRandomState,
        WorkspaceSamplingSource, quote_sampling_workspace_with_observer,
    },
};

/// Actual sampler policy and history borrowed alongside projected random state.
///
/// The equation traversal supplies the score layout and complete backing
/// envelope, and the request geometry supplies the entire future output
/// allowance. Inspection advances scalar metadata only; it neither copies nor
/// mutates the sampler history, adaptive state or native random key. The random
/// projection must use the same workspace context as the equation traversal and
/// must describe the actual backing, including unknown capacity when necessary.
///
/// This descriptor does not validate a new request against the saved policy,
/// retain native source custody, price a destination copy or authorize inference.
/// Those checks belong to the enclosing prepared consumer.
#[derive(Clone, Copy)]
pub struct BorrowedTextSamplingWorkspace<'a> {
    sampler: &'a ConfiguredTextSampler,
    temperature: f32,
    random: Option<&'a WorkspaceSamplingRandomState>,
    filter: TextFilterWorkspace<'a>,
}

impl<'a> BorrowedTextSamplingWorkspace<'a> {
    /// Borrows the existing sampler and the selected metadata random state.
    /// Temperature and context compatibility are checked during quotation by
    /// the ordinary runtime sampling mechanism. Stochastic policies need their
    /// selected random state; deterministic policies may omit it.
    pub fn new(
        sampler: &'a ConfiguredTextSampler,
        temperature: f32,
        random: Option<&'a WorkspaceSamplingRandomState>,
        filter: impl Into<TextFilterWorkspace<'a>>,
    ) -> Self {
        Self {
            sampler,
            temperature,
            random,
            filter: filter.into(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum TextSamplingInput<'a> {
    Configured(TextGenerationConfig, TextFilterWorkspace<'a>),
    Borrowed(BorrowedTextSamplingWorkspace<'a>),
}

impl TextSamplingInput<'_> {
    pub(super) fn quote(
        self,
        logits: WorkspaceSamplingSource<'_>,
        steps: u64,
        context: &WorkspaceContext,
        observer: Option<&mut dyn SamplingWorkspaceObserver>,
    ) -> Result<SamplingWorkspaceReport, Error> {
        match self {
            Self::Configured(config, filter) => {
                let sampler = ConfiguredTextSampler::from_config(config)
                    .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
                // Equation quotation has finished its last report. The fresh
                // key descriptor belongs to sampling construction, before the
                // runtime worker quotes its complete initial state span. The
                // transition preserves cumulative metadata funding; the runtime
                // preparation report still prices actual key creation once.
                context.begin_span();
                let random = (config.sampling().temperature > 0.0)
                    .then(|| WorkspaceSamplingRandomState::from_seed(context))
                    .transpose()?;
                quote_sampling_workspace_with_observer(
                    &sampler,
                    config.sampling().temperature,
                    random.as_ref(),
                    logits,
                    filter,
                    steps,
                    context,
                    observer,
                )
            }
            Self::Borrowed(input) => quote_sampling_workspace_with_observer(
                input.sampler,
                input.temperature,
                input.random,
                logits,
                input.filter,
                steps,
                context,
                observer,
            ),
        }
    }
}

/// Actual score layout and complete backing union shared by text and media
/// equation drivers. No logical-row estimate substitutes for view backing.
pub(super) struct SamplingScores {
    layout: Option<eredu_nn::workspace::WorkspaceLayout>,
    capacity: Option<u64>,
    allocations: usize,
}
impl Default for SamplingScores {
    fn default() -> Self {
        Self {
            layout: None,
            capacity: Some(0),
            allocations: 0,
        }
    }
}
impl SamplingScores {
    pub(super) fn observe(
        &mut self,
        scores: Option<&eredu_nn::workspace::WorkspaceTensor>,
        uniform: bool,
        context: &WorkspaceContext,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceStoragePopulation>, Error> {
        self.observe_storage(scores, uniform, context, false)
    }
    /// Media sources may already own some score backing. Only the native
    /// carryover result excludes the exact selection; the sampler still gets
    /// the complete backing envelope required to consume those scores.
    pub(super) fn observe_prepared(
        &mut self,
        scores: Option<&eredu_nn::workspace::WorkspaceTensor>,
        uniform: bool,
        context: &WorkspaceContext,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceStoragePopulation>, Error> {
        self.observe_storage(scores, uniform, context, true)
    }
    fn observe_storage(
        &mut self,
        scores: Option<&eredu_nn::workspace::WorkspaceTensor>,
        uniform: bool,
        context: &WorkspaceContext,
        exclude_borrowed: bool,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceStoragePopulation>, Error> {
        let Some(scores) = scores else {
            return Ok(Some(eredu_nn::workspace::WorkspaceStoragePopulation::EMPTY));
        };
        if uniform
            && self
                .layout
                .as_ref()
                .is_some_and(|layout| layout != scores.layout())
        {
            return Err(super::workspace_message(
                context,
                format_args!("selected final-position score geometry changed during generation"),
            ));
        }
        let report = context.report_scalars(std::slice::from_ref(scores))?;
        self.allocations = self
            .allocations
            .max(report.closing_storage.maximum_allocations);
        if uniform {
            let backing = report.state.and_then(|state| state.retained_bytes);
            self.capacity = self
                .capacity
                .zip(backing)
                .map(|(prior, current)| prior.max(current));
            self.layout = Some(scores.layout().clone());
        }
        Ok(if exclude_borrowed {
            report.residual.map_or(Some(report.closing_storage), |residual| {
                residual.closing_storage
            })
        } else {
            Some(report.closing_storage)
        })
    }
    pub(super) fn finish(
        self,
        sampling: Option<TextSamplingInput<'_>>,
        steps: u64,
        context: &WorkspaceContext,
        observer: Option<&mut dyn SamplingWorkspaceObserver>,
    ) -> Result<Option<SamplingWorkspaceReport>, Error> {
        sampling
            .map(|sampling| {
                sampling.quote(
                    eredu_runtime::working_memory::WorkspaceSamplingInput {
                        layout: self.layout.as_ref().ok_or_else(|| {
                            super::workspace_message(
                                context,
                                format_args!("selected generation produced no score geometry"),
                            )
                        })?,
                        backing_capacity_bytes: self.capacity,
                    }
                    .with_backing_population(self.allocations),
                    steps,
                    context,
                    observer,
                )
            })
            .transpose()
    }
}

#[cfg(test)]
#[path = "sampling/tests.rs"]
mod tests;
