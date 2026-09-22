//! Source-labelled incompleteness retained by the same ordinary caller census.
use super::*;

#[derive(Debug, Default)]
pub(crate) struct OrdinaryTraceCallControls {
    pub(crate) controls: Option<OrdinaryCallControls>,
    pub(crate) parallel: Option<super::super::super::parallel::OrdinaryParallelControls>,
    pub(crate) failure: Option<OrdinaryCallFailure>,
    pub(crate) paged_source: bool,
}

#[derive(Debug, thiserror::Error)]
#[error("ordinary native caller attribution is incomplete: {description}")]
pub(crate) struct OrdinaryCallFailure {
    description: String,
    #[source]
    cause: eredu_runtime::working_memory::WorkingMemoryError,
}

impl OrdinaryTraceCallControls {
    pub(super) fn complete(
        controls: OrdinaryCallControls,
        parallel: super::super::super::parallel::OrdinaryParallelControls,
        paged_source: bool,
    ) -> Self {
        Self {
            controls: Some(controls),
            parallel: Some(parallel),
            failure: None,
            paged_source,
        }
    }
}

impl ResidentRecipeRecorder {
    pub(super) fn missing_ordinary_caller(
        &self,
        producer: &'static str,
        report: &WorkspaceTraceReport,
        operation: Option<usize>,
    ) -> Result<OrdinaryTraceCallControls, Error> {
        self.missing_ordinary_caller_detail(format_args!("{producer}"), report, operation)
    }
    pub(super) fn missing_ordinary_caller_detail(
        &self,
        producer: std::fmt::Arguments<'_>,
        report: &WorkspaceTraceReport,
        operation: Option<usize>,
    ) -> Result<OrdinaryTraceCallControls, Error> {
        let operation =
            operation.and_then(|index| report.operations.get(index).map(|op| (index, op)));
        let description = match (&self.context, operation) {
            (Some(context), Some((index, op))) => context
                .metadata_string(format_args!("{producer}; operation {index}: {:?}", op.kind,))?,
            (Some(context), None) => context.metadata_string(format_args!("{producer}"))?,
            (None, Some((index, op))) => format!("{producer}; operation {index}: {:?}", op.kind),
            (None, None) => producer.to_string(),
        };
        Ok(OrdinaryTraceCallControls {
            controls: None,
            parallel: None,
            failure: Some(OrdinaryCallFailure {
                description,
                cause: eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            }),
            paged_source: false,
        })
    }
}

impl ResidentNativeRecipe {
    /// Takes the first missing caller source in retained execution order. The
    /// caller keeps this diagnostic under the same planning metadata custody.
    pub(crate) fn take_ordinary_call_failure(&mut self) -> Option<OrdinaryCallFailure> {
        for row in &mut self.records {
            if row.ordinary_call_failure.is_some() {
                return row.ordinary_call_failure.take();
            }
        }
        self.sampling
            .rows
            .iter_mut()
            .find_map(|row| row.ordinary_call_failure.take())
    }
}
