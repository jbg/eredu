//! Full-prompt storage preceding bounded text execution.

use eredu_core::{CapabilityError, ExecutionWorkspaceEstimate, InferenceGeometry, WorkspaceBound};
use eredu_nn::{Error, workspace::*};

/// Native prompt storage plus its source host payload. Unlike an equation span,
/// preparation retains the full token input across all prefill chunks. This
/// report includes construction and retention of its exact text cache identity,
/// but excludes tokenizer/application strings.
#[derive(Debug, Clone)]
pub struct TextPromptWorkspaceReport {
    geometry: InferenceGeometry,
    peak: WorkspaceBound,
    tensor_peak_bytes: Option<u64>,
    host_peak_bytes: Option<u64>,
}

impl TextPromptWorkspaceReport {
    /// Exact request whose complete input was priced.
    pub const fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    /// Complete simultaneously live input preparation payload, or a gap.
    pub const fn peak(&self) -> &WorkspaceBound {
        &self.peak
    }
    /// Native buffer domain, independently comparable with allocator telemetry.
    pub const fn tensor_peak_bytes(&self) -> Option<u64> {
        self.tensor_peak_bytes
    }
    /// Legacy caller capacity (zero when source bytes are separately sealed in I),
    /// plus unchanged host staging and text identity construction.
    pub const fn host_peak_bytes(&self) -> Option<u64> {
        self.host_peak_bytes
    }

    /// Adds this preparation report to the enclosing workspace. Original input I
    /// is sealed separately in protected host custody; the legacy report includes
    /// caller capacity here. Equation/sampling overlap and unknown bounds remain.
    pub fn compose(
        &self,
        outside: ExecutionWorkspaceEstimate,
    ) -> Result<ExecutionWorkspaceEstimate, CapabilityError> {
        self.compose_metadata(outside, super::WorkspaceReportMetadata::ordinary())
            .map_err(super::WorkspaceReportError::into_capability)
    }

    /// The same prompt composition with counted owning report destinations.
    pub fn compose_metadata(
        &self,
        mut outside: ExecutionWorkspaceEstimate,
        metadata: super::WorkspaceReportMetadata<'_>,
    ) -> Result<ExecutionWorkspaceEstimate, super::WorkspaceReportError> {
        metadata.admit::<ExecutionWorkspaceEstimate>()?;
        if outside.geometry != self.geometry {
            return Err(eredu_core::AdmissionPolicyError::InvalidConfiguration {
                field: "text_prompt_workspace",
                detail: "prepared input quote differs from the admitted request geometry",
            }
            .into());
        }
        outside.activations = match (&outside.activations, &self.peak) {
            (
                WorkspaceBound::Bounded { bytes, assumptions },
                WorkspaceBound::Bounded {
                    bytes: input,
                    assumptions: input_assumptions,
                },
            ) => metadata.bounded(
                bytes.checked_add(*input).ok_or(
                    eredu_core::AdmissionPolicyError::ArithmeticOverflow {
                        operation: "prompt and enclosing workspace",
                    },
                )?,
                format_args!("{assumptions}; {input_assumptions}"),
            )?,
            (WorkspaceBound::Unknown { .. }, _) => outside.activations,
            (_, unknown @ WorkspaceBound::Unknown { .. }) => metadata.clone_bound(unknown)?,
        };
        Ok(outside)
    }
}

/// Quotes copying a complete contiguous host U32 prompt into native storage.
/// `source_capacity_bytes` is the actual owned host allocation capacity, not
/// just the token slice's length. Unknown backing stays unknown. The caller
/// supplies the same geometry to architecture, sampling and final admission.
///
/// This starts a new metadata span. It allocates no native input and does not
/// inspect token values, reconstruct tokenizer state, or prepare media. Native
/// facts for eager final-shape initialization must include all copies and staging.
pub fn quote_text_prompt_workspace(
    geometry: InferenceGeometry,
    source_capacity_bytes: Option<u64>,
    context: &WorkspaceContext,
) -> Result<TextPromptWorkspaceReport, Error> {
    quote_prompt(geometry, source_capacity_bytes, false, context)
}

/// Native upload/identity report paired with the same sealed original input I.
/// The source bytes are in I, never caller capacity or a second activation term.
pub fn quote_original_token_prompt_workspace(
    geometry: InferenceGeometry,
    input: &super::OriginalTokenInputLayout,
    context: &WorkspaceContext,
) -> Result<TextPromptWorkspaceReport, Error> {
    if geometry.batch_size != 1 || geometry.input_positions != input.token_count() as u64 {
        return Err(context.metadata_error(format_args!("original token input geometry mismatch")));
    }
    quote_prompt(geometry, Some(0), true, context)
}
// Fixed ownership beside the already-priced text identity payload. The private
// text worker initializes its mutex before publishing aliases, so there is one
// PAL allocation rather than an unbounded concurrent first-lock population.
fn text_identity_control_bytes() -> Option<u64> {
    let (inner, identity, controls) =
        crate::input::SharedPreparedInputCacheIdentity::text_control_request()?;
    let shared = super::qualified_storage::shared_layout_bytes(inner)
        .ok()?
        .checked_sub(std::mem::size_of::<crate::input::PreparedInputCacheIdentity>() as u64)?;
    shared
        .checked_add(super::qualified_storage::shared_layout_bytes(identity).ok()?)?
        .checked_add(super::fixed_baseline::pal_mutex_bytes()?)?
        .checked_add(u64::try_from(controls).ok()?)?
        .checked_add(std::mem::size_of::<crate::input::TextInputIdentityPlan>() as u64)?
        .checked_add(std::mem::size_of::<crate::input::BoundTextInputIdentityPlan<'_>>() as u64)?
        .checked_add(std::mem::size_of::<crate::input::TextInputIdentityError>() as u64)
}
fn quote_prompt(
    geometry: InferenceGeometry,
    source_capacity_bytes: Option<u64>,
    original: bool,
    context: &WorkspaceContext,
) -> Result<TextPromptWorkspaceReport, Error> {
    if context.uses_checked_metadata() {
        geometry
            .validate_fixed()
            .map_err(|cause| context.metadata_source(cause))?;
    } else {
        geometry.validate().map_err(Error::backend)?;
    }
    let metadata = super::WorkspaceReportMetadata::new(context);
    metadata
        .admit::<TextPromptWorkspaceReport>()
        .map_err(|cause| metadata.error(cause))?;
    let count = geometry
        .batch_size
        .checked_mul(geometry.input_positions)
        .ok_or_else(|| context.metadata_error(format_args!("prompt token count overflow")))?;
    let logical = count
        .checked_mul(4)
        .ok_or_else(|| context.metadata_error(format_args!("prompt byte count overflow")))?;
    if !original && source_capacity_bytes.is_some_and(|bytes| bytes < logical) {
        return Err(context.metadata_error(format_args!(
            "owned prompt capacity is smaller than its token payload"
        )));
    }
    i32::try_from(count)
        .map_err(|_| context.metadata_error(format_args!("flat prompt exceeds tensor extent")))?;
    let batch =
        i32::try_from(geometry.batch_size).map_err(|cause| context.metadata_source(cause))?;
    let positions =
        i32::try_from(geometry.input_positions).map_err(|cause| context.metadata_source(cause))?;
    metadata
        .admit::<crate::input::TextInputIdentityPlan>()
        .map_err(|cause| metadata.error(cause))?;
    metadata
        .admit::<crate::input::TextInputIdentityError>()
        .map_err(|cause| metadata.error(cause))?;
    let identity =
        crate::input::TextInputIdentityPlan::new(geometry.batch_size, geometry.input_positions)
            .map_err(|cause| context.metadata_source(cause))?;
    context.begin_state_span([])?;
    // Original I is authenticated by quote_original_token_prompt_workspace;
    // its independently admitted native input producer supplies this complete
    // integer matrix. Retain that source distinction in the enclosing quote,
    // just as the shared model driver does for the exact input ordinal.
    let _tokens = if original {
        Some(WorkspaceTensor::prepared_token_input(&[batch,positions],WorkspaceDtype::Uint32,context)?)
    } else {
        let mut outputs = context.metadata_vec(1)?;
        outputs.push(context.layout(&[batch, positions], WorkspaceDtype::Uint32)?);
        let _tokens = context.execute(WorkspaceOperationKind::Initialize, &[], outputs)?.remove(0);
        None
    };
    let trace = if context.uses_checked_metadata() {
        context.finish_report(&[])?
    } else {
        context.report(&[])?
    };
    let tensor_peak_bytes = trace.tensor_buffers.total_bytes;
    let host_peak_bytes = source_capacity_bytes
        .zip(trace.host_workspace_bytes)
        .zip(text_identity_control_bytes())
        .map(|((source, staging), identity_controls)| {
            source
                .checked_add(staging)
                .and_then(|bytes| bytes.checked_add(identity.peak_bytes()))
                .and_then(|bytes| bytes.checked_add(identity_controls))
                .ok_or_else(|| {
                    context.metadata_error(format_args!("prompt host workspace overflow"))
                })
        })
        .transpose()?;
    let peak = match tensor_peak_bytes.zip(host_peak_bytes) {
        Some((tensor, host)) => WorkspaceBound::bounded(
            tensor.checked_add(host).ok_or_else(|| context.metadata_error(format_args!("prompt workspace overflow")))?,
            context.metadata_string(format_args!("host source mode original_input={original}; legacy host U32 capacity or separately sealed original I, and native input retained across all chunks; input/shape copies, host staging and closed text identity construction/retention including qualified shared headers and pre-share mutex storage included, excluding tokenizer/application strings; {}", PromptAssumptions(&trace.assumptions)))?,
        ),
        None => WorkspaceBound::Unknown { reason: context.metadata_string(format_args!("full prompt backing, native initialization, identity owner or host staging has no complete bound"))? },
    };
    Ok(TextPromptWorkspaceReport {
        geometry,
        peak,
        tensor_peak_bytes,
        host_peak_bytes,
    })
}

#[cfg(test)]
mod tests;

struct PromptAssumptions<'a>(&'a [String]);
impl std::fmt::Display for PromptAssumptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, text) in self.0.iter().enumerate() {
            if index != 0 {
                f.write_str("; ")?;
            }
            f.write_str(text)?;
        }
        Ok(())
    }
}
