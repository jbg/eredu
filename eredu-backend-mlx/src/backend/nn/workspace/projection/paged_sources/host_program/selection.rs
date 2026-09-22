//! Select the submitted generation subset of the same complete quote trace.
use super::*;
use eredu_runtime::working_memory::{
    InferenceSpanWorkspacePlan, InferenceSpanWorkspaceRecord, InferenceWorkspaceSpan,
};

/// Descriptive access to the actual retained append row. Implementations lend
/// existing publication IDs; neither an ordinal nor a range issues a source.
pub(in super::super) trait HostAppendSource {
    fn source_index(&self) -> usize;
    fn ordinal(&self) -> usize;
    fn scan_coordinates(&self) -> Option<(i64, i64)>;
    fn publication(&self, range: &std::ops::Range<i64>) -> Option<&CacheBlockId>;
}
impl HostAppendSource for super::super::programs::PagedAppendProgram {
    fn source_index(&self) -> usize {
        self.source
    }
    fn ordinal(&self) -> usize {
        self.ordinal
    }
    fn scan_coordinates(&self) -> Option<(i64, i64)> {
        self.scan
            .as_ref()
            .map(|scan| (scan.query_start, scan.context_end))
    }
    fn publication(&self, range: &std::ops::Range<i64>) -> Option<&CacheBlockId> {
        self.publications
            .iter()
            .find(|row| row.id.start == range.start && row.id.end == range.end)
            .map(|row| &row.id)
    }
}

fn coordinates(record: &InferenceSpanWorkspaceRecord) -> Option<(i64, i64)> {
    let (start, length) = match record.span() {
        InferenceWorkspaceSpan::Sampling(_) => return None,
        InferenceWorkspaceSpan::Prefill(chunk) => (
            chunk.position,
            chunk.input.end.checked_sub(chunk.input.start)?,
        ),
        InferenceWorkspaceSpan::Decode { position, .. } => (*position, 1),
    };
    Some((
        i64::try_from(start).ok()?,
        i64::try_from(start.checked_add(length)?).ok()?,
    ))
}
pub(super) fn submitted(
    plan: &InferenceSpanWorkspacePlan,
    invocation: (i64, i64),
    context: &WorkspaceContext,
) -> Result<bool, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    let mut selected = plan.generation_records().ok_or_else(|| {
        fail(CacheSourceError::HostIdentity(
            "generation schedule availability",
        ))
    })?;
    let records = plan.records().iter();
    context
        .charge_metadata(
            size_of::<(
                &InferenceSpanWorkspacePlan,
                (i64, i64),
                &WorkspaceContext,
                Option<&InferenceSpanWorkspaceRecord>,
                Result<bool, CacheSourceFailure>,
                (u64, u64),
            )>()
            .checked_add(std::mem::size_of_val(&selected))
            .and_then(|n| n.checked_add(std::mem::size_of_val(&records)))
            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let mut actual = None;
    for record in records {
        if coordinates(record) == Some(invocation) {
            if actual.is_some() {
                return Err(fail(CacheSourceError::HostIdentity(
                    "unique diagnostic invocation",
                )));
            }
            actual = Some(record);
        }
    }
    let actual = actual.ok_or_else(|| {
        fail(CacheSourceError::HostIdentity(
            "retained diagnostic invocation",
        ))
    })?;
    // Compare the actual immutable record loan. No duplicate rule for which
    // final decode generation omits, and no shape/byte equality authorization.
    for record in &mut selected {
        if std::ptr::eq(record, actual) {
            return Ok(true);
        }
    }
    Ok(false)
}
pub(in super::super) fn program<'a, P: HostAppendSource>(
    source: usize,
    invocation: (i64, i64),
    programs: &'a [Option<P>],
    plan: &InferenceSpanWorkspacePlan,
    context: &WorkspaceContext,
) -> Result<Option<&'a P>, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    context
        .charge_metadata(size_of::<(
            usize,
            (i64, i64),
            &[Option<P>],
            &InferenceSpanWorkspacePlan,
            &WorkspaceContext,
            std::slice::Iter<'_, Option<P>>,
            Result<Option<&P>, CacheSourceFailure>,
        )>())
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    if !submitted(plan, invocation, context)? {
        return Ok(None);
    }
    for program in programs.iter().flatten() {
        if program.source_index() == source && program.scan_coordinates() == Some(invocation) {
            return Ok(Some(program));
        }
    }
    Err(fail(CacheSourceError::HostIdentity(
        "submitted scan program",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_nn::workspace::{WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound};
    #[derive(Debug)]
    struct NoEquations;
    impl WorkspaceMechanisms for NoEquations {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
            Ok(None)
        }
    }
    #[test]
    fn host_inventory_selects_actual_generation_and_rejects_unknown_invocations() {
        for chunk in [1, 2, 5] {
            for outputs in [1, 4] {
                let context = WorkspaceContext::new(NoEquations);
                let geometry = eredu_core::InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: 5,
                    max_output_tokens: outputs,
                    prefill_chunk_positions: chunk,
                    output: eredu_core::OutputDemand::LastPosition,
                };
                let report =
                    eredu_runtime::working_memory::quote_inference_workspace(geometry, |_| {
                        context.begin_state_span([])?;
                        context.report(&[])
                    })
                    .unwrap();
                let plan = report.span_workspace_plan();
                let final_start = i64::try_from(5 + outputs - 1).unwrap();
                assert!(!submitted(plan, (final_start, final_start + 1), &context).unwrap());
                assert_eq!(
                    plan.records().len(),
                    plan.generation_forward_count().unwrap() + 1
                );
                for record in plan.generation_records().unwrap() {
                    let actual = coordinates(record).unwrap();
                    assert!(submitted(plan, actual, &context).unwrap());
                    // A selected span still needs its actual native program.
                    assert!(
                        program::<super::super::super::programs::PagedAppendProgram>(
                            0,
                            actual,
                            &[],
                            plan,
                            &context
                        )
                        .is_err()
                    );
                }
                assert!(
                    program::<super::super::super::programs::PagedAppendProgram>(
                        0,
                        (final_start, final_start + 1),
                        &[],
                        plan,
                        &context
                    )
                    .unwrap()
                    .is_none()
                );
                assert!(submitted(plan, (final_start + 1, final_start + 2), &context).is_err());
            }
        }
    }
}
