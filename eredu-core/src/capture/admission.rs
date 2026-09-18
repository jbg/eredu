//! Shared capture declaration admission and its concrete source producers.
use super::*;
pub(crate) mod allocation;
use allocation::Allocation;
pub use allocation::CaptureAdmissionStorageError;

pub(super) fn admit_geometry(
    plan: CapturePlan,
    catalog: &ObservationCatalog,
    support: &ObservationSupportReport,
    capabilities: &CaptureCapabilities,
    request: CaptureRequestShape,
    invocation_bounds: Option<CaptureInvocationBounds>,
    text_origin: CaptureTextOrigin,
    allocation: Allocation<'_>,
) -> Result<AdmittedCapturePlan, CaptureError> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<(
            CapturePlan,
            &ObservationCatalog,
            &ObservationSupportReport,
            &CaptureCapabilities,
            CaptureRequestShape,
            Option<CaptureInvocationBounds>,
            CaptureTextOrigin,
            Allocation<'_>,
        )>(),
        size_of::<Result<AdmittedCapturePlan, CaptureError>>(),
        size_of::<(&CaptureSelection, &ObservationPoint)>(),
        size_of::<Vec<ObservationPoint>>(),
        size_of::<Vec<(&str, usize)>>(),
        size_of::<(CapturePhase, u64, u64, u64, CaptureInvocationShape)>(),
        size_of::<ResolvedCaptureSlice>(),
        size_of::<super::plan_copy::Worker>(),
        size_of::<ObservationPoint>(),
        crate::HostMetadataFunding::reservation_control_bytes(),
    ];
    allocation.reserve(
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(CaptureError::Overflow)?,
    )?;
    if let Some(bounds) = invocation_bounds {
        bounds
            .maximum_fixed()
            .map_err(|cause| cause.legacy_parts_with(None, allocation))?;
    }
    revalidation::validate_schema_with(&plan, catalog, support, capabilities, allocation)?;
    if request.batch == 0 || request.prompt_tokens == 0 || request.max_predictions == 0 {
        return Err(CaptureError::Invalid(
            allocation.text("batch, prompt, and prediction limits must be positive")?,
        ));
    }
    if invocation_bounds.is_none() {
        // Preserve the legacy zero-origin request check. The additional
        // origin span uses the last actual prediction, not a phantom decode.
        add(request.prompt_tokens, request.max_predictions)?;
        text_origin.validate_request(request)?;
    } else if text_origin != CaptureTextOrigin::default() {
        return Err(CaptureError::Invalid(allocation.text(
            "ordinary text origin cannot replace invocation authority",
        )?));
    }
    mul(request.batch, request.prompt_tokens)?;
    let ids = ordered_keys(
        allocation,
        plan.selections
            .iter()
            .map(|selection| selection.id.as_str()),
    )?;
    let mut points = allocation.vector(plan.selections.len())?;
    for (selection_index, selection) in plan.selections.iter().enumerate() {
        if selection.id.is_empty() || repeated_key(&ids, &selection.id, selection_index) {
            return Err(CaptureError::Invalid(
                allocation.text("capture IDs must be nonempty and unique")?,
            ));
        }
        let point = match catalog.get(&selection.path) {
            Some(point) => point,
            None => return Err(CaptureError::MissingPath(allocation.text(&selection.path)?)),
        };
        let sparse = matches!(
            point.value_type,
            crate::ObservationValueType::RoutedUnits { .. }
        );
        if sparse != matches!(selection.transform, CaptureTransform::RoutedUnits) {
            return Err(CaptureError::Unsupported(allocation.text(
                "routed-unit boundaries require the routed_units transform",
            )?));
        }
        if let crate::ObservationValueType::RoutedUnits { geometry, .. } = &point.value_type {
            geometry.components_with(allocation)?;
        }
        revalidation::validate_transform_with(selection, capabilities, allocation)?;
        if selection.schedule.every == 0
            || selection
                .schedule
                .end_prediction
                .is_some_and(|end| end <= selection.schedule.first_prediction)
        {
            return Err(CaptureError::Invalid(
                allocation.text("invalid capture schedule")?,
            ));
        }
        revalidation::validate_phase_support_with(selection, support, allocation)?;
        if matches!(selection.transform, CaptureTransform::Slice) && selection.slices.is_empty() {
            return Err(CaptureError::Invalid(allocation.text(
                "slice capture requires an explicit axis slice; use FullTensor to opt in",
            )?));
        }
        revalidation::validate_histogram_with(selection, capabilities, allocation)?;
        if let CaptureTransform::TopCandidates { count } = selection.transform {
            if count == 0
                || selection.path != crate::MODEL_LOGITS_OBSERVATION_PATH
                || !selection.slices.is_empty()
            {
                return Err(CaptureError::Invalid(allocation.text("candidate capture requires positive count, unsliced model.logits, and single-sequence execution")?));
            }
            if request.batch != 1 {
                return Err(CaptureError::Unsupported(
                    allocation.text("candidate capture requires batch one")?,
                ));
            }
            if let Some(SymbolicDimension::Known(vocabulary)) = point
                .axes
                .as_ref()
                .and_then(|axes| axes.last())
                .map(|a| &a.dimension)
            {
                if count > *vocabulary as u64 {
                    return Err(CaptureError::Invalid(
                        allocation.text("candidate count exceeds vocabulary")?,
                    ));
                }
            }
        }
        if let CaptureTransform::TokenScores { token_ids } = &selection.transform {
            if token_ids.is_empty()
                || token_ids.len() > 64
                || repeated_score_id(token_ids, allocation)?
                || selection.path != crate::MODEL_LOGITS_OBSERVATION_PATH
                || !selection.slices.is_empty()
            {
                return Err(CaptureError::Invalid(allocation.text(
                    "token scoring requires 1..=64 unique IDs and unsliced model.logits",
                )?));
            }
            if request.batch != 1 {
                return Err(CaptureError::Unsupported(
                    allocation.text("token scoring requires batch one")?,
                ));
            }
            if let Some(SymbolicDimension::Known(vocabulary)) = point
                .axes
                .as_ref()
                .and_then(|axes| axes.last())
                .map(|axis| &axis.dimension)
            {
                if token_ids.iter().any(|id| *id as usize >= *vocabulary) {
                    return Err(CaptureError::Invalid(
                        allocation.text("selected score ID exceeds model vocabulary")?,
                    ));
                }
            }
        }
        let axes = ordered_keys(
            allocation,
            selection.slices.iter().map(|slice| slice.axis.as_str()),
        )?;
        for (slice_index, slice) in selection.slices.iter().enumerate() {
            if slice.stride == 0
                || slice.start > slice.end
                || repeated_key(&axes, &slice.axis, slice_index)
            {
                return Err(CaptureError::Invalid(
                    allocation.text("invalid or duplicate axis slice")?,
                ));
            }
            if !point
                .axes
                .as_ref()
                .is_some_and(|axes| axes.iter().any(|axis| axis.name == slice.axis))
            {
                return Err(CaptureError::Invalid(
                    allocation.format(format_args!("unknown axis {}", slice.axis))?,
                ));
            }
        }
        // Resolve every known shape before execution, and defer only genuinely
        // runtime-dependent dimensions. Unknown never becomes a zero extent.
        for phase in [CapturePhase::Prefill, CapturePhase::Decode] {
            let range = if invocation_bounds.is_some() {
                selection
                    .schedule
                    .count_coordinates(phase, 0, request.max_predictions)?
            } else {
                selection
                    .schedule
                    .count_and_last(phase, request.max_predictions)?
            };
            if let Some((count, last)) = range {
                let first = last - mul(count - 1, selection.schedule.every)?;
                for prediction in [first, last] {
                    for slice in &selection.slices {
                        let axis = point
                            .axes
                            .as_ref()
                            .and_then(|axes| axes.iter().find(|axis| axis.name == slice.axis))
                            .expect("axis was validated");
                        let geometry = match invocation_bounds {
                            Some(bounds) => bounds
                                .maximum_fixed()
                                .map_err(|cause| cause.legacy_parts_with(None, allocation))?,
                            None => text_origin.invocation_shape(request, phase, prediction)?,
                        };
                        if geometry
                            .extent_fixed(&axis.dimension)
                            .map_err(|cause| cause.legacy_parts_with(None, allocation))?
                            .is_some_and(|extent| slice.end > extent)
                        {
                            return Err(CaptureError::Invalid(allocation.format(format_args!(
                                "slice {} exceeds known request extent",
                                slice.axis
                            ))?));
                        }
                    }
                    let geometry = match invocation_bounds {
                        Some(bounds) => bounds
                            .maximum_fixed()
                            .map_err(|cause| cause.legacy_parts_with(None, allocation))?,
                        None => text_origin.invocation_shape(request, phase, prediction)?,
                    };
                    if let Some(shape) = geometry.resolve_with(point, allocation)? {
                        resolve_slice_with(point, selection, &shape, allocation)?;
                    }
                }
            }
        }
        points.push(allocation.point(point)?);
    }
    // Identity includes catalog semantics and request shape, not just caller labels.
    allocation.reserve(identity::control_bytes().ok_or(CaptureError::Overflow)?)?;
    let digest = match identity::digest(&plan, &points, request, invocation_bounds, text_origin) {
        Ok(digest) => digest,
        Err(cause) => {
            return Err(CaptureError::Invalid(
                allocation.format(format_args!("{cause}"))?,
            ));
        }
    };
    let identity = allocation.text(std::str::from_utf8(&digest).expect("hex capture identity"))?;
    drop(ids);
    Ok(AdmittedCapturePlan {
        plan,
        points,
        request,
        invocation_bounds,
        text_origin,
        identity,
    })
}

// Sort borrowed keys once. Keeping original indices preserves the first duplicate
// reported by the existing source-order validation loop.
fn ordered_keys<'a>(
    allocation: Allocation<'_>,
    keys: impl ExactSizeIterator<Item = &'a str>,
) -> Result<Vec<(&'a str, usize)>, CaptureError> {
    let mut rows = allocation.vector(keys.len())?;
    rows.extend(keys.enumerate().map(|(index, key)| (key, index)));
    sort(allocation, &mut rows)?;
    Ok(rows)
}
fn repeated_key(rows: &[(&str, usize)], key: &str, index: usize) -> bool {
    let first = rows.partition_point(|(value, _)| *value < key);
    rows[first].1 < index
}
fn repeated_score_id(ids: &[u32], allocation: Allocation<'_>) -> Result<bool, CaptureError> {
    // The caller checks the published 64-ID score bound first.
    allocation.reserve(std::mem::size_of::<[u32; 64]>() + std::mem::size_of::<&[u32]>())?;
    let mut sorted = [0; 64];
    sorted[..ids.len()].copy_from_slice(ids);
    sort(allocation, &mut sorted[..ids.len()])?;
    Ok(sorted[..ids.len()]
        .windows(2)
        .any(|pair| pair[0] == pair[1]))
}

// Iterative heap sort has fixed control storage for every declaration length.
// Both admission policies use it; source order remains in each key's index.
fn sort<T: Ord>(allocation: Allocation<'_>, rows: &mut [T]) -> Result<(), CaptureError> {
    allocation.reserve(std::mem::size_of::<(&mut [T], [usize; 6], bool)>())?;
    fn sift<T: Ord>(rows: &mut [T], mut root: usize, end: usize) {
        while root < end / 2 {
            let mut child = root * 2 + 1;
            if child + 1 < end && rows[child] < rows[child + 1] {
                child += 1;
            }
            if rows[root] >= rows[child] {
                break;
            }
            rows.swap(root, child);
            root = child;
        }
    }
    for start in (0..rows.len() / 2).rev() {
        sift(rows, start, rows.len());
    }
    for end in (1..rows.len()).rev() {
        rows.swap(0, end);
        sift(rows, 0, end);
    }
    Ok(())
}
