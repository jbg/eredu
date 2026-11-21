//! Portable pricing of capture geometry and records using native allocation facts.

use super::*;
use std::num::NonZeroU64;

/// Side-effect-free limits and allocation bounds for native capture transforms.
#[derive(Debug, Clone, Copy)]
pub struct CaptureWorkspaceModel {
    /// Maximum addressable native dimension or linear index.
    pub max_index: u64,
    /// Number of elements reduced in one native scratch chunk.
    pub reduction_chunk: NonZeroU64,
    /// Native retained input storage per source element.
    pub source_bytes_per_element: u64,
    /// Native retained selection storage per selected element.
    pub selected_bytes_per_element: u64,
    /// Fixed native workspace allowance per transform.
    pub workspace_base_bytes: u64,
    /// Native scratch allowance per element in the admitted temporary.
    pub workspace_bytes_per_element: u64,
    /// Optional bounds for compact routed-unit capture.
    pub routed: Option<RoutedCaptureWorkspace>,
}

/// Native storage facts for the compact routed-unit transform.
#[derive(Debug, Clone, Copy)]
pub struct RoutedCaptureWorkspace {
    /// Fixed device-side workspace.
    pub base_bytes: u64,
    /// Source and conversion storage per element.
    pub source_bytes_per_element: u64,
    /// Index and routing storage per candidate route.
    pub bytes_per_route: u64,
}

impl CaptureWorkspaceModel {
    /// Prices the logical storage and encoded record for the supplied geometry.
    pub fn estimate(
        &self,
        source: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        if matches!(selection.transform, CaptureTransform::RoutedUnits) {
            return self.estimate_routed(source, slice, false);
        }
        let source_elements = elements(source)?;
        let selected = match selection.transform {
            CaptureTransform::TopCandidates { .. } | CaptureTransform::TokenScores { .. } => {
                source.last().copied().unwrap_or(0)
            }
            _ => elements(&slice.shape)?,
        };
        if source_elements > self.max_index
            || source
                .iter()
                .chain(slice.strides.iter())
                .any(|n| *n > self.max_index)
        {
            return Err(CaptureError::Unsupported(
                "capture shape/index exceeds native range".into(),
            ));
        }
        let chunks = selected.div_ceil(self.reduction_chunk.get());
        let (host_bytes, encoded_bytes, temporary_elements) = match &selection.transform {
            CaptureTransform::RoutedUnits => unreachable!("handled before dense estimation"),
            CaptureTransform::TokenScores { token_ids, .. } => {
                if token_ids.is_empty()
                    || token_ids.len() > 64
                    || source
                        .last()
                        .is_none_or(|width| token_ids.iter().any(|id| u64::from(*id) >= *width))
                {
                    return Err(CaptureError::Invalid(
                        "selected score IDs exceed the runtime vocabulary or count limit".into(),
                    ));
                }
                let count = token_ids.len() as u64;
                (
                    add(256, add(mul(count, 128)?, mul(chunks, 8)?)?)?,
                    add(1024, mul(count, 512)?)?,
                    selected.min(self.reduction_chunk.get()),
                )
            }
            CaptureTransform::TopCandidates { count } => {
                if *count == 0 || source.last().is_none_or(|vocabulary| count > vocabulary) {
                    return Err(CaptureError::Invalid(
                        "candidate count exceeds the runtime vocabulary".into(),
                    ));
                }
                (
                    add(80, mul(*count, 20)?)?,
                    add(384, mul(*count, 80)?)?,
                    *count,
                )
            }
            CaptureTransform::Preview { max_elements } => {
                let n = selected.min(*max_elements);
                (mul(n, 16)?, add(128, mul(n, 32)?)?, n)
            }
            CaptureTransform::Slice | CaptureTransform::FullTensor => {
                (mul(selected, 16)?, add(128, mul(selected, 32)?)?, selected)
            }
            CaptureTransform::Summary => (
                add(128, mul(chunks, 128)?)?,
                512,
                selected.min(self.reduction_chunk.get()),
            ),
            CaptureTransform::Histogram { edges } => {
                let bins = edges.len().saturating_sub(1) as u64;
                if bins == 0 || bins > 128 {
                    return Err(CaptureError::Unsupported("histogram bin count".into()));
                }
                (
                    add(mul(bins, 16)?, mul(chunks, mul(add(bins, 3)?, 8)?)?)?,
                    add(256, mul(add(bins, 1)?, 64)?)?,
                    selected.min(self.reduction_chunk.get()),
                )
            }
        };
        // Candidate capture touches only the selected logits row: its temporaries
        // (dtype copy, finite mask, argsort indices) are all row-sized, and the
        // source backing is an inference allocation that exists regardless of
        // capture. Charging the whole source would veto prefill candidate capture
        // — where the source is [batch, prompt, vocabulary] — for any real prompt.
        let (retained_source, retained_selected) = match &selection.transform {
            CaptureTransform::TopCandidates { .. } | CaptureTransform::TokenScores { .. } => {
                let vocabulary = source.last().copied().unwrap_or(0);
                (vocabulary, vocabulary)
            }
            _ => (source_elements, selected),
        };
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: add(
                add(
                    mul(retained_source, self.source_bytes_per_element)?,
                    mul(retained_selected, self.selected_bytes_per_element)?,
                )?,
                add(
                    self.workspace_base_bytes,
                    mul(temporary_elements, self.workspace_bytes_per_element)?,
                )?,
            )?,
            host_bytes,
            encoded_bytes,
        })
    }

    /// Validates and prices an exact partitioned routed capture fragment.
    pub fn estimate_partition_routed(
        &self,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        request.validate()?;
        self.estimate_routed(&request.native_shape()?, request.slice, true)
    }

    /// Prices compact routed geometry, including partition fragments with no local columns.
    pub fn estimate_routed(
        &self,
        shape: &[u64],
        slice: &ResolvedCaptureSlice,
        partitioned: bool,
    ) -> Result<CaptureUsage, CaptureError> {
        let workspace = self
            .routed
            .ok_or_else(|| CaptureError::Unsupported("routed-unit collector unavailable".into()))?;
        if !partitioned && slice.strides.iter().any(|n| *n > self.max_index) {
            return Err(CaptureError::Unsupported(
                "routed-unit stride exceeds native range".into(),
            ));
        }
        if shape.len() != 3
            || slice.shape.len() != 3
            || shape.iter().any(|n| *n > self.max_index)
            || elements(shape)? > self.max_index
        {
            return Err(CaptureError::Unsupported(
                "routed-unit geometry exceeds native range".into(),
            ));
        }
        let all_routes = mul(shape[0], shape[1])?;
        let routes = if partitioned && slice.shape[2] == 0 {
            0
        } else {
            mul(slice.shape[0], slice.shape[1])?
        };
        let values = mul(routes, slice.shape[2])?;
        if routes == 0 {
            return Ok(CaptureUsage {
                captures: 1,
                retained_bytes: 0,
                host_bytes: add(4096, mul(shape[0], 32)?)?,
                encoded_bytes: add(4096, mul(shape[0], 64)?)?,
            });
        }
        // Full invocation source and conversion backing, bounded route-index arrays,
        // every row envelope and duplicate-detection/sorting storage. Charged once
        // before the first chunk, including when the selected rows occur much later.
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: add(
                workspace.base_bytes,
                add(
                    mul(elements(shape)?, workspace.source_bytes_per_element)?,
                    mul(all_routes, workspace.bytes_per_route)?,
                )?,
            )?,
            host_bytes: add(
                4096,
                add(
                    mul(all_routes, 128)?,
                    add(mul(routes, 512)?, mul(values, 16)?)?,
                )?,
            )?,
            // Native chunk evidence covers the whole invocation, even when the
            // exported token selection is tiny or occurs only in its last chunk.
            encoded_bytes: add(
                4096,
                add(
                    mul(shape[0], 64)?,
                    add(mul(routes, 768)?, mul(values, 32)?)?,
                )?,
            )?,
        })
    }
}
