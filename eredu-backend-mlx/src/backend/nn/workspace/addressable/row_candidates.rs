//! Finite receive-row classes for the existing indexed and grouped workers.
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceExpertKernel, WorkspaceMetadataAllocation, WorkspaceMetadataError,
};
use eredu_nn::Error;
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy)]
struct TailBand {
    start: usize,
    lower: usize,
    upper: usize,
}

/// For a tail r in 1..C, the largest admitted full-chunk count is
/// floor((N-r)/C). It changes only at N%C, yielding at most two bands.
fn tail_bands(maximum: usize, chunk_rows: usize) -> [Option<TailBand>; 2] {
    let full = maximum / chunk_rows;
    let tail = maximum % chunk_rows;
    [
        (tail != 0).then_some(TailBand {
            start: full * chunk_rows,
            lower: 1,
            upper: tail,
        }),
        if full != 0 && tail < chunk_rows - 1 {
            Some(TailBand {
                start: (full - 1) * chunk_rows,
                lower: tail + 1,
                upper: chunk_rows - 1,
            })
        } else {
            None
        },
    ]
}

fn candidates(
    kernel: WorkspaceExpertKernel<'_>,
    maximum: usize,
    chunk_rows: usize,
    bands: [Option<TailBand>; 2],
) -> impl Iterator<Item = usize> + '_ {
    [1, maximum, (maximum / chunk_rows) * chunk_rows]
        .into_iter()
        .filter(move |rows| *rows != 0 && *rows <= maximum)
        .chain(bands.into_iter().flatten().flat_map(move |band| {
            super::super::grouped::expert_row_candidates(kernel, band.upper)
                .filter(move |tail| *tail >= band.lower)
                // By construction, start + upper <= maximum, including at
                // usize::MAX. This sum cannot overflow.
                .map(move |tail| band.start + tail)
        }))
}

/// Each band adds the same number of full indexed chunks. Within that band,
/// reuse the grouped worker's proven singleton, unchunked and remainder classes.
/// Adding full chunks adds nonnegative retained buffers, controls, constructor
/// calls and capture publications; parent joins grow with total rows. Exact
/// full-chunk and singleton invocations are retained separately. Every returned
/// candidate still passes through the ordinary numerical/source quote worker.
///
/// Storage is bounded by two grouped candidate populations plus three scalar
/// cases; it does not grow with the prompt or selected outer chunk width.
pub(super) fn local_row_candidates(
    kernel: WorkspaceExpertKernel<'_>,
    maximum: usize,
    chunk_rows: usize,
    context: &WorkspaceContext,
) -> Result<Vec<usize>, Error> {
    let frames = [
        size_of::<(WorkspaceExpertKernel<'_>, usize, usize, &WorkspaceContext)>(),
        size_of::<[Option<TailBand>; 2]>(),
        size_of::<Vec<usize>>(),
        size_of::<Result<Vec<usize>, Error>>(),
        size_of::<[usize; 4]>(),
        size_of::<Option<usize>>(),
        size_of::<std::slice::Iter<'_, usize>>(),
    ];
    context.charge_metadata(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    if chunk_rows == 0 {
        return Err(WorkspaceMetadataError::Unqualified.into());
    }
    let bands = tail_bands(maximum, chunk_rows);
    let population = candidates(kernel, maximum, chunk_rows, bands);
    context.charge_metadata(
        size_of_val(&population)
            .checked_mul(2)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    let mut values = context.metadata_vec(population.count())?;
    for rows in candidates(kernel, maximum, chunk_rows, bands) {
        let index = values
            .iter()
            .position(|prior| *prior >= rows)
            .unwrap_or(values.len());
        if values.get(index) != Some(&rows) {
            // The buffer already covers every yielded value before deduplication.
            values.insert(index, rows);
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::nn::grouped::{
        GROUPED_PROJECTION_CHUNK_THRESHOLD as THRESHOLD, GROUPED_PROJECTION_CHUNK_TOKENS as CHUNK,
    };
    use eredu_checkpoint::LinearFormat;
    use eredu_nn::{
        GatedProductGroupLayout, GatedProductPolicy, GroupedGatedProductSpec,
        GroupedLinearActivation, GroupedLinearSpec, GroupedProjectionSpec, LinearFormatSpec,
        ParameterSpec,
    };

    fn projection(name: &str) -> GroupedProjectionSpec {
        GroupedProjectionSpec::new(
            ParameterSpec::trainable(name).unwrap(),
            None,
            LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        )
        .unwrap()
    }
    fn gated() -> GroupedGatedProductSpec {
        GroupedGatedProductSpec::new(
            4,
            8,
            16,
            8,
            GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: projection("gate_up"),
                down: projection("down"),
            },
        )
        .unwrap()
    }
    fn linear() -> GroupedLinearSpec {
        GroupedLinearSpec::new(
            4,
            8,
            8,
            GroupedLinearActivation::Identity,
            projection("weight"),
        )
        .unwrap()
    }
    fn dominates(actual: usize, candidate: usize, outer: usize, gated: bool) -> bool {
        if candidate < actual || candidate / outer < actual / outer {
            return false;
        }
        let (a, c) = (actual % outer, candidate % outer);
        if a <= 1 {
            return c == a;
        }
        if !gated {
            return c >= a;
        }
        if a <= THRESHOLD as usize {
            return c >= a && c <= THRESHOLD as usize;
        }
        c >= a && c % CHUNK as usize == a % CHUNK as usize
    }

    #[test]
    fn indexed_row_candidates_cover_every_small_receive_count() {
        let gated = gated();
        let linear = linear();
        for (kernel, chunked) in [
            (WorkspaceExpertKernel::Gated(&gated), true),
            (WorkspaceExpertKernel::Linear(&linear), false),
        ] {
            for outer in 1..=96 {
                for maximum in 0..=192 {
                    let values = candidates(kernel, maximum, outer, tail_bands(maximum, outer))
                        .collect::<Vec<_>>();
                    assert!(values.iter().all(|value| *value > 0 && *value <= maximum));
                    for actual in 1..=maximum {
                        assert!(
                            values
                                .iter()
                                .any(|candidate| dominates(actual, *candidate, outer, chunked)),
                            "actual={actual}, maximum={maximum}, outer={outer}, gated={chunked}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn indexed_row_candidate_population_is_bounded_at_large_dimensions() {
        let kernel = gated();
        let kernel = WorkspaceExpertKernel::Gated(&kernel);
        for maximum in [0, 1, 63, 64, 65, 1_000_000, usize::MAX] {
            for outer in [1, 2, 31, 32, 33, 65, 1_000_003, usize::MAX] {
                let values = candidates(kernel, maximum, outer, tail_bands(maximum, outer))
                    .collect::<Vec<_>>();
                assert!(values.len() <= 3 + 2 * (2 + CHUNK as usize));
                assert!(values.iter().all(|value| *value > 0 && *value <= maximum));
                if maximum > 0 {
                    assert!(values.contains(&maximum));
                } else {
                    assert!(values.is_empty());
                }
            }
        }
    }
}
