//! Fixed facts from the unchanged exact layout/population comparison.
use super::*;

/// A real partial differs from the retained collective source. No variant
/// allocates a diagnostic destination or carries execution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GroupCpuBindingError {
    /// The native input rank differs from the quoted rank.
    #[error("collective input rank differs: expected {expected}, actual {actual}")]
    Rank {
        /// Retained equation rank.
        expected: usize,
        /// Actual native input rank.
        actual: usize,
    },
    /// The first mismatching dimension of equal-rank inputs.
    #[error("collective input dimension {axis} differs: expected {expected}, actual {actual}")]
    Dimension {
        /// First differing axis.
        axis: usize,
        /// Retained equation dimension.
        expected: i32,
        /// Actual native dimension.
        actual: i32,
    },
    /// The native scalar type differs from the quoted type.
    #[error("collective input scalar differs: expected {expected:?}, actual {actual:?}")]
    Scalar {
        /// Retained equation scalar type.
        expected: Dtype,
        /// Actual native scalar type.
        actual: Dtype,
    },
    /// The existing native source query refused this actual partial.
    #[error("actual collective source query failed")]
    NativeSource(#[source] GroupStorageUnavailable),
    /// The first differing source field, including the exact allocation-class
    /// index when it comes from an array of request facts.
    #[error(
        "collective {population:?} field {field}, index {index:?}, differs: expected {expected}, actual {actual}"
    )]
    Population {
        /// Native worker whose exact source differed.
        population: GroupCpuBindingPopulation,
        /// Native scalar or request-array field name.
        field: &'static str,
        /// Exact request-array index, or none for a scalar.
        index: Option<usize>,
        /// Retained cold source value.
        expected: i128,
        /// Recomputed actual input source value.
        actual: i128,
    },
}

/// The unchanged native producer whose exact population did not match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupCpuBindingPopulation {
    /// Lazy collective primitive construction.
    Constructor,
    /// CPU copy/evaluation and retained communication worker.
    Evaluation,
}

pub(super) fn geometry(
    expected: &[i32],
    dtype: Dtype,
    actual: &[i32],
    scalar: Dtype,
) -> Result<(), GroupCpuBindingError> {
    if expected.len() != actual.len() {
        return Err(GroupCpuBindingError::Rank {
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    for (axis, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        if expected != actual {
            return Err(GroupCpuBindingError::Dimension {
                axis,
                expected,
                actual,
            });
        }
    }
    if dtype != scalar {
        return Err(GroupCpuBindingError::Scalar {
            expected: dtype,
            actual: scalar,
        });
    }
    Ok(())
}

pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<GroupCpuBindingError>(),
        size_of::<Result<(), GroupCpuBindingError>>(),
        size_of::<(&[i32], Dtype, &[i32], Dtype)>(),
        size_of::<
            std::iter::Enumerate<
                std::iter::Zip<std::slice::Iter<'_, i32>, std::slice::Iter<'_, i32>>,
            >,
        >(),
        size_of::<Option<(usize, (&i32, &i32))>>(),
        size_of::<(usize, i32, i32)>(),
        size_of::<(&[usize], &[usize])>(),
        size_of::<
            std::iter::Enumerate<
                std::iter::Zip<std::slice::Iter<'_, usize>, std::slice::Iter<'_, usize>>,
            >,
        >(),
        size_of::<Option<(usize, (&usize, &usize))>>(),
        size_of::<(usize, usize, usize)>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

// Keep one comparison per source field. No diagnostic vector or clone is
// needed: the first failed exact comparison carries its fixed scalar facts.
macro_rules! fields {
    ($kind:ident, $a:ident, $b:ident; $($field:ident $(.$nested:ident)?),* $(,)?) => {$(
        if $a.$field $(.$nested)? != $b.$field $(.$nested)? {
            return Err(GroupCpuBindingError::Population {
                population: GroupCpuBindingPopulation::$kind,
                field: stringify!($field $(.$nested)?), index: None,
                expected: $a.$field $(.$nested)? as i128,
                actual: $b.$field $(.$nested)? as i128,
            });
        }
    )*};
}
macro_rules! arrays {
    ($kind:ident, $a:ident, $b:ident; $($field:ident),* $(,)?) => {$(
        for (index, (&expected, &actual)) in $a.$field.iter().zip(&$b.$field).enumerate() {
            if expected != actual {
                return Err(GroupCpuBindingError::Population {
                    population: GroupCpuBindingPopulation::$kind,
                    field: stringify!($field), index: Some(index),
                    expected: expected as i128, actual: actual as i128,
                });
            }
        }
    )*};
}
pub(super) fn constructor(
    a: &safemlx_sys::mlx_distributed_constructor_storage,
    b: &safemlx_sys::mlx_distributed_constructor_storage,
) -> Result<(), GroupCpuBindingError> {
    // Named inspection frames legitimately differ between layout and Array.
    // All semantic and allocation population facts must remain identical.
    fields!(Constructor, a, b; output_rank, output_elements, primitives,
        input_edges, blocks, header_bytes, header_alignment, slots_bytes,
        slots_alignment, reserved_alignment, requested_bytes, allocation_extents);
    arrays!(Constructor, a, b; request_bytes, request_alignments, request_counts);
    Ok(())
}
pub(super) fn evaluation(
    a: &safemlx_sys::mlx_distributed_cpu_eval_storage,
    b: &safemlx_sys::mlx_distributed_cpu_eval_storage,
) -> Result<(), GroupCpuBindingError> {
    fields!(Evaluation, a, b; operation, peer, input_rank, output_rank, inputs,
        tracer, possible_copy, backing_births, data_captures, temporary_batches,
        logical_backing_bytes, copy_backing_bytes, output_backing_bytes,
        copy_worker_graph_extent, communication_worker_graph_extent, blocks,
        header_bytes, header_alignment, slots_bytes, slots_alignment,
        reserved_alignment, requested_bytes, allocation_extents);
    arrays!(Evaluation, a, b; request_bytes, request_alignments, request_counts);
    fields!(Evaluation, a, b; communication.pool_jobs, communication.socket_attempts,
        communication.destination_arrays, communication.task_graph_extent,
        communication.destination_graph_extent);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_binding_reports_geometry_and_distinguishes_worker_population() {
        assert_eq!(
            geometry(&[2, 3], Dtype::Float32, &[6], Dtype::Float32),
            Err(GroupCpuBindingError::Rank {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            geometry(&[2, 3], Dtype::Float32, &[2, 4], Dtype::Float32),
            Err(GroupCpuBindingError::Dimension {
                axis: 1,
                expected: 3,
                actual: 4
            })
        );
        assert!(matches!(
            geometry(&[2, 3], Dtype::Float32, &[2, 3], Dtype::Float16),
            Err(GroupCpuBindingError::Scalar { .. })
        ));
        let a = safemlx_sys::mlx_distributed_constructor_storage::default();
        let mut b = a;
        b.named_control_bytes = 7;
        assert_eq!(constructor(&a, &b), Ok(()));
        b.request_counts[1] = 3;
        assert_eq!(
            constructor(&a, &b),
            Err(GroupCpuBindingError::Population {
                population: GroupCpuBindingPopulation::Constructor,
                field: "request_counts",
                index: Some(1),
                expected: 0,
                actual: 3,
            })
        );
        let a = safemlx_sys::mlx_distributed_cpu_eval_storage::default();
        let mut b = a;
        b.communication.pool_jobs = 2;
        assert_eq!(
            evaluation(&a, &b),
            Err(GroupCpuBindingError::Population {
                population: GroupCpuBindingPopulation::Evaluation,
                field: "communication.pool_jobs",
                index: None,
                expected: 0,
                actual: 2,
            })
        );
    }
}
