//! Non-owning observations of already available native array backing.

use super::Array;
use crate::{
    error::Result,
    utils::{guard::Guarded, runtime_lock},
};
use std::collections::BTreeMap;

/// One backing observed during a single [`inspect_storage`] call.
///
/// Distinct entries do not prove disjoint memory: separate external storage
/// wrappers may overlap. This record holds no native allocation or lifetime pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayBackingStorage {
    /// The native backing uses the standard MLX allocator deleter.
    /// Custom deleters remain false even if they happen to wrap MLX storage.
    pub allocator_owned: bool,
    /// Bytes charged by MLX for this complete backing, when safely queryable.
    /// Includes cached buffer capacity, but excludes CPU allocation headers,
    /// malloc overhead, graph descriptors and other runtime storage. It is not
    /// process RSS or an upper bound for future allocations.
    pub allocator_capacity_bytes: Option<usize>,
}

/// Metadata from one storage inspection, without retained native resources.
///
/// Backing indices have meaning only inside this survey. They must never be
/// compared across calls or used as completion, ownership or retirement proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayStorageSurvey {
    /// One backing index per input; unavailable or absent storage is `None`.
    pub values: Vec<Option<usize>>,
    /// Distinct native backing wrappers observed in input order.
    pub backings: Vec<ArrayBackingStorage>,
}

/// Observes storage of already available arrays without evaluating or waiting.
///
/// All roots remain borrowed under the runtime guard while native identities
/// are remapped to survey-local indices. No pointer identity leaves this call,
/// and the returned metadata retains no arrays or allocations. Queries neither
/// initialize streams nor copy device data. Unavailable graphs remain unknown.
pub fn inspect_storage(arrays: &[&Array]) -> Result<ArrayStorageSurvey> {
    let _guard = runtime_lock::enter_for_metadata();
    let mut identities = BTreeMap::new();
    let mut survey = ArrayStorageSurvey {
        values: Vec::with_capacity(arrays.len()),
        backings: Vec::new(),
    };
    for array in arrays {
        let mut identity = 0usize;
        let mut allocator_owned = false;
        let mut capacity = 0usize;
        // SAFETY: every input is borrowed for the complete survey, and the
        // runtime guard serializes native access. Outputs are valid initialized
        // pointers. Native inspection checks availability and never reads data.
        <() as Guarded>::try_from_op(|_| unsafe {
            safemlx_sys::_mlx_array_storage_metadata(
                &mut identity,
                &mut allocator_owned,
                &mut capacity,
                array.as_ptr(),
            )
        })?;
        if identity == 0 {
            survey.values.push(None);
            continue;
        }
        let index = *identities.entry(identity).or_insert_with(|| {
            let index = survey.backings.len();
            survey.backings.push(ArrayBackingStorage {
                allocator_owned,
                allocator_capacity_bytes: allocator_owned.then_some(capacity),
            });
            index
        });
        survey.values.push(Some(index));
    }
    Ok(survey)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{broadcast_to, indexing::TryIndexOp};

    #[test]
    fn lazy_storage_queries_never_evaluate_and_empty_surveys_are_empty() {
        let stream = crate::test_stream();
        let source = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let lazy = source.transpose(stream).unwrap();
        assert!(!lazy.is_available().unwrap());
        for _ in 0..2 {
            let report = inspect_storage(&[&lazy]).unwrap();
            assert_eq!(report.values, [None]);
            assert!(report.backings.is_empty());
            assert!(!lazy.is_available().unwrap());
        }
        assert_eq!(
            inspect_storage(&[]).unwrap(),
            ArrayStorageSurvey {
                values: vec![],
                backings: vec![],
            }
        );
    }

    #[test]
    fn views_share_survey_backing_and_metadata_survives_all_roots() {
        let report = {
            let stream = crate::test_stream();
            let source = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            let cloned = source.clone();
            let transposed = source.transpose(stream).unwrap();
            let slice = source.try_index_device(0..1, stream).unwrap();
            let row = source.try_index_device(0, stream).unwrap();
            let broadcast = broadcast_to(&row, &[4, 3], stream).unwrap();
            let separate = Array::from_slice(&[7.0f32, 8.0, 9.0], &[3]);
            transposed.evaluated().unwrap();
            slice.evaluated().unwrap();
            row.evaluated().unwrap();
            broadcast.evaluated().unwrap();
            let report = inspect_storage(&[
                &source,
                &cloned,
                &transposed,
                &slice,
                &row,
                &broadcast,
                &separate,
            ])
            .unwrap();
            assert_eq!(
                report.values,
                [
                    Some(0),
                    Some(0),
                    Some(0),
                    Some(0),
                    Some(1),
                    Some(1),
                    Some(2)
                ]
            );
            // A range is a view; integer indexing uses a copied gather. The
            // broadcast shares that gather's backing, not the original source.
            assert_eq!(report.backings.len(), 3);
            assert!(report.backings.iter().all(|value| value.allocator_owned));
            assert!(report.backings[0].allocator_capacity_bytes.unwrap() >= source.nbytes());
            assert!(report.backings[1].allocator_capacity_bytes.unwrap() >= row.nbytes());
            assert!(report.backings[2].allocator_capacity_bytes.unwrap() >= separate.nbytes());
            assert_eq!(row.nbytes(), 12);
            assert_eq!(broadcast.nbytes(), 48);
            assert_eq!(
                source.evaluated().unwrap().as_slice::<f32>(),
                &[1., 2., 3., 4., 5., 6.]
            );
            report
        };
        assert_eq!(report.values[0], report.values[3]);
        assert_eq!(report.values[4], report.values[5]);
        assert_eq!(report.clone(), report);
    }

    #[test]
    fn managed_input_reports_actual_adopted_or_copied_backing() {
        let values = vec![0x0102_0304u32, u32::MAX, 17, 42];
        let source = Array::try_from_owned_data(values.clone(), &[4]).unwrap();
        let report = inspect_storage(&[&source, &source]).unwrap();
        assert_eq!(report.values, [Some(0), Some(0)]);
        let backing = &report.backings[0];
        // CPU copies external inputs; other allocators can adopt them using a
        // custom deleter. Either result must preserve the ownership distinction.
        assert_eq!(
            backing.allocator_owned,
            backing.allocator_capacity_bytes.is_some()
        );
        if let Some(capacity) = backing.allocator_capacity_bytes {
            assert!(capacity >= source.nbytes());
        }
        assert_eq!(source.evaluated().unwrap().as_slice::<u32>(), values);
    }

    #[test]
    fn metadata_inspection_does_not_invoke_runtime_housekeeping() {
        use std::cell::Cell;
        thread_local! {
            static CALLS: Cell<usize> = const { Cell::new(0) };
        }
        fn hook() {
            CALLS.with(|calls| calls.set(calls.get() + 1));
        }
        struct Registered;
        impl Drop for Registered {
            fn drop(&mut self) {
                crate::unregister_thread_runtime_housekeeping(hook);
            }
        }
        let source = Array::from_slice(&[1.0f32, 2.0], &[2]);
        crate::register_thread_runtime_housekeeping(hook);
        let _registered = Registered;
        // Both metadata entry points skip hooks. An ordinary runtime entry
        // still invokes this registered hook, even for an available array.
        source.is_available().unwrap();
        assert_eq!(CALLS.with(Cell::get), 0);
        source.evaluated().unwrap();
        assert_eq!(CALLS.with(Cell::get), 1);
        CALLS.with(|calls| calls.set(0));
        inspect_storage(&[&source]).unwrap();
        assert_eq!(CALLS.with(Cell::get), 0);
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an accessible Metal device"]
    fn metal_storage_survey_preserves_lazy_gpu_work_and_shared_backing() {
        let report = {
            let stream =
                crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0));
            let source = Array::from_slice(&[1.0f32, -2.0, 3.5, 4.0, -5.0, 6.0], &[2, 3]);
            let output = source.add(&source, &stream).unwrap();
            assert!(!output.is_available().unwrap());
            let lazy = inspect_storage(&[&output]).unwrap();
            assert_eq!(lazy.values, [None]);
            assert!(lazy.backings.is_empty());
            assert!(!output.is_available().unwrap());

            let expected = [2.0f32, -4.0, 7.0, 8.0, -10.0, 12.0];
            assert_eq!(output.evaluated().unwrap().as_slice::<f32>(), expected);
            let alias = output.clone();
            let slice = output.try_index_device(0..1, &stream).unwrap();
            let transposed = output.transpose(&stream).unwrap();
            slice.evaluated().unwrap();
            transposed.evaluated().unwrap();
            let report = inspect_storage(&[&output, &alias, &slice, &transposed]).unwrap();
            assert_eq!(report.values, [Some(0), Some(0), Some(0), Some(0)]);
            assert_eq!(report.backings.len(), 1);
            assert!(report.backings[0].allocator_owned);
            assert!(report.backings[0].allocator_capacity_bytes.unwrap() >= output.nbytes());
            assert!(slice.nbytes() < output.nbytes());
            assert_eq!(output.evaluated().unwrap().as_slice::<f32>(), expected);
            report
        };
        // The completed survey contains no tensor, stream or allocation owner.
        assert_eq!(report.values, [Some(0), Some(0), Some(0), Some(0)]);
        assert_eq!(report.backings.len(), 1);
    }
}
