use super::*;
use crate::memory_fixture::LedgerFixture as _;
use eredu_runtime::{
    with_routed_unit_invocation, RoutedUnitBatch, RoutedUnitInvocation, RoutedUnitObserver,
};

#[test]
fn host_observation_preserves_logical_strides_for_all_supported_dtypes() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for dtype in [
        Dtype::Bool,
        Dtype::Uint8,
        Dtype::Uint16,
        Dtype::Uint32,
        Dtype::Uint64,
        Dtype::Int8,
        Dtype::Int16,
        Dtype::Int32,
        Dtype::Int64,
        Dtype::Float16,
        Dtype::Bfloat16,
        Dtype::Float32,
        Dtype::Float64,
    ] {
        let array = Array::from_slice(&[0_f32, 1., 2., 3., 4., 5.], &[2, 3])
            .as_dtype(dtype, &stream)
            .unwrap();
        let reverse = array
            .as_strided(&[2, 3][..], &[-3, -1][..], 5, &stream)
            .unwrap();
        let transposed = array.transpose_axes(&[1, 0], &stream).unwrap();
        let row = array.try_index_device(..1, &stream).unwrap();
        let broadcast = safemlx::ops::broadcast_to(&row, &[2, 3], &stream).unwrap();
        let empty = array.try_index_device(..0, &stream).unwrap();
        let scalar = array.try_index_device((0, 1), &stream).unwrap();
        for (view, expected) in [
            (reverse, vec![5_i64, 4, 3, 2, 1, 0]),
            (transposed, vec![0, 3, 1, 4, 2, 5]),
            (broadcast, vec![0, 1, 2, 0, 1, 2]),
            (empty, vec![]),
            (scalar, vec![1]),
        ] {
            let shape = view.shape().iter().map(|n| *n as usize).collect::<Vec<_>>();
            let observed = observe_tensor(&MlxTensor::from_array(view), &stream).unwrap();
            assert_eq!(observed.shape(), shape);
            let expected = match dtype {
                Dtype::Bool => {
                    TensorObservationData::Bool(expected.iter().map(|n| *n != 0).collect())
                }
                Dtype::Uint8 | Dtype::Uint16 | Dtype::Uint32 | Dtype::Uint64 => {
                    TensorObservationData::U64(expected.iter().map(|n| *n as u64).collect())
                }
                Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64 => {
                    TensorObservationData::I64(expected)
                }
                _ => TensorObservationData::F32(expected.iter().map(|n| *n as f32).collect()),
            };
            assert_eq!(observed.data(), &expected, "{dtype:?}");
            let (length, capacity) = match observed.data() {
                TensorObservationData::Bool(v) => (v.len(), v.capacity()),
                TensorObservationData::U64(v) => (v.len(), v.capacity()),
                TensorObservationData::I64(v) => (v.len(), v.capacity()),
                TensorObservationData::F32(v) => (v.len(), v.capacity()),
            };
            assert_eq!(
                length, capacity,
                "{dtype:?} host capacity exceeds logical payload"
            );
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("composed native observer failure")]
struct Sentinel;

struct Observer<'a> {
    input: &'a MlxTensor,
    units: &'a eredu_core::component::ComponentCoordinateMap,
    active: bool,
    fail: bool,
    started: usize,
    finished: Vec<bool>,
}
impl RuntimeActivationObserver<MlxTensor, Error> for Observer<'_> {
    fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), Error> {
        unreachable!()
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<MlxTensor>>, Error> {
        assert_eq!(path, "layer.0.units");
        Ok(Some(self))
    }
}
impl RoutedUnitObserver<MlxTensor> for Observer<'_> {
    fn invocation_active(&self) -> bool {
        self.active
    }
    fn begin_invocation(
        &mut self,
        invocation: &RoutedUnitInvocation<'_, MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        assert!(std::ptr::eq(invocation.input, self.input));
        assert!(std::ptr::eq(
            invocation.unit_coordinates.unwrap(),
            self.units
        ));
        let origin = invocation.origins.unwrap().resolve(0).unwrap();
        assert_eq!(
            (origin.source_peer, origin.token, origin.slot),
            (Some(1), 1, 1)
        );
        self.started += 1;
        if self.fail {
            Err(eredu_nn::Error::backend_retained_source(Sentinel))
        } else {
            Ok(())
        }
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), eredu_nn::Error> {
        self.finished.push(success);
        if self.fail {
            Err(eredu_nn::Error::backend("later finish failure"))
        } else {
            Ok(())
        }
    }
    fn observe(&mut self, _: &RoutedUnitBatch<'_, MlxTensor>) -> Result<(), eredu_nn::Error> {
        unreachable!()
    }
}

#[test]
#[ignore = "requires native MLX access"]
fn composed_array_and_neutral_adapters_preserve_routed_invocation_scope() {
    let input = MlxTensor::from_array(Array::from_slice(&[0.3f32, -0.7], &[1, 2]));
    let units = eredu_core::component::ComponentCoordinateMap::indices(5, vec![3, 1]).unwrap();
    let origins = eredu_runtime::RoutedUnitOrigins::new(&[0, 1], &[3], 2).unwrap();
    for (active, fail) in [(false, false), (false, true), (true, false)] {
        let mut inner = Observer {
            input: &input,
            units: &units,
            active,
            fail,
            started: 0,
            finished: Vec::new(),
        };
        let mut arrays = ArrayObserverAdapter {
            inner: &mut inner,
            routed_path: None,
            routed_invocation_active: false,
            allocation_authority: None,
        };
        let mut neutral = crate::composition::NeutralActivationObserver::new(&mut arrays);
        let observer = neutral.routed_unit_observer("layer.0.units").unwrap();
        assert_eq!(observer.as_ref().unwrap().invocation_active(), active);
        let mut called = false;
        let result = with_routed_unit_invocation(
            observer,
            RoutedUnitInvocation {
                input: &input,
                origins: Some(origins),
                unit_coordinates: Some(&units),
            },
            |observer| {
                assert!(observer.unwrap().invocation_active());
                called = true;
                Ok::<_, eredu_nn::Error>(())
            },
            |error| error,
        );
        assert_eq!(neutral.invocation_active(), active);
        assert_eq!(arrays.invocation_active(), active);
        assert_eq!(called, !fail);
        assert_eq!(inner.started, usize::from(!active));
        assert_eq!(inner.finished, if active { vec![] } else { vec![!fail] });
        if fail {
            let error = result.unwrap_err();
            let mut cause: &dyn std::error::Error = &error;
            while let Some(source) = cause.source() {
                cause = source;
            }
            assert!(cause.is::<Sentinel>());
        } else {
            result.unwrap();
        }
    }
}

#[test]
fn composed_observer_adapters_preserve_readout_demand_without_native_allocation() {
    struct SequenceObserver;
    impl RuntimeActivationObserver<MlxTensor, Error> for SequenceObserver {
        fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), Error> {
            Ok(())
        }
    }
    for sequence in [false, true] {
        let mut noop = eredu_runtime::NoopObserver;
        let mut capture = SequenceObserver;
        let inner: &mut dyn RuntimeActivationObserver<MlxTensor, Error> =
            if sequence { &mut capture } else { &mut noop };
        let mut arrays = ArrayObserverAdapter {
            inner,
            routed_path: None,
            routed_invocation_active: false,
            allocation_authority: None,
        };
        let neutral = crate::composition::NeutralActivationObserver::new(&mut arrays);
        assert_eq!(neutral.requires_sequence_readout(), sequence);
    }
}

#[test]
fn retained_observation_aliases_share_host_charge_until_final_owner() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
    let value = MlxTensor::from_array(Array::from_slice(&[0.5f32, -1.25, 3.5, 7.0], &[2, 2]));
    let current = || {
        pool.snapshot()
            .unwrap()
            .domains
            .iter()
            .find(|domain| domain.domain == pool.topology().host_domain())
            .unwrap()
            .current_charge_bytes
    };
    let baseline = current();
    let observed = readback::observe_tensor_in(&value, &stream, &pool).unwrap();
    assert_eq!(
        observed.data(),
        &TensorObservationData::F32(vec![0.5, -1.25, 3.5, 7.0])
    );
    assert_eq!(observed.shape(), [2, 2]);
    let charged = current();
    assert!(charged > baseline + 4 * std::mem::size_of::<f32>() as u64);
    drop(value);
    let alias = observed.clone();
    assert!(alias.same_storage(&observed));
    assert_eq!(current(), charged);
    drop(observed);
    assert_eq!(current(), charged);
    drop(alias);
    assert_eq!(current(), baseline);
}

#[test]
fn retained_observation_checks_host_limit_before_readback() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = crate::memory_fixture::ledger(0, 0).unwrap();
    let value = MlxTensor::from_array(Array::from_slice(&[0.5f32, -1.25], &[2]));
    let before = pool.snapshot().unwrap();
    let error = readback::observe_tensor_in(&value, &stream, &pool).unwrap_err();
    let mut cause: &dyn std::error::Error = &error;
    let mut budget = false;
    loop {
        if let Some(error) =
            cause.downcast_ref::<eredu_runtime::working_memory::WorkingMemoryError>()
        {
            budget |= matches!(error, eredu_runtime::working_memory::WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { domain, .. }
            ) if *domain == pool.topology().host_domain());
        }
        match cause.source() {
            Some(source) => cause = source,
            None => break,
        }
    }
    assert!(budget, "{error:?}");
    assert_eq!(pool.snapshot().unwrap(), before);
}

#[cfg(feature = "metal")]
#[test]
fn retained_observation_copies_noncontiguous_metal_output() {
    let pool = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let source = Array::from_slice(&[1_f32, 2., 3., 4., 5., 6.], &[2, 3]);
    let produced = source
        .add(&source, &stream)
        .unwrap()
        .transpose_axes(&[1, 0], &stream)
        .unwrap();
    let tensor = MlxTensor::from_array(produced);
    let before = pool.fixture_host_current().unwrap();
    let observed = readback::observe_tensor_in(&tensor, &stream, &pool).unwrap();
    assert_eq!(observed.shape(), &[3, 2]);
    assert_eq!(
        observed.data(),
        &TensorObservationData::F32(vec![2., 8., 4., 10., 6., 12.])
    );
    drop(tensor);
    assert!(pool.fixture_host_current().unwrap() > before);
    drop(observed);
    assert_eq!(pool.fixture_host_current().unwrap(), before);
}
