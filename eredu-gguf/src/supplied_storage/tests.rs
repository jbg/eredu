use super::*;
use crate::*;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
#[allow(dead_code)]
#[path = "../convert/destination/old.rs"]
mod old;
#[derive(Debug)]
pub(crate) struct Family;
#[derive(Debug)]
pub(crate) struct Buffer<T> {
    values: Vec<T>,
    retired: Rc<Cell<usize>>,
}
impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.values));
        self.retired.set(self.retired.get() + 1)
    }
}
impl<T> AsRef<[T]> for Buffer<T> {
    fn as_ref(&self) -> &[T] {
        &self.values
    }
}
impl<T> AsMut<[T]> for Buffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}
impl<T: fmt::Debug + 'static> InitializedStorage<T> for Buffer<T> {
    fn capacity(&self) -> usize {
        self.values.capacity()
    }
}
#[derive(Debug)]
pub(crate) struct Refusal(usize);
impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "refused reached slot {}", self.0)
    }
}
impl std::error::Error for Refusal {}
impl StorageFamily for Family {
    type Buffer<T: Copy + fmt::Debug + 'static> = Buffer<T>;
    type Error = Refusal;
}
#[derive(Default)]
pub(crate) struct Provider {
    calls: Rc<RefCell<Vec<(usize, usize, usize)>>>,
    retired: Rc<Cell<usize>>,
    fail: Option<usize>,
}
impl StorageProvider for Provider {
    type Family = Family;
    fn prepare<T: Copy + fmt::Debug + 'static>(
        &mut self,
        n: usize,
        initial: T,
    ) -> std::result::Result<Buffer<T>, (Refusal, Option<Buffer<T>>)> {
        let values = vec![initial; n];
        let ordinal = self.calls.borrow().len();
        self.calls
            .borrow_mut()
            .push((n, std::mem::size_of::<T>(), values.as_ptr() as usize));
        let out = Buffer {
            values,
            retired: self.retired.clone(),
        };
        if self.fail == Some(ordinal) {
            Err((Refusal(ordinal), Some(out)))
        } else {
            Ok(out)
        }
    }
}
fn input(ty: GgmlType, blocks: usize) -> (TensorDescriptor, Vec<u8>) {
    let (v, b) = ty.block_and_bytes().unwrap();
    (
        TensorDescriptor {
            name: "actual.α.weight".into(),
            dimensions: vec![v, blocks as u64],
            ggml_type: ty,
            relative_offset: 32,
            data_offset: 128,
            byte_len: b * blocks as u64,
        },
        (0..b as usize * blocks)
            .map(|i| (i.wrapping_mul(73).wrapping_add(29) % 256) as u8)
            .collect(),
    )
}
pub(crate) fn own(value: StoredConvertedTensor<Family>) -> ConvertedTensor {
    match value {
        StoredConvertedTensor::Dense { shape, dtype, data } => {
            ConvertedTensor::Dense(DenseTensor {
                shape: shape.as_slice().to_vec(),
                dtype,
                data: data.as_slice().to_vec(),
            })
        }
        StoredConvertedTensor::IQuant {
            shape,
            ggml_type,
            endian,
            data,
        } => ConvertedTensor::IQuant(IQuantTensor {
            shape: shape.as_slice().to_vec(),
            ggml_type,
            endian,
            data: data.as_slice().to_vec(),
        }),
        StoredConvertedTensor::Affine {
            weight_shape,
            scale_shape,
            bits,
            group_size,
            weights,
            scales,
            biases,
        } => ConvertedTensor::Affine(AffineTensor {
            weight_shape: weight_shape.as_slice().to_vec(),
            scale_shape: scale_shape.as_slice().to_vec(),
            bits,
            group_size,
            weights: weights.as_slice().to_vec(),
            scales: scales.as_slice().to_vec(),
            biases: biases.as_slice().to_vec(),
        }),
        StoredConvertedTensor::MxFp4 {
            weight_shape,
            scale_shape,
            weights,
            scales,
        } => ConvertedTensor::MxFp4(MxFp4Tensor {
            weight_shape: weight_shape.as_slice().to_vec(),
            scale_shape: scale_shape.as_slice().to_vec(),
            weights: weights.as_slice().to_vec(),
            scales: scales.as_slice().to_vec(),
        }),
    }
}
#[test]
fn supplied_all_conversion_branches_match_independent_old_bits_and_keep_real_owners() {
    for ty in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Bf16,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::IQ2XXS,
        GgmlType::IQ2XS,
        GgmlType::IQ3XXS,
        GgmlType::IQ1S,
        GgmlType::IQ4NL,
        GgmlType::IQ3S,
        GgmlType::IQ2S,
        GgmlType::IQ4XS,
        GgmlType::IQ1M,
        GgmlType::MxFp4,
    ] {
        for endian in [Endian::Little, Endian::Big] {
            let (d, raw) = input(ty, 3);
            let mut provider = Provider::default();
            let selected =
                StoredPhysicalDescriptor::prepare(&d, MetadataSelection::Full, &mut provider)
                    .unwrap();
            let mut conversion =
                StoredConversion::prepare(selected.into_descriptor(), endian, &mut provider)
                    .unwrap();
            let calls = provider.calls.borrow().clone();
            let bound = crate::ConversionPlan::new(&d, endian)
                .unwrap()
                .supplied_storage_bound()
                .unwrap();
            let actual = &calls[2..]; // actual full physical descriptor name/dimensions
            assert_eq!(bound.calls(), actual.len());
            assert_eq!(
                bound.bytes(),
                actual
                    .iter()
                    .map(|(count, size, _)| count * size)
                    .sum::<usize>()
            );
            let capacities = conversion.capacities();
            conversion.fill(&raw, d.view(), endian).unwrap();
            assert_eq!(conversion.capacities(), capacities);
            assert_eq!(*provider.calls.borrow(), calls);
            assert_eq!(provider.retired.get(), 0);
            let output = conversion.finish();
            let address = match &output {
                StoredConvertedTensor::Dense { data, .. }
                | StoredConvertedTensor::IQuant { data, .. } => data.as_slice().as_ptr() as usize,
                StoredConvertedTensor::Affine { weights, .. }
                | StoredConvertedTensor::MxFp4 { weights, .. } => {
                    weights.as_slice().as_ptr() as usize
                }
            };
            assert!(calls.iter().any(|(_, _, p)| *p == address));
            assert_eq!(
                format!("{:?}", own(output)),
                format!("{:?}", old::convert(&d, &raw, endian).unwrap())
            );
            assert_eq!(provider.retired.get(), calls.len());
        }
    }
}
#[test]
fn supplied_reached_refusal_keeps_every_allocated_prefix_and_never_calls_next_slot() {
    let (d, _) = input(GgmlType::Q4_0, 3);
    let mut baseline = Provider::default();
    let physical =
        StoredPhysicalDescriptor::prepare(&d, MetadataSelection::Full, &mut baseline).unwrap();
    let prepared =
        StoredConversion::prepare(physical.into_descriptor(), Endian::Big, &mut baseline).unwrap();
    let total = baseline.calls.borrow().len();
    drop(prepared);
    for fail in 0..total {
        let mut provider = Provider {
            fail: Some(fail),
            ..Provider::default()
        };
        match StoredPhysicalDescriptor::prepare(&d, MetadataSelection::Full, &mut provider) {
            Err(error) => {
                assert_eq!(provider.retired.get(), 0);
                assert_eq!(provider.calls.borrow().len(), fail + 1);
                drop(error)
            }
            Ok(physical) => {
                let error = StoredConversion::prepare(
                    physical.into_descriptor(),
                    Endian::Big,
                    &mut provider,
                )
                .unwrap_err();
                assert_eq!(provider.retired.get(), 0);
                assert_eq!(provider.calls.borrow().len(), fail + 1);
                assert!(
                    matches!(error.cause(),SuppliedStorageError::Provider(Refusal(i))if *i==fail)
                );
                drop(error)
            }
        }
        assert_eq!(provider.retired.get(), fail + 1);
    }
}
#[test]
fn supplied_malformed_affine_keeps_emission_ceiling_and_old_final_shape_error() {
    for ty in [
        GgmlType::Q4_0,
        GgmlType::Q5_0,
        GgmlType::Q8_0,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ] {
        let (mut d, raw) = input(ty, 3);
        d.dimensions[1] = 1;
        let mut provider = Provider::default();
        let physical =
            StoredPhysicalDescriptor::prepare(&d, MetadataSelection::Full, &mut provider).unwrap();
        let mut conversion =
            StoredConversion::prepare(physical.into_descriptor(), Endian::Big, &mut provider)
                .unwrap();
        let capacities = conversion.capacities();
        let bound = ConversionPlan::new(&d, Endian::Big)
            .unwrap()
            .supplied_storage_bound()
            .unwrap();
        let calls = provider.calls.borrow();
        assert_eq!(bound.calls(), calls.len() - 2);
        assert_eq!(
            bound.bytes(),
            calls[2..].iter().map(|(n, s, _)| n * s).sum::<usize>()
        );
        drop(calls);
        let error = conversion.fill(&raw, d.view(), Endian::Big).unwrap_err();
        assert_eq!(
            error.to_string(),
            old::convert(&d, &raw, Endian::Big).unwrap_err().to_string()
        );
        assert_eq!(conversion.capacities(), capacities);
        assert!(!conversion.completed());
        assert_eq!(provider.retired.get(), 0);
        drop(conversion);
        assert_eq!(provider.retired.get(), provider.calls.borrow().len());
    }
}
#[test]
fn supplied_physical_validation_precedes_any_unreached_provider_refusal() {
    let (d, _) = input(GgmlType::F32, 3);
    let bad = TensorSelection::Range {
        axis: 8,
        start: 0,
        end: 1,
    };
    let mut provider = Provider {
        fail: Some(0),
        ..Provider::default()
    };
    let failure =
        StoredPhysicalDescriptor::prepare(&d, MetadataSelection::Axis(&bad), &mut provider)
            .unwrap_err();
    assert_eq!(
        failure.to_string(),
        TensorSelectionPlan::new(&d, bad).unwrap_err().to_string()
    );
    assert!(provider.calls.borrow().is_empty());
    let selection = TensorSelection::Indices {
        axis: 0,
        indices: vec![2, 0, 2],
    };
    let mut provider = Provider::default();
    let selected =
        StoredPhysicalDescriptor::prepare(&d, MetadataSelection::Axis(&selection), &mut provider)
            .unwrap();
    assert_eq!(
        selected.descriptor(),
        TensorSelectionPlan::new(&d, selection)
            .unwrap()
            .selected_descriptor()
            .view()
    );
    drop(selected);
    assert_eq!(provider.retired.get(), provider.calls.borrow().len());
}

#[test]
fn supplied_scalar_g2_counts_the_actual_zero_shape_request() {
    let (mut d, _) = input(GgmlType::F32, 1);
    d.dimensions.clear();
    d.byte_len = 4;
    let raw = 1.5f32.to_le_bytes();
    let plan = ConversionPlan::new(&d, Endian::Little).unwrap();
    let bound = plan.supplied_storage_bound().unwrap();
    let mut provider = Provider::default();
    let physical =
        StoredPhysicalDescriptor::prepare(&d, MetadataSelection::Full, &mut provider).unwrap();
    let mut conversion =
        StoredConversion::prepare(physical.into_descriptor(), Endian::Little, &mut provider)
            .unwrap();
    let calls = provider.calls.borrow().clone();
    assert!(calls[2..].iter().any(|(n, _, _)| *n == 0));
    assert_eq!(bound.calls(), calls.len() - 2);
    assert_eq!(
        bound.bytes(),
        calls[2..]
            .iter()
            .map(|(n, size, _)| n * size)
            .sum::<usize>()
    );
    conversion.fill(&raw, d.view(), Endian::Little).unwrap();
    assert_eq!(
        format!("{:?}", own(conversion.finish())),
        format!("{:?}", old::convert(&d, &raw, Endian::Little).unwrap())
    );
    assert_eq!(provider.retired.get(), calls.len());
}
