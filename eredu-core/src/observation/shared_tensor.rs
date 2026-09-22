//! Shared read-only ownership; finite construction remains a runtime contract.
use super::*;
use std::{fmt, sync::Arc};

trait TensorCustody: Send + Sync {
    fn retire(self: Box<Self>);
}
fn unbox_custody<T>(owner: Box<T>) -> T {
    *owner
}
impl<T: Send + Sync> TensorCustody for T {
    fn retire(self: Box<Self>) {
        let value = unbox_custody(self);
        drop(value);
    }
}
struct TensorCustodyOwner(Option<Box<dyn TensorCustody>>);
impl Drop for TensorCustodyOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}

struct OwnedTensorObservation {
    // Payload drops before any final custody, including during unwinding.
    observation: TensorObservation,
    #[cfg(test)]
    _payload_retired: Option<PayloadRetired>,
    _custody: TensorCustodyOwner,
}

/// Read-only aliases of one actual host tensor observation.
///
/// Clone shares the allocation. Serialization is identical to TensorObservation.
/// This type alone certifies no funding, native completion, source provenance or
/// permission to capture. Runtime-owned closed constructors can retain their
/// independent allocation custody here. There is no mutable or owning export.
#[derive(Clone)]
pub struct SharedTensorObservation(TensorOwner);

#[derive(Clone)]
struct TensorOwner(Option<Arc<OwnedTensorObservation>>);
impl TensorOwner {
    fn get(&self) -> &OwnedTensorObservation {
        self.0.as_ref().expect("live tensor observation")
    }
}
impl Drop for TensorOwner {
    fn drop(&mut self) {
        // Every strong alias consumes its Arc and no Weak/raw owner escapes.
        // The winning call deallocates Arc's control before returning its value.
        // The moved value then retires its buffers and unboxes final custody.
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

impl SharedTensorObservation {
    /// Complete shared-control and concrete custody-Box allocation layouts.
    ///
    /// The Arc layout includes the inline TensorObservation exactly once. Its
    /// separately allocated shape/data capacities are not included. The pinned
    /// standard library uses two atomic counters followed by the concrete value.
    /// This is a construction fact, not a funding grant or payload diagnostic.
    #[doc(hidden)]
    pub fn retained_control_bytes<T: Send + Sync + 'static>() -> Option<u64> {
        let arc = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<OwnedTensorObservation>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        u64::try_from(arc.checked_add(std::mem::size_of::<T>())?).ok()
    }

    /// Backend/runtime ownership bridge, not an allocation or funding grant.
    ///
    /// The caller must already own and account for the entire observation before
    /// calling this. The supplied owner does not establish a byte allowance or
    /// certify completion. It must not retain this shared owner itself.
    #[doc(hidden)]
    pub fn retain(observation: TensorObservation, custody: impl Send + Sync + 'static) -> Self {
        Self(TensorOwner(Some(Arc::new(OwnedTensorObservation {
            observation,
            #[cfg(test)]
            _payload_retired: None,
            _custody: TensorCustodyOwner(Some(Box::new(custody))),
        }))))
    }

    /// Shares a caller-owned observation without granting execution or funding.
    /// Deserialized and independently constructed portable evidence uses this
    /// ownership boundary; native producers retain their actual admitted owner.
    pub fn from_caller_owned(observation: TensorObservation) -> Self {
        Self::retain(observation, ())
    }

    /// Borrowing or cloning this raw DTO does not transfer allocation authority.
    /// A deep clone of the borrowed DTO is a separate caller-owned allocation.
    pub fn as_observation(&self) -> &TensorObservation {
        &self.0.get().observation
    }

    /// Logical row-major shape of the retained allocation.
    pub fn shape(&self) -> &[usize] {
        self.0.get().observation.shape()
    }

    /// Read-only retained values, with no mutable or owning export.
    pub fn data(&self) -> &TensorObservationData {
        self.0.get().observation.data()
    }

    /// Whether both handles retain the same physical host payload owner.
    pub fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0 .0.as_ref().expect("live tensor observation"),
            other.0 .0.as_ref().expect("live tensor observation"),
        )
    }

    /// Exact DTO/shape/data capacity inspection; shared controls are excluded.
    /// This diagnostic does not establish funding.
    pub fn retained_payload_bytes(&self) -> Option<u64> {
        self.0.get().observation.retained_payload_bytes()
    }
}
impl AsRef<TensorObservation> for SharedTensorObservation {
    fn as_ref(&self) -> &TensorObservation {
        self.as_observation()
    }
}
impl PartialEq for SharedTensorObservation {
    fn eq(&self, other: &Self) -> bool {
        self.as_observation() == other.as_observation()
    }
}
impl fmt::Debug for SharedTensorObservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedTensorObservation")
            .field("shape", &self.shape())
            .field("retained_payload_bytes", &self.retained_payload_bytes())
            .finish_non_exhaustive()
    }
}
impl Serialize for SharedTensorObservation {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_observation().serialize(serializer)
    }
}

impl From<TensorObservation> for SharedTensorObservation {
    fn from(observation: TensorObservation) -> Self {
        Self::from_caller_owned(observation)
    }
}
impl<'de> Deserialize<'de> for SharedTensorObservation {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // The serialized values cannot carry a process-local allocation grant.
        TensorObservation::deserialize(deserializer).map(Self::from_caller_owned)
    }
}

#[cfg(test)]
struct PayloadRetired(std::sync::Arc<std::sync::atomic::AtomicUsize>);
#[cfg(test)]
impl Drop for PayloadRetired {
    fn drop(&mut self) {
        self.0.store(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Retired(Arc<AtomicUsize>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn aliases_share_values_wire_and_final_custody() {
        let retired = Arc::new(AtomicUsize::new(0));
        let mut shape = Vec::with_capacity(7);
        shape.extend([1, 3]);
        let mut data = Vec::with_capacity(11);
        data.extend([0.25, -2.0, 7.0]);
        let raw = TensorObservation::new(shape, TensorObservationData::F32(data)).unwrap();
        let bytes =
            std::mem::size_of::<TensorObservation>() + 7 * std::mem::size_of::<usize>() + 11 * 4;
        assert_eq!(raw.retained_payload_bytes(), Some(bytes as u64));
        let wire = serde_json::to_value(&raw).unwrap();
        let shared = SharedTensorObservation::retain(raw, Retired(retired.clone()));
        let alias = shared.clone();
        assert!(alias.same_storage(&shared));
        assert_eq!(alias.shape().as_ptr(), shared.shape().as_ptr());
        assert_eq!(serde_json::to_value(&alias).unwrap(), wire);
        let decoded: TensorObservation = serde_json::from_value(wire).unwrap();
        assert_eq!(alias.as_ref(), &decoded);
        let independent = SharedTensorObservation::retain(decoded, ());
        assert_eq!(alias, independent);
        assert!(!alias.same_storage(&independent));
        drop(shared);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn capacity_inspection_preserves_empty_and_integer_storage() {
        for data in [
            TensorObservationData::Bool(Vec::with_capacity(9)),
            TensorObservationData::I64(Vec::with_capacity(9)),
            TensorObservationData::U64(Vec::with_capacity(9)),
        ] {
            let width = match &data {
                TensorObservationData::Bool(_) => 1,
                _ => 8,
            };
            let raw = TensorObservation::new(vec![0], data).unwrap();
            assert_eq!(
                raw.retained_payload_bytes(),
                Some(
                    (std::mem::size_of::<TensorObservation>()
                        + std::mem::size_of::<usize>()
                        + 9 * width) as u64
                )
            );
        }
    }
    #[test]
    fn actual_payload_retires_before_final_custody() {
        struct Charge(Arc<AtomicUsize>);
        impl Drop for Charge {
            fn drop(&mut self) {
                assert_eq!(self.0.load(Ordering::SeqCst), 1);
                self.0.store(2, Ordering::SeqCst);
            }
        }
        let order = Arc::new(AtomicUsize::new(0));
        let shared = SharedTensorObservation(TensorOwner(Some(Arc::new(OwnedTensorObservation {
            observation: TensorObservation::new(
                vec![2],
                TensorObservationData::F32(vec![3.0, 4.0]),
            )
            .unwrap(),
            _payload_retired: Some(PayloadRetired(order.clone())),
            _custody: TensorCustodyOwner(Some(Box::new(Charge(order.clone())))),
        }))));
        let last = shared.clone();
        drop(shared);
        assert_eq!(order.load(Ordering::SeqCst), 0);
        drop(last);
        assert_eq!(order.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn concurrent_final_aliases_retire_one_payload_then_one_custody() {
        struct Charge(Arc<AtomicUsize>);
        impl Drop for Charge {
            fn drop(&mut self) {
                assert_eq!(self.0.swap(2, Ordering::SeqCst), 1);
            }
        }
        let order = Arc::new(AtomicUsize::new(0));
        let shared = SharedTensorObservation(TensorOwner(Some(Arc::new(OwnedTensorObservation {
            observation: TensorObservation::new(
                vec![3],
                TensorObservationData::F32(vec![1.25, -0.0, f32::INFINITY]),
            )
            .unwrap(),
            _payload_retired: Some(PayloadRetired(order.clone())),
            _custody: TensorCustodyOwner(Some(Box::new(Charge(order.clone())))),
        }))));
        let barrier = Arc::new(std::sync::Barrier::new(9));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let alias = shared.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    assert_eq!(alias.shape(), &[3]);
                    let TensorObservationData::F32(values) = alias.data() else {
                        unreachable!()
                    };
                    assert_eq!(values[0], 1.25);
                    assert_eq!(values[1].to_bits(), (-0.0f32).to_bits());
                    assert_eq!(values[2], f32::INFINITY);
                    drop(alias);
                })
            })
            .collect();
        drop(shared);
        assert_eq!(order.load(Ordering::SeqCst), 0);
        barrier.wait();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(order.load(Ordering::SeqCst), 2);
    }
    #[test]
    fn portable_observation_values_share_live_custody_and_deserialize_independently() {
        let retired = Arc::new(AtomicUsize::new(0));
        let tensor = SharedTensorObservation::retain(
            TensorObservation::new(vec![2], TensorObservationData::F32(vec![1.5, -3.0])).unwrap(),
            Retired(retired.clone()),
        );
        let original = ObservationValue::Tensor(tensor);
        let alias = original.clone();
        let wire = serde_json::to_value(&original).unwrap();
        let decoded: ObservationValue = serde_json::from_value(wire.clone()).unwrap();
        let (
            ObservationValue::Tensor(first),
            ObservationValue::Tensor(second),
            ObservationValue::Tensor(decoded_tensor),
        ) = (&original, &alias, &decoded)
        else {
            unreachable!()
        };
        assert!(first.same_storage(second));
        assert!(!first.same_storage(decoded_tensor));
        assert_eq!(decoded, original);
        assert_eq!(serde_json::to_value(decoded).unwrap(), wire);
        drop(original);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
