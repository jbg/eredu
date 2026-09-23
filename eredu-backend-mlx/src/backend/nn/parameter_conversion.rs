//! Reusable promotions owned by fully resident parameter materializations.
//!
//! The lookup table is weak: it never extends a model's residency. Original
//! immutable native identities keep edits, restoration and aliases distinct.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, OnceLock, Weak},
};

use safemlx::{error::Exception, Array, Dtype, Stream};

struct Conversion {
    _source: Array,
    state: Mutex<ConversionState>,
}

struct ConversionState {
    active: bool,
    value: Option<Array>,
}

type Registry = BTreeMap<usize, Weak<Conversion>>;
fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(Mutex::default)
}

/// Cache ownership follows one permanently resident device materialization.
/// Bounded-residency managers deliberately never enable this cache: their
/// admission ledger reserves original parameter storage only.
#[derive(Default)]
pub(crate) struct ResidentParameterConversions {
    entries: BTreeMap<usize, Arc<Conversion>>,
}

impl ResidentParameterConversions {
    pub(crate) fn register<'a>(&mut self, arrays: impl IntoIterator<Item = &'a Array>) {
        // Native descriptor access and shallow cloning happen before taking
        // the registry lock; that lock never encloses a native runtime call.
        let mut candidates = arrays
            .into_iter()
            .filter(|array| matches!(array.dtype(), Dtype::Bfloat16 | Dtype::Float16))
            .map(|array| (array.graph_identity(), Some(array.clone())))
            .collect::<Vec<_>>();
        let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
        registry.retain(|_, entry| entry.strong_count() > 0);
        for (identity, source) in &mut candidates {
            self.entries.entry(*identity).or_insert_with(|| {
                let entry = registry
                    .get(identity)
                    .and_then(Weak::upgrade)
                    .unwrap_or_else(|| {
                        Arc::new(Conversion {
                            _source: source.take().expect("registered source value"),
                            state: Mutex::new(ConversionState {
                                active: true,
                                value: None,
                            }),
                        })
                    });
                registry.insert(*identity, Arc::downgrade(&entry));
                entry
            });
        }
        drop(registry);
    }

    /// Retained conversion payload, separate from logical source parameters.
    /// One entry per immutable identity counts tied aliases only once.
    pub(crate) fn resident_bytes(&self) -> BTreeMap<usize, u64> {
        self.entries
            .iter()
            .map(|(&identity, entry)| {
                let state = entry
                    .state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                (
                    identity,
                    state
                        .value
                        .as_ref()
                        .map_or(0, |array| array.nbytes() as u64),
                )
            })
            .collect()
    }
}

impl Drop for ResidentParameterConversions {
    fn drop(&mut self) {
        let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
        for (identity, entry) in &self.entries {
            if Arc::strong_count(entry) == 1
                && registry
                    .get(identity)
                    .is_some_and(|registered| registered.ptr_eq(&Arc::downgrade(entry)))
            {
                registry.remove(identity);
            }
        }
    }
}

/// Revokes eligibility before publishing a replacement parameter value.
/// Clearing and unregistering prevents an old alias from repopulating a cache
/// after publication replaced the value represented by this residency owner.
/// Restoring the original value remains numerically exact and uses the ordinary
/// uncached path until it is materialized under a new resident owner.
pub(crate) fn invalidate(weight: &Array) {
    let identity = weight.graph_identity();
    let entry = registry()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .remove(&identity)
        .and_then(|entry| entry.upgrade());
    if let Some(entry) = entry {
        let mut state = entry
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.active = false;
        state.value = None;
    }
}

/// Match MLX's F32 promotion exactly, while reusing the result when its source
/// belongs to an enabled permanently resident owner. No cast is introduced for
/// BF16/F16 inputs, and unregistered/replacement parameters use ordinary matmul.
pub(crate) fn promoted_weight(
    input: &Array,
    weight: &Array,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    if input.dtype() != Dtype::Float32
        || !matches!(weight.dtype(), Dtype::Bfloat16 | Dtype::Float16)
    {
        return Ok(None);
    }
    let identity = weight.graph_identity();
    let entry = registry()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&identity)
        .and_then(Weak::upgrade);
    let Some(entry) = entry else {
        return Ok(None);
    };
    let mut state = entry
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !state.active {
        return Ok(None);
    }
    if state.value.is_none() {
        let converted = weight.as_dtype(Dtype::Float32, stream)?;
        // Materialize once before publishing across streams. This also detaches
        // the conversion graph from the source parameter after evaluation.
        safemlx::transforms::eval([&converted])?;
        state.value = Some(converted);
    }
    Ok(state.value.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType};

    #[test]
    fn parameter_conversion_reuses_aliases_preserves_values_and_expires_with_owner() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let input = Array::from_slice(&[0.375f32, -0.625, 1.125, 0.25], &[2, 2]);
        for dtype in [Dtype::Bfloat16, Dtype::Float16] {
            let weight = Array::from_slice(&[0.1234f32, -2.375, 0.53125, 1.0625], &[2, 2])
                .as_dtype(dtype, &stream)
                .unwrap();
            let alias = weight.clone();
            assert_eq!(weight.graph_identity(), alias.graph_identity());
            assert!(promoted_weight(&input, &weight, &stream).unwrap().is_none());
            let mut owner = ResidentParameterConversions::default();
            owner.register([&weight, &alias]);
            assert_eq!(owner.resident_bytes().values().sum::<u64>(), 0);
            let promoted = promoted_weight(&input, &weight, &stream).unwrap().unwrap();
            let reused = promoted_weight(&input, &alias, &stream).unwrap().unwrap();
            assert_eq!(promoted.graph_identity(), reused.graph_identity());
            assert_eq!(owner.resident_bytes().values().sum::<u64>(), 16);
            let expected = input
                .matmul(weight.transpose(&stream).unwrap(), &stream)
                .unwrap();
            let actual = input
                .matmul(promoted.transpose(&stream).unwrap(), &stream)
                .unwrap();
            assert_eq!(
                expected.evaluated().unwrap().as_slice::<f32>(),
                actual.evaluated().unwrap().as_slice::<f32>()
            );
            let narrow_input = input.as_dtype(dtype, &stream).unwrap();
            let gain = Array::from_slice(&[1.0f32, 1.0], &[2]);
            let normalized = crate::backend::nn::normalization::input_precision_rms(
                &narrow_input,
                &gain,
                1e-5,
                &stream,
            )
            .unwrap();
            assert_eq!(normalized.dtype(), Dtype::Float32);

            assert!(promoted_weight(&narrow_input, &weight, &stream)
                .unwrap()
                .is_none());
            let replacement = weight
                .add(Array::from_f32(0.5), &stream)
                .unwrap()
                .as_dtype(dtype, &stream)
                .unwrap();
            assert_ne!(replacement.graph_identity(), weight.graph_identity());
            assert!(promoted_weight(&input, &replacement, &stream)
                .unwrap()
                .is_none());
            assert_eq!(
                promoted_weight(&input, &weight, &stream)
                    .unwrap()
                    .unwrap()
                    .graph_identity(),
                promoted.graph_identity()
            );
            // Publication revokes retired conversions so they cannot reduce
            // the future-cast allowance while an edit is active. Merely
            // borrowing a mutable slot for catalog inspection preserves reuse.
            use crate::module::{PhysicalParam, PhysicalParameter};
            use eredu_nn::Tensor;
            let mut physical = PhysicalParam::new(weight.clone());
            let _ = physical.as_nested_value_mut();
            assert_eq!(owner.resident_bytes().values().sum::<u64>(), 16);
            let mut slot = crate::MlxTensor::from_array(weight.clone());
            slot.publish_parameter(&crate::MlxTensor::from_array(replacement));
            assert_eq!(owner.resident_bytes().values().sum::<u64>(), 0);
            assert!(promoted_weight(&input, &alias, &stream).unwrap().is_none());
            slot.publish_parameter(&crate::MlxTensor::from_array(weight.clone()));
            assert!(promoted_weight(&input, slot.as_array(), &stream)
                .unwrap()
                .is_none());
            let mut replacement_owner = ResidentParameterConversions::default();
            replacement_owner.register([&weight]);
            let newly_cached = promoted_weight(&input, &weight, &stream).unwrap().unwrap();
            assert_eq!(owner.resident_bytes().values().sum::<u64>(), 0);
            assert_eq!(replacement_owner.resident_bytes().values().sum::<u64>(), 16);
            drop(owner);
            assert_eq!(
                promoted_weight(&input, &weight, &stream)
                    .unwrap()
                    .unwrap()
                    .graph_identity(),
                newly_cached.graph_identity()
            );
            drop(replacement_owner);
            assert!(promoted_weight(&input, &weight, &stream).unwrap().is_none());
        }
    }
}
