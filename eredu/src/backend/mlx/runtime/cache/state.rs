//! Reusable MLX realization of architecture-declared key/value state.

use std::path::Path;

use eredu_core::cache::{
    LayerCachePolicy, PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions,
};
use eredu_runtime::{
    CacheResidencyReport, RuntimeLayerState, RuntimeState, StateError, StateLayout,
};
use safemlx::{error::Exception, Array, Stream};

use crate::{
    backend::mlx::{
        nn::shared::MlxBackend,
        runtime::cache::{
            kv::{ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, RetainedArrayIter},
            residency::CacheResidencyManager,
        },
    },
    core::cache::CacheRankIdentity,
};

/// One concrete MLX key/value layer state selected by runtime residency policy.
#[derive(Debug, Clone)]
pub enum MlxKeyValueLayerState {
    /// Contiguous execution-device keys and values.
    Device(ConcatKeyValueCache),
    /// Block-addressable keys and values managed by finite residency budgets.
    Paged(PagedKeyValueCache),
}

impl MlxKeyValueLayerState {
    fn clear(&mut self) -> Result<(), Exception> {
        match self {
            Self::Device(cache) => {
                cache.clear();
                Ok(())
            }
            Self::Paged(cache) => cache.clear(),
        }
    }
}

impl KeyValueCache for MlxKeyValueLayerState {
    fn offset(&self) -> i32 {
        match self {
            Self::Device(cache) => cache.offset(),
            Self::Paged(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Device(cache) => cache.max_size(),
            Self::Paged(cache) => cache.max_size(),
        }
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Device(cache) => cache.retained_arrays(),
            Self::Paged(cache) => cache.retained_arrays(),
        }
    }

    fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_))
    }

    fn paged_attention(
        &mut self,
        queries: &Array,
        scale: f32,
        mask: Option<&Array>,
        sinks: Option<&Array>,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        match self {
            Self::Device(cache) => cache.paged_attention(queries, scale, mask, sinks, stream),
            Self::Paged(cache) => cache.paged_attention(queries, scale, mask, sinks, stream),
        }
    }

    fn update_for_attention(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Device(cache) => cache.update_for_attention(keys, values, stream),
            Self::Paged(cache) => cache.update_for_attention(keys, values, stream),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Device(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Paged(cache) => cache.update_and_fetch(keys, values, stream),
        }
    }
}

impl RuntimeLayerState<MlxBackend> for MlxKeyValueLayerState {
    type RetainedValues<'a> = RetainedArrayIter<'a>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        match self {
            Self::Device(cache) => RuntimeLayerState::<MlxBackend>::retained_values(cache),
            Self::Paged(cache) => RuntimeLayerState::<MlxBackend>::retained_values(cache),
        }
    }
}

/// Model-wide MLX key/value state created solely from a neutral layout.
#[derive(Debug, Clone)]
pub struct MlxKeyValueState {
    layout: StateLayout,
    layers: Vec<MlxKeyValueLayerState>,
}

impl MlxKeyValueState {
    /// Creates contiguous execution-device state for every declared layer.
    pub fn device(layout: StateLayout) -> Result<Self, Exception> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                let window = key_value_window(layer, policy)?;
                Ok(MlxKeyValueLayerState::Device(match window {
                    Some(window) => ConcatKeyValueCache::new_for_sliding_attention(window),
                    None => ConcatKeyValueCache::new(),
                }))
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        Ok(Self { layout, layers })
    }

    /// Creates block-addressable state using one shared residency manager.
    pub fn paged(
        layout: StateLayout,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                let window = key_value_window(layer, policy)?;
                PagedKeyValueCache::new_with_layout(manager.clone(), layer, window, 0, rank)
                    .map(MlxKeyValueLayerState::Paged)
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        Ok(Self { layout, layers })
    }

    /// Returns the common absolute token offset, or zero for an empty state.
    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, KeyValueCache::offset)
    }

    /// Clears retained arrays without changing residency or attention windows.
    pub fn clear(&mut self) -> Result<(), Exception> {
        for layer in &mut self.layers {
            layer.clear()?;
        }
        Ok(())
    }

    /// Returns aggregate telemetry when this is paged state.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.layers
            .iter()
            .find_map(|layer| match layer {
                MlxKeyValueLayerState::Paged(cache) => Some(cache.report()),
                MlxKeyValueLayerState::Device(_) => None,
            })
            .transpose()
    }

    /// Finalizes paged tails and atomically persists a completed prefix.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Exception> {
        let mut manager = None;
        for layer in &mut self.layers {
            let MlxKeyValueLayerState::Paged(cache) = layer else {
                return Err(Exception::custom(
                    "prompt-cache persistence requires explicitly paged state",
                ));
            };
            cache.finalize()?;
            manager.get_or_insert_with(|| cache.manager().clone());
        }
        manager
            .ok_or_else(|| Exception::custom("cannot persist empty paged state"))?
            .save_prompt_cache(destination, descriptor, prefix_token_ids, &[], options)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

impl RuntimeState<MlxBackend> for MlxKeyValueState {
    type LayerState = MlxKeyValueLayerState;

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError> {
        let count = self.layers.len();
        self.layers
            .get_mut(layer)
            .ok_or(StateError::UnknownLayer { layer, count })
    }

    fn retained_values(
        &self,
        layer: usize,
    ) -> Result<<Self::LayerState as RuntimeLayerState<MlxBackend>>::RetainedValues<'_>, StateError>
    {
        self.layers
            .get(layer)
            .map(RuntimeLayerState::retained_values)
            .ok_or(StateError::UnknownLayer {
                layer,
                count: self.layers.len(),
            })
    }
}

impl AsRef<[MlxKeyValueLayerState]> for MlxKeyValueState {
    fn as_ref(&self) -> &[MlxKeyValueLayerState] {
        &self.layers
    }
}

impl AsMut<[MlxKeyValueLayerState]> for MlxKeyValueState {
    fn as_mut(&mut self) -> &mut [MlxKeyValueLayerState] {
        &mut self.layers
    }
}

fn key_value_window(layer: usize, policy: &LayerCachePolicy) -> Result<Option<i32>, Exception> {
    match policy {
        LayerCachePolicy::KeyValue { attention, .. } => attention
            .sliding_window_i32()
            .map_err(|error| Exception::custom(error.to_string())),
        _ => Err(Exception::custom(format!(
            "MLX key/value state cannot realize non-key/value policy at layer {layer}: {policy:?}"
        ))),
    }
}
