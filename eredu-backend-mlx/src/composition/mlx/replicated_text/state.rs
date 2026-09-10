use super::*;

pub(crate) trait MlxStateMechanisms: LayerRuntimeState<MlxNeuralBackend> + Sized {
    fn supports_isolated_snapshot(&self) -> bool {
        false
    }
    fn isolated_snapshot(&self, _stream: &Stream) -> Result<Self, Exception> {
        Err(Exception::custom(
            "complete isolated snapshot is unsupported for this state realization",
        ))
    }
    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        Some(0)
    }
    fn isolated_snapshot_auxiliary_growth(&self, _additional: u64) -> Option<u64> {
        Some(0)
    }
    fn isolated_snapshot_estimate(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        if !self.supports_isolated_snapshot() {
            return None;
        }
        let mut retained = self
            .layout()
            .logical_metadata_bytes()?
            .checked_add(u64::try_from(std::mem::size_of::<Self>()).ok()?)?
            .checked_add(u64::try_from(self.layout().len()).ok()?.checked_mul(1024)?)?;
        for layer in 0..self.layout().len() {
            // Native role maps also retain slots whose tensor is still absent.
            retained = retained.checked_add(
                u64::try_from(self.layout().components(layer)?.len())
                    .ok()?
                    .checked_mul(256)?,
            )?;
        }
        for array in self.retained_arrays() {
            // Logical data plus conservative native-array descriptor/shape allowance.
            // Include the destination and a possible contiguous materialization.
            retained = retained
                .checked_add(u64::try_from(array.nbytes()).ok()?.checked_mul(2)?)?
                .checked_add(4096)?
                .checked_add(u64::try_from(array.shape().len()).ok()?.checked_mul(16)?)?;
        }
        retained = retained.checked_add(self.isolated_snapshot_auxiliary_bytes()?)?;
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: retained,
            copy_bytes: retained,
        })
    }
    fn continuation_capacity_bound(&self, _additional: u64) -> Option<u64> {
        None
    }
    fn isolated_snapshot_growth(&self, additional: u64) -> Option<u64> {
        if !self.supports_isolated_snapshot() {
            return None;
        }
        // Ordinary text is a single sequence. Interpret only the declared
        // component geometry, including absent fixed tensors. MLX floating
        // storage is at most eight bytes; explicit integer components use four.
        // Charge the full future payload as an additional conservative allowance
        // (rather than subtracting currently retained data), including the same
        // materialization/descriptor allowance as an immutable copy.
        use eredu_core::cache::{StateTensorDimension as Dim, StateTensorDtype};
        let absolute = u64::try_from(self.offset()).ok()?.checked_add(additional)?;
        i32::try_from(absolute).ok()?;
        let prefix = absolute.max(self.continuation_capacity_bound(additional)?);
        i32::try_from(prefix).ok()?;
        let mut bytes = 0u64;
        for layer in 0..self.layout().len() {
            for component in self.layout().components(layer)? {
                let mut elements = 1u64;
                for dimension in component.shape() {
                    let extent = match dimension {
                        Dim::Batch | Dim::Scalar => 1,
                        Dim::Fixed(n) => u64::from(n.get()),
                        Dim::PrefixTokens => prefix,
                        Dim::PrefixTokensDiv(n) => prefix / u64::from(n.get()),
                        // The final remainder is not the maximum over a run.
                        Dim::PrefixTokensRem(n) => prefix.min(u64::from(n.get()) - 1),
                    };
                    elements = elements.checked_mul(extent)?;
                }
                let width = match component.dtype() {
                    StateTensorDtype::Floating => 8,
                    StateTensorDtype::Float32
                    | StateTensorDtype::Int32
                    | StateTensorDtype::Uint32 => 4,
                };
                bytes = bytes
                    .checked_add(elements.checked_mul(width)?.checked_mul(2)?)?
                    .checked_add(4096)?
                    .checked_add(
                        u64::try_from(component.shape().len())
                            .ok()?
                            .checked_mul(16)?,
                    )?;
            }
        }
        bytes.checked_add(self.isolated_snapshot_auxiliary_growth(additional)?)
    }
    fn offset(&self) -> i32;
    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error>;
    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error>;
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error>;
    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception>;
    fn retained_arrays(&self) -> Vec<&Array>;
    fn deep_checkpoint(&self) -> Result<Self, Exception>;
    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception>;
    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception>;
    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>;
    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    >;
    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception>;
}

pub(super) fn fork_mlx_prediction_target_state<S: MlxStateMechanisms>(
    state: &S,
    stream: &Stream,
) -> Result<S, Error> {
    state
        .fork_prediction_target_state(stream)
        .map_err(Into::into)
}

pub(super) fn selected_state_manager(
    selected: &SelectedStateRealization,
) -> Result<Option<CacheResidencyManager>, Error> {
    let needs_paging = selected
        .components()
        .iter()
        .any(|component| component.placement() == StateComponentPlacement::Paged);
    match (needs_paging, selected.policy()) {
        (_, CacheResidencyPolicy::Paged(options)) => CacheResidencyManager::new(options.clone())
            .map(Some)
            .map_err(|error| Error::Parallel(error.to_string())),
        (false, CacheResidencyPolicy::Device) => Ok(None),
        (true, CacheResidencyPolicy::Device) => Err(Error::Parallel(
            "selected paged state component has no paging policy".into(),
        )),
    }
}

impl MlxStateMechanisms for MlxKeyValueState {
    fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.continuation_capacity_bound(additional)
    }
    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        self.isolated_snapshot_auxiliary_bytes()
    }
    fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        self.isolated_snapshot_auxiliary_growth(additional)
    }

    fn supports_isolated_snapshot(&self) -> bool {
        self.supports_isolated_snapshot()
    }
    fn isolated_snapshot(&self, stream: &Stream) -> Result<Self, Exception> {
        self.isolated_snapshot(stream)
    }
    fn offset(&self) -> i32 {
        MlxKeyValueState::offset(self)
    }

    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected)?;
        MlxKeyValueState::from_selected_with_global_layer_start(
            selected,
            manager,
            rank,
            global_layer_start,
        )
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        _stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = MlxKeyValueState::from_selected_with_global_layer_start(
            selected,
            Some(manager),
            expected.topology().cache_rank_identity(),
            identity.global_layer_start(),
        )?;
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        MlxKeyValueState::save_prompt_cache(
            self,
            destination,
            descriptor,
            prefix_token_ids,
            options,
        )
        .map_err(Into::into)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        MlxKeyValueState::residency_report(self)
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        MlxKeyValueState::retained_arrays(self)
    }

    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        self.deep_clone_state()
    }

    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        MlxKeyValueState::fork_prediction_target_state(self, stream)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        MlxKeyValueState::restore_checkpoint(self, checkpoint, stream)
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.as_ref()
            .iter()
            .map(|layer| (eredu_nn::AttentionCache::offset(layer), Vec::new()))
            .collect()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        Ok(Vec::new())
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        Ok(Vec::new())
    }
}

impl MlxStateMechanisms for MlxHybridState {
    fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.continuation_capacity_bound(additional)
    }
    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        self.isolated_snapshot_auxiliary_bytes()
    }
    fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        self.isolated_snapshot_auxiliary_growth(additional)
    }
    fn supports_isolated_snapshot(&self) -> bool {
        self.supports_isolated_snapshot()
    }
    fn isolated_snapshot(&self, stream: &Stream) -> Result<Self, Exception> {
        self.isolated_snapshot(stream)
    }
    fn offset(&self) -> i32 {
        MlxHybridState::offset(self)
    }

    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected)?;
        MlxHybridState::from_selected_with_global_layer_start(
            selected,
            manager,
            rank,
            global_layer_start,
        )
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut state = MlxHybridState::from_selected_with_global_layer_start(
            selected,
            Some(manager),
            expected.topology().cache_rank_identity(),
            identity.global_layer_start(),
        )?;
        state.restore_prompt_cache_state(
            tensors,
            i32::try_from(prefix_token_ids.len())
                .map_err(|_| Error::Parallel("prompt-cache prefix exceeds i32".into()))?,
            identity.layer_prefix_offsets(),
        )?;
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        MlxHybridState::save_prompt_cache(self, destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        MlxHybridState::residency_report(self)
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        MlxHybridState::retained_arrays(self)
    }

    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        self.deep_clone_state()
    }

    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        MlxHybridState::fork_prediction_target_state(self, stream)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        MlxHybridState::restore_checkpoint(self, checkpoint, stream)
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.semantic_snapshot()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        self.fixed_numeric_snapshot()
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        self.retained_numeric_snapshot()
    }
}

impl MlxStateMechanisms for MlxPoolingAttentionState {
    fn offset(&self) -> i32 {
        self.as_ref().first().map_or(0, |layer| layer.offset())
    }

    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected)?;
        match manager {
            Some(manager) => MlxPoolingAttentionStateFactory::paged(
                selected.layout().clone(),
                manager,
                global_layer_start,
                0,
                rank,
            ),
            None => MlxPoolingAttentionStateFactory::device(selected.layout().clone()),
        }
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let prefix = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("prompt-cache prefix exceeds i32".into()))?;
        let mut state = MlxPoolingAttentionStateFactory::paged(
            selected.layout().clone(),
            manager,
            identity.global_layer_start(),
            prefix,
            expected.topology().cache_rank_identity(),
        )?;
        let mut tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .into_iter()
            .map(|tensor| ((tensor.owner, tensor.role), tensor.array))
            .collect::<BTreeMap<_, _>>();
        for (layer, cache) in state.as_mut().iter_mut().enumerate() {
            let processed = prefix
                .checked_add(identity.layer_prefix_offsets()[layer])
                .ok_or_else(|| Error::Parallel("prompt-cache layer offset overflowed".into()))?;
            cache.restore_prompt_cache_state(
                identity.global_layer_start() + layer,
                &mut tensors,
                processed,
            )?;
        }
        if !tensors.is_empty() {
            return Err(Error::Parallel(
                "prompt cache contains unexpected state tensors".into(),
            ));
        }
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let mut manager = None;
        for layer in self.as_mut() {
            layer.finalize()?;
            manager.get_or_insert_with(|| layer.residency_manager().cloned());
        }
        let fixed = self
            .as_ref()
            .iter()
            .enumerate()
            .flat_map(|(layer, cache)| cache.prompt_cache_state_arrays(layer))
            .collect::<Vec<_>>();
        manager
            .flatten()
            .ok_or_else(|| Error::Parallel("prompt-cache persistence requires paged state".into()))?
            .save_prompt_cache(destination, descriptor, prefix_token_ids, &fixed, options)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.as_ref()
            .iter()
            .find_map(MlxPoolingAttentionCache::residency_manager)
            .map(CacheResidencyManager::report)
            .transpose()
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        self.as_ref()
            .iter()
            .flat_map(MlxPoolingAttentionCache::retained_arrays)
            .collect()
    }

    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        eredu_runtime::DeviceState::create(self.layout().clone(), |layer, _| {
            self.as_ref()[layer].deep_clone_state()
        })
    }

    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        MlxPoolingAttentionStateFactory::fork_prediction_target_state(self, stream)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        if self.layout() != checkpoint.layout() || self.as_ref().len() != checkpoint.as_ref().len()
        {
            return Err(Exception::custom(
                "pooling-attention checkpoint layout does not match canonical state",
            ));
        }
        for (current, previous) in self.as_mut().iter_mut().zip(checkpoint.as_ref()) {
            PoolingAttentionCache::restore(current, previous, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.as_ref()
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let present = layer
                    .prompt_cache_state_arrays(index)
                    .into_iter()
                    .map(|state| state.role)
                    .collect::<std::collections::BTreeSet<_>>();
                let components = self
                    .layout()
                    .components(index)
                    .expect("pooling state layout contains each realized layer")
                    .iter()
                    .filter_map(|component| match component.role() {
                        eredu_core::cache::StateComponentRole::Fixed(role) => {
                            Some((role, present.contains(&role)))
                        }
                        _ => None,
                    })
                    .collect();
                (PoolingAttentionCache::offset(layer), components)
            })
            .collect()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        let mut snapshot = Vec::new();
        for (layer, cache) in self.as_ref().iter().enumerate() {
            for state in cache.prompt_cache_state_arrays(layer) {
                let evaluated = state.array.evaluated()?;
                snapshot.push((
                    layer,
                    state.role,
                    state.array.shape().to_vec(),
                    evaluated.as_slice::<f32>().to_vec(),
                ));
            }
        }
        Ok(snapshot)
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        self.retained_arrays()
            .into_iter()
            .map(|array| {
                let evaluated = array.evaluated()?;
                Ok((array.shape().to_vec(), evaluated.as_slice::<f32>().to_vec()))
            })
            .collect()
    }
}
