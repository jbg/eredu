//! Portable external drafting cache and observer ownership.

use crate::{SpeculativeCaptureEnvelope, SpeculativeCaptureMetadata, SpeculativeIdentity};

/// Portable external target-cache envelope.
///
/// The native cache remains opaque. Prepared-input ownership, the selected realization, and the
/// committed target frontier are portable lifecycle state and therefore live above every backend.
pub struct ExternalAssistantCache<C> {
    native: C,
    selected: crate::SelectedSpeculativeRealization,
    prepared_input: Option<SpeculativeIdentity>,
    frontier: u64,
}

impl<C> ExternalAssistantCache<C> {
    /// Couples one opaque native cache to the realization selected before materialization.
    pub fn new(native: C, selected: crate::SelectedSpeculativeRealization) -> Self {
        Self {
            native,
            selected,
            prepared_input: None,
            frontier: 0,
        }
    }

    /// Borrows opaque backend state for one native mechanism call.
    pub const fn native(&self) -> &C {
        &self.native
    }

    /// Mutably borrows opaque backend state for one native mechanism call.
    pub const fn native_mut(&mut self) -> &mut C {
        &mut self.native
    }

    /// Binds one prepared semantic input to the reusable lane cache.
    pub fn bind_prepared_input(&mut self, identity: SpeculativeIdentity) -> Result<(), String> {
        match self.prepared_input.as_ref() {
            Some(bound) if bound != &identity => {
                Err("external speculative cache belongs to a different prepared input".into())
            }
            Some(_) => Ok(()),
            None => {
                self.prepared_input = Some(identity);
                Ok(())
            }
        }
    }

    /// Derives and binds the speculative identity of one portable prepared-input cache key.
    pub fn bind_prepared_input_cache_identity(
        &mut self,
        prepared: &crate::PreparedInputCacheIdentity,
    ) -> Result<(), String> {
        let identity = SpeculativeIdentity::new(format!(
            "prepared-input/{}",
            prepared.prefix_content_fingerprint()
        ))
        .map_err(|error| error.to_string())?;
        self.bind_prepared_input(identity)
    }

    /// Advances the portable committed frontier from a backend-reported native offset.
    pub fn advance_frontier(&mut self, frontier: i32) -> Result<(), String> {
        let frontier = u64::try_from(frontier)
            .map_err(|_| "external target cache frontier is negative".to_owned())?;
        if frontier < self.frontier {
            return Err(format!(
                "external target cache frontier regressed from {} to {frontier}",
                self.frontier
            ));
        }
        self.frontier = frontier;
        Ok(())
    }

    /// Returns the committed target frontier.
    pub fn frontier(&self) -> Result<i32, String> {
        i32::try_from(self.frontier)
            .map_err(|_| "external target cache frontier exceeds i32".to_owned())
    }

    /// Returns architecture-declared capture paths in their validated tensor order.
    pub fn capture_paths(&self) -> Vec<&str> {
        self.selected
            .requirements()
            .capture()
            .entries()
            .iter()
            .map(|entry| entry.path().as_str())
            .collect()
    }

    /// Closes and validates one ordered architecture capture at the current frontier.
    pub fn validate_capture_shapes(&self, shapes: &[Vec<usize>]) -> Result<(), String> {
        let prepared_input = self
            .prepared_input
            .clone()
            .ok_or_else(|| "external target capture precedes prepared-input binding".to_owned())?;
        let schema = self
            .selected
            .requirements()
            .capture()
            .instantiate(shapes.iter().cloned())
            .map_err(|error| format!("{error}; received {shapes:?}"))?;
        let values = vec![(); schema.entries().len()];
        let envelope = SpeculativeCaptureEnvelope::new(
            SpeculativeCaptureMetadata::new(schema, self.frontier),
            values,
        )
        .map_err(|error| error.to_string())?;
        let lane = self.selected.lane_identity(prepared_input, self.frontier);
        self.selected
            .validate_capture(&lane, &envelope)
            .map_err(|error| error.to_string())
    }

    /// Heap and envelope storage copied with a native checkpoint.
    pub fn control_metadata_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<ExternalAssistantCacheCheckpoint<C>>() as u64).checked_add(
            self.prepared_input
                .as_ref()
                .map_or(0, |id| id.as_str().len() as u64),
        )
    }

    /// Wraps one opaque native checkpoint in the exact semantic cache boundary.
    pub fn checkpoint<N>(&self, native: N) -> ExternalAssistantCacheCheckpoint<N> {
        ExternalAssistantCacheCheckpoint {
            native,
            prepared_input: self.prepared_input.clone(),
            frontier: self.frontier,
        }
    }

    /// Restores the semantic half after its opaque native checkpoint was restored.
    pub fn restore_semantics<N>(&mut self, checkpoint: &ExternalAssistantCacheCheckpoint<N>) {
        self.prepared_input.clone_from(&checkpoint.prepared_input);
        self.frontier = checkpoint.frontier;
    }
}

/// Exact portable checkpoint envelope around opaque native cache storage.
pub struct ExternalAssistantCacheCheckpoint<C> {
    native: C,
    prepared_input: Option<SpeculativeIdentity>,
    frontier: u64,
}

impl<C> ExternalAssistantCacheCheckpoint<C> {
    /// Borrows the opaque backend checkpoint.
    pub const fn native(&self) -> &C {
        &self.native
    }

    /// Returns the committed frontier retained by this checkpoint.
    pub fn frontier(&self) -> Result<i32, String> {
        i32::try_from(self.frontier)
            .map_err(|_| "external target checkpoint frontier exceeds i32".to_owned())
    }
}

/// Explicit production observers for external-assistant activations and logits.
///
/// They are installed on a reusable materialized assistant and invoked by its architecture-owned
/// executor. The default is the neutral no-op observer used by ordinary generation.
pub struct ExternalAssistantObservers<T, L, E> {
    tensors: Box<dyn crate::ActivationObserver<T, E>>,
    logits: Box<dyn crate::ActivationObserver<L, E>>,
}

impl<T, L, E> Default for ExternalAssistantObservers<T, L, E>
where
    E: 'static,
{
    fn default() -> Self {
        Self {
            tensors: Box::new(crate::NoopObserver),
            logits: Box::new(crate::NoopObserver),
        }
    }
}

impl<T, L, E> ExternalAssistantObservers<T, L, E> {
    /// Installs explicit typed observers used by the production executor.
    pub fn new(
        tensors: impl crate::ActivationObserver<T, E> + 'static,
        logits: impl crate::ActivationObserver<L, E> + 'static,
    ) -> Self {
        Self {
            tensors: Box::new(tensors),
            logits: Box::new(logits),
        }
    }

    /// Observes and optionally replaces one tensor activation.
    pub fn observe_tensor(&mut self, path: &str, value: &T) -> Result<T, E>
    where
        T: Clone,
    {
        crate::observe_and_intervene(self.tensors.as_mut(), path, value)
    }

    /// Observes and optionally replaces one logits value.
    pub fn observe_logits(&mut self, path: &str, value: &L) -> Result<L, E>
    where
        L: Clone,
    {
        crate::observe_and_intervene(self.logits.as_mut(), path, value)
    }
}
