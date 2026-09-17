//! Cold composition of storage owned by the enclosing native session payload.

use super::{MlxModelSession, SessionPayload};
use crate::backend::error::Error;
use crate::backend::runtime::residency::storage::RetainedStorage;
use crate::composition::mlx::model::RetainedIdleModelStorage;

impl MlxModelSession {
    /// Called only while constructing a fresh session, after its initial reset
    /// has settled. Publication registers existing storage; future operations
    /// still enter through operation_memory and acquire their own authority.
    pub(super) fn publish_initial_idle_storage(&mut self) -> Result<bool, Error> {
        // The outer owner contains declarations, layouts, identities and
        // accounting metadata. PreparedModelDiscovery's deferred fingerprint
        // keeps paths/file metadata, not source payloads or open files. Capture
        // layout caches contain topology declarations, not captured tensors.
        // Keep the pattern exhaustive so new outer owners require classification.
        let Self {
            payload: _,
            poison: _,
            failure: _,
            authority: _,
            capacity_handoffs: _,
            floating_state_dtype_bytes: _,
            capabilities: _,
            capture_discovery: _,
            speculative_capture_layouts: _,
            partition_capture: _,
            intervention_session_identity: _,
            state_residency: _,
        } = self;
        // Do not reap, evaluate or materialize during this checked transition.
        self.ensure_healthy()?;
        self.authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let Some(payload) = self.payload.get_mut() else {
            return Ok(false);
        };
        let Some(owner) = payload._memory_owner.clone() else {
            return Ok(false);
        };
        // The initial reset only retains clones of the loading authority. A
        // later or independently created authority must never be discarded by
        // this construction-only transition.
        if !payload.operation_memory.borrow().retains_only(&owner) {
            return Ok(false);
        }
        let enclosing = payload.retained_enclosing_storage()?;
        if !payload
            .model
            .publish_initial_idle_storage(enclosing, &owner)?
        {
            return Ok(false);
        }
        payload._memory_owner = None;
        *payload.operation_memory.get_mut() = Default::default();
        // The local owner guard retires last. Immutable clones held by loader
        // recovery, extracted values or other descendants remain authoritative.
        Ok(true)
    }
}

impl SessionPayload {
    /// Collects this SessionPayload while the caller preserves an idle session.
    ///
    /// This inspection neither enters native work nor changes accounting
    /// authority. The executable checks its own idle state and keeps decoder
    /// storage separate from model, source and parameter-overlay storage.
    /// This is not a complete MlxModelSession inventory: its outer discovery,
    /// capture-layout caches and other fields require their own classification.
    /// The caller must also check the session's submission authority without
    /// turning this inspection into a reaping or native preparation operation.
    pub(super) fn retained_idle_storage(&self) -> Result<RetainedIdleModelStorage, Error> {
        self.model
            .retained_idle_storage(Some(self.retained_enclosing_storage()?))
    }

    /// Uses inventories prepared by the current Work before entering a producer.
    /// Successful prefixes stay in these destinations on every later refusal.
    pub(super) fn collect_retained_idle_storage(
        &self,
        nonstate: &mut RetainedStorage,
        decoder: &mut RetainedStorage,
    ) -> Result<(), Error> {
        self.collect_retained_enclosing_storage(nonstate)?;
        self.model.collect_retained_idle_storage(nonstate, decoder)
    }

    /// Covers only owners outside the executable, with no full-model claim.
    fn retained_enclosing_storage(&self) -> Result<RetainedStorage, Error> {
        let mut enclosing = RetainedStorage::default();
        self.collect_retained_enclosing_storage(&mut enclosing)?;
        Ok(enclosing)
    }

    fn collect_retained_enclosing_storage(
        &self,
        enclosing: &mut RetainedStorage,
    ) -> Result<(), Error> {
        // Keep this exhaustive: adding another payload owner requires deciding
        // how its storage participates in the enclosing inventory.
        let Self {
            model: _,
            parameter_state,
            target,
            distributed,
            #[cfg(any(feature = "image", feature = "audio"))]
            processor,
            #[cfg(test)]
            _retirement_probe,
            _memory_owner: _,
            state_memory: _,
            memory_pool: _,
            operation_memory: _,
            nonstate_publication: _,
        } = self;
        parameter_state.collect_retained_storage(enclosing)?;

        // Setup captured each actual fixed communicator buffer source. Borrow
        // those descriptors without native queries, payload copies or inferred
        // zero coverage from a missing outer wrapper.
        target.collect_retained_buffer(enclosing)?;
        if let Some(distributed)=distributed {distributed.collect_retained_buffers(enclosing)?;}
        // Consume the actual architecture-owned declaration. Inline policy is
        // already part of this enclosing metadata object; media inputs, feature
        // buffers and native conversion results have independent request owners.
        #[cfg(any(feature = "image", feature = "audio"))]
        if let Some(processor) = processor {
            match processor.retained_storage() {
                eredu_architectures::processor_execution::PreparedProcessorStorage::Inline { .. } => {}
            }
        }
        #[cfg(test)]
        if _retirement_probe.is_some() {
            // A test-injected Any owner can itself retain numerical payload.
            enclosing.mark_incomplete();
        }

        // NativeMemoryOwner, NativeMemoryRetention and WorkingMemoryPool retain
        // accounting authority, not an additional tensor/source payload owned by
        // this session. Their handles and bookkeeping are metadata; the actual
        // model and state allocations are traversed by the executable below.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::tests::support::path_instrumentation as paths;
    use eredu_runtime::working_memory::WorkingMemoryPool;

    fn fixture() -> (Stream, ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let backend = MlxBackend::new(&stream, &stream)
            .with_memory_pool(WorkingMemoryPool::new(u64::MAX, 0).unwrap());
        let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
        let model = eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default())
            .unwrap();
        let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        (stream, runtime, root)
    }

    #[test]
    fn idle_payload_inventory_is_cold_and_keeps_decoder_storage_separate() {
        let (_stream, mut runtime, _root) = fixture();
        let payload = &runtime.session().payload;
        let before_state = payload.model.erased().state_snapshot();
        let before_paths = paths::snapshot();
        let before_owners = payload.memory_pool.unquoted_owner_count().unwrap();
        let before_used = payload.memory_pool.used_bytes().unwrap();
        let before_peak = payload.memory_pool.peak_bytes().unwrap();
        assert_eq!(
            payload
                .retained_enclosing_storage()
                .unwrap()
                .byte_bound()
                .unwrap(),
            Some(0)
        );
        let empty = payload.retained_idle_storage().unwrap();
        let nonstate = empty.nonstate_bytes().unwrap().unwrap();
        assert!(nonstate > 0);
        assert_eq!(empty.decoder_state_bytes().unwrap(), Some(0));
        assert!(empty.has_empty_decoder_storage().unwrap());
        assert_eq!(payload.model.erased().state_snapshot(), before_state);
        assert_eq!(paths::snapshot(), before_paths);
        assert_eq!(
            payload.memory_pool.unquoted_owner_count().unwrap(),
            before_owners
        );
        assert_eq!(payload.memory_pool.used_bytes().unwrap(), before_used);
        assert_eq!(payload.memory_pool.peak_bytes().unwrap(), before_peak);
        drop(empty);

        let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2, 3]).unwrap();
        let identity_bytes = prompt
            .shared_cache_identity()
            .unwrap()
            .capacity_bytes()
            .unwrap();
        let output = runtime.prefill(prompt).unwrap().wait().unwrap();
        assert!(output
            .logits()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .any(|value| value.abs() > 1e-6));
        drop(output);
        runtime.session().ensure_no_submission_in_flight().unwrap();
        let payload = &runtime.session().payload;
        let before_state = payload.model.erased().state_snapshot();
        let before_paths = paths::snapshot();
        let populated = payload.retained_idle_storage().unwrap();
        assert_eq!(
            populated.nonstate_bytes().unwrap(),
            Some(nonstate + identity_bytes)
        );
        assert!(populated.decoder_state_bytes().unwrap().unwrap() > 0);
        assert!(!populated.has_empty_decoder_storage().unwrap());
        assert_eq!(payload.model.erased().state_snapshot(), before_state);
        assert_eq!(paths::snapshot(), before_paths);
    }

    #[test]
    fn initialized_communication_and_target_aliases_preserve_complete_buffer_coverage() {
        let (stream, mut runtime, _root) = fixture();
        let world =
            safemlx::distributed::Group::init(false, safemlx::distributed::Backend::Ring).unwrap();
        let manifest = eredu_runtime::CommunicationManifest::new(1, 0, vec![], vec![])
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
        let communication =
            MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
        let payload = runtime.session_mut().payload.get_mut().unwrap();
        assert_eq!(
            payload
                .retained_enclosing_storage()
                .unwrap()
                .byte_bound()
                .unwrap(),
            Some(0)
        );
        let baseline=payload.retained_idle_storage().unwrap().nonstate_bytes().unwrap();
        payload.distributed = Some(communication.clone());
        assert_eq!(
            payload
                .retained_enclosing_storage()
                .unwrap()
                .byte_bound()
                .unwrap(),
            Some(0)
        );
        let distributed = payload.retained_idle_storage().unwrap();
        assert_eq!(distributed.nonstate_bytes().unwrap(), baseline);
        assert_eq!(distributed.decoder_state_bytes().unwrap(), Some(0));
        drop(distributed);

        // Target identity can retain the communicator independently of the
        // outer communication owner; the same fixed source remains valid.
        payload.target =
            crate::backend::MlxPreparedTarget::new(&stream, Some(&communication)).unwrap();
        payload.distributed = None;
        assert_eq!(
            payload
                .retained_enclosing_storage()
                .unwrap()
                .byte_bound()
                .unwrap(),
            Some(0)
        );
        let target_only = payload.retained_idle_storage().unwrap();
        assert_eq!(target_only.nonstate_bytes().unwrap(), baseline);
        assert_eq!(target_only.decoder_state_bytes().unwrap(), Some(0));
    }

    #[test]
    fn erased_test_owner_is_not_reported_as_empty_storage() {
        let (_stream, mut runtime, _root) = fixture();
        runtime
            .session_mut()
            .set_retirement_probe(Box::new(vec![3_f32, 5., 7.]));
        assert_eq!(
            runtime
                .session()
                .payload
                .retained_enclosing_storage()
                .unwrap()
                .byte_bound()
                .unwrap(),
            None
        );
        let storage = runtime.session().payload.retained_idle_storage().unwrap();
        assert_eq!(storage.nonstate_bytes().unwrap(), None);
        assert_eq!(storage.decoder_state_bytes().unwrap(), Some(0));
    }
}
