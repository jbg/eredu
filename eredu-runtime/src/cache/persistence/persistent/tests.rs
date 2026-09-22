use super::*;
use crate::cache::worker::test_support::context;
use crate::working_memory::DependencyMemoryPolicy;
use std::sync::atomic::Ordering;

#[test]
fn persistent_import_authenticates_identity_and_retains_both_accounts_without_unlinking() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("import.safetensors");
    let mut manifest = super::super::tests::manifest(&path);
    let original = fs::read(&path).unwrap();
    let (context, construction) = context();
    let (dependency_context, dependency) = crate::cache::worker::test_support::context();
    let funding = PromptCachePersistenceFunding::new(
        &context,
        dependency_context.metadata_funding().unwrap(),
        DependencyMemoryPolicy::default(),
        DEFAULT_PROMPT_CACHE_MANIFEST_BYTE_LIMIT,
    )
    .unwrap();
    manifest.blocks[0].payload_sha256.replace_range(..2, "ff");
    let bad = PersistentCacheBlockSource::prepare(
        File::open(&path).unwrap(),
        &path,
        &manifest.blocks[0],
        &funding,
    )
    .unwrap_err();
    assert!(bad.is_digest_mismatch());
    drop(bad);
    manifest.blocks[0].payload_sha256 = hash_prompt_cache_shard_payload(&path).unwrap();
    let source = PersistentCacheBlockSource::prepare(
        File::open(&path).unwrap(),
        &path,
        &manifest.blocks[0],
        &funding,
    )
    .unwrap();
    let foreign = root.path().join("foreign.safetensors");
    fs::write(&foreign, &original).unwrap();
    let refused = source
        .prepare_read_from(File::open(&foreign).unwrap(), &context)
        .unwrap_err();
    assert!(matches!(
        refused.binding_error(),
        Some(ArtifactFileReadError::Changed)
    ));
    drop(refused);
    let read = source
        .prepare_read_from(File::open(&path).unwrap(), &context)
        .unwrap();
    let alias = source.clone();
    assert!(alias.same_source(&source));
    let failure = read.read_into(&mut [0; 1]).unwrap_err();
    assert_eq!(failure.read_error().unwrap().filled_bytes(), 0);
    drop(source);
    drop(alias);
    drop(funding);
    drop(context);
    drop(dependency_context);
    assert!(!construction.retired.load(Ordering::SeqCst));
    assert!(!dependency.retired.load(Ordering::SeqCst));
    assert!(failure.source_file().is_some());
    drop(failure);
    assert!(construction.retired.load(Ordering::SeqCst));
    assert!(dependency.retired.load(Ordering::SeqCst));
    assert_eq!(fs::read(&path).unwrap(), original);
}

#[test]
fn persistent_stream_copy_preserves_payload_and_changed_source_refuses_before_destination_write() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("import.safetensors");
    let mut manifest = super::super::tests::manifest(&path);
    let mut bytes = fs::read(&path).unwrap();
    let end = bytes.len();
    bytes[end - 4..].copy_from_slice(&[7, 11, 19, 31]);
    fs::write(&path, &bytes).unwrap();
    manifest.blocks[0].payload_sha256 = hash_prompt_cache_shard_payload(&path).unwrap();
    let (context, _) = context();
    let (dependency, _) = crate::cache::worker::test_support::context();
    let funding = PromptCachePersistenceFunding::new(
        &context,
        dependency.metadata_funding().unwrap(),
        DependencyMemoryPolicy::default(),
        DEFAULT_PROMPT_CACHE_MANIFEST_BYTE_LIMIT,
    )
    .unwrap();
    let source = PersistentCacheBlockSource::prepare(
        File::open(&path).unwrap(),
        &path,
        &manifest.blocks[0],
        &funding,
    )
    .unwrap();
    let destination = root.path().join("copied.safetensors");
    let mut file = File::create(&destination).unwrap();
    source
        .prepare_read_from(File::open(&path).unwrap(), &context)
        .unwrap()
        .copy_to(&mut file)
        .unwrap();
    drop(file);
    assert_eq!(fs::read(&destination).unwrap(), bytes);
    let mut output = vec![0; source.file_bytes()];
    source
        .prepare_read_from(File::open(&path).unwrap(), &context)
        .unwrap()
        .read_into(&mut output)
        .unwrap();
    assert_eq!(output, bytes);
    let pending = source
        .prepare_read_from(File::open(&path).unwrap(), &context)
        .unwrap();
    let mut changed = bytes.clone();
    changed.push(41);
    fs::write(&path, changed).unwrap();
    let empty = root.path().join("refused.safetensors");
    let mut file = File::create(&empty).unwrap();
    let failure = pending.copy_to(&mut file).unwrap_err();
    assert_eq!(failure.read_error().unwrap().filled_bytes(), 0);
    assert_eq!(file.metadata().unwrap().len(), 0);
    assert!(matches!(
        source
            .prepare_read_from(File::open(&path).unwrap(), &context)
            .unwrap_err()
            .binding_error(),
        Some(ArtifactFileReadError::Changed)
    ));
    assert!(path.exists());
}

#[test]
fn fixed_state_source_checks_declarations_before_storage_and_retires_refused_payloads() {
    use eredu_core::cache::{StateTensorOwner, StateTensorRole};
    use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
    use std::cell::Cell;
    use std::rc::Rc;

    #[derive(Debug)]
    struct Payload {
        bytes: Vec<u8>,
        retired: Rc<Cell<bool>>,
    }
    impl AsRef<[u8]> for Payload {
        fn as_ref(&self) -> &[u8] {
            &self.bytes
        }
    }
    impl AsMut<[u8]> for Payload {
        fn as_mut(&mut self) -> &mut [u8] {
            &mut self.bytes
        }
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            self.retired.set(true);
        }
    }
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("state.safetensors");
    let payload: Vec<_> = [2.5f32, -7.0, 11.25]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    serialize_to_file(
        [(
            "recurrent",
            TensorView::new(Dtype::F32, vec![1, 3], &payload).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let declaration = PromptCacheStateTensor {
        owner: StateTensorOwner::Layer(0),
        role: StateTensorRole::Recurrent,
        shard: "state.safetensors".into(),
        array: "recurrent".into(),
        shape: vec![1, 3],
        dtype: "Float32".into(),
        logical_bytes: 12,
        payload_sha256: hash_prompt_cache_shard_payload(&path).unwrap(),
    };
    let (context, construction) = context();
    let (dependency_context, dependency) = crate::cache::worker::test_support::context();
    let funding = PromptCachePersistenceFunding::new(
        &context,
        dependency_context.metadata_funding().unwrap(),
        DependencyMemoryPolicy::default(),
        DEFAULT_PROMPT_CACHE_MANIFEST_BYTE_LIMIT,
    )
    .unwrap();
    let mut malformed = declaration.clone();
    malformed.shape = vec![3, 1];
    let allocations = Cell::new(0);
    let rejected = PersistentCacheStateTensor::<Vec<u8>>::read_with(
        File::open(&path).unwrap(),
        &path,
        &malformed,
        &funding,
        |bytes| {
            allocations.set(allocations.get() + 1);
            Ok(vec![0; bytes])
        },
    )
    .unwrap_err();
    assert!(matches!(rejected.cause, Cause::Declaration));
    assert_eq!(allocations.get(), 0);
    drop(rejected);

    let refused_retired = Rc::new(Cell::new(false));
    let mut wrong_digest = declaration.clone();
    wrong_digest.payload_sha256.replace_range(
        ..2,
        if &wrong_digest.payload_sha256[..2] == "ff" {
            "00"
        } else {
            "ff"
        },
    );
    let refused = PersistentCacheStateTensor::read_with(
        File::open(&path).unwrap(),
        &path,
        &wrong_digest,
        &funding,
        |bytes| {
            Ok(Payload {
                bytes: vec![0; bytes],
                retired: refused_retired.clone(),
            })
        },
    )
    .unwrap_err();
    assert!(refused.is_digest_mismatch());
    assert!(refused_retired.get());
    drop(refused);

    let retired = Rc::new(Cell::new(false));
    let loaded = PersistentCacheStateTensor::read_with(
        File::open(&path).unwrap(),
        &path,
        &declaration,
        &funding,
        |bytes| {
            Ok(Payload {
                bytes: vec![0; bytes],
                retired: retired.clone(),
            })
        },
    )
    .unwrap();
    assert_eq!(loaded.payload(), payload);
    assert_eq!(loaded.shape(), [1, 3]);
    assert_eq!(loaded.dtype(), Dtype::F32);
    drop((funding, context, dependency_context));
    assert!(!construction.retired.load(Ordering::SeqCst));
    assert!(!dependency.retired.load(Ordering::SeqCst));
    assert!(!retired.get());
    drop(loaded);
    assert!(retired.get());
    assert!(construction.retired.load(Ordering::SeqCst));
    assert!(dependency.retired.load(Ordering::SeqCst));
    assert!(path.exists());
}
