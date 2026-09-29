//! Source-free vision recipes retain exact physical identity until explicit binding.
use super::*;
use eredu_checkpoint::{
    recipe::RecipeCatalog,
    store::{
        CheckpointLease, CheckpointSource, StoreError, TensorMetadata, TensorReadRequest,
        TensorSourceProvenance, WeightStoreDiagnostics,
    },
    StoredDtype,
};
use eredu_runtime::ReplicatedTextPhysicalSource;
use std::sync::atomic::{AtomicUsize, Ordering};

const POSITION: &str = "model.visual.pos_embed.weight";
pub(super) struct Headers(BTreeMap<String, TensorMetadata>);
impl RecipeCatalog for Headers {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.0
            .get(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
}
pub(super) fn headers(
    source: &dyn CheckpointSource,
) -> (Headers, BTreeMap<String, ReplicatedTextPhysicalSource>) {
    let mut catalog = BTreeMap::new();
    let mut physical = BTreeMap::new();
    for key in source.source_keys() {
        let metadata = source.source_metadata(&key).unwrap();
        let provenance = source.source_provenance(&key).unwrap();
        physical.insert(
            key.clone(),
            ReplicatedTextPhysicalSource::new(
                provenance.catalog_key,
                provenance.physical_tensor,
                provenance.backing_shard.unwrap(),
                provenance.output,
                provenance.source_encoding,
                metadata.encoded_byte_len,
            )
            .unwrap(),
        );
        catalog.insert(key, metadata);
    }
    (Headers(catalog), physical)
}

#[derive(Clone, Copy, Debug)]
enum Change {
    None,
    Shape,
    PhysicalShape,
    ByteLength,
    Dtype,
    CatalogKey,
    PhysicalTensor,
    Output,
    Encoding,
    Shard,
    Missing,
}
struct CheckedSource {
    source: SharedCheckpointSource,
    change: Change,
    reads: AtomicUsize,
}
impl CheckedSource {
    fn new(source: SharedCheckpointSource, change: Change) -> Arc<Self> {
        Arc::new(Self {
            source,
            change,
            reads: AtomicUsize::new(0),
        })
    }
    fn assert_unread(&self) {
        assert_eq!(self.reads.load(Ordering::Relaxed), 0);
        assert_eq!(self.source.source_diagnostics().unwrap().physical_reads, 0);
    }
}
impl CheckpointSource for CheckedSource {
    fn source_keys(&self) -> Vec<String> {
        self.source
            .source_keys()
            .into_iter()
            .filter(|key| !(matches!(self.change, Change::Missing) && key == POSITION))
            .collect()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        let mut metadata = self.source.source_metadata(key)?;
        if key == POSITION {
            match self.change {
                Change::Shape => metadata.logical_shape[0] += 1,
                Change::PhysicalShape => metadata.physical_shape[0] += 1,
                Change::ByteLength => metadata.encoded_byte_len += 4,
                Change::Dtype => metadata.stored_dtype = StoredDtype::BF16,
                _ => (),
            }
        }
        Ok(metadata)
    }
    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        let mut provenance = self.source.source_provenance(key)?;
        if key == POSITION {
            match self.change {
                Change::CatalogKey => provenance.catalog_key = "other.logical.weight".into(),
                Change::PhysicalTensor => {
                    provenance.physical_tensor = "other.physical.weight".into()
                }
                Change::Output => provenance.output = "other.output".into(),
                Change::Encoding => {
                    provenance.source_encoding =
                        eredu_checkpoint::SourceTensorEncoding::Safetensors(StoredDtype::BF16)
                }
                Change::Shard => provenance.backing_shard = Some("different-projector.bin".into()),
                _ => (),
            }
        }
        Ok(provenance)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.source.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }
}
fn reject_changed_sources(plan: &VisionPlan, source: SharedCheckpointSource) {
    for change in [
        Change::Shape,
        Change::PhysicalShape,
        Change::ByteLength,
        Change::Dtype,
        Change::CatalogKey,
        Change::PhysicalTensor,
        Change::Output,
        Change::Encoding,
        Change::Shard,
        Change::Missing,
    ] {
        let altered = CheckedSource::new(source.clone(), change);
        assert!(
            plan.clone().bind(altered.clone()).is_err(),
            "accepted {change:?}"
        );
        altered.assert_unread();
    }
    let exact = CheckedSource::new(source, Change::None);
    let bound = plan.clone().bind(exact.clone()).unwrap();
    exact.assert_unread();
    assert_eq!(bound.static_parameters().recipes(), plan.static_recipes());
    for layer in 0..plan.config().layer_count() {
        assert_eq!(
            bound.block(layer).unwrap().recipes(),
            plan.block_recipes(layer).unwrap()
        );
    }
}

#[test]
fn gguf_vision_headers_need_no_source_and_binding_preserves_exact_projector_identity() {
    let (directory, target, _) = super::super::gguf::fixtures(false);
    let path = directory.path().join("vision.gguf");
    for (quantized, mixed) in [(false, false), (true, false), (false, true)] {
        write(&path, false, quantized, 32, mixed);
        let checkpoint = eredu_gguf::Checkpoint::open(&path).unwrap();
        let hidden = path.with_extension("hidden");
        std::fs::rename(&path, &hidden).unwrap();
        let plan = GgufVisionPlan::prepare(target.spec().configuration(), &checkpoint)
            .and_then(|plan| plan.bind_media_tokens(media(), None))
            .unwrap();
        assert!(plan.processor().is_some());
        assert_eq!(plan.config().layer_count(), 2);
        assert!(plan.block_recipes(2).is_err());
        // Recipe shape/dtype inference has only metadata, even for packed rows and
        // mixed temporal patch dtypes. No readable projector is available here.
        let catalog = plan.gguf_source().unwrap().catalog();
        for recipe in plan.static_recipes().values() {
            recipe.infer(catalog).unwrap();
        }
        std::fs::rename(&hidden, &path).unwrap();
        let source = gguf_vision_source(&plan);
        reject_changed_sources(&plan, source.clone());
        let other_path = directory.path().join("other-vision.gguf");
        write(&other_path, false, quantized, 32, mixed);
        let other = GgufVisionPlan::prepare(
            target.spec().configuration(),
            &eredu_gguf::Checkpoint::open(&other_path).unwrap(),
        )
        .and_then(|plan| plan.bind_media_tokens(media(), None))
        .unwrap();
        let other_source = CheckedSource::new(gguf_vision_source(&other), Change::None);
        assert!(plan.clone().bind(other_source.clone()).is_err());
        other_source.assert_unread();
        // Binding may use only retained metadata and leases, not reopen a path.
        std::fs::remove_file(&path).unwrap();
        let counted = CheckedSource::new(source, Change::None);
        plan.bind(counted.clone()).unwrap();
        counted.assert_unread();
    }
}

#[test]
fn safetensors_vision_headers_bind_without_payloads_and_reject_changed_physical_sources() {
    let (directory, target, _) = super::super::gguf::fixtures(false);
    let projector_path = directory.path().join("geometry.gguf");
    write(&projector_path, false, false, 32, false);
    let geometry = GgufVisionPlan::prepare(
        target.spec().configuration(),
        &eredu_gguf::Checkpoint::open(&projector_path).unwrap(),
    )
    .and_then(|plan| plan.bind_media_tokens(media(), None))
    .unwrap();
    let mut config = target.spec().configuration().clone();
    config.vision = Some(geometry.config().clone());
    config.media = Some(media());
    let (directory, source) = transforms::fixture(&values());
    let (catalog, physical) = headers(source.as_ref());
    let path = directory.path().join("model.safetensors");
    let hidden = path.with_extension("hidden");
    std::fs::rename(&path, &hidden).unwrap();
    let plan = VisionPlan::safetensors(&config, &catalog, physical.clone()).unwrap();
    assert!(plan.gguf_source().is_none());
    for recipe in plan.static_recipes().values() {
        recipe.infer(&catalog).unwrap();
    }
    reject_changed_sources(&plan, source.clone());
    // Missing or altered cold headers cannot be admitted even with matching names.
    let mut missing = physical.clone();
    missing.remove(POSITION);
    assert!(VisionPlan::safetensors(&config, &catalog, missing).is_err());
    let mut malformed = catalog.0.clone();
    malformed.get_mut(POSITION).unwrap().logical_shape[0] += 1;
    assert!(VisionPlan::safetensors(&config, &Headers(malformed), physical.clone()).is_err());
    let mut malformed = catalog.0.clone();
    malformed.get_mut(POSITION).unwrap().physical_shape[0] += 1;
    assert!(VisionPlan::safetensors(&config, &Headers(malformed), physical.clone()).is_err());
    let mut malformed = catalog.0.clone();
    malformed.get_mut(POSITION).unwrap().encoded_byte_len += 4;
    let mut wrong_bytes = physical;
    let previous = &wrong_bytes[POSITION];
    let replacement = ReplicatedTextPhysicalSource::new(
        previous.catalog_key(),
        previous.tensor(),
        previous.shard().to_path_buf(),
        previous.output(),
        previous.source_encoding().clone(),
        previous.encoded_byte_len() + 4,
    )
    .unwrap();
    wrong_bytes.insert(POSITION.into(), replacement);
    assert!(VisionPlan::safetensors(&config, &Headers(malformed), wrong_bytes).is_err());
    std::fs::rename(&hidden, &path).unwrap();
    let (_other_directory, other) = transforms::fixture(&values());
    let other = CheckedSource::new(other, Change::None);
    assert!(plan.bind(other.clone()).is_err());
    other.assert_unread();
}

#[test]
fn gguf_vision_declaration_defers_all_token_ids_until_checked_binding() {
    let (directory, target, _) = super::super::gguf::fixtures(false);
    let path = directory.path().join("token-independent.gguf");
    write(&path, false, true, 32, false);
    let checkpoint = eredu_gguf::Checkpoint::open(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let declaration = GgufVisionPlan::prepare(target.spec().configuration(), &checkpoint).unwrap();
    assert_eq!(
        declaration.config().out_hidden_size,
        target.spec().configuration().hidden_size
    );
    assert!(!declaration.physical_sources().is_empty());
    for recipe in declaration.recipes().values() {
        recipe.infer(declaration.gguf_source().catalog()).unwrap();
    }
    // A structural projector carries no invented framing IDs. Raw preparation
    // cannot proceed until the target tokenizer supplies its exact identities.
    assert!(declaration.processor().unwrap().image(4, 4).is_err());
    let bound = declaration
        .clone()
        .bind_media_tokens(media(), Some(media().image))
        .unwrap();
    let image = bound.processor().unwrap().image(4, 4).unwrap();
    assert_eq!(image.framing.start_token_id, media().start);
    assert_eq!(image.framing.end_token_id, media().end);
    assert_eq!(bound.physical_sources(), declaration.physical_sources());
    assert_eq!(bound.recipes(), declaration.recipes());
    assert!(matches!(
        bound
            .clone()
            .with_processor(declaration.processor().unwrap().clone()),
        Err(PreparationError::VisionMismatch {
            field: "processor media framing"
        })
    ));
    let mut other_tokens = media();
    std::mem::swap(&mut other_tokens.start, &mut other_tokens.end);
    let other = declaration
        .clone()
        .bind_media_tokens(other_tokens, None)
        .unwrap();
    assert!(matches!(
        bound
            .clone()
            .with_processor(other.processor().unwrap().clone()),
        Err(PreparationError::VisionMismatch {
            field: "processor media framing"
        })
    ));
    let retained_processor = bound.processor().unwrap().clone();
    bound.clone().with_processor(retained_processor).unwrap();
    let base = [media().image, media().video, media().start, media().end];
    let tokens = |ids: [u32; 4]| MediaTokens {
        image: ids[0],
        video: ids[1],
        start: ids[2],
        end: ids[3],
    };
    let reject = |ids: [u32; 4], reset| {
        assert!(matches!(
            declaration.clone().bind_media_tokens(tokens(ids), reset),
            Err(PreparationError::VisionMismatch {
                field: "media token IDs"
            })
        ));
    };
    for first in 0..4 {
        for second in first + 1..4 {
            let mut ids = base;
            ids[second] = ids[first];
            reject(ids, None);
        }
        for outside in [target.spec().configuration().vocabulary as u32, u32::MAX] {
            let mut ids = base;
            ids[first] = outside;
            reject(ids, None);
        }
    }
    // The n-gram reset ID has model semantics: it must be the image ID,
    // independently of valid vocabulary membership or visual framing roles.
    for reset in [media().video, media().start, media().end, u32::MAX] {
        reject(base, Some(reset));
    }
    let mut declared_target = target.spec().configuration().clone();
    declared_target.media = Some(media());
    let exact = GgufVisionPlan::prepare(&declared_target, &checkpoint).unwrap();
    let mut swapped = media();
    std::mem::swap(&mut swapped.image, &mut swapped.video);
    assert!(matches!(
        exact.bind_media_tokens(swapped, None),
        Err(PreparationError::VisionMismatch {
            field: "media token IDs"
        })
    ));
}

#[test]
fn gguf_vision_declaration_rejects_structural_errors_before_token_resolution() {
    let (directory, target, _) = super::super::gguf::fixtures(false);
    let path = directory.path().join("malformed-projector.gguf");
    for (missing_patch, output) in [(true, 32), (false, 31)] {
        write(&path, missing_patch, false, output, false);
        let checkpoint = eredu_gguf::Checkpoint::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(GgufVisionPlan::prepare(target.spec().configuration(), &checkpoint).is_err());
    }
    write(&path, false, false, 32, false);
    let checkpoint = eredu_gguf::Checkpoint::open(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let declaration = GgufVisionPlan::prepare(target.spec().configuration(), &checkpoint).unwrap();
    let mut mismatched = target.spec().configuration().clone();
    mismatched.vision = Some(declaration.config().clone());
    mismatched.vision.as_mut().unwrap().patch_size += 1;
    assert!(matches!(
        GgufVisionPlan::prepare(&mismatched, &checkpoint),
        Err(PreparationError::VisionMismatch {
            field: "vision geometry"
        })
    ));
}
