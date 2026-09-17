use super::*;
use eredu_gguf::{GgmlType, TensorInput, Writer};
use std::{cell::RefCell, io::Write};

#[derive(Default)]
struct Resolver {
    requirements: Vec<GgufCompanionRequirement>,
    calls: RefCell<Vec<&'static str>>,
    first_primary: RefCell<Option<GgufCheckpoint>>,
}

impl Resolver {
    fn resolved() -> ResolvedModelConfiguration<()> {
        ResolvedModelConfiguration::new(
            ModelConfiguration {
                declared_model_type: "fixture".into(),
                effective_model_type: "fixture".into(),
                family: "fixture".into(),
                loading_protocol: LoadingProtocol::Model,
                json: None,
            },
            (),
        )
    }
}

impl ModelConfigurationResolver for Resolver {
    type ArtifactPlan = ();

    fn resolve_safetensors(
        &self,
        _: &Value,
    ) -> Result<ResolvedModelConfiguration<()>, ArtifactError> {
        self.calls.borrow_mut().push("safetensors");
        Ok(Self::resolved())
    }

    fn resolve_gguf(
        &self,
        name: &str,
        checkpoint: &GgufCheckpoint,
    ) -> Result<ResolvedModelConfiguration<()>, ArtifactError> {
        assert_eq!(name, "fixture");
        self.calls.borrow_mut().push("resolve");
        *self.first_primary.borrow_mut() = Some(checkpoint.clone());
        Ok(Self::resolved())
    }

    fn gguf_companion_requirements(
        &self,
        _: &str,
        _: &GgufCheckpoint,
    ) -> Result<Vec<GgufCompanionRequirement>, ArtifactError> {
        self.calls.borrow_mut().push("requirements");
        Ok(self.requirements.clone())
    }

    fn artifact_plan(
        &self,
        _: &Path,
        _: ArtifactFormat,
        _: &ModelConfiguration,
        _: &TensorCatalog,
        _: Option<&ValidatedGguf>,
        _: (),
    ) -> Result<(), ArtifactError> {
        self.calls.borrow_mut().push("plan");
        Ok(())
    }
}

fn write(path: &Path, name: &str, kind: GgmlType, split: Option<usize>) {
    let mut metadata = BTreeMap::from([(
        "general.architecture".into(),
        MetadataValue::String("fixture".into()),
    )]);
    if let Some(index) = split {
        metadata.insert("split.no".into(), MetadataValue::Uint16(index as u16));
        metadata.insert("split.count".into(), MetadataValue::Uint16(2));
        metadata.insert("split.tensors.count".into(), MetadataValue::Uint64(2));
    }
    let (dimensions, data) = match kind {
        GgmlType::F32 => (
            vec![2],
            [1.25f32.to_le_bytes(), (-3.5f32).to_le_bytes()].concat(),
        ),
        GgmlType::Q8_0 => (vec![32], vec![0; 34]),
        _ => unreachable!(),
    };
    Writer::default()
        .write(
            File::create(path).unwrap(),
            &metadata,
            &[TensorInput {
                name,
                dimensions: &dimensions,
                ggml_type: kind,
                data: &data,
            }],
        )
        .unwrap();
}

fn inspect(
    path: &Path,
    resolver: &Resolver,
    captured: bool,
) -> Result<ArtifactInspection<()>, ArtifactError> {
    if captured {
        inspect_artifact_with_prepared_gguf_headers(path, resolver)
    } else {
        inspect_artifact(path, resolver)
    }
}

fn requirement(
    required: bool,
    encoding: GgufCompanionEncoding,
    depth: usize,
) -> GgufCompanionRequirement {
    GgufCompanionRequirement::new(
        GgufCompanionRole::MediaProjector,
        required,
        "mmproj",
        depth,
        encoding,
    )
    .unwrap()
}

fn captured(checkpoint: &GgufCheckpoint, expected: bool) {
    assert!(!checkpoint.shards().is_empty());
    assert!(checkpoint
        .shards()
        .iter()
        .all(|shard| shard.prepared_header().is_some() == expected));
}

#[test]
fn captured_split_inspection_preserves_first_owners_and_admission_through_consumption() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("model-00001-of-00002.gguf");
    write(&path, "a", GgmlType::F32, Some(0));
    write(
        &root.path().join("model-00002-of-00002.gguf"),
        "b",
        GgmlType::F32,
        Some(1),
    );
    let resolver = Resolver::default();
    let original = inspect_artifact_with_prepared_gguf_headers(&path, &resolver).unwrap();
    assert_eq!(
        *resolver.calls.borrow(),
        ["resolve", "requirements", "plan"]
    );
    let ordinary = inspect_artifact(&path, &Resolver::default()).unwrap();
    assert_eq!(
        ordinary.configuration().family,
        original.configuration().family
    );
    assert_eq!(ordinary.tensors().get("a").unwrap().shape, [2]);
    assert_eq!(
        ordinary.tensors().get("b").unwrap().dtype,
        original.tensors().get("b").unwrap().dtype
    );
    captured(ordinary.gguf_checkpoint().unwrap(), false);
    captured(original.gguf_checkpoint().unwrap(), true);
    assert_eq!(original.gguf_checkpoint().unwrap().shards().len(), 2);
    let token = original.admission_token();
    let projected = original.clone().map_architecture_plan(|()| 17);
    assert!(token.same_admission(&projected.admission_token()));
    assert!(!token.same_admission(&ordinary.admission_token()));
    let other = inspect_artifact_with_prepared_gguf_headers(&path, &Resolver::default()).unwrap();
    assert!(!token.same_admission(&other.admission_token()));
    let first = resolver.first_primary.borrow();
    for (a, b) in first
        .as_ref()
        .unwrap()
        .shards()
        .iter()
        .zip(projected.gguf_checkpoint().unwrap().shards())
    {
        assert!(std::ptr::eq(
            a.prepared_header().unwrap(),
            b.prepared_header().unwrap()
        ));
    }
    let plan = plan_model_preparation(
        projected,
        PreparationPolicy::default(),
        crate::backend::SessionCapabilities::default(),
    )
    .unwrap();
    assert!(token.same_admission(&plan.inspection().admission_token()));
    let ModelArtifact::Gguf { validated, .. } = plan.into_artifact() else {
        panic!("GGUF expected")
    };
    drop(first);
    drop(resolver);
    drop(original);
    drop(other);
    let (checkpoint, companions) = validated.into_parts();
    assert!(companions.is_empty());
    let mut materializer = checkpoint.into_materializer();
    let mut reference = ordinary.gguf_checkpoint().unwrap().materializer();
    for name in ["a", "b"] {
        assert_eq!(
            materializer.converted_tensor(name).unwrap(),
            reference.converted_tensor(name).unwrap()
        );
    }
}

#[test]
fn captured_companion_selection_preserves_dense_precedence_nearest_directory_and_absence() {
    for capture in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        let primary = nested.join("model.gguf");
        write(&primary, "weight", GgmlType::F32, None);
        let mut resolver = Resolver {
            requirements: vec![requirement(false, GgufCompanionEncoding::DensePreferred, 1)],
            ..Resolver::default()
        };
        assert!(inspect(&primary, &resolver, capture)
            .unwrap()
            .validated_gguf()
            .unwrap()
            .companions()
            .next()
            .is_none());
        let far = root.path().join("mmproj-dense.gguf");
        write(&far, "projector", GgmlType::F32, None);
        let near = nested.join("MMPROJ-quant.gguf");
        write(&near, "projector", GgmlType::Q8_0, None);
        let selected = inspect(&primary, &resolver, capture).unwrap();
        let companion = selected
            .validated_gguf()
            .unwrap()
            .companion(&GgufCompanionRole::MediaProjector)
            .unwrap();
        assert_eq!(companion.path(), near);
        captured(companion.checkpoint(), capture);
        resolver.requirements = vec![requirement(true, GgufCompanionEncoding::DenseRequired, 1)];
        let refusal = inspect(&primary, &resolver, capture).unwrap_err();
        assert!(
            matches!(&refusal, ArtifactError::InvalidArtifact(message) if message.contains("all 1 matching candidates are quantized"))
        );
        let dense = nested.join("mmproj-dense.gguf");
        write(&dense, "projector", GgmlType::F32, None);
        for encoding in [
            GgufCompanionEncoding::DenseRequired,
            GgufCompanionEncoding::DensePreferred,
        ] {
            resolver.requirements = vec![requirement(true, encoding, 1)];
            let selected = inspect(&primary, &resolver, capture).unwrap();
            let companion = selected
                .validated_gguf()
                .unwrap()
                .companion(&GgufCompanionRole::MediaProjector)
                .unwrap();
            assert_eq!(companion.path(), dense);
            captured(companion.checkpoint(), capture);
        }
    }
}

#[test]
fn captured_companion_refusals_preserve_actual_errors_and_callback_order() {
    let root = tempfile::tempdir().unwrap();
    let primary = root.path().join("model.gguf");
    write(&primary, "weight", GgmlType::F32, None);
    for scenario in ["missing", "duplicate", "ambiguous", "malformed"] {
        let a = root.path().join("mmproj-a.gguf");
        let b = root.path().join("mmproj-b.gguf");
        if scenario == "ambiguous" {
            write(&a, "a", GgmlType::F32, None);
            write(&b, "b", GgmlType::F32, None);
        }
        if scenario == "malformed" {
            std::fs::remove_file(&b).unwrap();
            std::fs::write(&a, b"not GGUF").unwrap();
        }
        let mut messages = Vec::new();
        for capture in [false, true] {
            let optional = scenario == "duplicate";
            let req = requirement(!optional, GgufCompanionEncoding::DensePreferred, 0);
            let resolver = Resolver {
                requirements: if optional {
                    vec![req.clone(), req]
                } else {
                    vec![req]
                },
                ..Resolver::default()
            };
            let failure = inspect(&primary, &resolver, capture).unwrap_err();
            match scenario {
                "missing" => assert!(matches!(
                    &failure,
                    ArtifactError::MissingRequiredGgufCompanion {
                        role: GgufCompanionRole::MediaProjector,
                        ..
                    }
                )),
                "duplicate" => assert!(
                    matches!(&failure, ArtifactError::InvalidArtifact(m) if m.contains("declared more than once"))
                ),
                "ambiguous" => assert!(
                    matches!(&failure, ArtifactError::InvalidArtifact(m) if m.contains("ambiguous"))
                ),
                "malformed" => assert!(matches!(&failure, ArtifactError::Gguf(_))),
                _ => unreachable!(),
            }
            assert_eq!(*resolver.calls.borrow(), ["resolve", "requirements"]);
            messages.push(failure.to_string());
        }
        assert_eq!(messages[0], messages[1], "{scenario}");
    }
}

#[test]
fn captured_entry_keeps_safetensors_and_early_container_errors_ordinary() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("config.json"), b"{}").unwrap();
    let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut file = File::create(root.path().join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header).unwrap();
    file.write_all(&7.0f32.to_le_bytes()).unwrap();
    drop(file);
    for capture in [false, true] {
        let resolver = Resolver::default();
        let actual = inspect(root.path(), &resolver, capture).unwrap();
        assert_eq!(actual.format(), ArtifactFormat::SafeTensors);
        assert!(actual.validated_gguf().is_none());
        assert_eq!(actual.tensors().get("weight").unwrap().shape, [1]);
        assert_eq!(*resolver.calls.borrow(), ["safetensors", "plan"]);
        resolver.calls.borrow_mut().clear();
        assert!(matches!(
            inspect(&root.path().join("missing"), &resolver, capture),
            Err(ArtifactError::MissingArtifact(_))
        ));
        assert!(resolver.calls.borrow().is_empty());
    }
}
