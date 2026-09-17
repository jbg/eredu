use super::*;
use std::{cell::RefCell, rc::Rc};

mod destinations;
mod formats;

type Events = Rc<RefCell<Vec<&'static str>>>;
#[derive(Debug)]
struct Owner {
    label: &'static str,
    events: Events,
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.events.borrow_mut().push(self.label);
    }
}
#[derive(Debug)]
struct Failure {
    shared: Option<StaticConstructionError>,
    cause: Option<Owner>,
}
impl From<StaticConstructionError> for Failure {
    fn from(shared: StaticConstructionError) -> Self {
        Self {
            shared: Some(shared),
            cause: None,
        }
    }
}
struct TraceSink {
    events: Events,
    fail: Option<&'static str>,
    offset_bits: Option<u32>,
}
impl TraceSink {
    fn create(&self, call: &'static str, retired: &'static str) -> Result<Owner, Failure> {
        self.events.borrow_mut().push(call);
        if self.fail == Some(call) {
            return Err(Failure {
                shared: None,
                cause: Some(Owner {
                    label: "cause-retire",
                    events: self.events.clone(),
                }),
            });
        }
        Ok(Owner {
            label: retired,
            events: self.events.clone(),
        })
    }
}
impl<'a> StaticModuleConstruction<'a> for TraceSink {
    type Embedding = Owner;
    type Normalization = Owner;
    type Linear = Owner;
    type Error = Failure;
    fn embedding(&mut self, d: StaticEmbeddingDeclaration<'a>) -> Result<Owner, Failure> {
        assert_eq!(
            (d.vocabulary, d.dimensions, d.weight),
            (11, 4, "model.embed_tokens.weight")
        );
        assert_eq!(d.format, LinearFormat::Dense);
        self.create("embedding", "embedding-retire")
    }
    fn normalization(&mut self, d: StaticNormalizationDeclaration<'a>) -> Result<Owner, Failure> {
        assert_eq!(
            (d.groups, d.dimensions, d.weight),
            (Some(2), 4, "model.norm.weight")
        );
        assert_eq!(d.epsilon.to_bits(), 1e-5_f32.to_bits());
        self.offset_bits = Some(d.offset.to_bits());
        self.create("norm", "norm-retire")
    }
    fn head(&mut self, d: StaticHeadDeclaration<'a>) -> Result<Owner, Failure> {
        assert_eq!((d.input, d.output, d.weight), (4, 11, "lm_head.weight"));
        assert_eq!(d.format, LinearFormat::Dense);
        self.create("head", "head-retire")
    }
}
fn source(tied: bool) -> StaticModuleSpec {
    StaticModuleSpec {
        normalization_groups: Some(2),
        embedding_weight: "model.embed_tokens.weight".into(),
        normalization_weight: "model.norm.weight".into(),
        head_weight: "lm_head.weight".into(),
        vocabulary: 11,
        hidden_size: 4,
        normalization_epsilon: 1e-5,
        normalization_offset: 0.0,
        embedding_quantization: None,
        head_format: LinearFormat::Dense,
        tied_head: tied,
    }
}
fn trace(fail: Option<&'static str>) -> TraceSink {
    TraceSink {
        events: Rc::default(),
        fail,
        offset_bits: None,
    }
}

#[test]
fn shared_static_order_preserves_partial_failures_and_successful_field_retirement() {
    let spec = source(false);
    for (fail, expected) in [
        ("embedding", vec!["embedding"]),
        ("norm", vec!["embedding", "norm", "embedding-retire"]),
        (
            "head",
            vec![
                "embedding",
                "norm",
                "head",
                "norm-retire",
                "embedding-retire",
            ],
        ),
    ] {
        let mut sink = trace(Some(fail));
        let error = spec
            .borrowed()
            .construct_with(StaticModulePlacement::Replicated, &mut sink)
            .unwrap_err();
        assert_eq!(*sink.events.borrow(), expected);
        assert!(error.cause.is_some());
        assert!(!sink.events.borrow().contains(&"cause-retire"));
        drop(error);
        assert_eq!(sink.events.borrow().last(), Some(&"cause-retire"));
    }
    let mut sink = trace(None);
    let parts = spec
        .borrowed()
        .construct_with(StaticModulePlacement::Replicated, &mut sink)
        .unwrap();
    assert_eq!(*sink.events.borrow(), ["embedding", "norm", "head"]);
    drop(parts);
    assert_eq!(
        *sink.events.borrow(),
        [
            "embedding",
            "norm",
            "head",
            "embedding-retire",
            "norm-retire",
            "head-retire"
        ]
    );
    let mut tied = source(true);
    // A tied head does not declare, validate or allocate unused head metadata.
    tied.head_weight = "   ".into();
    tied.head_format = LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
        group_size: 0,
        ..Default::default()
    });
    let mut sink = trace(None);
    let parts = tied
        .borrowed()
        .construct_with(StaticModulePlacement::Replicated, &mut sink)
        .unwrap();
    assert!(parts.lm_head.is_none());
    assert_eq!(*sink.events.borrow(), ["embedding", "norm"]);
}

#[test]
fn shared_parallel_checks_stay_before_embedding_and_after_norm_as_applicable() {
    let valid = VocabularyParallelRange {
        global_vocabulary: 11,
        local: 4..8,
    };
    let invalid = VocabularyParallelRange {
        global_vocabulary: 11,
        local: 4..12,
    };
    let spec = source(false);
    let mut sink = trace(None);
    let error = spec
        .borrowed()
        .construct_with(
            StaticModulePlacement::Vocabulary {
                embedding: &invalid,
                output: None,
            },
            &mut sink,
        )
        .unwrap_err();
    assert!(matches!(
        error.shared,
        Some(StaticConstructionError::Vocabulary(_))
    ));
    assert!(sink.events.borrow().is_empty());
    for (tied, output, expected) in [
        (true, Some(&valid), StaticConstructionError::TiedOutputRange),
        (false, None, StaticConstructionError::MissingOutputRange),
        (
            false,
            Some(&invalid),
            StaticConstructionError::Vocabulary(
                invalid.validate_global_rows_fixed(11).unwrap_err(),
            ),
        ),
    ] {
        let spec = source(tied);
        let mut sink = trace(None);
        let error = spec
            .borrowed()
            .construct_with(
                StaticModulePlacement::Vocabulary {
                    embedding: &valid,
                    output,
                },
                &mut sink,
            )
            .unwrap_err();
        assert_eq!(error.shared, Some(expected));
        assert_eq!(
            *sink.events.borrow(),
            ["embedding", "norm", "norm-retire", "embedding-retire"]
        );
    }
    let mut sink = trace(Some("norm"));
    let error = spec
        .borrowed()
        .construct_with(
            StaticModulePlacement::Vocabulary {
                embedding: &valid,
                output: None,
            },
            &mut sink,
        )
        .unwrap_err();
    assert!(error.cause.is_some()); // Native/sink norm failure precedes missing output.
    for tied in [false, true] {
        let spec = source(tied);
        let mut sink = trace(None);
        let parts = spec
            .borrowed()
            .construct_with(
                StaticModulePlacement::Vocabulary {
                    embedding: &valid,
                    output: (!tied).then_some(&valid),
                },
                &mut sink,
            )
            .unwrap();
        assert_eq!(parts.lm_head.is_none(), tied);
    }
}

#[test]
fn static_owned_source_scope_and_offset_policy_preserve_original_lifetimes() {
    struct Source {
        spec: StaticModuleSpec,
        events: Events,
    }
    impl Drop for Source {
        fn drop(&mut self) {
            self.events.borrow_mut().push("source-retire");
        }
    }
    fn consume(
        source: Source,
        sink: &mut TraceSink,
    ) -> Result<StaticModuleParts<Owner, Owner, Owner>, Failure> {
        // Same owned-argument/borrowed-worker scope as the actual public adapters.
        source
            .spec
            .borrowed()
            .construct_with(StaticModulePlacement::Replicated, sink)
    }
    for fail in [None, Some("head")] {
        let mut sink = trace(fail);
        let source = Source {
            spec: source(false),
            events: sink.events.clone(),
        };
        let result = consume(source, &mut sink);
        assert_eq!(sink.events.borrow().last(), Some(&"source-retire"));
        if fail.is_some() {
            assert_eq!(
                *sink.events.borrow(),
                [
                    "embedding",
                    "norm",
                    "head",
                    "norm-retire",
                    "embedding-retire",
                    "source-retire"
                ]
            );
        }
        drop(result);
    }
    for offset in [0.0_f32, -0.0, 1.0, f32::NAN] {
        let mut spec = source(true);
        spec.normalization_offset = offset;
        let view = spec.borrowed();
        assert!(std::ptr::eq(
            view.embedding_weight.as_ptr(),
            spec.embedding_weight.as_ptr()
        ));
        let declaration = StaticNormalizationDeclaration {
            groups: view.normalization_groups,
            dimensions: view.hidden_size,
            epsilon: view.normalization_epsilon,
            weight: view.normalization_weight,
            offset: view.normalization_offset,
        };
        let ordinary = declaration.into_ordinary().unwrap();
        match ordinary.scale {
            NormalizationScale::Learned(weight) => {
                assert!(offset == 0.0);
                assert_eq!(weight.id.as_str(), spec.normalization_weight);
            }
            NormalizationScale::LearnedOffset {
                weight,
                offset: actual,
            } => {
                assert_ne!(offset, 0.0);
                assert_eq!(actual.to_bits(), offset.to_bits());
                assert_eq!(weight.id.as_str(), spec.normalization_weight);
            }
            NormalizationScale::Unit => panic!("unexpected unit scale"),
        }
        let mut sink = trace(None);
        let _parts = view
            .construct_with(StaticModulePlacement::Replicated, &mut sink)
            .unwrap();
        assert_eq!(sink.offset_bits, Some(offset.to_bits()));
    }
}
