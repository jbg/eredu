use super::*;
use eredu_nn::{
    Parameter, ParameterSourceError, ParameterSourceVisitor, ParameterSpec, ParameterVisitor,
    ParameterVisitorMut,
};
use safemlx::Array;
use std::time::Duration;

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "MlxTensor")]
struct StaticFixture {
    weight: Parameter<MlxTensor>,
    #[parameter(skip, retained_value)]
    helper: MlxTensor,
}

struct Partial(StaticFixture);
impl Parameterized<MlxTensor> for Partial {
    fn visit_parameter_sources<'a, V: ParameterSourceVisitor<'a, MlxTensor>>(
        &'a self,
        visitor: &mut V,
    ) -> Result<(), ParameterSourceError> {
        self.0.visit_parameter_sources(visitor)?;
        Err(ParameterSourceError::UnclassifiedRetainedField)
    }

    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, MlxTensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        self.0.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.0.set_trainable(trainable);
    }
}
fn fixture() -> StaticFixture {
    let mut spec = ParameterSpec::trainable("weight").unwrap();
    spec.alias_of = Some(eredu_nn::ParameterId::new("root").unwrap());
    spec.group = Some("group".into());
    spec.linear_companion_of = Some(eredu_nn::ParameterId::new("linear").unwrap());
    spec.linear_companion = Some(eredu_nn::LinearCompanionRole::Scale);
    let mut result = StaticFixture {
        weight: Parameter::new(
            spec,
            MlxTensor::from_array(Array::from_slice(&[3_i32, 7], &[1, 2])),
        ),
        helper: MlxTensor::from_array(Array::from_slice(&[11_i32], &[1])),
    };
    result.set_trainable(false);
    result
}

#[test]
fn actual_metadata_and_auxiliary_descriptors_match_ordinary_static_rows() {
    let source = fixture();
    struct Ordinary {
        named: usize,
        bytes: usize,
        shape: usize,
        frozen: bool,
    }
    impl<'a> ParameterVisitor<'a, MlxTensor> for Ordinary {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a MlxTensor) {
            self.named += 1;
            self.bytes += metadata.id().as_str().len()
                + metadata.alias_of().as_ref().unwrap().as_str().len()
                + metadata.group().as_ref().unwrap().len()
                + metadata
                    .linear_companion_of()
                    .as_ref()
                    .unwrap()
                    .as_str()
                    .len();
            self.shape += value.as_array().shape().len();
            self.frozen &= !metadata.trainable();
        }
    }
    let mut ordinary = Ordinary {
        named: 0,
        bytes: 0,
        shape: 0,
        frozen: true,
    };
    source.visit_parameters(&mut ordinary).unwrap();
    let mut guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let mut count = ParameterOwnerCounts::default();
    count
        .observe_source(ParameterOwnerRole::Static, None, &source, &mut guard)
        .unwrap();
    let actual = count.role(ParameterOwnerRole::Static).parameters;
    assert!(ordinary.frozen);
    assert_eq!(actual.named_slots, ordinary.named);
    assert_eq!(actual.metadata_name_bytes, ordinary.bytes);
    assert_eq!(actual.shape_elements, ordinary.shape + 1);
    assert_eq!(actual.auxiliary_slots, 1);
    assert_eq!(actual.unknown_backings, 0);
    assert_eq!(count.prediction_modules, 0);
}

#[test]
fn failed_source_and_overflow_never_publish_partial_role_counts() {
    let source = Partial(fixture());
    let mut guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let mut counts = ParameterOwnerCounts::default();
    let before = counts;
    assert_eq!(
        counts.observe_source(ParameterOwnerRole::Static, None, &source, &mut guard),
        Err(ParameterOwnerSourceError::Observation {
            role: ParameterOwnerRole::Static,
            module: None,
            source: ParameterCountError::Traversal(ParameterSourceError::UnclassifiedRetainedField)
        })
    );
    assert_eq!(counts, before);
    counts.roles[ParameterOwnerRole::PredictionInner.index()]
        .parameters
        .named_slots = usize::MAX;
    let before = counts;
    assert_eq!(
        counts.observe_source(
            ParameterOwnerRole::PredictionInner,
            Some(3),
            &source,
            &mut guard
        ),
        Err(ParameterOwnerSourceError::Observation {
            role: ParameterOwnerRole::PredictionInner,
            module: Some(3),
            source: ParameterCountError::CountOverflow
        })
    );
    assert_eq!(counts, before);
    let map = BTreeMap::from([("actual".into(), source.0.helper.clone())]);
    counts.roles[ParameterOwnerRole::PredictionReplacement.index()].map_key_bytes = usize::MAX;
    let before = counts;
    assert!(matches!(
        counts.observe_map(
            ParameterOwnerRole::PredictionReplacement,
            Some(3),
            &map,
            &mut guard
        ),
        Err(ParameterOwnerSourceError::Observation {
            source: ParameterCountError::CountOverflow,
            ..
        })
    ));
    assert_eq!(counts, before);
    counts.model_prototypes = usize::MAX;
    assert_eq!(
        counts.model_prototype(),
        Err(ParameterOwnerSourceError::OwnerCountOverflow)
    );
    assert_eq!(counts.model_prototypes, usize::MAX);
    counts.prediction_managers_installed = usize::MAX;
    let before = counts;
    assert_eq!(
        counts.prediction_module(true, true),
        Err(ParameterOwnerSourceError::OwnerCountOverflow)
    );
    assert_eq!(counts, before);
}

#[test]
fn fixed_owner_errors_retain_actual_causes_and_report_concrete_layouts() {
    let error = ParameterOwnerSourceError::Observation {
        role: ParameterOwnerRole::PredictionInner,
        module: Some(7),
        source: ParameterCountError::Descriptor {
            slot: 2,
            source: safemlx::ArrayDescriptorError::SourceChanged,
        },
    };
    let inner = std::error::Error::source(&error).unwrap();
    assert!(inner.downcast_ref::<ParameterCountError>().is_some());
    assert_eq!(
        inner
            .source()
            .unwrap()
            .downcast_ref::<safemlx::ArrayDescriptorError>(),
        Some(&safemlx::ArrayDescriptorError::SourceChanged)
    );
    macro_rules! layout {
        ($ty:ty) => {
            println!(
                "{} size={} align={}",
                stringify!($ty),
                std::mem::size_of::<$ty>(),
                std::mem::align_of::<$ty>()
            );
        };
    }
    layout!(ParameterOwnerCounts);
    layout!(ParameterOwnerRoleCounts);
    layout!(ParameterOwnerSourceError);
    layout!(ParameterCountError);
    layout!(ParameterSourceCounter<'_>);
    layout!(NativeParameterOwnerSource<'_>);
    layout!(CountedNativeParameterOwnerSource<'_>);
    layout!(Result<CountedNativeParameterOwnerSource<'_>, ParameterOwnerSourceError>);
    assert!(!std::mem::needs_drop::<ParameterOwnerSourceError>());
    assert!(!std::mem::needs_drop::<CountedNativeParameterOwnerSource<'_>>());
}
