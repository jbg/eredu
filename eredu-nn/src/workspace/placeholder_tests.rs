use super::*;
use crate::{
    EmbeddingSpec, GroupedNeuralBackend, GroupedProjectionSpec, GroupedRelu2Spec, LinearFormat,
    LinearSpec, NeuralBackend, ParameterSpec, Parameterized, Tensor,
};

#[derive(Debug)]
struct Seeds {
    tensor: bool,
    host: bool,
}
impl WorkspaceMechanisms for Seeds {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        assert!(matches!(
            operation.kind,
            WorkspaceOperationKind::ParameterPlaceholder
        ));
        assert!(operation.inputs.is_empty());
        assert_eq!(operation.outputs.len(), 1);
        assert!(operation.outputs[0].shape().is_empty());
        Ok(self.tensor.then(|| WorkspaceOperationBound {
            outputs: vec![WorkspaceOutputStorage::Allocate(
                operation.outputs[0].dtype().bytes(),
            )],
            scratch_bytes: 0,
            assumptions: "fixture copies one scalar into exact dtype-sized backing".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(self.host.then(|| WorkspaceHostBound {
            bytes: 0,
            assumptions: "fixture has no separate scalar staging".into(),
        }))
    }
}
fn context() -> WorkspaceContext {
    WorkspaceContext::new(Seeds {
        tensor: true,
        host: true,
    })
}
fn slot(name: &str) -> ParameterSpec {
    ParameterSpec::trainable(name).unwrap()
}
fn scalar_dtypes(context: &WorkspaceContext) -> Vec<WorkspaceDtype> {
    context
        .report(&[])
        .unwrap()
        .operations
        .iter()
        .map(|op| {
            assert!(matches!(
                op.kind,
                WorkspaceOperationKind::ParameterPlaceholder
            ));
            assert!(op.inputs.is_empty());
            assert_eq!(op.outputs.len(), 1);
            assert_eq!(op.outputs[0].shape(), &[] as &[i32]);
            op.outputs[0].dtype()
        })
        .collect()
}

#[test]
fn unloaded_construction_prices_distinct_scalar_seeds_not_logical_weights() {
    let context = context();
    let large = WorkspaceTensor::unloaded_f32(&[65_536, 65_536], &context).unwrap();
    let empty = WorkspaceTensor::unloaded_f32(&[0, 65_536], &context).unwrap();
    let integers = WorkspaceTensor::unloaded_i32(&[8_192], &context).unwrap();
    assert_eq!(large.shape(), [65_536, 65_536]);
    assert_eq!(integers.layout().dtype(), WorkspaceDtype::Int32);
    let alias = large.clone();
    let replacement = WorkspaceTensor::existing(large.layout().clone(), &context).unwrap();
    drop((large, empty, integers, alias, replacement));
    assert_eq!(
        scalar_dtypes(&context),
        [
            WorkspaceDtype::Float32,
            WorkspaceDtype::Float32,
            WorkspaceDtype::Int32
        ]
    );
    let report = context.report(&[]).unwrap();
    assert_eq!(report.total_bytes, Some(12));
    assert_eq!(report.retained_bytes, Some(0));
    assert_eq!(report.transient_bytes, Some(12));
    assert_eq!(report.host_workspace_bytes, Some(0));
    context.begin_state_span([]).unwrap();
    assert_eq!(context.report(&[]).unwrap().total_bytes, Some(0));
    assert!(context.report(&[]).unwrap().operations.is_empty());
    WorkspaceTensor::unloaded_f32(&[1], &context).unwrap();
    assert_eq!(context.report(&[]).unwrap().total_bytes, Some(4));
}

#[test]
fn neural_factories_price_each_packed_companion_in_its_native_placeholder_dtype() {
    use eredu_checkpoint::{AffineQuantization, BlockFp8Format, BlockFp8ScaleEncoding};
    use WorkspaceDtype::{Float32 as F, Uint32 as U32, Uint8 as U8};
    let formats = [
        (
            LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
            vec![F, F],
        ),
        (
            LinearFormatSpec::affine(
                LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap()),
                slot("scale"),
                slot("affine_bias"),
            )
            .unwrap(),
            vec![U32, F, F, F],
        ),
        (
            LinearFormatSpec::scaled(LinearFormat::MxFp4, slot("scale")).unwrap(),
            vec![U32, U8, F],
        ),
        (
            LinearFormatSpec::scaled(
                LinearFormat::E4M3BlockFp8(
                    BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
                ),
                slot("scale"),
            )
            .unwrap(),
            vec![U8, F, F],
        ),
        (
            LinearFormatSpec::scaled(
                LinearFormat::E4M3BlockFp8(
                    BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::Ue8m0).unwrap(),
                ),
                slot("scale"),
            )
            .unwrap(),
            vec![U8, U8, F],
        ),
    ];
    for (format, expected) in formats {
        let context = context();
        let mut module = WorkspaceBackend::linear(
            LinearSpec {
                input: 256,
                output: 128,
                weight: slot("weight"),
                bias: Some(slot("bias")),
                format,
            },
            &context,
        )
        .unwrap();
        assert_eq!(scalar_dtypes(&context), expected);
        assert_eq!(
            context.report(&[]).unwrap().total_bytes,
            Some(expected.iter().map(|dtype| dtype.bytes()).sum())
        );
        let mut retained = Vec::new();
        assert!(module.visit_retained_values(&mut |value| retained.push(value.clone())));
        assert_eq!(retained.len(), expected.len());
        assert!(retained.iter().all(|value| !value.shape().is_empty()));
        // Ordinary mutable parameter replacement does not construct a new seed.
        struct Replace<'a>(&'a WorkspaceContext);
        impl<'a> crate::ParameterVisitorMut<'a, WorkspaceTensor> for Replace<'_> {
            fn visit_mut(&mut self, _: crate::ParameterMetadataView<'_>, value: &'a mut WorkspaceTensor) {
                *value = WorkspaceTensor::existing(value.layout().clone(), self.0).unwrap();
            }
        }
        module.visit_parameters_mut(&mut Replace(&context));
        assert_eq!(scalar_dtypes(&context), expected);
    }
}

#[test]
fn grouped_geometry_and_alias_clones_do_not_invent_additional_constructor_seeds() {
    let context = context();
    let projection = |name| {
        GroupedProjectionSpec::new(
            slot(name),
            None,
            LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        )
        .unwrap()
    };
    let grouped = WorkspaceBackend::grouped_relu2(
        GroupedRelu2Spec::new(7, 64, 32, projection("up"), projection("down")).unwrap(),
        &context,
    )
    .unwrap();
    let mut shapes = Vec::new();
    assert!(grouped.visit_retained_values(&mut |value| shapes.push(value.shape().to_vec())));
    assert_eq!(shapes, [vec![7, 32, 64], vec![7, 64, 32]]);
    assert_eq!(scalar_dtypes(&context), [WorkspaceDtype::Float32; 2]);
    let embedding = WorkspaceBackend::embedding(
        EmbeddingSpec {
            vocabulary: 128,
            dimensions: 64,
            weight: slot("embedding"),
            format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        },
        &context,
    )
    .unwrap();
    let _alias = embedding.clone();
    assert_eq!(scalar_dtypes(&context), [WorkspaceDtype::Float32; 3]);
    // A resident construction preceding a span contributes no new span work.
    context.begin_span();
    let _group_alias = grouped.clone();
    assert!(context.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn missing_seed_facts_stay_unknown_and_invalid_geometry_records_nothing() {
    for (tensor, host) in [(false, true), (true, false), (false, false)] {
        let context = WorkspaceContext::new(Seeds { tensor, host });
        WorkspaceTensor::unloaded_f32(&[4_096, 4_096], &context).unwrap();
        let report = context.report(&[]).unwrap();
        assert_eq!(report.total_bytes, None);
        assert_eq!(report.unpriced_operations.is_empty(), tensor);
        assert_eq!(report.unpriced_host_operations.is_empty(), host);
    }
    let context = context();
    assert!(WorkspaceTensor::unloaded_f32(&[-1], &context).is_err());
    assert!(context.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn placeholder_mechanism_failure_preserves_the_original_source() {
    #[derive(Debug)]
    struct Failed;
    impl WorkspaceMechanisms for Failed {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            Err(Error::backend_retained_source(std::io::Error::other(
                "scalar allocator fact failed",
            )))
        }
    }
    let context = WorkspaceContext::new(Failed);
    let error = WorkspaceTensor::unloaded_i32(&[16], &context).unwrap_err();
    assert!(std::error::Error::source(&error)
        .unwrap()
        .is::<std::io::Error>());
    assert!(context.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn named_unloaded_parameters_preserve_only_exact_retained_representation() {
    let context = context();
    let spec = slot("learned.scalar");
    let expected = WorkspaceRepresentation::new(WorkspaceFloatingType::Bfloat16, true);
    context.install_parameter_representations(vec![
        WorkspaceParameterRepresentation::new(
            spec.id.clone(),
            context.layout(&[1], WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(expected)),
        ),
    ]).unwrap();
    let scalar = crate::Parameter::<WorkspaceTensor>::unloaded(spec.clone(), &[1], &context).unwrap();
    let changed_shape = crate::Parameter::<WorkspaceTensor>::unloaded(spec, &[2], &context).unwrap();
    let other = crate::Parameter::<WorkspaceTensor>::unloaded(slot("other.scalar"), &[1], &context).unwrap();
    let anonymous = WorkspaceTensor::unloaded_f32(&[1], &context).unwrap();
    assert_eq!(scalar.as_ref().layout().representation(), Some(expected));
    assert_eq!(changed_shape.as_ref().layout().representation(), None);
    assert_eq!(other.as_ref().layout().representation(), None);
    assert_eq!(anonymous.layout().representation(), None);
    assert_eq!(scalar_dtypes(&context), [WorkspaceDtype::Float32; 4]);
    assert_eq!(context.report(&[]).unwrap().total_bytes, Some(16));
}

#[test]
fn grouped_parameter_sources_require_complete_bank_geometry() {
    for dtype in [WorkspaceFloatingType::Float32, WorkspaceFloatingType::Float16,
        WorkspaceFloatingType::Bfloat16] {
        let context = context();
        let expected = WorkspaceRepresentation::new(dtype, true);
        context.install_parameter_representations(vec![
            WorkspaceParameterRepresentation::new(slot("up").id,
                context.layout(&[7,32,64],WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(expected))),
            WorkspaceParameterRepresentation::new(slot("down").id,
                context.layout(&[7,64,32],WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(expected))),
        ]).unwrap();
        let projection=|name| GroupedProjectionSpec::new(slot(name), None,
            LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()).unwrap();
        for (groups, name, supplied) in [(7,"up",true),(8,"up",false),(7,"other",false)] {
            let bank=WorkspaceBackend::grouped_relu2(GroupedRelu2Spec::new(groups,64,32,
                projection(name),projection("down")).unwrap(),&context).unwrap();
            let mut actual=Vec::new();
            assert!(bank.visit_retained_values(&mut |value|
                actual.push(value.layout().representation())));
            assert_eq!(actual, [supplied.then_some(expected),
                (groups==7).then_some(expected)]);
        }
        // Complete parameter geometry changes neither the placeholder count
        // nor its source-independent scalar construction.
        assert_eq!(scalar_dtypes(&context), [WorkspaceDtype::Float32;6]);
    }
}
