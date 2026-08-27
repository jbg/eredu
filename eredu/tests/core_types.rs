use std::any::TypeId;

#[test]
fn facade_exports_are_curated_application_types() {
    assert_eq!(
        TypeId::of::<eredu::ModelKind>(),
        TypeId::of::<eredu_architectures::ModelKind>(),
    );
    assert_eq!(
        TypeId::of::<eredu::PreparationPolicy>(),
        TypeId::of::<eredu_core::artifact::PreparationPolicy>(),
    );
    assert_eq!(
        TypeId::of::<eredu::ExecutionPlan>(),
        TypeId::of::<eredu_core::ExecutionPlan>(),
    );
    assert_eq!(
        TypeId::of::<eredu::ModelInspectionReport>(),
        TypeId::of::<eredu_core::ModelInspectionReport>(),
    );
    assert_eq!(
        TypeId::of::<eredu::MultimodalRequest>(),
        TypeId::of::<eredu_core::MultimodalRequest>(),
    );
    assert_eq!(
        TypeId::of::<eredu::QuantizationRequest>(),
        TypeId::of::<eredu_core::QuantizationRequest>(),
    );
}
