use super::*;

#[test]
fn residency_controllers_share_the_selected_execution_budget() {
    use crate::residency::ResidencyController;
    use eredu_checkpoint::{
        recipe::RecipeCatalog,
        store::{StoreError, TensorMetadata},
    };
    use eredu_core::residency::{
        parameter_binding::OffloadUnit, ParameterConversionRetentionPolicy,
    };
    use eredu_core::residency::{OffloadConfig, OffloadPlan};
    struct EmptyCatalog;
    impl RecipeCatalog for EmptyCatalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            Err(StoreError::UnknownTensor { key: key.into() })
        }
    }
    let registry = ConversionRetentionRegistry::default();
    let group = execution_budget(ParameterConversionRetentionPolicyReport::resolve(
        Some(ParameterConversionRetentionPolicy::Bounded { max_bytes: 16 }),
        Eligibility::Eligible,
    ))
    .unwrap();
    let partition = || {
        ResidencyController::new(
            &EmptyCatalog,
            OffloadPlan::new(OffloadConfig::default(), []).unwrap(),
            [] as [OffloadUnit; 0],
        )
        .unwrap()
    };
    assert!(partition().conversion_retention().is_none());
    let target = partition().with_conversion_retention(group.clone());
    let embedded = partition().with_conversion_retention(group.clone());
    let source = registry
        .parameter(
            ResourceIdentity {
                scope: "test".into(),
                key: "source".into(),
            },
            16,
        )
        .unwrap();
    let pending = source
        .reserve(target.conversion_retention().unwrap())
        .unwrap();
    let reserved = || {
        embedded
            .conversion_retention()
            .unwrap()
            .report()
            .usage
            .value()
            .unwrap()
            .reserved_payload_bytes
    };
    assert_eq!(reserved(), 16);
    drop(pending);
    assert_eq!(reserved(), 0);
}
