use super::*;
use eredu_core::{ControllerDeclarationData, HostPreparationAuthority};
use std::sync::atomic::{AtomicBool, Ordering};

struct Output {
    recipe: SharedControllerBytes,
    declaration: SharedControllerDeclaration,
    validation: SharedControllerDeclaration,
}
impl ControllerCompilationOutput for Output {
    fn controller_sources(&self) -> ControllerCompilationSources<'_> {
        ControllerCompilationSources {
            recipe: Some(&self.recipe),
            grammar: Some(&self.declaration),
            validation: Some(&self.validation),
        }
    }
}
struct Data([u8; 13]);
impl ControllerDeclarationData for Data {
    fn owned_capacity_bytes(&self) -> Option<u64> {
        Some(self.0.len() as u64)
    }
}
struct Compiler<'a> {
    entered: &'a AtomicBool,
    refuse: bool,
}
impl OriginalControllerCompiler for Compiler<'_> {
    type Output = Output;
    type Error = HostMetadataFundingError;
    fn compile(self, funding: &HostMetadataFunding) -> Result<Output, Self::Error> {
        self.entered.store(true, Ordering::SeqCst);
        // Actual retained byte destination, shared shells, and authority shell.
        let bytes = 7
            + SharedControllerBytes::source_shell_bytes().unwrap()
            + 2 * SharedControllerDeclaration::source_constructor_bytes::<Data>().unwrap()
            + HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().unwrap();
        funding.reserve_metadata(bytes)?;
        let authority = HostPreparationAuthority::retain(funding.clone());
        let recipe = SharedControllerBytes::new(vec![17; 7], authority.clone());
        if self.refuse {
            return Err(HostMetadataFundingError::Capacity {
                required: 31,
                available: 0,
            });
        }
        Ok(Output {
            recipe,
            declaration: SharedControllerDeclaration::new(Data([23; 13]), authority.clone()),
            validation: SharedControllerDeclaration::new(Data([29; 13]), authority),
        })
    }
}
fn preparation() -> (WorkingMemoryPool, OriginalChatProfilePreparation) {
    let pool = WorkingMemoryPool::new(64 << 20, 0).unwrap();
    let tokenizer = crate::working_memory::original_chat::tests::tokenizer(&pool);
    let template = pool
        .compile_chat_template(crate::working_memory::original_chat::tests::source_plan())
        .unwrap();
    let preparation = OriginalChatProfilePreparation::new(
        &template,
        &tokenizer,
        &crate::working_memory::InferenceExecutionIdentity::default(),
        64 << 20,
    )
    .unwrap();
    (pool, preparation)
}

#[test]
fn compilation_receipt_authenticates_exact_sources_without_copy_or_registration() {
    let (pool, preparation) = preparation();
    let entered = AtomicBool::new(false);
    let (output, receipt) = preparation
        .compile_controller(Compiler {
            entered: &entered,
            refuse: false,
        })
        .unwrap();
    assert!(entered.load(Ordering::SeqCst));
    assert!(
        receipt
            .metadata_funding()
            .same_account(preparation.metadata_funding())
    );
    receipt
        .validate_sources(output.controller_sources(), &pool)
        .unwrap();
    receipt
        .validate_validation_sources(&output.recipe, &output.validation)
        .unwrap();
    assert!(matches!(
        receipt.validate_validation_sources(&output.recipe, &output.declaration),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    // Equal values and an equal-sized foreign pool cannot authenticate sources.
    let (other, other_receipt) = preparation
        .compile_controller(Compiler {
            entered: &entered,
            refuse: false,
        })
        .unwrap();
    assert!(!receipt.same_compilation(&other_receipt));
    assert!(
        receipt
            .validate_sources(other.controller_sources(), &pool)
            .is_err()
    );
    let foreign = WorkingMemoryPool::new(64 << 20, 0).unwrap();
    assert!(
        receipt
            .validate_sources(output.controller_sources(), &foreign)
            .is_err()
    );
    assert!(
        !pool
            .validate_shared_controller_source(eredu_core::SharedControllerSource::Bytes(
                &output.recipe
            ))
            .is_ok()
    );
    let alias = output.declaration.clone();
    let receipt_alias = receipt.clone();
    drop((output, receipt, other, other_receipt, preparation));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(receipt_alias);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn semantic_consumer_requires_the_exact_tokenizer_and_selected_execution() {
    use crate::working_memory::{InferenceExecutionIdentity, PreparedSemanticSource};
    let pool = WorkingMemoryPool::new(64 << 20, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let tokenizer = crate::working_memory::original_chat::tests::tokenizer(&pool);
    let template = pool
        .compile_chat_template(crate::working_memory::original_chat::tests::source_plan())
        .unwrap();
    let preparation =
        OriginalChatProfilePreparation::new(&template, &tokenizer, &execution, 64 << 20).unwrap();
    let entered = AtomicBool::new(false);
    let (output, receipt) = preparation
        .compile_controller(Compiler {
            entered: &entered,
            refuse: false,
        })
        .unwrap();
    let semantic = PreparedSemanticSource::new(&tokenizer, &execution, 64 << 20).unwrap();
    receipt
        .validate_preparation(output.controller_sources(), &semantic)
        .unwrap();
    let (foreign_output, foreign_receipt) = preparation
        .compile_controller(Compiler {
            entered: &entered,
            refuse: false,
        })
        .unwrap();
    assert!(matches!(
        receipt.validate_preparation(foreign_output.controller_sources(), &semantic),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop((foreign_output, foreign_receipt));
    let other_execution =
        PreparedSemanticSource::new(&tokenizer, &InferenceExecutionIdentity::default(), 64 << 20)
            .unwrap();
    assert!(matches!(
        receipt.validate_preparation(output.controller_sources(), &other_execution),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let other_tokenizer = crate::working_memory::original_chat::tests::tokenizer(&pool);
    let other_source = PreparedSemanticSource::new(&other_tokenizer, &execution, 64 << 20).unwrap();
    assert!(matches!(
        receipt.validate_preparation(output.controller_sources(), &other_source),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop((
        output,
        receipt,
        semantic,
        other_execution,
        other_source,
        preparation,
        template,
        tokenizer,
        other_tokenizer,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn refused_compiler_keeps_its_original_account_through_the_error() {
    let (pool, preparation) = preparation();
    let entered = AtomicBool::new(false);
    let failure = match preparation.compile_controller(Compiler {
        entered: &entered,
        refuse: true,
    }) {
        Err(failure) => failure,
        Ok(_) => panic!("compiler refusal ignored"),
    };
    assert!(entered.load(Ordering::SeqCst));
    assert!(matches!(
        failure
            .source()
            .unwrap()
            .downcast_ref::<HostMetadataFundingError>(),
        Some(HostMetadataFundingError::Capacity {
            required: 31,
            available: 0
        })
    ));
    drop(preparation);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn insufficient_constructor_funding_never_enters_the_compiler() {
    let (pool, preparation) = preparation();
    let used = pool.used_bytes().unwrap();
    preparation
        .metadata_funding()
        .reserve_metadata((64 * 1024 * 1024 - used) as usize)
        .unwrap();
    let entered = AtomicBool::new(false);
    let failure = match preparation.compile_controller(Compiler {
        entered: &entered,
        refuse: false,
    }) {
        Err(failure) => failure,
        Ok(_) => panic!("unfunded compiler ran"),
    };
    assert!(!entered.load(Ordering::SeqCst));
    drop(preparation);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn existing_owners_without_the_actual_payer_cannot_be_published() {
    struct Adopt(Output);
    impl OriginalControllerCompiler for Adopt {
        type Output = Output;
        type Error = HostMetadataFundingError;
        fn compile(self, _: &HostMetadataFunding) -> Result<Output, Self::Error> {
            Ok(self.0)
        }
    }
    let (pool, preparation) = preparation();
    let entered = AtomicBool::new(false);
    let (output, receipt) = preparation
        .compile_controller(Compiler {
            entered: &entered,
            refuse: false,
        })
        .unwrap();
    let (_, other_preparation) = self::preparation();
    let failure = match other_preparation.compile_controller(Adopt(output)) {
        Err(failure) => failure,
        Ok(_) => panic!("foreign source was adopted"),
    };
    assert!(matches!(
        failure
            .source()
            .unwrap()
            .downcast_ref::<WorkingMemoryError>(),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    drop((receipt, preparation));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
