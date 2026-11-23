//! Portable transactions for a prediction extension installed in the target.

use eredu_core::{
    run_preparation::{finish_preparation, TextPreparationStage},
    speculative::DynGenerationExecutor,
    BackendError, GenerationError, ModelRuntime, SpeculativeDriverError,
    SpeculativeExecutionTopology, SpeculativeGenerationBatchOutput, SpeculativeGenerationLane,
    SpeculativeGenerationVisitor, SpeculativeTokenFilterController, TextGenerationBackend,
};

use super::ResidentBatchMechanisms;
use crate::speculative::{prepare_speculative_batch_lanes, run_speculative_batch};

/// Runs the installed predictor and ordinary target in one portable transaction.
pub fn execute_embedded_batch<B, M, C, V>(
    runtime: &mut ModelRuntime<B, super::BatchSession<B, M>>,
    lanes: Vec<SpeculativeGenerationLane<'_, crate::text_session_driver::TextPromptFor<B>, C>>,
    visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, B::Error>
where
    B: TextGenerationBackend,
    B::Error: From<BackendError>
        + From<GenerationError>
        + From<eredu_core::BackendFailure>
        + From<SpeculativeDriverError<M::Error>>,
    C: SpeculativeTokenFilterController + 'static,
    V: SpeculativeGenerationVisitor,
    M: ResidentBatchMechanisms<B>,
{
    let binding = super::batch::BatchBinding::<B, M>::new(runtime);
    let agree = |stage, status| M::agree_text_preparation(stage, status, binding.context());
    let map_agreement =
        |error| B::Error::from(SpeculativeDriverError::<M::Error>::Preparation(error));
    runtime.session_mut().with_prepared_model_transaction(
        |local| {
            finish_preparation(
                TextPreparationStage::Request,
                local,
                |status| agree(TextPreparationStage::Request, status),
                map_agreement,
            )
        },
        |model, ()| {
            let local = (|| {
                let executor = model.as_mut().bind_embedded_prediction().ok_or_else(|| {
                    B::Error::from(BackendError::Execution {
                        session: "embedded-prediction".into(),
                        operation: "bind installed executable".into(),
                        message: "target has no installed prediction extension".into(),
                    })
                })?;
                let prepared = prepare_speculative_batch_lanes(
                    lanes,
                    executor.max_proposals(),
                    M::prepare_sampling,
                )?;
                Ok((executor, prepared))
            })();
            let (mut executor, prepared) = finish_preparation(
                TextPreparationStage::Sampling,
                local,
                |status| agree(TextPreparationStage::Sampling, status),
                map_agreement,
            )?;
            let mut executor = DynGenerationExecutor::new(executor.as_mut());
            Ok(run_speculative_batch(
                &mut executor,
                prepared,
                &(),
                binding.context(),
                SpeculativeExecutionTopology::Single,
                eredu_core::speculative::timing::component_timing_enabled(),
                visitor,
            )?)
        },
    )
}
