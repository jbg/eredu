//! Native output measurement and completion for replicated text.

use super::*;

impl eredu_core::checkpoint_mechanisms::NativeTextWeights for MlxTextValues {
    type Parameters = MlxNeuralBackend;

    fn observe_text_construction() {}

    fn observe_text_binding(ranked: bool) {
        let _ = ranked;
    }

    fn observe_text_state_allocation() {}

    fn text_output_row(
        output: MlxTensor,
        sequence_index: i32,
        context: &Stream,
    ) -> Result<MlxTensor, Error> {
        output
            .as_array()
            .try_index_device((.., sequence_index, ..), context)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }

    fn text_parameter_memory() -> Result<Option<eredu_core::capability::ParameterMemoryFacts>, Error>
    {
        Ok(Some(eredu_core::capability::ParameterMemoryFacts {
            native: crate::composition::mlx::capability::native_memory_facts(),
            available: crate::composition::mlx::capability::available_memory()
                .map_err(|error| Error::Other(Box::new(error)))?,
            allocator_cache_limit: match crate::allocator_cache_policy() {
                Ok(policy) => eredu_core::Observed::exact(
                    policy.limit_bytes,
                    "current MLX allocator-cache policy",
                ),
                Err(error) => eredu_core::Observed::unavailable(error.to_string()),
            },
        }))
    }

    fn complete_text_output(
        output: &MlxTensor,
        state_roots: &[&MlxTensor],
        _context: &Stream,
    ) -> Result<(), Error> {
        let token_validations = active_token_validation_arrays();
        async_eval_with_event(
            std::iter::once(output.as_array())
                .chain(state_roots.iter().map(|root| root.as_array()))
                .chain(token_validations.iter()),
        )?
        .synchronize()?;
        validate_active_token_validations().map_err(Into::into)
    }
}
