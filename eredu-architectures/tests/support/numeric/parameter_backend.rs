// Scalar adapter to the real selected-task preflight. Execution delegates to
// the existing prepared-reference recipe evaluator; no second equation set.
impl eredu_runtime::ParameterBackend for NumericBackend {
    type Parameter = NumericTensor;
    type MaterializedWeight = Arc<NumericTensor>;
    type MaterializationContext = NumericContext;
    type Materialization = Arc<NumericTensor>;
    type ParameterError = Error;
    fn preflight_recipe(
        recipe: &DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<(), Error> {
        let output = recipe.infer(source).map_err(Error::backend)?;
        for d in output.shape {
            i32::try_from(d).map_err(Error::backend)?;
        }
        if matches!(recipe, DerivedWeightRecipe::View { .. }) {
            return Err(Error::backend(
                "numeric dense payload reinterpretation is not admitted",
            ));
        }
        Ok(())
    }
    fn materialize(
        lease: eredu_checkpoint::store::CheckpointLease,
        _: &NumericContext,
    ) -> Result<Self::Materialization, Error> {
        let value = payload::decode(
            lease
                .encoded_bytes()
                .ok_or_else(|| Error::backend("numeric lease has no scalar bytes"))?,
            &lease.metadata().stored_dtype,
            lease.output_shape(),
        )?;
        Ok(Arc::new(value))
    }
    fn materialize_recipe(
        recipe: &DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
        context: &NumericContext,
    ) -> Result<Self::Materialization, Error> {
        payload::recipe_value(recipe, source, context).map(Arc::new)
    }
    fn materialized_weight(value: &Self::Materialization) -> &Self::MaterializedWeight {
        value
    }
    fn finish_materialization(
        value: Self::Materialization,
    ) -> Result<Self::MaterializedWeight, Error> {
        Ok(value)
    }
    fn share_materialized_weight(
        value: &Self::MaterializedWeight,
    ) -> Result<Self::MaterializedWeight, Error> {
        Ok(Arc::clone(value))
    }
    fn validate_bind(
        parameter: &Self::Parameter,
        weight: &Self::MaterializedWeight,
    ) -> Result<(), Error> {
        if parameter.shape != weight.shape || parameter.dtype != weight.dtype {
            return Err(Error::backend(
                "numeric parameter binding shape/type mismatch",
            ));
        }
        Ok(())
    }
    fn bind(parameter: &mut Self::Parameter, weight: Self::MaterializedWeight) {
        *parameter = (*weight).clone();
    }
}
