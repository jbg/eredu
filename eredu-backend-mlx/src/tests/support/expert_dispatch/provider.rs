impl LocalExpertBank for crate::backend::nn::shared::MlxGroupedRelu2 {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_group_indices.reshape(&[-1, 1], stream)?;
        let weights = unit_coefficients(hidden.dim(0), hidden.dtype(), stream)?;
        let routes = eredu_nn::GroupSelection::new(
            crate::MlxTensor::from_array(ids),
            crate::MlxTensor::from_array(weights.clone()),
            crate::MlxTensor::from_array(weights),
        );
        eredu_nn::GroupedRelu2Operator::forward_grouped(
            self,
            &crate::MlxTensor::from_array(hidden.clone()),
            &routes,
            stream,
        )
        .map(|value| value.as_array().clone())
        .map_err(Error::from)
    }
}

impl LocalExpertBank for crate::backend::nn::shared::MlxGroupedGatedProduct {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_group_indices.reshape(&[-1, 1], stream)?;
        let weights = unit_coefficients(hidden.dim(0), hidden.dtype(), stream)?;
        let routes = eredu_nn::GroupSelection::new(
            crate::MlxTensor::from_array(ids),
            crate::MlxTensor::from_array(weights.clone()),
            crate::MlxTensor::from_array(weights),
        );
        eredu_nn::GroupedGatedProductOperator::forward_grouped(
            self,
            &crate::MlxTensor::from_array(hidden.clone()),
            &routes,
            stream,
        )
        .map(|value| value.as_array().clone())
        .map_err(Error::from)
    }
}

/// Architecture-specific execution behind the common route dispatcher.
pub trait LocalExpertBank {
    /// Executes compact hidden rows using dense owner-local expert ids.
    /// Returned route rows must be unweighted and retain input order.
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error>;

    /// Executes owner-local rows while separating replicated TP down bias.
    fn execute_local_routes_tensor_parallel(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<TensorParallelGroupedOutput<Array>, Error> {
        if partitions == 0 {
            return Err(Error::Parallel(
                "tensor-parallel partition count must be positive".into(),
            ));
        }
        Ok(TensorParallelGroupedOutput::new(
            self.execute_local_routes(hidden, local_group_indices, stream)?,
            None,
        ))
    }
}

/// Creates one unit weight for every routed token.
pub fn unit_coefficients(routes: i32, dtype: Dtype, stream: &Stream) -> Result<Array, Error> {
    Ok(safemlx::ops::ones_dtype(&[routes, 1], dtype, stream)?)
}

impl LocalExpertBank for PackedGatedProductGroups {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_group_indices.reshape(&[-1, 1], stream)?;
        let weights = unit_coefficients(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }

    fn execute_local_routes_tensor_parallel(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<TensorParallelGroupedOutput<Array>, Error> {
        let ids = local_group_indices.reshape(&[-1, 1], stream)?;
        let weights = unit_coefficients(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward_tensor_parallel(hidden, &ids, &weights, partitions, stream)?)
    }
}

impl LocalExpertBank for PackedRelu2Groups {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_group_indices.reshape(&[-1, 1], stream)?;
        let weights = unit_coefficients(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }
}
